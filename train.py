#!/usr/bin/env python3
import functools
from collections import namedtuple
from typing import Tuple
from datetime import timedelta, datetime

import tqdm

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import flax
import flax.linen as nn
from flax.core import FrozenDict
from flax.training import train_state

import optax

from esmjax import tokenizer as esm_tokenizer
from esmjax.modules import models
from esmjax.data import ESM2MaskedResidueDataset

from torch.utils.data import DataLoader


ModelConfig = namedtuple(
    "ModelConfig",
    ["num_layers", "embed_dim", "num_heads"]
    )

MODEL_CONFIGS = {
    "esm2_t6_8M_UR50D": ModelConfig(
        num_layers=6, embed_dim=320, num_heads=20
        ),
    "esm2_t12_35M_UR50D": ModelConfig(
        num_layers=12, embed_dim=480, num_heads=20
        ),
    "esm2_t30_150M_UR50D": ModelConfig(
        num_layers=30, embed_dim=640, num_heads=20
        ),
    "esm2_t33_650M_UR50D": ModelConfig(
        num_layers=33, embed_dim=1280, num_heads=20
        ),
    "esm2_t36_3B_UR50D": ModelConfig(
        num_layers=36, embed_dim=2560, num_heads=40
        ),
    "esm2_t48_15B_UR50D": ModelConfig(
        num_layers=48, embed_dim=5120, num_heads=40
        ),
}

def format_timedelta(td: timedelta) -> str:
    # Calculate total seconds from the timedelta
    total_seconds = int(td.total_seconds())

    # Compute days, hours, minutes, and seconds
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)        # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)          # 60 seconds in a minute

    # Format string depending on whether there are days or not
    if days > 0:
        return f"{days}d {hours:02}h:{minutes:02}m:{seconds:02}s"
    else:
        return f"{hours:02}h:{minutes:02}m:{seconds:02}s"


def human_readable_bytes(num_bytes):
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

    for unit in units:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024

    return f"{num_bytes:.2f} YB"


def auto_shard_params(params, mesh):
    """Automatically shards model parameters based on size & shape."""

    def shard_rule(x):
        if not isinstance(x, jnp.ndarray):
            if isinstance(x, np.ndarray):
                raise ValueError("np.ndarrays found in parameters PyTree. They will not be sharded!")
            return x  # Skip non-JAX types

        # Define a heuristic: If dimension is large, partition it across devices
        shard_spec = P("X") if x.shape[0] >= 1024 and x.shape[0] % 2 == 0 else P()  # Large tensors get sharded

        # Apply NamedSharding
        return jax.device_put(x, NamedSharding(mesh, shard_spec))

    # Apply sharding rules recursively to all parameters
    print("Sharding parameters...")
    sharded_params = jax.tree_util.tree_map(shard_rule, params)
    print("Parameters sharded.")

    return sharded_params


def get_1d_gpu_mesh():
    """Create 1D GPU mesh."""

    mesh_shape = (jax.local_device_count(),)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, ("X",))

    return mesh


def load_pretrained_params(params_file):
    """Load pretrained parameters.

    flax_params/{model_name}.bfloat16.msgpack
    """

    esm_params = flax.serialization.msgpack_restore(
        open(params_file, "rb").read()
    )

    # flax msgpack_restore does not automatically create jax.Arrays
    # instead, it creates numpy arrays, which prevents them
    # from being correctly sharded; therefore, we have to map all
    # parameters to jax.Arrays manually with tree_map
    esm_params = FrozenDict(jax.tree_util.tree_map(functools.partial(jax.device_put, device=jax.devices("cpu")[0]), esm_params))

    print("param memory usage:", human_readable_bytes(sum(arr.nbytes for arr in jax.tree_util.tree_leaves(esm_params))))

    return esm_params


def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float,
    input_shape: Tuple[int],
    *,
    params=None,
    mesh=None
):
    """Create a training state to hold model parameters and the optimizer.
    """

    if params is None:
        # hack to force initialization of params on CPU
        # cpu_device = jax.devices("cpu")[0]
        # params = jax.jit(model.init, device=cpu_device)(rng, jnp.ones(input_shape, dtype=int))["params"]
        params = model.init(rng, jnp.empty(input_shape, dtype=int))["params"]

    # Convert param dtype to bfloat16
    # params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16) if isinstance(x, jnp.ndarray) else x, params)

    # move params to desired accelerator
    # params = jax.tree_util.tree_map(lambda x: jax.device_put(x), params)

    # TODO: autoshard
    if mesh is not None:
        params = auto_shard_params(params, mesh)

    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    return state


def loss_fn(state, params, masked_ids, ids, special_tokens_mask):
    logits = state.apply_fn({"params": params}, masked_ids)
    scores = optax.softmax_cross_entropy_with_integer_labels(logits, ids)

    # only tokens with real meaning should contribute to the loss
    loss = (scores * (1 - special_tokens_mask)).mean()

    return loss


@jax.jit
def train_step(state, masked_ids, ids, special_tokens_mask):
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
        state,
        state.params,
        masked_ids,
        ids,
        special_tokens_mask
    )

    state = state.apply_gradients(grads=grads)
    return state, loss


def train(
    state: train_state.TrainState,
    dataloader: DataLoader,
    *,
    step_report: int = 100,
    start_step: int = 0,
    end_step: int,
):
    dataloader_it = iter(cycle(dataloader))

    recent_loss = 0.0

    step = start_step

    with tqdm.tqdm(total=end_step - start_step) as pbar:
        while step < end_step:
            pbar.set_description(f"Step {step} of {start_step}->{end_step}")

            start_time = datetime.now()
            batch = next(dataloader_it)

            masked_ids = batch["masked_ids"]
            ids = batch["ids"]
            special_tokens_mask = batch["special_tokens_mask"]

            masked_ids = jnp.array(masked_ids, dtype=int)
            ids = jnp.array(ids, dtype=int)
            special_tokens_mask = jnp.array(special_tokens_mask, dtype=bool)

            state, loss = train_step(state, masked_ids, ids, special_tokens_mask)

            end_time = datetime.now()

            elapsed_time = end_time - start_time

            print(f"step {step}: loss {loss} ({format_timedelta(elapsed_time)})")

            recent_loss += loss
            step += 1
            pbar.update(1)

            if (step - start_step) % step_report == 0:
                avg_loss = recent_loss / step_report
                print(f"\nAverage loss over {step_report} steps: {avg_loss:0.4f}")
                recent_loss = 0.0

    return state


def main(*, model_name):
    rng = jax.random.PRNGKey(0)

    # SETUP HYPERPARAMETERS
    # =====================

    input_seq_len = 256
    batch_size = 1
    learning_rate = 1e-3


    # SETUP DATASET
    # =============

    hdf5_file_path = "/home/gridsan/mattfeng/datasets/esm2_pretrain_nemo2_fulldata_v1.0/full_esm2_pretrain.h5"

    dataset = ESM2MaskedResidueDataset(
        hdf5_file_path,
        tokenizer=esm_tokenizer.protein_tokenizer(input_seq_len),
        seed=100,
        seq_len=input_seq_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )

    # SETUP MODEL
    # ===========

    cfg = MODEL_CONFIGS[model_name]

    esm = models.ESM2MLM(
        nn.Embed(dataset.tokenizer.get_vocab_size(), cfg.embed_dim),
        functools.partial(
            models.EncoderLayer,
            cfg.num_heads,
            cfg.embed_dim,
            cfg.embed_dim * 4
            ),
        cfg.num_layers
        )
    print(esm)

    # SETUP TRAINING
    # ==============

    print("Creating train state...")

    mesh = get_1d_gpu_mesh()

    state = create_train_state(rng, esm, learning_rate, (batch_size, input_seq_len), mesh=mesh)

    print("Created train state.")

    final_state = train(state, dataloader, end_step=50, step_report=5)


if __name__ == "__main__":
    # main(
    #     model_name="esm2_t48_15B_UR50D"
    # )

    main(
        model_name="esm2_t36_3B_UR50D"
    )

    # main(
    #     model_name="esm2_t33_650M_UR50D"
    # )

    # main(
    #     model_name="esm2_t30_150M_UR50D"
    # )

    # main(
    #     model_name="esm2_t12_35M_UR50D"
    # )