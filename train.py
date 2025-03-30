#!/usr/bin/env python3
import functools
from collections import namedtuple
from typing import Tuple
from datetime import timedelta, datetime
from collections import deque
import time

from termcolor import colored

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

import pynvml


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
    "esm2_t18_medium_800M_UR50D": ModelConfig(
        num_layers=18, embed_dim=1920, num_heads=30
        ),
    "esm2_t22_medium_1B_UR50D": ModelConfig(
        num_layers=22, embed_dim=1920, num_heads=30
        ),
    "esm2_t24_wide_2B_UR50D": ModelConfig(
        num_layers=24, embed_dim=2560, num_heads=40
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
    # Extract days, seconds, and microseconds from the timedelta
    days = td.days
    total_seconds = td.seconds
    microseconds = td.microseconds

    # Calculate hours, minutes, and seconds from total_seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format with always showing six digits for microseconds
    if days > 0:
        return f"{days}d:{hours:02}h:{minutes:02}m:{seconds:02}.{microseconds:03}s"
    else:
        return f"{hours:02}h {minutes:02}m {seconds:02}.{microseconds:03}s"


def human_readable_bytes(num_bytes):
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

    for unit in units:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024

    return f"{num_bytes:.2f} YB"


def diff(tokens1, tokens2):
    if len(tokens1) != len(tokens2):
        raise ValueError("Both lists must have the same number of tokens.")
    
    highlighted = []
    
    # Compare tokens from both lists
    for token1, token2 in zip(tokens1, tokens2):
        if token1 != token2:
            # Highlight both tokens using termcolor.colored
            highlighted.append(colored(f"{token1}/{token2}", 'red'))
        else:
            # If tokens are identical, add them without modification
            highlighted.append(token1)
    
    return " ".join(highlighted)


def auto_shard_params(params, mesh):
    """Automatically shards model parameters based on the largest tensor dimension."""

    def shard_rule(x):
        if not isinstance(x, jnp.ndarray):
            if isinstance(x, np.ndarray):
                raise ValueError("np.ndarrays found in parameters PyTree. They will not be sharded!")
            return x  # Skip non-JAX types

        # For scalars (0-dimensional arrays), there's nothing to shard.
        if x.ndim == 0:
            return x

        # Find the axis with the largest size
        largest_axis = max(range(x.ndim), key=lambda i: x.shape[i])
        largest_dim = x.shape[largest_axis]

        # Define a heuristic: if the largest dimension meets criteria, shard along that axis.
        # (For example, if the largest dimension is at least 1024 and is even.)
        if largest_dim >= 1024 and largest_dim % 2 == 0:
            # Create a PartitionSpec: mark the largest axis as sharded ("X") and others unsharded.
            spec = P(*tuple("X" if i == largest_axis else None for i in range(x.ndim)))
        else:
            # Otherwise, no sharding is applied.
            spec = P()

        # Apply NamedSharding using the computed PartitionSpec.
        return jax.device_put(x, NamedSharding(mesh, spec))

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


def load_pretrained_params(params_file: str):
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
    optim_params,
    input_shape: Tuple[int],
    *,
    grad_acc_steps: int = 1,
    params_file=None,
    mesh=None
):
    """Create a training state to hold model parameters and the optimizer.
    """

    if params_file is None:
        # hack to force initialization of params on CPU
        # cpu_device = jax.devices("cpu")[0]
        # params = jax.jit(model.init, device=cpu_device)(rng, jnp.ones(input_shape, dtype=int))["params"]
        print("Initializing model parameters...")
        params = model.init(rng, jnp.empty(input_shape, dtype=int))["params"]
    else:
        print(f"Using pretrained parameters from {params_file}...")
        params = load_pretrained_params(params_file)["params"]

    # Convert param dtype to bfloat16
    # params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16) if isinstance(x, jnp.ndarray) else x, params)

    # move params to desired accelerator
    # params = jax.tree_util.tree_map(lambda x: jax.device_put(x), params)

    if mesh is not None:
        params = auto_shard_params(params, mesh)

    # TODO: try using gradient accumulation
    # gradient acc uses additional memory even when
    # every_k_schedule = 1
    if grad_acc_steps > 1:
        tx = optax.MultiSteps(optax.adam(**optim_params), every_k_schedule=grad_acc_steps)
    else:
        tx = optax.adam(**optim_params)

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
    print("Compiling train_step...")

    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
        state,
        state.params,
        masked_ids,
        ids,
        special_tokens_mask
    )

    state = state.apply_gradients(grads=grads)
    return state, loss


def get_memory_stats():
    stats = []

    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = human_readable_bytes(mem_info.total)
        used = human_readable_bytes(mem_info.used)
        stat_str =  f"device {i}: {used}/{total}"
        stats.append(stat_str)

    return " | ".join(stats)


def train(
    state: train_state.TrainState,
    dataloader: DataLoader,
    *,
    avg_loss_range: int = 10,
    start_step: int = 0,
    end_step: int,
):

    dataloader_it = iter(cycle(dataloader))

    recent_losses = deque()

    step = start_step

    # precompile train_step
    batch = next(dataloader_it)

    masked_ids = batch["masked_ids"]
    ids = batch["ids"]
    special_tokens_mask = batch["special_tokens_mask"]

    # TODO: dataloader should move arrays onto device (pipeline)
    masked_ids = jnp.array(masked_ids, dtype=int)
    ids = jnp.array(ids, dtype=int)
    special_tokens_mask = jnp.array(special_tokens_mask, dtype=bool)

    start_time = datetime.now()
    jax.block_until_ready(train_step(state, masked_ids, ids, special_tokens_mask))
    end_time = datetime.now()

    print(f"Compilation time: {format_timedelta(end_time - start_time)}")

    while step < end_step:
        batch = next(dataloader_it)

        masked_ids = batch["masked_ids"]
        ids = batch["ids"]
        special_tokens_mask = batch["special_tokens_mask"]

        masked_ids = jnp.array(masked_ids, dtype=int)
        ids = jnp.array(ids, dtype=int)
        special_tokens_mask = jnp.array(special_tokens_mask, dtype=bool)

        start_time = datetime.now()
        state, loss = jax.block_until_ready(train_step(state, masked_ids, ids, special_tokens_mask))
        end_time = datetime.now()

        elapsed_time = end_time - start_time

        loss = loss.item()

        print(loss)
        print(type(loss))
        # print(loss.device())

        avg_loss = f"{sum(recent_losses) / avg_loss_range:0.6f}"

        print(f"step {colored(step, 'red')}: loss {loss:0.6f} avg_loss ({avg_loss_range} steps): {colored(avg_loss, 'cyan')} ({format_timedelta(elapsed_time)}) ({get_memory_stats()})")

        recent_losses.append(loss)

        if len(recent_losses) > avg_loss_range:
            recent_losses.popleft()

        # jax.profiler.save_device_memory_profile(f"memory_{step}.prof")

        step += 1

    return state


def main(*, model_name):
    print(f"Model: {model_name}")

    rng = jax.random.PRNGKey(0)

    # SETUP HYPERPARAMETERS
    # =====================

    input_seq_len = 256
    batch_size = 4
    optim_params = {
        # "learning_rate": 1e-4,
        "learning_rate": 0,
        "b1": 0.9,
        "b2": 0.98
    }
    grad_acc_steps = 200
    # grad_acc_steps = 1
    model_dtype = jnp.bfloat16
    # model_dtype = jnp.float32

    mesh = get_1d_gpu_mesh()
    # mesh = None

    params_file = "./flax_params/esm2_t33_650M_UR50D.bfloat16.msgpack"
    # params = None

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
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    # SETUP MODEL
    # ===========

    cfg = MODEL_CONFIGS[model_name]

    esm = models.ESM2MLM(
        nn.Embed(
            dataset.tokenizer.get_vocab_size(),
            cfg.embed_dim,
            param_dtype=model_dtype,
            dtype=model_dtype
            ),
        functools.partial(
            models.EncoderLayer,
            cfg.num_heads,
            cfg.embed_dim,
            cfg.embed_dim * 4,
            model_dtype
            ),
        cfg.num_layers,
        model_dtype
        )

    print(esm.tabulate(rng, jnp.array([[0, 1, 2]], dtype=int)))

    # SETUP TRAINING
    # ==============

    print("Creating train state...")

    state = create_train_state(
        rng,
        esm,
        optim_params,
        (batch_size, input_seq_len),
        params_file=params_file,
        grad_acc_steps=grad_acc_steps,
        mesh=mesh
    )

    print("Created train state.")

    final_state = train(
        state,
        dataloader,
        end_step=100000,
        avg_loss_range=grad_acc_steps
    )



if __name__ == "__main__":
    pynvml.nvmlInit()

    # main(
    #     model_name="esm2_t48_15B_UR50D"
    # )

    # main(
    #     model_name="esm2_t36_3B_UR50D"
    # )

    main(
        model_name="esm2_t33_650M_UR50D"
    )

    # main(
    #     model_name="esm2_t30_150M_UR50D"
    # )

    # main(
    #     model_name="esm2_t12_35M_UR50D"
    # )

    # ---

    # main(
    #     model_name="esm2_t18_medium_800M_UR50D"
    # )

    # main(
    #     model_name="esm2_t22_medium_1B_UR50D"
    # )

    # main(
    #     model_name="esm2_t24_wide_2B_UR50D"
    # )


    pynvml.nvmlShutdown()
