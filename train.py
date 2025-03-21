#!/usr/bin/env python3
import functools
from collections import namedtuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import flax
import flax.linen as nn
from flax.core import FrozenDict

from esmjax import tokenizer as esm_tokenizer
from esmjax.modules import models


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
    return jax.tree_util.tree_map(shard_rule, params)


def process_samples(apply_fn, esm_params, seqs):
    tokenizer = esm_tokenizer.protein_tokenizer(
        pad_to_multiple_of=128
        )
    tokens = [x.ids for x in tokenizer.encode_batch(seqs)]
    batch = np.array(tokens)

    embeds = apply_fn(esm_params, batch)

    return embeds


def train():
    pass


def train_step():
    pass


def main(model_name):
    cfg = MODEL_CONFIGS[model_name]

    # esm = models.ESM2(
    #     nn.Embed(33, cfg.embed_dim),
    #     functools.partial(
    #         models.EncoderLayer,
    #         cfg.num_heads,
    #         cfg.embed_dim,
    #         cfg.embed_dim * 4
    #         ),
    #     cfg.num_layers
    #     )

    esm = models.ESM2MLM(
        nn.Embed(33, cfg.embed_dim),
        functools.partial(
            models.EncoderLayer,
            cfg.num_heads,
            cfg.embed_dim,
            cfg.embed_dim * 4
            ),
        cfg.num_layers
        )
    print(esm)

    # get shapes (just informative; not needed)
    key = jax.random.PRNGKey(0)
    arr = jnp.array([[0, 1, 2]])
    shapes = jax.eval_shape(esm.init, key, arr)

    print(shapes)

    esm_params = flax.serialization.msgpack_restore(
        open(f"flax_params/{model_name}.bfloat16.msgpack", "rb").read()
    )

    # flax msgpack_restore does not automatically create jax.Arrays
    # instead, it creates numpy arrays, which prevents them
    # from being correctly sharded; therefore, we have to map all
    # parameters to jax.Arrays manually with tree_map
    esm_params = FrozenDict(jax.tree_util.tree_map(functools.partial(jax.device_put, device=jax.devices("cpu")[0]), esm_params))

    print("param memory usage:", human_readable_bytes(sum(arr.nbytes for arr in jax.tree_util.tree_leaves(esm_params))))

    # Create 1D GPU mesh
    mesh_shape = (2,)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    print("device_mesh", device_mesh)

    mesh = Mesh(device_mesh, ("X",))
    print("mesh", mesh)

    print("sharding parameters")
    esm_params = auto_shard_params(esm_params, mesh)
    print("parameters sharded")

    apply_fn = jax.jit(esm.apply)

    p53 = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

    sequences = [p53]

    embeds = process_samples(apply_fn, esm_params, sequences)

    print(embeds)
    print(embeds.shape)


if __name__ == "__main__":
    main("esm2_t48_15B_UR50D")