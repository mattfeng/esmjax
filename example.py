# General imports
import numpy as np

from flax.core import frozen_dict
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# esmjax imports
from esmjax import io, tokenizer as esm_tokenizer
from esmjax.modules import models

def convert_params_to_bfloat16(params):
    return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), jax.devices("cpu")[0]) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)


def get_esm2_model(cfg, lm_head: bool = False):
    """Given original PyTorch state cfg dict, return JAX model using the spec.
    Args:
        cfg (dict): Original PyTorch state cfg dict
        lm_head (bool, optional): If True, returns model with the language model
        head on top (will compute logits instead of embeddings). Defaults to False.
    Returns:
        Tuple[nn.Module, FrozenDict]: First value is the ESM2 nn.Module, second
            is the sharding spec for all params where a constraint is specified.
    """
    num_layers = cfg["model"].encoder_layers
    embed_dim = cfg["model"].encoder_embed_dim
    num_heads = cfg["model"].encoder_attention_heads

    embedding = nn.Embed(33, embed_dim)
    # num_heads = 40
    # embed_dim = 5120
    block_fn = functools.partial(EncoderLayer, num_heads, embed_dim, embed_dim * 4)
    esm_fn = ESM2MLM if lm_head else ESM2
    esm2 = esm_fn(embedding, block_fn, num_layers)

    return esm2

# MODEL_NAME = "esm2_t6_8M_UR50D"
# MODEL_NAME = "esm2_t12_35M_UR50D"
# MODEL_NAME = "esm2_t30_150M_UR50D"
# MODEL_NAME = "esm2_t33_650M_UR50D"
# MODEL_NAME = "esm2_t36_3B_UR50D"
MODEL_NAME = "esm2_t48_15B_UR50D"

# Load in the original PyTorch state; will download if first time.
state = io.get_torch_state(MODEL_NAME)

print("loaded model into CPU memory")

esm = models.get_esm2_model(state["cfg"])
print(esm)
esm_params = io.convert_encoder(state["model"], state["cfg"])
esm_params = frozen_dict.FrozenDict({"params": esm_params})
esm_params = convert_params_to_bfloat16(esm_params)

print("converted parameters to bfloat16")

print("param memory usage:", sum(arr.nbytes for arr in jax.tree_util.tree_leaves(esm_params)))

# Create 1D GPU mesh
mesh_shape = (2,)
device_mesh = mesh_utils.create_device_mesh(mesh_shape)
print("device_mesh", device_mesh)

mesh = Mesh(device_mesh, ("X",))
print("mesh", mesh)

# get shapes
# key = jax.random.PRNGKey(0)
# arr = jnp.array([[0, 1, 2]])
# abstract_params = jax.eval_shape(esm.init, key, arr)

# print(abstract_params)

def auto_shard_params(params, mesh):
    """Automatically shards model parameters based on size & shape."""
    def shard_rule(x):
        if not isinstance(x, jnp.ndarray):
            return x  # Skip non-JAX types

        # Define a heuristic: If dimension is large, partition it across devices
        # generally, shard along the last dimension (output dim)
        pspec = (None,) * (len(x.shape) - 1) + ("X",)
        shard_spec = P(*pspec) if x.shape[-1] >= 1024 and x.shape[-1] % 2 == 0 else P()  # Large tensors get sharded

        # Apply NamedSharding
        return jax.device_put(x, NamedSharding(mesh, shard_spec))

    # Apply sharding rules recursively to all parameters
    return jax.tree_util.tree_map(shard_rule, params)

esm_params = auto_shard_params(esm_params, mesh)

print(esm_params)

# Step 1. Tokenize input protein

p53_seq = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP\
    DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK\
    SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE\
    RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS\
    SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP\
    PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG\
    GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

insulin_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED\
    LQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

random_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED"

tokenizer = esm_tokenizer.protein_tokenizer(pad_to_multiple_of=128)
tokens = [x.ids for x in tokenizer.encode_batch([p53_seq, insulin_seq,random_seq])]
batch = np.array(tokens)

print("Generated batch")

# Step 2. Get embeddings

# Create fn for inference.
apply_fn = jax.jit(esm.apply)

# Note that the first call takes a *while*
embeds = apply_fn(esm_params, batch)

print(embeds)
print(embeds.shape)