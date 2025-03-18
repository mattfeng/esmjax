# General imports
import numpy as np

from flax.core import frozen_dict
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# esmjax imports
from esmjax import io, tokenizer as esm_tokenizer
from esmjax.modules import modules

def convert_params_to_bfloat16(params):
    return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), jax.devices("cpu")[0]) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)

def convert_abstract_params_to_bfloat16(params):
    return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, dtype=jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)

# MODEL_NAME = "esm2_t6_8M_UR50D"
# MODEL_NAME = "esm2_t12_35M_UR50D"
# MODEL_NAME = "esm2_t30_150M_UR50D"
# MODEL_NAME = "esm2_t36_3B_UR50D"
MODEL_NAME = "esm2_t48_15B_UR50D"

# Load in the original PyTorch state; will download if first time.
state = io.get_torch_state(MODEL_NAME)

print("loaded model into CPU memory")

esm = modules.get_esm2_model(state["cfg"])
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
key = jax.random.PRNGKey(0)
arr = jnp.array([[0, 1, 2]])
abstract_params = jax.eval_shape(esm.init, key, arr)

print(abstract_params)

def auto_shard_params(params, mesh):
    """Automatically shards model parameters based on size & shape."""
    def shard_rule(x):
        if not isinstance(x, jnp.ndarray):
            return x  # Skip non-JAX types

        # Define a heuristic: If dimension is large, partition it across devices

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