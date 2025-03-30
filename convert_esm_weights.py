"""
Converts PyTorch ESM weights to JAX/Flax weights.
"""

import jax
import jax.numpy as jnp

import flax

from esmjax import io

def convert_params_to_bfloat16(params):
    return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), jax.devices("cpu")[0]) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)

# MODEL_NAME = "esm2_t6_8M_UR50D"
# MODEL_NAME = "esm2_t12_35M_UR50D"
# MODEL_NAME = "esm2_t30_150M_UR50D"
MODEL_NAME = "esm2_t33_650M_UR50D"
# MODEL_NAME = "esm2_t36_3B_UR50D"
# MODEL_NAME = "esm2_t48_15B_UR50D"

print(f"converting model {MODEL_NAME}")
# Load in the original PyTorch state; will download if first time.
state = io.get_torch_state(MODEL_NAME)

print("loaded model into CPU memory")

esm_params = io.convert_encoder(state["model"], state["cfg"], lm_head=True)
esm_params = {"params": esm_params}
esm_params = convert_params_to_bfloat16(esm_params)

print("converted parameters to bfloat16")
print("param memory usage:", sum(arr.nbytes for arr in jax.tree_util.tree_leaves(esm_params)))

esm_params_bytes = flax.serialization.msgpack_serialize(esm_params)
with open(f"flax_params/{MODEL_NAME}.bfloat16.msgpack", "wb") as f:
    f.write(esm_params_bytes)

print("done.")