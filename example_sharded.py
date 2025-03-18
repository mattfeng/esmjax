# General imports
import numpy as np

from flax.core import frozen_dict
import jax

# esmjax imports
from esmjax import io, tokenizer as esm_tokenizer
from esmjax.modules import modules

# Imports specifically for multi-device sharding
from esmjax.modules import partitioning
from flax.linen import partitioning as nn_partitioning
from jax.experimental import maps, PartitionSpec as P, pjit


MODEL_NAME = "esm2_t48_15B_UR50D"
# Load in the original PyTorch state; will download if first time.
state = io.get_torch_state(MODEL_NAME)

esm, params_axes = modules.get_esm2_model(state["cfg"])
esm_params = io.convert_encoder(state["model"], state["cfg"])
esm_params = frozen_dict.FrozenDict({"params": esm_params})

esm_axes = partitioning.get_params_axes(esm_params, params_axes, rules=partitioning.DEFAULT_TPU_RULES)

# Create 2D TPU mesh
mesh_shape = (2, 4)  # X=2, Y=4, 8 TPUs total
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ("X", "Y"))

# Create fn for inference.
preshard_fn = pjit.pjit(
    lambda x: x,  # this function does nothing
    in_axis_resources=(esm_axes,),  # but this spec "pre-shards" the params
    out_axis_resources=esm_axes,
)

# There's two contexts: one for the mesh, the other specifying the translation
# rules for named sharding axis -> TPU mesh logical axis
with maps.Mesh(mesh.devices, mesh.axis_names), nn_partitioning.axis_rules(
    partitioning.DEFAULT_TPU_RULES
):
    esm_sharded_params = preshard_fn(esm_params)

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

tokenizer = esm_tokenizer.protein_tokenizer(pad_to_multiple_of=128)
tokens = [x.ids for x in tokenizer.encode_batch([p53_seq, insulin_seq])]
batch = np.array(tokens)

# Step 2. Get embeddings

# Create fn for inference.
apply_fn = pjit.pjit(
    esm.apply,
    in_axis_resources=(esm_axes, P("X", None)),
    out_axis_resources=P("X", None, "Y"),
)

# Note that the first call takes a *while*, about 50s on a TPUv2-8
with maps.Mesh(mesh.devices, mesh.axis_names), nn_partitioning.axis_rules(
    partitioning.DEFAULT_TPU_RULES
):
    embeds = apply_fn(esm_sharded_params, batch)

print(embeds)
print(embeds.shape)