import functools

from termcolor import colored

import numpy as np

from flax.core import frozen_dict
import flax.linen as nn

import optax

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from torch.utils.data import DataLoader

# esmjax imports
from esmjax import io, tokenizer as esm_tokenizer
from esmjax.modules import models
from esmjax.data import ESM2MaskedResidueDataset


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
    block_fn = functools.partial(models.EncoderLayer, num_heads, embed_dim, embed_dim * 4, jnp.float32)
    esm_fn = models.ESM2MLM if lm_head else models.ESM2

    esm2 = esm_fn(embedding, block_fn, num_layers, jnp.float32)

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

esm = get_esm2_model(state["cfg"], lm_head=True)
print(esm)
esm_params = io.convert_encoder(state["model"], state["cfg"], lm_head=True)
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


# ---

# p53_seq = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"


def diff(tokens1, tokens2, masked=None):
    if len(tokens1) != len(tokens2):
        raise ValueError("Both lists must have the same number of tokens.")
    
    highlighted = []
    
    # Compare tokens from both lists
    for i, (token1, token2) in enumerate(zip(tokens1, tokens2)):
        attrs = ["underline"] if (masked and masked[i]) else None

        if token1 != token2:
            # Highlight both tokens using termcolor.colored
            highlighted.append(colored(token1, "red", attrs=attrs) + "/" + colored(token2, "dark_grey", attrs=attrs))
        else:
            # If tokens are identical, add them without modification
            highlighted.append(colored(token1, attrs=attrs))
    
    return " ".join(highlighted)


def loss_fn(logits, labels, special_tokens_mask):
    # logits as token ids
    # labels as token ids
    scores = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # only tokens with real meaning should contribute to the loss
    loss = (scores * (1 - special_tokens_mask)).mean()

    return loss


hdf5_file_path = "/home/gridsan/mattfeng/datasets/esm2_pretrain_nemo2_fulldata_v1.0/full_esm2_pretrain.h5"

tokenizer = esm_tokenizer.protein_tokenizer(512)

dataset = ESM2MaskedResidueDataset(
    hdf5_file_path,
    tokenizer=tokenizer,
    seed=100,
    seq_len=512
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    drop_last=False,
    persistent_workers=True,
    num_workers=4,
    collate_fn=dataset.collate_fn
)

dataloader_it = iter(dataloader)

batch = next(dataloader_it)
print(batch)

# inference
apply_fn = jax.jit(esm.apply)
logits = apply_fn(esm_params, batch["masked_ids"])

output_probs = jax.nn.softmax(logits, axis=-1)
output_tokens = jnp.argmax(output_probs, axis=-1)

outputs_decoded = tokenizer.decode_batch(output_tokens, skip_special_tokens=False)
labels_decoded = tokenizer.decode_batch(batch["ids"], skip_special_tokens=False)

print(outputs_decoded)

for output, label, mask in zip(outputs_decoded, labels_decoded, batch["mask"]):
    output = output.split(" ")
    label = label.split(" ")
    mask = mask.tolist()

    print(diff(output, label, masked=mask))

loss = loss_fn(logits, batch["ids"], batch["special_tokens_mask"])
print(loss)