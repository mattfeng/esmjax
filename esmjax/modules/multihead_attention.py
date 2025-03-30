import functools
from typing import Callable, Optional

import einops

import flax.linen as nn

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from . import rotary_embedding

class RoPEMultiHeadDotProductAttention(nn.Module):
    """Implementation of multi-head dot product attention, with
    rotary embeddings (RoPE) applied to key and query. Also uses
    sharding from https://arxiv.org/abs/2105.04663, Table 1.

    Attributes:
        num_heads (int): Number of attention heads.
        dense_gen (Callable[[], nn.Module]): Callable to create query, key
            value and output layers from.
        qkv_features (int): Dimension of query, key and value feature projections.
        out_features (int): Dimension of output features.

    """

    num_heads: int
    dtype: jnp.dtype
    dense_gen: Callable[[], nn.Module] = nn.DenseGeneral
    qkv_features: int = None
    out_features: int = None

    @nn.compact
    def __call__(
        self,
        inputs_q: Float[Array, "batch q_len q_dim"],
        inputs_kv: Float[Array, "batch kv_len kv_dim"],
        mask: Optional[Bool[Array, "batch q_len kv_len"]] = None,
    ) -> Float[Array, "batch q_len outdim"]:

        # Pt. 1: Compute the query, key and value vectors
        # modified from https://github.com/google/flax/blob/main/flax/linen/attention.py#L232
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        # Create layer_fn for query, key and value.
        qkv_dense = functools.partial(
            self.dense_gen,
            features=(self.num_heads, head_dim),
            param_dtype=self.dtype,
            dtype=self.dtype
        )

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            qkv_dense(name="q_proj")(inputs_q),
            qkv_dense(name="k_proj")(inputs_kv),
            qkv_dense(name="v_proj")(inputs_kv),
        )

        # Pt. 2: Apply the rotary embedding to query and key.
        rotary = rotary_embedding.RotaryEmbedding(head_dim)
        query, key = rotary(query), rotary(key)

        # Pt. 3: Compute the attention weights and store them with sow.
        if mask is not None:
            mask = einops.rearrange(mask, "batch q_len kv_len -> batch () q_len kv_len")

        attn_weights = nn.attention.dot_product_attention_weights(
            query,
            key,
            mask=mask,
            dtype=self.dtype
            )
        self.sow("intermediates", "attn_weights", attn_weights)
        # modified from https://github.com/google/flax/blob/main/flax/linen/attention.py#L186
        x = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)

        # Pt. 4: Merge heads and project to output dimension.
        out = self.dense_gen(
            features=features,
            axis=(-2, -1),
            name="out_proj",
            param_dtype=self.dtype,
            dtype=self.dtype
        )(x)

        return out


class RoPEMultiHeadDotProductSelfAttention(RoPEMultiHeadDotProductAttention):
    """Special case of RoPEMultiHeadDotProductAttention, where input attends
    over itself."""

    @nn.compact
    def __call__(
        self,
        inputs: Float[Array, "batch len dim"],
        mask: Optional[Bool[Array, "batch len len"]] = None,
    ) -> Float[Array, "batch len outdim"]:
        return super().__call__(inputs, inputs, mask)
