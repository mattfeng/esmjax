import functools
from typing import Callable, Optional

import einops

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxtyping import Array, Bool, Float, Integer

from . import multihead_attention


class EncoderLayer(nn.Module):
    """Layer definition for each layer of the encoder tower.

    Attributes:
        num_heads (int): Number of attention heads in self-attention.
        embed_dim (int): Dimensionality of the embedding vectors.
        ffn_embed_dim (int): Dimensionality of the hidden layer vectors
            in the feedforward network.
    """

    num_heads: int
    embed_dim: int
    ffn_embed_dim: int

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch len embed"],
        mask: Optional[Bool[Array, "batch len len"]] = None,
    ) -> Float[Array, "batch len embed"]:
        # Create first residual block (LayerNorm + MHA)
        residual = x
        x = nn.LayerNorm(name="self_attn_layer_norm", epsilon=1e-5)(x)

        # Note that MHA weight sharding is defined inside the layer.
        x = multihead_attention.RoPEMultiHeadDotProductSelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            name="self_attn",
        )(x, mask=mask)
        x = residual + x

        # Create second residual block (LayerNorm + MLP)
        residual = x
        x = nn.LayerNorm(name="final_layer_norm", epsilon=1e-5)(x)

        # Create + apply first MLP layer with weight + activation sharding constraints.
        x = nn.Dense(
            self.ffn_embed_dim,
            name="fc1",
        )(x)
        # Don't approximate gelu to avoid divergence with original PyTorch.
        x = nn.gelu(x, approximate=False)

        # Create + apply second MLP layer with weight + activation sharding constraints.
        x = nn.Dense(
            self.embed_dim,
            name="fc2",
        )(x)
        x = residual + x

        return x


class ESM2(nn.Module):
    """The ESM2 protein language model, outputting final layer embeddings.

    Attributes:
        embedding (nn.Module): Flax layer to use as embedding layer.
        block_gen (Callable[[], nn.Module]): Callable to create encoder layers from.
        num_layers (int): Number of layers in the encoder.
        pad_idx (int): idx of the padding token in input.
    """

    embedding: nn.Module
    block_gen: Callable[[], nn.Module]
    num_layers: int
    pad_idx: int = 1
    mask_idx: int = 32

    @nn.compact
    def __call__(
        self, tokens: Integer[Array, "batch len"]
    ) -> Float[Array, "batch len embed"]:
        # Extract out the padding mask.
        pad_embed_mask, pad_att_mask = self.get_pad_masks(tokens)

        # Get vector embeddings, and scale to account of <mask> token use in training.
        embeds = self.embedding(tokens)
        embeds = self.rescale_masked_tokens(tokens, embeds)

        # Apply all of the layer blocks in sequence, with pad masking.
        for idx in range(self.num_layers):
            embeds = self.block_gen(name=f"{idx}")(embeds, mask=pad_att_mask)

        # Apply final layer norm, and set pad embed vectors to zero.
        embeds = nn.LayerNorm(name="post_norm", epsilon=1e-5)(embeds)
        embeds = embeds * pad_embed_mask

        return embeds

    def get_pad_masks(self, tokens):
        # Get tokens which are *not* padding
        pad_embed_mask = tokens != self.pad_idx
        pad_embed_mask = einops.rearrange(pad_embed_mask, "batch len -> batch len ()")
        # For use with self-attention (assign no weight to anything to/from <pad>)
        pad_att_mask = jnp.einsum("bxh,byh->bxy", pad_embed_mask, pad_embed_mask)

        return pad_embed_mask, pad_att_mask

    def rescale_masked_tokens(self, tokens, embeds):
        embeds = embeds * (tokens != self.mask_idx)[:, :, None]

        mask_ratio_train = 0.15 * 0.8
        src_lengths = (tokens != self.pad_idx).sum(axis=1)
        # Get the mask token to sequence length ratio, per sequence.
        mask_ratio_observed = (tokens == self.mask_idx).sum(axis=1) / src_lengths
        # Rescale by the ratio between theoretical and observed.
        embeds = (
            embeds * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        )

        return embeds


class ESM2MLM(ESM2):
    """Modified version of ESM2 that returns logits."""

    @nn.compact
    def __call__(self, tokens):
        embeds = super().__call__(tokens)

        final_ffn_dim = self.embedding.features
        num_embeddings = self.embedding.num_embeddings

        x = nn.Dense(final_ffn_dim, name="lm_head_fc")(embeds)
        x = nn.gelu(x, approximate=False)
        x = nn.LayerNorm(epsilon=1e-5, name="lm_head_layer_norm")(x)

        bias = self.param("logit_bias", nn.initializers.zeros, (num_embeddings,))
        x = self.embedding.attend(x) + bias

        return x


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
