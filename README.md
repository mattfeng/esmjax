# esmjax

This project is a JAX/Equinox re-implementation of the ESM-2 protein language model ([Lin et al. (2022)](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)).

It is adapted from [Irhum Shafkat's implementation of ESM-2 in JAX/Flax](https://github.com/irhum/esmjax), itself a re-implementation of the [original model written in PyTorch](https://github.com/facebookresearch/esm).

## Directory organization

- `esmjax/`
    - `modules/`
    - `io.py`: Weight porting of all ESM-2 models (8M to 15B) to JAX from original PyTorch weights (from [irhum/esmjax](https://github.com/irhum/esmjax)).
    - `tokenizer.py`: A protein tokenizer matching the output of the original, but re-written with HuggingFace's `tokenizers` library (from [irhum/esmjax](https://github.com/irhum/esmjax)).

## Scripts

```bash
python convert_esm_weights.py
```

## Developer notes

### Numerical precision
- bfloat16 matmul precision: Work to validate the model perplexity on TPUs (and identify potential degradation) is WIP. Detailed results + plots coming soon and will be updated here.
- https://docs.jax.dev/en/latest/sharded-computation.html
- https://flax-linen.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#setup