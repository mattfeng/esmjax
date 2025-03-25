from typing import Any
import functools

import jax

from flax.core import FrozenDict


def print_if_compiling(func):
    # A simple cache keyed on argument shapes and dtypes.
    compiled_cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key based on argument shapes and dtypes where possible.
        key = tuple(
            (arg.shape, arg.dtype) if hasattr(arg, "shape") and hasattr(arg, "dtype") else arg
            for arg in args
        )
        if key not in compiled_cache:
            print(f"Compiling '{func.__name__}' for input signature: {key}")
            compiled_cache[key] = True
        return func(*args, **kwargs)

    # Return a jitted version of the wrapper.
    return jax.jit(wrapper)


def print_param_dtypes(params: Any, prefix: str = ""):
    """Recursively print the dtype of each parameter in a PyTree."""
    if isinstance(params, (dict, FrozenDict)):
        for key, value in params.items():
            print_param_dtypes(value, prefix=f"{prefix}/{key}" if prefix else key)
    elif isinstance(params, (list, tuple)):
        for idx, item in enumerate(params):
            print_param_dtypes(item, prefix=f"{prefix}[{idx}]")
    elif hasattr(params, 'dtype'):
        print(f"{prefix}: {params.dtype}")
    else:
        print(f"{prefix}: (non-array or unsupported type: {type(params)})")