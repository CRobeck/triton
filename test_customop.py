import torch

import pytest

import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton.language.core import builtin
from typing import TypeVar, Type
from functools import wraps
import builtins
import os

T = TypeVar('T')
TensorTy = TypeVar('TensorTy')

triton.language.__all__.append("custom_add")
tensor: Type[TensorTy] = tl.tensor
builder: ir.builder

TRITON_BUILTIN = "__triton_builtin__"

def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return builtins.tuple(_unwrap_if_constexpr(x) for x in o)
    if isinstance(o, tuple):
        return tuple(_unwrap_if_constexpr(x) for x in o)
    return o.value if isinstance(o, tl.constexpr) else o

def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper

@builtin
def custom_add(x, y, sanitize_overflow: tl.constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    builder = _semantic.getBuilder()
    return tl.tensor(builder.create_custom_fadd(x.handle, y.handle), x.type)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = custom_add(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)

if __name__ == "__main__":

    size = 98432
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    output_torch = x + y
    output_triton = torch.empty_like(x)
    n_elements = output_triton.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    h = add_kernel[grid](x, y, output_triton, n_elements, BLOCK_SIZE=1024)

    print(f'The maximum difference between torch and custom triton op is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
