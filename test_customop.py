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
from typing import List, Optional, Sequence, Tuple, TypeVar, Generic, Type

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

def _str_to_sem(sem_option):
    sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
    if sem_option:
        if sem_option == "acquire":
            sem = ir.MEM_SEMANTIC.ACQUIRE
        elif sem_option == "release":
            sem = ir.MEM_SEMANTIC.RELEASE
        elif sem_option == "acq_rel":
            sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
        elif sem_option == "relaxed":
            sem = ir.MEM_SEMANTIC.RELAXED
        else:
            raise ValueError(f"Memory semantic {sem_option} not supported")
    return sem

def _str_to_scope(scope_option):
    scope = ir.MEM_SYNC_SCOPE.GPU
    if scope_option:
        if scope_option == "gpu":
            scope = ir.MEM_SYNC_SCOPE.GPU
        elif scope_option == "cta":
            scope = ir.MEM_SYNC_SCOPE.CTA
        elif scope_option == "sys":
            scope = ir.MEM_SYNC_SCOPE.SYSTEM
        else:
            raise ValueError(f"Memory semantic {scope_option} not supported")
    return scope

def atom_red_typechecking_impl(ptr: TensorTy, mask: TensorTy) -> Tuple[TensorTy, TensorTy]:
    if ptr.type.is_block():
        if mask is not None:
            mask = self.broadcast_impl_shape(mask, ptr.type.get_block_shapes())
    if mask is None:
        mask_ir = self.builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ty = ptr.type.with_element_ty(tl.int1)
            mask_ir = self.builder.create_splat(mask_ty.to_ir(self.builder), mask_ir)
        mask = self.tensor(mask_ir, mask_ty)
    return ptr, mask

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

@builtin
def custom_atomic(ptr, mask=None, sem=None, scope=None, _semantic=None):
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    builder = _semantic.getBuilder()
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)

    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    if ptr.type.is_block():
        dst_ty = ptr.type.with_element_ty(elt_ty)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = elt_ty

    ptr, mask = atom_red_typechecking_impl(ptr, mask)
    return tl.tensor(builder.create_nvgpu_loadacquire(ptr.handle, mask.handle, sem, scope), dst_ty)




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
    # a = custom_atomic(x_ptr, mask=mask, sem='acquire', scope='gpu')
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
