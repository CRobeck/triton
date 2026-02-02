"""isort:skip_file"""
# Triton Nano: Minimal language module for vector-add
# Import order is significant here.

from .core import (
    arange,
    bfloat16,
    block_type,
    constexpr,
    constexpr_type,
    dtype,
    float16,
    float32,
    float64,
    int1,
    int16,
    int32,
    int64,
    int8,
    load,
    pointer_type,
    program_id,
    store,
    tensor,
    tuple,
    tuple_type,
    uint16,
    uint32,
    uint64,
    uint8,
    void,
)

__all__ = [
    "arange",
    "bfloat16",
    "block_type",
    "constexpr",
    "dtype",
    "float16",
    "float32",
    "float64",
    "int1",
    "int16",
    "int32",
    "int64",
    "int8",
    "load",
    "pointer_type",
    "program_id",
    "store",
    "tensor",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "void",
]


def str_to_ty(name, c):
    from builtins import tuple

    if name[0] == "*":
        name = name[1:]
        const = False
        if name[0] == "k":
            name = name[1:]
            const = True
        ty = str_to_ty(name, c)
        return pointer_type(element_ty=ty, const=const)

    if name.startswith("constexpr"):
        from .core import constexpr_type
        return constexpr_type(c)

    tys = {
        "fp16": float16,
        "bf16": bfloat16,
        "fp32": float32,
        "fp64": float64,
        "i1": int1,
        "i8": int8,
        "i16": int16,
        "i32": int32,
        "i64": int64,
        "u1": int1,
        "u8": uint8,
        "u16": uint16,
        "u32": uint32,
        "u64": uint64,
        "B": int1,
    }
    return tys[name]
