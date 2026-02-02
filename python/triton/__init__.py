"""isort:skip_file"""
# Triton Nano: Minimal compiler for vector-add
__version__ = '3.6.0'

# ---------------------------------------
# Note: import order is significant here.

# submodules
from .runtime import (
    JITFunction,
    KernelInterface,
)
from .runtime.jit import jit
from .compiler import compile

from . import language
from . import runtime

__all__ = [
    "cdiv",
    "compile",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "runtime",
]

# -------------------------------------
# misc. utilities
# -------------------------------------


def cdiv(x: int, y: int):
    return (x + y - 1) // y
