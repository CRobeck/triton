from __future__ import annotations

import math
from warnings import warn
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps, cached_property
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional, Tuple
from dataclasses import dataclass
import builtins
from .. import knobs
from ..runtime.jit import JITCallable
import inspect

from .._C.libtriton import ir
from .._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape, get_primitive_bitwidth, _tuple_create

T = TypeVar('T')

TRITON_BUILTIN = "__triton_builtin__"

PropagateNan = ir.PROPAGATE_NAN


def must_use_result(x, s=True):
    """If the result of this function is unused, throw an error."""
    if isinstance(x, str):
        return (lambda fn: must_use_result(fn, x))
    x._must_use_result = s
    return x


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
    wrapper.signature = inspect.signature(fn)

    return wrapper


def _tensor_member_fn(fn: T) -> T:
    """Decorator that adds this free function as a member fn on class tensor.

    When called as a member function on class tensor, the first argument to `fn`
    is `self`, i.e. the tensor object.

    If there are multiple decorators on a function, you probably want this one
    to be the highest one (i.e. furthest from the function's `def`), so it's
    applied last.

    Unfortunately you still need to add a type stub to the body of class tensor
    in order for pytype to know about it.
    """
    assert callable(fn)
    orig_sig = inspect.signature(fn)
    # Does fn take args other than _semantic, _generator, and the tensor itself?
    has_args = len(orig_sig.parameters.keys() - {"_semantic", "_generator"}) > 1

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f"""
    This function can also be called as a member function on :py:class:`tensor`,
    as :code:`x.{fn.__name__}({"..." if has_args else ""})` instead of
    :code:`{fn.__name__}(x{", ..." if has_args else ""})`.
    """

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Match the signature of `fn`, but change the first arg to `self` so the
    # docs are a little less weird.
    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name='self')
    new_sig = orig_sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    wrapper.signature = new_sig
    wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"
    # If fn is a builtin, mark the wrapper as a builtin too.
    if is_builtin(fn):
        setattr(wrapper, TRITON_BUILTIN, True)

    setattr(tensor, fn.__name__, fn if isinstance(fn, JITCallable) else wrapper)
    return fn


def _unwrap_iterable(x):
    """Returns x[0] if x has one element and x[0] is iterable."""
    if len(x) == 1:
        # Determine whether x[0] is iterable.
        #
        # You might want to use collections.abc.Iterable instead of this
        # try/except block.  Unfortunately, this doesn't work with constexpr.
        #
        # The problem is that abc.Iterable checks for __iter__ on the *class*.
        # But we want constexpr to expose an __iter__ method if and only if the
        # wrapped *object* (i.e. self.value) is iterable.  Therefore there's no
        # right answer for whether the class constexpr defines __iter__, and
        # abc.Iterable doesn't work (at least not without some metaclass magic).
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass

    return x


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


@builtin
def to_tensor(x, _semantic=None):
    return _semantic.to_tensor(x)


# -----------------------
# constexpr
# -----------------------


class const:
    """
    This class is used as a type annotation to mark pointers to constant data.
    The `store` function cannot be called with a pointer to const. Constness
    is part of the pointer type and the usual Triton type consistency rules
    apply. For example you cannot have a function that returns constant pointer
    in one return statement and non-constant pointer in another.
    """
    pass


class base_value:
    """Base class of values that exist in the triton IR (i.e. not constexprs).
    """
    type: base_type

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        """Flatten frontend value into a sequence of mlir handles, which are appended
        to the output list
        """
        raise NotImplementedError


class base_type:

    def __eq__(self, other) -> bool:
        raise NotImplementedError("Types must implement __eq__")

    def __ne__(self, other) -> bool:
        return not (self == other)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError

    def mangle(self) -> str:
        raise NotImplementedError(f"NYI: Type mangling for type {self.__class__}")

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        raise NotImplementedError


class constexpr_type(base_type):

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, constexpr_type) and self.value == other.value

    def __repr__(self) -> str:
        return f"constexpr_type[{self.value}]"

    def __hash__(self):
        return hash(self.value)

    def mangle(self) -> str:
        if hasattr(self.value, "mangle"):
            val = self.value.mangle()
        else:
            val = repr(self.value)
        return f"c{val}"

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        return

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return constexpr(self.value), cursor


class constexpr(base_value):
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        while isinstance(value, constexpr):
            value = value.value
        self.value = value
        self.type = constexpr_type(value)

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __hash__(self):
        return hash((self.value, self.type))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        return

    def __index__(self):
        return self.value

    # In interpreter mode, constant values are not wrapped in constexpr,
    # and therefore do not have a .value attribute.
    # As a result, from here and below, we need to call the _unwrap_if_constexpr
    # function to obtain either constexpr.value or the value itself.
    def __add__(self, other):
        return constexpr(self.value + _unwrap_if_constexpr(other))

    def __radd__(self, other):
        return constexpr(_unwrap_if_constexpr(other) + self.value)

    def __sub__(self, other):
        return constexpr(self.value - _unwrap_if_constexpr(other))

    def __rsub__(self, other):
        return constexpr(_unwrap_if_constexpr(other) - self.value)

    def __mul__(self, other):
        return constexpr(self.value * _unwrap_if_constexpr(other))

    def __mod__(self, other):
        return constexpr(self.value % _unwrap_if_constexpr(other))

    def __rmul__(self, other):
        return constexpr(_unwrap_if_constexpr(other) * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / _unwrap_if_constexpr(other))

    def __rtruediv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // _unwrap_if_constexpr(other))

    def __rfloordiv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) // self.value)

    def __gt__(self, other):
        return constexpr(self.value > _unwrap_if_constexpr(other))

    def __rgt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= _unwrap_if_constexpr(other))

    def __rge__(self, other):
        return constexpr(_unwrap_if_constexpr(other) >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < _unwrap_if_constexpr(other))

    def __rlt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) < self.value)

    def __le__(self, other):
        return constexpr(self.value <= _unwrap_if_constexpr(other))

    def __rle__(self, other):
        return constexpr(_unwrap_if_constexpr(other) <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == _unwrap_if_constexpr(other))

    def __ne__(self, other):
        return constexpr(self.value != _unwrap_if_constexpr(other))

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & _unwrap_if_constexpr(other))

    def logical_and(self, other):
        return constexpr(self.value and _unwrap_if_constexpr(other))

    def __or__(self, other):
        return constexpr(self.value | _unwrap_if_constexpr(other))

    def __xor__(self, other):
        return constexpr(self.value ^ _unwrap_if_constexpr(other))

    def logical_or(self, other):
        return constexpr(self.value or _unwrap_if_constexpr(other))

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value**_unwrap_if_constexpr(other))

    def __rpow__(self, other):
        return constexpr(_unwrap_if_constexpr(other)**self.value)

    def __rshift__(self, other):
        return constexpr(self.value >> _unwrap_if_constexpr(other))

    def __lshift__(self, other):
        return constexpr(self.value << _unwrap_if_constexpr(other))

    def __not__(self):
        return constexpr(not self.value)

    def __iter__(self):
        return iter(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def __getitem__(self, *args):
        args = (_unwrap_if_constexpr(x) for x in _normalize_tuple(args))
        return self.value.__getitem__(*args)


CONSTEXPR_0 = constexpr(0)


# Triton Nano: Minimal tuple class for code_generator compatibility
class tuple_type(base_type):
    """Stub tuple_type for compatibility."""
    pass


class tuple(base_value):
    """Minimal tuple class for code_generator compatibility."""

    def __init__(self, values, type=None):
        self._values = list(values)
        self._type = type

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def _setitem(self, idx, val):
        """Set item at index (for mutable tuple semantics)."""
        self._values[idx] = val

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self._values == other._values
        if isinstance(other, (list, builtins.tuple)):
            return self._values == list(other)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def type(self):
        return self._type


def _tuple_create(original, values):
    """Helper to preserve tuple type when unwrapping."""
    return builtins.tuple(values)


def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return _tuple_create(o, [_unwrap_if_constexpr(x) for x in o])
    if isinstance(o, tuple):
        return tuple([_unwrap_if_constexpr(x) for x in o], o.type)
    return o.value if isinstance(o, constexpr) else o


def _normalize_tuple(t):
    normalized_tuple = _unwrap_if_constexpr(t)
    if isinstance(normalized_tuple, (list, builtins.tuple)):
        normalized_tuple = tuple(normalized_tuple)
    return normalized_tuple


def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.type.scalar.primitive_bitwidth
        if shift_value.value >= bitwidth:
            warn(
                f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) for type '{value.dtype}'. This may result in undefined behavior."
            )


# -----------------------
# dtype
# -----------------------


class dtype(base_type):
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    class KIND(Enum):
        BOOLEAN = 0
        INTEGRAL = 1
        FLOATING = 2

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        self.primitive_bitwidth = get_primitive_bitwidth(name)
        self.itemsize = self.primitive_bitwidth // 8
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.exponent_bias = 16
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 52
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def kind(self):
        # Return int value following the type ordering bool < integer < fp
        if self.is_bool():
            return dtype.KIND.BOOLEAN
        elif self.is_int():
            return dtype.KIND.INTEGRAL
        else:
            assert self.is_floating()
            return dtype.KIND.FLOATING

    def get_int_max_value(self):
        if self.is_int_signed():
            return 2**(self.int_bitwidth - 1) - 1
        if self.is_int_unsigned():
            return 2**self.int_bitwidth - 1
        assert False

    def get_int_min_value(self):
        if self.is_int_signed():
            return -2**(self.int_bitwidth - 1)
        if self.is_int_unsigned():
            return 0
        assert False

    @staticmethod
    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other) -> bool:
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, ))

    @property
    def scalar(self):
        return self

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name.startswith("fp8"):
            if hasattr(builder, "options") and self.name not in builder.options.supported_fp8_dtypes:
                raise ValueError(f'type {self} not supported in this architecture. '
                                 f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')

        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_ty()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_ty()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_ty()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):
        return self.name

    def codegen_name(self):
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        if self.is_int():
            SIGNED = dtype.SIGNEDNESS.SIGNED
            prefix = 'i' if self.int_signedness == SIGNED else 'u'
            return prefix + str(self.int_bitwidth)
        if self.is_floating():
            return str(self)
        if self.is_void():
            return 'V'
        return super().mangle()

    def with_element_ty(self, element_ty: dtype):
        assert not self.is_block()
        return element_ty


# Some functions have a param named `dtype`, which shadows the `dtype` class.
# We can't change the param name because it is part of function's public API.
# Declare an alias so those functions can still reference the dtype class.
_DtypeClass = dtype


class pointer_type(dtype):

    def __init__(self, element_ty: dtype, address_space: int = 1, const: bool = False):
        element_ty = _unwrap_if_constexpr(element_ty)
        if not isinstance(element_ty, dtype):
            raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
        self.element_ty = element_ty
        self.address_space = address_space
        self.const = const
        self.name = f'pointer<{element_ty}>' if not const else f'const_pointer<{element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def is_const(self):
        return self.const

    def __eq__(self, other) -> bool:
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space and self.const == other.const

    @property
    def scalar(self):
        return self

    def mangle(self) -> str:
        return f"P{self.element_ty.mangle()}"


class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty

        # Note that block_type's shape is a list of int
        # while tensor's shape is a list of constexpr.
        assert (isinstance(shape, (list, tuple)))

        # shape can be empty ([]) when an input is a 0D tensor.
        self.shape = tuple(_unwrap_shape(shape))
        if not self.shape:
            raise TypeError('0d block_type is forbidden')

        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> Tuple[int]:
        return self.shape

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return block_type(scalar_ty, self.shape)

    def __eq__(self, other) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    @property
    def scalar(self):
        return self.element_ty

    @property
    def nbytes(self):
        return self.numel * (self.element_ty.primitive_bitwidth // 8)

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        return f'{elt}S{shape}S'


# scalar types
void = dtype('void')
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8e5 = dtype('fp8e5')
float8e5b16 = dtype('fp8e5b16')
float8e4nv = dtype('fp8e4nv')
float8e4b8 = dtype('fp8e4b8')
float8e4b15 = dtype('fp8e4b15')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_type(int32)


def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    if bitwidth == 1:
        return int1
    elif bitwidth == 8 and signed:
        return int8
    elif bitwidth == 8 and not signed:
        return uint8
    elif bitwidth == 16 and signed:
        return int16
    elif bitwidth == 16 and not signed:
        return uint16
    elif bitwidth == 32 and signed:
        return int32
    elif bitwidth == 32 and not signed:
        return uint32
    elif bitwidth == 64 and signed:
        return int64
    elif bitwidth == 64 and not signed:
        return uint64
    else:
        raise ValueError(f'Unsupported bitwidth {bitwidth} and signedness {signed}')


# -----------------------
# tensor
# -----------------------


class tensor(base_value):
    """Represents an N-dimensional array of values or pointers.

    :code:`tensor` is the fundamental data structure in Triton programs.  Most
    functions in :py:mod:`triton.language` operate on and return tensors.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    :code:`tensor` also defines most of the magic/dunder methods, so you can
    write :code:`x+y`, :code:`x << 2`, etc.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, type: dtype):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.numel = constexpr(math.prod(self.shape))
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        self.shape = tuple([constexpr(s) for s in self.shape])

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        # ex. "float32[16, 32]"
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _semantic=None):
        return add(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __radd__(self, other, _semantic=None):
        return add(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __sub__(self, other, _semantic=None):
        return sub(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rsub__(self, other, _semantic=None):
        return sub(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __mul__(self, other, _semantic=None):
        return mul(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rmul__(self, other, _semantic=None):
        return mul(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __truediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(self, other)

    @builtin
    def __rtruediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(other, self)

    @builtin
    def __floordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(self, other)

    @builtin
    def __rfloordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(other, self)

    @builtin
    def __mod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(self, other)

    @builtin
    def __rmod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(other, self)

    # unary operators
    @builtin
    def __neg__(self, _semantic=None):
        return _semantic.minus(self)

    @builtin
    def __invert__(self, _semantic=None):
        return _semantic.invert(self)

    # bitwise operators

    @builtin
    def __and__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(self, other)

    @builtin
    def __rand__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(other, self)

    @builtin
    def __or__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(self, other)

    @builtin
    def __ror__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(other, self)

    @builtin
    def __xor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(self, other)

    @builtin
    def __rxor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(other, self)

    @builtin
    def __lshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(self, other)

    @builtin
    def __rlshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(other, self)

    @builtin
    def __rshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(self, other)
        else:
            return _semantic.lshr(self, other)

    @builtin
    def __rrshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(other, self)
        else:
            return _semantic.lshr(other, self)

    # >
    @builtin
    def __gt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(self, other)

    @builtin
    def __rgt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(other, self)

    # >=
    @builtin
    def __ge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(self, other)

    @builtin
    def __rge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(other, self)

    # <
    @builtin
    def __lt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(self, other)

    @builtin
    def __rlt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(other, self)

    # <=
    @builtin
    def __le__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(self, other)

    @builtin
    def __rle__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(other, self)

    # ==
    @builtin
    def __eq__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(self, other)

    @builtin
    def __req__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(other, self)

    @builtin
    def __ne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(self, other)

    @builtin
    def __rne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(other, self)

    @builtin
    def logical_and(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_and(self, other)

    @builtin
    def logical_or(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_or(self, other)

    # note: __not__ isn't actually a magic method in python
    # but it's ok because our ASTVisitor handles it
    @builtin
    def __not__(self, _semantic=None):
        return _semantic.not_(self)

    @builtin
    def __getitem__(self, slices, _semantic=None):
        if isinstance(slices, (builtins.slice, slice, constexpr)) or slices is None:
            slices = [slices]
        if isinstance(slices, tuple):
            slices = slices.values
        ret = self
        for dim, sl in enumerate(slices):
            if _unwrap_if_constexpr(sl) is None:
                ret = _semantic.expand_dims(ret, dim)
            elif isinstance(sl, (builtins.slice, slice)) and all(
                    _unwrap_if_constexpr(arg) is None for arg in (sl.start, sl.stop, sl.step)):
                pass  # an unsqueeze
            else:
                raise ValueError(f"unsupported tensor index: {sl}")
        return ret

    @property
    def T(self):
        """Transposes a 2D tensor."""
        assert False, "Transposition must be created by the AST Visitor"

    @builtin
    def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None):
        """
        Alias for :py:func:`tensor.cast`.
        """
        return cast(self, dtype, fp_downcast_rounding, bitcast, _semantic=_semantic)

    # Type stubs for functions added by the _tensor_member_fn decorator.
    # (Unfortunately these can't be created automatically.)
    #
    # We couldn't write these definitions out even if we wanted to, because some
    # of these functions are defined in standard.py.
    def broadcast_to(self, *shape) -> tensor:
        ...

    def trans(self, *dims) -> tensor:
        ...

    def permute(self, *dims) -> tensor:
        ...

    def split(self) -> tuple[tensor, tensor]:
        ...

    def view(self, *shape) -> tensor:
        ...

    def reshape(self, *shape) -> tensor:
        ...

    def expand_dims(self, axis) -> tensor:
        ...

    def cast(self, dtype, fp_downcast_rounding=None, bitcast=False) -> tensor:
        ...

    def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="") -> tensor:
        ...

    def advance(self, offsets) -> tensor:
        ...

    def atomic_cas(self, cmp, val, sem=None, scope=None) -> tensor:
        ...

    def atomic_xchg(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_add(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_max(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_min(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_and(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_or(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_xor(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def exp(self) -> tensor:
        ...

    def log(self) -> tensor:
        ...

    def cos(self) -> tensor:
        ...

    def sin(self) -> tensor:
        ...

    def sqrt(self) -> tensor:
        ...

    def rsqrt(self) -> tensor:
        ...

    def abs(self) -> tensor:
        ...

    def reduce(self, axis, combine_fn, keep_dims=False) -> tensor:
        ...

    def associative_scan(self, axis, combine_fn, reverse=False) -> tensor:
        ...

    def gather(self, indices, axis) -> tensor:
        ...

    def histogram(self, num_bins) -> tensor:
        ...

    def cdiv(self, div) -> tensor:
        ...

    def sigmoid(self) -> tensor:
        ...

    def softmax(self, dim=None, keep_dims=False, ieee_rounding=False) -> tensor:
        ...

    def ravel(self) -> tensor:
        ...

    def max(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmax(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def min(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmin(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def sum(self, axis=None, keep_dims=False, dtype=None) -> tensor:
        ...

    def xor_sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def reduce_or(self, axis=None, keep_dims=False) -> tensor:
        ...

    def cumsum(self, axis=0, reverse=False) -> tensor:
        ...

    def cumprod(self, axis=0, reverse=False) -> tensor:
        ...

    def sort(self, dim: constexpr = None, descending: constexpr = CONSTEXPR_0) -> tensor:
        ...

    def flip(self, dim=None) -> tensor:
        ...



@builtin
def program_id(axis, _semantic=None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = _semantic.program_id(0)
    #     pid1 = _semantic.program_id(1)
    #     pid2 = _semantic.program_id(2)
    #     npg0 = _semantic.num_programs(0)
    #     npg1 = _semantic.num_programs(1)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = _unwrap_if_constexpr(axis)
    return _semantic.program_id(axis)



# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _semantic=None):
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    return _semantic.arange(start, end)


arange.__doc__ = f"""
    Returns contiguous values within the half-open interval :code:`[start,
    end)`.  :code:`end - start` must be less than or equal to
    :code:`TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}`

    :param start: Start of the interval. Must be a power of two.
    :type start: int32
    :param end: End of the interval. Must be a power of two greater than
        :code:`start`.
    :type end: int32
"""


def _unwrap_shape(shape):
    shape = _unwrap_if_constexpr(shape)
    return [_unwrap_if_constexpr(s) for s in shape]


def _shape_check_impl(shape):
    shape = _unwrap_shape(shape)
    validate_block_shape(shape)
    return shape


# -----------------------
# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # `mask` and `other` can be constexpr
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)
    padding_option = _unwrap_if_constexpr(padding_option)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    return _semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                          volatile)



@_tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None):
    """
    Store a tensor of data into memory locations defined by `pointer`.

        (1) If `pointer` is a single element pointer, a scalar is stored.  In
            this case:

            - `mask` must also be scalar, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional block is stored.  In this case:

            - `mask` is implicitly broadcast to `pointer.shape`, and
            - `boundary_check` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a block
            of data is stored.  In this case:

            - `mask` must be None, and
            - `boundary_check` can be specified to control the behavior of out-of-bound access.

    `value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_ty`.

    :param pointer: The memory location where the elements of `value` are stored
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param value: The tensor of elements to be stored
    :type value: Block
    :param mask: If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`
    :type mask: Block of triton.int1, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for
        cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt"
        stands for cache write-through, see `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional, should be one of {"", "evict_first", "evict_last"}
    """
    # `value` can be constexpr
    value = _semantic.to_tensor(value)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    return _semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy)


@builtin
def add(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.add(x, y, sanitize_overflow)


@builtin
def sub(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.sub(x, y, sanitize_overflow)


@builtin
def mul(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.mul(x, y, sanitize_overflow)


@builtin
def static_print(*values, sep: str = " ", end: str = "\n", file=None, flush=False, _semantic=None):
    '''
    Print the values at compile time.  The parameters are the same as the builtin :code:`print`.

    NOTE: Calling the Python builtin :code:`print` is not the same as calling this, it instead maps to :code:`device_print`,
    which has special requirements for the arguments.

    .. highlight:: python
    .. code-block:: python

        tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
    '''
    pass


@builtin
def static_assert(cond, msg="", _semantic=None):
    '''
    Assert the condition at compile time.  Does not require that the :code:`TRITON_DEBUG` environment variable
    is set.

    .. highlight:: python
    .. code-block:: python

        tl.static_assert(BLOCK_SIZE == 1024)
    '''
    pass


