from __future__ import annotations  # remove after python 3.11
import builtins
import warnings

from typing import List, Optional, Sequence, Tuple, TypeVar, Generic, Type
import numbers

from triton.runtime import driver

from .._C.libtriton import ir
from . import core as tl

T = TypeVar('T')
TensorTy = TypeVar('TensorTy')


class IncompatibleTypeErrorImpl(Exception):

    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = "invalid operands of type " + self.type_a.__repr__() + " and " + self.type_b.__repr__()
        super(IncompatibleTypeErrorImpl, self).__init__(self.message)


class TritonSemantic(Generic[TensorTy]):
    tensor: Type[TensorTy] = tl.tensor
    lang = tl

    builder: ir.builder

    def __init__(self, builder):
        self.builder = builder

    def device_assert(self, cond, msg, file_line):
        # Triton Nano: Stub - assertions disabled for minimal build
        pass

# ===----------------------------------------------------------------------===##
# Programming Model
# ===----------------------------------------------------------------------===##

    def program_id(self, axis: int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
        return self.tensor(self.builder.create_get_program_id(axis), tl.int32)

# ===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
# ===----------------------------------------------------------------------===//

    def integer_promote_impl(self, a_ty: tl.dtype, b_ty: tl.dtype) -> tl.dtype:
        a_rank = a_ty.int_bitwidth
        b_rank = b_ty.int_bitwidth
        a_sn = a_ty.int_signedness
        b_sn = b_ty.int_signedness
        # Rules for signedness taken from "Usual arithmetic conversions" on
        # https://en.cppreference.com/w/c/language/conversion.
        if a_sn == b_sn:
            return a_ty if a_rank > b_rank else b_ty
        elif a_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
            return a_ty if a_rank >= b_rank else b_ty
        elif b_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
            return b_ty if b_rank >= a_rank else a_ty
        raise TypeError(f"unexpected signedness {a_sn} and {b_sn}")

    def computation_type_impl(self, a_ty: tl.dtype, a_is_scalar: bool, b_ty: tl.dtype, b_is_scalar: bool,
                              div_or_mod: bool) -> tl.dtype:
        # 0) For scalars we follow semantics similar to PyTorch, namely:
        # - If the scalar is of a lower or equal kind (bool < uint < int < fp),
        #   it doesn't participate in the promotion
        if a_is_scalar != b_is_scalar:
            scalar_ty, tensor_ty = (a_ty, b_ty) if a_is_scalar else (b_ty, a_ty)
            if scalar_ty.kind().value <= tensor_ty.kind().value:
                # Upcast because of 3) and 4) below!
                if div_or_mod and (tensor_ty in (tl.float16, tl.bfloat16)):
                    return tl.float32
                return tensor_ty

        # 1) if one operand is double, the other is implicitly
        #    converted to double
        if a_ty.is_fp64() or b_ty.is_fp64():
            return tl.float64
        # 2) if one operand is float, the other is implicitly
        #    converted to float
        if a_ty.is_fp32() or b_ty.is_fp32():
            return tl.float32
        # 3 ) if one operand is half, the other is implicitly converted to half
        #     unless we're doing / or %, which do not exist natively in PTX for fp16.
        #     Supported PTX op: add, sub, mul, fma, neg, abs, min, max, tanh, ex2, setp
        if a_ty.is_fp16() or b_ty.is_fp16():
            if div_or_mod:
                return tl.float32
            else:
                return tl.float16
        # 4) return bf16 only if both operands are of bf16
        if a_ty.is_bf16() and b_ty.is_bf16():
            if div_or_mod:
                return tl.float32
            else:
                return tl.bfloat16
        if a_ty.is_bf16() or b_ty.is_bf16():
            return tl.float32
        # 5) return fp16 if operands are different fp8
        if a_ty.is_fp8() and b_ty.is_fp8():
            return a_ty if a_ty == b_ty else tl.float16
        if not a_ty.is_int() or not b_ty.is_int():
            raise TypeError(f"unexpected type {a_ty} and {b_ty}")
        # 6 ) both operands are integer and undergo
        #    integer promotion
        if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
            raise TypeError("Cannot use /, #, or % with " + a_ty.__repr__() + " and " + b_ty.__repr__() +
                            " because they have different signedness;"
                            "this is unlikely to result in a useful answer. Cast them to the same signedness.")
        return self.integer_promote_impl(a_ty, b_ty)

    def to_tensor(self, x, check_type=True):
        if isinstance(x, self.tensor):
            return x
        x = x.value if isinstance(x, tl.constexpr) else x
        if isinstance(x, (int, float, bool)):
            dtype = self.to_tensor_type(x)
            return self.scalar_constant(x, dtype=dtype)
        elif check_type:
            raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")
        return x

    def to_tensor_type(self, x):
        if isinstance(x, tl.dtype):
            return x
        elif isinstance(x, tl.constexpr_type):
            x = x.value

        if isinstance(x, bool):
            return tl.int1
        elif isinstance(x, int):
            if -2**31 <= x < 2**31:
                return tl.int32
            elif 2**31 <= x < 2**32:
                return tl.uint32
            elif -2**63 <= x < 2**63:
                return tl.int64
            elif 2**63 <= x < 2**64:
                return tl.uint64
            raise ValueError(f'Nonrepresentable integer {x}.')
        elif isinstance(x, float):
            min_float32 = 2**-126
            max_float32 = (2 - 2**-23) * 2**127
            abs_x = builtins.abs(x)
            if abs_x == float("inf") or\
               abs_x == 0.0 or \
               x != x or \
               min_float32 <= abs_x <= max_float32:
                return tl.float32
            else:
                return tl.float64
        raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")

# ===----------------------------------------------------------------------===//
#                               Binary Operators
# ===----------------------------------------------------------------------===//

    def check_ptr_type_impl(self, type_a: tl.dtype, type_b: tl.dtype, allow_ptr_a: bool) -> None:
        if type_a.is_ptr():
            if not allow_ptr_a:
                raise IncompatibleTypeErrorImpl(type_a, type_b)
            # T* + U* with T != U
            if type_b.is_ptr() and (type_a != type_b):
                raise IncompatibleTypeErrorImpl(type_a, type_b)
            # T* + float
            if type_b.is_floating():
                raise IncompatibleTypeErrorImpl(type_a, type_b)

    def binary_op_type_checking_impl(self, lhs: TensorTy | numbers.Number, rhs: TensorTy | numbers.Number,
                                     allow_lhs_ptr=False, allow_rhs_ptr=False, arithmetic_check=True,
                                     div_or_mod=False) -> Tuple[TensorTy, TensorTy]:
        lhs_is_scalar = isinstance(lhs, numbers.Number)
        rhs_is_scalar = isinstance(rhs, numbers.Number)
        if lhs_is_scalar:
            lhs_scalar = lhs
            lhs = self.to_tensor(lhs)
        if rhs_is_scalar:
            rhs_scalar = rhs
            rhs = self.to_tensor(rhs)

        # implicit typecasting
        lhs_sca_ty = lhs.type.scalar
        rhs_sca_ty = rhs.type.scalar
        self.check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
        self.check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
        if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
            ret_sca_ty = self.computation_type_impl(lhs_sca_ty, lhs_is_scalar, rhs_sca_ty, rhs_is_scalar, div_or_mod)
            if (lhs_is_scalar and lhs_scalar < 0 and ret_sca_ty.is_int_unsigned()
                    or rhs_is_scalar and rhs_scalar < 0 and ret_sca_ty.is_int_unsigned()):
                raise ValueError("Cannot perform a binary operation between an unsigned tensor and a negative scalar. "
                                 "Perform a explicit cast on one of them.")
            if ret_sca_ty.is_int():
                if lhs_is_scalar and not (ret_sca_ty.get_int_min_value() <= lhs_scalar <=
                                          ret_sca_ty.get_int_max_value()):
                    raise ValueError(f"Scalar {lhs_scalar} is out of range for type {ret_sca_ty}")
                if rhs_is_scalar and not (ret_sca_ty.get_int_min_value() <= rhs_scalar <=
                                          ret_sca_ty.get_int_max_value()):
                    raise ValueError(f"Scalar {rhs_scalar} is out of range for type {ret_sca_ty}")
            lhs = self.scalar_constant(lhs_scalar, dtype=ret_sca_ty) if lhs_is_scalar else self.cast(lhs, ret_sca_ty)
            rhs = self.scalar_constant(rhs_scalar, dtype=ret_sca_ty) if rhs_is_scalar else self.cast(rhs, ret_sca_ty)

        # implicit broadcasting
        lhs, rhs = self.broadcast_impl_value(lhs, rhs)
        return lhs, rhs

    def binary_op_sanitize_overflow_impl(self, lhs: TensorTy, rhs: TensorTy, binary_op: callable):
        if lhs.type.scalar.int_bitwidth >= 64 or not self.builder.options.sanitize_overflow:
            return
        lhs_sca_ty = lhs.type.scalar
        rhs_sca_ty = rhs.type.scalar
        assert lhs_sca_ty == rhs_sca_ty
        assert lhs_sca_ty.is_int()
        lhs = self.cast(lhs, tl.int64)
        rhs = self.cast(rhs, tl.int64)
        ret = binary_op(lhs, rhs, False)
        max_value = lhs_sca_ty.get_int_max_value()
        max_value = self.scalar_constant(max_value, tl.int64)
        min_value = lhs_sca_ty.get_int_min_value()
        min_value = self.scalar_constant(min_value, tl.int64)
        cond = self.and_(self.less_equal(ret, max_value), self.greater_equal(ret, min_value))
        msg = f"int{lhs_sca_ty.int_bitwidth} overflow detected for operation {binary_op.__name__}"
        self.device_assert(cond, msg, None)

    def add(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number,
            sanitize_overflow: bool) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, True, True)
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        if input_scalar_ty.is_ptr() and other_scalar_ty.is_ptr():
            raise TypeError("cannot add pointers together")

        # offset + ptr
        # ptr + offset
        if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
            input, other = other, input
            input_scalar_ty = input.type.scalar
            other_scalar_ty = other.type.scalar
        if input_scalar_ty.is_ptr():
            other_handle = other.handle
            if other.dtype.is_int_unsigned() and other.dtype.int_bitwidth < 64:
                # addptr treats offset as signed. Zero-extend unsigned offsets to ensure they're positive
                i64_ty = other.type.with_element_ty(tl.int64).to_ir(self.builder)
                other_handle = self.builder.create_int_cast(other.handle, i64_ty, False)
            return self.tensor(self.builder.create_addptr(input.handle, other_handle), input.type)
        # float + float
        elif input_scalar_ty.is_floating():
            return self.tensor(self.builder.create_fadd(input.handle, other.handle), input.type)
        # int + int
        elif input_scalar_ty.is_int():
            if sanitize_overflow:
                self.binary_op_sanitize_overflow_impl(input, other, self.add)
            return self.tensor(self.builder.create_add(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {input_scalar_ty}")

    def sub(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number,
            sanitize_overflow: bool) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, True, False)
        scalar_ty = input.type.scalar
        # ptr - offset
        if scalar_ty.is_ptr():
            return self.add(input, self.minus(other), sanitize_overflow=False)
        # float - float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fsub(input.handle, other.handle), input.type)
        # int - int
        elif scalar_ty.is_int():
            if sanitize_overflow:
                self.binary_op_sanitize_overflow_impl(input, other, self.sub)
            return self.tensor(self.builder.create_sub(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

    def mul(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number,
            sanitize_overflow: bool) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float * float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fmul(input.handle, other.handle), input.type)
        # int * int
        elif scalar_ty.is_int():
            if sanitize_overflow:
                self.binary_op_sanitize_overflow_impl(input, other, self.mul)
            return self.tensor(self.builder.create_mul(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

    def truediv(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        # float / int
        if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
            other = self.cast(other, input_scalar_ty)
        # int / float
        elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
            input = self.cast(input, other_scalar_ty)
        # int / int (cast to tl.float32)
        elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
            input = self.cast(input, tl.float32)
            other = self.cast(other, tl.float32)
        # float / float (cast to the highest exponent type)
        elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
            if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
                other = self.cast(other, input_scalar_ty)
            else:
                input = self.cast(input, other_scalar_ty)
        # unreachable
        else:
            raise TypeError(f"unexpected type {input_scalar_ty}")
        return self.tensor(self.builder.create_fdiv(input.handle, other.handle), input.type)

    def floordiv(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        if input_scalar_ty.is_int() and other_scalar_ty.is_int():
            ret_ty = self.integer_promote_impl(input_scalar_ty, other_scalar_ty)
            input = self.cast(input, ret_ty)
            other = self.cast(other, ret_ty)
            if ret_ty.is_int_signed():
                return self.tensor(self.builder.create_sdiv(input.handle, other.handle), input.type)
            else:
                return self.tensor(self.builder.create_udiv(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {input_scalar_ty}")

    def fdiv(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number, ieee_rounding: bool) -> TensorTy:
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
            raise TypeError("both operands of fdiv must have floating scalar type")
        input, other = self.binary_op_type_checking_impl(input, other, False, False, False, True)
        ret = self.builder.create_fdiv(input.handle, other.handle)
        return self.tensor(ret, input.type)

    def mod(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        # float % float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_frem(input.handle, other.handle), input.type)
        # % int
        elif scalar_ty.is_int():
            if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
                raise TypeError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                                "because they have different signedness;"
                                "this is unlikely to result in a useful answer. Cast them to the same signedness.")
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_srem(input.handle, other.handle), input.type)
            else:
                return self.tensor(self.builder.create_urem(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

##############
# other arithmetic ops
##############

##############
# bitwise ops
##############

    def bitwise_op_type_checking_impl(self, input: TensorTy, other: TensorTy) -> Tuple[TensorTy, TensorTy]:
        input, other = self.binary_op_type_checking_impl(input, other)
        input_sca_ty = input.type.scalar
        other_sca_ty = other.type.scalar
        if not input_sca_ty.is_int() or not other_sca_ty.is_int():
            raise IncompatibleTypeErrorImpl(input_sca_ty, other_sca_ty)
        ret_sca_ty = self.integer_promote_impl(input_sca_ty, other_sca_ty)
        if ret_sca_ty != input_sca_ty:
            input = self.cast(input, ret_sca_ty)
        if ret_sca_ty != other_sca_ty:
            other = self.cast(other, ret_sca_ty)
        return input, other

    def and_(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_and(input.handle, other.handle), input.type)

    def or_(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_or(input.handle, other.handle), input.type)

    def xor_(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_xor(input.handle, other.handle), input.type)

    def logical_and(self, input: TensorTy, other: TensorTy) -> TensorTy:
        if not input.type.is_int1():
            input = self.bitcast(input, tl.int1)
        if not other.type.is_int1():
            other = self.bitcast(other, tl.int1)
        return self.and_(input, other)

    def logical_or(self, input: TensorTy, other: TensorTy) -> TensorTy:
        if not input.type.is_int1():
            input = self.bitcast(input, tl.int1)
        if not other.type.is_int1():
            other = self.bitcast(other, tl.int1)
        return self.or_(input, other)

    def not_(self, input: TensorTy):
        if not input.type.is_int1():
            input = self.bitcast(input, tl.int1)
        return self.invert(input)

    def lshr(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_lshr(input.handle, other.handle), input.type)

    def ashr(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_ashr(input.handle, other.handle), input.type)

    def shl(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_shl(input.handle, other.handle), input.type)

# ===----------------------------------------------------------------------===//
#                               Unary Operators
# ===----------------------------------------------------------------------===//

    def plus(self, input: TensorTy) -> TensorTy:
        return input

    def minus(self, input: TensorTy) -> TensorTy:
        input_sca_ty = input.type.scalar
        if input_sca_ty.is_ptr():
            raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
        _0 = self.tensor(self.builder.get_null_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.sub(_0, input, True)

    def invert(self, input: TensorTy) -> TensorTy:
        input_sca_ty = input.type.scalar
        if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
            raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
        _1 = self.tensor(self.builder.get_all_ones_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.xor_(input, _1)

# ===----------------------------------------------------------------------===//
#                               Comparison Operators
# ===----------------------------------------------------------------------===//

    def _bool_like(self, v: TensorTy) -> tl.block_type:
        return v.type.with_element_ty(tl.int1)

    def greater_than(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float > float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOGT(input.handle, other.handle), self._bool_like(input))
        # > int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSGT(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpUGT(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def greater_equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float >= float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOGE(input.handle, other.handle), self._bool_like(input))
        # >= int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSGE(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpUGE(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def less_than(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float < float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOLT(input.handle, other.handle), self._bool_like(input))
        # < int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSLT(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpULT(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def less_equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float < float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOLE(input.handle, other.handle), self._bool_like(input))
        # < int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSLE(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpULE(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float == float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOEQ(input.handle, other.handle), self._bool_like(input))
        # == int
        elif scalar_ty.is_int():
            return self.tensor(self.builder.create_icmpEQ(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def not_equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float == float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpUNE(input.handle, other.handle), self._bool_like(input))
        # == int
        elif scalar_ty.is_int():
            return self.tensor(self.builder.create_icmpNE(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

# ===----------------------------------------------------------------------===//
#                               Block Creation
# ===----------------------------------------------------------------------===//

    def arange(self, start: int, end: int, *, ret_ty: tl.block_type = None) -> TensorTy:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        is_start_int64 = bool(start >> 32)
        is_end_int64 = bool(end >> 32)
        if is_start_int64 or is_end_int64:
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError("arange's end argument must be greater than the start argument")
        range = end - start
        if (range & (range - 1)) != 0:
            raise ValueError("arange's range must be a power of 2")
        shape = [range]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)

    def scalar_constant(self, value, dtype: tl.dtype) -> TensorTy:
        # scalar
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = self.builder.get_null_value(dtype.to_ir(self.builder))
        elif dtype.is_fp8():
            value = self.builder.get_fp32(value)
            value = self.builder.create_fp_trunc(value, dtype.to_ir(self.builder))
        else:
            get_value_fn = getattr(self.builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        return self.tensor(value, dtype)

    def make_scalar(self, value, dtype: tl.dtype) -> TensorTy:
        if isinstance(value, tl.tensor):
            assert value.numel.value == 1, "only accepts size-1 tensor"
            return self.cast(value, dtype)
        # scalar
        return self.scalar_constant(value, dtype)

    def full(self, shape: List[int], value, dtype: tl.dtype) -> TensorTy:
        return self.splat(self.make_scalar(value, dtype), shape)

# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//

    def splat(self, value: TensorTy, shape: List[int]) -> TensorTy:
        assert not value.type.is_block(), "Cannot splat a block tensor"
        if len(shape) == 0:
            return value
        ret_ty = tl.block_type(value.dtype, shape)
        return self.tensor(self.builder.create_splat(ret_ty.to_ir(self.builder), value.handle), ret_ty)

    def unsplat(self, value: TensorTy) -> TensorTy:
        return self.tensor(self.builder.create_unsplat(value.handle), value.dtype)

    def broadcast_impl_shape(self, input: TensorTy, shape: Tuple[int]) -> TensorTy:
        if not input.type.is_block():
            return self.splat(input, shape)
        src_shape = input.type.get_block_shapes()
        if len(src_shape) != len(shape):
            raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
        if shape == src_shape:
            return input
        for i, item in enumerate(src_shape):
            if shape[i] != item and item != 1:
                raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                                 f" must match the existing size ({item}) at non-singleton dimension"
                                 f" {i}: {src_shape}, {shape}")
        ret_ty = tl.block_type(input.type.scalar, shape)
        return self.tensor(self.builder.create_broadcast(input.handle, shape), ret_ty)

    def broadcast_impl_value(self, lhs: TensorTy, rhs: TensorTy) -> TensorTy:
        lhs_ty = lhs.type
        rhs_ty = rhs.type

        # make_shape_compatible(block, scalar)
        if lhs_ty.is_block() and not rhs_ty.is_block():
            rhs_ty = lhs_ty.with_element_ty(rhs_ty.scalar)
            rhs = self.tensor(self.builder.create_splat(rhs_ty.to_ir(self.builder), rhs.handle), rhs_ty)
        # make_shape_compatible(scalar, block)
        elif not lhs_ty.is_block() and rhs_ty.is_block():
            lhs_ty = rhs_ty.with_element_ty(lhs_ty.scalar)
            lhs = self.tensor(self.builder.create_splat(lhs_ty.to_ir(self.builder), lhs.handle), lhs_ty)
        # make_shape_compatible(block, block)
        elif lhs_ty.is_block() and rhs_ty.is_block():
            lhs_shape = lhs_ty.get_block_shapes()
            rhs_shape = rhs_ty.get_block_shapes()

            if len(lhs_shape) < len(rhs_shape):
                # Add new axes to lhs
                for _ in range(len(lhs_shape), len(rhs_shape)):
                    lhs = self.tensor(self.builder.create_expand_dims(lhs.handle, 0),
                                      tl.block_type(lhs_ty.scalar, [1] + lhs_shape.values))
                    lhs_ty = lhs.type
                    lhs_shape = lhs_ty.get_block_shapes()
            elif len(rhs_shape) < len(lhs_shape):
                # Add new axes to rhs
                for _ in range(len(rhs_shape), len(lhs_shape)):
                    rhs = self.tensor(self.builder.create_expand_dims(rhs.handle, 0),
                                      tl.block_type(rhs_ty.scalar, [1] + rhs_shape.values))
                    rhs_ty = rhs.type
                    rhs_shape = rhs_ty.get_block_shapes()
            assert len(rhs_shape) == len(lhs_shape)

            ret_shape = []
            for i, left in enumerate(lhs_shape):
                right = rhs_shape[i]
                if left == 1:
                    ret_shape.append(right)
                elif (right == 1) or (right == left):
                    ret_shape.append(left)
                else:
                    raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                     "at index " + str(i) + ": " + str(left) + " and " + str(right))
            if lhs_shape != ret_shape:
                ret_ty = tl.block_type(lhs_ty.scalar, ret_shape)
                lhs = self.tensor(self.builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
            if rhs_shape != ret_shape:
                ret_ty = tl.block_type(rhs_ty.scalar, ret_shape)
                rhs = self.tensor(self.builder.create_broadcast(rhs.handle, ret_shape), ret_ty)
        # (scalar, scalar) => returns original blocks
        return lhs, rhs

    def broadcast_tensors(self, *tensors: TensorTy) -> list:
        """Broadcast tensors to compatible shapes."""
        if len(tensors) == 2:
            return list(self.broadcast_impl_value(tensors[0], tensors[1]))
        elif len(tensors) == 3:
            # Broadcast first two, then the third
            a, b = self.broadcast_impl_value(tensors[0], tensors[1])
            a, c = self.broadcast_impl_value(a, tensors[2])
            b, c = self.broadcast_impl_value(b, c)
            return [a, b, c]
        else:
            raise ValueError(f"broadcast_tensors expects 2 or 3 tensors, got {len(tensors)}")

#######
# cast
#######

    def _str_to_rounding_mode(self, rounding_mode: Optional[str]):
        if rounding_mode is None:
            return None
        if rounding_mode == 'rtne':
            return ir.ROUNDING_MODE.RTNE
        if rounding_mode == 'rtz':
            return ir.ROUNDING_MODE.RTZ
        raise ValueError(f"Invalid rounding mode: {rounding_mode}. Supported rounding modes are 'rtne' and 'rtz'.")

    def bitcast(self, input: TensorTy, dst_ty: tl.dtype) -> TensorTy:
        src_ty = input.type
        if src_ty.is_block():
            dst_ty = src_ty.with_element_ty(dst_ty.scalar)
        if src_ty == dst_ty:
            return input
        src_sca_ty = src_ty.scalar
        dst_sca_ty = dst_ty.scalar
        if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
            return self.cast(input, dst_ty)
        # Bitcast
        src_bits = src_sca_ty.primitive_bitwidth
        dst_bits = dst_sca_ty.primitive_bitwidth
        if src_bits != dst_bits:
            raise ValueError("Cannot bitcast data-type of size " + str(src_bits) + " to "
                             "data-type of size " + str(dst_bits))
        return self.tensor(self.builder.create_bitcast(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

    def cast(self, input: TensorTy, dst_ty: tl.dtype, fp_downcast_rounding: Optional[str] = None) -> TensorTy:
        src_ty = input.type
        src_sca_ty = src_ty.scalar
        dst_sca_ty = dst_ty.scalar
        if src_sca_ty == dst_sca_ty:
            return input
        if src_ty.is_block():
            dst_ty = src_ty.with_element_ty(dst_sca_ty)

        # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
        # not be set
        fp_downcast_rounding = self._str_to_rounding_mode(fp_downcast_rounding)
        use_custom_rounding = False
        if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
        ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
            if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
            elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
        else:
            if fp_downcast_rounding is not None:
                raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                                 "Source scalar type is " + str(src_sca_ty) + " and destination type is " +
                                 str(dst_sca_ty))

        if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
            assert self.builder.codegen_fns.get(
                "convert_custom_types") is not None, "target doesn't provide conversion for this type."
            return self.builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _semantic=self)
        # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
        # and non-default rounding modes for downcasting
        if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
           (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
           use_custom_rounding:
            return self.tensor(
                self.builder.create_fp_to_fp(input.handle, dst_ty.to_ir(self.builder), fp_downcast_rounding), dst_ty)

        # bf16 <=> (not fp32)
        if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
           (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
            return self.cast(self.cast(input, tl.float32), dst_sca_ty)

        # Standard floating types' casting: truncation
        #   fp64 => fp32, fp16, bf16
        #   fp32 => fp16, bf16
        truncate_fp = src_sca_ty.is_floating() and \
            dst_sca_ty.is_floating() and \
            src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
        if truncate_fp:
            return self.tensor(self.builder.create_fp_trunc(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Standard floating types' casting: extension
        #   fp32 => fp64
        #   fp16 => fp32, fp64
        #   bf16 => fp32, fp64
        ext_fp = src_sca_ty.is_floating() and \
            dst_sca_ty.is_floating() and \
            src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
        if ext_fp:
            return self.tensor(self.builder.create_fp_ext(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting between integer types
        if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
           (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
            sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
            if dst_sca_ty.is_bool():
                ty = input.dtype.to_ir(self.builder)
                _0 = self.tensor(self.builder.get_null_value(ty), input.dtype)
                return self.not_equal(input, _0)
            else:
                return self.tensor(self.builder.create_int_cast(input.handle, dst_ty.to_ir(self.builder), sign_extend),
                                   dst_ty)

        # Casting standard floating types to integer types
        if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
            if dst_sca_ty.is_bool():
                ty = input.dtype.to_ir(self.builder)
                _0 = self.tensor(self.builder.get_null_value(ty), input.dtype)
                return self.not_equal(input, _0)
            elif dst_sca_ty.is_int_signed():
                return self.tensor(self.builder.create_fp_to_si(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            else:
                return self.tensor(self.builder.create_fp_to_ui(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting integer types to standard floating types
        if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
            if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
                return self.tensor(self.builder.create_ui_to_fp(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            else:
                return self.tensor(self.builder.create_si_to_fp(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting pointer types to integer types
        if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
            bitwidth = dst_sca_ty.int_bitwidth
            if bitwidth == 64:
                return self.tensor(self.builder.create_ptr_to_int(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            if bitwidth == 1:
                return self.not_equal(self.cast(input, tl.int64), self.tensor(self.builder.get_int64(0), tl.int64))

        # Casting integer types to pointer types
        if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
            return self.tensor(self.builder.create_int_to_ptr(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting pointer types to pointer types
        if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
            return self.tensor(self.builder.create_bitcast(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        assert False, f'cannot cast {input} to {dst_ty}'

# ===----------------------------------------------------------------------===//
#                               Memory Operators
# ===----------------------------------------------------------------------===//

    def _str_to_load_cache_modifier(self, cache_modifier):
        cache = ir.CACHE_MODIFIER.NONE  # default
        if cache_modifier:
            if cache_modifier == ".ca":
                cache = ir.CACHE_MODIFIER.CA
            elif cache_modifier == ".cg":
                cache = ir.CACHE_MODIFIER.CG
            elif cache_modifier == ".cv":
                cache = ir.CACHE_MODIFIER.CV
            else:
                raise ValueError(f"Cache modifier {cache_modifier} not supported")
        return cache

    def _str_to_store_cache_modifier(self, cache_modifier):
        cache = ir.CACHE_MODIFIER.NONE  # default
        if cache_modifier:
            if cache_modifier == ".wb":
                cache = ir.CACHE_MODIFIER.WB
            elif cache_modifier == ".cg":
                cache = ir.CACHE_MODIFIER.CG
            elif cache_modifier == ".cs":
                cache = ir.CACHE_MODIFIER.CS
            elif cache_modifier == ".wt":
                cache = ir.CACHE_MODIFIER.WT
            else:
                raise ValueError(f"Cache modifier {cache_modifier} not supported")
        return cache

    def _str_to_eviction_policy(self, eviction_policy):
        eviction = ir.EVICTION_POLICY.NORMAL  # default
        if eviction_policy:
            if eviction_policy == "evict_last":
                eviction = ir.EVICTION_POLICY.EVICT_LAST
            elif eviction_policy == "evict_first":
                eviction = ir.EVICTION_POLICY.EVICT_FIRST
            else:
                raise ValueError(f"Eviction policy {eviction_policy} not supported")
        return eviction

    def _str_to_padding_option(self, padding_option):
        padding = None  # default
        if padding_option:
            if padding_option == "zero":
                padding = ir.PADDING_OPTION.PAD_ZERO
            elif padding_option == "nan":
                padding = ir.PADDING_OPTION.PAD_NAN
            else:
                raise ValueError(f"Padding option {padding_option} not supported")
        return padding

    def _str_to_sem(self, sem_option):
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

    def _str_to_scope(self, scope_option):
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

    def _load_legacy(self, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile):
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

        # Check `mask`, `other`, `boundary_check`, and `padding` arguments
        if mask is None and other is not None:
            raise ValueError("`other` cannot be provided without `mask`")
        if padding or boundary_check:
            raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                             "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                             "use block pointers (defined by `make_block_ptr`) instead")

        # For a pointer of scalar, check the type of `mask` and `other`
        if not ptr.type.is_block():
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
            if other and other.type.is_block():
                raise ValueError("Other argument cannot be block type if pointer argument is not a block")

        # Make `mask` and `other` into the same shape as `ptr`
        if ptr.type.is_block():
            if mask is not None:
                ptr, mask = self.broadcast_impl_value(ptr, mask)
            if other is not None:
                ptr, other = self.broadcast_impl_value(ptr, other)

        # Get `pointer_type<elt_ty>` and `elt_ty`
        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        is_bool = elt_ty == tl.int1
        if is_bool:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast `other` into `elt_ty` type
        if other is not None:
            other = self.cast(other, elt_ty)

        # Create loaded result type `dst_ty`
        if ptr.type.is_block():
            dst_ty = ptr.type.with_element_ty(elt_ty)
        else:
            # Load by de-referencing the pointer of scalar
            dst_ty = elt_ty

        # Build IR
        if mask is None:
            ret = self.tensor(self.builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
        else:
            ret = self.tensor(
                self.builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache,
                                                eviction, is_volatile), dst_ty)
        if is_bool:
            ret = self.cast(ret, tl.int1)
        return ret

    def load(self, ptr: TensorTy, mask: Optional[TensorTy], other: Optional[TensorTy], boundary_check: Tuple,
             padding_option: str, cache_modifier: str, eviction_policy: str, is_volatile: bool) -> TensorTy:
        # Triton Nano: Simplified load, only legacy pointers
        cache = self._str_to_load_cache_modifier(cache_modifier)
        eviction = self._str_to_eviction_policy(eviction_policy)
        padding = self._str_to_padding_option(padding_option)

        return self._load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile)

    def _store_legacy(self, ptr, val, mask, boundary_check, cache, eviction):
        # Store by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.store`")

        # Check `boundary_check` argument
        if boundary_check:
            raise ValueError("`boundary_check` argument is not supported for storing a tensor of pointers or storing a "
                             "scalar. Because the compiler does not know the boundary; please use block pointers "
                             "(defined by `make_block_ptr`) instead")

        # For a pointer of scalar, check the type of `val` and `mask`
        if not ptr.type.is_block():
            if val.type.is_block():
                raise ValueError("Value argument cannot be block type if pointer argument is not a block")
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")

        # Make `mask` and `val` into the same shape as `ptr`
        if ptr.type.is_block():
            ptr_shape = ptr.shape
            if mask is None:
                ptr, val = self.broadcast_tensors(ptr, val)
            else:
                ptr, val, mask = self.broadcast_tensors(ptr, val, mask)
            if ptr_shape != ptr.shape:
                raise ValueError(f"Expected pointer argument to have shape {ptr.shape} but got {ptr_shape}")

        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        if elt_ty == tl.int1:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast to target data type
        val = self.cast(val, elt_ty)

        # Build IR
        if mask is None:
            return self.tensor(self.builder.create_store(ptr.handle, val.handle, cache, eviction), tl.void)
        if not mask.type.scalar.is_bool():
            raise ValueError("Mask must have boolean scalar type")
        return self.tensor(self.builder.create_masked_store(ptr.handle, val.handle, mask.handle, cache, eviction),
                           tl.void)

    def store(self, ptr: TensorTy, val: TensorTy, mask: Optional[TensorTy], boundary_check, cache_modifier: str,
              eviction_policy: str) -> TensorTy:
        # Triton Nano: Simplified store, only legacy pointers
        cache = self._str_to_store_cache_modifier(cache_modifier)
        eviction = self._str_to_eviction_policy(eviction_policy)

        if ptr.type.is_const() or ptr.type.scalar.is_const():
            raise ValueError("Cannot store to a constant pointer")

        return self._store_legacy(ptr, val, mask, boundary_check, cache, eviction)

