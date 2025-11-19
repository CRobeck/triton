#ifndef CUSTOM_DIALECT_H_
#define CUSTOM_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "CustomOps.h.inc"

#endif // CUSTOM_DIALECT_H_
