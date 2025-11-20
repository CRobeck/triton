#ifndef PLUGIN_DIALECT_H_
#define PLUGIN_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "PluginOps.h.inc"

#endif // PLUGIN_DIALECT_H_
