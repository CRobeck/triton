#ifndef CUSTOM_DIALECT
#define CUSTOM_DIALECT

include "mlir/IR/OpBase.td"

def Custom_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::mlir::triton::custom";
}

#endif
