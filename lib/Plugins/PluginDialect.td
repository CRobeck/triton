#ifndef PLUGIN_DIALECT
#define PLUGIN_DIALECT

include "mlir/IR/OpBase.td"

def Plugin_Dialect : Dialect {
  let name = "plugin";
  let cppNamespace = "::mlir::triton::plugin";
}

#endif
