#ifndef PLUGIN_OPS
#define PLUGIN_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PluginDialect.td"

class Plugin_Op<string mnemonic, list<Trait> traits = []> :
  Op<Plugin_Dialect, mnemonic, !listconcat(traits, [])> {
}

def Plugin_AddFOp : Plugin_Op<"addf", [Commutative]> {
}

#endif // PLUGIN_OPS
