#ifndef CUSTOM_OPS
#define CUSTOM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "CustomDialect.td"

class Custom_Op<string mnemonic, list<Trait> traits = []> :
  Op<Custom_Dialect, mnemonic, !listconcat(traits, [])> {
}

def Custom_AddFOp : Custom_Op<"addf", [Commutative]> {
}

#endif // CUSTOM_OPS
