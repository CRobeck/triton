#ifndef TRITON_THIRD_PARTY_PROTON_INCLUDE_TRITONPROTONTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_PROTON_INCLUDE_TRITONPROTONTOLLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

//#define GEN_PASS_DECL
//#include "TritonProtonToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton::proton {

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateProtonSMEMBufferPass(StringRef arch, int32_t customLDSLimit = 0);
} // namespace mlir::triton::proton

#endif // TRITON_THIRD_PARTY_PROTON_INCLUDE_TRITONPROTONTOLLVM_PASSES_H_

