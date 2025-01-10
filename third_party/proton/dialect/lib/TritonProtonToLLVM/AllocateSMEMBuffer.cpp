#include "TargetInfo.h"
#include "TritonProtonToLLVM/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEPROTONSMEMBUFFER
#include "TritonProtonToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

class AllocateProtonSMEMBuffer
    : public mlir::triton::impl::AllocateProtonSMEMBufferBase<AllocateProtonSMEMBuffer> {

  int LDSLimit;

public:
  AllocateProtonSMEMBuffer(StringRef targetArch, int customLDSLimit)
      : AllocateProtonSMEMBufferBase<AllocateProtonSMEMBuffer>() {
    this->targetArch = targetArch.str();
    this->customLDSLimit = customLDSLimit;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

  }
};

} // namespace

namespace mlir::triton::proton {

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateProtonSMEMBufferPass(StringRef targetArch, int customLDSLimit) {
  return std::make_unique<AllocateProtonSMEMBuffer>(targetArch, customLDSLimit);
}

} // namespace mlir::triton::proton
