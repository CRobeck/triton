#include "OptimizeLDSUtility.h"
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
#define GEN_PASS_DEF_OPTIMIZEPROTONLDSUSAGE
#include "TritonProtonToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

class OptimizeProtonLDSUsage
    : public mlir::triton::impl::OptimizeProtonLDSUsageBase<OptimizeProtonLDSUsage> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    }
};

} // namespace

namespace mlir::triton::Proton {

std::unique_ptr<OperationPass<ModuleOp>>
createOptimizeLDSUsagePass(StringRef targetArch, int customLDSLimit) {
  return std::make_unique<OptimizeProtonLDSUsage>(targetArch, customLDSLimit);
}

} // namespace mlir::triton::Proton
