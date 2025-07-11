#include "Analysis/AMDGPUAllocation.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
#define GEN_PASS_DEF_OPTIMIZEATOMICS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct OptimizeAtomics
    : public mlir::triton::impl::OptimizeAtomicsBase<OptimizeAtomics> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    mod.walk([&](triton::AtomicRMWOp op) {
      if (isa_and_nonnull<triton::AtomicRMWOp>(op) &&
          !isa_and_nonnull<triton::AtomicRMWOp>(op->getPrevNode()) &&
          !isa_and_nonnull<triton::AtomicRMWOp>(op->getNextNode()) &&
          op.getSem() == triton::MemSemantic::ACQUIRE_RELEASE) {
        op.setSem(triton::MemSemantic::RELAXED);
        OpBuilder rewriter(op);
        GCNBuilder gcnBuilder;
        rewriter.setInsertionPoint(op);
        gcnBuilder.create<>("buffer_wbl2 sc1")->operator()();
        auto buffer_wbl2 =
            gcnBuilder.launch(rewriter, op->getLoc(), void_ty(ctx));
        buffer_wbl2->moveBefore(op);

        GCNBuilder bufferinvBuilder;
        rewriter.setInsertionPointAfter(op);
        bufferinvBuilder.create<>("s_waitcnt vmcnt(0)")->operator()();
        bufferinvBuilder.create<>("buffer_inv sc1")->operator()();
        bufferinvBuilder.launch(rewriter, op->getLoc(), void_ty(ctx));
      }
    });
  };
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>> createOptimizeAtomicsPass() {
  return std::make_unique<OptimizeAtomics>();
}

} // namespace mlir::triton::AMD
