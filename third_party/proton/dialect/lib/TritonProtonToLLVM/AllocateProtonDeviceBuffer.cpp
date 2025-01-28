#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATEPROTONDEVICEBUFFER
#include "../third_party/proton/dialect/include/TritonProtonToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
struct AllocateProtonDeviceBuffer
    : public mlir::triton::impl::AllocateProtonDeviceBufferBase<
          AllocateProtonDeviceBuffer> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);

    OpBuilder b(mod.getBodyRegion());
    MLIRContext *context = &getContext();
    auto loc = mod.getLoc();

    bool hasProtonRecordOp = false;
    mod.walk([&](FunctionOpInterface funcOp) {
      funcOp.walk(
          [&](mlir::triton::proton::RecordOp op) { hasProtonRecordOp = true; });
    });
    if (hasProtonRecordOp) {
      FuncOp func = *mod.getOps<triton::FuncOp>().begin();
      b.setInsertionPointToStart(&func.getBody().front());
      // For now just hard code the device buffer size we want to use.
      int deviceBufferSizeInBytes = 1024;
      auto ptrTy = PointerType::get(IntegerType::get(context, 8), 1);
      auto buffer = b.create<triton::proton::BufferAllocOp>(
          loc, ptrTy, deviceBufferSizeInBytes);
    }
  }
};

} // namespace

namespace mlir {

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateProtonDeviceBuffer() {
  return std::make_unique<AllocateProtonDeviceBuffer>();
}

} // namespace triton

} // namespace mlir
