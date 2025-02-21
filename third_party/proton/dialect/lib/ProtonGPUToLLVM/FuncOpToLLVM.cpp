#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

namespace {

static void filterFuncAttributes(LLVM::LLVMFuncOp op, bool filterArgAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {

  for (const auto &attr : op->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == op.getFunctionTypeAttrName() ||
        attr.getName() == "std.varargs" ||
        (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
      continue;
    result.push_back(attr);
  }
}

struct FuncOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::RecordOp> {
  explicit FuncOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::RecordOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::RecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto m =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();	  
    int numFuncs = llvm::range_size(m.getOps<LLVM::LLVMFuncOp>());
    auto func = *m.getOps<LLVM::LLVMFuncOp>().begin();
    if(!op.getIsStart()){
    	rewriter.replaceOp(op, op);
    	return success();
    }
    	
    auto loc = func.getLoc();
    auto ctx = func->getContext();    
    auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto funcTy = func.getFunctionType();
    auto amendedInputTy = llvm::to_vector(func.getArgumentTypes());
    amendedInputTy.push_back(globalPtrTy);
    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, func.getResultTypes());
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(func, /*filterArgAttrs=*/true, amendedAttrs);
//    SmallVector<NamedAttribute> amendedAttrs;    
//    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    llvm::errs() << func << "!!!!\n";
    rewriter.replaceOp(op, op);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateFuncOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, targetInfo, benefit);
}
