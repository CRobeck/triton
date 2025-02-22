#ifndef PROTON_GPU_TO_LLVM_PATTERN_PROTON_OP_TO_LLVM_H
#define PROTON_GPU_TO_LLVM_PATTERN_PROTON_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton {
class TargetInfoBase;
namespace proton {

void populateRecordOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);

void populateGlobalScratchAllocOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);

void populateProtonOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit);

} // namespace proton
} // namespace mlir::triton

#endif // PROTON_TO_LLVM_PATTERN_PROTON_OP_TO_LLVM_H
