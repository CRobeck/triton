#ifndef TRITONPROTON_CONVERSION_PASSES
#define TRITONPROTON_CONVERSION_PASSES
include "mlir/Pass/PassBase.td"
def AllocateProtonSMEMBuffer : Pass<"allocate-proton-smem-buffer", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::proton::createAllocateProtonSMEMBufferPass(\"\")";
    let options = [
        Option<"targetArch", "target-arch", "std::string", /*default*/"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"customLDSLimit", "lds-limit", "int", /*default*/"0",
               "custom limit of LDS consumption, if not provided, maximum LDS size is used">,
    ];
}
#endif
