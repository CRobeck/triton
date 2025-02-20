add_triton_library(ProtonGPUToLLVM
    RecordOpToLLVM.cpp
    FuncOpToLLVM.cpp
    ProtonGPUToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)
