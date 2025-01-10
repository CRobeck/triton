add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    AllocateSMEMBuffer.cpp

    LINK_LIBS PUBLIC
    ProtonIR
)
