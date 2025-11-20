set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Plugins)
add_public_tablegen_target(TritonPluginsIncGen)

set(LLVM_TARGET_DEFINITIONS PluginOps.td)
mlir_tablegen(PluginOps.h.inc -gen-op-decls)
mlir_tablegen(PluginOps.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=plugin)
add_public_tablegen_target(PluginTableGen)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

set(TRITON_PLUGIN_PASSES
    TritonPluginsTestLib
    )

set(TritonPluginsTestLib_SOURCES
    TritonPlugin.cpp
    PluginOps.cpp
    )


foreach( plugin ${TRITON_PLUGIN_PASSES} )
    add_mlir_library(${plugin}
        ${${plugin}_SOURCES}
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_BINARY_DIR}/lib

        DEPENDS
        TritonTableGen
        TritonCanonicalizeIncGen
        TritonPluginsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFunctionInterfaces
        MLIRFuncDialect
        MLIRPass
        LLVMSupport
        MLIRSupport
        TritonGPUIR
        TritonNvidiaGPUIR
        TritonInstrumentIR
        MLIROptLib
        MLIRPass
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRTransforms
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
        )

    set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
    include_directories(${CMAKE_CURRENT_BINARY_DIR})


    target_compile_options(${plugin} PRIVATE -fvisibility=hidden)
endforeach()
