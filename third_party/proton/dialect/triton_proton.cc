#include "Dialect/Proton/IR/Dialect.h"
#include "TritonProtonToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_amd_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_2("add_allocate_proton_smem_buffer",
                     mlir::triton::proton::createAllocateProtonSMEMBufferPass,
                     const std::string &, int32_t);
}


void init_triton_proton(py::module &&m) {
  auto passes = m.def_submodule("passes");

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
