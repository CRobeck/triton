#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "ir.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_proton(py::module &&m) {
  auto passes = m.def_submodule("passes");

  py::class_<TritonOpBuilder>(m, "builder", py::module_local(),
                              py::dynamic_attr())
      .def(py::init<MLIRContext *>());

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
