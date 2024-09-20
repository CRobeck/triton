#include "Neutron.h"

#include <map>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace neutron;

void initNeutron(pybind11::module &&m) {
  using ret = pybind11::return_value_policy;
  using namespace pybind11::literals;

  m.def("activate", [](size_t sessionId) {
    SessionManager::instance().activateSession(sessionId);
  });

  m.def("deactivate", [](size_t sessionId) {
    SessionManager::instance().deactivateSession(sessionId);
  });  

}

PYBIND11_MODULE(libneutron, m) {
  m.doc() = "Python bindings to the Neutron API";
  initNeutron(std::move(m.def_submodule("neutron")));
}
