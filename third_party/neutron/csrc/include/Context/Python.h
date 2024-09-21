#ifndef NEUTRON_CONTEXT_PYTHON_H_
#define NEUTRON_CONTEXT_PYTHON_H_

#include "Context.h"

namespace neutron {

/// Unwind the Python stack and early return a list of contexts.
class PythonContextSource : public ContextSource {
public:
  std::vector<Context> getContexts() override;
};

} // namespace neutron

#endif // NEUTRON_CONTEXT_PYTHON_H_
