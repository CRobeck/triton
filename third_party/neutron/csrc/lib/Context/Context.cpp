#include "Context/Context.h"

namespace neutron {

std::atomic<size_t> Scope::scopeIdCounter{1};

/*static*/ thread_local std::map<ThreadLocalOpInterface *, bool>
    ThreadLocalOpInterface::opInProgress;

} // namespace neutron
