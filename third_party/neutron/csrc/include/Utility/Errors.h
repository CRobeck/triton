#include <stdexcept>

namespace neutron {

class NotImplemented : public std::logic_error {
public:
  NotImplemented() : std::logic_error("Not yet implemented"){};
};

} // namespace neutron
