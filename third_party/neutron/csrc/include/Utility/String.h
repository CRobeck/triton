#ifndef NEUTRON_UTILITY_STRING_H_
#define NEUTRON_UTILITY_STRING_H_

#include <string>

namespace neutron {

inline std::string toLower(const std::string &str) {
  std::string lower;
  for (auto c : str) {
    lower += tolower(c);
  }
  return lower;
}

} // namespace neutron

#endif // NEUTRON_UTILITY_STRING_H_
