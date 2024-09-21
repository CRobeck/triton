#ifndef NEUTRON_UTILITY_TRAITS_H_
#define NEUTRON_UTILITY_TRAITS_H_

#include <type_traits>
#include <variant>

namespace neutron {
template <class T, class... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};
} // namespace neutron

#endif
