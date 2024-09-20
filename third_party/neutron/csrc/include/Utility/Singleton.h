#ifndef NEUTRON_UTILITY_SINGLETON_H_
#define NEUTRON_UTILITY_SINGLETON_H_

namespace neutron {

template <typename T> class Singleton {
public:
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static T &instance() {
    static T _;
    return _;
  }

protected:
  Singleton() = default;
};

} // namespace neutron

#endif // NEUTRON_UTILITY_SINGLETON_H_
