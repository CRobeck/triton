#ifndef NEUTRON_SESSION_H_
#define NEUTRON_SESSION_H_

#include "Utility/Singleton.h"

#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

namespace neutron {

class Session {
public:
  ~Session() = default;

  void activate();

  void deactivate();


private:
  Session(size_t id)
      : id(id) {}

  size_t id{};
  friend class SessionManager;
};

/// A session manager is responsible for managing the lifecycle of sessions.
/// There's a single and unique session manager in the system.
class SessionManager : public Singleton<SessionManager> {
public:
  SessionManager() = default;
  ~SessionManager() = default;

  void activateSession(size_t sesssionId);
  void deactivateSession(size_t sessionId);

private:
  void activateSessionImpl(size_t sesssionId);
  void deActivateSessionImpl(size_t sessionId);

};

} // namespace neutron

#endif // NEUTRON_SESSION_H_
