#include <iostream>
#include "Session/Session.h"

namespace neutron {

void Session::activate() {
  return;
}

void Session::deactivate() {
  return;
}

void SessionManager::activateSession(size_t sessionId) {
  std::cout << "Activated Neutron Session" << std::endl;
  activateSessionImpl(sessionId);
}

void SessionManager::deactivateSession(size_t sessionId) {
  std::cout << "Deactivated Neutron Session" << std::endl;  
  deActivateSessionImpl(sessionId);
}

void SessionManager::activateSessionImpl(size_t sessionId) {
    return;
}

void SessionManager::deActivateSessionImpl(size_t sessionId) {
    return;
}

} // namespace neutron
