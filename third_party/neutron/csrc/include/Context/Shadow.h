#ifndef NEUTRON_CONTEXT_SHADOW_H_
#define NEUTRON_CONTEXT_SHADOW_H_

#include "Context.h"
#include <vector>

namespace neutron {

/// Incrementally build a list of contexts by shadowing the stack with
/// user-defined scopes.
class ShadowContextSource : public ContextSource, public ScopeInterface {
public:
  ShadowContextSource() = default;

  std::vector<Context> getContexts() override { return contextStack; }

  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

private:
  std::vector<Context> contextStack;
};

} // namespace neutron

#endif // NEUTRON_CONTEXT_CONTEXT_H_
