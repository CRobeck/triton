#ifndef NEUTRON_PROFILER_ROCTRACER_PROFILER_H_
#define NEUTRON_PROFILER_ROCTRACER_PROFILER_H_

#include "GPUProfiler.h"

namespace neutron {

class RoctracerProfiler : public GPUProfiler<RoctracerProfiler> {
public:
  RoctracerProfiler();
  virtual ~RoctracerProfiler();

private:
  struct RoctracerProfilerPimpl;
};

} // namespace neutron

#endif // NEUTRON_PROFILER_ROCTRACER_PROFILER_H_
