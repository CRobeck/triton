#ifndef NEUTRON_PROFILER_CUPTI_PROFILER_H_
#define NEUTRON_PROFILER_CUPTI_PROFILER_H_

#include "GPUProfiler.h"

namespace neutron {

class CuptiProfiler : public GPUProfiler<CuptiProfiler> {
public:
  CuptiProfiler();
  virtual ~CuptiProfiler();

private:
  struct CuptiProfilerPimpl;
};

} // namespace neutron

#endif // NEUTRON_PROFILER_CUPTI_PROFILER_H_
