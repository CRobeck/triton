#ifndef NEUTRON_DATA_TRACE_DATA_H_
#define NEUTRON_DATA_TRACE_DATA_H_

#include "Data.h"

namespace neutron {

class TraceData : public Data {
public:
  using Data::Data;
  virtual ~TraceData() = default;

  size_t addScope(size_t scopeId, const std::string &name) override;

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override;

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics,
                  bool aggregable) override;

protected:
  void startOp(const Scope &scope) override final;

  void stopOp(const Scope &scope) override final;

private:
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;
};

} // namespace neutron

#endif // NEUTRON_DATA_TRACE_DATA_H_
