#ifndef NEUTRON_DRIVER_GPU_HSA_H_
#define NEUTRON_DRIVER_GPU_HSA_H_

#include "Driver/Device.h"
#include "hsa/hsa_ext_amd.h"

namespace neutron {

namespace hsa {

template <bool CheckSuccess>
hsa_status_t agentGetInfo(hsa_agent_t agent, hsa_agent_info_t attribute,
                          void *value);

hsa_status_t iterateAgents(hsa_status_t (*callback)(hsa_agent_t agent,
                                                    void *data),
                           void *data);

} // namespace hsa

} // namespace neutron

#endif // NEUTRON_DRIVER_GPU_HSA_H_
