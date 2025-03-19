#include "hip/hip_runtime.h"

#define UINT64_T unsigned long long

//__global__ void load_acquire_system(UINT64_T *input, UINT64_T *output) {
//  *output = __hip_atomic_load(input, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
//}

__attribute__((used))
__device__ UINT64_T
 load_acquire_system(UINT64_T *input) {
  UINT64_T output = __hip_atomic_load(input, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return output;
}
