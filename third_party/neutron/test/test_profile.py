import torch
import triton
#import triton.instrument as neutron
import triton.profiler as proton

import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_activate():
#    neutron.activate(0)
    torch.zeros((10, 10), device="cuda")
#    neutron.deactivate(0)
