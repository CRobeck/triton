import torch
import triton
import triton.profiler as proton
import triton.instrument as neutron
import tempfile
import json
import pytest
from typing import NamedTuple

import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def test_activate():
    proton.activate(0)
    torch.zeros((10, 10), device="cuda")
    proton.deactivate(0)    

    neutron.activate(0)
    print("test_profile")
    torch.zeros((10, 10), device="cuda")
    neutron.deactivate(0)
