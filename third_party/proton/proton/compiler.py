from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path


class ProtonBackend(BaseBackend):

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    @staticmethod
    def make_llir(src, arch):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        custom_lds_size = 0
        passes.ttgpuir.add_allocate_shared_memory(pm, arch, custom_lds_size)

    def add_stages(self, stages, options):
        #stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        #stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, arch)
        #stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        #stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
