import triton
from triton import knobs
import inspect
from triton.backends.compiler import Language


import os
import re
import pathlib

def dump_stages(self, stages, options, language, capability):
    source_code = "# This is generated from Triton compiler.py"
    source_code = source_code + '\n' + "from triton._C.libtriton import ir, passes, llvm, amd, nvidia"
    source_code = source_code + '\n' + "class GPUOverrideBackend:"
    source_code = source_code + '\n' + inspect.getsource(self.make_ttir)
    source_code = source_code + '\n' + inspect.getsource(self.make_ttgir)
    full_name = "compiler_override.py"
    if knobs.cache.dump_dir:
        full_name = os.path.join(knobs.cache.dump_dir, full_name)

    with open(full_name, "w") as file:
        file.write(source_code)
def override_stages(self, stages, options, language, capability):
    # Limit to TTIR and TTGIR for now
    if language != Language.TRITON: return
    full_name = "compiler_override.py"
    if knobs.cache.override_dir:
        full_name = os.path.join(knobs.cache.override_dir, full_name)
# def inspect_stages(self, stages, options, language, capability):
#     print('INSPECT STAGES')
#     dump_stages(self, stages, options, language, capability)
    # override_stages(self, stages, options, language, capability)
def inspect_stages_hook(self, stages, options, language, capability):
    print('INSPECT STAGES')
    dump_stages(self, stages, options, language, capability)

def test_inspection(monkeypatch, tmp_path: pathlib.Path):
        # dump_stages(self, stages, options, language, capability)
    # stage_name = 'make_ttgir'
    # curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
    # repro_path = tmp_path / "repro_prefix"
    # # "repro_prefix.make_ttgir.repro.mlir"
    # full_name = os.path.join(str(tmp_path), "repro_prefix." + stage_name + ".repro.mlir")
    # # print(str(tmp_path))
    # with open(full_name, "r") as infile:
    #     file_str = infile.readlines()
    # print(file_str)

    # # full_name = os.path.join(knobs.cache.override_dir, full_name)

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(tmp_path))
    # monkeypatch.setenv("TRITON_DUMP_PASS_STAGES", "1")
    # monkeypatch.setenv("TRITON_INSPECT_PASS_STAGES", "1")



    # inspect_stages_hook_called = False
    # make_ttgir_wrapper_called = False

    # def inspect_stages_hook(self, stages, options, language, capability):
    #     print('INSPECT STAGES')
    #     dump_stages(self, stages, options, language, capability)
        # override_stages(self, stages, options, language, capability)
    #     nonlocal inspect_stages_hook_called
    #     inspect_stages_hook_called = True

    #     def make_ttgir_wrapper(src, metadata, options, capability):
    #         nonlocal make_ttgir_wrapper_called
    #         make_ttgir_wrapper_called = True
    #         return self.make_ttgir(src, metadata, options, capability)

    #     stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

    @triton.jit
    def k1():
        return

    @triton.jit
    def k2():
        return

    # Run once to get the clean/golden repro dump
    # k1[(1, )]()
    # assert not inspect_stages_hook_called and not make_ttgir_wrapper_called
    # assert os.path.exists(curr_repro_path)
    # golden_repro = curr_repro_path.read_text()
    # curr_repro_path.unlink()

    # Setup hook and call again, check if hooks got called
    knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    # knobs.runtime.add_stages_inspection_hook = inspect_stages
    k2[(1, )]()
    curr_repro_path = tmp_path / ("../test_inspection0." + "make_ttgir" + ".repro.mlir")
    repro = curr_repro_path.read_text()
    m = re.search(r"pipeline: \"(.*" + "convert-triton-to-tritongpu" + ".*)\"", repro)
    print(m.group(1))
    # assert "tritongpu-prefetch" not in m.group(1)
    # assert inspect_stages_hook_called and make_ttgir_wrapper_called
    # assert os.path.exists(curr_repro_path)
    # hook_repro = curr_repro_path.read_text()

    # Check that repros match
    # assert golden_repro.replace('k1', 'dummy') == hook_repro.replace('k2', 'dummy')
