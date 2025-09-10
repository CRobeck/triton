import os
import subprocess
import pathlib
import json

import triton


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_override(tmp_path: pathlib.Path):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run once to get the file dumps
    first_env = os.environ.copy()
    first_env["TRITON_ALWAYS_COMPILE"] = "1"
    first_env["TRITON_DUMP_PASS_STAGES"] = "1"
    first_env["TRITON_DUMP_DIR"] = str(tmp_path)
    first_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)


    subprocess.run(["python3", dir_path + "/override_helper.py", str(tmp_path)], env=first_env)
    filename = tmp_path / "override_compiler.py"

    print(str(filename))

    with open(filename, "r") as infile:
        file_str = infile.readlines()

    with open(filename, "w") as outfile:
        for line in file_str:
            # turn off pre-fetching
            if "add_prefetch" in line:
                continue
            outfile.write(line)

    # Run again with pipeline override
    second_env = os.environ.copy()
    second_env["TRITON_ALWAYS_COMPILE"] = "1"
    second_env["TRITON_OVERRIDE_PASS_STAGES"] = "1"
    second_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    second_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)
    subprocess.run(["python3", dir_path + "/override_helper.py", str(tmp_path)], env=second_env)

    triton_[(1, )]()

    stages = {
        'make_ttir': "triton-combine",
        'make_ttgir': "triton.*-coalesce",
        'make_llir': "convert-triton-.*gpu-to-llvm",
    }


    for stage_name, stage_pipeline_check in stages.items():
        assert os.path.exists(str(repro_path) + '.' + stage_name + '.repro.mlir')
    #     curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
    #     repro = curr_repro_path.read_text()
    #     assert "mlir_reproducer" in repro, f"Expected MLIR reproducer in {curr_repro_path}. Got:\n{repro}"
    #     m = re.search(r"pipeline: \"(.*" + stage_pipeline_check + ".*)\"", repro)
    #     assert m, "Expected to match pass pipeline after \"pipeline:\" in MLIR reproducer"
    #     pipeline_str = m.group(1)
    #     assert pipeline_str, "Expected non-empty pass pipeline in MLIR reproducer"
