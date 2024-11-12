import tempfile
import triton
from triton.compiler import IRSource
from triton._C.libtriton import ir

target = triton.runtime.driver.active.get_current_target()


def test_mlir_attribute_parsing() -> None:
    '''
    Tests that MLIR attributes are parsed correctly from input ttir/ttgir.

    Checks for the following:
    1. Name and type signature are parsed correctly
    2. _get_num_warps_from_ir_str() works
    3. tt.nv_tma_desc attribute is parsed correctly
    '''

    sample_ttgir = r"""
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg3: i32 {tt.divisibility = 16 : i32},
                                %arg4: i32 {tt.divisibility = 16 : i32},
                                %arg5: i32 {tt.divisibility = 16 : i32},
                                %arg6: i32 {tt.divisibility = 16 : i32},
                                %arg7: i32 {tt.divisibility = 16 : i32},
                                %arg8: i32 {tt.divisibility = 16 : i32, tt.nv_tma_desc = 0 : i32},
                                %desc: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}) attributes {noinline = false} {
    tt.return
  }
}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(sample_ttgir)
        f.flush()
        context = ir.context()
        src = IRSource(f.name, context)

        # check name and type signature
        # should match ty_to_cpp(...)
        assert  src.signature == \
                    {0: "*f32", 1: "*f32", 2: "*f32", 3: "i32", \
                           4: "i32", 5: "i32", 6: "i32", 7: "i32", 8: "nvTmaDesc", 9: "nvTmaDesc"}
        assert src.name == "@matmul_kernel"

        # check num warps
        assert src.parse_options()['num_warps'] == 8

    sample_ttgir_vector_add = r"""
    #blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
       tt.func public @add_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
       %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
       %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32},
       %arg3: i32 {tt.divisibility = 16 : i32})
        attributes {noinline = false} {
         %c1024_i32 = arith.constant 1024 : i32
         %0 = tt.get_program_id x : i32
         %1 = arith.muli %0, %c1024_i32 : i32
         %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
         %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
         %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
         %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
         %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
         %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
         %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
         %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<i32>, #blocked>
         %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
         %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
         %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<i32>, #blocked>
         %13 = arith.addi %9, %12 : tensor<1024xi32, #blocked>
         %14 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
         %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
         tt.store %15, %13, %6 : tensor<1024x!tt.ptr<i32>, #blocked>
         tt.return
       }
     }
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(sample_ttgir_vector_add)
        f.flush()
        context = ir.context()
        src = IRSource(f.name, context)

        # now test compilation
        triton.compile(f.name, target=target)