// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=2 try_to_unguard_epilogue=0" -canonicalize | FileCheck %s --check-prefixes=COMMON,SYNC,GUARD-EPILOGUE
// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=2 try_to_unguard_epilogue=1" -canonicalize | FileCheck %s --check-prefixes=COMMON,SYNC


// Check that we can pipeline a simple matmul kernel

// COMMON-LABEL: simple_matmul_kernel

// Prologue
// COMMON-COUNT-2: ttg.local_alloc
  // SYNC-COUNT-2: tt.load
  // SYNC-COUNT-2: ttg.local_store
  //

// Main loop
//         COMMON:   scf.for
//
  // SYNC-COUNT-2:   ttg.local_load
  //         SYNC:   tt.dot
  //         SYNC:   scf.yield
  //
// Epilogue
// COMMON-COUNT-2: ttg.local_load
//         GUARD-EPILOGUE: scf.if
//         COMMON:   tt.dot
// GUARD-EPILOGUE:   scf.yield
// COMMON-COUNT-2: ttg.local_dealloc

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @simple_matmul_kernel(%test: tensor<1x64xi32, #blocked1>, %arg0: tensor<64x64x!tt.ptr<f16>, #mma>, %arg1: i32, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<32x64xi32, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %1 = arith.muli %arg1, %c64_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %3 = arith.addi %2, %0 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = arith.remsi %3, %4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %11 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %13 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
    %14 = tt.addptr %13, %12 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %15:3 = scf.for %arg11 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg12 = %cst_1, %arg13 = %10, %arg14 = %14) -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %17 = tt.load %arg13 : tensor<64x32x!tt.ptr<f16>, #blocked>
      %18 = tt.load %arg14 : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %19 = ttg.convert_layout %17 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %20 = ttg.convert_layout %18 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %21 = tt.dot %19, %20, %arg12, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      %22 = tt.addptr %arg13, %cst : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %23 = tt.addptr %arg14, %cst_0 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      scf.yield %21, %22, %23 : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>
    }
    %16 = arith.truncf %15#0 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    tt.store %arg0, %16 : tensor<64x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}
