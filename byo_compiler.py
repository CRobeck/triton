
@staticmethod
def byo_make_ttgir(pm, mod, metadata, opt, capability, cluster_info, dump_enabled, passes, nvidia):

    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
    # optimize TTGIR
    passes.ttgpuir.add_coalesce(pm)
    if capability // 10 >= 8:
        passes.ttgpuir.add_f32_dot_tc(pm)
    # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
    nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    passes.ttgpuir.add_accelerate_matmul(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
    passes.ttir.add_loop_aware_cse(pm)
    if capability // 10 in [8, 9]:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
    elif capability // 10 >= 10:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.ttgpuir.add_optimize_accumulator_init(pm)
        passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
        nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        # hoist again and allow hoisting out of if statements
        passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
        nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
    else:
        passes.ttir.add_triton_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.ttgpuir.add_prefetch(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.ttgpuir.add_coalesce_async_copy(pm)
    nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.common.add_symbol_dce(pm)
    if capability // 10 >= 9:
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
    nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
    nvidia.passes.ttnvgpuir.add_lower_mma(pm)
    passes.common.add_sccp(pm)
    passes.common.add_cse(pm)
    passes.common.add_canonicalizer(pm)

