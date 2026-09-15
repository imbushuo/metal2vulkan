//! Subgroup and quad shuffle AIR call lowering, plus the lane-index builtins the shuffles and the
//! simdgroup matrix families both read.

use super::*;
pub(in crate::passes) fn subgroup_shuffle_index_u32(
    ctx: &mut Ctx,
    value: Word,
    insts: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let Some(ty) = value_result_type(ctx, value) else {
        return Err("subgroup shuffle index has no type".to_string());
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return Err("subgroup shuffle index type is undefined".to_string());
    };
    if def.class.opcode != Op::TypeInt {
        return Err("subgroup shuffle index is not an integer".to_string());
    }
    if def.operands.first() == Some(&Operand::LiteralBit32(32)) {
        return Ok(value);
    }
    let uint = ctx.ty_uint();
    let converted = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(converted),
        vec![Operand::IdRef(value)],
    ));
    Ok(converted)
}

pub(in crate::passes) fn lower_quad_shuffle(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let requested_lane = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let quad_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_base),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(!3u32)),
        ],
    ));
    let quad_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_local),
        vec![
            Operand::IdRef(requested_lane),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(quad_base), Operand::IdRef(quad_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.quad_shuffle_rotate_down(value, delta)` rotates values DOWN within the 4-lane quad, wrapping:
/// result on lane L = `value` from lane `(L + delta) % 4`. Verified bit-exact on Apple Metal (lane L,
/// delta d -> lane (L+d)%4 for all L,d in 0..3). Lowered as a masked `GroupNonUniformShuffle`:
/// `source_lane = (quad_base) + ((lane & 3) + delta) % 4`.
pub(in crate::passes) fn lower_quad_shuffle_rotate_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let quad_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_base),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(!3u32)),
        ],
    ));
    let local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(local),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let local_plus_delta = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(local_plus_delta),
        vec![Operand::IdRef(local), Operand::IdRef(delta)],
    ));
    // (local + delta) % 4 — the quad size is a power of two, so a bitwise AND is the modulo.
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(source_local),
        vec![
            Operand::IdRef(local_plus_delta),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(quad_base), Operand::IdRef(source_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.simd_shuffle_rotate_down(value, delta)` rotates values down within Metal's 32-lane simdgroup,
/// wrapping at the simdgroup boundary: source lane = `(lane + delta) & 31`.
pub(in crate::passes) fn lower_simd_shuffle_rotate_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let simd_lane = metal_simd_lane_local_u32(ctx, subgroup_lane, &mut insts);
    let simd_base = metal_simd_lane_base_u32(ctx, subgroup_lane, simd_lane, &mut insts);
    let local_plus_delta = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(local_plus_delta),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(source_local),
        vec![
            Operand::IdRef(local_plus_delta),
            Operand::IdRef(ctx.const_uint(31)),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(simd_base), Operand::IdRef(source_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.simd_shuffle_down(value, delta)` reads `value` from lane `lane + delta` within Metal's
/// 32-lane simdgroup. Lower via absolute shuffle instead of `ShuffleDown` so wider Vulkan subgroups
/// cannot cross the Metal simdgroup boundary. Out-of-range reads select the current lane to avoid
/// emitting an undefined SPIR-V source lane.
pub(in crate::passes) fn lower_simd_shuffle_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let simd_lane = metal_simd_lane_local_u32(ctx, lane, &mut insts);
    let simd_base = metal_simd_lane_base_u32(ctx, lane, simd_lane, &mut insts);
    let remaining = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(remaining),
        vec![
            Operand::IdRef(ctx.const_uint(32)),
            Operand::IdRef(simd_lane),
        ],
    ));
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ULessThan,
        Some(ctx.ty_bool()),
        Some(in_bounds),
        vec![Operand::IdRef(delta), Operand::IdRef(remaining)],
    ));
    let shifted_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(shifted_lane),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let shifted_subgroup_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(shifted_subgroup_lane),
        vec![Operand::IdRef(simd_base), Operand::IdRef(shifted_lane)],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(source_lane),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(shifted_subgroup_lane),
            Operand::IdRef(lane),
        ],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.simd_shuffle_up(value, delta)` reads `value` from lane `lane - delta` within Metal's
/// 32-lane simdgroup. Use absolute shuffle to keep semantics independent of the Vulkan subgroup
/// width.
pub(in crate::passes) fn lower_simd_shuffle_up(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let simd_lane = metal_simd_lane_local_u32(ctx, lane, &mut insts);
    let simd_base = metal_simd_lane_base_u32(ctx, lane, simd_lane, &mut insts);
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UGreaterThanEqual,
        Some(ctx.ty_bool()),
        Some(in_bounds),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let shifted_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(shifted_lane),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let shifted_subgroup_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(shifted_subgroup_lane),
        vec![Operand::IdRef(simd_base), Operand::IdRef(shifted_lane)],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(source_lane),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(shifted_subgroup_lane),
            Operand::IdRef(lane),
        ],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.quad_shuffle_up(value, delta)` reads `value` from lane `local - delta` within the 4-lane quad;
/// when `local - delta` underflows the quad, Apple Metal returns the lane's OWN value (verified: lane
/// 0 up(1) and lane 1 up(2) return their own value, not a cross-quad lane). Lowered quad-boundary-safe
/// via `quad_base = lane & ~3`: the source local index is clamped to the lane's own local when it would
/// underflow, so the shuffle reads a valid in-quad lane and yields the own value out of bounds.
pub(in crate::passes) fn lower_quad_shuffle_up(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let quad_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_base),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(!3u32)),
        ],
    ));
    let local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(local),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UGreaterThanEqual,
        Some(bool_ty),
        Some(in_bounds),
        vec![Operand::IdRef(local), Operand::IdRef(delta)],
    ));
    let lowered = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(lowered),
        vec![Operand::IdRef(local), Operand::IdRef(delta)],
    ));
    // Out of bounds -> read own local lane, which yields the lane's own value (Metal's behaviour).
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(source_local),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(lowered),
            Operand::IdRef(local),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(quad_base), Operand::IdRef(source_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `simd_shuffle_and_fill_down(data, fill, delta, modulo)`: within each `modulo`-wide cluster, lane
/// `l` reads `data[l+delta]`; lanes whose `l+delta` leaves the cluster read `fill[l+delta-modulo]`
/// instead. Both reads share the cluster base, the data read keeps the unwrapped offset and the
/// fill read takes it modulo the cluster width, and a final `Select` on `l + delta < modulo` picks
/// between them — the mirror of `lower_simd_shuffle_and_fill_up`. Device-verified by the authored
/// cases on `filter_error_map_SIMD` (`91f91ae2dc143089`), which read one 32-lane simdgroup through
/// this call at `modulo` 32 and at a literal 4, where the cluster base is not zero.
pub(in crate::passes) fn lower_simd_shuffle_and_fill_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[2], &mut insts)?;
    let modulo = subgroup_shuffle_index_u32(ctx, args[3], &mut insts)?;
    let uint = ctx.ty_uint();
    let modulo_is_zero = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IEqual,
        Some(ctx.ty_bool()),
        Some(modulo_is_zero),
        vec![Operand::IdRef(modulo), Operand::IdRef(ctx.const_uint(0))],
    ));
    let safe_modulo = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(safe_modulo),
        vec![
            Operand::IdRef(modulo_is_zero),
            Operand::IdRef(ctx.const_uint(1)),
            Operand::IdRef(modulo),
        ],
    ));
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let local_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(local_lane),
        vec![Operand::IdRef(lane), Operand::IdRef(safe_modulo)],
    ));
    let cluster_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(cluster_base),
        vec![Operand::IdRef(lane), Operand::IdRef(local_lane)],
    ));
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_local),
        vec![Operand::IdRef(local_lane), Operand::IdRef(delta)],
    ));
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ULessThan,
        Some(ctx.ty_bool()),
        Some(in_bounds),
        vec![Operand::IdRef(source_local), Operand::IdRef(safe_modulo)],
    ));
    let wrapped_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(wrapped_local),
        vec![Operand::IdRef(source_local), Operand::IdRef(safe_modulo)],
    ));
    let data_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(data_local),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(source_local),
            Operand::IdRef(wrapped_local),
        ],
    ));
    let data_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(data_lane),
        vec![Operand::IdRef(cluster_base), Operand::IdRef(data_local)],
    ));
    let fill_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(fill_lane),
        vec![Operand::IdRef(cluster_base), Operand::IdRef(wrapped_local)],
    ));
    let data = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(data),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(data_lane),
        ],
    ));
    let fill = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(fill),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[1]),
            Operand::IdRef(fill_lane),
        ],
    ));
    insts.push(Instruction::new(
        Op::Select,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(data),
            Operand::IdRef(fill),
        ],
    ));
    Ok(insts)
}

/// `simd_shuffle_and_fill_up(data, fill, delta, modulo)`: within each `modulo`-wide cluster, the
/// in-cluster lane `l` reads `data[l-delta]`; lanes with `l < delta` instead read
/// `fill[l-delta+modulo]`. The shuffle source index `cluster_base + ((l + modulo - (delta%modulo))
/// % modulo)` is identical for both the data and fill reads (it wraps to the in-cluster position),
/// and a final `Select` on `l >= delta` picks data vs fill — mirroring
/// `lower_simd_shuffle_and_fill_down` with the up direction.
///
/// **The `Select` compares the lane against the UNREDUCED delta.** `delta == modulo` shifts the
/// whole cluster out, so every lane must answer `fill` at its own position, and the two-argument
/// overload — which passes `modulo = __metal_get_simdgroup_size()` — spells that ordinary input as
/// `simd_shuffle_and_fill_up(data, fill, 32)`. Comparing against `delta % modulo` answered zero
/// there, found every lane in bounds and returned `data`, where its own mirror returns `fill` and
/// so does the device. Device-verified by the authored case
/// `shift-a-whole-simd-cluster-out-through-the-fill` (`1f9792852fdfe2bf`), which reads one 32-lane
/// simdgroup through both directions at `delta == modulo` for `modulo` 32 and 8, and at three
/// in-range deltas that pin the rest of the model.
///
/// The reduced `delta % modulo` survives in the wrap arithmetic only, where it keeps
/// `l + modulo - delta` from underflowing an unsigned subtraction above `delta == modulo`; on
/// `[0, modulo]` the two spellings agree, so nothing this case measures depends on the difference.
pub(in crate::passes) fn lower_simd_shuffle_and_fill_up(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[2], &mut insts)?;
    let modulo = subgroup_shuffle_index_u32(ctx, args[3], &mut insts)?;
    // safe_modulo = modulo == 0 ? 1 : modulo (avoid UMod-by-zero).
    let modulo_is_zero = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IEqual,
        Some(bool_ty),
        Some(modulo_is_zero),
        vec![Operand::IdRef(modulo), Operand::IdRef(ctx.const_uint(0))],
    ));
    let safe_modulo = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(safe_modulo),
        vec![
            Operand::IdRef(modulo_is_zero),
            Operand::IdRef(ctx.const_uint(1)),
            Operand::IdRef(modulo),
        ],
    ));
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let local_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(local_lane),
        vec![Operand::IdRef(lane), Operand::IdRef(safe_modulo)],
    ));
    let cluster_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(cluster_base),
        vec![Operand::IdRef(lane), Operand::IdRef(local_lane)],
    ));
    // delta_mod = delta % safe_modulo (bounds the arithmetic so the lift stays in [0, 2*modulo)).
    let delta_mod = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(delta_mod),
        vec![Operand::IdRef(delta), Operand::IdRef(safe_modulo)],
    ));
    // in_bounds = local_lane >= delta (no wrap needed -> use data). Against the UNREDUCED delta, so
    // that a delta of a whole cluster leaves every lane out of bounds rather than none of them.
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UGreaterThanEqual,
        Some(bool_ty),
        Some(in_bounds),
        vec![Operand::IdRef(local_lane), Operand::IdRef(delta)],
    ));
    // lifted = local_lane + safe_modulo - delta_mod, in [0, 2*modulo); wrapped = lifted % modulo is
    // the in-cluster source position for both data (when in bounds) and fill (when wrapped).
    let lifted_hi = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(lifted_hi),
        vec![Operand::IdRef(local_lane), Operand::IdRef(safe_modulo)],
    ));
    let lifted = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(lifted),
        vec![Operand::IdRef(lifted_hi), Operand::IdRef(delta_mod)],
    ));
    let wrapped_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(wrapped_local),
        vec![Operand::IdRef(lifted), Operand::IdRef(safe_modulo)],
    ));
    let src_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(src_lane),
        vec![Operand::IdRef(cluster_base), Operand::IdRef(wrapped_local)],
    ));
    let data = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(data),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(src_lane),
        ],
    ));
    let fill = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(fill),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[1]),
            Operand::IdRef(src_lane),
        ],
    ));
    insts.push(Instruction::new(
        Op::Select,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(data),
            Operand::IdRef(fill),
        ],
    ));
    Ok(insts)
}

pub(in crate::passes) fn subgroup_lane_index_u32(
    ctx: &mut Ctx,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let var = subgroup_local_invocation_id_input_var(ctx, uint);
    let lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Load,
        Some(uint),
        Some(lane),
        vec![Operand::IdRef(var)],
    ));
    lane
}

pub(in crate::passes) fn metal_simd_lane_local_u32(
    ctx: &mut Ctx,
    subgroup_lane: Word,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let simd_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(simd_lane),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(31)),
        ],
    ));
    simd_lane
}

pub(in crate::passes) fn metal_simd_lane_base_u32(
    ctx: &mut Ctx,
    subgroup_lane: Word,
    simd_lane: Word,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let simd_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(simd_base),
        vec![Operand::IdRef(subgroup_lane), Operand::IdRef(simd_lane)],
    ));
    simd_base
}

/// The absolute subgroup lane naming Metal simd-local index `index` within the CALLING lane's own
/// 32-lane simdgroup: `(lane & !31) + (index & 31)`.
///
/// The base is the semantic half. It keeps a shuffle inside the caller's Metal simdgroup on a driver
/// whose subgroup is wider than 32 -- without it, lane 40 asking for simd-local 3 reads absolute
/// lane 3, which belongs to a different simdgroup. Every other member of this family already
/// resolves its lane this way.
///
/// The mask is a well-definedness choice, and it is NOT Metal's answer. Measured on an M3 Max:
/// `simd_shuffle(v, lane + 32)` returns lane `lane & !3` -- quad-aligned garbage, neither the index
/// modulo 32 nor the lane's own value -- so Metal has no defined out-of-range behaviour to match.
/// What the mask buys is that `base + (index & 31)` is always a real lane of the caller's own
/// simdgroup, where `base + index` could name an id past the end of the subgroup entirely, which
/// SPIR-V leaves undefined. This is the same defined refinement of an undefined Metal case that
/// `simd_shuffle_up`/`_down` make when they select the caller's own lane out of range.
pub(in crate::passes) fn metal_simd_absolute_lane_u32(
    ctx: &mut Ctx,
    index: Word,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let lane = subgroup_lane_index_u32(ctx, insts);
    let simd_lane = metal_simd_lane_local_u32(ctx, lane, insts);
    let simd_base = metal_simd_lane_base_u32(ctx, lane, simd_lane, insts);
    let masked = metal_simd_lane_local_u32(ctx, index, insts);
    let absolute = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(absolute),
        vec![Operand::IdRef(simd_base), Operand::IdRef(masked)],
    ));
    absolute
}

pub(in crate::passes) fn subgroup_local_invocation_id_input_var(ctx: &mut Ctx, uint: Word) -> Word {
    let key = SynthCacheKey::SubgroupLocalInvocationIdInputVar;
    if let Some(&var) = ctx.synth_cache.get(&key) {
        return var;
    }
    if let Some(var) = existing_builtin_input_var(ctx, BuiltIn::SubgroupLocalInvocationId, uint) {
        decorate_fragment_integer_input_flat(ctx, var);
        ctx.synth_cache.insert(key, var);
        return var;
    }
    let ptr_ty = ctx.ty_ptr(StorageClass::Input, uint);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::Input)],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(var),
            Operand::Decoration(Decoration::BuiltIn),
            Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId),
        ],
    ));
    decorate_fragment_integer_input_flat(ctx, var);
    ctx.interface.push(var);
    ctx.synth_cache.insert(key, var);
    var
}

fn decorate_fragment_integer_input_flat(ctx: &mut Ctx, var: Word) {
    if ctx.stage != Stage::Fragment
        || ctx.module.annotations.iter().any(|instruction| {
            instruction.class.opcode == Op::Decorate
                && instruction.operands.first() == Some(&Operand::IdRef(var))
                && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Flat))
        })
    {
        return;
    }
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![Operand::IdRef(var), Operand::Decoration(Decoration::Flat)],
    ));
}

pub(in crate::passes) fn existing_builtin_input_var(
    ctx: &Ctx,
    builtin: BuiltIn,
    pointee: Word,
) -> Option<Word> {
    ctx.module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find_map(|inst| {
            if inst.class.opcode != Op::Variable {
                return None;
            }
            if inst.operands.first() != Some(&Operand::StorageClass(StorageClass::Input)) {
                return None;
            }
            let var = inst.result_id?;
            let ptr_ty = inst.result_type?;
            let ptr_def = type_def_of(ctx, ptr_ty)?;
            if ptr_def.class.opcode != Op::TypePointer
                || ptr_def.operands.first() != Some(&Operand::StorageClass(StorageClass::Input))
                || ptr_def.operands.get(1) != Some(&Operand::IdRef(pointee))
            {
                return None;
            }
            has_builtin_decoration(ctx, var, builtin).then_some(var)
        })
}

pub(in crate::passes) fn has_builtin_decoration(ctx: &Ctx, var: Word, builtin: BuiltIn) -> bool {
    ctx.module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && inst.operands.first() == Some(&Operand::IdRef(var))
            && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && inst.operands.get(2) == Some(&Operand::BuiltIn(builtin))
    })
}
