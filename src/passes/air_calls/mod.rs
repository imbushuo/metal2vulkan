//! Residual AIR call dispatch and scalar/vector helper lowering.

pub(in crate::passes) mod conversions;

use super::*;

mod images;
use images::*;

mod dispatch_texture;
pub(in crate::passes) use dispatch_texture::*;
mod integer_simd;
pub(in crate::passes) use integer_simd::*;
mod float_imageblock;
pub(in crate::passes) use float_imageblock::*;
mod rawbyte_unary;
pub(in crate::passes) use rawbyte_unary::*;
mod matrix;
pub(in crate::passes) use matrix::*;
mod shuffle;
pub(in crate::passes) use shuffle::*;
mod reduce_bitops;
pub(in crate::passes) use reduce_bitops::*;
mod bfloat_glsl;
pub(in crate::passes) use bfloat_glsl::*;
mod tensor;
pub(in crate::passes) use tensor::*;
mod agx_emask;
/// Splitting one AIR call site into its own region of basic blocks (shared by the lowerings that
/// need control flow).
mod block_split;
mod imageblock_block_copy;
pub(in crate::passes) use agx_emask::*;
pub(in crate::passes) use imageblock_block_copy::*;

fn copy_or_bitcast_result(
    result_type: Word,
    result: Word,
    source_type: Word,
    source: Word,
) -> Instruction {
    let opcode = if result_type == source_type {
        Op::CopyObject
    } else {
        Op::Bitcast
    };
    Instruction::new(
        opcode,
        Some(result_type),
        Some(result),
        vec![Operand::IdRef(source)],
    )
}

/// Build clamp edge operands for FClamp on possibly-vector `rty`. Scalar -> the scalar consts;
/// vector -> OpConstantComposite splat.
fn clamp_edges(ctx: &mut Ctx, rty: Word, zero: Word, one: Word) -> (Word, Word) {
    let defs = type_defs(&ctx.module);
    let is_vec = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(rty))
        .map(|g| g.class.opcode == Op::TypeVector)
        .or_else(|| defs.get(&rty).map(|g| g.class.opcode == Op::TypeVector))
        .unwrap_or(false);
    if !is_vec {
        return (zero, one);
    }
    let n = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(rty))
        .and_then(|g| g.operands.get(1).cloned())
        .or_else(|| defs.get(&rty).and_then(|g| g.operands.get(1).cloned()))
        .and_then(|o| match o {
            Operand::LiteralBit32(n) => Some(n),
            _ => None,
        })
        .unwrap_or(4);
    let lo = splat(ctx, rty, zero, n);
    let hi = splat(ctx, rty, one, n);
    (lo, hi)
}

/// Component count of `ty`: vector -> N, scalar -> 1.
fn vector_len(ctx: &Ctx, ty: Word) -> u32 {
    let find = |id: Word| {
        ctx.new_globals
            .iter()
            .chain(ctx.module.types_global_values.iter())
            .find(|g| g.result_id == Some(id))
            .cloned()
    };
    if let Some(def) = find(ty) {
        if def.class.opcode == Op::TypeVector {
            if let Some(Operand::LiteralBit32(n)) = def.operands.get(1) {
                return *n;
            }
        }
    }
    1
}

/// A constant of value `v` (0.0/1.0) shaped like `rty`: scalar -> the const; vector -> a splat. The
/// constant's type matches `rty`'s element type (half element -> a `half` const; else `float`), so a
/// `v3half` select arm splats half constants — not floats (which would mistype the OpConstantComposite).
fn splat_or_scalar(ctx: &mut Ctx, rty: Word, v: f32, n: u32) -> Word {
    let elem = element_type(ctx, rty);
    let s = if is_half_scalar(ctx, elem) {
        ctx.const_half(v)
    } else {
        ctx.const_float(v)
    };
    if n <= 1 {
        s
    } else {
        splat(ctx, rty, s, n)
    }
}

fn splat(ctx: &mut Ctx, vty: Word, scalar: Word, n: u32) -> Word {
    ctx.const_composite(vty, vec![scalar; n as usize])
}

/// f32 (scalar or vN) type id for a lane count.
pub(super) fn ty_f32_shaped(ctx: &mut Ctx, n: u32) -> Word {
    if n > 1 {
        ctx.ty_vecf(n)
    } else {
        ctx.ty_float()
    }
}

/// u32 (scalar or vN) type id for a lane count.
pub(super) fn ty_u32_shaped(ctx: &mut Ctx, n: u32) -> Word {
    if n > 1 {
        ctx.ty_vec_uint(n)
    } else {
        ctx.ty_uint()
    }
}

pub(super) fn ty_bool_shaped(ctx: &mut Ctx, n: u32) -> Word {
    if n > 1 {
        ctx.ty_vec_bool(n)
    } else {
        ctx.ty_bool()
    }
}

/// A shift amount of 16, shaped to match an `n`-lane operand (vector shifts need a vector amount).
pub(super) fn shift_amount_16(ctx: &mut Ctx, n: u32) -> Word {
    let s = ctx.const_uint(16);
    if n > 1 {
        let vty = ctx.ty_vec_uint(n);
        splat(ctx, vty, s, n)
    } else {
        s
    }
}

pub(super) fn shaped_u32_const(ctx: &mut Ctx, n: u32, value: u32) -> Word {
    let scalar = ctx.const_uint(value);
    if n > 1 {
        let vty = ctx.ty_vec_uint(n);
        splat(ctx, vty, scalar, n)
    } else {
        scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv::GroupOperation;

    #[test]
    fn zero_count_steps_cover_all_supported_widths() {
        assert_eq!(
            trailing_zero_steps(32),
            vec![(0xffff, 16), (0xff, 8), (0xf, 4), (0x3, 2), (0x1, 1)]
        );
        assert_eq!(
            trailing_zero_steps(16),
            vec![(0xff, 8), (0xf, 4), (0x3, 2), (0x1, 1)]
        );
        assert_eq!(trailing_zero_steps(8), vec![(0xf, 4), (0x3, 2), (0x1, 1)]);
        assert_eq!(
            leading_zero_steps(32),
            vec![
                (0xffff_0000, 16),
                (0xff00_0000, 8),
                (0xf000_0000, 4),
                (0xc000_0000, 2),
                (0x8000_0000, 1),
            ]
        );
        assert_eq!(
            leading_zero_steps(16),
            vec![(0xff00, 8), (0xf000, 4), (0xc000, 2), (0x8000, 1)]
        );
        // 64-bit high mask exercises the full i64 bit pattern (top word set).
        assert_eq!(
            leading_zero_steps(64)[0],
            (0xffff_ffff_0000_0000u64 as i64, 32)
        );
    }

    #[test]
    fn glsl_extinst_does_not_match_explicit_memory_order() {
        assert!(glsl_extinst("air.atomic_fetch_max_explicit_texture_2d.i16.u.v4i32").is_none());
        assert!(matches!(
            glsl_extinst("air.fast_exp.v4f32"),
            Some(GLSLstd450::Exp)
        ));
        assert!(matches!(
            glsl_extinst("air.fast_powr.v4f32"),
            Some(GLSLstd450::Pow)
        ));
    }

    /// A plain min/max/clamp symbol names Metal's NaN-aware function and must reach the GLSL op
    /// that defines the NaN case; a `fast_` symbol names `fast::`, which defines neither NaN nor
    /// infinity, and keeps the op whose NaN result SPIR-V leaves undefined.
    #[test]
    fn plain_minmax_is_nan_aware_and_fast_minmax_is_not() {
        for (name, expected) in [
            ("air.fmax.f32", GLSLstd450::NMax),
            ("air.fmin.v4f16", GLSLstd450::NMin),
            ("air.max.f16", GLSLstd450::NMax),
            ("air.min.v3f32", GLSLstd450::NMin),
            ("air.clamp.f16", GLSLstd450::NClamp),
            ("llvm.maxnum.f32", GLSLstd450::NMax),
            ("llvm.minnum.f16", GLSLstd450::NMin),
            ("air.fast_fmax.f32", GLSLstd450::FMax),
            ("air.fast_fmin.v4f32", GLSLstd450::FMin),
            ("air.fast_clamp.f32", GLSLstd450::FClamp),
        ] {
            assert_eq!(glsl_extinst(name), Some(expected), "{name}");
        }
        // The rule rewrites only the three min/max/clamp ops; every other entry is untouched.
        for name in ["air.sqrt.f32", "air.fast_exp.v4f32", "air.mix.f16"] {
            let op = glsl_extinst(name).expect(name);
            assert_eq!(nan_aware_if_precise(name, op), op, "{name}");
        }
    }

    // ---- simd reduce clustering onto Metal's 32-lane simdgroup ---------------------------------

    /// Look up the u32 value of an `OpConstant` id in the module (new_globals or types).
    fn const_uint_value(ctx: &Ctx, id: Word) -> Option<u32> {
        ctx.new_globals
            .iter()
            .chain(ctx.module.types_global_values.iter())
            .find(|g| g.result_id == Some(id) && g.class.opcode == Op::Constant)
            .and_then(|g| match g.operands.first() {
                Some(Operand::LiteralBit32(v)) => Some(*v),
                _ => None,
            })
    }

    /// A whole-subgroup `Reduce` becomes a `ClusteredReduce` with a trailing cluster-size operand
    /// referencing the constant 32 (Metal's simdgroup width).
    #[test]
    fn group_reduce_operands_clusters_every_reduce() {
        let mut ctx = Ctx::new(Module::new());
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let value = ctx.const_uint(7);
        let ops = group_reduce_operands(&mut ctx, scope, GroupOperation::Reduce, value);
        assert_eq!(
            ops.len(),
            4,
            "clustered reduce carries a cluster-size operand"
        );
        assert_eq!(ops[0], Operand::IdScope(scope));
        assert_eq!(
            ops[1],
            Operand::GroupOperation(GroupOperation::ClusteredReduce)
        );
        assert_eq!(ops[2], Operand::IdRef(value));
        let Operand::IdRef(cluster) = ops[3] else {
            panic!("cluster-size operand is an IdRef");
        };
        assert_eq!(
            const_uint_value(&ctx, cluster),
            Some(32),
            "cluster size is 32 lanes"
        );
    }

    /// A scan is never turned into a clustered reduce — `ClusteredReduce` is a reduce-only group
    /// operation, so `lower_simd_sum` corrects a scan by subtracting its partition prefix instead.
    #[test]
    fn group_reduce_operands_never_clusters_scan() {
        let mut ctx = Ctx::new(Module::new());
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let value = ctx.const_uint(7);
        let ops = group_reduce_operands(&mut ctx, scope, GroupOperation::InclusiveScan, value);
        assert_eq!(ops.len(), 3, "a scan is not turned into a clustered reduce");
        assert_eq!(
            ops[1],
            Operand::GroupOperation(GroupOperation::InclusiveScan)
        );
    }
}
