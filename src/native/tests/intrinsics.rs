#![allow(unused_imports)]
use super::super::cfg::{
    id_ref_operand, infer_branch_merges, infer_loop_merges, infer_switch_merges,
    lower_unstructured_switches, split_body_blocks, BodyBlock,
};
use super::super::emit_vulkan_spirv;
use super::super::emitter::Emitter;
use super::super::ir::{LlType, LlValue};
use super::super::parse::{parse_type, parse_typed_value};
use super::*;
use crate::passes::{self, Stage};
use crate::spirv_module::load_bytes;
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Instruction};
use crate::{disassemble, meta, tools};
use spirv::{Decoration, Op, Scope, SelectionControl, StorageClass, Word};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[test]
fn native_llvm_umax_lowers_to_compare_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define i64 @main(i64 %a, i64 %b) {
entry:
  %m = tail call i64 @llvm.umax.i64(i64 %a, i64 %b)
  ret i64 %m
}

declare i64 @llvm.umax.i64(i64, i64)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUGreaterThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.umax"), "{asm}");
}

#[test]
fn native_llvm_usub_sat_lowers_to_compare_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define i32 @sat32(i32 %a, i32 %b) {
entry:
  %m = call i32 @llvm.usub.sat.i32(i32 %a, i32 %b)
  ret i32 %m
}

define i64 @sat64(i64 %a, i64 %b) {
entry:
  %m = call i64 @llvm.usub.sat.i64(i64 %a, i64 %b)
  ret i64 %m
}

declare i32 @llvm.usub.sat.i32(i32, i32)
declare i64 @llvm.usub.sat.i64(i64, i64)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpULessThan"), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.usub.sat"), "{asm}");
}

#[test]
fn native_parse_type_skips_leading_fast_math_flags() {
    assert_eq!(
        parse_type("nnan ninf nsz arcp afn <2 x half>").expect("parse flagged vector type"),
        LlType::Vector(Box::new(LlType::Half), 2)
    );
    assert_eq!(
        parse_type("volatile <4 x half>").expect("parse volatile vector type"),
        LlType::Vector(Box::new(LlType::Half), 4)
    );
}

#[test]
fn native_fast_multiply_add_contracts_without_changing_precise_arithmetic() {
    let ll = r#"
define float @long_fast_multiply_add(float %a, float %b, float %c) {
entry:
  %p0 = fmul fast float %a, %b
  %p1 = fmul fast float %a, %b
  %p2 = fmul fast float %a, %b
  %p3 = fmul fast float %a, %b
  %p4 = fmul fast float %a, %b
  %p5 = fmul fast float %a, %b
  %p6 = fmul fast float %a, %b
  %p7 = fmul fast float %a, %b
  %p8 = fmul fast float %a, %b
  %p9 = fmul fast float %a, %b
  %s0 = fadd fast float %c, %p0
  %s1 = fadd fast float %s0, %p1
  %s2 = fadd fast float %s1, %p2
  %s3 = fadd fast float %s2, %p3
  %s4 = fadd fast float %s3, %p4
  %s5 = fadd fast float %s4, %p5
  %s6 = fadd fast float %s5, %p6
  %s7 = fadd fast float %s6, %p7
  %s8 = fadd fast float %s7, %p8
  %s9 = fadd fast float %s8, %p9
  ret float %s9
}

define float @precise_multiply_add(float %a, float %b, float %c) {
entry:
  %product = fmul float %a, %b
  %sum = fadd float %product, %c
  ret float %sum
}

define float @shared_fast_product(float %a, float %b, float %c) {
entry:
  %product = fmul fast float %a, %b
  %sum = fadd fast float %product, %c
  %both = fadd fast float %sum, %product
  ret float %both
}

define float @product_rooted_sum(float %a, float %b, float %c, float %d) {
entry:
  %left = fmul fast float %a, %b
  %right = fmul fast float %c, %d
  %root = fadd fast float %left, %right
  %tail = fmul fast float %a, %d
  %sum = fadd fast float %root, %tail
  ret float %sum
}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load spv");
    let fmas = module
        .all_inst_iter()
        .filter(|instruction| {
            instruction.class.opcode == Op::ExtInst
                && matches!(
                    instruction.operands.get(1),
                    Some(Operand::LiteralExtInstInteger(number))
                        if *number == spirv::GlslStd450Op::Fma as u32
                )
        })
        .count();
    assert_eq!(fmas, 10);
    assert_eq!(
        module
            .all_inst_iter()
            .filter(|instruction| instruction.class.opcode == Op::FAdd)
            .count(),
        5
    );
}

/// `air.is_uniform` is Metal's `uniform<T>` assertion, so the module carries the constant it
/// asserts -- not a subgroup vote, and not the capability one needs.
///
/// AIR feeds the predicate straight to `llvm.assume`: all 35143 occurrences across the 14579 local
/// corpus sources are consumed by an assume or by a `phi i1` that is, so the program supplies the
/// answer rather than observing it. Lowering it to `OpGroupNonUniformAllEqual` answered a question
/// at the subgroup scope AIR never mentioned -- every Metal intrinsic scoped to an execution group
/// is spelled `air.simd_*` or `air.quad_*`, and this one names no group -- while placing a
/// convergent instruction wherever the frontend had put a free assumption and demanding
/// `GroupNonUniformVote` of every device running the module. 684 corpus modules carried it.
#[test]
fn native_air_is_uniform_is_the_assertion_it_states_not_a_subgroup_vote() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %same = tail call i1 @air.is_uniform.i32(i32 %v)
  tail call void @llvm.assume(i1 %same)
  %word = select i1 %same, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.is_uniform.i32(i32)
declare void @llvm.assume(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_is_uniform_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpGroupNonUniformAllEqual"), "{asm}");
    assert!(!asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpConstantTrue"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_llvm_assume_is_dropped_after_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %same = icmp eq i32 %v, %v
  tail call void @llvm.assume(i1 %same)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.assume(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_assume_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("llvm.assume"), "{asm}");
    assert!(!asm.contains("llvm_assume"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_agx3_yield_scheduling_hint_is_dropped_after_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i16 %delay, ptr addrspace(1) %out) {
entry:
  tail call void @llvm.agx3.yield(i16 %delay)
  store i16 %delay, ptr addrspace(1) %out, align 2
  ret void
}

declare void @llvm.agx3.yield(i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"ushort", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_agx3_yield_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("llvm.agx3.yield"), "{asm}");
    assert!(!asm.contains("llvm_agx3_yield"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_pack_unorm4x8_lowers_to_glsl_extinst() {
    let ll = r#"
source_filename = "pack_unorm"
target datalayout = "e-p:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @main(ptr addrspace(1) %out) {
entry:
  %packed = tail call i32 @air.pack.unorm4x8.v4f32(<4 x float> <float 0.000000e+00, float 5.000000e-01, float 1.000000e+00, float 2.500000e-01>)
  store i32 %packed, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.pack.unorm4x8.v4f32(<4 x float>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_unorm4x8_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(
        module,
        Stage::Kernel,
        None,
        None,
        meta::parse_air_kernel_meta(ll).as_ref(),
        meta::entry_name(ll, "kernel").as_deref(),
    )
    .expect("interface transform")
    .assemble()
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("PackUnorm4x8"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_air_srgb_pack_carries_the_transfer_and_leaves_alpha_alone() {
    // `air.pack.unorm4x8.srgb` CONTAINS `unorm4x8`, so the substring format table answers for it
    // and drops the transfer: Metal packs 0.5 as byte 188, a plain PackUnorm4x8 writes 128.
    let ll = r#"
source_filename = "pack_srgb"
target datalayout = "e-p:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @main(ptr addrspace(1) %out) {
entry:
  %packed = tail call i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float> <float 0.000000e+00, float 5.000000e-01, float 1.000000e+00, float 5.000000e-01>)
  store i32 %packed, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_srgb_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(
        module,
        Stage::Kernel,
        None,
        None,
        meta::parse_air_kernel_meta(ll).as_ref(),
        meta::entry_name(ll, "kernel").as_deref(),
    )
    .expect("interface transform")
    .assemble()
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("PackUnorm4x8"), "{asm}");
    // The transfer itself: both arms, the 0.0031308 knee, and the per-lane select between them.
    assert!(asm.contains(" Pow "), "{asm}");
    assert!(asm.contains("0.0031308"), "{asm}");
    assert!(asm.contains("12.92"), "{asm}");
    assert!(asm.contains("OpFOrdLessThanEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    // Alpha comes back from the untransformed vector -- lane 3 of the second shuffle operand.
    assert!(asm.contains("OpVectorShuffle"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_air_pack_unorm_rgb565_f16_lowers_to_exact_fields() {
    // The rounding is `RoundEven`, not `Round`. SPIR-V leaves `Round`'s tie direction to the
    // implementation and SPIRV-Cross spells the two as different MSL functions -- `round` rounds
    // ties away from zero, `rint` rounds them to even. Metal rounds these ties to EVEN, verified on
    // device by `the-rgb565-pack-at-the-exact-tie`: components whose f32 products with 31, 63 and
    // 31 are exactly 0.5, 2.5 and 4.5 pack to 0, 2 and 4, and the `Round` form answered 1, 3 and 5.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %out) {
entry:
  %packed = call i16 @air.pack.unorm.rgb565.v3f16(<3 x half> <half 0xH3C00, half 0xH3800, half 0xH0000>)
  store i16 %packed, ptr addrspace(1) %out, align 2
  ret void
}

declare i16 @air.pack.unorm.rgb565.v3f16(<3 x half>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_unorm_rgb565_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert_eq!(asm.matches(" NClamp ").count(), 3, "{asm}");
    assert_eq!(asm.matches(" FClamp ").count(), 0, "{asm}");
    assert_eq!(asm.matches(" RoundEven ").count(), 3, "{asm}");
    assert_eq!(asm.matches(" Round ").count(), 0, "{asm}");
    // Metal answers 0 for a NaN component of every normalized pack -- device-measured across all
    // seven families by `pack-every-normalized-format-at-its-edges`. `NClamp(NaN, 0, 1)` is 0 by
    // specification where `FClamp` of a NaN is undefined, so the clamp produces the zero on its
    // own; there is no per-lane `OpIsNan`/`OpSelect` in front of it any more.
    assert_eq!(asm.matches("OpIsNan").count(), 0, "{asm}");
    assert_eq!(asm.matches("OpShiftLeftLogical").count(), 2, "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_air_pack_unorm_rgb10a2_rounds_the_exact_product() {
    // Metal rounds the EXACT real product `x * 1023`, not the `float` product -- measured on device
    // by `round-the-exact-product-into-ten-ten-ten-two`, where the float-product model gets 8 of 14
    // deliberately chosen arguments wrong. The exact product is recovered per lane as `p = x * M`
    // plus the FMA residual `r = fma(x, M, -p)`, so the module must carry one `Fma` per lane and the
    // multiply feeding it must be decorated `NoContraction`: a backend allowed to fuse that multiply
    // into the FMA would make the residual zero and silently restore the wrong model.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %v = load <4 x float>, ptr addrspace(1) %in, align 16
  %p = call i32 @air.pack.unorm.rgb10a2.v4f32(<4 x float> %v)
  store i32 %p, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.pack.unorm.rgb10a2.v4f32(<4 x float>)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"in"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_rgb10a2_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches(" NClamp ").count(), 4, "{asm}");
    assert_eq!(asm.matches(" FClamp ").count(), 0, "{asm}");
    assert_eq!(asm.matches(" Fma ").count(), 4, "{asm}");
    assert_eq!(asm.matches(" Floor ").count(), 4, "{asm}");
    // The NaN lane is the clamp's own answer now; see the rgb565 sibling.
    assert_eq!(asm.matches("OpIsNan").count(), 0, "{asm}");
    // Three shifts, because the red lane sits at bit zero.
    assert_eq!(asm.matches("OpShiftLeftLogical").count(), 3, "{asm}");

    let module = load_bytes(&spv).expect("load native spv");
    let contracted_free: HashSet<Word> = module
        .annotations
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.get(1)
                    == Some(&Operand::Decoration(spirv::Decoration::NoContraction))
        })
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::IdRef(target)) => Some(*target),
            _ => None,
        })
        .collect();
    let scaling_multiplies: Vec<Word> = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::FMul)
        .filter_map(|inst| inst.result_id)
        .collect();
    assert_eq!(scaling_multiplies.len(), 4, "one scaling multiply per lane");
    for multiply in scaling_multiplies {
        assert!(
            contracted_free.contains(&multiply),
            "the scaling multiply must be NoContraction or the residual collapses to zero"
        );
    }
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_sinpi_reduces_its_argument_instead_of_scaling_it_by_pi() {
    // `sinpi(x)` is not `sin(float(pi) * x)`. `float(pi)` is 8.74e-08 above pi, so the scaled form
    // misses every half-integer -- `sinpi(1)` came back -8.74e-08 where Metal answers -0.0 -- and it
    // hands a large argument to `sin`, which flushes to zero, so `cospi(1e7)` came back 0.0 where
    // Metal answers 1.0. The reduction below keeps the argument inside [-0.25, 0.25]: two RoundEven
    // steps, and BOTH of Sin and Cos, because the quadrant chooses between them.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %x = load float, ptr addrspace(1) %in, align 4
  %r = call float @air.sinpi.f32(float %x)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.sinpi.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_sinpi_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(
        asm.matches("RoundEven").count(),
        2,
        "argument is not reduced modulo 2 and then to a quadrant: {asm}"
    );
    for opcode in ["Sin", "Cos", "OpSelect", "OpFOrdEqual"] {
        assert!(asm.contains(opcode), "missing {opcode}: {asm}");
    }
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_fast_fmod_keeps_the_trunc_expression_and_not_a_remainder() {
    // Metal's `fmod` is the expression `x - y * trunc(x / y)`, not the exact C remainder, so the
    // primitives that look like it are both wrong: `OpFRem` is the exact remainder carrying the
    // dividend's sign and `OpFMod` is the floor-style modulus. Once the quotient rounds the two
    // answers are nowhere near each other -- `fast::fmod(1e10, 3)` is -512 on device where the
    // exact remainder is 1 -- and 504 corpus sources call this family.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %x = load float, ptr addrspace(1) %in, align 4
  %p = getelementptr inbounds float, ptr addrspace(1) %in, i64 1
  %y = load float, ptr addrspace(1) %p, align 4
  %r = call float @air.fast_fmod.f32(float %x, float %y)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.fast_fmod.f32(float, float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fast_fmod_trunc_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    for opcode in ["OpFDiv", "Trunc", "OpFMul", "OpFSub"] {
        assert!(asm.contains(opcode), "missing {opcode}: {asm}");
    }
    assert!(
        !asm.contains("OpFRem"),
        "exact remainder substituted: {asm}"
    );
    assert!(!asm.contains("OpFMod"), "floor modulus substituted: {asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_atomic_fence_takes_its_memory_classes_from_its_flags_operand() {
    // `atomic_thread_fence(flags, order, scope)`. The memory a fence covers is the FIRST operand,
    // the same `mem_flags` word the two barriers read; the THIRD is the scope it is ordered over.
    // Deriving the classes from the scope answered `mem_threadgroup` with 584 (device memory, no
    // workgroup memory at all) and `mem_texture` with no image memory.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out) {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  tail call void @air.atomic.fence(i32 1, i32 5, i32 2)
  tail call void @air.atomic.fence(i32 2, i32 5, i32 2)
  tail call void @air.atomic.fence(i32 4, i32 5, i32 1)
  ret void
}

declare void @air.atomic.fence(i32, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_atomic_fence_flags_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let mut constants = std::collections::HashMap::new();
    for line in asm.lines() {
        let mut words = line.split_whitespace();
        if let (Some(result), Some("="), Some("OpConstant"), Some(_), Some(value)) = (
            words.next(),
            words.next(),
            words.next(),
            words.next(),
            words.next(),
        ) {
            constants.insert(result.to_string(), value.to_string());
        }
    }
    let barriers = asm
        .lines()
        .filter_map(|line| line.strip_prefix("OpMemoryBarrier "))
        .map(|operands| {
            operands
                .split_whitespace()
                .map(|id| constants.get(id).cloned().unwrap_or_else(|| id.to_string()))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>();
    // Scope 1 is Device and 2 is Workgroup; 584 is AcquireRelease | UniformMemory |
    // CrossWorkgroupMemory, 264 is AcquireRelease | WorkgroupMemory, 2056 is AcquireRelease |
    // ImageMemory.
    assert_eq!(
        barriers,
        vec![
            "1 584".to_string(),
            "1 264".to_string(),
            "2 2056".to_string()
        ],
        "{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_coherent_air_store_and_fence_lower_to_memory_ops() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @fence_update(ptr addrspace(1) %timestamp, ptr addrspace(2) %value) {
entry:
  %v = load i32, ptr addrspace(2) %value, align 4
  %old = tail call i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1) %timestamp)
  %next = add i32 %v, %old
  tail call void @air.store.system_coherent.volatile.i32.p1i32(i32 %next, ptr addrspace(1) %timestamp)
  tail call void @air.atomic.fence(i32 1, i32 5, i32 3)
  ret void
}

declare i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1))
declare void @air.store.system_coherent.volatile.i32.p1i32(i32, ptr addrspace(1))
declare void @air.atomic.fence(i32, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @fence_update, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"timestamp"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"value"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_coherent_air_store_fence_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(asm.contains("OpMemoryBarrier"), "{asm}");
    assert!(!asm.contains("air.load.system_coherent"), "{asm}");
    assert!(!asm.contains("air.store.system_coherent"), "{asm}");
    assert!(!asm.contains("air.atomic.fence"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_call_pointee_propagation_is_monotonic_for_conflicting_helpers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %p) {
entry:
  call void @takes_i32(ptr addrspace(1) %p)
  call void @takes_float(ptr addrspace(1) %p)
  ret void
}

define void @takes_i32(ptr addrspace(1) %q) {
entry:
  %g = getelementptr inbounds i32, ptr addrspace(1) %q, i64 0
  ret void
}

define void @takes_float(ptr addrspace(1) %q) {
entry:
  %g = getelementptr inbounds float, ptr addrspace(1) %q, i64 0
  ret void
}
"#;
    let ir = super::super::ir::LlModule::parse(ll).expect("parse");
    assert_eq!(
        ir.ptr_pointees
            .get(&("takes_i32".to_string(), "%q".to_string())),
        Some(&LlType::Int(32))
    );
    assert_eq!(
        ir.ptr_pointees
            .get(&("takes_float".to_string(), "%q".to_string())),
        Some(&LlType::Float)
    );
    assert!(
        !ir.ptr_pointees
            .contains_key(&("k".to_string(), "%p".to_string())),
        "{:?}",
        ir.ptr_pointees
    );
}

#[test]
fn native_air_declaration_call_is_named_for_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define float @k(float %x) {
entry:
  %y = tail call fast float @air.fast_ceil.f32(float %x)
  ret float %y
}

declare float @air.fast_ceil.f32(float)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpName"), "{asm}");
    assert!(asm.contains("air.fast_ceil.f32"), "{asm}");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_agx2_cluster_number_is_one_uniform_physical_cluster() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
define void @k(ptr addrspace(1) %out) {
entry:
  %cluster = tail call i32 @llvm.agx2.cluster.num()
  store i32 %cluster, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.agx2.cluster.num()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_agx2_cluster_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [10, 8, 1],
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // Device-measured on an M3 Max: the value is the same for every thread of a threadgroup and
    // varies between threadgroups over 0..=36. It is NOT a function of the invocation's position,
    // so nothing about the local id or the local size may appear in it — a per-thread answer would
    // make a predicate the hardware guarantees uniform diverge inside a threadgroup.
    assert!(!asm.contains("BuiltIn LocalInvocationId"), "{asm}");
    assert!(!asm.contains("llvm.agx2.cluster.num"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("OpIMul"), "{asm}");
    assert!(!asm.contains("OpCompositeExtract"), "{asm}");
    // The stored value is a copy of the zero constant, so it is uniform by construction.
    let zero = asm
        .lines()
        .find_map(|line| {
            let mut words = line.split_whitespace();
            let result = words.next()?;
            (words.next()? == "=" && words.next()? == "OpConstant" && words.nth(1)? == "0")
                .then_some(result)
        })
        .unwrap_or_else(|| panic!("{asm}"));
    assert!(
        asm.lines()
            .any(|line| line.contains("OpCopyObject")
                && line.split_whitespace().last() == Some(zero)),
        "{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_void_air_call_is_emitted_for_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @discard() {
entry:
  tail call void @air.discard_fragment()
  ret void
}

declare void @air.discard_fragment()
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("air.discard_fragment"), "{asm}");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_visible_function_table_indirect_group_call_is_noop() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %table = inttoptr i64 0 to ptr addrspace(1)
  %size = tail call i32 @air.get_size_visible_function_table(ptr addrspace(1) %table)
  %fp = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %size)
  %cast = bitcast ptr %fp to ptr
  tail call void %cast(ptr addrspace(2) null) #1, !air.function_groups !0
  ret void
}

declare i32 @air.get_size_visible_function_table(ptr addrspace(1))
declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)

attributes #1 = { convergent nobuiltin nounwind "no-builtins" }
!air.function_groups = !{!0}
!0 = !{!"air.function_group", !"rayGen"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_visible_function_group_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(
        !asm.contains("air.get_size_visible_function_table"),
        "{asm}"
    );
    assert!(
        !asm.contains("air.get_function_pointer_visible_function_table"),
        "{asm}"
    );
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_visible_function_table_value_call_reports_indirect_function_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <4 x float> @main() {
entry:
  %table = inttoptr i64 0 to ptr addrspace(1)
  %fp = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)
  %cast = bitcast ptr %fp to ptr
  %result = call fast <4 x float> %cast(ptr addrspace(2) null)
  ret <4 x float> %result
}

declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)
"#;
    let err = emit_vulkan_spirv(ll).expect_err("indirect value call should be unsupported");
    assert!(err.contains("unsupported indirect call"), "{err}");
    assert!(err.contains("function pointer %cast"), "{err}");
    assert!(!err.contains("graph_walk_unmigrated_opcode"), "{err}");
}

#[test]
fn native_visible_function_reference_fails_before_retry_cascade() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  tail call void @postfixPrimary_f.MTL_VISIBLE_FN_REF(ptr addrspace(2) null)
  ret void
}

declare void @postfixPrimary_f.MTL_VISIBLE_FN_REF(ptr addrspace(2)) section "air.externally_defined"
!air.visible_function_references = !{!0}
!0 = !{!"air.visible_function_reference", ptr @postfixPrimary_f.MTL_VISIBLE_FN_REF, !"postfixPrimary_f"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_visible_function_ref_{}",
        std::process::id()
    ));
    let err = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp)
        .expect_err("direct Metal visible function references are unsupported");
    assert!(
        err.contains("unsupported Metal visible function reference"),
        "{err}"
    );
    assert!(err.contains("Logical SPIR-V"), "{err}");
}

/// A post-tessellation vertex function: `air.patch` states the domain and control-point count, and
/// `air.patch_control_point_input` names the per-control-point fetch the evaluation stage calls.
const PATCH_CONTROL_POINT_LL: &str = r#"
target triple = "spirv-unknown-vulkan1.2"
define <{ <4 x float> }> @main(ptr %patch, <2 x float> %position_in_patch) {
entry:
  %cp = tail call { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32 0, ptr %patch)
  %pos = insertvalue <{ <4 x float> }> undef, <4 x float> zeroinitializer, 0
  ret <{ <4 x float> }> %pos
}

declare { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32, ptr) section "air.externally_defined"
!air.vertex = !{!0}
!0 = !{ptr @main, !1, !2, !7}
!1 = !{!3}
!2 = !{!4, !8}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{!"air.patch", !"triangle", !"air.patch_control_point", i32 3}
!8 = !{i32 1, !"air.position_in_patch", !"air.arg_type_name", !"float2"}
"#;

#[test]
fn native_patch_control_point_reference_lowers_to_tessellation_evaluation() {
    let ll = PATCH_CONTROL_POINT_LL;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_patch_control_point_ref_{}",
        std::process::id()
    ));
    let out = crate::translate_sanitized_native(ll, Stage::Vertex, &tmp)
        .expect("patch control points lower through tessellation evaluation");
    let asm = disassemble(&out).expect("disassemble tessellation evaluation module");
    assert!(asm.contains("OpEntryPoint TessellationEvaluation"), "{asm}");
    assert!(
        asm.contains("OpExecutionMode") && asm.contains("Triangles"),
        "{asm}"
    );
    assert!(asm.contains("BuiltIn TessCoord"), "{asm}");
    assert!(!asm.contains("MTL_CONTROL_POINT_FN"), "{asm}");
}

/// The same post-tessellation function, reading `[[patch_id]]` -- which is `PrimitiveId`.
///
/// `PrimitiveId` is enabled by any of `Geometry`, `Tessellation`, `RayTracingKHR` or
/// `MeshShadingEXT`. A tessellation-evaluation entry already declares `Tessellation`, so the
/// requirement is satisfied before anything is added for the builtin -- and adding `Geometry`
/// anyway, as this used to, demanded `geometryShader` of every consumer for nothing. 226 of the 236
/// corpus modules that carried `Geometry` were exactly this shape, and Metal has no geometry stage,
/// so none of them could be loaded on the platform they came from.
#[test]
fn native_tessellation_patch_id_adds_no_geometry_capability() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <{ <4 x float> }> @main(ptr %patch, <2 x float> %position_in_patch, i32 %pid) {
entry:
  %cp = tail call { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32 0, ptr %patch)
  %f = uitofp i32 %pid to float
  %v = insertelement <4 x float> zeroinitializer, float %f, i64 0
  %pos = insertvalue <{ <4 x float> }> undef, <4 x float> %v, 0
  ret <{ <4 x float> }> %pos
}

declare { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32, ptr) section "air.externally_defined"
!air.vertex = !{!0}
!0 = !{ptr @main, !1, !2, !7}
!1 = !{!3}
!2 = !{!4, !8, !9}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{!"air.patch", !"triangle", !"air.patch_control_point", i32 3}
!8 = !{i32 1, !"air.position_in_patch", !"air.arg_type_name", !"float2"}
!9 = !{i32 2, !"air.patch_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"pid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_tess_primitive_id_{}",
        std::process::id()
    ));
    let out = crate::translate_sanitized_native(ll, Stage::Vertex, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble");
    assert!(asm.contains("OpEntryPoint TessellationEvaluation"), "{asm}");
    assert!(asm.contains("BuiltIn PrimitiveId"), "{asm}");
    assert!(asm.contains("OpCapability Tessellation"), "{asm}");
    assert!(!asm.contains("OpCapability Geometry"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

/// The same function with an `air.patch` node naming a domain the translator does not model. Metal
/// would still run it as a post-tessellation evaluation shader; dropping the shape turns it into an
/// ordinary vertex shader that validates, binds and reflects while drawing the wrong geometry.
#[test]
fn native_unreadable_patch_domain_is_refused_rather_than_dropped() {
    let ll = PATCH_CONTROL_POINT_LL.replace("!\"triangle\"", "!\"tetrahedron\"");
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_unreadable_patch_domain_{}",
        std::process::id()
    ));
    let err = crate::translate_sanitized_native(&ll, Stage::Vertex, &tmp)
        .expect_err("a patch domain with no lowering must not become an ordinary vertex shader");
    assert!(err.contains("tessellation patch"), "{err}");
    assert!(err.contains("no tessellation domain"), "{err}");
}

/// An `air.patch` node that states a domain but no control-point count is the same hazard: the
/// count is what sizes every per-patch input the pipeline wires.
#[test]
fn native_patch_without_a_control_point_count_is_refused() {
    let ll = PATCH_CONTROL_POINT_LL.replace(
        "!\"air.patch\", !\"triangle\", !\"air.patch_control_point\", i32 3",
        "!\"air.patch\", !\"triangle\"",
    );
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_patch_no_control_point_{}",
        std::process::id()
    ));
    let err = crate::translate_sanitized_native(&ll, Stage::Vertex, &tmp).expect_err(
        "a patch with no control-point count must not become an ordinary vertex shader",
    );
    assert!(err.contains("air.patch_control_point"), "{err}");
}

/// The patch node is found by what it says, not by where it sits. Metal writes it third in the
/// vertex root today; a root that grows another entry must not turn a tessellation shader into a
/// plain vertex one, nor read a non-patch node as a patch.
#[test]
fn native_patch_node_is_found_by_its_marker_not_its_position() {
    let ll = PATCH_CONTROL_POINT_LL
        .replace("!0 = !{ptr @main, !1, !2, !7}", "!0 = !{ptr @main, !1, !2, !9, !7}")
        .replace(
            "!8 = !{i32 1, !\"air.position_in_patch\", !\"air.arg_type_name\", !\"float2\"}",
            "!8 = !{i32 1, !\"air.position_in_patch\", !\"air.arg_type_name\", !\"float2\"}\n!9 = !{}",
        );
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_patch_marker_position_{}",
        std::process::id()
    ));
    let out = crate::translate_sanitized_native(&ll, Stage::Vertex, &tmp)
        .expect("the patch node is still found past an unrelated root entry");
    let asm = disassemble(&out).expect("disassemble tessellation evaluation module");
    assert!(asm.contains("OpEntryPoint TessellationEvaluation"), "{asm}");
    assert!(asm.contains("Triangles"), "{asm}");
}

#[test]
fn native_reverse_bits_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %rev = tail call i32 @air.reverse_bits.i32(i32 305419896)
  ret void
}

declare i32 @air.reverse_bits.i32(i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_reverse_bits_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpBitReverse"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_llvm_bswap_i32_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i32 %x) {
entry:
  %swapped = tail call i32 @llvm.bswap.i32(i32 %x)
  ret void
}

declare i32 @llvm.bswap.i32(i32)
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_bswap_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_llvm_abs_i32_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i32 %x) {
entry:
  %abs = tail call i32 @llvm.abs.i32(i32 %x, i1 true)
  ret void
}

declare i32 @llvm.abs.i32(i32, i1 immarg)
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_native_abs_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpExtInst"), "{asm}");
    assert!(asm.contains("SAbs"), "{asm}");
    assert!(!asm.contains("llvm.abs"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_rotate_i32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i32 %x, i32 %shift) {
entry:
  %rot = tail call i32 @air.rotate.i32(i32 %x, i32 %shift)
  ret void
}

declare i32 @air.rotate.i32(i32, i32)
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_rotate_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_llvm_fshl_lowers_every_spirv_integer_width() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %rot8 = call i8 @llvm.fshl.i8(i8 18, i8 52, i8 3)
  %rot16 = call i16 @llvm.fshl.i16(i16 4660, i16 22136, i16 5)
  %rot32 = call i32 @llvm.fshl.i32(i32 305419896, i32 2596069104, i32 7)
  %rot64 = call i64 @llvm.fshl.i64(i64 1311768467463790320, i64 -81985529216486896, i64 11)
  ret void
}

declare i8 @llvm.fshl.i8(i8, i8, i8)
declare i16 @llvm.fshl.i16(i16, i16, i16)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i64 @llvm.fshl.i64(i64, i64, i64)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_fshl_widths_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.matches("OpShiftLeftLogical").count() >= 4, "{asm}");
    assert!(asm.matches("OpShiftRightLogical").count() >= 4, "{asm}");
    assert!(asm.matches("OpBitwiseOr").count() >= 4, "{asm}");
    assert!(!asm.contains("llvm_fshl"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_fragment_derivatives_and_dot_are_typed_at_construction() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <4 x float> @frag() {
entry:
  %dx = tail call float @air.dfdx.f32(float 2.000000e+00)
  %width_half = tail call half @air.fwidth.f16(half 0xH3C00)
  %width = fpext half %width_half to float
  %dot = tail call float @air.dot.v2f32(<2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float> <float 3.000000e+00, float 4.000000e+00>)
  %sum0 = fadd float %dx, %width
  %sum1 = fadd float %sum0, %dot
  %lane = insertelement <4 x float> poison, float %sum1, i32 0
  %color = shufflevector <4 x float> %lane, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %color
}

declare float @air.dfdx.f32(float)
declare half @air.fwidth.f16(half)
declare float @air.dot.v2f32(<2 x float>, <2 x float>)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_derivative_dot_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpDPdx"), "{asm}");
    assert!(asm.contains("OpFwidth"), "{asm}");
    assert!(asm.contains("OpDot"), "{asm}");
    assert!(asm.matches("OpFConvert").count() >= 2, "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_dot_rejects_nonvector_air_operands_before_assembly() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <4 x float> @frag() {
entry:
  %dot = tail call float @air.dot.f32(float 1.000000e+00, float 2.000000e+00)
  %lane = insertelement <4 x float> poison, float %dot, i32 0
  %color = shufflevector <4 x float> %lane, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %color
}

declare float @air.dot.f32(float, float)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_invalid_dot_{}",
        std::process::id()
    ));
    let error = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp)
        .expect_err("scalar air.dot operands must fail during construction");
    assert!(
        error.contains("operands are not identical vectors of its result component type"),
        "{error}"
    );
}

#[test]
fn native_extract_bits_u32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i32 @air.extract_bits.u.i32(i32 305419896, i32 4, i32 8)
  ret void
}

declare i32 @air.extract_bits.u.i32(i32, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_extract_bits_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpBitFieldUExtract"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_wide_multiply_family_splits_on_sign_and_on_what_it_takes_back() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %hi_s = tail call i32 @air.mul_hi.s.i32(i32 -7, i32 305419896)
  %hi_u = tail call i32 @air.mul_hi.u.i32(i32 -7, i32 305419896)
  %mad_hi = tail call i32 @air.mad_hi.s.i32(i32 -7, i32 305419896, i32 %hi_u)
  %mad_sat = tail call i32 @air.mad_sat.u.i32(i32 -7, i32 305419896, i32 %mad_hi)
  ret void
}

declare i32 @air.mul_hi.s.i32(i32, i32)
declare i32 @air.mul_hi.u.i32(i32, i32)
declare i32 @air.mad_hi.s.i32(i32, i32, i32)
declare i32 @air.mad_sat.u.i32(i32, i32, i32)
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    // The sign lives in the widening, the narrowing and the shift: an unsigned member must never
    // sign-extend its operands into the 64-bit product, and a signed one must never zero-extend.
    assert!(asm.contains("OpSConvert"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpShiftRightArithmetic"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    // `mad_hi` wraps its addend into the narrowed high half; `mad_sat` clamps the 64-bit sum.
    assert!(asm.contains("UClamp"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_narrow_bitfield_intrinsics_widen_the_vulkan_base_to_i32() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %extracted = tail call i16 @air.extract_bits.s.i16(i16 4660, i32 4, i32 8)
  %inserted = tail call i16 @air.insert_bits.u.i16(i16 %extracted, i16 10, i32 4, i32 8)
  %vector_extracted = tail call <2 x i16> @air.extract_bits.u.v2i16(<2 x i16> <i16 4660, i16 22136>, i32 4, i32 8)
  %vector_inserted = tail call <2 x i16> @air.insert_bits.u.v2i16(<2 x i16> %vector_extracted, <2 x i16> <i16 10, i16 11>, i32 4, i32 8)
  ret void
}

declare i16 @air.extract_bits.s.i16(i16, i32, i32)
declare i16 @air.insert_bits.u.i16(i16, i16, i32, i32)
declare <2 x i16> @air.extract_bits.u.v2i16(<2 x i16>, i32, i32)
declare <2 x i16> @air.insert_bits.u.v2i16(<2 x i16>, <2 x i16>, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_narrow_bitfields_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpBitFieldSExtract"), "{asm}");
    assert!(asm.contains("OpBitFieldUExtract"), "{asm}");
    assert!(asm.matches("OpBitFieldInsert").count() >= 2, "{asm}");
    assert!(asm.matches("OpUConvert").count() >= 10, "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_extract_bits_u64_avoids_maintenance9_bitfield_opcode() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i64 @air.extract_bits.u.i64(i64 81985529216486895, i32 40, i32 8)
  ret void
}

declare i64 @air.extract_bits.u.i64(i64, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_extract_bits_u64_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpBitFieldUExtract"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_insert_bits_u32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i32 @air.insert_bits.u.i32(i32 305419896, i32 10, i32 4, i32 8)
  ret void
}

declare i32 @air.insert_bits.u.i32(i32, i32, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_insert_bits_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpBitFieldInsert"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_insert_bits_u64_avoids_maintenance9_bitfield_opcode() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i64 @air.insert_bits.u.i64(i64 81985529216486895, i64 10, i32 40, i32 8)
  ret void
}

declare i64 @air.insert_bits.u.i64(i64, i64, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_insert_bits_u64_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpBitFieldInsert"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_ctz_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @air.ctz.i32(i32 305419896, i1 false)
  ret void
}

declare i32 @air.ctz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_native_ctz_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpBitCount"), "{asm}");
    assert!(!asm.contains("FindILsb"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_llvm_cttz_i32_zero_undef_lowers_to_find_lsb() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @llvm.cttz.i32(i32 305419896, i1 true)
  ret void
}

declare i32 @llvm.cttz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_cttz_zero_undef_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpExtInst"), "{asm}");
    assert!(asm.contains("FindILsb"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_llvm_cttz_i32_defined_zero_lowers_to_select_width() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @llvm.cttz.i32(i32 305419896, i1 false)
  ret void
}

declare i32 @llvm.cttz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_cttz_defined_zero_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpExtInst"), "{asm}");
    assert!(asm.contains("FindILsb"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_clz_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @air.clz.i32(i32 305419896, i1 false)
  ret void
}

declare i32 @air.clz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_native_clz_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpBitCount"), "{asm}");
    assert!(!asm.contains("FindUMsb"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_fast_tan_and_fmod_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %tan = tail call fast float @air.fast_tan.f32(float 5.000000e-01)
  %v0 = insertelement <2 x float> poison, float 5.500000e+00, i32 0
  %v1 = insertelement <2 x float> %v0, float -5.500000e+00, i32 1
  %d0 = insertelement <2 x float> poison, float 2.000000e+00, i32 0
  %d1 = insertelement <2 x float> %d0, float 2.000000e+00, i32 1
  %fmodv = tail call fast <2 x float> @air.fast_fmod.v2f32(<2 x float> %v1, <2 x float> %d1)
  %fmods = tail call fast float @air.fast_fmod.f32(float -5.500000e+00, float 2.000000e+00)
  %lane = extractelement <2 x float> %fmodv, i32 0
  %sum0 = fadd fast float %tan, %fmods
  %sum1 = fadd fast float %sum0, %lane
  %sink = fcmp oge float %sum1, 0.000000e+00
  ret void
}

declare float @air.fast_tan.f32(float)
declare <2 x float> @air.fast_fmod.v2f32(<2 x float>, <2 x float>)
declare float @air.fast_fmod.f32(float, float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_tan_fmod_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" Tan "), "{asm}");
    assert!(asm.contains(" Trunc "), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpFSub"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

/// `cospi`, `sinpi` and `tanpi` all reduce; none of them pre-multiplies by `float(pi)`.
#[test]
fn native_fast_pi_transcendentals_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %cos = tail call fast float @air.fast_cospi.f32(float 5.000000e-01)
  %sin = tail call fast float @air.fast_sinpi.f32(float 2.500000e-01)
  %tan = tail call fast float @air.fast_tanpi.f32(float 1.250000e-01)
  %sum0 = fadd fast float %cos, %sin
  %sum1 = fadd fast float %sum0, %tan
  %sink = fcmp oge float %sum1, 0.000000e+00
  ret void
}

declare float @air.fast_cospi.f32(float)
declare float @air.fast_sinpi.f32(float)
declare float @air.fast_tanpi.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_pi_transcendentals_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" Cos "), "{asm}");
    assert!(asm.contains(" Sin "), "{asm}");
    // None of the three is a pre-multiplied base transcendental. `tanpi` in particular is the
    // quotient of the sinpi and cospi quadrant arms, not `tan(pi*x)`: `tan` has no pole where
    // `tanpi` does, and `tan(pi*0.5)` is -2.28e7 where Metal answers +inf.
    assert!(!asm.contains(" Tan "), "{asm}");
    assert_eq!(asm.matches("OpFDiv").count(), 1, "{asm}");
    assert!(asm.matches("OpFMul").count() >= 3, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_fast_sincos_f32_zeroes_large_arguments() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(float %x) {
entry:
  %cos = tail call fast float @air.fast_cos.f32(float %x)
  %sin = tail call fast float @air.fast_sin.f32(float %x)
  %sum = fadd fast float %cos, %sin
  %sink = fcmp oge float %sum, 0.000000e+00
  ret void
}

declare float @air.fast_cos.f32(float)
declare float @air.fast_sin.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_sincos_large_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" FAbs "), "{asm}");
    assert!(asm.contains(" Cos "), "{asm}");
    assert!(asm.contains(" Sin "), "{asm}");
    // The flush is the whole difference from GLSL Sin/Cos: the argument reaches the trig call
    // unreduced. A `x - trunc(x / 2pi) * 2pi` reduction ahead of it -- which this lowering used to
    // emit -- rounds twice and put `fast::sin(100)` 38 ULP from Metal's answer.
    assert!(!asm.contains(" Trunc "), "{asm}");
    assert!(!asm.contains("OpFDiv"), "{asm}");
    assert!(!asm.contains("OpFSub"), "{asm}");
    // Both trig calls and both flush compares read the same one value, the entry's own argument.
    let operand_of = |needle: &str| {
        asm.lines()
            .find(|line| line.contains(needle))
            .and_then(|line| line.split_whitespace().last())
            .map(str::to_string)
            .unwrap_or_else(|| panic!("no {needle} in {asm}"))
    };
    let argument = operand_of(" FAbs ");
    assert_eq!(operand_of(" Sin "), argument, "{asm}");
    assert_eq!(operand_of(" Cos "), argument, "{asm}");
    // The flush compare is UNORDERED so a NaN argument takes the zero arm, as the hardware does,
    // and the edge is the last magnitude the hardware still evaluates: float(pi/2) * 2^22.
    assert!(asm.contains("OpFUnordGreaterThan"), "{asm}");
    // The only ordered compare left is the shader's own `fcmp oge` sink.
    assert_eq!(asm.matches("OpFOrdGreaterThanEqual").count(), 1, "{asm}");
    assert_eq!(asm.matches("OpFUnordGreaterThan").count(), 2, "{asm}");
    assert!(asm.contains("6588397"), "{asm}");
    assert!(asm.matches("OpSelect").count() >= 2, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_fast_ldexp_lowers_to_the_glsl_ldexp() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %scaled = tail call fast float @air.fast_ldexp.f32(float 1.250000e+00, i32 3)
  %sink = fcmp oge float %scaled, 0.000000e+00
  ret void
}

declare float @air.fast_ldexp.f32(float, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_ldexp_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    // GLSL scales the exponent field. Materializing the scale as its own float32 --
    // `x * exp2(float(n))`, which this used to emit -- overflows for n above 127 and flushes to
    // zero below -126 even when the product is an ordinary number, so `ldexp(1e-30, 200)` came back
    // `inf` where Metal answers 1.6069381e30.
    assert!(asm.contains(" Ldexp "), "{asm}");
    assert!(!asm.contains("OpConvertSToF"), "{asm}");
    assert!(!asm.contains(" Exp2 "), "{asm}");
    assert!(!asm.contains("OpFMul"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_llvm_fabs_lowers_to_extinst() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %abs = tail call fast float @llvm.fabs.f32(float -1.250000e+00)
  %sink = fcmp oge float %abs, 0.000000e+00
  ret void
}

declare float @llvm.fabs.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_fabs_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" FAbs "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_fast_atan_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %atan = tail call fast float @air.fast_atan.f32(float 5.000000e-01)
  %sink = fcmp oge float %atan, 0.000000e+00
  ret void
}

declare float @air.fast_atan.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_atan_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" Atan "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

#[test]
fn native_ignores_lifetime_intrinsics_with_generic_ptrs() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @life() {
entry:
  call void @llvm.lifetime.start.p0(ptr undef)
  call void @llvm.lifetime.end.p0(ptr undef)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(!asm.contains("llvm.lifetime"), "{asm}");
    assert!(asm.contains("OpReturn"), "{asm}");
}

/// Metal `round` breaks a `.5` tie away from zero, which GLSL.std.450 `Round` leaves to the
/// implementation. The lowering must therefore be the explicit trunc/remainder form and must NOT
/// reach for the ext-inst; `rint` keeps `RoundEven`, whose tie rule already matches Metal's.
#[test]
fn native_round_breaks_ties_away_from_zero_without_glsl_round() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(float %x, <4 x float> %v) {
entry:
  %r = tail call float @air.round.f32(float %x)
  %rv = tail call <4 x float> @air.fast_round.v4f32(<4 x float> %v)
  %e = tail call float @air.rint.f32(float %x)
  %lane = extractelement <4 x float> %rv, i32 0
  %sum = fadd float %r, %lane
  %sum2 = fadd float %sum, %e
  %sink = fcmp oge float %sum2, 0.000000e+00
  ret void
}

declare float @air.round.f32(float)
declare <4 x float> @air.fast_round.v4f32(<4 x float>)
declare float @air.rint.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_round_ties_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(
        !asm.contains(" Round "),
        "GLSL Round leaves the .5 tie direction to the implementation: {asm}"
    );
    assert!(
        asm.contains(" RoundEven "),
        "air.rint keeps GLSL RoundEven: {asm}"
    );
    // Two calls (scalar + vector) each expand to trunc, remainder, both edge compares and two
    // selects feeding one add.
    assert_eq!(asm.matches(" Trunc ").count(), 2, "{asm}");
    assert_eq!(
        asm.matches("OpFOrdLessThanEqual").count(),
        2,
        "one lower edge compare per call: {asm}"
    );
    // Two upper edge compares plus the shader's own `fcmp oge` sink.
    assert_eq!(asm.matches("OpFOrdGreaterThanEqual").count(), 3, "{asm}");
    assert_eq!(asm.matches("OpSelect").count(), 4, "{asm}");
    assert!(asm.contains("OpFSub"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A half `air.round` widens to f32 for the tie arithmetic and narrows back, so the emitted module
/// carries the round-trip conversions and still avoids GLSL `Round`.
#[test]
fn native_round_half_round_trips_through_float() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(half %x, <2 x half> %v) {
entry:
  %r = tail call half @air.round.f16(half %x)
  %rv = tail call <2 x half> @air.round.v2f16(<2 x half> %v)
  %lane = extractelement <2 x half> %rv, i32 0
  %sum = fadd half %r, %lane
  %sink = fcmp oge half %sum, 0xH0000
  ret void
}

declare half @air.round.f16(half)
declare <2 x half> @air.round.v2f16(<2 x half>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_round_half_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains(" Round "), "{asm}");
    assert_eq!(asm.matches(" Trunc ").count(), 2, "{asm}");
    assert!(
        asm.matches("OpFConvert").count() >= 4,
        "widen in and narrow out for both calls: {asm}"
    );
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `air.pow` and `air.fast_pow` share one lowering at every float width — the precise f32 spelling
/// used to fall through to a bare GLSL `Pow`, which is undefined for the negative bases Metal
/// defines. `air.powr` is a different contract (`x >= 0`) and keeps the bare ext-inst.
#[test]
fn native_pow_and_powr_take_different_paths_at_f32() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(float %x, float %y) {
entry:
  %p = tail call float @air.pow.f32(float %x, float %y)
  %r = tail call float @air.powr.f32(float %x, float %y)
  %sum = fadd float %p, %r
  %sink = fcmp oge float %sum, 0.000000e+00
  ret void
}

declare float @air.pow.f32(float, float)
declare float @air.powr.f32(float, float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pow_powr_f32_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    // Two Pow ext-insts: one bare (powr), one wrapped in the parity/domain form (pow).
    assert_eq!(asm.matches(" Pow ").count(), 2, "{asm}");
    // Two: `air.pow` takes the magnitude of its base to keep `Pow` in its domain, and the
    // magnitude of its EXPONENT to ask whether the exponent is finite. `air.powr` takes neither.
    assert_eq!(
        asm.matches(" FAbs ").count(),
        2,
        "only air.pow takes a magnitude: {asm}"
    );
    assert_eq!(
        asm.matches("OpFNegate").count(),
        1,
        "only air.pow reapplies the parity sign: {asm}"
    );
    assert!(!asm.contains("OpFConvert"), "f32 needs no widening: {asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Every `FPFastMathMode` decoration in a disassembly, as `result id -> mask spelling`.
fn float_math_modes(asm: &str) -> std::collections::HashMap<String, String> {
    asm.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("OpDecorate ")?;
            let (target, decoration) = rest.split_once(' ')?;
            let mask = decoration.trim().strip_prefix("FPFastMathMode")?;
            Some((target.to_string(), mask.trim().to_string()))
        })
        .collect()
}

/// The result ids of a disassembly's arithmetic instructions, in source order.
fn float_arithmetic_ids(asm: &str, opcodes: &[&str]) -> Vec<String> {
    asm.lines()
        .filter(|line| opcodes.iter().any(|op| line.contains(op)))
        .filter_map(|line| line.trim().split(' ').next().map(str::to_string))
        .collect()
}

#[test]
fn a_float_op_states_exactly_the_relaxations_its_flag_run_grants() {
    // SPIR-V has no per-module math mode, so an undecorated `OpFMul` permits everything and
    // MoltenVK -- whose own fast-math default is on -- takes the permission. `NoContraction` used
    // to stand in for the whole flag run, and it can only say one of the six things a flag run
    // says: it forbids fusion and leaves REASSOCIATION permitted, which Metal does not.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define float @sums(float %a, float %b, float %c) {
entry:
  %plain = fmul float %a, %b
  %plain_sum = fadd float %plain, %c
  %precise = fmul nnan ninf nsz arcp afn float %a, %b
  %precise_sum = fsub nnan ninf nsz arcp afn float %c, %precise
  %allowed = fmul reassoc nsz arcp contract afn float %a, %b
  %allowed_sum = fadd reassoc nsz arcp contract afn float %allowed, %c
  %s0 = fadd fast float %plain_sum, %precise_sum
  %s1 = fadd fast float %s0, %allowed_sum
  ret float %s1
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let modes = float_math_modes(&asm);
    let arithmetic = float_arithmetic_ids(&asm, &["OpFMul", "OpFAdd", "OpFSub"]);
    // In source order: the two unflagged ops, the two `nnan ninf nsz arcp afn` ops, the two spelled
    // `reassoc nsz arcp contract afn`, then the two spelled `fast`.
    assert_eq!(arithmetic.len(), 8, "{asm}");
    for id in &arithmetic[..2] {
        assert_eq!(modes.get(id).map(String::as_str), Some("None"), "{asm}");
    }
    for id in &arithmetic[2..4] {
        assert_eq!(
            modes.get(id).map(String::as_str),
            Some("NotNaN|NotInf|NSZ|AllowRecip"),
            "{asm}"
        );
    }
    // `reassoc nsz arcp contract afn` and `fast` both grant every rewrite, so an undecorated
    // instruction already says what they say. `nnan`/`ninf` are withheld by the first group and are
    // not worth the extension: censused corpus-wide, a withheld one is observable on exactly three
    // instructions in three sources.
    for id in &arithmetic[4..] {
        assert!(!modes.contains_key(id), "{id} wrongly decorated in {asm}");
    }
    assert_eq!(modes.len(), 4, "{asm}");
    assert!(!asm.contains("NoContraction"), "{asm}");
}

#[test]
fn native_inlined_helper_keeps_its_float_math_decorations() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @scale_add(ptr addrspace(1) %out, ptr addrspace(1) %input) {
entry:
  tail call fastcc void @helper(ptr addrspace(1) %out, ptr addrspace(1) %input)
  ret void
}

define internal fastcc void @helper(ptr addrspace(1) %out, ptr addrspace(1) %input) {
entry:
  %a = load float, ptr addrspace(1) %input, align 4
  %p = fmul float %a, %a
  %s = fadd float %p, %a
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @scale_add, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let modes = float_math_modes(&asm);
    // The multiply and the add live in the helper, which the emitted inliner splices into the entry
    // under fresh ids before the now-unreferenced helper is pruned. Both survivors must still say
    // Metal was given no permission to fuse or regroup them.
    let arithmetic = float_arithmetic_ids(&asm, &["OpFMul", "OpFAdd"]);
    assert_eq!(arithmetic.len(), 2, "{asm}");
    for id in &arithmetic {
        assert_eq!(modes.get(id).map(String::as_str), Some("None"), "{asm}");
    }
    assert_eq!(modes.len(), 2, "{asm}");
}

#[test]
fn native_division_without_reciprocal_permission_pins_its_fast_math_mode() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define float @quotients(float %a, float %b) {
entry:
  %exact = fdiv float %a, %b
  %approx = fdiv fast float %a, %b
  %also_approx = fdiv nnan ninf nsz arcp afn float %a, %b
  %s0 = fadd fast float %exact, %approx
  %s1 = fadd fast float %s0, %also_approx
  ret float %s1
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let modes = float_math_modes(&asm);
    let divisions = float_arithmetic_ids(&asm, &["OpFDiv"]);
    // In source order: the unflagged division, the one spelled `fast`, and the one that grants the
    // reciprocal by name while still withholding fusion and reassociation.
    assert_eq!(divisions.len(), 3, "{asm}");
    assert_eq!(
        modes.get(&divisions[0]).map(String::as_str),
        Some("None"),
        "{asm}"
    );
    assert!(!modes.contains_key(&divisions[1]), "{asm}");
    assert_eq!(
        modes.get(&divisions[2]).map(String::as_str),
        Some("NotNaN|NotInf|NSZ|AllowRecip"),
        "{asm}"
    );
    assert!(asm.contains("OpCapability FloatControls2"), "{asm}");
    assert!(asm.contains("SPV_KHR_float_controls2"), "{asm}");
}

#[test]
fn native_device_atomic_on_an_unmodelled_pointer_keeps_device_scope() {
    // A CAS-loop `atomic_float_sum` helper: the caller GEPs into a device buffer, the helper
    // bitcasts the argument and atomics through it. The bitcast mints a Private placeholder, so the
    // pointer's name lands in BOTH `unmodeled_pointers` and `raw_offsets` -- and the raw entry is
    // the one that carries the real addressing, here a device address.
    //
    // `atomic_i32_pointer_id` tests `raw_offsets` first and emits a PhysicalStorageBuffer pointer.
    // `atomic_i32_scope_for_arg` used to test `unmodeled_pointers` first and answer `Workgroup`, so
    // the atomic was scoped to one threadgroup -- not atomic against any other threadgroup -- and
    // its semantics named WorkgroupMemory for a device-memory access. 8 corpus modules, 32 atomics.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out, <3 x i32> %tid) {
entry:
  %i = extractelement <3 x i32> %tid, i64 0
  %idx = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx
  tail call fastcc void @helper(ptr addrspace(1) %p)
  ret void
}

define internal fastcc void @helper(ptr addrspace(1) %0) unnamed_addr {
entry:
  %e = alloca i32, align 4
  %b = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %old = tail call i32 @air.atomic.global.load.i32(ptr addrspace(1) %b, i32 0, i32 2, i1 true)
  %sum = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %b, i32 %old, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_device_atomic_unmodelled_scope_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    // Sanitize first: the pre-sanitized text takes a different pointer path and never reaches the
    // branch this test guards, so translating it directly would assert nothing.
    let (san, _) = crate::tools::sanitize_ll_text_with_datalayout(ll);
    let spv = crate::translate_sanitized_native(&san, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");

    let mut constants = std::collections::HashMap::new();
    let mut pointer_class = std::collections::HashMap::new();
    let mut result_type = std::collections::HashMap::new();
    for line in asm.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [result, "=", "OpConstant", _, value] => {
                constants.insert(result.to_string(), value.to_string());
            }
            [result, "=", "OpTypePointer", storage, _] => {
                pointer_class.insert(result.to_string(), storage.to_string());
            }
            [result, "=", opcode, ty, ..] if opcode.starts_with("Op") => {
                result_type.insert(result.to_string(), ty.to_string());
            }
            _ => {}
        }
    }

    // Both atomics address the buffer through a device address, so both must name Device scope (1)
    // and relaxed semantics (0) -- never Workgroup (2) with the WorkgroupMemory bit (256) set.
    let mut seen = 0usize;
    for line in asm.lines() {
        let Some((_, rest)) = line.trim().split_once(" = OpAtomic") else {
            continue;
        };
        // <op> <result type> <pointer> <scope> <semantics> ...
        let words: Vec<&str> = rest.split_whitespace().collect();
        let (pointer, scope, semantics) = (words[2], words[3], words[4]);
        let storage = result_type
            .get(pointer)
            .and_then(|ty| pointer_class.get(ty))
            .map(String::as_str);
        assert_eq!(
            storage,
            Some("PhysicalStorageBuffer"),
            "the atomic must address the device-address pointer, not {storage:?}\n{asm}"
        );
        assert_eq!(
            constants.get(scope).map(String::as_str),
            Some("1"),
            "a device-address atomic must be Device-scoped\n{asm}"
        );
        assert_eq!(
            constants.get(semantics).map(String::as_str),
            Some("0"),
            "a device-address atomic must not name WorkgroupMemory\n{asm}"
        );
        seen += 1;
    }
    assert_eq!(seen, 2, "expected both atomics to survive\n{asm}");

    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_load_through_a_selected_device_address_is_a_load_not_a_zero() {
    // A `select` between two pointers RECOVERED from memory, under the physical-address model.
    //
    // Two facts put the module in that model: the pointers are parked in a local aggregate and read
    // back, so each is its own 64-bit address, and the integer atomic over a float slot is what
    // makes `requires_device_address_model` choose the physical representation at all -- Logical
    // SPIR-V cannot form the integer pointer view of a float slot.
    //
    // `PhysicalStorageBuffer` was missing from the select's list of selectable storage classes, so
    // the select fell through to an unmodelled placeholder and the load through it became
    // `OpCopyObject` of `OpConstantNull`: BOTH addresses computed, both discarded, the word read as
    // zero, and spirv-val happy. A generated-shader differential
    // (`pointer_select_fuzz`) found it on the device; this holds it without a GPU.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %d, i32 %tid) {
entry:
  %cache = alloca { ptr addrspace(1), ptr addrspace(1) }, align 8
  %f0 = getelementptr inbounds { ptr addrspace(1), ptr addrspace(1) }, ptr %cache, i64 0, i32 0
  store ptr addrspace(1) %a, ptr %f0, align 8
  %f1 = getelementptr inbounds { ptr addrspace(1), ptr addrspace(1) }, ptr %cache, i64 0, i32 1
  store ptr addrspace(1) %b, ptr %f1, align 8
  %la = load ptr addrspace(1), ptr %f0, align 8
  %lb = load ptr addrspace(1), ptr %f1, align 8
  %slot = zext i32 %tid to i64
  %dp = getelementptr inbounds float, ptr addrspace(1) %d, i64 %slot
  %db = bitcast ptr addrspace(1) %dp to ptr addrspace(1)
  %seed = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %db, i32 1, i32 0, i32 2, i1 true)
  %cd = icmp eq i32 %seed, 1
  %pa = getelementptr inbounds i32, ptr addrspace(1) %la, i64 %slot
  %pb = getelementptr inbounds i32, ptr addrspace(1) %lb, i64 %slot
  %p = select i1 %cd, ptr addrspace(1) %pa, ptr addrspace(1) %pb
  %v = load i32, ptr addrspace(1) %p, align 4
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %slot
  store i32 %v, ptr addrspace(1) %op, align 4
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"d"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_device_address_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");

    let mut pointer_class = std::collections::HashMap::new();
    let mut result_type = std::collections::HashMap::new();
    for line in asm.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [result, "=", "OpTypePointer", storage, _] => {
                pointer_class.insert(result.to_string(), storage.to_string());
            }
            [result, "=", opcode, ty, ..] if opcode.starts_with("Op") => {
                result_type.insert(result.to_string(), ty.to_string());
            }
            _ => {}
        }
    }

    // The chosen pointer is a physical one, and it is chosen by an OpSelect rather than replaced.
    let selected: Vec<&str> = asm
        .lines()
        .filter_map(|line| line.trim().split_once(" = OpSelect "))
        .filter(|(result, _)| {
            result_type
                .get(*result)
                .and_then(|ty| pointer_class.get(ty))
                .is_some_and(|storage| storage == "PhysicalStorageBuffer")
        })
        .map(|(result, _)| result)
        .collect();
    assert_eq!(
        selected.len(),
        1,
        "expected exactly one physical pointer select\n{asm}"
    );

    // And the load reads THROUGH it, with the Aligned operand a physical access must carry.
    let loaded = asm
        .lines()
        .map(str::trim)
        .find(|line| {
            line.contains("= OpLoad ") && line.split_whitespace().any(|w| w == selected[0])
        })
        .unwrap_or_else(|| panic!("no load through the selected pointer\n{asm}"));
    assert!(
        loaded.contains("Aligned"),
        "a physical load must carry Aligned: {loaded}\n{asm}"
    );

    // Nothing in the module folds a value to a null constant: that fold IS the old bug.
    assert!(
        !asm.contains("OpConstantNull"),
        "the load was replaced by a null rather than performed\n{asm}"
    );

    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_a_buffer_address_round_trip_still_names_its_buffer() {
    // `ptrtoint` a bound buffer, add a byte offset the host supplies, `inttoptr` it back, index.
    // Apple's DilateFilter4 and ErodeFilter4 reach their structuring element exactly this way.
    //
    // Logical SPIR-V has no pointer address, so the integer itself is correctly folded to zero -- it
    // is never observed. What used to be lost with it is which buffer it names: the `inttoptr` took
    // an unmodelled placeholder, and every load through it became `OpConstantNull`. Both halves of
    // the answer were wrong at once and spirv-val was happy, so only the loaded VALUE can hold this.
    //
    // The buffer is a 6-byte struct of three `ushort`s, which is what makes the root a word-array
    // view rather than a byte one, and which puts every odd row at a different sub-word phase.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

%struct.Row = type { i16, i16, i16 }

define void @k(ptr addrspace(2) %base, ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %gid) {
entry:
  %addr = ptrtoint ptr addrspace(2) %base to i64
  %offp = getelementptr inbounds i32, ptr addrspace(2) %params, i64 0
  %off = load i32, ptr addrspace(2) %offp, align 4
  %offw = zext i32 %off to i64
  %moved = add i64 %offw, %addr
  %row = inttoptr i64 %moved to ptr addrspace(2)
  %slot = zext i32 %gid to i64
  %ap = getelementptr inbounds %struct.Row, ptr addrspace(2) %row, i64 %slot, i32 0
  %a = load i16, ptr addrspace(2) %ap, align 2
  %av = zext i16 %a to i32
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %slot
  store i32 %av, ptr addrspace(1) %op, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 6, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"Row", !"air.arg_name", !"base"}
!4 = !{i32 0, i32 2, i32 0, !"ushort", !"a", i32 2, i32 2, i32 0, !"ushort", !"b", i32 4, i32 2, i32 0, !"ushort", !"c"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"byte_offset"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_buffer_address_round_trip_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");

    let bound_variable = |binding: &str| -> String {
        asm.lines()
            .map(str::trim)
            .find_map(
                |line| match line.split_whitespace().collect::<Vec<_>>()[..] {
                    ["OpDecorate", id, "Binding", value] if value == binding => {
                        Some(id.to_string())
                    }
                    _ => None,
                },
            )
            .unwrap_or_else(|| panic!("no variable at binding {binding}\n{asm}"))
    };
    // Every id the given seeds reach, following operands forward through the whole module.
    let reached_from = |seeds: &[&str]| -> std::collections::HashSet<String> {
        let mut reached: std::collections::HashSet<String> =
            seeds.iter().map(|id| id.to_string()).collect();
        loop {
            let before = reached.len();
            for line in asm.lines().map(str::trim) {
                let Some((result, rest)) = line.split_once(" = Op") else {
                    continue;
                };
                // Skip the result type, which is the first word after the opcode.
                if rest
                    .split_whitespace()
                    .skip(2)
                    .any(|word| reached.contains(word))
                {
                    reached.insert(result.to_string());
                }
            }
            if reached.len() == before {
                return reached;
            }
        }
    };

    let stored = asm
        .lines()
        .map(str::trim)
        .find_map(
            |line| match line.split_whitespace().collect::<Vec<_>>()[..] {
                ["OpStore", _, value, ..] => Some(value.to_string()),
                _ => None,
            },
        )
        .unwrap_or_else(|| panic!("nothing was stored\n{asm}"));

    let from_offset = reached_from(&[bound_variable("1").as_str()]);
    assert!(
        from_offset.contains(&stored),
        "the stored word does not depend on the byte offset the round trip added\n{asm}"
    );
    let from_base = reached_from(&[bound_variable("0").as_str()]);
    assert!(
        from_base.contains(&stored),
        "the stored word does not come from the buffer the address named\n{asm}"
    );

    // The integer address is still folded, and folding it is still correct -- but nothing the store
    // depends on may be built from that fold.
    let nulls: Vec<&str> = asm
        .lines()
        .map(str::trim)
        .filter_map(|line| line.split_once(" = OpConstantNull "))
        .map(|(result, _)| result)
        .collect();
    if !nulls.is_empty() {
        assert!(
            !reached_from(&nulls).contains(&stored),
            "the stored word is a folded zero rather than a load\n{asm}"
        );
    }

    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_a_buffer_address_round_trip_refuses_an_untyped_byte_root() {
    // Same round trip as
    // `native_a_buffer_address_round_trip_still_names_its_buffer`, on a buffer whose
    // ONLY use is `ptrtoint`. Nothing then names its element type, so the emitter declares it as a
    // bare `uchar` pointer and its descriptor block becomes a BYTE array -- while the raw word path
    // would index it in WORDS. `plan_raw_word_pointer_rewrite` reads that same chain's index as a
    // byte index and divides it by four a second time, and the load lands a quarter of the way into
    // the buffer with nothing invalid to see: two `OpUDiv`s in a row and spirv-val happy.
    //
    // No corpus source reaches this (14579 translate identically with and without the refusal, and
    // zero of them contain a chained `OpUDiv` at all), so the honest answer is to refuse rather
    // than to answer the wrong word.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(2) %base, ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %gid) {
entry:
  %addr = ptrtoint ptr addrspace(2) %base to i64
  %offp = getelementptr inbounds i32, ptr addrspace(2) %params, i64 0
  %off = load i32, ptr addrspace(2) %offp, align 4
  %offw = zext i32 %off to i64
  %moved = add i64 %offw, %addr
  %word = inttoptr i64 %moved to ptr addrspace(2)
  %slot = zext i32 %gid to i64
  %wp = getelementptr inbounds i32, ptr addrspace(2) %word, i64 %slot
  %v = load i32, ptr addrspace(2) %wp, align 4
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %slot
  store i32 %v, ptr addrspace(1) %op, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"base"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"byte_offset"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_buffer_address_untyped_root_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let error = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp)
        .expect_err("an untyped byte root must not be addressed as words");
    assert!(
        error.contains("untyped byte root"),
        "refused for the wrong reason: {error}"
    );
}

#[test]
fn native_two_call_sites_at_one_uniform_member_both_read_it() {
    // A `constant`-space helper parameter reaches the callee as a byte CURSOR on the uniform
    // buffer's descriptor root, and only one cursor per callee parameter can be recorded. The gate
    // that kept a second call site from contradicting the first was "the callee has exactly one
    // call site", which refuses a helper called twice at the SAME member -- and then every load in
    // it folds to `OpConstantNull`, silently.
    //
    // The two call sites here name that member through two DISTINCT `getelementptr` locals in two
    // different blocks, which is the shape Apple's `VCPRender` uses and the shape a "both sites
    // pass the same local" gate would still refuse. Identical chains address identical bytes, so
    // no cursor can conflict.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

%struct.Coef = type { float, float, float, float }
%struct.Params = type { %struct.Coef, %struct.Coef }

define void @k(ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %gid) {
entry:
  %odd = and i32 %gid, 1
  %cond = icmp eq i32 %odd, 0
  br i1 %cond, label %even, label %oddblk

even:
  %ma = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %ra = call fastcc float @coef0(ptr addrspace(2) %ma, float 1.000000e+00)
  br label %join

oddblk:
  %mb = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %rb = call fastcc float @coef0(ptr addrspace(2) %mb, float -1.000000e+00)
  br label %join

join:
  %r = phi float [ %ra, %even ], [ %rb, %oddblk ]
  %slot = zext i32 %gid to i64
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %slot
  store float %r, ptr addrspace(1) %op, align 4
  ret void
}

; Two blocks, so the AIR-text inliner leaves it as a called function.
define internal fastcc float @coef0(ptr addrspace(2) %c, float %x) {
head:
  %neg = fcmp olt float %x, 0.000000e+00
  br i1 %neg, label %lo, label %hi

lo:
  %p0 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %c, i64 0, i32 0
  %v0 = load float, ptr addrspace(2) %p0, align 4
  ret float %v0

hi:
  %p3 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %c, i64 0, i32 3
  %v3 = load float, ptr addrspace(2) %p3, align 4
  ret float %v3
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 16, i32 0, !"Coef", !"a", !5, i32 16, i32 16, i32 0, !"Coef", !"b", !6}
!5 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!6 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_uniform_member_two_sites_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");

    // Both members are read out of the uniform buffer, and neither read is a folded zero.
    let uniform = asm
        .lines()
        .map(str::trim)
        .find_map(
            |line| match line.split_whitespace().collect::<Vec<_>>()[..] {
                ["OpDecorate", id, "Binding", "0"] => Some(id.to_string()),
                _ => None,
            },
        )
        .expect("no variable at binding 0");
    let chains = asm
        .lines()
        .filter(|line| line.contains("AccessChain") && line.contains(&format!(" {uniform} ")))
        .count();
    assert!(
        chains >= 2,
        "expected both helper reads to address the uniform buffer, saw {chains}\n{asm}"
    );
    assert!(
        !asm.contains("OpConstantNull"),
        "the uniform member folded to a null rather than being read\n{asm}"
    );

    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_a_cloned_helper_parameter_keeps_its_buffer_cursor() {
    // The structurizer CLONES blocks (`cfg::clone_crossarm`) and renames every value it duplicates.
    // `raw_buffer_params` is keyed by NAME and is derived before the CFG is structured, so the clone
    // of an inlined helper's buffer parameter is absent from it -- and that absence is what carries
    // the parameter's byte cursor. Every load through the cloned parameter folded to
    // `OpConstantNull`, in a module that still passes spirv-val.
    //
    // This CFG is the shape `privatize_deep_shared_continuations` exists for: an in-arm continuation
    // (`cont`) that the ENCLOSING arm (`outerF`) also reaches, so the inner selection's merge is not
    // dominated by its header and `cont` has to be privatized for the inner entries. `cont` is where
    // the helper call is, so the clone renames its inlined parameter.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

%struct.Coef = type { float, float, float, float }
%struct.Params = type { %struct.Coef, %struct.Coef }

define void @k(ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %gid) {
entry:
  %c0 = icmp ult i32 %gid, 4
  br i1 %c0, label %outerT, label %outerF

outerT:
  %c1 = icmp ult i32 %gid, 2
  br i1 %c1, label %innerA, label %innerB

innerA:
  br label %cont

innerB:
  br label %merge

outerF:
  br label %cont

cont:
  %m = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %r = call fastcc float @coef(ptr addrspace(2) %m, float 5.000000e-01)
  br label %merge

merge:
  %v = phi float [ %r, %cont ], [ 7.000000e+00, %innerB ]
  %slot = zext i32 %gid to i64
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %slot
  store float %v, ptr addrspace(1) %op, align 4
  ret void
}

define internal fastcc float @coef(ptr addrspace(2) %c, float %x) {
head:
  %p2 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %c, i64 0, i32 2
  %v2 = load float, ptr addrspace(2) %p2, align 4
  %s = fadd float %v2, %x
  ret float %s
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 16, i32 0, !"Coef", !"a", !5, i32 16, i32 16, i32 0, !"Coef", !"b", !6}
!5 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!6 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_cloned_helper_parameter_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");

    assert!(
        asm.contains("OpCopyObject") || asm.contains("OpLoad"),
        "nothing was emitted\n{asm}"
    );
    assert!(
        !asm.contains("OpConstantNull"),
        "the cloned helper read its uniform member as a folded zero\n{asm}"
    );
    let uniform = asm
        .lines()
        .map(str::trim)
        .find_map(
            |line| match line.split_whitespace().collect::<Vec<_>>()[..] {
                ["OpDecorate", id, "Binding", "0"] => Some(id.to_string()),
                _ => None,
            },
        )
        .expect("no variable at binding 0");
    let reads = asm
        .lines()
        .filter(|line| line.contains("AccessChain") && line.contains(&format!(" {uniform} ")))
        .count();
    assert!(
        reads >= 2,
        "expected both copies of the helper to read the uniform buffer, saw {reads}\n{asm}"
    );

    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

/// A narrow integer constant with a negative value is not a 32-bit literal.
///
/// `Ctx::const_int_of` wrote `v as u32` for every width below 64, so `const_int_of(i16, -8)` --
/// the alignment mask `air.get_descriptor_size_tensor` rounds its payload up with -- emitted
/// `OpConstant %ushort 4294967288`. SPIR-V 2.2.1 requires a literal narrower than a word to have
/// zero high-order bits when the type's Signedness is 0, and spirv-val rejects the whole module on
/// it, so every consumer of that family got a FALLBACK instead of a shader.
#[test]
fn native_narrow_negative_constant_fits_its_type() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out) {
entry:
  %word = load i32, ptr addrspace(1) %out, align 4
  %rank = trunc i32 %word to i16
  %v = call i16 @air.get_descriptor_size_tensor(i16 %rank, i16 %rank)
  %w = zext i16 %v to i32
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}

declare i16 @air.get_descriptor_size_tensor(i16, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_narrow_negative_constant_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");

    // Widths of the integer types, and the literal every constant of one carries.
    let mut width = std::collections::HashMap::new();
    for line in asm.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if let [result, "=", "OpTypeInt", bits, _sign] = words.as_slice() {
            width.insert(result.to_string(), bits.parse::<u32>().expect("int width"));
        }
    }
    let mut checked = 0;
    for line in asm.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let [_result, "=", "OpConstant", ty, literal] = words.as_slice() else {
            continue;
        };
        let Some(&bits) = width.get(*ty) else {
            continue;
        };
        if bits >= 32 {
            continue;
        }
        let value: u64 = literal.parse().expect("constant literal");
        assert!(
            value < (1u64 << bits),
            "{line} does not fit its {bits}-bit unsigned type\n{asm}"
        );
        checked += 1;
    }
    assert!(checked > 0, "expected narrow constants in\n{asm}");

    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

/// `air.convert` picks its SPIR-V opcode from the kind letters in the name and never checks the
/// operand it actually has. The 8-bit float formats arrive as `i8` -- SPIR-V has no type for them --
/// so `air.convert.f.f32.f.f8e5m2` read as `f`->`f` and emitted an `OpFConvert` over an integer.
/// The owned check refused it several phases later with a shape complaint naming neither the family
/// nor the reason.
///
/// Modelling them would be an exact bit rule, not an approximation, and it cannot be settled here:
/// the installed Metal toolchain has no 8-bit float type, so there is no oracle for a decode.
/// Unknown stays unknown, and says which family it does not know.
#[test]
fn native_eight_bit_float_convert_is_refused_by_name() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out) {
entry:
  %b = load i8, ptr addrspace(1) %out, align 1
  %v = call float @air.convert.f.f32.f.f8e5m2(i8 %b)
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.convert.f.f32.f.f8e5m2(i8)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_eight_bit_float_convert_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let error = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp)
        .expect_err("an 8-bit float convert has no lowering");
    assert!(
        error.contains("air.convert.f.f32.f.f8e5m2") && error.contains("f8e5m2"),
        "the refusal must name the family and the format, got {error}"
    );
    assert!(
        !error.contains("FConvert"),
        "the refusal must not be the downstream shape complaint, got {error}"
    );
}
