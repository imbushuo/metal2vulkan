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
fn fragment_quad_active_mask_and_narrow_popcount_are_vulkan_valid() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define <4 x half> @frag() {
entry:
  %mask = call i16 @air.quad_active_threads_mask()
  %quad = and i16 %mask, 15
  %count = call i16 @air.popcount.i16(i16 %quad)
  %value = uitofp i16 %count to half
  %v0 = insertelement <4 x half> poison, half %value, i32 0
  %v1 = shufflevector <4 x half> %v0, <4 x half> poison, <4 x i32> zeroinitializer
  ret <4 x half> %v1
}

declare i16 @air.quad_active_threads_mask()
declare i16 @air.popcount.i16(i16)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_quad_active_mask_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpGroupNonUniformBallot"), "{asm}");
    let bitcount = asm
        .lines()
        .find(|line| line.contains("OpBitCount"))
        .expect("OpBitCount");
    assert!(bitcount.contains(&uint32_type_id(&asm)), "{asm}");
    assert!(asm.contains("Flat"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_workgroup_array_vector_load_uses_first_element() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

@tile = internal unnamed_addr addrspace(3) global [256 x <4 x half>] undef, align 8

define void @k(ptr addrspace(1) %out) {
entry:
  %v = load <4 x half>, ptr addrspace(3) @tile, align 8
  store <4 x half> %v, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_workgroup_array_vec_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpBitcast"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_kernel_ushort2_threads_per_threadgroup_uses_specialized_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(<2 x i16> %lsize) {
entry:
  %x = extractelement <2 x i16> %lsize, i32 0
  %ok = icmp uge i16 %x, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lsize"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_ushort2_threads_per_threadgroup_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSpecConstantComposite"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_simd_shuffle_and_barrier_lower_to_subgroup_ops() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, i16 %lane) {
entry:
  %out = tail call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %lane)
  %signed = tail call i32 @air.simd_shuffle.s.i32(i32 %out, i16 %lane)
  %ok = icmp uge i32 %signed, 0
  tail call void @air.simdgroup.barrier(i32 0, i32 4)
  ret void
}

declare i32 @air.simd_shuffle.u.i32(i32, i16)
declare i32 @air.simd_shuffle.s.i32(i32, i16)
declare void @air.simdgroup.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_simd_shuffle_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 2, "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_quad_sum_lowers_to_shuffle_xor_butterfly() {
    // quad_sum has no quad-scoped reduction op in SPIR-V; it lowers to an XOR butterfly over the two
    // intra-quad swap axes (mask 1, then mask 2), so every lane ends with the full quad sum. Two
    // GroupNonUniformShuffleXor + two FAdd, using only the Shuffle capability (no GroupNonUniformQuad).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %v, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.quad_sum.f32(float %v)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.quad_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_quad_sum_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm.contains("OpCapability GroupNonUniformQuad"),
        "quad_sum must not require the GroupNonUniformQuad capability: {asm}"
    );
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleXor").count(),
        2,
        "{asm}"
    );
    assert_eq!(asm.matches("OpFAdd").count(), 2, "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_quad_integer_extrema_stay_inside_aligned_quad() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %max = tail call i32 @air.quad_max.u.i32(i32 %v)
  %min = tail call i32 @air.quad_min.u.i32(i32 %v)
  %sum = add i32 %max, %min
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.quad_max.u.i32(i32)
declare i32 @air.quad_min.u.i32(i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_quad_integer_extrema_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleXor").count(),
        4,
        "{asm}"
    );
    assert_eq!(asm.matches("OpUGreaterThan").count(), 2, "{asm}");
    assert_eq!(asm.matches("OpULessThan").count(), 2, "{asm}");
    assert!(!asm.contains("OpGroupNonUniformUMax"), "{asm}");
    assert!(!asm.contains("OpGroupNonUniformUMin"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_simd_sum_uses_metal_32_lane_cluster() {
    // AIR simd operations structurally select Metal's 32-lane simdgroup semantics. This cannot be
    // left to a driver's native subgroup width: MoltenVK may expose a 64-lane subgroup.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %v, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_sum.f32(float %v)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_simd_sum_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("ClusteredReduce"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformClustered"),
        "{asm}"
    );
}

#[test]
fn native_air_simd_is_first_selects_each_32_lane_partition() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %out) {
entry:
  %first = tail call i1 @air.simd_is_first()
  %word = select i1 %first, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.simd_is_first()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_is_first_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(asm.contains("BuiltIn SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_quad_is_first_selects_each_four_lane_partition() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %out) {
entry:
  %first = tail call i1 @air.quad_is_first()
  %word = select i1 %first, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.quad_is_first()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_is_first_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(asm.contains("BuiltIn SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.trim_end().ends_with(" 3")),
        "{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_air_quad_all_lowers_to_four_lane_xor_butterfly() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %all = tail call i1 @air.quad_all(i1 %ok)
  %word = select i1 %all, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.quad_all(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_all_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformShuffle"), "{asm}");
    assert!(!asm.contains("OpCapability GroupNonUniformVote"), "{asm}");
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleXor").count(),
        2,
        "{asm}"
    );
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_quad_any_lowers_to_four_lane_xor_butterfly() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %any = tail call i1 @air.quad_any(i1 %ok)
  %word = select i1 %any, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.quad_any(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_any_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformShuffle"), "{asm}");
    assert!(!asm.contains("OpCapability GroupNonUniformVote"), "{asm}");
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleXor").count(),
        2,
        "{asm}"
    );
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn air_simd_any_all_vote_over_the_lane_s_own_thirty_two_lane_partition() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %any = tail call i1 @air.simd_any(i1 %ok)
  %all = tail call i1 @air.simd_all(i1 %ok)
  %any_word = select i1 %any, i32 1, i32 0
  %all_word = select i1 %all, i32 2, i32 0
  %word = or i32 %any_word, %all_word
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.simd_any(i1)
declare i1 @air.simd_all(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_any_all_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformBallot"), "{asm}");
    // The vote is Metal's 32-lane simdgroup, not the physical subgroup: ballot the predicate,
    // select this lane's own partition word, and read the answer out of it. `any` is a non-zero
    // word; `all` compares against the same partition's ballot of `true`, the active mask.
    assert!(asm.contains("OpGroupNonUniformBallot"), "{asm}");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(asm.contains("OpINotEqual"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(!asm.contains("OpGroupNonUniformAny"), "{asm}");
    assert!(!asm.contains("OpGroupNonUniformAll"), "{asm}");
    assert!(!asm.contains("OpCapability GroupNonUniformVote"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn air_simd_ballot_i64_is_the_lane_s_own_thirty_two_lane_partition_word() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %mask = tail call i64 @air.simd_ballot.i64(i1 %ok)
  %lo = trunc i64 %mask to i32
  store i32 %lo, ptr addrspace(1) %out, align 4
  ret void
}

declare i64 @air.simd_ballot.i64(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_ballot_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformBallot"), "{asm}");
    // `simd_vote` bits are indexed by `simd_lane_id`, so the mask is this lane's own aligned
    // 32-lane ballot word zero-extended -- never the physical subgroup's low 64 lanes joined.
    assert!(asm.contains("OpGroupNonUniformBallot"), "{asm}");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(!asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(!asm.contains("OpBitwiseOr"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_get_simdgroup_size_i16_lowers_to_width_constant() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %out) {
entry:
  %width16 = tail call i16 @air.get_simdgroup_size.i16()
  %width = zext i16 %width16 to i32
  store i32 %width, ptr addrspace(1) %out, align 4
  ret void
}

declare i16 @air.get_simdgroup_size.i16()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_size_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.trim_end().ends_with(" 32")),
        "{asm}"
    );
    assert!(
        !asm.contains("air.get_simdgroup_size"),
        "intrinsic call survived lowering:\n{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

/// `simd_shuffle_and_fill_up` picks data vs fill by comparing the in-cluster lane against the
/// UNREDUCED delta.
///
/// Comparing against `delta % modulo` reads identically on `[0, modulo)` and answers the opposite
/// thing at `delta == modulo`, which is the whole cluster shifted out: it finds every lane in
/// bounds and returns `data` where the device -- and this lowering's own mirror -- returns `fill`.
/// The two-argument Metal overload passes `modulo = __metal_get_simdgroup_size()`, so that input is
/// ordinary. Device evidence is `shift-a-whole-simd-cluster-out-through-the-fill`; this asserts the
/// shape a GPU-less run can still see, because the reduced delta is computed either way for the
/// wrap arithmetic and only the comparison distinguishes them.
#[test]
fn native_air_simd_shuffle_and_fill_up_tests_the_unreduced_delta() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(half %x, half %fill, i16 %delta, i16 %width, ptr addrspace(1) %out) {
entry:
  %up = tail call half @air.simd_shuffle_and_fill_up.f16(half %x, half %fill, i16 %delta, i16 %width)
  %f = fpext half %up to float
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare half @air.simd_shuffle_and_fill_up.f16(half, half, i16, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_fill_up_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let defs = module
        .all_inst_iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst)))
        .collect::<std::collections::HashMap<_, _>>();
    let operand = |inst: &Instruction, index: usize| match inst.operands.get(index) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    };
    let condition = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::Select)
        .filter_map(|inst| operand(inst, 0))
        .filter_map(|id| defs.get(&id).copied())
        .find(|inst| inst.class.opcode == Op::UGreaterThanEqual)
        .unwrap_or_else(|| panic!("the data-vs-fill select has no lane bound test:\n{spv:?}"));
    let bound = operand(condition, 1).and_then(|id| defs.get(&id).copied());
    assert!(
        bound.is_none_or(|inst| inst.class.opcode != Op::UMod),
        "the lane is compared against a reduced delta, so a delta of one whole cluster leaves \
         every lane in bounds instead of none of them"
    );
    // The reduced delta is still computed, for the wrap. Its absence would mean the test above
    // passed because the lowering stopped reducing at all, which underflows above delta == modulo.
    assert!(
        module
            .all_inst_iter()
            .any(|inst| inst.class.opcode == Op::UMod),
        "the wrap still needs the reduced delta"
    );
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_air_simd_shuffle_and_fill_down_lowers_to_clustered_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(half %x, half %fill, i16 %delta, ptr addrspace(1) %out) {
entry:
  %width = tail call i16 @air.get_simdgroup_size.i16()
  %down = tail call half @air.simd_shuffle_and_fill_down.f16(half %x, half %fill, i16 %delta, i16 %width)
  %f = fpext half %down to float
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare i16 @air.get_simdgroup_size.i16()
declare half @air.simd_shuffle_and_fill_down.f16(half, half, i16, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_fill_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert_eq!(
        asm.lines()
            .filter(|line| line.contains("= OpGroupNonUniformShuffle "))
            .count(),
        2,
        "{asm}"
    );
    assert!(asm.contains("OpUMod"), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
    assert!(asm.contains("OpULessThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(
        !asm.contains("air.simd_shuffle_and_fill_down"),
        "intrinsic call survived lowering:\n{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_broadcast_lowers_to_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, i16 %lane, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_broadcast.f32(float %x, i16 %lane)
  store float %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_broadcast.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_broadcast_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    // The operand is a lane of the CALLER's simdgroup, not an absolute subgroup lane, so the
    // lowering has to read SubgroupLocalInvocationId and rebase -- the same shape
    // `simd_shuffle_down` is checked for below.
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_shuffle_lowers_to_the_callers_own_simdgroup_lane() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, i16 %lane, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_shuffle.f32(float %x, i16 %lane)
  store float %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_shuffle.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_shuffle_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    // Passing the AIR operand straight through as an absolute subgroup lane reads another
    // simdgroup's lane whenever the driver's subgroup is wider than Metal's 32.
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_shuffle_resolves_the_same_lane_for_every_element_type() {
    // `air.simd_shuffle` names a lane of the CALLER's own 32-lane Metal simdgroup, so the lowering
    // has to add the caller's simdgroup base -- see `metal_simd_absolute_lane_u32`. The integer
    // element types used to be emitted natively with the AIR operand passed straight through as an
    // absolute subgroup lane, so a module calling both spellings computed two different source
    // lanes for one intrinsic on any driver whose subgroup is wider than 32.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, float %f, i16 %lane, ptr addrspace(1) %out) {
entry:
  %si = tail call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %lane)
  %ss = tail call i32 @air.simd_shuffle.s.i32(i32 %si, i16 %lane)
  %sf = tail call float @air.simd_shuffle.f32(float %f, i16 %lane)
  %fi = sitofp i32 %ss to float
  %sum = fadd float %sf, %fi
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.simd_shuffle.u.i32(i32, i16)
declare i32 @air.simd_shuffle.s.i32(i32, i16)
declare float @air.simd_shuffle.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_lane_model_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 3, "{asm}");

    // Every shuffle's lane operand must be the `simd_base + (index & 31)` sum, not the AIR operand.
    let adds: Vec<&str> = asm
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("= OpIAdd "))
        .map(|l| l.split_whitespace().next().expect("result id"))
        .collect();
    let mut lanes = Vec::new();
    for line in asm.lines().map(str::trim) {
        if let Some(rest) = line.split("= OpGroupNonUniformShuffle ").nth(1) {
            lanes.push(rest.split_whitespace().last().expect("lane operand"));
        }
    }
    assert_eq!(lanes.len(), 3, "{asm}");
    for lane in &lanes {
        assert!(
            adds.contains(lane),
            "lane {lane} is not an OpIAdd result\n{asm}"
        );
    }
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_shuffle_down_lowers_to_32_lane_absolute_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, half %h, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_shuffle_down.f32(float %x, i16 %delta)
  %sh = tail call half @air.simd_shuffle_down.f16(half %h, i16 %delta)
  %hf = fpext half %sh to float
  %sum = fadd float %sx, %hf
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_shuffle_down.f32(float, i16)
declare half @air.simd_shuffle_down.f16(half, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 2, "{asm}");
    assert!(!asm.contains("OpGroupNonUniformShuffleDown"), "{asm}");
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_shuffle_rotate_down_lowers_to_wrapping_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_shuffle_rotate_down.f32(float %x, i16 %delta)
  store float %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_shuffle_rotate_down.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_rotate_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 1, "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(!asm.contains("air.simd_shuffle_rotate_down"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_shuffle_up_lowers_to_32_lane_absolute_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %x, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call i32 @air.simd_shuffle_up.u.i32(i32 %x, i16 %delta)
  store i32 %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.simd_shuffle_up.u.i32(i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_up_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(!asm.contains("OpGroupNonUniformShuffleUp"), "{asm}");
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_shuffle_xor_lowers_to_subgroup_shuffle_xor() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(half %x, i16 %mask, ptr addrspace(1) %out) {
entry:
  %sx = tail call half @air.simd_shuffle_xor.f16(half %x, i16 %mask)
  %f = fpext half %sx to float
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare half @air.simd_shuffle_xor.f16(half, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_xor_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformShuffleXor"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_quad_shuffle_down_vector_lowers_to_subgroup_shuffle_down() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(<4 x float> %x, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call <4 x float> @air.quad_shuffle_down.v4f32(<4 x float> %x, i16 %delta)
  store <4 x float> %sx, ptr addrspace(1) %out, align 16
  ret void
}

declare <4 x float> @air.quad_shuffle_down.v4f32(<4 x float>, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_shuffle_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleDown").count(),
        1,
        "{asm}"
    );
    assert!(asm.contains("OpUConvert"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_prefix_exclusive_sum_lowers_to_subgroup_scan() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_prefix_exclusive_sum.f32(float %x)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_prefix_exclusive_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_prefix_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("ExclusiveScan"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_vector_u16_prefix_exclusive_sum_lowers_componentwise() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(2) %input, ptr addrspace(1) %out) {
entry:
  %x = load <4 x i16>, ptr addrspace(2) %input, align 8
  %sum = tail call <4 x i16> @air.simd_prefix_exclusive_sum.u.v4i16(<4 x i16> %x)
  store <4 x i16> %sum, ptr addrspace(1) %out, align 8
  ret void
}

declare <4 x i16> @air.simd_prefix_exclusive_sum.u.v4i16(<4 x i16>)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_vector_u16_prefix_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability Int16"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformIAdd"), "{asm}");
    assert!(asm.contains("ExclusiveScan"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_sum_lowers_to_subgroup_reduce() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_sum.f32(float %x)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_sum_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    // `air.simd_sum` is a 32-lane simdgroup reduction, so it emits `ClusteredReduce` -- and that
    // group operation takes `GroupNonUniformClustered`, not `GroupNonUniformArithmetic`. The
    // capability follows the OPERATION, not the opcode. Asserting the substring "Reduce" cannot
    // tell the two apart, which is how the wrong capability went unnoticed on 244 corpus modules.
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformClustered"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("ClusteredReduce"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_or_i8_lowers_to_subgroup_bitwise_reduce() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i8 %x, ptr addrspace(1) %out) {
entry:
  %mask = tail call i8 @air.simd_or.u.i8(i8 %x)
  %wide = zext i8 %mask to i32
  store i32 %wide, ptr addrspace(1) %out, align 4
  ret void
}

declare i8 @air.simd_or.u.i8(i8)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_or_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    // Clustered to 32 lanes like every other `air.simd_*` reduction, so the enabling capability is
    // `GroupNonUniformClustered` and not `GroupNonUniformArithmetic`.
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformClustered"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpCapability Int8"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformBitwiseOr"), "{asm}");
    assert!(asm.contains("ClusteredReduce"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_air_simd_inclusive_sum_min_max_lower_to_subgroup_arithmetic() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(float %x, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_prefix_inclusive_sum.f32(float %x)
  %min = tail call float @air.simd_min.f32(float %x)
  %max = tail call float @air.simd_max.f32(float %x)
  %a = fadd fast float %sum, %min
  %b = fadd fast float %a, %max
  store float %b, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_prefix_inclusive_sum.f32(float)
declare float @air.simd_min.f32(float)
declare float @air.simd_max.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_extrema_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("InclusiveScan"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformFMin"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformFMax"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

/// The `.s.`/`.u.` marker on a subgroup extrema decides the comparison, not the result type.
///
/// An LLVM `i16` carries no sign, so the emitter types it `%ushort` and reading the signedness off
/// the SPIR-V type made every `air.simd_min.s.*` and `air.simd_max.s.*` an unsigned reduction. The
/// marker is the only statement AIR makes about it, and it is the same one `lower_integer_op` reads
/// for the ordinary `air.min`/`air.max`. A signed reduction therefore runs in the signed sibling
/// type and bitcasts back, which also keeps this crate's rule that an S-op's result type is signed.
#[test]
fn native_air_signed_subgroup_extrema_take_their_comparison_from_the_air_name() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i16 %x, ptr addrspace(1) %out) {
entry:
  %smin = tail call i16 @air.simd_min.s.i16(i16 %x)
  %umin = tail call i16 @air.simd_min.u.i16(i16 %x)
  %smax = tail call i16 @air.simd_max.s.i16(i16 %x)
  %a = add i16 %smin, %umin
  %b = add i16 %a, %smax
  store i16 %b, ptr addrspace(1) %out, align 2
  ret void
}

declare i16 @air.simd_min.s.i16(i16)
declare i16 @air.simd_min.u.i16(i16)
declare i16 @air.simd_max.s.i16(i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"short", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_signed_simd_extrema_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // The signed pair takes the S ops, the unsigned one the U op; all three share a source value, so
    // a lowering that read the type would emit three U ops.
    assert!(asm.contains("OpGroupNonUniformSMin"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformSMax"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformUMin"), "{asm}");
    assert!(!asm.contains("OpGroupNonUniformUMax"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_kernel_threads_per_grid_shares_num_workgroups_builtin() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(<2 x i32> %grid_size, <2 x i32> %group_count) {
entry:
  %sx = extractelement <2 x i32> %grid_size, i64 0
  %sy = extractelement <2 x i32> %grid_size, i64 1
  %gx = extractelement <2 x i32> %group_count, i64 0
  %gy = extractelement <2 x i32> %group_count, i64 1
  %sum0 = add i32 %sx, %sy
  %sum1 = add i32 %gx, %gy
  %sum = add i32 %sum0, %sum1
  %ok = icmp uge i32 %sum, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
!4 = !{i32 1, !"air.threadgroups_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"group_count"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threads_per_grid_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [32, 2, 1],
            kernel_dispatch: Some(crate::reflect::KernelDispatch::Workgroups),
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("LocalSize 32 2 1"), "{asm}");
    assert!(asm.contains("BuiltIn NumWorkgroups"), "{asm}");
    assert_eq!(asm.matches("BuiltIn NumWorkgroups").count(), 1, "{asm}");
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(!asm.contains("OpUndef"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("32")),
        "{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("2")),
        "{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_kernel_threads_per_grid_accepts_exact_dispatch_threads_shape() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(<2 x i32> %grid_size) {
entry:
  %sx = extractelement <2 x i32> %grid_size, i64 0
  %sy = extractelement <2 x i32> %grid_size, i64 1
  %sum = add i32 %sx, %sy
  %ok = icmp uge i32 %sum, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_exact_threads_per_grid_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [5, 2, 1],
            kernel_dispatch: Some(crate::reflect::KernelDispatch::ThreadsFixed {
                threads_per_grid: [21, 3, 1],
            }),
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn WorkgroupSize"), "{asm}");
    assert_eq!(asm.matches("SpecId").count(), 3, "{asm}");
    assert!(asm.contains("PushConstant"), "{asm}");
    assert!(!asm.contains("BuiltIn NumWorkgroups"), "{asm}");
    assert!(!asm.contains("OpIMul"), "{asm}");
    assert_eq!(asm.matches("OpUGreaterThanEqual").count(), 1, "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_kernel_fixed_dispatch_threads_uses_specialized_boundary_regions() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k() {
entry:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_fixed_dispatch_guard_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [8, 8, 1],
            kernel_dispatch: Some(crate::reflect::KernelDispatch::ThreadsFixed {
                threads_per_grid: [57, 9, 1],
            }),
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn WorkgroupSize"), "{asm}");
    assert_eq!(asm.matches("SpecId").count(), 3, "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");

    let divisible = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [8, 8, 1],
            kernel_dispatch: Some(crate::reflect::KernelDispatch::ThreadsFixed {
                threads_per_grid: [64, 16, 1],
            }),
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate divisible grid");
    let divisible_asm = disassemble(&divisible).expect("disassemble divisible grid");
    assert!(
        !divisible_asm.contains("OpBranchConditional"),
        "{divisible_asm}"
    );
    assert!(
        divisible_asm.contains("BuiltIn WorkgroupSize"),
        "{divisible_asm}"
    );
}

#[test]
fn native_kernel_dynamic_dispatch_threads_shares_reflected_push_constant_grid() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(<3 x i32> %grid_size) {
entry:
  %x = extractelement <3 x i32> %grid_size, i64 0
  %ok = icmp uge i32 %x, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"grid_size"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_dynamic_dispatch_guard_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let options = passes::TransformOptions {
        kernel_local_size: [8, 8, 1],
        kernel_dispatch: Some(crate::reflect::KernelDispatch::ThreadsDynamic { offset: 16 }),
        ..passes::TransformOptions::default()
    };
    let (spv, reflection) =
        crate::translate_sanitized_native_reflected(ll, Stage::Kernel, &tmp, options)
            .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("PushConstant"), "{asm}");
    for offset in (16..64).step_by(4) {
        assert!(asm.contains(&format!("Offset {offset}")), "{asm}");
    }
    assert_eq!(asm.matches("OpUGreaterThanEqual").count(), 1, "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    assert_eq!(
        reflection.kernel_dispatch,
        Some(crate::reflect::KernelDispatch::ThreadsDynamic { offset: 16 })
    );
    assert_eq!(
        reflection
            .kernel_dispatch
            .and_then(crate::reflect::KernelDispatch::push_constant_range),
        Some(crate::reflect::KernelDispatchPushConstantRange {
            offset: 16,
            size: 48,
        })
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_kernel_default_dispatch_uses_dynamic_grid_and_requires_explicit_workgroups() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k() {
entry:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_default_dispatch_guard_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let (guarded, reflection) = crate::translate_sanitized_native_reflected(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions::default(),
    )
    .expect("default translation");
    let guarded_asm = disassemble(&guarded).expect("disassemble guarded default");
    assert!(
        guarded_asm.contains("BuiltIn WorkgroupSize"),
        "{guarded_asm}"
    );
    assert_eq!(guarded_asm.matches("SpecId").count(), 3);
    assert!(!guarded_asm.contains("OpBranchConditional"));
    assert_eq!(
        reflection.kernel_dispatch,
        Some(crate::reflect::KernelDispatch::ThreadsDynamic {
            offset: crate::reflect::DEFAULT_KERNEL_DISPATCH_PUSH_CONSTANT_OFFSET,
        })
    );
    tools::spirv_val_bytes(&guarded, &tmp).expect("spirv-val guarded default");

    let workgroups = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_dispatch: Some(crate::reflect::KernelDispatch::Workgroups),
            ..passes::TransformOptions::default()
        },
    )
    .expect("explicit whole-workgroup translation");
    let workgroups_asm = disassemble(&workgroups).expect("disassemble whole workgroups");
    assert!(!workgroups_asm.contains("PushConstant"), "{workgroups_asm}");
    assert!(
        !workgroups_asm.contains("OpBranchConditional"),
        "{workgroups_asm}"
    );
}

#[test]
fn native_kernel_partial_dispatch_threads_preserves_source_workgroup_barrier() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k() {
entry:
  tail call void @air.wg.barrier(i32 0, i32 1)
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_partial_barrier_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions::default(),
    )
    .expect("boundary decomposition keeps every workgroup barrier uniform");
    let asm = disassemble(&spv).expect("disassemble exact-thread barrier");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val exact-thread barrier");

    let workgroups = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_dispatch: Some(crate::reflect::KernelDispatch::Workgroups),
            ..passes::TransformOptions::default()
        },
    )
    .expect("an explicit whole-workgroup proof keeps the barrier uniform");
    tools::spirv_val_bytes(&workgroups, &tmp).expect("spirv-val whole-workgroup barrier");
}

#[test]
fn native_kernel_partial_dispatch_ignores_barrier_in_unreachable_helper() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k() {
entry:
  ret void
}

define internal void @dead_helper() {
entry:
  tail call void @air.wg.barrier(i32 0, i32 1)
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_dead_barrier_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions::default(),
    )
    .expect("a dead helper's barrier does not constrain entry dispatch");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn WorkgroupSize"), "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    assert!(!asm.contains("OpControlBarrier"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val dead barrier helper");
}

#[test]
fn native_kernel_partial_dispatch_preserves_barrier_in_reachable_helper() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k() {
entry:
  tail call void @barrier_helper()
  ret void
}

define internal void @barrier_helper() {
entry:
  tail call void @air.wg.barrier(i32 0, i32 1)
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_reachable_barrier_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions::default(),
    )
    .expect("boundary decomposition keeps a reachable helper barrier uniform");
    let asm = disassemble(&spv).expect("disassemble reachable barrier");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val reachable barrier");
}

#[test]
fn native_kernel_generated_workgroup_initialization_needs_no_dispatch_cull() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
@scratch = internal addrspace(3) global [1 x i32] zeroinitializer, align 4

define void @k() {
entry:
  %slot = getelementptr [1 x i32], ptr addrspace(3) @scratch, i64 0, i64 0
  store i32 7, ptr addrspace(3) %slot, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_generated_barrier_guard_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [8, 1, 1],
            kernel_dispatch: Some(crate::reflect::KernelDispatch::ThreadsFixed {
                threads_per_grid: [9, 1, 1],
            }),
            ..passes::TransformOptions::default()
        },
    )
    .expect("translator-owned barrier remains uniform");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    assert!(!asm.contains("OpBranchConditional"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_kernel_threadgroup_packed_float1_record_array_validates() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
%struct.Temp = type { [1 x float], i32 }

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %value = getelementptr inbounds %struct.Temp, ptr addrspace(3) %temp, i64 %idx, i32 0, i64 0
  store float 1.000000e+00, ptr addrspace(3) %value, align 4
  %count = getelementptr inbounds %struct.Temp, ptr addrspace(3) %temp, i64 %idx, i32 1
  store i32 %i, ptr addrspace(3) %count, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !5, !"air.arg_type_name", !"Temp", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!5 = !{i32 0, i32 4, i32 0, !"packed_float1", !"value", i32 4, i32 4, i32 0, !"uint", !"count"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_packed_float1_record_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Workgroup"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("512")),
        "{asm}"
    );
    assert!(!asm.contains("DescriptorSet"), "{asm}");
    assert!(!asm.contains("Binding"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_raw_workgroup_vector_view_is_valid_before_retries() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(3) %tile, ptr addrspace(1) %out, i32 %index) {
entry:
  %value = call <3 x half> @load3(ptr addrspace(3) %tile, i32 %index)
  store <3 x half> %value, ptr addrspace(1) %out, align 8
  ret void
}

define internal <3 x half> @load3(ptr addrspace(3) %tile, i32 %index) {
entry:
  %wide = getelementptr <4 x half>, ptr addrspace(3) %tile, i32 %index
  %alias = bitcast ptr addrspace(3) %wide to ptr addrspace(3)
  %value = load <3 x half>, ptr addrspace(3) %alias, align 8
  ret <3 x half> %value
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"tile"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"half3*", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_raw_workgroup_vector_view_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_native_no_retry(ll, Stage::Kernel).expect("primary translate");
    tools::spirv_val_bytes(&spv, &tmp).expect("primary SPIR-V validates");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Workgroup"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
}

#[test]
fn native_kernel_threadgroup_param_direct_scalar_load_uses_element_zero() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %slot = getelementptr inbounds float, ptr addrspace(3) %temp, i64 %idx
  store float 2.000000e+00, ptr addrspace(3) %temp, align 4
  %root = load float, ptr addrspace(3) %temp, align 4
  store float %root, ptr addrspace(3) %slot, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threadgroup_direct_scalar_root_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let workgroup_vars = module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            (inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup)))
            .then_some(inst.result_id?)
        })
        .collect::<HashSet<_>>();
    assert!(!workgroup_vars.is_empty(), "{asm}");
    let direct_workgroup_load = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            inst.class.opcode == Op::Load
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_vars.contains(&id))
        });
    assert!(!direct_workgroup_load, "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

/// SPIRV-Cross emits a SPIR-V value with a single reader inline in the expression that reads it.
/// When the reader is an `OpSelect`, a subgroup instruction becomes an MSL ternary arm and only
/// the lanes taking that arm execute it -- which breaks Metal's requirement that a shuffle's
/// source lane take part in the call. Every such result is spilled through a `Function` variable,
/// because an `OpStore` is a statement and can never become a ternary arm.
#[test]
fn subgroup_results_a_select_reads_are_spilled_to_a_variable() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out, i16 %lane) {
entry:
  %x = uitofp i16 %lane to float
  %up = tail call float @air.simd_shuffle_up.f32(float %x, i16 1)
  %down = tail call float @air.simd_shuffle_down.f32(float %x, i16 2)
  %hi = fadd float %down, 1.000000e+00
  %c = icmp ult i16 %lane, 30
  %pick = select i1 %c, float %hi, float %up
  store float %pick, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_shuffle_up.f32(float, i16)
declare float @air.simd_shuffle_down.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_subgroup_materialize_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let shuffles = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        // Both AIR shuffles lower to `OpGroupNonUniformShuffle` with a computed source lane.
        .filter(|inst| inst.class.opcode == Op::GroupNonUniformShuffle)
        .filter_map(|inst| inst.result_id)
        .collect::<HashSet<_>>();
    assert_eq!(shuffles.len(), 2, "{asm}");
    // Each shuffle's only reader must be the store that materializes it.
    for shuffle in &shuffles {
        let readers = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.operands.contains(&Operand::IdRef(*shuffle)))
            .collect::<Vec<_>>();
        assert_eq!(readers.len(), 1, "{asm}");
        assert_eq!(readers[0].class.opcode, Op::Store, "{asm}");
    }
    // ... and no OpSelect may read a subgroup result directly any more.
    let select_reads_a_shuffle = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::Select)
        .any(|inst| {
            inst.operands
                .iter()
                .any(|operand| matches!(operand, Operand::IdRef(id) if shuffles.contains(id)))
        });
    assert!(!select_reads_a_shuffle, "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

/// A `[[threadgroup(n)]]` buffer's `Workgroup` array is sized by the caller, because Metal binds
/// its length at encode time and Vulkan fixes it in the module. With no length the array keeps the
/// translator's 512-element default; with one it holds exactly `length / element_size` elements. A
/// kernel that indexes past whatever length is emitted writes nowhere and reads zero, so the two
/// arms below are the difference between a dispatch that computes and one that silently does not.
#[test]
fn kernel_threadgroup_array_length_follows_the_caller() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %slot = getelementptr inbounds i32, ptr addrspace(3) %temp, i64 %idx
  store i32 7, ptr addrspace(3) %slot, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_array_length_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let elements = |options: crate::passes::TransformOptions| -> u32 {
        let spv = crate::translate_sanitized_native_with_options(ll, Stage::Kernel, &tmp, options)
            .expect("translate");
        let module = load_bytes(&spv).expect("load spv");
        let constants = module
            .types_global_values
            .iter()
            .filter_map(|inst| match (inst.class.opcode, inst.operands.first()) {
                (Op::Constant, Some(Operand::LiteralBit32(value))) => {
                    Some((inst.result_id?, *value))
                }
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        module
            .types_global_values
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == Op::TypeArray)
                    .then(|| {
                        constants
                            .get(&id_ref_operand(inst.operands.get(1)?)?)
                            .copied()
                    })
                    .flatten()
            })
            .expect("a Workgroup array")
    };
    assert_eq!(elements(whole_workgroup_options()), 512);
    assert_eq!(
        elements(
            whole_workgroup_options()
                .with_threadgroup_memory_length(0, 4352)
                .expect("length")
        ),
        1088
    );
    // A trailing partial element is not addressable, so the length rounds DOWN.
    assert_eq!(
        elements(
            whole_workgroup_options()
                .with_threadgroup_memory_length(0, 4354)
                .expect("length")
        ),
        1088
    );
    // Too small to hold one element, and a zero length, are caller mistakes rather than shapes to
    // emit -- SPIR-V has no zero-length array.
    let too_small = whole_workgroup_options()
        .with_threadgroup_memory_length(0, 2)
        .expect("length");
    let error = crate::translate_sanitized_native_with_options(ll, Stage::Kernel, &tmp, too_small)
        .expect_err("a two-byte binding holds no uint");
    assert!(error.contains("less than one 4-byte element"), "{error}");
    assert!(
        crate::passes::TransformOptions::default()
            .with_threadgroup_memory_length(0, 0)
            .is_err(),
        "a zero length must be refused"
    );
}

/// A threadgroup element the layout pass builds is sized by the AIR struct, not by the one word
/// `spirv_size_align` answers for an id it cannot find.
///
/// `stage_input` collects its type definitions from the module before it starts synthesizing
/// types, so the element type `build_workgroup_air_type` just made is not in that map, and
/// `spirv_size_align` returns `(4, 4)` for anything it cannot look up. The caller's byte length was
/// therefore divided by 4 for every AIR-layout element: this 80-byte struct measured 4 bytes, so a
/// 4000-byte binding named 1000 elements -- 80000 bytes -- rather than 50.
///
/// The uncapped guess is the same fact seen from the other side. 512 elements of this struct is
/// 40960 bytes, and Metal refuses any pipeline declaring more than 32768: three corpus sources are
/// modules no driver can load. The guess is now capped by what can exist; a length the caller named
/// is still honoured as given.
#[test]
fn kernel_threadgroup_element_size_comes_from_the_air_struct() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

%struct.Wide = type { i32, [3 x <4 x float>], i32, i32, i32, i32 }

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %slot = getelementptr inbounds %struct.Wide, ptr addrspace(3) %temp, i64 %idx, i32 0
  store i32 7, ptr addrspace(3) %slot, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !4, !"air.arg_type_size", i32 80, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Wide", !"air.arg_name", !"temp"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"type", i32 16, i32 48, i32 0, !"float3x3", !"transform", i32 64, i32 4, i32 0, !"uint", !"numInliers", i32 68, i32 4, i32 0, !"int", !"err"}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_element_size_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    // The declared element count of the Workgroup variable, whose element is the 80-byte AIR
    // struct. Walk the variable to its array rather than taking the first array in the module --
    // the struct itself contains one.
    let elements = |options: crate::passes::TransformOptions| -> u32 {
        let spv = crate::translate_sanitized_native_with_options(ll, Stage::Kernel, &tmp, options)
            .expect("translate");
        let module = load_bytes(&spv).expect("load spv");
        let constants = module
            .types_global_values
            .iter()
            .filter_map(|inst| match (inst.class.opcode, inst.operands.first()) {
                (Op::Constant, Some(Operand::LiteralBit32(value))) => {
                    Some((inst.result_id?, *value))
                }
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        let types = module
            .types_global_values
            .iter()
            .filter_map(|inst| Some((inst.result_id?, inst)))
            .collect::<std::collections::HashMap<_, _>>();
        // Walk the Workgroup variable to its array rather than taking the first array in the
        // module: the AIR struct itself contains one.
        let pointer = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.first()
                        == Some(&Operand::StorageClass(spirv::StorageClass::Workgroup))
            })
            .and_then(|inst| inst.result_type)
            .expect("a Workgroup variable");
        let array = types
            .get(&pointer)
            .and_then(|inst| id_ref_operand(inst.operands.get(1)?))
            .expect("the variable's pointee");
        let length = types
            .get(&array)
            .filter(|inst| inst.class.opcode == Op::TypeArray)
            .and_then(|inst| id_ref_operand(inst.operands.get(1)?))
            .expect("a Workgroup array");
        constants.get(&length).copied().expect("its length")
    };
    // 4000 bytes of an 80-byte element is 50 elements. Before this, it was 1000 -- 80000 bytes.
    assert_eq!(
        elements(
            whole_workgroup_options()
                .with_threadgroup_memory_length(0, 4000)
                .expect("length")
        ),
        50
    );
    // Rounds DOWN to whole elements: 4079 bytes still names 50.
    assert_eq!(
        elements(
            whole_workgroup_options()
                .with_threadgroup_memory_length(0, 4079)
                .expect("length")
        ),
        50
    );
    // Below one element is a caller mistake, and the message now names the real element size.
    let error = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options()
            .with_threadgroup_memory_length(0, 40)
            .expect("length"),
    )
    .expect_err("40 bytes holds no 80-byte element");
    assert!(error.contains("less than one 80-byte element"), "{error}");
    // With no caller length the guess is capped at what a Metal pipeline may declare: 512 of these
    // would be 40960 bytes, which no device accepts.
    let guessed = elements(whole_workgroup_options());
    assert_eq!(guessed, 409);
    assert!(
        guessed * 80 <= 32768,
        "the guess must name memory a pipeline can declare: {} bytes",
        guessed * 80
    );
}

/// The same rule holds when native emission gave the parameter its own raw word array rather than
/// letting `stage_input` build one from the AIR layout. A body that stores a word and reads a
/// narrow scalar through one `[[threadgroup]]` pointer decodes as raw words, and that array used
/// to keep the emitter's fixed 2048-word length no matter what the caller bound -- so a caller who
/// bound 12288 bytes got a module whose accesses past 8192 wrote nowhere and read zero, with no
/// diagnostic and a clean `spirv-val`. 142 of 14579 corpus sources emit this array.
#[test]
fn kernel_raw_threadgroup_array_length_also_follows_the_caller() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %wide = getelementptr inbounds i32, ptr addrspace(3) %temp, i64 0
  store i32 0, ptr addrspace(3) %wide, align 4
  %idx = zext i32 %i to i64
  %slot = getelementptr inbounds i16, ptr addrspace(3) %temp, i64 %idx
  %v = load i16, ptr addrspace(3) %slot, align 2
  %w = add i16 %v, 1
  store i16 %w, ptr addrspace(3) %slot, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_raw_threadgroup_array_length_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let elements = |options: crate::passes::TransformOptions| -> u32 {
        let spv = crate::translate_sanitized_native_with_options(ll, Stage::Kernel, &tmp, options)
            .expect("translate");
        let module = load_bytes(&spv).expect("load spv");
        let constants = module
            .types_global_values
            .iter()
            .filter_map(|inst| match (inst.class.opcode, inst.operands.first()) {
                (Op::Constant, Some(Operand::LiteralBit32(value))) => {
                    Some((inst.result_id?, *value))
                }
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        module
            .types_global_values
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == Op::TypeArray)
                    .then(|| {
                        constants
                            .get(&id_ref_operand(inst.operands.get(1)?)?)
                            .copied()
                    })
                    .flatten()
            })
            .expect("a Workgroup array")
    };
    // No caller fact: the emitter's own length stands, so nothing moves for a caller who names
    // none -- which is every CLI translation.
    assert_eq!(elements(whole_workgroup_options()), 2048);
    assert_eq!(
        elements(
            whole_workgroup_options()
                .with_threadgroup_memory_length(0, 12288)
                .expect("length")
        ),
        3072
    );
    // Shorter than the emitter's guess is just as much a caller fact as longer.
    assert_eq!(
        elements(
            whole_workgroup_options()
                .with_threadgroup_memory_length(0, 1024)
                .expect("length")
        ),
        256
    );
}

#[test]
fn native_wg_barrier_device_flag_orders_storage_buffer_memory() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %out) {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  tail call void @air.wg.barrier(i32 1, i32 1)
  %v = load i32, ptr addrspace(1) %out, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_wg_barrier_device_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    // 584 = AcquireRelease | UniformMemory | CrossWorkgroupMemory.
    assert!(asm.contains(" 584"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_threadgroup_atomic_i32_lowers_to_workgroup_spirv() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
%"struct.metal::_atomic" = type { i32 }
@local_counts = internal addrspace(3) global [1 x %"struct.metal::_atomic"] zeroinitializer, align 4

define void @k() {
entry:
  %p = getelementptr inbounds [1 x %"struct.metal::_atomic"], ptr addrspace(3) @local_counts, i64 0, i64 0, i32 0
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) %p, i32 0, i32 0, i32 1, i1 true)
  tail call void @air.wg.barrier(i32 2, i32 1)
  %v = tail call i32 @air.atomic.local.load.i32(ptr addrspace(3) %p, i32 0, i32 1, i1 true)
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %p, i32 1, i32 0, i32 1, i1 true)
  %sub_old = tail call i32 @air.atomic.local.sub.u.i32(ptr addrspace(3) nonnull captures(none) inttoptr (i64 1024 to ptr addrspace(3)), i32 1, i32 0, i32 1, i1 true)
  %signed_old = tail call i32 @air.atomic.local.add.s.i32(ptr addrspace(3) %p, i32 -1, i32 0, i32 1, i1 true)
  %signed_max_old = tail call i32 @air.atomic.local.max.s.i32(ptr addrspace(3) %p, i32 -3, i32 0, i32 1, i1 true)
  %max_old = tail call i32 @air.atomic.local.max.u.i32(ptr addrspace(3) %p, i32 9, i32 0, i32 1, i1 true)
  %signed_min_old = tail call i32 @air.atomic.local.min.s.i32(ptr addrspace(3) %p, i32 -9, i32 0, i32 1, i1 true)
  %min_old = tail call i32 @air.atomic.local.min.u.i32(ptr addrspace(3) %p, i32 3, i32 0, i32 1, i1 true)
  %masked = tail call i32 @air.atomic.local.and.u.i32(ptr addrspace(3) %p, i32 255, i32 0, i32 1, i1 true)
  %mask = tail call i32 @air.atomic.local.or.u.i32(ptr addrspace(3) %p, i32 2, i32 0, i32 1, i1 true)
  ret void
}

declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)
declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.load.i32(ptr addrspace(3), i32, i32, i1)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.sub.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.add.s.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.max.s.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.max.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.min.s.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.min.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.and.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.or.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Workgroup"), "{asm}");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    assert!(asm.contains("OpAtomicStore"), "{asm}");
    assert!(asm.contains("OpAtomicLoad"), "{asm}");
    assert!(asm.matches("OpAtomicIAdd").count() >= 2, "{asm}");
    assert!(asm.contains("OpAtomicISub"), "{asm}");
    assert!(asm.contains("OpAtomicSMax"), "{asm}");
    assert!(asm.contains("OpAtomicUMax"), "{asm}");
    assert!(asm.contains("OpAtomicSMin"), "{asm}");
    assert!(asm.contains("OpAtomicUMin"), "{asm}");
    assert!(asm.contains("OpAtomicAnd"), "{asm}");
    assert!(asm.contains("OpAtomicOr"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_threadgroup_atomic_fixed_loop_is_flattened_and_unrolled() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
%"struct.metal::_atomic" = type { i32 }
@local_counts = internal addrspace(3) global [4 x %"struct.metal::_atomic"] zeroinitializer, align 4

define void @k() {
entry:
  %initp = getelementptr inbounds [4 x %"struct.metal::_atomic"], ptr addrspace(3) @local_counts, i64 0, i64 0, i32 0
  store i32 0, ptr addrspace(3) %initp, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %bin64 = zext i32 %i to i64
  %p = getelementptr inbounds [4 x %"struct.metal::_atomic"], ptr addrspace(3) @local_counts, i64 0, i64 %bin64, i32 0
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %p, i32 1, i32 0, i32 1, i1 true)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  tail call void @air.wg.barrier(i32 2, i32 1)
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_unroll_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpTypeStruct %uint"), "{asm}");
    assert_eq!(asm.matches("OpAtomicIAdd").count(), 4, "{asm}");
    assert!(!asm.contains("OpLoopMerge"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_simdgroup_matrix_8x8_lowers_through_scalar_array() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %out) {
entry:
  %a = insertelement <64 x float> zeroinitializer, float 1.000000e+00, i64 0
  %b = insertelement <64 x float> zeroinitializer, float 2.000000e+00, i64 0
  %c = insertelement <64 x float> zeroinitializer, float 3.000000e+00, i64 0
  %m = tail call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %b, <64 x float> %c)
  %lane = extractelement <64 x float> %m, i64 0
  store float %lane, ptr addrspace(1) %out, align 4
  ret void
}

declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_matrix_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpTypeVector %float 64"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_agx2_distributed_matmad_lowers_to_partitioned_subgroup_shuffles() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %out_f, ptr addrspace(1) %out_h) {
entry:
  %f8 = call <2 x float> @llvm.agx2.f32matmad8x8.v2f32(<2 x float> zeroinitializer, <2 x float> zeroinitializer, <2 x float> zeroinitializer)
  %f4 = call <2 x float> @llvm.agx2.f32matmad4x4.v2f32(<2 x float> %f8, <2 x float> %f8, <2 x float> %f8)
  %h8 = call <2 x half> @llvm.agx2.f16matmad8x8.v2f16(<2 x half> zeroinitializer, <2 x half> zeroinitializer, <2 x half> zeroinitializer)
  %h4 = call <2 x half> @llvm.agx2.f16matmad4x4.v2f16(<2 x half> %h8, <2 x half> %h8, <2 x half> %h8)
  store <2 x float> %f4, ptr addrspace(1) %out_f, align 8
  store <2 x half> %h4, ptr addrspace(1) %out_h, align 4
  ret void
}

declare <2 x float> @llvm.agx2.f32matmad8x8.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <2 x float> @llvm.agx2.f32matmad4x4.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <2 x half> @llvm.agx2.f16matmad8x8.v2f16(<2 x half>, <2 x half>, <2 x half>)
declare <2 x half> @llvm.agx2.f16matmad4x4.v2f16(<2 x half>, <2 x half>, <2 x half>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float2*", !"air.arg_name", !"out_f"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"half2*", !"air.arg_name", !"out_h"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_agx2_distributed_matmad_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("llvm.agx2."), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(asm.contains("BuiltIn SubgroupLocalInvocationId"), "{asm}");
    assert_eq!(
        asm.lines()
            .filter(|line| line.contains("= OpGroupNonUniformShuffle "))
            .count(),
        72,
        "{asm}"
    );
    assert_eq!(asm.matches(" Fma ").count(), 48, "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_simdgroup_matrix_16x16_distributed_mac_lowers_and_validates() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i16 %lane, ptr addrspace(1) %out_f, ptr addrspace(1) %out_i) {
entry:
  %transpose = icmp eq i16 %lane, 0
  %f = call <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32(<8 x half> zeroinitializer, i1 %transpose, <8 x half> zeroinitializer, i1 true, <8 x float> zeroinitializer)
  %i = call <8 x i32> @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.u.v8i32.v8i8.v8i8.v8i32(<8 x i8> zeroinitializer, i1 false, <8 x i8> zeroinitializer, i1 %transpose, <8 x i32> zeroinitializer)
  store <8 x float> %f, ptr addrspace(1) %out_f, align 32
  store <8 x i32> %i, ptr addrspace(1) %out_i, align 32
  ret void
}

declare <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32(<8 x half>, i1, <8 x half>, i1, <8 x float>)
declare <8 x i32> @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.u.v8i32.v8i8.v8i8.v8i32(<8 x i8>, i1, <8 x i8>, i1, <8 x i32>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float8*", !"air.arg_name", !"out_f"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"int8*", !"air.arg_name", !"out_i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_matrix_16x16_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("simdgroup_matrix_16x16x16"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(asm.contains("BuiltIn SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpSConvert"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(asm.contains(" Fma "), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_agx3_packed_igemm_reuses_distributed_matrix_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %out) {
entry:
  %result = call <8 x i32> @llvm.agx3.igemm.v8i32.i64.i64.v8i32(i8 16, i8 16, i8 16, i16 9, i64 578437695752307201, i16 75, i64 1157159078456920585, i16 75, <8 x i32> zeroinitializer, i16 9)
  store <8 x i32> %result, ptr addrspace(1) %out, align 32
  ret void
}

declare <8 x i32> @llvm.agx3.igemm.v8i32.i64.i64.v8i32(i8, i8, i8, i16, i64, i16, i64, i16, <8 x i32>, i16)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"int8*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_agx3_packed_igemm_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("llvm.agx3.igemm"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(asm.contains("OpSConvert"), "{asm}");
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_simdgroup_matrix_16x16_all_observed_float_encodings_validate() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %out) {
entry:
  %f32 = call <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f32.v8f32.v8f32(<8 x float> zeroinitializer, i1 false, <8 x float> zeroinitializer, i1 false, <8 x float> zeroinitializer)
  store <8 x float> %f32, ptr addrspace(1) %out, align 32
  %bf16 = call <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8bf16.v8f32(<8 x bfloat> zeroinitializer, i1 false, <8 x bfloat> zeroinitializer, i1 false, <8 x float> zeroinitializer)
  store <8 x float> %bf16, ptr addrspace(1) %out, align 32
  %e4m3 = call <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3.v8f16.v8f32(<8 x i8> zeroinitializer, i1 false, <8 x half> zeroinitializer, i1 false, <8 x float> zeroinitializer)
  store <8 x float> %e4m3, ptr addrspace(1) %out, align 32
  %e4m3fn = call <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e4m3fn.v8f32(<8 x i8> zeroinitializer, i1 false, <8 x i8> zeroinitializer, i1 false, <8 x float> zeroinitializer)
  store <8 x float> %e4m3fn, ptr addrspace(1) %out, align 32
  %e5m2 = call <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e5m2.v8f8e5m2.v8f32(<8 x i8> zeroinitializer, i1 false, <8 x i8> zeroinitializer, i1 false, <8 x float> zeroinitializer)
  store <8 x float> %e5m2, ptr addrspace(1) %out, align 32
  ret void
}

declare <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f32.v8f32.v8f32(<8 x float>, i1, <8 x float>, i1, <8 x float>)
declare <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8bf16.v8f32(<8 x bfloat>, i1, <8 x bfloat>, i1, <8 x float>)
declare <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3.v8f16.v8f32(<8 x i8>, i1, <8 x half>, i1, <8 x float>)
declare <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e4m3fn.v8f32(<8 x i8>, i1, <8 x i8>, i1, <8 x float>)
declare <8 x float> @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e5m2.v8f8e5m2.v8f32(<8 x i8>, i1, <8 x i8>, i1, <8 x float>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float8*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_matrix_16x16_float_types_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("simdgroup_matrix_16x16x16"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains(" Ldexp "), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_simdgroup_matrix_8x8_full_pipeline_lowers_and_validates() {
    // Exercises all four simdgroup_matrix 8x8 lowerings end to end: load (f16 and f32) from device
    // buffers, init_diag, mixed-precision multiply_accumulate with both f32 and f16 results, and store
    // to a device buffer. The descriptor vectors carry the documented `<elements_per_row, 8>` / `<1,
    // elements_per_row>` shape (leading dimension = component 0 of the first descriptor vector).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %in_h, ptr addrspace(1) %in_f, ptr addrspace(1) %out) {
entry:
  %pv1 = insertelement <2 x i64> <i64 poison, i64 8>, i64 8, i64 0
  %pv2 = insertelement <2 x i64> <i64 1, i64 poison>, i64 8, i64 1
  %ph = getelementptr inbounds half, ptr addrspace(1) %in_h, i64 0
  %pf = getelementptr inbounds float, ptr addrspace(1) %in_f, i64 0
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i64 0
  %A = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) %ph, <2 x i64> %pv1, <2 x i64> %pv2, <2 x i64> zeroinitializer)
  %B = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %pf, <2 x i64> %pv1, <2 x i64> %pv2, <2 x i64> zeroinitializer)
  %C = call <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float 1.000000e+00)
  %D = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %B, <64 x half> %A, <64 x float> %C)
  %Dh = call <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f32.v64f16.v64f32(<64 x float> %B, <64 x half> %A, <64 x float> %C)
  %dh0 = extractelement <64 x half> %Dh, i64 0
  %dhf = fpext half %dh0 to float
  store float %dhf, ptr addrspace(1) %po
  %scaled = fmul fast <64 x float> %D, %D
  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %scaled, ptr addrspace(1) %po, <2 x i64> %pv1, <2 x i64> %pv2, <2 x i64> zeroinitializer)
  ret void
}

declare <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float>, <64 x half>, <64 x float>)
declare <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f32.v64f16.v64f32(<64 x float>, <64 x half>, <64 x float>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"in_h"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"in_f"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_matrix_full_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    // The <64 x float> matrix must be modeled as an array, and the wide fmul must be scalarized
    // (no OpFMul over a 64-lane array type).
    assert!(!asm.contains("OpTypeVector %float 64"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
}

/// A kernel's DECLARED `[[thread_index_in_simdgroup]]` names the same lane the shuffle beside it
/// reads, so both come from one variable.
///
/// `air.simd_shuffle(v, lane)` is defined to return `v` when `lane` is the caller's own index in
/// the simdgroup, and this module asks for exactly that. The shuffle lowering has always
/// partitioned `SubgroupLocalInvocationId` into Metal-width runs; the entry-parameter role instead
/// masked `LocalInvocationIndex`, which names the same lane only on a driver that happens to cut
/// subgroups out of the threadgroup in index order -- something Vulkan leaves
/// implementation-defined. Two derivations of one fact, and the module carried both.
#[test]
fn native_declared_simdgroup_lane_is_the_lane_the_shuffle_reads() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, i16 %lane, ptr addrspace(1) %out) {
entry:
  %same = tail call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %lane)
  %first = tail call i1 @air.simd_is_first()
  %word = select i1 %first, i32 %same, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.simd_shuffle.u.i32(i32, i16)
declare i1 @air.simd_is_first()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!4 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_declared_simd_lane_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    // `air.simd_is_first` synthesizes the lane variable too. One decoration is therefore the
    // statement that the declared role and the intrinsic read the SAME lane, not two.
    assert_eq!(
        asm.matches("BuiltIn SubgroupLocalInvocationId").count(),
        1,
        "one lane variable, shared by the declared role and the intrinsic\n{asm}"
    );
    assert!(
        !asm.contains("BuiltIn LocalInvocationIndex"),
        "the lane inside a simdgroup is not a fact about the threadgroup\n{asm}"
    );
    let lane_var = asm
        .lines()
        .map(str::trim)
        .find(|line| line.contains("BuiltIn SubgroupLocalInvocationId"))
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("lane variable decoration")
        .to_string();
    assert!(
        asm.lines()
            .any(|line| line.contains(" = OpLoad ") && line.trim_end().ends_with(&lane_var)),
        "the declared lane loads that variable\n{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// AIR states a barrier's execution scope in its second operand, and both barrier intrinsics take
/// both values: 42 calls across 13 corpus sources are `air.simdgroup.barrier(flags, 1)`, a
/// threadgroup-wide barrier spelled through the simdgroup intrinsic. Reading the scope off the
/// callee name instead synchronized one simdgroup where Metal synchronizes the whole threadgroup.
#[test]
fn native_barrier_execution_scope_comes_from_the_operand_not_the_callee() {
    fn barrier_scopes(ll: &str, label: &str) -> Vec<(u32, u32, u32)> {
        let tmp = std::env::temp_dir().join(format!(
            "metal2vulkan_barrier_scope_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
        let module = load_bytes(&spv).expect("load spv");
        let constants: HashMap<Word, u32> = module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::Constant)
            .filter_map(|inst| match (inst.result_id, inst.operands.first()) {
                (Some(id), Some(Operand::LiteralBit32(value))) => Some((id, *value)),
                _ => None,
            })
            .collect();
        let resolve = |operand: &Operand| match operand {
            Operand::IdScope(id) | Operand::IdMemorySemantics(id) | Operand::IdRef(id) => {
                *constants.get(id).expect("barrier operand is a constant")
            }
            other => panic!("unexpected barrier operand {other:?}"),
        };
        let scopes = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == Op::ControlBarrier)
            .map(|inst| {
                (
                    resolve(&inst.operands[0]),
                    resolve(&inst.operands[1]),
                    resolve(&inst.operands[2]),
                )
            })
            .collect();
        let _ = std::fs::remove_dir_all(&tmp);
        scopes
    }

    let ll = |calls: &str| {
        format!(
            r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %out) {{
entry:
{calls}
  store i32 0, ptr addrspace(1) %out, align 4
  ret void
}}

declare void @air.wg.barrier(i32, i32)
declare void @air.simdgroup.barrier(i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"int*", !"air.arg_name", !"out"}}
"#
        )
    };

    let workgroup_memory = {
        use spirv::MemorySemantics;
        (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits()
    };
    // Same intrinsic, same memory flags, different stated scope: only the operand decides.
    assert_eq!(
        barrier_scopes(
            &ll("  tail call void @air.simdgroup.barrier(i32 2, i32 1)\n  \
                 tail call void @air.simdgroup.barrier(i32 2, i32 4)"),
            "simd"
        ),
        vec![
            (
                Scope::Workgroup as u32,
                Scope::Workgroup as u32,
                workgroup_memory
            ),
            (
                Scope::Subgroup as u32,
                Scope::Subgroup as u32,
                workgroup_memory
            ),
        ]
    );
    // `air.wg.barrier` is unchanged, including the imageblock memory flag, which names threadgroup
    // memory because the shared-cell imageblock lowering puts its cells in `Workgroup` storage.
    assert_eq!(
        barrier_scopes(&ll("  tail call void @air.wg.barrier(i32 8, i32 1)"), "wg"),
        vec![(
            Scope::Workgroup as u32,
            Scope::Workgroup as u32,
            workgroup_memory
        )]
    );

    // A scope AIR does not define fails visibly rather than picking one.
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_barrier_scope_bad_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let err = crate::translate_sanitized_native(
        &ll("  tail call void @air.wg.barrier(i32 2, i32 2)"),
        Stage::Kernel,
        &tmp,
    )
    .expect_err("undefined barrier scope");
    assert!(err.contains("execution scope 2"), "{err}");
    let _ = std::fs::remove_dir_all(&tmp);
}
