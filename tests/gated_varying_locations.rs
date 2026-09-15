//! One Metal varying struct, two stages, one set of `Location`s.
//!
//! A vertex shader's return struct and the fragment shader's parameter list are the same Metal
//! declaration compiled twice. Vulkan links the two by `Location`, and this translator assigns
//! those locations positionally from the AIR interface list — so the numbering rule has to be a
//! pure function of that list, identical in both stages. It was not. A `[[function_constant]]`
//! member reached the fragment decode through a reader that resolves the `air.function_constant`
//! wrapper and reached the vertex output decode through one that did not, so the vertex module
//! dropped the member and renumbered every varying after it one slot lower.
//!
//! Nothing downstream notices. Both modules pass `spirv-val`, both report the same reflection
//! varyings by name, and the pipeline links: the fragment simply reads a different interpolant
//! than the vertex wrote, for every varying declared after a gated one. 106 vertex and 209
//! fragment sources of a 14579-source local corpus declare a gated varying with a later ungated
//! one behind it — the exact shape that shifts.
//!
//! The oracle here is the pair, not either module: neither `Location` set is wrong on its own.

use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_gated_varying_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// The gate both modules share: one predicate, no static initializer, so it is unresolved at
/// translation time — the state every unspecialized translation of a `[[function_constant]]`
/// shader is in.
const GATE: &str = r#"@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
"#;

/// `struct V { float4 position [[position]]; float2 a [[function_constant(enabled)]]; float2 b; }`
/// as a vertex return value.
const VERTEX: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, <2 x float>, <2 x float> }> @vert(<4 x float> %p, <2 x float> %a, <2 x float> %b) {
entry:
  %r0 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> undef, <4 x float> %p, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> %r0, <2 x float> %a, 1
  %r2 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> %r1, <2 x float> %b, 2
  ret <{ <4 x float>, <2 x float>, <2 x float> }> %r2
}

!air.vertex = !{!0}
!air.function_constants = !{!10}
!0 = !{ptr @vert, !1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.function_constant", !10, !"air.vertex_output", !"generated(a)", !"air.arg_type_name", !"float2", !"air.arg_name", !"a"}
!4 = !{!"air.vertex_output", !"generated(b)", !"air.arg_type_name", !"float2", !"air.arg_name", !"b"}
!5 = !{!6, !7, !8}
!6 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!7 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"a"}
!8 = !{i32 2, !"air.vertex_input", !"air.location_index", i32 2, !"air.arg_type_name", !"float2", !"air.arg_name", !"b"}
!10 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, !"bool", !"enabled", i32 0, i1 false}
"#;

/// The same struct on the consuming side.
const FRAGMENT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @frag(<4 x float> %pos, <2 x float> %a, <2 x float> %b) {
entry:
  %x = extractelement <2 x float> %b, i32 0
  %y = extractelement <2 x float> %a, i32 1
  %v0 = insertelement <4 x float> undef, float %x, i32 0
  %v1 = insertelement <4 x float> %v0, float %y, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!air.function_constants = !{!10}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.function_constant", !10, !"air.fragment_input", !"generated(a)", !"air.center", !"air.arg_type_name", !"float2", !"air.arg_name", !"a"}
!6 = !{i32 2, !"air.fragment_input", !"generated(b)", !"air.center", !"air.arg_type_name", !"float2", !"air.arg_name", !"b"}
!10 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, !"bool", !"enabled", i32 0, i1 false}
"#;

/// Every `Location` in the entry point's interface, by storage class, sorted.
fn interface_locations(asm: &str, storage_class: &str) -> Vec<u32> {
    let ids: Vec<&str> = asm
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            let (id, rest) = line.split_once(" = OpVariable ")?;
            rest.split_whitespace()
                .last()
                .filter(|class| *class == storage_class)
                .map(|_| id)
        })
        .collect();
    let mut out: Vec<u32> = asm
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            let rest = line.strip_prefix("OpDecorate ")?;
            let (id, rest) = rest.split_once(' ')?;
            let location = rest.strip_prefix("Location ")?;
            ids.contains(&id).then(|| location.trim().parse().ok())?
        })
        .collect();
    out.sort_unstable();
    out
}

fn asm(ll: &str, stage: Stage) -> String {
    let spv = translate_sanitized_native(ll, stage, &tmp()).expect("translate");
    disassemble(&spv).expect("disassemble")
}

/// The vertex outputs and the fragment inputs of one struct occupy the same locations.
#[test]
fn a_gated_varying_occupies_the_same_location_in_both_stages() {
    let vertex = asm(&format!("{VERTEX}{GATE}"), Stage::Vertex);
    let fragment = asm(&format!("{FRAGMENT}{GATE}"), Stage::Fragment);

    let written = interface_locations(&vertex, "Output");
    let read = interface_locations(&fragment, "Input");
    assert_eq!(
        written, read,
        "the vertex writes {written:?} and the fragment reads {read:?} for the same varying \
         struct\n--- vertex ---\n{vertex}\n--- fragment ---\n{fragment}"
    );
    assert_eq!(
        written,
        vec![0, 1],
        "both varyings are declared, the gated one first:\n{vertex}"
    );
}

/// The ungated varying behind the gated one is the member that moves, so it is pinned by name.
///
/// A test that only compared the two location *sets* would still pass if both stages dropped the
/// gated member — and dropping it loses the varying outright whenever the predicate is enabled at
/// pipeline creation, which is the failure the set comparison cannot see.
#[test]
fn the_varying_behind_a_gated_one_keeps_its_declared_slot() {
    let vertex = asm(&format!("{VERTEX}{GATE}"), Stage::Vertex);
    let fragment = asm(&format!("{FRAGMENT}{GATE}"), Stage::Fragment);
    for (stage, asm, class) in [
        ("vertex", &vertex, "Output"),
        ("fragment", &fragment, "Input"),
    ] {
        assert!(
            interface_locations(asm, class).contains(&1),
            "{stage}: `b` is the second varying of the struct and keeps Location 1:\n{asm}"
        );
    }
}
