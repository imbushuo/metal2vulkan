//! A varying AIR declares reaches the module even when the shader never writes it.
//!
//! `stage_output` skipped the store for an output whose value is statically `OpUndef` — nothing
//! useful to write, so nothing written. The Output variable was then unreferenced, and
//! `module_cleanup`'s unreferenced-global rule removed it: variable, `Location` decoration and
//! entry-point interface entry together. That rule is right for a descriptor (a `Binding` no
//! instruction touches is a binding a consumer would have to satisfy for nothing) and wrong for a
//! stage output, which is one half of a linkage contract with the next stage.
//!
//! The fragment shader compiled from the same Metal varying struct declares the matching Input, and
//! Vulkan requires every consumed input to have a producing output at that `Location`. AIR declares
//! the member because Metal declares it; that the shader leaves it undefined makes its VALUE
//! undefined, not its existence. 27 vertex and 3 tessellation-evaluation sources of a 2880-source
//! corpus sample declared a varying the emitted module did not contain.
//!
//! Fragment outputs keep the old rule: an attachment is memory, and one nothing writes is better
//! left unwritten than written with garbage.

use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_declared_out_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// `struct V { float4 position [[position]]; float2 uv; float2 unwritten; }`. The shader writes
/// `position` and `uv` and leaves `unwritten` at its `undef` initial value — legal Metal, and the
/// shape a `[[function_constant]]`-gated member collapses to once the gated arm folds away.
const VERTEX: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, <2 x float>, <2 x float> }> @vert(<4 x float> %p, <2 x float> %uv) {
entry:
  %r0 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> undef, <4 x float> %p, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float>, <2 x float> }> %r0, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float>, <2 x float> }> %r1
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(uv)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!"air.vertex_output", !"generated(unwritten)", !"air.arg_type_name", !"float2", !"air.arg_name", !"unwritten"}
!5 = !{!6, !7}
!6 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!7 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

/// The consuming side of the same struct.
const FRAGMENT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @frag(<4 x float> %pos, <2 x float> %uv, <2 x float> %unwritten) {
entry:
  %x = extractelement <2 x float> %uv, i32 0
  %y = extractelement <2 x float> %unwritten, i32 1
  %v0 = insertelement <4 x float> undef, float %x, i32 0
  %v1 = insertelement <4 x float> %v0, float %y, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.fragment_input", !"generated(uv)", !"air.center", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!6 = !{i32 2, !"air.fragment_input", !"generated(unwritten)", !"air.center", !"air.arg_type_name", !"float2", !"air.arg_name", !"unwritten"}
"#;

/// A fragment that returns a struct whose second render target is never written.
const FRAGMENT_TWO_TARGETS: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, <4 x float> }> @frag(<4 x float> %pos) {
entry:
  %r0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %pos, 0
  ret <{ <4 x float>, <4 x float> }> %r0
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{!5}
!5 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
"#;

/// Every `Location` decorating a variable of `storage_class`, sorted.
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

#[test]
fn an_unwritten_vertex_varying_still_reaches_the_module() {
    let vertex = asm(VERTEX, Stage::Vertex);
    assert_eq!(
        interface_locations(&vertex, "Output"),
        vec![0, 1],
        "both declared varyings are Output variables, `unwritten` included:\n{vertex}"
    );
}

/// The property that matters is the pair: what the vertex produces and what the fragment consumes.
#[test]
fn the_fragment_reading_an_unwritten_varying_has_a_producer_for_it() {
    let vertex = asm(VERTEX, Stage::Vertex);
    let fragment = asm(FRAGMENT, Stage::Fragment);
    let written = interface_locations(&vertex, "Output");
    let read = interface_locations(&fragment, "Input");
    assert_eq!(
        written, read,
        "the vertex writes {written:?} and the fragment reads {read:?} for the same varying \
         struct\n--- vertex ---\n{vertex}\n--- fragment ---\n{fragment}"
    );
}

/// An attachment is not a linkage slot. A fragment output nothing writes stays unwritten rather
/// than being written with an undefined value, which is the rule this change deliberately keeps.
#[test]
fn an_unwritten_fragment_render_target_stays_unwritten() {
    let fragment = asm(FRAGMENT_TWO_TARGETS, Stage::Fragment);
    assert_eq!(
        interface_locations(&fragment, "Output"),
        vec![0],
        "only the render target the shader writes is emitted:\n{fragment}"
    );
}
