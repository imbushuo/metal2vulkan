//! An `air.function_constant` wrapper does not change a stage input's interface.
//!
//! A `[[function_constant]]`-gated entry parameter is the same parameter, declared conditionally.
//! Metal's own reflection reports it, the application binds it, and if the constant is set at
//! pipeline creation the shader reads it. The wrapper says WHEN it is live, not WHETHER it exists.
//!
//! The fragment decode has always read a wrapped role straight through the marker, so a gated
//! `air.fragment_input` was always the varying it declares. The vertex decode read its roles through
//! a resource-promotion list that named no stage-input role, so a gated `air.vertex_input` became no
//! input variable at all: the parameter lowered to `OpUndef`, reflection omitted the attribute, and
//! a pipeline created with the constant enabled read undefined data where the application had bound
//! a vertex buffer. Nothing reports it -- the module validates, and the missing attribute is missing
//! from reflection too, so the two agree with each other about the wrong answer. 573 gated
//! `air.vertex_input` arguments in 89 of 14579 local corpus sources.
//!
//! The oracle is the PAIR of modules, not either one: same declaration, one with the wrapper and one
//! without, must translate to the same interface.

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::ResourceKind;
use metal2vulkan::{disassemble, reflect_sanitized, translate_sanitized_native};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_gated_inputs_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// The gate the wrapped variants reference: one predicate with no static initializer, so it is
/// unresolved at translation time -- the state every unspecialized translation is in.
const GATE: &str = r#"
@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
!air.function_constants = !{!20}
!20 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, !"bool", !"enabled", i32 0, i1 false}
"#;

/// A vertex shader with two attributes; `WRAP` marks where the wrapper goes on the second.
const VERTEX: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, <2 x float> }> @vert(<4 x float> %p, <2 x float> %uv) {
entry:
  %r0 = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %p, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float> }> %r0, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float> }> %r1
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(uv)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!6 = !{i32 1, WRAP!"air.vertex_input", !"air.location_index", i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

/// The mirror on the fragment side, whose decode already answered this correctly.
const FRAGMENT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @frag(<4 x float> %pos, <2 x float> %uv) {
entry:
  %x = extractelement <2 x float> %uv, i32 0
  %v0 = insertelement <4 x float> undef, float %x, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, WRAP!"air.fragment_input", !"generated(uv)", !"air.center", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

/// A skinning-shaped kernel: an ungated `[[stage_in]]` attribute and a gated one, in the shape the
/// corpus declares them (`position` always, `normal` behind a `needNormal` constant).
const KERNEL: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define void @skin(i32 %index, <3 x float> %position, <3 x float> %normal, ptr addrspace(1) %out) {
entry:
  %sum = fadd <3 x float> %position, %normal
  %slot = getelementptr <3 x float>, ptr addrspace(1) %out, i32 %index
  store <3 x float> %sum, ptr addrspace(1) %slot, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @skin, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
!4 = !{i32 1, !"air.stage_in", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
!5 = !{i32 2, WRAP!"air.stage_in", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"normal"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"packed_float3", !"air.arg_name", !"out"}
"#;

/// The template with and without the `air.function_constant` wrapper on the second parameter.
fn pair(template: &str) -> (String, String) {
    (
        format!("{}{GATE}", template.replace("WRAP", "")),
        format!(
            "{}{GATE}",
            template.replace("WRAP", "!\"air.function_constant\", !20, ")
        ),
    )
}

fn asm(ll: &str, stage: Stage) -> String {
    let spv = translate_sanitized_native(ll, stage, &tmp()).expect("translate");
    disassemble(&spv).expect("disassemble")
}

/// A gated vertex attribute is the attribute it declares, in the module and in reflection.
#[test]
fn a_gated_vertex_attribute_declares_the_same_interface_as_an_ungated_one() {
    let (bare, gated) = pair(VERTEX);
    assert_eq!(
        asm(&gated, Stage::Vertex),
        asm(&bare, Stage::Vertex),
        "the wrapper is when the attribute is live, not whether it exists"
    );

    let reflect = |ll: &str| {
        reflect_sanitized(ll, Stage::Vertex, TransformOptions::default())
            .expect("reflect")
            .vertex_attributes
    };
    let attributes = reflect(&gated);
    assert_eq!(attributes, reflect(&bare));
    assert_eq!(
        attributes.iter().map(|a| a.location).collect::<Vec<_>>(),
        vec![0, 1],
        "both attributes are reported, so an application binds a vertex buffer for both"
    );
}

/// The fragment side of the same rule, pinned so the two decodes cannot drift apart again.
#[test]
fn a_gated_fragment_varying_declares_the_same_interface_as_an_ungated_one() {
    let (bare, gated) = pair(FRAGMENT);
    assert_eq!(
        asm(&gated, Stage::Fragment),
        asm(&bare, Stage::Fragment),
        "the fragment decode has always read the wrapped role, and still must"
    );
}

/// A kernel `[[stage_in]]` attribute is a descriptor-backed stream, and gating it does not make the
/// stream go away -- it made the parameter `OpUndef`, so the kernel read nothing where the runtime
/// had a vertex stream. 120 gated `air.stage_in` arguments in 52 of 14579 local corpus sources; 12
/// of the 2880-source sample gain 24 real module descriptors, not just 24 reflected ones.
#[test]
fn a_gated_kernel_stage_input_declares_the_same_interface_as_an_ungated_one() {
    let (bare, gated) = pair(KERNEL);
    assert_eq!(
        asm(&gated, Stage::Kernel),
        asm(&bare, Stage::Kernel),
        "the wrapper is when the stream is live, not whether it exists"
    );

    let streams = |ll: &str| {
        reflect_sanitized(ll, Stage::Kernel, TransformOptions::default())
            .expect("reflect")
            .bindings
            .into_iter()
            .filter(|binding| binding.kind == ResourceKind::KernelStageInput)
            .count()
    };
    assert_eq!(streams(&gated), streams(&bare));
    assert_eq!(streams(&gated), 2, "both declared streams are descriptors");
}
