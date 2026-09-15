//! Negative / must-FALLBACK suite (refactor safety net T4).
//!
//! The crate's floor-safety argument is that unsupported inputs FALLBACK *cleanly* — a
//! `translate_sanitized_native` `Err`, never wrong-but-valid SPIR-V that a downstream gate would
//! wave through. These tests pin that behaviour for the known-unsupported classes so a refactor
//! (especially S23, panic→Result) can't silently turn a clean FALLBACK into a translate that
//! "succeeds" with garbage — or into a process abort.
//!
//! Covers known-unsupported classes: `air.intersect.*` raytracing, `llvm.agx3.*` emask
//! intrinsics, texture atomics — plus structural malformations (no definitions, truncated).
//!
//! One of them is a GAP rather than a limit: the byte-view vector store below has a lowering, it
//! just has not been written, and the reason it is pinned is that the obvious ways to write it are
//! wrong. Closing it must be a deliberate change to this test, not a silent pass.
//!
//! One class here is not an unsupported *construct* but an unmodelled *role*: an AIR stage output
//! the emitter has no case for. Those used to be skipped silently, which is the one outcome the
//! floor-safety argument does not survive — the module validates and reflects identically while the
//! value the shader computed reaches nothing. They are pinned here with their positive controls.
//!
//! One case here is deliberately POSITIVE. A rejection is only pinned from one side: a test that
//! asserts a class FALLBACKs cannot notice a change that starts rejecting inputs that used to
//! translate. The function-constant-gated pair below asserts both directions over a single
//! template, so the boundary itself is what is pinned, not just the far side of it.
//!
//! Two of these are the classes that actually turn up. Measured over a 2880-source local corpus
//! sample, 204 sources do not translate, and 191 of them are one of two shapes: a call through a
//! Metal visible function table, or an intersection whose custom intersection functions come from a
//! function buffer. Both are function pointers, which Logical SPIR-V has none of — no lowering is
//! coming, so what matters is that the rejection stays a rejection. Each is pinned below on the
//! exact diagnostic those corpus sources produce.

use metal2vulkan::passes::Stage;
use metal2vulkan::translate_sanitized_native;
use std::env;
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = env::temp_dir().join(format!("m2v_must_fallback_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Assert the input FALLBACKs cleanly (Err) and the diagnostic mentions `needle` — i.e. it is the
/// *intended* rejection, not an unrelated parse failure.
fn assert_fallback(ll: &str, needle: &str) {
    match translate_sanitized_native(ll, Stage::Kernel, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK (Err) but translate succeeded ({} bytes); \
             wrong-but-valid SPIR-V defeats the floor-safety guarantee",
            spv.len()
        ),
        Err(e) => assert!(
            e.contains(needle),
            "FALLBACK diagnostic should mention {needle:?}; got: {e}"
        ),
    }
}

// A minimal, genuinely-translatable kernel; each negative case injects exactly one unsupported
// construct into `%OP` so the FALLBACK is attributable to that construct and nothing else.
const HEAD: &str = r#"
target triple = "air64_v28-apple-macosx26.5.0"

%Input = type { [4 x i32] }
%Output = type { [4 x i32] }

define void @k(ptr addrspace(2) %in, ptr addrspace(1) %out) {
entry:
  %a0p = getelementptr inbounds %Input, ptr addrspace(2) %in, i64 0, i32 0, i64 0
  %a0 = load i32, ptr addrspace(2) %a0p
"#;

const TAIL: &str = r#"
  %o0 = getelementptr inbounds %Output, ptr addrspace(1) %out, i64 0, i32 0, i64 0
  store i32 %r, ptr addrspace(1) %o0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"Input", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"Output", !"air.arg_name", !"out"}
!5 = !{i32 0, i32 16, i32 0, !"uint", !"v0", i32 4, i32 4, i32 0, !"uint", !"v1", i32 8, i32 4, i32 0, !"uint", !"v2", i32 12, i32 4, i32 0, !"uint", !"v3"}
"#;

fn kernel_with(op: &str) -> String {
    format!("{HEAD}{op}{TAIL}")
}

/// As [`kernel_with`], for a construct that also needs its callee declared.
fn kernel_with_declarations(op: &str, declarations: &str) -> String {
    format!("{HEAD}{op}{TAIL}{declarations}")
}

/// Sanity anchor: the base kernel (op = a plain add) DOES translate, so each negative below is
/// attributable to its injected construct rather than a broken template.
#[test]
fn base_kernel_translates() {
    let base = kernel_with("  %r = add i32 %a0, %a0\n");
    assert!(
        translate_sanitized_native(&base, Stage::Kernel, &tmp()).is_ok(),
        "the negative-suite base template must itself translate"
    );
}

#[test]
fn raytracing_intersect_intrinsic_fallbacks() {
    let ll = kernel_with("  %r = call i32 @air.intersect.f32.i32(i32 %a0, i32 %a0, i32 %a0)\n");
    assert_fallback(&ll, "@air.intersect.");
}

#[test]
fn agx3_emask_intrinsic_fallbacks() {
    let ll = kernel_with("  %r = call i32 @llvm.agx3.emask.i32(i32 %a0)\n");
    assert_fallback(&ll, "@llvm.agx3.");
}

#[test]
fn texture_atomic_fallbacks() {
    let ll = kernel_with(
        "  %r = call i32 @air.atomic_fetch_add.explicit.texture.2d.i32(i32 %a0, i32 %a0)\n",
    );
    assert_fallback(&ll, "@air.atomic_fetch_add.explicit.texture");
}

/// A call through a Metal visible function table: the callee is an SSA value, not a symbol.
///
/// The largest unsupported class in the corpus sample, 136 sources of the 204. Logical SPIR-V has no
/// function pointers, so this cannot become a lowering; it can only become a wrong one. The
/// diagnostic names the pointer so the author can find the call.
#[test]
fn visible_function_table_call_fallbacks() {
    let ll = kernel_with_declarations(
        "  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %out, i32 0)\n\
         \x20 %r = call i32 %fp(ptr addrspace(2) %in)\n",
        "declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)\n",
    );
    assert_fallback(
        &ll,
        "unsupported indirect call through function pointer %fp",
    );
}

/// The same visible-function-table call, reached only when a `[[function_constant]]` predicate is
/// true — which pins the *edge* of the rejection above rather than its interior.
///
/// Nothing supplies function-constant values at translate time (`fc_air_specialize` bakes any
/// caller-supplied ones into the AIR before parsing, and the emitted modules carry no
/// `OpSpecConstant` for them), so `air.is_function_constant_defined` folds to `false` and the
/// module's static initializer stores `0` into the gating global. The gated region is then
/// statically dead, gets pruned, and the indirect call never reaches the emitter.
///
/// That fold is load-bearing for real shaders: corpus sources that call through a visible function
/// table inside an off-by-default region translate today *because of it*. It is also fragile —
/// it spans `fold_static_initializer_constants` and `prune_unreachable_function_bodies`
/// (`src/native/ir/static_init.rs`) and depends on the initializer being recognised as one. So the
/// two directions are pinned as a pair over one template that differs in exactly one token, the
/// order of the branch arms:
///
/// - dead side taken → the call is folded away and the kernel translates;
/// - live side taken → the same call FALLBACKs with the same diagnostic as above.
///
/// A regression that stops recognising the initializer turns the first into a FALLBACK; one that
/// prunes too eagerly turns the second into a silent success. Neither is caught by either half
/// alone.
const FC_GATED_VFT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@kEnabled = internal unnamed_addr addrspace(2) global i8 0, align 1

declare i1 @air.is_function_constant_defined(ptr addrspace(2))
declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)

define internal void @_GLOBAL__sub_I_fc() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, align 1
  %2 = call i1 @air.is_function_constant_defined(ptr addrspace(2) @enabled.MTL_FC_INIT_0_b)
  %3 = icmp ne i8 %1, 0
  %4 = select i1 %2, i1 %3, i1 false
  %5 = zext i1 %4 to i8
  store i8 %5, ptr addrspace(2) @kEnabled, align 1
  ret void
}

define internal fastcc float @fetch(ptr addrspace(1) %table, ptr addrspace(1) %data) {
  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)
  %r = call float %fp(ptr addrspace(1) %data)
  ret float %r
}

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %table) {
entry:
  %e = load i8, ptr addrspace(2) @kEnabled, align 1
  %c = icmp eq i8 %e, 0
  br i1 %c, label %ARMS

use:
  %v = call fastcc float @fetch(ptr addrspace(1) %table, ptr addrspace(1) %out)
  br label %done

done:
  %r = phi float [ 0.000000e+00, %entry ], [ %v, %use ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table", !"air.arg_name", !"table"}
!6 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, !"bool", !"enabled", i32 0, i1 false}
"#;

/// `FC_GATED_VFT` with the branch arms in `arms` order — the sole difference between the two cases.
fn fc_gated_vft(arms: &str) -> String {
    assert!(
        FC_GATED_VFT.contains("label %ARMS"),
        "the branch placeholder must survive edits to the template"
    );
    FC_GATED_VFT.replace("label %ARMS", arms)
}

#[test]
fn function_constant_gated_visible_function_table_call_is_folded_away() {
    let ll = fc_gated_vft("label %done, label %use");
    let spv = translate_sanitized_native(&ll, Stage::Kernel, &tmp()).expect(
        "an off-by-default function constant makes the visible-function-table region dead; \
         folding it is what lets such shaders translate at all",
    );
    assert!(
        !spv.is_empty(),
        "the folded kernel must still emit a module"
    );
}

/// The initializer is recognised by its `air.static_init` section, not by its Itanium name.
///
/// `_GLOBAL__sub_I…` is how clang mangles a translation-unit initializer; `section
/// "air.static_init"` is how AIR declares what the function *is*. Over the corpus the two always
/// travel together, so only an authored pair can tell which one the readers act on — and the answer
/// decides real translations, since the fold above is what makes the gated call disappear. Renaming
/// the function must change nothing; removing the section must stop the fold, leaving the same
/// module with the same dead-side branch to reject on the call it can no longer prune.
#[test]
fn a_static_initializer_is_recognised_by_its_air_section_not_its_name() {
    let dead_side = fc_gated_vft("label %done, label %use");

    let renamed = dead_side.replace("_GLOBAL__sub_I_fc", "air_static_ctor");
    assert_ne!(
        renamed, dead_side,
        "the initializer name must be substituted"
    );
    translate_sanitized_native(&renamed, Stage::Kernel, &tmp())
        .expect("an initializer keeps its meaning when only its name changes");

    let unsectioned = dead_side.replace(" section \"air.static_init\"", "");
    assert_ne!(unsectioned, dead_side, "the section must be removed");
    assert_fallback(
        &unsectioned,
        "unsupported indirect call through function pointer %fp",
    );
}

#[test]
fn live_visible_function_table_call_still_fallbacks_under_a_function_constant() {
    let ll = fc_gated_vft("label %use, label %done");
    assert_fallback(
        &ll,
        "unsupported indirect call through function pointer %fp",
    );
}

/// An intersection whose intersection functions come from a function buffer.
///
/// The second largest class, about 55 sources. `air.intersect.*` with a null function table lowers
/// (`ray_intersection.rs`); the `intersection_function_buffer` variant is a dispatch through custom
/// intersection functions, so it is the same function-pointer wall as the case above wearing a
/// raytracing hat. The tag suffix is one of a combinatorial family -- `instancing`, `triangle_data`,
/// `world_space_data`, `user_data`, `primitive_motion`, `instance_motion`,
/// `multi_level_instancing` -- and the rejection is on the `intersection_function_buffer` stem, not
/// on any one combination, which is why one of them stands for all of them here.
#[test]
fn intersection_function_buffer_fallbacks() {
    let signature = "{ i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 }";
    let ll = kernel_with_declarations(
        &format!(
            "  %hit = call {signature} @air.intersect.intersection_function_buffer.triangle_data(\
             <3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, \
             ptr addrspace(1) %out, ptr addrspace(1) %out, i64 0, i64 1, ptr null, i64 0, i32 0, \
             i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)\n\
             \x20 %r = extractvalue {signature} %hit, 0\n"
        ),
        "declare { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } \
         @air.intersect.intersection_function_buffer.triangle_data(<3 x float>, <3 x float>, float, \
         float, ptr addrspace(1), ptr addrspace(1), i64, i64, ptr, i64, i32, i32, i32, i32, i32, \
         i32, i32, i32, i32, i32, i1, i1)\n",
    );
    assert_fallback(&ll, "air.intersect.intersection_function_buffer");
}

/// A return member whose AIR output role has no lowering.
///
/// AIR states an output because the shader writes it, so a member the emitter has no case for is a
/// computed value that reaches nothing. That used to be a `continue` in the output-member walk: the
/// module still validated, still exposed the same descriptors, and still reflected identically —
/// only the write was missing. `[[sample_mask]]` sat in exactly that hole until it was noticed by
/// eye, and the next role to be added to AIR would sit there too.
///
/// The rejection is on the *role*, not on any particular name, so an authored marker AIR does not
/// define stands in for whatever that next role turns out to be. Its positive control is the same
/// module with the member declared a role that does have a lowering.
#[test]
fn an_output_member_with_no_lowering_fallbacks() {
    const FRAGMENT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, i32 }> @frag(<4 x float> %pos, i32 %extra) {
entry:
  %r0 = insertvalue <{ <4 x float>, i32 }> undef, <4 x float> %pos, 0
  %r1 = insertvalue <{ <4 x float>, i32 }> %r0, i32 %extra, 1
  ret <{ <4 x float>, i32 }> %r1
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!"air.ROLE", !"air.arg_type_name", !"uint", !"air.arg_name", !"extra"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!6 = !{i32 1, !"air.fragment_input", !"generated(e)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"extra"}
"#;

    let unknown = FRAGMENT.replace("air.ROLE", "air.coverage_of_a_role_that_does_not_exist");
    match translate_sanitized_native(&unknown, Stage::Fragment, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK but translate succeeded ({} bytes); an output member \
             nothing writes is a silently wrong module",
            spv.len()
        ),
        Err(e) => {
            assert!(
                e.contains("air.coverage_of_a_role_that_does_not_exist"),
                "the diagnostic should name the role it cannot lower; got: {e}"
            );
            assert!(
                e.contains("return member 1"),
                "and the member it sits on; got: {e}"
            );
        }
    }

    let known = FRAGMENT.replace(r#"!"air.ROLE""#, r#"!"air.sample_mask""#);
    assert!(
        translate_sanitized_native(&known, Stage::Fragment, &tmp()).is_ok(),
        "the same module with a role that does have a lowering must translate"
    );
}

/// The same guard on the vertex side, where an unmodelled member is worse than dropped: the output
/// walk hands it the next free user Location, so it displaces a real varying instead of vanishing.
#[test]
fn an_unmodelled_vertex_output_member_fallbacks() {
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
!3 = !{!"air.ROLE", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!6 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

    let unknown = VERTEX.replace("air.ROLE", "air.output_role_that_does_not_exist");
    match translate_sanitized_native(&unknown, Stage::Vertex, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK but translate succeeded ({} bytes)",
            spv.len()
        ),
        Err(e) => assert!(
            e.contains("air.output_role_that_does_not_exist"),
            "the diagnostic should name the role; got: {e}"
        ),
    }

    let known = VERTEX.replace(
        r#"!"air.ROLE""#,
        r#"!"air.vertex_output", !"generated(uv)""#,
    );
    assert!(
        translate_sanitized_native(&known, Stage::Vertex, &tmp()).is_ok(),
        "the same module with a modelled role must translate"
    );
}

/// An entry parameter whose AIR role has no lowering.
///
/// The input side of the same hole. An unrecognised parameter is bound to a zero value so the body
/// stays well formed, which is right for a function-constant-disabled resource — Metal defines that
/// one as absent — and wrong for a system value nothing models: the shader reads zero where the
/// hardware would have supplied a value, in a module that validates, binds and reflects exactly as
/// if nothing were missing.
///
/// The three cases below are the boundary. An unmodelled role rejects; a modelled one translates;
/// and a bare `air.function_constant` parameter — which has no role marker behind the wrapper at
/// all, so reading past it lands on the node's own `air.arg_type_name` — must not be mistaken for a
/// role and rejected.
#[test]
fn an_entry_parameter_with_no_lowering_fallbacks() {
    const KERNEL: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define void @k(ptr addrspace(1) %out, i32 %sys) {
entry:
  %p = getelementptr i32, ptr addrspace(1) %out, i64 0
  store i32 %sys, ptr addrspace(1) %p
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.ROLE", !"air.arg_type_name", !"uint", !"air.arg_name", !"sys"}
"#;

    let unknown = KERNEL.replace("air.ROLE", "air.system_value_that_does_not_exist");
    match translate_sanitized_native(&unknown, Stage::Kernel, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK but translate succeeded ({} bytes); a system value read \
             as zero is a silently wrong module",
            spv.len()
        ),
        Err(e) => {
            assert!(
                e.contains("air.system_value_that_does_not_exist"),
                "the diagnostic should name the role; got: {e}"
            );
            assert!(
                e.contains("entry parameter 1"),
                "and the parameter it sits on; got: {e}"
            );
        }
    }

    let known = KERNEL.replace(r#"!"air.ROLE""#, r#"!"air.thread_index_in_threadgroup""#);
    assert!(
        translate_sanitized_native(&known, Stage::Kernel, &tmp()).is_ok(),
        "the same module with a modelled role must translate"
    );

    // A parameter that IS a function constant carries the wrapper and nothing else. Metal supplies
    // no value here, so binding it to its default is the defined behaviour, not a missing lowering.
    let constant = KERNEL.replace(r#"!"air.ROLE""#, r#"!"air.function_constant""#);
    assert!(
        translate_sanitized_native(&constant, Stage::Kernel, &tmp()).is_ok(),
        "a bare function-constant parameter has no role to reject"
    );

    // ...and the wrapper in front of a real role is not the same thing. Only a gate this module's
    // own initializers drive to ZERO leaves the parameter out of the variant; behind a predicate the
    // evaluator cannot fold, the unmodelled role is still one the emitter reads as a zero.
    let gated = |constructor: &str| {
        KERNEL
            .replace(
                r#"!"air.ROLE""#,
                r#"!"air.function_constant", !5, !"air.system_value_that_does_not_exist""#,
            )
            .replace(
                "!air.kernel",
                &format!(
                    "@gate = internal addrspace(2) global i8 0, align 1\n\
                     @mirror = internal addrspace(2) global i8 undef, align 1\n\
                     define internal void @ctor() #0 section \"air.static_init\" {{\n\
                     {constructor}\n  ret void\n}}\n\
                     !5 = !{{ptr addrspace(2) @gate, !\"bool\", !\"gate\"}}\n!air.kernel"
                ),
            )
    };
    let disabled = gated("  store i8 0, ptr addrspace(2) @gate, align 1");
    assert!(
        translate_sanitized_native(&disabled, Stage::Kernel, &tmp()).is_ok(),
        "a gate this module drives to zero leaves the parameter out; there is nothing to reject"
    );
    // Read before written, out of a mirror global with no initializer to fold: the corpus's own
    // spelling of a predicate this evaluator cannot decide.
    let unresolved = gated(
        "  %v = load i8, ptr addrspace(2) @mirror, align 1\n  \
         store i8 %v, ptr addrspace(2) @gate, align 1",
    );
    match translate_sanitized_native(&unresolved, Stage::Kernel, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK but translate succeeded ({} bytes); a function-constant \
             wrapper is not evidence the parameter is absent",
            spv.len()
        ),
        Err(e) => assert!(
            e.contains("air.system_value_that_does_not_exist"),
            "the diagnostic should name the wrapped role; got: {e}"
        ),
    }
}

/// A builtin entry parameter whose type is not the builtin's type.
///
/// The third face of the same hole. `position`, `point_coord` and `front_facing` each declare a
/// SPIR-V builtin of a fixed shape; a parameter of any other shape used to be bound to a zero, so a
/// `[[position]]` the emitter could not wire put every fragment at the origin and a
/// `[[front_facing]]` it could not wire called every triangle back-facing — in a module that
/// validated and reflected as though the builtin were connected.
///
/// Unlike an unmodelled role, this branch is unreachable on real AIR: Metal fixes each attribute's
/// type, and over 2880 corpus sources no module takes it. So the rejection costs nothing and exists
/// to keep the silent zero from coming back.
#[test]
fn a_builtin_parameter_of_the_wrong_type_fallbacks() {
    const FRAGMENT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @frag(TYPE %sys) {
entry:
  %v0 = insertelement <4 x float> undef, float 0.000000e+00, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.ROLE", !"air.center", !"air.arg_type_name", !"NAME"}
"#;

    let build = |role: &str, name: &str, ty: &str| {
        FRAGMENT
            .replace("air.ROLE", &format!("air.{role}"))
            .replace("NAME", name)
            .replace("TYPE", ty)
    };

    // Each attribute's own type translates.
    for (role, name, ty) in [
        ("position", "float4", "<4 x float>"),
        ("point_coord", "float2", "<2 x float>"),
        ("front_facing", "bool", "i1"),
    ] {
        assert!(
            translate_sanitized_native(&build(role, name, ty), Stage::Fragment, &tmp()).is_ok(),
            "[[{role}]] of type {name} must translate"
        );
    }

    // Any other type has no wiring, and a zero in its place is a lie about the geometry.
    for (role, name, ty, needle) in [
        ("position", "float2", "<2 x float>", "float4"),
        ("point_coord", "float4", "<4 x float>", "float2"),
        ("front_facing", "uint", "i32", "bool"),
    ] {
        match translate_sanitized_native(&build(role, name, ty), Stage::Fragment, &tmp()) {
            Ok(spv) => panic!(
                "expected a FALLBACK for [[{role}]] typed {name}, got {} bytes",
                spv.len()
            ),
            Err(e) => assert!(
                e.contains(role) && e.contains(needle),
                "the diagnostic should name the attribute and the type it needs; got: {e}"
            ),
        }
    }
}

#[test]
fn no_function_definitions_fallbacks() {
    assert_fallback(
        "target triple = \"air64_v28-apple-macosx26.5.0\"\n",
        "no function definitions found",
    );
}

#[test]
fn truncated_module_fallbacks() {
    // a `define` block that never closes / never returns
    let ll = "target triple = \"air64_v28-apple-macosx26.5.0\"\n\
              define void @k(ptr %x) {\n\
              entry:\n\
              \x20 %a = load i32, ptr %x\n";
    assert_fallback(ll, "unterminated function");
}

/// A `<3 x float>` store through a byte view of a raw word buffer, reached across a call.
///
/// This one is a GAP, not a limit — unlike the function-pointer classes above, the lowering is
/// expressible; nothing has written it. It is the largest remaining non-function-pointer rejection
/// class in the corpus sample (8 of the 12), and it is pinned here because it is the class most
/// likely to be closed WRONGLY. A `device uchar*` view of a `{ RuntimeArray<uint> }` block writing
/// four, eight or twelve bytes at a runtime offset has two tempting lowerings that are not the AIR's
/// semantics: a non-atomic read-modify-write of the surrounding words, which races another thread
/// writing the neighbouring bytes (`emit_scalar_narrowing_store` restricts exactly that to
/// thread-local slots, and `emit_raw_byte_store_from_u32` uses atomics on shared storage); and a
/// plain word store, which is only byte-exact if the offset is four-byte aligned — the AIR asserts
/// that with `align`, but the alignment does not reach the SPIR-V pass that would need it. Until one
/// of those is resolved honestly, this must stay a clean `Err`.
///
/// Reduced from a corpus source with `llvm-reduce` (708 lines to 44) and then renamed, so it carries
/// no third-party identifiers. Every remaining part is load-bearing: dropping the `i32` load or the
/// byte `getelementptr` in `@k` (which together fix the block's element width at 32 bits and open
/// the byte view), inlining `@store_vec` into `@k`, flattening its pass-through blocks, or shrinking
/// the argument list each make the module translate.
#[test]
fn byte_view_vector_store_into_a_word_block_fallbacks() {
    let ll = r#"target triple = "air64_v28-apple-macosx26.5.0"

%struct.view = type { ptr addrspace(2), ptr addrspace(1) }

define void @k(ptr addrspace(1) %0) {
  %2 = alloca %struct.view, align 8
  %3 = getelementptr %struct.view, ptr %2, i64 0, i32 1
  store ptr addrspace(1) %0, ptr %3, align 8
  %4 = getelementptr i8, ptr addrspace(1) %0, i64 0
  %5 = load i32, ptr addrspace(1) %0, align 16
  call fastcc void @store_vec(ptr %2)
  ret void
}

define fastcc void @store_vec(ptr %0) {
  br label %2

2:                                                ; preds = %1
  br label %4

4:                                                ; preds = %2
  %5 = getelementptr %struct.view, ptr %0, i64 0, i32 1
  %6 = load ptr addrspace(1), ptr %5, align 8
  %7 = zext i32 0 to i64
  %8 = getelementptr i8, ptr addrspace(1) %6, i64 %7
  store <3 x float> zeroinitializer, ptr addrspace(1) %8, align 16
  br label %9

9:                                                ; preds = %4
  ret void
}

!air.kernel = !{!0}

!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7, !8, !9}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 280, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 280, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"hdr", !"air.arg_name", !"header"}
!5 = !{!"air.struct_type_info", !6, i32 0, i32 8, i32 35, !"hdr_entry", !"entries"}
!6 = !{i32 0, i32 4, i32 0, !"int", !"offset", i32 4, i32 2, i32 0, !"short", !"type", i32 6, i32 2, i32 0, !"short", !"stride"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}
!8 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 280, !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 280, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"hdr", !"air.arg_name", !"header"}
!9 = !{i32 4, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}"#;
    assert_fallback(ll, "no dynamic-struct-index rewrite repaired");
}

/// A sampler whose exact state the shader picks at run time.
///
/// The emitter cannot represent a select between two `__air_sampler_state` globals as a SPIR-V
/// pointer value, so it leaves a private placeholder and both states are gone by the time the
/// sample lowers. The only sampler left to hand `OpSampledImage` is the translator's own
/// nearest/clamp default -- a different filter and address mode than either branch asked for, in a
/// module that validates, binds and reflects as though the shader had asked for that default. Nine
/// of 14579 local corpus sources reach it.
///
/// Pinned from both sides over one template: sampling through either state UNCONDITIONALLY is the
/// ordinary constexpr-sampler path and must keep translating, so a future recovery of the selected
/// form has to change this test rather than pass it by accident.
#[test]
fn a_runtime_selected_sampler_state_fallbacks() {
    const FRAGMENT: &str = r#"target triple = "spirv-unknown-vulkan1.2"

@__air_sampler_state = internal addrspace(2) constant i64 -9188470239253757879, align 8
@__air_sampler_state.1 = internal addrspace(2) constant i64 -9188470239253755831, align 8

define <4 x float> @frag(<4 x float> %position, <2 x float> %coord, ptr addrspace(1) %tex) {
entry:
  %edge = extractelement <2 x float> %coord, i64 0
  %wide = fcmp oge float %edge, 1.000000e+00
SAMPLER_CHOICE
  %sample = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) SAMPLER_OPERAND, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %color
}
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.fragment = !{!0}
!air.sampler_states = !{!7, !8}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.fragment_input", !"generated(coord)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"coord"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!7 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
"#;

    let selected = FRAGMENT
        .replace(
            "SAMPLER_CHOICE",
            "  %s = select i1 %wide, ptr addrspace(2) @__air_sampler_state, \
             ptr addrspace(2) @__air_sampler_state.1",
        )
        .replace("SAMPLER_OPERAND", "%s");
    match translate_sanitized_native(&selected, Stage::Fragment, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK but translate succeeded ({} bytes); a sample through a \
             default sampler neither branch asked for is a silently wrong module",
            spv.len()
        ),
        Err(e) => assert!(
            e.contains("sampler operand is a pointer"),
            "FALLBACK diagnostic should name the unrecovered sampler operand; got: {e}"
        ),
    }

    for state in ["@__air_sampler_state", "@__air_sampler_state.1"] {
        let fixed = FRAGMENT
            .replace("SAMPLER_CHOICE", "")
            .replace("SAMPLER_OPERAND", state);
        assert!(
            translate_sanitized_native(&fixed, Stage::Fragment, &tmp()).is_ok(),
            "sampling unconditionally through {state} must keep translating"
        );
    }
}

/// A sampler argument AIR states occupies more than one descriptor.
///
/// `air.location_index` carries two operands, the Metal slot and the descriptor count, and Metal
/// spells `array<sampler, 8>` with a count of 8. There is no sampler descriptor-array lowering:
/// the element loads resolve to nothing, so the sample falls back to the synthesized default
/// sampler and the state the shader selected is dropped -- in a module that validates, binds and
/// reflects as though one sampler at that slot were the whole story. Two of 14579 local corpus
/// sources declare one.
///
/// Pinned from both sides over one template, so the boundary is what is fixed: the ordinary
/// single-sampler spelling of the same kernel -- one declared descriptor, sampled straight off the
/// entry parameter -- must keep translating.
#[test]
fn a_sampler_descriptor_array_fallbacks() {
    const KERNEL: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(1) %tex, SAMPLER_PARAM, ptr addrspace(1) %out) {
entry:
SAMPLER_ELEMENT
  %sample = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %s, <2 x float> zeroinitializer, i1 false, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 COUNT, !"air.arg_type_name", !"SAMPLER_TYPE", !"air.arg_name", !"samps"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
"#;

    let array = KERNEL
        .replace("i32 COUNT", "i32 8")
        .replace("SAMPLER_TYPE", "array<sampler, 8>")
        .replace(
            "SAMPLER_PARAM",
            "ptr readonly byval([8 x ptr addrspace(2)]) captures(none) %samps",
        )
        .replace(
            "SAMPLER_ELEMENT",
            "  %slot = getelementptr inbounds [8 x ptr addrspace(2)], ptr %samps, i32 0, i32 5\n               %s = load ptr addrspace(2), ptr %slot, align 8",
        );
    match translate_sanitized_native(&array, Stage::Kernel, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK but translate succeeded ({} bytes); a sample through a \
             default sampler the shader never asked for is a silently wrong module",
            spv.len()
        ),
        Err(e) => {
            assert!(
                e.contains("array of 8 samplers"),
                "the diagnostic should name the declared count; got: {e}"
            );
            assert!(
                e.contains("entry parameter 1"),
                "and the parameter it sits on; got: {e}"
            );
        }
    }

    let single = KERNEL
        .replace("i32 COUNT", "i32 1")
        .replace("SAMPLER_TYPE", "sampler")
        .replace("SAMPLER_PARAM", "ptr addrspace(2) %s")
        .replace("SAMPLER_ELEMENT", "");
    assert!(
        translate_sanitized_native(&single, Stage::Kernel, &tmp()).is_ok(),
        "the same kernel with an ordinary single sampler must translate"
    );
}

/// A texture handle array whose two statements of its own length disagree.
///
/// AIR states the length twice: as the `air.location_index` count operand and inside the
/// `array<texture..., N>` type name. The interface pass sizes its `OpTypeArray` from the name and
/// reflection reports the same number, so a name the ABI contradicts would bind and publish a
/// descriptor array of the wrong size. Over 14579 local corpus sources all 76 declarations agree,
/// which is exactly what makes the ABI usable as a check on the name parse: it costs nothing today
/// and fails the moment the parse drifts.
#[test]
fn a_texture_array_whose_declared_length_contradicts_its_type_name_fallbacks() {
    const KERNEL: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr readonly captures(none) %imgs, ptr addrspace(1) %out) {
entry:
  %slot = getelementptr inbounds ptr addrspace(1), ptr %imgs, i32 3
  %tex = load ptr addrspace(1), ptr %slot, align 8
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 COUNT, !"air.sample", !"air.arg_type_name", !"TEXTURE_TYPE", !"air.arg_name", !"imgs"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let fixed = |count: &str, name: &str| {
        KERNEL
            .replace("i32 COUNT", count)
            .replace("TEXTURE_TYPE", name)
    };

    for (count, name, why) in [
        (
            "i32 3",
            "array<texture2d<float, sample>, 4>",
            "a length the ABI contradicts",
        ),
        (
            "i32 4",
            "texture2d<float, sample>",
            "a count on a name that is not an array at all",
        ),
    ] {
        match translate_sanitized_native(&fixed(count, name), Stage::Kernel, &tmp()) {
            Ok(spv) => panic!(
                "expected a clean FALLBACK for {why} but translate succeeded ({} bytes)",
                spv.len()
            ),
            Err(e) => assert!(
                e.contains("descriptors per `air.location_index`"),
                "the diagnostic should name the ABI count for {why}; got: {e}"
            ),
        }
    }

    assert!(
        translate_sanitized_native(
            &fixed("i32 4", "array<texture2d<float, sample>, 4>"),
            Stage::Kernel,
            &tmp()
        )
        .is_ok(),
        "the same kernel whose two statements of the length agree must translate"
    );
}

/// A texture write reachable only when a `[[function_constant]]` VALUE matches, which pins the
/// difference between a predicate AIR answers and a value AIR does not.
///
/// `air.is_function_constant_defined` folding to `false` is AIR's own answer for a constant nobody
/// supplied, and the pair above depends on it. Reading the constant's *value* is a different
/// question: AIR leaves the `air.fc_initializer` global `undef`, and `meta::globals` reads that as
/// zero so the same dead-region fold can run. Zero is not a default the shader declared, so when it
/// is what removed the entry's every texture write, the module we would emit cannot write the
/// texture its AIR writes -- 55 of the 14579 local corpus sources, 28 of which regain an image
/// write the moment any value is put in place of the `undef`.
///
/// Pinned as a pair over one template differing in a single token, the sentinel the folded zero is
/// compared against:
///
/// - sentinel 0 -> the folded value selects the write, and the kernel translates;
/// - sentinel 1 -> it selects nothing, and the kernel must FALLBACK rather than report success.
///
/// Deleting the check turns the second into a silent success that writes nothing; widening it to
/// fire whenever a module has function constants turns the first into a FALLBACK.
const FC_VALUE_GATED_WRITE: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

@size.MTL_FC_INIT_0_t = internal addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@kSamplingSize = internal unnamed_addr addrspace(2) global i16 0, align 2
@tg = internal addrspace(3) global float undef, align 4
@tgi = internal addrspace(3) global i32 undef, align 4

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

define internal void @_GLOBAL__sub_I_fc() section "air.static_init" {
  %1 = load i16, ptr addrspace(2) @size.MTL_FC_INIT_0_t, align 2
  store i16 %1, ptr addrspace(2) @kSamplingSize, align 2
  ret void
}

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out, <2 x i32> %gid) {
entry:
  %s = load i16, ptr addrspace(2) @kSamplingSize, align 2
  %c = icmp eq i16 %s, SENTINEL
  br i1 %c, label %write, label %done

write:
  GATED
  br label %done

done:
  ALWAYS
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{ptr addrspace(2) @size.MTL_FC_INIT_0_t, !"ushort", !"kSamplingSize", i32 0, i1 true}
"#;

const TEXTURE_WRITE: &str = "call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %gid, <4 x float> zeroinitializer, i32 0, i32 2)";
const DEVICE_STORE: &str = "store float 1.000000e+00, ptr addrspace(1) %out, align 4";
/// Threadgroup scratch: a real store, to memory that dies with the dispatch.
const THREADGROUP_STORE: &str = "store float 1.000000e+00, ptr addrspace(3) @tg, align 4";
/// The same, as a threadgroup counter rather than a store -- an atomic read-modify-write is the
/// other arm of [`module_has_an_observable_effect`], and the one the corpus does not exercise.
const THREADGROUP_ATOMIC: &str = "%bump = call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) \
                                  @tgi, i32 1, i32 0, i32 1, i1 true)";

/// `FC_VALUE_GATED_WRITE` with the sentinel the folded constant is compared against, the write the
/// constant gates, and any write that runs unconditionally.
fn fc_value_gated_write(sentinel: &str, gated: &str, ungated: &str) -> String {
    for placeholder in ["SENTINEL", "GATED", "ALWAYS"] {
        assert!(
            FC_VALUE_GATED_WRITE.contains(placeholder),
            "the {placeholder} placeholder must survive edits to the template"
        );
    }
    FC_VALUE_GATED_WRITE
        .replace("SENTINEL", sentinel)
        .replace("GATED", gated)
        .replace("ALWAYS", ungated)
}

fn assert_translates(ll: &str, why: &str) {
    let spv = translate_sanitized_native(ll, Stage::Kernel, &tmp())
        .unwrap_or_else(|error| panic!("{why}; got FALLBACK: {error}"));
    assert!(!spv.is_empty(), "{why}; got an empty module");
}

const ERASED: &str =
    "no write survived folding 1 function constant(s) the caller supplied no value for";

#[test]
fn a_function_constant_value_that_selects_the_write_still_translates() {
    assert_translates(
        &fc_value_gated_write("0", TEXTURE_WRITE, ""),
        "the folded constant selects the write, so the module writes its texture",
    );
}

#[test]
fn a_function_constant_value_that_erases_every_write_fallbacks() {
    assert_fallback(&fc_value_gated_write("1", TEXTURE_WRITE, ""), ERASED);
}

/// The same erasure reached through a device buffer store rather than a texture write. 81 of the
/// 117 corpus sources the check names are this form, so pinning only the texture one would leave
/// most of it untested.
#[test]
fn a_function_constant_value_that_erases_every_device_store_fallbacks() {
    assert_fallback(&fc_value_gated_write("1", DEVICE_STORE, ""), ERASED);
}

/// A store into threadgroup memory is not the write the AIR promised. `Workgroup` scratch dies with
/// the dispatch, so a folded module that keeps only its staging stores can no more be observed by
/// the caller than one that keeps only its allocas -- and it is the shape that actually occurs:
/// 17 corpus sources emit no store outside `Function`, `Private` and `Workgroup` while their AIR
/// writes a device buffer or a texture (`b118683b` writes 157 textures in its AIR and emits 852
/// Private stores, 4 Function and 2 Workgroup, and no image write at all).
///
/// Counting `Workgroup` was what let all 17 report success.
#[test]
fn a_threadgroup_store_is_not_the_write_the_air_promised() {
    assert_fallback(
        &fc_value_gated_write("1", DEVICE_STORE, THREADGROUP_STORE),
        ERASED,
    );
}

/// The atomic arm of the same rule. `OpAtomicIAdd` on a `Workgroup` pointer is a threadgroup
/// counter: no more visible to a caller than the threadgroup store above, and the emitted module
/// really does contain one (`%52 = OpAtomicIAdd %uint %tgi ...`) with nothing else surviving.
///
/// No corpus source needs this -- measured over all 14579, none has a local-storage atomic or
/// `OpCopyMemory` as its only surviving effect, and making the rule uniform moves nothing. It is
/// here because the asymmetry is what caused the bug the test above pins: stores were classified by
/// storage class and atomics were counted wherever they appeared, so a folded module that kept a
/// threadgroup counter would answer "something survived" for the same wrong reason 17 sources did.
#[test]
fn a_threadgroup_atomic_is_not_the_write_the_air_promised() {
    assert_fallback(
        &fc_value_gated_write("1", DEVICE_STORE, THREADGROUP_ATOMIC),
        ERASED,
    );
}

/// The control for the test above, and the reason it is not simply a wider refusal: the same module
/// with the constant selecting its device store keeps translating. The threadgroup store is not
/// what is being refused -- having nothing else is.
#[test]
fn a_threadgroup_store_alongside_a_live_device_store_keeps_the_module() {
    assert_translates(
        &fc_value_gated_write("0", DEVICE_STORE, THREADGROUP_STORE),
        "the folded constant selects the device store, so threadgroup scratch beside it is \
         irrelevant",
    );
}

/// And the direction that keeps the check from being a blanket refusal of function-constant
/// shaders: one surviving observable write is enough, whatever kind it is and whether or not the
/// gated one died. Narrowing the emitted-side predicate back to image writes alone fails this,
/// which is how four corpus modules were over-refused before the predicate was widened.
#[test]
fn one_surviving_write_of_another_kind_keeps_the_module() {
    assert_translates(
        &fc_value_gated_write("1", TEXTURE_WRITE, DEVICE_STORE),
        "the device store runs unconditionally, so the module still writes",
    );
    assert_translates(
        &fc_value_gated_write("1", DEVICE_STORE, TEXTURE_WRITE),
        "the texture write runs unconditionally, so the module still writes",
    );
}

/// A Metal imageblock is threadgroup tile memory the render pass resolves into its attachments when
/// the tile finishes. Vulkan has no resolve step and this translator has no attachment to resolve
/// into, so imageblock cells are staged in per-invocation `Private` (or per-threadgroup
/// `Workgroup`) memory. That is the right model for a kernel that stages a cell and then writes a
/// texture or a device buffer -- but a Metal tile CLEAR kernel writes nothing else, and it used to
/// translate into a module that validates, reflects no resource at all, and clears nothing. 16 of
/// the 14579 local corpus sources are that shape (`vst::splat::hw_rasterizer::clearColor*Kernel`,
/// `xdr::*_block`, `DaVinci::resetLM`, `CC_WarpedDataClear`).
///
/// `ALWAYS` is any write that runs beside the imageblock store.
const IMAGEBLOCK_ONLY_WRITE: &str = r#"target triple = "spirv-unknown-vulkan1.2"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %blk, <2 x i16> %tid, ptr addrspace(1) %out) {
entry:
  %cell = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x half> zeroinitializer, ptr addrspace(4) %cell, align 8
  ALWAYS
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ColorBlock, layout_explicit>", !"air.arg_name", !"colorBlock"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"color"}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;

fn imageblock_only_write(ungated: &str) -> String {
    assert!(
        IMAGEBLOCK_ONLY_WRITE.contains("ALWAYS"),
        "the ALWAYS placeholder must survive edits to the template"
    );
    IMAGEBLOCK_ONLY_WRITE.replace("ALWAYS", ungated)
}

#[test]
fn an_imageblock_is_not_a_write_a_caller_can_observe() {
    assert_fallback(
        &imageblock_only_write(""),
        "the entry's only write is into an imageblock",
    );
}

/// The control, and the reason this is not a blanket refusal of tile kernels: the same imageblock
/// store beside one device store keeps translating. 66 corpus sources stage a cell and then write
/// something a caller owns, and none of them may be refused for staging one.
#[test]
fn an_imageblock_store_alongside_a_device_store_keeps_the_module() {
    assert_translates(
        &imageblock_only_write(DEVICE_STORE),
        "the device store runs unconditionally, so staging an imageblock cell beside it is \
         irrelevant",
    );
}

/// A write the AIR spells only as an INTRINSIC. `air_declares_an_observable_write` used to know
/// two spellings -- `air.write_texture*` and a store through an `addrspace(1)` pointer -- and a
/// kernel whose every write is an `air.atomic.global.*` has neither, so the AIR-side half of the
/// pair answered "this module never claimed to write" and the emitted-side half was never asked.
/// Three corpus kernels (`binFragmentsKernel`, `binFragmentsSpatialKernel`,
/// `gatherNDGradient_base`) shipped that way, each with every function constant it folds declared
/// REQUIRED, so the zero-variant they translated to is one Metal itself refuses to build.
///
/// `GATED` is the write the folded constant gates and `ALWAYS` one that runs regardless.
const FC_GATED_DEVICE_ATOMIC: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

@size.MTL_FC_INIT_0_t = internal addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@kSamplingSize = internal unnamed_addr addrspace(2) global i16 0, align 2

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1)
declare i32 @air.atomic.global.load.u.i32(ptr addrspace(1) captures(none), i32, i32)

define internal void @_GLOBAL__sub_I_fc() section "air.static_init" {
  %1 = load i16, ptr addrspace(2) @size.MTL_FC_INIT_0_t, align 2
  store i16 %1, ptr addrspace(2) @kSamplingSize, align 2
  ret void
}

define void @k(ptr addrspace(1) %out, <2 x i32> %gid) {
entry:
  %s = load i16, ptr addrspace(2) @kSamplingSize, align 2
  %c = icmp eq i16 %s, SENTINEL
  br i1 %c, label %write, label %done

write:
  GATED
  br label %done

done:
  ALWAYS
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!4}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!4 = !{ptr addrspace(2) @size.MTL_FC_INIT_0_t, !"ushort", !"kSamplingSize", i32 0, i1 true}
"#;

const DEVICE_ATOMIC_ADD: &str =
    "%bump = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %out, i32 1, i32 0, i32 2, i1 true)";
/// The one device atomic that leaves no value behind.
const DEVICE_ATOMIC_LOAD: &str =
    "%seen = call i32 @air.atomic.global.load.u.i32(ptr addrspace(1) %out, i32 0, i32 2)";

fn fc_gated_device_atomic(sentinel: &str, gated: &str, ungated: &str) -> String {
    for placeholder in ["SENTINEL", "GATED", "ALWAYS"] {
        assert!(
            FC_GATED_DEVICE_ATOMIC.contains(placeholder),
            "the {placeholder} placeholder must survive edits to the template"
        );
    }
    FC_GATED_DEVICE_ATOMIC
        .replace("SENTINEL", sentinel)
        .replace("GATED", gated)
        .replace("ALWAYS", ungated)
}

#[test]
fn a_device_atomic_is_a_write_the_air_promised() {
    assert_fallback(&fc_gated_device_atomic("1", DEVICE_ATOMIC_ADD, ""), ERASED);
}

/// The control: the same atomic selected by the folded constant keeps translating, so this is not
/// a blanket refusal of kernels that use device atomics.
#[test]
fn a_surviving_device_atomic_keeps_the_module() {
    assert_translates(
        &fc_gated_device_atomic("0", DEVICE_ATOMIC_ADD, ""),
        "the folded constant selects the atomic, so the module still bumps its counter",
    );
}

/// And the other direction of the one exclusion the marker list makes. A kernel whose only device
/// atomic is a LOAD leaves no value behind; it is a no-op in Metal too, and refusing it would be
/// refusing a shader that does exactly what it says.
#[test]
fn a_kernel_whose_only_device_atomic_is_a_load_still_translates() {
    assert_translates(
        &fc_gated_device_atomic("1", DEVICE_ATOMIC_LOAD, ""),
        "an atomic load is not a write, so nothing was erased",
    );
}
