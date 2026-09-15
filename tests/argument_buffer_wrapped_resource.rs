//! A texture an argument buffer holds inside a nested struct binds like a flat one.
//!
//! Metal lets an argument-buffer member be a user struct that itself holds a texture, sampler or
//! buffer -- `struct texture2d_wrapper { texture2d<half> t; };` as one member of the buffer. AIR
//! spells that with an `air.struct_type_info` PREFIX naming the wrapper's own member list, and the
//! member's `air.indirect_argument` suffix is then an `i32` where a flat member carries the node ref
//! describing the resource. That `i32` is the wrapper's own Metal argument id, and each member
//! inside it carries an `air.location_index` RELATIVE to it, so the absolute `[[id(n)]]` is the SUM
//! down the nesting path.
//!
//! The embedded-argument walk used to read only the flat form. The wrapped texture never became an
//! embedded descriptor: the member decoded as opaque storage, the handle load produced a Private
//! placeholder, and the sample took the path that answers zero for a resource the pipeline does not
//! provide -- the right answer for a `[[function_constant]]`-gated texture whose constant is off,
//! and the wrong one here. The result was a fragment shader that samples black in a module that
//! passes `spirv-val` and reports a consistent reflection.
//!
//! The walk now descends, so a wrapped texture is surfaced at the summed argument id, in whatever
//! dimension its type name reads. What it still cannot express -- a wrapped buffer, sampler or
//! function table, and any member of a wrapper declared as an ARRAY of structs, whose elements
//! repeat at offsets this walk does not enumerate -- stays a named refusal rather than a black
//! sample.

use metal2vulkan::meta::parse_air_kernel_meta;
use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_wrapped_arg_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// The flat form: `struct Args { texture2d<float, write> output; short r; }`.
const FLAT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

%Args = type <{ %"struct.metal::texture2d", i16, [6 x i8] }>
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @k(ptr addrspace(2) %args, <2 x i32> %coord) local_unnamed_addr #0 {
entry:
  %field = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %field, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %coord, <4 x float> zeroinitializer, i32 0, i32 2) #3
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr #3

attributes #0 = { convergent nounwind }
attributes #3 = { convergent nounwind memory(argmem: write) }

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"output", !"air.indirect_argument", !5, i32 8, i32 2, i32 0, !"short", !"radius", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"output"}
!6 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"short", !"air.arg_name", !"radius"}
!7 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"coord"}
"#;

/// The same buffer with the texture one struct deeper, spelled the way the corpus spells it: an
/// `air.struct_type_info` prefix on the member, and an `i32` where the flat member has a node ref.
/// The wrapper's id is 7 and the texture's own `air.location_index` is 3, so `[[id(10)]]`.
const WRAPPED: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

%Args = type <{ %Wrapper, i16, [6 x i8] }>
%Wrapper = type { %"struct.metal::texture2d" }
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @k(ptr addrspace(2) %args, <2 x i32> %coord) local_unnamed_addr #0 {
entry:
  %field = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %field, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %coord, <4 x float> zeroinitializer, i32 0, i32 2) #3
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr #3

attributes #0 = { convergent nounwind }
attributes #3 = { convergent nounwind memory(argmem: write) }

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{!"air.struct_type_info", !8, i32 0, i32 8, i32 0, !"texture2d_wrapper", !"wrapper", !"air.indirect_argument", i32 7, i32 8, i32 2, i32 0, !"short", !"radius", !"air.indirect_argument", !6}
!8 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"output", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"output"}
!6 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"short", !"air.arg_name", !"radius"}
!7 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"coord"}
"#;

/// The wrapped buffer, sampled rather than written. This is the pair that mattered: the write
/// lowering already refused a non-storage image, while the sample lowering answered zero. The
/// sampler is a top-level argument, so nothing but the texture is under test.
const WRAPPED_SAMPLE: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

%Args = type <{ %Wrapper }>
%Wrapper = type { %"struct.metal::texture2d" }
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define <4 x float> @k(ptr addrspace(2) %args, ptr addrspace(2) %smp) local_unnamed_addr #0 {
entry:
  %tf = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %tf, align 8
  %s = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %c = extractvalue { <4 x float>, i8 } %s, 0
  ret <4 x float> %c
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

attributes #0 = { convergent nounwind }

!air.fragment = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{!"air.struct_type_info", !8, i32 0, i32 8, i32 0, !"texture2d_wrapper", !"wrapper", !"air.indirect_argument", i32 0}
!8 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
"#;

/// A wrapped `texture1d`: the dimension the embedded lowering used to drop on the floor.
const WRAPPED_1D: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

%Args = type <{ %Wrapper }>
%Wrapper = type { %"struct.metal::texture1d" }
%"struct.metal::texture1d" = type { ptr addrspace(1) }

define <4 x float> @k(ptr addrspace(2) %args, ptr addrspace(2) %smp) local_unnamed_addr #0 {
entry:
  %tf = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %tf, align 8
  %s = call { <4 x float>, i8 } @air.sample_texture_1d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, float 0.000000e+00, i1 false, float 0.000000e+00, i32 0)
  %c = extractvalue { <4 x float>, i8 } %s, 0
  ret <4 x float> %c
}

declare { <4 x float>, i8 } @air.sample_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), float, i1, float, i32)

attributes #0 = { convergent nounwind }

!air.fragment = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{!"air.struct_type_info", !8, i32 0, i32 8, i32 0, !"texture1d_wrapper", !"wrapper", !"air.indirect_argument", i32 0}
!8 = !{i32 0, i32 8, i32 0, !"texture1d<float, sample>", !"tex", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d<float, sample>", !"air.arg_name", !"tex"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
"#;

/// The same buffer whose wrapper is declared as an ARRAY of two structs. Its members repeat once
/// per element, at offsets and argument ids this walk does not enumerate.
const WRAPPED_ARRAY: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

%Args = type <{ [2 x %Wrapper] }>
%Wrapper = type { %"struct.metal::texture2d" }
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define <4 x float> @k(ptr addrspace(2) %args, ptr addrspace(2) %smp) local_unnamed_addr #0 {
entry:
  %tf = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i32 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %tf, align 8
  %s = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %c = extractvalue { <4 x float>, i8 } %s, 0
  ret <4 x float> %c
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

attributes #0 = { convergent nounwind }

!air.fragment = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{!"air.struct_type_info", !8, i32 0, i32 8, i32 2, !"texture2d_wrapper", !"wrappers", !"air.indirect_argument", i32 0}
!8 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
"#;

/// A FLAT member declared as a C array -- `texture2d<float, sample> slices[32]` -- indexed at
/// runtime. The length is stated only in the member tuple's `array_len` slot: the type name is the
/// element type, and the node's own `air.location_index` count operand stays `1`, unlike the
/// top-level `array<texture2d<...>, N>` spelling where name and count operand agree.
const FLAT_C_ARRAY: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

%Args = type { [32 x %"struct.metal::texture2d"] }
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define <4 x float> @k(ptr addrspace(2) %args, ptr addrspace(2) %smp, i32 %slice) local_unnamed_addr #0 {
entry:
  %w = zext i32 %slice to i64
  %tf = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i64 %w, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %tf, align 8
  %s = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %c = extractvalue { <4 x float>, i8 } %s, 0
  ret <4 x float> %c
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

attributes #0 = { convergent nounwind }

!air.fragment = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!3, !6, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 8, i32 32, !"texture2d<float, sample>", !"slices", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"slices"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!7 = !{i32 2, !"air.fragment_input", !"generated(5slicej)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"slice"}
"#;

/// The control: the flat spelling still binds its embedded texture at the same slot.
#[test]
fn a_flat_embedded_texture_still_binds() {
    let spv = translate_sanitized_native(FLAT, Stage::Kernel, &tmp()).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("Binding 480") && asm.contains("OpImageWrite"),
        "the flat member is surfaced as a storage image and written:\n{asm}"
    );
    let meta = parse_air_kernel_meta(FLAT).expect("parse");
    assert!(
        meta.unsurfaced_embedded_resources.is_empty(),
        "nothing about the flat form is unsurfaced: {:?}",
        meta.unsurfaced_embedded_resources
    );
}

/// The wrapped texture is a descriptor, at the argument id the nesting path sums to, and the store
/// reaches it -- the same storage image the flat spelling produces.
#[test]
fn a_wrapped_embedded_texture_binds_at_the_summed_argument_id() {
    let meta = parse_air_kernel_meta(WRAPPED).expect("parse");
    assert!(
        meta.unsurfaced_embedded_resources.is_empty(),
        "a wrapped 2D texture is surfaced, not refused: {:?}",
        meta.unsurfaced_embedded_resources
    );
    let texture = match meta.embedded_textures.as_slice() {
        [only] => *only,
        other => panic!("exactly one embedded texture, got {other:?}"),
    };
    // Wrapper id 7 + the texture's own relative `air.location_index` 3.
    assert_eq!(texture.argument_index, 10);
    // The byte offset is absolute within the argument buffer, and the ordinal is the OUTER member's
    // -- the first index of the GEP that reaches the handle.
    assert_eq!(texture.field_offset, 0);
    assert_eq!(texture.field_ordinal, 0);

    let spv = translate_sanitized_native(WRAPPED, Stage::Kernel, &tmp()).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("Binding 480") && asm.contains("OpImageWrite"),
        "the wrapped member is surfaced as a storage image and written:\n{asm}"
    );
}

/// The sample side: the pair that mattered, because this one used to answer zero silently.
#[test]
fn a_wrapped_embedded_texture_samples_its_own_image() {
    let spv =
        translate_sanitized_native(WRAPPED_SAMPLE, Stage::Fragment, &tmp()).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("OpSampledImage") && asm.contains("OpImageSample"),
        "the wrapped texture is sampled through its own descriptor:\n{asm}"
    );
}

/// The dimension is read from the member's type name, not assumed to be 2D: a wrapped `texture1d`
/// gets a 1D image rather than no descriptor at all.
#[test]
fn a_wrapped_texture_keeps_the_dimension_its_type_name_names() {
    let spv = translate_sanitized_native(WRAPPED_1D, Stage::Fragment, &tmp()).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains(" 1D 0 0 0 1 Unknown") && asm.contains("OpImageSample"),
        "the wrapped 1D texture is sampled through a 1D image:\n{asm}"
    );
}

/// A wrapper declared as an ARRAY of structs stays a named refusal. Binding element 0 alone would
/// leave every other element reading black in a module that validates and binds cleanly.
#[test]
fn an_arrayed_wrapper_still_refuses() {
    let error = translate_sanitized_native(WRAPPED_ARRAY, Stage::Fragment, &tmp())
        .expect_err("one of several declared textures must not be the only one bound");
    assert!(
        error.contains("does not surface") && error.contains("air.texture"),
        "the refusal names the resource that was missed: {error}"
    );
}

/// A C-array member is a descriptor ARRAY the runtime index selects into, not one descriptor.
///
/// The length lives in the member tuple, so reading the type name alone surfaced a single image and
/// emitted a sample of it for every value of the index -- element 0 whatever the shader asked for,
/// in a module that validates and whose reflection agrees with it. The index was still computed and
/// still compared against null; the comparison was simply dropped on the floor.
#[test]
fn a_flat_c_array_member_is_indexed_rather_than_collapsed_to_element_zero() {
    let meta = metal2vulkan::meta::parse_air_fragment_meta(FLAT_C_ARRAY).expect("parse");
    assert!(
        meta.unsurfaced_embedded_resources.is_empty(),
        "a C-array texture member is surfaced, not refused: {:?}",
        meta.unsurfaced_embedded_resources
    );
    let texture = match meta.embedded_textures.as_slice() {
        [only] => *only,
        other => panic!("exactly one embedded texture argument, got {other:?}"),
    };
    assert_eq!(texture.array_length, Some(32));

    let spv = translate_sanitized_native(FLAT_C_ARRAY, Stage::Fragment, &tmp()).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = Disassembly::parse(&asm);

    // A 32-element array of the image type, bound as one UniformConstant variable.
    let (array_ty, image_ty) = module
        .results()
        .find_map(|(id, op, args)| {
            let [element, length] = args else { return None };
            (op == "OpTypeArray" && module.constant_value(length) == Some(32))
                .then_some((id, *element))
        })
        .unwrap_or_else(|| panic!("a 32-element descriptor array is declared:\n{asm}"));
    assert_eq!(
        module.opcode(image_ty),
        Some("OpTypeImage"),
        "the array's element is the image type:\n{asm}"
    );
    let array_ptr = module
        .results()
        .find_map(|(id, op, args)| {
            (op == "OpTypePointer" && args.first() == Some(&array_ty)).then_some(id)
        })
        .unwrap_or_else(|| panic!("a pointer to the array type is declared:\n{asm}"));
    let array_var = module
        .results()
        .find_map(|(id, op, args)| {
            (op == "OpVariable" && args.first() == Some(&array_ptr)).then_some(id)
        })
        .unwrap_or_else(|| panic!("the array is bound as one variable, not one image:\n{asm}"));

    // Indexed by an SSA value -- the shader's own `slice` -- rather than by element 0.
    let element = module
        .results()
        .find_map(|(id, op, args)| match args {
            [_, base, index] if op == "OpAccessChain" && *base == array_var => {
                assert!(
                    module.constant_value(index).is_none(),
                    "the index is the runtime value, not a constant:\n{asm}"
                );
                Some(id)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("the array is indexed before it is loaded:\n{asm}"));
    let loaded = module
        .results()
        .find_map(|(id, op, args)| (op == "OpLoad" && args.get(1) == Some(&element)).then_some(id))
        .unwrap_or_else(|| panic!("the selected element is loaded:\n{asm}"));
    assert!(
        module
            .results()
            .any(|(_, op, args)| op == "OpSampledImage" && args.get(1) == Some(&loaded)),
        "and the loaded element is what gets sampled:\n{asm}"
    );
}

/// Just enough of a SPIR-V disassembly to ask which id defines what. The emitted text names ids
/// numerically, so matching on substrings cannot tell a descriptor array from an unrelated one.
struct Disassembly {
    /// `(result id, opcode, id operands)` for every instruction that defines a result. The
    /// operands are as printed, so for a value-producing instruction the first one is its RESULT
    /// TYPE and the instruction's own operands start at index 1.
    defs: Vec<(u32, String, Vec<u32>)>,
    /// The literal an `OpConstant` defines.
    constants: std::collections::HashMap<u32, u32>,
}

impl Disassembly {
    fn parse(asm: &str) -> Self {
        let mut defs = vec![];
        let mut constants = std::collections::HashMap::new();
        for line in asm.lines() {
            let Some((result, rest)) = line.split_once('=') else {
                continue;
            };
            let Some(result) = result
                .trim()
                .strip_prefix('%')
                .and_then(|id| id.parse().ok())
            else {
                continue;
            };
            let mut tokens = rest.split_whitespace();
            let Some(opcode) = tokens.next() else {
                continue;
            };
            // An `OpConstant`'s operands are its type then its literal value.
            if opcode == "OpConstant" {
                if let Some(value) = tokens.clone().nth(1).and_then(|tok| tok.parse().ok()) {
                    constants.insert(result, value);
                }
            }
            let ids = tokens
                .filter_map(|tok| tok.strip_prefix('%')?.parse().ok())
                .collect();
            defs.push((result, opcode.to_string(), ids));
        }
        Self { defs, constants }
    }

    fn results(&self) -> impl Iterator<Item = (u32, &str, &[u32])> {
        self.defs
            .iter()
            .map(|(id, op, args)| (*id, op.as_str(), args.as_slice()))
    }

    fn opcode(&self, id: u32) -> Option<&str> {
        self.results()
            .find(|(def, ..)| *def == id)
            .map(|(_, op, _)| op)
    }

    fn constant_value(&self, id: &u32) -> Option<u32> {
        self.constants.get(id).copied()
    }
}
