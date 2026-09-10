use super::*;
use crate::passes::Stage;

fn float_shape(module: &Module, ty: Word) -> Option<(u32, u32)> {
    let definition = module
        .types_global_values
        .iter()
        .find(|i| i.result_id == Some(ty))?;
    match (definition.class.opcode, definition.operands.as_slice()) {
        (Op::TypeFloat, [Operand::LiteralBit32(bits)]) => Some((*bits, 1)),
        (Op::TypeVector, [Operand::IdRef(component), Operand::LiteralBit32(lanes)]) => {
            float_shape(module, *component).map(|(bits, _)| (bits, *lanes))
        }
        _ => None,
    }
}

fn interface_shapes(module: &Module, storage: StorageClass) -> Vec<(u32, u32)> {
    module
        .types_global_values
        .iter()
        .filter(|i| {
            i.class.opcode == Op::Variable
                && i.operands.first() == Some(&Operand::StorageClass(storage))
        })
        .map(|variable| {
            let pointer = module
                .types_global_values
                .iter()
                .find(|i| i.result_id == variable.result_type)
                .unwrap();
            let Operand::IdRef(pointee) = pointer.operands[1] else {
                panic!("pointee")
            };
            float_shape(module, pointee).expect("floating-point interface")
        })
        .collect()
}

#[test]
fn half_stage_io_preserves_interpolation_and_accepts_unused_inputs() {
    let source = r#"
target triple = "spirv-unknown-vulkan1.2"
define half @f(half %a, half %b, half %c, half %unused) {
entry:
  %ab = fadd half %a, %b
  %abc = fadd half %ab, %c
  ret half %abc
}
!air.fragment = !{!0}
!0 = !{ptr @f, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half"}
!3 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.fragment_input", !"generated(a)", !"air.flat", !"air.arg_type_name", !"half"}
!5 = !{i32 1, !"air.fragment_input", !"generated(b)", !"air.centroid", !"air.no_perspective", !"air.arg_type_name", !"half"}
!6 = !{i32 2, !"air.fragment_input", !"generated(c)", !"air.sample", !"air.perspective", !"air.arg_type_name", !"half"}
!7 = !{i32 3, !"air.fragment_input", !"generated(unused)", !"air.center", !"air.perspective", !"air.arg_type_name", !"half"}
"#;
    let directory = std::env::temp_dir().join(format!(
        "metal2vulkan-half-interpolation-{}",
        std::process::id(),
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let bytes = crate::translate_sanitized_native(source, Stage::Fragment, &directory).unwrap();
    let module = load_bytes(&bytes).unwrap();
    for (location, decoration) in [
        (0, Decoration::Flat),
        (1, Decoration::Centroid),
        (1, Decoration::NoPerspective),
        (2, Decoration::Sample),
    ] {
        let input = module.annotations.iter().find_map(|i| match i.operands.as_slice() {
            [Operand::IdRef(id), Operand::Decoration(Decoration::Location),
                Operand::LiteralBit32(loc)] if *loc == location && module.types_global_values.iter()
                    .any(|v| v.result_id == Some(*id) && v.operands.first()
                        == Some(&Operand::StorageClass(StorageClass::Input))) => Some(*id),
            _ => None,
        }).expect("input location");
        assert!(
            module
                .annotations
                .iter()
                .any(|i| i.operands == [Operand::IdRef(input), Operand::Decoration(decoration),]),
            "location {location} must retain {decoration:?}"
        );
    }
    assert!(interface_shapes(&module, StorageClass::Input)
        .iter()
        .all(|shape| *shape == (32, 1)));
    crate::tools::spirv_val_bytes(&bytes, &directory).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn half_stage_io_uses_float_interfaces_without_widening_shader_math_or_buffers() {
    let directory =
        std::env::temp_dir().join(format!("metal2vulkan-half-stage-io-{}", std::process::id(),));
    std::fs::create_dir_all(&directory).unwrap();
    for lanes in 1..=4 {
        let ty = if lanes == 1 {
            "half".to_owned()
        } else {
            format!("<{lanes} x half>")
        };
        let name = if lanes == 1 {
            "half".to_owned()
        } else {
            format!("half{lanes}")
        };
        let vertex = format!(
            r#"
target triple = "spirv-unknown-vulkan1.2"
define <{{ <4 x float>, {ty} }}> @v(<4 x float> %position, {ty} %attribute, ptr addrspace(1) %buffer) {{
entry:
  %stored = load {ty}, ptr addrspace(1) %buffer, align 2
  %sum = fadd {ty} %stored, %attribute
  %a = insertvalue <{{ <4 x float>, {ty} }}> undef, <4 x float> %position, 0
  %b = insertvalue <{{ <4 x float>, {ty} }}> %a, {ty} %sum, 1
  ret <{{ <4 x float>, {ty} }}> %b
}}
!air.vertex = !{{!0}}
!0 = !{{ptr @v, !1, !4}}
!1 = !{{!2, !3}}
!2 = !{{!"air.position", !"air.arg_type_name", !"float4"}}
!3 = !{{!"air.vertex_output", !"generated(color)", !"air.arg_type_name", !"{name}"}}
!4 = !{{!5, !6, !7}}
!5 = !{{i32 0, !"air.vertex_input", i32 0, !"air.arg_type_name", !"float4"}}
!6 = !{{i32 1, !"air.vertex_input", i32 1, !"air.arg_type_name", !"{name}"}}
!7 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"{name}"}}
"#
        );
        let fragment = format!(
            r#"
target triple = "spirv-unknown-vulkan1.2"
define {ty} @f({ty} %color) {{
entry:
  %sum = fadd {ty} %color, %color
  ret {ty} %sum
}}
!air.fragment = !{{!0}}
!0 = !{{ptr @f, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"{name}"}}
!3 = !{{!4}}
!4 = !{{i32 0, !"air.fragment_input", !"generated(color)", !"air.center", !"air.perspective", !"air.arg_type_name", !"{name}"}}
"#
        );
        for (source, stage) in [(&vertex, Stage::Vertex), (&fragment, Stage::Fragment)] {
            let bytes = crate::translate_sanitized_native(source, stage, &directory)
                .expect("half stage interface translates");
            let module = load_bytes(&bytes).unwrap();
            for storage in [StorageClass::Input, StorageClass::Output] {
                let shapes = interface_shapes(&module, storage);
                assert!(
                    shapes.contains(&(32, lanes)),
                    "{stage:?} {storage:?}: {shapes:?}"
                );
                assert!(shapes.iter().all(|(bits, _)| *bits == 32), "{shapes:?}");
            }
            let conversions: Vec<_> = module
                .all_inst_iter()
                .filter(|i| i.class.opcode == Op::FConvert)
                .filter_map(|i| float_shape(&module, i.result_type?))
                .collect();
            assert!(
                conversions.contains(&(16, lanes)),
                "entry rounds to half: {conversions:?}"
            );
            assert!(
                conversions.contains(&(32, lanes)),
                "exit widens exactly: {conversions:?}"
            );
            assert!(
                module.all_inst_iter().any(|i| {
                    i.class.opcode == Op::FAdd
                        && i.result_type.and_then(|ty| float_shape(&module, ty))
                            == Some((16, lanes))
                }),
                "shader arithmetic must remain half"
            );
            if stage == Stage::Vertex {
                assert!(
                    module.all_inst_iter().any(|i| {
                        matches!(i.class.opcode, Op::Load | Op::Bitcast)
                            && i.result_type
                                .and_then(|ty| float_shape(&module, ty))
                                .is_some_and(|(bits, _)| bits == 16)
                    }),
                    "buffer loads retain half storage, directly or unpacked from integer words"
                );
            }
            crate::tools::spirv_val_bytes(&bytes, &directory).expect("valid Vulkan shader");
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}
