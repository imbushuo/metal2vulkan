//! Seam-neutral structural queries over retained SPIR-V types and values.

use super::*;

/// Snapshot result types for one function plus module-scope values. Passes that inspect every
/// instruction in a function use this instead of repeatedly scanning the complete module.
pub(in crate::passes) fn function_value_types(
    ctx: &Ctx,
    function_idx: usize,
) -> HashMap<Word, Word> {
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .chain(ctx.module.functions[function_idx].parameters.iter())
        .chain(
            ctx.module.functions[function_idx]
                .blocks
                .iter()
                .flat_map(|block| block.instructions.iter()),
        )
        .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
        .collect()
}

/// Find the defining type instruction of `ty` among the module's globals and any synthesized ones.
pub(in crate::passes) fn type_def_of(ctx: &Ctx, ty: Word) -> Option<Instruction> {
    if let Some((in_new_globals, index)) = ctx
        .phase_type_positions
        .as_ref()
        .and_then(|positions| positions.get(&ty))
        .copied()
    {
        let definition = if in_new_globals {
            ctx.new_globals.get(index)
        } else {
            ctx.module.types_global_values.get(index)
        };
        if definition.is_some_and(|instruction| instruction.result_id == Some(ty)) {
            return definition.cloned();
        }
    }
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|g| g.result_id == Some(ty))
        .cloned()
}

/// Find the defining instruction for a value id in globals, parameters, or function bodies.
pub(in crate::passes) fn value_def_instruction(ctx: &Ctx, value: Word) -> Option<Instruction> {
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|inst| inst.result_id == Some(value))
        .cloned()
        .or_else(|| {
            ctx.module.functions.iter().find_map(|func| {
                func.parameters
                    .iter()
                    .find(|param| param.result_id == Some(value))
                    .cloned()
                    .or_else(|| {
                        func.blocks.iter().find_map(|block| {
                            block
                                .instructions
                                .iter()
                                .find(|inst| inst.result_id == Some(value))
                                .cloned()
                        })
                    })
            })
        })
}

/// The `result_type` of a value id `v`: search global constants/types and every function body.
pub(in crate::passes) fn value_result_type(ctx: &Ctx, v: Word) -> Option<Word> {
    if let Some(result_type) = ctx
        .phase_value_types
        .as_ref()
        .and_then(|types| types.get(&v))
    {
        return Some(*result_type);
    }
    for g in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if g.result_id == Some(v) {
            return g.result_type;
        }
    }
    for f in &ctx.module.functions {
        for p in &f.parameters {
            if p.result_id == Some(v) {
                return p.result_type;
            }
        }
        for b in &f.blocks {
            for i in &b.instructions {
                if i.result_id == Some(v) {
                    return i.result_type;
                }
            }
        }
    }
    None
}

/// True if `v` is an integer scalar or vector rather than a sampler or pointer.
pub(in crate::passes) fn value_is_int_or_intvec(ctx: &Ctx, v: Word) -> bool {
    let Some(ty) = value_result_type(ctx, v) else {
        return false;
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return false;
    };
    match def.class.opcode {
        Op::TypeInt => true,
        Op::TypeVector => def
            .operands
            .first()
            .and_then(|o| match o {
                Operand::IdRef(e) => type_def_of(ctx, *e),
                _ => None,
            })
            .map(|e| e.class.opcode == Op::TypeInt)
            .unwrap_or(false),
        _ => false,
    }
}

/// True if `ty` is `OpTypeVector` whose element is `OpTypeFloat 16`.
pub(in crate::passes) fn is_half_vector(ctx: &Ctx, ty: Word) -> bool {
    let Some(def) = type_def_of(ctx, ty) else {
        return false;
    };
    if def.class.opcode != Op::TypeVector {
        return false;
    }
    let Some(Operand::IdRef(elem)) = def.operands.first() else {
        return false;
    };
    type_def_of(ctx, *elem)
        .map(|e| {
            e.class.opcode == Op::TypeFloat
                && e.operands.first() == Some(&Operand::LiteralBit32(16))
        })
        .unwrap_or(false)
}

/// The element type of `rty` when it is a vector, or `rty` itself otherwise.
pub(in crate::passes) fn element_type(ctx: &Ctx, rty: Word) -> Word {
    if let Some(def) = type_def_of(ctx, rty) {
        if def.class.opcode == Op::TypeVector {
            if let Some(Operand::IdRef(e)) = def.operands.first() {
                return *e;
            }
        }
    }
    rty
}

/// Scalar (or vector-element) bit width of an int/float type, or zero for other types.
pub(in crate::passes) fn scalar_bit_width(ctx: &Ctx, ty: Word) -> u32 {
    let elem = element_type(ctx, ty);
    type_def_of(ctx, elem)
        .and_then(|d| match d.class.opcode {
            Op::TypeInt | Op::TypeFloat => match d.operands.first() {
                Some(Operand::LiteralBit32(b)) => Some(*b),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(0)
}

/// `(element type, vector length)` for a vector-typed value.
pub(in crate::passes) fn vector_shape(ctx: &Ctx, v: Word) -> Option<(Word, u32)> {
    let ty = value_result_type(ctx, v)?;
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode != Op::TypeVector {
        return None;
    }
    let elem = match def.operands.first() {
        Some(Operand::IdRef(e)) => *e,
        _ => return None,
    };
    let n = match def.operands.get(1) {
        Some(Operand::LiteralBit32(n)) => *n,
        _ => return None,
    };
    Some((elem, n))
}

/// True if the scalar `OpType*` id is a float of the given width.
pub(in crate::passes) fn is_float_width(ctx: &Ctx, ty: Word, width: u32) -> bool {
    type_def_of(ctx, ty)
        .map(|d| {
            d.class.opcode == Op::TypeFloat
                && d.operands.first() == Some(&Operand::LiteralBit32(width))
        })
        .unwrap_or(false)
}

/// Whether `value` is an INTEGER constant that is non-zero in every lane. The answer is one-sided:
/// `false` means "cannot prove non-zero", which covers `OpConstantNull`, a genuinely zero literal,
/// and any non-constant. Callers use it to drop a guard whose predicate is a compile-time false, so
/// a wrong `true` would delete a live guard while a wrong `false` only leaves one standing.
///
/// The integer restriction is load-bearing, not incidental: float `-0.0` has a non-zero bit pattern
/// but compares equal to zero, so reading its literal as "never zero" would be exactly the wrong
/// answer for a float guard.
pub(in crate::passes) fn integer_constant_is_never_zero(ctx: &Ctx, value: Word) -> bool {
    let Some(def) = value_def_instruction(ctx, value) else {
        return false;
    };
    if def
        .result_type
        .is_none_or(|ty| integer_scalar_or_vector_element(ctx, ty).is_none())
    {
        return false;
    }
    match def.class.opcode {
        // A wide literal spans several words; the constant is non-zero if any of them is.
        Op::Constant => def.operands.iter().any(|operand| match operand {
            Operand::LiteralBit32(bits) => *bits != 0,
            Operand::LiteralBit64(bits) => *bits != 0,
            _ => false,
        }),
        Op::ConstantComposite => {
            !def.operands.is_empty()
                && def.operands.iter().all(|operand| match operand {
                    Operand::IdRef(component) => integer_constant_is_never_zero(ctx, *component),
                    _ => false,
                })
        }
        _ => false,
    }
}

/// The `TypeInt` element of `ty` when `ty` is an integer scalar or a vector of them, else `None`.
pub(in crate::passes) fn integer_scalar_or_vector_element(ctx: &Ctx, ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeInt => Some(ty),
        Op::TypeVector => match def.operands.first()? {
            Operand::IdRef(element) => {
                (type_def_of(ctx, *element)?.class.opcode == Op::TypeInt).then_some(*element)
            }
            _ => None,
        },
        _ => None,
    }
}
