//! Materialization of subgroup results a select would otherwise make conditional.
//!
//! SPIRV-Cross (MoltenVK's MSL frontend) forwards a single-use SPIR-V value into the expression
//! that reads it rather than emitting a temporary. When the reader is an `OpSelect`, the value
//! becomes an arm of an MSL ternary, and an arm is evaluated only by the lanes that take it. A
//! subgroup instruction moved into an arm that way stops being executed by the whole subgroup, and
//! Metal's contract for `simd_shuffle` — the source lane must participate in the call — no longer
//! holds: a lane whose source sits in the *other* arm reads undefined data.
//!
//! The failure is silent and its shape is exact. In a prefix-sum window (`shuffle_down(scan, d)`
//! for the near lanes, a broadcast-and-`shuffle_up` for the far ones, selected on
//! `lane < width - d`), every lane at or above `width - 2*d` comes back wrong: the near arm's
//! lanes with `lane + d >= width - d` source a lane that took the far arm, and the far arm's lanes
//! source lanes that took the near one.
//!
//! Spilling the result through a `Function` variable is what fixes it. An `OpStore` is a
//! statement, never an expression, so SPIRV-Cross has to emit the subgroup call where it stands —
//! unconditionally — and the ternary reads a plain local. On a driver that consumes SPIR-V
//! directly the spill is a trivially promotable copy, so this costs nothing where nothing is
//! wrong.
//!
//! Only the values that can actually be forwarded are spilled: the result must have exactly one
//! reader in the whole function, and the chain from it to the select must be single-use pure
//! arithmetic inside one block. A value read twice, or read from another block, is already a
//! SPIRV-Cross temporary.

use super::Ctx;
use crate::spirv_module::{Instruction, Operand};
use spirv::{Op, StorageClass, Word};
use std::collections::HashMap;

/// Subgroup instructions whose result carries an execution effect, so hoisting the call into a
/// conditional changes which lanes take part in it.
fn is_subgroup_op(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::GroupNonUniformAll
            | Op::GroupNonUniformAllEqual
            | Op::GroupNonUniformAny
            | Op::GroupNonUniformBallot
            | Op::GroupNonUniformBallotBitCount
            | Op::GroupNonUniformBallotBitExtract
            | Op::GroupNonUniformBallotFindLSB
            | Op::GroupNonUniformBallotFindMSB
            | Op::GroupNonUniformBitwiseAnd
            | Op::GroupNonUniformBitwiseOr
            | Op::GroupNonUniformBitwiseXor
            | Op::GroupNonUniformBroadcast
            | Op::GroupNonUniformBroadcastFirst
            | Op::GroupNonUniformElect
            | Op::GroupNonUniformFAdd
            | Op::GroupNonUniformFMax
            | Op::GroupNonUniformFMin
            | Op::GroupNonUniformFMul
            | Op::GroupNonUniformIAdd
            | Op::GroupNonUniformIMul
            | Op::GroupNonUniformLogicalAnd
            | Op::GroupNonUniformLogicalOr
            | Op::GroupNonUniformLogicalXor
            | Op::GroupNonUniformRotateKHR
            | Op::GroupNonUniformSMax
            | Op::GroupNonUniformSMin
            | Op::GroupNonUniformShuffle
            | Op::GroupNonUniformShuffleDown
            | Op::GroupNonUniformShuffleUp
            | Op::GroupNonUniformShuffleXor
            | Op::GroupNonUniformUMax
            | Op::GroupNonUniformUMin
    )
}

/// Instructions SPIRV-Cross emits as a pure expression, so a single-use chain of them carries the
/// forwarding all the way from the subgroup call to the select. Anything else (a load, a call, a
/// phi) either has its own temporary or is a block-boundary value.
fn forwards_as_expression(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::CopyObject
            | Op::Bitcast
            | Op::FNegate
            | Op::SNegate
            | Op::Not
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::IAdd
            | Op::ISub
            | Op::IMul
            | Op::UDiv
            | Op::SDiv
            | Op::UMod
            | Op::SMod
            | Op::SRem
            | Op::FMod
            | Op::FRem
            | Op::BitwiseAnd
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::ConvertFToS
            | Op::ConvertFToU
            | Op::ConvertSToF
            | Op::ConvertUToF
            | Op::FConvert
            | Op::SConvert
            | Op::UConvert
            | Op::CompositeConstruct
            | Op::CompositeExtract
            | Op::CompositeInsert
            | Op::VectorShuffle
            | Op::VectorTimesScalar
            | Op::ExtInst
            | Op::Select
    )
}

/// Spill every subgroup result an `OpSelect` would otherwise turn into a conditionally evaluated
/// MSL ternary arm. Modules whose subgroup results are already materialized come back untouched.
pub(in crate::passes) fn materialize_selected_subgroup_results(ctx: &mut Ctx) {
    for function_idx in 0..ctx.module.functions.len() {
        materialize_in_function(ctx, function_idx);
    }
}

fn materialize_in_function(ctx: &mut Ctx, function_idx: usize) {
    let function = &ctx.module.functions[function_idx];
    // Function-wide read counts: a value read twice, or read from a block other than its own, is
    // already a SPIRV-Cross temporary and cannot be forwarded into a ternary arm.
    let mut reads: HashMap<Word, usize> = HashMap::new();
    for block in &function.blocks {
        for inst in &block.instructions {
            for operand in &inst.operands {
                if let Operand::IdRef(id) = operand {
                    *reads.entry(*id).or_default() += 1;
                }
            }
        }
    }

    let mut spill: Vec<(usize, usize, Word, Word)> = Vec::new(); // (block, index, result, type)
    for (block_idx, block) in function.blocks.iter().enumerate() {
        let defs: HashMap<Word, (usize, Op, Word)> = block
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(index, inst)| {
                Some((
                    inst.result_id?,
                    (index, inst.class.opcode, inst.result_type?),
                ))
            })
            .collect();
        let mut queue: Vec<Word> = block
            .instructions
            .iter()
            .filter(|inst| inst.class.opcode == Op::Select)
            // Operand 0 is the condition; the two arms follow.
            .flat_map(|inst| inst.operands.iter().skip(1))
            .filter_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect();
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = queue.pop() {
            if !seen.insert(id) || reads.get(&id).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(&(index, opcode, result_type)) = defs.get(&id) else {
                continue;
            };
            if is_subgroup_op(opcode) {
                spill.push((block_idx, index, id, result_type));
                continue;
            }
            if !forwards_as_expression(opcode) {
                continue;
            }
            queue.extend(
                block.instructions[index]
                    .operands
                    .iter()
                    .filter_map(|operand| match operand {
                        Operand::IdRef(operand) => Some(*operand),
                        _ => None,
                    }),
            );
        }
    }
    if spill.is_empty() {
        return;
    }

    // A SPIR-V Function variable must be declared in the function's first block, so collect the
    // declarations and the in-place edits separately and apply each once.
    let mut variables = Vec::new();
    let mut edits: HashMap<usize, Vec<(usize, Word, Word, Word)>> = HashMap::new();
    for (block_idx, index, result, result_type) in spill {
        let ptr_type = ctx.ty_ptr(StorageClass::Function, result_type);
        let variable = ctx.module.fresh_id();
        let reload = ctx.module.fresh_id();
        variables.push(Instruction::new(
            Op::Variable,
            Some(ptr_type),
            Some(variable),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        edits
            .entry(block_idx)
            .or_default()
            .push((index, result, variable, reload));
    }

    let function = &mut ctx.module.functions[function_idx];
    for (block_idx, mut block_edits) in edits {
        // Later insertions first, so an earlier one does not shift the index of a later one.
        block_edits.sort_by_key(|(index, ..)| std::cmp::Reverse(*index));
        for (index, result, variable, reload) in block_edits {
            let result_type = function.blocks[block_idx].instructions[index].result_type;
            function.blocks[block_idx].instructions.splice(
                index + 1..index + 1,
                [
                    Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(variable), Operand::IdRef(result)],
                    ),
                    Instruction::new(
                        Op::Load,
                        result_type,
                        Some(reload),
                        vec![Operand::IdRef(variable)],
                    ),
                ],
            );
            // The single reader is by construction later in this same block.
            for inst in function.blocks[block_idx]
                .instructions
                .iter_mut()
                .skip(index + 3)
            {
                for operand in inst.operands.iter_mut() {
                    if *operand == Operand::IdRef(result) {
                        *operand = Operand::IdRef(reload);
                    }
                }
            }
        }
    }
    let entry = &mut function.blocks[0];
    let after_variables = entry
        .instructions
        .iter()
        .position(|inst| inst.class.opcode != Op::Variable)
        .unwrap_or(entry.instructions.len());
    entry
        .instructions
        .splice(after_variables..after_variables, variables);
}
