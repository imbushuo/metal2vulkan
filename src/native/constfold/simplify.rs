//! Byte-neutral responsibility split of the former monolith; see the parent module.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

pub(in crate::native) fn block_id(blk: &crate::spirv_module::Block) -> Option<Word> {
    blk.label.as_ref().and_then(|l| l.result_id)
}

/// Fold every `OpBranchConditional` whose condition is a known constant into an `OpBranch` to the
/// taken target, dropping the preceding `OpSelectionMerge`. Loop-header conditionals (preceded by
/// `OpLoopMerge`) are left untouched — removing a loop merge would break the loop construct.
pub(in crate::native) fn fold_branches(
    f: &mut crate::spirv_module::Function,
    vals: &HashMap<Word, i128>,
) -> bool {
    let mut changed = false;
    for blk in &mut f.blocks {
        let n = blk.instructions.len();
        if n == 0 {
            continue;
        }
        let term = &blk.instructions[n - 1];
        if term.class.opcode != Op::BranchConditional {
            continue;
        }
        let Some(Operand::IdRef(cond)) = term.operands.first() else {
            continue;
        };
        let Some(c) = vals.get(cond).copied() else {
            continue;
        };
        let (Some(Operand::IdRef(t)), Some(Operand::IdRef(fl))) =
            (term.operands.get(1), term.operands.get(2))
        else {
            continue;
        };
        let taken = if c != 0 { *t } else { *fl };
        // Inspect the merge instruction preceding the terminator.
        let has_selection_merge =
            n >= 2 && blk.instructions[n - 2].class.opcode == Op::SelectionMerge;
        let has_loop_merge = n >= 2 && blk.instructions[n - 2].class.opcode == Op::LoopMerge;
        if has_loop_merge {
            continue; // never fold a loop header conditional
        }
        blk.instructions[n - 1] =
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(taken)]);
        if has_selection_merge {
            blk.instructions.remove(n - 2);
        }
        changed = true;
    }
    changed
}

/// Successor block ids of a terminator.
pub(in crate::native) fn successors(term: &Instruction) -> Vec<Word> {
    let id = |o: Option<&Operand>| match o {
        Some(Operand::IdRef(w)) => Some(*w),
        _ => None,
    };
    match term.class.opcode {
        Op::Branch => id(term.operands.first()).into_iter().collect(),
        Op::BranchConditional => id(term.operands.get(1))
            .into_iter()
            .chain(id(term.operands.get(2)))
            .collect(),
        Op::Switch => {
            // operands: selector, default, then (literal, label) pairs.
            let mut out: Vec<Word> = id(term.operands.get(1)).into_iter().collect();
            let mut i = 2;
            while i < term.operands.len() {
                if let Some(Operand::IdRef(w)) = term.operands.get(i + 1) {
                    out.push(*w);
                }
                i += 2;
            }
            out
        }
        _ => vec![],
    }
}

/// Remove blocks unreachable from the entry block, and fix phis in survivors to list exactly their
/// surviving predecessors.
pub(in crate::native) fn prune_unreachable(f: &mut crate::spirv_module::Function) -> bool {
    if f.blocks.is_empty() {
        return false;
    }
    let entry = match block_id(&f.blocks[0]) {
        Some(e) => e,
        None => return false,
    };
    let succ: HashMap<Word, Vec<Word>> = f
        .blocks
        .iter()
        .filter_map(|b| {
            let id = block_id(b)?;
            let term = b.instructions.last()?;
            Some((id, successors(term)))
        })
        .collect();
    // BFS reachability.
    let mut reach: HashSet<Word> = HashSet::new();
    let mut stack = vec![entry];
    while let Some(b) = stack.pop() {
        if !reach.insert(b) {
            continue;
        }
        if let Some(ss) = succ.get(&b) {
            for s in ss {
                if !reach.contains(s) {
                    stack.push(*s);
                }
            }
        }
    }
    let before = f.blocks.len();
    f.blocks
        .retain(|b| block_id(b).is_some_and(|id| reach.contains(&id)));
    let removed = f.blocks.len() != before;

    // Drop dangling structured-merge instructions: an `OpSelectionMerge`/`OpLoopMerge` whose merge
    // (or, for a loop, continue) target block was just pruned as unreachable is a forward reference
    // to a non-existent id — invalid SPIR-V that even the relooper cannot parse. Removing the merge
    // dissolves the (now incomplete) structured construct, leaving a well-formed — if unstructured —
    // module that the prune-then-relooper retry re-structures. This is what folding a function-
    // constant branch whose taken arm convergence was the merge produces (the merge block becomes
    // reachable only through the pruned not-taken path).
    let alive: HashSet<Word> = f.blocks.iter().filter_map(block_id).collect();
    let mut merge_fixed = false;
    for b in &mut f.blocks {
        let n = b.instructions.len();
        if n < 2 {
            continue;
        }
        let mi = n - 2;
        let op = b.instructions[mi].class.opcode;
        if op != Op::SelectionMerge && op != Op::LoopMerge {
            continue;
        }
        let merge_gone = matches!(
            b.instructions[mi].operands.first(),
            Some(Operand::IdRef(m)) if !alive.contains(m)
        );
        let cont_gone = op == Op::LoopMerge
            && matches!(
                b.instructions[mi].operands.get(1),
                Some(Operand::IdRef(c)) if !alive.contains(c)
            );
        if merge_gone || cont_gone {
            b.instructions.remove(mi);
            merge_fixed = true;
        }
    }

    // Recompute actual predecessors among survivors, then fix every phi to those preds.
    let mut preds: HashMap<Word, HashSet<Word>> = HashMap::new();
    for b in &f.blocks {
        let Some(id) = block_id(b) else { continue };
        if let Some(term) = b.instructions.last() {
            for s in successors(term) {
                preds.entry(s).or_default().insert(id);
            }
        }
    }
    let mut phi_fixed = false;
    for b in &mut f.blocks {
        let Some(id) = block_id(b) else { continue };
        let allowed = preds.get(&id).cloned().unwrap_or_default();
        for inst in &mut b.instructions {
            if inst.class.opcode != Op::Phi {
                continue;
            }
            // operands: (value, parent) pairs.
            let mut kept: Vec<Operand> = Vec::new();
            let mut i = 0;
            while i + 1 < inst.operands.len() {
                if let Operand::IdRef(parent) = inst.operands[i + 1] {
                    if allowed.contains(&parent) {
                        kept.push(inst.operands[i].clone());
                        kept.push(inst.operands[i + 1].clone());
                    } else {
                        phi_fixed = true;
                    }
                }
                i += 2;
            }
            inst.operands = kept;
        }
    }
    removed || phi_fixed || merge_fixed
}

/// Replace a phi that has only one possible outcome -- every incoming value is the same id -- with
/// that value, then substitute it throughout the function. The single-incoming phi is the one-arm
/// instance of this, not a separate rule.
///
/// The listed parents must be EXACTLY the block's actual CFG predecessors. A phi with a stale
/// partial arm list is not an SSA identity: replacing it could expose the listed value along an
/// incoming edge that its definition does not dominate. When the lists do agree, the value
/// dominates the end of every predecessor, so it dominates the phi's own block and every use the
/// phi had. A phi that names ITSELF as an incoming value is excluded -- collapsing it to its own
/// result id would delete the only definition of that id.
pub(in crate::native) fn collapse_trivial_phis(f: &mut crate::spirv_module::Function) -> bool {
    let mut predecessors: HashMap<Word, HashSet<Word>> = HashMap::new();
    for block in &f.blocks {
        let Some(parent) = block_id(block) else {
            continue;
        };
        if let Some(terminator) = block.instructions.last() {
            for successor in successors(terminator) {
                predecessors.entry(successor).or_default().insert(parent);
            }
        }
    }
    let mut repl: HashMap<Word, Word> = HashMap::new();
    for b in &f.blocks {
        let Some(block) = block_id(b) else { continue };
        for inst in &b.instructions {
            if inst.class.opcode != Op::Phi
                || inst.operands.is_empty()
                || !inst.operands.len().is_multiple_of(2)
            {
                continue;
            }
            let Some(rid) = inst.result_id else { continue };
            let mut only_value: Option<Word> = None;
            let mut parents: HashSet<Word> = HashSet::new();
            let identity = inst.operands.chunks_exact(2).all(|pair| {
                let (Some(Operand::IdRef(value)), Some(Operand::IdRef(parent))) =
                    (pair.first(), pair.get(1))
                else {
                    return false;
                };
                parents.insert(*parent);
                *value != rid && *only_value.get_or_insert(*value) == *value
            });
            if let (true, Some(value)) = (identity, only_value) {
                if predecessors.get(&block) == Some(&parents) {
                    repl.insert(rid, value);
                }
            }
        }
    }
    if repl.is_empty() {
        return false;
    }
    drop_and_substitute(f, &repl);
    true
}

/// Delete every instruction whose result is a key of `repl` and rewrite every remaining reference
/// to it as the value it collapsed to, following chains so a value that itself collapsed resolves
/// to its final replacement. Shared by the phi and the select collapse: both reduce a merge with
/// one possible outcome to that outcome, and the bookkeeping afterwards is the same.
fn drop_and_substitute(f: &mut crate::spirv_module::Function, repl: &HashMap<Word, Word>) {
    let resolve = |mut x: Word| -> Word {
        let mut guard = 0;
        while let Some(&n) = repl.get(&x) {
            if n == x || guard > repl.len() {
                break;
            }
            x = n;
            guard += 1;
        }
        x
    };
    for b in &mut f.blocks {
        b.instructions
            .retain(|i| !i.result_id.is_some_and(|r| repl.contains_key(&r)));
    }
    for b in &mut f.blocks {
        for inst in &mut b.instructions {
            for op in &mut inst.operands {
                if let Operand::IdRef(id) = op {
                    if repl.contains_key(id) {
                        *op = Operand::IdRef(resolve(*id));
                    }
                }
            }
        }
    }
}

/// Replace an `OpSelect` that has only one possible outcome with that outcome: either its two arms
/// are the same id, or `vals` proves its condition constant. This is the value-side twin of
/// `fold_branches` (which does the same for `OpBranchConditional`) and of `collapse_trivial_phis`
/// (which does the same for a merge in phi form). Unlike the phi there is no dominance side
/// condition to check: both arms of a select must already dominate the select itself, so the
/// surviving arm dominates every use the select had.
pub(in crate::native) fn collapse_constant_selects(
    f: &mut crate::spirv_module::Function,
    vals: &HashMap<Word, i128>,
    lane_conditions: &HashSet<Word>,
) -> bool {
    let mut repl: HashMap<Word, Word> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.instructions {
            if inst.class.opcode != Op::Select {
                continue;
            }
            let (
                Some(rid),
                Some(Operand::IdRef(cond)),
                Some(Operand::IdRef(a)),
                Some(Operand::IdRef(b)),
            ) = (
                inst.result_id,
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            )
            else {
                continue;
            };
            if a == b {
                repl.insert(rid, *a);
                continue;
            }
            // A vector condition selects per lane, and `vals` is a SCALAR lattice: an entry for
            // a vector id (a `Bitcast` of a scalar constant can put one there) says nothing about
            // the individual lanes. `fold_branches` never meets this because a branch condition is
            // always a scalar bool; a select's need not be, so it is checked.
            if lane_conditions.contains(cond) {
                continue;
            }
            if let Some(c) = vals.get(cond) {
                repl.insert(rid, if *c != 0 { *a } else { *b });
            }
        }
    }
    if repl.is_empty() {
        return false;
    }
    drop_and_substitute(f, &repl);
    true
}

/// Whether an instruction is a pure value computation safe to delete when its result is unused.
pub(in crate::native) fn is_pure(op: Op) -> bool {
    matches!(
        op,
        Op::Undef
            | Op::AccessChain
            | Op::InBoundsAccessChain
            | Op::PtrAccessChain
            | Op::Load
            | Op::CopyObject
            | Op::Bitcast
            | Op::UConvert
            | Op::SConvert
            | Op::FConvert
            | Op::ConvertSToF
            | Op::ConvertUToF
            | Op::ConvertFToS
            | Op::ConvertFToU
            | Op::IAdd
            | Op::ISub
            | Op::IMul
            | Op::UDiv
            | Op::SDiv
            | Op::UMod
            | Op::SMod
            | Op::SRem
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FNegate
            | Op::SNegate
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::BitwiseAnd
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::Not
            | Op::IEqual
            | Op::INotEqual
            | Op::ULessThan
            | Op::SLessThan
            | Op::UGreaterThan
            | Op::SGreaterThan
            | Op::ULessThanEqual
            | Op::SLessThanEqual
            | Op::UGreaterThanEqual
            | Op::SGreaterThanEqual
            | Op::FOrdEqual
            | Op::FOrdNotEqual
            | Op::FOrdLessThan
            | Op::FOrdGreaterThan
            | Op::FOrdLessThanEqual
            | Op::FOrdGreaterThanEqual
            | Op::LogicalNot
            | Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalEqual
            | Op::LogicalNotEqual
            | Op::Select
            | Op::Phi
            | Op::CompositeExtract
            | Op::CompositeConstruct
            | Op::CompositeInsert
            | Op::VectorShuffle
            | Op::VectorExtractDynamic
            | Op::VectorTimesScalar
            | Op::MatrixTimesVector
            | Op::MatrixTimesMatrix
            | Op::Transpose
            | Op::Dot
    )
}

/// Remove pure, result-bearing instructions whose result is dead. Liveness is computed
/// TRANSITIVELY from roots (non-pure sinks — terminators, stores, merge instructions, impure
/// result ops, plus decorations/debug-names/entry-points/types), not by "any operand reference =
/// used": a pure instruction's operands only count as a use when the instruction itself is live.
/// This is what lets a self-referential DEAD CYCLE be collected — e.g. a loop-carried pointer
/// phi whose only remaining reference is its own back-edge `OpPtrAccessChain` (the consumer load
/// having been pruned with its statically-dead FC arm). The naive mark "an id is used if it
/// appears in any operand" keeps such a cycle alive forever (the phi marks the access chain used,
/// the access chain marks the phi used), so the mistyped pointer phi survives and the module stays
/// invalid; transitive liveness from sinks drops the whole cycle in one pass.
#[cfg(test)]
pub(in crate::native) fn dce(module: &mut Module) -> bool {
    dce_preserving(module, &HashSet::new())
}

pub(in crate::native) fn dce_preserving(
    module: &mut Module,
    preserved_global_ids: &HashSet<Word>,
) -> bool {
    // Pure, result-bearing definitions: result id -> the operand ids it depends on. A live result
    // propagates liveness to these; a result reachable from no sink is dead.
    let mut pure_def: HashMap<Word, Vec<Word>> = HashMap::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                if is_pure(inst.class.opcode) {
                    if let Some(r) = inst.result_id {
                        let deps = inst
                            .operands
                            .iter()
                            .filter_map(|op| match op {
                                Operand::IdRef(id) => Some(*id),
                                _ => None,
                            })
                            .collect();
                        pure_def.insert(r, deps);
                    }
                }
            }
        }
    }

    // Seed the worklist from every reference that is NOT an operand of a pure result-bearing
    // instruction: module roots (decorations, names, entry points, exec modes, type/global
    // operands) and, in function bodies, the operands of sinks (terminators, stores, merges,
    // OpExtInst, impure result ops, labels).
    let mut work: Vec<Word> = Vec::new();
    let seed = |ops: &[Operand], work: &mut Vec<Word>| {
        for op in ops {
            if let Operand::IdRef(id) = op {
                work.push(*id);
            }
        }
    };
    for s in [
        &module.entry_points,
        &module.debug_names,
        &module.annotations,
        &module.execution_modes,
    ] {
        for inst in s {
            seed(&inst.operands, &mut work);
        }
    }
    work.extend(preserved_global_ids.iter().copied());
    for inst in &module.types_global_values {
        seed(&inst.operands, &mut work);
    }
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                let is_pure_def = is_pure(inst.class.opcode) && inst.result_id.is_some();
                if !is_pure_def {
                    seed(&inst.operands, &mut work);
                }
            }
        }
    }

    // Transitive closure: a live pure result keeps its operands live.
    let mut live = HashSet::new();
    while let Some(id) = work.pop() {
        if !live.insert(id) {
            continue;
        }
        if let Some(deps) = pure_def.get(&id) {
            for &d in deps {
                if !live.contains(&d) {
                    work.push(d);
                }
            }
        }
    }

    let mut any = false;
    for f in &mut module.functions {
        for b in &mut f.blocks {
            let before = b.instructions.len();
            b.instructions.retain(|inst| {
                let removable = is_pure(inst.class.opcode)
                    && inst.result_id.is_some_and(|r| !live.contains(&r));
                !removable
            });
            if b.instructions.len() != before {
                any = true;
            }
        }
    }
    // Module-scope dead-constant sweep, restricted to the value producers a dead arm leaves behind:
    // an `OpConstantNull`/`OpUndef` (e.g. the null base of a pruned pointer-induction walk). These
    // are pure and operand-free, so removing an unreferenced one is always safe — and necessary: a
    // pointer-typed `OpConstantNull %_ptr_UniformConstant_*` is itself invalid SPIR-V ("may only
    // return a logical pointer in StorageBuffer/Workgroup"), so leaving the orphan blocks the
    // relooper's structured output from validating even after the dead arm that used it is gone.
    // Types and global `OpVariable`s (interface/descriptors) are never swept.
    let before = module.types_global_values.len();
    module.types_global_values.retain(|inst| {
        let sweepable = matches!(inst.class.opcode, Op::ConstantNull | Op::Undef);
        let dead = inst.result_id.is_some_and(|r| !live.contains(&r));
        !(sweepable && dead)
    });
    if module.types_global_values.len() != before {
        any = true;
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    #[test]
    fn single_incoming_phi_substitutes_its_concrete_value() {
        let mut entry = Block::new();
        entry.label = Some(Instruction::new(Op::Label, None, Some(7), vec![]));
        entry.instructions = vec![Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(8)],
        )];
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(8), vec![]));
        block.instructions = vec![
            // The deliberately stale result type models an interface-refined image value still
            // wrapped in the pointer carrier recorded before CFG edge splitting.
            Instruction::new(
                Op::Phi,
                Some(2),
                Some(10),
                vec![Operand::IdRef(5), Operand::IdRef(7)],
            ),
            Instruction::new(
                Op::SampledImage,
                Some(3),
                Some(11),
                vec![Operand::IdRef(10), Operand::IdRef(6)],
            ),
        ];
        let mut function = Function::new();
        function.blocks = vec![entry, block];

        assert!(collapse_trivial_phis(&mut function));
        assert_eq!(function.blocks[1].instructions.len(), 1);
        assert_eq!(
            function.blocks[1].instructions[0].operands.first(),
            Some(&Operand::IdRef(5))
        );
    }

    /// A phi is an identity when every incoming edge carries the SAME value, not only when there is
    /// one edge. Three predecessors all handing over `%5` make the phi `%5`. The two refusals are
    /// the safety of that: a parent list that does not name every real predecessor is not an SSA
    /// identity (the value need not dominate the unlisted edge), and a phi that names its own result
    /// would lose the only definition of that id.
    #[test]
    fn phi_collapses_when_every_edge_carries_one_value() {
        let label = |id: Word| Some(Instruction::new(Op::Label, None, Some(id), vec![]));
        let branch =
            |target: Word| Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(target)]);
        let phi = |result: Word, arms: &[(Word, Word)]| {
            Instruction::new(
                Op::Phi,
                Some(2),
                Some(result),
                arms.iter()
                    .flat_map(|(value, parent)| [Operand::IdRef(*value), Operand::IdRef(*parent)])
                    .collect(),
            )
        };
        // Three predecessors (7, 8, 9) all branch to the join block 10.
        let join = |body: Vec<Instruction>| {
            let mut function = Function::new();
            let mut blocks = Vec::new();
            for id in [7, 8, 9] {
                let mut block = Block::new();
                block.label = label(id);
                block.instructions = vec![branch(10)];
                blocks.push(block);
            }
            let mut merge = Block::new();
            merge.label = label(10);
            merge.instructions = body;
            blocks.push(merge);
            function.blocks = blocks;
            function
        };

        let mut all_three = join(vec![
            phi(11, &[(5, 7), (5, 8), (5, 9)]),
            Instruction::new(Op::CopyObject, Some(2), Some(12), vec![Operand::IdRef(11)]),
        ]);
        assert!(collapse_trivial_phis(&mut all_three));
        let body = &all_three.blocks[3].instructions;
        assert_eq!(body.len(), 1, "the phi is gone");
        assert_eq!(
            body[0].operands.first(),
            Some(&Operand::IdRef(5)),
            "its use reads the one incoming value"
        );

        // Block 9 is a real predecessor the arm list does not name.
        let mut partial = join(vec![phi(11, &[(5, 7), (5, 8)])]);
        assert!(
            !collapse_trivial_phis(&mut partial),
            "a partial parent list is not an SSA identity"
        );

        // Every arm carries the phi's own result.
        let mut circular = join(vec![phi(11, &[(11, 7), (11, 8), (11, 9)])]);
        assert!(
            !collapse_trivial_phis(&mut circular),
            "a self-referential phi keeps its definition"
        );

        // Two distinct incoming values are a real merge.
        let mut real = join(vec![phi(11, &[(5, 7), (5, 8), (6, 9)])]);
        assert!(!collapse_trivial_phis(&mut real), "a real merge survives");
    }

    #[test]
    fn dce_keeps_only_typed_sidecar_rooted_dead_global() {
        let ulong = 1;
        let sentinel = 2;
        let dead = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(ulong),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(sentinel), vec![]),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(dead), vec![]),
        ];

        assert!(dce_preserving(&mut module, &HashSet::from([sentinel])));

        let ids = module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&sentinel));
        assert!(!ids.contains(&dead));
    }
}
