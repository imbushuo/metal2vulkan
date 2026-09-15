//! Splitting one AIR call site into its own region of basic blocks.
//!
//! Most AIR lowerings return a flat `Vec<Instruction>` that the driver splices into the block the
//! call was in. A lowering that needs control flow -- a per-lane guard, a copy loop -- cannot: it
//! has to end the call's block, insert blocks of its own, and continue the rest of the block
//! afterwards. That is why such a lowering runs as a PRE-PASS, before `lower_air_calls` walks the
//! blocks and caches the block graph it walks.
//!
//! The mechanical part is the same whichever lowering wants it, and it is the part that is easy to
//! get subtly wrong: a split must not strand a terminator, must not fall between a structured merge
//! and the terminator it claims, must keep a loop header on its original label, and must repoint
//! every successor `OpPhi` that named the old block at whichever block now branches there.
//! [`CallSiteSplit`] owns exactly that and nothing about any particular intrinsic.

use crate::spirv_module::{Block, Function, Instruction, Operand};
use spirv::{Op, Word};
use std::collections::HashSet;

/// One call site opened up for a lowering to build blocks in.
///
/// `prefix` is everything before the call and `suffix` everything after it, including the original
/// terminator. A lowering appends its branch to `prefix`, builds its own blocks, and calls
/// [`CallSiteSplit::finish`] with them; `suffix` becomes the continuation block, whose label the
/// lowering can read from [`CallSiteSplit::continuation`] before it builds anything.
pub(in crate::passes) struct CallSiteSplit {
    /// Instructions before the call, for the lowering to append its own setup and branch to.
    pub(in crate::passes) prefix: Vec<Instruction>,
    /// Instructions after the call, which become the continuation block.
    pub(in crate::passes) suffix: Vec<Instruction>,
    block: usize,
    old_label: Word,
    continuation: Option<Word>,
    successors: HashSet<Word>,
    /// Set when the split block was a loop header: the private selection merge that stands in for
    /// the loop merge on the continuation, and the real loop merge it branches to.
    loop_exit_passthrough: Option<(Word, Word)>,
    /// The `OpLoopMerge` lifted off the split block, replayed by [`Self::branch_prefix_to`]. It has
    /// to be the instruction before the terminator, so it cannot go into `prefix` until the
    /// lowering has finished appending its own setup.
    pending_loop_merge: Option<Instruction>,
    /// How many leading `OpVariable`s the split block had when it was opened. A lowering may
    /// allocate more entry-block scratch while the split is open -- AGX emask allocates its load
    /// destination that way -- and `prefix` is a snapshot taken before that, so `finish` has to
    /// carry the new ones across instead of writing the stale snapshot back over them.
    leading_variables: usize,
}

impl CallSiteSplit {
    /// Open the block containing `inst` at `block`, taking the call out of it.
    ///
    /// `what` names the lowering for the refusal messages, which stay honest refusals: a call site
    /// this cannot split is one the lowering must not emit for.
    pub(in crate::passes) fn open(
        ctx: &mut crate::passes::Ctx,
        entry_idx: usize,
        block: usize,
        inst: usize,
        what: &str,
    ) -> Result<Self, String> {
        let old_label = ctx.module.functions[entry_idx].blocks[block]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
            .ok_or_else(|| format!("{what} appears in a block without a label"))?;
        let old_insts = ctx.module.functions[entry_idx].blocks[block]
            .instructions
            .clone();
        let prefix = old_insts[..inst].to_vec();
        let mut suffix = old_insts[inst + 1..].to_vec();
        if suffix.is_empty()
            || !suffix
                .last()
                .is_some_and(|inst| is_block_terminator(inst.class.opcode))
        {
            return Err(format!(
                "{what} lowering requires the source block to retain its terminator"
            ));
        }
        if prefix
            .last()
            .is_some_and(|inst| matches!(inst.class.opcode, Op::SelectionMerge | Op::LoopMerge))
        {
            return Err(format!(
                "{what} lowering cannot split between a structured merge and its terminator"
            ));
        }

        // Splitting a loop-header call must not move OpLoopMerge onto the new continuation block:
        // the original label remains the backedge target and therefore remains the loop header.
        // Keep the claim on that label, and turn the former loop-exit conditional into an ordinary
        // selection with a private pass-through merge. This carries CFG ownership through intrinsic
        // lowering instead of asking a later validator-triggered prune to erase the malformed loop.
        let loop_merge = suffix
            .iter()
            .position(|inst| inst.class.opcode == Op::LoopMerge)
            .map(|index| suffix.remove(index));
        let mut loop_exit_passthrough = None;
        if let Some(loop_merge) = &loop_merge {
            let merge_target = loop_merge
                .operands
                .first()
                .and_then(|operand| match operand {
                    Operand::IdRef(target) => Some(*target),
                    _ => None,
                })
                .ok_or_else(|| format!("{what} loop split found a malformed OpLoopMerge"))?;
            let terminator = suffix
                .last_mut()
                .ok_or_else(|| format!("{what} loop split lost its terminator"))?;
            match terminator.class.opcode {
                Op::Branch => {}
                Op::BranchConditional => {
                    let exits_at_merge = terminator
                        .operands
                        .iter()
                        .skip(1)
                        .any(|operand| *operand == Operand::IdRef(merge_target));
                    if !exits_at_merge {
                        return Err(format!(
                            "{what} loop-header conditional does not target its loop merge"
                        ));
                    }
                    let private_merge = ctx.module.fresh_id();
                    for operand in terminator.operands.iter_mut().skip(1) {
                        if *operand == Operand::IdRef(merge_target) {
                            *operand = Operand::IdRef(private_merge);
                        }
                    }
                    let terminator_index = suffix.len() - 1;
                    suffix.insert(
                        terminator_index,
                        Instruction::new(
                            Op::SelectionMerge,
                            None,
                            None,
                            vec![
                                Operand::IdRef(private_merge),
                                Operand::SelectionControl(spirv::SelectionControl::NONE),
                            ],
                        ),
                    );
                    loop_exit_passthrough = Some((private_merge, merge_target));
                }
                _ => {
                    return Err(format!(
                        "{what} lowering cannot split this loop-header terminator honestly"
                    ));
                }
            }
        }
        let successors = terminator_successors(suffix.last().expect("suffix terminator"));
        Ok(Self {
            prefix,
            suffix,
            block,
            old_label,
            continuation: None,
            successors,
            loop_exit_passthrough,
            pending_loop_merge: loop_merge,
            leading_variables: leading_variable_count(&old_insts),
        })
    }

    /// Close the opened block by branching to `target`, replaying a lifted `OpLoopMerge` first.
    ///
    /// Call this after appending whatever setup the lowering needs to `prefix`: `OpLoopMerge` is
    /// required to sit immediately before its block's terminator, so it can only be replaced once
    /// nothing more will be appended.
    pub(in crate::passes) fn branch_prefix_to(&mut self, target: Word) {
        if let Some(loop_merge) = self.pending_loop_merge.take() {
            self.prefix.push(loop_merge);
        }
        self.prefix.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(target)],
        ));
    }

    /// Label of the block the call was in, which the prefix keeps. A lowering whose first built
    /// block is an `OpPhi` target needs it: the phi's incoming edge still comes from this label.
    pub(in crate::passes) fn entry_label(&self) -> Word {
        self.old_label
    }

    /// Label of the block the rest of the original block becomes, allocated on first ask so a
    /// lowering can branch to it from the blocks it builds. Allocating lazily also lets a lowering
    /// keep the id order it had before it used this helper.
    pub(in crate::passes) fn continuation(&mut self, ctx: &mut crate::passes::Ctx) -> Word {
        *self
            .continuation
            .get_or_insert_with(|| ctx.module.fresh_id())
    }

    /// Splice `blocks` between the opened block and its continuation, and repoint successor phis.
    ///
    /// `blocks` must already branch to [`Self::continuation`] wherever the lowering is done; this
    /// only places them and repairs the phi predecessors the split invalidated.
    pub(in crate::passes) fn finish(
        mut self,
        ctx: &mut crate::passes::Ctx,
        entry_idx: usize,
        mut blocks: Vec<Block>,
    ) {
        let continuation = self.continuation(ctx);
        let Self {
            mut prefix,
            suffix,
            block,
            old_label,
            successors,
            loop_exit_passthrough,
            leading_variables,
            ..
        } = self;
        // A lowering is free to allocate entry-block `OpVariable` scratch between `open` and
        // `finish` -- AGX emask allocates its per-lane load destination that way -- and when the
        // call site is itself in the entry block that allocation lands in the very block `prefix`
        // is a stale snapshot of. Writing the snapshot back would silently drop the variable and
        // leave every reference to it dangling, so carry across whatever grew on the leading
        // `OpVariable` run instead.
        let current = &ctx.module.functions[entry_idx].blocks[block].instructions;
        let grown = leading_variable_count(current);
        if grown > leading_variables {
            let added = current[leading_variables..grown].to_vec();
            prefix.splice(leading_variables..leading_variables, added);
        }
        blocks.push(labelled_block(continuation, suffix));
        if let Some((private_merge, merge_target)) = loop_exit_passthrough {
            blocks.push(labelled_block(
                private_merge,
                vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(merge_target)],
                )],
            ));
        }
        ctx.module.functions[entry_idx].blocks[block].instructions = prefix;
        ctx.module.functions[entry_idx]
            .blocks
            .splice(block + 1..block + 1, blocks);
        // The blocks that used to be reached from the old label are now reached from the
        // continuation -- except the loop merge, which the private pass-through reaches instead.
        if let Some((private_merge, merge_target)) = loop_exit_passthrough {
            let merge_successor = HashSet::from([merge_target]);
            rewrite_successor_phi_predecessors(
                &mut ctx.module.functions[entry_idx],
                &merge_successor,
                old_label,
                private_merge,
            );
            let ordinary_successors = successors
                .difference(&merge_successor)
                .copied()
                .collect::<HashSet<_>>();
            rewrite_successor_phi_predecessors(
                &mut ctx.module.functions[entry_idx],
                &ordinary_successors,
                old_label,
                continuation,
            );
        } else {
            rewrite_successor_phi_predecessors(
                &mut ctx.module.functions[entry_idx],
                &successors,
                old_label,
                continuation,
            );
        }
    }
}

pub(in crate::passes) fn labelled_block(label: Word, instructions: Vec<Instruction>) -> Block {
    Block {
        label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
        instructions,
    }
}

pub(in crate::passes) use crate::spirv_module::is_block_terminator;

/// How many `OpVariable`s a block opens with. Function-storage variables have to be the first
/// instructions of a function's first block, so this run is where any of them live.
fn leading_variable_count(instructions: &[Instruction]) -> usize {
    instructions
        .iter()
        .position(|inst| inst.class.opcode != Op::Variable)
        .unwrap_or(instructions.len())
}

fn terminator_successors(inst: &Instruction) -> HashSet<Word> {
    let mut out = HashSet::new();
    match inst.class.opcode {
        Op::Branch => {
            if let Some(Operand::IdRef(label)) = inst.operands.first() {
                out.insert(*label);
            }
        }
        Op::BranchConditional => {
            for operand in inst.operands.iter().skip(1).take(2) {
                if let Operand::IdRef(label) = operand {
                    out.insert(*label);
                }
            }
        }
        Op::Switch => {
            for operand in inst.operands.iter().skip(1) {
                if let Operand::IdRef(label) = operand {
                    out.insert(*label);
                }
            }
        }
        _ => {}
    }
    out
}

fn rewrite_successor_phi_predecessors(
    function: &mut Function,
    successors: &HashSet<Word>,
    old_label: Word,
    new_label: Word,
) {
    if successors.is_empty() {
        return;
    }
    for block in &mut function.blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        if !successors.contains(&label) {
            continue;
        }
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::Phi {
                break;
            }
            for pair in inst.operands.chunks_mut(2) {
                if pair.len() == 2 && pair[1] == Operand::IdRef(old_label) {
                    pair[1] = Operand::IdRef(new_label);
                }
            }
        }
    }
}
