//! Pre-submission loop-iteration budget for translated SPIR-V.
//!
//! A committed GPU command buffer cannot be cancelled. Once a compute kernel enters an unbounded
//! loop it pins the GPU until the machine is rebooted; killing the submitting CPU process does not
//! stop it, and on macOS a saturated GPU starves WindowServer past its 40-second watchdog checkin,
//! which kills the user's login session.
//!
//! The corpus declares `execution_safety = loop_free | authored_bounded`, but that bound is prose
//! attached to the *AIR*. Nothing re-checks it after translation, so a bound the translator drops
//! becomes a live wedge on the candidate (MoltenVK) side only: Metal runs the original AIR and
//! finishes, MoltenVK runs our SPIR-V and never signals its fence.
//!
//! Proving finiteness is undecidable, so this pass does not try. It makes loops finite by
//! construction: each `OpLoopMerge` header gets a private counter, incremented once per iteration,
//! and the header is steered out of the loop once the counter reaches the budget. A run that trips
//! the budget is not a valid byte-comparison — the caller is expected to treat it as budget-exceeded
//! rather than as match/mismatch — but it cannot wedge the machine.
//!
//! # Why the existing exit edge is reused
//!
//! The obvious transform — branch from the header straight to the declared merge block when the
//! budget is spent — introduces a CFG edge that did not exist before, and that edge breaks SSA.
//! Values defined inside the loop dominate the merge block only because every path to the merge ran
//! through the body; a header-to-merge bypass destroys that, and `spirv-val` rejects the result with
//! "definition does not dominate its parent". Repairing it means full LCSSA reconstruction.
//!
//! So the common case adds no edges at all. A counted loop's header already ends in
//! `OpBranchConditional %continue_test %body %merge`, and folding the budget into `%continue_test`
//! forces the loop out through the exit it already had, leaving the CFG — and therefore every
//! dominance relation and every phi — exactly as the translator emitted it.
//!
//! A header with no edge to its own merge block cannot be steered that way — and that is the shape
//! our relooper emits, `OpLoopMerge` followed by a plain `OpBranch` into the body with every break
//! living inside it. Those exhaust their budget by **returning from the function** instead. A return
//! adds no edge to the merge block, so the merge keeps exactly the predecessors the translator gave
//! it and no phi anywhere needs repair. The invocation abandons its output, which shows up
//! downstream as a mismatch rather than as a hung queue.

use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Function, Instruction};
use spirv::{Op, StorageClass, Word};
use std::collections::HashMap;

/// Default iterations allowed per loop header before the budget forces an exit.
///
/// The budget has to clear every authored corpus loop — these are unit-test-scale kernels, so their
/// trip counts are in the hundreds — while still bounding total GPU occupancy on a runaway one. The
/// ceiling that matters is WindowServer's 40-second watchdog: the cost of a trip is the budget times
/// the thread count, so a budget in the tens of thousands keeps even a wide dispatch under a second,
/// where 2^20 could run into seconds of stutter before it gave up.
pub const DEFAULT_LOOP_BUDGET: u32 = 1 << 16;

/// Outcome of instrumenting one module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopBudgetReport {
    /// Loops bounded by folding the budget into an exit branch the header already had. The CFG is
    /// unchanged, so these are always safe to dispatch.
    pub loops_bounded_in_place: usize,
    /// Loops bounded by returning from the function, because the header had no exit of its own.
    pub loops_bounded_via_early_return: usize,
    /// Loop headers left alone because their terminator does not branch (`OpReturn`,
    /// `OpUnreachable`, `OpKill`), so they cannot iterate.
    pub loops_skipped: usize,
}

impl LoopBudgetReport {
    /// Whether the module contained any loop at all.
    pub fn had_loops(&self) -> bool {
        self.loops_bounded_in_place + self.loops_bounded_via_early_return + self.loops_skipped > 0
    }

    /// Loops that now carry a budget.
    pub fn loops_instrumented(&self) -> usize {
        self.loops_bounded_in_place + self.loops_bounded_via_early_return
    }

    /// Whether any loop needed a control-flow edge the translator did not emit. Neither form is
    /// expected to disturb SSA, but the added-edge shapes are the ones worth re-validating before a
    /// module reaches a real device.
    pub fn needs_revalidation(&self) -> bool {
        self.loops_bounded_via_early_return > 0
    }
}

/// Bound every loop in `module` to at most `budget` iterations per entry.
pub fn instrument_loop_budget(module: &mut Module, budget: u32) -> LoopBudgetReport {
    let mut report = LoopBudgetReport::default();
    if !module.functions.iter().any(function_has_loop) {
        return report;
    }

    let mut pool = ConstantPool::intern(module, budget);
    for index in 0..module.functions.len() {
        let mut function = std::mem::take(&mut module.functions[index]);
        instrument_function(&mut function, &mut pool, &mut report);
        module.functions[index] = function;
    }
    pool.flush(module);
    module.sync_id_bound_from_instructions();
    report
}

fn function_has_loop(function: &Function) -> bool {
    function
        .blocks
        .iter()
        .any(|block| loop_merge_index(block).is_some())
}

/// Index of a block's `OpLoopMerge`, if it heads a loop.
fn loop_merge_index(block: &Block) -> Option<usize> {
    block
        .instructions
        .iter()
        .position(|inst| inst.class.opcode == Op::LoopMerge)
}

fn block_label(block: &Block) -> Option<Word> {
    block.label.as_ref().and_then(|label| label.result_id)
}

/// Successor block ids of a terminator, in operand order.
fn terminator_targets(inst: &Instruction) -> Vec<Word> {
    let skip = match inst.class.opcode {
        Op::Branch => 0,
        // Both carry a leading non-label id: the condition, and the switch selector.
        Op::BranchConditional | Op::Switch => 1,
        _ => return Vec::new(),
    };
    inst.operands
        .iter()
        .skip(skip)
        .filter_map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect()
}

/// How the budget steers a given loop header out of its loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExitForm {
    /// `OpBranchConditional %c %body %merge`: the loop continues on true, so the budget is ANDed in.
    ContinueOnTrue,
    /// `OpBranchConditional %c %merge %body`: the loop exits on true, so an exhausted budget is ORed in.
    ExitOnTrue,
    /// `OpBranch %body`: the header has one successor, so the budget picks between it and a return.
    ReturnFromBranch,
    /// Any other header without an edge to its own merge block. The original terminator moves into a
    /// guard block so the budget can pick between the guard and a return.
    ReturnViaGuard,
}

/// Interns the scalar types, constants and undef values the instrumentation needs, creating only
/// what the module does not already declare.
struct ConstantPool {
    next_id: Word,
    uint: Word,
    bool_ty: Word,
    ptr_function_uint: Word,
    const_zero: Word,
    const_one: Word,
    const_budget: Word,
    void_ty: Option<Word>,
    undef_by_type: HashMap<Word, Word>,
    additions: Vec<Instruction>,
}

impl ConstantPool {
    fn intern(module: &mut Module, budget: u32) -> Self {
        let mut pool = Self {
            next_id: module.id_bound().max(1),
            uint: 0,
            bool_ty: 0,
            ptr_function_uint: 0,
            const_zero: 0,
            const_one: 0,
            const_budget: 0,
            void_ty: module
                .types_global_values
                .iter()
                .find(|inst| inst.class.opcode == Op::TypeVoid)
                .and_then(|inst| inst.result_id),
            undef_by_type: HashMap::new(),
            additions: Vec::new(),
        };
        pool.uint = pool.find_or_create(module, Op::TypeInt, None, || {
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
        });
        pool.bool_ty = pool.find_or_create(module, Op::TypeBool, None, Vec::new);
        let uint = pool.uint;
        pool.ptr_function_uint = pool.find_or_create(module, Op::TypePointer, None, || {
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(uint),
            ]
        });
        pool.const_zero = pool.constant(module, 0);
        pool.const_one = pool.constant(module, 1);
        pool.const_budget = pool.constant(module, budget);
        pool
    }

    fn fresh_id(&mut self) -> Word {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// An existing declaration matching `(opcode, result_type, operands)`, or a freshly appended one.
    fn find_or_create(
        &mut self,
        module: &Module,
        opcode: Op,
        result_type: Option<Word>,
        operands: impl FnOnce() -> Vec<Operand>,
    ) -> Word {
        let operands = operands();
        let matches = |inst: &Instruction| {
            inst.class.opcode == opcode
                && inst.result_type == result_type
                && inst.operands == operands
        };
        if let Some(id) = module
            .types_global_values
            .iter()
            .chain(&self.additions)
            .find(|inst| matches(inst))
            .and_then(|inst| inst.result_id)
        {
            return id;
        }
        let id = self.fresh_id();
        self.additions
            .push(Instruction::new(opcode, result_type, Some(id), operands));
        id
    }

    fn constant(&mut self, module: &Module, value: u32) -> Word {
        let uint = self.uint;
        self.find_or_create(module, Op::Constant, Some(uint), || {
            vec![Operand::LiteralBit32(value)]
        })
    }

    /// An `OpUndef` of `result_type`, for phi incomings on a forced-exit edge.
    fn undef(&mut self, result_type: Word) -> Word {
        if let Some(id) = self.undef_by_type.get(&result_type) {
            return *id;
        }
        let id = self.fresh_id();
        self.additions.push(Instruction::new(
            Op::Undef,
            Some(result_type),
            Some(id),
            Vec::new(),
        ));
        self.undef_by_type.insert(result_type, id);
        id
    }

    fn flush(self, module: &mut Module) {
        module.types_global_values.extend(self.additions);
        module.set_id_bound(self.next_id);
    }
}

struct LoopPlan {
    header_id: Word,
    continue_id: Word,
    counter: Word,
    guard_id: Word,
    check_id: Word,
    exit_block_id: Word,
    exit: ExitForm,
}

fn instrument_function(
    function: &mut Function,
    pool: &mut ConstantPool,
    report: &mut LoopBudgetReport,
) {
    // Plans are collected up front so that inserting a guard block cannot invalidate the indices.
    let mut plans = Vec::new();
    for block in &function.blocks {
        let Some(merge_index) = loop_merge_index(block) else {
            continue;
        };
        let Some(header_id) = block_label(block) else {
            continue;
        };
        let loop_merge = &block.instructions[merge_index];
        let (Some(Operand::IdRef(merge_id)), Some(Operand::IdRef(continue_id))) =
            (loop_merge.operands.first(), loop_merge.operands.get(1))
        else {
            continue;
        };
        let Some(terminator) = block.instructions.last() else {
            continue;
        };
        let Some(exit) = classify_exit(terminator, *merge_id) else {
            // A header that returns or is unreachable cannot iterate.
            report.loops_skipped += 1;
            continue;
        };
        plans.push(LoopPlan {
            header_id,
            continue_id: *continue_id,
            counter: pool.fresh_id(),
            guard_id: pool.fresh_id(),
            check_id: pool.fresh_id(),
            exit_block_id: pool.fresh_id(),
            exit,
        });
    }

    for plan in plans {
        apply_plan(function, &plan, pool);
        match plan.exit {
            ExitForm::ContinueOnTrue | ExitForm::ExitOnTrue => report.loops_bounded_in_place += 1,
            ExitForm::ReturnFromBranch | ExitForm::ReturnViaGuard => {
                report.loops_bounded_via_early_return += 1
            }
        }
    }
}

/// Decide how a header can be steered out of its loop, or `None` if it cannot iterate at all.
fn classify_exit(terminator: &Instruction, merge_id: Word) -> Option<ExitForm> {
    match terminator.class.opcode {
        Op::BranchConditional => {
            let targets = terminator_targets(terminator);
            match (targets.first(), targets.get(1)) {
                // Reusing an exit the header already has keeps the CFG, and therefore SSA, intact.
                (Some(_), Some(false_target)) if *false_target == merge_id => {
                    Some(ExitForm::ContinueOnTrue)
                }
                (Some(true_target), Some(_)) if *true_target == merge_id => {
                    Some(ExitForm::ExitOnTrue)
                }
                _ => Some(ExitForm::ReturnViaGuard),
            }
        }
        Op::Branch => Some(ExitForm::ReturnFromBranch),
        Op::Switch => Some(ExitForm::ReturnViaGuard),
        _ => None,
    }
}

fn apply_plan(function: &mut Function, plan: &LoopPlan, pool: &mut ConstantPool) {
    declare_counter(function, plan, pool);
    reset_counter_in_preheaders(function, plan, pool);

    let Some(header_index) = function
        .blocks
        .iter()
        .position(|block| block_label(block) == Some(plan.header_id))
    else {
        return;
    };

    // The counter is read, bumped and tested ahead of OpLoopMerge, which must stay immediately
    // before the terminator.
    let in_budget = bump_counter(&mut function.blocks[header_index], plan, pool);

    match plan.exit {
        ExitForm::ContinueOnTrue | ExitForm::ExitOnTrue => {
            fold_budget_into_condition(&mut function.blocks[header_index], plan, pool, in_budget);
        }
        ExitForm::ReturnFromBranch | ExitForm::ReturnViaGuard => {
            divert_to_return(function, header_index, plan, pool, in_budget);
        }
    }
}

/// Emit the per-iteration counter bump into the header and return the id of the "still in budget"
/// boolean. Instructions land before `OpLoopMerge`, which must remain adjacent to the terminator.
fn bump_counter(header: &mut Block, plan: &LoopPlan, pool: &mut ConstantPool) -> Word {
    let Some(merge_index) = loop_merge_index(header) else {
        return pool.const_zero;
    };
    let current = pool.fresh_id();
    let next = pool.fresh_id();
    let in_budget = pool.fresh_id();
    let emitted = [
        Instruction::new(
            Op::Load,
            Some(pool.uint),
            Some(current),
            vec![Operand::IdRef(plan.counter)],
        ),
        Instruction::new(
            Op::IAdd,
            Some(pool.uint),
            Some(next),
            vec![Operand::IdRef(current), Operand::IdRef(pool.const_one)],
        ),
        Instruction::new(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(plan.counter), Operand::IdRef(next)],
        ),
        Instruction::new(
            Op::ULessThan,
            Some(pool.bool_ty),
            Some(in_budget),
            vec![Operand::IdRef(current), Operand::IdRef(pool.const_budget)],
        ),
    ];
    for (offset, inst) in emitted.into_iter().enumerate() {
        header.instructions.insert(merge_index + offset, inst);
    }
    in_budget
}

/// Rewrite the header's branch condition so an exhausted budget takes the exit the loop already had.
/// No edge is added, so dominance and every phi in the function are untouched.
fn fold_budget_into_condition(
    header: &mut Block,
    plan: &LoopPlan,
    pool: &mut ConstantPool,
    in_budget: Word,
) {
    let Some(merge_index) = loop_merge_index(header) else {
        return;
    };
    let Some(original) = header
        .instructions
        .last()
        .and_then(|inst| inst.operands.first().cloned())
    else {
        return;
    };
    let Operand::IdRef(original_condition) = original else {
        return;
    };

    let combined = pool.fresh_id();
    let mut emitted = Vec::new();
    match plan.exit {
        // Continue only while the original test holds AND budget remains.
        ExitForm::ContinueOnTrue => emitted.push(Instruction::new(
            Op::LogicalAnd,
            Some(pool.bool_ty),
            Some(combined),
            vec![
                Operand::IdRef(original_condition),
                Operand::IdRef(in_budget),
            ],
        )),
        // Exit as soon as the original test holds OR the budget is spent.
        ExitForm::ExitOnTrue => {
            let exhausted = pool.fresh_id();
            emitted.push(Instruction::new(
                Op::LogicalNot,
                Some(pool.bool_ty),
                Some(exhausted),
                vec![Operand::IdRef(in_budget)],
            ));
            emitted.push(Instruction::new(
                Op::LogicalOr,
                Some(pool.bool_ty),
                Some(combined),
                vec![
                    Operand::IdRef(original_condition),
                    Operand::IdRef(exhausted),
                ],
            ));
        }
        ExitForm::ReturnFromBranch | ExitForm::ReturnViaGuard => return,
    }
    for (offset, inst) in emitted.into_iter().enumerate() {
        header.instructions.insert(merge_index + offset, inst);
    }
    if let Some(terminator) = header.instructions.last_mut() {
        terminator.operands[0] = Operand::IdRef(combined);
    }
}

/// Bound a header that has no exit of its own by returning from the function when the budget is
/// spent.
///
/// The return is the whole point: branching to the loop's merge block instead would give the merge a
/// predecessor that bypasses the body, and every value the body defines would stop dominating its
/// uses at and after the merge. A return leaves the merge's predecessors exactly as the translator
/// emitted them, so no phi in the function needs repair.
///
/// The test cannot live in the header itself. A loop header's conditional branch may only choose
/// between the body and the loop's own merge block; selecting between the body and a return block
/// there is an unstructured selection. So the header branches unconditionally into a check block,
/// and the check block is a well-formed selection whose merge is the block that continues the loop.
///
/// ```text
///   header:  <counter bump>  OpLoopMerge %merge %continue    OpBranch %check
///   check:   OpSelectionMerge %stay       OpBranchConditional %in_budget %stay %exit
///   exit:    OpReturn
///   stay:    the loop body, or a guard holding the header's original terminator
/// ```
fn divert_to_return(
    function: &mut Function,
    header_index: usize,
    plan: &LoopPlan,
    pool: &mut ConstantPool,
    in_budget: Word,
) {
    // An OpBranch header already has exactly one successor to continue into; anything else has to
    // keep its original terminator, which is what the guard block is for.
    let guard_terminator = match plan.exit {
        ExitForm::ReturnFromBranch => None,
        _ => function.blocks[header_index].instructions.pop(),
    };

    let stay = match &guard_terminator {
        Some(_) => plan.guard_id,
        None => {
            let Some(body) = function.blocks[header_index]
                .instructions
                .last()
                .and_then(|terminator| terminator_targets(terminator).first().copied())
            else {
                return;
            };
            function.blocks[header_index].instructions.pop();
            body
        }
    };

    function.blocks[header_index]
        .instructions
        .push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(plan.check_id)],
        ));

    let mut check = Block::new();
    check.label = Some(Instruction::new(
        Op::Label,
        None,
        Some(plan.check_id),
        Vec::new(),
    ));
    check.instructions.push(Instruction::new(
        Op::SelectionMerge,
        None,
        None,
        vec![
            Operand::IdRef(stay),
            Operand::SelectionControl(spirv::SelectionControl::NONE),
        ],
    ));
    check.instructions.push(Instruction::new(
        Op::BranchConditional,
        None,
        None,
        vec![
            Operand::IdRef(in_budget),
            Operand::IdRef(stay),
            Operand::IdRef(plan.exit_block_id),
        ],
    ));

    let exit = build_return_block(function, plan.exit_block_id, pool);

    // Structured block ordering: the selection's blocks precede its merge, which is `stay`.
    let mut insert_at = header_index + 1;
    for block in [check, exit] {
        function.blocks.insert(insert_at, block);
        insert_at += 1;
    }

    // Whichever block now continues the loop is entered from the check, not from the header.
    match guard_terminator {
        Some(original) => {
            let original_targets = terminator_targets(&original);
            let mut guard = Block::new();
            guard.label = Some(Instruction::new(
                Op::Label,
                None,
                Some(plan.guard_id),
                Vec::new(),
            ));
            guard.instructions.push(original);
            function.blocks.insert(insert_at, guard);
            for target in &original_targets {
                repoint_phi_predecessor(function, *target, plan.header_id, plan.guard_id);
            }
        }
        None => repoint_phi_predecessor(function, stay, plan.header_id, plan.check_id),
    }
}

/// A block that leaves the function, matching its declared return type.
fn build_return_block(function: &Function, label: Word, pool: &mut ConstantPool) -> Block {
    let return_type = function.def.as_ref().and_then(|def| def.result_type);
    let terminator = match return_type {
        Some(ty) if Some(ty) != pool.void_ty => {
            let undef = pool.undef(ty);
            Instruction::new(Op::ReturnValue, None, None, vec![Operand::IdRef(undef)])
        }
        _ => Instruction::new(Op::Return, None, None, Vec::new()),
    };
    let mut block = Block::new();
    block.label = Some(Instruction::new(Op::Label, None, Some(label), Vec::new()));
    block.instructions.push(terminator);
    block
}

/// Add the counter as a function-scope variable, zero-initialized so a loop with no identifiable
/// preheader still starts from a defined value.
fn declare_counter(function: &mut Function, plan: &LoopPlan, pool: &ConstantPool) {
    let Some(entry) = function.blocks.first_mut() else {
        return;
    };
    let insertion = entry
        .instructions
        .iter()
        .position(|inst| inst.class.opcode != Op::Variable)
        .unwrap_or(entry.instructions.len());
    entry.instructions.insert(
        insertion,
        Instruction::new(
            Op::Variable,
            Some(pool.ptr_function_uint),
            Some(plan.counter),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(pool.const_zero),
            ],
        ),
    );
}

/// Store zero on every edge that enters the loop from outside, so re-entering a nested loop starts a
/// fresh budget rather than inheriting the outer loop's count.
fn reset_counter_in_preheaders(function: &mut Function, plan: &LoopPlan, pool: &ConstantPool) {
    for block in &mut function.blocks {
        let Some(label) = block_label(block) else {
            continue;
        };
        if label == plan.header_id || label == plan.continue_id {
            continue;
        }
        let Some(terminator) = block.instructions.last() else {
            continue;
        };
        if !terminator_targets(terminator).contains(&plan.header_id) {
            continue;
        }
        // A structured merge instruction must stay adjacent to the terminator it heads.
        let mut insertion = block.instructions.len() - 1;
        if insertion > 0
            && matches!(
                block.instructions[insertion - 1].class.opcode,
                Op::SelectionMerge | Op::LoopMerge
            )
        {
            insertion -= 1;
        }
        block.instructions.insert(
            insertion,
            Instruction::new(
                Op::Store,
                None,
                None,
                vec![
                    Operand::IdRef(plan.counter),
                    Operand::IdRef(pool.const_zero),
                ],
            ),
        );
    }
}

/// Rewrite `(value, from)` phi pairs in `target` to name `to` instead.
fn repoint_phi_predecessor(function: &mut Function, target: Word, from: Word, to: Word) {
    let Some(block) = function
        .blocks
        .iter_mut()
        .find(|block| block_label(block) == Some(target))
    else {
        return;
    };
    for inst in &mut block.instructions {
        if inst.class.opcode != Op::Phi {
            break;
        }
        for pair in inst.operands.chunks_mut(2) {
            if let [_, Operand::IdRef(predecessor)] = pair {
                if *predecessor == from {
                    *predecessor = to;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MERGE: Word = 20;
    const CONTINUE: Word = 30;
    const BODY: Word = 40;

    fn terminator(opcode: Op, operands: Vec<Operand>) -> Instruction {
        Instruction::new(opcode, None, None, operands)
    }

    #[test]
    fn a_header_that_exits_on_false_folds_the_budget_with_and() {
        let inst = terminator(
            Op::BranchConditional,
            vec![
                Operand::IdRef(7),
                Operand::IdRef(BODY),
                Operand::IdRef(MERGE),
            ],
        );
        assert_eq!(classify_exit(&inst, MERGE), Some(ExitForm::ContinueOnTrue));
    }

    #[test]
    fn a_header_that_exits_on_true_folds_the_budget_with_or() {
        let inst = terminator(
            Op::BranchConditional,
            vec![
                Operand::IdRef(7),
                Operand::IdRef(MERGE),
                Operand::IdRef(BODY),
            ],
        );
        assert_eq!(classify_exit(&inst, MERGE), Some(ExitForm::ExitOnTrue));
    }

    /// A `while (true)` header reaches its merge only through breaks in the body, so the budget has
    /// no existing edge to ride out on and has to leave the function instead.
    #[test]
    fn an_unconditional_header_exits_by_returning() {
        let inst = terminator(Op::Branch, vec![Operand::IdRef(BODY)]);
        assert_eq!(
            classify_exit(&inst, MERGE),
            Some(ExitForm::ReturnFromBranch)
        );
    }

    #[test]
    fn a_conditional_header_with_no_exit_returns_through_a_guard() {
        let inst = terminator(
            Op::BranchConditional,
            vec![
                Operand::IdRef(7),
                Operand::IdRef(BODY),
                Operand::IdRef(BODY + 1),
            ],
        );
        assert_eq!(classify_exit(&inst, MERGE), Some(ExitForm::ReturnViaGuard));
    }

    #[test]
    fn a_header_that_returns_cannot_iterate() {
        assert_eq!(
            classify_exit(&terminator(Op::Return, Vec::new()), MERGE),
            None
        );
        assert_eq!(
            classify_exit(&terminator(Op::Unreachable, Vec::new()), MERGE),
            None
        );
    }

    #[test]
    fn branch_conditional_targets_skip_the_condition_operand() {
        let inst = terminator(
            Op::BranchConditional,
            vec![
                Operand::IdRef(7),
                Operand::IdRef(BODY),
                Operand::IdRef(MERGE),
            ],
        );
        assert_eq!(terminator_targets(&inst), vec![BODY, MERGE]);
    }

    #[test]
    fn switch_targets_skip_the_selector_operand() {
        let inst = terminator(
            Op::Switch,
            vec![
                Operand::IdRef(7),
                Operand::IdRef(MERGE),
                Operand::LiteralBit32(1),
                Operand::IdRef(BODY),
            ],
        );
        assert_eq!(terminator_targets(&inst), vec![MERGE, BODY]);
    }

    #[test]
    fn a_non_iterating_header_is_counted_as_skipped() {
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(10), Vec::new()));
        block.instructions.push(terminator(
            Op::LoopMerge,
            vec![
                Operand::IdRef(MERGE),
                Operand::IdRef(CONTINUE),
                Operand::LoopControl(spirv::LoopControl::NONE),
            ],
        ));
        block.instructions.push(terminator(Op::Return, Vec::new()));
        let mut function = Function::new();
        function.blocks.push(block);

        let mut module = Module::new();
        let mut pool = ConstantPool::intern(&mut module, 16);
        let mut report = LoopBudgetReport::default();
        instrument_function(&mut function, &mut pool, &mut report);

        assert_eq!(report.loops_instrumented(), 0);
        assert_eq!(report.loops_skipped, 1);
        assert!(report.had_loops());
        assert!(!report.needs_revalidation());
    }
}
