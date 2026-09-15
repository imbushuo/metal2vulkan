//! Passes-side SPIR-V (`crate::spirv_module::Block`, `Word` label) CFG primitives.
//!
//! The final passes walk the same retained owned-`Block` control-flow graph the native emitter
//! does. The native side has its own equivalents in `native::cfg::graph`; these cannot be shared
//! because the passes layer must not depend on `native` (that would invert the ownership
//! direction). This is the single
//! passes-side home for the Word-layer successor scan, replacing the byte-identical
//! copies formerly open-coded in `inline/mod.rs` and `lower/access.rs`.

use crate::spirv_module::Block;
use spirv::Word;
use std::collections::HashMap;

pub(in crate::passes) use crate::spirv_module::block_successors;

pub(in crate::passes) use crate::spirv_module::block_successors_by_label;

/// Block dominance of one owned SPIR-V function body, indexed by block position.
///
/// The passes layer needs the same question the owned-CFG check asks — does the block that defines
/// a value dominate the block that uses it — whenever it substitutes an existing SSA value for an
/// operand it did not find in place. Built from [`block_successors`] over the positional block
/// order, so `0` is the entry block, and answered through the crate's single dominator computation
/// rather than a second copy of it.
pub(in crate::passes) struct BlockDominance {
    reachable: Vec<bool>,
    intervals: Vec<Option<(usize, usize)>>,
}

impl BlockDominance {
    pub(in crate::passes) fn of(blocks: &[Block]) -> Self {
        let positions: HashMap<Word, usize> = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| Some((block.label.as_ref()?.result_id?, index)))
            .collect();
        let successors = blocks
            .iter()
            .map(|block| {
                block_successors(block)
                    .into_iter()
                    .filter_map(|label| positions.get(&label).copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let predecessors = crate::dominators::build_predecessors(&successors);
        let (reachable, intervals, _) = crate::dominators::dominance(&successors, &predecessors);
        Self {
            reachable,
            intervals,
        }
    }

    /// Whether `dominator` dominates `block`. An unreachable use is dominated by nothing, which is
    /// the conservative answer: a substitution there cannot be justified by dominance either.
    pub(in crate::passes) fn dominates(&self, dominator: usize, block: usize) -> bool {
        if !self.reachable.get(block).copied().unwrap_or(false) {
            return false;
        }
        crate::dominators::dominates_interval(&self.intervals, dominator, block)
    }
}
