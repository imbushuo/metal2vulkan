//! Byte-neutral responsibility split of the former monolith; see the parent module.

use crate::spirv_module::Instruction;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

/// SSA values in `f` proven `!= 0` for every executing invocation. Seeded from a `NumWorkgroups`
/// load (each grid dimension is `>= 1`), grown through the value-preserving ops the dispatch-index
/// arithmetic uses: component extracts, integer conversions, and a product of two nonzero terms. A
/// nonzero MODULE constant is also a seed. (Narrowing conversions and wide products are treated as
/// nonzero-preserving under the kernel's own grid-fits-its-index-type contract — the same contract
/// the AGX kernel encodes by truncating the grid size to `ushort`; a grid that overflowed that type
/// would already miscompile in Apple's own code.)
pub(in crate::native) fn compute_nonzero(
    f: &crate::spirv_module::Function,
    consts: &HashMap<Word, i128>,
    numworkgroups: &HashSet<Word>,
) -> HashSet<Word> {
    let mut nz: HashSet<Word> = consts
        .iter()
        .filter(|(_, v)| **v != 0)
        .map(|(k, _)| *k)
        .collect();
    let is_nz = |nz: &HashSet<Word>, op: Option<&Operand>| -> bool {
        matches!(op, Some(Operand::IdRef(id)) if nz.contains(id))
    };
    loop {
        let mut changed = false;
        for b in &f.blocks {
            for inst in &b.instructions {
                let Some(rid) = inst.result_id else { continue };
                if nz.contains(&rid) {
                    continue;
                }
                let seed = match inst.class.opcode {
                    Op::Load => is_nz_load(inst, numworkgroups),
                    Op::CompositeExtract | Op::VectorExtractDynamic => {
                        is_nz(&nz, inst.operands.first())
                    }
                    Op::UConvert | Op::SConvert | Op::CopyObject | Op::Bitcast => {
                        is_nz(&nz, inst.operands.first())
                    }
                    Op::IMul => {
                        is_nz(&nz, inst.operands.first()) && is_nz(&nz, inst.operands.get(1))
                    }
                    _ => false,
                };
                if seed {
                    nz.insert(rid);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    nz
}

/// Whether a `Load` reads a `NumWorkgroups` builtin variable (directly).
pub(in crate::native) fn is_nz_load(inst: &Instruction, numworkgroups: &HashSet<Word>) -> bool {
    matches!(inst.operands.first(), Some(Operand::IdRef(p)) if numworkgroups.contains(p))
}

/// Resolve `id` to `(base, offset)` where `id == base + offset (mod 2^width)`, threading through
/// `IAdd`/`ISub` whose other operand is a known module constant. Non-affine ids are their own base
/// with offset 0.
pub(in crate::native) fn affine(
    id: Word,
    def: &HashMap<Word, &Instruction>,
    consts: &HashMap<Word, i128>,
    widths: &HashMap<Word, u32>,
) -> (Word, u128) {
    let mask = |w: u32| -> u128 {
        if w >= 128 {
            u128::MAX
        } else {
            (1u128 << w) - 1
        }
    };
    let mut base = id;
    let mut off: u128 = 0;
    let mut guard = 0;
    while guard < 64 {
        guard += 1;
        let Some(inst) = def.get(&base) else { break };
        let w = match inst.result_id.and_then(|r| widths.get(&r)) {
            Some(w) => *w,
            None => break,
        };
        let m = mask(w);
        let a = inst.operands.first();
        let b = inst.operands.get(1);
        let cst = |o: Option<&Operand>| -> Option<u128> {
            match o {
                Some(Operand::IdRef(c)) => consts.get(c).map(|v| (*v as u128) & m),
                _ => None,
            }
        };
        match inst.class.opcode {
            Op::IAdd => {
                if let (Some(Operand::IdRef(x)), Some(c)) = (a, cst(b)) {
                    off = off.wrapping_add(c) & m;
                    base = *x;
                } else if let (Some(c), Some(Operand::IdRef(x))) = (cst(a), b) {
                    off = off.wrapping_add(c) & m;
                    base = *x;
                } else {
                    break;
                }
            }
            Op::ISub => {
                if let (Some(Operand::IdRef(x)), Some(c)) = (a, cst(b)) {
                    off = off.wrapping_sub(c) & m;
                    base = *x;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    (base, off)
}

/// Fold grid-stride early-return guards to `true`: an `OpUGreaterThan(X, Y)` where `Y == X - 1`
/// (affine, same base) and `X` is proven nonzero. `X > X-1` holds for every unsigned `X >= 1`, so
/// once the FC-derived work count folds the offset to exactly `-1` this guard is statically taken,
/// which lets branch-folding prune the guarded MXU compute nest. Returns `guard_id -> 1`.
pub(in crate::native) fn nonzero_self_minus_one_guards(
    f: &crate::spirv_module::Function,
    consts: &HashMap<Word, i128>,
    widths: &HashMap<Word, u32>,
    numworkgroups: &HashSet<Word>,
) -> HashMap<Word, i128> {
    if numworkgroups.is_empty() {
        return HashMap::new();
    }
    let nz = compute_nonzero(f, consts, numworkgroups);
    let mut def: HashMap<Word, &Instruction> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.instructions {
            if let Some(r) = inst.result_id {
                def.insert(r, inst);
            }
        }
    }
    let mut out = HashMap::new();
    for b in &f.blocks {
        for inst in &b.instructions {
            if inst.class.opcode != Op::UGreaterThan {
                continue;
            }
            let Some(rid) = inst.result_id else { continue };
            let (Some(Operand::IdRef(x)), Some(Operand::IdRef(y))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            // X must be a proven-nonzero base (offset 0); Y must be exactly X - 1.
            let (bx, ox) = affine(*x, &def, consts, widths);
            let (by, oy) = affine(*y, &def, consts, widths);
            let w = widths.get(x).copied();
            let Some(w) = w else { continue };
            let mask = if w >= 128 {
                u128::MAX
            } else {
                (1u128 << w) - 1
            };
            if bx == by && ox == 0 && oy == mask && nz.contains(&bx) {
                out.insert(rid, 1);
            }
        }
    }
    out
}

/// Unsigned comparison kind for [`ucmp_fold`].
#[derive(Clone, Copy)]
pub(in crate::native) enum Cmp {
    Lt,
    Gt,
    Le,
    Ge,
}

/// Fold an unsigned integer comparison over lattice operands. Beyond the both-constant case, it
/// applies the exact unsigned bound `0 <= x` to fold a comparison with a known-0 operand REGARDLESS
/// of the other (even Bottom): `x < 0` and `0 > x` are always false; `0 <= x` and `x >= 0` are always
/// true. Values are interpreted unsigned (widths <= 64; the lattice stores width-masked non-negative
/// integers).
pub(in crate::native) fn ucmp_fold(x: Option<Lat>, y: Option<Lat>, kind: Cmp) -> Option<Lat> {
    let is_zero = |v: Option<Lat>| matches!(v, Some(Lat::Const(0)));
    match kind {
        Cmp::Lt if is_zero(y) => return Some(Lat::Const(0)), // x < 0 = false
        Cmp::Gt if is_zero(x) => return Some(Lat::Const(0)), // 0 > y = false
        Cmp::Le if is_zero(x) => return Some(Lat::Const(1)), // 0 <= y = true
        Cmp::Ge if is_zero(y) => return Some(Lat::Const(1)), // x >= 0 = true
        _ => {}
    }
    match (x, y) {
        (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => Some(Lat::Bottom),
        (Some(Lat::Const(a)), Some(Lat::Const(b))) => {
            let (a, b) = (a as u128, b as u128);
            let r = match kind {
                Cmp::Lt => a < b,
                Cmp::Gt => a > b,
                Cmp::Le => a <= b,
                Cmp::Ge => a >= b,
            };
            Some(Lat::Const(r as i128))
        }
        _ => None,
    }
}

/// Fold a SIGNED integer comparison over lattice operands. The lattice stores an integer in its
/// type's UNSIGNED representation -- `module_scalar_constants` zero-extends the literal and
/// `IAdd`/`IMul`/`ISub` mask back to the width -- so `-1 : i32` is held as `0xFFFFFFFF` and the bits
/// must be sign-extended from `width` before they mean anything signed. That is why this cannot
/// share `ucmp_fold`'s body, and why an unknown or implausible width refuses to fold rather than
/// guessing 128 and reading every negative operand as a large positive one.
///
/// There is no zero-operand shortcut here. `x >= 0` is a tautology for UNSIGNED `x` only; signed,
/// both `x >= 0` and `x < 0` depend on the value, so a one-constant operand proves nothing.
pub(in crate::native) fn scmp_fold(
    x: Option<Lat>,
    y: Option<Lat>,
    width: Option<u32>,
    kind: Cmp,
) -> Option<Lat> {
    match (x, y) {
        (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => Some(Lat::Bottom),
        (Some(Lat::Const(a)), Some(Lat::Const(b))) => {
            let width = width.filter(|w| (1..=128).contains(w))?;
            let shift = 128 - width;
            let sign_extend = |v: i128| (v << shift) >> shift;
            let (a, b) = (sign_extend(a), sign_extend(b));
            let result = match kind {
                Cmp::Lt => a < b,
                Cmp::Gt => a > b,
                Cmp::Le => a <= b,
                Cmp::Ge => a >= b,
            };
            Some(Lat::Const(result as i128))
        }
        _ => None,
    }
}

/// A value in the constant lattice. Absent-from-the-map = TOP (optimistically undefined). This is
/// the classic SCCP lattice: TOP -> Const -> Bottom, monotone downward, so the fixpoint terminates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::native) enum Lat {
    Const(i128),
    Bottom,
}

/// Meet two lattice values (TOP is represented by `None`).
pub(in crate::native) fn meet(a: Option<Lat>, b: Option<Lat>) -> Option<Lat> {
    match (a, b) {
        (None, x) | (x, None) => x, // TOP ∧ x = x
        (Some(Lat::Bottom), _) | (_, Some(Lat::Bottom)) => Some(Lat::Bottom),
        (Some(Lat::Const(x)), Some(Lat::Const(y))) => {
            Some(if x == y { Lat::Const(x) } else { Lat::Bottom })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lattice holds an integer in its type's UNSIGNED representation, so the WIDTH decides
    /// what a signed comparison means: `0 < 65534` is true read as 32-bit and false read as 16-bit,
    /// where 65534 is -2. Getting this wrong would fold a real branch to the opposite arm, so the
    /// width is not optional -- an absent or implausible one refuses instead of guessing.
    #[test]
    fn signed_comparison_reads_the_operands_at_their_own_width() {
        let c = |v: i128| Some(Lat::Const(v));
        assert_eq!(
            scmp_fold(c(0), c(65534), Some(32), Cmp::Lt),
            Some(Lat::Const(1)),
            "at 32 bits 65534 is 65534"
        );
        assert_eq!(
            scmp_fold(c(0), c(65534), Some(16), Cmp::Lt),
            Some(Lat::Const(0)),
            "at 16 bits 65534 is -2"
        );
        assert_eq!(
            scmp_fold(c(0), c(65534), Some(16), Cmp::Gt),
            Some(Lat::Const(1)),
            "and 0 > -2"
        );
        // 0xFFFFFFFF is -1 at 32 bits and 4294967295 at 64.
        assert_eq!(
            scmp_fold(c(0xFFFF_FFFF), c(0), Some(32), Cmp::Lt),
            Some(Lat::Const(1))
        );
        assert_eq!(
            scmp_fold(c(0xFFFF_FFFF), c(0), Some(64), Cmp::Lt),
            Some(Lat::Const(0))
        );
        // The signed minimum compares below everything else at its width.
        assert_eq!(
            scmp_fold(c(0x8000_0000), c(0x7FFF_FFFF), Some(32), Cmp::Lt),
            Some(Lat::Const(1))
        );
        assert_eq!(
            scmp_fold(c(7), c(7), Some(32), Cmp::Le),
            Some(Lat::Const(1))
        );
        assert_eq!(
            scmp_fold(c(7), c(7), Some(32), Cmp::Ge),
            Some(Lat::Const(1))
        );
    }

    /// Unlike `ucmp_fold` there is no zero shortcut: `x >= 0` is a tautology only for an UNSIGNED
    /// `x`. A lattice value that is still TOP or a width the caller could not determine both leave
    /// the comparison unresolved rather than assuming one.
    #[test]
    fn signed_comparison_refuses_what_it_cannot_prove() {
        let c = |v: i128| Some(Lat::Const(v));
        assert_eq!(
            scmp_fold(None, c(0), Some(32), Cmp::Ge),
            None,
            "x >= 0 is not a signed tautology"
        );
        assert_eq!(scmp_fold(c(0), None, Some(32), Cmp::Lt), None);
        assert_eq!(
            scmp_fold(c(0), c(1), None, Cmp::Lt),
            None,
            "an unknown width refuses"
        );
        assert_eq!(scmp_fold(c(0), c(1), Some(0), Cmp::Lt), None);
        assert_eq!(scmp_fold(c(0), c(1), Some(129), Cmp::Lt), None);
        assert_eq!(
            scmp_fold(Some(Lat::Bottom), c(1), Some(32), Cmp::Lt),
            Some(Lat::Bottom),
            "a poisoned operand poisons the result"
        );
        // The unsigned twin DOES have the zero shortcut, and keeps it.
        assert_eq!(ucmp_fold(None, c(0), Cmp::Ge), Some(Lat::Const(1)));
    }
}
