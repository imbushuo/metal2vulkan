//! Pre-emit AIR lowering for a `llvm.memcpy` whose LENGTH is not a compile-time constant.
//!
//! The emitter copies a constant-length `llvm.memcpy` by unrolling it into that many word or byte
//! accesses (`emitter::body::memcpy_memset`), which needs the count in hand. A dynamic length has
//! no unrolling, and until this pass the call fell through every arm: in a module the emitter
//! otherwise accepted, the copy was silently DROPPED and the bytes it should have written kept
//! their previous value; in the rest it surfaced as an internal-contract failure against a bodiless
//! `llvm.memcpy` declaration. Both are measured — `fastGPUMemcpyKernel` copies its `len % 16` tail
//! this way, and before this pass the authored case
//! `a-byte-copy-whose-tail-is-shorter-than-its-vector` disagreed with Metal on exactly those bytes.
//!
//! The lowering is a byte loop, which is what a dynamic length admits: nothing about the count is
//! known, so neither alignment nor a wider step can be assumed. `llvm.memcpy` requires its operands
//! not to overlap, so ascending order needs no argument.
//!
//! The loop is spliced into the CALLER's block rather than called as a helper. A device-buffer
//! pointer does not survive a SPIR-V function boundary here — routed through a helper parameter it
//! arrives as a Private placeholder and the copy lands in scratch memory, which is the same silent
//! loss in a new costume. Splicing keeps both pointers on the raw-cursor path the surrounding code
//! already uses. That costs one restriction: the containing block must be LABELLED, because the
//! split renames it for any successor phi, and an implicit entry block has no name to rewrite. All
//! ten dynamic-length copies in the corpus are in labelled blocks; an entry-block one is left
//! exactly as it is today.
//!
//! Like [`super::async_copy`], this runs on the sanitized AIR TEXT before
//! [`super::ir::LlModule::parse`], and is a no-op — borrowing its input — unless a call with a
//! non-constant length is actually present.

use super::air_text::{call_arguments, last_token, split_args, LineBuffer};
use std::borrow::Cow;
use std::collections::HashMap;

/// One `llvm.memcpy` call with a dynamic length, decoded from its argument list.
struct DynamicMemcpy {
    dst_ty: String,
    src_ty: String,
    len_ty: String,
    dst: String,
    src: String,
    len: String,
}

/// Rewrite every dynamic-length `llvm.memcpy` in `san_ll` into an in-place byte-copy loop.
pub(crate) fn lower_dynamic_length_memcpy(san_ll: &str) -> Cow<'_, str> {
    let splits = plan_splits(san_ll);
    if splits.is_empty() {
        return Cow::Borrowed(san_ll);
    }
    let mut out = LineBuffer::with_capacity(san_ll.len().saturating_add(san_ll.len() / 8));
    let mut label: Option<String> = None;
    let mut next = 0usize;
    for line in san_ll.lines() {
        if let Some(block) = block_label(line) {
            label = Some(block.to_string());
            out.push(line);
            continue;
        }
        let trimmed = line.trim_start();
        let Some(call) = parse_dynamic_memcpy(trimmed) else {
            out.push(&rewrite_phi_predecessors(line, &splits));
            continue;
        };
        let Some(predecessor) = label.clone() else {
            // An implicit entry block has no name for a successor phi to be re-pointed at.
            out.push(line);
            continue;
        };
        let indent = &line[..line.len() - trimmed.len()];
        emit_copy_loop(&mut out, indent, &call, &predecessor, next);
        label = Some(format!("__dmc_done{next}"));
        next += 1;
    }
    if san_ll.ends_with('\n') {
        out.text.push('\n');
    }
    Cow::Owned(out.text)
}

/// Owned counterpart: once rewriting has produced a replacement, returning it drops the superseded
/// input before typed parsing begins, and a no-op returns the original allocation unchanged.
pub(crate) fn lower_dynamic_length_memcpy_owned(san_ll: String) -> String {
    match lower_dynamic_length_memcpy(&san_ll) {
        Cow::Borrowed(_) => san_ll,
        Cow::Owned(lowered) => lowered,
    }
}

/// The label each split block ENDS at after rewriting, keyed by its original label. A phi in a
/// successor names the predecessor block, and after a split the branch it names comes from the
/// loop's exit block instead. Computed up front because a phi can precede its predecessor in the
/// text; empty when the module has no dynamic-length copy to split on.
fn plan_splits(san_ll: &str) -> HashMap<String, String> {
    let mut splits = HashMap::new();
    let mut label: Option<&str> = None;
    let mut original: Option<String> = None;
    let mut next = 0usize;
    for line in san_ll.lines() {
        if let Some(block) = block_label(line) {
            label = Some(block);
            original = None;
            continue;
        }
        if parse_dynamic_memcpy(line.trim_start()).is_none() {
            continue;
        }
        let Some(block) = label else { continue };
        let start = original.get_or_insert_with(|| block.to_string()).clone();
        splits.insert(start, format!("__dmc_done{next}"));
        next += 1;
    }
    splits
}

/// The label a basic-block header line declares, if it is one.
fn block_label(line: &str) -> Option<&str> {
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let name = line.split(':').next()?;
    (!name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$'))
        && line[name.len()..].starts_with(':'))
    .then_some(name)
}

/// Re-point a phi's incoming block at the loop exit that now terminates the split predecessor.
/// Only phi lines are touched: a `br`/`switch` still targets the block's unchanged START.
fn rewrite_phi_predecessors(line: &str, splits: &HashMap<String, String>) -> String {
    if !line.contains(" phi ") {
        return line.to_string();
    }
    let mut rewritten = line.to_string();
    for (from, to) in splits {
        rewritten = rewritten.replace(&format!(", %{from} ]"), &format!(", %{to} ]"));
    }
    rewritten
}

/// Emit the byte loop replacing one call, leaving the emitter positioned in `__dmc_done{n}` so the
/// rest of the original block follows it.
fn emit_copy_loop(
    out: &mut LineBuffer,
    indent: &str,
    call: &DynamicMemcpy,
    predecessor: &str,
    n: usize,
) {
    let (dst_ty, src_ty, len_ty) = (&call.dst_ty, &call.src_ty, &call.len_ty);
    let (dst, src, len) = (&call.dst, &call.src, &call.len);
    out.push_fmt(format_args!("{indent}br label %__dmc_head{n}"));
    out.push_fmt(format_args!("__dmc_head{n}:"));
    out.push_fmt(format_args!(
        "{indent}%__dmc_i{n} = phi {len_ty} [ 0, %{predecessor} ], [ %__dmc_next{n}, %__dmc_body{n} ]"
    ));
    // `llvm.memcpy` takes an unsigned byte count.
    out.push_fmt(format_args!(
        "{indent}%__dmc_more{n} = icmp ult {len_ty} %__dmc_i{n}, {len}"
    ));
    out.push_fmt(format_args!(
        "{indent}br i1 %__dmc_more{n}, label %__dmc_body{n}, label %__dmc_done{n}"
    ));
    out.push_fmt(format_args!("__dmc_body{n}:"));
    out.push_fmt(format_args!(
        "{indent}%__dmc_src{n} = getelementptr i8, {src_ty} {src}, {len_ty} %__dmc_i{n}"
    ));
    out.push_fmt(format_args!(
        "{indent}%__dmc_byte{n} = load i8, {src_ty} %__dmc_src{n}, align 1"
    ));
    out.push_fmt(format_args!(
        "{indent}%__dmc_dst{n} = getelementptr i8, {dst_ty} {dst}, {len_ty} %__dmc_i{n}"
    ));
    out.push_fmt(format_args!(
        "{indent}store i8 %__dmc_byte{n}, {dst_ty} %__dmc_dst{n}, align 1"
    ));
    out.push_fmt(format_args!(
        "{indent}%__dmc_next{n} = add {len_ty} %__dmc_i{n}, 1"
    ));
    out.push_fmt(format_args!("{indent}br label %__dmc_head{n}"));
    out.push_fmt(format_args!("__dmc_done{n}:"));
}

/// Decode `call void @llvm.memcpy.<dst>.<src>.<len>(<dst>, <src>, <len>, i1 <volatile>)`, returning
/// `None` for anything this pass must not touch: a declaration, a constant length (the emitter
/// unrolls those), or a volatile copy, whose ordering a plain loop does not promise.
fn parse_dynamic_memcpy(line: &str) -> Option<DynamicMemcpy> {
    if !line.contains("call") || line.starts_with("declare") {
        return None;
    }
    let suffix = line.split("@llvm.memcpy.").nth(1)?;
    let mut spaces = suffix.split('(').next()?.split('.');
    let dst_ty = pointer_type(spaces.next()?)?;
    let src_ty = pointer_type(spaces.next()?)?;
    let len_ty = spaces.next()?.to_string();
    if !len_ty.starts_with('i') || len_ty[1..].parse::<u32>().is_err() {
        return None;
    }
    let args = split_args(call_arguments(line)?);
    if args.len() != 4 || last_token(&args[3]) != "false" {
        return None;
    }
    let len = last_token(&args[2]).to_string();
    if !len.starts_with('%') {
        return None;
    }
    Some(DynamicMemcpy {
        dst_ty,
        src_ty,
        len_ty,
        dst: last_token(&args[0]).to_string(),
        src: last_token(&args[1]).to_string(),
        len,
    })
}

/// The LLVM spelling of the `pN` address-space token in the intrinsic's name.
fn pointer_type(space: &str) -> Option<String> {
    match space.strip_prefix('p')?.parse::<u32>().ok()? {
        0 => Some("ptr".to_string()),
        n => Some(format!("ptr addrspace({n})")),
    }
}

#[cfg(test)]
mod tests {
    use super::lower_dynamic_length_memcpy;
    use std::borrow::Cow;

    fn module(body: &str) -> String {
        format!("define void @k() {{\nentry:\n  br label %19\n19:\n{body}  br label %24\n24:\n  ret void\n}}\n")
    }

    const DYNAMIC: &str = "  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 1 %21, \
                           ptr addrspace(1) align 1 %22, i64 %23, i1 false) #2, !alias.scope !30\n";

    #[test]
    fn a_constant_length_copy_is_left_for_the_emitter_to_unroll() {
        let ll = module(
            "  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %a, ptr addrspace(1) %b, \
             i64 48, i1 false)\n",
        );
        assert!(matches!(
            lower_dynamic_length_memcpy(&ll),
            Cow::Borrowed(value) if std::ptr::eq(value, ll.as_str())
        ));
    }

    #[test]
    fn a_volatile_copy_is_not_rewritten_into_a_plain_loop() {
        let ll = module(&DYNAMIC.replace("i1 false", "i1 true"));
        assert!(matches!(
            lower_dynamic_length_memcpy(&ll),
            Cow::Borrowed(value) if std::ptr::eq(value, ll.as_str())
        ));
    }

    #[test]
    fn a_lone_declaration_is_not_a_call() {
        let ll =
            "declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)\n";
        assert!(matches!(
            lower_dynamic_length_memcpy(ll),
            Cow::Borrowed(value) if std::ptr::eq(value, ll)
        ));
    }

    #[test]
    fn a_dynamic_length_copy_becomes_a_byte_loop_in_the_calling_block() {
        let ll = module(DYNAMIC);
        let lowered = lower_dynamic_length_memcpy(&ll);
        assert!(!lowered.contains("@llvm.memcpy.p1.p1.i64(ptr"), "{lowered}");
        // The loop is spliced in place: it must NOT become a call, whose parameters would strip the
        // pointers of their device-buffer provenance.
        assert!(!lowered.contains("call void @__"), "{lowered}");
        assert!(
            lowered.contains("%__dmc_i0 = phi i64 [ 0, %19 ], [ %__dmc_next0, %__dmc_body0 ]"),
            "{lowered}"
        );
        assert!(lowered.contains("icmp ult i64 %__dmc_i0, %23"), "{lowered}");
        assert!(
            lowered.contains("%__dmc_src0 = getelementptr i8, ptr addrspace(1) %22, i64 %__dmc_i0"),
            "{lowered}"
        );
        assert!(
            lowered.contains("store i8 %__dmc_byte0, ptr addrspace(1) %__dmc_dst0, align 1"),
            "{lowered}"
        );
        assert!(lowered.contains("__dmc_done0:"), "{lowered}");
    }

    #[test]
    fn a_successor_phi_follows_the_split_block_to_its_loop_exit() {
        let ll = format!(
            "define void @k() {{\nentry:\n  br label %19\n19:\n{DYNAMIC}  br label %24\n24:\n  \
             %26 = phi i32 [ 0, %entry ], [ 1, %19 ]\n  ret void\n}}\n"
        );
        let lowered = lower_dynamic_length_memcpy(&ll);
        assert!(
            lowered.contains("%26 = phi i32 [ 0, %entry ], [ 1, %__dmc_done0 ]"),
            "{lowered}"
        );
        // The unconditional branch still targets the block's unchanged START.
        assert!(lowered.contains("  br label %19\n"), "{lowered}");
    }

    #[test]
    fn an_implicit_entry_block_is_left_alone() {
        let ll = format!("define void @k() {{\n{DYNAMIC}  ret void\n}}\n");
        assert!(matches!(
            lower_dynamic_length_memcpy(&ll),
            Cow::Borrowed(value) if std::ptr::eq(value, ll.as_str())
        ));
    }
}
