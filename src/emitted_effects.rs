//! Whether a finished module can still do what its AIR said it does.
//!
//! Every lowering decision that removes work is safe only while something the dispatch can observe
//! survives it. Metal's `[[function_constant]]` folding is the one that can remove all of it: AIR
//! declares no default for a function constant, so `meta::globals` reads its `undef`
//! `air.fc_initializer` as zero, and a shader that dispatches on the constant's VALUE — rather than
//! on `air.is_function_constant_defined`, which AIR does answer — can have every arm fall away. The
//! result validates, reflects, and writes nothing.
//!
//! Metal imageblocks are the second way it happens, and the cause is ours rather than the shader's:
//! an imageblock is tile memory that Metal's render pass resolves into its attachments, Vulkan has
//! no such step, and this translator therefore stages imageblock cells in per-invocation `Private`
//! or per-threadgroup `Workgroup` memory. A kernel that reads back what it staged and then writes a
//! texture is fine; a Metal tile CLEAR kernel, whose only write is the imageblock, becomes a module
//! that validates, reflects no resource at all, and clears nothing.
//!
//! The predicates here are the halves of noticing both: what the AIR plainly does — through either
//! path — and what the emitted module still can.

use crate::spirv_module::Module;
use crate::spirv_operand::Operand;
use spirv::{Op, StorageClass, Word};
use std::collections::HashMap;

/// Every `air.*` marker that names a write into memory the dispatch's caller owns, other than the
/// device atomics (see [`declares_a_device_atomic_write`], which has to exclude one operation).
///
/// A store through an `addrspace(1)` pointer is the spelling this used to know, and an AIR that
/// writes only through an INTRINSIC has no such line -- which is how three corpus kernels whose
/// every write is an `air.atomic.global.*` got past the pair of predicates below. Censused over all
/// 14579 local corpus sources by `@air.*` symbol, these are the write-side families that exist:
///
/// - `air.write_texture_{1d,1d_array,2d,2d_array,3d,cube,buffer_1d}` — one prefix covers them all.
/// - `air.write_imageblock_slice_to_texture_2d{,_array}`, 160 sources. NOT covered by the prefix
///   above: the substring is `write_imageblock_slice_to_texture`.
/// - `air.atomic_fetch_max_explicit_texture_2d`, 2 sources — a texture atomic, so an image write.
/// - `air.store.device_coherent` (19) and `air.store.system_coherent.volatile` (4).
/// - `air.atomic.global.{add,store,cmpxchg.weak,max,min,xchg,or,sub,xor,and}`.
///
/// Deliberately left out: `air.atomic.local.*` and `air.store.implicit_imageblock` write memory
/// that dies with the dispatch or the tile, which is what
/// [`module_has_an_observable_effect`] and [`air_declares_an_imageblock_write`] answer for;
/// `air.simdgroup_async_copy_2d` and `air.simdgroup_matrix_8x8_store` take a pointer whose address
/// space the AIR spells out at the call, so the `addrspace(1)` arm already sees the device cases.
const AIR_OBSERVABLE_WRITE_MARKERS: &[&str] = &[
    "@air.write_texture",
    "@air.write_imageblock_slice_to_texture",
    "@air.atomic_fetch_",
    "@air.store.device_coherent",
    "@air.store.system_coherent",
    // The `llvm.*` half of the same contract. `llvm.agx3.*.with.emask.global.*` is the AGX masked
    // device store (37 sources; the `.local` sibling is threadgroup and stays out), and LLVM's
    // memory intrinsics mangle their DESTINATION address space first, so `p1` is the device one --
    // `llvm.memcpy.p0.p1` reads a device buffer into an alloca and must not count.
    "@llvm.agx3.store.with.emask.global",
    "@llvm.memcpy.p1.",
    "@llvm.memset.p1.",
];

/// Whether the AIR plainly writes something a dispatch can observe: a device-write intrinsic, or a
/// store through a device (`addrspace(1)`) pointer.
///
/// Deliberately textual and deliberately generous — it only decides whether the finished module is
/// worth asking about, and [`module_has_an_observable_effect`] is what actually decides.
pub(crate) fn air_declares_an_observable_write(air_ll: &str) -> bool {
    air_ll.lines().any(|line| {
        let line = line.trim_start();
        // A `declare` names a symbol; only a call performs it. AIR declares every intrinsic it
        // references and some it no longer calls, so reading the declarations would report a
        // module that dropped its last write as one that still has one.
        if line.starts_with("declare ") {
            return false;
        }
        if line.starts_with("store ") && line.contains("ptr addrspace(1)") {
            return true;
        }
        AIR_OBSERVABLE_WRITE_MARKERS
            .iter()
            .any(|marker| line.contains(marker))
            || calls_a_device_atomic_write(line)
    })
}

/// Whether the line performs a device atomic that leaves a new value behind.
///
/// `air.atomic.global.load` does not, and the prefix alone would count a kernel that only reads a
/// device counter as one that writes — refusing a shader that really is a no-op in Metal too. This
/// is the same distinction [`is_atomic_read_modify_write`] draws on the emitted side.
fn calls_a_device_atomic_write(line: &str) -> bool {
    const MARKER: &str = "@air.atomic.global.";
    line.match_indices(MARKER)
        .any(|(start, _)| !line[start + MARKER.len()..].starts_with("load"))
}

/// Whether the AIR writes an imageblock cell: a store through the imageblock address space.
///
/// `addrspace(4)` is that space. Measured over all 14579 local corpus sources: 729 stores through
/// it in 143 sources, and every one of those 143 declares an `air.imageblock` entry parameter --
/// no source stores there without one. Textual and generous for the same reason as
/// [`air_declares_an_observable_write`]: it only decides whether the finished module is worth
/// asking about.
///
/// Separate from [`air_declares_an_observable_write`] because a caller observes the two through
/// different machinery. A device store lands in memory the dispatch's caller already owns; an
/// imageblock cell is tile memory that Metal's render pass RESOLVES into its attachments when the
/// tile completes, and there is no such step to lower.
pub(crate) fn air_declares_an_imageblock_write(air_ll: &str) -> bool {
    air_ll.lines().any(|line| {
        let line = line.trim_start();
        if line.starts_with("declare ") {
            return false;
        }
        (line.starts_with("store ") && line.contains("ptr addrspace(4)"))
            // LLVM's memory intrinsics mangle their destination address space first.
            || line.contains("@llvm.memcpy.p4.")
            || line.contains("@llvm.memset.p4.")
    })
}

/// Whether the module contains any instruction that can affect memory the DISPATCH'S CALLER can
/// observe -- which is the question [`air_declares_an_observable_write`] asked, so it is the
/// question this has to answer.
///
/// Every opcode that writes through a pointer answers to one rule: the write counts unless the
/// pointer's storage class dies with the dispatch. `Function` and `Private` are per-invocation and
/// `Workgroup` is per-threadgroup. Staging a value in an alloca or in threadgroup memory is how
/// ordinary code gets work done, so counting either would report an emptied module as live -- and a
/// folded shader that keeps its threadgroup scratch and loses every buffer write is exactly the
/// shape that happens: 17 corpus sources reached here with no store outside those three classes,
/// and the largest of them writes 157 textures in its AIR.
///
/// The rule is one rule deliberately. Applying it to stores and exempting atomics and
/// `OpCopyMemory` is the asymmetry that let those 17 through in the first place: a threadgroup
/// counter is no more visible to a caller than a threadgroup store. No corpus source depends on
/// either arm today -- none has a local-storage atomic or copy as its only surviving effect -- so
/// this costs nothing and closes the same hole in the arms where it has not been dug yet.
///
/// `OpImageWrite` is the exception, and only because it takes an image rather than a pointer: an
/// image operand is a descriptor or a placeholder for one, never `Function`/`Private`/`Workgroup`.
pub(crate) fn module_has_an_observable_effect(module: &Module) -> bool {
    let mut pointer_storage: HashMap<Word, StorageClass> = HashMap::new();
    for instruction in module.types_global_values.iter() {
        if instruction.class.opcode != Op::TypePointer {
            continue;
        }
        if let (Some(id), Some(Operand::StorageClass(storage))) =
            (instruction.result_id, instruction.operands.first())
        {
            pointer_storage.insert(id, *storage);
        }
    }
    let mut value_storage: HashMap<Word, StorageClass> = HashMap::new();
    for instruction in module.all_inst_iter() {
        if let (Some(id), Some(result_type)) = (instruction.result_id, instruction.result_type) {
            if let Some(storage) = pointer_storage.get(&result_type) {
                value_storage.insert(id, *storage);
            }
        }
    }
    module.all_inst_iter().any(|instruction| {
        let opcode = instruction.class.opcode;
        if opcode == Op::ImageWrite {
            return true;
        }
        if !matches!(opcode, Op::Store | Op::CopyMemory | Op::AtomicStore)
            && !is_atomic_read_modify_write(opcode)
        {
            return false;
        }
        // Every one of those opcodes names the pointer it writes through first (`OpCopyMemory`
        // names the TARGET first and the source second, which is the one that matters here).
        let Some(Operand::IdRef(pointer)) = instruction.operands.first() else {
            // A write whose pointer this cannot read is not evidence that nothing survived.
            return true;
        };
        !matches!(
            value_storage.get(pointer),
            Some(StorageClass::Function | StorageClass::Private | StorageClass::Workgroup)
        )
    })
}

// What is left after this check, re-measured over all 14579 corpus sources by disassembling every
// translating module and asking this predicate of it directly. 68 of the 13231 that translate have
// no caller-observable effect, and every one of them is accounted for:
//
//   - 42 kernels whose AIR entry is literally `ret void`. They really do nothing; every argument is
//     `air.arg_unused`.
//   - 20 FRAGMENT entries. A fragment's result is an `Output` variable, not a store, so this
//     predicate says "no" about all of them and means nothing by it -- which is why
//     `air_declares_an_observable_write` gates the question and no fragment reaches it.
//   - 2 `fence_wait` kernels whose only call is `air.atomic.fence`. Nothing to write.
//   - the 4 `MatrixMultiplyNNA11_M*` below.
//
// So the four ARE the whole residue now, but they were not when that was first written: the
// AIR-side predicate knew two spellings of a write, and 24 more modules -- 16 imageblock-only tile
// kernels, 3 whose every write was a device atomic, 5 whose every write was
// `llvm.agx3.store.with.emask.global` -- were passing it by never being asked.
//
// Four modules -- `MatrixMultiplyNNA11_M1..M4` -- still translate to something that cannot write.
// They carry no function constants at all, so the arm above never names them; the pair of
// predicates WOULD refuse them if the function-constant precondition in
// `reject_function_constant_erased_effects` were dropped, and that widening was tried and not
// taken. It costs exactly those 4 and moves nothing else in the corpus, but it also refuses every
// stage-MISMATCHED translation whose write cannot bind, which halves the sweep in
// tests/live_module_scope_variables.rs -- 55 fixture/stage pairs and 123 module-scope globals down
// to 31 and 53. Four modules out of 14579 did not look worth that.
//
// What is actually wrong with the four, so the next person does not re-derive it: the entry
// advances two device buffer parameters by a DYNAMIC byte offset and passes the cursors to one of
// eight `internal fastcc` helpers chosen by a runtime switch. `raw_call_arg_id`
// (`native/emitter/body/calls.rs`) declines to propagate the caller's buffer root through a
// parameter whose offset carries dynamic terms and returns `Ok(None)`; the helper parameter then
// stays a Private zero placeholder and every raw store through it lands there. M1 emits 154
// `OpStore`, 132 of which target an `OpCopyObject %_ptr_Private_uint`, and declares one
// StorageBuffer variable for the three buffers its own reflection reports.
//
// Three independent sweeps agree those four are the whole non-function-constant residue: AIR
// device-writes with no StorageBuffer store in the module; reflection reporting `Unused` for a
// buffer an AIR dataflow shows dereferenced; and `OpStore` into a Private pointer placeholder.
//
// A trap in that third sweep, which faked a 121-module class out of an 8-module one: the
// `air.static_init` constructor is INLINED into the entry, so restricting the scan to the entry
// body does not exclude it, and its stores go into module-scope `Private` globals. Tell them apart
// by `OpName` -- the AIR globals the constructor mirrors carry one, the zero variable synthesised
// for a possibly-absent parameter does not.
//
// Turning that `Ok(None)` into an `Err` is NOT the fix: 72 of the 14579 reach that arm and 68 are
// unharmed by the decline. A twenty-line reduction of the shape -- one helper, one dynamic byte
// cursor, one store -- lowers correctly and keeps its `OpStore`, so the trigger is something the
// four have that the reduction does not.

/// The atomic opcodes that leave a new value behind. `OpAtomicLoad` does not, and counting it once
/// cost a whole invented failure class.
fn is_atomic_read_modify_write(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::AtomicExchange
            | Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
            | Op::AtomicIIncrement
            | Op::AtomicIDecrement
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor
            | Op::AtomicFAddEXT
            | Op::AtomicFMinEXT
            | Op::AtomicFMaxEXT
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_store_is_an_observable_write() {
        assert!(air_declares_an_observable_write(
            "  store i32 %v, ptr addrspace(1) %p, align 4\n"
        ));
        assert!(air_declares_an_observable_write(
            "  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t)\n"
        ));
    }

    /// A write spelled only as an intrinsic has no `store ... addrspace(1)` line, and three corpus
    /// kernels whose every write is a device atomic reached the emitted-side check with the AIR-side
    /// one answering "this module never claimed to write".
    #[test]
    fn a_device_write_spelled_as_an_intrinsic_is_an_observable_write() {
        for call in [
            "  %r = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 1)\n",
            "  call void @air.atomic.global.store.u.i32(ptr addrspace(1) %p, i32 1)\n",
            "  %r = call i32 @air.atomic.global.max.s.i32(ptr addrspace(1) %p, i32 1)\n",
            "  call void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1) %t)\n",
            "  call void @air.store.device_coherent.i32(ptr addrspace(1) %p, i32 1)\n",
            "  call void @air.store.system_coherent.volatile.i32(ptr addrspace(1) %p, i32 1)\n",
            "  %r = call i32 @air.atomic_fetch_max_explicit_texture_2d.u.i32(ptr addrspace(1) %t)\n",
        ] {
            assert!(
                air_declares_an_observable_write(call),
                "{call} names a write the dispatch's caller owns"
            );
        }
    }

    /// The `llvm.*` half of the same contract, and the address-space mangling that decides it.
    #[test]
    fn an_llvm_intrinsic_write_is_read_from_its_destination_address_space() {
        assert!(air_declares_an_observable_write(
            "  tail call void @llvm.agx3.store.with.emask.global.v4i8(ptr addrspace(1) %p, \
             <4 x i8> %v, i16 15, i16 15, i16 1)\n"
        ));
        assert!(air_declares_an_observable_write(
            "  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) %dst, ptr %src, i64 16, i1 false)\n"
        ));
        assert!(air_declares_an_observable_write(
            "  call void @llvm.memset.p1.i64(ptr addrspace(1) %dst, i8 0, i64 16, i1 false)\n"
        ));
        // Destination first: copying a device buffer INTO an alloca is a read.
        assert!(!air_declares_an_observable_write(
            "  call void @llvm.memcpy.p0.p1.i64(ptr %dst, ptr addrspace(1) %src, i64 16, i1 false)\n"
        ));
        // The threadgroup sibling of the AGX store dies with the dispatch.
        assert!(!air_declares_an_observable_write(
            "  tail call void @llvm.agx3.store.with.emask.local.v4i8(ptr addrspace(3) %p, \
             <4 x i8> %v, i16 15, i16 15, i16 1)\n"
        ));
        // And an imageblock destination is the other predicate's question.
        assert!(!air_declares_an_observable_write(
            "  call void @llvm.memset.p4.i64(ptr addrspace(4) %dst, i8 0, i64 8, i1 false)\n"
        ));
        assert!(air_declares_an_imageblock_write(
            "  call void @llvm.memset.p4.i64(ptr addrspace(4) %dst, i8 0, i64 8, i1 false)\n"
        ));
    }

    /// The one device atomic that leaves no value behind. Counting it would refuse a kernel that
    /// only reads a device counter -- which really is a no-op in Metal too.
    #[test]
    fn a_device_atomic_load_is_not_an_observable_write() {
        assert!(!air_declares_an_observable_write(
            "  %r = call i32 @air.atomic.global.load.u.i32(ptr addrspace(1) %p)\n"
        ));
        // ... and it must not mask a real one beside it.
        assert!(air_declares_an_observable_write(
            "  %r = call i32 @air.atomic.global.load.u.i32(ptr addrspace(1) %p)\n  \
             %s = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 1)\n"
        ));
    }

    /// Threadgroup atomics and implicit-imageblock stores are deliberately not on the list: they
    /// write memory that dies with the dispatch or the tile.
    #[test]
    fn a_local_atomic_is_not_an_observable_write() {
        assert!(!air_declares_an_observable_write(
            "  %r = call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %p, i32 1)\n"
        ));
        assert!(!air_declares_an_observable_write(
            "  call void @air.store.implicit_imageblock.v4f16(ptr addrspace(4) %p)\n"
        ));
    }

    #[test]
    fn an_imageblock_store_is_recognised_by_its_address_space() {
        assert!(air_declares_an_imageblock_write(
            "  store <4 x half> zeroinitializer, ptr addrspace(4) %5, align 8\n"
        ));
        // The imageblock address space is not one of the observable ones, and vice versa: the two
        // predicates answer different questions and must not collapse into each other.
        assert!(!air_declares_an_observable_write(
            "  store <4 x half> zeroinitializer, ptr addrspace(4) %5, align 8\n"
        ));
        assert!(!air_declares_an_imageblock_write(
            "  store i32 %v, ptr addrspace(1) %p, align 4\n"
        ));
        // A load through an imageblock pointer is not a write.
        assert!(!air_declares_an_imageblock_write(
            "  %v = load <4 x half>, ptr addrspace(4) %p, align 8\n"
        ));
    }

    #[test]
    fn a_threadgroup_or_stack_store_is_not_an_observable_write() {
        assert!(!air_declares_an_observable_write(
            "  store i32 %v, ptr addrspace(3) %p, align 4\n"
        ));
        assert!(!air_declares_an_observable_write(
            "  store i32 %v, ptr %alloca, align 4\n"
        ));
        // A load through a device pointer is not a write.
        assert!(!air_declares_an_observable_write(
            "  %v = load i32, ptr addrspace(1) %p, align 4\n"
        ));
    }
}
