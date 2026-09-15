//! Which descriptors the finished module provably cannot write, stated in the module.
//!
//! Vulkan does not read the absence of `NonWritable` as silence. A graphics-stage module that
//! declares an undecorated storage buffer, storage texel buffer or storage image requires its
//! consumer to enable `fragmentStoresAndAtomics` or `vertexPipelineStoresAndAtomics`, whether or not
//! any store exists (`VUID-RuntimeSpirv-NonWritable-06340` and `-06341`). Leaving a read-only
//! descriptor undecorated therefore does not merely omit information, it asks every consumer for a
//! device feature the shader does not use -- and the VUID is per-variable, so one undecorated
//! descriptor keeps the demand for the whole module.
//!
//! `spirv-val` does not check this decoration against the writes in the module, so a wrong answer
//! here is silent all the way to the device. That is why each rule below is a proof and not a
//! heuristic, and why `tests/nonwritable_covers_the_module.rs` performs the check `spirv-val` does
//! not.
//!
//! The two descriptor classes are proved differently because they are reached differently. A buffer
//! is reached through a pointer, so the proof is the pointer walk reflection already runs; an image
//! is reached by loading the variable itself, which is a much shorter walk and needs no addressing
//! model to bound it.

use super::footprint::{addressing_is_logical, descriptor_keys, Analyzer, DescriptorKey};
use crate::spirv_module::{Instruction, Module, Operand};
use spirv::{Decoration, Op, StorageClass, Word};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Decorate every descriptor the finished module provably never writes `NonWritable`.
pub(crate) fn decorate_unwritten_descriptors(module: &mut Module) {
    decorate_unwritten_storage_buffers(module);
    decorate_unwritten_storage_images(module);
}

/// Every descriptor `OpVariable` the module declares in the `StorageBuffer` storage class, paired
/// with the `(set, binding)` it is decorated with. Several variables can share one binding -- the
/// stage-input builder emits a typed alias per element type over a single collapsed buffer -- so the
/// key, not the id, is what a write is attributed to.
fn storage_buffer_descriptor_variables(module: &Module) -> Vec<(Word, DescriptorKey)> {
    let decorations = descriptor_keys(module);
    module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
        .filter(|instruction| {
            matches!(
                instruction.operands.first(),
                Some(Operand::StorageClass(StorageClass::StorageBuffer))
            )
        })
        .filter_map(|instruction| {
            let id = instruction.result_id?;
            Some((id, *decorations.get(&id)?))
        })
        .collect()
}

/// Decorate every storage-buffer descriptor the finished module provably never writes `NonWritable`.
///
/// Vulkan reads the absence of this decoration as a demand, not as silence: a graphics-stage module
/// declaring an undecorated storage buffer requires the consumer to enable
/// `fragmentStoresAndAtomics` or `vertexPipelineStoresAndAtomics` whether or not any store exists
/// (`VUID-RuntimeSpirv-NonWritable-06340` and `-06341`). Leaving a read-only buffer undecorated
/// therefore does not merely omit information, it asks every consumer for a device feature the
/// shader does not use.
///
/// The proof is the same walk reflection uses to widen a declared access, for the same reason it is
/// the same walk: "did any instruction write through this descriptor" is one fact about the module,
/// and deriving it twice is how the two answers drift apart. Three conditions make the walk's
/// silence a proof rather than an absence of evidence:
///
/// * Logical addressing. Under `PhysicalStorageBuffer64` a store can reach a buffer through an
///   address the walk cannot attribute to any descriptor, so no descriptor is provably unwritten.
/// * The walk saw no write. `observed` records reads and writes separately; a read-only descriptor
///   and an untouched one are both eligible.
/// * No pointer rooted at the descriptor reached an operand slot the walk does not model. That is
///   what `escaped` records, and it is the only remaining way a Logical module can write a buffer
///   without the walk seeing the store -- passing the pointer to a function, storing the pointer
///   itself, or handing it to an instruction the walk has no rule for.
///
/// `spirv-val` does not check this decoration against the stores in the module, so a wrong answer
/// here is silent. That is why the condition is a proof and not a heuristic.
fn decorate_unwritten_storage_buffers(module: &mut Module) {
    if !addressing_is_logical(module) {
        return;
    }
    let variables = storage_buffer_descriptor_variables(module);
    if variables.is_empty() {
        return;
    }
    let targets = variables
        .iter()
        .map(|(_, key)| *key)
        .collect::<BTreeSet<_>>();
    let analysis = Analyzer::new(module, &targets).analyze();
    let unwritten = targets
        .iter()
        .filter(|key| !analysis.escaped.contains(key))
        .filter(|key| !analysis.observed.get(key).is_some_and(|seen| seen.writes))
        .copied()
        .collect::<BTreeSet<_>>();

    for (id, key) in variables {
        if !unwritten.contains(&key) {
            continue;
        }
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::Decoration(Decoration::NonWritable),
            ],
        ));
    }
}

/// Every descriptor `OpVariable` the module declares whose type is a STORAGE image, paired with the
/// `(set, binding)` it is decorated with.
///
/// `OpTypeImage`'s `Sampled` operand is 2 for an image used without a sampler, which is the only
/// kind a shader can write and the only kind the VUID is about. A sampled image is not writable in
/// the first place, so decorating one would state nothing.
fn storage_image_descriptor_variables(module: &Module) -> Vec<(Word, DescriptorKey)> {
    let storage_images = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::TypeImage)
        .filter(|instruction| instruction.operands.get(5) == Some(&Operand::LiteralBit32(2)))
        .filter_map(|instruction| instruction.result_id)
        .collect::<HashSet<_>>();
    let pointers = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::TypePointer)
        .filter(|instruction| {
            matches!(
                instruction.operands.first(),
                Some(Operand::StorageClass(StorageClass::UniformConstant))
            )
        })
        .filter(|instruction| {
            matches!(instruction.operands.get(1), Some(Operand::IdRef(pointee)) if storage_images.contains(pointee))
        })
        .filter_map(|instruction| instruction.result_id)
        .collect::<HashSet<_>>();
    let decorations = descriptor_keys(module);
    module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
        .filter(|instruction| {
            instruction
                .result_type
                .is_some_and(|ty| pointers.contains(&ty))
        })
        .filter_map(|instruction| {
            Some((
                instruction.result_id?,
                *decorations.get(&instruction.result_id?)?,
            ))
        })
        .collect()
}

/// Whether an operand slot holding an image id is one this walk understands, and if so whether
/// reaching it writes the image.
///
/// The list is the whole rule: an id in a slot that is not here has left the walk, and a descriptor
/// whose image reaches one is not provably unwritten however few writes were seen.
fn image_operand_role(opcode: Op, position: usize) -> Option<ImageUse> {
    match opcode {
        // The variable itself. `OpLoad` produces the image object every other instruction consumes.
        Op::Load if position == 0 => Some(ImageUse::Propagates),
        // `OpImageTexelPointer Result Image Coordinate Sample` takes the VARIABLE, and exists to
        // hand a texel to an atomic. Every atomic but `OpAtomicLoad` writes, so this is a write.
        Op::ImageTexelPointer if position == 0 => Some(ImageUse::Writes),
        // Image objects that carry the same underlying image.
        Op::CopyObject | Op::Image | Op::SampledImage if position == 0 => {
            Some(ImageUse::Propagates)
        }
        Op::ImageWrite if position == 0 => Some(ImageUse::Writes),
        Op::ImageRead
        | Op::ImageFetch
        | Op::ImageSparseRead
        | Op::ImageSparseFetch
        | Op::ImageQuerySize
        | Op::ImageQuerySizeLod
        | Op::ImageQueryLevels
        | Op::ImageQuerySamples
        | Op::ImageQueryFormat
        | Op::ImageQueryOrder
        | Op::ImageQueryLod
            if position == 0 =>
        {
            Some(ImageUse::Reads)
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImageUse {
    Propagates,
    Reads,
    Writes,
}

/// Decorate every storage-image descriptor the finished module provably never writes.
///
/// The proof is shorter than the buffer one and needs no addressing model. An image lives in
/// `UniformConstant`; there is no such thing as a device address to one, so the only way to reach it
/// is to name the variable, and the only way to use it is to load an image object from it. Gating on
/// Logical anyway, as the buffer half must, would exclude the corpus's 387
/// `PhysicalStorageBuffer64` modules -- but that is not worth 387 modules: measured, only **4** of
/// the 340 that gain a decoration are among them. This walk
/// follows exactly that: the variable, the objects loaded from it, and the objects copied from those.
/// A descriptor is decorated when no write reached it AND no image rooted at it ever appeared in an
/// operand slot `image_operand_role` has no rule for -- a function argument, a stored value, an
/// instruction the walk does not know.
fn decorate_unwritten_storage_images(module: &mut Module) {
    let variables = storage_image_descriptor_variables(module);
    if variables.is_empty() {
        return;
    }
    let mut roots = variables
        .iter()
        .map(|(id, key)| (*id, *key))
        .collect::<HashMap<_, _>>();
    // Image objects derive from the variable in definition order inside a block, but a phi's
    // incoming edge is not, so iterate to a fixed point rather than assuming one pass suffices.
    loop {
        let mut grew = false;
        for instruction in module.all_inst_iter() {
            let Some(result) = instruction.result_id else {
                continue;
            };
            if roots.contains_key(&result) {
                continue;
            }
            let source = instruction
                .operands
                .iter()
                .enumerate()
                .find_map(|(position, operand)| {
                    let Operand::IdRef(id) = operand else {
                        return None;
                    };
                    (image_operand_role(instruction.class.opcode, position)
                        == Some(ImageUse::Propagates))
                    .then(|| roots.get(id).copied())
                    .flatten()
                });
            if let Some(key) = source {
                roots.insert(result, key);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    let mut disqualified = BTreeSet::<DescriptorKey>::new();
    for instruction in module
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter())
        .flat_map(|block| block.instructions.iter())
    {
        for (position, operand) in instruction.operands.iter().enumerate() {
            let Operand::IdRef(id) = operand else {
                continue;
            };
            let Some(key) = roots.get(id).copied() else {
                continue;
            };
            match image_operand_role(instruction.class.opcode, position) {
                Some(ImageUse::Reads | ImageUse::Propagates) => {}
                // A write, or a slot the walk cannot follow: either way this descriptor is not
                // provably unwritten.
                Some(ImageUse::Writes) | None => {
                    disqualified.insert(key);
                }
            }
        }
    }

    for (id, key) in variables {
        if disqualified.contains(&key) {
            continue;
        }
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::Decoration(Decoration::NonWritable),
            ],
        ));
    }
}
