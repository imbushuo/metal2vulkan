//! Give an explicit imageblock aliased onto the implicit one the storage the alias promises.
//!
//! An explicit Metal imageblock is tile-local scratch, and the native emitter stages it in a
//! `Workgroup` array of cells. `air.alias_implicit_imageblock` says the storage is not scratch at
//! all: it *is* the implicit imageblock -- the render targets the rasterizer already wrote -- so
//! the kernel's first read of a cell is a read of framebuffer content and its last write has to
//! land back there. The staging array is neither. Emitted on its own the module validates, resolves
//! an uninitialised array, and reaches no attachment, which is why every one of the 30 corpus
//! sources carrying the marker was refused before this pass existed.
//!
//! This pass keeps the staging array and gives it the two halves it was missing:
//!
//! * a **prologue** in the entry block that fills the calling thread's cell from the
//!   descriptor-backed planes, one `air.load.implicit_imageblock.*` lowering per member;
//! * an **epilogue** before the return that writes the cell back, one
//!   `air.store.implicit_imageblock.*` lowering per member.
//!
//! Both are bracketed by a `Workgroup` control barrier, so a thread that reads a neighbour's cell
//! sees a filled one and a thread whose cell another thread wrote does not race the write-back.
//!
//! Reusing the implicit-imageblock lowerings rather than emitting `OpImageRead`/`OpImageWrite`
//! directly is deliberate: they own the plane's descriptor allocation, its storage format and the
//! narrowing conversions between a plane's four-lane texel and a one-, two- or four-lane member.
//! A second derivation of any of those would be a second answer to the same question.
//!
//! The member -> attachment mapping is `meta::aliased_imageblock_planes`: member `i` in declaration
//! order is colour attachment `i`, which is what "the explicit layout IS the implicit layout"
//! means. Nothing keys on a member's source name.

use super::*;

/// Fill and write back the cells of an aliased explicit imageblock.
pub(in crate::passes) fn lower_aliased_imageblock(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
    planes: &[crate::meta::AliasedImageblockPlane],
) -> Result<(), String> {
    let Some(storage) = ctx.emit_sidecar.aliased_imageblock_staging else {
        return Ok(());
    };
    ctx.emit_sidecar.aliased_imageblock_staging = None;
    if planes.is_empty() {
        return Ok(());
    }
    if !matches!(stage, Stage::Kernel) {
        return Err(
            "an imageblock aliased onto the implicit imageblock is only valid in a tile kernel"
                .to_string(),
        );
    }
    let member_types = staging_member_types(ctx, storage, planes.len())?;

    // Both halves address the calling thread's own cell, as
    // `LocalInvocationId.y * local_size_x + LocalInvocationId.x`. That is the same index the
    // emitter computes for `air.imageblock_data`, which linearises the AIR coordinate `y * width +
    // x` with the threadgroup extent as the width -- but only because the AIR coordinate is the
    // calling thread's own threadgroup-local position, and that is an assumption inherited rather
    // than checked here. `tile_coordinate_scale` (`native::ir`) treats ANY entry parameter reaching
    // a cell coordinate as a thread position of coefficient one without asking which position role
    // it carries, and the shared-cell lowering's whole cell budget already rests on that: it
    // allocates one cell per thread of the threadgroup. An entry whose coordinate were a
    // grid-relative position would index past that array in the emitter's own linearisation, before
    // this pass is reached. All 30 corpus sources carrying the alias marker pass
    // `[[thread_position_in_threadgroup]]`, and the authored cases measure the correspondence.
    let uint = ctx.ty_uint();
    let v2uint = ctx.ty_vec_uint(2);
    let v3uint = ctx.ty_vec_uint(3);
    // This runs after `build_stage_input`, which binds the builtin whenever the entry declares
    // `[[thread_position_in_threadgroup]]`. A second Input variable with the same BuiltIn in one
    // OpEntryPoint is invalid SPIR-V, so reuse whichever one is already there.
    let local_id_var =
        air_calls::existing_builtin_input_var(ctx, BuiltIn::LocalInvocationId, v3uint)
            .unwrap_or_else(|| {
                stage_input::bind_kernel_v3uint_builtin(ctx, BuiltIn::LocalInvocationId)
            });
    let [local_x, _, _] = ctx.kernel_local_size_ids();

    let local_id = ctx.module.fresh_id();
    let x = ctx.module.fresh_id();
    let y = ctx.module.fresh_id();
    let row = ctx.module.fresh_id();
    let index = ctx.module.fresh_id();
    let coord = ctx.module.fresh_id();
    let mut preamble = vec![
        Instruction::new(
            Op::Load,
            Some(v3uint),
            Some(local_id),
            vec![Operand::IdRef(local_id_var)],
        ),
        Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(x),
            vec![Operand::IdRef(local_id), Operand::LiteralBit32(0)],
        ),
        Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(y),
            vec![Operand::IdRef(local_id), Operand::LiteralBit32(1)],
        ),
        Instruction::new(
            Op::IMul,
            Some(uint),
            Some(row),
            vec![Operand::IdRef(y), Operand::IdRef(local_x)],
        ),
        Instruction::new(
            Op::IAdd,
            Some(uint),
            Some(index),
            vec![Operand::IdRef(row), Operand::IdRef(x)],
        ),
        Instruction::new(
            Op::CompositeConstruct,
            Some(v2uint),
            Some(coord),
            vec![Operand::IdRef(x), Operand::IdRef(y)],
        ),
    ];

    // The prologue and epilogue are built detached from the module, so the implicit-imageblock
    // lowerings cannot find the coordinate or the loaded member by scanning for their definitions.
    // `phase_value_types` is the established channel for exactly that: it is `None` between passes,
    // and every entry below names an id this function just created.
    let mut value_types = ctx.phase_value_types.take().unwrap_or_default();
    value_types.insert(coord, v2uint);

    let mut epilogue = Vec::new();
    for (plane, member_ty) in planes.iter().zip(&member_types) {
        let cell = cell_member_pointer(ctx, storage, index, *member_ty, plane, planes.len());
        preamble.extend(cell.chain);
        let texel = ctx.module.fresh_id();
        let name = format!("air.load.implicit_imageblock.{}", plane.intrinsic_suffix);
        let args = plane_args(ctx, plane, coord);
        ctx.phase_value_types = Some(value_types);
        let loaded =
            air_calls::lower_implicit_imageblock_load(ctx, &name, texel, *member_ty, &args);
        value_types = ctx.phase_value_types.take().unwrap_or_default();
        preamble.extend(loaded?);
        preamble.push(Instruction::new(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(cell.pointer), Operand::IdRef(texel)],
        ));

        // A second chain with its own result id: the prologue's is defined in the entry block and
        // an access chain does not reach across one in Logical SPIR-V.
        let cell = cell_member_pointer(ctx, storage, index, *member_ty, plane, planes.len());
        epilogue.extend(cell.chain);
        let value = ctx.module.fresh_id();
        epilogue.push(Instruction::new(
            Op::Load,
            Some(*member_ty),
            Some(value),
            vec![Operand::IdRef(cell.pointer)],
        ));
        value_types.insert(value, *member_ty);
        let name = format!("air.store.implicit_imageblock.{}", plane.intrinsic_suffix);
        let mut store_args = vec![value];
        store_args.extend(plane_args(ctx, plane, coord));
        ctx.phase_value_types = Some(value_types);
        let stored = air_calls::lower_implicit_imageblock_store(ctx, &name, &store_args);
        value_types = ctx.phase_value_types.take().unwrap_or_default();
        epilogue.extend(stored?);
    }
    drop(value_types);
    preamble.push(workgroup_barrier(ctx));
    epilogue.insert(0, workgroup_barrier(ctx));

    let return_blocks = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block
                .instructions
                .last()
                .is_some_and(|instruction| instruction.class.opcode == Op::Return)
        })
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    // Both barriers have to be reached by every thread of the threadgroup. One return block is
    // reached by all of them; several are the arms of a branch, and putting a barrier in each is a
    // barrier in non-uniform control flow, which Vulkan leaves undefined.
    let [return_block] = return_blocks[..] else {
        return Err(format!(
            "an imageblock aliased onto the implicit imageblock needs one return the whole \
             threadgroup reaches to write its cells back through, and the entry has {}",
            return_blocks.len()
        ));
    };

    let entry_block = &mut ctx.module.functions[entry_idx].blocks[0];
    let after_variables = entry_block
        .instructions
        .iter()
        .take_while(|instruction| instruction.class.opcode == Op::Variable)
        .count();
    entry_block
        .instructions
        .splice(after_variables..after_variables, preamble);

    let block = &mut ctx.module.functions[entry_idx].blocks[return_block];
    let terminator = block.instructions.len() - 1;
    block.instructions.splice(terminator..terminator, epilogue);
    Ok(())
}

struct CellMember {
    /// Instructions producing `pointer`, re-emitted per use because the prologue and the epilogue
    /// live in different blocks and a Logical-SPIR-V access chain does not cross one.
    chain: Vec<Instruction>,
    pointer: Word,
}

fn cell_member_pointer(
    ctx: &mut Ctx,
    storage: Word,
    index: Word,
    member_ty: Word,
    plane: &crate::meta::AliasedImageblockPlane,
    plane_count: usize,
) -> CellMember {
    let ptr_ty = ctx.ty_ptr(StorageClass::Workgroup, member_ty);
    let pointer = ctx.module.fresh_id();
    // A one-member cell keeps no struct around the member (`imageblock_data_pointee_from_air_type`
    // collapses it), so the array element IS the member and there is no second index to give.
    let mut operands = vec![Operand::IdRef(storage), Operand::IdRef(index)];
    if plane_count > 1 {
        let member = ctx.const_uint(plane.attachment);
        operands.push(Operand::IdRef(member));
    }
    CellMember {
        chain: vec![Instruction::new(
            Op::InBoundsAccessChain,
            Some(ptr_ty),
            Some(pointer),
            operands,
        )],
        pointer,
    }
}

/// `(attachment, coordinate, index, data rate)`, the trailing operands both implicit-imageblock
/// intrinsics take. An aliased cell is one texel of one sample of attachment `i`, so the array
/// index and the data rate are zero.
fn plane_args(
    ctx: &mut Ctx,
    plane: &crate::meta::AliasedImageblockPlane,
    coord: Word,
) -> Vec<Word> {
    let attachment = ctx.const_uint(plane.attachment);
    let zero = ctx.const_uint(0);
    vec![attachment, coord, zero, zero]
}

fn workgroup_barrier(ctx: &mut Ctx) -> Instruction {
    let scope = ctx.const_uint(Scope::Workgroup as u32);
    let semantics = ctx
        .const_uint((MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits());
    Instruction::new(
        Op::ControlBarrier,
        None,
        None,
        vec![
            Operand::IdScope(scope),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
        ],
    )
}

/// The emitted type of each cell member, in plane order.
///
/// Read off the staging array rather than re-derived from the AIR: the emitter chose the cell type
/// and the access chains this pass builds have to index the type it chose.
fn staging_member_types(ctx: &Ctx, storage: Word, plane_count: usize) -> Result<Vec<Word>, String> {
    let missing = || "aliased imageblock staging variable has no array type".to_string();
    let variable = type_def_of(ctx, value_result_type(ctx, storage).ok_or_else(missing)?)
        .ok_or_else(missing)?;
    let Some(Operand::IdRef(array)) = variable.operands.get(1) else {
        return Err(missing());
    };
    let array = type_def_of(ctx, *array).ok_or_else(missing)?;
    if array.class.opcode != Op::TypeArray {
        return Err(missing());
    }
    let Some(Operand::IdRef(cell)) = array.operands.first() else {
        return Err(missing());
    };
    let cell_def = type_def_of(ctx, *cell).ok_or_else(missing)?;
    if cell_def.class.opcode != Op::TypeStruct {
        // A single-member cell is collapsed to the member itself.
        if plane_count != 1 {
            return Err(format!(
                "aliased imageblock declares {plane_count} render-target planes and its cell is \
                 not a struct, so no member of it corresponds to a plane"
            ));
        }
        return Ok(vec![*cell]);
    }
    let members = cell_def
        .operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<Vec<_>>();
    if members.len() != plane_count {
        return Err(format!(
            "aliased imageblock declares {plane_count} render-target planes and the cell the \
             emitter staged has {} members",
            members.len()
        ));
    }
    Ok(members)
}
