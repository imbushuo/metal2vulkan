//! Copying a whole imageblock block to a texture, one texel per loop iteration.
//!
//! Metal's `imageblock_slice` carries a cell pointer, a size and a validity flag and nothing else
//! (`metal_imageblocks`, `imageblock_slice_base`), and
//! `air.write_imageblock_slice_to_texture_*` copies `size.x * size.y` cells to the same-shaped
//! region of the texture at the destination coordinate. **The block starts at the imageblock's own
//! origin**; the pointer selects which FIELD of the cell is being copied and its cell coordinate is
//! ignored -- see `imageblock_cell_chain` for the device measurement. The per-texel lowering in
//! `float_imageblock` writes exactly one cell, and refuses anything larger rather than write
//! 1/(WxH) of the block.
//!
//! This pass rewrites the calls it refuses into a loop whose body is the same call restricted to a
//! single cell, so the per-texel lowering stays the only place that knows how to convert and write
//! one. It runs as a pre-pass because a loop needs blocks: `lower_air_calls` splices a lowering's
//! instructions into the block the call was in, and caches the block graph it walks.

use super::block_split::{labelled_block, CallSiteSplit};
use super::*;

/// Rewrite every static-extent block copy in the entry into a per-texel loop.
pub(in crate::passes) fn lower_imageblock_block_copies(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> Result<(), String> {
    // Rewriting one site inserts blocks, so re-scan from the top rather than hold indices across
    // the mutation. Each rewrite leaves a residual call whose region operand is a constant 1x1,
    // which the scan does not select, so this terminates.
    loop {
        let names = air_names(&ctx.module);
        let Some(site) = find_block_copy_call(ctx, entry_idx, &names) else {
            return Ok(());
        };
        rewrite_block_copy(ctx, entry_idx, site)?;
    }
}

/// One selected call: its block and instruction index, the call itself, and its extent.
struct BlockCopySite {
    block: usize,
    inst: usize,
    call: Instruction,
    /// The extent in cells when AIR spelled it as a constant, and `None` when the size operand only
    /// exists at runtime. Every runtime one in the corpus is arithmetic on
    /// `[[threads_per_threadgroup]]`, which reaches SPIR-V as `OpSpecConstant`, so it is a loop
    /// bound rather than something to fold.
    region: Option<[u32; 2]>,
}

/// The next slice write that copies more than one cell.
///
/// A call whose extent is a constant one cell is what the per-texel lowering already handles. A call
/// whose extent only exists at runtime is selected too -- the loop bound is an id either way -- but
/// only when AIR spelled the size out: the implicit form's extent is the imageblock's own, and there
/// is no operand to read it from when that is not static.
fn find_block_copy_call(
    ctx: &Ctx,
    entry_idx: usize,
    names: &HashMap<Word, String>,
) -> Option<BlockCopySite> {
    for (block, blk) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (inst, call) in blk.instructions.iter().enumerate() {
            if call.class.opcode != Op::FunctionCall {
                continue;
            }
            let Some(Operand::IdRef(callee)) = call.operands.first() else {
                continue;
            };
            if !names
                .get(callee)
                .is_some_and(|name| name.starts_with("air.write_imageblock_slice_to_texture"))
            {
                continue;
            }
            let args = call_args(call);
            if args.len() < 6 {
                continue;
            }
            // Skipping a call must not end the scan: a module can name one this pass leaves alone
            // alongside the copies it does rewrite.
            let region = static_imageblock_region(ctx, &args);
            match region {
                Some([width, height])
                    if width.checked_mul(height).is_none_or(|cells| cells <= 1) =>
                {
                    continue
                }
                Some(_) => {}
                None if !imageblock_region_is_explicit(ctx, &args) => continue,
                None => {}
            }
            return Some(BlockCopySite {
                block,
                inst,
                call: call.clone(),
                region,
            });
        }
    }
    None
}

fn call_args(inst: &Instruction) -> Vec<Word> {
    inst.operands[1..]
        .iter()
        .filter_map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect()
}

/// The Workgroup cell array a slice-write pointer names, as `(root, cell index, trailing indices)`.
///
/// The native emitter builds every imageblock cell pointer as `OpInBoundsAccessChain %cells
/// %linearIndex` over the threadgroup cell array, optionally followed by member indices selecting a
/// field of the cell (`native/emitter/body/calls.rs`). Offsetting that pointer by a block coordinate
/// is not expressible in Logical SPIR-V, so the loop re-forms the chain at the block coordinate
/// instead, which needs the chain taken apart. Anything else -- notably the `Private`
/// per-invocation staging array, which holds one cell and therefore has no block to copy -- is
/// refused rather than guessed at.
///
/// # The cell index is a stride, not an origin
///
/// The trailing indices are the field path and they are kept. The cell index is NOT: Metal starts
/// the block at the imageblock's own origin whatever cell the pointer names, so the index is used
/// only to recover the row stride the emitter linearised the array with, and the copy walks from
/// cell zero.
///
/// Device-measured on `apple-m3-max-macos26.5.2` over an 8x8 imageblock of 32-byte cells holding
/// `x + 8y`, in the case `imageblock-runtime-slice-region-weight-plane`. Three writes of the same
/// field, at three sizes, from three different cells:
///
/// | slice pointer | size | Metal wrote |
/// |---|---|---|
/// | cell (4, 4) | 4x4 | cells (0..3, 0..3) |
/// | cell (0, 0) | 2x2 | cells (0..1, 0..1) |
/// | cell (5, 3) | 1x1 | cell (0, 0) |
///
/// Not one of them started where the pointer pointed, and the 1x1 row rules out a rounding or
/// clamping story: a copy anchored at the pointer would have answered 29 there and answered 0.
/// The header agrees once read for what it says rather than what it suggests --
/// `_imageblock_slice_base` stores the pointer as `_imgblkptr` alongside the size, and the field
/// offset is all a plane copy needs from it.
pub(in crate::passes) fn imageblock_cell_chain(
    ctx: &Ctx,
    pointer: Word,
) -> Result<(Word, Word, Vec<Word>), String> {
    // A cell pointer arrives as one chain or as a chain of chains: the emitter indexes the cell
    // array, and AIR may then select a field of that cell through a second access chain. Walk back
    // to the array variable collecting indices in source order, so index 0 is the cell index and the
    // rest select within the cell.
    let mut indices: Vec<Word> = Vec::new();
    let mut current = pointer;
    while let Some(def) = value_def_instruction(ctx, current) {
        if !matches!(def.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
            break;
        }
        let mut ids = def.operands.iter().filter_map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        });
        let Some(base) = ids.next() else {
            break;
        };
        let mut leading = ids.collect::<Vec<_>>();
        leading.append(&mut indices);
        indices = leading;
        current = base;
    }
    let storage = value_result_type(ctx, current)
        .filter(|_| !indices.is_empty())
        .and_then(|ty| ptr_storage(&type_defs(&ctx.module), ty));
    if storage != Some(StorageClass::Workgroup) {
        return Err(format!(
            "air.write_imageblock_slice_to_texture copies a block of cells, but its pointer does \
             not index a threadgroup imageblock cell array (it roots in {storage:?}), and this \
             translator has no other storage a neighbouring cell could come from"
        ));
    }
    Ok((current, indices.remove(0), indices))
}

/// The declared length of the imageblock cell array a pointer roots in.
fn imageblock_cell_array_len(ctx: &Ctx, cell_array: Word) -> Option<u32> {
    let ptr_ty = value_result_type(ctx, cell_array)?;
    let array = type_def_of(ctx, pointer_pointee_type(ctx, ptr_ty)?)?;
    if array.class.opcode != Op::TypeArray {
        return None;
    }
    let Some(Operand::IdRef(length)) = array.operands.get(1) else {
        return None;
    };
    match value_def_instruction(ctx, *length)?.operands.first() {
        Some(Operand::LiteralBit32(literal)) => Some(*literal),
        _ => None,
    }
}

/// The row stride of the imageblock cell array, read off the index the emitter built.
///
/// The cell array is linearised `y * width + x`, and the width is either the APV imageblock extent
/// or the threadgroup extent -- a choice the native emitter makes and this layer must not make a
/// second time. So recover the exact id it multiplied by instead of re-deriving the quantity: the
/// index is `OpIAdd (OpIMul %y %width) %x`, and the width is that multiplier.
pub(in crate::passes) fn imageblock_row_stride(ctx: &Ctx, cell_index: Word) -> Option<Word> {
    let add = value_def_instruction(ctx, cell_index)?;
    if add.class.opcode != Op::IAdd {
        return None;
    }
    let Some(Operand::IdRef(row)) = add.operands.first() else {
        return None;
    };
    let mul = value_def_instruction(ctx, *row)?;
    if mul.class.opcode != Op::IMul {
        return None;
    }
    match mul.operands.get(1) {
        Some(Operand::IdRef(width)) => Some(*width),
        _ => None,
    }
}

/// A constant vector of `ty` with every component set to `value`, for re-spelling a call operand
/// whose declared type the residual call has to keep.
fn constant_splat(ctx: &mut Ctx, ty: Word, value: u32) -> Option<Word> {
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode != Op::TypeVector {
        return None;
    }
    let Some(Operand::IdRef(elem)) = def.operands.first() else {
        return None;
    };
    let Some(Operand::LiteralBit32(lanes)) = def.operands.get(1) else {
        return None;
    };
    let (elem, lanes) = (*elem, *lanes);
    let component = ctx.get_or_create(Op::Constant, Some(elem), vec![Operand::LiteralBit32(value)]);
    Some(ctx.get_or_create(
        Op::ConstantComposite,
        Some(ty),
        vec![Operand::IdRef(component); lanes as usize],
    ))
}

fn rewrite_block_copy(ctx: &mut Ctx, entry_idx: usize, site: BlockCopySite) -> Result<(), String> {
    const WHAT: &str = "air.write_imageblock_slice_to_texture block copy";
    let BlockCopySite {
        block,
        inst,
        call,
        region,
    } = site;
    let args = call_args(&call);
    // How the extent reads in a refusal, and whether it can be more than one row. A runtime extent
    // cannot prove either, so it takes the general shape of both.
    let extent = match region {
        Some([width, height]) => format!("{width}x{height}"),
        None => "runtime-sized".to_string(),
    };
    let multi_row = region.is_none_or(|[_, height]| height > 1);
    if let Some([width, height]) = region {
        width
            .checked_mul(height)
            .ok_or("air.write_imageblock_slice_to_texture block extent overflows a cell count")?;
    }

    let (cell_array, linearised_index, tail) = imageblock_cell_chain(ctx, args[1])?;
    // The array is sized by the emitter (an APV extent, else the tile byte budget divided by the
    // cell size -- `imageblock::cell_capacity`), and a block that does not fit is a disagreement
    // with that sizing rather than something to write past. A constant extent larger than the array
    // is a disagreement this pass can PROVE, so it refuses; everything else is clipped at runtime by
    // the loop's own continue test, which is the only answer available once the extent is an id.
    let capacity = imageblock_cell_array_len(ctx, cell_array);
    if let (Some([width, height]), Some(capacity)) = (region, capacity) {
        if width * height > capacity {
            return Err(format!(
                "air.write_imageblock_slice_to_texture copies a {extent} block, but the \
                 imageblock holds {capacity} cells, so the block names cells that do not exist"
            ));
        }
    }
    let cell_ptr_ty = value_result_type(ctx, args[1])
        .ok_or("air.write_imageblock_slice_to_texture cell pointer has no result type")?;
    // Only a multi-row block needs the array's row stride; a 1-row one walks consecutive cells.
    let row_stride = if multi_row {
        Some(imageblock_row_stride(ctx, linearised_index).ok_or_else(|| {
            format!(
                "air.write_imageblock_slice_to_texture copies a {extent} block, but its \
                 cell index is not the `y * width + x` the imageblock cell array is linearised by, \
                 so the stride from one block row to the next is unknown"
            )
        })?)
    } else {
        None
    };
    let coord_ty = value_result_type(ctx, args[5])
        .ok_or("air.write_imageblock_slice_to_texture destination coordinate has no type")?;
    let coord_lanes = vector_type_shape(ctx, coord_ty).map_or(1, |(_, lanes)| lanes);
    if coord_lanes < 2 && multi_row {
        return Err(format!(
            "air.write_imageblock_slice_to_texture copies a {extent} block to a texture \
             whose destination coordinate has {coord_lanes} component(s), which cannot name the \
             second block axis"
        ));
    }
    // Re-spell the region operand as a constant 1x1 of its own declared type: the residual call
    // keeps the callee's signature, and `static_imageblock_region` has to read one cell off it.
    let region_ty = value_result_type(ctx, args[4])
        .ok_or("air.write_imageblock_slice_to_texture size operand has no type")?;
    let unit_region = constant_splat(ctx, region_ty, 1).ok_or_else(|| {
        format!(
            "air.write_imageblock_slice_to_texture size operand type %{region_ty} is not a vector"
        )
    })?;
    let flag_ty = value_result_type(ctx, args[2]).unwrap_or_else(|| ctx.ty_bool());
    let flag_true = ctx.const_bool_of(flag_ty, true);

    let mut split = CallSiteSplit::open(ctx, entry_idx, block, inst, WHAT)?;
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let zero = ctx.const_uint(0);
    let one = ctx.const_uint(1);
    // The trip count and the block width, as ids the loop can use whichever way AIR spelled them.
    let (total, width_id) = match region {
        Some([width, height]) => (ctx.const_uint(width * height), ctx.const_uint(width)),
        None => {
            // The size operand is a `<2 x i16>` in every corpus module that has a runtime one.
            // Widen it once so the trip count and the block coordinate are computed in the
            // counter's type, then read the pair off it.
            let region32 = if scalar_bit_width(ctx, region_ty) == 32 {
                args[4]
            } else {
                let id = ctx.module.fresh_id();
                let wide = ctx.ty_vec_uint(2);
                split.prefix.push(Instruction::new(
                    Op::UConvert,
                    Some(wide),
                    Some(id),
                    vec![Operand::IdRef(args[4])],
                ));
                id
            };
            let extract = |ctx: &mut Ctx, split: &mut CallSiteSplit, lane: u32| {
                let id = ctx.module.fresh_id();
                split.prefix.push(Instruction::new(
                    Op::CompositeExtract,
                    Some(uint),
                    Some(id),
                    vec![Operand::IdRef(region32), Operand::LiteralBit32(lane)],
                ));
                id
            };
            let width_id = extract(ctx, &mut split, 0);
            let height_id = extract(ctx, &mut split, 1);
            let total = ctx.module.fresh_id();
            split.prefix.push(Instruction::new(
                Op::IMul,
                Some(uint),
                Some(total),
                vec![Operand::IdRef(width_id), Operand::IdRef(height_id)],
            ));
            (total, width_id)
        }
    };
    // The destination coordinate in 32-bit form, so the per-texel offset adds in one type. The
    // residual call carries it back to `build_fetch_coord`, which accepts an already-32-bit one.
    let (coord, coord32_ty) = if scalar_bit_width(ctx, coord_ty) == 32 {
        (args[5], coord_ty)
    } else {
        let wide = if coord_lanes == 1 {
            uint
        } else {
            ctx.ty_vec_uint(coord_lanes)
        };
        let id = ctx.module.fresh_id();
        split.prefix.push(Instruction::new(
            Op::UConvert,
            Some(wide),
            Some(id),
            vec![Operand::IdRef(args[5])],
        ));
        (id, wide)
    };

    let entry_label = split.entry_label();
    let header = ctx.module.fresh_id();
    let cond = ctx.module.fresh_id();
    let body = ctx.module.fresh_id();
    let latch = ctx.module.fresh_id();
    let merge = ctx.module.fresh_id();
    split.branch_prefix_to(header);
    let continuation = split.continuation(ctx);

    let counter = ctx.module.fresh_id();
    let next_counter = ctx.module.fresh_id();
    let header_block = labelled_block(
        header,
        vec![
            Instruction::new(
                Op::Phi,
                Some(uint),
                Some(counter),
                vec![
                    Operand::IdRef(zero),
                    Operand::IdRef(entry_label),
                    Operand::IdRef(next_counter),
                    Operand::IdRef(latch),
                ],
            ),
            Instruction::new(
                Op::LoopMerge,
                None,
                None,
                vec![
                    Operand::IdRef(merge),
                    Operand::IdRef(latch),
                    Operand::LoopControl(spirv::LoopControl::NONE),
                ],
            ),
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(cond)]),
        ],
    );

    let binary = |ctx: &mut Ctx, out: &mut Vec<Instruction>, op, ty, a: Word, b: Word| {
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            op,
            Some(ty),
            Some(id),
            vec![Operand::IdRef(a), Operand::IdRef(b)],
        ));
        id
    };

    // The block coordinate and the cell it names are computed in the CONTINUE TEST, not the body,
    // because the test needs the cell index: the block origin is a runtime value (the writing
    // thread's own cell), so no extent -- constant or not -- proves that `base + offset` stays
    // inside the cell array. Walking off the end would be an `OpInBoundsAccessChain` out of range.
    // The index rises with the counter for any block no wider than the array row it walks, so
    // leaving through the exit the loop already has clips exactly the tail that does not fit, and
    // costs no CFG edge and no phi repair.
    let mut cond_insts = Vec::new();
    // A 1-row block walks the counter directly, which is also the only shape that needs no stride.
    let (dx, dy) = match row_stride {
        None => (counter, zero),
        Some(_) => (
            binary(ctx, &mut cond_insts, Op::UMod, uint, counter, width_id),
            binary(ctx, &mut cond_insts, Op::UDiv, uint, counter, width_id),
        ),
    };
    let offset = match row_stride {
        None => dx,
        Some(stride) => {
            let row = binary(ctx, &mut cond_insts, Op::IMul, uint, dy, stride);
            binary(ctx, &mut cond_insts, Op::IAdd, uint, row, dx)
        }
    };
    // `offset` IS the cell index: the block starts at the imageblock's own origin, not at the cell
    // the slice pointer names. See `imageblock_cell_chain` for the measurement.
    let cell_index = offset;
    let counted = binary(ctx, &mut cond_insts, Op::ULessThan, bool_ty, counter, total);
    let in_range = match capacity {
        None => counted,
        Some(capacity) => {
            let limit = ctx.const_uint(capacity);
            let fits = binary(
                ctx,
                &mut cond_insts,
                Op::ULessThan,
                bool_ty,
                cell_index,
                limit,
            );
            binary(ctx, &mut cond_insts, Op::LogicalAnd, bool_ty, counted, fits)
        }
    };
    cond_insts.push(Instruction::new(
        Op::BranchConditional,
        None,
        None,
        vec![
            Operand::IdRef(in_range),
            Operand::IdRef(body),
            Operand::IdRef(merge),
        ],
    ));
    let cond_block = labelled_block(cond, cond_insts);

    let mut body_insts = Vec::new();
    let cell = ctx.module.fresh_id();
    let mut chain = vec![Operand::IdRef(cell_array), Operand::IdRef(cell_index)];
    chain.extend(tail.into_iter().map(Operand::IdRef));
    body_insts.push(Instruction::new(
        Op::InBoundsAccessChain,
        Some(cell_ptr_ty),
        Some(cell),
        chain,
    ));
    let texel_coord = if coord_lanes == 1 {
        binary(ctx, &mut body_insts, Op::IAdd, coord32_ty, coord, dx)
    } else {
        // A block only ever offsets the two spatial axes; any further component (an array slice)
        // stays where it is.
        let mut components = vec![Operand::IdRef(dx), Operand::IdRef(dy)];
        components.resize(coord_lanes as usize, Operand::IdRef(zero));
        let delta = ctx.module.fresh_id();
        body_insts.push(Instruction::new(
            Op::CompositeConstruct,
            Some(coord32_ty),
            Some(delta),
            components,
        ));
        binary(ctx, &mut body_insts, Op::IAdd, coord32_ty, coord, delta)
    };

    // The residual call is the original restricted to the single cell this iteration names: an
    // explicit 1x1 region, so the per-texel lowering accepts it and clips it against the texture
    // exactly as it clips a one-cell write today.
    let mut residual = call.clone();
    for (arg, operand) in residual.operands.iter_mut().skip(1).enumerate() {
        match arg {
            1 => *operand = Operand::IdRef(cell),
            2 => *operand = Operand::IdRef(flag_true),
            4 => *operand = Operand::IdRef(unit_region),
            5 => *operand = Operand::IdRef(texel_coord),
            _ => {}
        }
    }
    body_insts.push(residual);
    body_insts.push(Instruction::new(
        Op::Branch,
        None,
        None,
        vec![Operand::IdRef(latch)],
    ));

    let latch_block = labelled_block(
        latch,
        vec![
            Instruction::new(
                Op::IAdd,
                Some(uint),
                Some(next_counter),
                vec![Operand::IdRef(counter), Operand::IdRef(one)],
            ),
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(header)]),
        ],
    );
    let merge_block = labelled_block(
        merge,
        vec![Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(continuation)],
        )],
    );

    split.finish(
        ctx,
        entry_idx,
        vec![
            header_block,
            cond_block,
            labelled_block(body, body_insts),
            latch_block,
            merge_block,
        ],
    );
    Ok(())
}
