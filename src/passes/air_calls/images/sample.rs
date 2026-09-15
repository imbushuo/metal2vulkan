//! Color texture sampling and pixel-filter lowering.

use super::*;

use crate::reflect::SamplerAddressMode;

/// Lower air.sample_texture_<dim>: combine the texture+sampler operands via OpSampledImage and emit
/// OpImageSampleExplicitLod when AIR carries a scalar float LOD operand, when integer textures require
/// it, or when a compute kernel cannot legally use implicit derivatives.
/// AIR call args: a0=texture(loaded image), a1=sampler(loaded sampler), a2=coord, then optional
/// layer/LOD/flags. Result is AIR's {vecN,i8}; we reproduce it as OpCompositeConstruct so the
/// downstream CompositeExtract 0 still yields the color.
/// Whether the AIR family names Metal's explicit-derivative sample.
fn is_gradient_family(name: &str) -> bool {
    // `air.sample_texture_2d_grad.v4f16`: the family is the token before the return-type suffix, and
    // a suffix can itself be several tokens (`.u.v4i16`), so test every token rather than a position.
    name.split('.').any(|token| token.ends_with("_grad"))
}

/// Metal's `gradient2d(dPdx, dPdy)` operands, when the AIR family names the gradient form.
///
/// The `_grad` symbol places the two derivative vectors immediately after the coordinate (and, for
/// an arrayed family, its layer), then a single scalar `min_lod_clamp` float. Without this the
/// generic LOD search reads that clamp as an explicit level and the derivatives are dropped, which
/// is a sample of the wrong mip in every module that asked for one.
fn sample_gradient_pair(
    ctx: &Ctx,
    name: &str,
    arrayed: bool,
    args: &[Word],
    spatial: usize,
) -> Result<Option<(Word, Word)>, String> {
    if !is_gradient_family(name) {
        return Ok(None);
    }
    let start = if arrayed { 4 } else { 3 };
    let is_derivative = |arg: Word| {
        value_result_type(ctx, arg)
            .and_then(|ty| vector_type_shape(ctx, ty))
            .is_some_and(|(elem, lanes)| lanes as usize == spatial && is_f32_scalar(ctx, elem))
    };
    let (Some(&dx), Some(&dy)) = (args.get(start), args.get(start + 1)) else {
        return Err(format!("{name} has no gradient operands"));
    };
    if !is_derivative(dx) || !is_derivative(dy) {
        return Err(format!(
            "{name} does not carry two {spatial}-component float derivatives after its coordinate"
        ));
    }
    // The clamp that follows selects a floor for the derived level. Zero is no clamp at all, which
    // is what every gradient sample in the corpus asks for; a real one would need `MinLod`.
    if let Some(&clamp) = args.get(start + 2) {
        let is_zero = value_def_instruction(ctx, clamp).is_some_and(|def| {
            matches!(def.class.opcode, Op::ConstantNull)
                || (def.class.opcode == Op::Constant
                    && def.operands.first() == Some(&Operand::LiteralBit32(0)))
        });
        if !is_zero {
            return Err(format!(
                "{name} carries a nonzero minimum-LOD clamp, which this translator does not model"
            ));
        }
    }
    Ok(Some((dx, dy)))
}

pub(in crate::passes) fn lower_sample(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err("air.sample has no result".into()),
    };
    if args.len() < 3 {
        return Err("air.sample missing texture/sampler/coord".into());
    }
    let (mut img, samp, coord) = (resolve_image_value(ctx, args[0]), args[1], args[2]);
    if texture_operand_is_absent(ctx, img) {
        if let Some(sampled_img) =
            recovered_image_for_private_operand(ctx, img, name, ImageOperandUse::Sampled)
        {
            img = sampled_img;
        } else {
            return lower_null_texture_result(ctx, res, rty);
        }
    }
    let mut out = vec![];
    img = load_image_if_pointer(ctx, img, &mut out);
    // Build the sampled-image type matching the texture's dimension (recorded when the image var was
    // created). Defaults to 2D for the common case.
    let (fallback_dim, fallback_arrayed, fallback_comp) = image_shape_or_recorded(ctx, img);
    let (img_ty, dim, arrayed, comp) =
        sampled_operand_image_info(ctx, img, fallback_dim, fallback_arrayed, fallback_comp);
    // The sample result vector component type must match the image's sampled type (v4uint/v4int for
    // integer textures); a float sample with explicit LOD is needed at fragment scope for integer
    // textures (implicit-LOD sampling of integer images is invalid).
    let sample_v4 = image_fetch_v4(ctx, img, v4);
    let is_int_tex = comp != crate::passes::ImageComp::Float;

    if let Some(sampler_state) = ctx
        .sampler_states
        .get(&samp)
        .copied()
        .filter(|state| state.uses_pixel_coordinates())
    {
        // Pixel-coordinate linear float sampling is emulated in-shader (per-tap fetch + lerp) for
        // 2D AND 3D: Vulkan forbids an unnormalized-coordinates sampler on a Dim3d image view, so
        // a plain OpImageSample with raw pixel coords can never be the 3D lowering (Vulkan consumers reject
        // the pipeline/descriptor pairing at dispatch).
        let pixel_linear = comp == crate::passes::ImageComp::Float
            && sampler_state.uses_linear_filter()
            && matches!(dim, Dim::Dim1D | Dim::Dim2D | Dim::Dim3D);
        let pixel_bicubic = comp == crate::passes::ImageComp::Float
            && sampler_state.uses_bicubic_filter()
            && matches!(dim, Dim::Dim1D | Dim::Dim2D);
        let pixel_fetch = comp != crate::passes::ImageComp::Float
            || sampler_state.uses_pixel_nearest()
            || arrayed;
        if pixel_linear || pixel_bicubic || pixel_fetch {
            if is_gradient_family(name) {
                // These paths pick one mip level and fetch taps from it. A gradient sample chooses
                // its level from the derivatives, which nothing here reads.
                return Err(format!(
                    "{name} samples through a pixel-coordinate sampler, whose emulation has no \
                     place for the gradient the call carries"
                ));
            }
            let lod = match find_sample_lod(ctx, arrayed, args, &mut out) {
                Some(lod) => sample_lod_to_fetch_lod(ctx, lod, &mut out)?,
                None => ctx.const_uint(0),
            };
            if pixel_linear {
                let color = lower_pixel_linear_sample(
                    ctx,
                    sampler_state,
                    img,
                    dim,
                    arrayed,
                    coord,
                    args,
                    lod,
                    sample_v4,
                    &mut out,
                )?;
                return finish_sample_result(ctx, res, rty, color, sample_v4, out);
            }
            if pixel_bicubic {
                let color = lower_pixel_bicubic_sample(
                    ctx,
                    sampler_state,
                    img,
                    dim,
                    arrayed,
                    coord,
                    args,
                    lod,
                    sample_v4,
                    false,
                    &mut out,
                )?;
                return finish_sample_result(ctx, res, rty, color, sample_v4, out);
            }
            let fetch = build_pixel_fetch_coord(
                ctx,
                sampler_state,
                img,
                dim,
                arrayed,
                coord,
                args,
                lod,
                &mut out,
            )?;
            let mut color = ctx.module.fresh_id();
            push_image_read_or_fetch(ctx, &mut out, img, fetch.coord, Some(lod), sample_v4, color)?;
            if let Some(in_bounds) = fetch.in_bounds {
                let guarded = ctx.module.fresh_id();
                let zero_color = const_null_of(ctx, sample_v4);
                out.push(Instruction::new(
                    Op::Select,
                    Some(sample_v4),
                    Some(guarded),
                    vec![
                        Operand::IdRef(in_bounds),
                        Operand::IdRef(color),
                        Operand::IdRef(zero_color),
                    ],
                ));
                color = guarded;
            }
            return finish_sample_result(ctx, res, rty, color, sample_v4, out);
        }
        return Err(format!(
            "pixel-coordinate sampling does not support {dim:?} with {:?} filtering",
            sampler_state.min_filter
        ));
    }
    if let Some(sampler_state) = ctx
        .sampler_states
        .get(&samp)
        .copied()
        .filter(|state| state.uses_bicubic_filter())
    {
        if comp == crate::passes::ImageComp::Float && matches!(dim, Dim::Dim1D | Dim::Dim2D) {
            let lod = match find_sample_lod(ctx, arrayed, args, &mut out) {
                Some(lod) => sample_lod_to_fetch_lod(ctx, lod, &mut out)?,
                None => ctx.const_uint(0),
            };
            let color = lower_pixel_bicubic_sample(
                ctx,
                sampler_state,
                img,
                dim,
                arrayed,
                coord,
                args,
                lod,
                sample_v4,
                true,
                &mut out,
            )?;
            return finish_sample_result(ctx, res, rty, color, sample_v4, out);
        }
        return Err(
            "air.sample_texture bicubic filtering is supported only for 1D/2D float textures"
                .into(),
        );
    }
    if is_int_tex && is_gradient_family(name) {
        return Err(format!(
            "{name} samples an integer texture, whose nearest-fetch emulation has no place for \
             the gradient the call carries"
        ));
    }
    if is_int_tex {
        if let Some(sampler_state) = ctx.sampler_states.get(&samp).copied() {
            let lod = match find_sample_lod(ctx, arrayed, args, &mut out) {
                Some(lod) => sample_lod_to_fetch_lod(ctx, lod, &mut out)?,
                None => ctx.const_uint(0),
            };
            let fetch = build_normalized_nearest_fetch_coord(
                ctx,
                sampler_state,
                img,
                dim,
                arrayed,
                coord,
                args,
                lod,
                &mut out,
            )?;
            let mut color = ctx.module.fresh_id();
            push_image_read_or_fetch(ctx, &mut out, img, fetch.coord, Some(lod), sample_v4, color)?;
            if let Some(in_bounds) = fetch.in_bounds {
                let guarded = ctx.module.fresh_id();
                let zero_color = const_null_of(ctx, sample_v4);
                out.push(Instruction::new(
                    Op::Select,
                    Some(sample_v4),
                    Some(guarded),
                    vec![
                        Operand::IdRef(in_bounds),
                        Operand::IdRef(color),
                        Operand::IdRef(zero_color),
                    ],
                ));
                color = guarded;
            }
            return finish_sample_result(ctx, res, rty, color, sample_v4, out);
        }
    }

    let si_ty = ctx.ty_sampled_image(img_ty);
    let si = ctx.module.fresh_id();
    let color = ctx.module.fresh_id();
    let samp = valid_sampler_value(ctx, samp, &mut out)?;
    out.push(Instruction::new(
        Op::SampledImage,
        Some(si_ty),
        Some(si),
        vec![Operand::IdRef(img), Operand::IdRef(samp)],
    ));

    // Compose the sample coordinate to match the image dimension. AIR passes the spatial coord in
    // arg[2]; for *_array samples the layer index is the NEXT integer arg (arg[3]). The combined
    // coordinate has (#spatial-dims + arrayed) float components.
    let mut coord_for_sample = build_sample_coord(ctx, dim, arrayed, coord, args, &mut out)?;
    let spatial = sample_spatial_dims(dim);
    let grad = match spatial {
        Some(spatial) => sample_gradient_pair(ctx, name, arrayed, args, spatial)?,
        None => sample_gradient_pair(ctx, name, arrayed, args, 0)?,
    };
    let level = if grad.is_some() {
        None
    } else {
        find_sample_level(ctx, arrayed, args, &mut out)
    };
    let (const_offset, dynamic_offset) = if let Some(spatial) = spatial {
        let (const_offset, dynamic_offset) =
            sample_const_or_dynamic_offset(ctx, arrayed, args, spatial as u32)?;
        let const_offset = const_offset.and_then(|offset| {
            if offset.iter().all(|delta| *delta == 0) {
                None
            } else {
                Some(const_sint_vec(ctx, &offset))
            }
        });
        (const_offset, dynamic_offset)
    } else {
        (None, None)
    };
    if let (Some(spatial), Some(offset)) = (spatial, dynamic_offset) {
        coord_for_sample = apply_dynamic_sample_offset(
            ctx,
            img,
            arrayed,
            coord_for_sample,
            offset,
            spatial,
            &mut out,
        )?;
    }
    push_image_sample(
        ctx,
        &mut out,
        sample_v4,
        color,
        si,
        coord_for_sample,
        level,
        is_int_tex,
        const_offset,
        grad,
    );

    finish_sample_result(ctx, res, rty, color, sample_v4, out)
}

#[allow(clippy::too_many_arguments)]
fn lower_pixel_bicubic_sample(
    ctx: &mut Ctx,
    sampler_state: StaticSamplerState,
    img: Word,
    dim: Dim,
    arrayed: bool,
    coord: Word,
    args: &[Word],
    lod: Word,
    sample_v4: Word,
    normalized_coordinates: bool,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let spatial = match dim {
        Dim::Dim1D => 1,
        Dim::Dim2D => 2,
        _ => return Err("air.sample_texture bicubic sample unsupported dimension".into()),
    };
    let (offset, dynamic_offset) =
        sample_const_or_dynamic_offset(ctx, arrayed, args, spatial as u32)?;
    let dynamic_offset = dynamic_offset
        .map(|offset| dynamic_i32_integer_offset_components(ctx, offset, spatial, out))
        .transpose()?;
    let layer = if arrayed {
        Some(
            args.get(3)
                .copied()
                .ok_or("air.sample_texture bicubic array texture missing layer")?,
        )
    } else {
        None
    };
    let size = query_image_size(ctx, img, spatial, arrayed, lod, out);
    let float_ty = ctx.ty_float();
    let sint = ctx.ty_sint();
    let bool_ty = ctx.ty_bool();
    let glsl = ctx.glsl();
    let zero_f = ctx.const_float(0.0);
    let half_f = ctx.const_float(0.5);
    let mut base = Vec::with_capacity(spatial);
    let mut axis_weights = Vec::with_capacity(spatial);
    for (axis, component) in sample_coord_components(ctx, coord, spatial as u32, out)?
        .into_iter()
        .enumerate()
    {
        let Operand::IdRef(component) = component else {
            return Err("air.sample_texture bicubic coord component is not an id".into());
        };
        let component = if normalized_coordinates {
            let extent = if spatial == 1 {
                size
            } else {
                let extent = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::CompositeExtract,
                    Some(ctx.ty_uint()),
                    Some(extent),
                    vec![Operand::IdRef(size), Operand::LiteralBit32(axis as u32)],
                ));
                extent
            };
            let extent_f = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ConvertUToF,
                Some(float_ty),
                Some(extent_f),
                vec![Operand::IdRef(extent)],
            ));
            let pixel_component = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FMul,
                Some(float_ty),
                Some(pixel_component),
                vec![Operand::IdRef(component), Operand::IdRef(extent_f)],
            ));
            pixel_component
        } else {
            component
        };
        let component = clamp_pixel_coord_component_finite(
            ctx,
            sampler_state,
            component,
            size,
            spatial > 1,
            axis as u32,
            out,
        );
        let biased = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(biased),
            vec![Operand::IdRef(component), Operand::IdRef(half_f)],
        ));
        let floor = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(floor),
            vec![
                Operand::IdRef(glsl),
                Operand::LiteralExtInstInteger(8),
                Operand::IdRef(biased),
            ],
        ));
        let mut base_i = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ConvertFToS,
            Some(sint),
            Some(base_i),
            vec![Operand::IdRef(floor)],
        ));
        if let Some(offset) = &offset {
            if offset[axis] != 0 {
                let shifted = ctx.module.fresh_id();
                let delta = ctx.const_int_of(sint, offset[axis] as i64);
                out.push(Instruction::new(
                    Op::IAdd,
                    Some(sint),
                    Some(shifted),
                    vec![Operand::IdRef(base_i), Operand::IdRef(delta)],
                ));
                base_i = shifted;
            }
        } else if let Some(offset) = &dynamic_offset {
            let shifted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::IAdd,
                Some(sint),
                Some(shifted),
                vec![Operand::IdRef(base_i), Operand::IdRef(offset[axis])],
            ));
            base_i = shifted;
        }
        let frac = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(frac),
            vec![Operand::IdRef(biased), Operand::IdRef(floor)],
        ));
        base.push(base_i);
        axis_weights.push(catmull_rom_weights(ctx, frac, out));
    }

    let zero_color = const_null_of(ctx, sample_v4);
    let mut acc = zero_color;
    for tap in 0..4usize.pow(spatial as u32) {
        let tap_indices = (0..spatial)
            .map(|axis| (tap / 4usize.pow(axis as u32)) % 4)
            .collect::<Vec<_>>();
        let tap_offsets = tap_indices
            .iter()
            .map(|index| *index as i32 - 1)
            .collect::<Vec<_>>();
        let tap_coord = pixel_linear_tap_coord(
            ctx,
            sampler_state,
            dim,
            arrayed,
            &base,
            &tap_offsets,
            layer,
            size,
            out,
        )?;
        let fetched = ctx.module.fresh_id();
        push_image_read_or_fetch(
            ctx,
            out,
            img,
            tap_coord.coord,
            Some(lod),
            sample_v4,
            fetched,
        )?;
        let color = if let Some(in_bounds) = tap_coord.in_bounds {
            let guarded = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Select,
                Some(sample_v4),
                Some(guarded),
                vec![
                    Operand::IdRef(in_bounds),
                    Operand::IdRef(fetched),
                    Operand::IdRef(zero_color),
                ],
            ));
            guarded
        } else {
            fetched
        };
        let mut weight = axis_weights[0][tap_indices[0]];
        for (axis, weights) in axis_weights.iter().enumerate().skip(1) {
            let combined = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FMul,
                Some(float_ty),
                Some(combined),
                vec![
                    Operand::IdRef(weight),
                    Operand::IdRef(weights[tap_indices[axis]]),
                ],
            ));
            weight = combined;
        }
        let weighted = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::VectorTimesScalar,
            Some(sample_v4),
            Some(weighted),
            vec![Operand::IdRef(color), Operand::IdRef(weight)],
        ));
        let weight_is_zero = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(weight_is_zero),
            vec![Operand::IdRef(weight), Operand::IdRef(zero_f)],
        ));
        let contribution = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Select,
            Some(sample_v4),
            Some(contribution),
            vec![
                Operand::IdRef(weight_is_zero),
                Operand::IdRef(zero_color),
                Operand::IdRef(weighted),
            ],
        ));
        let sum = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FAdd,
            Some(sample_v4),
            Some(sum),
            vec![Operand::IdRef(acc), Operand::IdRef(contribution)],
        ));
        acc = sum;
    }
    Ok(acc)
}

fn catmull_rom_weights(ctx: &mut Ctx, t: Word, out: &mut Vec<Instruction>) -> [Word; 4] {
    let float_ty = ctx.ty_float();
    let c_half = ctx.const_float(0.5);
    let c_neg_half = ctx.const_float(-0.5);
    let c_one = ctx.const_float(1.0);
    let c_one_half = ctx.const_float(1.5);
    let c_two = ctx.const_float(2.0);
    let c_two_half = ctx.const_float(2.5);
    let mul = |ctx: &mut Ctx, a, b, out: &mut Vec<Instruction>| {
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FMul,
            Some(float_ty),
            Some(id),
            vec![Operand::IdRef(a), Operand::IdRef(b)],
        ));
        id
    };
    let add = |ctx: &mut Ctx, op, a, b, out: &mut Vec<Instruction>| {
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            op,
            Some(float_ty),
            Some(id),
            vec![Operand::IdRef(a), Operand::IdRef(b)],
        ));
        id
    };
    let t2 = mul(ctx, t, t, out);
    let t3 = mul(ctx, t2, t, out);
    let w0a = mul(ctx, c_neg_half, t, out);
    let w0b = mul(ctx, c_one, t2, out);
    let w0c = mul(ctx, c_neg_half, t3, out);
    let w0ab = add(ctx, Op::FAdd, w0a, w0b, out);
    let w0 = add(ctx, Op::FAdd, w0ab, w0c, out);
    let w1a = mul(ctx, c_two_half, t2, out);
    let w1b = mul(ctx, c_one_half, t3, out);
    let w1a = add(ctx, Op::FSub, c_one, w1a, out);
    let w1 = add(ctx, Op::FAdd, w1a, w1b, out);
    let w2a = mul(ctx, c_half, t, out);
    let w2b = mul(ctx, c_two, t2, out);
    let w2c = mul(ctx, c_one_half, t3, out);
    let w2ab = add(ctx, Op::FAdd, w2a, w2b, out);
    let w2 = add(ctx, Op::FSub, w2ab, w2c, out);
    let w3a = mul(ctx, c_neg_half, t2, out);
    let w3b = mul(ctx, c_half, t3, out);
    let w3 = add(ctx, Op::FAdd, w3a, w3b, out);
    [w0, w1, w2, w3]
}

/// Clamp a float pixel-space sample coordinate component to a finite range before it is floored
/// and converted to an integer texel index. `OpConvertFToS` of inf/NaN is undefined and the
/// linear-filter fraction `inf - floor(inf)` is NaN, which no zero-weight guard can mask, while a
/// hardware sampler resolves a non-finite coordinate through the address mode like any other
/// far-out-of-range one. `NClamp` maps a NaN coordinate to the low bound deterministically.
///
/// *Which* finite range is safe depends on the axis address mode, because it is that mode which
/// resolves the integer texel downstream:
///
/// * The three CLAMP modes saturate, so `[-9.0, size + 9.0]` is free: sample offsets are bounded
///   to `[-8, 7]` texels by the AIR contract, so any value at or below -9 (or at or beyond
///   size + 9) still resolves purely through the address mode after the offset is applied, and
///   the downstream per-tap clamp / in-bounds guard yields the identical edge texel or border
///   zero for the clamped value.
/// * The two REPEAT modes wrap modulo the extent, so saturating first destroys the coordinate's
///   phase and answers the wrong texel — on an 8-wide texture a pixel coordinate of 25.5 wraps to
///   texel 1, and clamped to 17.0 it does not. Wrap by whole periods instead (`size` for `Repeat`,
///   `2 * size` for `MirroredRepeat`). `x - period * floor(x / period)` subtracts an exact integer
///   multiple of the period, so the fraction the filter weights are built from survives bit for
///   bit, and the result lands in `[0, period]` — which the saturating clamp would then break
///   again for a mirrored axis, so the two are alternatives and never composed.
///
/// The wrap does not make the downstream `SMod` redundant: the half-texel bias can still push the
/// base texel to `-1`, and a bicubic footprint still reaches a tap or two past the extent.
pub(in crate::passes) fn clamp_pixel_coord_component_finite(
    ctx: &mut Ctx,
    sampler_state: StaticSamplerState,
    comp: Word,
    size: Word,
    size_is_vector: bool,
    axis: u32,
    out: &mut Vec<Instruction>,
) -> Word {
    let float_ty = ctx.ty_float();
    let uint = ctx.ty_uint();
    let glsl = ctx.glsl();
    let size_axis = if size_is_vector {
        let c = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(c),
            vec![Operand::IdRef(size), Operand::LiteralBit32(axis)],
        ));
        c
    } else {
        size
    };
    let size_f = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ConvertUToF,
        Some(float_ty),
        Some(size_f),
        vec![Operand::IdRef(size_axis)],
    ));
    let period = match sampler_state.spatial_address_mode(axis as usize) {
        Some(SamplerAddressMode::Repeat) => Some(size_f),
        Some(SamplerAddressMode::MirroredRepeat) => {
            let two = ctx.const_float(2.0);
            let period = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FMul,
                Some(float_ty),
                Some(period),
                vec![Operand::IdRef(size_f), Operand::IdRef(two)],
            ));
            Some(period)
        }
        _ => None,
    };
    let Some(period) = period else {
        let slack = ctx.const_float(9.0);
        let hi = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FAdd,
            Some(float_ty),
            Some(hi),
            vec![Operand::IdRef(size_f), Operand::IdRef(slack)],
        ));
        let lo = ctx.const_float(-9.0);
        return nclamp_float(ctx, comp, lo, hi, out);
    };
    // Only to make the coordinate finite: 2^22 is far past any texture extent, and keeping
    // `period * floor(x / period)` well inside the f32 integer range is what keeps the wrap exact.
    let hi = ctx.const_float(4_194_304.0);
    let lo = ctx.const_float(-4_194_304.0);
    let finite = nclamp_float(ctx, comp, lo, hi, out);
    let quotient = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FDiv,
        Some(float_ty),
        Some(quotient),
        vec![Operand::IdRef(finite), Operand::IdRef(period)],
    ));
    let whole = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(whole),
        vec![
            Operand::IdRef(glsl),
            Operand::LiteralExtInstInteger(8), // Floor
            Operand::IdRef(quotient),
        ],
    ));
    let periods = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FMul,
        Some(float_ty),
        Some(periods),
        vec![Operand::IdRef(period), Operand::IdRef(whole)],
    ));
    let wrapped = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FSub,
        Some(float_ty),
        Some(wrapped),
        vec![Operand::IdRef(finite), Operand::IdRef(periods)],
    ));
    wrapped
}

/// `GLSLstd450NClamp(x, lo, hi)`, which resolves a NaN `x` to `lo`.
fn nclamp_float(ctx: &mut Ctx, x: Word, lo: Word, hi: Word, out: &mut Vec<Instruction>) -> Word {
    let float_ty = ctx.ty_float();
    let glsl = ctx.glsl();
    let clamped = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(clamped),
        vec![
            Operand::IdRef(glsl),
            Operand::LiteralExtInstInteger(81), // NClamp
            Operand::IdRef(x),
            Operand::IdRef(lo),
            Operand::IdRef(hi),
        ],
    ));
    clamped
}

#[allow(clippy::too_many_arguments)]
pub(in crate::passes) fn lower_pixel_linear_sample(
    ctx: &mut Ctx,
    sampler_state: StaticSamplerState,
    img: Word,
    dim: Dim,
    arrayed: bool,
    coord: Word,
    args: &[Word],
    lod: Word,
    sample_v4: Word,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let spatial: usize = match dim {
        Dim::Dim1D => 1,
        Dim::Dim2D => 2,
        Dim::Dim3D => 3,
        _ => return Err("air.sample_texture pixel linear sample unsupported dimension".into()),
    };
    let (offset, dynamic_offset) =
        sample_const_or_dynamic_offset(ctx, arrayed, args, spatial as u32)?;
    let dynamic_offset = dynamic_offset
        .map(|offset| dynamic_i32_integer_offset_components(ctx, offset, spatial, out))
        .transpose()?;
    let layer = if arrayed {
        Some(
            args.get(3)
                .copied()
                .ok_or("air.sample_texture array texture missing layer")?,
        )
    } else {
        None
    };
    let size = query_image_size(ctx, img, spatial, arrayed, lod, out);
    let float_ty = ctx.ty_float();
    let sint = ctx.ty_sint();
    let bool_ty = ctx.ty_bool();
    let glsl = ctx.glsl();
    let zero_f = ctx.const_float(0.0);
    let one_f = ctx.const_float(1.0);
    let half_f = ctx.const_float(0.5);
    let mut base = Vec::with_capacity(spatial);
    let mut frac = Vec::with_capacity(spatial);
    for (idx, comp) in sample_coord_components(ctx, coord, spatial as u32, out)?
        .into_iter()
        .enumerate()
    {
        let Operand::IdRef(comp) = comp else {
            return Err("air.sample_texture pixel coord component is not an id".into());
        };
        let comp = clamp_pixel_coord_component_finite(
            ctx,
            sampler_state,
            comp,
            size,
            spatial > 1 || arrayed,
            idx as u32,
            out,
        );
        let biased_coord = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(biased_coord),
            vec![Operand::IdRef(comp), Operand::IdRef(half_f)],
        ));
        let floor = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(floor),
            vec![
                Operand::IdRef(glsl),
                Operand::LiteralExtInstInteger(8),
                Operand::IdRef(biased_coord),
            ],
        ));
        let mut base_i = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ConvertFToS,
            Some(sint),
            Some(base_i),
            vec![Operand::IdRef(floor)],
        ));
        if let Some(offset) = &offset {
            let delta = offset[idx];
            if delta != 0 {
                let shifted = ctx.module.fresh_id();
                let delta = ctx.const_int_of(sint, delta as i64);
                out.push(Instruction::new(
                    Op::IAdd,
                    Some(sint),
                    Some(shifted),
                    vec![Operand::IdRef(base_i), Operand::IdRef(delta)],
                ));
                base_i = shifted;
            }
        } else if let Some(offset) = &dynamic_offset {
            let shifted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::IAdd,
                Some(sint),
                Some(shifted),
                vec![Operand::IdRef(base_i), Operand::IdRef(offset[idx])],
            ));
            base_i = shifted;
        }
        let frac_i = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(frac_i),
            vec![Operand::IdRef(biased_coord), Operand::IdRef(floor)],
        ));
        base.push(base_i);
        frac.push(frac_i);
    }

    // Per-axis lerp weights: [1 - frac, frac] for each spatial axis. The tap loop walks the
    // 2^spatial corner taps (bilinear for 2D, trilinear for 3D).
    let mut axis_weights = Vec::with_capacity(spatial);
    for &frac_axis in frac.iter().take(spatial) {
        let one_minus = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(one_minus),
            vec![Operand::IdRef(one_f), Operand::IdRef(frac_axis)],
        ));
        axis_weights.push([one_minus, frac_axis]);
    }
    let zero_color = const_null_of(ctx, sample_v4);
    let mut acc = zero_color;
    for tap in 0..(1usize << spatial) {
        let tap_offset: Vec<i32> = (0..spatial)
            .map(|axis| ((tap >> axis) & 1) as i32)
            .collect();
        let tap_coord = pixel_linear_tap_coord(
            ctx,
            sampler_state,
            dim,
            arrayed,
            &base,
            &tap_offset,
            layer,
            size,
            out,
        )?;
        let fetched = ctx.module.fresh_id();
        push_image_read_or_fetch(
            ctx,
            out,
            img,
            tap_coord.coord,
            Some(lod),
            sample_v4,
            fetched,
        )?;
        let color = if let Some(in_bounds) = tap_coord.in_bounds {
            let guarded = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Select,
                Some(sample_v4),
                Some(guarded),
                vec![
                    Operand::IdRef(in_bounds),
                    Operand::IdRef(fetched),
                    Operand::IdRef(zero_color),
                ],
            ));
            guarded
        } else {
            fetched
        };
        let mut weight = axis_weights[0][tap & 1];
        for (axis, weights) in axis_weights.iter().enumerate().skip(1) {
            let combined = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FMul,
                Some(float_ty),
                Some(combined),
                vec![
                    Operand::IdRef(weight),
                    Operand::IdRef(weights[(tap >> axis) & 1]),
                ],
            ));
            weight = combined;
        }
        let weighted = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::VectorTimesScalar,
            Some(sample_v4),
            Some(weighted),
            vec![Operand::IdRef(color), Operand::IdRef(weight)],
        ));
        let weight_is_zero = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(weight_is_zero),
            vec![Operand::IdRef(weight), Operand::IdRef(zero_f)],
        ));
        let contribution = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Select,
            Some(sample_v4),
            Some(contribution),
            vec![
                Operand::IdRef(weight_is_zero),
                Operand::IdRef(zero_color),
                Operand::IdRef(weighted),
            ],
        ));
        let sum = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FAdd,
            Some(sample_v4),
            Some(sum),
            vec![Operand::IdRef(acc), Operand::IdRef(contribution)],
        ));
        acc = sum;
    }
    Ok(acc)
}

pub(in crate::passes) fn finish_sample_result(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    color: Word,
    color_ty: Word,
    mut out: Vec<Instruction>,
) -> Result<Vec<Instruction>, String> {
    // The AIR sample result shape varies by texture component type:
    //   * f32 textures: `air.sample_texture_*.v4f32` returns a `{ <4 x float>, i8 }` STRUCT — the body
    //     then `CompositeExtract`s member 0. We rebuild that struct from the v4float color + undef i8.
    //   * half textures: `air.sample_texture_2d.v4f16` (and `air.read_texture_2d.v4f16`) return a bare
    //     `<4 x half>` VECTOR — we `OpFConvert` the v4float sample down to v4half and yield it directly.
    // Classify `rty` (struct vs vector) and emit the matching tail so both validate.
    let rdef = type_def_of(ctx, rty);
    let is_struct = rdef
        .as_ref()
        .map(|d| d.class.opcode == Op::TypeStruct)
        .unwrap_or(false);
    if is_struct {
        // Determine the struct's member-0 (color) type; if it is a half vector, convert first.
        let member0 = rdef
            .as_ref()
            .and_then(|d| d.operands.first())
            .and_then(|o| match o {
                Operand::IdRef(m) => Some(*m),
                _ => None,
            });
        let color_for_struct = match member0 {
            Some(m) if m != color_ty && is_half_vector(ctx, m) => {
                let c = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::FConvert,
                    Some(m),
                    Some(c),
                    vec![Operand::IdRef(color)],
                ));
                c
            }
            Some(m) => coerce_same_shape_integer(ctx, &mut out, color, color_ty, m)?,
            _ => color,
        };
        let i8u = ctx.ty_int8();
        let undef8 = ctx.module.fresh_id();
        out.push(Instruction::new(Op::Undef, Some(i8u), Some(undef8), vec![]));
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color_for_struct), Operand::IdRef(undef8)],
        ));
    } else if rty != color_ty && is_half_vector(ctx, rty) {
        // Bare half-vector result: FConvert the v4float sample to v4half.
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color)],
        ));
    } else if resolve::integer_shape(ctx, rty).is_some() {
        let c = coerce_same_shape_integer(ctx, &mut out, color, color_ty, rty)?;
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(c)],
        ));
    } else {
        // Bare v4float result: copy the color through.
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color)],
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Module;

    #[test]
    fn normalized_bicubic_scales_coordinates_and_emits_sixteen_fetches() {
        let mut ctx = Ctx::new(Module::new());
        let image_ty = ctx.ty_image(Dim::Dim2D, false, crate::passes::ImageComp::Float);
        let image = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Undef,
            Some(image_ty),
            Some(image),
            vec![],
        ));
        let coord_ty = ctx.ty_vecf(2);
        let x = ctx.const_float(0.25);
        let y = ctx.const_float(0.75);
        let coord = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::ConstantComposite,
            Some(coord_ty),
            Some(coord),
            vec![Operand::IdRef(x), Operand::IdRef(y)],
        ));
        let sample_v4 = ctx.ty_vech(4);
        let lod = ctx.const_uint(0);
        let sampler_state = StaticSamplerState::from_air_words([34901797601023049, 0])
            .expect("normalized bicubic AIR sampler state");
        assert!(sampler_state.uses_bicubic_filter());
        assert!(!sampler_state.uses_pixel_coordinates());

        let mut out = Vec::new();
        lower_pixel_bicubic_sample(
            &mut ctx,
            sampler_state,
            image,
            Dim::Dim2D,
            false,
            coord,
            &[image, 0, coord],
            lod,
            sample_v4,
            true,
            &mut out,
        )
        .expect("normalized bicubic lowering");

        assert_eq!(
            out.iter()
                .filter(|inst| inst.class.opcode == Op::ImageFetch)
                .count(),
            16
        );
        let size_ids = out
            .iter()
            .filter(|inst| inst.class.opcode == Op::ImageQuerySizeLod)
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        let extent_float_ids = out
            .iter()
            .filter(|inst| inst.class.opcode == Op::ConvertUToF)
            .filter(|inst| {
                let Some(Operand::IdRef(source)) = inst.operands.first() else {
                    return false;
                };
                out.iter().any(|candidate| {
                    candidate.result_id == Some(*source)
                        && candidate.class.opcode == Op::CompositeExtract
                        && matches!(candidate.operands.first(), Some(Operand::IdRef(id)) if size_ids.contains(id))
                })
            })
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        let scaled_extent_ids = extent_float_ids
            .iter()
            .filter(|extent| {
                out.iter().any(|inst| {
                    inst.class.opcode == Op::FMul
                        && inst
                            .operands
                            .iter()
                            .any(|operand| operand == &Operand::IdRef(**extent))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(scaled_extent_ids.len(), 2);
    }

    #[test]
    fn a_repeating_bicubic_wraps_the_coordinate_rather_than_clamping_it() {
        // A saturating clamp to `[-9, size + 9]` is only free for the three CLAMP address modes.
        // Under `address::repeat` it destroys the coordinate's phase, so both axes must wrap by
        // whole periods instead — and must not do both, since a wrap already lands in range.
        let mut ctx = Ctx::new(Module::new());
        let image_ty = ctx.ty_image(Dim::Dim2D, false, crate::passes::ImageComp::Float);
        let image = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Undef,
            Some(image_ty),
            Some(image),
            vec![],
        ));
        let coord_ty = ctx.ty_vecf(2);
        let x = ctx.const_float(3.0625);
        let y = ctx.const_float(0.4375);
        let coord = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::ConstantComposite,
            Some(coord_ty),
            Some(coord),
            vec![Operand::IdRef(x), Operand::IdRef(y)],
        ));
        let sample_v4 = ctx.ty_vecf(4);
        let lod = ctx.const_uint(0);
        // The word `constant sampler(filter::bicubic, address::repeat, coord::normalized)` compiles
        // to, taken from `validation/fixtures/public/kernel_bicubic_repeat_wrap.metal`.
        let sampler_state = StaticSamplerState::from_air_words([34901797601023122, 0])
            .expect("bicubic repeat AIR sampler state");
        assert!(sampler_state.uses_bicubic_filter());
        assert!(!sampler_state.uses_pixel_coordinates());
        for axis in 0..2 {
            assert_eq!(
                sampler_state.spatial_address_mode(axis),
                Some(SamplerAddressMode::Repeat)
            );
        }

        let mut out = Vec::new();
        lower_pixel_bicubic_sample(
            &mut ctx,
            sampler_state,
            image,
            Dim::Dim2D,
            false,
            coord,
            &[image, 0, coord],
            lod,
            sample_v4,
            true,
            &mut out,
        )
        .expect("bicubic repeat lowering");

        let float_constant = |value: f32, ctx: &Ctx| {
            ctx.new_globals.iter().any(|inst| {
                inst.class.opcode == Op::Constant
                    && inst.operands.first() == Some(&Operand::LiteralBit32(value.to_bits()))
            })
        };
        for slack in [9.0f32, -9.0f32] {
            assert!(
                !float_constant(slack, &ctx),
                "a repeating axis must not saturate to size {slack:+}"
            );
        }

        // One `x - period * floor(x / period)` per axis, and each period is the axis extent
        // rather than a constant, so the wrap tracks the texture it is actually sampling.
        let floored_quotients = out
            .iter()
            .filter(|inst| {
                inst.class.opcode == Op::ExtInst
                    && inst.operands.get(1) == Some(&Operand::LiteralExtInstInteger(8))
            })
            .filter(|inst| {
                let Some(Operand::IdRef(source)) = inst.operands.get(2) else {
                    return false;
                };
                out.iter().any(|candidate| {
                    candidate.result_id == Some(*source) && candidate.class.opcode == Op::FDiv
                })
            })
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        assert_eq!(floored_quotients.len(), 2);
        let wraps = out
            .iter()
            .filter(|inst| inst.class.opcode == Op::FSub)
            .filter(|inst| {
                let Some(Operand::IdRef(subtrahend)) = inst.operands.get(1) else {
                    return false;
                };
                out.iter().any(|candidate| {
                    candidate.result_id == Some(*subtrahend)
                        && candidate.class.opcode == Op::FMul
                        && candidate.operands.iter().any(|operand| {
                            matches!(operand, Operand::IdRef(id) if floored_quotients.contains(id))
                        })
                })
            })
            .count();
        assert_eq!(wraps, 2);
    }

    #[test]
    fn normalized_bicubic_1d_emits_four_fetches() {
        let mut ctx = Ctx::new(Module::new());
        let image_ty = ctx.ty_image(Dim::Dim1D, false, crate::passes::ImageComp::Float);
        let image = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Undef,
            Some(image_ty),
            Some(image),
            vec![],
        ));
        let coord = ctx.const_float(0.25);
        let sample_v4 = ctx.ty_vech(4);
        let lod = ctx.const_uint(0);
        let sampler_state = StaticSamplerState::from_air_words([34901797601023049, 0])
            .expect("normalized bicubic AIR sampler state");
        let mut out = Vec::new();
        lower_pixel_bicubic_sample(
            &mut ctx,
            sampler_state,
            image,
            Dim::Dim1D,
            false,
            coord,
            &[image, 0, coord],
            lod,
            sample_v4,
            true,
            &mut out,
        )
        .expect("normalized 1D bicubic lowering");
        assert_eq!(
            out.iter()
                .filter(|inst| inst.class.opcode == Op::ImageFetch)
                .count(),
            4
        );
    }
}
