//! Floating-point and imageblock AIR call lowering.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImageblockConversion {
    None,
    Float,
    Bitcast,
}

pub(in crate::passes) fn lower_implicit_imageblock_load(
    ctx: &mut Ctx,
    name: &str,
    result: Word,
    result_ty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 4 {
        return Err(format!(
            "{name} expects attachment, coordinate, index, and data rate"
        ));
    }
    let (attachment, data_rate) = implicit_imageblock_constants(ctx, name, args)?;
    let (format, comp, storage_ty, conversion, lanes) =
        implicit_imageblock_format(ctx, name, result_ty)?;
    let (var, image_ty) = ctx.implicit_imageblock_var(attachment, data_rate, format, comp)?;
    let mut out = Vec::new();
    let image = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Load,
        Some(image_ty),
        Some(image),
        vec![Operand::IdRef(var)],
    ));
    ctx.image_dims.insert(image, (Dim::Dim2D, true));
    ctx.image_comp.insert(image, comp);
    ctx.image_storage.insert(image);
    let coord = build_fetch_coord(ctx, Dim::Dim2D, true, args[1], Some(args[2]), &mut out)?;
    let read_result = if conversion != ImageblockConversion::None || lanes != 4 {
        ctx.module.fresh_id()
    } else {
        result
    };
    out.push(Instruction::new(
        Op::ImageRead,
        Some(storage_ty),
        Some(read_result),
        vec![Operand::IdRef(image), Operand::IdRef(coord)],
    ));
    let (converted_source, converted_source_ty) = if lanes == 1 {
        let component = if conversion != ImageblockConversion::None {
            ctx.module.fresh_id()
        } else {
            result
        };
        let component_ty = match comp {
            ImageComp::Float => ctx.ty_float(),
            ImageComp::Uint => ctx.ty_uint(),
            ImageComp::Sint => ctx.ty_sint(),
        };
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(component_ty),
            Some(component),
            vec![Operand::IdRef(read_result), Operand::LiteralBit32(0)],
        ));
        (component, component_ty)
    } else if lanes < 4 {
        let prefix = ctx.module.fresh_id();
        let prefix_ty = match comp {
            ImageComp::Float => ctx.ty_vecf(lanes),
            ImageComp::Uint => ctx.ty_vec_uint(lanes),
            ImageComp::Sint => ctx.ty_vec_sint(lanes),
        };
        out.push(Instruction::new(
            Op::VectorShuffle,
            Some(prefix_ty),
            Some(prefix),
            std::iter::once(Operand::IdRef(read_result))
                .chain(std::iter::once(Operand::IdRef(read_result)))
                .chain((0..lanes).map(Operand::LiteralBit32))
                .collect(),
        ));
        (prefix, prefix_ty)
    } else {
        (read_result, storage_ty)
    };
    if conversion != ImageblockConversion::None {
        let instruction = match conversion {
            ImageblockConversion::Float => Instruction::new(
                Op::FConvert,
                Some(result_ty),
                Some(result),
                vec![Operand::IdRef(converted_source)],
            ),
            ImageblockConversion::Bitcast => {
                copy_or_bitcast_result(result_ty, result, converted_source_ty, converted_source)
            }
            ImageblockConversion::None => unreachable!(),
        };
        out.push(instruction);
    }
    Ok(out)
}

pub(in crate::passes) fn lower_implicit_imageblock_store(
    ctx: &mut Ctx,
    name: &str,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 5 {
        return Err(format!(
            "{name} expects value, attachment, coordinate, index, and data rate"
        ));
    }
    let abi_args = &args[1..];
    let (attachment, data_rate) = implicit_imageblock_constants(ctx, name, abi_args)?;
    let value_ty = value_result_type(ctx, args[0])
        .ok_or_else(|| format!("{name} value has no result type"))?;
    let (format, comp, storage_ty, conversion, lanes) =
        implicit_imageblock_format(ctx, name, value_ty)?;
    let (var, image_ty) = ctx.implicit_imageblock_var(attachment, data_rate, format, comp)?;
    let mut out = Vec::new();
    let image = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Load,
        Some(image_ty),
        Some(image),
        vec![Operand::IdRef(var)],
    ));
    ctx.image_dims.insert(image, (Dim::Dim2D, true));
    ctx.image_comp.insert(image, comp);
    ctx.image_storage.insert(image);
    let coord = build_fetch_coord(
        ctx,
        Dim::Dim2D,
        true,
        abi_args[1],
        Some(abi_args[2]),
        &mut out,
    )?;
    let value = if conversion != ImageblockConversion::None {
        let storage_value_ty = if lanes == 1 {
            match conversion {
                ImageblockConversion::Float => ctx.ty_float(),
                ImageblockConversion::Bitcast => match comp {
                    ImageComp::Float => ctx.ty_float(),
                    ImageComp::Uint => ctx.ty_uint(),
                    ImageComp::Sint => ctx.ty_sint(),
                },
                ImageblockConversion::None => unreachable!(),
            }
        } else if lanes < 4 {
            match comp {
                ImageComp::Float => ctx.ty_vecf(lanes),
                ImageComp::Uint => ctx.ty_vec_uint(lanes),
                ImageComp::Sint => ctx.ty_vec_sint(lanes),
            }
        } else {
            storage_ty
        };
        let converted = ctx.module.fresh_id();
        let instruction = match conversion {
            ImageblockConversion::Float => Instruction::new(
                Op::FConvert,
                Some(storage_value_ty),
                Some(converted),
                vec![Operand::IdRef(args[0])],
            ),
            ImageblockConversion::Bitcast => {
                copy_or_bitcast_result(storage_value_ty, converted, value_ty, args[0])
            }
            ImageblockConversion::None => unreachable!(),
        };
        out.push(instruction);
        converted
    } else {
        args[0]
    };
    let value = if lanes == 1 {
        let texel = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(storage_ty),
            Some(texel),
            vec![Operand::IdRef(value); 4],
        ));
        texel
    } else if lanes < 4 {
        let texel = ctx.module.fresh_id();
        let operands = std::iter::once(Operand::IdRef(value))
            .chain(std::iter::once(Operand::IdRef(value)))
            .chain(
                (0..4)
                    .map(|lane| Operand::LiteralBit32(if lane < lanes { lane } else { u32::MAX })),
            )
            .collect();
        out.push(Instruction::new(
            Op::VectorShuffle,
            Some(storage_ty),
            Some(texel),
            operands,
        ));
        texel
    } else {
        value
    };
    out.push(Instruction::new(
        Op::ImageWrite,
        None,
        None,
        vec![
            Operand::IdRef(image),
            Operand::IdRef(coord),
            Operand::IdRef(value),
        ],
    ));
    Ok(out)
}

fn implicit_imageblock_constants(
    ctx: &Ctx,
    name: &str,
    args: &[Word],
) -> Result<(u32, u32), String> {
    let attachment = constant_u32(ctx, args[0])
        .ok_or_else(|| format!("{name} requires a constant render-target attachment"))?;
    let data_rate = constant_u32(ctx, args[3])
        .ok_or_else(|| format!("{name} requires a constant imageblock data rate"))?;
    Ok((attachment, data_rate))
}

fn implicit_imageblock_format(
    ctx: &mut Ctx,
    name: &str,
    value_ty: Word,
) -> Result<(ImageFormat, ImageComp, Word, ImageblockConversion, u32), String> {
    use crate::meta::TextureFormat;

    let format = crate::meta::implicit_imageblock_texture_format(name)?
        .ok_or_else(|| format!("{name} is not an implicit imageblock intrinsic"))?;
    let lowered = match format {
        TextureFormat::R16f => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::R16f,
                ImageComp::Float,
                ctx.ty_vecf(4),
                ImageblockConversion::Float,
                1,
            )
        }
        TextureFormat::Rg16f => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::Rg16f,
                ImageComp::Float,
                ctx.ty_vecf(4),
                ImageblockConversion::Float,
                2,
            )
        }
        TextureFormat::Rgba16f => (
            ImageFormat::Rgba16f,
            ImageComp::Float,
            ctx.ty_vecf(4),
            ImageblockConversion::Float,
            4,
        ),
        TextureFormat::R32f => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::R32f,
                ImageComp::Float,
                ctx.ty_vecf(4),
                ImageblockConversion::None,
                1,
            )
        }
        TextureFormat::Rgba32f => (
            ImageFormat::Rgba32f,
            ImageComp::Float,
            value_ty,
            ImageblockConversion::None,
            4,
        ),
        TextureFormat::R32ui => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::R32ui,
                ImageComp::Uint,
                ctx.ty_vec_uint(4),
                ImageblockConversion::Bitcast,
                1,
            )
        }
        _ => return Err(format!("{name} has unsupported implicit imageblock format")),
    };
    Ok(lowered)
}

fn require_storage_image_extended_formats(ctx: &mut Ctx) {
    let capability = spirv::Capability::StorageImageExtendedFormats;
    if ctx
        .module
        .capabilities
        .iter()
        .any(|instruction| instruction.operands.as_slice() == [Operand::Capability(capability)])
    {
        return;
    }
    ctx.module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(capability)],
    ));
}

/// Lower the residual floating-point / transcendental AIR intrinsic family and the generic
/// GLSL.std.450 ext-inst tail (sincos, cospi/sinpi/tanpi, exp10, fmod, log10, integer & float
/// min/max/clamp, min3/max3/fmedian3, bfloat fmuladd, ldexp, fast_tanh, the `glsl_extinst` map,
/// saturate). This is the terminal dispatch stage of `lower_one`: every call that reached here is
/// either handled or rejected with the "unhandled air.* intrinsic" error. Guard order is preserved.
pub(in crate::passes) fn lower_float_math(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if (name.starts_with("air.fast_sincos.") || name.starts_with("air.sincos.")) && args.len() == 2
    {
        let ext = ctx.glsl();
        let cos = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Sin as u32),
                    Operand::IdRef(args[0]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(cos),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Cos as u32),
                    Operand::IdRef(args[0]),
                ],
            ),
            Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(args[1]), Operand::IdRef(cos)],
            ),
        ]);
    }
    // `sinpi`/`cospi` are NOT `sin(pi*x)`/`cos(pi*x)`; see `lower_metal_pi_scaled_sin_cos`.
    if (name.starts_with("air.cospi.") || name.starts_with("air.fast_cospi.")) && args.len() == 1 {
        return lower_metal_pi_scaled_sin_cos(ctx, res, rty, args[0], PiScaled::Cos);
    }
    if (name.starts_with("air.sinpi.") || name.starts_with("air.fast_sinpi.")) && args.len() == 1 {
        return lower_metal_pi_scaled_sin_cos(ctx, res, rty, args[0], PiScaled::Sin);
    }
    // `tanpi` is NOT `tan(pi*x)` either, and by a wider margin than sinpi/cospi; see
    // `lower_metal_pi_scaled_sin_cos`.
    if (name.starts_with("air.tanpi.") || name.starts_with("air.fast_tanpi.")) && args.len() == 1 {
        return lower_metal_pi_scaled_sin_cos(ctx, res, rty, args[0], PiScaled::Tan);
    }
    // GLSL.std.450 has no Exp10, so this composes one. Which composition does not matter for
    // accuracy: `powr(10,x)`, `pow(10,x)` and `exp2(x * log2(10))` are all BIT-IDENTICAL to Metal's
    // `exp10` -- on all 2077 float arguments of a sweep over [-40,40] plus the round integers, and
    // on all 63488 finite halves (device-measured, Apple M3 Max / macOS 26.5.2, one spelling per
    // kernel). `Pow(10, x)` is chosen because it is one instruction instead of two and works on
    // vectors, which the pre-multiply form did not.
    //
    // Metal does NOT return the exact round powers: `exp10(2)` is 99.99999 and `exp10(3)` is
    // 999.9998, under both the default fast math and `Pow`. Reproducing that is the contract.
    if (name.starts_with("air.exp10.") || name.starts_with("air.fast_exp10.")) && args.len() == 1 {
        let ext = ctx.glsl();
        let n = vector_len(ctx, rty);
        let ten = splat_or_scalar(ctx, rty, 10.0, n);
        return Ok(vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Pow as u32),
                Operand::IdRef(ten),
                Operand::IdRef(args[0]),
            ],
        )]);
    }
    // Metal's fmod is the EXPRESSION `x - y * trunc(x / y)`, not the exact C remainder, and this
    // is the one hand-rolled wrapper here that must NOT be replaced by the primitive that looks
    // like it. SPIR-V's `OpFRem` is the exact remainder with the dividend's sign -- textbook C
    // `fmod` -- and `OpFMod` is the floor-style modulus; both would be wrong. Once the quotient is
    // large enough to round, Metal's answer is the residue of a ROUNDED trunc, which is nowhere
    // near the exact remainder: measured on this repository's reference GPU by the authored
    // `kernel_fast_fmod_large_quotient`, `fast::fmod(1e10, 3)` is -512 where the exact remainder is
    // 1, `fast::fmod(1e20, 3)` is -4398046511104 where it is 2, and `fast::fmod(1e6, 0.1)` is
    // -0.0149 where it is 0.0851. SPIRV-Cross renders these four instructions back into literally
    // `x - (y * trunc(x / y))`, which is why the candidate reproduces Metal bit for bit. 504 corpus
    // sources call this family. The same operations are valid for scalar and vector float types.
    if name.starts_with("air.fast_fmod.") || name.starts_with("air.fmod.") {
        if args.len() != 2 {
            return Err(format!("{name} expects two operands"));
        }
        let ext = ctx.glsl();
        let quotient = ctx.module.fresh_id();
        let truncated = ctx.module.fresh_id();
        let product = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(
                Op::FDiv,
                Some(rty),
                Some(quotient),
                vec![Operand::IdRef(args[0]), Operand::IdRef(args[1])],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(truncated),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Trunc as u32),
                    Operand::IdRef(quotient),
                ],
            ),
            Instruction::new(
                Op::FMul,
                Some(rty),
                Some(product),
                vec![Operand::IdRef(args[1]), Operand::IdRef(truncated)],
            ),
            Instruction::new(
                Op::FSub,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(args[0]), Operand::IdRef(product)],
            ),
        ]);
    }
    // GLSL.std.450 has no Log10 either, and here the composition DOES matter. Metal's `log10` is
    // not `log(x) / ln(10)`: device-measured on an Apple M3 Max / macOS 26.5.2, that form
    // disagrees with `log10` on 1995 of a 2011-argument float sweep, and it is visibly wrong at
    // the arguments a shader is most likely to pass -- `log10(100)` comes back 1.9999999 where
    // Metal returns exactly 2. `log2(x) * log10(2)` agrees on all 2011, and reproduces every edge:
    // `+-0` -> `-inf`, `+inf` -> `+inf`, a negative argument and `-inf` -> NaN.
    //
    // At half width the multiply must NOT be done in half -- see `lower_scaled_glsl_unary`.
    if (name.starts_with("air.fast_log10.") || name.starts_with("air.log10.")) && args.len() == 1 {
        return lower_post_scaled_glsl_unary(
            ctx,
            res,
            rty,
            args[0],
            std::f32::consts::LOG10_2,
            GLSLstd450::Log2,
        );
    }
    // Integer min/max use GLSL's integer ext-inst variants. The generic `air.min`/`air.max`
    // matcher below is float-only; using FMin/FMax for e.g. `air.min.s.i16` emits invalid
    // SPIR-V (`FMin` with a ushort result).
    if args.len() == 2 {
        let int_minmax = if name.starts_with("air.min.s.") {
            Some(GLSLstd450::SMin)
        } else if name.starts_with("air.min.u.") {
            Some(GLSLstd450::UMin)
        } else if name.starts_with("air.max.s.") {
            Some(GLSLstd450::SMax)
        } else if name.starts_with("air.max.u.") {
            Some(GLSLstd450::UMax)
        } else {
            None
        };
        if let Some(op) = int_minmax {
            if matches!(op, GLSLstd450::SMin | GLSLstd450::SMax) {
                return lower_signed_integer_minmax(ctx, res, rty, args, op);
            }
            let ext = ctx.glsl();
            return Ok(vec![Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(op as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            )]);
        }
    }
    // Integer clamp uses GLSL's integer clamp variants. The generic `clamp` matcher below is
    // float-only; using FClamp for e.g. `air.clamp.u.i32` emits invalid SPIR-V.
    if args.len() == 3 {
        let int_clamp = if name.starts_with("air.clamp.s.") {
            Some(GLSLstd450::SClamp)
        } else if name.starts_with("air.clamp.u.") {
            Some(GLSLstd450::UClamp)
        } else {
            None
        };
        if let Some(op) = int_clamp {
            let ext = ctx.glsl();
            return Ok(vec![Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(op as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                    Operand::IdRef(args[2]),
                ],
            )]);
        }
    }
    // AIR's `*min3`/`*max3` helpers are ternary. GLSL.std.450 min/max ops are binary, so
    // fold them into two ext-inst operations instead of forwarding all three operands to one
    // invalid OpExtInst.
    if args.len() == 3 {
        let ternary_minmax = if name.contains("fmax3") {
            Some(GLSLstd450::FMax)
        } else if name.contains("fmin3") {
            Some(GLSLstd450::FMin)
        } else if name.starts_with("air.max3.u.") {
            Some(GLSLstd450::UMax)
        } else if name.starts_with("air.min3.u.") {
            Some(GLSLstd450::UMin)
        } else if name.starts_with("air.max3.s.") {
            Some(GLSLstd450::SMax)
        } else if name.starts_with("air.min3.s.") {
            Some(GLSLstd450::SMin)
        } else {
            None
        }
        .map(|op| nan_aware_if_precise(name, op));
        if let Some(op) = ternary_minmax {
            let ext = ctx.glsl();
            let tmp = ctx.module.fresh_id();
            return Ok(vec![
                Instruction::new(
                    Op::ExtInst,
                    Some(rty),
                    Some(tmp),
                    vec![
                        Operand::IdRef(ext),
                        Operand::LiteralExtInstInteger(op as u32),
                        Operand::IdRef(args[0]),
                        Operand::IdRef(args[1]),
                    ],
                ),
                Instruction::new(
                    Op::ExtInst,
                    Some(rty),
                    Some(res),
                    vec![
                        Operand::IdRef(ext),
                        Operand::LiteralExtInstInteger(op as u32),
                        Operand::IdRef(tmp),
                        Operand::IdRef(args[2]),
                    ],
                ),
            ]);
        }
    }
    if name.contains("fmedian3") && args.len() == 3 {
        let ext = ctx.glsl();
        // `fmedian3` is built out of the same `fmin`/`fmax` the plain symbol names, so it carries
        // the same NaN contract; see `nan_aware_if_precise`.
        let fmin = nan_aware_if_precise(name, GLSLstd450::FMin);
        let fmax = nan_aware_if_precise(name, GLSLstd450::FMax);
        let min_ab = ctx.module.fresh_id();
        let max_ab = ctx.module.fresh_id();
        let min_max_ab_c = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(min_ab),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(fmin as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(max_ab),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(fmax as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(min_max_ab_c),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(fmin as u32),
                    Operand::IdRef(max_ab),
                    Operand::IdRef(args[2]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(fmax as u32),
                    Operand::IdRef(min_ab),
                    Operand::IdRef(min_max_ab_c),
                ],
            ),
        ]);
    }
    if is_llvm_bfloat_fmuladd(name) {
        return lower_bfloat_fmuladd(ctx, name, res, rty, args);
    }
    if is_air_math(name, "ldexp") && args.len() == 2 {
        return lower_fast_ldexp(ctx, name, res, rty, args);
    }
    // Metal's FAST tanh is exp2-based — tanh(x) = (t - 1)/(t + 1) with t = exp2(x * 2*log2(e)) —
    // so t overflows to +inf for large positive x and the quotient is inf/inf = NaN. Apple goldens
    // carry those NaNs (the Espresso batchnorm family), where GLSL Tanh saturates to 1.0. Emit the
    // faithful formula for the fast variant only; precise air.tanh stays GLSL Tanh below.
    if name.starts_with("air.fast_tanh.") && args.len() == 1 {
        return lower_fast_tanh(ctx, res, rty, args[0]);
    }
    if name.starts_with("air.fast_cos.") && args.len() == 1 && is_f32_scalar_or_vector(ctx, rty) {
        return Ok(lower_fast_trig(ctx, res, rty, args[0], GLSLstd450::Cos));
    }
    if name.starts_with("air.fast_sin.") && args.len() == 1 && is_f32_scalar_or_vector(ctx, rty) {
        return Ok(lower_fast_trig(ctx, res, rty, args[0], GLSLstd450::Sin));
    }
    // GLSL.std.450 ext-inst math.
    if let Some(glsl_op) = glsl_extinst(name) {
        if args.len() == 1
            && glsl_op == GLSLstd450::Round
            && (is_f32_scalar_or_vector(ctx, rty) || is_half_scalar_or_vector(ctx, rty))
        {
            return Ok(lower_metal_round(ctx, res, rty, args[0]));
        }
        if args.len() == 1 && matches!(glsl_op, GLSLstd450::Round | GLSLstd450::RoundEven) {
            return Ok(half_glsl_op(ctx, glsl_op, res, rty, args));
        }
        // A half `Tanh`/`Atan2` reaches Metal's `fast::` variant through SPIRV-Cross while the float
        // one reaches `precise::`; see `half_glsl_op`. `fast::tanh(44)` is 0 and `fast::tanh(50)` is
        // NaN where the oracle's `tanh(half)` is 1.0, so the half width is a different function and
        // the op has to be computed at float width to be the one AIR named. Only for the PRECISE
        // symbol: `air.fast_atan2.f16` is asking for exactly the `fast::` variant the half width
        // already reaches, and widening it would walk away from the function it named.
        if matches!(glsl_op, GLSLstd450::Tanh | GLSLstd450::Atan2)
            && !name.starts_with("air.fast_")
            && args.len() == usize::from(glsl_op == GLSLstd450::Atan2) + 1
        {
            return Ok(half_glsl_op(ctx, glsl_op, res, rty, args));
        }
        // GLSL Pow cannot be handed a negative base; Metal's `pow` defines one. `powr` (x >= 0 by
        // contract) is a different AIR symbol and keeps the plain ext-inst below.
        if args.len() == 2
            && is_air_math(name, "pow")
            && (is_half_scalar_or_vector(ctx, rty) || is_f32_scalar_or_vector(ctx, rty))
        {
            return Ok(lower_metal_pow(ctx, res, rty, args[0], args[1]));
        }
        let ext = ctx.glsl();
        let mut ops = vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(glsl_op as u32),
        ];
        for a in args {
            ops.push(Operand::IdRef(*a));
        }
        return Ok(vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            ops,
        )]);
    }
    // saturate(x) = clamp(x, 0, 1); needs synthesized 0/1 constants of the result ELEMENT type
    // (half `air.saturate.f16` -> half 0/1; float -> float 0/1; else spirv-val rejects the mismatch).
    if name.contains("saturate") && args.len() == 1 {
        let ext = ctx.glsl();
        // `saturate` is `clamp(x, 0, 1)` and carries the plain clamp's NaN contract with it.
        let clamp = nan_aware_if_precise(name, GLSLstd450::FClamp);
        let (zero, one) = scalar_zero_one(ctx, rty);
        // The clamp on a vector takes vector clamp operands; GLSL accepts scalar edges only for
        // scalar x. For vectors we must splat — build constant composites.
        let (lo, hi) = clamp_edges(ctx, rty, zero, one);
        return Ok(vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(clamp as u32),
                Operand::IdRef(args[0]),
                Operand::IdRef(lo),
                Operand::IdRef(hi),
            ],
        )]);
    }

    Err(format!("unhandled air.* intrinsic: {name}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PiScaled {
    Sin,
    Cos,
    Tan,
}

/// Metal's `sinpi(x)` and `cospi(x)` are NOT `sin(float(pi) * x)` and `cos(float(pi) * x)`.
///
/// `float(pi)` is 3.1415927410125732, which is 8.74e-08 above pi, so the pre-multiply puts the
/// argument that far past the zero crossing and the answer at every half-integer -- the arguments
/// a shader is most likely to pass -- is a small number instead of an exact 0 or +-1. Measured on
/// this repository's reference GPU against `precise::sinpi`/`precise::cospi`: `sinpi(1)` is -0.0 on
/// device and the pre-multiply gives -8.74e-08, `cospi(0.5)` is 0.0 and gives -4.37e-08,
/// `cospi(100.5)` is 0.0 and gives -1.03e-05. Worse, the pre-multiply hands a LARGE argument to
/// `sin`/`cos`, which flushes to 0.0 above float(pi/2)*2^22 (see `lower_fast_trig`), so
/// `cospi(1e7)` came back 0.0 where Metal answers 1.0. Over an 80-point grid of the two functions
/// the pre-multiply was wrong in 40 lanes; the reduction below is wrong in 6, all by one ULP at
/// arguments that are not multiples of 0.5.
///
/// Both functions have period 2, and 2 is a power of two, so the reduction is EXACT:
///
/// ```text
/// x2 = x - 2*roundEven(x/2)          in [-1, 1], exact
/// n  = roundEven(2*x2)               in {-2,-1,0,1,2}
/// r  = x2 - n/2                      in [-0.25, 0.25], exact
/// sinpi(x) = [sin, cos, -sin, -cos][n mod 4](pi*r)
/// cospi(x) = [cos, -sin, -cos, sin][n mod 4](pi*r)
/// ```
///
/// `n` never leaves {-2,-1,0,1,2}, so the quadrant is selected by three float compares rather than
/// an integer conversion that would overflow for a large `x`. Only `pi*r` is inexact, and `|r|` is
/// at most 0.25, which is why a half-integer argument lands on an exact 0 or +-1: `r` is 0 there
/// and `sin(0)`/`cos(0)` are exact.
///
/// Two measured sign details. `sinpi` returns -0.0 at odd integers, which the `-sin(pi*r)` arm
/// reproduces because IEEE negation of +0.0 is -0.0. `cospi` returns +0.0 at EVERY half-integer,
/// including the ones whose quadrant arm is a negated sine, where the sine is +0.0; that arm
/// therefore selects the +0.0 constant when the residue is 0. Writing it as `0.0 - sin(pi*r)`
/// instead is not enough: Metal's compiler folds `0.0 - x` back into a negation under its default
/// no-signed-zeros fast math, and five lanes came back -0.0 when it was tried.
///
/// A non-finite argument answers the canonical quiet NaN 0x7fc00000, which is what Metal returns
/// for all of `+-inf` and NaN. Without the guard the reduction would answer +-0.0 there, because
/// `x - x` is NaN and MSL's `sin` flushes a NaN to 0.0.
///
/// `tanpi(x)` is the quotient of the two, and is where the pre-multiply is most visibly wrong.
/// `tan(float(pi) * x)` is not merely imprecise: it has no pole at all, because `float(pi)*0.5`
/// misses pi/2. Device-measured over 4091 arguments -- a 1/1000 grid on [-2,2] plus the quarter
/// integers and the edges -- `tan(pi*x)` disagrees with `tanpi` on 3184, and at the half-integers
/// it returns a large finite number of the WRONG SIGN where Metal returns an infinity:
/// `tanpi(0.5)` is +inf and `tan(pi*0.5)` is -2.28e7. `sinpi(x)/cospi(x)` disagrees on 98, none by
/// more than 1e-5 relative, and reproduces every pole, both signed zeros, the large arguments
/// (`tanpi(12345.5)` is -inf), `+-inf` and NaN exactly.
///
/// Dividing the two selected quadrant arms is what puts the poles in: at a half-integer the
/// residue is 0, so the numerator is an exact +-1 and the denominator is the +0.0 the cospi arm
/// already selects there, and IEEE division gives the correctly signed infinity. At an integer the
/// numerator is the -0.0 the sinpi arm gives and the denominator is +-1, so `tanpi` is an exact
/// signed zero. Nothing about the pole handling is written down here twice -- it falls out of the
/// two arms that were already device-verified.
///
/// `precise::tanpi` and `fast::tanpi` agree on all 4092 measured arguments, so both AIR spellings
/// take this one lowering.
fn lower_metal_pi_scaled_sin_cos(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    x: Word,
    which: PiScaled,
) -> Result<Vec<Instruction>, String> {
    let float_ty = float_equivalent(ctx, rty);
    match type_def_of(ctx, float_ty) {
        Some(def) if def.class.opcode == Op::TypeFloat => {}
        _ => return Err("pi-scaled sine/cosine currently supports scalar float only".to_string()),
    }
    let ext = ctx.glsl();
    let bool_ty = ctx.ty_bool();
    let half = ctx.const_float(0.5);
    let two = ctx.const_float(2.0);
    let pi = ctx.const_float(std::f32::consts::PI);
    let zero = ctx.const_float(0.0);
    let one = ctx.const_float(1.0);
    let minus_one = ctx.const_float(-1.0);
    let infinity = ctx.const_float(f32::INFINITY);
    let quiet_nan = ctx.const_float(f32::from_bits(0x7fc0_0000));
    let mut out = Vec::new();

    let emit =
        |ctx: &mut Ctx, out: &mut Vec<Instruction>, op: Op, operands: Vec<Operand>| -> Word {
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(op, Some(float_ty), Some(id), operands));
            id
        };
    let xf = if float_ty == rty {
        x
    } else {
        emit(ctx, &mut out, Op::FConvert, vec![Operand::IdRef(x)])
    };

    let round_even = |ctx: &mut Ctx, out: &mut Vec<Instruction>, value: Word| -> Word {
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(id),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::RoundEven as u32),
                Operand::IdRef(value),
            ],
        ));
        id
    };

    // x2 = x - 2*roundEven(x/2): both multiplications are by a power of two and the subtraction
    // cancels to at most 1, so nothing here rounds.
    let scaled = emit(
        ctx,
        &mut out,
        Op::FMul,
        vec![Operand::IdRef(xf), Operand::IdRef(half)],
    );
    let whole = round_even(ctx, &mut out, scaled);
    let doubled = emit(
        ctx,
        &mut out,
        Op::FMul,
        vec![Operand::IdRef(whole), Operand::IdRef(two)],
    );
    let reduced = emit(
        ctx,
        &mut out,
        Op::FSub,
        vec![Operand::IdRef(xf), Operand::IdRef(doubled)],
    );
    let twice = emit(
        ctx,
        &mut out,
        Op::FMul,
        vec![Operand::IdRef(reduced), Operand::IdRef(two)],
    );
    let quadrant = round_even(ctx, &mut out, twice);
    let offset = emit(
        ctx,
        &mut out,
        Op::FMul,
        vec![Operand::IdRef(quadrant), Operand::IdRef(half)],
    );
    let residue = emit(
        ctx,
        &mut out,
        Op::FSub,
        vec![Operand::IdRef(reduced), Operand::IdRef(offset)],
    );
    let angle = emit(
        ctx,
        &mut out,
        Op::FMul,
        vec![Operand::IdRef(pi), Operand::IdRef(residue)],
    );

    let sine = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(sine),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::Sin as u32),
            Operand::IdRef(angle),
        ],
    ));
    let cosine = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(cosine),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::Cos as u32),
            Operand::IdRef(angle),
        ],
    ));

    let compare =
        |ctx: &mut Ctx, out: &mut Vec<Instruction>, quadrant: Word, against: Word| -> Word {
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FOrdEqual,
                Some(bool_ty),
                Some(id),
                vec![Operand::IdRef(quadrant), Operand::IdRef(against)],
            ));
            id
        };
    let at_zero = compare(ctx, &mut out, quadrant, zero);
    let at_one = compare(ctx, &mut out, quadrant, one);
    let at_minus_one = compare(ctx, &mut out, quadrant, minus_one);

    // `quadrant` is +-2 in the arm each of these falls through to, and +-2 select the same value.
    let sine_arms = |ctx: &mut Ctx, out: &mut Vec<Instruction>| -> (Word, Word, Word, Word) {
        let negated_sine = emit(ctx, out, Op::FNegate, vec![Operand::IdRef(sine)]);
        let negated_cosine = emit(ctx, out, Op::FNegate, vec![Operand::IdRef(cosine)]);
        (sine, cosine, negated_cosine, negated_sine)
    };
    // Metal's cospi answers +0.0 at EVERY half-integer, including the ones whose quadrant arm is a
    // negated sine, and the sine is +0.0 there. Neither `-sin` nor `0.0 - sin` survives that:
    // negation makes -0.0, and Metal's compiler folds `0.0 - x` back into a negation under its
    // default no-signed-zeros fast math. Select the +0.0 constant on the residue instead, which is
    // the only argument at which the two differ.
    let cosine_arms = |ctx: &mut Ctx, out: &mut Vec<Instruction>| -> (Word, Word, Word, Word) {
        let negated_sine = emit(ctx, out, Op::FNegate, vec![Operand::IdRef(sine)]);
        let at_axis = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(at_axis),
            vec![Operand::IdRef(residue), Operand::IdRef(zero)],
        ));
        let signed_zero = emit(
            ctx,
            out,
            Op::Select,
            vec![
                Operand::IdRef(at_axis),
                Operand::IdRef(zero),
                Operand::IdRef(negated_sine),
            ],
        );
        let negated_cosine = emit(ctx, out, Op::FNegate, vec![Operand::IdRef(cosine)]);
        (cosine, signed_zero, sine, negated_cosine)
    };
    let pick_quadrant = |ctx: &mut Ctx,
                         out: &mut Vec<Instruction>,
                         (arm_zero, arm_one, arm_minus_one, arm_two): (Word, Word, Word, Word)|
     -> Word {
        let inner = emit(
            ctx,
            out,
            Op::Select,
            vec![
                Operand::IdRef(at_minus_one),
                Operand::IdRef(arm_minus_one),
                Operand::IdRef(arm_two),
            ],
        );
        let middle = emit(
            ctx,
            out,
            Op::Select,
            vec![
                Operand::IdRef(at_one),
                Operand::IdRef(arm_one),
                Operand::IdRef(inner),
            ],
        );
        emit(
            ctx,
            out,
            Op::Select,
            vec![
                Operand::IdRef(at_zero),
                Operand::IdRef(arm_zero),
                Operand::IdRef(middle),
            ],
        )
    };
    let selected = match which {
        PiScaled::Sin => {
            let arms = sine_arms(ctx, &mut out);
            pick_quadrant(ctx, &mut out, arms)
        }
        PiScaled::Cos => {
            let arms = cosine_arms(ctx, &mut out);
            pick_quadrant(ctx, &mut out, arms)
        }
        PiScaled::Tan => {
            let sin_arms = sine_arms(ctx, &mut out);
            let numerator = pick_quadrant(ctx, &mut out, sin_arms);
            let cos_arms = cosine_arms(ctx, &mut out);
            let denominator = pick_quadrant(ctx, &mut out, cos_arms);
            emit(
                ctx,
                &mut out,
                Op::FDiv,
                vec![Operand::IdRef(numerator), Operand::IdRef(denominator)],
            )
        }
    };

    let magnitude = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(magnitude),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
            Operand::IdRef(xf),
        ],
    ));
    let finite = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FOrdLessThan,
        Some(bool_ty),
        Some(finite),
        vec![Operand::IdRef(magnitude), Operand::IdRef(infinity)],
    ));
    let guarded = if float_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    out.push(Instruction::new(
        Op::Select,
        Some(float_ty),
        Some(guarded),
        vec![
            Operand::IdRef(finite),
            Operand::IdRef(selected),
            Operand::IdRef(quiet_nan),
        ],
    ));
    if float_ty != rty {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(guarded)],
        ));
    }
    Ok(out)
}

/// Metal's `round(x)` rounds a `.5` fraction AWAY FROM ZERO — `round(2.5) == 3.0` and
/// `round(-2.5) == -3.0` — while GLSL.std.450 `Round` leaves the tie direction to the
/// implementation ("presumably the direction that is fastest"), so a round-half-to-even driver
/// answers `2.0` for the same input. Emit the deterministic form instead:
///
/// ```text
/// t = trunc(x); d = x - t; round(x) = t + (d >= 0.5 ? 1.0 : d <= -0.5 ? -1.0 : +0.0)
/// ```
///
/// The `+0.0` arm also reproduces the measured Apple-GPU detail that a zero-magnitude result is
/// `+0.0` even for a negative input (`round(-1e-20)` and `round(-0.0)` both answer `+0.0`), because
/// IEEE `-0.0 + 0.0` is `+0.0`. NaN and the infinities pass through: both compares are ordered, so
/// `adj` is `+0.0` and `t + 0.0` is `t`. A half operand computes in f32 and narrows back, which is
/// exact — every half whose magnitude is below 2048 rounds to an integer a half can hold, and every
/// half at or above 2048 is already integral.
fn lower_metal_round(ctx: &mut Ctx, res: Word, rty: Word, x: Word) -> Vec<Instruction> {
    let float_ty = float_equivalent(ctx, rty);
    let lanes = vector_len(ctx, float_ty);
    let bool_ty = if lanes == 1 {
        ctx.ty_bool()
    } else {
        ctx.ty_vec_bool(lanes)
    };
    let ext = ctx.glsl();
    let up_edge = splat_or_scalar(ctx, float_ty, 0.5, lanes);
    let down_edge = splat_or_scalar(ctx, float_ty, -0.5, lanes);
    let plus_one = splat_or_scalar(ctx, float_ty, 1.0, lanes);
    let minus_one = splat_or_scalar(ctx, float_ty, -1.0, lanes);
    let zero = splat_or_scalar(ctx, float_ty, 0.0, lanes);
    let mut out = Vec::new();
    // A half operand widens to f32 for the arithmetic; an f32 operand is already in shape.
    let xf = if float_ty == rty {
        x
    } else {
        let widened = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(widened),
            vec![Operand::IdRef(x)],
        ));
        widened
    };
    let truncated = ctx.module.fresh_id();
    let remainder = ctx.module.fresh_id();
    let reaches_up = ctx.module.fresh_id();
    let reaches_down = ctx.module.fresh_id();
    let down_or_zero = ctx.module.fresh_id();
    let adjust = ctx.module.fresh_id();
    let rounded = if float_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    out.extend([
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(truncated),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Trunc as u32),
                Operand::IdRef(xf),
            ],
        ),
        Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(remainder),
            vec![Operand::IdRef(xf), Operand::IdRef(truncated)],
        ),
        Instruction::new(
            Op::FOrdGreaterThanEqual,
            Some(bool_ty),
            Some(reaches_up),
            vec![Operand::IdRef(remainder), Operand::IdRef(up_edge)],
        ),
        Instruction::new(
            Op::FOrdLessThanEqual,
            Some(bool_ty),
            Some(reaches_down),
            vec![Operand::IdRef(remainder), Operand::IdRef(down_edge)],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(down_or_zero),
            vec![
                Operand::IdRef(reaches_down),
                Operand::IdRef(minus_one),
                Operand::IdRef(zero),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(adjust),
            vec![
                Operand::IdRef(reaches_up),
                Operand::IdRef(plus_one),
                Operand::IdRef(down_or_zero),
            ],
        ),
        Instruction::new(
            Op::FAdd,
            Some(float_ty),
            Some(rounded),
            vec![Operand::IdRef(truncated), Operand::IdRef(adjust)],
        ),
    ]);
    if float_ty != rty {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(rounded)],
        ));
    }
    out
}

/// Metal's `fast::sin` / `fast::cos` reduce the argument and then GIVE UP, answering exactly `0.0`
/// once the quadrant index would exceed 2^22. Bisected on this repository's reference GPU (Apple M3
/// Max, macOS 26.5.2): `fast::cos(6588397.5)` still tracks `precise::cos`, and the very next float,
/// `6588398.0`, answers `0.0` — 6588397.5 is `float(pi/2) * 2^22`, so the boundary is the quadrant
/// count, not a round power of two. The same input flushes both `sin` and `cos`, and the negatives
/// mirror it.
///
/// The infinities and NaN flush too (`fast::cos(inf)`, `fast::cos(NaN)` are both `0.0`), which an
/// ORDERED compare would miss for NaN — hence `FUnordGreaterThan`, whose NaN arm is the flush.
///
/// The flush is the WHOLE difference from GLSL `Sin`/`Cos`. This lowering used to reduce the
/// argument first — `x - trunc(x / 2pi) * 2pi` — which is not something Metal does and which
/// destroys precision for any argument big enough to reduce at all: `trunc(x / 2pi) * 2pi` rounds
/// twice, so the reduced argument carries the absolute error of a number the size of `x`, not of
/// the residue. Measured on the authored `kernel_fast_trig_range` fixture: with the reduction
/// `fast::sin(100)` came back **38 ULP** from Metal's answer; without it every probed argument
/// agrees with Metal bit for bit, including 1e5, 1e6 and the threshold itself, where MSL's own
/// `sin` is correctly rounded and the reduction was purely destructive. Dropping it changed bytes
/// in 564 of 14579 corpus modules and the status of none.
///
/// **Why `air.fast_tan` is NOT given this treatment, measured rather than assumed.** The guard is
/// only sound because `fast::sin`/`fast::cos` and `precise::sin`/`precise::cos` are otherwise the
/// SAME function: device-measured on an Apple M3 Max / macOS 26.5.2, they are bit-identical on all
/// 10001 arguments of a 1/5 grid over [-1000, 1000], and differ on 165 and 162 of 10001 over
/// [0, 6e6] by at most 2 ULP. The flush is the whole difference, so modelling the flush closes the
/// whole gap -- which matters because SPIRV-Cross emits the UNQUALIFIED MSL `sin`/`cos` and
/// MoltenVK answers those precisely.
///
/// `fast::tan` is a different approximation, not a flushed one. It differs from `precise::tan` on
/// **5828 of the same 10001 arguments** at |x| <= 1000, by up to 4 ULP, and past roughly 5.09e7
/// (positive) or -2.79e7 (negative) it returns NaN where `precise::tan` returns a finite value --
/// and not at a clean threshold either, since 5 of 24 sampled arguments BELOW the positive bound
/// are NaN too. No select-shaped guard reproduces that. `air.fast_tan.f32` is therefore closed as
/// a MoltenVK limitation on the same terms as `air.fast_atan2.f32`: 85 corpus call sites that
/// cannot be made byte-exact, not a lowering that is missing a guard.
fn lower_fast_trig(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    x: Word,
    op: GLSLstd450,
) -> Vec<Instruction> {
    let ext = ctx.glsl();
    let lanes = vector_len(ctx, rty);
    let bool_ty = if lanes == 1 {
        ctx.ty_bool()
    } else {
        ctx.ty_vec_bool(lanes)
    };
    let zero = ctx.const_float(0.0);
    // float(pi/2) * 2^22, the largest magnitude the hardware still evaluates.
    let threshold = ctx.const_float(6_588_397.5);
    let (zero, threshold) = clamp_edges(ctx, rty, zero, threshold);
    let abs = ctx.module.fresh_id();
    let raw = ctx.module.fresh_id();
    let too_large = ctx.module.fresh_id();
    vec![
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(abs),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
                Operand::IdRef(x),
            ],
        ),
        Instruction::new(
            Op::FUnordGreaterThan,
            Some(bool_ty),
            Some(too_large),
            vec![Operand::IdRef(abs), Operand::IdRef(threshold)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(raw),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(op as u32),
                Operand::IdRef(x),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(too_large),
                Operand::IdRef(zero),
                Operand::IdRef(raw),
            ],
        ),
    ]
}

pub(in crate::passes) fn lower_signed_integer_minmax(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    op: GLSLstd450,
) -> Result<Vec<Instruction>, String> {
    let signed_ty =
        signed_integer_type_like(ctx, rty).ok_or("air signed integer min/max result is not int")?;
    let mut out = Vec::new();
    let lhs = bitcast_integer_to_type(ctx, &mut out, args[0], signed_ty)?;
    let rhs = bitcast_integer_to_type(ctx, &mut out, args[1], signed_ty)?;
    let signed_res = if signed_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    let ext = ctx.glsl();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(signed_ty),
        Some(signed_res),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(op as u32),
            Operand::IdRef(lhs),
            Operand::IdRef(rhs),
        ],
    ));
    if signed_res != res {
        out.push(Instruction::new(
            Op::Bitcast,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(signed_res)],
        ));
    }
    Ok(out)
}

pub(in crate::passes) fn bitcast_integer_to_type(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    target_ty: Word,
) -> Result<Word, String> {
    let value_ty = value_result_type(ctx, value).ok_or("air signed min/max arg has no type")?;
    if value_ty == target_ty {
        return Ok(value);
    }
    let value_shape =
        integer_shape(ctx, value_ty).ok_or("air signed min/max arg is not integer-shaped")?;
    let target_shape =
        integer_shape(ctx, target_ty).ok_or("air signed min/max target is not integer-shaped")?;
    if value_shape != target_shape {
        return Err("air signed min/max arg shape does not match result".into());
    }
    let cast = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(target_ty),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    Ok(cast)
}

pub(in crate::passes) fn signed_integer_type_like(ctx: &mut Ctx, ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeInt => {
            let bits = match def.operands.first()? {
                Operand::LiteralBit32(bits) => *bits,
                _ => return None,
            };
            Some(signed_integer_scalar_type(ctx, bits))
        }
        Op::TypeVector => {
            let elem = match def.operands.first()? {
                Operand::IdRef(elem) => *elem,
                _ => return None,
            };
            let lanes = match def.operands.get(1)? {
                Operand::LiteralBit32(lanes) => *lanes,
                _ => return None,
            };
            let signed_elem = signed_integer_type_like(ctx, elem)?;
            Some(integer_vector_type(ctx, signed_elem, lanes))
        }
        _ => None,
    }
}

pub(in crate::passes) fn signed_integer_scalar_type(ctx: &mut Ctx, bits: u32) -> Word {
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(bits))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(1))
        {
            if let Some(id) = inst.result_id {
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeInt,
        id,
        vec![Operand::LiteralBit32(bits), Operand::LiteralBit32(1)],
    ));
    id
}

pub(in crate::passes) fn integer_vector_type(ctx: &mut Ctx, elem: Word, lanes: u32) -> Word {
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeVector
            && inst.operands.first() == Some(&Operand::IdRef(elem))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(lanes))
        {
            if let Some(id) = inst.result_id {
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeVector,
        id,
        vec![Operand::IdRef(elem), Operand::LiteralBit32(lanes)],
    ));
    id
}

use crate::passes::air_calls::images::integer_shape;

/// Whether this call's block extent comes from the explicit size operand rather than the implicit
/// imageblock (= threadgroup) extent.
///
/// Metal's `imageblock_slice` carries a size and a validity flag; the flag says whether the size is
/// the one to use. A flag that is neither constant leaves the choice to runtime, and the bounds gate
/// -- the only reader that can express both answers at once -- selects per axis. Every reader that
/// needs one answer takes the explicit operand, which is the one AIR bothered to compute.
///
/// This is the one derivation of that choice. It used to be spelled twice, and the copy in the
/// bounds gate and the copy in the acceptance check are exactly the pair that has to agree.
pub(in crate::passes) fn imageblock_region_is_explicit(ctx: &Ctx, args: &[Word]) -> bool {
    let has_size_flag = value_def_instruction(ctx, args[2]).map(|def| def.class.opcode);
    let explicit_size_def = value_def_instruction(ctx, args[4]).map(|def| def.class.opcode);
    let explicit_usable = !matches!(explicit_size_def, Some(Op::Undef) | None);
    explicit_usable && !matches!(has_size_flag, Some(Op::ConstantFalse))
}

/// The block extent in cells, when it is a compile-time constant.
///
/// `None` is a call whose extent only exists at runtime: a non-constant explicit size operand. The
/// implicit extent is the whole imageblock, and the imageblock is `emitted_tile_cell_scale` cells
/// per thread in each axis, so the implicit form is static whenever that scale is.
pub(in crate::passes) fn static_imageblock_region(ctx: &Ctx, args: &[Word]) -> Option<[u32; 2]> {
    if imageblock_region_is_explicit(ctx, args) {
        return constant_uvec2_components(ctx, args[4]);
    }
    let scale = emitted_tile_cell_scale(ctx, *args.get(1)?)?;
    Some([
        ctx.kernel_local_size[0].checked_mul(scale)?,
        ctx.kernel_local_size[1].checked_mul(scale)?,
    ])
}

/// How many imageblock cells the emitter gave each thread in each axis, read back off the cell index
/// it built.
///
/// The implicit block extent is the imageblock's, and whether that is the threadgroup extent or a
/// multiple of it is a choice `infer_imageblock_cell_scale` made and the native emitter acted on.
/// Re-deriving it here from `TransformOptions` would be a second derivation of the same fact, and it
/// was wrong for every entry that stages a `k`x`k` block per thread -- it named a `1/k^2` corner of
/// the block Metal copies. So recover what the emitter actually multiplied the threadgroup row
/// stride by, exactly as [`imageblock_row_stride`] recovers the stride itself: an unscaled tile
/// leaves the stride alone, a scaled one is `OpIMul` by a constant.
fn emitted_tile_cell_scale(ctx: &Ctx, cell_pointer: Word) -> Option<u32> {
    let (_, cell_index, _) = imageblock_cell_chain(ctx, cell_pointer).ok()?;
    let Some(width) = imageblock_row_stride(ctx, cell_index) else {
        // A single-row tile never multiplied, so no scale was applied to a stride that is not there.
        return Some(1);
    };
    let def = value_def_instruction(ctx, width)?;
    if def.class.opcode != Op::IMul {
        return Some(1);
    }
    let Some(Operand::IdRef(scale)) = def.operands.get(1) else {
        return None;
    };
    match value_def_instruction(ctx, *scale)?.operands.first() {
        Some(Operand::LiteralBit32(literal)) => Some(*literal),
        _ => None,
    }
}

/// The imageblock-slice write copies a whole WxH block of cells to the texture; this translator
/// emits a single `OpImageWrite` of the one cell the pointer names.
///
/// `gate_imageblock_region_in_bounds` already derives that region -- the explicit size operand when
/// the has-size flag is set, otherwise the imageblock (= threadgroup) extent -- and uses it only to
/// clip the write. The two derivations of "how much does this call write" disagreed: the gate said
/// WxH and the write said one texel. The single-texel form is exact when, and only when, the region
/// is one cell, so accept that shape and refuse the rest rather than write 1/(WxH) of the block.
///
/// Device-measured on `copyTexture` (four 2x2 tiles over a 4x4 texture): Metal returns the input
/// byte for byte, the single-texel lowering leaves 12 of 16 texels untouched.
fn imageblock_slice_region_is_one_texel(ctx: &Ctx, args: &[Word]) -> Result<(), String> {
    match static_imageblock_region(ctx, args) {
        // One cell, or a degenerate region the bounds gate already clips to nothing.
        Some([width, height]) if width * height <= 1 => Ok(()),
        Some([width, height]) => Err(format!(
            "air.write_imageblock_slice_to_texture copies a {width}x{height} block of imageblock \
             cells, and this translator stages one cell per invocation; emitting the module would \
             write a single texel where Metal writes the block"
        )),
        None => Err(
            "air.write_imageblock_slice_to_texture carries a runtime block extent, and this \
             translator stages one cell per invocation; emitting the module would write a single \
             texel where Metal writes the block"
                .into(),
        ),
    }
}

/// The two components of a constant `<2 x i16>`/`<2 x i32>` region operand, when both are constant.
pub(in crate::passes) fn constant_uvec2_components(ctx: &Ctx, value: Word) -> Option<[u32; 2]> {
    let def = value_def_instruction(ctx, value)?;
    match def.class.opcode {
        Op::ConstantNull => Some([0, 0]),
        Op::ConstantComposite => {
            let mut out = [0u32; 2];
            for (slot, operand) in out.iter_mut().zip(def.operands.iter()) {
                let Operand::IdRef(component) = operand else {
                    return None;
                };
                let component = value_def_instruction(ctx, *component)?;
                match (component.class.opcode, component.operands.first()) {
                    (Op::Constant, Some(Operand::LiteralBit32(literal))) => *slot = *literal,
                    (Op::ConstantNull, _) => *slot = 0,
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

pub(in crate::passes) fn lower_imageblock_slice_write(
    ctx: &mut Ctx,
    name: &str,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    if args.len() < 6 {
        return Err("air.write_imageblock_slice_to_texture missing operands".into());
    }
    imageblock_slice_region_is_one_texel(ctx, args)?;
    let ptr_ty = value_result_type(ctx, args[1])
        .ok_or("air.write_imageblock_slice_to_texture pointer has no result type")?;
    let texel_ty = pointer_pointee_type(ctx, ptr_ty)
        .ok_or("air.write_imageblock_slice_to_texture pointer is not typed")?;
    let storage = ptr_storage(&type_defs(&ctx.module), ptr_ty).unwrap_or(StorageClass::Private);
    let write_texel_ty = imageblock_slice_texel_type(ctx, name, v4).unwrap_or(texel_ty);
    let texel_ptr = if write_texel_ty == texel_ty {
        args[1]
    } else {
        // Logical SPIR-V cannot reinterpret one pointer type as another. An imageblock cell may
        // carry several metadata-described fields while the AIR write intrinsic names the first
        // texel field directly (for example `{ half, half2 }` written through the `.f16` form).
        // Follow only an exact zero-offset aggregate path to that field; this is a real typed
        // subobject, unlike the former pointer OpBitcast. If the AIR type has no such field, keep
        // the layout gap visible instead of inventing a byte-level reinterpretation.
        let path = imageblock_zero_offset_subobject_path(ctx, texel_ty, write_texel_ty)
            .ok_or_else(|| {
                format!(
                    "air.write_imageblock_slice_to_texture cannot view pointee type %{texel_ty} \
                     as texel type %{write_texel_ty} through a zero-offset aggregate field"
                )
            })?;
        let retyped_ptr_ty = ctx.ty_ptr(storage, write_texel_ty);
        let retyped = ctx.module.fresh_id();
        let mut operands = Vec::with_capacity(path.len() + 1);
        operands.push(Operand::IdRef(args[1]));
        operands.extend(
            path.into_iter()
                .map(|index| Operand::IdRef(ctx.const_uint(index))),
        );
        let mut out = vec![Instruction::new(
            Op::InBoundsAccessChain,
            Some(retyped_ptr_ty),
            Some(retyped),
            operands,
        )];
        let texel = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Load,
            Some(write_texel_ty),
            Some(texel),
            vec![Operand::IdRef(retyped)],
        ));
        return lower_imageblock_slice_write_texel(ctx, name, args, v4, texel, write_texel_ty, out);
    };
    let texel = ctx.module.fresh_id();
    let out = vec![Instruction::new(
        Op::Load,
        Some(write_texel_ty),
        Some(texel),
        vec![Operand::IdRef(texel_ptr)],
    )];
    lower_imageblock_slice_write_texel(ctx, name, args, v4, texel, write_texel_ty, out)
}

/// Return the exact zero-offset aggregate-member path from `source` to `target`.
///
/// Imageblock write intrinsics encode their texel type in the stable AIR ABI symbol, while the
/// pointer itself can still name the complete metadata-described cell. The only legal Logical
/// SPIR-V narrowing is an access chain to a real contained subobject. Struct member zero and array
/// element zero share the parent's byte address, so recursively following their first child is
/// sufficient and deliberately does not model arbitrary same-size reinterprets.
pub(in crate::passes) fn imageblock_zero_offset_subobject_path(
    ctx: &Ctx,
    source: Word,
    target: Word,
) -> Option<Vec<u32>> {
    if source == target {
        return Some(Vec::new());
    }
    let source_def = type_def_of(ctx, source)?;
    let first_child = match source_def.class.opcode {
        Op::TypeStruct | Op::TypeArray => match source_def.operands.first() {
            Some(Operand::IdRef(child)) => *child,
            _ => return None,
        },
        _ => return None,
    };
    let mut path = imageblock_zero_offset_subobject_path(ctx, first_child, target)?;
    path.insert(0, 0);
    Some(path)
}

pub(in crate::passes) fn lower_imageblock_slice_write_texel(
    ctx: &mut Ctx,
    name: &str,
    args: &[Word],
    v4: Word,
    texel: Word,
    texel_ty: Word,
    mut out: Vec<Instruction>,
) -> Result<Vec<Instruction>, String> {
    let mut img = resolve_image_value(ctx, args[0]);
    if !image_is_storage(ctx, img) {
        img = recovered_image_for_private_operand(ctx, img, name, ImageOperandUse::Storage)
            .ok_or_else(|| {
                format!("air.write_imageblock_slice_to_texture on non-storage image id {img}")
            })?;
    }
    ctx.require_runtime_storage_image_use(img, RuntimeStorageImageUse::Write)?;
    let (dim, arrayed) = ctx
        .image_dims
        .get(&img)
        .copied()
        .unwrap_or((Dim::Dim2D, false));
    // The imageblock-slice ABI puts the destination coordinate at operand 5 and, for an array
    // texture, its slice at operand 6.
    let layer = if arrayed {
        Some(
            args.get(6)
                .copied()
                .ok_or("air.write_imageblock_slice_to_texture array texture missing layer")?,
        )
    } else {
        None
    };
    let coord32 = build_fetch_coord(ctx, dim, arrayed, args[5], layer, &mut out)?;
    let region_gate =
        gate_imageblock_region_in_bounds(ctx, args, img, dim, arrayed, coord32, &mut out)?;
    let comp = ctx
        .image_comp
        .get(&img)
        .copied()
        .unwrap_or(ImageComp::Float);
    // A scalar texel (single-channel imageblock write, e.g. `air.write_imageblock_slice_to_texture_2d
    // .f16`) has no vector shape; treat it as 1 lane. OpImageWrite always takes a 4-component texel,
    // so a sub-v4 texel is padded to v4 with zeros (the extra channels are ignored by single/2/3-
    // channel storage formats).
    let (elem, lanes) = vector_type_shape(ctx, texel_ty).unwrap_or((texel_ty, 1));
    let texel32 = if comp != ImageComp::Float {
        // Integer (Sint/Uint) imageblock write, e.g. `air.write_imageblock_slice_to_texture_2d
        // .i16.v4i16` into an Rgba16Sint/Uint storage image: build a v4 of the image's 32-bit
        // integer sampled type (`ty_sint`/`ty_uint`), sign/zero-extending the narrower channels
        // per the format's signedness. OpImageWrite's texel type must match the storage image's
        // integer sampled type, exactly as the float arm matches its float sampled type.
        build_int_write_texel(ctx, comp, texel, elem, lanes, &mut out)?
    } else if lanes == 4 {
        if is_f32_scalar(ctx, elem) {
            texel
        } else if is_half_scalar(ctx, elem) {
            let converted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(v4),
                Some(converted),
                vec![Operand::IdRef(texel)],
            ));
            converted
        } else {
            return Err(
                "air.write_imageblock_slice_to_texture: unsupported texel component".into(),
            );
        }
    } else if lanes == 1 {
        // Convert the scalar channel to f32, then build a v4 (channel 0 = value, 1..3 = 0).
        let f32_ty = ctx.ty_float();
        let scalar32 = if is_f32_scalar(ctx, elem) {
            texel
        } else if is_half_scalar(ctx, elem) {
            let converted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(f32_ty),
                Some(converted),
                vec![Operand::IdRef(texel)],
            ));
            converted
        } else {
            return Err(
                "air.write_imageblock_slice_to_texture: unsupported texel component".into(),
            );
        };
        let zero = ctx.const_float(0.0);
        let padded = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(v4),
            Some(padded),
            vec![
                Operand::IdRef(scalar32),
                Operand::IdRef(zero),
                Operand::IdRef(zero),
                Operand::IdRef(zero),
            ],
        ));
        padded
    } else if lanes == 2 || lanes == 3 {
        // A 2- or 3-channel texel (e.g. `.v2f16`): convert to a float vector of the same lane count,
        // extract its channels, and build a v4 padding the unused channels with 0 (OpImageWrite always
        // takes a 4-component texel; the storage format ignores the extra channels).
        let f32_ty = ctx.ty_float();
        let src32 = if is_f32_scalar(ctx, elem) {
            texel
        } else if is_half_scalar(ctx, elem) {
            let fvec = ctx.ty_vecf(lanes);
            let converted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(fvec),
                Some(converted),
                vec![Operand::IdRef(texel)],
            ));
            converted
        } else {
            return Err(
                "air.write_imageblock_slice_to_texture: unsupported texel component".into(),
            );
        };
        let zero = ctx.const_float(0.0);
        let mut comps = Vec::with_capacity(4);
        for i in 0..lanes {
            let c = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::CompositeExtract,
                Some(f32_ty),
                Some(c),
                vec![Operand::IdRef(src32), Operand::LiteralBit32(i)],
            ));
            comps.push(Operand::IdRef(c));
        }
        for _ in lanes..4 {
            comps.push(Operand::IdRef(zero));
        }
        let padded = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(v4),
            Some(padded),
            comps,
        ));
        padded
    } else {
        return Err("air.write_imageblock_slice_to_texture: texel is not v4".into());
    };
    let texel32_ty = match comp {
        ImageComp::Float => v4,
        ImageComp::Sint => ctx.ty_vec_sint(4),
        ImageComp::Uint => ctx.ty_vec_uint(4),
    };
    let texel32 = zero_texel_for_empty_imageblock_region(
        ctx,
        texel32,
        texel32_ty,
        region_gate.empty,
        &mut out,
    )?;
    let texel32 = if comp == ImageComp::Float {
        ctx.module.types_global_values.append(&mut ctx.new_globals);
        ctx.phase_type_positions = None;
        ctx.texture_write_rounding.preserve_imageblock_slice(
            &mut ctx.module,
            &mut out,
            texel32,
            texel32_ty,
        )?
    } else {
        texel32
    };
    out.push(Instruction::new(
        Op::ImageWrite,
        None,
        None,
        vec![
            Operand::IdRef(img),
            Operand::IdRef(region_gate.coord),
            Operand::IdRef(texel32),
        ],
    ));
    Ok(out)
}

fn zero_texel_for_empty_imageblock_region(
    ctx: &mut Ctx,
    texel: Word,
    texel_ty: Word,
    region_empty: Word,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let zero_texel = ctx.get_or_create(Op::ConstantNull, Some(texel_ty), vec![]);
    let selected = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(texel_ty),
        Some(selected),
        vec![
            Operand::IdRef(region_empty),
            Operand::IdRef(zero_texel),
            Operand::IdRef(texel),
        ],
    ));
    Ok(selected)
}

/// Build the `OpImageWrite` texel for a non-float (Sint/Uint) imageblock slice write: the storage
/// image's integer sampled type is 32-bit (`ty_sint`/`ty_uint`), so a narrower channel texel (the
/// common `.v4i16` form) is widened to a v4 of that 32-bit int, sign-extended for Sint and
/// zero-extended for Uint. Only the v4 shape is handled (both regression cases are `.v4i16`); a sub-v4
/// integer texel is left as an honest Err so a mis-lowering never ships instead of failing loudly.
pub(in crate::passes) fn build_int_write_texel(
    ctx: &mut Ctx,
    comp: ImageComp,
    texel: Word,
    elem: Word,
    lanes: u32,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    if lanes != 4 {
        return Err(
            "air.write_imageblock_slice_to_texture: non-float imageblock write with non-v4 texel"
                .into(),
        );
    }
    let (v4int_ty, signed) = match comp {
        ImageComp::Sint => (ctx.ty_vec_sint(4), true),
        ImageComp::Uint => (ctx.ty_vec_uint(4), false),
        ImageComp::Float => {
            return Err("build_int_write_texel called for a float image".into());
        }
    };
    // Already a 32-bit int of the target signedness → OpImageWrite accepts it directly (a same-width
    // SConvert/UConvert would be spirv-val-invalid, so this guard is required, not just an optimization).
    let already32 = (signed && is_int_scalar_width(ctx, elem, 32))
        || (!signed && is_uint_scalar_width(ctx, elem, 32));
    if already32 {
        return Ok(texel);
    }
    let op = if signed { Op::SConvert } else { Op::UConvert };
    let converted = ctx.module.fresh_id();
    out.push(Instruction::new(
        op,
        Some(v4int_ty),
        Some(converted),
        vec![Operand::IdRef(texel)],
    ));
    Ok(converted)
}

struct ImageblockRegionGate {
    coord: Word,
    empty: Word,
}

/// Gate an `air.write_imageblock_slice_to_texture_*` store on the region it names, and report
/// whether that region has a zero spatial extent.
///
/// This lowering emits one `OpImageWrite`, so the region it is asked about is one cell: a block copy
/// has already been split into per-cell calls carrying a constant 1x1 region
/// (`imageblock_block_copy`). The gate then says "is this texel inside the texture", and where it is
/// not, ORs the write coordinate with all-ones so Vulkan's out-of-bounds store rule drops it.
///
/// **Metal clips such a write per texel; it does not discard the whole block.** Device-measured on
/// `copyTexture` by the authored case `imageblock-slice-block-hangs-off-the-texture`: four 2x2 tiles
/// over a 3x3 texture, three of whose blocks hang off an edge, and Metal writes all nine texels that
/// fit. The claim this comment used to make -- that the whole implicit-region write is discarded --
/// was never measured and is wrong.
///
/// A zero-area region keeps its destination coordinate and writes a transparent-zero texel. That is
/// the one part of this still unmeasured: no corpus module names a statically zero region, and the
/// runtime-extent calls that could reach one at runtime are refused before they get here.
fn gate_imageblock_region_in_bounds(
    ctx: &mut Ctx,
    args: &[Word],
    img: Word,
    dim: Dim,
    arrayed: bool,
    coord32: Word,
    out: &mut Vec<Instruction>,
) -> Result<ImageblockRegionGate, String> {
    let spatial: u32 = match dim {
        Dim::Dim1D | Dim::DimBuffer => 1,
        Dim::Dim3D => 3,
        _ => 2,
    };
    let ncomp = spatial + u32::from(arrayed);
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    // Texture dimensions. The image is a storage image (checked by the caller), so the query has
    // no LOD operand; the bound storage view is single-mip.
    let size_ty = if ncomp == 1 {
        uint
    } else {
        ctx.ty_vec_uint(ncomp)
    };
    let dims = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ImageQuerySize,
        Some(size_ty),
        Some(dims),
        vec![Operand::IdRef(img)],
    ));
    let extract = |ctx: &mut Ctx, out: &mut Vec<Instruction>, composite: Word, index: u32| {
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(id),
            vec![Operand::IdRef(composite), Operand::LiteralBit32(index)],
        ));
        id
    };
    // Block region size per axis: the explicit size operand when the has-size flag is set, else the
    // imageblock (= threadgroup) dimensions. A non-constant flag selects at runtime.
    let has_size_flag = value_def_instruction(ctx, args[2]).map(|def| def.class.opcode);
    let local_size = ctx.kernel_local_size_ids();
    let implicit = [local_size[0], local_size[1]];
    let explicit = if imageblock_region_is_explicit(ctx, args) {
        // args[4] is a `<2 x i16>` size; widen to uint2 and split into components.
        let src_ty = value_result_type(ctx, args[4])
            .ok_or("air.write_imageblock_slice_to_texture: size operand has no type")?;
        let wide = if scalar_bit_width(ctx, src_ty) == 32 {
            args[4]
        } else {
            let uint2 = ctx.ty_vec_uint(2);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::UConvert,
                Some(uint2),
                Some(id),
                vec![Operand::IdRef(args[4])],
            ));
            id
        };
        Some([extract(ctx, out, wide, 0), extract(ctx, out, wide, 1)])
    } else {
        None
    };
    let mut region = implicit;
    if let Some(explicit) = explicit {
        match has_size_flag {
            Some(Op::ConstantTrue) => region = explicit,
            Some(Op::ConstantFalse) => {}
            _ => {
                // Runtime flag: select per axis.
                for axis in 0..2 {
                    let id = ctx.module.fresh_id();
                    out.push(Instruction::new(
                        Op::Select,
                        Some(uint),
                        Some(id),
                        vec![
                            Operand::IdRef(args[2]),
                            Operand::IdRef(explicit[axis]),
                            Operand::IdRef(implicit[axis]),
                        ],
                    ));
                    region[axis] = id;
                }
            }
        }
    }
    // fits = origin + region_size <= dims, per spatial axis (x, and y for 2D/3D images). The origin
    // comes from the already-widened write coordinate; both operands are <= 0xffff so the u32 add
    // cannot wrap. In parallel, track whether any spatial size component is zero so the caller can
    // model Apple's transparent-zero texel for zero-area explicit writes.
    let zero = ctx.const_uint(0);
    let mut fits: Option<Word> = None;
    let mut empty: Option<Word> = None;
    for axis in 0..spatial.min(2) {
        let origin = if ncomp == 1 {
            coord32
        } else {
            extract(ctx, out, coord32, axis)
        };
        let dim_axis = if ncomp == 1 {
            dims
        } else {
            extract(ctx, out, dims, axis)
        };
        let end = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::IAdd,
            Some(uint),
            Some(end),
            vec![
                Operand::IdRef(origin),
                Operand::IdRef(region[axis as usize]),
            ],
        ));
        let axis_empty = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::IEqual,
            Some(bool_ty),
            Some(axis_empty),
            vec![Operand::IdRef(region[axis as usize]), Operand::IdRef(zero)],
        ));
        let axis_fits = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ULessThanEqual,
            Some(bool_ty),
            Some(axis_fits),
            vec![Operand::IdRef(end), Operand::IdRef(dim_axis)],
        ));
        fits = Some(match fits {
            None => axis_fits,
            Some(prev) => {
                let both = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::LogicalAnd,
                    Some(bool_ty),
                    Some(both),
                    vec![Operand::IdRef(prev), Operand::IdRef(axis_fits)],
                ));
                both
            }
        });
        empty = Some(match empty {
            None => axis_empty,
            Some(prev) => {
                let either = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::LogicalOr,
                    Some(bool_ty),
                    Some(either),
                    vec![Operand::IdRef(prev), Operand::IdRef(axis_empty)],
                ));
                either
            }
        });
    }
    let fits = fits.ok_or("gate_imageblock_region_in_bounds: at least one spatial axis")?;
    let empty = empty.ok_or("gate_imageblock_region_in_bounds: at least one spatial axis")?;
    // mask = fits ? 0 : 0xffffffff; OR-ing it into the coordinate forces the store out of bounds
    // (and thus discarded) exactly when the implicit/runtime-selected block region does not fit.
    let all_ones = ctx.const_uint(u32::MAX);
    let mask = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(mask),
        vec![
            Operand::IdRef(fits),
            Operand::IdRef(zero),
            Operand::IdRef(all_ones),
        ],
    ));
    let (mask_vec, coord_ty) = if ncomp == 1 {
        (mask, uint)
    } else {
        let vec_ty = ctx.ty_vec_uint(ncomp);
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(vec_ty),
            Some(id),
            vec![Operand::IdRef(mask); ncomp as usize],
        ));
        (id, vec_ty)
    };
    let gated = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseOr,
        Some(coord_ty),
        Some(gated),
        vec![Operand::IdRef(coord32), Operand::IdRef(mask_vec)],
    ));
    Ok(ImageblockRegionGate {
        coord: gated,
        empty,
    })
}

pub(in crate::passes) fn imageblock_slice_texel_type(
    ctx: &mut Ctx,
    name: &str,
    v4: Word,
) -> Option<Word> {
    // The texel component type is the final dotted suffix of the intrinsic name, e.g.
    // `air.write_imageblock_slice_to_texture_2d.i16.v2f16` -> `v2f16`. `vNfM` is an M-bit-float
    // N-vector (N in 2..=4); a bare `fM` is a scalar (1 lane). The half/float scalar+vector forms are
    // all handled by `lower_imageblock_slice_write_texel`; the name is authoritative for the texel
    // type so the field pointer is reinterpreted to it before the load.
    let suffix = name.rsplit('.').next()?;
    let (lanes, elem) = match suffix.strip_prefix('v') {
        Some(rest) => {
            let split = rest.find('f')?;
            (rest[..split].parse::<u32>().ok()?, &rest[split..])
        }
        None => (1, suffix),
    };
    if !(1..=4).contains(&lanes) {
        return None;
    }
    let half = match elem {
        "f16" => true,
        "f32" => false,
        _ => return None,
    };
    Some(match (half, lanes) {
        (true, 1) => ctx.ty_half(),
        (true, n) => ctx.ty_vech(n),
        (false, 1) => ctx.ty_float(),
        (false, 4) => v4,
        (false, n) => ctx.ty_vecf(n),
    })
}

pub(in crate::passes) fn pointer_pointee_type(ctx: &Ctx, ptr_ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ptr_ty)?;
    if def.class.opcode != Op::TypePointer {
        return None;
    }
    match def.operands.get(1) {
        Some(Operand::IdRef(pointee)) => Some(*pointee),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    /// The bisected boundary of Metal's `fast::sin` / `fast::cos` flush, pinned as the arithmetic it
    /// actually is. On an Apple M3 Max (macOS 26.5.2) `fast::cos(6588397.5)` still tracks
    /// `precise::cos` and the next representable float answers exactly `0.0`; that boundary is the
    /// float `pi/2` scaled by 2^22, i.e. the largest argument whose quadrant index fits in 22 bits.
    #[test]
    fn fast_trig_flush_threshold_is_the_last_evaluated_quadrant() {
        let threshold = 6_588_397.5f32;
        assert_eq!(threshold, std::f32::consts::FRAC_PI_2 * (1u32 << 22) as f32);
        assert_eq!(threshold.to_bits(), 0x4AC9_0FDB);
        // The first magnitude the hardware flushes is the very next float.
        assert_eq!(f32::from_bits(0x4AC9_0FDC), 6_588_398.0);
    }
}
