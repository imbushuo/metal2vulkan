//! Raw-byte, descriptor-alias, and unary AIR call lowering.

use super::*;

pub(in crate::passes) fn atomic_i32_pointer(
    ctx: &mut Ctx,
    ptr: Word,
    out: &mut Vec<Instruction>,
) -> Word {
    let Some(ptr_ty) = value_result_type(ctx, ptr) else {
        return ptr;
    };
    let Some(pointee) = pointer_pointee_type(ctx, ptr_ty) else {
        return ptr;
    };
    if is_uint_scalar_width(ctx, pointee, 32) {
        return ptr;
    }
    if !is_uint_scalar_width(ctx, pointee, 8) {
        return ptr;
    }
    let Some((root, byte_index)) = raw_byte_access_root_and_index(ctx, ptr) else {
        return ptr;
    };
    let Some(binding) = descriptor_binding(ctx, root) else {
        return ptr;
    };
    let alias = raw_uint_alias_buffer(ctx, binding);
    let Some(word_index) = raw_byte_index_to_word_index(ctx, byte_index, out) else {
        return ptr;
    };
    let uint = ctx.ty_uint();
    let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let fixed = ctx.module.fresh_id();
    let zero = ctx.const_uint(0);
    out.push(Instruction::new(
        Op::AccessChain,
        Some(ptr_ty),
        Some(fixed),
        vec![
            Operand::IdRef(alias),
            Operand::IdRef(zero),
            Operand::IdRef(word_index),
        ],
    ));
    fixed
}

pub(in crate::passes) fn raw_byte_access_root_and_index(
    ctx: &Ctx,
    ptr: Word,
) -> Option<(Word, Word)> {
    let inst = value_def_instruction(ctx, ptr)?;
    if !matches!(
        inst.class.opcode,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) {
        return None;
    }
    let Some(Operand::IdRef(root)) = inst.operands.first() else {
        return None;
    };
    let indices = inst.operands[1..]
        .iter()
        .map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if let Some(byte_index) = raw_byte_buffer_index(ctx, *root, &indices) {
        return Some((*root, byte_index));
    }
    let (root, mut base_indices) = raw_byte_access_path(ctx, *root)?;
    base_indices.extend(indices);
    raw_byte_buffer_index(ctx, root, &base_indices).map(|byte_index| (root, byte_index))
}

pub(in crate::passes) fn raw_byte_access_path(ctx: &Ctx, ptr: Word) -> Option<(Word, Vec<Word>)> {
    let inst = value_def_instruction(ctx, ptr)?;
    if !matches!(
        inst.class.opcode,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) {
        return None;
    }
    let Some(Operand::IdRef(base)) = inst.operands.first() else {
        return None;
    };
    let indices = inst.operands[1..]
        .iter()
        .map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if raw_byte_buffer_index(ctx, *base, &indices).is_some() {
        return Some((*base, indices));
    }
    let (root, mut base_indices) = raw_byte_access_path(ctx, *base)?;
    base_indices.extend(indices);
    Some((root, base_indices))
}

pub(in crate::passes) fn raw_byte_buffer_index(
    ctx: &Ctx,
    root: Word,
    indices: &[Word],
) -> Option<Word> {
    let [member, byte_index] = indices else {
        return None;
    };
    if constant_u32(ctx, *member) != Some(0) {
        return None;
    }
    let root_ty = value_result_type(ctx, root)?;
    let block_ty = pointer_pointee_type(ctx, root_ty)?;
    let block_def = type_def_of(ctx, block_ty)?;
    if block_def.class.opcode != Op::TypeStruct {
        return None;
    }
    let runtime = match block_def.operands.first() {
        Some(Operand::IdRef(runtime)) => *runtime,
        _ => return None,
    };
    let runtime_def = type_def_of(ctx, runtime)?;
    if runtime_def.class.opcode != Op::TypeRuntimeArray {
        return None;
    }
    let elem = match runtime_def.operands.first() {
        Some(Operand::IdRef(elem)) => *elem,
        _ => return None,
    };
    is_uint_scalar_width(ctx, elem, 8).then_some(*byte_index)
}

pub(in crate::passes) fn raw_byte_index_to_word_index(
    ctx: &mut Ctx,
    byte_index: Word,
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    if let Some(byte_index) = constant_u32(ctx, byte_index) {
        return (byte_index % 4 == 0).then(|| ctx.const_uint(byte_index / 4));
    }
    let byte_ty = value_result_type(ctx, byte_index)?;
    let byte_index = if is_uint_scalar_width(ctx, byte_ty, 32) {
        byte_index
    } else if int_scalar_width(ctx, byte_ty) == Some(64) {
        let converted = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::UConvert,
            Some(ctx.ty_uint()),
            Some(converted),
            vec![Operand::IdRef(byte_index)],
        ));
        converted
    } else {
        return None;
    };
    let word = ctx.module.fresh_id();
    let divisor = ctx.const_uint(4);
    out.push(Instruction::new(
        Op::UDiv,
        Some(ctx.ty_uint()),
        Some(word),
        vec![Operand::IdRef(byte_index), Operand::IdRef(divisor)],
    ));
    Some(word)
}

pub(in crate::passes) fn raw_uint_alias_buffer(ctx: &mut Ctx, binding: u32) -> Word {
    if let Some(var) = find_raw_uint_alias_buffer(ctx, binding) {
        return var;
    }
    let uint = ctx.ty_uint();
    let runtime = ctx.ty_runtime_array(uint);
    let block = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeStruct,
        block,
        vec![Operand::IdRef(runtime)],
    ));
    decorate_raw_uint_block(ctx, block, runtime);

    let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, block);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
    ));
    decorate_descriptor_binding(ctx, var, binding);
    ctx.interface.push(var);
    var
}

pub(in crate::passes) fn find_raw_uint_alias_buffer(ctx: &Ctx, binding: u32) -> Option<Word> {
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .filter(|inst| inst.class.opcode == Op::Variable)
        .filter_map(|inst| inst.result_id)
        .find(|var| {
            descriptor_binding(ctx, *var) == Some(binding) && is_raw_uint_buffer_var(ctx, *var)
        })
}

pub(in crate::passes) fn is_raw_uint_buffer_var(ctx: &Ctx, var: Word) -> bool {
    let Some(var_ty) = value_result_type(ctx, var) else {
        return false;
    };
    let Some(block_ty) = pointer_pointee_type(ctx, var_ty) else {
        return false;
    };
    let Some(block_def) = type_def_of(ctx, block_ty) else {
        return false;
    };
    if block_def.class.opcode != Op::TypeStruct {
        return false;
    }
    let Some(Operand::IdRef(runtime)) = block_def.operands.first() else {
        return false;
    };
    let Some(runtime_def) = type_def_of(ctx, *runtime) else {
        return false;
    };
    if runtime_def.class.opcode != Op::TypeRuntimeArray {
        return false;
    }
    let Some(Operand::IdRef(elem)) = runtime_def.operands.first() else {
        return false;
    };
    is_uint_scalar_width(ctx, *elem, 32)
}

pub(in crate::passes) fn decorate_raw_uint_block(ctx: &mut Ctx, block: Word, runtime: Word) {
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(block),
            Operand::Decoration(Decoration::Block),
        ],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::MemberDecorate,
        None,
        None,
        vec![
            Operand::IdRef(block),
            Operand::LiteralBit32(0),
            Operand::Decoration(Decoration::Offset),
            Operand::LiteralBit32(0),
        ],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(runtime),
            Operand::Decoration(Decoration::ArrayStride),
            Operand::LiteralBit32(4),
        ],
    ));
}

pub(in crate::passes) fn decorate_descriptor_binding(ctx: &mut Ctx, var: Word, binding: u32) {
    let set = ctx.descriptor_layout.set;
    decorate_binding(&mut ctx.module, var, set, binding);
}

pub(in crate::passes) fn descriptor_binding(ctx: &Ctx, var: Word) -> Option<u32> {
    ctx.module.annotations.iter().find_map(|inst| {
        if inst.class.opcode == Op::Decorate
            && inst.operands.first() == Some(&Operand::IdRef(var))
            && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
        {
            match inst.operands.get(2) {
                Some(Operand::LiteralBit32(binding)) => Some(*binding),
                _ => None,
            }
        } else {
            None
        }
    })
}

pub(in crate::passes) fn vector_type_shape(ctx: &Ctx, ty: Word) -> Option<(Word, u32)> {
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode != Op::TypeVector {
        return None;
    }
    let elem = match def.operands.first() {
        Some(Operand::IdRef(elem)) => *elem,
        _ => return None,
    };
    let lanes = match def.operands.get(1) {
        Some(Operand::LiteralBit32(lanes)) => *lanes,
        _ => return None,
    };
    Some((elem, lanes))
}

pub(in crate::passes) fn is_image_size_query(name: &str) -> bool {
    name.starts_with("air.get_width_texture")
        || name.starts_with("air.get_height_texture")
        || name.starts_with("air.get_depth_texture")
        || name.starts_with("air.get_array_size_texture")
        || name.starts_with("air.get_width_depth")
        || name.starts_with("air.get_height_depth")
        || name.starts_with("air.get_depth_depth")
}

/// Lower a derivative intrinsic (`OpDPdx`/`OpDPdy`/`OpFwidth`). On a 32-bit-float operand it is the
/// op directly; on a HALF operand (`rty` is half/half-vector) it round-trips through float, since
/// Vulkan requires these to operate on 32-bit floats. `arg` is the (half or float) operand.
pub(in crate::passes) fn half_deriv(
    ctx: &mut Ctx,
    op: Op,
    res: Word,
    rty: Word,
    arg: Word,
) -> Result<Vec<Instruction>, String> {
    if !is_f32_scalar_or_vector(ctx, rty) && !is_half_scalar_or_vector(ctx, rty) {
        return Err(format!(
            "{op:?} AIR result is not a half/float scalar or vector"
        ));
    }
    if value_result_type(ctx, arg) != Some(rty) {
        return Err(format!("{op:?} AIR operand does not match its result type"));
    }
    let float_ty = float_equivalent(ctx, rty);
    if float_ty == rty {
        // already a float type -> emit the op directly.
        return Ok(vec![Instruction::new(
            op,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(arg)],
        )]);
    }
    // half: FConvert up, derivative in float, FConvert down.
    let argf = ctx.module.fresh_id();
    let derivf = ctx.module.fresh_id();
    Ok(vec![
        Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(argf),
            vec![Operand::IdRef(arg)],
        ),
        Instruction::new(op, Some(float_ty), Some(derivf), vec![Operand::IdRef(argf)]),
        Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(derivf)],
        ),
    ])
}

/// Lower `scale * op(x)` -- a transcendental scaled by a constant AFTER the fact, at half or float
/// width, scalar or vector. `log10(x)` is `log10(2) * log2(x)`.
///
/// The composition is always evaluated in 32-bit float, converting a half operand in and the
/// result back, because Metal composes it at float width too: rounding the INTERMEDIATE to half
/// does not reproduce it. Device-measured on an Apple M3 Max / macOS 26.5.2 over all 63488 finite
/// halves, `log2(x) * log10(2)` evaluated entirely in half disagrees with Metal's `log10(half)` on
/// 8081 of them, while the same expression evaluated in float and rounded once agrees on all 63488.
pub(in crate::passes) fn lower_post_scaled_glsl_unary(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    arg: Word,
    scale: f32,
    op: GLSLstd450,
) -> Result<Vec<Instruction>, String> {
    let float_ty = float_equivalent(ctx, rty);
    if !is_f32_scalar_or_vector(ctx, float_ty) {
        return Err(
            "scaled transcendental result is not a half/float scalar or vector".to_string(),
        );
    }
    let ext = ctx.glsl();
    let n = vector_len(ctx, float_ty);
    let scale_c = splat_or_scalar(ctx, float_ty, scale, n);
    let mut out = Vec::new();
    // Convert a half operand up to f32 for the transcendental and the multiply.
    let argf = if float_ty == rty {
        arg
    } else {
        let f = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(f),
            vec![Operand::IdRef(arg)],
        ));
        f
    };
    // The multiply writes the final result directly when no half round-trip is needed.
    let scaled = if float_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    let transcendental = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(transcendental),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(op as u32),
            Operand::IdRef(argf),
        ],
    ));
    out.push(Instruction::new(
        Op::FMul,
        Some(float_ty),
        Some(scaled),
        vec![Operand::IdRef(transcendental), Operand::IdRef(scale_c)],
    ));
    if float_ty != rty {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(scaled)],
        ));
    }
    Ok(out)
}

/// Lower a GLSL.std.450 op at 32-bit float width, converting a half result and its half arguments
/// across. A float result is emitted as the plain ext-inst, so this is only a widening for half.
///
/// Two independent reasons a half op is emitted this way:
///
/// * SPIR-V Tools rejects at least Round/RoundEven directly on half vectors.
/// * SPIRV-Cross, which is what MoltenVK runs our modules through, chooses Metal's `fast::` variant
///   for a HALF `Tanh`/`Atan2` and its `precise::` variant for the float one (measured with
///   `spirv-cross --msl` on synthetic modules). Apple's `fast::tanh` returns 0 past about 44 and NaN
///   past 50, where the Metal oracle's `tanh(half)` is 1.0, so the half form is not the same
///   function at all. `Sinh`/`Cosh` are `fast::` at BOTH widths and this does not reach them.
pub(in crate::passes) fn half_glsl_op(
    ctx: &mut Ctx,
    op: GLSLstd450,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Vec<Instruction> {
    let float_ty = float_equivalent(ctx, rty);
    let ext = ctx.glsl();
    let extinst = |result_ty: Word, result: Word, operands: &[Word]| {
        let mut ops = vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(op as u32),
        ];
        ops.extend(operands.iter().map(|arg| Operand::IdRef(*arg)));
        Instruction::new(Op::ExtInst, Some(result_ty), Some(result), ops)
    };
    if float_ty == rty {
        return vec![extinst(rty, res, args)];
    }
    let mut out = Vec::with_capacity(args.len() + 2);
    let widened = args
        .iter()
        .map(|arg| {
            let argf = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(float_ty),
                Some(argf),
                vec![Operand::IdRef(*arg)],
            ));
            argf
        })
        .collect::<Vec<_>>();
    let outf = ctx.module.fresh_id();
    out.push(extinst(float_ty, outf, &widened));
    out.push(Instruction::new(
        Op::FConvert,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(outf)],
    ));
    out
}

/// Metal's `pow(x, y)` is the C `powf` contract. GLSL.std.450 `Pow` is a DIFFERENT function twice
/// over, and this lowering closes both gaps.
///
/// **`Pow` is `powr`, not `pow`.** SPIRV-Cross renders `GLSLstd450Pow` as MSL `powr` -- correctly,
/// since `Pow` is undefined for a negative base and that is exactly `powr`'s domain -- so a module
/// that emitted the ext-inst alone would be running `powr` on the device. Measured on this
/// repository's reference GPU (Apple M3 Max, macOS 26.5.2), for a NON-NEGATIVE base the two
/// disagree on exactly two rules, and on nothing else:
///
/// * `pow(x, +-0) == 1` for every `x`. `powr(0, 0)`, `powr(inf, 0)` and `powr(NaN, 0)` are all NaN.
/// * `pow(1, y) == 1` for every `y`. `powr(1, +-inf)` and `powr(1, NaN)` are NaN.
///
/// So a zero exponent answers 1 outright -- the former `0^0` guard was the one row of that rule
/// this lowering had noticed -- and a unit base answers 1 **when the exponent is not finite**.
///
/// The second guard is deliberately conjoined with `!isfinite(y)` rather than applied to the
/// magnitude, and that is not an optimization. `powr(1, y)` is 1 for every finite `y` already, so
/// the conjunction changes no answer; what it changes is that at a finite CONSTANT exponent the
/// predicate is a compile-time false and the whole select disappears. Emitting it unconditionally
/// put one extra `OpSelect` between the `Pow` and the arithmetic consuming it, and that moved a
/// real module's answer by a float ULP: `copyCubeMapWithLinearToSRGB` computes
/// `1.055 * pow(v, 1/2.4) - 0.055`, and at `v = 1` the result came back one half-ULP low, breaking
/// the device-verified case `srgb-encode-both-arms-in-one-cube-face` on 14 of its 16 texels. The
/// selected values were mathematically identical; only the instruction sequence MoltenVK's
/// compiler chose was not. **A guard that cannot fire must also not be emitted.**
///
/// **`Pow` has no negative base.** For `x < 0` Metal answers
///
/// * `pow(-2, 3) == -8` and `pow(-8, -3) == -1/512` -- an ODD integer exponent keeps the base's sign,
/// * `pow(-2, 2) == 4` and `pow(-1, inf) == 1` -- an EVEN integer exponent (infinity included) does not,
/// * `pow(-2, 2.6) == NaN` and `pow(-2, 0.5) == NaN` -- a non-integer exponent is out of the domain,
///
/// identically for `half` and `float`, for `air.pow` and `air.fast_pow`, and under both ordinary and
/// `-ffast-math` compilation. So compute `pow(|x|, y)` and reapply the sign the exponent's parity
/// earns.
///
/// Two edges of that rule are not what a `x < 0` test alone answers, and both are device-measured:
///
/// * **The sign is the sign BIT, not `x < 0`.** `pow(-0, 3)` is `-0` and `pow(-0, -3)` is `-inf`;
///   negative zero fails `x < 0` and still carries its sign through an odd integer exponent. It is
///   read with an `OpBitcast`, conjoined with `|x| == 0` so a signed NaN base can never reach it --
///   Metal's own NaN-sign behaviour is inconsistent -- `trunc(-NaN)` drops the sign and
///   `round(-NaN)` keeps it -- and this lowering does not claim anything about it.
/// * **Only a FINITE negative base leaves the domain.** `pow(-inf, 2.6)` and `pow(-inf, 0.5)` are
///   `+inf`, not NaN: an infinite base has no fractional part to be ambiguous about. Refusing every
///   non-integer exponent under a negative base answered NaN for both.
///
/// ```text
/// magnitude = powr(|x|, y)
/// integral  = trunc(y) == y;  odd = trunc(y * 0.5) * 2 != y
/// negative  = x < 0 || (|x| == 0 && signbit(x))
/// signed    = negative && integral && odd  ? -magnitude : magnitude
/// pow(x, y) = y == 0                                  ? 1
///           : (x < 0 && |x| != inf && !integral)      ? NaN
///           : (|x| == 1 && !isfinite(y))              ? 1
///           : signed
/// ```
///
/// The NaN arm sits ABOVE the unit-base arm on purpose: `pow(-1, NaN)` is NaN while
/// `pow(+1, NaN)` is 1, and a non-integer exponent under a finite negative base is what separates
/// them.
///
/// A half operand computes in f32 and narrows back. `validation/fixtures/public/
/// kernel_pow_signed_zero_and_infinite_base` is the 7 x 12 device-verified grid over all of this.
pub(in crate::passes) fn lower_metal_pow(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    base: Word,
    exponent: Word,
) -> Vec<Instruction> {
    let float_ty = float_equivalent(ctx, rty);
    let n = vector_len(ctx, float_ty);
    let bool_ty = if n > 1 {
        ctx.ty_vec_bool(n)
    } else {
        ctx.ty_bool()
    };
    let ext = ctx.glsl();
    let zero = splat_or_scalar(ctx, float_ty, 0.0, n);
    let one = splat_or_scalar(ctx, float_ty, 1.0, n);
    let two = splat_or_scalar(ctx, float_ty, 2.0, n);
    let one_half = splat_or_scalar(ctx, float_ty, 0.5, n);
    let not_a_number = splat_or_scalar(ctx, float_ty, f32::NAN, n);
    let infinity = splat_or_scalar(ctx, float_ty, f32::INFINITY, n);
    // The sign BIT, not `x < 0`: negative zero is the one base whose sign an odd integer exponent
    // still carries and whose comparison against zero is false. `OpBitcast` is exact on every
    // encoding including NaN, and the `base_is_zero` conjunction below keeps a signed NaN base out
    // of this path entirely -- Metal's own NaN-sign behaviour is inconsistent and unmeasured here.
    let int_scalar_ty = ctx.ty_sint();
    let int_ty = if n > 1 {
        ctx.ty_vec_sint(n)
    } else {
        int_scalar_ty
    };
    let zero_int_scalar = ctx.const_int_of(int_scalar_ty, 0);
    let zero_int = if n > 1 {
        splat(ctx, int_ty, zero_int_scalar, n)
    } else {
        zero_int_scalar
    };
    let mut out = Vec::new();
    let widen = |ctx: &mut Ctx, out: &mut Vec<Instruction>, value: Word| {
        if float_ty == rty {
            return value;
        }
        let wide = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(wide),
            vec![Operand::IdRef(value)],
        ));
        wide
    };
    let basef = widen(ctx, &mut out, base);
    let exponentf = widen(ctx, &mut out, exponent);
    let magnitude_base = ctx.module.fresh_id();
    let magnitude = ctx.module.fresh_id();
    let base_is_unit = ctx.module.fresh_id();
    let exponent_magnitude = ctx.module.fresh_id();
    let exponent_is_finite = ctx.module.fresh_id();
    let exponent_is_not_finite = ctx.module.fresh_id();
    let answers_unit = ctx.module.fresh_id();
    let unit_or_signed = ctx.module.fresh_id();
    let integral_exponent = ctx.module.fresh_id();
    let is_integral = ctx.module.fresh_id();
    let halved = ctx.module.fresh_id();
    let halved_integral = ctx.module.fresh_id();
    let doubled = ctx.module.fresh_id();
    let is_odd = ctx.module.fresh_id();
    let odd_integer = ctx.module.fresh_id();
    let negated = ctx.module.fresh_id();
    let base_is_zero = ctx.module.fresh_id();
    let base_bits = ctx.module.fresh_id();
    let base_sign_bit = ctx.module.fresh_id();
    let base_is_negative_zero = ctx.module.fresh_id();
    let base_is_negative = ctx.module.fresh_id();
    let base_sign_is_negative = ctx.module.fresh_id();
    let apply_sign = ctx.module.fresh_id();
    let signed = ctx.module.fresh_id();
    let base_is_finite = ctx.module.fresh_id();
    let is_not_integral = ctx.module.fresh_id();
    let base_is_finite_negative = ctx.module.fresh_id();
    let out_of_domain = ctx.module.fresh_id();
    let powered = ctx.module.fresh_id();
    let exponent_is_zero = ctx.module.fresh_id();
    let guarded = if float_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    out.extend([
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(magnitude_base),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
                Operand::IdRef(basef),
            ],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(magnitude),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Pow as u32),
                Operand::IdRef(magnitude_base),
                Operand::IdRef(exponentf),
            ],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(integral_exponent),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Trunc as u32),
                Operand::IdRef(exponentf),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(is_integral),
            vec![Operand::IdRef(exponentf), Operand::IdRef(integral_exponent)],
        ),
        Instruction::new(
            Op::FMul,
            Some(float_ty),
            Some(halved),
            vec![Operand::IdRef(exponentf), Operand::IdRef(one_half)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(halved_integral),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Trunc as u32),
                Operand::IdRef(halved),
            ],
        ),
        Instruction::new(
            Op::FMul,
            Some(float_ty),
            Some(doubled),
            vec![Operand::IdRef(halved_integral), Operand::IdRef(two)],
        ),
        Instruction::new(
            Op::FOrdNotEqual,
            Some(bool_ty),
            Some(is_odd),
            vec![Operand::IdRef(doubled), Operand::IdRef(exponentf)],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(odd_integer),
            vec![Operand::IdRef(is_integral), Operand::IdRef(is_odd)],
        ),
        Instruction::new(
            Op::FNegate,
            Some(float_ty),
            Some(negated),
            vec![Operand::IdRef(magnitude)],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(base_is_zero),
            vec![Operand::IdRef(magnitude_base), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::Bitcast,
            Some(int_ty),
            Some(base_bits),
            vec![Operand::IdRef(basef)],
        ),
        Instruction::new(
            Op::SLessThan,
            Some(bool_ty),
            Some(base_sign_bit),
            vec![Operand::IdRef(base_bits), Operand::IdRef(zero_int)],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(base_is_negative_zero),
            vec![Operand::IdRef(base_is_zero), Operand::IdRef(base_sign_bit)],
        ),
        Instruction::new(
            Op::FOrdLessThan,
            Some(bool_ty),
            Some(base_is_negative),
            vec![Operand::IdRef(basef), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::LogicalOr,
            Some(bool_ty),
            Some(base_sign_is_negative),
            vec![
                Operand::IdRef(base_is_negative),
                Operand::IdRef(base_is_negative_zero),
            ],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(apply_sign),
            vec![
                Operand::IdRef(base_sign_is_negative),
                Operand::IdRef(odd_integer),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(signed),
            vec![
                Operand::IdRef(apply_sign),
                Operand::IdRef(negated),
                Operand::IdRef(magnitude),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(base_is_unit),
            vec![Operand::IdRef(magnitude_base), Operand::IdRef(one)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(exponent_magnitude),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
                Operand::IdRef(exponentf),
            ],
        ),
        Instruction::new(
            Op::FOrdLessThan,
            Some(bool_ty),
            Some(exponent_is_finite),
            vec![Operand::IdRef(exponent_magnitude), Operand::IdRef(infinity)],
        ),
        Instruction::new(
            Op::LogicalNot,
            Some(bool_ty),
            Some(exponent_is_not_finite),
            vec![Operand::IdRef(exponent_is_finite)],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(answers_unit),
            vec![
                Operand::IdRef(base_is_unit),
                Operand::IdRef(exponent_is_not_finite),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(unit_or_signed),
            vec![
                Operand::IdRef(answers_unit),
                Operand::IdRef(one),
                Operand::IdRef(signed),
            ],
        ),
        Instruction::new(
            Op::FOrdNotEqual,
            Some(bool_ty),
            Some(base_is_finite),
            vec![Operand::IdRef(magnitude_base), Operand::IdRef(infinity)],
        ),
        Instruction::new(
            Op::LogicalNot,
            Some(bool_ty),
            Some(is_not_integral),
            vec![Operand::IdRef(is_integral)],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(base_is_finite_negative),
            vec![
                Operand::IdRef(base_is_negative),
                Operand::IdRef(base_is_finite),
            ],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(out_of_domain),
            vec![
                Operand::IdRef(base_is_finite_negative),
                Operand::IdRef(is_not_integral),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(powered),
            vec![
                Operand::IdRef(out_of_domain),
                Operand::IdRef(not_a_number),
                Operand::IdRef(unit_or_signed),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(exponent_is_zero),
            vec![Operand::IdRef(exponentf), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(guarded),
            vec![
                Operand::IdRef(exponent_is_zero),
                Operand::IdRef(one),
                Operand::IdRef(powered),
            ],
        ),
    ]);
    if float_ty != rty {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(guarded)],
        ));
    }
    out
}

pub(in crate::passes) fn lower_fast_ldexp(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if !is_f32_scalar(ctx, rty) {
        return Err(format!("{name} currently supports scalar f32 results"));
    }
    let mantissa_ty =
        value_result_type(ctx, args[0]).ok_or_else(|| format!("{name} mantissa has no type"))?;
    if mantissa_ty != rty {
        return Err(format!("{name} mantissa/result type mismatch"));
    }
    let exponent_ty =
        value_result_type(ctx, args[1]).ok_or_else(|| format!("{name} exponent has no type"))?;
    if !is_int_scalar_width(ctx, exponent_ty, 32) {
        return Err(format!("{name} exponent is not scalar i32"));
    }
    // GLSL.std.450 scales the exponent field directly, which is what Metal's `ldexp` does. Building
    // it as `x * exp2(float(n))` instead — which this used to do — materializes the scale factor as
    // its own float32, and that factor overflows for n above 127 and flushes to zero below -126
    // even when the PRODUCT is an ordinary number: `ldexp(1e-30, 200)` came back `inf` against
    // Metal's 1.6069381e30, and `ldexp(1e30, -200)` came back `0` against Metal's 6.2230154e-31.
    let ext = ctx.glsl();
    Ok(vec![Instruction::new(
        Op::ExtInst,
        Some(rty),
        Some(res),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::Ldexp as u32),
            Operand::IdRef(args[0]),
            Operand::IdRef(args[1]),
        ],
    )])
}

/// The 32-bit-float type matching `ty`'s shape: `half` -> `float`, `<N x half>` -> `<N x float>`,
/// an already-float type -> itself.
pub(in crate::passes) fn float_equivalent(ctx: &mut Ctx, ty: Word) -> Word {
    if let Some(def) = type_def_of(ctx, ty) {
        match def.class.opcode {
            Op::TypeFloat => {
                if def.operands.first() == Some(&Operand::LiteralBit32(16)) {
                    return ctx.ty_float();
                }
                return ty; // float32 already
            }
            Op::TypeVector => {
                if let (Some(Operand::IdRef(elem)), Some(Operand::LiteralBit32(n))) =
                    (def.operands.first(), def.operands.get(1))
                {
                    let n = *n;
                    if is_half_scalar(ctx, *elem) {
                        return ctx.ty_vecf(n);
                    }
                }
                return ty;
            }
            _ => {}
        }
    }
    ty
}

/// Scalar 0/1 constants matching the ELEMENT type of `rty`: half element -> half 0/1, else float
/// 0/1. Used for `saturate`/`clamp` edges so FClamp's operand types match its half/float result.
/// See the `air.fast_tanh` arm in `lower_call`: (exp2(x*2log2e) - 1) / (exp2(x*2log2e) + 1),
/// byte-faithful to Metal's overflow-to-NaN fast tanh.
pub(in crate::passes) fn lower_fast_tanh(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    x: Word,
) -> Result<Vec<Instruction>, String> {
    let elem = element_type(ctx, rty);
    let (scale_scalar, one_scalar) = if is_half_scalar(ctx, elem) {
        (ctx.const_half(2.885_39), ctx.const_half(1.0))
    } else {
        (ctx.const_float(2.885_39), ctx.const_float(1.0))
    };
    let (scale, one) = clamp_edges(ctx, rty, scale_scalar, one_scalar);
    let ext = ctx.glsl();
    let scaled = ctx.module.fresh_id();
    let t = ctx.module.fresh_id();
    let num = ctx.module.fresh_id();
    let den = ctx.module.fresh_id();
    Ok(vec![
        Instruction::new(
            Op::FMul,
            Some(rty),
            Some(scaled),
            vec![Operand::IdRef(x), Operand::IdRef(scale)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(t),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Exp2 as u32),
                Operand::IdRef(scaled),
            ],
        ),
        Instruction::new(
            Op::FSub,
            Some(rty),
            Some(num),
            vec![Operand::IdRef(t), Operand::IdRef(one)],
        ),
        Instruction::new(
            Op::FAdd,
            Some(rty),
            Some(den),
            vec![Operand::IdRef(t), Operand::IdRef(one)],
        ),
        Instruction::new(
            Op::FDiv,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(num), Operand::IdRef(den)],
        ),
    ])
}

pub(in crate::passes) fn scalar_zero_one(ctx: &mut Ctx, rty: Word) -> (Word, Word) {
    let elem = element_type(ctx, rty);
    if is_half_scalar(ctx, elem) {
        (ctx.const_half(0.0), ctx.const_half(1.0))
    } else {
        (ctx.const_float(0.0), ctx.const_float(1.0))
    }
}

#[cfg(test)]
mod tests {
    /// The arithmetic [`super::lower_metal_pow`] emits, written out in host f32 so the formula can
    /// be checked against hardware without a GPU.
    ///
    /// `powr` stands in for GLSL `Pow`, which is what SPIRV-Cross renders it as and therefore what
    /// runs on the device. Rust has no `powr`, so it is spelled out: `powr` agrees with `powf` on a
    /// non-negative base except that a zero exponent under a zero, infinite or NaN base is NaN, and
    /// `powr(1, y)` is NaN for a non-finite `y`. Modelling it as `powf` would hide the two guards
    /// this lowering exists to supply.
    fn powr(a: f32, y: f32) -> f32 {
        if y == 0.0 && (a == 0.0 || a.is_infinite() || a.is_nan()) {
            return f32::NAN;
        }
        if a == 1.0 && !y.is_finite() {
            return f32::NAN;
        }
        a.powf(y)
    }

    fn emitted_pow(x: f32, y: f32) -> f32 {
        let magnitude_base = x.abs();
        let magnitude = powr(magnitude_base, y);
        let is_integral = y == y.trunc();
        let is_odd = (y * 0.5).trunc() * 2.0 != y;
        let base_is_negative = x < 0.0;
        let sign_is_negative =
            base_is_negative || (magnitude_base == 0.0 && (x.to_bits() as i32) < 0);
        let signed = if sign_is_negative && is_integral && is_odd {
            -magnitude
        } else {
            magnitude
        };
        // `powr(1, y)` is NaN only for a non-finite `y`, and the conjunction is what lets the whole
        // select fold away wherever the exponent is a finite constant. See the doc comment.
        let unit_or_signed = if magnitude_base == 1.0 && !y.is_finite() {
            1.0
        } else {
            signed
        };
        if y == 0.0 {
            1.0
        } else if base_is_negative && magnitude_base != f32::INFINITY && !is_integral {
            f32::NAN
        } else {
            unit_or_signed
        }
    }

    /// The whole 7 x 12 grid of `validation/fixtures/public/
    /// kernel_pow_signed_zero_and_infinite_base`, read back from a Metal compute kernel on an Apple
    /// M3 Max (macOS 26.5.2) for `pow`, `precise::pow` and `fast::pow`, at both `half` and `float`,
    /// compiled with and without `-ffast-math`. All eight spellings agreed on every row.
    ///
    /// `NAN` here means "any NaN": the emitted NaN is a constant `0x7fc00000` and so is Metal's
    /// answer on every one of these rows, but this model does not carry a payload.
    #[test]
    fn emitted_pow_reproduces_metal_on_every_edge() {
        const NAN: f32 = f32::NAN;
        const INF: f32 = f32::INFINITY;
        // (base, exponent, Metal's answer). Read down: the exponent list is the same for every
        // base except `+2`, which substitutes 4 for 0.5 and -1 for 2.6 -- the only two lanes whose
        // answer would be irrational and so unwritable here.
        let measured: &[(f32, f32, f32)] = &[
            // +0: a zero exponent is 1, a negative exponent is +inf, the sign never appears.
            (0.0, 3.0, 0.0),
            (0.0, -3.0, INF),
            (0.0, 2.0, 0.0),
            (0.0, -2.0, INF),
            (0.0, 0.5, 0.0),
            (0.0, 2.6, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -0.0, 1.0),
            (0.0, INF, 0.0),
            (0.0, -INF, INF),
            (0.0, NAN, NAN),
            (0.0, 1.0, 0.0),
            // -0: identical EXCEPT that an odd integer exponent carries the sign bit.
            (-0.0, 3.0, -0.0),
            (-0.0, -3.0, -INF),
            (-0.0, 2.0, 0.0),
            (-0.0, -2.0, INF),
            (-0.0, 0.5, 0.0),
            (-0.0, 2.6, 0.0),
            (-0.0, 0.0, 1.0),
            (-0.0, -0.0, 1.0),
            (-0.0, INF, 0.0),
            (-0.0, -INF, INF),
            (-0.0, NAN, NAN),
            (-0.0, 1.0, -0.0),
            // -2: a finite negative base, the only one a non-integer exponent takes out of domain.
            (-2.0, 3.0, -8.0),
            (-2.0, -3.0, -0.125),
            (-2.0, 2.0, 4.0),
            (-2.0, -2.0, 0.25),
            (-2.0, 0.5, NAN),
            (-2.0, 2.6, NAN),
            (-2.0, 0.0, 1.0),
            (-2.0, -0.0, 1.0),
            (-2.0, INF, INF),
            (-2.0, -INF, 0.0),
            (-2.0, NAN, NAN),
            (-2.0, 1.0, -2.0),
            // -1: |x| == 1, where `powr` answers NaN for a non-finite exponent and `pow` answers 1.
            (-1.0, 3.0, -1.0),
            (-1.0, -3.0, -1.0),
            (-1.0, 2.0, 1.0),
            (-1.0, -2.0, 1.0),
            (-1.0, 0.5, NAN),
            (-1.0, 2.6, NAN),
            (-1.0, 0.0, 1.0),
            (-1.0, -0.0, 1.0),
            (-1.0, INF, 1.0),
            (-1.0, -INF, 1.0),
            (-1.0, NAN, NAN),
            (-1.0, 1.0, -1.0),
            // -inf: negative, and NOT out of domain for a non-integer exponent.
            (-INF, 3.0, -INF),
            (-INF, -3.0, -0.0),
            (-INF, 2.0, INF),
            (-INF, -2.0, 0.0),
            (-INF, 0.5, INF),
            (-INF, 2.6, INF),
            (-INF, 0.0, 1.0),
            (-INF, -0.0, 1.0),
            (-INF, INF, INF),
            (-INF, -INF, 0.0),
            (-INF, NAN, NAN),
            (-INF, 1.0, -INF),
            // +inf: where `powr(inf, 0)` is NaN and `pow` is 1.
            (INF, 3.0, INF),
            (INF, -3.0, 0.0),
            (INF, 2.0, INF),
            (INF, -2.0, 0.0),
            (INF, 0.5, INF),
            (INF, 2.6, INF),
            (INF, 0.0, 1.0),
            (INF, -0.0, 1.0),
            (INF, INF, INF),
            (INF, -INF, 0.0),
            (INF, NAN, NAN),
            (INF, 1.0, INF),
            // +2: the ordinary base, to prove the guards cost it nothing.
            (2.0, 3.0, 8.0),
            (2.0, -3.0, 0.125),
            (2.0, 2.0, 4.0),
            (2.0, -2.0, 0.25),
            (2.0, 4.0, 16.0),
            (2.0, -1.0, 0.5),
            (2.0, 0.0, 1.0),
            (2.0, -0.0, 1.0),
            (2.0, INF, INF),
            (2.0, -INF, 0.0),
            (2.0, NAN, NAN),
            (2.0, 1.0, 2.0),
        ];
        assert_eq!(measured.len(), 7 * 12);
        for &(x, y, expected) in measured {
            let got = emitted_pow(x, y);
            if expected.is_nan() {
                assert!(got.is_nan(), "pow({x}, {y}) = {got}, expected NaN");
            } else {
                assert_eq!(got.to_bits(), expected.to_bits(), "pow({x}, {y})");
            }
        }
    }

    /// The two rules that make GLSL `Pow` a different function from Metal's `pow`, pinned on the
    /// model's own `powr` so a later simplification that dropped either guard fails here as well as
    /// on the device. Both are device-measured: `powr(inf, 0)` and `powr(1, inf)` are NaN.
    #[test]
    fn the_two_powr_guards_are_what_separate_pow_from_powr() {
        assert!(powr(f32::INFINITY, 0.0).is_nan());
        assert!(powr(0.0, 0.0).is_nan());
        assert!(powr(1.0, f32::INFINITY).is_nan());
        assert_eq!(emitted_pow(f32::INFINITY, 0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(emitted_pow(0.0, 0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(emitted_pow(1.0, f32::INFINITY).to_bits(), 1.0f32.to_bits());
    }

    /// The unit-base override is conjoined with "the exponent is not finite" so the select FOLDS
    /// AWAY at a finite constant exponent, and this is why. It is not an optimization: an extra
    /// `OpSelect` between the `Pow` and the arithmetic that consumes it moved a real corpus
    /// module's answer by one float ULP -- `copyCubeMapWithLinearToSRGB`, whose
    /// `1.055 * pow(v, 1/2.4) - 0.055` at `v = 1` came back one half-ULP low and broke the
    /// device-verified case `srgb-encode-both-arms-in-one-cube-face`. The selected values were
    /// mathematically identical; only the instruction sequence MoltenVK's compiler chose was not.
    /// So a guard that cannot fire must also not be emitted.
    ///
    /// This test states the condition rather than the bytes: at a finite exponent the override's
    /// predicate is false whatever the base is, so nothing downstream of it can move.
    #[test]
    fn the_unit_base_override_cannot_fire_at_a_finite_exponent() {
        for &y in &[0.416_666_66f32, 2.4, 1.0, -3.0, 1e30, f32::MIN_POSITIVE] {
            assert!(y.is_finite());
            for &x in &[1.0f32, -1.0, 0.0, 2.0, f32::INFINITY] {
                // The emitted predicate, spelled the way the lowering builds it: `answers_unit`.
                let answers_unit = x.abs() == 1.0 && !y.is_finite();
                assert!(!answers_unit, "pow({x}, {y})");
            }
        }
    }
}
