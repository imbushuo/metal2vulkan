//! `air.convert` lowering helpers owned by the AIR-call subsystem.

use super::bfloat_glsl::{narrow_f32_to_bf16, widen_bf16_to_f32};
use super::*;

/// True if a type token (e.g. `i1`, `v3i1`) denotes a BOOL: an `i1` scalar/vector, with the `1` not
/// followed by another digit (so `i16`/`v2i16` don't match).
pub(in crate::passes) fn is_i1_type_token(tok: &str) -> bool {
    let t = tok.strip_prefix('v').map_or(tok, |rest| {
        // skip the leading vector count digits.
        rest.trim_start_matches(|c: char| c.is_ascii_digit())
    });
    t == "i1"
}

/// True if `rty`'s (element) type is an integer.
fn is_int_result(ctx: &Ctx, rty: Word) -> bool {
    let elem = element_type(ctx, rty);
    type_def_of(ctx, elem)
        .map(|d| d.class.opcode == Op::TypeInt)
        .unwrap_or(false)
}

/// Component count of value `v` (from its result type): vector -> N, scalar -> 1.
fn vector_len_of_value(ctx: &Ctx, v: Word) -> u32 {
    value_result_type(ctx, v)
        .map(|t| vector_len(ctx, t))
        .unwrap_or(1)
}

/// An integer constant of value `iv` shaped like `rty` (scalar -> the int const; vector -> a splat),
/// matching `rty`'s element width/signedness.
fn int_splat_or_scalar(ctx: &mut Ctx, rty: Word, iv: i64, n: u32) -> Word {
    let elem = element_type(ctx, rty);
    let s = ctx.const_int_of(elem, iv);
    if n <= 1 {
        s
    } else {
        splat(ctx, rty, s, n)
    }
}

/// air.convert.<dstkind><...>.<srckind>... -> OpConvert* on the result type. Kinds: f=float,
/// s=signed int, u=unsigned int. We read the first and last kind letters off the mangled name.
/// `air.convert.<dstkind><dsttype>.<srckind><srctype>` -> the SPIR-V conversion its kinds name.
/// Wide (>4-lane) AIR vectors are handled here and everything else in [`lower_convert_narrow`].
pub(super) fn lower_convert(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    // AIR vectors wider than four lanes have no SPIR-V vector type, so the emitter represents them
    // as arrays -- and no conversion instruction accepts an aggregate. Replay the conversion one
    // exact lane at a time and reconstruct the wide value. The lane conversion is the SAME
    // lowering the narrow path uses, applied to a name whose type tokens have had their `v<N>`
    // prefix stripped: this arm used to carry its own kind-to-opcode table, which was a second
    // derivation of the same fact and disagreed with the first one about bfloat (whose SPIR-V type
    // is `OpTypeInt 16`, so a wide `f`->`f` convert into it picked `OpFConvert` and was rejected)
    // and about a signed/unsigned pair of equal width (which is a bitcast only when the widths
    // match).
    if args.len() == 1
        && type_def_of(ctx, rty).is_some_and(|definition| definition.class.opcode == Op::TypeArray)
    {
        let source_ty = value_result_type(ctx, args[0])
            .ok_or_else(|| format!("{name} wide source has no result type"))?;
        let (result_element, result_lanes) = composite_shape(ctx, rty)
            .ok_or_else(|| format!("{name} wide result has no fixed element shape"))?;
        let (source_element, source_lanes) = composite_shape(ctx, source_ty)
            .ok_or_else(|| format!("{name} wide source has no fixed element shape"))?;
        if source_lanes != result_lanes {
            return Err(format!(
                "{name} wide source/result lane mismatch: {source_lanes} vs {result_lanes}"
            ));
        }
        let lane_name = scalarized_convert_name(name);
        let mut out = Vec::new();
        let mut converted = Vec::with_capacity(result_lanes as usize);
        for lane in 0..result_lanes {
            let source = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::CompositeExtract,
                Some(source_element),
                Some(source),
                vec![Operand::IdRef(args[0]), Operand::LiteralBit32(lane)],
            ));
            // The extract is not in the module yet, so publish its type on the phase map the
            // narrow lowering reads its source shape from.
            ctx.phase_value_types
                .get_or_insert_with(Default::default)
                .insert(source, source_element);
            let result = ctx.module.fresh_id();
            out.extend(lower_convert_narrow(
                ctx,
                &lane_name,
                result,
                result_element,
                &[source],
            )?);
            converted.push(Operand::IdRef(result));
        }
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(rty),
            Some(res),
            converted,
        ));
        return Ok(out);
    }
    lower_convert_narrow(ctx, name, res, rty, args)
}

/// One `air.convert` whose result is a scalar or a SPIR-V vector (at most four lanes).
fn lower_convert_narrow(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    // `air.convert` names are `air.convert.<dstkind>.<dsttype>.<srckind>.<srctype>`. An `i1` token is a
    // BOOL; whether it is the DEST type or the SOURCE type decides the lowering direction.
    // e.g. air.convert.f.v3f32.u.v3i1 (i1 = src) / air.convert.u.i1.f.f32 (i1 = dst).
    let parts: Vec<&str> = name.trim_start_matches("air.convert.").split('.').collect();
    // The type tokens are at even-ish positions; precisely, dst type = parts[1], src type = last part.
    let dst_type = parts.get(1).copied().unwrap_or("");
    let src_type = parts.last().copied().unwrap_or("");

    // Bool SOURCE (`...u.v3i1`): a vector/scalar of i1 -> numeric via OpSelect of 1/0 (OpConvert*
    // rejects bool input). The result element type may be float OR int (`air.convert.s.i32.u.i1`).
    if is_i1_type_token(src_type) {
        let n = vector_len(ctx, rty);
        let elem_is_int = is_int_result(ctx, rty);
        let (one, zero) = if elem_is_int {
            (
                int_splat_or_scalar(ctx, rty, 1, n),
                int_splat_or_scalar(ctx, rty, 0, n),
            )
        } else {
            (
                splat_or_scalar(ctx, rty, 1.0, n),
                splat_or_scalar(ctx, rty, 0.0, n),
            )
        };
        return Ok(vec![Instruction::new(
            Op::Select,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(args[0]),
                Operand::IdRef(one),
                Operand::IdRef(zero),
            ],
        )]);
    }
    // Bool DEST (`air.convert.u.i1.f.f32`): a numeric -> i1 (bool) via a `!= 0` comparison. The source
    // kind (the last single-letter token) picks float vs int compare. `rty` here is `%bool`.
    if is_i1_type_token(dst_type) {
        // src kind = the kind letter immediately before the src type.
        let src_kind = parts
            .get(parts.len().saturating_sub(2))
            .and_then(|p| p.chars().next())
            .unwrap_or('f');
        let n = vector_len_of_value(ctx, args[0]);
        if src_kind == 'f' {
            let zero = splat_or_scalar(ctx, value_result_type(ctx, args[0]).unwrap_or(rty), 0.0, n);
            return Ok(vec![Instruction::new(
                Op::FUnordNotEqual,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(args[0]), Operand::IdRef(zero)],
            )]);
        }
        let src_ty = value_result_type(ctx, args[0]).unwrap_or(rty);
        let zero = int_splat_or_scalar(ctx, src_ty, 0, n);
        return Ok(vec![Instruction::new(
            Op::INotEqual,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0]), Operand::IdRef(zero)],
        )]);
    }
    // find dst kind and src kind by scanning for single-letter f/s/u tokens.
    let kinds: Vec<char> = parts
        .iter()
        .filter(|p| p.len() == 1 && matches!(p.chars().next().unwrap(), 'f' | 's' | 'u'))
        .map(|p| p.chars().next().unwrap())
        .collect();
    let (dst, src) = match (kinds.first(), kinds.last()) {
        (Some(d), Some(s)) if kinds.len() >= 2 => (*d, *s),
        _ => return Err(format!("cannot parse convert kinds from {name}")),
    };
    // bfloat16 has no SPIR-V type, so the emitter models a bf16 value as its raw `OpTypeInt 16` bit
    // pattern (the top 16 bits of an f32). A convert whose source or dest is bf16 (`...f.bf16` /
    // `f.bf16...`) therefore can't go straight through OpConvert*: the int16-typed operand fails
    // "expected float input". Widen the bf16 bits to f32 at the source / narrow f32 to bf16 bits at
    // the dest, around the existing float<->int conversion. `bf16` is a stable AIR type token.
    if token_is_bf16(src_type) || token_is_bf16(dst_type) {
        return lower_convert_bf16(ctx, res, rty, args, dst_type, src_type, dst, src);
    }
    // The 8-bit float formats are the same shape of problem as bf16 -- no SPIR-V type, carried as
    // `i8` bits -- and unlike bf16 nothing here models them. Left to the `f`/`f` arm below they
    // become an `OpFConvert` over an integer operand, which the owned check refuses several phases
    // later with a shape complaint that names neither the family nor the reason. Refuse by name.
    //
    // What it would take to model them is an exact bit rule, not an approximation: e5m2 shares
    // fp16's five exponent bits and bias, e4m3fn does not and has no infinities. Neither can be
    // settled on this machine -- the installed Metal toolchain has no 8-bit float type, so there is
    // no oracle to check a decode against.
    if let Some(token) = [src_type, dst_type]
        .into_iter()
        .find(|token| token_is_narrow_float(token))
    {
        return Err(format!(
            "{name} converts the 8-bit float format {}, which has no SPIR-V type and no modelled \
             bit layout here",
            scalarize_convert_token(token)
        ));
    }
    if (dst, src) == ('f', 's') {
        let mut out = Vec::new();
        let (signed, _) = bitcast_to_integer_signedness(ctx, &mut out, args[0], true)?;
        out.push(Instruction::new(
            Op::ConvertSToF,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(signed)],
        ));
        return Ok(out);
    }
    if (dst, src) == ('u', 's') || (dst, src) == ('s', 'u') {
        let mut out = Vec::new();
        let signed_source = src == 's';
        let (input, input_ty) =
            bitcast_to_integer_signedness(ctx, &mut out, args[0], signed_source)?;
        let same_shape = scalar_bit_width(ctx, input_ty) == scalar_bit_width(ctx, rty)
            && vector_len(ctx, input_ty) == vector_len(ctx, rty);
        let instruction = if same_shape {
            copy_or_bitcast_result(rty, res, input_ty, input)
        } else {
            let opcode = if signed_source {
                Op::SConvert
            } else {
                Op::UConvert
            };
            Instruction::new(opcode, Some(rty), Some(res), vec![Operand::IdRef(input)])
        };
        out.push(instruction);
        return Ok(out);
    }
    // A float-to-integer convert is emitted bare, at EVERY destination width. SPIR-V leaves an
    // out-of-range or NaN input undefined, so this is a statement about Metal, and it is
    // device-measured rather than assumed (Apple M3 Max / macOS 26.5.2): Metal's cast SATURATES at
    // every width and sends NaN to zero. `uint(+inf)` is 4294967295, `int(-inf)` is -2147483648,
    // `ushort(70000)` is 65535, `short(-32769)` is -32768, `char(255)` is 127, and every NaN is 0
    // -- and SPIRV-Cross renders these as exactly that MSL cast, so the contract is reproduced by
    // construction. All 72 rows of `float-to-integer-saturates-at-every-width` Match on that.
    //
    // The narrow widths used to wrap the input in an `FClamp` first, which was wrong twice. Metal's
    // `clamp(NaN, lo, hi)` returns `lo`, so a signed narrow destination answered -32768 or -128
    // where Metal answers 0. And the clamp edges are spelled in the SOURCE float type, which for a
    // half source cannot hold a 16-bit destination's own maximum: `short(half(40000))` came back
    // 32752 instead of 32767, on 3071 of the 65536 halves. Both are gone with the clamp.
    let op = match (dst, src) {
        ('f', 'u') => Op::ConvertUToF,
        ('u', 'f') => Op::ConvertFToU,
        ('s', 'f') => Op::ConvertFToS,
        ('u', 'u') => Op::UConvert,
        ('s', 's') => Op::SConvert,
        // float<->float of differing widths (half<->float): a single OpFConvert handles both fpext and
        // fptrunc, scalar or vector (`air.convert.f.v4f32.f.v4f16`, `...v4f16.f.v4f32`, `v2f32.f.v2f16`).
        ('f', 'f') => Op::FConvert,
        _ => return Err(format!("unhandled convert kinds {dst}->{src} in {name}")),
    };
    Ok(vec![Instruction::new(
        op,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(args[0])],
    )])
}

fn bitcast_to_integer_signedness(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    signed: bool,
) -> Result<(Word, Word), String> {
    let ty = value_result_type(ctx, value).ok_or("air.convert integer source has no type")?;
    let target_ty = integer_type_like(ctx, ty, signed)?;
    if target_ty == ty {
        return Ok((value, ty));
    }
    let cast = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(target_ty),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    Ok((cast, target_ty))
}

fn integer_type_like(ctx: &mut Ctx, ty: Word, signed: bool) -> Result<Word, String> {
    let def = type_def_of(ctx, ty).ok_or("air.convert integer source type is undefined")?;
    match def.class.opcode {
        Op::TypeInt => {
            let bits = literal_u32(def.operands.first())
                .ok_or("air.convert integer source int missing width")?;
            let current_signed = literal_u32(def.operands.get(1))
                .ok_or("air.convert integer source int missing signedness")?;
            if current_signed == u32::from(signed) {
                Ok(ty)
            } else {
                Ok(integer_type(ctx, bits, signed))
            }
        }
        Op::TypeVector => {
            let elem = id_ref(def.operands.first())
                .ok_or("air.convert integer source vector missing element type")?;
            let lanes = literal_u32(def.operands.get(1))
                .ok_or("air.convert integer source vector missing length")?;
            let target_elem = integer_type_like(ctx, elem, signed)?;
            if target_elem == elem {
                Ok(ty)
            } else {
                Ok(vector_type(ctx, target_elem, lanes))
            }
        }
        _ => Err("air.convert source is not an integer scalar/vector".into()),
    }
}

fn integer_type(ctx: &mut Ctx, bits: u32, signed: bool) -> Word {
    let key = SynthCacheKey::IntType { bits, signed };
    if let Some(&id) = ctx.synth_cache.get(&key) {
        return id;
    }
    let signedness = u32::from(signed);
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(bits))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(signedness))
        {
            if let Some(rid) = inst.result_id {
                ctx.synth_cache.insert(key, rid);
                return rid;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::TypeInt,
        None,
        Some(id),
        vec![
            Operand::LiteralBit32(bits),
            Operand::LiteralBit32(signedness),
        ],
    ));
    ctx.synth_cache.insert(key, id);
    id
}

fn vector_type(ctx: &mut Ctx, elem: Word, lanes: u32) -> Word {
    let key = SynthCacheKey::VecType { elem, lanes };
    if let Some(&id) = ctx.synth_cache.get(&key) {
        return id;
    }
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
            if let Some(rid) = inst.result_id {
                ctx.synth_cache.insert(key, rid);
                return rid;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(id),
        vec![Operand::IdRef(elem), Operand::LiteralBit32(lanes)],
    ));
    ctx.synth_cache.insert(key, id);
    id
}

fn id_ref(op: Option<&Operand>) -> Option<Word> {
    match op {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

fn literal_u32(op: Option<&Operand>) -> Option<u32> {
    match op {
        Some(Operand::LiteralBit32(v)) => Some(*v),
        _ => None,
    }
}

/// True if a convert type token denotes a bf16 (`bf16`, `v2bf16`, `v4bf16`, ...). bf16 is modeled as
/// `OpTypeInt 16` storage, so it needs widen/narrow around float conversions.
/// A convert type token with its vector prefix removed: `v8bf16` -> `bf16`, `v3i1` -> `i1`,
/// `f32` -> `f32`. Used to re-spell a wide `air.convert` as the per-lane convert it decomposes into.
fn scalarize_convert_token(tok: &str) -> &str {
    let Some(rest) = tok.strip_prefix('v') else {
        return tok;
    };
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        tok
    } else {
        &rest[digits..]
    }
}

/// The same `air.convert` name with every type token scalarized.
fn scalarized_convert_name(name: &str) -> String {
    let tokens = name
        .trim_start_matches("air.convert.")
        .split('.')
        .map(scalarize_convert_token)
        .collect::<Vec<_>>();
    format!("air.convert.{}", tokens.join("."))
}

fn token_is_bf16(tok: &str) -> bool {
    tok.contains("bf16")
}

/// True if a convert type token denotes one of LLVM's 8-bit float formats (`f8e5m2`, `f8e4m3`,
/// `f8e4m3fn`, and their `v8...` vector spellings). They reach AIR as `i8` and SPIR-V has no type
/// for them.
fn token_is_narrow_float(tok: &str) -> bool {
    tok.contains("f8e")
}

/// Lane count encoded in a convert type token: `v4f32` -> 4, `bf16` / `f32` -> 1.
fn token_lanes(tok: &str) -> u32 {
    tok.strip_prefix('v')
        .map(|rest| {
            rest.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(1)
        })
        .unwrap_or(1)
}

/// The largest f32 strictly below `2^k`, as its bit pattern: exponent `k - 1` with an all-ones
/// significand. Used as the clamp bound for a float -> integer round trip of a `k`-bit range.
fn largest_f32_below_pow2(k: u32) -> f32 {
    f32::from_bits(((k - 1 + 127) << 23) | 0x007f_ffff)
}

/// An integer zero shaped like `int_ty` (scalar, or a splat for an `n`-lane vector).
fn shaped_int_zero(ctx: &mut Ctx, int_ty: Word, n: u32) -> Word {
    let elem = element_type(ctx, int_ty);
    let scalar = ctx.const_int_of(elem, 0);
    if n > 1 {
        splat(ctx, int_ty, scalar, n)
    } else {
        scalar
    }
}

/// Round `f32val` -- the round-to-nearest-even f32 of the integer `int_val` -- TO ODD, so that a
/// following round-to-nearest-even narrowing to bf16 lands on the same value a single correctly
/// rounded integer -> bf16 conversion would.
///
/// bf16 keeps 8 significant bits and f32 keeps 24, so converting a wide integer through f32 rounds
/// twice. An integer just past a bf16 midpoint can round back exactly ONTO that midpoint in f32 and
/// then be sent the other way by ties-to-even: `bfloat(33685505u)` is 0x4C01 on Metal and 0x4C00
/// through an f32 intermediate. Round-to-odd is the standard cure -- an odd f32 significand is never
/// itself a bf16 value or a bf16 midpoint (those have at least the low 15 significand bits clear),
/// so the second rounding can no longer see a tie that the first one manufactured, and it keeps the
/// side of the midpoint the true integer was on.
///
/// The two bracketing f32 values of an inexact conversion are `t` (toward zero) and `t + ulp`, and
/// exactly one of them has an odd significand; this picks it. `int_val` is only inexact when it has
/// more than 24 significant bits, so callers skip this for sources narrower than 32 bits.
fn round_int_to_odd_f32(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    int_val: Word,
    int_ty: Word,
    f32val: Word,
    signed: bool,
    n: u32,
) -> Word {
    let f32_ty = ty_f32_shaped(ctx, n);
    let u32_ty = ty_u32_shaped(ctx, n);
    let bool_ty = ty_bool_shaped(ctx, n);
    let width = scalar_bit_width(ctx, int_ty);

    // Clamp before the round trip: nearest-even can carry a source at the very top of the integer
    // range up to 2^width (unsigned) or 2^(width-1) (signed), and OpConvertFToU/S does not define
    // an out-of-range operand. Every value the clamp moves is inexact and rounds to odd anyway --
    // the bound's own significand is all ones -- so clamping does not change any answer.
    let bound_pow2 = if signed { width - 1 } else { width };
    let bound = splat_or_scalar(ctx, f32_ty, largest_f32_below_pow2(bound_pow2), n);
    let ext = ctx.glsl();
    let clamped = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(f32_ty),
        Some(clamped),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::FMin as u32),
            Operand::IdRef(f32val),
            Operand::IdRef(bound),
        ],
    ));

    // OpConvertFToS/U truncates toward zero, so `back` is the exact integer value of `clamped`.
    let back = ctx.module.fresh_id();
    out.push(Instruction::new(
        if signed {
            Op::ConvertFToS
        } else {
            Op::ConvertFToU
        },
        Some(int_ty),
        Some(back),
        vec![Operand::IdRef(clamped)],
    ));
    let exact = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IEqual,
        Some(bool_ty),
        Some(exact),
        vec![Operand::IdRef(back), Operand::IdRef(int_val)],
    ));

    // Did nearest-even round AWAY from zero? For a non-negative source that is `back > int_val`;
    // for a negative one the magnitude grew when `back < int_val`.
    let away = if signed {
        let zero = shaped_int_zero(ctx, int_ty, n);
        let non_negative = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::SGreaterThanEqual,
            Some(bool_ty),
            Some(non_negative),
            vec![Operand::IdRef(int_val), Operand::IdRef(zero)],
        ));
        let greater = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::SGreaterThan,
            Some(bool_ty),
            Some(greater),
            vec![Operand::IdRef(back), Operand::IdRef(int_val)],
        ));
        let less = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::SLessThan,
            Some(bool_ty),
            Some(less),
            vec![Operand::IdRef(back), Operand::IdRef(int_val)],
        ));
        let away = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Select,
            Some(bool_ty),
            Some(away),
            vec![
                Operand::IdRef(non_negative),
                Operand::IdRef(greater),
                Operand::IdRef(less),
            ],
        ));
        away
    } else {
        let away = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::UGreaterThan,
            Some(bool_ty),
            Some(away),
            vec![Operand::IdRef(back), Operand::IdRef(int_val)],
        ));
        away
    };

    // f32 is sign-magnitude, so subtracting one from the bit pattern steps the MAGNITUDE down one
    // ulp for either sign (a zero significand borrows into the exponent, which is exactly the next
    // smaller magnitude). Step down when nearest-even rounded away, then set the significand's low
    // bit: that names the odd member of the bracketing pair.
    let bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(u32_ty),
        Some(bits),
        vec![Operand::IdRef(clamped)],
    ));
    let one = shaped_u32_const(ctx, n, 1);
    let stepped_down = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ISub,
        Some(u32_ty),
        Some(stepped_down),
        vec![Operand::IdRef(bits), Operand::IdRef(one)],
    ));
    let toward_zero = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(u32_ty),
        Some(toward_zero),
        vec![
            Operand::IdRef(away),
            Operand::IdRef(stepped_down),
            Operand::IdRef(bits),
        ],
    ));
    let odd = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseOr,
        Some(u32_ty),
        Some(odd),
        vec![Operand::IdRef(toward_zero), Operand::IdRef(one)],
    ));
    let selected = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(u32_ty),
        Some(selected),
        vec![
            Operand::IdRef(exact),
            Operand::IdRef(bits),
            Operand::IdRef(odd),
        ],
    ));
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(f32_ty),
        Some(result),
        vec![Operand::IdRef(selected)],
    ));
    result
}

/// Lower an `air.convert` whose source and/or dest is bf16. The conversion is split around an f32
/// intermediate: the source is brought to f32 (widening bf16 bits, or an ordinary int/float->f32
/// convert), then f32 is taken to the dest (narrowing to bf16 bits, or an ordinary f32->int/float
/// convert). This keeps the bf16 leg honest — a real f32 value flows through the arithmetic convert.
#[allow(clippy::too_many_arguments)]
fn lower_convert_bf16(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    dst_type: &str,
    src_type: &str,
    dst_kind: char,
    src_kind: char,
) -> Result<Vec<Instruction>, String> {
    let arg = *args
        .first()
        .ok_or("air.convert bf16: missing source operand")?;
    let src_is_bf16 = token_is_bf16(src_type);
    let dst_is_bf16 = token_is_bf16(dst_type);
    let n = if src_is_bf16 {
        token_lanes(src_type)
    } else {
        token_lanes(dst_type)
    };
    let mut out = Vec::new();

    // 1. Bring the source to f32.
    let f32_ty = ty_f32_shaped(ctx, n);
    let f32val = if src_is_bf16 {
        widen_bf16_to_f32(ctx, &mut out, arg, n)
    } else {
        let src_ty = value_result_type(ctx, arg).unwrap_or(rty);
        match src_kind {
            // f16/f32 source: widen/copy to f32 (FConvert rejects equal widths, so copy when already f32).
            'f' => {
                if scalar_bit_width(ctx, src_ty) == 32 {
                    arg
                } else {
                    let id = ctx.module.fresh_id();
                    out.push(Instruction::new(
                        Op::FConvert,
                        Some(f32_ty),
                        Some(id),
                        vec![Operand::IdRef(arg)],
                    ));
                    id
                }
            }
            // Integer source. The f32 hop is exact for anything narrower than 32 bits (f32 keeps
            // 24 significant bits), but a wider source can round twice on the way to bf16 -- so
            // round the intermediate TO ODD first when the destination is bf16.
            _ => {
                let signed = src_kind == 's';
                let id = ctx.module.fresh_id();
                out.push(Instruction::new(
                    if signed {
                        Op::ConvertSToF
                    } else {
                        Op::ConvertUToF
                    },
                    Some(f32_ty),
                    Some(id),
                    vec![Operand::IdRef(arg)],
                ));
                if dst_is_bf16 && scalar_bit_width(ctx, src_ty) >= 32 {
                    round_int_to_odd_f32(ctx, &mut out, arg, src_ty, id, signed, n)
                } else {
                    id
                }
            }
        }
    };

    // 2. Take the f32 intermediate to the dest.
    if dst_is_bf16 {
        narrow_f32_to_bf16(ctx, &mut out, f32val, n, rty, res);
    } else {
        let op = match dst_kind {
            's' => Op::ConvertFToS,
            'u' => Op::ConvertFToU,
            // float dest: f16 narrows via FConvert; f32 is identity (CopyObject, FConvert rejects it).
            _ => {
                if scalar_bit_width(ctx, rty) == 32 {
                    Op::CopyObject
                } else {
                    Op::FConvert
                }
            }
        };
        out.push(Instruction::new(
            op,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(f32val)],
        ));
    }
    Ok(out)
}
