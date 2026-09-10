//! Half scalar/vector stage transport is 32-bit; AIR values stay half inside the shader.

use super::*;

pub(super) fn float_shape(ctx: &Ctx, ty: Word) -> Option<(u32, u32)> {
    let definition = type_def_of(ctx, ty)?;
    match (definition.class.opcode, definition.operands.as_slice()) {
        (Op::TypeFloat, [Operand::LiteralBit32(bits)]) => Some((*bits, 1)),
        (Op::TypeVector, [Operand::IdRef(component), Operand::LiteralBit32(lanes)]) => {
            float_shape(ctx, *component).map(|(bits, _)| (bits, *lanes))
        }
        _ => None,
    }
}

/// Float16 arithmetic and 16-bit buffer storage do not authorize 16-bit Input/Output storage.
/// Widen half values exactly on output and narrow the interpolated/fetched value at shader entry,
/// using the same representation for independently translated stages.
pub(super) fn float_interface_type(ctx: &mut Ctx, ty: Word) -> Word {
    match float_shape(ctx, ty) {
        Some((16, 1)) => ctx.ty_float(),
        Some((16, lanes)) => ctx.ty_vecf(lanes),
        _ => ty,
    }
}
