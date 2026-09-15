//! Format-aware storage-write conversion, applied after runtime image-format specialization.
//!
//! Quantizing to an exactly representable destination value before `OpImageWrite` avoids relying
//! on a driver's implicit float conversion mode. Binary16 conversion uses only 32-bit operations;
//! it requires neither shaderFloat16 nor optional Vulkan float-control execution modes.

use crate::spirv_module::{load_bytes, Block, Function, Instruction, Module, Operand};
use spirv::{Decoration, FunctionControl, ImageFormat, Op, Word};
use std::collections::HashMap;

/// Native conversion policy of the source Metal device, not a pipeline descriptor override.
///
/// An AIR `.rte`/`.rtz` write already selected its mode and ignores this policy. `Default` retains
/// the target's native conversion and makes no cross-device parity claim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureWriteRoundingMode {
    #[default]
    Default,
    TowardZero,
    ToNearestEven,
}

/// Component encoding of a runtime storage-image view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureWriteFormat {
    Float16,
    Float32,
    Integer,
    /// Normalized integer formats are outside Metal's floating-point texture rounding control.
    Normalized,
}

/// Runtime format facts for one descriptor. Required for formatless images.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureWriteTarget {
    pub descriptor_set: u32,
    pub binding: u32,
    pub format: TextureWriteFormat,
}

/// Runtime specialization ABI. Local-size specializations occupy 0..=2. These constants are
/// executable operands: stripping debug information cannot discard a write's rounding contract.
pub const NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID: u32 = 3;
/// First per-write destination-precision SpecId. Values are zero (no narrowing) or 16 (binary16).
pub const TEXTURE_WRITE_FORMAT_SPEC_ID_BASE: u32 = 16;

const _: () = {
    assert!(NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID < TEXTURE_WRITE_FORMAT_SPEC_ID_BASE);
    let mut dimension = 0;
    while dimension < crate::reflect::KERNEL_LOCAL_SIZE_SPEC_IDS.len() {
        assert!(
            crate::reflect::KERNEL_LOCAL_SIZE_SPEC_IDS[dimension]
                < NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID
        );
        dimension += 1;
    }
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AirWriteRounding {
    Native,
    TowardZero,
    ToNearestEven,
}

impl AirWriteRounding {
    pub(crate) fn from_intrinsic(name: &str) -> Result<Self, String> {
        let suffix = name
            .strip_prefix("air.write_texture_")
            .and_then(|name| name.split_once('.').map(|(_, suffix)| suffix))
            .ok_or("texture rounding: malformed AIR write intrinsic")?;
        match suffix.split('.').next() {
            Some("rte") => Ok(Self::ToNearestEven),
            Some("rtz") => Ok(Self::TowardZero),
            Some("rtp" | "rtn" | "rtne") => {
                Err(format!("texture rounding: unsupported AIR mode in {name}"))
            }
            _ => Ok(Self::Native),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WriteConversion {
    Air(AirWriteRounding),
    PreserveImageblockSlice,
}

#[derive(Default)]
pub(crate) struct WriteRoundingLowering {
    native_mode: Option<Word>,
    next_format: u32,
    helpers: HashMap<(Word, WriteConversion), Word>,
    quantizers: HashMap<(Word, TextureWriteRoundingMode), Word>,
}

impl WriteRoundingLowering {
    pub(crate) fn wrap(
        &mut self,
        module: &mut Module,
        out: &mut Vec<Instruction>,
        mode: AirWriteRounding,
        texel: Word,
        ty: Word,
    ) -> Result<Word, String> {
        self.wrap_producer(module, out, WriteConversion::Air(mode), texel, ty)
    }

    pub(crate) fn preserve_imageblock_slice(
        &mut self,
        module: &mut Module,
        out: &mut Vec<Instruction>,
        texel: Word,
        ty: Word,
    ) -> Result<Word, String> {
        self.wrap_producer(
            module,
            out,
            WriteConversion::PreserveImageblockSlice,
            texel,
            ty,
        )
    }

    fn wrap_producer(
        &mut self,
        module: &mut Module,
        out: &mut Vec<Instruction>,
        mode: WriteConversion,
        texel: Word,
        ty: Word,
    ) -> Result<Word, String> {
        let uint = declaration(
            module,
            Op::TypeInt,
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        );
        let native_mode = match self.native_mode {
            Some(id) => id,
            None => {
                let id = specialization(module, uint, NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID, 0)?;
                self.native_mode = Some(id);
                id
            }
        };
        let format_id = TEXTURE_WRITE_FORMAT_SPEC_ID_BASE
            .checked_add(self.next_format)
            .ok_or("texture rounding: specialization index overflow")?;
        self.next_format = self
            .next_format
            .checked_add(1)
            .ok_or("texture rounding: specialization index overflow")?;
        let format = specialization(module, uint, format_id, 0)?;
        let helper = match self.helpers.get(&(ty, mode)) {
            Some(id) => *id,
            None => {
                let helper = match mode {
                    WriteConversion::Air(mode) => {
                        let zero = *self
                            .quantizers
                            .entry((ty, TextureWriteRoundingMode::TowardZero))
                            .or_insert_with(|| {
                                half_quantizer(module, ty, TextureWriteRoundingMode::TowardZero)
                            });
                        let nearest = *self
                            .quantizers
                            .entry((ty, TextureWriteRoundingMode::ToNearestEven))
                            .or_insert_with(|| {
                                half_quantizer(module, ty, TextureWriteRoundingMode::ToNearestEven)
                            });
                        write_quantizer(module, ty, uint, mode, zero, nearest)
                    }
                    WriteConversion::PreserveImageblockSlice => {
                        imageblock_slice_passthrough(module, ty, uint)
                    }
                };
                self.helpers.insert((ty, mode), helper);
                helper
            }
        };
        let result = module.fresh_id();
        out.push(Instruction::new(
            Op::FunctionCall,
            Some(ty),
            Some(result),
            vec![
                Operand::IdRef(helper),
                Operand::IdRef(texel),
                Operand::IdRef(format),
                Operand::IdRef(native_mode),
            ],
        ));
        Ok(result)
    }
}

fn specialization(module: &mut Module, ty: Word, index: u32, value: u32) -> Result<Word, String> {
    if module.annotations.iter().any(|inst| {
        matches!(inst.operands.as_slice(),
            [Operand::IdRef(_), Operand::Decoration(Decoration::SpecId), Operand::LiteralBit32(existing)]
            if *existing == index)
    }) {
        return Err(format!("texture rounding: specialization id {index} already occupied"));
    }
    let id = module.fresh_id();
    module.types_global_values.push(Instruction::new(
        Op::SpecConstant,
        Some(ty),
        Some(id),
        vec![Operand::LiteralBit32(value)],
    ));
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(id),
            Operand::Decoration(Decoration::SpecId),
            Operand::LiteralBit32(index),
        ],
    ));
    Ok(id)
}

/// Specialize storage writes in a translated module. The caller must first specialize explicit
/// image formats to the actual bound views and must include the source-native `mode` and `targets`
/// in cache identity. Pipeline descriptor rounding must not replace the mode encoded by AIR.
///
/// Unknown image formats without runtime facts, conflicting formatless types, and unimplemented
/// destination encodings return an error rather than silently using Vulkan's native rounding.
pub fn specialize_texture_write_rounding(
    words: &[u32],
    mode: TextureWriteRoundingMode,
    targets: &[TextureWriteTarget],
) -> Result<Vec<u32>, String> {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut module = load_bytes(&bytes).map_err(|error| error.to_string())?;
    specialize_module(&mut module, mode, targets, true)?;
    Ok(module.assemble())
}

pub(crate) fn initialize_declared_write_formats(module: &mut Module) -> Result<(), String> {
    specialize_module(module, TextureWriteRoundingMode::Default, &[], false)
}

fn specialize_module(
    module: &mut Module,
    mode: TextureWriteRoundingMode,
    targets: &[TextureWriteTarget],
    require_runtime_formats: bool,
) -> Result<(), String> {
    let mut definitions = HashMap::new();
    for inst in module.all_inst_iter() {
        if let Some(id) = inst.result_id {
            definitions.insert(id, inst.clone());
        }
    }
    let mut bindings = HashMap::<Word, (Option<u32>, Option<u32>)>::new();
    for inst in &module.annotations {
        if let [Operand::IdRef(id), Operand::Decoration(decoration), Operand::LiteralBit32(value)] =
            inst.operands.as_slice()
        {
            let entry = bindings.entry(*id).or_default();
            match decoration {
                Decoration::DescriptorSet => entry.0 = Some(*value),
                Decoration::Binding => entry.1 = Some(*value),
                _ => {}
            }
        }
    }
    let mut target_formats = HashMap::new();
    for target in targets {
        if let Some(previous) =
            target_formats.insert((target.descriptor_set, target.binding), target.format)
        {
            if previous != target.format {
                return Err("texture rounding: descriptor has conflicting runtime formats".into());
            }
        }
    }
    let mut formats = HashMap::new();
    for (variable, (set, binding)) in bindings {
        let (Some(set), Some(binding)) = (set, binding) else {
            continue;
        };
        let Some(&format) = target_formats.get(&(set, binding)) else {
            continue;
        };
        let pointer = definitions
            .get(&variable)
            .and_then(|inst| inst.result_type)
            .and_then(|id| definitions.get(&id))
            .filter(|inst| inst.class.opcode == Op::TypePointer);
        let Some(Operand::IdRef(image_ty)) = pointer.and_then(|inst| inst.operands.get(1)) else {
            return Err("texture rounding target is not an image descriptor".into());
        };
        let mut image_ty = *image_ty;
        let mut seen = std::collections::HashSet::new();
        loop {
            if !seen.insert(image_ty) {
                return Err("texture rounding: recursive descriptor type".into());
            }
            let ty = definitions
                .get(&image_ty)
                .ok_or("texture rounding: missing descriptor type")?;
            match ty.class.opcode {
                Op::TypeImage => break,
                Op::TypeArray | Op::TypeRuntimeArray => {
                    let Some(Operand::IdRef(element)) = ty.operands.first() else {
                        return Err("texture rounding: descriptor array has no element type".into());
                    };
                    image_ty = *element;
                }
                _ => return Err("texture rounding target is not an image descriptor".into()),
            }
        }
        if let Some(previous) = formats.insert(image_ty, format) {
            if previous != format {
                return Err(
                    "texture rounding: shared formatless image type has conflicting views".into(),
                );
            }
        }
    }
    let spec_ids: HashMap<_, _> = module.annotations.iter().filter_map(|inst| {
        match inst.operands.as_slice() {
            [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralBit32(index)] => Some((*id, *index)),
            _ => None,
        }
    }).collect();
    let mut values = HashMap::new();
    if let Some((&id, _)) = spec_ids
        .iter()
        .find(|(_, index)| **index == NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID)
    {
        values.insert(
            id,
            match mode {
                TextureWriteRoundingMode::Default => 0,
                TextureWriteRoundingMode::TowardZero => 1,
                TextureWriteRoundingMode::ToNearestEven => 2,
            },
        );
    }
    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::ImageWrite {
                    continue;
                }
                let Some(Operand::IdRef(image)) = inst.operands.first() else {
                    return Err("texture rounding: malformed image write".into());
                };
                let image_ty = definitions
                    .get(image)
                    .and_then(|inst| inst.result_type)
                    .ok_or("texture rounding: image write has no image type")?;
                let image = definitions
                    .get(&image_ty)
                    .filter(|inst| inst.class.opcode == Op::TypeImage)
                    .ok_or("texture rounding: write operand is not an image")?;
                let Some(Operand::ImageFormat(format)) = image.operands.get(6) else {
                    return Err("texture rounding: image has no format".into());
                };
                let declared = component_format(*format)?;
                let format = match (declared, formats.get(&image_ty).copied()) {
                    (Some(declared), Some(runtime)) if declared != runtime => {
                        return Err(
                            "texture rounding: image format must be specialized before conversion"
                                .into(),
                        );
                    }
                    (Some(declared), _) => declared,
                    (None, Some(runtime)) => runtime,
                    (None, None) if !require_runtime_formats => continue,
                    (None, None) => {
                        return Err(
                            "texture rounding: formatless image requires runtime format".into()
                        )
                    }
                };
                let Some(Operand::IdRef(texel)) = inst.operands.get(2) else {
                    return Err("texture rounding: image write has no texel".into());
                };
                let value = definitions
                    .get(texel)
                    .ok_or("texture rounding: texel has no definition")?;
                let selector = if value.class.opcode == Op::FunctionCall {
                    match value.operands.as_slice() {
                        [Operand::IdRef(_), Operand::IdRef(_), Operand::IdRef(format), Operand::IdRef(native)]
                            if spec_ids
                                .get(format)
                                .is_some_and(|id| *id >= TEXTURE_WRITE_FORMAT_SPEC_ID_BASE)
                                && spec_ids.get(native)
                                    == Some(&NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID) =>
                        {
                            Some(*format)
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let Some(selector) = selector else {
                    if format == TextureWriteFormat::Float16
                        && mode != TextureWriteRoundingMode::Default
                    {
                        return Err(
                            "texture rounding: floating write lacks AIR rounding provenance".into(),
                        );
                    }
                    continue;
                };
                let precision = if format == TextureWriteFormat::Float16 {
                    16
                } else {
                    0
                };
                if let Some(previous) = values.insert(selector, precision) {
                    if previous != precision {
                        return Err("texture rounding: one write selector reaches conflicting image formats".into());
                    }
                }
            }
        }
    }
    for inst in &mut module.types_global_values {
        if let Some(value) = inst.result_id.and_then(|id| values.get(&id)) {
            if inst.class.opcode != Op::SpecConstant {
                return Err("texture rounding: invalid specialization operand".into());
            }
            inst.operands = vec![Operand::LiteralBit32(*value)];
        }
    }
    Ok(())
}

fn component_format(format: ImageFormat) -> Result<Option<TextureWriteFormat>, String> {
    use TextureWriteFormat as F;
    Ok(Some(match format {
        ImageFormat::Unknown => return Ok(None),
        ImageFormat::R16f | ImageFormat::Rg16f | ImageFormat::Rgba16f => F::Float16,
        ImageFormat::R32f | ImageFormat::Rg32f | ImageFormat::Rgba32f => F::Float32,
        // MSL §1.6.7 applies the rounding control to floating-point pixel types, not UNORM/SNORM.
        ImageFormat::R8
        | ImageFormat::Rg8
        | ImageFormat::Rgba8
        | ImageFormat::R16
        | ImageFormat::Rg16
        | ImageFormat::Rgba16
        | ImageFormat::R8Snorm
        | ImageFormat::Rg8Snorm
        | ImageFormat::Rgba8Snorm
        | ImageFormat::R16Snorm
        | ImageFormat::Rg16Snorm
        | ImageFormat::Rgba16Snorm
        | ImageFormat::Rgb10A2 => F::Normalized,
        ImageFormat::R8ui
        | ImageFormat::R16ui
        | ImageFormat::R32ui
        | ImageFormat::Rg8ui
        | ImageFormat::Rg16ui
        | ImageFormat::Rg32ui
        | ImageFormat::Rgba8ui
        | ImageFormat::Rgba16ui
        | ImageFormat::Rgba32ui
        | ImageFormat::R8i
        | ImageFormat::R16i
        | ImageFormat::R32i
        | ImageFormat::Rg8i
        | ImageFormat::Rg16i
        | ImageFormat::Rg32i
        | ImageFormat::Rgba8i
        | ImageFormat::Rgba16i
        | ImageFormat::Rgba32i
        | ImageFormat::Rgb10a2ui => F::Integer,
        other => {
            return Err(format!(
                "texture rounding: unsupported destination format {other:?}"
            ))
        }
    }))
}

struct Quantizer<'a> {
    module: &'a mut Module,
    instructions: Vec<Instruction>,
    uint: Word,
    uvec: Word,
    bvec: Word,
    fvec: Word,
    constants: HashMap<u32, Word>,
}

impl Quantizer<'_> {
    fn constant(&mut self, value: u32) -> Word {
        if let Some(id) = self.constants.get(&value) {
            return *id;
        }
        let scalar = self.module.fresh_id();
        self.module.types_global_values.push(Instruction::new(
            Op::Constant,
            Some(self.uint),
            Some(scalar),
            vec![Operand::LiteralBit32(value)],
        ));
        let id = self.module.fresh_id();
        self.module.types_global_values.push(Instruction::new(
            Op::ConstantComposite,
            Some(self.uvec),
            Some(id),
            vec![Operand::IdRef(scalar); 4],
        ));
        self.constants.insert(value, id);
        id
    }

    fn op(&mut self, opcode: Op, ty: Word, args: &[Word]) -> Word {
        let id = self.module.fresh_id();
        self.instructions.push(Instruction::new(
            opcode,
            Some(ty),
            Some(id),
            args.iter().copied().map(Operand::IdRef).collect(),
        ));
        id
    }

    fn binary(&mut self, opcode: Op, a: Word, b: Word) -> Word {
        self.op(opcode, self.uvec, &[a, b])
    }

    fn literal(&mut self, opcode: Op, a: Word, b: u32) -> Word {
        let b = self.constant(b);
        self.binary(opcode, a, b)
    }

    fn compare(&mut self, opcode: Op, a: Word, b: u32) -> Word {
        let b = self.constant(b);
        self.op(opcode, self.bvec, &[a, b])
    }

    fn select(&mut self, predicate: Word, yes: Word, no: Word) -> Word {
        self.op(Op::Select, self.uvec, &[predicate, yes, no])
    }

    fn rounded_shift(&mut self, value: Word, shift: Word, mode: TextureWriteRoundingMode) -> Word {
        let quotient = self.binary(Op::ShiftRightLogical, value, shift);
        if mode == TextureWriteRoundingMode::TowardZero {
            return quotient;
        }
        let one = self.constant(1);
        let limit = self.binary(Op::ShiftLeftLogical, one, shift);
        let mask = self.literal(Op::ISub, limit, 1);
        let remainder = self.binary(Op::BitwiseAnd, value, mask);
        let halfway = self.literal(Op::ShiftRightLogical, limit, 1);
        let greater = self.op(Op::UGreaterThan, self.bvec, &[remainder, halfway]);
        let equal = self.op(Op::IEqual, self.bvec, &[remainder, halfway]);
        let odd = self.literal(Op::BitwiseAnd, quotient, 1);
        let odd = self.compare(Op::INotEqual, odd, 0);
        let tie_up = self.op(Op::LogicalAnd, self.bvec, &[equal, odd]);
        let up = self.op(Op::LogicalOr, self.bvec, &[greater, tie_up]);
        let zero = self.constant(0);
        let increment = self.select(up, one, zero);
        self.binary(Op::IAdd, quotient, increment)
    }
}

fn declaration(module: &mut Module, opcode: Op, operands: Vec<Operand>) -> Word {
    if let Some(id) = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == opcode && inst.operands == operands)
        .and_then(|inst| inst.result_id)
    {
        return id;
    }
    let id = module.fresh_id();
    module
        .types_global_values
        .push(Instruction::new(opcode, None, Some(id), operands));
    id
}

fn scalar_constant(module: &mut Module, ty: Word, value: u32) -> Word {
    if let Some(id) = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Constant
                && inst.result_type == Some(ty)
                && inst.operands == [Operand::LiteralBit32(value)]
        })
        .and_then(|inst| inst.result_id)
    {
        return id;
    }
    let id = module.fresh_id();
    module.types_global_values.push(Instruction::new(
        Op::Constant,
        Some(ty),
        Some(id),
        vec![Operand::LiteralBit32(value)],
    ));
    id
}

fn emit(module: &mut Module, out: &mut Vec<Instruction>, op: Op, ty: Word, args: &[Word]) -> Word {
    let id = module.fresh_id();
    out.push(Instruction::new(
        op,
        Some(ty),
        Some(id),
        args.iter().copied().map(Operand::IdRef).collect(),
    ));
    id
}

// The existing slice producer has its own conversion semantics, not air.write_texture's policy.
// An executable identity carries that distinction through serialization without debug markers.
fn imageblock_slice_passthrough(module: &mut Module, fvec: Word, uint: Word) -> Word {
    let function_type = declaration(
        module,
        Op::TypeFunction,
        vec![
            Operand::IdRef(fvec),
            Operand::IdRef(fvec),
            Operand::IdRef(uint),
            Operand::IdRef(uint),
        ],
    );
    let function = module.fresh_id();
    let input = module.fresh_id();
    let format = module.fresh_id();
    let native = module.fresh_id();
    let label = module.fresh_id();
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(fvec),
            Some(function),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(function_type),
            ],
        )),
        parameters: [(fvec, input), (uint, format), (uint, native)]
            .into_iter()
            .map(|(ty, id)| Instruction::new(Op::FunctionParameter, Some(ty), Some(id), vec![]))
            .collect(),
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: vec![Instruction::new(
                Op::ReturnValue,
                None,
                None,
                vec![Operand::IdRef(input)],
            )],
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });
    function
}

fn write_quantizer(
    module: &mut Module,
    fvec: Word,
    uint: Word,
    mode: AirWriteRounding,
    zero: Word,
    nearest: Word,
) -> Word {
    let boolean = declaration(module, Op::TypeBool, vec![]);
    let bvec = declaration(
        module,
        Op::TypeVector,
        vec![Operand::IdRef(boolean), Operand::LiteralBit32(4)],
    );
    let function_type = declaration(
        module,
        Op::TypeFunction,
        vec![
            Operand::IdRef(fvec),
            Operand::IdRef(fvec),
            Operand::IdRef(uint),
            Operand::IdRef(uint),
        ],
    );
    let function = module.fresh_id();
    let input = module.fresh_id();
    let format = module.fresh_id();
    let native = module.fresh_id();
    let label = module.fresh_id();
    let mut out = Vec::new();
    let sixteen = scalar_constant(module, uint, 16);
    let half = emit(module, &mut out, Op::IEqual, boolean, &[format, sixteen]);
    let (quantized, enabled) = match mode {
        AirWriteRounding::TowardZero => (
            emit(module, &mut out, Op::FunctionCall, fvec, &[zero, input]),
            half,
        ),
        AirWriteRounding::ToNearestEven => (
            emit(module, &mut out, Op::FunctionCall, fvec, &[nearest, input]),
            half,
        ),
        AirWriteRounding::Native => {
            let rtz = emit(module, &mut out, Op::FunctionCall, fvec, &[zero, input]);
            let rte = emit(module, &mut out, Op::FunctionCall, fvec, &[nearest, input]);
            let two = scalar_constant(module, uint, 2);
            let use_rte = emit(module, &mut out, Op::IEqual, boolean, &[native, two]);
            let use_rte = emit(
                module,
                &mut out,
                Op::CompositeConstruct,
                bvec,
                &[use_rte; 4],
            );
            let value = emit(module, &mut out, Op::Select, fvec, &[use_rte, rte, rtz]);
            let zero = scalar_constant(module, uint, 0);
            let explicit = emit(module, &mut out, Op::INotEqual, boolean, &[native, zero]);
            let enabled = emit(module, &mut out, Op::LogicalAnd, boolean, &[half, explicit]);
            (value, enabled)
        }
    };
    let enabled = emit(
        module,
        &mut out,
        Op::CompositeConstruct,
        bvec,
        &[enabled; 4],
    );
    let result = emit(
        module,
        &mut out,
        Op::Select,
        fvec,
        &[enabled, quantized, input],
    );
    out.push(Instruction::new(
        Op::ReturnValue,
        None,
        None,
        vec![Operand::IdRef(result)],
    ));
    module.functions.push(Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(fvec),
            Some(function),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(function_type),
            ],
        )),
        parameters: [(fvec, input), (uint, format), (uint, native)]
            .into_iter()
            .map(|(ty, id)| Instruction::new(Op::FunctionParameter, Some(ty), Some(id), vec![]))
            .collect(),
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: out,
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    });
    function
}

fn half_quantizer(module: &mut Module, fvec: Word, mode: TextureWriteRoundingMode) -> Word {
    let uint = declaration(
        module,
        Op::TypeInt,
        vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
    );
    let boolean = declaration(module, Op::TypeBool, vec![]);
    let uvec = declaration(
        module,
        Op::TypeVector,
        vec![Operand::IdRef(uint), Operand::LiteralBit32(4)],
    );
    let bvec = declaration(
        module,
        Op::TypeVector,
        vec![Operand::IdRef(boolean), Operand::LiteralBit32(4)],
    );
    let function_type = declaration(
        module,
        Op::TypeFunction,
        vec![Operand::IdRef(fvec), Operand::IdRef(fvec)],
    );
    let function_id = module.fresh_id();
    let input = module.fresh_id();
    let label = module.fresh_id();
    let mut q = Quantizer {
        module,
        instructions: Vec::new(),
        uint,
        uvec,
        bvec,
        fvec,
        constants: HashMap::new(),
    };
    let bits = q.op(Op::Bitcast, uvec, &[input]);
    let magnitude = q.literal(Op::BitwiseAnd, bits, 0x7fff_ffff);
    let sign = q.literal(Op::BitwiseAnd, bits, 0x8000_0000);
    let exponent = q.literal(Op::ShiftRightLogical, magnitude, 23);
    let small = q.compare(Op::ULessThan, exponent, 113);
    let tiny = q.compare(Op::ULessThan, exponent, 102);
    let large = q.compare(Op::UGreaterThanEqual, exponent, 143);
    let special = q.compare(Op::IEqual, exponent, 255);
    let significand = q.literal(Op::BitwiseAnd, magnitude, 0x007f_ffff);
    let significand = q.literal(Op::BitwiseOr, significand, 0x0080_0000);
    // Every evaluated shift is in 13..24, even on lanes whose result is discarded.
    let base = q.constant(126);
    let shift = q.binary(Op::ISub, base, exponent);
    let thirteen = q.constant(13);
    let twenty_four = q.constant(24);
    let shift = q.select(small, shift, thirteen);
    let shift = q.select(tiny, twenty_four, shift);
    let subnormal = q.rounded_shift(significand, shift, mode);
    let zero = q.constant(0);
    let subnormal = q.select(tiny, zero, subnormal);
    let normal = q.literal(Op::ISub, magnitude, 0x3800_0000);
    let normal = q.rounded_shift(normal, thirteen, mode);
    let half = q.select(small, subnormal, normal);
    let overflow = q.constant(if mode == TextureWriteRoundingMode::TowardZero {
        0x7bff
    } else {
        0x7c00
    });
    let half = q.select(large, overflow, half);
    let half_small = q.compare(Op::ULessThan, half, 0x400);
    let half_infinite = q.compare(Op::IEqual, half, 0x7c00);
    let widened = q.literal(Op::ShiftLeftLogical, half, 13);
    let widened = q.literal(Op::IAdd, widened, 0x3800_0000);
    let sub_float = q.op(Op::ConvertUToF, q.fvec, &[half]);
    let scale_bits = q.constant(0x3380_0000); // 2^-24: binary16's subnormal quantum.
    let scale = q.op(Op::Bitcast, q.fvec, &[scale_bits]);
    let sub_float = q.op(Op::FMul, q.fvec, &[sub_float, scale]);
    let sub_bits = q.op(Op::Bitcast, q.uvec, &[sub_float]);
    let widened = q.select(half_small, sub_bits, widened);
    let infinity = q.constant(0x7f80_0000);
    let widened = q.select(half_infinite, infinity, widened);
    let widened = q.binary(Op::BitwiseOr, widened, sign);
    // NaN/Inf retain their classes and signs; image-format conversion owns NaN payload encoding.
    let widened = q.select(special, bits, widened);
    let result = q.op(Op::Bitcast, q.fvec, &[widened]);
    q.instructions.push(Instruction::new(
        Op::ReturnValue,
        None,
        None,
        vec![Operand::IdRef(result)],
    ));
    let function = Function {
        def: Some(Instruction::new(
            Op::Function,
            Some(fvec),
            Some(function_id),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(function_type),
            ],
        )),
        parameters: vec![Instruction::new(
            Op::FunctionParameter,
            Some(fvec),
            Some(input),
            vec![],
        )],
        blocks: vec![Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: q.instructions,
        }],
        end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
    };
    q.module.functions.push(function);
    function_id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_values(words: &[u32]) -> HashMap<u32, u32> {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let module = load_bytes(&bytes).unwrap();
        let constants: HashMap<_, _> = module
            .types_global_values
            .iter()
            .filter_map(|inst| match (inst.result_id, inst.operands.as_slice()) {
                (Some(id), [Operand::LiteralBit32(value)])
                    if inst.class.opcode == Op::SpecConstant =>
                {
                    Some((id, *value))
                }
                _ => None,
            })
            .collect();
        module.annotations.iter().filter_map(|inst| {
            match inst.operands.as_slice() {
                [Operand::IdRef(id), Operand::Decoration(Decoration::SpecId), Operand::LiteralBit32(index)] => {
                    constants.get(id).map(|value| (*index, *value))
                }
                _ => None,
            }
        }).collect()
    }

    fn evaluate_function(module: &Module, values: &mut [u32], id: Word, args: &[u32]) -> u32 {
        let function = module
            .functions
            .iter()
            .find(|function| function.def.as_ref().and_then(|inst| inst.result_id) == Some(id))
            .unwrap();
        for (parameter, argument) in function.parameters.iter().zip(args) {
            values[parameter.result_id.unwrap() as usize] = *argument;
        }
        for inst in &function.blocks[0].instructions {
            let ids = inst
                .operands
                .iter()
                .map(|operand| {
                    let Operand::IdRef(id) = operand else {
                        panic!("instruction operand")
                    };
                    *id
                })
                .collect::<Vec<_>>();
            if inst.class.opcode == Op::ReturnValue {
                return values[ids[0] as usize];
            }
            let a = values[ids[0] as usize];
            let b = ids.get(1).map_or(0, |id| values[*id as usize]);
            let result = match inst.class.opcode {
                Op::FunctionCall => {
                    let args = ids[1..]
                        .iter()
                        .map(|id| values[*id as usize])
                        .collect::<Vec<_>>();
                    evaluate_function(module, values, ids[0], &args)
                }
                Op::CompositeConstruct | Op::Bitcast => a,
                Op::BitwiseAnd => a & b,
                Op::BitwiseOr => a | b,
                Op::ShiftRightLogical => a >> b,
                Op::ShiftLeftLogical => a << b,
                Op::ISub => a.wrapping_sub(b),
                Op::IAdd => a.wrapping_add(b),
                Op::ULessThan => u32::from(a < b),
                Op::UGreaterThan => u32::from(a > b),
                Op::UGreaterThanEqual => u32::from(a >= b),
                Op::IEqual => u32::from(a == b),
                Op::INotEqual => u32::from(a != b),
                Op::LogicalAnd => u32::from(a != 0 && b != 0),
                Op::LogicalOr => u32::from(a != 0 || b != 0),
                Op::Select => {
                    if a != 0 {
                        b
                    } else {
                        values[ids[2] as usize]
                    }
                }
                Op::ConvertUToF => (a as f32).to_bits(),
                Op::FMul => (f32::from_bits(a) * f32::from_bits(b)).to_bits(),
                other => panic!("unimplemented interpreter opcode {other:?}"),
            };
            values[inst.result_id.unwrap() as usize] = result;
        }
        panic!("function did not return")
    }

    fn constant_values(module: &Module) -> Vec<u32> {
        let mut values = vec![0; module.id_bound() as usize];
        for inst in &module.types_global_values {
            if matches!(inst.class.opcode, Op::Constant | Op::SpecConstant) {
                if let [Operand::LiteralBit32(value)] = inst.operands.as_slice() {
                    values[inst.result_id.unwrap() as usize] = *value;
                }
            } else if inst.class.opcode == Op::ConstantComposite {
                let Operand::IdRef(scalar) = inst.operands[0] else {
                    panic!("constant splat")
                };
                values[inst.result_id.unwrap() as usize] = values[scalar as usize];
            }
        }
        values
    }

    fn evaluate_image_writes(module: &Module, input: u32) -> Vec<u32> {
        module.all_inst_iter().filter(|inst| inst.class.opcode == Op::ImageWrite).map(|write| {
            let Operand::IdRef(texel) = write.operands[2] else { panic!("write texel") };
            let call = module.all_inst_iter().find(|inst| inst.result_id == Some(texel)).unwrap();
            let [Operand::IdRef(callee), Operand::IdRef(_), Operand::IdRef(format), Operand::IdRef(native)] =
                call.operands.as_slice() else { panic!("rounding call") };
            let mut values = constant_values(module);
            let args = [input, values[*format as usize], values[*native as usize]];
            evaluate_function(module, &mut values, *callee, &args)
        }).collect()
    }

    #[test]
    fn imageblock_slice_provenance_preserves_existing_conversion_for_every_policy() {
        let mut module = Module::new();
        let float = declaration(&mut module, Op::TypeFloat, vec![Operand::LiteralBit32(32)]);
        let vector = declaration(
            &mut module,
            Op::TypeVector,
            vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
        );
        let input = module.fresh_id();
        let mut lowering = WriteRoundingLowering::default();
        let mut out = Vec::new();
        lowering
            .preserve_imageblock_slice(&mut module, &mut out, input, vector)
            .unwrap();
        let Operand::IdRef(function) = out[0].operands[0] else {
            panic!("producer wrapper")
        };
        assert_eq!(
            module.functions.len(),
            1,
            "a slice must not acquire either AIR quantizer"
        );
        for native in 0..=2 {
            for format in [0, 16] {
                for bits in [
                    0x3f80_3000,
                    0xbf80_3000,
                    0x477f_f000,
                    0x3300_0000,
                    0,
                    0x8000_0000,
                    0x7f80_0000,
                    0x7fc0_1234,
                ] {
                    let mut values = constant_values(&module);
                    assert_eq!(
                        evaluate_function(&module, &mut values, function, &[bits, format, native]),
                        bits,
                        "slice conversion must remain owned by its existing producer"
                    );
                }
            }
        }
    }

    #[test]
    fn per_write_air_mode_survives_conflicting_native_policy_and_non_half_formats() {
        for air in [
            AirWriteRounding::Native,
            AirWriteRounding::TowardZero,
            AirWriteRounding::ToNearestEven,
        ] {
            let mut module = Module::new();
            let float = declaration(&mut module, Op::TypeFloat, vec![Operand::LiteralBit32(32)]);
            let vector = declaration(
                &mut module,
                Op::TypeVector,
                vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
            );
            let input = module.fresh_id();
            let mut lowering = WriteRoundingLowering::default();
            let mut out = Vec::new();
            lowering
                .wrap(&mut module, &mut out, air, input, vector)
                .unwrap();
            let Operand::IdRef(function) = out[0].operands[0] else {
                panic!("wrapper")
            };
            for native in 0..=2 {
                for format in [0, 16] {
                    for bits in [
                        0x3f80_3000u32,
                        0xbf80_3000,
                        65520.0f32.to_bits(),
                        (-65520.0f32).to_bits(),
                        0x3300_0000,
                        0x8000_0000,
                        0x7f80_0000,
                    ] {
                        let mut values = constant_values(&module);
                        let source_mode = match air {
                            AirWriteRounding::Native => native,
                            AirWriteRounding::TowardZero => 1,
                            AirWriteRounding::ToNearestEven => 2,
                        };
                        let value = f32::from_bits(bits);
                        let expected = if format != 16 || source_mode == 0 {
                            value
                        } else {
                            let mut half = crate::float16::f32_to_f16_bits(value);
                            if source_mode == 1
                                && value.is_finite()
                                && widen_half(half).abs() > value.abs()
                            {
                                half -= 1;
                            }
                            widen_half(half)
                        };
                        assert_eq!(
                            evaluate_function(
                                &module,
                                &mut values,
                                function,
                                &[bits, format, native]
                            ),
                            expected.to_bits(),
                            "air={air:?} native={native} format={format} value={value:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn air_write_rounding_is_an_intrinsic_operand_contract() {
        for shape in ["1d", "2d", "2d_array", "3d", "cube", "buffer"] {
            for suffix in ["v4f32", "v4f16"] {
                assert_eq!(
                    AirWriteRounding::from_intrinsic(&format!(
                        "air.write_texture_{shape}.{suffix}"
                    ))
                    .unwrap(),
                    AirWriteRounding::Native
                );
                assert_eq!(
                    AirWriteRounding::from_intrinsic(&format!(
                        "air.write_texture_{shape}.rte.{suffix}"
                    ))
                    .unwrap(),
                    AirWriteRounding::ToNearestEven
                );
                assert_eq!(
                    AirWriteRounding::from_intrinsic(&format!(
                        "air.write_texture_{shape}.rtz.{suffix}"
                    ))
                    .unwrap(),
                    AirWriteRounding::TowardZero
                );
                assert!(AirWriteRounding::from_intrinsic(&format!(
                    "air.write_texture_{shape}.rtn.{suffix}"
                ))
                .is_err());
            }
        }
    }

    fn widen_half(bits: u16) -> f32 {
        let magnitude = u32::from(bits & 0x7fff);
        let value = if magnitude < 0x400 {
            (magnitude as f32) * 2.0f32.powi(-24)
        } else if magnitude >= 0x7c00 {
            f32::from_bits(0x7f80_0000 | ((magnitude & 0x3ff) << 13))
        } else {
            f32::from_bits((magnitude << 13) + 0x3800_0000)
        };
        f32::from_bits(value.to_bits() | (u32::from(bits & 0x8000) << 16))
    }

    /// Execute the actual generated instruction stream, not a parallel copy of the lowering.
    struct Oracle {
        values: Vec<u32>,
        input: usize,
        output: usize,
        operations: Vec<(Op, usize, Vec<usize>)>,
    }

    impl Oracle {
        fn new(mode: TextureWriteRoundingMode) -> Self {
            let mut module = Module::new();
            let float = declaration(&mut module, Op::TypeFloat, vec![Operand::LiteralBit32(32)]);
            let vector = declaration(
                &mut module,
                Op::TypeVector,
                vec![Operand::IdRef(float), Operand::LiteralBit32(4)],
            );
            half_quantizer(&mut module, vector, mode);
            let mut values = vec![0; module.id_bound() as usize];
            for inst in &module.types_global_values {
                if inst.class.opcode == Op::Constant {
                    let Operand::LiteralBit32(value) = inst.operands[0] else {
                        panic!("constant")
                    };
                    values[inst.result_id.unwrap() as usize] = value;
                } else if inst.class.opcode == Op::ConstantComposite {
                    let Operand::IdRef(scalar) = inst.operands[0] else {
                        panic!("splat")
                    };
                    values[inst.result_id.unwrap() as usize] = values[scalar as usize];
                }
            }
            let function = &module.functions[0];
            let input = function.parameters[0].result_id.unwrap() as usize;
            let mut operations = Vec::new();
            let mut output = 0;
            for inst in &function.blocks[0].instructions {
                let operands = inst
                    .operands
                    .iter()
                    .map(|operand| {
                        let Operand::IdRef(id) = operand else {
                            panic!("operation operand")
                        };
                        *id as usize
                    })
                    .collect::<Vec<_>>();
                if inst.class.opcode == Op::ReturnValue {
                    output = operands[0];
                } else {
                    operations.push((
                        inst.class.opcode,
                        inst.result_id.unwrap() as usize,
                        operands,
                    ));
                }
            }
            Self {
                values,
                input,
                output,
                operations,
            }
        }

        fn run(&mut self, value: f32) -> f32 {
            self.values[self.input] = value.to_bits();
            for (op, result, args) in &self.operations {
                let a = self.values[args[0]];
                let b = args.get(1).map_or(0, |id| self.values[*id]);
                self.values[*result] = match op {
                    Op::Bitcast => a,
                    Op::BitwiseAnd => a & b,
                    Op::BitwiseOr => a | b,
                    Op::ShiftRightLogical => a >> b,
                    Op::ShiftLeftLogical => a << b,
                    Op::ISub => a.wrapping_sub(b),
                    Op::IAdd => a.wrapping_add(b),
                    Op::ULessThan => u32::from(a < b),
                    Op::UGreaterThan => u32::from(a > b),
                    Op::UGreaterThanEqual => u32::from(a >= b),
                    Op::IEqual => u32::from(a == b),
                    Op::INotEqual => u32::from(a != b),
                    Op::LogicalAnd => u32::from(a != 0 && b != 0),
                    Op::LogicalOr => u32::from(a != 0 || b != 0),
                    Op::Select => {
                        if a != 0 {
                            b
                        } else {
                            self.values[args[2]]
                        }
                    }
                    Op::ConvertUToF => (a as f32).to_bits(),
                    Op::FMul => (f32::from_bits(a) * f32::from_bits(b)).to_bits(),
                    other => panic!("unimplemented oracle instruction {other:?}"),
                };
            }
            f32::from_bits(self.values[self.output])
        }
    }

    #[test]
    fn half_quantizer_covers_every_binary16_rounding_boundary() {
        let mut nearest = Oracle::new(TextureWriteRoundingMode::ToNearestEven);
        let mut zero = Oracle::new(TextureWriteRoundingMode::TowardZero);
        for lower in 0u16..0x7bff {
            let lo = widen_half(lower);
            let hi = widen_half(lower + 1);
            let midpoint = (lo + hi) * 0.5;
            for bits in [
                midpoint.to_bits() - 1,
                midpoint.to_bits(),
                midpoint.to_bits() + 1,
            ] {
                for sign in [0, 0x8000_0000] {
                    let value = f32::from_bits(bits | sign);
                    let expected = widen_half(crate::float16::f32_to_f16_bits(value));
                    assert_eq!(
                        nearest.run(value).to_bits(),
                        expected.to_bits(),
                        "RTE {value:?}"
                    );
                    assert_eq!(
                        zero.run(value).to_bits(),
                        lo.to_bits() | sign,
                        "RTZ {value:?}"
                    );
                }
            }
        }
        for value in [
            0.0,
            -0.0,
            65504.0,
            65520.0,
            70000.0,
            -70000.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let expected = widen_half(crate::float16::f32_to_f16_bits(value));
            assert_eq!(
                nearest.run(value).to_bits(),
                expected.to_bits(),
                "RTE {value:?}"
            );
            let expected_zero = if value.is_finite() && value.abs() > 65504.0 {
                65504.0f32.copysign(value)
            } else {
                expected
            };
            assert_eq!(
                zero.run(value).to_bits(),
                expected_zero.to_bits(),
                "RTZ {value:?}"
            );
        }
        for bits in [0x7fc0_0000, 0x7f80_0001, 0xff80_0001] {
            assert!(nearest.run(f32::from_bits(bits)).is_nan());
            assert!(zero.run(f32::from_bits(bits)).is_nan());
        }
    }

    #[test]
    fn rounding_specializes_authored_float_image_writes_without_float16_capability() {
        let source = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @write_image(ptr addrspace(1) %image, ptr addrspace(1) %values) {
entry:
  %value = load <4 x float>, ptr addrspace(1) %values, align 16
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %image, <2 x i32> zeroinitializer, <4 x float> %value, i32 0, i32 2)
  ret void
}
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0}
!0 = !{ptr @write_image, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"image"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"values"}
"#;
        let scratch = std::env::temp_dir().join("m2v-authored-write-rounding");
        std::fs::create_dir_all(&scratch).unwrap();
        let bytes = crate::translate_sanitized_native_with_options(
            source,
            crate::passes::Stage::Kernel,
            &scratch,
            crate::passes::TransformOptions::default()
                .with_runtime_storage_image(
                    0,
                    crate::reflect::RuntimeStorageImageState {
                        format: crate::reflect::RuntimeStorageImageFormat::Rgba16Float,
                        capabilities: crate::reflect::RuntimeStorageImageCapabilities {
                            storage_image: true,
                            ..Default::default()
                        },
                    },
                )
                .unwrap(),
        )
        .expect("authored translation");
        let original = load_bytes(&bytes).unwrap().assemble();
        let mut unmarked = load_bytes(&bytes).unwrap();
        let inputs: HashMap<_, _> = unmarked
            .all_inst_iter()
            .filter(|inst| inst.class.opcode == Op::FunctionCall && inst.operands.len() == 4)
            .map(|inst| (inst.result_id.unwrap(), inst.operands[1].clone()))
            .collect();
        for function in &mut unmarked.functions {
            for block in &mut function.blocks {
                for inst in &mut block.instructions {
                    if inst.class.opcode == Op::ImageWrite {
                        let Operand::IdRef(texel) = inst.operands[2] else {
                            panic!("write texel")
                        };
                        inst.operands[2] = inputs[&texel].clone();
                    }
                }
            }
        }
        let unmarked = unmarked.assemble();
        assert_eq!(
            specialize_texture_write_rounding(&unmarked, TextureWriteRoundingMode::Default, &[],)
                .unwrap(),
            unmarked
        );
        for mode in [
            TextureWriteRoundingMode::TowardZero,
            TextureWriteRoundingMode::ToNearestEven,
        ] {
            assert!(specialize_texture_write_rounding(&unmarked, mode, &[])
                .unwrap_err()
                .contains("lacks AIR rounding provenance"));
        }
        for (suffix, expected_rtz, expected_rte) in [
            ("", 0x3f80_2000, 0x3f80_4000),
            ("rte.", 0x3f80_4000, 0x3f80_4000),
            ("rtz.", 0x3f80_2000, 0x3f80_2000),
        ] {
            let source = source.replace(
                "air.write_texture_2d.v4f32",
                &format!("air.write_texture_2d.{suffix}v4f32"),
            );
            let compiled = crate::translate_sanitized_native_with_options(
                &source,
                crate::passes::Stage::Kernel,
                &scratch,
                crate::passes::TransformOptions::default()
                    .with_runtime_storage_image(
                        0,
                        crate::reflect::RuntimeStorageImageState {
                            format: crate::reflect::RuntimeStorageImageFormat::Rgba16Float,
                            capabilities: crate::reflect::RuntimeStorageImageCapabilities {
                                storage_image: true,
                                ..Default::default()
                            },
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
            let words = load_bytes(&compiled).unwrap().assemble();
            for (native, expected) in [
                (TextureWriteRoundingMode::TowardZero, expected_rtz),
                (TextureWriteRoundingMode::ToNearestEven, expected_rte),
            ] {
                let specialized = specialize_texture_write_rounding(&words, native, &[]).unwrap();
                let bytes: Vec<_> = specialized
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                crate::tools::spirv_val_bytes(&bytes, &scratch).unwrap();
                let module = load_bytes(&bytes).unwrap();
                assert_eq!(evaluate_image_writes(&module, 0x3f80_3000), [expected],
                    "per-write {suffix:?} must survive module serialization and native-policy specialization");
            }
        }
        let call = source
            .lines()
            .find(|line| line.contains("call void @air.write_texture_"))
            .unwrap();
        let air_declaration = source
            .lines()
            .find(|line| line.starts_with("declare void @air.write_texture_"))
            .unwrap();
        let expand = |line: &str| {
            ["", "rte.", "rtz."]
                .map(|suffix| {
                    line.replace(
                        "air.write_texture_2d.v4f32",
                        &format!("air.write_texture_2d.{suffix}v4f32"),
                    )
                })
                .join("\n")
        };
        let mixed = source
            .replace(call, &expand(call))
            .replace(air_declaration, &expand(air_declaration));
        let mixed = crate::translate_sanitized_native_with_options(
            &mixed,
            crate::passes::Stage::Kernel,
            &scratch,
            crate::passes::TransformOptions::default()
                .with_runtime_storage_image(
                    0,
                    crate::reflect::RuntimeStorageImageState {
                        format: crate::reflect::RuntimeStorageImageFormat::Rgba16Float,
                        capabilities: crate::reflect::RuntimeStorageImageCapabilities {
                            storage_image: true,
                            ..Default::default()
                        },
                    },
                )
                .unwrap(),
        )
        .unwrap();
        let mixed = specialize_texture_write_rounding(
            &load_bytes(&mixed).unwrap().assemble(),
            TextureWriteRoundingMode::TowardZero,
            &[],
        )
        .unwrap();
        let mixed_bytes = mixed
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        crate::tools::spirv_val_bytes(&mixed_bytes, &scratch).unwrap();
        assert_eq!(
            evaluate_image_writes(&load_bytes(&mixed_bytes).unwrap(), 0x3f80_3000),
            [0x3f80_2000, 0x3f80_4000, 0x3f80_2000],
            "mode is per write, not per binding or module"
        );
        assert_eq!(
            spec_values(&mixed)[&(TEXTURE_WRITE_FORMAT_SPEC_ID_BASE + 2)],
            16
        );
        for mode in [
            TextureWriteRoundingMode::TowardZero,
            TextureWriteRoundingMode::ToNearestEven,
        ] {
            let words = specialize_texture_write_rounding(&original, mode, &[]).unwrap();
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            let asm = crate::disassemble(&bytes).unwrap();
            assert!(asm.contains("OpShiftRightLogical"), "{asm}");
            assert!(!asm.contains("OpCapability Float16"), "{asm}");
            crate::tools::spirv_val_bytes(&bytes, &scratch).expect("validate rounded image write");
        }
        assert_eq!(
            specialize_texture_write_rounding(&original, TextureWriteRoundingMode::Default, &[])
                .unwrap(),
            original
        );
        for format in [
            ImageFormat::Rgba32f,
            ImageFormat::Rgba8,
            ImageFormat::Rgba8Snorm,
            ImageFormat::Rgba16,
            ImageFormat::Rgb10A2,
        ] {
            let mut module = load_bytes(&bytes).unwrap();
            for inst in &mut module.types_global_values {
                if inst.class.opcode == Op::TypeImage {
                    inst.operands[6] = Operand::ImageFormat(format);
                }
            }
            let words = module.assemble();
            for mode in [
                TextureWriteRoundingMode::TowardZero,
                TextureWriteRoundingMode::ToNearestEven,
            ] {
                let specialized = specialize_texture_write_rounding(&words, mode, &[]).unwrap();
                assert_eq!(
                    spec_values(&specialized)[&TEXTURE_WRITE_FORMAT_SPEC_ID_BASE],
                    0,
                    "{format:?} must disable floating-point narrowing"
                );
            }
        }
        let mut module = load_bytes(&bytes).unwrap();
        let image_type = module
            .types_global_values
            .iter_mut()
            .find(|inst| inst.class.opcode == Op::TypeImage)
            .unwrap();
        image_type.operands[6] = Operand::ImageFormat(ImageFormat::Unknown);
        let image_id = image_type.result_id.unwrap();
        let pointer_id = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == Op::TypePointer
                    && inst.operands.get(1) == Some(&Operand::IdRef(image_id))
            })
            .unwrap()
            .result_id
            .unwrap();
        let variable = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::Variable && inst.result_type == Some(pointer_id))
            .unwrap()
            .result_id
            .unwrap();
        let binding = module.annotations.iter().find_map(|inst| {
            match inst.operands.as_slice() {
                [Operand::IdRef(id), Operand::Decoration(Decoration::Binding), Operand::LiteralBit32(binding)]
                    if *id == variable => Some(*binding),
                _ => None,
            }
        }).unwrap();
        let words = module.assemble();
        assert!(specialize_texture_write_rounding(
            &words,
            TextureWriteRoundingMode::ToNearestEven,
            &[]
        )
        .unwrap_err()
        .contains("requires runtime format"));
        let target = TextureWriteTarget {
            descriptor_set: 0,
            binding,
            format: TextureWriteFormat::Normalized,
        };
        let normalized = specialize_texture_write_rounding(
            &words,
            TextureWriteRoundingMode::ToNearestEven,
            &[target],
        )
        .unwrap();
        assert_eq!(
            spec_values(&normalized)[&TEXTURE_WRITE_FORMAT_SPEC_ID_BASE],
            0
        );
        let half_target = TextureWriteTarget {
            format: TextureWriteFormat::Float16,
            ..target
        };
        for targets in [[target, half_target], [half_target, target]] {
            assert!(specialize_texture_write_rounding(
                &words,
                TextureWriteRoundingMode::TowardZero,
                &targets,
            )
            .unwrap_err()
            .contains("conflicting runtime formats"));
        }
        let mut array_module = module.clone();
        let variable_position = array_module
            .types_global_values
            .iter()
            .position(|inst| inst.result_id == Some(variable))
            .unwrap();
        let mut array_variable = array_module.types_global_values.remove(variable_position);
        let uint = declaration(
            &mut array_module,
            Op::TypeInt,
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        );
        let one = scalar_constant(&mut array_module, uint, 1);
        let zero = scalar_constant(&mut array_module, uint, 0);
        let array = declaration(
            &mut array_module,
            Op::TypeArray,
            vec![Operand::IdRef(image_id), Operand::IdRef(one)],
        );
        let pointer = declaration(
            &mut array_module,
            Op::TypePointer,
            vec![
                Operand::StorageClass(spirv::StorageClass::UniformConstant),
                Operand::IdRef(array),
            ],
        );
        array_variable.result_type = Some(pointer);
        array_module.types_global_values.push(array_variable);
        let load_count = array_module
            .all_inst_iter()
            .filter(|inst| {
                inst.class.opcode == Op::Load
                    && inst.operands.first() == Some(&Operand::IdRef(variable))
            })
            .count();
        let mut ids = (0..load_count)
            .map(|_| array_module.fresh_id())
            .collect::<Vec<_>>()
            .into_iter();
        for function in &mut array_module.functions {
            for block in &mut function.blocks {
                for mut inst in std::mem::take(&mut block.instructions) {
                    if inst.class.opcode == Op::Load
                        && inst.operands.first() == Some(&Operand::IdRef(variable))
                    {
                        let id = ids.next().unwrap();
                        block.instructions.push(Instruction::new(
                            Op::AccessChain,
                            Some(pointer_id),
                            Some(id),
                            vec![Operand::IdRef(variable), Operand::IdRef(zero)],
                        ));
                        inst.operands[0] = Operand::IdRef(id);
                    }
                    block.instructions.push(inst);
                }
            }
        }
        array_module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(
                spirv::Capability::StorageImageWriteWithoutFormat,
            )],
        ));
        for target in [target, half_target] {
            let specialized = specialize_texture_write_rounding(
                &array_module.assemble(),
                TextureWriteRoundingMode::TowardZero,
                &[target],
            )
            .unwrap();
            let precision = if target.format == TextureWriteFormat::Float16 {
                16
            } else {
                0
            };
            assert_eq!(
                spec_values(&specialized)[&TEXTURE_WRITE_FORMAT_SPEC_ID_BASE],
                precision
            );
            let bytes: Vec<_> = specialized
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            crate::tools::spirv_val_bytes(&bytes, &scratch).unwrap();
        }
        for inst in &mut module.types_global_values {
            if inst.class.opcode == Op::TypeImage {
                inst.operands[6] = Operand::ImageFormat(ImageFormat::R11fG11fB10f);
            }
        }
        assert!(specialize_texture_write_rounding(
            &module.assemble(),
            TextureWriteRoundingMode::ToNearestEven,
            &[]
        )
        .unwrap_err()
        .contains("unsupported destination format"));
        std::fs::remove_dir_all(scratch).unwrap();
    }
}
