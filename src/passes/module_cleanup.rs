//! Dead-global/debug cleanup and capability/extension closure.

use super::*;

/// Declare `ext` unless the module already does.
///
/// Every capability that is not core Vulkan needs its extension named alongside it, and each of
/// them wants the same idempotent push — one place to get it right rather than one copy per
/// extension.
fn require_extension(ctx: &mut Ctx, ext: &str) {
    let have = ctx
        .module
        .extensions
        .iter()
        .any(|e| matches!(e.operands.first(), Some(Operand::LiteralString(s)) if s == ext));
    if !have {
        ctx.module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString(ext.into())],
        ));
    }
}

fn needed_variable_pointer_capabilities(ctx: &Ctx) -> (bool, bool) {
    crate::spirv_variable_ptr::variable_pointer_requirement(&ctx.module)
}

/// Collect every id DEFINED in the module (globals + function defs/params/labels/instruction
/// results). Used to spot dangling debug-name / decoration references.
fn defined_ids(module: &Module) -> HashSet<Word> {
    let mut s = HashSet::new();
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            s.insert(id);
        }
    }
    for inst in &module.ext_inst_imports {
        if let Some(id) = inst.result_id {
            s.insert(id);
        }
    }
    for f in &module.functions {
        if let Some(id) = f.def.as_ref().and_then(|d| d.result_id) {
            s.insert(id);
        }
        for p in &f.parameters {
            if let Some(id) = p.result_id {
                s.insert(id);
            }
        }
        for b in &f.blocks {
            if let Some(id) = b.label.as_ref().and_then(|l| l.result_id) {
                s.insert(id);
            }
            for inst in &b.instructions {
                if let Some(id) = inst.result_id {
                    s.insert(id);
                }
            }
        }
    }
    s
}

/// Collect ids referenced from function bodies. Entry-point interface variables should be live
/// because code loads/stores/access-chains them, not because an earlier pass registered them before a
/// later raw-alias rewrite replaced every use.
pub(super) fn function_referenced_ids(module: &Module) -> HashSet<Word> {
    let mut s = HashSet::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                for op in &inst.operands {
                    if let Operand::IdRef(r) | Operand::IdScope(r) | Operand::IdMemorySemantics(r) =
                        op
                    {
                        s.insert(*r);
                    }
                }
            }
        }
    }
    s
}

/// Remove global variables that are no longer referenced by code and no longer listed on the filtered
/// entry interface. Decorations/debug names for the removed ids are swept by drop_dangling_debug.
pub(super) fn drop_dead_unreferenced_variables(
    ctx: &mut Ctx,
    function_refs: &HashSet<Word>,
    interface_ids: &HashSet<Word>,
) {
    ctx.module.types_global_values.retain(|inst| {
        if inst.class.opcode != Op::Variable {
            return true;
        }
        let Some(id) = inst.result_id else {
            return true;
        };
        function_refs.contains(&id) || interface_ids.contains(&id)
    });
}

/// Whether a non-semantic record (`OpName`, `OpDecorate`, `OpMemberName`, `OpMemberDecorate`) is
/// about one of `removed`. Every one of them names its target first.
fn targets_any(instruction: &Instruction, removed: &HashSet<Word>) -> bool {
    matches!(instruction.operands.first(), Some(Operand::IdRef(id)) if removed.contains(id))
}

/// Drop every global `OpVariable` no instruction references, together with its entry-point interface
/// entry, its decorations, and its debug names. Returns whether anything was removed.
///
/// `finalize` applies this rule once, against the bodies as they stand when it builds the entry
/// point. Instruction-deleting work runs after it at two boundaries — the pointer and access-chain
/// closures at the end of `transform`, and dead-value elimination plus constant-CFG pruning in
/// `finish_module` — and when one of those deletes a variable's last use, nothing downstream can
/// remove the variable: `gc_dead_globals` roots liveness AT the entry-point interface, so the
/// interface entry keeps the variable alive and the variable justifies the interface entry. The
/// module then ships an `OpVariable` with a `DescriptorSet`/`Binding` no instruction touches, and a
/// consumer building a descriptor-set layout from the interface has to satisfy a binding the shader
/// never reads.
///
/// Re-applying the rule after the last instruction-deleting step breaks that cycle. It can only
/// shrink the interface — an id kept here is one an instruction still names — so SPIR-V 1.4's
/// requirement that the interface list every global the entry point references is preserved.
pub(crate) fn drop_unreferenced_global_variables(module: &mut Module) -> bool {
    /// `OpEntryPoint` operands: execution model, entry function id, name, then the interface ids.
    const INTERFACE_START: usize = 3;

    let referenced = function_referenced_ids(module);
    let mut changed = false;
    for entry_point in &mut module.entry_points {
        let before = entry_point.operands.len();
        entry_point.operands = std::mem::take(&mut entry_point.operands)
            .into_iter()
            .enumerate()
            .filter(|(index, operand)| match operand {
                Operand::IdRef(id) if *index >= INTERFACE_START => referenced.contains(id),
                _ => true,
            })
            .map(|(_, operand)| operand)
            .collect();
        changed |= entry_point.operands.len() != before;
    }
    let removed = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
        .filter_map(|instruction| instruction.result_id)
        .filter(|id| !referenced.contains(id))
        .collect::<HashSet<_>>();
    if removed.is_empty() {
        return changed;
    }
    module.types_global_values.retain(|instruction| {
        instruction
            .result_id
            .is_none_or(|id| !removed.contains(&id))
    });
    module
        .annotations
        .retain(|instruction| !targets_any(instruction, &removed));
    module
        .debug_names
        .retain(|instruction| !targets_any(instruction, &removed));
    true
}

/// Remove OpName/OpDecorate/OpMemberDecorate whose target id is not defined anywhere.
pub(crate) fn drop_dangling_debug(module: &mut Module) {
    let defined = defined_ids(module);
    let keep = |inst: &Instruction| -> bool {
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => defined.contains(id),
            _ => true,
        }
    };
    module.debug_names.retain(&keep);
    module.annotations.retain(&keep);
}

/// Decorations that only DESCRIBE the explicit memory layout of the declaration they name. They are
/// metadata owned by their target, not an independent reason for it to exist, so `gc_dead_globals`
/// does not root liveness at them.
///
/// This is an allowlist on purpose. Most decorations do give their target a module-level role that
/// nothing else records -- `BuiltIn WorkgroupSize` on an otherwise unreferenced constant is the
/// local-size declaration, `SpecId` is the specialization contract -- and treating every decoration
/// as description broke 9,857 of the 14,579 corpus sources on exactly that constant.
const LAYOUT_DECORATIONS: &[spirv::Decoration] = &[
    spirv::Decoration::ArrayStride,
    spirv::Decoration::Block,
    spirv::Decoration::BufferBlock,
    spirv::Decoration::ColMajor,
    spirv::Decoration::MatrixStride,
    spirv::Decoration::Offset,
    spirv::Decoration::RowMajor,
];

/// Iteratively drop types/global-values whose result id is referenced nowhere else in the module
/// (instruction operands, result-types, decorations, entry-point interface, function bodies). Keeps
/// variables referenced by the entry interface. Repeats to a fixpoint so chains die cleanly.
pub(super) fn gc_dead_globals(ctx: &mut Ctx) {
    let collect = |inst: &Instruction, live: &mut HashSet<Word>| {
        if let Some(ty) = inst.result_type {
            live.insert(ty);
        }
        for operand in &inst.operands {
            if let Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) =
                operand
            {
                live.insert(*id);
            }
        }
    };

    // Executable/module linkage, semantic decorations, and typed sidecars own liveness. Debug names
    // follow those definitions; treating names as roots retains dead constants (including pointer
    // nulls that Vulkan's Logical addressing model cannot represent) forever.
    let mut live = HashSet::new();
    live.extend(
        ctx.emit_sidecar
            .local_pointer_field_stores
            .iter()
            .map(|fact| fact.id),
    );
    live.extend(ctx.emit_sidecar.buffer_root_source_types.values().copied());
    for section in [&ctx.module.entry_points, &ctx.module.execution_modes] {
        for instruction in section {
            collect(instruction, &mut live);
        }
    }
    for instruction in &ctx.module.annotations {
        // A layout decoration describes its target; it does not use it. Rooting liveness at the
        // target lets a `Block`/`Offset`-decorated struct keep itself, and every member type it
        // names, alive on the strength of its own description, and lets an `ArrayStride`-decorated
        // physical pointer retain a capability-requiring declaration in a Logical module. The
        // decoration operand sits at a different position on `OpDecorate` than on
        // `OpMemberDecorate`, so look for it anywhere in the instruction.
        if instruction.operands.iter().any(|operand| {
            matches!(operand, Operand::Decoration(decoration) if LAYOUT_DECORATIONS.contains(decoration))
        }) {
            continue;
        }
        collect(instruction, &mut live);
    }
    for function in &ctx.module.functions {
        if let Some(definition) = &function.def {
            collect(definition, &mut live);
        }
        for parameter in &function.parameters {
            collect(parameter, &mut live);
        }
        for block in &function.blocks {
            if let Some(label) = &block.label {
                collect(label, &mut live);
            }
            for instruction in &block.instructions {
                collect(instruction, &mut live);
            }
        }
    }

    let definitions = ctx
        .module
        .types_global_values
        .iter()
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
        .collect::<HashMap<_, _>>();
    let mut pending = live.iter().copied().collect::<Vec<_>>();
    while let Some(id) = pending.pop() {
        let Some(definition) = definitions.get(&id) else {
            continue;
        };
        let mut dependencies = HashSet::new();
        collect(definition, &mut dependencies);
        for dependency in dependencies {
            if live.insert(dependency) {
                pending.push(dependency);
            }
        }
    }

    ctx.module
        .types_global_values
        .retain(|instruction| instruction.result_id.is_none_or(|id| live.contains(&id)));
    drop_dangling_debug(&mut ctx.module);
}

/// Remove every `OpCapability Int8`/`Int16`/`Int64`/`Float16`/`Float64` whose scalar type does not
/// survive (i.e. is used nowhere after access-chain narrowing + dead-global gc).
///
/// Leaving a capability whose feature is unused is legal for spirv-val, but it is not free. It is
/// what cues NVIDIA's compiler down the 64-bit path that crashes, and every one of these is a
/// Vulkan device feature -- `shaderInt8`, `shaderInt16`, `shaderFloat16`, `shaderFloat64` -- so a
/// module that declares one it never uses refuses to load on hardware that would otherwise have run
/// it. **Measured before this generalized past `Int64`: 656 of the 14,579 corpus sources declared a
/// width capability with no type of that width -- 591 `Int8`, 32 `Float16`, 29 `Int16`.**
///
/// The width->capability table is `native::scalar_width_capability`, the same one the owned-module
/// check reads in the opposite direction to REFUSE a module that declares such a type without the
/// capability. If genuine narrow math remains the type is still present, so the capability stays
/// (and `add_needed_capabilities` would re-add it).
pub(crate) fn drop_unused_scalar_width_capabilities(module: &mut Module) {
    let used: HashSet<spirv::Capability> = module
        .types_global_values
        .iter()
        .filter(|i| matches!(i.class.opcode, Op::TypeInt | Op::TypeFloat))
        .filter_map(|i| match i.operands.first() {
            Some(&Operand::LiteralBit32(width)) => {
                crate::native::scalar_width_capability(i.class.opcode, width)
            }
            _ => None,
        })
        .collect();
    module.capabilities.retain(|c| match c.operands.first() {
        Some(&Operand::Capability(capability)) => {
            !matches!(
                capability,
                spirv::Capability::Int8
                    | spirv::Capability::Int16
                    | spirv::Capability::Int64
                    | spirv::Capability::Float16
                    | spirv::Capability::Float64
            ) || used.contains(&capability)
        }
        _ => true,
    });
}

/// Capabilities whose ENTIRE requirement is expressed in the SPIR-V grammar: each one is enabling
/// for a fixed set of opcodes or enumerants and nothing else, so "no instruction and no operand in
/// the finished module names it" is a proof that it is unused.
///
/// This is deliberately an ALLOWLIST rather than the complement of a blocklist. Plenty of
/// capabilities are demanded by validation RULES the grammar cannot see -- `VariablePointers` by a
/// pointer reaching an `OpPhi`, `StorageBuffer8BitAccess` by a narrow type landing in a block,
/// `StorageImageWriteWithoutFormat` by an `Unknown` format being written -- and for those the
/// grammar's silence means nothing. Getting this list short is safe (a capability we could have
/// dropped survives); getting it wrong in the other direction would emit a module that does not
/// validate. `VariablePointers`/`VariablePointersStorageBuffer` are excluded on purpose:
/// `drop_unused_variable_pointer_capabilities` computes them from the rule that actually governs
/// them. So are `Image1D`/`ImageBuffer`/`Sampled1D`/`SampledBuffer`: the grammar attaches those to
/// the `Dim` enumerant of `OpTypeImage`, but the validator demands them to ACCESS such an image
/// ("Capability ImageBuffer is required to access storage image"), and a module can reach an image
/// whose type it did not spell. Dropping them broke 9 of the 14,579 corpus sources, measured;
/// `add_needed_capabilities` computes those from the finished module already.
const GRAMMAR_COMPLETE_CAPABILITIES: &[spirv::Capability] = &[
    spirv::Capability::DemoteToHelperInvocation,
    spirv::Capability::FloatControls2,
    spirv::Capability::GroupNonUniformArithmetic,
    spirv::Capability::GroupNonUniformBallot,
    spirv::Capability::GroupNonUniformClustered,
    spirv::Capability::GroupNonUniformQuad,
    spirv::Capability::GroupNonUniformShuffle,
    spirv::Capability::GroupNonUniformShuffleRelative,
    spirv::Capability::GroupNonUniformVote,
    spirv::Capability::ImageQuery,
    spirv::Capability::StorageImageExtendedFormats,
];

/// Remove an `OpTypeInt`/`OpTypeFloat` no instruction references, so
/// [`drop_unused_scalar_width_capabilities`] sees the widths the module actually uses.
///
/// A dead scalar type is not free: the width drop keys on the type EXISTING, so one that survives
/// keeps its capability, and with it the Vulkan device feature. **Measured: 30 of the 14,579 corpus
/// sources declared `Int64` (22) or `Int16` (8) purely through a type no instruction touched** --
/// and a declared-but-unused `Int64` is the exact thing that dropper's comment says cues NVIDIA's
/// compiler down the 64-bit path that crashes.
///
/// `gc_dead_globals` does not reach these because it runs before CFG construction deletes the last
/// user. Its own carve-out covers layout decorations (`LAYOUT_DECORATIONS`), which is what stops a
/// `Block`/`Offset`-decorated struct keeping its member types alive on the strength of its own
/// description; this pass is the late sweep for what dies afterwards.
pub(crate) fn drop_unreferenced_scalar_types(module: &mut Module) {
    let scalars: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|instruction| matches!(instruction.class.opcode, Op::TypeInt | Op::TypeFloat))
        .filter_map(|instruction| instruction.result_id)
        .collect();
    if scalars.is_empty() {
        return;
    }
    let mut referenced: HashSet<Word> = HashSet::new();
    for instruction in module.all_inst_iter() {
        // A name or a decoration describes a type; it does not use one.
        if matches!(
            instruction.class.opcode,
            Op::Name | Op::MemberName | Op::Decorate | Op::MemberDecorate | Op::DecorateId
        ) {
            continue;
        }
        referenced.extend(instruction.result_type.filter(|ty| scalars.contains(ty)));
        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                if scalars.contains(id) && instruction.result_id != Some(*id) {
                    referenced.insert(*id);
                }
            }
        }
    }
    module.types_global_values.retain(|instruction| {
        instruction
            .result_id
            .is_none_or(|id| !scalars.contains(&id) || referenced.contains(&id))
    });
    drop_dangling_debug(module);
}

/// Drop an allowlisted capability that nothing in the finished module requires.
///
/// The same shape as `drop_unused_scalar_width_capabilities` and for the same reason: a capability
/// is requested where the need is PREDICTED -- by the emitter, or by an earlier phase -- and a later
/// phase can delete the thing that needed it. `add_needed_capabilities` already computes several of
/// these exact predicates from the finished module and uses each in the ADD direction only, so a
/// folded-away `OpDemoteToHelperInvocation` left its capability behind.
///
/// The requirement is read from the grammar rather than restated, so there is no second table to
/// drift: an instruction's own capabilities, every operand enumerant's, and the scalar-width one.
/// A kept capability's transitive prerequisites are kept with it.
///
/// **Measured before this existed: 118 of the 14,579 corpus sources declared a capability nothing in
/// the module asked for** -- 56 `DemoteToHelperInvocation`, 40 `ImageQuery`, 27 `FloatControls2`,
/// 13 `GroupNonUniform*`, 9 `StorageImageExtendedFormats`, 9 `Image1D`, 1 `ImageBuffer`, plus the
/// variable-pointer rows below.
///
/// `VariablePointers`/`VariablePointersStorageBuffer` cannot join the allowlist because the grammar
/// does not encode their need: what asks for them is a POINTER-TYPED `OpPhi`/`OpSelect`/
/// `OpPtrAccessChain`, and the grammar attaches nothing to those opcodes. They get their own
/// predicate here instead, recomputed from the finished module, because the pipeline computes the
/// requirement once and reapplies the SNAPSHOT afterwards -- so a pass that deleted the last pointer
/// merge left the capability behind. **Measured: 1 of the 14,579 corpus sources.** A stale
/// declaration is legal SPIR-V, but it demands the strictly stronger `variablePointers` feature of
/// every consumer and cues native drivers down a variable-pointer compiler path.
pub(crate) fn drop_unrequired_capabilities(module: &mut Module) {
    let mut required: HashSet<spirv::Capability> = HashSet::new();
    let mut required_extensions: HashSet<&'static str> = HashSet::new();
    for instruction in module.all_inst_iter() {
        // A DECLARATION is not a use. `OpCapability X` carries `Operand::Capability(X)`, whose
        // grammar entry names the extension that enables X -- so counting it here would let every
        // declaration justify its own extension, and nothing would ever be dropped.
        if matches!(instruction.class.opcode, Op::Capability | Op::Extension) {
            continue;
        }
        if let Some((capabilities, extensions)) =
            crate::spirv_binary::instruction_declaration_requirements(instruction.class.opcode)
        {
            required.extend(capabilities.iter().copied());
            required_extensions.extend(extensions.iter().copied());
        }
        for operand in &instruction.operands {
            for requirement in crate::spirv_binary::operand_declaration_requirements(operand) {
                required.extend(requirement.capabilities.iter().copied());
                required_extensions.extend(requirement.extensions.iter().copied());
            }
        }
        if let (Op::TypeInt | Op::TypeFloat, Some(&Operand::LiteralBit32(width))) =
            (instruction.class.opcode, instruction.operands.first())
        {
            required.extend(crate::native::scalar_width_capability(
                instruction.class.opcode,
                width,
            ));
        }
    }
    // A capability that survives keeps whatever it is built on, however deep.
    let mut frontier: Vec<spirv::Capability> = module
        .capabilities
        .iter()
        .filter_map(|instruction| match instruction.operands.first() {
            Some(&Operand::Capability(capability)) => Some(capability),
            _ => None,
        })
        .filter(|capability| {
            !GRAMMAR_COMPLETE_CAPABILITIES.contains(capability) || required.contains(capability)
        })
        .collect();
    while let Some(capability) = frontier.pop() {
        for requirement in
            crate::spirv_binary::operand_declaration_requirements(&Operand::Capability(capability))
        {
            // A lone prerequisite is implied. Several are enabling alternatives, and the module
            // already chose one by declaring it, so they are not prerequisites of this one.
            if let [prerequisite] = requirement.capabilities {
                if required.insert(*prerequisite) {
                    frontier.push(*prerequisite);
                }
            }
            required_extensions.extend(requirement.extensions.iter().copied());
        }
    }
    let (needs_storage_buffer_pointers, needs_other_pointers) =
        crate::spirv_variable_ptr::variable_pointer_requirement(module);
    module
        .capabilities
        .retain(|instruction| match instruction.operands.first() {
            Some(&Operand::Capability(spirv::Capability::VariablePointersStorageBuffer)) => {
                needs_storage_buffer_pointers || needs_other_pointers
            }
            Some(&Operand::Capability(spirv::Capability::VariablePointers)) => needs_other_pointers,
            Some(&Operand::Capability(capability)) => {
                !GRAMMAR_COMPLETE_CAPABILITIES.contains(&capability)
                    || required.contains(&capability)
            }
            _ => true,
        });
    // Dropping a capability strands whatever `OpExtension` existed only to enable it, and an
    // unused extension is the same demand on the driver the capability was. Only extensions an
    // allowlisted capability can ask for are candidates, so an extension a validation rule
    // demands -- which the grammar cannot see -- is never a candidate.
    let droppable_extensions: HashSet<&'static str> = GRAMMAR_COMPLETE_CAPABILITIES
        .iter()
        .flat_map(|capability| {
            crate::spirv_binary::operand_declaration_requirements(&Operand::Capability(*capability))
        })
        .flat_map(|requirement| requirement.extensions.iter().copied())
        .collect();
    module
        .extensions
        .retain(|instruction| match instruction.operands.first() {
            Some(Operand::LiteralString(extension)) => {
                !droppable_extensions.contains(extension.as_str())
                    || required_extensions.contains(extension.as_str())
            }
            _ => true,
        });
}

pub(super) fn drop_unused_variable_pointer_capabilities(ctx: &mut Ctx) -> (bool, bool) {
    crate::spirv_variable_ptr::lower_storage_buffer_pointer_phis(&mut ctx.module);
    crate::spirv_variable_ptr::lower_zero_base_storage_buffer_ptr_access_chains(&mut ctx.module);
    crate::spirv_variable_ptr::rewrite_storage_buffer_atomic_scopes(&mut ctx.module);
    let (needs_storage_buffer, needs_other) = needed_variable_pointer_capabilities(ctx);
    ctx.module
        .capabilities
        .retain(|c| match c.operands.first() {
            Some(Operand::Capability(spirv::Capability::VariablePointersStorageBuffer)) => {
                needs_storage_buffer
            }
            Some(Operand::Capability(spirv::Capability::VariablePointers)) => needs_other,
            _ => true,
        });
    (needs_storage_buffer, needs_other)
}

pub(super) fn add_needed_capabilities(ctx: &mut Ctx, variable_pointer_requirements: (bool, bool)) {
    use spirv::Capability;
    let mut want: Vec<Capability> = vec![];
    if ctx.module.entry_points.iter().any(|instruction| {
        instruction.operands.first()
            == Some(&Operand::ExecutionModel(
                spirv::ExecutionModel::TessellationEvaluation,
            ))
    }) {
        want.push(Capability::Tessellation);
    }
    let has_demote = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::DemoteToHelperInvocation);
    if has_demote {
        want.push(Capability::DemoteToHelperInvocation);
    }
    // `Sampled1D`/`SampledBuffer` cover a SAMPLED image of that dimensionality; `Image1D`/
    // `ImageBuffer` cover a storage one. They are separate capabilities, so a module holding only a
    // storage image of that shape needs only the storage one. Asking for the sampled capability
    // beside it asks the consumer for something the module cannot do.
    //
    // The storage arm always read operand 5 -- `Sampled`, where 2 means "used without a sampler" --
    // and the sampled arm never did, so a `texture1d<..., access::write>` claimed both.
    let image_dim = |dim: Dim, storage: bool| {
        ctx.module.types_global_values.iter().any(|i| {
            i.class.opcode == Op::TypeImage
                && i.operands.get(1) == Some(&Operand::Dim(dim))
                && (i.operands.get(5) == Some(&Operand::LiteralBit32(2))) == storage
        })
    };
    if image_dim(Dim::Dim1D, false) {
        want.push(Capability::Sampled1D);
    }
    if image_dim(Dim::Dim1D, true) {
        want.push(Capability::Image1D);
    }
    if image_dim(Dim::DimBuffer, false) {
        want.push(Capability::SampledBuffer);
    }
    if image_dim(Dim::DimBuffer, true) {
        want.push(Capability::ImageBuffer);
    }
    let has_input_attachment_type = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::DimSubpassData))
    });
    let has_input_attachment_decor = ctx.module.annotations.iter().any(|i| {
        i.class.opcode == Op::Decorate
            && i.operands.get(1) == Some(&Operand::Decoration(Decoration::InputAttachmentIndex))
    });
    if has_input_attachment_type || has_input_attachment_decor {
        want.push(Capability::InputAttachment);
    }
    let has_viewport_index = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ViewportIndex))
    });
    if has_viewport_index {
        want.push(Capability::ShaderViewportIndex);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 5) {
                header.set_version(1, 5);
            }
        }
    }
    let has_layer = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::Layer))
    });
    if has_layer {
        want.push(Capability::ShaderLayer);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 5) {
                header.set_version(1, 5);
            }
        }
    }
    // `ViewIndex` is `[[amplification_id]]`: the view a multiview pass is rasterizing. The
    // capability is core since SPIR-V 1.3, which every module this translator emits already is.
    let has_view_index = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ViewIndex))
    });
    if has_view_index {
        want.push(Capability::MultiView);
    }
    let has_clip_distance = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ClipDistance))
    });
    if has_clip_distance {
        want.push(Capability::ClipDistance);
    }
    let has_primitive_id = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::PrimitiveId))
    });
    // `PrimitiveId` is enabled by ANY of `Geometry`, `Tessellation`, `RayTracingKHR` or
    // `MeshShadingEXT`. That is a disjunction, and choosing `Geometry` from it was the worst
    // available answer twice over.
    //
    // A tessellation-evaluation entry already declares `Tessellation` above, so for 226 of the 236
    // corpus modules that carried `Geometry` the requirement was ALREADY satisfied and the
    // declaration bought nothing. What it cost is that Vulkan reads a declared capability as a
    // demand: the consumer must enable `geometryShader`, and no Metal-backed implementation has one
    // -- Metal has no geometry stage at all. Every one of those modules was unloadable on the very
    // platform this translator targets, for a capability nothing in it uses.
    //
    // The remaining 10 are fragment entries reading `[[primitive_id]]`, where one disjunct does
    // have to be declared. Neither `Geometry` nor `Tessellation` describes a fragment shader, so
    // the choice is not between a right answer and a wrong one; it is between a disjunct every
    // Metal-backed implementation supports and one none of them does.
    if has_primitive_id
        && !want.iter().any(|capability| {
            matches!(
                capability,
                Capability::Geometry
                    | Capability::Tessellation
                    | Capability::RayTracingKHR
                    | Capability::MeshShadingEXT
            )
        })
    {
        want.push(Capability::Tessellation);
    }
    let has_sample_id = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::SampleId))
    });
    // `Sample` on an Input variable is per-sample interpolation (Metal `[[sample_perspective]]`),
    // which requests sample-rate shading just as reading `SampleId` does.
    let has_sample_interpolation = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Sample))
    });
    if has_sample_id || has_sample_interpolation {
        want.push(Capability::SampleRateShading);
    }
    // `BaryCoordKHR` / `BaryCoordNoPerspKHR` are `[[barycentric_coord]]`; both come from the
    // fragment-barycentric extension rather than core Vulkan.
    let has_barycentric = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && matches!(
                instruction.operands.get(2),
                Some(&Operand::BuiltIn(BuiltIn::BaryCoordKHR))
                    | Some(&Operand::BuiltIn(BuiltIn::BaryCoordNoPerspKHR))
            )
    });
    if has_barycentric {
        want.push(Capability::FragmentBarycentricKHR);
    }
    let has_stencil_export = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::FragStencilRefEXT))
    });
    if has_stencil_export {
        want.push(Capability::StencilExportEXT);
    }
    // Texture query opcodes need the ImageQuery capability.
    let has_query = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::ImageQuerySize
                    | Op::ImageQuerySizeLod
                    | Op::ImageQueryLod
                    | Op::ImageQueryLevels
                    | Op::ImageQuerySamples
            )
        });
    if has_query {
        want.push(Capability::ImageQuery);
    }
    let has_group_shuffle = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformShuffle | Op::GroupNonUniformShuffleXor
            )
        });
    if has_group_shuffle {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformShuffle);
    }
    let has_group_shuffle_relative = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformShuffleUp | Op::GroupNonUniformShuffleDown
            )
        });
    if has_group_shuffle_relative {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformShuffleRelative);
    }
    let has_group_elect = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::GroupNonUniformElect);
    if has_group_elect {
        want.push(Capability::GroupNonUniform);
    }
    let has_group_vote = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformAll | Op::GroupNonUniformAny | Op::GroupNonUniformAllEqual
            )
        });
    if has_group_vote {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformVote);
    }
    let has_group_ballot = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformBallot
                    | Op::GroupNonUniformBallotBitExtract
                    | Op::GroupNonUniformBallotBitCount
                    | Op::GroupNonUniformBallotFindLSB
                    | Op::GroupNonUniformBallotFindMSB
            )
        });
    if has_group_ballot {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformBallot);
    }
    let has_group_broadcast = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformBroadcast | Op::GroupNonUniformBroadcastFirst
            )
        });
    if has_group_broadcast {
        // BroadcastFirst (and Broadcast) are gated by the Ballot capability in core SPIR-V.
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformBallot);
    }
    let has_subgroup_invocation_builtin = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.iter().any(|operand| {
                matches!(
                    operand,
                    Operand::BuiltIn(spirv::BuiltIn::SubgroupLocalInvocationId)
                )
            })
    });
    if has_subgroup_invocation_builtin {
        want.push(Capability::GroupNonUniform);
    }
    let has_group_arithmetic = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformIAdd
                    | Op::GroupNonUniformFAdd
                    | Op::GroupNonUniformSMin
                    | Op::GroupNonUniformUMin
                    | Op::GroupNonUniformFMin
                    | Op::GroupNonUniformSMax
                    | Op::GroupNonUniformUMax
                    | Op::GroupNonUniformFMax
                    | Op::GroupNonUniformBitwiseAnd
                    | Op::GroupNonUniformBitwiseOr
                    | Op::GroupNonUniformBitwiseXor
            )
            // These opcodes take their capability from the GROUP OPERATION, not from the opcode:
            // `Reduce`/`InclusiveScan`/`ExclusiveScan` need `GroupNonUniformArithmetic`, and
            // `ClusteredReduce` needs `GroupNonUniformClustered` instead. Every `air.simd_*`
            // whole-simdgroup reduction emits the clustered form, so keying on the opcode alone
            // demanded arithmetic subgroup support of 244 corpus modules that use none.
            && !i.operands.iter().any(|o| {
                matches!(
                    o,
                    Operand::GroupOperation(spirv::GroupOperation::ClusteredReduce)
                )
            })
        });
    if has_group_arithmetic {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformArithmetic);
    }
    // A `ClusteredReduce` group operation (emitted by every `air.simd_*` whole-simdgroup
    // reduction) additionally needs the `GroupNonUniformClustered` capability.
    let has_clustered_reduce = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            i.operands.iter().any(|o| {
                matches!(
                    o,
                    Operand::GroupOperation(spirv::GroupOperation::ClusteredReduce)
                )
            })
        });
    if has_clustered_reduce {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformClustered);
    }
    let has_atomic_fadd = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::AtomicFAddEXT);
    if has_atomic_fadd {
        want.push(Capability::AtomicFloat32AddEXT);
    }
    // Width-based scalar capabilities: a `half` (OpTypeFloat 16) needs Float16; an 8-/16-bit int
    // needs Int8/Int16. Lowering can synthesize narrow constants and types (saturate edges, FConvert
    // results, and flat-buffer extracts), so assert the corresponding capabilities here.
    let int_width = |w: u32| {
        ctx.module.types_global_values.iter().any(|i| {
            i.class.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::LiteralBit32(w))
        })
    };
    let has_half = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeFloat && i.operands.first() == Some(&Operand::LiteralBit32(16))
    });
    if has_half {
        want.push(Capability::Float16);
    }
    if int_width(8) {
        want.push(Capability::Int8);
    }
    if int_width(16) {
        want.push(Capability::Int16);
    }
    if int_width(64) {
        want.push(Capability::Int64);
    }
    let (has_storage_buffer_pointer_merge, has_other_pointer_merge) = variable_pointer_requirements;
    if has_storage_buffer_pointer_merge {
        want.push(Capability::VariablePointersStorageBuffer);
    }
    if has_other_pointer_merge {
        want.push(Capability::VariablePointers);
    }
    for cap in &want {
        let already = ctx.module.capabilities.iter().any(|c| {
            matches!(c.operands.first(), Some(Operand::Capability(existing)) if existing == cap)
        });
        if !already {
            ctx.module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(*cap)],
            ));
        }
    }
    // DemoteToHelperInvocation needs its SPIR-V extension declared (core only from SPIR-V 1.6).
    if want.contains(&Capability::DemoteToHelperInvocation) {
        let ext = "SPV_EXT_demote_to_helper_invocation";
        require_extension(ctx, ext);
    }
    if want.contains(&Capability::AtomicFloat32AddEXT) {
        let ext = "SPV_EXT_shader_atomic_float_add";
        require_extension(ctx, ext);
    }
    if want.contains(&Capability::FragmentBarycentricKHR) {
        let ext = "SPV_KHR_fragment_shader_barycentric";
        require_extension(ctx, ext);
    }
    if want.contains(&Capability::StencilExportEXT) {
        let ext = "SPV_EXT_shader_stencil_export";
        require_extension(ctx, ext);
        // `StencilRefReplacingEXT` is to `FragStencilRefEXT` what `DepthReplacing` is to
        // `FragDepth`: the declaration that this fragment shader replaces the value, without which
        // a driver is entitled to keep the pipeline's. spirv-val does not demand it, so nothing
        // caught its absence; glslang emits it for `gl_FragStencilRefARB` alongside exactly the
        // capability and extension above, which is the answer this matches.
        let entry_points: Vec<Word> = ctx
            .module
            .entry_points
            .iter()
            .filter(|instruction| {
                instruction.operands.first()
                    == Some(&Operand::ExecutionModel(spirv::ExecutionModel::Fragment))
            })
            .filter_map(|instruction| match instruction.operands.get(1) {
                Some(&Operand::IdRef(entry)) => Some(entry),
                _ => None,
            })
            .collect();
        for entry in entry_points {
            let declared = ctx.module.execution_modes.iter().any(|instruction| {
                instruction.operands.as_slice()
                    == [
                        Operand::IdRef(entry),
                        Operand::ExecutionMode(spirv::ExecutionMode::StencilRefReplacingEXT),
                    ]
            });
            if !declared {
                ctx.module.execution_modes.push(Instruction::new(
                    Op::ExecutionMode,
                    None,
                    None,
                    vec![
                        Operand::IdRef(entry),
                        Operand::ExecutionMode(spirv::ExecutionMode::StencilRefReplacingEXT),
                    ],
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Module;
    use crate::spirv_module::ModuleHeader;

    /// A fragment module carrying `OpCapability DemoteToHelperInvocation` and its extension, with
    /// `body` for the entry's only block.
    fn module_with_demote_declared(body: Vec<Instruction>) -> Module {
        use crate::spirv_module::{Block, Function};
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(10));
        module.capabilities = vec![
            Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(spirv::Capability::Shader)],
            ),
            Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(
                    spirv::Capability::DemoteToHelperInvocation,
                )],
            ),
        ];
        module.extensions = vec![Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString(
                "SPV_EXT_demote_to_helper_invocation".to_string(),
            )],
        )];
        module.types_global_values = vec![
            Instruction::new(Op::TypeVoid, None, Some(1), vec![]),
            Instruction::new(Op::TypeFunction, None, Some(2), vec![Operand::IdRef(1)]),
            Instruction::new(Op::TypeBool, None, Some(3), vec![]),
        ];
        let mut instructions = body;
        instructions.push(Instruction::new(Op::Return, None, None, vec![]));
        module.functions = vec![Function {
            def: Some(Instruction::new(
                Op::Function,
                Some(1),
                Some(4),
                vec![
                    Operand::FunctionControl(spirv::FunctionControl::NONE),
                    Operand::IdRef(2),
                ],
            )),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![Block {
                label: Some(Instruction::new(Op::Label, None, Some(5), vec![])),
                instructions,
            }],
        }];
        module
    }

    fn declares(module: &Module, capability: spirv::Capability) -> bool {
        module.capabilities.iter().any(|instruction| {
            instruction.operands.first() == Some(&Operand::Capability(capability))
        })
    }

    fn declares_extension(module: &Module, extension: &str) -> bool {
        module.extensions.iter().any(|instruction| {
            matches!(instruction.operands.first(), Some(Operand::LiteralString(name)) if name == extension)
        })
    }

    /// A module declaring `Int64` plus an `OpTypeInt 64`, with `body` for the entry's only block.
    fn module_with_int64_declared(body: Vec<Instruction>) -> Module {
        use crate::spirv_module::{Block, Function};
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(10));
        module.capabilities = vec![
            Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(spirv::Capability::Shader)],
            ),
            Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(spirv::Capability::Int64)],
            ),
        ];
        module.types_global_values = vec![
            Instruction::new(Op::TypeVoid, None, Some(1), vec![]),
            Instruction::new(Op::TypeFunction, None, Some(2), vec![Operand::IdRef(1)]),
            Instruction::new(
                Op::TypeInt,
                None,
                Some(3),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
        ];
        // The kind of description that keeps a type alive through `gc_dead_globals` without
        // anything using it.
        module.debug_names = vec![Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(3),
                Operand::LiteralString("ulong".to_string()),
            ],
        )];
        let mut instructions = body;
        instructions.push(Instruction::new(Op::Return, None, None, vec![]));
        module.functions = vec![Function {
            def: Some(Instruction::new(
                Op::Function,
                Some(1),
                Some(4),
                vec![
                    Operand::FunctionControl(spirv::FunctionControl::NONE),
                    Operand::IdRef(2),
                ],
            )),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![Block {
                label: Some(Instruction::new(Op::Label, None, Some(5), vec![])),
                instructions,
            }],
        }];
        module
    }

    fn declares_type(module: &Module, opcode: Op, width: u32) -> bool {
        module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == opcode
                && instruction.operands.first() == Some(&Operand::LiteralBit32(width))
        })
    }

    /// The scalar type is named but never used, so it goes -- and the capability it was the only
    /// evidence for goes with it. Before the late pass existed, 30 of the 14,579 corpus sources
    /// shipped `Int64` or `Int16` on exactly this basis: CFG construction deleted the last user
    /// after `gc_dead_globals` had already run.
    #[test]
    fn a_scalar_type_nothing_uses_takes_its_capability_with_it() {
        let mut module = module_with_int64_declared(vec![]);

        drop_unreferenced_scalar_types(&mut module);
        drop_unused_scalar_width_capabilities(&mut module);

        assert!(!declares_type(&module, Op::TypeInt, 64));
        assert!(!declares(&module, spirv::Capability::Int64));
        assert!(
            module.debug_names.is_empty(),
            "the name that described it goes too"
        );
    }

    /// The control: one instruction whose RESULT TYPE is the scalar keeps both.
    #[test]
    fn a_scalar_type_an_instruction_results_in_is_kept() {
        let mut module =
            module_with_int64_declared(vec![Instruction::new(Op::Undef, Some(3), Some(6), vec![])]);

        drop_unreferenced_scalar_types(&mut module);
        drop_unused_scalar_width_capabilities(&mut module);

        assert!(declares_type(&module, Op::TypeInt, 64));
        assert!(declares(&module, spirv::Capability::Int64));
    }

    /// Nothing in the module demotes, so neither the capability nor the extension that exists only
    /// to enable it survives.
    #[test]
    fn a_capability_no_instruction_requires_is_dropped_with_its_extension() {
        let mut module = module_with_demote_declared(vec![]);

        drop_unrequired_capabilities(&mut module);

        assert!(!declares(
            &module,
            spirv::Capability::DemoteToHelperInvocation
        ));
        assert!(!declares_extension(
            &module,
            "SPV_EXT_demote_to_helper_invocation"
        ));
        assert!(
            declares(&module, spirv::Capability::Shader),
            "a capability outside the allowlist is never a candidate"
        );
    }

    /// `OpIsHelperInvocationEXT` needs the same capability `OpDemoteToHelperInvocation` does, and
    /// the grammar says so, so reading the requirement from the grammar rather than restating it
    /// keeps both. A hand-written "does the module demote?" predicate would drop them here.
    #[test]
    fn the_other_instruction_that_needs_a_capability_keeps_it() {
        let mut module = module_with_demote_declared(vec![Instruction::new(
            Op::IsHelperInvocationEXT,
            Some(3),
            Some(6),
            vec![],
        )]);

        drop_unrequired_capabilities(&mut module);

        assert!(declares(
            &module,
            spirv::Capability::DemoteToHelperInvocation
        ));
        assert!(declares_extension(
            &module,
            "SPV_EXT_demote_to_helper_invocation"
        ));
    }

    /// A module declaring `VariablePointers`, with `body` for the entry's only block.
    ///
    /// Types for a `Workgroup` pointer are present either way, so what decides the capability is
    /// the body -- which is the point: the grammar attaches nothing to `OpSelect`, so only a
    /// pointer-typed one asks for this.
    fn module_with_variable_pointers_declared(body: Vec<Instruction>) -> Module {
        use crate::spirv_module::{Block, Function};
        let (void, fn_ty, uint, pointer, a, b, cond) = (1, 2, 3, 4, 5, 6, 7);
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(20));
        module.capabilities = [
            spirv::Capability::Shader,
            spirv::Capability::VariablePointersStorageBuffer,
            spirv::Capability::VariablePointers,
        ]
        .into_iter()
        .map(|capability| {
            Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(capability)],
            )
        })
        .collect();
        module.types_global_values = vec![
            Instruction::new(Op::TypeVoid, None, Some(void), vec![]),
            Instruction::new(
                Op::TypeFunction,
                None,
                Some(fn_ty),
                vec![Operand::IdRef(void)],
            ),
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(spirv::StorageClass::Workgroup),
                    Operand::IdRef(uint),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(a),
                vec![Operand::StorageClass(spirv::StorageClass::Workgroup)],
            ),
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(b),
                vec![Operand::StorageClass(spirv::StorageClass::Workgroup)],
            ),
            Instruction::new(Op::TypeBool, None, Some(cond), vec![]),
            Instruction::new(Op::Undef, Some(cond), Some(8), vec![]),
        ];
        let mut instructions = body;
        instructions.push(Instruction::new(Op::Return, None, None, vec![]));
        module.functions = vec![Function {
            def: Some(Instruction::new(
                Op::Function,
                Some(void),
                Some(9),
                vec![
                    Operand::FunctionControl(spirv::FunctionControl::NONE),
                    Operand::IdRef(fn_ty),
                ],
            )),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![Block {
                label: Some(Instruction::new(Op::Label, None, Some(10), vec![])),
                instructions,
            }],
        }];
        module
    }

    /// The pipeline computes the variable-pointer requirement once and reapplies that snapshot after
    /// later passes run, so a pass that deletes the last pointer merge leaves the capability behind.
    /// Recomputing here is what makes the declaration a statement about the module that ships.
    /// Measured over the corpus: 1 of 14,579 sources carried a `VariablePointers` nothing asked for.
    #[test]
    fn variable_pointers_goes_when_the_last_pointer_merge_does() {
        // An OpSelect over plain integers is the same opcode with a non-pointer result type, which
        // is exactly the distinction the grammar does not draw and this predicate must.
        let mut module = module_with_variable_pointers_declared(vec![Instruction::new(
            Op::Select,
            Some(3),
            Some(11),
            vec![Operand::IdRef(8), Operand::IdRef(3), Operand::IdRef(3)],
        )]);

        drop_unrequired_capabilities(&mut module);

        assert!(!declares(&module, spirv::Capability::VariablePointers));
        assert!(!declares(
            &module,
            spirv::Capability::VariablePointersStorageBuffer
        ));
    }

    /// The converse, and the reason the predicate cannot simply be "no pointer type in the module":
    /// a `Workgroup` pointer select is what only full `VariablePointers` permits.
    #[test]
    fn a_workgroup_pointer_select_keeps_variable_pointers() {
        let mut module = module_with_variable_pointers_declared(vec![Instruction::new(
            Op::Select,
            Some(4),
            Some(11),
            vec![Operand::IdRef(8), Operand::IdRef(5), Operand::IdRef(6)],
        )]);

        drop_unrequired_capabilities(&mut module);

        assert!(declares(&module, spirv::Capability::VariablePointers));
        assert!(declares(
            &module,
            spirv::Capability::VariablePointersStorageBuffer
        ));
    }

    /// A module whose entry point lists two descriptor variables while only one is still loaded.
    /// The second is what a pipeline boundary strands when a later rewrite deletes its last use.
    fn module_with_one_stranded_descriptor() -> Module {
        use crate::spirv_module::{Block, Function};
        let (uint, pointer, live, stranded, entry, loaded) = (1, 2, 3, 4, 5, 6);
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(7));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(uint),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(live),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(stranded),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        for (variable, binding) in [(live, 0), (stranded, 1)] {
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(variable),
                    Operand::Decoration(spirv::Decoration::Binding),
                    Operand::LiteralBit32(binding),
                ],
            ));
            module.debug_names.push(Instruction::new(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(variable),
                    Operand::LiteralString(format!("buffer{binding}")),
                ],
            ));
        }
        module.entry_points.push(Instruction::new(
            Op::EntryPoint,
            None,
            None,
            vec![
                Operand::ExecutionModel(spirv::ExecutionModel::GLCompute),
                Operand::IdRef(entry),
                Operand::LiteralString("main".to_string()),
                Operand::IdRef(live),
                Operand::IdRef(stranded),
            ],
        ));
        let mut function = Function::new();
        let mut block = Block::new();
        block.instructions.push(Instruction::new(
            Op::Load,
            Some(uint),
            Some(loaded),
            vec![Operand::IdRef(live)],
        ));
        function.blocks.push(block);
        module.functions.push(function);
        module
    }

    /// The ids `module_with_one_stranded_descriptor` uses, in the order it names them.
    const LIVE_VARIABLE: Word = 3;
    const STRANDED_VARIABLE: Word = 4;

    fn variable_ids(module: &Module) -> HashSet<Word> {
        module
            .types_global_values
            .iter()
            .filter(|instruction| instruction.class.opcode == Op::Variable)
            .filter_map(|instruction| instruction.result_id)
            .collect()
    }

    fn ids_named_by(records: &[Instruction]) -> HashSet<Word> {
        records
            .iter()
            .filter_map(|instruction| match instruction.operands.first() {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_variable_no_instruction_references_leaves_with_its_interface_entry_and_records() {
        let mut module = module_with_one_stranded_descriptor();

        assert!(drop_unreferenced_global_variables(&mut module));

        assert_eq!(variable_ids(&module), HashSet::from([LIVE_VARIABLE]));
        let interface = module.entry_points[0].operands[3..]
            .iter()
            .filter_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect::<HashSet<_>>();
        assert_eq!(interface, HashSet::from([LIVE_VARIABLE]));
        assert!(!ids_named_by(&module.annotations).contains(&STRANDED_VARIABLE));
        assert!(!ids_named_by(&module.debug_names).contains(&STRANDED_VARIABLE));
        // The surviving descriptor keeps everything that describes it.
        assert!(ids_named_by(&module.annotations).contains(&LIVE_VARIABLE));
        assert!(ids_named_by(&module.debug_names).contains(&LIVE_VARIABLE));
    }

    #[test]
    fn the_sweep_reaches_a_fixed_point_in_one_pass() {
        let mut module = module_with_one_stranded_descriptor();
        assert!(drop_unreferenced_global_variables(&mut module));
        // Nothing left to remove, so a pipeline that runs it at more than one boundary pays a scan
        // and changes nothing.
        assert!(!drop_unreferenced_global_variables(&mut module));
    }

    #[test]
    fn gc_keeps_local_pointer_store_sentinel_rooted_by_typed_sidecar() {
        let ulong = 1;
        let sentinel = 2;
        let dead = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(ulong),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(sentinel), vec![]),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(dead), vec![]),
        ];
        let mut ctx = Ctx::new(module);
        ctx.emit_sidecar.local_pointer_field_stores.push(
            crate::emit_sidecar::LocalPointerFieldStore {
                id: sentinel,
                source: 99,
                root: 98,
                indices: Vec::new(),
            },
        );

        gc_dead_globals(&mut ctx);

        let ids = ctx
            .module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&ulong));
        assert!(ids.contains(&sentinel));
        assert!(!ids.contains(&dead));
    }

    #[test]
    fn gc_drops_dead_global_even_when_debug_metadata_names_it() {
        let byte = 1;
        let pointer = 2;
        let null = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(byte),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(byte),
                ],
            ),
            Instruction::new(Op::ConstantNull, Some(pointer), Some(null), vec![]),
        ];
        module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(null),
                Operand::LiteralString("dead pointer".to_string()),
            ],
        ));
        let mut ctx = Ctx::new(module);

        gc_dead_globals(&mut ctx);

        assert!(ctx.module.types_global_values.is_empty());
        assert!(ctx.module.debug_names.is_empty());
    }

    /// A `Block`-decorated struct with `Offset`-decorated members describes a buffer layout. That
    /// description is not a reason for the struct, or the member types it names, to exist: 26,233
    /// struct declarations survived in 10,588 of the 14,579 corpus sources on exactly this basis.
    #[test]
    fn gc_drops_dead_struct_owned_only_by_its_block_and_offset_layout() {
        let word = 1;
        let structure = 2;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(3));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(word),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypeStruct,
                None,
                Some(structure),
                vec![Operand::IdRef(word), Operand::IdRef(word)],
            ),
        ];
        module.annotations = vec![
            Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(structure),
                    Operand::Decoration(spirv::Decoration::Block),
                ],
            ),
            Instruction::new(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(structure),
                    Operand::LiteralBit32(1),
                    Operand::Decoration(spirv::Decoration::Offset),
                    Operand::LiteralBit32(4),
                ],
            ),
        ];
        let mut ctx = Ctx::new(module);

        gc_dead_globals(&mut ctx);

        assert!(ctx.module.types_global_values.is_empty());
        assert!(ctx.module.annotations.is_empty());
    }

    /// The control, and why the carve-out is an allowlist. `BuiltIn WorkgroupSize` on an otherwise
    /// unreferenced constant IS the local size -- nothing else in the module records it -- so that
    /// decoration has to stay a liveness root. Treating every decoration as description dropped the
    /// constant and broke 9,857 of the 14,579 corpus sources on
    /// `VUID-StandaloneSpirv-None-10685`.
    #[test]
    fn gc_keeps_the_constant_a_workgroup_size_decoration_names() {
        let word = 1;
        let vector = 2;
        let one = 3;
        let size = 4;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(5));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(word),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypeVector,
                None,
                Some(vector),
                vec![Operand::IdRef(word), Operand::LiteralBit32(3)],
            ),
            Instruction::new(
                Op::Constant,
                Some(word),
                Some(one),
                vec![Operand::LiteralBit32(1)],
            ),
            Instruction::new(
                Op::ConstantComposite,
                Some(vector),
                Some(size),
                vec![
                    Operand::IdRef(one),
                    Operand::IdRef(one),
                    Operand::IdRef(one),
                ],
            ),
        ];
        module.annotations = vec![Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(size),
                Operand::Decoration(spirv::Decoration::BuiltIn),
                Operand::BuiltIn(spirv::BuiltIn::WorkgroupSize),
            ],
        )];
        let mut ctx = Ctx::new(module);

        gc_dead_globals(&mut ctx);

        assert_eq!(ctx.module.types_global_values.len(), 4);
        assert_eq!(ctx.module.annotations.len(), 1);
    }

    #[test]
    fn gc_drops_dead_pointer_type_owned_only_by_array_stride() {
        let byte = 1;
        let pointer = 2;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(3));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(byte),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(pointer),
                vec![
                    Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                    Operand::IdRef(byte),
                ],
            ),
        ];
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(pointer),
                Operand::Decoration(spirv::Decoration::ArrayStride),
                Operand::LiteralBit32(1),
            ],
        ));
        let mut ctx = Ctx::new(module);

        gc_dead_globals(&mut ctx);

        assert!(ctx.module.types_global_values.is_empty());
        assert!(ctx.module.annotations.is_empty());
    }
}
