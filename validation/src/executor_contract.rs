use crate::case::{
    AuthoredCase, DeviceBufferArrayResource, OutputSelection, Primitive, Stage, TextureFormat,
    ThreadgroupMemoryResource,
};
use crate::requirement::ToolingRequirement;
use metal2vulkan::{
    meta::{TextureDimension, TextureFormat as ReflectedTextureFormat},
    reflect::{
        ResourceKind, SamplerAddressMode, SamplerCoordinates, SamplerFilter, SamplerMipFilter,
        SamplerReduction, ShaderReflection,
    },
};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

pub fn air_call_counts(ll: &str) -> BTreeMap<String, usize> {
    metal2vulkan::air_intrinsics::air_call_counts(ll)
}

pub fn unsupported_air_requirements(source_ll: &str) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    if air_call_counts(source_ll)
        .keys()
        .any(|name| metal2vulkan::meta::implicit_imageblock_texture_format(name).is_err())
    {
        requirements.insert(ToolingRequirement::ImplicitImageblockLiteral);
    }
    // Ask the product what an indirect-command-buffer encoder looks like rather than restating it.
    // This used to key on `!"air.indirect_command_buffer"`, a metadata string no AIR module the
    // harvest has ever produced carries -- so the requirement was never reported for any real
    // encoder. What they do carry is a call into one of the encoder families and an
    // `air.command_buffer` argument-buffer member.
    if air_call_counts(source_ll)
        .keys()
        .any(|name| metal2vulkan::air_intrinsics::is_command_encoder_helper(name))
        || source_ll.contains("!\"air.command_buffer\"")
    {
        requirements.insert(ToolingRequirement::IndirectCommandBuffer);
    }
    requirements
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthCompare {
    Always,
    Less,
    Greater,
}

pub fn depth_compare(reflection: &ShaderReflection) -> DepthCompare {
    match reflection.depth_qualifier {
        Some(metal2vulkan::meta::DepthQualifier::Less) => DepthCompare::Less,
        Some(metal2vulkan::meta::DepthQualifier::Greater) => DepthCompare::Greater,
        Some(metal2vulkan::meta::DepthQualifier::Any) | None => DepthCompare::Always,
    }
}

/// Structural reflection requirements that the shared Metal/Vulkan execution contract cannot yet
/// run. Keeping this inventory beside [`require_case`] prevents triage, checking, and the candidate
/// descriptor path from silently describing different capability boundaries.
pub fn unsupported_reflection_requirements(
    reflection: &ShaderReflection,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    if reflection.function_constants.iter().any(|constant| {
        crate::case::ScalarType::from_metal_abi_type_encoding(&constant.abi_type_encoding)
            .is_none_or(|(scalar, _)| !scalar.supports_metal_function_constant())
    }) {
        requirements.insert(ToolingRequirement::FunctionConstantLiteral);
    }
    if reflection.vertex_attributes.iter().any(|attribute| {
        attribute
            .type_name
            .as_deref()
            .and_then(crate::case::AttributeFormat::from_air_type_name)
            .is_none()
    }) {
        requirements.insert(ToolingRequirement::VertexAttributeLiteral);
    }
    if reflection.tessellation.as_ref().is_some_and(|interface| {
        interface
            .control_point_attributes
            .iter()
            .chain(&interface.patch_attributes)
            .any(|attribute| {
                attribute
                    .type_name
                    .as_deref()
                    .and_then(crate::case::AttributeFormat::from_air_type_name)
                    .is_none_or(|format| !format.supports_tessellation_interface())
            })
    }) {
        requirements.insert(ToolingRequirement::TessellationAttributeLiteral);
    }
    if reflection.tessellation.as_ref().is_some_and(|interface| {
        [
            interface.instance_id.as_ref(),
            interface.amplification_id.as_ref(),
            interface.amplification_count.as_ref(),
        ]
        .into_iter()
        .flatten()
        .any(|attribute| {
            attribute
                .type_name
                .as_deref()
                .and_then(crate::case::AttributeFormat::from_air_type_name)
                .is_none_or(|format| !format.supports_tessellation_system_value())
        })
    }) {
        requirements.insert(ToolingRequirement::TessellationSystemInputLiteral);
    }
    if reflection
        .fragment_imageblock
        .as_ref()
        .is_some_and(|imageblock| {
            imageblock.members.iter().any(|member| {
                crate::case::FragmentImageblockFormat::from_air_type(&member.type_name, member.size)
                    .is_none()
            })
        })
    {
        requirements.insert(ToolingRequirement::FragmentImageblockMemberLiteral);
    }
    for binding in &reflection.bindings {
        match binding.kind {
            ResourceKind::Buffer
            | ResourceKind::ThreadgroupBuffer
            | ResourceKind::KernelStageInput
            | ResourceKind::Texture
            | ResourceKind::TextureArray
            | ResourceKind::StorageImage
            | ResourceKind::Sampler
            | ResourceKind::StaticSampler
            | ResourceKind::ColorInput
            | ResourceKind::AccelerationStructureShadow
            | ResourceKind::PrimitiveAccelerationStructure
            | ResourceKind::VisibleFunctionTable
            | ResourceKind::IntersectionFunctionTable
            | ResourceKind::EmbeddedArgBufferTexture
            | ResourceKind::EmbeddedArgBufferBuffer
            | ResourceKind::BufferAddressTable => {}
            // A descriptor the translator invented to type an AIR value has no Metal argument
            // behind it, so there is no authored resource for the Vulkan executor to write there
            // and nothing for the Metal oracle to encode. The module still reads through it, so
            // leaving the binding out of the layout is not an option either -- the row is honestly
            // not executable until the harness can supply a deterministic placeholder resource.
            // The Vulkan executor supplies an all-zero one-texel placeholder for a synthesized
            // null texture, so that row is executable. A synthesized read sampler still has no
            // placeholder behind it.
            ResourceKind::SynthesizedNullTexture => {}
            ResourceKind::SynthesizedReadSampler => {
                requirements.insert(ToolingRequirement::SynthesizedPlaceholderDescriptor);
            }
        }
        if binding.kind == ResourceKind::KernelStageInput
            && binding
                .type_name
                .as_deref()
                .and_then(crate::case::AttributeFormat::from_air_type_name)
                .is_none()
        {
            requirements.insert(ToolingRequirement::KernelStageInputLiteral);
        }
        if let Some(shape) = binding.texture_shape {
            if shape.dimension == TextureDimension::Buffer
                && !matches!(
                    binding.kind,
                    ResourceKind::Texture | ResourceKind::StorageImage
                )
            {
                requirements.insert(ToolingRequirement::TextureBufferLiteral);
            }
            if shape.storage_format.is_some_and(|format| {
                !matches!(
                    format,
                    ReflectedTextureFormat::R8
                        | ReflectedTextureFormat::Rgba8
                        | ReflectedTextureFormat::R16f
                        | ReflectedTextureFormat::R16ui
                        | ReflectedTextureFormat::Rg16f
                        | ReflectedTextureFormat::Rg32f
                        | ReflectedTextureFormat::R32i
                        | ReflectedTextureFormat::R32f
                        | ReflectedTextureFormat::R32ui
                        | ReflectedTextureFormat::Rgba32i
                        | ReflectedTextureFormat::Rgba32ui
                        | ReflectedTextureFormat::Rgba32f
                        | ReflectedTextureFormat::Rgba16f
                        | ReflectedTextureFormat::Rgba8ui
                        | ReflectedTextureFormat::Rgba16ui
                        | ReflectedTextureFormat::Rgba8i
                        | ReflectedTextureFormat::Rgba16i
                )
            }) {
                requirements.insert(ToolingRequirement::StorageTextureFormatLiteral);
            }
        }
        if let Some(state) = binding.static_sampler {
            if state.reduction != SamplerReduction::WeightedAverage {
                requirements.insert(ToolingRequirement::StaticSamplerReduction);
            }
            if state.coordinates == SamplerCoordinates::Pixel
                && (state.min_filter != state.mag_filter
                    || state.mip_filter == SamplerMipFilter::Linear)
            {
                requirements.insert(ToolingRequirement::StaticSamplerPixelFilter);
            }
        }
    }
    requirements
}

/// Source-aware executor requirements for constexpr sampler states. Pixel addressing and bicubic
/// filtering are implemented by the product's fetch-based shader lowering for supported image-call
/// shapes; in those paths the Vulkan sampler descriptor is only an interface placeholder and can be
/// canonicalized legally. Calls that still consume native sampler state remain visible gaps.
pub fn unsupported_source_requirements(
    source_ll: &str,
    reflection: &ShaderReflection,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = unsupported_reflection_requirements(reflection);
    requirements.extend(unsupported_air_requirements(source_ll));
    let states = reflection
        .bindings
        .iter()
        .filter_map(|binding| binding.static_sampler)
        .collect::<Vec<_>>();
    let bicubic = states.iter().copied().filter(|state| {
        state.min_filter == SamplerFilter::Bicubic || state.mag_filter == SamplerFilter::Bicubic
    });
    if bicubic
        .clone()
        .any(|state| sampler_state_needs_native_descriptor(source_ll, state))
    {
        requirements.insert(ToolingRequirement::StaticSamplerBicubic);
    }
    let invalid_pixel_address = states.iter().copied().filter(|state| {
        state.coordinates == SamplerCoordinates::Pixel
            && (!matches!(
                state.address_mode_s,
                SamplerAddressMode::ClampToEdge | SamplerAddressMode::ClampToBorder
            ) || !matches!(
                state.address_mode_t,
                SamplerAddressMode::ClampToEdge | SamplerAddressMode::ClampToBorder
            ))
    });
    if invalid_pixel_address
        .clone()
        .any(|state| sampler_state_needs_native_descriptor(source_ll, state))
    {
        requirements.insert(ToolingRequirement::StaticSamplerPixelAddress);
    }
    requirements
}

/// Observation gaps shared by corpus classification and the authored graphics executors.
///
/// These are not shader-translation gaps: they mean the validation harness cannot yet construct
/// an observable experiment for an otherwise reflected interface.
pub fn unsupported_observation_requirements(
    stage: Stage,
    reflection: &ShaderReflection,
) -> BTreeSet<ToolingRequirement> {
    let mut requirements = BTreeSet::new();
    match stage {
        Stage::Vertex => {
            let writes_position = reflection
                .vertex_builtins
                .is_some_and(|builtins| builtins.writes_position);
            let has_shader_resource_output = reflection.bindings.iter().any(|binding| {
                matches!(
                    binding.kind,
                    ResourceKind::Buffer
                        | ResourceKind::StorageImage
                        | ResourceKind::EmbeddedArgBufferBuffer
                        | ResourceKind::EmbeddedArgBufferTexture
                ) && !matches!(
                    binding.access,
                    Some(metal2vulkan::reflect::ResourceAccess::ReadOnly)
                )
            });
            if !writes_position && !has_shader_resource_output {
                requirements.insert(ToolingRequirement::VertexSideEffectObservation);
            }
        }
        Stage::Fragment => {
            for varying in &reflection.varyings {
                if varying
                    .type_name
                    .as_deref()
                    .and_then(crate::observation_contract::ObservationType::parse)
                    .is_none()
                {
                    requirements.insert(ToolingRequirement::FragmentVaryingObservationType);
                }
                if crate::observation_contract::metal_field_name(
                    varying.location,
                    varying.name.as_deref(),
                    varying.user_semantic.as_deref(),
                )
                .is_err()
                {
                    requirements.insert(ToolingRequirement::FragmentVaryingLinkage);
                }
            }
            if reflection.render_targets.iter().any(|target| {
                target
                    .type_name
                    .as_deref()
                    .and_then(crate::observation_contract::ObservationType::parse)
                    .is_none()
            }) {
                requirements.insert(ToolingRequirement::FragmentOutputObservationType);
            }
        }
        Stage::Kernel => {}
    }
    requirements
}

fn sampler_state_needs_native_descriptor(
    source_ll: &str,
    state: metal2vulkan::reflect::StaticSamplerState,
) -> bool {
    let calls = source_ll.lines().filter_map(|line| {
        let call = line.split(" call ").nth(1)?;
        let name = call.split('@').nth(1)?.split('(').next()?;
        (name.starts_with("air.sample_")
            || name.starts_with("air.gather_")
            || name.starts_with("air.calculate_"))
        .then_some(name)
    });
    calls
        .map(|name| sampler_call_uses_emulated_state(name, state))
        .any(|emulated| !emulated)
}

fn sampler_call_uses_emulated_state(
    name: &str,
    state: metal2vulkan::reflect::StaticSamplerState,
) -> bool {
    if name.starts_with("air.gather_texture_2d") || name.starts_with("air.gather_depth_2d") {
        // Pixel gathers are reconstructed from four fetches. A normalized gather does not consume
        // min/mag filtering, so substituting a legal linear descriptor for bicubic is exact too.
        return state.coordinates == SamplerCoordinates::Pixel
            || state.min_filter == SamplerFilter::Bicubic
            || state.mag_filter == SamplerFilter::Bicubic;
    }
    if name.starts_with("air.sample_depth_2d") {
        return state.coordinates == SamplerCoordinates::Pixel
            && state.min_filter == SamplerFilter::Linear
            && state.mag_filter == SamplerFilter::Linear;
    }
    if name.starts_with("air.calculate_clamped_lod_texture_2d")
        || name.starts_with("air.calculate_unclamped_lod_texture_2d")
    {
        // ImageQueryLod observes coordinate derivatives, bias, and clamps; min/mag reconstruction
        // filtering is not an input. Replacing an unavailable bicubic descriptor filter with linear
        // therefore leaves the query result unchanged.
        return state.min_filter == SamplerFilter::Bicubic
            || state.mag_filter == SamplerFilter::Bicubic;
    }
    if !name.starts_with("air.sample_texture_") {
        return false;
    }
    let float_result = name.contains(".v4f16") || name.contains(".v4f32");
    if state.coordinates == SamplerCoordinates::Pixel {
        if state.min_filter == SamplerFilter::Nearest && state.mag_filter == SamplerFilter::Nearest
        {
            return true;
        }
        if name.contains("_array.") {
            return true;
        }
        if state.min_filter == SamplerFilter::Linear && state.mag_filter == SamplerFilter::Linear {
            return float_result
                && (name.starts_with("air.sample_texture_2d")
                    || name.starts_with("air.sample_texture_3d"));
        }
    }
    (state.min_filter == SamplerFilter::Bicubic || state.mag_filter == SamplerFilter::Bicubic)
        && float_result
        && (name.starts_with("air.sample_texture_1d") || name.starts_with("air.sample_texture_2d"))
}

pub fn require_reflection(
    case: &AuthoredCase,
    source_ll: &str,
    reflection: &ShaderReflection,
    executor: &str,
) -> Result<(), String> {
    let mut unsupported = unsupported_source_requirements(source_ll, reflection);
    unsupported.extend(unsupported_observation_requirements(case.stage, reflection));
    if !unsupported.is_empty() {
        return Err(format!(
            "{executor} lacks authored tooling for {}",
            unsupported
                .into_iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    let authored_depth = case
        .depth_stencil
        .as_ref()
        .is_some_and(|attachment| attachment.initial_depth_b64.is_some());
    let authored_stencil = case
        .depth_stencil
        .as_ref()
        .is_some_and(|attachment| attachment.initial_stencil_b64.is_some());
    if !reflection.depth_members.is_empty() && !authored_depth {
        return Err(format!(
            "{executor} fragment depth output requires an authored depth attachment"
        ));
    }
    if !reflection.stencil_members.is_empty() && !authored_stencil {
        return Err(format!(
            "{executor} fragment stencil output requires an authored stencil attachment"
        ));
    }
    if reflection.fragment_imageblock.is_some() && case.fragment_imageblock.is_none() {
        return Err(format!(
            "{executor} custom fragment imageblock requires authored fragment_imageblock planes"
        ));
    }
    let color_inputs = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::ColorInput)
        .collect::<Vec<_>>();
    for binding in &color_inputs {
        if !case
            .render_targets
            .iter()
            .any(|target| target.index == binding.metal_index)
        {
            return Err(format!(
                "{executor} framebuffer-fetch input {} requires an authored render target at the same index",
                binding.metal_index
            ));
        }
    }
    if !color_inputs.is_empty()
        && !case.draw.as_ref().is_some_and(|draw| {
            draw.primitive == Primitive::Triangle
                && draw.vertex_start == 0
                && draw.vertex_count == 3
                && draw.instance_count == 1
        })
    {
        return Err(format!(
            "{executor} framebuffer-fetch execution requires one authored fullscreen triangle (triangle, vertex_start=0, vertex_count=3, instance_count=1)"
        ));
    }
    Ok(())
}

/// Metal's granularity for a threadgroup-memory allocation, in bytes.
///
/// `MTLComputeCommandEncoder`'s `setThreadgroupMemoryLength:atIndex:` asserts on anything else --
/// measured, not assumed: running an authored case that declares 4 bytes under
/// `METAL_DEVICE_WRAPPER_TYPE=1` fails with `-[MTLDebugComputeCommandEncoder
/// setThreadgroupMemoryLength:atIndex:]:799: failed assertion 'length(4) must be a multiple of 16
/// bytes.'`. Without the validation layer the driver accepts the call, so an authored length that
/// Metal cannot allocate executes and records evidence anyway.
///
/// The Metal executor passes the authored length straight to that API, so the length a manifest
/// declares IS the length Metal is asked for and has to be one Metal can allocate. Rounding it up
/// inside the executor instead would hand Metal a different size from the one the manifest states
/// and from the one the Vulkan candidate reads.
pub const METAL_THREADGROUP_MEMORY_GRANULARITY: u32 = 16;

/// Why a threadgroup-memory declaration is not one a Metal compute encoder can bind, or `None`.
///
/// One rule, consulted by [`require_case`] at both executors' entry and by the authored-case
/// checker before install, so an unallocatable length cannot reach the store from either side.
pub fn threadgroup_memory_length_error(
    resource: &ThreadgroupMemoryResource,
    context: &str,
) -> Option<String> {
    if resource.length == 0 {
        return Some(format!(
            "{context} threadgroup memory {} declares zero bytes, which satisfies the reflected requirement without allocating anything",
            resource.binding
        ));
    }
    if !resource
        .length
        .is_multiple_of(METAL_THREADGROUP_MEMORY_GRANULARITY)
    {
        return Some(format!(
            "{context} threadgroup memory {} declares {} bytes, which Metal cannot allocate: a threadgroup-memory length must be a multiple of {METAL_THREADGROUP_MEMORY_GRANULARITY} bytes",
            resource.binding, resource.length
        ));
    }
    None
}

/// Why a device-buffer array does not author every element it declares, or `None`.
///
/// Metal binds element `i` of the array at buffer index `binding + i` and requires all `length` of
/// them: a slot the manifest leaves out is not bound at all, and
/// `validateComputeFunctionArguments` refuses the dispatch with `missing Buffer binding at index 2
/// for buffers.2[0]`. Without `MTL_DEBUG_LAYER=1` the driver dispatches anyway and the shader
/// reads undefined memory through the empty slot while the case still reports `qualified`.
///
/// There is no honest value to substitute for an unauthored operand, so this refuses. Contrast
/// [`buffer_bytes_padded_to_declared_size`], which pads a *declared size*: that padding lies past
/// everything the manifest describes, while a missing array element is a resource the manifest
/// claims exists and never says anything about.
///
/// One rule, consulted by [`require_case`] at both executors' entry and by the authored-case
/// checker before install, so an incomplete array cannot reach the store from either side.
pub fn device_buffer_array_completeness_error(
    resource: &DeviceBufferArrayResource,
    context: &str,
) -> Option<String> {
    let authored = resource
        .elements
        .iter()
        .map(|element| element.index)
        .collect::<BTreeSet<_>>();
    let missing = (0..resource.length)
        .filter(|index| !authored.contains(index))
        .map(|index| index.to_string())
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return None;
    }
    Some(format!(
        "{context} device-buffer array {} declares {} elements but authors no bytes for element{} {}",
        resource.binding,
        resource.length,
        if missing.len() == 1 { "" } else { "s" },
        missing.join(", ")
    ))
}

/// The bytes to bind for a buffer argument, padded to the size AIR declares for it.
///
/// A case authors only the bytes its shader reads, which is routinely far fewer than the
/// argument's declared struct -- one byte against the 264 of `params`, sixteen against the 1179628
/// of `spanTable`. Both APIs size a binding from the declaration rather than from the use, and
/// Metal's validation layer refuses the short one outright:
///
/// ```text
/// argument params[0] from Buffer(29) with offset(0) and length(1) has space for 1 bytes,
/// but argument has a length(264).
/// ```
///
/// Without that layer it binds the short buffer anyway and any read past the end is undefined.
/// Padding the *binding* leaves the authored bytes at the front and does not author the padding:
/// a case whose result depends on it is reading bytes it never declared, and now reads a stable
/// zero on both executors rather than whatever followed each allocation.
///
/// Never shortens. A runtime `array_ref` declares no size at all, and a case may legitimately
/// author more than the declaration.
pub fn buffer_bytes_padded_to_declared_size(
    authored: &[u8],
    declared_size: Option<usize>,
) -> Cow<'_, [u8]> {
    match declared_size {
        Some(size) if size > authored.len() => {
            let mut padded = Vec::with_capacity(size);
            padded.extend_from_slice(authored);
            padded.resize(size, 0);
            Cow::Owned(padded)
        }
        _ => Cow::Borrowed(authored),
    }
}

/// The declared byte size of the `[[buffer(n)]]` argument at `metal_index`, as AIR states it
/// through `air.arg_type_size`. `None` when the entry declares no buffer there, or declares one
/// whose extent is not a fixed object.
pub fn reflected_buffer_declared_size(
    reflection: &ShaderReflection,
    metal_index: u32,
) -> Option<usize> {
    reflection
        .bindings
        .iter()
        .find(|binding| binding.kind == ResourceKind::Buffer && binding.metal_index == metal_index)
        .and_then(|binding| binding.declared_size)
        .map(|size| size as usize)
}

/// Shared capability boundary for the literal Metal and Vulkan executors.
///
/// Unsupported manifest resources are rejected as a whole; no runner may drop them or synthesize
/// defaults while executing the supported literal-resource subset.
pub fn require_case(case: &AuthoredCase, executor: &str) -> Result<(), String> {
    for resource in &case.device_buffer_arrays {
        if let Some(error) = device_buffer_array_completeness_error(resource, executor) {
            return Err(error);
        }
    }
    match case.stage {
        Stage::Kernel => {
            for resource in &case.threadgroup_memory {
                if let Some(error) = threadgroup_memory_length_error(resource, executor) {
                    return Err(error);
                }
            }
            if !case.vertex_inputs.is_empty() {
                return Err(format!(
                    "{executor} kernel execution does not accept vertex inputs"
                ));
            }
            if !case.render_targets.is_empty() && case.imageblock.is_none() {
                return Err(format!(
                    "{executor} kernel render targets require an authored imageblock"
                ));
            }
            if !case.render_targets.is_empty()
                && case
                    .imageblock
                    .as_ref()
                    .is_some_and(|imageblock| imageblock.implicit_coverage.is_none())
            {
                return Err(format!(
                    "{executor} kernel render targets require authored implicit imageblock coverage"
                ));
            }
            if matches!(case.output, OutputSelection::RenderTarget { .. })
                && case.imageblock.is_none()
            {
                return Err(format!(
                    "{executor} kernel render-target output requires an authored imageblock"
                ));
            }
        }
        Stage::Fragment | Stage::Vertex => {
            // A custom fragment imageblock is an attachment. It lives in the pass's tile memory
            // and both executors already raster a fragment pass that has no color or depth
            // attachment at all -- the Metal one by setting explicit raster dimensions, the Vulkan
            // one by building a subpass with none. Requiring a color attachment beside it would
            // require authoring a render target the shader never writes, which the resource
            // contract rejects because reflection does not declare one: the two rules together
            // made every fragment entry whose only output is a custom imageblock unauthorable.
            if case.render_targets.is_empty()
                && case.depth_stencil.is_none()
                && case.fragment_imageblock.is_none()
                && !(case.stage == Stage::Fragment && matches!(case.output, OutputSelection::None))
                && !case.is_rasterization_disabled_vertex()
            {
                return Err(format!(
                    "{executor} graphics execution requires at least one attachment"
                ));
            }
            if !case.kernel_stage_inputs.is_empty()
                || !case.threadgroup_memory.is_empty()
                || case.imageblock.is_some()
            {
                return Err(format!(
                    "{executor} graphics execution does not accept kernel stage inputs, threadgroup memory, or imageblocks"
                ));
            }
            if case.fragment_imageblock.is_some() && case.stage != Stage::Fragment {
                return Err(format!(
                    "{executor} custom fragment imageblocks are valid only for fragment execution"
                ));
            }
            if case.stage == Stage::Fragment && !case.vertex_inputs.is_empty() {
                return Err(format!(
                    "{executor} fragment execution does not accept authored vertex inputs"
                ));
            }
            let dimensions = case
                .render_targets
                .first()
                .map(|target| target.dimensions)
                .or_else(|| {
                    case.depth_stencil
                        .as_ref()
                        .map(|attachment| attachment.dimensions)
                })
                .unwrap_or([1, 1]);
            // The imageblock is sized by the pass, not the other way round. Without this an
            // attachment-less case could author a plane larger than the 1x1 region both executors
            // raster and compare bytes the shader never covered.
            if case
                .fragment_imageblock
                .as_ref()
                .is_some_and(|imageblock| imageblock.dimensions != dimensions)
            {
                return Err(format!(
                    "{executor} custom fragment imageblock dimensions must match the pass attachments"
                ));
            }
            for target in &case.render_targets {
                if target.dimensions != dimensions {
                    return Err(format!(
                        "{executor} graphics render targets must have identical dimensions"
                    ));
                }
                if target.format == TextureFormat::Depth32Float {
                    return Err(format!(
                        "{executor} color render target {} cannot use depth32_float",
                        target.index
                    ));
                }
            }
            if case
                .depth_stencil
                .as_ref()
                .is_some_and(|attachment| attachment.dimensions != dimensions)
            {
                return Err(format!(
                    "{executor} graphics attachments must have identical dimensions"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_buffer_array(
        binding: u32,
        length: u32,
        indices: &[u32],
    ) -> DeviceBufferArrayResource {
        DeviceBufferArrayResource {
            binding,
            length,
            elements: indices
                .iter()
                .map(|index| crate::case::DeviceBufferArrayElementResource {
                    index: *index,
                    role: crate::case::ResourceRole::Input,
                    bytes_b64: Some("AAAAAA==".into()),
                    initial_bytes_b64: None,
                })
                .collect(),
        }
    }

    /// Metal requires every element of a device-buffer array bound, so a manifest that declares
    /// more elements than it authors has to be refused rather than left to read undefined memory.
    #[test]
    fn a_device_buffer_array_must_author_every_element_it_declares() {
        let complete = |length, indices: &[u32]| {
            device_buffer_array_completeness_error(
                &device_buffer_array(0, length, indices),
                "manifest",
            )
        };
        assert_eq!(complete(3, &[0, 1, 2]), None);
        assert_eq!(complete(0, &[]), None);
        assert_eq!(complete(2, &[1, 0]), None);
        let one = complete(3, &[0, 1]).expect("element 2 is unauthored");
        assert!(one.ends_with("authors no bytes for element 2"), "{one}");
        let two =
            device_buffer_array_completeness_error(&device_buffer_array(4, 4, &[1]), "manifest")
                .expect("elements 0, 2 and 3 are unauthored");
        assert!(
            two.starts_with("manifest device-buffer array 4 declares 4 elements"),
            "{two}"
        );
        assert!(two.ends_with("no bytes for elements 0, 2, 3"), "{two}");
    }

    /// Both APIs size a buffer binding from the argument's declaration, not from the bytes a case
    /// chose to author. Padding must extend the binding without disturbing the authored prefix,
    /// and must never shorten it: a runtime `array_ref` has no declared size at all, and a case
    /// may author more than the declaration.
    #[test]
    fn a_buffer_binding_is_padded_to_the_declared_size_and_never_shortened() {
        let padded = buffer_bytes_padded_to_declared_size;
        assert_eq!(padded(&[1, 2, 3], Some(6)).as_ref(), &[1, 2, 3, 0, 0, 0]);
        assert_eq!(padded(&[1, 2, 3], Some(3)).as_ref(), &[1, 2, 3]);
        assert_eq!(padded(&[1, 2, 3], Some(2)).as_ref(), &[1, 2, 3]);
        assert_eq!(padded(&[1, 2, 3], None).as_ref(), &[1, 2, 3]);
        assert!(matches!(padded(&[1, 2, 3], Some(3)), Cow::Borrowed(_)));
        assert!(matches!(padded(&[1, 2, 3], Some(4)), Cow::Owned(_)));
    }
    use crate::case::{
        BufferResource, Comparison, DepthStencilResource, Dispatch, Draw, ExecutionSafety,
        FragmentImageblockFormat, FragmentImageblockMemberResource, FragmentImageblockResource,
        FunctionConstant, OutputSelection, Primitive, RenderTargetResource, ResourceRole,
        ScalarType, TextureFormat, TextureResource, TextureType, VertexObservation,
    };

    fn case() -> AuthoredCase {
        AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: "22".repeat(32),
            name: "literal".into(),
            entry: "main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: None,
        }
    }

    #[test]
    fn unsupported_manifest_data_is_rejected_not_ignored_or_defaulted() {
        let mut case = case();
        assert!(require_case(&case, "test executor").is_ok());
        case.function_constants.push(FunctionConstant {
            index: 0,
            scalar_type: ScalarType::U32,
            lanes: 1,
            bytes_b64: "AQAAAA==".into(),
        });
        assert!(require_case(&case, "test executor").is_ok());
        case.textures.push(TextureResource {
            binding: 1,
            role: ResourceRole::Output,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1, 1],
            sample_count: 1,
            bytes_b64: None,
            initial_bytes_b64: Some("qqqqqqqqqqqqqqqqqqqqqg==".into()),
        });
        case.output = OutputSelection::Texture {
            binding: 1,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        };
        assert!(require_case(&case, "test executor").is_ok());
    }

    #[test]
    fn a_threadgroup_memory_length_metal_cannot_allocate_is_refused_at_the_executor() {
        let mut case = case();
        case.threadgroup_memory = vec![crate::case::ThreadgroupMemoryResource {
            binding: 0,
            length: METAL_THREADGROUP_MEMORY_GRANULARITY,
        }];
        assert!(require_case(&case, "test executor").is_ok());

        // Measured against Metal's own validation layer: the encoder asserts on a length that is
        // not a multiple of sixteen, so no runner may pass one through.
        case.threadgroup_memory[0].length = METAL_THREADGROUP_MEMORY_GRANULARITY - 1;
        assert!(require_case(&case, "test executor")
            .unwrap_err()
            .contains("must be a multiple of 16 bytes"));

        case.threadgroup_memory[0].length = 0;
        assert!(require_case(&case, "test executor")
            .unwrap_err()
            .contains("zero bytes"));
    }

    #[test]
    fn source_and_observation_requirements_are_enforced_by_the_shared_executor_gate() {
        let reflection = ShaderReflection::from_kernel(
            &metal2vulkan::meta::KernMeta::default(),
            Some("kernel"),
            [1, 1, 1],
        );
        for (source_ll, requirement) in [
            (
                "call void @air.store.implicit_imageblock.v3f16()",
                ToolingRequirement::ImplicitImageblockLiteral,
            ),
            (
                "call void @air.set_kernel_buffer_compute_command.p1i8()",
                ToolingRequirement::IndirectCommandBuffer,
            ),
            (
                "!0 = !{i32 0, !\"air.command_buffer\", !\"air.location_index\", i32 0}",
                ToolingRequirement::IndirectCommandBuffer,
            ),
        ] {
            assert!(unsupported_air_requirements(source_ll).contains(&requirement));
            assert!(
                require_reflection(&case(), source_ll, &reflection, "test executor")
                    .unwrap_err()
                    .contains(requirement.as_str())
            );
        }

        let vertex =
            ShaderReflection::from_vertex(&metal2vulkan::meta::VertMeta::default(), Some("vertex"));
        assert!(unsupported_observation_requirements(Stage::Vertex, &vertex)
            .contains(&ToolingRequirement::VertexSideEffectObservation));
        let mut authored = case();
        authored.stage = Stage::Vertex;
        assert!(require_reflection(&authored, "", &vertex, "test executor")
            .unwrap_err()
            .contains(ToolingRequirement::VertexSideEffectObservation.as_str()));
    }

    #[test]
    fn reflected_stage_input_types_use_the_authored_attribute_contract() {
        use metal2vulkan::meta::{KernMeta, KernRole, VertMeta, VertRole};

        let mut vertex_meta = VertMeta {
            roles: vec![(0, VertRole::VertexInput(3))],
            ..Default::default()
        };
        vertex_meta.vertex_input_types.insert(3, "uchar4".into());
        let vertex = ShaderReflection::from_vertex(&vertex_meta, Some("vertex"));
        assert!(!unsupported_reflection_requirements(&vertex)
            .contains(&ToolingRequirement::VertexAttributeLiteral));
        vertex_meta.vertex_input_types.insert(3, "ulong".into());
        let vertex = ShaderReflection::from_vertex(&vertex_meta, Some("vertex"));
        assert!(unsupported_reflection_requirements(&vertex)
            .contains(&ToolingRequirement::VertexAttributeLiteral));

        let mut kernel_meta = KernMeta {
            roles: vec![(0, KernRole::StageInput(2))],
            ..Default::default()
        };
        kernel_meta
            .stage_input_type_names
            .insert(0, "float3".into());
        let supported = ShaderReflection::from_kernel(&kernel_meta, Some("kernel"), [1, 1, 1]);
        assert!(!unsupported_reflection_requirements(&supported)
            .contains(&ToolingRequirement::KernelStageInputLiteral));

        kernel_meta.stage_input_type_names.insert(0, "ulong".into());
        let unsupported = ShaderReflection::from_kernel(&kernel_meta, Some("kernel"), [1, 1, 1]);
        assert!(unsupported_reflection_requirements(&unsupported)
            .contains(&ToolingRequirement::KernelStageInputLiteral));
    }

    #[test]
    fn fragment_cases_use_the_shared_render_contract() {
        let mut case = case();
        case.stage = Stage::Fragment;
        case.buffers.clear();
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
        });
        case.render_targets.push(RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [2, 2],
            initial_bytes_b64: "q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urqw==".into(),
        });
        case.output = OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [2, 2],
        };
        assert!(require_case(&case, "test executor").is_ok());

        let mut attachmentless = case.clone();
        attachmentless.render_targets.clear();
        attachmentless.output = OutputSelection::None;
        assert!(require_case(&attachmentless, "test executor").is_ok());
        attachmentless.output = OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        };
        assert!(require_case(&attachmentless, "test executor")
            .unwrap_err()
            .contains("at least one attachment"));

        case.render_targets.push(RenderTargetResource {
            index: 1,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 2],
            initial_bytes_b64: "q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urqw==".into(),
        });
        assert!(require_case(&case, "test executor")
            .unwrap_err()
            .contains("identical dimensions"));

        case.render_targets.pop();
        case.stage = Stage::Vertex;
        case.vertex_observation = Some(VertexObservation::Position);
        assert!(require_case(&case, "test executor").is_ok());
    }

    #[test]
    fn framebuffer_fetch_uses_the_authored_render_target_contract() {
        let ll = r#"define <4 x float> @frag(<4 x float> %color) { ret <4 x float> %color }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
"#;
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        assert_eq!(reflection.bindings[0].type_name.as_deref(), Some("float4"));
        let mut case = case();
        case.stage = Stage::Fragment;
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap_err()
            .contains("same index"));
        case.render_targets.push(RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1],
            initial_bytes_b64: "q6urq6urq6urq6urq6urqw==".into(),
        });
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
        });
        assert!(require_reflection(&case, ll, &reflection, "test executor").is_ok());
    }

    #[test]
    fn texture_shape_requirements_match_literal_executor_boundaries() {
        use metal2vulkan::meta::{KernMeta, KernRole};

        let names = [
            "texture2d_ms<float, read>",
            "texture_buffer<uint, read>",
            "texture2d<uint, write>",
        ];
        let mut meta = KernMeta {
            roles: names
                .iter()
                .enumerate()
                .map(|(index, _)| (index as u32, KernRole::Texture(index as u32)))
                .collect(),
            ..Default::default()
        };
        for (index, name) in names.into_iter().enumerate() {
            meta.texture_type_names.insert(index as u32, name.into());
        }
        let reflection = ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1]);
        assert!(unsupported_reflection_requirements(&reflection).is_empty());
    }

    #[test]
    fn a_synthesized_null_texture_is_executable_and_a_read_sampler_is_not() {
        // `air.get_null_texture_2d()` whose handle is read: the translator binds a real image at a
        // binding no Metal argument produces. The Vulkan executor now writes an all-zero one-texel
        // placeholder there, so the row is executable; only a synthesized read sampler still has
        // nothing behind it.
        let ll = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  %tex = call ptr addrspace(1) @air.get_null_texture_2d()
  %width = call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %width, ptr addrspace(1) %out, align 4
  ret void
}

declare ptr addrspace(1) @air.get_null_texture_2d()
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
        let tmp = std::env::temp_dir().join(format!(
            "m2v_validation_placeholder_requirement_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).expect("scratch directory");
        let (_spirv, reflection) = metal2vulkan::translate_sanitized_native_reflected(
            ll,
            metal2vulkan::passes::Stage::Kernel,
            &tmp,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .expect("the placeholder-reading kernel translates");
        let _ = std::fs::remove_dir_all(&tmp);
        let placeholder = reflection
            .bindings
            .iter()
            .position(|binding| binding.kind == ResourceKind::SynthesizedNullTexture)
            .expect("the placeholder the module reads through is reported");
        assert!(unsupported_reflection_requirements(&reflection).is_empty());

        let mut sampler = reflection.clone();
        sampler.bindings[placeholder].kind = ResourceKind::SynthesizedReadSampler;
        assert!(unsupported_reflection_requirements(&sampler)
            .contains(&ToolingRequirement::SynthesizedPlaceholderDescriptor));
    }

    #[test]
    fn custom_fragment_imageblock_accepts_every_product_storage_format() {
        let ll = include_str!("../fixtures/public/fragment_custom_imageblock.ll");
        let mut reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        {
            let members = &mut reflection.fragment_imageblock.as_mut().unwrap().members;
            for (member, (type_name, size)) in
                members
                    .iter_mut()
                    .zip([("half", 2), ("half4", 8), ("uchar4", 4), ("ushort", 2)])
            {
                member.type_name = type_name.into();
                member.size = size;
            }
        }
        assert!(!unsupported_reflection_requirements(&reflection)
            .contains(&ToolingRequirement::FragmentImageblockMemberLiteral));

        let member = &mut reflection.fragment_imageblock.as_mut().unwrap().members[0];
        member.type_name = "float4".into();
        member.size = 16;
        assert!(unsupported_reflection_requirements(&reflection)
            .contains(&ToolingRequirement::FragmentImageblockMemberLiteral));
    }

    #[test]
    fn fragment_depth_output_uses_the_shared_authored_attachment_contract() {
        let ll = r#"
define float @frag() { ret float 5.000000e-01 }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.depth", !"air.depth_qualifier", !"air.less", !"air.arg_type_name", !"float"}
"#;
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        assert_eq!(
            reflection.depth_qualifier,
            Some(metal2vulkan::meta::DepthQualifier::Less)
        );
        assert_eq!(depth_compare(&reflection), DepthCompare::Less);
        assert!(unsupported_source_requirements(ll, &reflection).is_empty());
        let mut case = case();
        case.stage = Stage::Fragment;
        assert!(require_reflection(&case, ll, &reflection, "test executor")
            .unwrap_err()
            .contains("authored depth attachment"));
        case.depth_stencil = Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: Some("AAAAAA==".into()),
            initial_stencil_b64: None,
        });
        assert!(require_reflection(&case, ll, &reflection, "test executor").is_ok());

        let stencil_ll = ll.replace(
            "!\"air.depth\", !\"air.depth_qualifier\", !\"air.less\", !\"air.arg_type_name\", !\"float\"",
            "!\"air.stencil\", !\"air.arg_type_name\", !\"uint\"",
        );
        let stencil_reflection = metal2vulkan::reflect_sanitized(
            &stencil_ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        assert!(unsupported_source_requirements(&stencil_ll, &stencil_reflection).is_empty());
        case.depth_stencil = Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: None,
            initial_stencil_b64: Some("AA==".into()),
        });
        assert!(
            require_reflection(&case, &stencil_ll, &stencil_reflection, "test executor").is_ok()
        );
    }

    /// A fragment entry whose only output is a custom imageblock has no attachment to author: AIR
    /// declares no render target, and the resource contract rejects one the reflection does not
    /// name. Both executors already raster an attachment-free fragment pass, so the imageblock is
    /// the attachment -- and it is sized by the pass, not the other way round.
    #[test]
    fn a_custom_fragment_imageblock_is_the_attachment_and_takes_the_pass_dimensions() {
        let imageblock = |dimensions: [u32; 2]| FragmentImageblockResource {
            dimensions,
            members: vec![FragmentImageblockMemberResource {
                semantic: "color".into(),
                format: FragmentImageblockFormat::Half4,
                role: ResourceRole::InOut,
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAAAAAAA=".into()),
            }],
        };
        let mut case = case();
        case.stage = Stage::Fragment;
        case.buffers = vec![];
        case.render_targets = vec![];
        case.output = OutputSelection::FragmentImageblock {
            semantic: "color".into(),
            origin: [0, 0],
            dimensions: [1, 1],
        };
        assert!(
            require_case(&case, "test executor")
                .unwrap_err()
                .contains("requires at least one attachment"),
            "an output with no attachment and no imageblock is still refused"
        );
        case.fragment_imageblock = Some(imageblock([1, 1]));
        require_case(&case, "test executor")
            .expect("a custom fragment imageblock is an attachment");
        // The attachment-free pass rasters 1x1, so a larger plane would compare bytes no fragment
        // ever covered.
        case.fragment_imageblock = Some(imageblock([2, 2]));
        assert!(
            require_case(&case, "test executor")
                .unwrap_err()
                .contains("imageblock dimensions must match"),
            "a plane larger than the pass is refused"
        );
    }

    #[test]
    fn shader_emulated_pixel_address_state_is_not_a_native_descriptor_gap() {
        let sample = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053330, i64 0], align 8
define void @k(ptr addrspace(1) %tex) {
entry:
  %sample = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> zeroinitializer)
  ret void
}
!air.kernel = !{!0}
!air.sampler_states = !{!4}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>"}
!4 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
        let reflection = metal2vulkan::reflect_sanitized(
            sample,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        assert!(!unsupported_source_requirements(sample, &reflection)
            .contains(&ToolingRequirement::StaticSamplerPixelAddress));

        let native_sample = sample.replace(
            "air.sample_texture_2d.v4f32",
            "air.sample_compare_depth_2d.f32",
        );
        assert!(unsupported_source_requirements(&native_sample, &reflection)
            .contains(&ToolingRequirement::StaticSamplerPixelAddress));
    }
}
