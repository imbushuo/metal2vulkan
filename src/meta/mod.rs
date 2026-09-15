//! AIR stage-interface metadata parsed from sanitized `.ll` before SPIR-V emission. The parser maps
//! each entry-function parameter to the role that the interface pass uses to synthesize Vulkan
//! bindings, stage inputs, and stage outputs.

use crate::passes::Stage;
use std::collections::HashMap;

mod embedded;
mod function_constants;
mod globals;
mod intersections;
mod samplers;
mod textures;
mod types;
use embedded::{
    body_uses_texture_intrinsic, detect_embedded_arguments, detect_embedded_textures,
    unsurfaced_embedded_resources, ArgumentBuffers,
};
pub use embedded::{embedded_synthetic_texture_index, EmbeddedArgument, EmbeddedTexture};
pub(crate) use function_constants::function_constants_without_a_supplied_value;
pub use function_constants::{parse_function_constants, FunctionConstant};
use globals::{location_index_with_static, static_init_int_global_values};
pub(crate) use globals::{static_init_foldable_global_values, StaticIntValue};
pub use intersections::{
    AirIntersectionFamily, AirIntersectionInstancing, AirIntersectionResultField,
};
pub use samplers::{is_static_sampler_global, static_sampler_name_order};
pub use textures::{
    texture_shape_from_name, TextureComponent, TextureDimension, TextureFormat, TextureShape,
    TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT,
};
pub(crate) use types::storage_air_type_for_size;
use types::{parse_struct_info, struct_info_ref, tokenize, Tok};
pub use types::{primitive_air_type_from_name, AirMember, AirScalar, AirType};

/// Whether stable AIR argument metadata describes a runtime array of device-buffer addresses.
pub fn is_device_buffer_array_type_name(name: &str) -> bool {
    name.chars()
        .filter(|character| !character.is_whitespace())
        .eq("array_ref<void>".chars())
}

/// Role of a single fragment-shader entry parameter, keyed by its parameter index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragRole {
    /// `[[position]]` -> Input BuiltIn FragCoord (often unused).
    Position,
    /// `[[point_coord]]` -> Input BuiltIn PointCoord.
    PointCoord,
    /// `[[front_facing]]` -> Input BuiltIn FrontFacing.
    FrontFacing,
    /// `[[barycentric_coord]]` -> Input BuiltIn `BaryCoordKHR` / `BaryCoordNoPerspKHR` (float3).
    ///
    /// AIR states the perspective axis on the same node, exactly as it does for a varying, and
    /// SPIR-V spells the two as different builtins rather than as a decoration — so the flag has to
    /// travel with the role.
    BarycentricCoord { no_perspective: bool },
    /// `[[primitive_id]]` -> Input BuiltIn PrimitiveId (32-bit uint).
    PrimitiveId,
    /// `[[sample_id]]` -> Input BuiltIn SampleId (32-bit uint).
    SampleId,
    /// `[[sample_mask]]` on an *argument* -> Input BuiltIn `SampleMask`, the coverage the
    /// rasterizer produced for this fragment. AIR spells the input form `air.sample_mask_in` to
    /// distinguish it from the return member; SPIR-V uses one builtin in two storage classes.
    SampleMaskIn,
    /// `[[viewport_array_index]]` -> Input BuiltIn ViewportIndex (32-bit uint).
    ViewportArrayIndex,
    /// `[[render_target_array_index]]` -> Input BuiltIn Layer (32-bit uint).
    RenderTargetArrayIndex,
    /// `[[amplification_id]]` -> Input BuiltIn ViewIndex (32-bit uint).
    ///
    /// Metal vertex amplification and Vulkan multiview are the same feature: one draw rasterized
    /// into several views, with the shader told which view it is running for. Metal spells that
    /// index `[[amplification_id]]`; Vulkan spells it `gl_ViewIndex`. A pass with a single view
    /// answers zero on both sides, so the mapping holds whether or not multiview is enabled.
    AmplificationId,
    /// `[[amplification_count]]` -> the count the render encoder set, bound as a constant.
    ///
    /// The count is a property of the draw, not of the stage that reads it: the same
    /// `setVertexAmplificationCount:viewMappings:` answers a fragment and a vertex alike. Only the
    /// vertex role was modelled, so a fragment declaring it fell to [`FragRole::Other`] and the
    /// module was refused for a value the stage-input pass already knew how to bind.
    AmplificationCount,
    /// `[[stage_in]]` interpolated input -> Input var at Location N (N = order among fragment_inputs).
    Varying(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// A Metal visible-function table resolved during authored dependency linking.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table resolved during authored dependency linking.
    IntersectionFunctionTable(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[color(n)]]` framebuffer-fetch input -> Vulkan input attachment (out of scope this milestone).
    ColorInput(u32),
    /// A custom fragment `[[imageblock_data]]` projection. Its fields are mapped to the master
    /// imageblock layout by AIR user semantic, never by the source struct or argument name.
    ImageblockData,
    /// A fact about the execution group this fragment's thread runs in. Only the facts that need no
    /// threadgroup reach a fragment; see [`ExecutionGroupFact::needs_a_threadgroup`].
    ExecutionGroup {
        fact: ExecutionGroupFact,
        lanes: u32,
    },
    /// A texture argument the variant this module describes declares no slot for, so no descriptor
    /// exists for it. Metal reads zero through such a resource and stores nowhere; naming the role
    /// is what lets the lowering tell that operand apart from a handle it merely lost track of.
    /// See the internal `variant_texture_slot` helper.
    VariantAbsentTexture,
    /// Anything we don't model.
    Other,
}

/// Declared AIR access on a buffer argument. This is a conservative contract: it may be broader
/// than the specialized body actually uses, but it never understates reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Where inside a pixel AIR asked a `fragment_input` varying to be sampled.
///
/// Metal spells this as the first half of an interpolation attribute — `[[center_perspective]]`,
/// `[[centroid_no_perspective]]`, `[[sample_perspective]]` — and AIR emits it as its own marker
/// alongside the perspective marker. `air.center` is the default and needs no SPIR-V decoration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VaryingSampling {
    /// `air.center` — sampled at the pixel center. Vulkan's default; no decoration.
    #[default]
    Center,
    /// `air.centroid` — sampled inside the covered area of the primitive (`Centroid`).
    Centroid,
    /// `air.sample` — sampled per covered sample, which forces per-sample shading (`Sample`,
    /// capability `SampleRateShading`).
    Sample,
}

/// How AIR asked a `fragment_input` varying to be interpolated.
///
/// One record per varying rather than one set per qualifier: AIR states the whole interpolation
/// attribute on the argument node, and reading only the part the emitter happened to support is
/// how `air.no_perspective` and `air.centroid` were silently dropped for as long as only
/// `air.flat` was decoded. `interpolation_markers_are_decoded_or_deliberately_ignored` in
/// `src/meta/tests.rs` pins the full marker inventory against this record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaryingInterpolation {
    /// `air.flat` — not interpolated at all. AIR states this instead of, not alongside, the
    /// perspective/sampling pair, so it takes precedence over both fields below.
    pub flat: bool,
    /// `air.no_perspective` — interpolated linearly in screen space rather than
    /// perspective-correct (`NoPerspective`). Its complement `air.perspective` is Vulkan's
    /// default and needs no decoration.
    pub no_perspective: bool,
    /// Where in the pixel the interpolated value is taken from.
    pub sampling: VaryingSampling,
}

impl VaryingInterpolation {
    /// Decode the interpolation attribute from an argument node's `air.*` marker list.
    fn from_role_strings(strs: &[String]) -> Self {
        let has = |marker: &str| strs.iter().any(|s| s == marker);
        Self {
            flat: has("flat"),
            no_perspective: has("no_perspective"),
            sampling: if has("centroid") {
                VaryingSampling::Centroid
            } else if has("sample") {
                VaryingSampling::Sample
            } else {
                VaryingSampling::Center
            },
        }
    }
}

/// A fragment shader's decoded parameter roles + render-target count/indices.
#[derive(Clone, Debug, Default)]
pub struct FragMeta {
    /// `(param_idx, role)` — one per fragment input, in declaration order.
    pub roles: Vec<(u32, FragRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering.
    ///
    /// An unrecognised parameter is bound to a zero value so the body stays well formed. That is
    /// right for a function-constant-disabled resource, which Metal itself defines as absent, and
    /// wrong for anything else: a `[[barycentric_coord]]` the emitter does not model becomes a
    /// silent zero, and everything computed from it is wrong in a module that validates. Emission
    /// rejects on a non-empty list instead.
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Every attribute on the `!air.fragment` root this stage has no model for, as it reads in
    /// AIR. See [`FragMeta::early_fragment_tests`]: the root's attribute tail changes what the
    /// stage does, so an entry carrying an attribute nothing consumes must be refused rather than
    /// emitted as if the root had carried nothing.
    pub unmodelled_stage_attributes: Vec<String>,
    /// `[[early_fragment_tests]]`: depth and stencil testing happens before the fragment body.
    ///
    /// This is not an optimisation hint. A fragment the depth test rejects runs no part of the
    /// body, so none of its buffer, texture or imageblock stores happen; under the default late
    /// test the same shader performs every store and only its color output is discarded.
    pub early_fragment_tests: bool,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    /// Detected from the module's intrinsic calls, which is a property of the body rather than of
    /// the stage — the interface pass materializes the plane wherever it lowers one of those calls,
    /// so every stage that can carry them has to be able to report them.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// `fragment_input` Location -> AIR type name (`float2`, `float4`, ...). Used by passthrough
    /// vertex synthesis when the pipeline binds a built-in vertex slot.
    pub varying_types: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal field/argument name, when AIR metadata carries one. The
    /// Metal oracle uses this to generate a vertex struct that Apple's pipeline linker matches.
    pub varying_names: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal user semantic, such as `user(texturecoord)`, when AIR
    /// metadata carries one.
    pub varying_user_semantics: HashMap<u32, String>,
    /// `fragment_input` Location -> the interpolation attribute AIR declared for it. Absent means
    /// AIR said nothing, which is Vulkan's default (perspective-correct, pixel center).
    pub varying_interpolation: HashMap<u32, VaryingInterpolation>,
    /// number of `air.render_target` outputs (MRT count; 1 for the common single-output case).
    pub n_render_targets: u32,
    /// Return-struct member index -> color attachment Location for actual `air.render_target`
    /// outputs. Non-color outputs such as `air.stencil` are deliberately absent.
    pub render_target_members: Vec<(u32, u32)>,
    /// Return-struct member index -> AIR render-target type name (`float4`, `int4`, ...).
    pub render_target_type_names: HashMap<u32, String>,
    /// Return-struct member indices tagged as `air.depth` (`[[depth(...)]]`).
    pub depth_members: Vec<u32>,
    /// Conservative depth-test relation declared by `[[depth(...)]]`.
    pub depth_qualifier: Option<DepthQualifier>,
    /// Return-struct member indices tagged as `air.stencil` (`[[stencil]]`).
    pub stencil_members: Vec<u32>,
    /// Return-struct member indices tagged as `air.sample_mask` (`[[sample_mask]]`).
    ///
    /// The value is a coverage mask: a sample whose bit the shader clears is not written, which is
    /// how alpha-to-coverage and custom MSAA resolves are expressed. Vulkan spells it as the
    /// `SampleMask` builtin, an array of `uint` rather than the scalar Metal returns.
    pub sample_mask_members: Vec<u32>,
    /// `(member index, AIR role)` for every enabled return member whose role the emitter has no
    /// lowering for. See [`FRAGMENT_OUTPUT_ROLES`].
    pub unmodelled_output_members: Vec<(u32, String)>,
    /// Custom per-pixel fragment imageblock master plus the input/output projections that expose
    /// subsets of its fields. `None` when the fragment carries no `air.imageblock_master` contract.
    pub fragment_imageblock: Option<FragmentImageblock>,
    /// Render-target locations in AIR output metadata order. A single-output fragment can legally
    /// write a nonzero MRT slot, e.g. coverage shaders writing `[[color(1)]]`.
    pub render_target_indices: Vec<u32>,
    /// `param_idx -> reconstructed struct layout` for buffer args that carry `air.struct_type_info`.
    /// Used to rebuild the real struct when native emission represents the buffer as a bare pointer.
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR node carries it (device=1,
    /// constant=2). Populated only from `air.address_space` / the function param pointer address
    /// space — absent (not guessed) when the IR does not state it. Mirrors
    /// [`KernMeta::buffer_address_spaces`] for the fragment stage.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` (`air.arg_type_size` / `air.buffer_size`), when
    /// the AIR node carries it. Mirrors [`KernMeta::buffer_type_sizes`] for the fragment stage.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> air.buffer_size` for reference-like arguments whose metadata bounds the
    /// reachable object to exactly that many bytes. Kept separate from `buffer_type_sizes`, which
    /// may only be an unbounded pointer's element/pointee size.
    pub buffer_object_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name` for every stage.
    pub buffer_type_names: HashMap<u32, String>,
    /// `param_idx -> declared AIR read/write qualifier`.
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    /// Framebuffer-fetch color input Location -> AIR render-target type name, e.g. `float4`.
    pub color_input_type_names: HashMap<u32, String>,
    pub embedded_textures: Vec<EmbeddedTexture>,
    pub embedded_arguments: Vec<EmbeddedArgument>,
    /// Resource handles an argument buffer declares inside a nested `air.struct_type_info` member,
    /// which the embedded-argument walk does not surface. Non-empty means every texture operation on
    /// an unrecovered handle has to refuse rather than take the "resource is absent" path -- see
    /// `embedded::unsurfaced_embedded_resources`.
    pub unsurfaced_embedded_resources: Vec<String>,
}

/// One field in AIR's custom fragment-imageblock master layout.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockMember {
    pub offset: u32,
    pub size: u32,
    pub type_name: String,
    pub semantic: String,
    pub raster_order_group: u32,
}

/// One field exposed by an entry input or return-value imageblock projection.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockProjectionMember {
    pub projection_member: u32,
    pub master_member: u32,
}

/// A partial struct view of a custom fragment imageblock.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockProjection {
    /// Entry parameter index for an input projection, or return-struct member index for an output.
    pub interface_index: u32,
    pub members: Vec<FragmentImageblockProjectionMember>,
}

/// AIR's exact custom fragment-imageblock ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblock {
    pub sample_size: u32,
    pub members: Vec<FragmentImageblockMember>,
    pub inputs: Vec<FragmentImageblockProjection>,
    pub outputs: Vec<FragmentImageblockProjection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthQualifier {
    Any,
    Less,
    Greater,
}

impl FragMeta {
    pub fn role_of(&self, idx: u32) -> Option<&FragRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
    /// See [`Self::declared_descriptor_counts`].
    pub fn declared_descriptor_count(&self, idx: u32) -> Option<u32> {
        self.declared_descriptor_counts.get(&idx).copied()
    }
    pub fn color_input_type_name(&self, location: u32) -> Option<&str> {
        self.color_input_type_names
            .get(&location)
            .map(String::as_str)
    }
    pub fn varying_type(&self, loc: u32) -> Option<&str> {
        self.varying_types.get(&loc).map(String::as_str)
    }
    pub fn varying_name(&self, loc: u32) -> Option<&str> {
        self.varying_names.get(&loc).map(String::as_str)
    }
    pub fn varying_user_semantic(&self, loc: u32) -> Option<&str> {
        self.varying_user_semantics.get(&loc).map(String::as_str)
    }
    /// The interpolation attribute AIR declared for the varying at `loc`, defaulting to Vulkan's
    /// own default when AIR declared none.
    pub fn varying_interpolation(&self, loc: u32) -> VaryingInterpolation {
        self.varying_interpolation
            .get(&loc)
            .copied()
            .unwrap_or_default()
    }
    pub fn varying_is_flat(&self, loc: u32) -> bool {
        self.varying_interpolation(loc).flat
    }
    pub fn render_target_location_for_member(&self, member_idx: u32) -> Option<u32> {
        self.render_target_members
            .iter()
            .find_map(|(member, location)| (*member == member_idx).then_some(*location))
    }
    pub fn render_target_type_name(&self, member_idx: u32) -> Option<&str> {
        self.render_target_type_names
            .get(&member_idx)
            .map(String::as_str)
    }
    pub fn is_depth_member(&self, member_idx: u32) -> bool {
        self.depth_members.contains(&member_idx)
    }
    pub fn is_stencil_member(&self, member_idx: u32) -> bool {
        self.stencil_members.contains(&member_idx)
    }
    pub fn is_sample_mask_member(&self, member_idx: u32) -> bool {
        self.sample_mask_members.contains(&member_idx)
    }
}

/// Role of a single vertex-shader entry parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertRole {
    /// `[[stage_in]]` / `[[attribute(n)]]` vertex attribute -> Input var at Location N.
    VertexInput(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[texture(n)]]` -> sampled image (vertex texture fetch). WindowServer vertices read data via
    /// textures, so the vertex stage needs the same texture/sampler handling as the fragment stage.
    Texture(u32),
    /// `[[sampler(n)]]` -> sampler.
    Sampler(u32),
    /// A Metal visible-function table resolved during authored dependency linking.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table resolved during authored dependency linking.
    IntersectionFunctionTable(u32),
    /// `[[vertex_id]]` -> Input BuiltIn VertexIndex (32-bit uint).
    VertexId,
    /// `[[instance_id]]` -> Input BuiltIn InstanceIndex (32-bit uint).
    InstanceId,
    /// Opaque Metal patch handle consumed only by the metadata-named control-point accessor.
    PatchControlPoints,
    /// Per-patch user input at the AIR location.
    PatchInput(u32),
    /// `[[position_in_patch]]` -> the leading components of Vulkan `TessCoord`.
    PositionInPatch,
    /// `[[patch_id]]` -> Vulkan `PrimitiveId` in tessellation evaluation.
    PatchId,
    /// Metal vertex-amplification identifiers have no Vulkan tessellation builtin and are exposed
    /// through the translator's per-patch system-input locations.
    AmplificationId,
    AmplificationCount,
    /// A fact about the execution group this vertex's thread runs in. See
    /// [`FragRole::ExecutionGroup`].
    ExecutionGroup {
        fact: ExecutionGroupFact,
        lanes: u32,
    },
    /// A texture argument the variant this module describes declares no slot for, so no descriptor
    /// exists for it. Metal reads zero through such a resource and stores nowhere; naming the role
    /// is what lets the lowering tell that operand apart from a handle it merely lost track of.
    /// See the internal `variant_texture_slot` helper.
    VariantAbsentTexture,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PatchDomain {
    Triangle,
    Quad,
    Isoline,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchControlPointField {
    pub location: u32,
    pub type_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessellationMeta {
    pub domain: PatchDomain,
    pub control_point_count: u32,
    pub control_point_function: Option<String>,
    pub control_point_fields: Vec<PatchControlPointField>,
}

/// Role of a single vertex return-struct member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertOutRole {
    /// `[[position]]` -> Output BuiltIn Position.
    Position,
    /// `[[point_size]]` -> Output BuiltIn PointSize. Does not consume a Location.
    PointSize,
    /// `[[clip_distance]]` -> Output BuiltIn ClipDistance. Does not consume a Location.
    ClipDistance,
    /// `[[viewport_array_index]]` -> Output BuiltIn ViewportIndex. Does not consume a Location.
    ViewportArrayIndex,
    /// `[[render_target_array_index]]` -> Output BuiltIn Layer. Does not consume a Location.
    RenderTargetArrayIndex,
    /// User varying output -> Output var at Location N.
    Varying(u32),
    /// Function-constant-gated output disabled by the translator's default-zero FC model.
    FunctionConstantDisabled,
    Other,
}

/// A vertex shader's decoded parameter roles. The OUTPUT struct (member 0 = `[[position]]`, the rest
/// follows `output_roles`; point-size-style builtins do not consume varying locations.
#[derive(Clone, Debug, Default)]
pub struct VertMeta {
    pub roles: Vec<(u32, VertRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering.
    ///
    /// An unrecognised parameter is bound to a zero value so the body stays well formed. That is
    /// right for a function-constant-disabled resource, which Metal itself defines as absent, and
    /// wrong for anything else: a `[[barycentric_coord]]` the emitter does not model becomes a
    /// silent zero, and everything computed from it is wrong in a module that validates. Emission
    /// rejects on a non-empty list instead.
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    /// Detected from the module's intrinsic calls, which is a property of the body rather than of
    /// the stage — the interface pass materializes the plane wherever it lowers one of those calls,
    /// so every stage that can carry them has to be able to report them.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// Entry parameter index -> AIR type name. Tessellation system values use this to expose the
    /// exact cross-stage scalar type instead of forcing executors to infer it from a location.
    pub parameter_type_names: HashMap<u32, String>,
    pub output_roles: Vec<VertOutRole>,
    /// `(member index, AIR role)` for every output member decoded as [`VertOutRole::Other`].
    ///
    /// The vertex output walk gives an unrecognised member the next free user Location, so an
    /// unmodelled role does not vanish — it becomes a varying the pipeline never wired, at a
    /// Location it takes from a real one. Reporting it lets emission reject instead.
    pub unmodelled_output_members: Vec<(u32, String)>,
    /// Output member indices AIR marked `air.invariant` — Metal `[[position, invariant]]`.
    ///
    /// The guarantee is bit-exact: the same vertex fed through two pipelines that both declare it
    /// must produce the identical clip position, which is what lets a depth-prepass and the pass
    /// that tests against it agree instead of z-fighting. Vulkan spells it `OpDecorate … Invariant`.
    /// A translation that drops it stays valid and reflects identically, so nothing but this record
    /// carries the request across.
    pub invariant_outputs: Vec<u32>,
    /// User-varying output Location -> AIR type name.
    pub output_varying_types: HashMap<u32, String>,
    /// User-varying output Location -> Metal field name.
    pub output_varying_names: HashMap<u32, String>,
    /// User-varying output Location -> Metal linker semantic.
    pub output_varying_user_semantics: HashMap<u32, String>,
    /// Vertex input Location -> AIR type name (`float2`, `float4`, ...). Used by conformance
    /// oracles that must synthesize a Metal vertex descriptor before pipeline reflection exists.
    pub vertex_input_types: HashMap<u32, String>,
    /// Vertex input Location -> Metal argument name, when AIR metadata carries one.
    pub vertex_input_names: HashMap<u32, String>,
    /// Per-patch tessellation input Location -> AIR type name.
    pub patch_input_types: HashMap<u32, String>,
    /// Per-patch tessellation input Location -> Metal argument name.
    pub patch_input_names: HashMap<u32, String>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_address_spaces`]).
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_type_sizes`]).
    pub buffer_type_sizes: HashMap<u32, u32>,
    pub buffer_object_sizes: HashMap<u32, u32>,
    pub buffer_type_names: HashMap<u32, String>,
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    pub embedded_textures: Vec<EmbeddedTexture>,
    pub embedded_arguments: Vec<EmbeddedArgument>,
    /// Resource handles an argument buffer declares inside a nested `air.struct_type_info` member,
    /// which the embedded-argument walk does not surface. Non-empty means every texture operation on
    /// an unrecovered handle has to refuse rather than take the "resource is absent" path -- see
    /// `embedded::unsurfaced_embedded_resources`.
    pub unsurfaced_embedded_resources: Vec<String>,
    pub tessellation: Option<TessellationMeta>,
    /// Why an `air.patch` node the function does carry could not be decoded into
    /// [`VertMeta::tessellation`], if that happened.
    ///
    /// The two are not interchangeable with `tessellation: None`. A vertex function with no patch
    /// node is an ordinary vertex shader; one whose patch node did not decode is a post-tessellation
    /// evaluation shader whose domain, spacing, winding and per-patch inputs would all go missing,
    /// and the module that results is valid, binds, reflects, and draws the wrong geometry. The
    /// stage-input pass refuses on this rather than emitting it.
    pub undecoded_patch_shape: Option<String>,
    /// Every attribute on the `!air.vertex` root this stage has no model for, as it reads in AIR.
    /// Mirrors [`FragMeta::unmodelled_stage_attributes`].
    pub unmodelled_stage_attributes: Vec<String>,
}

impl VertMeta {
    pub fn is_tessellation_evaluation(&self) -> bool {
        self.tessellation.is_some()
    }

    pub fn tessellation_system_input_location(&self, role: &VertRole) -> Option<u32> {
        let base = self
            .roles
            .iter()
            .filter_map(|(_, role)| match role {
                VertRole::PatchInput(location) => Some(*location),
                _ => None,
            })
            .chain(
                self.tessellation
                    .iter()
                    .flat_map(|meta| meta.control_point_fields.iter().map(|field| field.location)),
            )
            .max()
            .map_or(0, |location| location + 1);
        match role {
            VertRole::InstanceId => Some(base),
            VertRole::AmplificationId => Some(base + 1),
            VertRole::AmplificationCount => Some(base + 2),
            _ => None,
        }
    }
}

impl VertMeta {
    pub fn role_of(&self, idx: u32) -> Option<&VertRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
    /// See [`Self::declared_descriptor_counts`].
    pub fn declared_descriptor_count(&self, idx: u32) -> Option<u32> {
        self.declared_descriptor_counts.get(&idx).copied()
    }
    pub fn output_role_of(&self, idx: u32) -> Option<&VertOutRole> {
        self.output_roles.get(idx as usize)
    }
    /// Whether AIR marked output member `idx` `air.invariant`.
    pub fn output_is_invariant(&self, idx: u32) -> bool {
        self.invariant_outputs.contains(&idx)
    }
    pub fn output_varying_type(&self, loc: u32) -> Option<&str> {
        self.output_varying_types.get(&loc).map(String::as_str)
    }
    pub fn output_varying_name(&self, loc: u32) -> Option<&str> {
        self.output_varying_names.get(&loc).map(String::as_str)
    }
    pub fn output_varying_user_semantic(&self, loc: u32) -> Option<&str> {
        self.output_varying_user_semantics
            .get(&loc)
            .map(String::as_str)
    }
    pub fn vertex_input_type(&self, loc: u32) -> Option<&str> {
        self.vertex_input_types.get(&loc).map(String::as_str)
    }
    pub fn vertex_input_name(&self, loc: u32) -> Option<&str> {
        self.vertex_input_names.get(&loc).map(String::as_str)
    }
}

/// Role of a single compute-kernel entry parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernRole {
    /// AIR `buffer` argument. `KernMeta::buffer_address_space` distinguishes device/constant
    /// resource buffers from threadgroup scratch buffers.
    Buffer(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// Host-populated StorageBuffer shadow for an opaque Metal acceleration structure used by AIR
    /// introspection intrinsics. Bound at the resource's `air.location_index`; see `as_shadow`.
    AccelerationStructureShadow(u32),
    /// An AIR primitive acceleration structure that is not consumed by an AIR intersection
    /// intrinsic. Metal still binds the native object; Vulkan needs no descriptor.
    PrimitiveAccelerationStructure(u32),
    /// An AIR primitive acceleration structure consumed by AIR intersection lowering. Metal binds
    /// the native object and Vulkan exposes authored triangle geometry through a StorageBuffer.
    PrimitiveAccelerationStructureShadow(u32),
    /// A Metal visible-function table. Logical SPIR-V has no descriptor for the opaque table;
    /// authored linking resolves its entries before ordinary interface lowering.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table. Like visible tables, this is a link-time authored
    /// resource rather than a Vulkan descriptor.
    IntersectionFunctionTable(u32),
    /// `[[threads_per_threadgroup]]` (`uint` or `uint3`) -> the size of the threadgroup this thread
    /// actually runs in. Scalar params receive component .x.
    ///
    /// This is the EXECUTING size, which a decomposed exact-thread dispatch makes smaller in a tail
    /// region: `KernelDispatchPlan` gives each region its own `LocalSize`, so the fact follows the
    /// region's specialized local size rather than the one the dispatch asked for.
    ThreadsPerThreadgroup,
    /// `[[dispatch_threads_per_threadgroup]]` (`uint` or `uint3`) -> the threadgroup size the
    /// dispatch ASKED FOR, which a short final threadgroup does not change.
    ///
    /// It is a separate role from [`KernRole::ThreadsPerThreadgroup`] because the two differ exactly
    /// where this translator emulates `dispatchThreads:`. A ten-thread grid in groups of four gives
    /// Metal `threads_per_threadgroup == 2` and `dispatch_threads_per_threadgroup == 4` in the last
    /// group, and `KernelDispatchPlan` reproduces that split by specializing the tail region's local
    /// size to two -- so aliasing the two roles reports the tail size for both. The requested size is
    /// `TransformOptions::kernel_local_size`, which is the same value the plan decomposes.
    DispatchThreadsPerThreadgroup,
    /// `[[thread_position_in_threadgroup]]` (`uint` or `uint3`) -> LocalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInThreadgroup,
    /// `[[threadgroups_per_grid]]` (`uint` or `uint3`) -> NumWorkgroups.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupsPerGrid,
    /// `[[threads_per_grid]]` (`uint` or `uint3`) -> the selected kernel dispatch grid. Whole-
    /// workgroup dispatches derive it as NumWorkgroups * LocalSize; exact-thread dispatches read
    /// the complete logical grid from the region payload.
    ThreadsPerGrid,
    /// `[[threadgroup_position_in_grid]]` (`uint` or `uint3`) -> WorkgroupId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupPositionInGrid,
    /// `[[thread_index_in_threadgroup]]` (`uint`) -> LocalInvocationIndex.
    ThreadIndexInThreadgroup,
    /// One fact about the AIR execution group this thread runs in, over a group `lanes` threads
    /// wide. See [`ExecutionGroupFact`] and [`AIR_EXECUTION_GROUPS`].
    ExecutionGroup {
        fact: ExecutionGroupFact,
        lanes: u32,
    },
    /// `[[thread_position_in_grid]]` (`uint` or `uint3`) -> GlobalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInGrid,
    /// Kernel `[[stage_in]]` attribute data. Metal feeds this through a stage-input descriptor keyed
    /// by `air.location_index`; Vulkan lowering needs an explicit per-invocation data ABI and must
    /// not silently bind it to zero.
    StageInput(u32),
    /// A texture argument the variant this module describes declares no slot for, so no descriptor
    /// exists for it. Metal reads zero through such a resource and stores nowhere; naming the role
    /// is what lets the lowering tell that operand apart from a handle it merely lost track of.
    /// See the internal `variant_texture_slot` helper.
    VariantAbsentTexture,
    Other,
}

/// A compute kernel's decoded parameter roles. The body has no return value (a `void` kernel that
/// writes through its buffer pointers), so no output handling is needed.
#[derive(Clone, Debug, Default)]
pub struct KernMeta {
    pub roles: Vec<(u32, KernRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering. See [`FragMeta::unmodelled_input_params`].
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Parameter indices of explicit imageblocks whose AIR node carries
    /// `air.alias_implicit_imageblock`.
    ///
    /// An explicit imageblock is ordinarily tile-local scratch, and the emitter gives it Private
    /// storage. This marker says the opposite: the storage *is* the implicit imageblock -- the
    /// render targets the rasterizer already wrote -- so the kernel's first read is of framebuffer
    /// content, and its writes have to land back there. Private scratch is neither, so the marker
    /// cannot be dropped.
    pub aliased_implicit_imageblock_params: Vec<u32>,
    /// The render-target planes an aliased explicit imageblock's members resolve to, keyed by the
    /// same parameter index. One derivation, read by reflection (which must declare a descriptor
    /// for every plane) and by the emitter (which must fill the cell from them and write it back).
    pub aliased_implicit_imageblock_planes: HashMap<u32, Vec<AliasedImageblockPlane>>,
    /// Every attribute on the `!air.kernel` root this stage has no model for, as it reads in AIR.
    /// Mirrors [`FragMeta::unmodelled_stage_attributes`].
    pub unmodelled_stage_attributes: Vec<String>,
    /// `[[max_total_threads_per_threadgroup(N)]]`: the largest threadgroup the entry was compiled
    /// to run. A dispatch wider than this is outside what the AIR body was built for, so the
    /// requested `LocalSize` is checked against it rather than emitted unread.
    pub max_work_group_size: Option<u32>,
    /// Function-constant-wrapped buffer parameter index -> Metal buffer location. Multiple mutually
    /// exclusive typed alternatives may intentionally share one location.
    pub function_constant_buffer_locations: HashMap<u32, u32>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> reconstructed air.imageblock_data layout` for imageblock args.
    pub imageblock_layouts: HashMap<u32, AirType>,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// `param_idx -> AIR address space` for buffer args. Address space 3 is threadgroup memory.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer argument byte size`, from `air.arg_type_size` or
    /// `air.buffer_size` when present.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> air.buffer_size` when AIR declares one exact reference-object extent.
    pub buffer_object_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name`, e.g. `char`, `void`, or a struct name.
    pub buffer_type_names: HashMap<u32, String>,
    /// `param_idx -> declared AIR read/write qualifier`.
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    /// `param_idx -> AIR kernel stage-input scalar/vector type name`.
    pub stage_input_type_names: HashMap<u32, String>,
    /// Textures EMBEDDED inside an `air.indirect_buffer` argument buffer (via `air.indirect_argument`
    /// → nested `air.texture`) that the kernel body reads/writes with AIR texture intrinsics. Each is
    /// surfaced as a standalone image resource so the read/write lowers to a real descriptor instead of a
    /// private placeholder. See [`EmbeddedTexture`].
    pub embedded_textures: Vec<EmbeddedTexture>,
    /// Every resource-handle member carried by an `air.indirect_buffer`, including handles whose
    /// concrete kind is supplied by an authored manifest and verified from its AIR use.
    pub embedded_arguments: Vec<EmbeddedArgument>,
    /// Resource handles an argument buffer declares inside a nested `air.struct_type_info` member,
    /// which the embedded-argument walk does not surface. Non-empty means every texture operation on
    /// an unrecovered handle has to refuse rather than take the "resource is absent" path -- see
    /// `embedded::unsurfaced_embedded_resources`.
    pub unsurfaced_embedded_resources: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImplicitImageblockAttachment {
    pub attachment: u32,
    pub data_rate: u32,
    pub max_index: Option<u32>,
    pub format: TextureFormat,
    pub reads: bool,
    pub writes: bool,
}

/// One member of an explicit imageblock that `air.alias_implicit_imageblock` aliases onto the
/// implicit imageblock, resolved to the render-target plane it names.
///
/// The alias says the explicit layout IS the implicit layout, and the implicit layout is the colour
/// attachments in order, so member `i` in declaration order is attachment `i`. Nothing else in AIR
/// states the mapping: `air.struct_type_info` gives each member a source name (`color`, `depth`,
/// `customParams`), and those are names the shader author chose, not attachment indices -- a member
/// typed and named `depth` is still colour attachment 2 when it is declared third.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AliasedImageblockPlane {
    /// Colour attachment index, which is the member's position in declaration order.
    pub attachment: u32,
    /// The member's byte offset inside the imageblock cell.
    pub offset: u32,
    pub format: TextureFormat,
    /// The `air.load/store.implicit_imageblock.<suffix>` family that carries this texel type.
    pub intrinsic_suffix: &'static str,
}

/// Resolve an aliased explicit imageblock's members to the render-target planes they alias, or
/// state why they cannot be resolved.
///
/// `Ok(None)` means the parameter declares no member layout at all. An unsupported member type is
/// an error rather than a silently dropped plane: dropping one would stage that member in
/// threadgroup memory while its siblings reached the attachments, which is the exact silence the
/// alias marker exists to prevent.
pub fn aliased_imageblock_planes(
    layout: &AirType,
) -> Result<Option<Vec<AliasedImageblockPlane>>, String> {
    let AirType::Struct(members) = layout else {
        return Ok(None);
    };
    if members.is_empty() {
        return Ok(None);
    }
    let mut planes = Vec::with_capacity(members.len());
    for (index, member) in members.iter().enumerate() {
        let (format, intrinsic_suffix) = match &member.ty {
            AirType::Scalar(AirScalar::Half) => (TextureFormat::R16f, "f16"),
            AirType::Vec {
                scalar: AirScalar::Half,
                lanes: 2,
            } => (TextureFormat::Rg16f, "v2f16"),
            AirType::Vec {
                scalar: AirScalar::Half,
                lanes: 4,
            } => (TextureFormat::Rgba16f, "v4f16"),
            AirType::Scalar(AirScalar::Float) => (TextureFormat::R32f, "f32"),
            AirType::Vec {
                scalar: AirScalar::Float,
                lanes: 4,
            } => (TextureFormat::Rgba32f, "v4f32"),
            AirType::Scalar(AirScalar::UInt) => (TextureFormat::R32ui, "i32"),
            other => {
                return Err(format!(
                    "aliased imageblock member {index} has type {other:?}, which no implicit \
                     imageblock plane format represents"
                ))
            }
        };
        planes.push(AliasedImageblockPlane {
            attachment: index as u32,
            offset: member.offset,
            format,
            intrinsic_suffix,
        });
    }
    Ok(Some(planes))
}

impl KernRole {
    /// The Metal BUFFER-TABLE slot this role occupies, or `None` for a role that occupies none.
    ///
    /// This is the set [`KernMeta::stage_input_bindings`] must not allocate a synthetic slot on top
    /// of. Written as an exhaustive match rather than a list with a `_ => None` tail: a role added
    /// later that names a buffer-table index has to answer here, and the cost of forgetting is a
    /// synthetic stage-input descriptor decorated with a real buffer's binding.
    ///
    /// Deliberately conservative at both ends. The function-table and acceleration-structure roles
    /// reserve their slot even where they reflect with no descriptor of their own, because they
    /// still name a Metal buffer-table index the pipeline binds; reserving one nothing reads costs a
    /// synthetic slot, sharing one costs a wrong read. `StageInput` answers `None` on purpose --
    /// its own `u32` is the AIR attribute location, which is what the synthetic slot exists to
    /// stand in for.
    pub fn buffer_table_slot(&self) -> Option<u32> {
        match self {
            Self::Buffer(slot)
            | Self::AccelerationStructureShadow(slot)
            | Self::PrimitiveAccelerationStructure(slot)
            | Self::PrimitiveAccelerationStructureShadow(slot)
            | Self::VisibleFunctionTable(slot)
            | Self::IntersectionFunctionTable(slot) => Some(*slot),
            Self::Texture(_)
            | Self::Sampler(_)
            | Self::StageInput(_)
            | Self::ThreadsPerThreadgroup
            | Self::DispatchThreadsPerThreadgroup
            | Self::ThreadPositionInThreadgroup
            | Self::ThreadgroupsPerGrid
            | Self::ThreadsPerGrid
            | Self::ThreadgroupPositionInGrid
            | Self::ThreadIndexInThreadgroup
            | Self::ExecutionGroup { .. }
            | Self::ThreadPositionInGrid
            | Self::VariantAbsentTexture
            | Self::Other => None,
        }
    }
}

impl KernMeta {
    pub fn role_of(&self, idx: u32) -> Option<&KernRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn imageblock_layout_of(&self, idx: u32) -> Option<&AirType> {
        self.imageblock_layouts.get(&idx)
    }
    pub fn buffer_address_space(&self, idx: u32) -> Option<u32> {
        self.buffer_address_spaces.get(&idx).copied()
    }
    pub fn buffer_type_size(&self, idx: u32) -> Option<u32> {
        self.buffer_type_sizes.get(&idx).copied()
    }
    pub fn buffer_type_name(&self, idx: u32) -> Option<&str> {
        self.buffer_type_names.get(&idx).map(String::as_str)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
    /// See [`Self::declared_descriptor_counts`].
    pub fn declared_descriptor_count(&self, idx: u32) -> Option<u32> {
        self.declared_descriptor_counts.get(&idx).copied()
    }

    /// Synthetic buffer slots used for kernel `[[stage_in]]` attributes.
    ///
    /// Metal supplies stage-input data through ordinary buffer-table slots selected by the pipeline
    /// descriptor. Vulkan exposes the same arrays as read-only storage buffers. Keeping allocation
    /// here makes reflection and lowering consume one ABI decision instead of duplicating it.
    ///
    /// Allocation walks the stage inputs in PARAMETER-INDEX order, not in the order the AIR argument
    /// list happens to name them. The two coincide in all 14553 kernel argument lists of the local
    /// corpus, so ordering them costs nothing today; what it buys is that this ABI is a function of
    /// the entry's signature rather than of a metadata layout no other decision depends on. A
    /// producer that emitted the same kernel with its argument nodes reordered would otherwise hand
    /// a consumer different slots for the same shader.
    pub fn stage_input_bindings(&self) -> HashMap<u32, u32> {
        let mut occupied = self
            .roles
            .iter()
            .filter_map(|(_, role)| role.buffer_table_slot())
            .collect::<std::collections::HashSet<_>>();
        let mut stage_inputs = self
            .roles
            .iter()
            .filter(|(_, role)| matches!(role, KernRole::StageInput(_)))
            .map(|(param_index, _)| *param_index)
            .collect::<Vec<_>>();
        stage_inputs.sort_unstable();
        let mut next = 0u32;
        let mut bindings = HashMap::new();
        for param_index in stage_inputs {
            while occupied.contains(&next) {
                next = next.saturating_add(1);
            }
            occupied.insert(next);
            bindings.insert(param_index, next);
        }
        bindings
    }
}

/// Parse `!air.kernel` into a `KernMeta`. Structure (RE'd from the compute fixtures):
///   `!air.kernel = !{!N}`; `!N = !{ptr @k, !EMPTY, !IN}`;
///   `!IN = !{!a, !b, ...}` each `!{i32 idx, !"air.buffer"|"air.texture"|..., ...}`.
pub fn parse_air_kernel_meta(ll: &str) -> Option<KernMeta> {
    parse_air_kernel_meta_with(ll, false)
}

/// Like [`parse_air_kernel_meta`], but with `promote_fc_buffers` controlling whether a
/// `[[function_constant]]`-gated `air.buffer` param is classified as a REAL StorageBuffer binding
/// (`true`) or left as the default possibly-absent Private placeholder (`false`). Every production
/// entry point uses `false`; only the adopt-if-validates `fc_promote_psb` retry passes `true` (see
/// the internal `fc_promoted_role` classifier). Standalone callers can request either projection;
/// production parses both
/// projections from one shared metadata-node table before the transform pipeline runs.
pub fn parse_air_kernel_meta_with(ll: &str, promote_fc_buffers: bool) -> Option<KernMeta> {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "kernel", &nodes);
    parse_air_kernel_meta_with_nodes(ll, promote_fc_buffers, &nodes, entry.as_deref())
}

/// Parse the default and FC-buffer-promoted kernel projections from one metadata-node table. The
/// retry cascade needs both projections, but their only difference is how a stable
/// `air.function_constant` wrapper classifies a wrapped buffer; collecting and decoding the AIR
/// metadata twice is unnecessary.
pub(crate) fn parse_air_kernel_meta_variants(
    ll: &str,
) -> (Option<KernMeta>, Option<KernMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "kernel", &nodes);
    let default = parse_air_kernel_meta_with_nodes(ll, false, &nodes, entry.as_deref());
    let promoted = parse_air_kernel_meta_with_nodes(ll, true, &nodes, entry.as_deref());
    (default, promoted, entry)
}

fn parse_air_kernel_meta_with_nodes(
    ll: &str,
    promote_fc_buffers: bool,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<KernMeta> {
    let root = stage_root(ll, "kernel")?;
    let rootc = nodes.get(&root)?;
    let static_int_globals = static_init_int_global_values(ll);
    let resource_location =
        |node: &str, fallback: u32| location_index_with_static(node, fallback, &static_int_globals);
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let refs = refs_in(rootc);
    // The argument-info list is the SECOND ref (`!EMPTY` is the first — an empty placeholder node).
    let in_ref = *refs.get(1)?;
    let mut max_work_group_size = None;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::MaxWorkGroupSize(size) => max_work_group_size = Some(size),
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut aliased_implicit_imageblock_params: Vec<u32> = vec![];
    let mut function_constant_buffer_locations = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut imageblock_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut stage_input_type_names = HashMap::new();
    // `air.location_index` of every top-level `air.texture` arg — the basis for the synthetic
    // embedded-texture index K (see `embedded_synthetic_texture_index`).
    let mut top_level_texture_locations: Vec<u32> = vec![];
    // `(buffer_param_index, struct_type_info_node_ref)` for each `air.indirect_buffer` arg, so
    // embedded-texture detection can run once K is known (after the whole arg list is scanned).
    let mut indirect_buffer_struct_refs: Vec<(u32, u32, u32)> = vec![];
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        let layout = struct_info_ref(node).and_then(|sref| parse_struct_info(nodes, sref, 0));
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        if strs.first().map(String::as_str) == Some("function_constant")
            && primary_role(&strs) == Some("buffer")
        {
            function_constant_buffer_locations.insert(idx, resource_location(node, idx));
        }
        let Some(mut first) = fc_promoted_role(&strs, promote_fc_buffers) else {
            continue;
        };
        // The one demotion the promotion set cannot state: a wrapped texture this variant declares
        // no slot for has no binding to promote it to. See [`variant_texture_slot`].
        let texture_slot = variant_texture_slot(node, idx, nodes, &static_int_globals);
        if primary_role(&strs) == Some("texture") && texture_slot.is_none() {
            first = VARIANT_ABSENT_TEXTURE_ROLE;
        }
        let first = present_gated_buffer_role(first, &strs, node, nodes, &static_int_globals);
        let first = present_system_value_role(first, node, nodes, &static_int_globals);
        if let Some(declared) = unmodelled_declared_role(
            |role| air_input_role_is_modelled(Stage::Kernel, role),
            node,
            nodes,
            &static_int_globals,
        ) {
            unmodelled_input_params.push((idx, declared));
        }
        let role = match first {
            "buffer" | "indirect_buffer" => {
                if first == "indirect_buffer" {
                    if let Some(sref) = struct_info_ref(node) {
                        // Key by the buffer's `air.location_index` (the Metal `[[buffer(N)]]` slot the
                        // harness binds), NOT the AIR argument position — they differ (e.g. arg 2 but
                        // buffer(0)). The oracle/runner both index buffers by location.
                        indirect_buffer_struct_refs.push((idx, resource_location(node, idx), sref));
                    }
                }
                if let Some(t) = layout.clone() {
                    buffer_layouts.insert(idx, t);
                }
                buffer_address_spaces.insert(
                    idx,
                    address_space(node)
                        .or_else(|| param_address_spaces.get(&idx).copied())
                        .unwrap_or(1),
                );
                if let Some(name) = arg_type_name(node) {
                    buffer_type_names.insert(idx, name);
                }
                if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                    .or_else(|| i32_after_marker(node, "air.buffer_size"))
                {
                    buffer_type_sizes.insert(idx, size);
                }
                if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                    buffer_object_sizes.insert(idx, size);
                }
                if let Some(access) = declared_buffer_access(node) {
                    buffer_accesses.insert(idx, access);
                }
                KernRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
            }
            "texture" => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                let loc = texture_slot.unwrap_or_else(|| resource_location(node, idx));
                top_level_texture_locations.push(loc);
                KernRole::Texture(loc)
            }
            "instance_acceleration_structure" if body_uses_acceleration_structure_shadow(ll) => {
                KernRole::AccelerationStructureShadow(resource_location(node, idx))
            }
            "primitive_acceleration_structure" => {
                let binding = resource_location(node, idx);
                if ll.contains("@air.intersect.") {
                    KernRole::PrimitiveAccelerationStructureShadow(binding)
                } else {
                    KernRole::PrimitiveAccelerationStructure(binding)
                }
            }
            "visible_function_table" => {
                KernRole::VisibleFunctionTable(resource_location(node, idx))
            }
            "intersection_function_table" => {
                KernRole::IntersectionFunctionTable(resource_location(node, idx))
            }
            "imageblock" => {
                if let Some(t) = layout {
                    imageblock_layouts.insert(idx, t);
                }
                if strs.iter().any(|s| s == "alias_implicit_imageblock") {
                    aliased_implicit_imageblock_params.push(idx);
                }
                KernRole::Other
            }
            "sampler" => KernRole::Sampler(resource_location(node, idx)),
            "threads_per_threadgroup" => KernRole::ThreadsPerThreadgroup,
            "dispatch_threads_per_threadgroup" => KernRole::DispatchThreadsPerThreadgroup,
            "thread_position_in_threadgroup" => KernRole::ThreadPositionInThreadgroup,
            "threadgroups_per_grid" => KernRole::ThreadgroupsPerGrid,
            "threads_per_grid" => KernRole::ThreadsPerGrid,
            "threadgroup_position_in_grid" => KernRole::ThreadgroupPositionInGrid,
            "thread_index_in_threadgroup" => KernRole::ThreadIndexInThreadgroup,
            "thread_position_in_grid" => KernRole::ThreadPositionInGrid,
            "stage_in" => {
                if let Some(name) = arg_type_name(node) {
                    stage_input_type_names.insert(idx, name);
                }
                KernRole::StageInput(resource_location(node, idx))
            }
            VARIANT_ABSENT_TEXTURE_ROLE => KernRole::VariantAbsentTexture,
            // The specializer's spelling of the same fact. Once a VALUE is supplied,
            // `specialize_function_constant_metadata` rewrites the resolved-off wrapper, and past
            // that rewrite the wrapped role is unreadable -- so the demotion above never runs and
            // this is the only place the fact survives.
            _ if declares_disabled_texture(&strs) => KernRole::VariantAbsentTexture,
            other => stage_execution_group_role(Stage::Kernel, other).map_or(
                KernRole::Other,
                |(fact, lanes)| KernRole::ExecutionGroup { fact, lanes },
            ),
        };
        roles.push((idx, role));
    }
    // Detect argument-buffer-embedded textures that the body uses through AIR texture
    // intrinsics. Gated purely on AIR structure/semantics — the `air.indirect_argument` →
    // `air.texture` marker chain plus stable AIR intrinsic families — so it cannot key on any shader
    // name. The body must actually use a texture intrinsic for us to surface it.
    let argument_buffers = ArgumentBuffers::new(indirect_buffer_struct_refs);
    let embedded_textures = if body_uses_texture_intrinsic(ll) {
        detect_embedded_textures(nodes, &argument_buffers, &top_level_texture_locations)
    } else {
        vec![]
    };
    let embedded_arguments = detect_embedded_arguments(nodes, &argument_buffers);
    let unsurfaced = unsurfaced_embedded_resources(nodes, &argument_buffers);
    let mut implicit_imageblock_attachments = detect_implicit_imageblock_attachments(ll)?;
    // An aliased explicit imageblock reaches the same planes the implicit intrinsics do, and the
    // module the emitter builds for it loads and stores every one. Reflection has to declare those
    // descriptors from here or a consumer builds a set layout missing the bindings the module it
    // was handed reads.
    let mut aliased_implicit_imageblock_planes = HashMap::new();
    for param in &aliased_implicit_imageblock_params {
        let Some(layout) = imageblock_layouts.get(param) else {
            continue;
        };
        let Some(planes) = aliased_imageblock_planes(layout).ok().flatten() else {
            continue;
        };
        for plane in &planes {
            implicit_imageblock_attachments.push(ImplicitImageblockAttachment {
                attachment: plane.attachment,
                data_rate: 0,
                max_index: Some(0),
                format: plane.format,
                reads: true,
                writes: true,
            });
        }
        aliased_implicit_imageblock_planes.insert(*param, planes);
    }
    Some(KernMeta {
        roles,
        unmodelled_input_params,
        aliased_implicit_imageblock_params,
        aliased_implicit_imageblock_planes,
        unmodelled_stage_attributes,
        max_work_group_size,
        function_constant_buffer_locations,
        buffer_layouts,
        imageblock_layouts,
        implicit_imageblock_attachments,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        stage_input_type_names,
        embedded_textures,
        embedded_arguments,
        unsurfaced_embedded_resources: unsurfaced,
    })
}

/// Decode the stable AIR implicit-imageblock intrinsic suffix to its exact storage plane format.
/// `Ok(None)` means the symbol is not in this intrinsic family; an unknown family suffix is an
/// explicit error so reflection and corpus capability audits cannot silently omit a new ABI shape.
pub fn implicit_imageblock_texture_format(name: &str) -> Result<Option<TextureFormat>, String> {
    let suffix = name
        .strip_prefix("air.load.implicit_imageblock.")
        .or_else(|| name.strip_prefix("air.store.implicit_imageblock."));
    let Some(suffix) = suffix else {
        return Ok(None);
    };
    let format = match suffix {
        "f16" => TextureFormat::R16f,
        "v2f16" => TextureFormat::Rg16f,
        "v4f16" => TextureFormat::Rgba16f,
        "f32" => TextureFormat::R32f,
        "v4f32" => TextureFormat::Rgba32f,
        "i32" => TextureFormat::R32ui,
        _ => {
            return Err(format!(
                "{name} has unsupported implicit imageblock texel type"
            ))
        }
    };
    Ok(Some(format))
}

fn detect_implicit_imageblock_attachments(ll: &str) -> Option<Vec<ImplicitImageblockAttachment>> {
    let mut attachments =
        std::collections::BTreeMap::<(u32, u32, TextureFormat), ImplicitImageblockAttachment>::new(
        );
    for line in ll.lines() {
        let Some(at) = line.find("@air.") else {
            continue;
        };
        let call = &line[at + 1..];
        let Some(open) = call.find('(') else { continue };
        let name = &call[..open];
        let (reads, writes, value_prefix) = if name.starts_with("air.load.implicit_imageblock.") {
            (true, false, 0usize)
        } else if name.starts_with("air.store.implicit_imageblock.") {
            (false, true, 1usize)
        } else {
            continue;
        };
        let Some(close) = call[open + 1..].find(')') else {
            continue;
        };
        let args = split_top_level_commas(&call[open + 1..open + 1 + close]);
        let Some(attachment) = args
            .get(value_prefix)
            .and_then(|arg| typed_u32_constant(arg))
        else {
            continue;
        };
        let index = args
            .get(value_prefix + 2)
            .and_then(|arg| typed_u32_constant(arg));
        let Some(data_rate) = args
            .get(value_prefix + 3)
            .and_then(|arg| typed_u32_constant(arg))
        else {
            continue;
        };
        let format = implicit_imageblock_texture_format(name).ok().flatten()?;
        let entry = attachments
            .entry((attachment, data_rate, format))
            .or_insert(ImplicitImageblockAttachment {
                attachment,
                data_rate,
                max_index: index,
                format,
                reads: false,
                writes: false,
            });
        entry.reads |= reads;
        entry.writes |= writes;
        entry.max_index = match (entry.max_index, index) {
            (Some(left), Some(right)) => Some(left.max(right)),
            _ => None,
        };
    }
    Some(attachments.into_values().collect())
}

fn typed_u32_constant(value: &str) -> Option<u32> {
    value.split_whitespace().last()?.parse().ok()
}

fn body_uses_acceleration_structure_shadow(ll: &str) -> bool {
    ll.contains("@air.get_instance_count_instance_acceleration_structure")
        || ll.contains("@air.get_primitive_acceleration_structure_instance_acceleration_structure")
        || ll.lines().any(|line| {
            let Some(start) = line.find("@air.intersect.") else {
                return false;
            };
            let Some(end) = line[start + 1..].find('(') else {
                return false;
            };
            let callee = &line[start + 1..start + 1 + end];
            AirIntersectionFamily::parse(callee)
                .ok()
                .flatten()
                .is_some_and(|family| family.instancing != AirIntersectionInstancing::None)
        })
}

/// Whether every AIR intersection call in this module has an implemented structural lowering.
///
/// Validation tooling uses the same product-owned decision as translation so its authorability
/// inventory cannot drift into a second, independently maintained intrinsic allowlist.
pub fn air_intersection_calls_are_supported(ll: &str) -> bool {
    crate::native::ray_intersection::all_air_intersection_calls_are_lowerable(ll)
}

/// Collect every `!N = !{...}` metadata node body, keyed by N. Shared by both stage parsers.
fn collect_nodes(ll: &str) -> HashMap<u32, String> {
    #[cfg(test)]
    AIR_META_PARSE_COUNT.with(|count| count.set(count.get() + 1));
    let mut nodes = HashMap::new();
    for l in ll.lines() {
        let l = l.trim();
        let Some(rest) = l.strip_prefix('!') else {
            continue;
        };
        // expect "<digits> = !{<body>}" or "<digits> = distinct !{<body>}"
        let Some((eq, prefix_len)) =
            rest.find(" = !{")
                .map(|eq| (eq, " = !{".len()))
                .or_else(|| {
                    rest.find(" = distinct !{")
                        .map(|eq| (eq, " = distinct !{".len()))
                })
        else {
            continue;
        };
        let Ok(id) = rest[..eq].parse::<u32>() else {
            continue;
        };
        let body = &rest[eq + prefix_len..];
        let body = body.strip_suffix('}').unwrap_or(body);
        nodes.insert(id, body.to_string());
    }
    nodes
}

#[cfg(test)]
thread_local! {
    static AIR_META_PARSE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_air_meta_parse_count() {
    AIR_META_PARSE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn air_meta_parse_count() -> usize {
    AIR_META_PARSE_COUNT.with(std::cell::Cell::get)
}

/// The entry function NAME from `!air.<stage> = !{!N}; !N = !{ptr @<name>, ...}`. The Vulkan backend
/// does NOT inline helpers, so a module has many functions; this names the real entry.
pub fn entry_name(ll: &str, stage: &str) -> Option<String> {
    let nodes = collect_nodes(ll);
    entry_name_from_nodes(ll, stage, &nodes)
}

fn entry_name_from_nodes(ll: &str, stage: &str, nodes: &HashMap<u32, String>) -> Option<String> {
    let root = stage_root(ll, stage)?;
    let body = nodes.get(&root)?;
    // body like: `ptr @BlurComposite, !16, !18` or `ptr @"re::df::pack", !16, !18`.
    let at = body.find('@')?;
    let after = &body[at + 1..];
    let name = if let Some(quoted) = after.strip_prefix('"') {
        quoted_symbol_name(quoted)?
    } else {
        after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
            .collect()
    };
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn pointer_symbol(body: &str) -> Option<String> {
    let at = body.find('@')?;
    let after = &body[at + 1..];
    let name = if let Some(quoted) = after.strip_prefix('"') {
        quoted_symbol_name(quoted)?
    } else {
        after
            .chars()
            .take_while(|c| c.is_alphanumeric() || matches!(*c, '_' | '.' | '$'))
            .collect()
    };
    (!name.is_empty()).then_some(name)
}

fn quoted_symbol_name(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some(out),
            '\\' => {
                let hi = chars.peek().copied();
                let mut clone = chars.clone();
                let lo = {
                    clone.next();
                    clone.peek().copied()
                };
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                        chars.next();
                        chars.next();
                        let byte = u8::from_str_radix(&format!("{hi}{lo}"), 16).ok()?;
                        out.push(byte as char);
                        continue;
                    }
                }
                out.push(chars.next().unwrap_or('\\'));
            }
            _ => out.push(ch),
        }
    }
    None
}

fn function_param_pointer_address_spaces(ll: &str, name: &str) -> Option<HashMap<u32, u32>> {
    let params = function_param_list(ll, name)?;
    let mut out = HashMap::new();
    for (idx, param) in split_top_level_commas(&params).into_iter().enumerate() {
        let param = param.trim_start();
        if !param.starts_with("ptr") {
            continue;
        }
        if let Some(addrspace) = param.find("addrspace(").and_then(|pos| {
            let after = &param[pos + "addrspace(".len()..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<u32>().ok()
        }) {
            out.insert(idx as u32, addrspace);
        }
    }
    Some(out)
}

fn function_param_list(ll: &str, name: &str) -> Option<String> {
    let unquoted = format!("@{name}(");
    let quoted = format!("@\"{name}\"(");
    let (start, needle_len) = ll
        .find(&unquoted)
        .map(|start| (start, unquoted.len()))
        .or_else(|| ll.find(&quoted).map(|start| (start, quoted.len())))?;
    let start = start + needle_len;
    let mut depth = 1u32;
    let mut end = start;
    for (off, ch) in ll[start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    end = start + off;
                    break;
                }
            }
            _ => {}
        }
    }
    (depth == 0).then(|| ll[start..end].to_string())
}

fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut start = 0usize;
    let mut angle_depth = 0i32;
    let mut paren_depth = 0i32;
    let mut brace_depth = 0i32;
    for (idx, ch) in s.char_indices() {
        match ch {
            '<' => angle_depth += 1,
            '>' => angle_depth -= 1,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            ',' if angle_depth == 0 && paren_depth == 0 && brace_depth == 0 => {
                items.push(s[start..idx].trim().to_string());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        items.push(tail.to_string());
    }
    items
}

/// Find the single operand of `!air.<stage> = !{!N}` and return N.
fn stage_root(ll: &str, stage: &str) -> Option<u32> {
    let needle = format!("!air.{stage} = !{{!");
    for l in ll.lines() {
        let l = l.trim();
        if let Some(rest) = l.strip_prefix(&needle) {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            return digits.parse().ok();
        }
    }
    None
}

/// One per-entry attribute carried by a stage root node, past its
/// `(function, outputs, inputs)` operand triple.
///
/// All three roots -- `!air.kernel`, `!air.vertex`, `!air.fragment` -- grow their entry
/// attributes in that same tail, but not in the same form: some are references to a keyed node
/// (`!{!"air.patch", ...}`), some are bare strings (`!"early_fragment_tests"`). A stage that
/// scans the tail for only the one attribute it knows cannot tell "no attribute" from "an
/// attribute I have never seen", so a new one is dropped in silence. Decoding the whole tail in
/// one place is what makes the difference observable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StageAttribute {
    /// `!{!"air.patch", !"<domain>", !"air.patch_control_point", i32 N}` on a post-tessellation
    /// vertex function. Carries the node body; the vertex decode reads the shape out of it.
    Patch(String),
    /// `!"early_fragment_tests"` -- `[[early_fragment_tests]]`. Depth and stencil tests run
    /// before the fragment body, so a fragment the test rejects performs none of the body's
    /// stores. Vulkan spells it `OpExecutionMode ... EarlyFragmentTests`.
    EarlyFragmentTests,
    /// `!{!"air.max_work_group_size", i32 N}` -- `[[max_total_threads_per_threadgroup(N)]]`. The
    /// largest threadgroup this entry was compiled to run.
    MaxWorkGroupSize(u32),
    /// A tail operand this translator has no model for. Carried out so a stage can refuse rather
    /// than emit a module that silently lacks whatever the attribute asked for.
    Unrecognized(String),
}

impl StageAttribute {
    /// How the attribute reads back in a refusal message.
    pub(crate) fn describe(&self) -> String {
        match self {
            StageAttribute::Patch(_) => "air.patch".to_string(),
            StageAttribute::EarlyFragmentTests => "early_fragment_tests".to_string(),
            StageAttribute::MaxWorkGroupSize(_) => "air.max_work_group_size".to_string(),
            StageAttribute::Unrecognized(text) => text.clone(),
        }
    }
}

/// Decode the attribute tail of a stage root node body.
///
/// The first three operands are the entry function pointer, its output node and its input node;
/// every operand after them is an attribute. Recognition is by the attribute's own ABI marker,
/// never by its position in the tail.
fn stage_root_attributes(rootc: &str, nodes: &HashMap<u32, String>) -> Vec<StageAttribute> {
    let operands = split_top_level_commas(rootc);
    operands
        .iter()
        .skip(3)
        .filter_map(|operand| {
            let referenced = operand
                .strip_prefix('!')
                .and_then(|digits| digits.parse::<u32>().ok())
                .and_then(|id| nodes.get(&id));
            Some(match referenced {
                Some(node) if node.contains("!\"air.patch\"") => {
                    StageAttribute::Patch(node.clone())
                }
                Some(node) if node.contains("!\"air.max_work_group_size\"") => {
                    match i32_after_marker(node, "air.max_work_group_size") {
                        Some(size) => StageAttribute::MaxWorkGroupSize(size),
                        None => StageAttribute::Unrecognized(
                            "air.max_work_group_size states no thread count".to_string(),
                        ),
                    }
                }
                // A node with no operands states nothing, so there is nothing here to model and
                // nothing emission could drop. Refusing it would cost a translation for no safety.
                Some(node) if node.trim().is_empty() => return None,
                Some(node) => StageAttribute::Unrecognized(format!("!{{{node}}}")),
                None if operand == "!\"early_fragment_tests\"" => {
                    StageAttribute::EarlyFragmentTests
                }
                None => StageAttribute::Unrecognized(operand.clone()),
            })
        })
        .collect()
}

/// The metadata refs (`!N`) appearing in a node body, in order. Skips `ptr @func` operands.
fn refs_in(body: &str) -> Vec<u32> {
    body.split(',')
        .filter_map(|s| {
            s.trim()
                .strip_prefix('!')
                .and_then(|x| x.parse::<u32>().ok())
        })
        .collect()
}

/// The `i32 N` immediate (parameter index) inside a metadata node body, if present.
fn first_i32(body: &str) -> Option<u32> {
    let mut it = body.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        if tok == "i32" {
            if let Some(n) = it.peek() {
                let n = n.trim_end_matches(',');
                if let Ok(v) = n.parse::<u32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn i32_after_marker(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    first_i32(&body[pos + marker.len()..])
}

fn location_index(body: &str, fallback: u32) -> u32 {
    match location_operands(body).map(|operands| operands.index) {
        Some(LocationOperand::Literal(index)) => index,
        _ => fallback,
    }
}

/// How many descriptors an argument node states it occupies, when AIR states more than one.
///
/// `None` for the ordinary single-descriptor argument, and for a count spelled as a
/// function-constant global -- the consumer picks that length at pipeline creation, so there is no
/// compile-time array to size.
fn declared_descriptor_count(body: &str) -> Option<u32> {
    match location_operands(body)?.count? {
        LocationOperand::Literal(count) if count > 1 => Some(count),
        _ => None,
    }
}

/// One operand of the `air.location_index` pair: either a compile-time value or a pointer to a
/// function-constant global whose value the consumer chooses at pipeline creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LocationOperand {
    Literal(u32),
    Global(String),
}

/// The two operands `air.location_index` always carries: the resource's Metal slot, and how many
/// descriptors the argument occupies at that slot.
///
/// `!"air.location_index", i32 4, i32 2` is `[[texture(4)]]` holding two texture handles. The count
/// is `1` for an ordinary resource; above `1` the argument is a handle ARRAY, and every corpus
/// declaration above `1` agrees with the `array<..., N>` length in the type name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocationOperands {
    pub index: LocationOperand,
    pub count: Option<LocationOperand>,
}

/// The operands that follow `marker` in an argument node, POSITIONALLY, each decoded as a literal
/// or as the function-constant global that supplies it at pipeline creation.
///
/// AIR spells a slot either way, and a decoder that scans forward for "the next `i32`" or "the next
/// `@`" reads a DIFFERENT operand whenever the one it wants is spelled the other way. Measured over
/// 14579 corpus sources, `air.location_index`'s pair is 112946 `(i32, i32)`, 3404 `(ptr, i32)`,
/// 1447 `(ptr, ptr)` and 439 `(i32, ptr)`, and `air.render_target`'s first operand is 3213 `i32`
/// and 103 `ptr` -- so scanning finds the descriptor COUNT in place of the texture slot, or the
/// dual-source index in place of the color attachment.
fn marker_operands(body: &str, marker: &str) -> Vec<Option<LocationOperand>> {
    let marker = format!("!\"{marker}\"");
    let Some(position) = body.find(&marker) else {
        return vec![];
    };
    split_metadata_operands(&body[position + marker.len()..])
        .into_iter()
        .map(|operand| {
            if let Some(literal) = operand.strip_prefix("i32 ") {
                return literal.trim().parse().ok().map(LocationOperand::Literal);
            }
            let at = operand.find('@')?;
            let name = operand[at..]
                .chars()
                .take_while(|character| {
                    !character.is_whitespace() && !matches!(*character, ',' | ')' | '(' | '[' | ']')
                })
                .collect::<String>();
            (name.len() > 1).then_some(LocationOperand::Global(name))
        })
        .collect()
}

/// Decode the `air.location_index` operand pair.
pub(crate) fn location_operands(body: &str) -> Option<LocationOperands> {
    let mut operands = marker_operands(body, "air.location_index").into_iter();
    Some(LocationOperands {
        index: operands.next().flatten()?,
        count: operands.next().flatten(),
    })
}

/// The Metal slot a node declares under `marker`: `air.location_index` for a bound resource,
/// `air.render_target` for a color attachment.
///
/// Both are the FIRST operand after their marker, and both may name a function-constant global
/// instead of a literal. `fallback` -- the parameter or member ordinal -- stands in when the global
/// is one this module does not initialize, because the operands AFTER the slot describe something
/// else entirely and answering with one of those is answering the wrong question.
pub(crate) fn declared_slot(
    body: &str,
    marker: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    match marker_operands(body, marker).into_iter().next().flatten() {
        Some(LocationOperand::Literal(slot)) => slot,
        Some(LocationOperand::Global(global)) => {
            static_int_globals.get(&global).copied().unwrap_or(fallback)
        }
        None => fallback,
    }
}

/// Split an AIR metadata operand list on the commas that separate operands. A `!"..."` string
/// operand may itself contain commas (`!"array<texture2d<half, sample>, 2>"`), so a plain
/// `split(',')` would tear one operand into three and shift every position after it.
fn split_metadata_operands(tail: &str) -> Vec<String> {
    let mut operands = vec![];
    let mut current = String::new();
    let mut quoted = false;
    for character in tail.chars() {
        match character {
            '"' => {
                quoted = !quoted;
                current.push(character);
            }
            ',' if !quoted => {
                operands.push(std::mem::take(&mut current).trim().to_string());
            }
            _ => current.push(character),
        }
    }
    operands.push(current.trim().to_string());
    operands.retain(|operand| !operand.is_empty());
    operands
}

fn render_target_location(
    body: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    declared_slot(body, "air.render_target", fallback, static_int_globals)
}

fn address_space(body: &str) -> Option<u32> {
    i32_after_marker(body, "air.address_space")
}

/// The first `air.<role>` string literal in a node body (e.g. `air.texture`).
fn role_strings(body: &str) -> Vec<String> {
    let mut out = vec![];
    let mut rest = body;
    while let Some(p) = rest.find("!\"air.") {
        let after = &rest[p + 6..];
        let end = after.find('"').unwrap_or(after.len());
        out.push(after[..end].to_string());
        rest = &after[end..];
    }
    out
}

/// The primary arg-kind role string for an argument node, looking past a leading
/// `air.function_constant` wrapper. A conditionally-present (function-constant-gated) resource
/// emits `!"air.function_constant", !REF` BEFORE its real `!"air.<role>"` marker, so a naive
/// "first role string" sees `function_constant` and mis-classifies the argument as `Other`. The
/// real role is the first marker that isn't the wrapper. `air.function_constant` is a stable AIR
/// metadata-ABI symbol, not a shader identifier.
fn primary_role(strs: &[String]) -> Option<&str> {
    strs.iter()
        .map(String::as_str)
        .find(|s| *s != "function_constant")
}

/// The role marker a declaration carries, read POSITIONALLY past an `air.function_constant`
/// wrapper, or `None` when it carries none.
///
/// The wrapper is the operand PAIR `!"air.function_constant", !PREDICATE`; the role, when there is
/// one, is the operand that follows the predicate. Both other readings of this node are wrong here:
///
/// - Scanning for "the first `air.` marker that is not the wrapper" reads a BARE function-constant
///   parameter's own `air.arg_type_name` as though it were a role. That parameter *is* the function
///   constant -- there is no role behind the wrapper at all -- and the positional read answers
///   `None` for it, because the operand after its predicate is not an `air.` marker.
/// - Asking [`fc_promoted_role`] answers `None` for every gated role the emitter does not PROMOTE,
///   and that is precisely the set this predicate exists to name. A gated `air.amplification_id` on
///   a fragment entry was classified `Other`, bound to a zero and never reported -- in a module that
///   validated, bound and reflected as though nothing were missing. Whether the gate is on is a
///   DIFFERENT question (see [`unmodelled_declared_role`]), not a reason to stop reading.
///
/// Measured: every one of the 14000-odd function-constant-wrapped argument and return nodes across
/// the 14579 local corpus sources carries a real role marker in that position, and none carries a
/// qualifier -- [`ARGUMENT_IDENTITY_MARKERS`] is what answers the role-less parameter whichever way
/// AIR spells its wrapper.
fn declared_role(body: &str) -> Option<String> {
    let role_name = |operand: &str| {
        operand
            .strip_prefix("!\"air.")
            .and_then(|rest| rest.strip_suffix('"'))
            .map(str::to_string)
    };
    let mut operands = split_metadata_operands(body).into_iter();
    let first = operands.find_map(|operand| role_name(&operand))?;
    if first != "function_constant" {
        return (first != "function_constant_disabled").then_some(first);
    }
    // The predicate the wrapper gates on. A wrapper carrying none gates nothing and wraps nothing.
    operands.next().filter(|operand| {
        operand.strip_prefix('!').is_some_and(|rest| {
            !rest.is_empty() && rest.chars().all(|character| character.is_ascii_digit())
        })
    })?;
    operands
        .next()
        .as_deref()
        .and_then(role_name)
        .filter(|role| !ARGUMENT_IDENTITY_MARKERS.contains(&role.as_str()))
}

/// The two markers that name an argument's C-level identity rather than its role. Every argument
/// node carries both whatever else it declares, so one of them sitting where a role would sit means
/// the declaration has no role: it is the bare function constant itself.
const ARGUMENT_IDENTITY_MARKERS: &[&str] = &["arg_name", "arg_type_name"];

fn function_constant_gate_global(body: &str, nodes: &HashMap<u32, String>) -> Option<String> {
    if role_strings(body).first().map(String::as_str) != Some("function_constant") {
        return None;
    }
    refs_in(body)
        .into_iter()
        .filter_map(|r| nodes.get(&r))
        .find_map(|node| {
            let at = node.find('@')?;
            let name = node[at..]
                .chars()
                .take_while(|c| !c.is_whitespace() && !matches!(*c, ',' | ')' | '(' | '[' | ']'))
                .collect::<String>();
            (name.len() > 1).then_some(name)
        })
}

/// True when the node's `air.function_constant` gate is one this module's own static initializers
/// drive to ZERO -- the argument is absent from the variant this module describes.
///
/// This is deliberately NOT [`metadata_enabled_by_default`] negated. That predicate answers "may
/// this declaration be present", so an UNRESOLVED gate is false for it. Here the question is the
/// opposite one -- "did the variant the initializers describe leave this argument out" -- and an
/// unresolved gate is no evidence of that.
fn function_constant_gate_disabled(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .and_then(|global| static_int_globals.get(&global).copied())
        == Some(0)
}

/// True when the node's `air.function_constant` gate is one this module's own static initializers
/// drive NONZERO -- the argument is PRESENT in the variant this module describes.
///
/// The mirror of [`function_constant_gate_disabled`], and it has to be a separate question from
/// [`metadata_enabled_by_default`] for the same reason that one is: an UNRESOLVED gate reads as
/// "may be present" there, and "may be present" is not evidence of presence any more than it is
/// evidence of absence. Only the resolved-nonzero case is.
fn function_constant_gate_enabled(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .and_then(|global| static_int_globals.get(&global).copied())
        .is_some_and(|value| value != 0)
}

/// The role a gated BUFFER argument carries once the module's own initializers have answered
/// whether it is there.
///
/// [`FC_PROMOTED_RESOURCE_ROLES`] deliberately omits `buffer`: a gated buffer whose gate this
/// module cannot resolve may or may not be bound, and the "possibly-absent -> Private zero
/// placeholder" default is the honest answer for it. But that argument is about an UNRESOLVED gate,
/// and it was being applied to a resolved-ON one too. A gate this module's own static initializers
/// drive nonzero states that the variant this module describes DOES carry the buffer -- the same
/// evidence standard [`function_constant_gate_disabled`] uses to drop a descriptor in the other
/// direction -- and binding a definitely-present buffer to a Private zero sends its every store
/// into per-invocation scratch.
///
/// It also settles a split reading: the fragment decode reaches the wrapped role through
/// [`primary_role`], so a gated buffer on a fragment entry has always been a buffer, while kernel
/// and vertex read it through [`gated_role`] and left it a placeholder. One declaration, two
/// classifications.
///
/// The remaining half of that split -- a gated-OFF buffer, which the fragment still binds and the
/// other two still drop -- stays, and stays deliberately. Making the fragment agree costs 490
/// buffer bindings and 1060 loads across 100 corpus modules and rescues nothing; a gated-off
/// argument declaring a LITERAL slot is Metal's spelling of mutually exclusive alternatives, and
/// no gated fragment buffer in the corpus declares anything else. Measured in
/// `a_fragment_gated_buffer_keeps_its_literal_slot_whichever_way_the_gate_resolves`.
fn present_gated_buffer_role<'a>(
    role: &'a str,
    strs: &'a [String],
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> &'a str {
    if role != "function_constant" || primary_role(strs) != Some("buffer") {
        return role;
    }
    if function_constant_gate_enabled(body, nodes, static_int_globals) {
        "buffer"
    } else {
        role
    }
}

/// The Metal texture slot an argument node declares FOR THE VARIANT THIS MODULE DESCRIBES, or
/// `None` when it declares none and the argument therefore has no descriptor to bind.
///
/// Two spellings mean "no slot", and reading only the first of them decorated two textures with one
/// descriptor in 250 of the 14579 local corpus sources:
///
/// - `air.location_index` resolving to `-1`: AIR states outright that specialization has not
///   assigned a binding yet.
/// - A slot supplied by a function-constant global on an argument whose own gate the module's
///   initializers drive to zero. Metal compiles `[[texture(fc_expr)]]` into a RUNNING SUM over the
///   arguments the pipeline enables, so the global holds this argument's slot only when this
///   argument is one of them. For an argument the variant leaves out, the same global holds
///   wherever the sum stopped -- which is a LIVE argument's slot. One corpus fragment shader put
///   seventeen differently-shaped textures on `Binding 38` that way, and a kernel put a
///   `texture2d<float, write>` and a 128-element `array_ref` on `Binding 480`. Reflection then
///   asks a consumer to bind two different textures to one Metal index, and the emitted module
///   aliases descriptors whose only correct reading is the one the shader is not compiled for.
///
/// An argument gated off but declaring a LITERAL slot keeps it: `[[texture(0), function_constant(a)]]`
/// beside `[[texture(0), function_constant(!a)]]` is Metal's own way of spelling mutually exclusive
/// typed alternatives, the slot is stated rather than summed, and the alternatives are never live
/// together.
fn variant_texture_slot(
    body: &str,
    fallback: u32,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> Option<u32> {
    let slot = location_index_with_static(body, fallback, static_int_globals);
    if slot == u32::MAX {
        return None;
    }
    let summed_by_function_constant = matches!(
        location_operands(body).map(|operands| operands.index),
        Some(LocationOperand::Global(_))
    );
    if summed_by_function_constant
        && function_constant_gate_disabled(body, nodes, static_int_globals)
    {
        return None;
    }
    Some(slot)
}

/// The role marker a texture argument carries once [`variant_texture_slot`] has found the variant
/// declares no slot for it. It is deliberately not `"texture"` and deliberately not
/// `"function_constant"`: the first would ask for a descriptor that does not exist, and the second
/// throws away which argument was demoted, which is the only thing that tells an absent resource
/// apart from a handle the lowering merely lost.
/// True when this declaration is a TEXTURE the variant this module describes leaves out, spelled
/// the way [`specialize_function_constant_metadata`] spells it.
///
/// Two spellings state that one fact. `air.function_constant` beside a gate this module's own
/// static initializers drive to zero is what an unspecialized module carries, and
/// [`variant_texture_slot`] demotes it. Once a VALUE is supplied, the specializer rewrites that
/// same node to `air.function_constant_disabled`, after which the wrapped role is not readable at
/// all -- [`declared_role`] answers `None` for it by design, because a disabled declaration
/// declares nothing. That is right for the interface and wrong for the operand: the argument still
/// reaches the body, still binds to a Private placeholder, and is still the absent resource. Naming
/// only the first spelling is what left every function-constant-specialized translation reaching
/// the recovery in `recovered_image_for_private_operand` that the unspecialized one no longer does.
///
/// The role is read POSITIONALLY: `role_strings` skips the wrapper's `!PREDICATE` operand, so the
/// marker after the wrapper is the role. `texture` is never an [`ARGUMENT_IDENTITY_MARKERS`] entry,
/// so this cannot mistake a bare function-constant parameter for one.
fn declares_disabled_texture(strs: &[String]) -> bool {
    strs.first().map(String::as_str) == Some("function_constant_disabled")
        && strs.get(1).map(String::as_str) == Some("texture")
}

const VARIANT_ABSENT_TEXTURE_ROLE: &str = "variant_absent_texture";

/// The role a declaration carries that this stage's emitter does NOT model, or `None` when the
/// role is one it models -- or when the variant this module describes leaves the declaration out.
///
/// Reporting the role is what turns an unsupported entry parameter or return member into an honest
/// `FALLBACK` instead of a silently dropped input. The four callers (kernel, fragment and vertex
/// entry parameters, and fragment return members) differ only in which role list is theirs, so the
/// rule -- including which gates excuse a refusal -- is stated once here.
///
/// The excusing predicate is [`function_constant_gate_disabled`] and deliberately NOT
/// `!metadata_enabled_by_default`. Those two are not complements. `metadata_enabled_by_default`
/// answers "may this declaration be present", so an UNRESOLVED gate reads as "absent" for it --
/// and an unmodelled role behind a gate we could not evaluate is still unmodelled. Suppressing the
/// refusal there is precisely the case where the emitter goes on to bind a zero for a parameter the
/// pipeline may well have supplied a value for. Only a gate this module's own static initializers
/// drive to zero is evidence that the variant left the declaration out.
fn unmodelled_declared_role(
    is_modelled: impl Fn(&str) -> bool,
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> Option<String> {
    let declared = declared_role(body)?;
    (!is_modelled(&declared) && !function_constant_gate_disabled(body, nodes, static_int_globals))
        .then_some(declared)
}

fn metadata_enabled_by_default(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .map(|global| {
            static_int_globals
                .get(&global)
                .is_some_and(|value| *value != 0)
        })
        .unwrap_or(true)
}

/// Specialize metadata-only `air.function_constant` wrappers from statically evaluated AIR
/// predicates. A true predicate exposes the wrapped role as unconditional. A false predicate is
/// marked disabled so interface construction cannot retain an unreachable descriptor or output.
/// Unknown predicates keep the wrapper and the ordinary unspecialized projection.
pub(crate) fn specialize_function_constant_metadata(ll: &str) -> String {
    let nodes = collect_nodes(ll);
    let static_int_globals = static_init_int_global_values(ll);
    let marker = "!\"air.function_constant\"";
    let mut output = String::with_capacity(ll.len());

    for line in ll.split_inclusive('\n') {
        let value = function_constant_gate_global(line, &nodes)
            .and_then(|global| static_int_globals.get(&global).copied());
        match value {
            Some(0) => {
                output.push_str(&line.replacen(marker, "!\"air.function_constant_disabled\"", 1))
            }
            Some(_) => {
                let rewritten = line.find(marker).and_then(|start| {
                    let after_marker = &line[start + marker.len()..];
                    let next_role = after_marker.find("!\"air.")?;
                    Some(format!("{}{}", &line[..start], &after_marker[next_role..]))
                });
                if let Some(rewritten) = rewritten {
                    output.push_str(&rewritten);
                } else {
                    output.push_str(line);
                }
            }
            None => output.push_str(line),
        }
    }
    output
}

/// The wrapped roles `fc_promoted_role` looks past an `air.function_constant` wrapper for. One list
/// rather than one match arm each: separate arms are what let `sampler` go missing next to
/// `texture`, while the two are declared together and used through each other.
///
/// **A role belongs here when promoting it changes the emitted MODULE, not only reflection.** The
/// standing justification for keeping a gated descriptor is that the resource-using arm may be
/// enabled later — but an argument whose uses fold away leaves an unreferenced variable that
/// `module_cleanup` removes, and reflection then asks a consumer to create and bind a descriptor no
/// instruction touches. Measured over the 2880-source sample against the corpus:
///
/// - `stage_in`: 12 kernel modules, +24 reflected `KernelStageInput` AND +24 module descriptors.
///   The parameters were `OpUndef` — the kernel read nothing where the runtime had a vertex stream.
/// - `indirect_buffer`: 28 modules, +225 reflected bindings and **0** module bindings. Deliberately
///   NOT here; promoting it buys a consumer nothing but descriptors to satisfy.
/// - `buffer`: 147 modules, +499 reflected bindings and +132 module bindings -- but 14 module
///   bindings LOST, and one source of 14579 stops translating at all ("owned Store violates its
///   pointer-pointee and value-type contract"). Promoted only under the caller's `promote_buffers`,
///   for the reason below -- and, unconditionally but only on the evidence of a resolved-ON gate,
///   by [`present_gated_buffer_role`], which is the subset of that widening that costs nothing.
///
/// `texture`, `sampler` and `imageblock` are already wrapper-neutral: stripping the wrapper from
/// every one of them corpus-wide changes no module and no reflected field.
const FC_PROMOTED_RESOURCE_ROLES: &[&str] = &[
    "imageblock",
    "intersection_function_table",
    "sampler",
    "stage_in",
    "texture",
    "visible_function_table",
];

/// Like `primary_role`, but only looks past the `air.function_constant` wrapper when the wrapped
/// resource is one of [`FC_PROMOTED_RESOURCE_ROLES`], or a BUFFER (the latter only when
/// `promote_buffers` is set).
/// An imageblock has no descriptor binding to promote; recognizing its stable ABI marker merely keeps
/// its metadata-described cell type available to the native emitter. A wrapped texture with a real
/// location remains a descriptor even when its predicate defaults false: SPIR-V specialization may
/// enable the resource-using arm, and substituting another texture or erasing the binding would be
/// semantically wrong. A `-1` location remains absent until AIR specialization assigns a binding.
/// A wrapped SAMPLER is the same argument in the other descriptor band, and the two are declared
/// together — a gated `texture2d` is sampled through a gated `sampler`. Collapsing only the sampler
/// left the sample with no state to reach: its parameter never became a descriptor, so the operand
/// stayed pointer-shaped into image lowering. The fragment decode has always read the wrapped role
/// directly and so already promoted it; kernel and vertex read it through here and did not, which is
/// the same declaration classified two ways.
/// Promoting a wrapped BUFFER binds it as a
/// REAL StorageBuffer instead of the "possibly-absent → Private zero placeholder" default; on the
/// DEFAULT path this regresses byte-conformant goldens (a genuinely-absent fc buffer must stay Private,
/// and the conditionally-present binding emits an invalid `ArrayStride`-decorated array-of-Block), so
/// `promote_buffers` is FALSE on every default parse. It is set only by the adopt-if-validates
/// `fc_promote_psb` retry (an FC-multiplexed kernel whose live dtype variant's buffers ARE present and
/// hold real data — demoting them to Private zeros makes the cross-binding pointer merge read zeros,
/// byte-wrong; keeping them real StorageBuffer lets the FC prune + PSB lower the merge byte-correctly).
/// Dispatching on stable AIR ABI markers, not shader names.
fn fc_promoted_role(strs: &[String], promote_buffers: bool) -> Option<&str> {
    match gated_role(strs, |role| {
        FC_PROMOTED_RESOURCE_ROLES.contains(&role) || air_role_is_system_value(role)
    }) {
        // Only the `fc_promote_psb` retry asks for the extra role, so it is spelled as one extra
        // question rather than a second copy of the list that can drift from the first.
        Some("function_constant") if promote_buffers => gated_role(strs, |role| role == "buffer"),
        other => other,
    }
}

/// The role marker an interface node declares, resolving an `air.function_constant` wrapper against
/// the roles the reading stage models.
///
/// Every stage decode asks this one question — "a gated declaration: which role is it?" — and the
/// answer has to be the SAME for the same declaration, because a stage interface is matched across
/// stages by position and by `Location`, not by name. Each decode used to answer it its own way:
/// the fragment lists read the role past the wrapper unconditionally, the vertex output list read
/// it through a resource allowlist that named no output role at all. So one gated member of one
/// Metal varying struct was a varying on the fragment side and nothing on the vertex side, and
/// every later varying in that struct landed on a different `Location` in the two modules — a
/// silent read of the wrong interpolant, in two modules that each validate.
///
/// `modelled` answers "does the caller lower this role", asked of the caller's own lowering table,
/// so a role the stage handles cannot go missing from a second hand-maintained copy of that list.
/// A wrapper with anything else behind it
/// — an unmodelled role, or a BARE function constant whose next marker is its own
/// `air.arg_type_name` — stays the wrapper. `air.function_constant_disabled` (an
/// AIR-specialized-off predicate) is never a role: it is not the wrapper this looks past.
fn gated_role(strs: &[String], modelled: impl Fn(&str) -> bool) -> Option<&str> {
    let first = strs.first().map(String::as_str)?;
    if first != "function_constant" {
        return Some(first);
    }
    Some(match primary_role(strs) {
        Some(role) if modelled(role) => role,
        _ => first,
    })
}

fn string_after_marker(body: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    let after = &body[pos + marker.len()..];
    let pos = after.find("!\"")?;
    let value = &after[pos + 2..];
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

fn arg_type_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_type_name")
}

fn declared_buffer_access(body: &str) -> Option<BufferAccess> {
    if body.contains("air.read_write") {
        Some(BufferAccess::ReadWrite)
    } else if body.contains("air.write") {
        Some(BufferAccess::WriteOnly)
    } else if body.contains("air.read") {
        Some(BufferAccess::ReadOnly)
    } else {
        None
    }
}

fn arg_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_name")
}

fn ref_after_marker(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let after = body.get(body.find(&marker)? + marker.len()..)?;
    let bang = after.find('!')?;
    let digits = after[bang + 1..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().ok()
}

fn parse_fragment_imageblock_master(body: &str) -> Option<Vec<FragmentImageblockMember>> {
    let toks = tokenize(body);
    let mut members = Vec::new();
    let mut i = 0;
    while i + 4 < toks.len() {
        let (offset, size, type_name, semantic) = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ) {
            (
                Some(Tok::Int(offset)),
                Some(Tok::Int(size)),
                Some(Tok::Int(_array_len)),
                Some(Tok::Str(type_name)),
                Some(Tok::Str(semantic)),
            ) => (*offset, *size, type_name.clone(), semantic.clone()),
            _ => return None,
        };
        i += 5;
        let raster_order_group = match (toks.get(i), toks.get(i + 1)) {
            (Some(Tok::Str(marker)), Some(Tok::Int(group)))
                if marker == "air.raster_order_group" =>
            {
                i += 2;
                *group
            }
            _ => return None,
        };
        members.push(FragmentImageblockMember {
            offset,
            size,
            type_name,
            semantic,
            raster_order_group,
        });
    }
    (!members.is_empty() && i == toks.len()).then_some(members)
}

fn parse_fragment_imageblock_projection(
    nodes: &HashMap<u32, String>,
    node: &str,
    interface_index: u32,
    master_members: &[FragmentImageblockMember],
) -> Option<FragmentImageblockProjection> {
    let projection_ref = struct_info_ref(node)?;
    let projection = nodes.get(&projection_ref)?;
    let toks = tokenize(projection);
    let mut members = Vec::new();
    let mut i = 0;
    let mut projection_member = 0;
    while i + 4 < toks.len() {
        let semantic = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ) {
            (
                Some(Tok::Int(_)),
                Some(Tok::Int(_)),
                Some(Tok::Int(_)),
                Some(Tok::Str(_)),
                Some(Tok::Str(semantic)),
            ) => semantic,
            _ => return None,
        };
        let master_member = master_members
            .iter()
            .position(|member| member.semantic == *semantic)? as u32;
        members.push(FragmentImageblockProjectionMember {
            projection_member,
            master_member,
        });
        projection_member += 1;
        i += 5;
        if matches!(toks.get(i), Some(Tok::Str(marker)) if marker == "air.raster_order_group")
            && matches!(toks.get(i + 1), Some(Tok::Int(_)))
        {
            i += 2;
        }
    }
    (!members.is_empty() && i == toks.len()).then_some(FragmentImageblockProjection {
        interface_index,
        members,
    })
}

fn parse_fragment_imageblock(
    nodes: &HashMap<u32, String>,
    out_ref: u32,
    in_ref: u32,
) -> Option<FragmentImageblock> {
    let output_nodes = nodes
        .get(&out_ref)
        .map(|body| refs_in(body))
        .unwrap_or_default();
    let input_nodes = nodes
        .get(&in_ref)
        .map(|body| refs_in(body))
        .unwrap_or_default();
    let imageblock_node = output_nodes
        .iter()
        .chain(input_nodes.iter())
        .filter_map(|id| nodes.get(id))
        .find(|body| primary_role(&role_strings(body)) == Some("imageblock_data"))?;
    // Some AIR producers attach an explicit `air.imageblock_master` to a narrow projection. Others
    // pass the complete imageblock-data struct directly, in which case its own struct metadata is
    // the master layout. Both contracts carry the same offset/size/type/semantic/ROG tuple stream;
    // use that structure rather than requiring the optional indirection.
    let master_ref = ref_after_marker(imageblock_node, "air.imageblock_master")
        .or_else(|| struct_info_ref(imageblock_node))?;
    let master_members = parse_fragment_imageblock_master(nodes.get(&master_ref)?)?;
    let sample_size = i32_after_marker(imageblock_node, "air.imageblock_data_size")?;

    let outputs = output_nodes
        .iter()
        .enumerate()
        .filter_map(|(index, id)| {
            let node = nodes.get(id)?;
            (primary_role(&role_strings(node)) == Some("imageblock_data"))
                .then(|| {
                    parse_fragment_imageblock_projection(nodes, node, index as u32, &master_members)
                })
                .flatten()
        })
        .collect();
    let inputs = input_nodes
        .iter()
        .filter_map(|id| {
            let node = nodes.get(id)?;
            let index = first_i32(node)?;
            (primary_role(&role_strings(node)) == Some("imageblock_data"))
                .then(|| parse_fragment_imageblock_projection(nodes, node, index, &master_members))
                .flatten()
        })
        .collect();
    Some(FragmentImageblock {
        sample_size,
        members: master_members,
        inputs,
        outputs,
    })
}

/// Parse `!air.fragment` into a `FragMeta`. Structure (RE'd Phase 5.11):
///   `!air.fragment = !{!N}`; `!N = !{ptr @func, !OUT, !IN}`;
///   `!IN = !{!a, !b, ...}` each `!{i32 idx, !"air.<role>", ...}`; `!OUT = list of render targets`.
pub fn parse_air_fragment_meta(ll: &str) -> Option<FragMeta> {
    parse_air_fragment_meta_with_entry(ll).0
}

pub(crate) fn parse_air_fragment_meta_with_entry(ll: &str) -> (Option<FragMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "fragment", &nodes);
    let meta = parse_air_fragment_meta_with_nodes(ll, &nodes, entry.as_deref());
    (meta, entry)
}

/// The fragment entry-parameter roles the emitter models.
///
/// Any other enabled role lands in [`FragMeta::unmodelled_input_params`] and is rejected at
/// emission. `render_target` appears here because an `air.render_target` in the *input* list is
/// framebuffer fetch (`[[color(n)]]`), not an output.
/// The AIR entry-parameter roles that name a SPIR-V builtin rather than a bound resource, across
/// every stage. The execution-group family answers for itself -- see [`air_role_is_system_value`].
///
/// A resource keeps its descriptor whether or not its function constant is on: the pipeline layout
/// has to match what the application binds either way. A system value does not. When its constant
/// is off the parameter is absent, and declaring the builtin anyway puts a variable in the entry
/// point interface — and, for `viewport_array_index` and `render_target_array_index`, a device
/// capability in the module — for a value the shader cannot read.
///
/// The converse is the half that went missing for two stages. The fragment decode reads the role
/// past its wrapper unconditionally and consults this list only to drop the gated-OFF ones; the
/// kernel and vertex decodes asked `fc_promoted_role`, which collapses every role it does not
/// PROMOTE back to the wrapper -- gated-ON builtins included. Those bound a zero for a value the
/// pipeline had, which is the same silent read the list exists to prevent, from the other side.
pub const AIR_SYSTEM_VALUE_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "barycentric_coord",
    "dispatch_threads_per_threadgroup",
    "front_facing",
    "instance_id",
    "patch_id",
    "point_coord",
    "position",
    "position_in_patch",
    "primitive_id",
    "render_target_array_index",
    "sample_id",
    "sample_mask_in",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
    "vertex_id",
    "viewport_array_index",
];

/// Whether an AIR entry-parameter role names a system value. See [`AIR_SYSTEM_VALUE_ROLES`]; the
/// execution-group family is decoded from the marker's shape rather than listed, so it is asked
/// rather than enumerated and a group added to [`AIR_EXECUTION_GROUPS`] needs no second entry here.
pub fn air_role_is_system_value(role: &str) -> bool {
    AIR_SYSTEM_VALUE_ROLES.contains(&role) || execution_group_role(role).is_some()
}

/// The role a stage should classify a declaration as, once a gated-OFF system value is accounted
/// for: that parameter is ABSENT, and binding a zero for it is what Metal defines. Blanked rather
/// than dropped so the caller's own `match` falls through to its unmodelled arm.
///
/// One statement of the rule for all three stages. It is only about system values: a RESOURCE keeps
/// its descriptor whichever way its gate goes, because the pipeline layout has to match what the
/// application binds either way.
fn present_system_value_role<'a>(
    role: &'a str,
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> &'a str {
    if air_role_is_system_value(role)
        && !metadata_enabled_by_default(body, nodes, static_int_globals)
    {
        return "";
    }
    role
}

pub const FRAGMENT_INPUT_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "barycentric_coord",
    "buffer",
    "fragment_input",
    "front_facing",
    "imageblock_data",
    "indirect_buffer",
    "intersection_function_table",
    "point_coord",
    "position",
    "primitive_id",
    "render_target",
    "render_target_array_index",
    "sample_id",
    "sample_mask_in",
    "sampler",
    "texture",
    "viewport_array_index",
    "visible_function_table",
];

/// The vertex entry-parameter roles the emitter models. See [`FRAGMENT_INPUT_ROLES`].
pub const VERTEX_INPUT_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "buffer",
    "indirect_buffer",
    "instance_id",
    "intersection_function_table",
    "patch_control_point_input",
    "patch_id",
    "patch_input",
    "position_in_patch",
    "sampler",
    "texture",
    "vertex_id",
    "vertex_input",
    "visible_function_table",
];

/// The kernel entry-parameter roles the emitter models. See [`FRAGMENT_INPUT_ROLES`].
///
/// The execution-group family is NOT listed here. Its members are decoded from the marker's shape by
/// [`execution_group_role`], and [`air_input_role_is_modelled`] asks that decoder rather than a
/// second spelling of the same family -- a member could otherwise have a lowering and still be
/// refused for want of a line in a list.
pub const KERNEL_INPUT_ROLES: &[&str] = &[
    "buffer",
    "dispatch_threads_per_threadgroup",
    "imageblock",
    "indirect_buffer",
    "instance_acceleration_structure",
    "intersection_function_table",
    "primitive_acceleration_structure",
    "sampler",
    "stage_in",
    "texture",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
    "visible_function_table",
];

/// Whether the emitter models `role` on an entry parameter of `stage`.
///
/// The stage's own inventory, plus the execution-group facts that stage can answer. Those are read
/// off the marker's shape, so the family needs no entry in any inventory and cannot end up listed
/// for one stage and forgotten for another -- which is how it started, listed for the kernel alone
/// while a fragment declaring `[[thread_index_in_simdgroup]]` was refused outright.
pub fn air_input_role_is_modelled(stage: Stage, role: &str) -> bool {
    let inventory = match stage {
        Stage::Fragment => FRAGMENT_INPUT_ROLES,
        Stage::Vertex => VERTEX_INPUT_ROLES,
        Stage::Kernel => KERNEL_INPUT_ROLES,
    };
    inventory.contains(&role) || stage_execution_group_role(stage, role).is_some()
}

/// Metal's execution groups inside a threadgroup, and how many threads each holds.
///
/// A quadgroup is four threads and a simdgroup thirty-two, and EVERY role in the family --
/// a lane's index inside its group, its group's index inside the threadgroup, how many groups the
/// threadgroup holds, and the group's own width -- is one derivation over that single number.
/// Spelling the family out per group is what left `air.quadgroups_per_threadgroup` with no arm at
/// all beside `air.simdgroups_per_threadgroup`, so a kernel declaring it read a zero where the
/// hardware had a count.
pub const AIR_EXECUTION_GROUPS: &[(&str, u32)] = &[("quadgroup", 4), ("simdgroup", 32)];

/// One fact an entry parameter can ask about the execution group its thread runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionGroupFact {
    /// `[[thread_index_in_simdgroup]]`, `[[thread_index_in_quadgroup]]`: this lane's index inside
    /// its own group.
    ThreadIndexInGroup,
    /// `[[simdgroup_index_in_threadgroup]]`, `[[quadgroup_index_in_threadgroup]]`: which group
    /// inside the threadgroup this lane belongs to.
    GroupIndexInThreadgroup,
    /// `[[simdgroups_per_threadgroup]]`, `[[quadgroups_per_threadgroup]]`: how many whole groups the
    /// threadgroup's local size divides into.
    GroupsPerThreadgroup,
    /// `[[threads_per_simdgroup]]`: the group's width.
    ThreadsPerGroup,
}

impl ExecutionGroupFact {
    /// Whether a shader stage has to be dispatched as a threadgroup grid for this fact to mean
    /// anything.
    ///
    /// A lane's index inside its group and the group's own width are facts about the GROUP: every
    /// stage runs its threads in execution groups, and this translator states both off
    /// `SubgroupLocalInvocationId` and the group's width, neither of which mentions a workgroup.
    /// The other two are facts about the THREADGROUP the groups divide -- which one this thread's
    /// group is, and how many the local size holds -- and only a compute grid has one, so a
    /// fragment or vertex entry declaring them is refused rather than answered with a zero.
    pub fn needs_a_threadgroup(self) -> bool {
        match self {
            Self::ThreadIndexInGroup | Self::ThreadsPerGroup => false,
            Self::GroupIndexInThreadgroup | Self::GroupsPerThreadgroup => true,
        }
    }
}

/// The execution-group fact an AIR role marker names and the group's width, or `None` when the
/// marker names no execution group.
///
/// Read from the marker's SHAPE against [`AIR_EXECUTION_GROUPS`] rather than from an arm per
/// (group, fact) pair, so a member of the family cannot be modelled for one group and missing for
/// the other. The four shapes are exhaustive over the family and unambiguous: no marker outside it
/// matches one and then names a group in the table.
///
/// The answer carries no stage in it -- a Metal simdgroup is the same thirty-two threads whichever
/// entry declares it. [`ExecutionGroupFact::needs_a_threadgroup`] is where a stage enters.
///
/// `dispatch_` is Metal's "what the dispatch asked for" spelling, and it is stripped here only so a
/// member of THIS family cannot be modelled under one spelling and missing under the other. It does
/// not follow that the two spellings denote one value: for the threadgroup they do not, which is why
/// `air.dispatch_threads_per_threadgroup` is its own [`KernRole::DispatchThreadsPerThreadgroup`].
/// No group in [`AIR_EXECUTION_GROUPS`] is the threadgroup, so that role never reaches this decode.
pub fn execution_group_role(marker: &str) -> Option<(ExecutionGroupFact, u32)> {
    let marker = marker.strip_prefix("dispatch_").unwrap_or(marker);
    let (fact, group) = if let Some(group) = marker.strip_prefix("thread_index_in_") {
        (ExecutionGroupFact::ThreadIndexInGroup, group)
    } else if let Some(group) = marker.strip_suffix("_index_in_threadgroup") {
        (ExecutionGroupFact::GroupIndexInThreadgroup, group)
    } else if let Some(group) = marker.strip_suffix("s_per_threadgroup") {
        (ExecutionGroupFact::GroupsPerThreadgroup, group)
    } else {
        let group = marker.strip_prefix("threads_per_")?;
        (ExecutionGroupFact::ThreadsPerGroup, group)
    };
    let lanes = AIR_EXECUTION_GROUPS
        .iter()
        .find_map(|&(name, lanes)| (name == group).then_some(lanes))?;
    Some((fact, lanes))
}

/// The execution-group fact a stage models, if this marker names one it can answer.
///
/// One statement of "does this stage lower that marker", shared by the inventory that decides what
/// to refuse and by all three decodes. See [`ExecutionGroupFact::needs_a_threadgroup`].
fn stage_execution_group_role(stage: Stage, marker: &str) -> Option<(ExecutionGroupFact, u32)> {
    execution_group_role(marker)
        .filter(|(fact, _)| stage == Stage::Kernel || !fact.needs_a_threadgroup())
}

/// The fragment return-member roles the emitter lowers.
///
/// A member carrying any other role is reported through `FragMeta::unmodelled_output_members` and
/// rejected at emission rather than skipped. AIR states an output because the shader writes it, so
/// a member nothing knows how to write is a value that silently disappears — the shape that let
/// `[[sample_mask]]` be dropped without a diagnostic for as long as it was unrecognised.
pub const FRAGMENT_OUTPUT_ROLES: &[&str] = &[
    "render_target",
    "depth",
    "stencil",
    "sample_mask",
    "imageblock_data",
];

/// What a vertex return member is once its role marker is read. The `Varying` case still needs its
/// location, which the decode assigns positionally, so this names the arm rather than the value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VertexOutputKind {
    Position,
    PointSize,
    ClipDistance,
    ViewportArrayIndex,
    RenderTargetArrayIndex,
    Varying,
}

impl VertexOutputKind {
    /// Whether this member is a SPIR-V builtin rather than a user varying.
    ///
    /// The two answer the gating question differently, exactly as they do on an entry parameter
    /// list (see [`AIR_SYSTEM_VALUE_ROLES`]): a varying keeps its slot whatever its predicate
    /// says, because the stage consuming the same struct keeps it too, while a builtin the shader
    /// cannot have written is absent. `Position` is not a gated case — a vertex shader without a
    /// clip position is not a vertex shader, and AIR does not gate it.
    fn is_system_value(self) -> bool {
        !matches!(self, Self::Varying | Self::Position)
    }
}

/// The vertex return-member roles the emitter lowers, and what each one lowers to. See
/// [`FRAGMENT_OUTPUT_ROLES`] for the fragment side of the same contract.
///
/// One table, not a role list beside a `match`: this is BOTH the set an `air.function_constant`
/// wrapper is resolved against and the mapping the decode applies, so a role cannot be lowered
/// without being promoted past the wrapper, or promoted without being lowered. Those were two
/// spellings of one fact, and they disagreed — the wrapper list named no output role at all, so a
/// gated `air.vertex_output` was a varying to the fragment decode reading the same declaration and
/// nothing to this one, which renumbered every later varying in the struct on one side only.
const VERTEX_OUTPUT_ROLES: &[(&str, VertexOutputKind)] = &[
    ("clip_distance", VertexOutputKind::ClipDistance),
    ("point_size", VertexOutputKind::PointSize),
    ("position", VertexOutputKind::Position),
    (
        "render_target_array_index",
        VertexOutputKind::RenderTargetArrayIndex,
    ),
    ("vertex_output", VertexOutputKind::Varying),
    ("viewport_array_index", VertexOutputKind::ViewportArrayIndex),
];

/// The roles the VERTEX entry-parameter decode reads past an `air.function_constant` wrapper: the
/// shared resource band, plus the two stage-input roles this stage owns.
///
/// A stage input is not a resource, but promoting it is the same argument the resource band already
/// makes. SPIR-V specialization may enable the arm that reads it, and an argument that never became
/// an input variable reads `OpUndef` instead of the attribute the application bound and reflection
/// asked for — silently, in a module that validates. It is also what the fragment decode answers for
/// the mirror-image `air.fragment_input`, and one declaration must not be classified two ways.
/// All 573 gated `air.vertex_input` arguments in 89 of 14579 local corpus sources carry a real
/// `air.location_index`, and none collides with another attribute in its own module.
///
/// Stated as one promotion set rather than as patches applied to the resource classifier's answer
/// afterwards, which is how `patch_input` came to be corrected at the call site while `vertex_input`
/// was not corrected at all.
fn vertex_input_gated_role(strs: &[String]) -> Option<&str> {
    gated_role(strs, |role| {
        FC_PROMOTED_RESOURCE_ROLES.contains(&role)
            || air_role_is_system_value(role)
            || matches!(role, "patch_input" | "vertex_input")
    })
}

/// The [`VertexOutputKind`] a vertex return-member role marker names, if the emitter lowers it.
fn vertex_output_kind(role: &str) -> Option<VertexOutputKind> {
    VERTEX_OUTPUT_ROLES
        .iter()
        .find(|(name, _)| *name == role)
        .map(|(_, kind)| *kind)
}

fn parse_air_fragment_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<FragMeta> {
    let static_int_globals = static_init_int_global_values(ll);
    let root = stage_root(ll, "fragment")?;
    let rootc = nodes.get(&root)?;
    let refs = refs_in(rootc);
    let (out_ref, in_ref) = (*refs.first()?, *refs.get(1)?);
    let mut early_fragment_tests = false;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::EarlyFragmentTests => early_fragment_tests = true,
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }
    let fragment_imageblock = parse_fragment_imageblock(nodes, out_ref, in_ref);
    let render_target_members: Vec<(u32, u32)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let roles = role_strings(node);
                    let is_render_target = primary_role(&roles) == Some("render_target");
                    (is_render_target
                        && metadata_enabled_by_default(node, nodes, &static_int_globals))
                    .then(|| {
                        (
                            i as u32,
                            render_target_location(node, i as u32, &static_int_globals),
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    // Which return members carry a given non-color output role. One reader for every such role, so
    // adding one is a call rather than another copy of this walk — `air.sample_mask` was dropped
    // for as long as it was the role nobody had copied the walk for.
    let output_members_with_role = |role: &str| -> Vec<u32> {
        nodes
            .get(&out_ref)
            .map(|c| {
                refs_in(c)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let node = nodes.get(&r)?;
                        (primary_role(&role_strings(node)) == Some(role)
                            && metadata_enabled_by_default(node, nodes, &static_int_globals))
                        .then_some(i as u32)
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let depth_members = output_members_with_role("depth");
    let depth_qualifier = nodes.get(&out_ref).and_then(|c| {
        refs_in(c).into_iter().find_map(|r| {
            let node = nodes.get(&r)?;
            (primary_role(&role_strings(node)) == Some("depth"))
                .then(|| string_after_marker(node, "air.depth_qualifier"))
                .flatten()
                .and_then(|qualifier| match qualifier.as_str() {
                    "air.any" => Some(DepthQualifier::Any),
                    "air.less" => Some(DepthQualifier::Less),
                    "air.greater" => Some(DepthQualifier::Greater),
                    _ => None,
                })
        })
    });
    let stencil_members = output_members_with_role("stencil");
    let sample_mask_members = output_members_with_role("sample_mask");
    let unmodelled_output_members: Vec<(u32, String)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let role = unmodelled_declared_role(
                        |role| FRAGMENT_OUTPUT_ROLES.contains(&role),
                        node,
                        nodes,
                        &static_int_globals,
                    )?;
                    Some((i as u32, role))
                })
                .collect()
        })
        .unwrap_or_default();
    let render_target_indices: Vec<u32> = render_target_members
        .iter()
        .map(|(_, location)| *location)
        .collect();
    let n_render_targets = render_target_indices.len() as u32;
    let render_target_type_names: HashMap<u32, String> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let roles = role_strings(node);
                    let is_render_target = primary_role(&roles) == Some("render_target");
                    if is_render_target
                        && metadata_enabled_by_default(node, nodes, &static_int_globals)
                    {
                        arg_type_name(node).map(|name| (i as u32, name))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut buffer_layouts = HashMap::new();
    let mut varying_types = HashMap::new();
    let mut varying_names = HashMap::new();
    let mut varying_user_semantics = HashMap::new();
    let mut varying_interpolation = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut color_input_type_names = HashMap::new();
    let mut varying_loc = 0u32;
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut indirect_buffer_struct_refs: Vec<(u32, u32, u32)> = Vec::new();
    // AIR function-param pointer address spaces for the fragment entry, the fallback the kernel
    // parser uses when a buffer arg node omits `air.address_space`.
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        // Reconstruct any buffer struct layout used when native emission produces a bare pointer.
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        let Some(role_str) = primary_role(&strs) else {
            continue;
        };
        let texture_slot = variant_texture_slot(node, idx, nodes, &static_int_globals);
        let role_str = present_system_value_role(role_str, node, nodes, &static_int_globals);
        if let Some(declared) = unmodelled_declared_role(
            |role| air_input_role_is_modelled(Stage::Fragment, role),
            node,
            nodes,
            &static_int_globals,
        ) {
            unmodelled_input_params.push((idx, declared));
        }
        let role =
            match role_str {
                "position" => FragRole::Position,
                "point_coord" => FragRole::PointCoord,
                "front_facing" => FragRole::FrontFacing,
                "barycentric_coord" => FragRole::BarycentricCoord {
                    no_perspective: VaryingInterpolation::from_role_strings(&strs).no_perspective,
                },
                "primitive_id" => FragRole::PrimitiveId,
                "sample_id" => FragRole::SampleId,
                "sample_mask_in" => FragRole::SampleMaskIn,
                "viewport_array_index" => FragRole::ViewportArrayIndex,
                "render_target_array_index" => FragRole::RenderTargetArrayIndex,
                "amplification_id" => FragRole::AmplificationId,
                "amplification_count" => FragRole::AmplificationCount,
                "fragment_input" => {
                    let l = varying_loc;
                    varying_loc += 1;
                    if let Some(name) = arg_type_name(node) {
                        varying_types.insert(l, name);
                    }
                    if let Some(name) = arg_name(node) {
                        varying_names.insert(l, name);
                    }
                    if let Some(semantic) = string_after_marker(node, "air.fragment_input") {
                        varying_user_semantics.insert(l, semantic);
                    }
                    varying_interpolation.insert(l, VaryingInterpolation::from_role_strings(&strs));
                    FragRole::Varying(l)
                }
                "texture" => match texture_slot {
                    Some(slot) => {
                        if let Some(name) = arg_type_name(node) {
                            texture_type_names.insert(idx, name);
                        }
                        FragRole::Texture(slot)
                    }
                    // A texture this variant declares no slot for has no binding to report. See
                    // [`variant_texture_slot`].
                    None => FragRole::VariantAbsentTexture,
                },
                "sampler" => {
                    FragRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
                }
                "visible_function_table" => FragRole::VisibleFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "intersection_function_table" => FragRole::IntersectionFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "buffer" | "indirect_buffer" => {
                    if role_str == "indirect_buffer" {
                        if let Some(sref) = struct_info_ref(node) {
                            indirect_buffer_struct_refs.push((
                                idx,
                                location_index_with_static(node, idx, &static_int_globals),
                                sref,
                            ));
                        }
                    }
                    // Populate address space / declared size ONLY when the IR actually carries them
                    // (no invented default), so reflection never reports a guessed value.
                    if let Some(space) =
                        address_space(node).or_else(|| param_address_spaces.get(&idx).copied())
                    {
                        buffer_address_spaces.insert(idx, space);
                    }
                    if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                        .or_else(|| i32_after_marker(node, "air.buffer_size"))
                    {
                        buffer_type_sizes.insert(idx, size);
                    }
                    if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                        buffer_object_sizes.insert(idx, size);
                    }
                    if let Some(name) = arg_type_name(node) {
                        buffer_type_names.insert(idx, name);
                    }
                    if let Some(access) = declared_buffer_access(node) {
                        buffer_accesses.insert(idx, access);
                    }
                    FragRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
                }
                // an air.render_target in the INPUT list = framebuffer fetch ([[color(n)]]).
                "render_target" => {
                    let location = render_target_location(node, idx, &static_int_globals);
                    if let Some(name) = arg_type_name(node) {
                        color_input_type_names.insert(location, name);
                    }
                    FragRole::ColorInput(location)
                }
                "imageblock_data" => FragRole::ImageblockData,
                // The specializer's spelling of a texture this variant leaves out. The wrapped
                // role is unreadable past it, so this is the only place the fact survives.
                _ if declares_disabled_texture(&strs) => FragRole::VariantAbsentTexture,
                other => stage_execution_group_role(Stage::Fragment, other).map_or(
                    FragRole::Other,
                    |(fact, lanes)| FragRole::ExecutionGroup { fact, lanes },
                ),
            };
        roles.push((idx, role));
    }
    let top_level_texture_locations = roles
        .iter()
        .filter_map(|(_, role)| match role {
            FragRole::Texture(location) => Some(*location),
            _ => None,
        })
        .collect::<Vec<_>>();
    let argument_buffers = ArgumentBuffers::new(indirect_buffer_struct_refs);
    Some(FragMeta {
        roles,
        unmodelled_stage_attributes,
        early_fragment_tests,
        unmodelled_input_params,
        implicit_imageblock_attachments: detect_implicit_imageblock_attachments(ll)?,
        varying_types,
        varying_names,
        varying_user_semantics,
        varying_interpolation,
        n_render_targets,
        render_target_members,
        render_target_type_names,
        depth_members,
        depth_qualifier,
        stencil_members,
        sample_mask_members,
        unmodelled_output_members,
        fragment_imageblock,
        render_target_indices,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        color_input_type_names,
        embedded_textures: if body_uses_texture_intrinsic(ll) {
            detect_embedded_textures(nodes, &argument_buffers, &top_level_texture_locations)
        } else {
            Vec::new()
        },
        embedded_arguments: detect_embedded_arguments(nodes, &argument_buffers),
        unsurfaced_embedded_resources: unsurfaced_embedded_resources(nodes, &argument_buffers),
    })
}

/// Parse `!air.vertex` into a `VertMeta`. `!air.vertex = !{!N}`; `!N = !{ptr @func, !OUT, !IN}`;
/// `!IN` entries are `!{i32 idx, !"air.vertex_input"|"air.buffer", ...}`.
pub fn parse_air_vertex_meta(ll: &str) -> Option<VertMeta> {
    parse_air_vertex_meta_with_entry(ll).0
}

pub(crate) fn parse_air_vertex_meta_with_entry(ll: &str) -> (Option<VertMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "vertex", &nodes);
    let meta = parse_air_vertex_meta_with_nodes(ll, &nodes, entry.as_deref());
    (meta, entry)
}

fn parse_air_vertex_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<VertMeta> {
    let static_int_globals = static_init_int_global_values(ll);
    let root = stage_root(ll, "vertex")?;
    let rootc = nodes.get(&root)?;
    let refs = refs_in(rootc);
    let out_ref = *refs.first()?;
    let in_ref = *refs.get(1)?;
    // Read the whole attribute tail, not just the one attribute this stage models. A patch node is
    // recognised by what it says (`air.patch`) rather than by where it sits, and anything else the
    // root carries becomes a named refusal instead of silence.
    let mut patch_node = None;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::Patch(node) => patch_node = Some(node),
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }
    let mut undecoded_patch_shape = None;
    let patch_shape = patch_node.as_deref().and_then(|node| {
        let domain = if node.contains("!\"quad\"") {
            PatchDomain::Quad
        } else if node.contains("!\"triangle\"") {
            PatchDomain::Triangle
        } else if node.contains("!\"isoline\"") {
            PatchDomain::Isoline
        } else {
            // Dropping the shape here would emit an ordinary vertex shader: no domain, no spacing,
            // no winding, and a per-patch input set the pipeline never wires. Carry the failure out
            // so the passes can refuse instead.
            undecoded_patch_shape = Some("air.patch names no tessellation domain".to_string());
            return None;
        };
        match i32_after_marker(node, "air.patch_control_point") {
            Some(count) => Some((domain, count)),
            None => {
                undecoded_patch_shape =
                    Some("air.patch states no air.patch_control_point count".to_string());
                None
            }
        }
    });

    let mut output_roles = vec![];
    let mut invariant_outputs = vec![];
    let mut unmodelled_output_members = vec![];
    let mut output_varying_types = HashMap::new();
    let mut output_varying_names = HashMap::new();
    let mut output_varying_user_semantics = HashMap::new();
    let mut out_loc = 0u32;
    for r in refs_in(nodes.get(&out_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let strs = role_strings(node);
        let Some(first) = gated_role(&strs, |role| vertex_output_kind(role).is_some()) else {
            continue;
        };
        let kind = vertex_output_kind(first).filter(|kind| {
            // A gated builtin the shader cannot have written is absent. Declaring it anyway puts a
            // variable in the entry-point interface -- and, for `Layer` and `ViewportIndex`, a
            // device capability in the module -- for a value nothing produces.
            !kind.is_system_value() || metadata_enabled_by_default(node, nodes, &static_int_globals)
        });
        let role = match kind {
            Some(VertexOutputKind::Position) => VertOutRole::Position,
            Some(VertexOutputKind::PointSize) => VertOutRole::PointSize,
            Some(VertexOutputKind::ClipDistance) => VertOutRole::ClipDistance,
            Some(VertexOutputKind::ViewportArrayIndex) => VertOutRole::ViewportArrayIndex,
            Some(VertexOutputKind::RenderTargetArrayIndex) => VertOutRole::RenderTargetArrayIndex,
            Some(VertexOutputKind::Varying) => {
                let l = location_index_with_static(node, out_loc, &static_int_globals);
                out_loc += 1;
                if let Some(name) = arg_type_name(node) {
                    output_varying_types.insert(l, name);
                }
                if let Some(name) = arg_name(node) {
                    output_varying_names.insert(l, name);
                }
                if let Some(semantic) = string_after_marker(node, "air.vertex_output") {
                    output_varying_user_semantics.insert(l, semantic);
                }
                VertOutRole::Varying(l)
            }
            // The member is in the return struct but nothing writes it: an unresolved wrapper over
            // a role this stage does not lower, an AIR-specialized-off predicate, or a builtin the
            // filter above ruled absent.
            None if matches!(first, "function_constant" | "function_constant_disabled")
                || vertex_output_kind(first).is_some() =>
            {
                VertOutRole::FunctionConstantDisabled
            }
            // A role no vertex output lowering models. Reported, not skipped: AIR states an output
            // because the shader writes it, so a member nothing knows how to write is a value that
            // silently disappears.
            None => VertOutRole::Other,
        };
        if strs.iter().any(|s| s == "invariant") {
            invariant_outputs.push(output_roles.len() as u32);
        }
        if matches!(role, VertOutRole::Other) {
            unmodelled_output_members.push((output_roles.len() as u32, first.to_string()));
        }
        output_roles.push(role);
    }

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut parameter_type_names = HashMap::new();
    let mut vertex_input_types = HashMap::new();
    let mut vertex_input_names = HashMap::new();
    let mut patch_input_types = HashMap::new();
    let mut patch_input_names = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut indirect_buffer_struct_refs = Vec::new();
    let mut patch_control_point = None;
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let mut vin_loc = 0u32;
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        if let Some(name) = arg_type_name(node) {
            parameter_type_names.insert(idx, name);
        }
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        let Some(mut first) = vertex_input_gated_role(&strs) else {
            continue;
        };
        // The one demotion the promotion set cannot state: a wrapped texture this variant declares
        // no slot for has no binding to promote it to. See [`variant_texture_slot`].
        let texture_slot = variant_texture_slot(node, idx, nodes, &static_int_globals);
        if primary_role(&strs) == Some("texture") && texture_slot.is_none() {
            first = VARIANT_ABSENT_TEXTURE_ROLE;
        }
        let first = present_gated_buffer_role(first, &strs, node, nodes, &static_int_globals);
        let first = present_system_value_role(first, node, nodes, &static_int_globals);
        if let Some(declared) = unmodelled_declared_role(
            |role| air_input_role_is_modelled(Stage::Vertex, role),
            node,
            nodes,
            &static_int_globals,
        ) {
            unmodelled_input_params.push((idx, declared));
        }
        let role =
            match first {
                "vertex_input" => {
                    let l = location_index_with_static(node, vin_loc, &static_int_globals);
                    vin_loc += 1;
                    if let Some(name) = arg_type_name(node) {
                        vertex_input_types.insert(l, name);
                    }
                    if let Some(name) = arg_name(node) {
                        vertex_input_names.insert(l, name);
                    }
                    VertRole::VertexInput(l)
                }
                "buffer" | "indirect_buffer" => {
                    if first == "indirect_buffer" {
                        if let Some(sref) = struct_info_ref(node) {
                            indirect_buffer_struct_refs.push((
                                idx,
                                location_index_with_static(node, idx, &static_int_globals),
                                sref,
                            ));
                        }
                    }
                    if let Some(space) =
                        address_space(node).or_else(|| param_address_spaces.get(&idx).copied())
                    {
                        buffer_address_spaces.insert(idx, space);
                    }
                    if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                        .or_else(|| i32_after_marker(node, "air.buffer_size"))
                    {
                        buffer_type_sizes.insert(idx, size);
                    }
                    if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                        buffer_object_sizes.insert(idx, size);
                    }
                    if let Some(name) = arg_type_name(node) {
                        buffer_type_names.insert(idx, name);
                    }
                    if let Some(access) = declared_buffer_access(node) {
                        buffer_accesses.insert(idx, access);
                    }
                    VertRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
                }
                "texture" => {
                    if let Some(name) = arg_type_name(node) {
                        texture_type_names.insert(idx, name);
                    }
                    VertRole::Texture(texture_slot.unwrap_or_else(|| {
                        location_index_with_static(node, idx, &static_int_globals)
                    }))
                }
                "sampler" => {
                    VertRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
                }
                "visible_function_table" => VertRole::VisibleFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "intersection_function_table" => VertRole::IntersectionFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "vertex_id" => VertRole::VertexId,
                "instance_id" => VertRole::InstanceId,
                "patch_control_point_input" => {
                    let refs = refs_in(node);
                    let function = refs
                        .first()
                        .and_then(|reference| nodes.get(reference))
                        .and_then(|body| pointer_symbol(body));
                    let fields = refs
                        .iter()
                        .skip(1)
                        .filter_map(|reference| nodes.get(reference))
                        .map(|field| PatchControlPointField {
                            location: location_index_with_static(field, 0, &static_int_globals),
                            type_name: arg_type_name(field),
                        })
                        .collect::<Vec<_>>();
                    if let Some(function) = function {
                        patch_control_point = Some((function, fields));
                    }
                    VertRole::PatchControlPoints
                }
                "patch_input" => {
                    let location = location_index_with_static(node, idx, &static_int_globals);
                    if let Some(name) = arg_type_name(node) {
                        patch_input_types.insert(location, name);
                    }
                    if let Some(name) = arg_name(node) {
                        patch_input_names.insert(location, name);
                    }
                    VertRole::PatchInput(location)
                }
                "position_in_patch" => VertRole::PositionInPatch,
                "patch_id" => VertRole::PatchId,
                "amplification_id" => VertRole::AmplificationId,
                "amplification_count" => VertRole::AmplificationCount,
                VARIANT_ABSENT_TEXTURE_ROLE => VertRole::VariantAbsentTexture,
                // The specializer's spelling of the same fact. Once a VALUE is supplied,
                // `specialize_function_constant_metadata` rewrites the resolved-off wrapper, and past
                // that rewrite the wrapped role is unreadable -- so the demotion above never runs and
                // this is the only place the fact survives.
                _ if declares_disabled_texture(&strs) => VertRole::VariantAbsentTexture,
                other => stage_execution_group_role(Stage::Vertex, other).map_or(
                    VertRole::Other,
                    |(fact, lanes)| VertRole::ExecutionGroup { fact, lanes },
                ),
            };
        roles.push((idx, role));
    }
    let top_level_texture_locations = roles
        .iter()
        .filter_map(|(_, role)| match role {
            VertRole::Texture(location) => Some(*location),
            _ => None,
        })
        .collect::<Vec<_>>();
    let argument_buffers = ArgumentBuffers::new(indirect_buffer_struct_refs);
    Some(VertMeta {
        roles,
        unmodelled_input_params,
        implicit_imageblock_attachments: detect_implicit_imageblock_attachments(ll)?,
        parameter_type_names,
        output_roles,
        invariant_outputs,
        unmodelled_output_members,
        output_varying_types,
        output_varying_names,
        output_varying_user_semantics,
        vertex_input_types,
        vertex_input_names,
        patch_input_types,
        patch_input_names,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        embedded_textures: if body_uses_texture_intrinsic(ll) {
            detect_embedded_textures(nodes, &argument_buffers, &top_level_texture_locations)
        } else {
            Vec::new()
        },
        embedded_arguments: detect_embedded_arguments(nodes, &argument_buffers),
        unsurfaced_embedded_resources: unsurfaced_embedded_resources(nodes, &argument_buffers),
        undecoded_patch_shape,
        unmodelled_stage_attributes,
        tessellation: patch_shape.map(|(domain, control_point_count)| {
            let (control_point_function, control_point_fields) = patch_control_point
                .map(|(function, fields)| (Some(function), fields))
                .unwrap_or_default();
            TessellationMeta {
                domain,
                control_point_count,
                control_point_function,
                control_point_fields,
            }
        }),
    })
}

#[cfg(test)]
mod tests;
