use super::types::{member_tuples, tokenize};
use super::{arg_type_name, role_strings, texture_shape_from_name, BufferAccess, TextureFormat};
use crate::passes::ImageComp;
use spirv::Dim;
use std::collections::HashMap;

/// The `air.indirect_buffer` arguments of one entry, in the order the synthetic embedded-texture
/// index counts them: `(parameter index, Metal `[[buffer(N)]]` slot, `air.struct_type_info` node)`
/// per argument buffer, ordered by PARAMETER INDEX.
///
/// The order is part of an ABI, not a detail. [`embedded_synthetic_texture_index`] fixes where the
/// synthetic indices start and [`detect_embedded_textures`] hands out `K`, `K+1`, ... by walking
/// this list, so a consumer that binds by synthetic index has to reach the same order the
/// translator did. Walking the AIR metadata list in the order it happens to name its argument
/// nodes made that order a property of the metadata layout: reversing the list of one corpus kernel
/// moves the same embedded texture from `Binding 32` to `Binding 480` and from Sampled to Storage.
///
/// Ordered by PARAMETER index rather than by Metal buffer index, because the latter is not a total
/// order -- mutually exclusive function-constant alternatives share one `[[buffer(N)]]` -- while an
/// entry's parameter index is unique by construction.
///
/// The type exists so the order cannot be skipped: the walkers below take this and nothing else,
/// and [`ArgumentBuffers::new`] is the only way to make one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ArgumentBuffers(Vec<(u32, u32, u32)>);

impl ArgumentBuffers {
    pub(super) fn new(mut declared: Vec<(u32, u32, u32)>) -> Self {
        declared.sort_by_key(|(param_index, _, _)| *param_index);
        Self(declared)
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        self.0.iter().copied()
    }
}

/// A texture that lives inside an `air.indirect_buffer` argument buffer, marked
/// `air.indirect_argument` → `air.texture` in the buffer's `air.struct_type_info`, and used by the
/// shader body through an AIR texture intrinsic.
///
/// The translator materializes a synthetic UniformConstant sampled image for it and the validation
/// harness binds the SAME deterministically-seeded texture on both the Apple-oracle side (encoded
/// into the arg buffer via `MTLArgumentEncoder`) and the Vulkan-runner side (the reflected synthetic
/// descriptor). The authored identity remains the owning buffer plus field offset; the synthetic
/// texture index `K` is translator-owned (see [`embedded_synthetic_texture_index`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedTexture {
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_param_index: u32,
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct.
    pub field_offset: u32,
    pub field_ordinal: u32,
    /// Metal argument-encoder index (`[[id(n)]]`) of this texture inside the argument buffer.
    pub argument_index: u32,
    /// Texture dimensionality (2D, with `arrayed` carrying the 2D-array distinction).
    pub dim: Dim,
    pub arrayed: bool,
    /// Sampled/storage component type (Float for `texture2d<float, read/write>`).
    pub comp: ImageComp,
    /// Storage-image format for write/read_write textures; `None` for sampled/read textures.
    pub storage_format: Option<TextureFormat>,
    /// Fixed number of opaque texture handles in this argument-buffer field. `None` denotes one
    /// handle; embedded runtime `array_ref` fields are not yet representable by Metal's fixed
    /// argument-buffer layout metadata.
    pub array_length: Option<u32>,
    /// The synthetic texture index `K` this embedded texture binds at.
    pub synthetic_texture_index: u32,
}

/// One resource-handle member of an AIR indirect argument buffer. This is the common structural
/// coordinate used by authored embedded resources, Metal argument encoding, and static lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedArgument {
    pub buffer_param_index: u32,
    pub buffer_index: u32,
    pub field_ordinal: u32,
    pub field_offset: u32,
    pub argument_index: u32,
    /// Nested Metal buffer location for an `air.buffer` field; absent for constants, textures, and
    /// function-table handles represented by the same structural coordinate.
    pub resource_buffer_index: Option<u32>,
    pub resource_address_space: Option<u32>,
    pub resource_declared_size: Option<u32>,
    pub resource_access: Option<BufferAccess>,
}

/// The synthetic texture index `K` for embedded argument-buffer textures, derived so the translator
/// (binding decoration) and the validation harness (model + oracle) agree WITHOUT any name key.
///
/// **ABI convention:** `K = 1 + max(top-level air.texture air.location_index)`, or `0` when the
/// kernel has no top-level textures. This guarantees `K` never collides with a real top-level
/// texture's location index, so the synthetic image binds at index `K` in the sampled- or
/// storage-texture band. `texture_locations` is the set of `air.location_index` values of the
/// kernel's top-level texture args.
pub fn embedded_synthetic_texture_index(texture_locations: &[u32]) -> u32 {
    texture_locations.iter().copied().max().map_or(0, |m| m + 1)
}

/// True iff the shader body calls an AIR texture intrinsic. These are stable AIR intrinsic
/// symbols, not shader identifiers. A declaration line is only emitted when the body actually calls it,
/// so its presence in the `.ll` text is a sound structural signal.
pub(super) fn body_uses_texture_intrinsic(ll: &str) -> bool {
    ll.contains("@air.sample_")
        || ll.contains("@air.gather_")
        || ll.contains("@air.read_texture")
        || ll.contains("@air.write_texture")
        || ll.contains("@air.write_imageblock_slice_to_texture")
}

/// Scan each `air.indirect_buffer`'s `air.struct_type_info` for members marked
/// `air.indirect_argument` whose nested node is an `air.texture`. Each such texture is
/// surfaced as an `EmbeddedTexture` with a synthetic index K, K+1, … (K from
/// [`embedded_synthetic_texture_index`]) so the translator and harness agree on the binding without
/// a name key. Every texture shape [`texture_shape_from_name`] reads shares this contract; see
/// [`expressible_embedded_texture`] for what is left out.
pub(super) fn detect_embedded_textures(
    nodes: &HashMap<u32, String>,
    argument_buffers: &ArgumentBuffers,
    top_level_texture_locations: &[u32],
) -> Vec<EmbeddedTexture> {
    let mut out = vec![];
    let mut next_k = embedded_synthetic_texture_index(top_level_texture_locations);
    for (buffer_param_index, buffer_index, sref) in argument_buffers.iter() {
        for member in embedded_argument_members(nodes, sref) {
            let Some(tex_node) = nodes.get(&member.node_ref) else {
                continue;
            };
            let Some(shape) = expressible_embedded_texture(&member, tex_node) else {
                continue;
            };
            out.push(EmbeddedTexture {
                buffer_param_index,
                buffer_index,
                field_offset: member.field_offset,
                field_ordinal: member.field_ordinal,
                argument_index: member.argument_index(tex_node),
                dim: shape.dimension.to_spirv_dim(),
                arrayed: shape.arrayed,
                comp: shape.component.to_image_comp(),
                storage_format: shape.storage_format,
                array_length: shape.array_length,
                synthetic_texture_index: next_k,
            });
            next_k += 1;
        }
    }
    out
}

/// The texture shape of an `air.indirect_argument` member the embedded lowering can materialize a
/// descriptor for, or `None` for any other member.
///
/// One predicate, so [`detect_embedded_textures`] and [`unsurfaced_embedded_resources`] cannot
/// disagree about which handles are surfaced: whatever this refuses stays a refusal reason.
///
/// The shape is derived by [`texture_shape_from_name`], the same reading the top-level texture
/// arguments get, and every dimension it names is materialized -- `register_embedded_textures`
/// builds the image from `dim`/`arrayed`/`comp` without caring which. An earlier 2D-only gate here
/// silently dropped the embedded `texturecube`, `texturecube_array`, `texture1d` and `texture3d`
/// members of 97 corpus sources; 10 of those reached a sample and answered black.
fn expressible_embedded_texture(
    member: &ArgumentMember,
    node: &str,
) -> Option<super::TextureShape> {
    // A wrapper declared as an ARRAY of structs repeats its members once per element, at offsets
    // and argument ids this walk does not enumerate. Surfacing element 0 alone would bind one
    // descriptor and leave the rest reading black.
    if member.in_array_wrapper {
        return None;
    }
    let strs = role_strings(node);
    // Structural gate: the node must be an `air.texture` with an access class the read/write
    // lowerings can express.
    if !strs.iter().any(|s| s == "texture")
        || !strs
            .iter()
            .any(|s| matches!(s.as_str(), "sample" | "read" | "write" | "read_write"))
    {
        return None;
    }
    let mut shape = texture_shape_from_name(&arg_type_name(node).unwrap_or_default());
    // A descriptor-array member has TWO spellings and only one of them is in the type name.
    // `metal::array<texture2d<half, sample>, 2>` names the array, and `texture_shape_from_name`
    // reads the length out of it. A C array -- `texture2d<float, sample> slices[32]` -- names the
    // ELEMENT type, and states the length only in the member tuple's `array_len` slot; the node's
    // own `air.location_index` count operand stays `1` for it, unlike the top-level spelling
    // (see the corpus fact that count and `array<..., N>` always agree -- that holds outside an
    // argument buffer, not inside one). Reading the name alone surfaced one descriptor and let a
    // runtime index sample element 0 for every value of the index.
    if shape.array_length.is_none() && member.array_len != 0 {
        shape.array_ref = true;
        shape.array_length = Some(member.array_len);
    }
    Some(shape)
}

pub(super) fn detect_embedded_arguments(
    nodes: &HashMap<u32, String>,
    argument_buffers: &ArgumentBuffers,
) -> Vec<EmbeddedArgument> {
    argument_buffers
        .iter()
        .flat_map(|(buffer_param_index, buffer_index, sref)| {
            embedded_argument_members(nodes, sref)
                .into_iter()
                // Flat members only. A wrapped `air.buffer` is a device address a consumer has to
                // write into the owning argument buffer, and nothing binds one yet, so it stays a
                // refusal reason in `unsurfaced_embedded_resources` rather than a silent descriptor.
                .filter(|member| member.depth == 0)
                .flat_map(move |member| {
                    let node = nodes.get(&member.node_ref);
                    let is_buffer = node
                        .is_some_and(|node| role_strings(node).iter().any(|role| role == "buffer"));
                    // A C-array member -- `device half4* bufs[2]` -- declares ONE handle per
                    // element, at consecutive argument ids and consecutive 8-byte slots, and states
                    // its length only in the member tuple. A consumer given the first element alone
                    // leaves every later slot of the argument buffer unwritten, so a runtime index
                    // into the member reads a device address nothing ever populated. Only the
                    // handle roles repeat this way: `air.indirect_constant` is stored INLINE, so its
                    // array is one span of data rather than N resources.
                    let elements = match node {
                        // A texture array is already counted ONCE, by
                        // `EmbeddedTexture::array_length`, and every consumer derives its elements
                        // from that. Repeating them here would be a second derivation of one fact
                        // for no reader, and 512 entries wide in the corpus's largest atlas.
                        Some(node) if expressible_embedded_texture(&member, node).is_some() => 1,
                        Some(node) if member.repeats_a_handle(node) => member.array_len,
                        _ => 1,
                    };
                    (0..elements).filter_map(move |element| {
                        let node = node?;
                        let argument_index = member.argument_index(node).checked_add(element)?;
                        Some(EmbeddedArgument {
                            buffer_param_index,
                            buffer_index,
                            field_ordinal: member.field_ordinal,
                            field_offset: member
                                .field_offset
                                .checked_add(element.checked_mul(HANDLE_BYTES)?)?,
                            argument_index,
                            resource_buffer_index: is_buffer.then_some(argument_index),
                            resource_address_space: is_buffer
                                .then(|| super::address_space(node))
                                .flatten(),
                            resource_declared_size: is_buffer
                                .then(|| super::i32_after_marker(node, "air.arg_type_size"))
                                .flatten(),
                            resource_access: is_buffer
                                .then(|| super::declared_buffer_access(node))
                                .flatten(),
                        })
                    })
                })
        })
        .collect()
}

/// The AIR roles that name a resource an argument buffer references by HANDLE, rather than a value
/// it stores inline. `air.indirect_constant` is the one role that means "stored inline".
const EMBEDDED_RESOURCE_ROLES: &[&str] = &[
    "buffer",
    "command_buffer",
    "compute_pipeline_state",
    "indirect_buffer",
    "intersection_function_table",
    "primitive_acceleration_structure",
    "render_pipeline_state",
    "sampler",
    "texture",
    "visible_function_table",
];

/// Resource handles an argument buffer declares that [`detect_embedded_textures`] cannot
/// materialize a descriptor for, each named so a refusal can say what was missed.
///
/// Metal lets an argument-buffer member be a user struct that itself holds a texture, sampler or
/// buffer -- `struct texture2d_wrapper { texture2d<half> t; };` as one member. A wrapped texture is
/// surfaced like a flat one, at the argument id [`ArgumentMember::argument_index`] derives; anything
/// else wrapped -- a buffer, a sampler, a function table, or any member of a wrapper declared as an
/// ARRAY of structs -- has no descriptor. Nor does a flat `air.texture` whose access class the
/// read/write lowerings cannot express.
///
/// Such a member decodes as opaque storage, the handle load yields a Private placeholder, and every
/// texture operation on it takes the "resource is absent" path and folds to zero. That path is
/// right for a resource AIR says is absent and wrong for one it declares, so a module carrying one
/// of these refuses rather than sampling black.
///
/// A flat `air.buffer`, `air.sampler` or function-table handle is NOT named here: those have their
/// own representation as an [`EmbeddedArgument`], and the ones that lack a descriptor lack it for
/// reasons this list is not the diagnosis of.
pub(super) fn unsurfaced_embedded_resources(
    nodes: &HashMap<u32, String>,
    argument_buffers: &ArgumentBuffers,
) -> Vec<String> {
    let mut out = vec![];
    for (buffer_param_index, _, sref) in argument_buffers.iter() {
        for member in embedded_argument_members(nodes, sref) {
            let Some(node) = nodes.get(&member.node_ref) else {
                continue;
            };
            if expressible_embedded_texture(&member, node).is_some() {
                continue;
            }
            let Some(role) = role_strings(node)
                .into_iter()
                .find(|role| EMBEDDED_RESOURCE_ROLES.contains(&role.as_str()))
            else {
                continue;
            };
            let offset = member.field_offset;
            match (member.depth, role.as_str()) {
                (0, "texture") => out.push(format!(
                    "argument buffer parameter {buffer_param_index} holds an air.texture at byte \
                     offset {offset} the embedded lowering cannot express ({})",
                    arg_type_name(node).unwrap_or_else(|| "unnamed".to_string())
                )),
                (0, _) => {}
                (_, _) => out.push(format!(
                    "argument buffer parameter {buffer_param_index} holds an air.{role} at byte \
                     offset {offset} inside a nested struct member"
                )),
            }
        }
    }
    out
}

/// One `air.indirect_argument` resource member of an argument buffer, flattened out of the nested
/// `air.struct_type_info` wrappers Metal allows around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArgumentMember {
    /// Member ordinal of the OUTERMOST struct member that reaches this handle -- the first index of
    /// the LLVM GEP that loads it. Members of one wrapper share it; `field_offset` separates them.
    field_ordinal: u32,
    /// Byte offset of the handle from the start of the argument buffer.
    field_offset: u32,
    /// The `air.<role>` node describing the resource.
    node_ref: u32,
    /// Sum of the `air.indirect_argument` ids of the enclosing wrappers; zero for a flat member. A
    /// wrapped member's own `air.location_index` is RELATIVE to it, so the Metal argument-encoder id
    /// is `argument_base + air.location_index`.
    argument_base: u32,
    /// How many `air.struct_type_info` wrappers enclose this handle. Zero for a flat member.
    depth: u32,
    /// Whether any enclosing wrapper is declared as an ARRAY of structs. Such a wrapper repeats its
    /// members once per element, and this walk names element 0 only.
    in_array_wrapper: bool,
    /// The member's OWN declared C-array length (`texture2d<float> slices[32]`), or zero when it is
    /// a single handle. This is a third, independent statement of a descriptor-array length -- see
    /// [`expressible_embedded_texture`] for why the type name cannot answer it.
    array_len: u32,
}

impl ArgumentMember {
    /// The Metal `[[id(n)]]` this resource occupies in its argument buffer.
    ///
    /// Proved against the Metal front end, which refuses
    /// `struct Inner { texture2d<half> x [[id(3)]]; }` at `[[id(7)]]` beside a sibling at
    /// `[[id(10)]]` with "attribute 'id' set location to 10, but minimum is 11": 7 + 3 is taken, so
    /// the ids ADD down the nesting path.
    fn argument_index(&self, node: &str) -> u32 {
        self.argument_base
            .saturating_add(super::location_index(node, self.field_ordinal))
    }

    /// True when this member is a C array of RESOURCE HANDLES, each of which is its own argument id
    /// and its own [`HANDLE_BYTES`] slot. An `air.indirect_constant` array is stored inline and is
    /// one span of data, not N resources, so it is not one of these.
    fn repeats_a_handle(&self, node: &str) -> bool {
        self.array_len > 1
            && role_strings(node)
                .iter()
                .any(|role| EMBEDDED_RESOURCE_ROLES.contains(&role.as_str()))
    }
}

/// The size of one opaque resource handle in an argument buffer. Every consumer of an embedded
/// resource -- the Metal argument encoder, the Vulkan descriptor writer and `corpus-case-check` --
/// already derives an element from `field_offset` by this stride, so it is stated once here.
const HANDLE_BYTES: u32 = 8;

/// Metal's own argument-buffer nesting limit is far below this; the cap only stops a malformed node
/// graph whose `air.struct_type_info` reaches itself from recursing forever.
const MAX_ARGUMENT_NESTING: u32 = 8;

/// Yield every `air.indirect_argument` resource member of an `air.struct_type_info` node, descending
/// into the nested wrappers Metal allows around one.
///
/// The member grammar itself is [`member_tuples`]; this walk only decides what each tuple means.
/// A tuple whose `air.indirect_argument` suffix is a bare `i32` beside a nested member list is a
/// wrapper, and that `i32` is the wrapper's own argument id (see [`ArgumentMember::argument_index`]).
fn embedded_argument_members(nodes: &HashMap<u32, String>, sref: u32) -> Vec<ArgumentMember> {
    let mut out = vec![];
    collect_argument_members(nodes, sref, Nesting::default(), &mut out);
    out
}

/// What the enclosing `air.struct_type_info` wrappers contribute to every member inside them.
/// `Nesting::default()` is the argument buffer's own member list.
#[derive(Clone, Copy, Debug, Default)]
struct Nesting {
    field_ordinal: u32,
    field_offset: u32,
    argument_base: u32,
    depth: u32,
    in_array_wrapper: bool,
}

fn collect_argument_members(
    nodes: &HashMap<u32, String>,
    sref: u32,
    enclosing: Nesting,
    out: &mut Vec<ArgumentMember>,
) {
    if enclosing.depth > MAX_ARGUMENT_NESTING {
        return;
    }
    let Some(body) = nodes.get(&sref) else {
        return;
    };
    for (field_ordinal, member) in member_tuples(&tokenize(body)).into_iter().enumerate() {
        // A wrapper's members keep the OUTERMOST ordinal, which is the GEP index that reaches them,
        // so a flat member's ordinal is exactly what it was before nesting was walked at all.
        let outer_ordinal = if enclosing.depth == 0 {
            field_ordinal as u32
        } else {
            enclosing.field_ordinal
        };
        match (member.nested, member.argument_wrapper_id) {
            (Some(wrapper), Some(id)) => collect_argument_members(
                nodes,
                wrapper,
                Nesting {
                    field_ordinal: outer_ordinal,
                    field_offset: enclosing.field_offset + member.offset,
                    argument_base: enclosing.argument_base + id,
                    depth: enclosing.depth + 1,
                    in_array_wrapper: enclosing.in_array_wrapper || member.array_len != 0,
                },
                out,
            ),
            _ => {
                if let Some(node_ref) = member.argument_node {
                    out.push(ArgumentMember {
                        field_ordinal: outer_ordinal,
                        field_offset: enclosing.field_offset + member.offset,
                        node_ref,
                        argument_base: enclosing.argument_base,
                        depth: enclosing.depth,
                        in_array_wrapper: enclosing.in_array_wrapper,
                        array_len: member.array_len,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::TextureFormat;

    #[test]
    fn embedded_texture_use_gate_includes_sampling_and_gathering() {
        assert!(body_uses_texture_intrinsic(
            "call <4 x float> @air.sample_texture_2d(...)"
        ));
        assert!(body_uses_texture_intrinsic(
            "call <4 x float> @air.gather_texture_2d(...)"
        ));
        assert!(!body_uses_texture_intrinsic(
            "declare void @unrelated_texture_helper()"
        ));
    }

    #[test]
    fn embedded_texture_detection_includes_read_and_write_fields() {
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 8, i32 0, !"texture2d<float, read>", !"input", !"air.indirect_argument", !20, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"output", !"air.indirect_argument", !21"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>""#.to_string(),
        );
        nodes.insert(
            21,
            r#"i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>""#.to_string(),
        );

        let textures =
            detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(4, 0, 10)]), &[]);
        assert_eq!(textures.len(), 2);
        assert_eq!(textures[0].buffer_param_index, 4);
        assert_eq!(textures[0].field_offset, 0);
        assert_eq!(textures[0].argument_index, 0);
        assert_eq!(textures[0].storage_format, None);
        assert!(!textures[0].arrayed);
        assert_eq!(textures[0].synthetic_texture_index, 0);
        assert_eq!(textures[1].field_offset, 8);
        assert_eq!(textures[1].argument_index, 1);
        assert_eq!(textures[1].storage_format, Some(TextureFormat::R32f));
        assert_eq!(textures[1].array_length, None);
        assert!(!textures[1].arrayed);
        assert_eq!(textures[1].synthetic_texture_index, 1);
    }

    #[test]
    fn embedded_fixed_depth_array_preserves_handle_count_and_image_shape() {
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 16, i32 0, !"array<depth2d_array<float, sample>, 2>", !"depth", !"air.indirect_argument", !20"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 4, i32 2, !"air.sample", !"air.arg_type_name", !"array<depth2d_array<float, sample>, 2>""#.to_string(),
        );

        let textures =
            detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(1, 16, 10)]), &[]);
        assert_eq!(textures.len(), 1);
        assert_eq!(textures[0].argument_index, 4);
        assert_eq!(textures[0].array_length, Some(2));
        assert!(textures[0].arrayed);
        assert_eq!(textures[0].storage_format, None);
    }

    #[test]
    fn embedded_c_array_member_takes_its_length_from_the_member_tuple() {
        // `texture2d<float, sample> slices[32]` inside an argument buffer: the type name is the
        // ELEMENT type and the node's count operand stays 1, so only the tuple's `array_len` slot
        // says 32. Reading the name alone surfaced one descriptor and let a runtime index sample
        // element 0.
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 8, i32 32, !"texture2d<float, sample>", !"slices", !"air.indirect_argument", !20"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"slices""#.to_string(),
        );

        let textures =
            detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(5, 1, 10)]), &[]);
        assert_eq!(textures.len(), 1);
        assert_eq!(textures[0].array_length, Some(32));
        assert_eq!(textures[0].argument_index, 0);
        assert!(!textures[0].arrayed);
        // Expressible, so it is not also reported as a resource with no descriptor.
        assert!(
            unsurfaced_embedded_resources(&nodes, &ArgumentBuffers::new(vec![(5, 1, 10)]))
                .is_empty()
        );
    }

    #[test]
    fn a_c_array_of_buffer_handles_surfaces_one_resource_per_element() {
        // `device half4* bufs[2] [[id(12)]]` -- two device addresses, at consecutive argument ids
        // and consecutive 8-byte slots. Surfacing only the first left the second slot of the
        // argument buffer unwritten, so a runtime index into the member read an address nothing
        // ever populated.
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 8, i32 2, !"half4", !"bufs", !"air.indirect_argument", !20, i32 16, i32 4, i32 4, !"uint", !"counts", !"air.indirect_argument", !21"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.buffer", !"air.location_index", i32 12, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"bufs""#.to_string(),
        );
        // An inline `uint counts[4]` is one span of DATA, not four resources.
        nodes.insert(
            21,
            r#"i32 1, !"air.indirect_constant", !"air.location_index", i32 14, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"counts""#.to_string(),
        );

        let arguments = detect_embedded_arguments(&nodes, &ArgumentBuffers::new(vec![(0, 3, 10)]));
        let coordinates = arguments
            .iter()
            .map(|argument| {
                (
                    argument.field_offset,
                    argument.argument_index,
                    argument.resource_buffer_index,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            coordinates,
            vec![(0, 12, Some(12)), (8, 13, Some(13)), (16, 14, None)]
        );
    }

    #[test]
    fn embedded_argument_detection_classifies_only_nested_buffers_as_device_resources() {
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !20, i32 8, i32 8, i32 0, !"float", !"values", !"air.indirect_argument", !21"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.indirect_constant", !"air.location_index", i32 2, i32 1"#.to_string(),
        );
        nodes.insert(
            21,
            r#"i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.address_space", i32 1"#.to_string(),
        );

        let arguments = detect_embedded_arguments(&nodes, &ArgumentBuffers::new(vec![(3, 5, 10)]));
        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[0].resource_buffer_index, None);
        assert_eq!(arguments[1].buffer_param_index, 3);
        assert_eq!(arguments[1].buffer_index, 5);
        assert_eq!(arguments[1].field_offset, 8);
        assert_eq!(arguments[1].argument_index, 7);
        assert_eq!(arguments[1].resource_buffer_index, Some(7));
        assert_eq!(arguments[1].resource_address_space, Some(1));
    }

    /// A wrapper's members carry an `air.location_index` RELATIVE to the wrapper's own id, so the
    /// Metal `[[id(n)]]` adds down the nesting path -- 30 + 0 + 2 for a handle two wrappers deep.
    #[test]
    fn a_wrapped_texture_binds_at_the_summed_argument_id() {
        let nodes = HashMap::from([
            (
                10_u32,
                r#"!"air.struct_type_info", !11, i32 64, i32 8, i32 0, !"deep", !"d", !"air.indirect_argument", i32 30, i32 72, i32 8, i32 0, !"texture2d<float, sample>", !"flat", !"air.indirect_argument", !14"#.to_string(),
            ),
            (
                11_u32,
                r#"!"air.struct_type_info", !12, i32 0, i32 8, i32 0, !"inner", !"mid", !"air.indirect_argument", i32 0"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 8, i32 8, i32 0, !"texture2d<float, sample>", !"y", !"air.indirect_argument", !13"#.to_string(),
            ),
            (
                13_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
            (
                14_u32,
                r#"i32 1, !"air.texture", !"air.location_index", i32 40, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
        ]);
        let textures =
            detect_embedded_textures(&nodes, &ArgumentBuffers::new(vec![(0, 0, 10)]), &[]);
        assert_eq!(
            textures
                .iter()
                .map(|t| (t.field_ordinal, t.field_offset, t.argument_index))
                .collect::<Vec<_>>(),
            // The nested handle keeps the OUTER ordinal 0 (the first GEP index that reaches it) and
            // an ABSOLUTE byte offset 64 + 0 + 8; the flat sibling is untouched by the descent.
            vec![(0, 72, 32), (1, 72, 40)],
        );
    }

    /// A wrapper declared as an ARRAY of structs repeats its members once per element, at offsets
    /// and argument ids this walk does not enumerate. Binding element 0 alone would leave the rest
    /// reading black, so every member under it stays a refusal reason instead.
    #[test]
    fn an_arrayed_wrapper_is_refused_rather_than_bound_at_element_zero() {
        let nodes = HashMap::from([
            (
                10_u32,
                r#"!"air.struct_type_info", !11, i32 0, i32 48, i32 9, !"MaterialProperty", !"slot", !"air.indirect_argument", i32 0"#.to_string(),
            ),
            (
                11_u32,
                r#"i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !12"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
        ]);
        let buffers = ArgumentBuffers::new(vec![(0, 0, 10)]);
        assert_eq!(detect_embedded_textures(&nodes, &buffers, &[]), vec![]);
        assert_eq!(
            unsurfaced_embedded_resources(&nodes, &buffers),
            vec![
                "argument buffer parameter 0 holds an air.texture at byte offset 0 inside a nested \
                 struct member"
                    .to_string()
            ],
        );
    }

    /// A wrapped `air.buffer` is a device address a consumer has to write into the owning argument
    /// buffer, and nothing binds one, so it is named as unsurfaced rather than becoming a silent
    /// `EmbeddedArgument` beside the flat members.
    #[test]
    fn a_wrapped_buffer_is_named_unsurfaced_and_is_not_an_embedded_argument() {
        let nodes = HashMap::from([
            (
                10_u32,
                r#"!"air.struct_type_info", !11, i32 16, i32 8, i32 0, !"wrapper", !"w", !"air.indirect_argument", i32 4, i32 24, i32 8, i32 0, !"float", !"values", !"air.indirect_argument", !13"#.to_string(),
            ),
            (
                11_u32,
                r#"i32 0, i32 8, i32 0, !"float", !"inner", !"air.indirect_argument", !12"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.address_space", i32 1"#.to_string(),
            ),
            (
                13_u32,
                r#"i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.address_space", i32 1"#.to_string(),
            ),
        ]);
        let buffers = ArgumentBuffers::new(vec![(0, 0, 10)]);
        assert_eq!(
            detect_embedded_arguments(&nodes, &buffers)
                .iter()
                .map(|argument| (argument.field_offset, argument.resource_buffer_index))
                .collect::<Vec<_>>(),
            vec![(24, Some(7))],
        );
        assert_eq!(
            unsurfaced_embedded_resources(&nodes, &buffers),
            vec![
                "argument buffer parameter 0 holds an air.buffer at byte offset 16 inside a nested \
                 struct member"
                    .to_string()
            ],
        );
    }

    /// A nested struct with no `air.indirect_argument` suffix is an inline value, not a wrapped
    /// resource: the walk steps over it without disturbing the ordinals of the members around it.
    #[test]
    fn an_inline_nested_struct_does_not_shift_its_siblings() {
        let nodes = HashMap::from([
            (
                10_u32,
                r#"i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !11, !"air.struct_type_info", !12, i32 8, i32 16, i32 0, !"Params", !"params", i32 24, i32 8, i32 0, !"texture2d<float, sample>", !"tex", !"air.indirect_argument", !13"#.to_string(),
            ),
            (
                11_u32,
                r#"i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1"#.to_string(),
            ),
            (
                12_u32,
                r#"i32 0, i32 16, i32 0, !"float4", !"a""#.to_string(),
            ),
            (
                13_u32,
                r#"i32 2, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>""#.to_string(),
            ),
        ]);
        let buffers = ArgumentBuffers::new(vec![(0, 0, 10)]);
        assert!(unsurfaced_embedded_resources(&nodes, &buffers).is_empty());
        let textures = detect_embedded_textures(&nodes, &buffers, &[]);
        assert_eq!(textures.len(), 1);
        assert_eq!(textures[0].field_ordinal, 2);
        assert_eq!(textures[0].field_offset, 24);
        assert_eq!(textures[0].argument_index, 5);
    }

    /// The synthetic embedded-texture index is a function of the entry's SIGNATURE, not of the order
    /// the AIR metadata list names its argument-buffer nodes.
    ///
    /// The walk hands out `K`, `K+1`, ... in list order, and a consumer binds by that index, so
    /// listing the same two argument buffers the other way round used to move each embedded texture
    /// onto the other one's descriptor -- in one corpus kernel, from `Binding 32` (sampled) to
    /// `Binding 480` (storage). Parameter index orders them because it is unique per argument;
    /// the Metal `[[buffer(N)]]` slot is not, since mutually exclusive function-constant
    /// alternatives share one.
    #[test]
    fn the_synthetic_texture_index_does_not_depend_on_the_argument_list_order() {
        let nodes = HashMap::from([
            (
                10_u32,
                r#"i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"first", !"air.indirect_argument", !12"#
                    .to_string(),
            ),
            (
                11_u32,
                r#"i32 0, i32 8, i32 0, !"texture2d<float, write>", !"second", !"air.indirect_argument", !13"#
                    .to_string(),
            ),
            (
                12_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"first""#
                    .to_string(),
            ),
            (
                13_u32,
                r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"second""#
                    .to_string(),
            ),
        ]);
        // Parameter 0 binds `[[buffer(7)]]`, parameter 1 binds `[[buffer(2)]]`, so list order,
        // parameter order and Metal-slot order are three different orders.
        let declared = vec![(0_u32, 7_u32, 10_u32), (1, 2, 11)];
        let forward =
            detect_embedded_textures(&nodes, &ArgumentBuffers::new(declared.clone()), &[]);
        let reversed = detect_embedded_textures(
            &nodes,
            &ArgumentBuffers::new(declared.into_iter().rev().collect()),
            &[],
        );
        assert_eq!(forward, reversed);
        assert_eq!(
            forward
                .iter()
                .map(|texture| (texture.buffer_param_index, texture.synthetic_texture_index))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 1)],
        );
    }
}
