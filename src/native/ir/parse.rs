//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

#[derive(Default)]
pub(in crate::native) struct FunctionParseCache {
    types: HashMap<String, LlType>,
    functions: HashMap<String, ParsedFunction>,
    source_bytes: usize,
    #[cfg(test)]
    hits: usize,
}

struct ParsedFunction {
    header: String,
    body: Vec<String>,
    function: LlFunction,
    source_bytes: usize,
}

impl FunctionParseCache {
    fn set_types(&mut self, types: &HashMap<String, LlType>) {
        if self.types != *types {
            self.functions.clear();
            self.source_bytes = 0;
            self.types = types.clone();
        }
    }

    fn lower(
        &mut self,
        header: &str,
        body: &[&str],
        mut function: LlFunction,
        types: &HashMap<String, LlType>,
    ) -> Result<LlFunction, String> {
        if let Some(cached) = self.functions.get(&function.name) {
            if cached.header == header
                && cached.body.len() == body.len()
                && cached.body.iter().zip(body).all(|(a, b)| a == b)
            {
                #[cfg(test)]
                {
                    self.hits += 1;
                }
                return Ok(cached.function.clone());
            }
        }
        let entry = crate::native::cfg::implicit_entry_block_name(&function);
        function.blocks = crate::native::cfg::split_source_body_blocks(body, entry, types)?;
        if let Some(previous) = self.functions.remove(&function.name) {
            self.source_bytes -= previous.source_bytes;
        }
        // One translation owns this cache; retain at most one version of a
        // function and cap the source represented by shared typed carriers.
        let source_bytes = header.len().saturating_add(
            body.iter()
                .fold(0usize, |sum, line| sum.saturating_add(line.len())),
        );
        if self.functions.len() >= 256
            || self.source_bytes.saturating_add(source_bytes) > 4 * 1024 * 1024
        {
            return Ok(function);
        }
        self.source_bytes += source_bytes;
        self.functions.insert(
            function.name.clone(),
            ParsedFunction {
                header: header.to_owned(),
                body: body.iter().map(|line| (*line).to_owned()).collect(),
                function: function.clone(),
                source_bytes,
            },
        );
        Ok(function)
    }
}

impl LlModule {
    pub(in crate::native) fn parse(ll: &str) -> Result<Self, String> {
        let kern = meta::parse_air_kernel_meta(ll);
        let entry_name = meta::entry_name(ll, "kernel");
        Self::parse_inner(ll, false, kern.as_ref(), entry_name.as_deref())
    }

    /// Parse using stage metadata already owned by the translate path. Retry tiers re-emit the same
    /// AIR metadata with different buffer/CFG models, so reparsing that metadata inside every tier is
    /// pure waste and risks divergent inference. Direct native-emitter callers retain [`Self::parse`].
    pub(in crate::native) fn parse_with_stage_meta(
        ll: &str,
        kern: Option<&meta::KernMeta>,
        entry_name: Option<&str>,
    ) -> Result<Self, String> {
        Self::parse_inner(ll, false, kern, entry_name)
    }

    /// Parse with the exact primitive-metadata fallback enabled for a validation-gated re-emission.
    /// The normal emitter deliberately leaves this off: metadata alone is insufficient authority to
    /// change a primary module whose raw form has not first proved the relevant pointer-typing gap.
    #[cfg(test)]
    pub(in crate::native) fn parse_with_primitive_phi_metadata(ll: &str) -> Result<Self, String> {
        let kern = meta::parse_air_kernel_meta(ll);
        let entry_name = meta::entry_name(ll, "kernel");
        Self::parse_inner(ll, true, kern.as_ref(), entry_name.as_deref())
    }

    pub(in crate::native) fn parse_inner(
        ll: &str,
        primitive_phi_metadata: bool,
        kern: Option<&meta::KernMeta>,
        entry_name: Option<&str>,
    ) -> Result<Self, String> {
        Self::parse_cached(ll, primitive_phi_metadata, kern, entry_name, None)
    }

    pub(in crate::native) fn parse_cached(
        ll: &str,
        primitive_phi_metadata: bool,
        kern: Option<&meta::KernMeta>,
        entry_name: Option<&str>,
        mut cache: Option<&mut FunctionParseCache>,
    ) -> Result<Self, String> {
        let ray_lowered = crate::native::ray_intersection::lower_callback_free_triangle_queries(ll);
        let ll = ray_lowered.as_deref().unwrap_or(ll);
        let mut types = HashMap::new();
        // Named types may appear after a function that uses them. Collect the compact type table in
        // a first pass, then lower each function body directly into its typed carrier in a streaming
        // second pass. Only the current function's borrowed body lines are indexed; a generated module
        // never retains a whole-module line table beside all of its typed carriers.
        for raw_line in ll.lines() {
            let line = strip_comment(raw_line).trim();
            if line.starts_with('%') && line.contains(" = type ") {
                let (name, body) = line
                    .split_once(" = type ")
                    .ok_or_else(|| format!("native emitter: malformed type alias: {line}"))?;
                // `%struct._texture_2d_t = type opaque` declares a name whose body LLVM itself
                // does not know -- Metal's texture and sampler handles. Leaving it out of the
                // table is what keeps it opaque here too: a pointer to it still parses (the
                // pointee is dropped), and anything that reaches for the body gets an honest
                // "unknown named type" instead of a body we invented.
                if body.trim() == "opaque" {
                    continue;
                }
                types.insert(name.trim().to_string(), parse_type(body.trim())?);
            }
        }
        if let Some(cache) = cache.as_deref_mut() {
            cache.set_types(&types);
        }
        let mut functions = Vec::new();
        let mut declarations = Vec::new();
        let mut globals = Vec::new();
        let mut lines = ll.lines();
        while let Some(raw_line) = lines.next() {
            let line = strip_comment(raw_line).trim();
            if line.starts_with('%') && line.contains(" = type ") {
                continue;
            }
            if line.starts_with("define ") {
                let mut func = parse_function_header(line)?;
                let mut body = Vec::new();
                let mut terminated = false;
                for body_line in lines.by_ref() {
                    if strip_comment(body_line).trim() == "}" {
                        terminated = true;
                        break;
                    }
                    body.push(body_line);
                }
                if !terminated {
                    return Err(format!(
                        "native emitter: unterminated function {}",
                        func.name
                    ));
                }
                if let Some(cache) = cache.as_deref_mut() {
                    func = cache.lower(line, &body, func, &types)?;
                } else {
                    let entry = crate::native::cfg::implicit_entry_block_name(&func);
                    func.blocks =
                        crate::native::cfg::split_source_body_blocks(&body, entry, &types)?;
                }
                functions.push(func);
                continue;
            }
            if line.starts_with("declare ") {
                let decl = parse_declaration(line)?;
                if !is_ignored_intrinsic(&decl.name) {
                    declarations.push(decl);
                }
                continue;
            }
            if is_ignored_global(line) {
                continue;
            }
            if line.starts_with('@') && (line.contains(" constant ") || line.contains(" global ")) {
                globals.push(parse_global(line)?);
                continue;
            }
        }
        if functions.is_empty() {
            return Err("native emitter: no function definitions found".into());
        }
        if let Some(cache) = cache {
            let live = functions
                .iter()
                .map(|function| function.name.as_str())
                .collect::<HashSet<_>>();
            cache.functions.retain(|name, function| {
                if live.contains(name.as_str()) {
                    true
                } else {
                    cache.source_bytes -= function.source_bytes;
                    false
                }
            });
        }
        // Every body reader below consumes the typed carriers. Each borrowed function-body index was
        // released immediately after that function was lowered.
        let entry_functions = entry_name
            .map(|name| HashSet::from([name.to_string()]))
            .unwrap_or_else(|| infer_entry_functions(ll));
        let metadata_byte_buffer_params =
            infer_metadata_byte_buffer_params(kern, entry_name, &functions);
        let metadata_data_buffer_params =
            infer_metadata_data_buffer_params(kern, entry_name, &functions);
        let metadata_primitive_buffer_pointees =
            infer_metadata_primitive_buffer_pointees(kern, entry_name, &functions);
        let metadata_fc_buffer_locations =
            infer_metadata_fc_buffer_locations(kern, entry_name, &functions);
        let imageblock_dimensions = infer_apv_imageblock_dimensions(ll);
        let cross_coordinate_imageblock =
            infer_cross_coordinate_imageblock(&functions, &entry_functions);
        let imageblock_threads_per_threadgroup_param = kern
            .and_then(|meta| {
                meta.roles.iter().find_map(|(index, role)| {
                    matches!(role, KernRole::ThreadsPerThreadgroup).then_some(*index as usize)
                })
            })
            .and_then(|index| {
                functions
                    .iter()
                    .find(|function| entry_functions.contains(&function.name))
                    .and_then(|function| function.params.get(index))
                    .map(|(name, _)| name.clone())
            });
        // Whether the tile is shared is a property of the AIR, not of the stride the emitter can
        // reach for. Conjoining the `[[threads_per_threadgroup]]` parameter here answered "is this
        // imageblock shared?" with "can I linearize it?", and a kernel that addresses a second
        // coordinate without that parameter fell through to per-invocation Private staging, where
        // every coordinate resolves to the same private slot. `emit_imageblock_threadgroup_width`
        // already refuses the missing stride; let it.
        // An `air.alias_implicit_imageblock` explicit imageblock is not tile-local scratch at all:
        // its storage IS the implicit imageblock, so each cell is one render-target texel and the
        // tile is shared with whatever rasterized into it. Per-invocation staging, where every
        // coordinate resolves to one Private slot, cannot represent that however few coordinates
        // the entry happens to name -- so an aliased block takes the shared-cell path
        // unconditionally, and `aliased_imageblock` fills its cells from the planes and writes
        // them back.
        let aliased_imageblock_planes = kern
            .and_then(|meta| {
                meta.aliased_implicit_imageblock_params
                    .first()
                    .and_then(|param| meta.aliased_implicit_imageblock_planes.get(param))
            })
            .cloned()
            .unwrap_or_default();
        let imageblock_shared_cells = cross_coordinate_imageblock
            || !aliased_imageblock_planes.is_empty()
            || calls_imageblock_slice_write(&functions, &entry_functions);
        let imageblock_cell_scale = infer_imageblock_cell_scale(&functions, &entry_functions);
        // A shared tile is linearised `y * width + x`, and nothing in AIR states that width: it is
        // the threadgroup extent, a dispatch-time fact. A kernel that takes
        // `[[threads_per_threadgroup]]` hands it to us directly; one that does not used to leave
        // the emitter with no stride at all. But the pass layer already answers
        // `air.get_imageblock_width` with `TransformOptions::kernel_local_size` -- the same number,
        // from the one place that knows it. Declare that intrinsic so the emitter can ask the
        // question instead of inventing a second derivation of the tile extent.
        if imageblock_shared_cells
            && imageblock_dimensions.is_none()
            && imageblock_threads_per_threadgroup_param.is_none()
            && !declarations
                .iter()
                .any(|declaration| declaration.name == IMAGEBLOCK_WIDTH_INTRINSIC)
        {
            declarations.push(LlDeclaration {
                name: IMAGEBLOCK_WIDTH_INTRINSIC.to_string(),
                ret: LlType::Int(32),
                params: Vec::new(),
            });
        }
        // Private imageblock scratch normally keeps only its first metadata member, because a
        // single-coordinate slice never addresses another field. A byte GEP rooted at
        // `air.imageblock_data` proves that this module does address a later field, though; retain
        // the complete cell so the emitter can form that field through a typed access chain rather
        // than an illegal Logical-pointer reinterpret.
        let imageblock_nonzero_byte_field =
            infer_imageblock_nonzero_byte_field(&functions, &entry_functions);
        let imageblock_data_pointee = infer_imageblock_data_pointee(
            kern,
            imageblock_dimensions.is_some()
                || imageblock_shared_cells
                || imageblock_nonzero_byte_field,
        );
        let mut module = Self {
            air_data_layout: crate::layout::AirDataLayout::from_ir(ll)?,
            types,
            functions,
            declarations,
            globals,
            static_init_globals: meta::static_init_foldable_global_values(ll),
            entry_name: entry_name.map(str::to_string),
            preinlined_static_initializers: HashSet::new(),
            preinlined_helper_pointer_loads: HashSet::new(),
            preinlined_helper_type_capabilities: HashSet::new(),
            entry_functions,
            ptr_pointees: HashMap::new(),
            local_alloca_pointees: HashMap::new(),
            imageblock_data_pointee,
            imageblock_dimensions,
            imageblock_shared_cells,
            aliased_imageblock_planes,
            imageblock_threads_per_threadgroup_param,
            imageblock_cell_scale,
            metadata_pointee_params: HashSet::new(),
            metadata_pointee_sizes: HashMap::new(),
            metadata_byte_buffer_params,
            metadata_data_buffer_params,
            metadata_primitive_buffer_pointees,
            metadata_fc_buffer_locations,
            raw_buffer_params: HashSet::new(),
            call_connected_raw_params: HashSet::new(),
            param_connected_raw_params: HashSet::new(),
        };
        module.infer_metadata_buffer_pointees(kern, entry_name);
        module.infer_pointer_pointees();
        if primitive_phi_metadata {
            module.infer_metadata_primitive_buffer_pointees(kern, entry_name);
        }
        module.infer_local_alloca_pointees();
        module.infer_raw_buffer_params();
        module.infer_call_connected_raw_buffer_params();
        module.propagate_raw_buffer_params();
        module.propagate_data_buffer_params();
        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_reuses_only_unchanged_function_carriers_and_type_environment() {
        let original = "%T = type { i32 }\n\
            define i32 @helper(i32 %x) {\n %y = add i32 %x, 1\n ret i32 %y\n}\n\
            define i32 @main(i32 %x) {\n %y = call i32 @helper(i32 %x)\n ret i32 %y\n}\n";
        let mut cache = FunctionParseCache::default();
        let first = LlModule::parse_cached(original, false, None, Some("main"), Some(&mut cache))
            .expect("initial parse");
        let changed = original.replace("call i32 @helper(i32 %x)", "add i32 %x, 1");
        let next = LlModule::parse_cached(&changed, false, None, Some("main"), Some(&mut cache))
            .expect("rewritten parse");
        assert_eq!(cache.hits, 1);
        assert!(std::sync::Arc::ptr_eq(
            first.functions[0].blocks[0].typed.as_ref().unwrap(),
            next.functions[0].blocks[0].typed.as_ref().unwrap(),
        ));
        assert!(!std::sync::Arc::ptr_eq(
            first.functions[1].blocks[0].typed.as_ref().unwrap(),
            next.functions[1].blocks[0].typed.as_ref().unwrap(),
        ));
        let changed_types = changed.replace("%T = type { i32 }", "%T = type { i64 }");
        let third =
            LlModule::parse_cached(&changed_types, false, None, Some("main"), Some(&mut cache))
                .expect("changed type parse");
        assert_eq!(cache.hits, 1, "named type changes invalidate every carrier");
        assert!(!std::sync::Arc::ptr_eq(
            next.functions[0].blocks[0].typed.as_ref().unwrap(),
            third.functions[0].blocks[0].typed.as_ref().unwrap(),
        ));
    }

    #[test]
    fn threaded_stage_meta_preserves_kernel_inference() {
        let ll = r#"
define void @k(ptr addrspace(1) %input) {
entry:
  %value = load float, ptr addrspace(1) %input, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
"#;
        let reparsed = LlModule::parse(ll).expect("ordinary parse");
        let kern = meta::parse_air_kernel_meta(ll);
        let threaded =
            LlModule::parse_with_stage_meta(ll, kern.as_ref(), Some("k")).expect("threaded parse");

        assert_eq!(threaded.entry_functions, reparsed.entry_functions);
        assert_eq!(threaded.ptr_pointees, reparsed.ptr_pointees);
        assert_eq!(
            threaded.metadata_pointee_params,
            reparsed.metadata_pointee_params
        );
        assert_eq!(
            threaded.metadata_pointee_sizes,
            reparsed.metadata_pointee_sizes
        );
        assert_eq!(
            threaded.metadata_byte_buffer_params,
            reparsed.metadata_byte_buffer_params
        );
        assert_eq!(
            threaded.metadata_data_buffer_params,
            reparsed.metadata_data_buffer_params
        );
        assert_eq!(
            threaded.metadata_primitive_buffer_pointees,
            reparsed.metadata_primitive_buffer_pointees
        );
        assert_eq!(
            threaded.call_connected_raw_params,
            reparsed.call_connected_raw_params
        );
        assert_eq!(
            threaded.param_connected_raw_params,
            reparsed.param_connected_raw_params
        );
        assert_eq!(
            threaded.imageblock_data_pointee,
            reparsed.imageblock_data_pointee
        );
        assert_eq!(
            threaded.imageblock_threads_per_threadgroup_param,
            reparsed.imageblock_threads_per_threadgroup_param
        );
    }
}
