//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn drop_unmodeled_memcpy(&self, call: &LlCall) -> bool {
        call.callee.starts_with("llvm.memcpy.")
            && call.args.iter().take(2).any(|arg| match &arg.value {
                LlValue::Local(name) => self.unmodeled_pointers.contains(name),
                _ => false,
            })
    }

    pub(in crate::native::emitter) fn emit_raw_memcpy(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !call.callee.starts_with("llvm.memcpy.") {
            return Ok(false);
        }
        if call.args.len() != 4 || !matches!(call.args[3].value, LlValue::Bool(false)) {
            return Ok(false);
        }
        let Some(len) = typed_value_u64(&call.args[2]) else {
            return Ok(false);
        };
        if len == 0 {
            return Ok(false);
        }
        let (LlValue::Local(dst_name), LlValue::Local(src_name)) =
            (&call.args[0].value, &call.args[1].value)
        else {
            return Ok(false);
        };
        let dst_raw = self.raw_offsets.get(dst_name).cloned();
        let src_raw = self.raw_offsets.get(src_name).cloned();
        if let (Some(dst), Some(src)) = (dst_raw.as_ref(), src_raw.as_ref()) {
            if self.raw_pointer_word_aligned(dst)
                && self.raw_pointer_word_aligned(src)
                && len % 4 == 0
            {
                for byte in (0..len).step_by(4) {
                    let word = self.emit_raw_word_load(src, byte, instructions)?;
                    self.emit_raw_word_store(dst, byte, word, instructions)?;
                }
            } else {
                for byte in 0..len {
                    let value = self.emit_raw_byte_load_as_u32(src, byte, instructions)?;
                    self.emit_raw_byte_store_from_u32(value, dst, byte, instructions)?;
                }
            }
            return Ok(true);
        }

        if let Some(dst) = dst_raw {
            let dst_align = call.arg_aligns.first().copied().flatten();
            // A destination with no SPIR-V pointer value of its own has exactly one alternative to
            // this decomposition: `emit_typed_memcpy` copies the whole object into the `Private`
            // placeholder that stands in for it, which writes the buffer nothing at all. Let the
            // decomposition leave the source layout's inter-member padding alone rather than
            // decline, since `llvm.memcpy` of an aggregate leaves those bytes undefined anyway --
            // the surrounding shader writes the same struct member by member and never writes them.
            let leave_source_padding = self.unmodeled_pointers.contains(dst_name);
            if self.emit_typed_to_raw_memcpy(
                &dst,
                dst_align,
                &call.args[1],
                len,
                leave_source_padding,
                instructions,
            )? {
                return Ok(true);
            }
            return self.emit_raw_byte_run_memcpy(
                &dst,
                dst_align,
                &call.args[1],
                len,
                true,
                instructions,
            );
        }

        if let Some(src) = src_raw {
            if !self.raw_pointer_word_aligned(&src) {
                return Ok(false);
            }
            if let Some((dst_root, dst_addrspace, dst_base)) =
                self.byte_gep_root_and_const_offset(&call.args[0])?
            {
                if dst_base < 0 || dst_base as u64 > u32::MAX as u64 - len {
                    return Ok(false);
                }
                let dst_ptr_ty =
                    self.ptr_type_id(llvm_pointer_storage(dst_addrspace)?, &LlType::Int(8))?;
                for byte in (0..len).step_by(4) {
                    let word = self.emit_raw_word_load(&src, byte, instructions)?;
                    let dst_ptr = self.fresh();
                    let byte_offset = self.const_uint(dst_base as u32 + byte as u32)?;
                    instructions.push(Self::inst(
                        Op::InBoundsAccessChain,
                        Some(dst_ptr_ty),
                        Some(dst_ptr),
                        vec![Operand::IdRef(dst_root), Operand::IdRef(byte_offset)],
                    ));
                    instructions.push(Self::inst(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(dst_ptr), Operand::IdRef(word)],
                    ));
                }
                return Ok(true);
            }
            if self.emit_raw_to_typed_struct_memcpy(&call.args[0], &src, len, instructions)? {
                return Ok(true);
            }
            let dst_align = call.arg_aligns.first().copied().flatten();
            return self.emit_raw_byte_run_memcpy(
                &src,
                dst_align,
                &call.args[0],
                len,
                false,
                instructions,
            );
        }
        Ok(false)
    }

    /// Copy a run of BYTES between a raw byte cursor and a typed pointer that names one `i8`
    /// element of a longer local object -- `memcpy(&blob[0], &buffer_field, n)`, where the local is
    /// an `[n x i8]` staging alloca. Both whole-object decompositions see only the single byte the
    /// typed pointer names and decline, and the call then reached `drop_unmodeled_memcpy`, which
    /// discarded it: a copy of a real descriptor-backed buffer that simply did not happen.
    ///
    /// A byte at a time would be correct and unaffordable -- a subword store into a device buffer is
    /// an atomic read-modify-write loop. Group four consecutive `i8` lanes into one word instead, so
    /// the raw side keeps its plain word access. That needs the length to be a whole number of words
    /// and the cursor to be word-aligned; anything else declines, exactly as before.
    fn emit_raw_byte_run_memcpy(
        &mut self,
        raw: &RawBufferOffset,
        raw_align: Option<u64>,
        typed: &TypedValue,
        len: u64,
        raw_is_destination: bool,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !len.is_multiple_of(4) || !self.raw_pointer_word_aligned(raw) {
            return Ok(false);
        }
        staged_emit(instructions, |instructions| {
            let Some((element, count)) = self.gep_element_run(typed, len)? else {
                return Ok(false);
            };
            if self.resolve_type(&element)? != LlType::Int(8) || u64::from(count) != len {
                return Ok(false);
            }
            let uint_ty = self.type_id(&LlType::Int(32))?;
            let byte_ty = self.type_id(&LlType::Int(8))?;
            for word in 0..(len / 4) {
                let mut lane_ptrs = Vec::with_capacity(4);
                for lane in 0..4 {
                    let index = u32::try_from(word * 4 + lane).map_err(|_| {
                        "native emitter: byte-run memcpy lane index overflows".to_string()
                    })?;
                    let Some(ptr) =
                        self.emit_gep_provenance_lane_ptr(typed, &element, index, instructions)?
                    else {
                        return Ok(false);
                    };
                    lane_ptrs.push(ptr);
                }
                if raw_is_destination {
                    let mut assembled = None;
                    for (lane, ptr) in lane_ptrs.into_iter().enumerate() {
                        let byte = self.fresh();
                        instructions.push(Self::inst(
                            Op::Load,
                            Some(byte_ty),
                            Some(byte),
                            vec![Operand::IdRef(ptr)],
                        ));
                        let widened = self.fresh();
                        instructions.push(Self::inst(
                            Op::UConvert,
                            Some(uint_ty),
                            Some(widened),
                            vec![Operand::IdRef(byte)],
                        ));
                        let placed = if lane == 0 {
                            widened
                        } else {
                            let shift = self.const_uint(lane as u32 * 8)?;
                            let shifted = self.fresh();
                            instructions.push(Self::inst(
                                Op::ShiftLeftLogical,
                                Some(uint_ty),
                                Some(shifted),
                                vec![Operand::IdRef(widened), Operand::IdRef(shift)],
                            ));
                            shifted
                        };
                        assembled = Some(match assembled {
                            None => placed,
                            Some(previous) => {
                                let merged = self.fresh();
                                instructions.push(Self::inst(
                                    Op::BitwiseOr,
                                    Some(uint_ty),
                                    Some(merged),
                                    vec![Operand::IdRef(previous), Operand::IdRef(placed)],
                                ));
                                merged
                            }
                        });
                    }
                    let Some(assembled) = assembled else {
                        return Ok(false);
                    };
                    self.emit_raw_word_store_for_access(
                        raw,
                        word * 4,
                        assembled,
                        raw_align,
                        instructions,
                    )?;
                } else {
                    let value = self.emit_raw_word_load(raw, word * 4, instructions)?;
                    for (lane, ptr) in lane_ptrs.into_iter().enumerate() {
                        let placed = if lane == 0 {
                            value
                        } else {
                            let shift = self.const_uint(lane as u32 * 8)?;
                            let shifted = self.fresh();
                            instructions.push(Self::inst(
                                Op::ShiftRightLogical,
                                Some(uint_ty),
                                Some(shifted),
                                vec![Operand::IdRef(value), Operand::IdRef(shift)],
                            ));
                            shifted
                        };
                        let narrowed = self.fresh();
                        instructions.push(Self::inst(
                            Op::UConvert,
                            Some(byte_ty),
                            Some(narrowed),
                            vec![Operand::IdRef(placed)],
                        ));
                        instructions.push(Self::inst(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(ptr), Operand::IdRef(narrowed)],
                        ));
                    }
                }
            }
            Ok(true)
        })
    }

    /// Store a typed aggregate through a raw byte cursor, one 32-bit word per leaf.
    ///
    /// `leave_source_padding` admits a word set that does not cover every word of `len`. The walk
    /// visits every leaf of the source type at its own layout offset, so a gap is inter-member
    /// padding and never data; leaving those destination bytes unchanged is one of the undefined
    /// values `llvm.memcpy` of an aggregate may leave there.
    pub(in crate::native::emitter) fn emit_typed_to_raw_memcpy(
        &mut self,
        dst: &RawBufferOffset,
        dst_align: Option<u64>,
        src: &TypedValue,
        len: u64,
        leave_source_padding: bool,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            let Some(src_pointee) = self.pointer_pointee_for_value(&src.value)? else {
                return Ok(false);
            };
            let src_pointee = function_storage_local_type(&self.resolve_type(&src_pointee)?);
            let (src_size, _) = self.raw_type_size_align(&src_pointee)?;
            if len > src_size {
                return Ok(false);
            }
            let src_ptr = self.value_id(&src.value, &src.ty)?;
            let src_ty = self.type_id(&src_pointee)?;
            let src_value = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(src_ty),
                Some(src_value),
                vec![Operand::IdRef(src_ptr)],
            ));

            let mut words = Vec::new();
            if !self.collect_typed_memcpy_words(
                src_value,
                &src_pointee,
                0,
                len,
                &mut words,
                instructions,
            )? {
                return Ok(false);
            }
            words.sort_by_key(|(offset, _)| *offset);
            if words.len() != (len / 4) as usize {
                match self.typed_memcpy_words_with_raw_padding(src, len, words.clone())? {
                    Some(words_with_padding) => words = words_with_padding,
                    None if leave_source_padding => {}
                    None => return Ok(false),
                }
            }
            let mut previous = None;
            for (offset, word) in words {
                if !offset.is_multiple_of(4)
                    || offset + 4 > len
                    || previous.is_some_and(|previous| previous >= offset)
                {
                    return Ok(false);
                }
                previous = Some(offset);
                self.emit_raw_word_store_for_access(dst, offset, word, dst_align, instructions)?;
            }
            Ok(true)
        })
    }

    /// Copy a `memcpy` that spans MORE than the object either pointer names: `memcpy(&a[0], &b[k],
    /// n * sizeof(T))` names one element on each side and copies `n`. The whole-object path below is
    /// short by `n - 1` elements and says nothing about it — a 296-byte copy through two `i8` element
    /// pointers came out as a single-byte `OpCopyMemory`. Walk the run through both pointers' own GEP
    /// provenance, the same element-striding primitive [`Self::emit_element_run_zero_memset`] uses.
    ///
    /// Returns `false` — and `emit_typed_memcpy` then declines the call entirely, which hands it to
    /// `drop_unmodeled_memcpy` and the retry tiers — when either side has no element-striding
    /// provenance, when the two element types disagree, when the length does not divide evenly, or
    /// when the element is a pointer. Declining is the point: the four corpus modules whose source
    /// pointer is an unmodelled `Private` placeholder reach `drop_unmodeled_memcpy`, which is where
    /// they belonged, instead of copying one byte out of a variable that addresses nothing.
    fn emit_element_run_memcpy(
        &mut self,
        dst: &TypedValue,
        src: &TypedValue,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some((dst_element, count)) = self.gep_element_run(dst, len)? else {
            return Ok(false);
        };
        let Some((src_element, src_count)) = self.gep_element_run(src, len)? else {
            return Ok(false);
        };
        if count != src_count || !types_compatible(&dst_element, &src_element) {
            return Ok(false);
        }
        let LlType::Ptr(dst_addrspace) = self.resolve_type(&dst.ty)? else {
            return Ok(false);
        };
        let dst_storage = self.pointer_storage_for(&dst.value, dst_addrspace)?;
        let mut staged = Vec::new();
        for lane in 0..count {
            let Some(dst_ptr) =
                self.emit_gep_provenance_lane_ptr(dst, &dst_element, lane, &mut staged)?
            else {
                return Ok(false);
            };
            let Some(src_ptr) =
                self.emit_gep_provenance_lane_ptr(src, &src_element, lane, &mut staged)?
            else {
                return Ok(false);
            };
            self.emit_copy_memory(dst_ptr, src_ptr, dst_storage, &dst_element, &mut staged)?;
        }
        instructions.extend(staged);
        Ok(true)
    }

    pub(in crate::native::emitter) fn typed_memcpy_words_with_raw_padding(
        &self,
        src: &TypedValue,
        len: u64,
        words: Vec<(u64, Word)>,
    ) -> Result<Option<Vec<(u64, Word)>>, String> {
        if !len.is_multiple_of(4) {
            return Ok(None);
        }
        let LlValue::Local(src_name) = &src.value else {
            return Ok(None);
        };
        let Some(shadow) = self.raw_memcpy_shadows.get(src_name) else {
            return Ok(None);
        };
        let mut by_offset = HashMap::new();
        for (offset, word) in words {
            if offset % 4 != 0 || offset >= len {
                return Ok(None);
            }
            by_offset.insert(offset, word);
        }
        for (offset, word) in shadow {
            if *offset % 4 != 0 || *offset >= len {
                continue;
            }
            by_offset.entry(*offset).or_insert(*word);
        }
        let expected_words = (len / 4) as usize;
        if by_offset.len() != expected_words {
            return Ok(None);
        }
        let mut words = Vec::with_capacity(expected_words);
        for index in 0..expected_words {
            let offset = (index as u64) * 4;
            let Some(word) = by_offset.get(&offset).copied() else {
                return Ok(None);
            };
            words.push((offset, word));
        }
        Ok(Some(words))
    }

    pub(in crate::native::emitter) fn collect_typed_memcpy_words(
        &mut self,
        value: Word,
        ty: &LlType,
        base_offset: u64,
        len: u64,
        words: &mut Vec<(u64, Word)>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            if base_offset >= len {
                return Ok(true);
            }
            match self.resolve_type(ty)? {
                LlType::Int(32) | LlType::Float => {
                    if base_offset + 4 > len {
                        return Ok(false);
                    }
                    let word = self.typed_memcpy_word(value, ty, instructions)?;
                    words.push((base_offset, word));
                    Ok(true)
                }
                LlType::Vector(elem, lanes) => {
                    if !matches!(self.resolve_type(&elem)?, LlType::Int(32) | LlType::Float) {
                        return Ok(false);
                    }
                    for lane in 0..lanes {
                        let offset = base_offset + (lane as u64) * 4;
                        if offset >= len {
                            break;
                        }
                        if offset + 4 > len {
                            return Ok(false);
                        }
                        let elem_ty = self.resolve_type(&elem)?;
                        let elem_type = self.type_id(&elem_ty)?;
                        let elem_value = self.fresh();
                        instructions.push(Self::inst(
                            Op::CompositeExtract,
                            Some(elem_type),
                            Some(elem_value),
                            vec![Operand::IdRef(value), Operand::LiteralBit32(lane)],
                        ));
                        let word = self.typed_memcpy_word(elem_value, &elem_ty, instructions)?;
                        words.push((offset, word));
                    }
                    Ok(true)
                }
                LlType::Array(elem, count) => {
                    let elem = self.resolve_type(&elem)?;
                    let (elem_size, elem_align) = self.raw_type_size_align(&elem)?;
                    let stride = round_up_u64(elem_size, elem_align);
                    let elem_type = self.type_id(&elem)?;
                    for index in 0..count {
                        let offset = base_offset + stride * index as u64;
                        if offset >= len {
                            break;
                        }
                        let elem_value = self.fresh();
                        instructions.push(Self::inst(
                            Op::CompositeExtract,
                            Some(elem_type),
                            Some(elem_value),
                            vec![Operand::IdRef(value), Operand::LiteralBit32(index)],
                        ));
                        if !self.collect_typed_memcpy_words(
                            elem_value,
                            &elem,
                            offset,
                            len,
                            words,
                            instructions,
                        )? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                LlType::Struct(fields) => {
                    for index in 0..fields.len() {
                        let (offset, field) = self.raw_struct_member(&fields, index as u64)?;
                        let offset = base_offset + offset;
                        if offset >= len {
                            break;
                        }
                        let field_type = self.type_id(&field)?;
                        let field_value = self.fresh();
                        instructions.push(Self::inst(
                            Op::CompositeExtract,
                            Some(field_type),
                            Some(field_value),
                            vec![Operand::IdRef(value), Operand::LiteralBit32(index as u32)],
                        ));
                        if !self.collect_typed_memcpy_words(
                            field_value,
                            &field,
                            offset,
                            len,
                            words,
                            instructions,
                        )? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            }
        })
    }

    pub(in crate::native::emitter) fn typed_memcpy_word(
        &mut self,
        value: Word,
        ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        match self.resolve_type(ty)? {
            LlType::Int(32) => Ok(value),
            LlType::Float => {
                let uint_ty = self.type_id(&LlType::Int(32))?;
                let word = self.fresh();
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(uint_ty),
                    Some(word),
                    vec![Operand::IdRef(value)],
                ));
                Ok(word)
            }
            other => Err(format!(
                "native emitter: memcpy word from typed source {other:?} is not supported"
            )),
        }
    }

    pub(in crate::native::emitter) fn emit_raw_to_typed_struct_memcpy(
        &mut self,
        dst: &TypedValue,
        src: &RawBufferOffset,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some(dst_pointee) = self.pointer_pointee_for_value(&dst.value)? else {
            return Ok(false);
        };
        let dst_pointee = function_storage_local_type(&self.resolve_type(&dst_pointee)?);
        if !matches!(&dst_pointee, LlType::Struct(_) | LlType::Array(_, _)) {
            return Ok(false);
        }
        let LlType::Ptr(dst_addrspace) = self.resolve_type(&dst.ty)? else {
            return Ok(false);
        };
        let dst_storage = self.pointer_storage_for(&dst.value, dst_addrspace)?;
        let dst_id = self.value_id(&dst.value, &dst.ty)?;

        // Emit the whole copy into a scratch buffer first: a nested aggregate that turns out to carry
        // an unsupported leaf (e.g. a sub-word field) bails to `Ok(false)` and the caller falls back to
        // the typed memcpy path, so nothing partial may be committed to the real stream.
        let mut scratch = Vec::new();
        let word_shadow = if len.is_multiple_of(4) {
            let mut words = Vec::new();
            for byte in (0..len).step_by(4) {
                words.push((byte, self.emit_raw_word_load(src, byte, &mut scratch)?));
            }
            Some(words)
        } else {
            None
        };

        let mut copied_any = false;
        let mut index_path = Vec::new();
        let ok = self.emit_raw_to_typed_aggregate_fields(
            src,
            dst_id,
            dst_storage,
            &dst_pointee,
            0,
            &mut index_path,
            len,
            &mut scratch,
            &mut copied_any,
        )?;
        if !ok || !copied_any {
            return Ok(false);
        }
        instructions.extend(scratch);
        if let (LlValue::Local(dst_name), Some(words)) = (&dst.value, word_shadow) {
            self.raw_memcpy_shadows.insert(dst_name.clone(), words);
        }
        Ok(true)
    }

    /// Recursively store a raw device byte range into a typed (Function/Private) aggregate, one scalar
    /// or vector leaf at a time. Structs and arrays recurse, accumulating the access-chain index path
    /// and the running byte offset; scalar/vector leaves are read from `src` at their byte offset and
    /// stored through the full chain. This is the nesting generalization of the flat top-level struct
    /// copy — byte-faithful because every leaf lands at exactly the layout offset `raw_type_size_align`
    /// dictates, and the destination is per-invocation scratch (never the host-visible golden output).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native::emitter) fn emit_raw_to_typed_aggregate_fields(
        &mut self,
        src: &RawBufferOffset,
        dst_id: Word,
        dst_storage: StorageClass,
        ty: &LlType,
        base_offset: u64,
        index_path: &mut Vec<Word>,
        len: u64,
        out: &mut Vec<Instruction>,
        copied_any: &mut bool,
    ) -> Result<bool, String> {
        match self.resolve_type(ty)? {
            LlType::Struct(fields) => {
                for index in 0..fields.len() {
                    let (member_off, field) = self.raw_struct_member(&fields, index as u64)?;
                    let offset = base_offset + member_off;
                    if offset >= len {
                        break;
                    }
                    let field = function_storage_local_type(&field);
                    let (field_size, _) = self.raw_type_size_align(&field)?;
                    if field_size == 0 {
                        continue;
                    }
                    if offset + field_size > len {
                        return Ok(false);
                    }
                    let idx = self.const_uint(index as u32)?;
                    index_path.push(idx);
                    let ok = self.emit_raw_to_typed_aggregate_fields(
                        src,
                        dst_id,
                        dst_storage,
                        &field,
                        offset,
                        index_path,
                        len,
                        out,
                        copied_any,
                    )?;
                    index_path.pop();
                    if !ok {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            LlType::Array(elem, count) => {
                let elem = function_storage_local_type(&self.resolve_type(&elem)?);
                let (elem_size, elem_align) = self.raw_type_size_align(&elem)?;
                if elem_size == 0 {
                    return Ok(true);
                }
                let stride = round_up_u64(elem_size, elem_align);
                for i in 0..count as u64 {
                    let offset = base_offset + i * stride;
                    if offset >= len {
                        break;
                    }
                    if offset + elem_size > len {
                        return Ok(false);
                    }
                    let idx = self.const_uint(i as u32)?;
                    index_path.push(idx);
                    let ok = self.emit_raw_to_typed_aggregate_fields(
                        src,
                        dst_id,
                        dst_storage,
                        &elem,
                        offset,
                        index_path,
                        len,
                        out,
                        copied_any,
                    )?;
                    index_path.pop();
                    if !ok {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            leaf @ (LlType::Int(8 | 16 | 32)
            | LlType::Half
            | LlType::BFloat
            | LlType::Float
            | LlType::Vector(_, _)) => {
                let Some(value) =
                    self.emit_raw_word_as_typed_memcpy_field(src, base_offset, &leaf, out)?
                else {
                    return Ok(false);
                };
                let dst_ptr_ty = self.ptr_type_id(dst_storage, &leaf)?;
                let dst_field = self.fresh();
                let mut operands = Vec::with_capacity(index_path.len() + 1);
                operands.push(Operand::IdRef(dst_id));
                for idx in index_path.iter() {
                    operands.push(Operand::IdRef(*idx));
                }
                out.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(dst_ptr_ty),
                    Some(dst_field),
                    operands,
                ));
                out.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(dst_field), Operand::IdRef(value)],
                ));
                *copied_any = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn emit_raw_word_as_typed_memcpy_field(
        &mut self,
        src: &RawBufferOffset,
        offset: u64,
        field: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        if !offset.is_multiple_of(4)
            && !matches!(
                self.resolve_type(field)?,
                LlType::Int(8 | 16) | LlType::Half | LlType::BFloat
            )
        {
            return Ok(None);
        }
        match self.resolve_type(field)? {
            scalar @ (LlType::Int(8 | 16) | LlType::Half | LlType::BFloat) => {
                let value = self.fresh();
                self.emit_raw_scalar_load(value, &scalar, src, offset, None, instructions)?;
                Ok(Some(value))
            }
            LlType::Int(32) | LlType::Float => {
                let (field_size, _) = self.raw_type_size_align(field)?;
                if field_size != 4 {
                    return Ok(None);
                }
                let word = self.emit_raw_word_load(src, offset, instructions)?;
                self.emit_raw_word_as_typed_scalar(word, field, instructions)
                    .map(Some)
            }
            LlType::Vector(elem, lanes) => {
                let elem = self.resolve_type(&elem)?;
                if !matches!(elem, LlType::Int(32) | LlType::Float) {
                    return Ok(None);
                }
                let (elem_size, _) = self.raw_type_size_align(&elem)?;
                if elem_size != 4 {
                    return Ok(None);
                }
                let elem_ty = self.type_id(&elem)?;
                let vector_ty = self.type_id(field)?;
                let mut values = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let word =
                        self.emit_raw_word_load(src, offset + u64::from(lane) * 4, instructions)?;
                    let value = self.emit_raw_word_as_typed_scalar(word, &elem, instructions)?;
                    let value_ty = self.type_id(&elem)?;
                    if value_ty != elem_ty {
                        return Ok(None);
                    }
                    values.push(Operand::IdRef(value));
                }
                let vector = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(vector_ty),
                    Some(vector),
                    values,
                ));
                Ok(Some(vector))
            }
            _ => Ok(None),
        }
    }

    pub(in crate::native::emitter) fn emit_raw_word_as_typed_scalar(
        &mut self,
        word: Word,
        ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        match self.resolve_type(ty)? {
            LlType::Int(32) => Ok(word),
            LlType::Float => {
                let float_ty = self.type_id(&LlType::Float)?;
                let value = self.fresh();
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(float_ty),
                    Some(value),
                    vec![Operand::IdRef(word)],
                ));
                Ok(value)
            }
            other => Err(format!(
                "native emitter: raw memcpy scalar field {other:?} is not supported"
            )),
        }
    }

    pub(in crate::native::emitter) fn byte_gep_root_and_const_offset(
        &self,
        ptr: &TypedValue,
    ) -> Result<Option<(Word, u32, i64)>, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(None);
        };
        if self.resolve_type(&provenance.source_ty)? != LlType::Int(8) {
            return Ok(None);
        }
        if provenance.indices.len() != 1 {
            return Ok(None);
        }
        let Some(base) = const_index_i64(&provenance.indices[0]) else {
            return Ok(None);
        };
        Ok(Some((provenance.root, provenance.addrspace, base)))
    }

    /// The type a pointer's own `getelementptr` chain names, when it has one. Distinct from
    /// [`Self::pointer_pointee_for_value`]'s Function-storage view, which rewrites device pointers to
    /// `Int(64)`; an access chain built from this provenance is typed by THIS type.
    fn gep_provenance_element_type(&self, ptr: &TypedValue) -> Result<Option<LlType>, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(None);
        };
        if provenance.indices.is_empty() {
            return Ok(None);
        }
        Ok(Some(gep_pointee(
            &provenance.source_ty,
            &provenance.indices,
        )?))
    }

    /// The element RUN a byte length addresses through `ptr`: the element type and how many of them
    /// `len` covers, or `None` when the pointer does not address a contiguous run of them.
    ///
    /// The element type comes from the pointer's own `getelementptr` chain, not from
    /// [`Self::pointer_pointee_for_value`]'s Function-storage view, because the striding primitive
    /// ([`Self::emit_gep_provenance_lane_ptr`]) types its access chain from the GEP. A `ptr` element is
    /// refused for exactly that reason: a `[N x ptr addrspace(1)]` local declares `Int(64)` slots while
    /// its GEP names `Ptr`, so a chain built from the GEP cannot address the array the local is.
    fn gep_element_run(
        &mut self,
        ptr: &TypedValue,
        len: u64,
    ) -> Result<Option<(LlType, u32)>, String> {
        let Some(element) = self.gep_provenance_element_type(ptr)? else {
            return Ok(None);
        };
        let element = self.resolve_type(&element)?;
        if matches!(element, LlType::Ptr(_)) {
            return Ok(None);
        }
        let (element_size, _) = self.raw_type_size_align(&element)?;
        if element_size == 0 || !len.is_multiple_of(element_size) {
            return Ok(None);
        }
        let Ok(count) = u32::try_from(len / element_size) else {
            return Ok(None);
        };
        if !self.can_emit_gep_provenance_lane_ptrs(ptr, &element)? {
            return Ok(None);
        }
        Ok(Some((element, count)))
    }

    pub(in crate::native::emitter) fn emit_zero_memset(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !call.callee.starts_with("llvm.memset.") {
            return Ok(false);
        }
        if call.args.len() != 4
            || !typed_value_is_zero(&call.args[1])
            || typed_value_u64(&call.args[2]).is_none_or(|len| len == 0)
            || !matches!(call.args[3].value, LlValue::Bool(false))
        {
            return Ok(false);
        }
        let len = typed_value_u64(&call.args[2]).ok_or_else(|| {
            "native emitter: zero-memset length is not a positive u64 constant".to_string()
        })?;
        if let LlValue::Local(name) = &call.args[0].value {
            if let Some(raw) = self.raw_offsets.get(name).cloned() {
                let align = call.arg_aligns.first().copied().flatten();
                self.emit_raw_zero_memset(&raw, len, align, instructions)?;
                return Ok(true);
            }
        }
        if self.emit_byte_pointer_zero_memset(&call.args[0], len, instructions)? {
            return Ok(true);
        }
        let Some(pointee) = self.pointer_pointee_for_value(&call.args[0].value)? else {
            return Ok(false);
        };
        let storage_pointee = function_storage_local_type(&pointee);
        let (size, _) = self.raw_type_size_align(&storage_pointee)?;
        if len < size {
            // Logical SPIR-V cannot call LLVM's byte-pointer declaration with a differently typed
            // aggregate pointer. Lower a prefix clear through the real aggregate layout: fully
            // covered subobjects receive typed null stores, partial aggregates recurse, and padding
            // needs no store. A prefix cutting through a scalar remains an honest unsupported emit.
            let LlType::Ptr(addrspace) = self.resolve_type(&call.args[0].ty)? else {
                return Ok(false);
            };
            let storage = self.pointer_storage_for(&call.args[0].value, addrspace)?;
            let base = self.value_id(&call.args[0].value, &call.args[0].ty)?;
            if self.emit_typed_zero_prefix(
                base,
                storage,
                &storage_pointee,
                0,
                len,
                &mut Vec::new(),
                instructions,
            )? {
                return Ok(true);
            }
            return Err(format!(
                "native emitter: partial zero memset cuts through a scalar subobject of {storage_pointee:?}"
            ));
        }
        if len > size && self.emit_element_run_zero_memset(&call.args[0], len, instructions)? {
            return Ok(true);
        }
        let ptr = self.value_id(&call.args[0].value, &call.args[0].ty)?;
        let zero = self.const_null(&storage_pointee)?;
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(ptr), Operand::IdRef(zero)],
        ));
        Ok(true)
    }

    /// Clear a `memset` that spans MORE than its pointee object: `memset(&a[k], 0, n * sizeof(T))`
    /// names one element and clears `n` of them, and the single null store below is short by `n - 1`
    /// elements. Walk the run through the destination's own GEP provenance instead — the same
    /// element-striding primitive the vector loads use, which keeps `Function`/`Private` roots on
    /// `OpInBoundsAccessChain` where `OpPtrAccessChain` is illegal.
    ///
    /// Returns `false` (the caller keeps its single store) when the run does not divide evenly into
    /// elements, when the pointer's last index does not stride a contiguous array/vector element —
    /// striding a struct member would walk into a differently typed neighbour — or when the element is
    /// a pointer. That last one is the `[N x ptr addrspace(1)]` device-pointer table: the local
    /// declares `Int(64)` slots, the GEP names `Ptr`, and the striding primitive types its chain from
    /// the GEP, so it cannot address the array the local actually is. Zeroing a device pointer is the
    /// separate unmodelled-placeholder problem and is left exactly as it was.
    fn emit_element_run_zero_memset(
        &mut self,
        dst: &TypedValue,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some((element, count)) = self.gep_element_run(dst, len)? else {
            return Ok(false);
        };
        let zero = self.const_null(&element)?;
        let mut staged = Vec::new();
        for lane in 0..count {
            let Some(pointer) =
                self.emit_gep_provenance_lane_ptr(dst, &element, lane, &mut staged)?
            else {
                return Ok(false);
            };
            staged.push(Self::inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(pointer), Operand::IdRef(zero)],
            ));
        }
        instructions.extend(staged);
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_typed_zero_prefix(
        &mut self,
        base: Word,
        storage: StorageClass,
        ty: &LlType,
        offset: u64,
        end: u64,
        indices: &mut Vec<Word>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            let ty = function_storage_local_type(&self.resolve_type(ty)?);
            let (size, _) = self.raw_type_size_align(&ty)?;
            if size == 0 || offset >= end {
                return Ok(true);
            }
            if offset
                .checked_add(size)
                .is_some_and(|object_end| object_end <= end)
            {
                let pointer = if indices.is_empty() {
                    base
                } else {
                    let pointer_type = self.ptr_type_id(storage, &ty)?;
                    let pointer = self.fresh();
                    let mut operands = Vec::with_capacity(indices.len() + 1);
                    operands.push(Operand::IdRef(base));
                    operands.extend(indices.iter().copied().map(Operand::IdRef));
                    instructions.push(Self::inst(
                        Op::InBoundsAccessChain,
                        Some(pointer_type),
                        Some(pointer),
                        operands,
                    ));
                    pointer
                };
                let zero = self.const_null(&ty)?;
                instructions.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(pointer), Operand::IdRef(zero)],
                ));
                return Ok(true);
            }
            match ty {
                LlType::Struct(fields) => {
                    for index in 0..fields.len() {
                        let (member_offset, field) =
                            self.raw_struct_member(&fields, index as u64)?;
                        let index = self.const_uint(index as u32)?;
                        indices.push(index);
                        let supported = self.emit_typed_zero_prefix(
                            base,
                            storage,
                            &field,
                            offset.checked_add(member_offset).ok_or_else(|| {
                                "native emitter: partial memset struct offset overflows u64"
                                    .to_string()
                            })?,
                            end,
                            indices,
                            instructions,
                        )?;
                        indices.pop();
                        if !supported {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                LlType::Array(elem, count) | LlType::Vector(elem, count) => {
                    let elem = function_storage_local_type(&self.resolve_type(&elem)?);
                    let (elem_size, elem_align) = self.raw_type_size_align(&elem)?;
                    let stride = round_up_u64(elem_size, elem_align);
                    for index in 0..count {
                        let index_id = self.const_uint(index)?;
                        indices.push(index_id);
                        let element_offset = u64::from(index)
                            .checked_mul(stride)
                            .and_then(|relative| offset.checked_add(relative))
                            .ok_or_else(|| {
                                "native emitter: partial memset aggregate offset overflows u64"
                                    .to_string()
                            })?;
                        let supported = self.emit_typed_zero_prefix(
                            base,
                            storage,
                            &elem,
                            element_offset,
                            end,
                            indices,
                            instructions,
                        )?;
                        indices.pop();
                        if !supported {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            }
        })
    }

    pub(in crate::native::emitter) fn emit_raw_zero_memset(
        &mut self,
        raw: &RawBufferOffset,
        len: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let zero = self.const_uint(0)?;
        if self.raw_pointer_word_aligned(raw) {
            let word_bytes = len / 4 * 4;
            for byte in (0..word_bytes).step_by(4) {
                self.emit_raw_word_store_for_access(raw, byte, zero, access_align, instructions)?;
            }
            for byte in word_bytes..len {
                self.emit_raw_byte_store_from_u32(zero, raw, byte, instructions)?;
            }
            return Ok(());
        }
        for byte in 0..len {
            self.emit_raw_byte_store_from_u32(zero, raw, byte, instructions)?;
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_byte_pointer_zero_memset(
        &mut self,
        dst: &TypedValue,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if let LlValue::Local(name) = &dst.value {
            if let Some(padding) = self.workgroup_padding_byte_pointers.get(name).cloned() {
                if self.struct_range_is_padding(&padding.struct_ty, padding.byte_offset, len)? {
                    return Ok(true);
                }
                return Err(format!(
                    "native emitter: zero memset from Workgroup struct padding offset {} spans non-padding bytes",
                    padding.byte_offset
                ));
            }
        }
        let Some(pointee) = self.pointer_pointee_for_value(&dst.value)? else {
            return Ok(false);
        };
        if self.resolve_type(&pointee)? != LlType::Int(8) {
            return Ok(false);
        }
        let LlType::Ptr(addrspace) = self.resolve_type(&dst.ty)? else {
            return Ok(false);
        };
        let storage = self.pointer_storage_for(&dst.value, addrspace)?;
        let access_op = if ptr_access_chain_allowed_storage(storage) {
            Op::PtrAccessChain
        } else {
            Op::InBoundsAccessChain
        };
        let ptr_ty = self.ptr_type_id(storage, &LlType::Int(8))?;
        let ptr = self.value_id(&dst.value, &dst.ty)?;
        let zero = self.const_null(&LlType::Int(8))?;
        for byte in 0..len {
            let store_ptr = if byte == 0 {
                ptr
            } else {
                let index = self.const_uint(u32::try_from(byte).map_err(|_| {
                    format!("native emitter: memset byte offset {byte} exceeds u32")
                })?)?;
                let byte_ptr = self.fresh();
                instructions.push(Self::inst(
                    access_op,
                    Some(ptr_ty),
                    Some(byte_ptr),
                    vec![Operand::IdRef(ptr), Operand::IdRef(index)],
                ));
                byte_ptr
            };
            instructions.push(Self::inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(store_ptr), Operand::IdRef(zero)],
            ));
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn struct_range_is_padding(
        &self,
        struct_ty: &LlType,
        byte_offset: u64,
        len: u64,
    ) -> Result<bool, String> {
        let LlType::Struct(fields) = self.resolve_type(struct_ty)? else {
            return Ok(false);
        };
        let end = byte_offset.checked_add(len).ok_or_else(|| {
            "native emitter: Workgroup padding byte range overflows u64".to_string()
        })?;
        let (struct_size, _) = self.raw_type_size_align(struct_ty)?;
        if len == 0 || end > struct_size {
            return Ok(false);
        }
        let mut member_offset = 0u64;
        for field in &fields {
            let (field_size, field_align) = self.raw_type_size_align(field)?;
            member_offset = round_up_u64(member_offset, field_align);
            let member_end = member_offset.checked_add(field_size).ok_or_else(|| {
                "native emitter: Workgroup struct member range overflows u64".to_string()
            })?;
            if byte_offset < member_end && end > member_offset {
                return Ok(false);
            }
            member_offset = member_end;
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_typed_memcpy(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            if !call.callee.starts_with("llvm.memcpy.") {
                return Ok(false);
            }
            if call.args.len() != 4 || !matches!(call.args[3].value, LlValue::Bool(false)) {
                return Ok(false);
            }
            let Some(len) = typed_value_u64(&call.args[2]) else {
                return Ok(false);
            };
            if len == 0 {
                return Ok(true);
            }
            let Some(dst_pointee) = self.pointer_pointee_for_value(&call.args[0].value)? else {
                return Ok(false);
            };
            let Some(src_pointee) = self.pointer_pointee_for_value(&call.args[1].value)? else {
                return Ok(false);
            };
            let dst_pointee = self.resolve_type(&dst_pointee)?;
            let src_pointee = self.resolve_type(&src_pointee)?;
            let copy_pointee = function_storage_local_type(&src_pointee);
            let (copy_size, _) = self.raw_type_size_align(&copy_pointee)?;
            if len > copy_size {
                // The copy outruns the object its pointer names, so copying that object is short by
                // `len / copy_size - 1` elements. See `emit_element_run_zero_memset`.
                return self.emit_element_run_memcpy(
                    &call.args[0],
                    &call.args[1],
                    len,
                    instructions,
                );
            }
            if len < copy_size {
                if !types_compatible(&dst_pointee, &src_pointee) {
                    return self.emit_prefix_source_struct_memcpy(
                        &call.args[0],
                        &call.args[1],
                        &dst_pointee,
                        &src_pointee,
                        len,
                        instructions,
                    );
                }
                let LlType::Struct(fields) = &function_storage_local_type(&src_pointee) else {
                    return Ok(false);
                };
                let LlType::Ptr(dst_addrspace) = self.resolve_type(&call.args[0].ty)? else {
                    return Ok(false);
                };
                let LlType::Ptr(src_addrspace) = self.resolve_type(&call.args[1].ty)? else {
                    return Ok(false);
                };
                let dst_storage = self.pointer_storage_for(&call.args[0].value, dst_addrspace)?;
                let src_storage = self.pointer_storage_for(&call.args[1].value, src_addrspace)?;
                let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                let src = self.value_id(&call.args[1].value, &call.args[1].ty)?;
                return self.emit_prefix_struct_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    fields,
                    fields,
                    len,
                    instructions,
                );
            }

            let LlType::Ptr(dst_addrspace) = self.resolve_type(&call.args[0].ty)? else {
                return Ok(false);
            };
            let LlType::Ptr(src_addrspace) = self.resolve_type(&call.args[1].ty)? else {
                return Ok(false);
            };
            let dst_storage = self.pointer_storage_for(&call.args[0].value, dst_addrspace)?;
            let src_storage = self.pointer_storage_for(&call.args[1].value, src_addrspace)?;
            let src = self.value_id(&call.args[1].value, &call.args[1].ty)?;
            if self.can_emit_whole_copy_memory(
                &dst_pointee,
                &src_pointee,
                dst_storage,
                src_storage,
            )? {
                let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                self.emit_copy_memory(dst, src, dst_storage, &dst_pointee, instructions)?;
                return Ok(true);
            }

            // LLVM may pass an array object directly as the destination of an exact one-element
            // memcpy. Opaque pointers erase the source-level array-to-first-element decay, but SPIR-V
            // retains the aggregate pointee and therefore needs that access chain constructed
            // explicitly. Limit this to one complete source object; a longer byte range would require
            // source extent information that the erased pointer does not provide.
            if let LlType::Array(dst_element, count) = &dst_pointee {
                if *count > 0 && len == copy_size && types_compatible(dst_element, &src_pointee) {
                    let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                    let dst_element = function_storage_local_type(dst_element);
                    let dst_ptr_ty = self.ptr_type_id(dst_storage, &dst_element)?;
                    let dst_element_ptr = self.fresh();
                    let zero = self.const_uint(0)?;
                    instructions.push(Self::inst(
                        Op::InBoundsAccessChain,
                        Some(dst_ptr_ty),
                        Some(dst_element_ptr),
                        vec![Operand::IdRef(dst), Operand::IdRef(zero)],
                    ));
                    return self.emit_aggregate_memcpy(
                        dst_element_ptr,
                        src,
                        dst_storage,
                        src_storage,
                        &dst_element,
                        &src_pointee,
                        len,
                        instructions,
                    );
                }
            }

            // A named AIR wrapper can contain the exact aggregate copied into a bare local array
            // (`metal::matrix<T>` -> its `[N x vector<T>]` storage). Descend through the source's
            // offset-zero field just as the symmetric destination-wrapper path below does. The helper's
            // extent guard keeps this honest when a struct has trailing fields or padding covered by the
            // requested byte count.
            if matches!(src_pointee, LlType::Struct(_))
                && self.emit_prefix_source_struct_memcpy(
                    &call.args[0],
                    &call.args[1],
                    &dst_pointee,
                    &src_pointee,
                    len,
                    instructions,
                )?
            {
                return Ok(true);
            }

            if let (LlType::Struct(dst_fields), LlType::Struct(src_fields)) =
                (&dst_pointee, &src_pointee)
            {
                if self.struct_prefix_fields_compatible(dst_fields, src_fields)? {
                    let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                    return self.emit_prefix_struct_memcpy(
                        dst,
                        src,
                        dst_storage,
                        src_storage,
                        dst_fields,
                        src_fields,
                        len,
                        instructions,
                    );
                }
            }

            if let (LlType::Array(dst_elem, dst_len), LlType::Array(src_elem, src_len)) =
                (&dst_pointee, &src_pointee)
            {
                if dst_len != src_len || !types_compatible(dst_elem, src_elem) {
                    return Ok(false);
                }
                let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                return self.emit_prefix_array_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    dst_elem,
                    src_elem,
                    *src_len,
                    len,
                    instructions,
                );
            }

            if let LlType::Struct(fields) = &dst_pointee {
                let Some(field) = fields.first() else {
                    return Ok(false);
                };
                if !types_compatible(field, &src_pointee) {
                    return Ok(false);
                }
                let dst_field_pointee = function_storage_local_type(field);
                let field_ptr_ty = self.ptr_type_id(dst_storage, &dst_field_pointee)?;
                let base = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                let zero = self.const_uint(0)?;
                let field_ptr = self.fresh();
                instructions.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(field_ptr_ty),
                    Some(field_ptr),
                    vec![Operand::IdRef(base), Operand::IdRef(zero)],
                ));
                return self.emit_aggregate_memcpy(
                    field_ptr,
                    src,
                    dst_storage,
                    src_storage,
                    &dst_field_pointee,
                    &src_pointee,
                    len,
                    instructions,
                );
            }

            Ok(false)
        })
    }

    pub(in crate::native::emitter) fn struct_prefix_fields_compatible(
        &self,
        dst_fields: &[LlType],
        src_fields: &[LlType],
    ) -> Result<bool, String> {
        if src_fields.len() > dst_fields.len() {
            return Ok(false);
        }
        for index in 0..src_fields.len() {
            let (dst_offset, dst_field) = self.raw_struct_member(dst_fields, index as u64)?;
            let (src_offset, src_field) = self.raw_struct_member(src_fields, index as u64)?;
            if dst_offset != src_offset {
                return Ok(false);
            }
            let dst_field = function_storage_local_type(&dst_field);
            let src_field = function_storage_local_type(&src_field);
            if !types_compatible(&dst_field, &src_field) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn can_emit_whole_copy_memory(
        &mut self,
        dst_pointee: &LlType,
        src_pointee: &LlType,
        dst_storage: StorageClass,
        src_storage: StorageClass,
    ) -> Result<bool, String> {
        let dst_ty = function_storage_local_type(dst_pointee);
        let src_ty = function_storage_local_type(src_pointee);
        if self.type_id(&dst_ty)? != self.type_id(&src_ty)? {
            return Ok(false);
        }
        if (is_copy_memory_aggregate(&dst_ty) || is_copy_memory_aggregate(&src_ty))
            && (is_interface_backed_copy_storage(dst_storage)
                || is_interface_backed_copy_storage(src_storage))
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Copy one whole object from `src` to `dst`.
    ///
    /// `OpCopyMemory` is the natural spelling and is what this emits everywhere it can. It cannot be
    /// used when the DESTINATION is an interface-backed block: SPIRV-Cross's write analysis does not
    /// count an `OpCopyMemory` as a write, so a `StorageBuffer` written only that way is rendered
    /// `const device _N&` in MSL, the assignment into it does not compile, and MoltenVK answers
    /// `create compute pipeline: Initialization of an object has failed`. `spirv-val` is happy with
    /// the module either way, which is why this survived. An `OpLoad` plus an `OpStore` is the same
    /// copy and SPIRV-Cross does count that.
    pub(in crate::native::emitter) fn emit_copy_memory(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        pointee: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if !is_interface_backed_copy_storage(dst_storage) {
            instructions.push(Self::inst(
                Op::CopyMemory,
                None,
                None,
                vec![Operand::IdRef(dst), Operand::IdRef(src)],
            ));
            return Ok(());
        }
        let object_ty = self.type_id(&function_storage_local_type(pointee))?;
        let object = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(object_ty),
            Some(object),
            vec![Operand::IdRef(src)],
        ));
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(dst), Operand::IdRef(object)],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_aggregate_memcpy(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        src_storage: StorageClass,
        dst_pointee: &LlType,
        src_pointee: &LlType,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if len == 0 {
            return Ok(true);
        }
        if self.can_emit_whole_copy_memory(dst_pointee, src_pointee, dst_storage, src_storage)? {
            self.emit_copy_memory(dst, src, dst_storage, dst_pointee, instructions)?;
            return Ok(true);
        }

        let dst_pointee = function_storage_local_type(dst_pointee);
        let src_pointee = function_storage_local_type(src_pointee);
        match (&dst_pointee, &src_pointee) {
            (LlType::Struct(dst_fields), LlType::Struct(src_fields)) => {
                if !self.struct_prefix_fields_compatible(dst_fields, src_fields)? {
                    return Ok(false);
                }
                self.emit_prefix_struct_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    dst_fields,
                    src_fields,
                    len,
                    instructions,
                )
            }
            (LlType::Array(dst_elem, dst_len), LlType::Array(src_elem, src_len)) => {
                if dst_len != src_len || !types_compatible(dst_elem, src_elem) {
                    return Ok(false);
                }
                self.emit_prefix_array_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    dst_elem,
                    src_elem,
                    *src_len,
                    len,
                    instructions,
                )
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn emit_prefix_source_struct_memcpy(
        &mut self,
        dst_arg: &TypedValue,
        src_arg: &TypedValue,
        dst_pointee: &LlType,
        src_pointee: &LlType,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            let LlType::Struct(src_fields) = src_pointee else {
                return Ok(false);
            };
            if src_fields.is_empty() {
                return Ok(false);
            }
            let (src_offset, src_field) = self.raw_struct_member(src_fields, 0)?;
            if src_offset != 0 {
                return Ok(false);
            }
            let src_field = function_storage_local_type(&src_field);
            let dst_pointee = function_storage_local_type(dst_pointee);
            // A memcpy starting at an array object and spanning one element addresses element zero.
            // Opaque LLVM pointers erase that decay: `%dst = alloca [N x T]` is passed directly while
            // the source is a wrapper struct whose first field is `T`. Construct the element-zero
            // pointer explicitly instead of leaving a residual byte-pointer memcpy call whose declared
            // SPIR-V parameter cannot match the aggregate pointer value.
            let (dst_copy_pointee, decay_destination_array) = match &dst_pointee {
                LlType::Array(element, count)
                    if *count > 0 && types_compatible(element, &src_field) =>
                {
                    (function_storage_local_type(element), true)
                }
                _ if types_compatible(&dst_pointee, &src_field) => (dst_pointee.clone(), false),
                _ => return Ok(false),
            };
            let (field_size, _) = self.raw_type_size_align(&src_field)?;
            if len > field_size {
                return Ok(false);
            }

            let LlType::Ptr(dst_addrspace) = self.resolve_type(&dst_arg.ty)? else {
                return Ok(false);
            };
            let LlType::Ptr(src_addrspace) = self.resolve_type(&src_arg.ty)? else {
                return Ok(false);
            };
            let dst_storage = self.pointer_storage_for(&dst_arg.value, dst_addrspace)?;
            let src_storage = self.pointer_storage_for(&src_arg.value, src_addrspace)?;
            let mut dst = self.value_id(&dst_arg.value, &dst_arg.ty)?;
            let src = self.value_id(&src_arg.value, &src_arg.ty)?;
            let zero = self.const_uint(0)?;
            if decay_destination_array {
                let dst_ptr_ty = self.ptr_type_id(dst_storage, &dst_copy_pointee)?;
                let dst_element = self.fresh();
                instructions.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(dst_ptr_ty),
                    Some(dst_element),
                    vec![Operand::IdRef(dst), Operand::IdRef(zero)],
                ));
                dst = dst_element;
            }
            let src_ptr_ty = self.ptr_type_id(src_storage, &src_field)?;
            let src_field_ptr = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(src_ptr_ty),
                Some(src_field_ptr),
                vec![Operand::IdRef(src), Operand::IdRef(zero)],
            ));
            self.emit_aggregate_memcpy(
                dst,
                src_field_ptr,
                dst_storage,
                src_storage,
                &dst_copy_pointee,
                &src_field,
                len,
                instructions,
            )
        })
    }

    pub(in crate::native::emitter) fn emit_prefix_struct_memcpy(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        src_storage: StorageClass,
        dst_fields: &[LlType],
        src_fields: &[LlType],
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            let mut copied_any = false;
            for index in 0..src_fields.len() {
                let (dst_offset, dst_field) = self.raw_struct_member(dst_fields, index as u64)?;
                let (src_offset, src_field) = self.raw_struct_member(src_fields, index as u64)?;
                if dst_offset != src_offset {
                    return Ok(false);
                }
                let offset = src_offset;
                if offset >= len {
                    break;
                }
                let dst_field = function_storage_local_type(&dst_field);
                let src_field = function_storage_local_type(&src_field);
                let (field_size, _) = self.raw_type_size_align(&src_field)?;
                if field_size == 0 {
                    continue;
                }
                if offset + field_size > len {
                    return Ok(false);
                }

                let index_id = self.const_uint(index as u32)?;
                let dst_ptr_ty = self.ptr_type_id(dst_storage, &dst_field)?;
                let dst_field = self.fresh();
                instructions.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(dst_ptr_ty),
                    Some(dst_field),
                    vec![Operand::IdRef(dst), Operand::IdRef(index_id)],
                ));
                let src_ptr_ty = self.ptr_type_id(src_storage, &src_field)?;
                let src_field = self.fresh();
                instructions.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(src_ptr_ty),
                    Some(src_field),
                    vec![Operand::IdRef(src), Operand::IdRef(index_id)],
                ));
                if !self.emit_aggregate_memcpy(
                    dst_field,
                    src_field,
                    dst_storage,
                    src_storage,
                    &dst_fields[index],
                    &src_fields[index],
                    field_size,
                    instructions,
                )? {
                    return Ok(false);
                }
                copied_any = true;
            }
            Ok(copied_any)
        })
    }

    pub(in crate::native::emitter) fn emit_prefix_array_memcpy(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        src_storage: StorageClass,
        dst_elem: &LlType,
        src_elem: &LlType,
        count: u32,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        staged_emit(instructions, |instructions| {
            let dst_elem = function_storage_local_type(dst_elem);
            let src_elem = function_storage_local_type(src_elem);
            let (elem_size, elem_align) = self.raw_type_size_align(&src_elem)?;
            let stride = round_up_u64(elem_size, elem_align);
            let mut copied_any = false;
            for index in 0..count {
                let offset = stride * u64::from(index);
                if offset >= len {
                    break;
                }
                if offset + elem_size > len {
                    return Ok(false);
                }

                let index_id = self.const_uint(index)?;
                let dst_ptr_ty = self.ptr_type_id(dst_storage, &dst_elem)?;
                let dst_element = self.fresh();
                instructions.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(dst_ptr_ty),
                    Some(dst_element),
                    vec![Operand::IdRef(dst), Operand::IdRef(index_id)],
                ));
                let src_ptr_ty = self.ptr_type_id(src_storage, &src_elem)?;
                let src_element = self.fresh();
                instructions.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(src_ptr_ty),
                    Some(src_element),
                    vec![Operand::IdRef(src), Operand::IdRef(index_id)],
                ));
                if !self.emit_aggregate_memcpy(
                    dst_element,
                    src_element,
                    dst_storage,
                    src_storage,
                    &dst_elem,
                    &src_elem,
                    elem_size,
                    instructions,
                )? {
                    return Ok(false);
                }
                copied_any = true;
            }
            Ok(copied_any)
        })
    }
}
