//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_void_air_call(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        match call.callee.as_str() {
            "air.wg.barrier" | "air.simdgroup.barrier" => {
                if call.args.len() != 2 {
                    return Err(format!(
                        "native emitter: {} expects 2 operands",
                        call.callee
                    ));
                }
                // The execution scope is the second operand, not the intrinsic name. The two
                // intrinsics differ only in the scope Apple's `threadgroup_barrier` and
                // `simdgroup_barrier` happen to pass, and AIR spells a threadgroup-wide barrier
                // through the simdgroup intrinsic often enough to matter.
                let execution = air_barrier_execution_scope(call)?;
                let scope = self.const_uint(execution as u32)?;
                let memory_scope =
                    self.const_uint(air_barrier_memory_scope(call, execution) as u32)?;
                let semantics = self.const_uint(air_barrier_memory_semantics(call).bits())?;
                instructions.push(Self::inst(
                    Op::ControlBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdScope(memory_scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.fence" => {
                if call.args.len() != 3 {
                    return Err("native emitter: air.atomic.fence expects 3 operands".to_string());
                }
                // Which memory a fence covers is its FIRST operand, exactly as it is for the two
                // barriers above; the third names the scope over which that memory is ordered.
                // Deriving the memory classes from the scope instead answered `mem_threadgroup`
                // with device semantics and no `WORKGROUP_MEMORY` at all.
                let scope_kind = air_fence_memory_scope(call);
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics = self.const_uint(air_barrier_memory_semantics(call).bits())?;
                instructions.push(Self::inst(
                    Op::MemoryBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            callee if callee.starts_with("air.fence_texture") => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let scope = self.const_uint(Scope::Device as u32)?;
                let semantics = self.const_uint(
                    (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::IMAGE_MEMORY).bits(),
                )?;
                instructions.push(Self::inst(
                    Op::MemoryBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            callee if is_coherent_air_store(callee) => {
                if call.args.len() != 2 {
                    return Err(format!(
                        "native emitter: {} expects 2 operands",
                        call.callee
                    ));
                }
                let value =
                    self.value_id_in(&call.args[0].value, &call.args[0].ty, instructions)?;
                let ptr_arg = &call.args[1];
                if let LlValue::Local(name) = &ptr_arg.value {
                    if let Some(raw) = self.raw_offsets.get(name).cloned() {
                        self.emit_raw_store(&call.args[0].ty, value, &raw, None, instructions)?;
                        return Ok(true);
                    }
                    if self.unmodeled_pointers.contains(name) {
                        return Ok(true);
                    }
                }
                let ptr = self.value_id_in(&ptr_arg.value, &ptr_arg.ty, instructions)?;
                instructions.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(ptr), Operand::IdRef(value)],
                ));
                Ok(true)
            }
            "air.atomic.local.store.i32" | "air.atomic.global.store.i32" => {
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::RELEASE);
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicStore,
                    None,
                    None,
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(value),
                    ],
                ));
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn emit_value_air_call(
        &mut self,
        call: &LlCall,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        match call.callee.as_str() {
            // `llvm.agx2.cluster.num` names the PHYSICAL GPU cluster the threadgroup landed on,
            // not any position within it. Device-measured on an M3 Max (`xcrun metal -x ir` over a
            // hand-written AIR probe, since MSL has no spelling for the intrinsic): the value is
            // identical for every thread of a threadgroup, and over 32768 threadgroups it took every
            // value in 0..=36 with no dependence on the thread's coordinates. Vulkan exposes no
            // physical cluster, and nothing about the invocation could reproduce one — a per-thread
            // answer would make a predicate the hardware guarantees uniform diverge inside a
            // threadgroup, which is worse than any single answer. Zero is the answer given, and it
            // is one the device itself gives: it says this threadgroup ran on cluster 0, which is a
            // legal execution of a value the hardware picks nondeterministically. The corpus uses it
            // only as `icmp ult %n, 10` guarding a per-cluster slot claimed by an atomic cmpxchg,
            // so the effect is that every threadgroup contends for slot 0 and one wins, as it would
            // on a device with a single cluster.
            "llvm.agx2.cluster.num" => {
                if !call.args.is_empty() {
                    return Err(format!(
                        "native emitter: {} expects no operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}, expected i32",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let zero = self.const_uint(0)?;
                instructions.push(Self::inst(
                    Op::CopyObject,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(zero)],
                ));
                Ok(true)
            }
            // `air.is_null_texture_<dim>(%tex)` asks whether a texture handle is the null texture.
            // We consume it HERE only when `%tex` is a value we ourselves synthesized from
            // `air.get_null_texture_*` (tracked in `null_texture_values`): that value never crosses
            // the emitter->reparse seam as a recognizable image, so the passes-layer null tracking
            // (`ctx.null_image_values`) can't see it and would answer FALSE. A real bound texture is
            // NOT in the set, so it falls through to the default emission and the passes layer lowers
            // it exactly as before — keeping every regression case byte-identical. Dispatches on a stable
            // `air.*` ABI symbol plus our own data-flow set, never a shader name.
            callee if callee.starts_with("air.is_null_texture") => {
                if call.args.len() == 1 {
                    if let LlValue::Local(arg_name) = &call.args[0].value {
                        if self.null_texture_values.contains(arg_name) {
                            let result_ty = self.resolve_type(&call.ret)?;
                            let result_type = self.type_id(&result_ty)?;
                            let result = self.result_id(name, &result_ty)?;
                            let c = self.const_bool(true)?;
                            instructions.push(Self::inst(
                                Op::CopyObject,
                                Some(result_type),
                                Some(result),
                                vec![Operand::IdRef(c)],
                            ));
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            }
            "air.get_null_intersection_function_table" => {
                if !call.args.is_empty() {
                    return Err(format!(
                        "native emitter: {} expects no operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                let LlType::Ptr(addrspace) = result_ty else {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}, expected pointer",
                        call.callee
                    ));
                };
                self.define_unmodeled_byte_pointer_value(name, addrspace)?;
                let is_null = self.const_bool(true)?;
                self.record_pointer_nullness(name.to_string(), is_null);
                Ok(true)
            }
            "air.get_instance_count_instance_acceleration_structure" => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let LlValue::Local(shadow_name) = &call.args[0].value else {
                    return Err(format!(
                        "native emitter: {} shadow operand is not SSA",
                        call.callee
                    ));
                };
                let Some(mut raw) = self.raw_offsets.get(shadow_name).cloned() else {
                    return Ok(false);
                };
                raw.const_off += crate::as_shadow::INSTANCE_COUNT_BYTE_OFFSET as i64;
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}, expected i32",
                        call.callee
                    ));
                }
                let result = self.result_id(name, &result_ty)?;
                self.emit_raw_load(result, &result_ty, &raw, Some(4), instructions)?;
                Ok(true)
            }
            "air.get_primitive_acceleration_structure_instance_acceleration_structure" => {
                if call.args.len() != 2 {
                    return Err(format!(
                        "native emitter: {} expects 2 operands",
                        call.callee
                    ));
                }
                let LlValue::Local(shadow_name) = &call.args[0].value else {
                    return Err(format!(
                        "native emitter: {} shadow operand is not SSA",
                        call.callee
                    ));
                };
                let Some(mut raw) = self.raw_offsets.get(shadow_name).cloned() else {
                    return Ok(false);
                };
                let result_ty = self.resolve_type(&call.ret)?;
                let LlType::Ptr(addrspace) = result_ty else {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}, expected pointer",
                        call.callee
                    ));
                };
                raw.const_off += crate::as_shadow::CHILD_REFERENCES_BYTE_OFFSET as i64;
                raw.dyn_terms.push((
                    call.args[1].clone(),
                    crate::as_shadow::CHILD_REFERENCE_BYTE_STRIDE as i64,
                ));
                self.define_unmodeled_byte_pointer_value(name, addrspace)?;
                let (payload, is_null) =
                    self.emit_raw_pointer_payload(&raw, 0, Some(8), instructions)?;
                self.pointer_payload_words.insert(name.to_string(), payload);
                self.record_pointer_nullness(name.to_string(), is_null);
                Ok(true)
            }
            // `air.get_data_pointer_instance_acceleration_structure(%p)` returns the data pointer of an
            // instance acceleration structure. In the paravirt AS ABI (a design decision this stack
            // co-designs — see the AS-pointer-passthrough note) the instance AS's data pointer IS its
            // device address, so the intrinsic is an IDENTITY passthrough of `%p`. The argument is a
            // device pointer loaded from the instances buffer (BDA-eligible); when BDA mode has rooted
            // it at a device address, alias the result to that same device-address offset. A later
            // store of the result then copies the address verbatim and a field-offset GEP/deref reads
            // through it — exactly the plain-BDA path the 12 `store ptr addrspace(1)` cases use, with no
            // intervening tag-bit arithmetic. Byte-correct GIVEN the host rail lays out the instance AS
            // at this device address (the contract). Dispatches on a stable `air.*` ABI symbol, never a
            // shader name; outside BDA mode it returns `Ok(false)` so the default emit is untouched.
            "air.get_data_pointer_instance_acceleration_structure" => {
                if self.bda_device_pointers && call.args.len() == 1 {
                    if let LlValue::Local(arg_name) = &call.args[0].value {
                        if let Some(raw) = self.raw_offsets.get(arg_name).cloned() {
                            if raw.device_addr_base.is_some() {
                                self.used_device_address = true;
                                self.raw_offsets.insert(name.to_string(), raw);
                                self.pointer_storage
                                    .insert(name.to_string(), StorageClass::PhysicalStorageBuffer);
                                return Ok(true);
                            }
                        }
                    }
                }

                Ok(false)
            }
            callee if is_coherent_air_load(callee) => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr_arg = &call.args[0];
                if let LlValue::Local(ptr_name) = &ptr_arg.value {
                    if let Some(raw) = self.raw_offsets.get(ptr_name).cloned() {
                        self.emit_raw_load(result, &result_ty, &raw, None, instructions)?;
                        return Ok(true);
                    }
                    if self.unmodeled_pointers.contains(ptr_name) {
                        let zero = self.const_null(&result_ty)?;
                        instructions.push(Self::inst(
                            Op::CopyObject,
                            Some(result_type),
                            Some(result),
                            vec![Operand::IdRef(zero)],
                        ));
                        return Ok(true);
                    }
                }
                let ptr = self.value_id_in(&ptr_arg.value, &ptr_arg.ty, instructions)?;
                instructions.push(Self::inst(
                    Op::Load,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(ptr)],
                ));
                Ok(true)
            }
            "air.atomic.local.load.i32" | "air.atomic.global.load.i32" => {
                if call.args.len() != 4 {
                    return Err(format!(
                        "native emitter: {} expects 4 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE);
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicLoad,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.global.add.f32" => {
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Float {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                self.require_capability(Capability::AtomicFloat32AddEXT);
                self.require_extension("SPV_EXT_shader_atomic_float_add");
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_f32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let scope = self.const_uint(Scope::Device as u32)?;
                let semantics = self.const_uint(MemorySemantics::RELAXED.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicFAddEXT,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(value),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.global.sub.f32" => {
                // SPIR-V has no atomic float subtract; an atomic fetch-sub is exactly an atomic
                // fetch-add of the negated operand (both return the prior value), so negate then
                // AtomicFAddEXT.
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Float {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                self.require_capability(Capability::AtomicFloat32AddEXT);
                self.require_extension("SPV_EXT_shader_atomic_float_add");
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_f32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let negated = self.fresh();
                instructions.push(Self::inst(
                    Op::FNegate,
                    Some(result_type),
                    Some(negated),
                    vec![Operand::IdRef(value)],
                ));
                let scope = self.const_uint(Scope::Device as u32)?;
                let semantics = self.const_uint(MemorySemantics::RELAXED.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicFAddEXT,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(negated),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.local.cmpxchg.weak.i32" | "air.atomic.global.cmpxchg.weak.i32" => {
                if call.args.len() != 7 {
                    return Err(format!(
                        "native emitter: {} expects 7 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let compare_ptr_ty = self.resolve_type(&call.args[1].ty)?;
                if !matches!(compare_ptr_ty, LlType::Ptr(_)) {
                    return Err(format!(
                        "native emitter: {} compare operand is {compare_ptr_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let compare_ptr = self.value_id(&call.args[1].value, &call.args[1].ty)?;
                let compare = self.fresh();
                instructions.push(Self::inst(
                    Op::Load,
                    Some(result_type),
                    Some(compare),
                    vec![Operand::IdRef(compare_ptr)],
                ));
                let value =
                    self.value_id_in(&call.args[2].value, &call.args[2].ty, instructions)?;
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let success_semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE_RELEASE);
                let failure_semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE);
                let success_semantics = self.const_uint(success_semantics_kind.bits())?;
                let failure_semantics = self.const_uint(failure_semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicCompareExchange,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(success_semantics),
                        Operand::IdMemorySemantics(failure_semantics),
                        Operand::IdRef(value),
                        Operand::IdRef(compare),
                    ],
                ));
                instructions.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(compare_ptr), Operand::IdRef(result)],
                ));
                Ok(true)
            }
            "air.atomic.local.add.s.i32"
            | "air.atomic.local.add.u.i32"
            | "air.atomic.local.sub.s.i32"
            | "air.atomic.local.sub.u.i32"
            | "air.atomic.local.max.s.i32"
            | "air.atomic.local.max.u.i32"
            | "air.atomic.local.min.s.i32"
            | "air.atomic.local.min.u.i32"
            | "air.atomic.local.and.u.i32"
            | "air.atomic.local.or.u.i32"
            | "air.atomic.local.xchg.i32"
            | "air.atomic.global.add.s.i32"
            | "air.atomic.global.add.u.i32"
            | "air.atomic.global.and.u.i32"
            | "air.atomic.global.or.u.i32"
            | "air.atomic.global.xchg.i32"
            | "air.atomic.global.max.s.i32"
            | "air.atomic.global.max.u.i32"
            | "air.atomic.global.min.s.i32"
            | "air.atomic.global.min.u.i32"
            | "air.atomic.global.sub.s.i32"
            | "air.atomic.global.sub.u.i32" => {
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let op = match call.callee.as_str() {
                    "air.atomic.local.add.s.i32" => Op::AtomicIAdd,
                    "air.atomic.local.add.u.i32" => Op::AtomicIAdd,
                    "air.atomic.local.sub.s.i32" | "air.atomic.local.sub.u.i32" => Op::AtomicISub,
                    "air.atomic.local.max.s.i32" => Op::AtomicSMax,
                    "air.atomic.local.max.u.i32" => Op::AtomicUMax,
                    "air.atomic.local.min.s.i32" => Op::AtomicSMin,
                    "air.atomic.local.min.u.i32" => Op::AtomicUMin,
                    "air.atomic.local.and.u.i32" => Op::AtomicAnd,
                    "air.atomic.global.add.s.i32" => Op::AtomicIAdd,
                    "air.atomic.global.add.u.i32" => Op::AtomicIAdd,
                    "air.atomic.global.and.u.i32" => Op::AtomicAnd,
                    "air.atomic.global.xchg.i32" => Op::AtomicExchange,
                    "air.atomic.global.max.s.i32" => Op::AtomicSMax,
                    "air.atomic.global.max.u.i32" => Op::AtomicUMax,
                    "air.atomic.global.min.s.i32" => Op::AtomicSMin,
                    "air.atomic.global.min.u.i32" => Op::AtomicUMin,
                    "air.atomic.global.or.u.i32" => Op::AtomicOr,
                    "air.atomic.global.sub.s.i32" | "air.atomic.global.sub.u.i32" => Op::AtomicISub,
                    "air.atomic.local.or.u.i32" => Op::AtomicOr,
                    "air.atomic.local.xchg.i32" => Op::AtomicExchange,
                    _ => Op::AtomicIAdd,
                };
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE_RELEASE);
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    op,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(value),
                    ],
                ));
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

/// Which memories a barrier synchronizes, from AIR's `mem_flags` word.
///
/// `metal_compute`'s enum, confirmed one setter per probe kernel: `mem_device` is 1,
/// `mem_threadgroup` 2, `mem_texture` 4, `mem_threadgroup_imageblock` 8. An imageblock is
/// threadgroup memory -- the shared-cell lowering puts it in `Workgroup` storage -- so bit 3
/// names the same semantic bit as bit 1 rather than a new one.
fn air_barrier_memory_semantics(call: &LlCall) -> MemorySemantics {
    let flags = call
        .args
        .first()
        .and_then(|arg| air_i32_literal(&arg.value))
        .unwrap_or(0);
    let mut semantics = MemorySemantics::ACQUIRE_RELEASE;
    if flags & 1 != 0 {
        semantics |= MemorySemantics::UNIFORM_MEMORY | MemorySemantics::CROSS_WORKGROUP_MEMORY;
    }
    if flags & (2 | 8) != 0 {
        semantics |= MemorySemantics::WORKGROUP_MEMORY;
    }
    if flags & 4 != 0 {
        semantics |= MemorySemantics::IMAGE_MEMORY;
    }
    if semantics == MemorySemantics::ACQUIRE_RELEASE {
        semantics |= MemorySemantics::WORKGROUP_MEMORY;
    }
    semantics
}

/// The execution scope AIR states in a barrier's second operand: 1 threadgroup, 4 simdgroup.
///
/// Both barrier intrinsics carry it and both take both values. Apple's `threadgroup_barrier`
/// compiles to `air.wg.barrier(flags, 1)` and `simdgroup_barrier` to
/// `air.simdgroup.barrier(flags, 4)`, but AIR also spells a threadgroup-wide execution barrier
/// through the simdgroup intrinsic: 42 such calls across 13 corpus sources. Reading the scope off
/// the callee name instead of the operand made those synchronize one simdgroup where Metal
/// synchronizes the whole threadgroup.
///
/// Only 1 and 4 appear in the corpus and only those two are documented by the headers, so any
/// other value fails visibly rather than picking a scope for it.
fn air_barrier_execution_scope(call: &LlCall) -> Result<Scope, String> {
    match call.args.get(1).and_then(|arg| air_i32_literal(&arg.value)) {
        Some(1) => Ok(Scope::Workgroup),
        Some(4) => Ok(Scope::Subgroup),
        Some(other) => Err(format!(
            "native emitter: {} states execution scope {other}, which is neither AIR's threadgroup \
             (1) nor its simdgroup (4)",
            call.callee
        )),
        None => Err(format!(
            "native emitter: {} has no constant execution scope operand",
            call.callee
        )),
    }
}

/// The scope AIR states in `air.atomic.fence`'s third operand.
///
/// `atomic_thread_fence(flags, order, scope)` lowers to `air.atomic.fence(int(flags), int(order),
/// int(scope))`, and `thread_scope` is `thread_scope_thread` 0, `thread_scope_threadgroup` 1,
/// `thread_scope_device` 2 and `thread_scope_simdgroup` 4 — read one enumerator at a time out of
/// the Metal front end, not out of the enum's declaration order, which does not match the values.
///
/// Corpus fences state only 2 and a legacy 3, the OpenCL-derived all-devices scope this Metal has
/// dropped; both are Device, which is the widest scope a Vulkan shader can name. Anything else
/// unrecognised is Device too: a wider fence orders strictly more than a narrower one, so guessing
/// wide cannot lose an ordering the shader asked for.
fn air_fence_memory_scope(call: &LlCall) -> Scope {
    match call.args.get(2).and_then(|arg| air_i32_literal(&arg.value)) {
        Some(0) => Scope::Invocation,
        Some(1) => Scope::Workgroup,
        Some(4) => Scope::Subgroup,
        _ => Scope::Device,
    }
}

fn air_barrier_memory_scope(call: &LlCall, default_scope: Scope) -> Scope {
    let flags = call
        .args
        .first()
        .and_then(|arg| air_i32_literal(&arg.value))
        .unwrap_or(0);
    if flags & (1 | 4) != 0 {
        Scope::Device
    } else {
        default_scope
    }
}
