# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Product translation no longer launches LLVM/SPIR-V executables or writes their intermediate
  files. LLVM bitcode I/O uses a lazily loaded shared library; full SPIRV-Tools assembly and Vulkan
  1.2 validation are statically linked from pinned sources. macOS and Linux remain supported.
- `tools::llvm_disassemble`, `tools::llvm_assemble`, and `tools::spirv_assemble` provide in-memory
  operations. The generic `tools::run` / `run_with_timeout` subprocess APIs, their timeout markers,
  and executable-path overrides were removed. Use `METAL2VULKAN_LLVM_LIBRARY` for nonstandard LLVM
  installations. Existing translation signatures and `spirv_val_bytes`'s unused scratch argument
  are retained.
- Harvest disassembles through project-owned workers with fixed 20-second/500-MiB limits rather
  than spawning `llvm-dis`; `--llvm-dis` and `--llvm-dis-timeout-secs` were removed. Native calls in
  the public library run in the caller's process, not in implicitly spawned workers.
- Native Rust tests no longer probe for installed SPIR-V executables before validating. Their
  assembly, disassembly, and validation also run in process, so tool-less installations retain the
  same test coverage.

### Fixed

- A texture argument the pipeline variant does not provide is no longer answered by the module's
  only same-shaped live texture. `recovered_image_for_private_operand` exists for a handle the
  lowering lost track of -- a helper stored a texture in a Function aggregate and field replay
  missed it -- and it accepts a Private placeholder when exactly one image binding has the shape the
  intrinsic's own symbol names. A texture the variant leaves out arrives as the SAME placeholder, so
  wherever the live binding happened to match, the recovery stood it in: the shader read a texture it
  never named, and its store landed on one the pipeline did bind. The two are now told apart at the
  one point that knows which is which -- the parameter binding -- by naming the demotion
  `VariantAbsentTexture` instead of dropping it into `Other`, and recording the placeholder variable
  it creates.

  **Both spellings of "this variant leaves the argument out" had to be named, and only one of them
  was.** `air.function_constant` beside a gate the module's own static initializers drive to zero is
  what an unspecialized module carries; once a VALUE is supplied,
  `specialize_function_constant_metadata` rewrites that same node to
  `air.function_constant_disabled`, after which the wrapped role is unreadable by design and the
  demotion never runs. Every authored case with function constants is translated from the second
  spelling, so fixing only the first left the case path exactly as it was.

  **Measured reach over the 14,579 corpus sources: 0 modules change SPIR-V and none changes status.**
  The refusal is asked 3,451 times across 290 of them and never changes the answer -- 324 modules do
  build 2,985 absent-texture placeholders, so that is a real measurement and not a vacuous one; in
  every corpus case the recovery was already declining on candidate count, dominance or shape. On
  the specialized path it is a fix and not hardening, and the authored case that ships with it is
  the evidence: `read-and-write-a-texture-the-pipeline-does-not-provide` returns all `-1` before this
  change and matches Metal after it.

- The Vulkan executor enables `VK_KHR_maintenance5` where the device has it, so a point-topology
  pipeline built from a `vertex void` Metal function is legal. Three cases in the store draw points
  with a rasterization-disabled vertex function, which has no position output at all, let alone a
  `[[point_size]]`; without `maintenance5`,
  `VUID-VkGraphicsPipelineCreateInfo-topology-08773` makes a `POINT_LIST` pipeline demand that the
  last vertex-processing stage write `PointSize`. Two of the three violated it, and a result recorded
  from an invalid pipeline is not entitled to be trusted whatever it happens to be. `maintenance5` is
  the guarantee those cases actually need: it DEFINES the unwritten `PointSize` as 1.0 instead of
  leaving it undefined.

  Substituting a topology that demands nothing is not an alternative, and that was measured rather
  than assumed: under `TRIANGLE_LIST` both rasterization-disabled point cases flip to Mismatch,
  because a one-vertex draw assembles no triangle and the vertex shader never runs. The existing
  `rasterization_disabled_vertex_executes_narrow_attributes_without_a_companion` fails on the same
  substitution, so the topology is load-bearing and already guarded in-repo.

  `VK_KHR_maintenance5` requires `VK_KHR_dynamic_rendering` in the same enabled list
  (`VUID-vkCreateDevice-ppEnabledExtensionNames-01387`), so both are asked for or neither is;
  nothing here begins a dynamic-rendering pass. This closes the last messages the Vulkan validation
  layer reported over the store: **1010 before this session's executor work, 4 after it, 0 now.**

- `drop_unrequired_capabilities` now also decides the `VariablePointers` pair, recomputed from the
  finished module. The pipeline computes the variable-pointer requirement once and reapplies that
  SNAPSHOT after later passes run, so a pass that deleted the last pointer merge left the capability
  behind -- the same stale-declaration shape the `NonWritable` decoration is placed last to avoid.
  These two cannot join the grammar allowlist beside `ImageQuery` and the rest, because what asks
  for them is a POINTER-TYPED `OpPhi`/`OpSelect`/`OpPtrAccessChain` and the SPIR-V grammar attaches
  nothing to those opcodes; they get their own predicate instead.
  **Measured reach over the 14,579 corpus sources: 1 module changes SPIR-V, none changes status.**
  That is hardening, not a fix anyone was waiting for -- and it is worth stating why it is so small.
  The strip-one-capability oracle was previously run over the 332 *distinct capability sets* in the
  corpus, which is not a sweep of the corpus: whether a declared capability is needed is a fact about
  a module's instructions, not about the set it declares, so two modules with identical sets can
  differ on every member. Run properly over all 14,579 sources it reports **7** modules, and six of
  those are `VariablePointersStorageBuffer` declared beside `VariablePointers`, which implies it --
  redundant, not a demand. The seventh is this one. The oracle is now closed.

- A storage image the module never writes is decorated `NonWritable`, which is the other half of
  the VUID the storage-buffer decoration closed. `VUID-RuntimeSpirv-NonWritable-06340`/`-06341`
  covers "storage image, storage texel buffer, storage tensor, and storage buffer", and it is
  per-variable: one undecorated descriptor keeps `fragmentStoresAndAtomics` /
  `vertexPipelineStoresAndAtomics` demanded for the whole module. So decorating only the buffers
  left the demand standing wherever a storage image was declared beside them.
  **340 of the 14,579 corpus sources change SPIR-V**, carrying 513 image decorations. Of the 2655
  graphics-stage modules that declare a storage buffer or storage image, 264 still demanded the
  feature; **164 of them now stop**, and none starts. No status and no reflection changed.

  The image proof is shorter than the buffer one and deliberately not the same walk, because an
  image is not reached the same way. There is no device address to a `UniformConstant` variable, so
  no addressing-model gate is needed. Gating on Logical anyway, as the buffer half must, would have
  excluded the corpus's 387 `PhysicalStorageBuffer64` modules from consideration; **measured, 4 of
  the 340 modules that gain a decoration are among them**, so that is what the ungated rule is
  worth here -- 4 modules, not 387. Follow the variable, the image objects loaded from it, and the
  objects copied from those; disqualify on `OpImageWrite`, on `OpImageTexelPointer` (which takes the
  variable and exists to feed an atomic), or on **any operand slot the rule list has no entry for**.
  An array of images is not a root: its pointee is `OpTypeArray`, so it is never decorated.

  `spirv-val` accepts `NonWritable` on a storage-image variable and does not check it against
  `OpImageWrite`, so a wrong answer here is silent all the way to the device -- the same hole the
  buffer half found, measured again for images. `tests/nonwritable_covers_the_module.rs` performs the
  check it does not, now for images as well as buffers: an independent disassembly walk that roots
  every image object at the variable it was loaded from. A sweep of that shape found 0 violations
  across all 13,318 modules that translate. Both new tests are red without the decoration.

  Both halves now live in `src/reflect/nonwritable.rs` and share one derivation of "what descriptor
  is this variable" with the footprint walk, rather than reading `module.annotations` a second time.

- Four more capabilities are declared only for the construct that needs them. Each was a rule keyed
  on something coarser than the thing the capability enables, and `OpCapability` is a demand: the
  consumer must enable the device feature behind it before the module is legal to load.
  **283 of the 14,579 corpus sources change SPIR-V; none changes status or reflection.**
  - `GroupNonUniformArithmetic` was pushed for an arithmetic group *opcode*. The capability belongs
    to the group *operation*: `Reduce`/`InclusiveScan`/`ExclusiveScan` take it, `ClusteredReduce`
    takes `GroupNonUniformClustered` instead. Every `air.simd_*` whole-simdgroup reduction emits the
    clustered form, so 244 modules demanded arithmetic subgroup support they use none of. Two
    existing tests had asserted the old answer, one of them checking the substring `Reduce`, which
    `ClusteredReduce` contains.
  - `VariablePointers` was pushed for a pointer merge in any storage class but `StorageBuffer`. A
    `PhysicalStorageBuffer` pointer is an address; merging one is what `PhysicalStorageBufferAddresses`
    is for and neither variable-pointers capability governs it. 30 modules demanded the strictly
    stronger `variablePointers` feature for merging only addresses.
  - `Sampled1D` and `SampledBuffer` were pushed for an image of that dimensionality. `Dim 1D` is
    enabled by either `Sampled1D` or `Image1D`, and the type's `Sampled` operand — 2 means storage —
    says which one describes it. The storage arm read that operand and the sampled arm did not, so a
    `texture1d<..., access::write>` claimed both. 9 modules.
  These come from the same oracle as the `Geometry` fix above: strip one declared capability,
  reassemble, and ask `spirv-val` whether the module still validates.
  `tests/declared_capabilities_are_needed.rs` runs that oracle over every public fixture, and adds
  the three checks `spirv-val` cannot make — it enforces an opcode's capability disjunction but not
  the group operation's, the `Dim` enumerant's pair, or `ClipDistance` at all. The new fixture
  `kernel_write_one_dimensional_textures` is the carrier for the image half and the first authored
  device evidence for `air.write_texture_1d.u.v4i32` and `air.write_texture_buffer_1d.u.v4i32` — the
  latter appears in no corpus source, so no corpus host could ever have covered it.

- A module no longer declares `Capability Geometry`, which no Metal-backed Vulkan implementation can
  grant. Declaring a SPIR-V capability is a demand on the consumer — Vulkan requires the device
  feature behind it before the `VkShaderModule` is legal — and `Geometry` demands `geometryShader`,
  which Metal has no stage for at all. Nothing in the source language can produce a geometry shader,
  so the demand only ever arrived by accident, and it did: `PrimitiveId` is enabled by ANY of
  `Geometry`, `Tessellation`, `RayTracingKHR` or `MeshShadingEXT`, and the translator picked
  `Geometry` from that disjunction unconditionally. **236 of the 14,579 corpus sources carried it;
  226 were tessellation-evaluation entries that already declared `Tessellation`**, so the
  requirement was satisfied before the demand was added and the demand bought exactly nothing. The
  remaining 10 are fragment entries reading `[[primitive_id]]`, where one disjunct does have to be
  declared; neither names a fragment shader, so the choice is between a disjunct every Metal-backed
  implementation supports and one none of them does. **236 modules change SPIR-V, 10 of them by
  gaining `Tessellation`; no source changes status and no reflection changes.** The Vulkan
  validation layer reported this over the case store as
  `VUID-VkShaderModuleCreateInfo-pCode-08740`. `tests/no_module_needs_a_geometry_stage.rs` sweeps
  every public fixture for the invariant; the two arms of the rule are pinned by name in
  `native::tests::interface` (a fragment entry, which must add `Tessellation`) and
  `native::tests::intrinsics` (a tessellation-evaluation entry, which must add nothing).

- A buffer's reflected `access` is now what the module does, not that ORed onto what AIR declared,
  wherever the walk that answers the question is provably complete. `widen_access` has always
  refused to narrow, on the sound grounds that a store through a device address it cannot attribute
  would be missed — but that risk has a boundary, and the boundary was already being computed for
  the footprint. Under Logical addressing, with no pointer rooted at the descriptor escaping into an
  operand slot the walk does not model, there is nowhere else for an access to come from. That is
  the same proof under which the module now decorates the descriptor `NonWritable`, and reporting
  the wider answer beside that decoration made one function state two incompatible facts: the module
  told the driver the buffer is read-only while reflection told the consumer to upload it, barrier
  it, and read it back. AIR's `air.read_write` on a buffer whose body only stores to it describes
  the parameter, not the program. **Measured reach over the 14,579-source corpus: 2258 buffer
  bindings in 1727 modules change access — 1846 `ReadWrite`→`WriteOnly`, 388 `ReadWrite`→`ReadOnly`,
  21 `WriteOnly`→`Unused`, 3 `ReadOnly`→`Unused`.** No SPIR-V, no other reflected field, and no
  status changes; all 1209 device cases re-record byte-identical. `REFLECTION_VERSION` is 55.
  `tests/reflection_access_covers_the_module.rs` gains the direction it could not check before —
  its contradicted-access kernel stores through an `air.read` buffer it never loads, and that buffer
  is now `WriteOnly` rather than the declaration's read ORed onto the module's write — and
  `tests/nonwritable_covers_the_module.rs` pins the two derivations to each other: a binding the
  module decorates `NonWritable` may never reflect an access that includes writes.

- A storage buffer the module provably never writes is now decorated `NonWritable`. Vulkan reads the
  absence of that decoration as a demand rather than as silence: a graphics-stage module declaring an
  undecorated storage buffer requires its consumer to enable `fragmentStoresAndAtomics` or
  `vertexPipelineStoresAndAtomics` whether or not any store exists
  (`VUID-RuntimeSpirv-NonWritable-06340` and `-06341`). Running the candidate Vulkan executor under
  `VK_LAYER_KHRONOS_validation` over all 1209 qualified corpus cases reported those two VUIDs 302
  times; `grep -rn NonWritable src/` returned nothing, because the decoration had never been emitted
  at all. Read-only buffers were therefore asking every consumer for a device feature the shader does
  not use, and on a device without it the pipeline is simply unavailable.
  **11,193 of 14,579 corpus sources change bytes** (8617 kernel, 1474 fragment, 869 vertex, 233
  tessellation-evaluation). Of the 2608 graphics-stage modules that declare a storage buffer, **2558
  now decorate every one of them** and drop the feature demand entirely; 18 are decorated in part and
  32 not at all, and those keep the demand because the VUID is per-variable. 6199 of 6364
  graphics-stage storage buffers are decorated. No source changed status and no reflection changed.
  The proof is the access walk reflection already runs to widen a declared access
  (`reflect::footprint`), not a second one: "did any instruction write through this descriptor" is one
  fact about the module. Three conditions make that walk's silence a proof rather than an absence of
  evidence -- Logical addressing, no observed write, and no pointer rooted at the descriptor reaching
  an operand slot the walk does not model. The third was already computed and folded into
  `has_unbounded_access`; it is now reported separately as `Analysis::escaped`. 291 candidate
  descriptors are refused on the second or third condition. AIR's declared `air.read` is *not* the
  signal and cannot be: the corpus contains buffers declared read-only that the body stores through,
  and `tests/nonwritable_covers_the_module.rs` pins that case by name. That file is also the check
  `spirv-val` does not perform -- measured here, `spirv-val` accepts an `OpStore` straight through a
  `NonWritable` variable, so a wrong decoration is silent all the way to the device. It walks the
  disassembly independently of the translator's own analysis, over every public fixture; an
  independent sweep of the same shape found 0 violations across all 13,318 modules that translate.

- `air.simd_all`, `air.simd_any` and `air.simd_ballot.i64` now answer for Metal's 32-lane
  simdgroup instead of the whole physical subgroup. Every other member of the family already did:
  `air.simd_is_first`, `air.simd_broadcast_first` and the clustered `air.simd_{sum,min,max}`
  reductions all read `SubgroupLocalInvocationId` and partition on it, and corpus source
  `c3441ab4` emitted both models in one function -- an `OpGroupNonUniformFMax ... ClusteredReduce
  %uint_32` four hundred instructions above an `OpGroupNonUniformAny` over the physical subgroup.
  On a driver wider than 32 lanes the votes answered for lanes outside the caller's simdgroup, and
  the ballot's bits were indexed by subgroup lane rather than by `simd_lane_id`. All three were
  also the only members of the family emitted natively rather than as pass lowerings, so this
  closes a native/pass split as well. The vote now reads the caller's own 32-lane ballot word
  (`any` = non-zero, `all` = equal to the same partition's ballot of `true`), and the ballot is
  that word zero-extended; the shared word selection is `subgroup_partition_ballot_word`, which
  `air.quad_active_threads_mask` had open-coded. **Hardening on this hardware:** Apple's subgroup
  is exactly 32, so the answers are unchanged here, and the new authored fixture
  `kernel_simd_vote_and_ballot` confirms byte-identical device output for all five symbols. Four
  of 14,579 modules change bytes; two more change only type-declaration order. No status and no
  reflection changed, and the changed modules stop demanding `GroupNonUniformVote`.

- `air.simd_broadcast` and `air.simd_shuffle` now resolve their lane operand inside the CALLER's
  Metal simdgroup instead of treating it as an absolute subgroup lane. Every other member of the
  family already did -- `simd_shuffle_up`, `_down`, `_rotate_down`, `_and_fill_*` and the
  `simd_broadcast_first` arm immediately above these two all read `SubgroupLocalInvocationId` and
  rebase onto `lane & !31` -- so this was one fact derived two ways in one function. On a driver
  whose subgroup is wider than 32, lane 40 asking for simd-local 3 read absolute lane 3, which
  belongs to a different simdgroup. Both now go through one shared `metal_simd_absolute_lane_u32`,
  which also masks the index so `base + index` cannot name an id past the end of the subgroup.
  **Hardening, not a fix on any runtime here:** Apple's subgroup is exactly 32, so the rebase is a
  no-op on this hardware, and the two authored cases added on one of the changed modules confirm
  byte-identical device output. Ten of 14,579 modules change bytes; no status and no reflection
  changed. The mask is a defined refinement, not a Metal match -- measured on an M3 Max,
  `simd_shuffle(v, lane + 32)` returns lane `lane & !3`, so Metal has no out-of-range answer to
  reproduce.

- `air.dispatch_threads_per_threadgroup` no longer reports the executing threadgroup size. It was
  decoded as `KernRole::ThreadsPerThreadgroup` on the stated ground that "`vkCmdDispatch` issues
  whole workgroups only, so under Vulkan the two denote the same value". That premise is false for
  this translator, because `KernelDispatchPlan` emulates Metal's `dispatchThreads:` by giving each
  region its own specialized `LocalSize` -- so a tail region's threadgroup genuinely is shorter, and
  aliasing the roles made the requested size shrink with it. A Metal probe settles what the two
  attributes answer: for a ten-thread grid in groups of four, the last threadgroup reports
  `threads_per_threadgroup == 2` and `dispatch_threads_per_threadgroup == 4`. The requested size is
  now its own `KernRole::DispatchThreadsPerThreadgroup`, emitted as plain constants from
  `TransformOptions::kernel_local_size` -- the same value a caller decomposes the plan from, so it
  is fixed at translation time even where the grid is not, and deliberately not routed through the
  region spec constants. Two corpus modules change bytes; no status and no reflection changed.
  `exact_thread_regions_report_the_requested_threadgroup_size_unchanged` runs both facts through all
  four regions of a 10x3 grid in 4x2 threadgroups on the device and fails without the split.

- Recovering an inlined helper's local pointer field no longer forwards a source that has stopped
  being an opaque handle. The pass replaces a handle-typed load with the value that was stored into
  the same root/member slot, and it only filtered the load side. A device buffer parameter is
  pointer-typed when its store is recorded but a `ulong` address by the second replay, so the
  replacement handed every consumer a 64-bit integer where a handle belonged; an access chain rooted
  on it can descend nothing, and the native emitter refused sixteen corpus modules with
  `base %N has type TypeInt 64, which is not a pointer type`. The pass now declines only when the
  source is not one of the handle shapes the load side already accepts. Two narrower-looking
  variants are both wrong: requiring the two types to be *equal* trades four recoveries for eight
  regressions, because a pointer source legitimately changes pointer type across the boundary, and
  requiring only that a pointer load keep a pointer source breaks the case the second replay exists
  for -- an entry texture parameter is a loaded image id by then, and declining it folds a real
  `OpImageQuerySize` to zero with no status, reflection or byte-count signal at all. Four modules
  translate that did not (`4115808d`, `c159cd71`, `de9eb706`, `f0673eb9`), no module regressed, no
  reflection changed, and no other module's bytes move. The remaining twelve of the sixteen are
  refused earlier, for an atomic on a non-atomic storage class, which the forwarding was masking.
- `metal2vulkan` now rejects an unrecognized option instead of using it as the output path. Both
  positionals fell through a catch-all arm, so `metal2vulkan in.ll -o out.spv` wrote a file named
  `-o` into the working directory and never wrote `out.spv` -- and said `wrote -o` while doing it. A
  misspelled `--stag vertex` failed the same way. A leading dash is now always a flag, and an
  unknown one is an error naming the positional form.

- The synthetic index an argument-buffer-embedded texture binds at is now derived from the entry's
  parameter order rather than from the order the AIR metadata list names its `air.indirect_buffer`
  nodes. `embedded_synthetic_texture_index` fixes where the synthetic indices start and the walk
  hands out `K`, `K+1`, ... in list order, so a consumer binding by that index has to reach the same
  order the translator did -- and reversing the argument list of one corpus kernel moved the same
  embedded texture from `Binding 32` (sampled) to `Binding 480` (storage). Sixteen kernels were
  order-dependent this way under the metamorphic sweep. Parameter index orders them because it is
  unique per argument; the Metal `[[buffer(N)]]` slot is not, since mutually exclusive
  function-constant alternatives share one. The order now belongs to `ArgumentBuffers`, whose only
  constructor sorts, and the three walkers take that type and nothing else, so a fourth caller
  cannot pass an unsorted list. Every AIR argument list in the local corpus is already in parameter
  order, so the status A/B and the reflection diff over all 14579 sources are both zero changes.

- The AIR static-initializer evaluator reads 64-bit function constants, and a store keeps its own
  operand width. `ulong` is a Metal function-constant type -- 47 of the local corpus's
  `air.fc_initializer` globals are `i64` -- and the initializer reader accepted only 8, 16 and 32
  bits, so every predicate derived from one stayed unknown and each caller read that as "not enabled
  by default". Separately, every stored integer was narrowed to 32 bits regardless of the store's
  type, which discards exactly the half an `lshr i64 %bits, 63` predicate reads; `trunc ... to i32`
  had the same shape, masking only `i8` and `i16` destinations. Unresolved function-constant gates
  in a 1823-source sample fall from 37 to 28, in 9 modules to 7, with five declarations moving from
  "unknown, therefore treated as absent" to enabled. Status A/B and reflection diff over all 14579
  local corpus sources: zero changes either way -- the wrong values were being read into decisions
  that happened to land the same way, and the reason to fix them is that nothing guarantees the next
  one will.

- The AIR static-initializer evaluator folds the ordered `icmp` predicates (`ugt`, `uge`, `ult`,
  `ule`, `sgt`, `sge`, `slt`, `sle`), the integer opcodes it was missing (`sub`, `udiv`, `urem`,
  `sdiv`, `srem`, `ashr`) and the `llvm.umax`/`umin`/`smax`/`smin` intrinsics. It read only `eq` and
  `ne`, leaving 648 of the 10938 `icmp`s in the local corpus's static initializers unevaluated --
  and an unevaluated function-constant predicate reads as "not enabled by default", which every
  caller spends as a fact. Two corpus fragment shaders reported and emitted **no color attachment at
  all**: their only `air.render_target` sat behind
  `%15 = add nsw i32 %14, -3` / `%16 = icmp ult i32 %15, 2` / `xor i8 %17, 1`, a predicate that is
  TRUE under the all-constants-zero variant. Both now declare `Location 0` in the module and in
  reflection.

  Signedness is the part worth stating: the evaluator carries every integer as a masked `u64`, so
  `-1` compares GREATER than `1` unless the sign is recovered from the OPERAND's width, which is
  what the shader above depends on. A division by zero stays unknown rather than folding, as an
  over-wide shift already did. Status A/B over all 14579 local corpus sources: zero changes; the
  reflection diff moves exactly six modules -- two gaining the color attachment above, four kernels
  whose buffer footprint tightens because a newly folded constant changed which accesses are
  reachable.

- A kernel's synthetic `[[stage_in]]` buffer slots are now allocated in PARAMETER-INDEX order rather
  than in the order the AIR argument list names its nodes. Both `stage_input`'s lowering and
  reflection read this one allocation, so a producer that emitted the same kernel with its argument
  nodes reordered would have handed a consumer different slots for the same shader -- an ABI that
  depends on a metadata layout nothing else reads. All 14397 kernel, fragment and vertex argument
  lists in the local corpus are already in parameter-index order, so ordering them changes no
  output; the status A/B over 14579 sources is zero changes. The set of roles a synthetic slot must
  avoid is now `KernRole::buffer_table_slot`, an exhaustive match rather than a list with a
  `_ => None` tail: a role added later that names a buffer-table index has to answer it, instead of
  silently getting a stage input allocated on top of it.

- A texture the pipeline variant does not declare no longer takes a live texture's descriptor slot.
  Metal compiles `[[texture(fc_expr)]]` into a RUNNING SUM over the arguments the pipeline enables,
  emitted as a global the module's `air.static_init` constructor computes; the sum is a given
  argument's Metal slot only while that argument is one the variant enables. For an argument whose
  own `air.function_constant` gate the module's initializers drive to zero, the same global holds
  wherever the sum stopped -- which is a LIVE argument's slot. Reading it anyway put 17 differently
  shaped sampled textures on `Binding 38` of one corpus fragment shader, and a
  `texture2d<float, write>` beside a 128-element `array_ref` on `Binding 480` of a kernel.
  Reflection asked a consumer to bind two different textures to one Metal index -- no descriptor-set
  layout describes that -- and the store the shader aimed at the argument that is not there landed on
  the texture the pipeline did bind. 338 of 14579 local corpus sources emitted a module with two
  resources on one descriptor slot; the fix takes that to 92, all of which are either Metal's own
  mutually exclusive alternatives sharing a slot they STATE as a literal, or a module whose
  all-constants-zero variant is degenerate for a reason of its own.

  The rule keys on how the slot is spelled, not on the gate. `[[texture(0), function_constant(a)]]`
  beside `[[texture(0), function_constant(!a)]]` is Metal's way of declaring mutually exclusive typed
  alternatives; the slot is written down rather than summed, so it stays that argument's slot and the
  binding is kept. All 4851 ptr-spelled `air.location_index` operands in the local corpus are globals
  the static initializer writes; none is a constant global.

  Dropping those descriptors made three texture lowerings answer a resource that is no longer there,
  so the absent-resource contract now covers the whole texture-operation surface rather than only the
  reading half:

  - A `air.write_texture` whose image operand is an absent resource stores NOWHERE, the way
    `lower_null_texture_result` already reads zero for one, under the same
    `unsurfaced_embedded_resources` guard. The alternative is not "store somewhere harmless" but
    "overwrite the texture the pipeline did bind".
  - Recovering a Private placeholder onto "the module's only image binding" now also requires that
    image to have the shape the AIR intrinsic's own stable symbol states. Exactly one candidate is
    not evidence on its own, because an absent argument stands beside the live binding as a
    differently shaped alternative of it. The three near-identical recovery helpers
    (`single_storage_image_for_private_write`, `single_sampled_image_for_private_read`,
    `single_image_for_private_query`) are now one `recovered_image_for_private_operand`, so the rule
    is stated once, and one `intrinsic_texture_shape` parses the shape for it, for the write ABI and
    for `air.get_null_texture_*`.
  - A texture operand is absent whether it is the Private placeholder or the `OpConstantNull` that
    placeholder holds, which is what resolving a load through it reaches. Recognizing only the first
    left an `air.get_array_size_texture_2d_array` on the second to take the default non-arrayed 2D
    shape and refuse.

  Status A/B over all 14579 local corpus sources: **zero** changes.
  `tests/gated_texture_descriptor_slots.rs` pins each of the four parts from both sides, including
  the literal-slot control that scopes the rule.

- A texture an argument buffer holds inside a nested struct is no longer treated as an absent
  texture. Metal lets an argument-buffer member be a user struct that itself holds a texture,
  sampler or buffer; AIR spells that with an `air.struct_type_info` PREFIX naming the wrapper's own
  member list, and the member's `air.indirect_argument` suffix is then an `i32` where a flat member
  carries the node ref describing the resource. The embedded-argument walk reads only the flat form,
  so the wrapped texture never became an embedded descriptor: the member decoded as opaque storage,
  the handle load produced a Private placeholder, and the sample took the path that answers ZERO for
  a resource the pipeline does not provide. That answer is right for a `[[function_constant]]`-gated
  texture whose constant is off and wrong here -- the result was a fragment shader that samples
  black, in a module that passes `spirv-val`, reports a consistent reflection, and gives a consumer
  nothing to notice. The decode now reports each such resource, and a texture operand the resource
  binding could not recover refuses in a module that declares one rather than answering zero. 52 of
  14579 local corpus sources declare the shape; the status A/B moves exactly 12 of them from OK to an
  honest FALLBACK and changes nothing else.

  Surfacing them properly instead of refusing needs the argument-index rule for a nested member,
  which the corpus does not settle: three modules spell the outer `i32` as 0, 1 and 201 beside nested
  `air.location_index` values of 0, 10 and 0. `tests/argument_buffer_wrapped_resource.rs` pins the
  flat form still binding, the decode naming what it cannot surface, and both the sampled and written
  wrapped forms refusing.

- A `[[function_constant]]`-gated `air.stage_in` is the vertex stream it declares. The kernel decode
  collapsed a gated stage-in attribute back to the wrapper, so the parameter lowered to `OpUndef`:
  a skinning kernel that reads `position` always and `normal` behind a `needNormal` constant read
  nothing at all for the normal, in a module that validates and whose reflection agrees with it.
  120 gated `air.stage_in` arguments in 52 of 14579 local corpus sources; 12 of the 2880-source
  sample gain 24 `KernelStageInput` bindings AND the same 24 descriptors in the emitted module.
  Status A/B over all 14579 sources: zero changes.

  `FC_PROMOTED_RESOURCE_ROLES` now states the rule for belonging to it: **promoting a role has to
  change the emitted module, not only reflection.** The standing justification for keeping a gated
  descriptor is that the resource-using arm may be enabled later, but an argument whose uses fold
  away leaves an unreferenced variable that `module_cleanup` removes, and reflection then asks a
  consumer to create and bind a descriptor no instruction touches. Measured that way,
  `air.indirect_buffer` -- the other role the fragment decode promotes and the kernel and vertex
  decodes do not -- gains 225 reflected bindings across 28 modules and **zero** module bindings, so
  it stays out, with the measurement recorded next to the list. `texture`, `sampler` and `imageblock`
  are already wrapper-neutral corpus-wide. `tests/gated_stage_inputs.rs` now translates one
  declaration with and without the wrapper in all three stages and requires the two modules to be
  identical. `REFLECTION_VERSION` is now 45.

- A `[[function_constant]]`-gated `air.vertex_input` is the vertex attribute it declares. The wrapper
  says WHEN a parameter is live, not whether it exists: Metal reports the attribute, the application
  binds a vertex buffer for it, and a pipeline created with the constant enabled reads it. The
  fragment decode has always read a wrapped role straight through the marker, so the mirror-image
  `air.fragment_input` was always the varying it declares; the vertex decode read its roles through a
  resource-promotion list that named no stage-input role, so the argument became no `Input` variable
  at all -- it lowered to `OpUndef`, and reflection omitted the attribute. Nothing reports that: the
  module validates, and reflection agrees with the module about the wrong answer. 573 gated
  `air.vertex_input` arguments in 89 of 14579 local corpus sources, every one with a real
  `air.location_index` and none colliding with another attribute in its own module. Over the
  2880-source sample: 15 modules change, 59 `Input` `Location`s and 59 `vertex_attributes` gained and
  none lost, `vertex_attributes` the only reflected key that differs anywhere, and one vertex shader
  goes from reporting no attributes at all to reporting its eight. Status A/B over all 14579 sources:
  zero changes.

  The vertex entry-parameter decode now states the roles it reads past the wrapper as one promotion
  set, instead of patching the resource classifier's answer afterwards at the call site -- which is
  how `patch_input` came to be corrected there while `vertex_input` was not corrected at all. The one
  remaining call-site adjustment, a wrapped texture whose `air.location_index` is still unassigned, is
  a demotion the promotion set cannot express and is documented as such.
  `tests/gated_stage_inputs.rs` translates one declaration with and without the wrapper, in both
  stages, and requires the two modules to be identical. `REFLECTION_VERSION` is now 44.

- A varying AIR declares reaches the module even when the shader never writes it. `stage_output`
  skipped the store for an output whose value is statically `OpUndef` -- nothing useful to write, so
  nothing written -- which left the Output variable unreferenced, and `module_cleanup`'s
  unreferenced-global rule then removed the variable, its `Location` decoration and its entry-point
  interface entry together. That rule is right for a descriptor, whose `Binding` a consumer would
  otherwise have to satisfy for nothing, and wrong for a stage output, which is one half of a
  linkage contract: the fragment shader compiled from the same Metal varying struct declares the
  matching Input, and Vulkan requires every consumed input to have a producing output at that
  `Location`. AIR declares the member because Metal declares it; that the shader leaves it undefined
  makes its VALUE undefined, not its existence. Fragment outputs deliberately keep the old rule --
  an attachment is memory, and one nothing writes is better left unwritten than written with garbage.
  Over the 2880-source sample: 30 modules change (27 vertex, 3 tessellation evaluation), 87 Output
  `Location`s gained and none lost, no Input change and no reflected field change at all; the
  reflection-declares-a-varying-the-module-does-not oracle goes from 30 modules to 0. Status A/B over
  all 14579 sources: zero changes. `tests/vertex_output_declared_locations.rs` pins both halves of
  the rule, including the fragment attachment that stays unwritten.

- A `[[function_constant]]`-gated `air.vertex_output` member keeps the varying slot it declares. A
  vertex return struct and the fragment parameter list it feeds are one Metal declaration compiled
  twice, and Vulkan links the two by `Location`, which this translator assigns positionally from the
  AIR interface list -- so the numbering rule has to be a pure function of that list and identical in
  both stages. It was not: the fragment decode reads a wrapped role straight through the
  `air.function_constant` marker and numbered the gated member, while the vertex output decode read
  its roles through a promotion list that named no output role at all and dropped it, numbering every
  member behind it one slot lower. Both modules validate, both report the same varyings by name, and
  the pipeline links -- the fragment simply reads a different interpolant than the vertex wrote, for
  every varying declared after a gated one. 507 gated `air.vertex_output` members in 156 of 14579
  local corpus sources, 106 of them with an ungated member behind the gated one, which is the shape
  that shifts; 1240 gated `air.fragment_input` parameters in 292 sources on the other side. A
  reflection A/B over the 2880-source sample finds 30 modules differing and the `varyings` key is the
  only one that differs in any of them: 22 have a location that now names a different varying (the
  shift), and 8 declared every varying under a function constant and so reported none at all. Status
  A/B over all 14579 sources: zero changes.

  A gated BUILTIN return member still needs the module's own initializer to turn it on, which is the
  distinction the fragment decode draws with `FRAGMENT_SYSTEM_VALUE_ROLES` and for the same reason:
  nothing writes a builtin whose predicate is off, and declaring `Layer` or `ViewportIndex` anyway
  puts a device capability in the module for a value nothing produces. Four vertex sources declare
  TWO gated `air.point_size` members under mutually exclusive predicates, and promoting both emits
  two `PointSize` builtins in one entry point, which is invalid.

  Structurally, the vertex output roles are now one table, `VERTEX_OUTPUT_ROLES`, that is BOTH the
  set an `air.function_constant` wrapper is resolved against and the mapping the decode applies -- a
  role cannot be lowered without being read past the wrapper, or read past the wrapper without being
  lowered. Those were two spellings of one fact and they disagreed. `gated_role` is now the one place
  a wrapper is resolved, shared with the resource classifier.
  `tests/gated_varying_locations.rs` translates one gated varying struct as a vertex and as a
  fragment and compares the emitted `Location` decorations, which is the property that was broken and
  which neither module can fail on its own. `REFLECTION_VERSION` is now 43.

- A `[[function_constant]]`-gated `sampler` argument is a descriptor in every stage, not only in
  fragments. Metal declares a gated `texture2d` and the gated `sampler` it is sampled through
  together, and the fragment decode reads a wrapped argument's role straight through the
  `air.function_constant` marker, so it always bound both. The kernel and vertex decodes read theirs
  through a promoted-role classifier whose list named `texture`, `imageblock` and the function
  tables but not `sampler` -- so the same two lines of AIR produced a texture descriptor and no
  sampler descriptor, the sampler parameter never became one, and the sample fell through to the
  translator's own nearest/clamp default. 63 gated sampler arguments across 21 of 14579 local corpus
  sources; 9 of them, in 6 modules, are kernel or vertex. Status A/B over all 14579: exactly one
  module goes FALLBACK -> OK -- the one whose sample actually reached the substitution, which the
  previous entry now refuses -- and it reports its sampler at the Metal slot AIR declared. Four of
  the remaining five already translated and gain exactly the 7 `Sampler` bindings that were missing,
  with nothing removed and no other reflected field changed; the sixth still fails on a Metal visible
  function table. A reflection A/B over the 2880-source sample finds one module gaining one
  `Sampler` binding and no other content difference at all -- every remaining diff is the version
  field itself. The promotion list is now one named `FC_PROMOTED_RESOURCE_ROLES` rather than one
  match arm per role -- separate arms are what let `sampler` go missing next to `texture` --
  and `tests/gated_system_values.rs` translates one gated texture+sampler declaration in all three
  stages, which is the property that was broken. `REFLECTION_VERSION` is now 42.

- A sampler operand the translator could not resolve to a sampler is refused instead of being
  answered with the translator's own default. `valid_sampler_value` ended with "if it is still
  pointer-shaped, load the synthesized nearest/clamp sampler". That is correct for exactly one
  operand -- `air.get_read_sampler()`, which AIR declares stateless and whose consumer ignores the
  sampler -- and wrong for every other one, because a pointer-shaped sampler operand is a state the
  shader chose that an earlier pass failed to carry. The commonest form is a runtime `select`
  between two `__air_sampler_state` globals: the emitter has no SPIR-V pointer value for it and
  leaves a private placeholder, so both states are gone by the time the sample lowers and the module
  went on to validate, bind and reflect as though the shader had asked for nearest/clamp. Which ids
  `air.get_read_sampler()` produced is now recorded before AIR-call lowering rewrites them
  (`Ctx::read_sampler_values`), since after the rewrite the two forms are indistinguishable by type.
  Status A/B over all 14579 local corpus sources: 9 modules move from a silently substituted sampler
  to a clean FALLBACK, and nothing else changes. `tests/must_fallback.rs` pins the selected form
  from both sides, and a cube read through `air.get_read_sampler()` -- the one operand the default
  may still answer -- is pinned in `src/native/tests/textures.rs`.

- Reflected translation no longer asks a consumer to create a `VkSampler` at a binding the module
  does not declare. `StaticSampler` is a descriptor translation invents -- nothing in Metal's API
  puts one there, so the only reason a consumer creates it is that this reflection asked. Reflection
  named one per `!air.sampler_states` entry, which is every constexpr sampler state the source
  wrote, but a state no sample reads leaves a variable nothing references and the module drops it.
  112 such bindings in 60 of the 2880-source sample; `StaticSampler` was the last translator-
  invented kind still reported without the module's agreement, the other three already being
  reconciled by `reconcile_buffer_address_table` and `report_synthesized_placeholders`. The
  surviving samplers keep their bindings and their states -- the interface pass allocates before the
  drop -- and `tests/reflection_covers_declared_bindings.rs` now checks that direction for every
  invented kind over every public fixture, which is the one
  `assert_reflection_covers_declarations` does not. `REFLECTION_VERSION` is now 41.

- A color attachment whose Location is a function constant is read from the operand after its
  marker, like every other declared slot. `!"air.render_target", ptr addrspace(2) @loc, i32 0`
  states the Location as a global and the dual-source index after it; when the global was one the
  module does not initialize, the decode fell through to "the next `i32` after the marker" and
  answered with that index -- `0` in all 3213 literal-form corpus declarations -- so every such
  output landed on attachment 0 whatever the constant said. 103 of 3316 corpus `air.render_target`
  declarations spell the Location as a global. The parameter or member ordinal now stands in for an
  unknown slot, which is at least unique per member.

  `air.location_index` and `air.render_target` are now one decoder, `meta::declared_slot`, and the
  last scan-forward helper behind them is gone. A/B over the 2880-source sample: no change.

- Constexpr samplers are ordered by one shared, total key, so the state reflection reports at a
  binding is the state the module gave it. Two independent scans put the module's
  `__air_sampler_state` globals in a sequence and pair them off by position: the interface pass,
  which walks SPIR-V `OpName` and hands each global the next free sampler binding, and reflection,
  which walks the `!air.sampler_states` root and reports the decoded state at each of those
  bindings. Both sorted by a key that read the name suffix as a single integer -- which fails on the
  unsuffixed original and on the doubled form LLVM produces when it re-uniques an already uniqued
  name (`__air_sampler_state.163.1608`), collapsing both to `0`. A tie leaves the order to the input
  order, and the two scans have different ones, so every sample got a real sampler with the other
  one's filter, address modes and compare function, in a module that validates and binds cleanly.
  14 of the 5499 sampler-state globals across 14579 local corpus sources carry the doubled form and
  10 modules hold one beside the original. The key now orders by the whole dot-separated suffix with
  the name last, which cannot tie, and lives in one place
  (`meta::static_sampler_name_order`) with the prefix test beside it.

- A sampler argument AIR states is an ARRAY is refused instead of sampled through a default sampler
  nothing asked for. `air.location_index` carries two operands -- the Metal slot and the descriptor
  count -- and Metal spells `array<sampler, 8>` with a count of 8. Nothing read the count, so the
  argument bound as a single sampler; the element loads then resolved to nothing and the sample fell
  back to the synthesized read sampler, dropping the state the shader selected in a module that
  validates, binds and reflects as though one sampler at that slot were the whole story. Two of
  14579 local corpus sources declare one. There is no sampler descriptor-array lowering, so the
  honest answer is a `FALLBACK` naming the parameter and the count.

  The same count now checks the texture-array length the type name states. AIR states that length
  twice and all 76 corpus declarations agree, which is what makes the ABI usable as a standing check
  on the name parse: a `array<texture..., N>` whose two statements disagree, or a count above one on
  a name that is not an array at all, is refused rather than sized from the name alone.

- The `air.location_index` operand pair is decoded by POSITION rather than by scanning forward for
  the next operand of a given kind. Either operand may be a literal or a pointer to a
  function-constant global, and all four combinations occur (112946 / 3404 / 1447 / 439 over 14579
  local corpus sources). The slot decode looked for the first global after the marker, which is the
  COUNT whenever the slot is a literal -- so an argument at `[[texture(1)]]` with a
  function-constant array extent could bind at the extent's value instead of at 1. 439 corpus nodes
  are that shape, 69 of them textures; the buffer half was already dodged by a type-name special
  case for `array_ref<void>`, which the positional read makes unnecessary and which is now gone.
  Latent today -- no corpus module resolves the count global -- and no longer reachable.

- A fixed texture-handle array binds and reflects at the length AIR declares, not at the
  descriptor-ABI ceiling. `array<texture2d<half, sample>, 32>` states its length twice -- in the type
  name and as the second `air.location_index` operand -- and the decoded `TextureShape` already
  carried it, but the interface pass emitted `OpTypeArray` of 128 and reflection reported
  `count: 128` for every handle array, fixed or runtime. A consumer following the reflection had to
  supply 128 valid descriptors (or `descriptorBindingPartiallyBound`) for an array of 8, and the
  same `ResourceBinding` contradicted itself: `texture_shape.array_length` said 8 while
  `descriptor.count` said 128. Both sides now derive from `TextureShape::descriptor_count`; only a
  runtime `array_ref`, whose length is a function constant rather than part of the type name, still
  reports the capacity. 40 of 14579 local corpus sources declare a fixed array (lengths 2, 3, 4, 8,
  32 and 80); every one of the 76 declarations agrees with the second `air.location_index` operand.
  `REFLECTION_VERSION` is now 40.

  The same unification fixes a depth-texture array being bound as a single image. The interface
  pass asked `contains("array<texture")` while the shared type-name decoder asked about
  `array<depth` too, so `array<depth2d<float, sample>, 2>` was not recognised as a handle array at
  all: the module declared one `depth2d` where the shader indexes two, while reflection reported a
  `TextureArray` of 128 at that binding. There is now one predicate -- `TextureShape::array_ref` --
  and `tests/reflection_covers_declared_bindings.rs` requires the reported `count` to EQUAL the
  array the module declares rather than merely cover it, which is what makes over-reporting
  detectable at all.

- `ShaderReflection` reports `max_work_group_size`, the `[[max_total_threads_per_threadgroup(N)]]`
  ceiling translation now enforces. Enforcing a bound a consumer has no way to read leaves it
  guessing at the dispatch shape; reflection reports the ceiling and does not enforce it, because
  discovering the bound is the reason to ask. `REFLECTION_VERSION` is now 39.

- A kernel imageblock aliased onto the implicit one is refused instead of given tile-local scratch.
  An explicit imageblock is ordinarily per-tile scratch, undefined on entry, and the interface gives
  it a `Private` array -- an honest refinement. `air.alias_implicit_imageblock` on the same node
  states the opposite: the storage *is* the implicit imageblock, the render targets the rasterizer
  has already written, so the kernel's first read is of framebuffer content and its writes have to
  land back there. The marker was unread, so a tile-resolve kernel loaded an uninitialized `Private`
  array where the rasterized colors should have been, resolved it, and reported success -- with no
  descriptor for the framebuffer anywhere in the module or its reflection for a consumer to notice
  was missing. 30 of 14579 local corpus sources carry the marker; 124 non-aliased explicit
  imageblocks keep the scratch lowering unchanged.

- A shader that encodes into an indirect command buffer is refused instead of translated into one
  that does not. The `air.*_command` encoder families -- `set_pipeline_state_compute_command`,
  `set_kernel_buffer_compute_command`, `draw_primitives_render_command` and their siblings -- were
  lowered to nothing, on the reasoning that the conformance runner observes buffer bytes only. What
  that produced was a module that validates, binds and reflects exactly like the original while
  performing none of the encoding, which for a kernel whose body is entirely ICB encoding is the
  whole shader. Vulkan's device-generated-command extensions describe a different, driver-defined
  layout that no sequence of SPIR-V instructions can produce from these operands, so the honest
  answer is a `FALLBACK` naming the family. Twelve of 14579 local corpus sources reach it (two of
  the 2880-source A/B sample), all of which previously reported success.

  `AirIntrinsicDisposition::NoVulkanEquivalent` now distinguishes a family the translator has
  modelled and refuses from one nothing has modelled yet, so the refusal names the family rather
  than reporting it as unrecognised. Validation's `IndirectCommandBuffer` tooling requirement is
  derived from the same `is_command_encoder_helper` definition: it previously keyed on
  `!"air.indirect_command_buffer"`, a metadata string no harvested AIR module carries, so it never
  fired for a real encoder. It now fires on an encoder call or an `air.command_buffer` argument.

- One encoder produces every `half` constant the translator mints. The native emitter and the
  passes layer each carried their own `f32` -> binary16 conversion, and they disagreed: the passes
  copy rounded half-away-from-zero rather than to even, and combined the rounded significand into
  the encoding with `|` instead of `+`, so a value whose rounding carries out of the significand
  kept the exponent it started with whenever that exponent was odd. `1.999755859375` encoded as
  `1.0` instead of `2.0`.

  This reached emitted modules through the saturating bounds of `air.convert` from a `half` source
  to a narrow integer: the `32767.0` bound of a 16-bit signed convert encoded as `16384.0`, and the
  `8191.0` bound of a 14-bit one as `4096.0`, each halving the range the conversion could produce.
  A `half` above the halved bound saturated to it instead of to the destination's maximum. Across
  2880 corpus sources six modules change, each replacing the upper `FClamp` bound of a `half` ->
  `short` conversion: `OpConstant %half 0x1p+14` becomes `0x1p+15`, against a lower bound of
  `-0x1p+15` that was already right. (`32767` is not representable as a `half`; `0x1p+15` is the
  nearest one, which is what a correctly rounded bound means. `0x1p+14` was not near it.)

  The correct encoder now lives in `crate::float16` and is exercised at every rounding boundary in
  the format: for each pair of adjacent finite halves, the exact midpoint must land on the even one
  and the two neighbouring `f32` values must land on their own side.

- Every attribute an AIR stage root carries is read, and one the translator has no model for is
  refused instead of dropped. All three roots -- `!air.kernel`, `!air.vertex`, `!air.fragment` --
  state their per-entry attributes as extra operands past `(function, outputs, inputs)`, but not in
  one form: `air.patch` and `air.max_work_group_size` are references to a keyed node, while
  `early_fragment_tests` is a bare string on the root itself. Only the vertex decode looked at that
  tail, and only for `air.patch`, so:
  - `[[early_fragment_tests]]` reaches SPIR-V as `OpExecutionMode ... EarlyFragmentTests`. It was
    dropped on 51 of 14579 local corpus sources, which emitted them with Vulkan's default late
    depth test. Under early tests a fragment the depth test rejects performs none of the body's stores; under
    late tests the same shader performs every buffer, texture and imageblock write and only its
    color output is discarded. A fragment that both declares it and writes `air.depth` or
    `air.stencil` is refused: the test runs before the value it compares exists.
  - `[[max_total_threads_per_threadgroup(N)]]` (`air.max_work_group_size`, 439 of the same sources)
    bounds the requested kernel `LocalSize`. A dispatch wider than the ceiling the entry was
    compiled for is refused rather than emitted as a module that validates and runs a shape the
    source ruled out.

  A/B over a 2880-source sample: no status changes, and the only SPIR-V byte changes are the ten
  fragments that declare `early_fragment_tests`, each gaining exactly the one execution mode and
  still passing `spirv-val`. No source in that sample requests a dispatch past its ceiling.

- `reflect_sanitized` decides whether a kernel needs a buffer-address table with the emitter's own
  device-address predicate rather than a line-prefix scan of the AIR text. The scan disagreed with
  the finished module on 63 of 2880 corpus sources -- 49 by reporting a binding no module declares,
  and 13 by reporting none for a module that does, which leaves a consumer building a descriptor-set
  layout without a binding the shader reads. Asking the predicate the emitter itself asks leaves 8,
  four in each direction, and metadata-only reflection remains about three times cheaper than
  translating. Reflected translation, which reads the table off the module, is unchanged.
  `REFLECTION_VERSION` is now 38.
- `with_runtime_sampler` accepts a pixel-coordinate sampler whose LOD maximum is Metal's default.
  The emulation fetches level zero, so only a *minimum* above zero can exclude the level it reads --
  a maximum never can, since validation already requires it to be at least the minimum. Demanding
  both be zero refused the ordinary case: 531 of the 535 pixel-coordinate static samplers across
  2880 corpus sources carry a maximum of 65504, the half-precision limit AIR encodes "unclamped" as.
  The emulation constraints now live on `StaticSamplerState`, the type the lowering consumes, and
  the AIR constexpr-sampler path applies them too -- it previously applied none, so a static sampler
  the emulation cannot reproduce was lowered anyway.
- A vertex function carrying an `air.patch` node the decoder cannot read is refused instead of being
  emitted as an ordinary vertex shader. The node states a post-tessellation evaluation shader's
  domain and control-point count; dropping it dropped the `Quads`/`Triangles`/`Isolines`,
  `SpacingEqual` and `VertexOrderCcw` execution modes Vulkan requires of that stage along with every
  per-patch input the pipeline wires, and the module that came out validated, bound and reflected
  while drawing the wrong geometry. The node is also located by the `air.patch` marker it carries
  rather than by its position in the vertex root.
- An entry buffer whose declared AIR layout could not be attached to the emitted parameter now says
  which of the two things happened. A byte-addressed buffer is emitted as its raw contents -- a
  pointer to a word, or the block a storage buffer requires wrapping a runtime array of one -- and
  has no members for the declared offsets to land on, which is not the same as two structural
  descriptions disagreeing. Both reported `EmittedShapeMismatch`; over 2880 corpus sources 1733 of
  the 1805 unmapped parameters are the former, and reporting them as mismatches buried the 72 where
  the shapes really do differ.
- A buffer member `air.struct_type_info` names without describing -- a user struct or class
  mentioned by name only -- decodes as `AirType::Opaque { size }` at the byte size the member tuple
  declares, instead of as a 32-bit `Float`. The float leaf named a type AIR never stated and sized
  every such member at four bytes: over 2880 corpus sources, 2513 members across 817 modules are
  opaque, and 2357 of those declare a size other than four. The invented interior also failed to
  match the member the emitter produced, which discarded the declared offsets for the whole buffer
  rather than for the one member that provoked it. `REFLECTION_VERSION` is now 37.
- `validate_descriptor_abi` rejects a reported member layout that reaches past the argument holding
  it. AIR states an argument's size and its member layout as two independent facts and reflection
  reconstructs them independently, so a layout that over-runs means some member's storage was
  mistaken and every member after it names an offset the shader never reads. Nothing else catches
  it: a consumer packs its upload at those offsets, and a buffer whose reconstruction is that far
  off is emitted as raw bytes with no struct type to compare against. Reaching short stays valid,
  since the declared size is a `sizeof` and carries tail padding -- 1340 of the corpus's reported
  layouts legitimately stop short, and none over-run.
- A texture AIR declares write-capable binds as a storage image even when the shader body only
  queries its size. The binding class came from what the body did, and a size query counted as a
  sampled-image use, so such a texture bound in the sampled-texture band while reflection reported
  it in the storage-texture band -- a consumer wrote its descriptor where the shader does not read
  it. Only sampling decides the class now; every other AIR texture operation has a form for either.
- Scratch files under a caller-supplied `tmp` are named per process and per call, so two callers
  sharing one directory no longer overwrite and delete each other's input to `spirv-val` or
  `llvm-dis`. The shared name surfaced as a validation failure in a module that was fine.
- `get_width()`/`get_height()` on a `texture_buffer` translates instead of falling back. SPIR-V
  allows `OpImageQuerySizeLod` only on a 1D/2D/3D/Cube image with `MS` 0 and a `Sampled` operand
  that is not 2; a `Dim Buffer` image must use the LOD-less `OpImageQuerySize`. One
  `image_size_query_op` now states that rule for both size-query lowerings, which had drifted --
  only one of them handled multisample images and neither handled buffer textures.
- A descriptor the translator synthesizes only to type an AIR value is retracted whenever nothing
  consumes that value. This already covered `air.get_read_sampler()`; it now covers
  `air.get_null_texture_*()` too, so a shader that merely asks whether an optional attachment is
  bound no longer demands a texture descriptor it never reads. The rule is stated once, over the
  set of synthesized placeholders, rather than per variable.

### Removed

- `TransformOptions::simd_cluster32` and the `--simd-cluster32` CLI flag. The option could not be
  observed: `options_for_air` force-enabled it for any module whose text contained `@air.simd_`,
  and every arm it gated is reached only from an `air.simd_*` lowering, so a caller who passed
  `false` silently got clustering anyway and a caller who passed `true` changed nothing. The
  32-lane partition is now unconditional, which is what the shuffle half of the family already
  did -- `metal_simd_absolute_lane_u32` never consulted the flag. Removing it deletes the string
  sniff, five unreachable whole-subgroup arms, and one test that asserted a state that cannot
  occur; `air.simd_is_first` and `air.simd_broadcast_first` now take their partition base from the
  shared `metal_simd_lane_local_u32` / `metal_simd_lane_base_u32` helpers instead of open-coding
  it a third time. Measured: 0 of 14,579 corpus modules change a byte.

### Added

- `translate_sanitized_native_specialized_reflected_with_options` and
  `translate_sanitized_native_linked_specialized_reflected_with_options`: the function-constant
  specialized and linked translation entry points now have reflected forms, so a caller that binds
  the module can have the reflection OF that module. `reflect_sanitized_specialized` constructs no
  module and therefore answers the AIR type name for the facts only a finished module knows, which
  is what its doc comment has always said; before this the specialized paths left no other choice.

- Reflection schema v36: an argument-buffer member that holds a resource handle is reported at the
  eight bytes it occupies instead of as the type it points at. Metal spells such a member with its
  pointee's name -- `char` for a `device char *`, `float4x3` for a `device float4x3 *`, a
  `texture2d<...>` for a texture -- and reading that name as the member's storage put a 64-byte
  matrix where the buffer holds an address, shifting the meaning of every member after it.
  `MTLGenericBVHData` reported a 120-byte layout for its 72-byte argument; over 2880 corpus sources
  1736 members across 9 named types and 47 opaque ones were mis-sized this way. The member's
  `air.indirect_argument` node decides: every role but `air.indirect_constant` is a reference. The
  emitted SPIR-V follows for the 21 modules whose buffer was typed from this layout, which had
  declared four-byte floats at eight-byte member offsets.
- Reflection schema v35: a descriptor-backed buffer's `access` is widened to cover the loads and
  stores the finished module performs through it. AIR's declared access is not a guarantee about the
  body -- over 2880 corpus sources, 20 buffers reflected `ReadOnly` are stored through, 9 reflected
  `WriteOnly` are loaded from, and 3 reflected `Unused` are both -- and a consumer barriers and
  stages from this field. It only ever widens, since the analysis can miss an access through a
  device address it cannot attribute. `reflect_sanitized` builds no module and keeps the declared
  classification.
- Reflection schema v34: `texture_shape` describes the `OpTypeImage` the module declares at that
  binding, not the shape the AIR type name implies. A `texturecube` that is only texel-read binds as
  a `Dim2D` array, because SPIR-V has no cube texel fetch, and a consumer that created a cube view
  from the old shape would have built a view Vulkan rejects against that image variable. A binding
  whose image variables do not all declare the same type keeps the type-name-derived shape, as does
  `reflect_sanitized`, which builds no module.
- Reflection schema v33: reflected translation reports the descriptors the passes synthesize with no
  Metal argument behind them -- `SynthesizedNullTexture` for a read `air.get_null_texture_*()`
  placeholder and `SynthesizedReadSampler` for a consumed `air.get_read_sampler()` one. Nothing in
  the AIR metadata describes those bindings, so a descriptor-set layout built from reflection alone
  did not cover the module. The list comes from the passes and is filtered to what the finished
  module still declares; the module supplies each resource's class.
- Reflection schema v32: `ShaderReflection::bindings` reports each resource exactly once. A fragment
  entry with an `air.indirect_buffer` argument previously listed every argument-buffer resident
  twice, so a consumer sizing its work from that list allocated and wrote each of those descriptors
  twice. `validate_descriptor_abi`, which both reflection paths run, now rejects two entries that
  agree in every field.
- Reflection schema v31: implicit imageblock render-target planes are reported at every stage whose
  module calls the `air.load/store.implicit_imageblock.*` intrinsics, not only in compute. The
  interface pass materializes the descriptor from the call, so a fragment entry that reads its
  render target back through the imageblock now reflects the binding it declares. Reflected
  translation also reports the buffer-address table the finished module declares, rather than the
  one predicted from an AIR text scan; `reflect_sanitized`, which builds no module, keeps the
  prediction and documents it as an approximation.
- Bounded translation censuses now consume exact authored visible-function-table populations inside
  the isolated worker, using hash-local case lookup and the same checked linkage mapping as Vulkan
  candidate execution; only rows with no authored population remain linkage-required.
- Public `specialize_function_constants_zero` helper for baking discovered Metal function
  constants to their zero/default values, including branch pruning and removal of now-dead entry
  interface globals.
- Public byte-exact function-constant specialization, reflection-only AIR inspection, authored
  linked-function/table specialization, and vertex-observer generation APIs.
- AIR-level scalar/vector function-constant translation APIs that specialize stable
  `air.fc_initializer` globals before metadata, resource-interface, and CFG construction.
- Reflection schema v30: consumer metadata now covers decoded static samplers, texture shape and
  access, argument-buffer resources, kernel stage inputs, tessellation, imageblocks, exact function-
  constant ABI types, buffer extent/access classification, and conservative final-module static and
  invocation-strided buffer footprints, plus runtime sampler/storage-image specialization and the
  effective descriptor layout and kernel dispatch-grid ABI.
- Versioned, caller-selected descriptor layouts through `TransformOptions`, allowing independently
  translated stages to use distinct Vulkan descriptor sets and binding ranges while preserving the
  existing layout as the explicit default.
- Runtime pipeline-state specialization by Metal resource index for dynamically bound samplers and
  writable storage images, including embedded argument-buffer textures, exact reflected sampler and
  image-format state (including two-channel `Rg32Float`), and host storage-image feature checks.
- A typed exact-thread dispatch ABI that decomposes Metal boundary workgroups into at most eight
  Vulkan regions, with specialized local sizes, logical grid bases, public planning helpers, and
  fixed, dynamic, and explicitly proven whole-workgroup forms.
- A task-oriented translation and reflection integration guide plus a compiled serde reflection
  example.
- Additional stage-interface support for fragment `[[point_coord]]`, `[[primitive_id]]`,
  `[[sample_id]]`, and `[[render_target_array_index]]`, flat varyings, framebuffer-fetch color
  inputs, vertex builtins, and fragment outputs with nonzero render-target locations.
- Broader native translation support for texture arrays, storage-image arrays, texture
  gather/sample/read/write variants, half/integer render-target formats, scalar 64-bit integer
  arithmetic, and Workgroup memory patterns used by shared-memory reductions.
- Native lowering and reflection for linked functions, tessellation patch inputs, ray/intersection
  queries and result fields, argument-buffer resources, and implicit, custom, and direct-layout
  imageblocks.
- Native distributed `simdgroup_matrix` 16x16x16 multiply-accumulate lowering for the observed
  f32, f16, bf16, float8, and signed/unsigned i8 AIR element combinations, including dynamic
  transpose operands and 32-lane tile ownership.
- Authored validation contracts and executable Metal/Vulkan cases for tessellation, depth/stencil
  attachments, framebuffer fetch, multisample and buffer textures, narrow vertex attributes,
  vertex side effects, function constants, argument buffers, imageblocks, ray intersections, and
  exact empty observations for entries with no reflected writable output, including rejection of
  vertex positions and varyings as observable outputs.
- Byte-exact Metal/MoltenVK evidence now covers every public synthetic AIR fixture, including vector
  function-constant lanes, barrier-bearing boundary workgroups, narrow vertex attributes, combined
  depth/stencil, and custom and implicit imageblocks, plus indexed local identities for
  unconditional word stores, signed min/max initialization, grid-indexed byte clears, and exact
  grid-index sequence generation, parameter-free scalar and vector fragment returns (including a
  sparse color-location-2 target and scalar half targets), and constant-zero vertex positions
  observed through their exact non-rasterizing framebuffer result, plus two-channel float fragment
  targets, fragment position-depth extraction, and a pixel-coordinate sampled vertical convolution
  with explicit sampler state, weights, bounds, and bias. Constant scalar/vector fragment evidence
  now also covers half, float, and uint targets, while dual-output fragment clears qualify both
  attachments independently for half4 and float4 formats. Authored Vulkan draws now mirror Metal's
  default clockwise front-face winding, with byte-exact coverage for `[[front_facing]]`, primitive
  IDs, generated float/half fragment varyings across additional independently indexed AIR
  identities, scalar float-to-half varying conversion, and constant outputs whose unused stage
  inputs include viewport indices. Additional authored vertex evidence observes generated
  ushort-ID positions and independently captures passthrough position and float2 varying outputs;
  fragment evidence also covers flat uint varyings, scalar depth expansion, and independently
  selected sparse color attachments. Exact vertex observation now additionally covers forced-zero
  clip-space depth, paired position/float2 passthrough outputs, and deterministic position results
  beside explicitly undefined return members; fragment coverage includes alpha-to-half4 conversion,
  aggregate flat-uint returns, and viewport-bearing varying passthrough.
  The indexed evidence set additionally covers float2/float3 clip-position expansion, paired
  float2 vertex varyings, scalar and vector float-to-half fragment conversion, flat-boolean color
  selection, fog-alpha multiplication, empty texture kernels that preserve their selected bytes,
  and a rasterization-disabled vertex side-effect store.
  A complete indexed texture-copy family now has byte-exact coverage for flat and array-layer
  half-texture reads, including explicit function-constant specialization of the flat alternative.
  The complete indexed `backgroundFragment` family now has fresh byte-exact Metal/MoltenVK evidence
  for constant and sampled outputs, including two function-constant-selected gradient textures.
  The complete eight-identity `Clear::clear_fragment{,2,3,4}` suite now has byte-exact evidence for
  constant-buffer float-to-half conversion and aggregate output mapping through four render targets.
  Attachmentless fragment draws now execute on Metal with explicit render-pass dimensions, without
  synthesizing an observable attachment. The two-identity `Clear::clear_depth_stencil_fragment`
  family has exact empty-output Metal/MoltenVK evidence for its structural void-return contract.
  The complete four-identity `Clear::clear_vertex{,_mrt}` family has exact Metal/MoltenVK position
  observations for float2 vertex attributes expanded with buffer-supplied clip-space depth.
  The complete six-identity `FullscreenFragment{DepthStencil,Overlay,Texture}` suite has exact
  Metal/MoltenVK evidence for static-sampler half-texture reads, including red-lane replication,
  uniform-color multiplication, and RGB widening to RGBA32Float outputs.
  The complete six-identity `ARMesh::{mesh_depth_fragment,ar_mesh_shadow_fragment,ar_mesh_fragment}`
  suite has exact Metal/MoltenVK evidence for literal depth, edge-weighted shadow colors, and the
  full 2D/cube sampled-texture lighting path.
  A complete indexed tracking-area family now covers layered textures, fixed texture arrays, and
  function-constant-selected alternatives through eight-SIMD-group shared-memory reductions with
  barrier-synchronized exact uint output. Complete indexed vertex families now also cover
  vertex-ID-generated fullscreen positions across plain, viewport-indexed,
  render-target-layer-indexed, and single-view amplification interfaces.
- A sharded validation workflow with dependency-exact observations, an incremental SQLite source
  index, focused hash/shard selection, explicit full reclassification, capability audits, native
  Metal and Vulkan/MoltenVK A/B execution, and optional OpenRouter-authored case proposals.

### Changed

- Reducible control flow that the structured planner rejects is now nested into real SPIR-V loop
  and selection constructs before the bounded state-machine relooper is considered. Ordinary values
  stay in registers and demoted phis are promoted back at the block where their edges meet, so a
  function no longer arrives as one dispatch loop whose every crossing value is a function-scope
  variable. The nesting is adopted only when the emitted function satisfies the same construct,
  structured-exit, dominance, and phi contract the owned module is held to; anything else stays on
  the state machine.
- Vulkan 1.2 is now the mandatory translation and validation baseline; newer Vulkan features may
  only be exposed as optional performance paths with faithful Vulkan 1.2 fallbacks.
- Structured CFG plans now finalize loop-continue selection ownership before their completeness,
  ordering, and ownership checks; typed emission no longer runs detached continue, selection-arm,
  bypass, or reused-merge normalization after plan admission.
- Nested loop planning now materializes inner multi-exit dispatches before recomputing enclosing
  loop ownership, so newly synthesized exits are validly owned without a construct-tree retry.
- A nested natural loop that exits into its enclosing selection's sibling is now owned by a typed
  regional dispatcher before emission. Bounded scalar-only CFG rejects can use the same ownership
  contract across the whole function, while pointer state still declines rather than being guessed;
  terminal switches retain real dominated reconvergences and conflicting phi ownership rejects
  before instruction emission.
- Loop-local switches that target an enclosing loop role now lower to branch ladders before plan
  admission, preventing source-dominance false positives from emitting invalid case constructs.
- Imageblock scratch layout inference now follows reachable internal calls, preserving complete
  cells when a helper byte-addresses a nonzero field instead of relying on an inlining retry.
- Integer-width phi legalization now happens while pointer-phi transformations construct their
  replacement index phis; retained modules are no longer rescanned or repaired after emission or
  scalar-i64 lowering.
- Vulkan validation uploads every sampled-only literal through a staging buffer into an
  optimal-tiled image, so cube and other sampled shapes do not depend on optional linear-image
  support for their exact create flags.
- Authored output qualification now requires the shared checker to map the selection to a reflected
  shader write, then accepts deterministic byte-identical transformations instead of requiring
  every byte to differ from its initial value.
- The Metal validation executor now derives render-pipeline input topology from each authored draw
  and constructs matching array attachments and active-layer render passes for layered rendering.
- Generated Vulkan fragment companions preserve their authored layer-zero contract through the
  core first-layer rule, without requiring the optional Vulkan 1.2 `shaderOutputLayer` feature.
- Authored Vulkan execution now applies function constants before AIR translation, so nondefault
  values faithfully retain selected resources and CFG arms instead of trying to restore structure
  after default-valued SPIR-V emission.
- Exact Metal `dispatchThreads` kernels no longer round up and cull surplus lanes. The default
  contract now preserves partial-workgroup barriers by dispatching true boundary workgroup sizes;
  consumers must follow the reflected region plan and 48-byte dispatch payload.

- Native emitter wrapper APIs under `tools` now use `emit_vulkan_spirv*` names that match their
  implementation.
- The CLI accepts `--raster-samples` for AIR sample-count queries and derives a default `.vk.spv`
  output path when the output argument is omitted.
- Floating-point lowering is closer to AIR for the covered cases, including f32-to-f16 clamping,
  bf16 narrowing and NaN handling, fast `sin`/`cos`, `pow` zero edges, and exact `mix` endpoints.
- Buffer, pointer, control-flow, and access-chain lowering handles more structural cases, reducing
  fallbacks and invalid SPIR-V for shaders that use dynamic indices, pointer selects, aggregate
  copies, raw subword loads/stores, and local pointer tables.
- Opaque Metal buffers used through multiple scalar element types now retain each typed view as a
  descriptor alias at the same binding, while genuine array element-zero accesses gain their block
  descent and vector-stride pointers retain their vector pointee during interface construction;
  these preserve exact byte strides without late load/store pointer repairs.
- Exact byte-address provenance now composes through resource-wrapper collapse to the final buffer
  descriptor, so raw 32-bit vector loads are constructed from their word lanes before validation.
- Exact raw-word replay now derives synthesized access-chain pointer types from the concrete root's
  storage class, eliminating detached final access-chain storage-class repairs.
- Finalized typed CFG construction now recognizes a phi as an SSA identity only when its block has
  one actual predecessor and every incoming pair names that predecessor with the same structural
  value. All identities, including pointers, are substituted before SPIR-V IDs and representation
  sidecars are constructed. This eliminates the detached phi collapse and its access-chain and
  sampled-image type repairs.
- Buffer-interface discovery now follows transparent pointer aliases, includes pointer-arithmetic
  chains as typed element views, and gives mixed numeric scalar/vector views descriptor aliases at
  the same binding. Each logical pointer is therefore valid by construction instead of being
  retyped away from its byte loads during interface specialization.
- Null-derived access chains are neutralized once in the main access transform; redundant
  whole-module cleanup passes after native pointer rewrites have been removed.
- Exact raw-byte access is replayed at the descriptor-reconstruction boundary, eliminating the
  detached post-native replay and releasing source-only layout type graphs before final cleanup.
- Nondominating-value demotion now explicitly preserves the CFG successor contract, eliminating
  redundant whole-CFG repair passes after register spills.
- Pointer-phi legalization and mixed-storage value lowering now explicitly preserve CFG successors,
  so late structured-CFG repair runs only for an actual edge-producing loop split.
- Multi-entry loop splitting now owns its cloned region, header phis, redirected entry, and selection
  exit as one transaction, eliminating its detached whole-CFG repair pass.
- Multi-exit funnels, deep shared-arm refunnelling, and multi-entry loop splitting now preserve SSA
  dominance at their typed construction boundaries, eliminating the final emitted-module
  nondominating-value demotion pass.
- Construct-tree planning now assigns unreachable merge declarations for fully terminal selections
  alongside every other header, eliminating the late missing-terminal-header completion sweep.
- Dominated-region cloning now carries nested selection ownership through its structural rename map,
  so enclosing-route selections are materialized locally instead of dropped and rediscovered by a
  final missing-selection sweep.
- Construct-tree selection construction now preserves the local live-arm ownership of direct
  terminal guards through generic merge normalization, eliminating its repeated enclosing-escape
  and terminal-live repair sweeps.
- Ordinary selection construction now derives nested exits from the complete immutable source
  ownership map and materializes their private enclosing boundaries innermost-first, eliminating
  the late enclosing-region escape repair fixed point.
- Construct-tree selection construction now retains complete merge ownership through enclosing
  synthesis, eliminating its three post-construction nondominance, pass-through promotion, and
  bypass-refunneling repair stages from the product path.
- Construct-tree source ownership now includes nested terminal selections in its initial
  innermost-first header census, eliminating late post-synthesis terminal-convergence completion.
- Direct terminal ownership now consumes the complete proved linear tail before selection analysis,
  eliminating post-construction direct-guard refunneling from the product path.
- Selection construction now privatizes shared terminal returns when each merge owner is created,
  eliminating the later whole-owner terminal-return fixed-point sweeps.
- Innermost-first terminal-tail ownership now closes enclosing parents directly, removing the late
  parent/nested merge-composition pass from production.
- Switch merge construction now collapses proved terminal case tails when each switch owner is
  recorded, removing the late all-switch terminal finalization sweep from production.
- Loop-free terminal guards whose live convergence enters a later loop now defer a return shared
  with that loop's exit to the terminal owner, which privatizes both boundaries instead of admitting
  overlapping merge ownership.
- Loop-exit switch lowering now includes loop headers and multi-exit loops, producing a conditional
  ladder before planning so one block never needs both loop and selection merge ownership.
- Switch construction now privatizes intermediate continuations shared by a subset of cases even
  when the eventual switch merge is dominated, including loop-local suffixes whose clone does not
  cross a loop header, latch, exit, or nested-loop boundary.
- Construct-tree ownership now runs by default after an ordinary source-CFG planning rejection,
  while forward SSA allocation is scoped only to functions using that reordered plan. This removes
  a redundant emit, finish, validation, and source reparse from structurally owned large functions.
- Switch-tail ownership now preserves SPIR-V's legal fallthrough into the immediately following case,
  and shared-region cloning declines edges whose redirected predecessor crosses a natural-loop
  boundary, preventing non-adjacent case entries and non-dominating loop-exit values by construction.
- Enclosing selection owners now finish only their indexed dependent routes when the owner is
  recorded, removing the late whole-CFG enclosing-route materialization fixed point.
- Indexed translations retain natural emitted-loop ownership through the typed edge-producing
  transforms, allowing the detached stale-loop reclassification adapter to leave production.
- Translation audits now finish the 16-worker ordinary lane before starting the bounded costly-row
  lane, including sub-megabyte CFGs whose serialized AIR reaches 256 KiB. The costly lane is capped
  at two workers, with ≤384-KiB, 370-block/400-call CFGs isolated on one sublane while unrelated
  large and device-address/function-table rows continue on the other, so sustained concurrency does
  not consume the per-attempt 20-second budget. Cached outcomes now fingerprint the audit harness
  as well as the product translator.
- Translation audits can now resume a per-fingerprint retry-tier census in the disposable SQLite
  index, measure exact hash-file selections, and report the complete adoption histogram without
  reopening warm source shards.
- Resumable translation selection now materializes the historical-failure priority set once per
  uncursored batch instead of probing the complete audit history once for every indexed source.
- Function-constant-wrapped visible and intersection tables now retain their shared authored-linkage
  roles, and validation traces visible-table handles through internal helper parameters. A full
  `--reclassify-all` authoring census now reads every indexed source under the current analyzer ABI;
  the ordinary warm pass remains a zero-source-read incremental check. Translation workers classify
  these structurally proven authored-linkage dependencies before spawning the product/tool cascade.
- Direct AIR visible-function references now resolve automatically through an incremental
  same-library symbol index when the retained definition is unique, including transitive linked
  references; exact module byte locations keep warm and targeted lookups shard-local, while missing
  or ambiguous definitions remain explicit authored inputs. Translation audits can resumably replay
  historical linkage rows with `--retry-linkage`.
- Authored visible and intersection table slots now accept exact functions from separately
  harvested Metal libraries, matching the linked-functions API instead of imposing a false
  same-metallib provenance rule; exact module hashes, symbol definitions, and globally unique linked
  names remain mandatory.
- Partial zero-initialization of typed aggregates now lowers recursively into null stores for fully
  covered prefix subobjects, and final SPIR-V construction dependency-orders late synthesized
  module-scope types before existing users. Linked declaration/body pairs also select the bodied
  definition consistently during emitted-helper inlining, removing the associated empty-callee
  panic.
- Same-width scalar-array/vector reinterpret loads now rebuild the vector lane-by-lane with scalar
  bitcasts, preserving logical pointer types instead of requiring the all-buffer raw retry.
- Function-constant buffer alternatives that share one binding and mix scalar families through a
  recurrent pointer carrier now choose byte-addressed storage plus typed construct-tree planning
  during primary construction, instead of relying on the all-buffer raw and relooper retry cascade.
- Function-constant pruning now owns branch folding, reachability, phi, and loop-merge closure, which
  removes the final finish-time whole-module structured-CFG repair adapter.
- Primary construction now checks owned selection entry/exit, loop back-edge declarations,
  conditional merge declarations, and dominator serialization after function-constant CFG pruning,
  choosing bounded relooper form for only the affected functions before the first assembly while
  preserving unrelated functions and the hard downstream-driver state-machine cap. Relooper switch
  lowering now preserves both 32-bit and 64-bit selector literals.
- Function-constant pruning and entry-interface rebuilding now also own specialized Workgroup
  aggregate-stride access lowering instead of invoking a detached pointer repair afterward.
- The inline, SROA, and raw-access retry lowers dynamic typed accesses before constructing its
  relooper module, eliminating its post-relooper access-chain repair adapter.
- Helper inlining now completes address-preserving zero-offset aggregate descent once at its
  self-contained entry-closure boundary, eliminating the emitter's detached whole-module repair scan.
- Resource-select lowering assigns each duplicated `OpSampledImage` its branch image's exact type at
  construction, eliminating its final whole-module sampled-image type repair scan.
- Pointer-phi lowering now assigns synthesized incoming values directly to their predecessor edges,
  eliminating the final post-emission access-chain relocation and phi-order repair scan.
- Finalized typed CFG construction now redirects external entries around loop continue constructs,
  moves their exact phi values onto loop-header edges, and restores dominator serialization before
  emission, eliminating the corresponding numeric SPIR-V repair fixpoint.
- Emitted loop, selection, and switch merges that are reachable from outside their owning construct
  now get phi-aware private boundaries in the finalized typed CFG, eliminating the matching
  post-emission dominance repair.
- Instruction-local control flow is now materialized as real blocks inside the native emitter; when
  it splits a loop header, a dedicated source header retains the loop phis and ownership while any
  nested selection receives a private merge, eliminating the post-emission stale-loop downgrade.
- Finalized typed CFG construction now owns every loop header required by indexed inputs, eliminating
  both post-emission unmarked-loop synthesis and product-wide dominator-order normalization;
  structural loop tests define membership by dominated CFG predecessor edges rather than
  serialization order. Instruction-local control flow now carries each source block's real emitted
  exit into successor phis, eliminating final numeric-SPIR-V phi reconciliation and the product CFG
  repair module altogether.
- Selection boundaries that collide with a loop continue are now resolved in the finalized typed
  CFG, including enclosing selections, direct break/continue branches, and phi-carrying in-loop
  reconvergence; inner construct merges are likewise privatized in the typed plan, and emitted
  helper inlining now preserves the enclosing continue while giving its nested selection a private
  pass-through, eliminating both matching post-emission repairs.
- Structured emission now retains merge markers immediately before their terminators throughout
  lowering, eliminating the permissive pass that reordered already-malformed merge blocks.
- Direct-arm cloning and external loop-entry rewrites now preserve their newly established
  dominator order at the owning structural boundary.
- Loop plans now resolve emission-empty continue pass-through chains from the finalized typed CFG
  before emitting `OpLoopMerge`, eliminating the corresponding post-emission label repair.
- Finalized emission plans now give nested constructs phi-aware private merge targets while
  preserving loop-header backedges, eliminating the retained-SPIR-V shared-merge ownership pass.
- Finalized typed CFG emission now funnels selection/switch bypass edges through their declared
  phi-aware pass-through merges, eliminating that retained-SPIR-V product repair.
- Finalized typed CFG emission now clones shared direct arms only for headers with declared merges,
  then routes each clone through the merge with exact phi synthesis, eliminating the numeric-label
  shared-arm product repair.
- AIR `target datalayout` vector alignment and exact `air.struct_type_info` member offsets now flow
  through the primary emitter, aggregate byte walkers, reflection, and every retry tier instead of
  being reconstructed from generic SPIR-V layout assumptions.
- The default descriptor ABI now uses checked, non-overlapping bands for buffers, sampled textures,
  samplers, color inputs, imageblocks, storage textures, and translator-owned resources. Final
  modules reject descriptors outside their selected class range or set.
- Structured control-flow retries are bounded more tightly and reuse exact ordinary-planner
  rejection facts across the primary and construct-tree attempts, improving behavior on large
  shaders without reusing an accepting plan across retry semantics.
- Corpus translation and classification use bounded parallel workers, a 20-second per-item
  watchdog, a 500 MiB per-item memory ceiling, source-size and structural-cost-aware scheduling, and
  incremental index/cache reuse. Watchdog cleanup polls its own child instead of installing a
  process-global signal handler, and worker panics remain hash-attributed retryable failures instead
  of aborting a resumable census. Warm audits avoid reopening unchanged source shards; forced audits
  remain explicit.
- Large-module translation now streams function bodies into typed blocks, borrows unchanged
  normalization input, shares immutable block carriers, interns repeated instruction opcodes, and
  keeps mutually exclusive instruction and phi facts in one canonical representation. Resolved
  instructions derive def/use edges from those operands instead of allocating parallel name lists, and
  the emitter no longer clones every typed operand into a result-keyed table. Async-copy lowering
  streams into one bounded-growth buffer, skips dead declarations, and lets owned callers release
  superseded source text before typed parsing. Together these changes substantially reduce the
  measured largest local rows without changing their translated classification or skipping SPIR-V
  validation.
- Function-variable hoisting and integer-width normalization now preserve each block's instruction
  allocation, moving only matched variables and snapshotting only the instruction being rewritten.
  This removes full-block clones from universal large-module paths while preserving emitted order.
- Integer conversions now select their opcode from the actual emitted SPIR-V storage types. Legalized
  widths such as LLVM `i24` are masked at their producer, signed extensions restore the logical sign
  bit, and equal-width signed vertex inputs use bitcasts, eliminating the late width-convert repair.
- Validation capability checks, authored schema validation, backend execution gates, and cache
  identities now share one typed contract so a clean audit cannot hide a later executor rejection.
- Corpus capability audits now use the product's canonical AIR-call inventory and report every
  called intrinsic outside a recognized lowering or static-linkage family, including exact symbols
  and counts, instead of interpreting an unrelated clean authoring contract as complete support.
- Full AIR-intrinsic reclassification now updates every cached source in bounded keyset batches,
  independent of `--limit`, without reopening source shards; matrix capability recognition and
  lowering share one exact ABI parser.
- Translation census results now flow through a worker-count-bounded queue. A checkpoint failure
  cancels unclaimed rows and drains only in-flight results instead of allowing workers to retain an
  unbounded completed-row backlog while the scoped parent unwinds.
- Reflection documentation now describes the complete schema v30 descriptor, argument-buffer,
  runtime specialization, stage-interface, and conservative buffer-staging contracts.

### Removed

- Retired the monolithic validation ledgers and superseded mint/remint/run/why utilities in favor of
  sharded authored cases, dependency-exact observations, the source index, and unified corpus
  commands.
- Removed obsolete emitter naming and compatibility terminology; the product and tooling describe
  only the native AIR-to-SPIR-V pipeline.
- Removed the universal post-validation StorageBuffer/Workgroup pointer-phi rewrite, successful-
  module constant-branch reparse, and scalar-i64 module wrapper; same-root pointer merges and literal
  dead arms are now resolved on the typed source graph before emission.

### Fixed

- A buffer parameter whose body reads several distinct types through a one-index access chain is no
  longer wrapped as a flat `{ RuntimeArray<T> }` of whichever type happened to be seen first. A
  genuine `device T*` array has exactly one element view; several views mean the index selects
  members of a record, so the buffer's AIR struct layout is reconstructed instead. Wrapping such a
  buffer flatly kept the first view's stride and left every other access loading the wrong type.
- `air.get_num_samples_texture*` now lowers from the image type instead of an unconditional literal
  `1`: a multisampled 2D image queries `OpImageQuerySamples`, a single-sample image keeps the exact
  constant, and a multisampled non-2D dimensionality is refused. `ImageQuery` capability inference
  covers `OpImageQuerySamples`.
- Opaque buffer sources copied into local aggregates are now inferred as byte-addressed even when
  the destination is not another buffer parameter. AIR aggregate metadata whose explicit member
  offsets overlap Vulkan's naturally aligned block extents also selects the faithful byte view.
- Large CFGs now attempt their structurally valid primary emission before any retry. Missing switch
  merges and phi/predecessor mismatches route by their typed CFG failure class to the whole-CFG
  constructor, replacing the former block-count-triggered raw/relooper and construct-tree pre-route.
- Runtime device pointers now select the Vulkan 1.2 buffer-device-address model from typed AIR
  producer/use structure during primary emission. Opaque resource handles remain in the logical
  resource domain, and the redundant validation-triggered plain-BDA retries have been removed.
- Packed 32-bit reads through Private vector-backed helper views are now completed inside the
  memory-lowering transaction, so finalization no longer needs a repeated post-pass legalization.
- Interface binding now materializes direct loads selected between descriptor-backed and Private
  placeholder arms in the value domain, eliminating the corresponding module-wide finalizer repair.
- Descriptor-backed pointer-select closures, including dynamically indexed local pointer tables,
  are now constructed in the Logical value domain on the primary path instead of relying on
  validation-triggered value-select or physical-address repairs.
- Same-root pointer phis and selects now merge their integer access-chain indices and rematerialize
  one pointer for every storage class. Typed forward-select facts cover loop backedges whose
  advancing arm is a later chain of GEPs, so these recurrences need neither pointer SSA nor
  `VariablePointersStorageBuffer`.
- Scalar and vector function-constant initializer expressions are folded on the owned typed CFG
  before merge planning. Literal edges are pruned independently, surviving phi predecessor sets are
  rebuilt, and single-predecessor phis are substituted without letting an unrelated opaque aggregate
  phi suppress safe pruning. AGX execution-mask lowering now keeps loop ownership on the original
  backedge target and gives the lowered exit test a private structured merge.
- Function-constant-gated texture inputs with real Metal locations now retain their own descriptor
  and exact image shape even when their predicate defaults false, so later SPIR-V specialization
  cannot sample or write an unrelated texture. Only the `-1` location sentinel remains absent.
  Arrayed texture writes take their operand shape from the stable AIR intrinsic symbol even after
  the image handle passes through a local carrier.
- Scalar StorageBuffer remodeling now replays exact source GEP byte offsets as one stride-checked
  runtime-array index, preserving aggregate row/lane addresses after metadata collapses the
  descriptor element type.
- Unsupported image-texel value shapes now bypass buffer/CFG retry tiers that preserve the rejected
  SSA value, returning an honest fallback within the translation budget instead of repeatedly
  emitting the same large module.
- Added structural lowering for one-dimensional linear pixel sampling, logical pointer aliases,
  little-endian byte-aggregate/integer reinterpretation, and the packed signed-i8 AGX3 16x16x16
  matrix-MAC ABI. The matrix adapter validates its complete fixed descriptor contract and reuses
  the common distributed 32-lane matrix implementation.
- Unsupported Metal visible function references now fail fast with an explicit fallback instead of
  being treated as ordinary functions.
- Multiple `OpReturnValue` sites are rewritten consistently to stage outputs, and undefined
  fragment output stores are skipped rather than materialized.
- Kernel local-size options are validated as nonzero and are propagated into AIR local-size queries
  and imageblock lowering.
- SPIR-V generation avoids several invalid Logical-addressing forms by normalizing pointer phis,
  pointer selects, reinterpret loads/stores, access-chain index widths, and cross-binding pointer
  merges before final validation.
- Avoided a large-shader performance and size regression by measuring structured-CFG synthetic
  growth as added blocks rather than total module blocks, keeping large already-structured shaders
  on the primary emit path instead of falling back to the relooper tier.
- Removed inline denorm flush-to-zero emulation and avoided emitting `DenormFlushToZero`
  capability/execution modes on Vulkan devices that do not advertise float-controls support,
  restoring expected output size and compile time for affected shaders.
- Corrected imageblock explicit-slice stores so out-of-bounds oversized writes are discarded while
  zero-area writes still materialize transparent zero.
- Corrected integer `air.calculate_unclamped_lod_texture_2d` lowering so `OpImageQueryLod` uses a
  translator-owned default nearest sampler when the source AIR does not provide one.
- Fixed selected static sampler operands for AIR depth sampling and compare-depth lowering so
  `OpSampledImage` receives a valid SPIR-V sampler value even when AIR selects between sampler
  state globals.
- Fixed private indexed-container GEP lowering so dynamic accesses through palette-array pointer
  phis use ordinary access chains instead of invalid `OpPtrAccessChain` pointer-stride forms.
- Fixed function-constant-gated fragment outputs so the default-zero AIR predicate model does not
  advertise mutually exclusive render-target formats at the same Vulkan location.
- Fixed static-initializer evaluation so a specialized integer vector is retained as a typed vector
  and only a genuinely scalar value can enter the scalar global-fold map.
- Fixed fragment `[[sample_id]]` lowering to use `BuiltIn SampleId` with the required
  `SampleRateShading` capability, and lowered AIR `texture2d_ms` reads to MS `OpTypeImage` fetches
  with a `Sample` operand instead of treating the sample id as a mip LOD.
- Corrected native lowering for tessellation patch inputs; ray-intersection result typing and
  setters; custom, direct, narrow, and integer imageblocks; embedded texture arrays; array gather
  operands; array depth comparison sampling; and integer storage-image atomics.
- Corrected Metal SIMD/quad operations for vector `u16` prefix scans, integer extrema, active
  masks, and exact votes, and preserved signedness for same-width integer conversions and atomic
  subtraction.
- Repaired additional raw byte/word buffer, opaque-pointer, aggregate-pointee, record-layout,
  cross-storage select, and late pointer-typing cases without name-keyed workload exceptions.
- Corrected AIR struct member offsets, aggregate byte strides, and physical-pointer retry layouts
  for three-lane vectors and custom vector alignments, including preservation and conflict checking
  of explicit SPIR-V `ArrayStride` evidence.
- Runtime sampler specialization is now preserved through bounded SSA aliases and rejected when
  pointer selection or mixed joins make the exact state ambiguous; integer-image LOD queries and
  gather paths can no longer bypass the specialized state by substituting a default sampler.
- Runtime storage-image specialization now covers top-level and embedded writable textures across
  reads, writes, imageblock writes, emulated fetches, and atomics, with matching component and host
  capability checks in executable and metadata-only reflection paths.
- Vulkan validation now binds every compatible reflected sampled/storage texture alias to the same
  allocation and validates texture-array and argument-buffer alternatives as complete sets.
- Exact Metal `dispatchThreads` grids share one typed region payload across every logical grid
  builtin, including synthesized stage-input indexing and reflected buffer footprints.
- Module, function, block, instruction, and analysis transformations now express ownership by
  consuming values, returning replacements, mutating existing allocations transactionally, or
  appending directly. Fallible late SPIR-V repairs return their module explicitly, so no path can
  leave caller-visible state replaced by an empty/default ownership placeholder. ID-deleting native
  rewrite adapters now remove debug and annotation records only for the exact result IDs deleted by
  their own transaction, eliminating detached product cleanup sweeps without masking unrelated
  invalid metadata.
- Removed the late module-wide integer arithmetic width repair; packed scalar-slot addressing now
  scales dynamic indices in their declared integer type, so 64-bit access-chain indices remain
  valid and value-preserving without after-the-fact truncation.
- Deep shared-arm refunneling now gives every synthesized value phi an explicit incoming on every
  join predecessor, using an unobservable `undef` only on edges whose route branches to the other
  target.
  Value-domain pointer lowering likewise proves post-merge indices and its complete planned value-phi
  graph satisfy every predecessor dominance edge before commit. These construction contracts remove
  the duplicate post-primary non-dominating-value demotion scan.
- Cross-binding pointer-phi lowering now runs before the whole-function relooper size gate. The
  bounded in-memory address rewrite does not depend on CFG re-emission and remains available to large
  functions whose refunnelled value flow produces a complete cross-binding phi.
- Reduced worst-case translation time and memory growth by caching retry verdicts, pruning dead CFG
  before source re-emission, using linear CFG ordering and candidate scans, bounding generated CFG
  growth, and applying resource limits from worker startup.
- Validation now uploads sampled 3D images through bounded transfer buffers instead of relying on
  non-portable linear-tiling 3D images, restoring exact issue-4 and issue-5 Metal/MoltenVK checks.
- Opaque private-tensor descriptor intrinsics now have an explicit exact static-linkage contract,
  keeping their Apple-defined layout paired with the externally defined tensor-operation helper
  that consumes it instead of inventing a partial native representation.

## v0.1.0

### Added

- First public release of the `metal2vulkan` crate and CLI: native Metal AIR / sanitized LLVM IR →
  Vulkan SPIR-V.
- Library entry points for stage-aware translate, optional reflection metadata, and
  function-constant specialization.
- CLI: `metal2vulkan <in.air|.ll> <out.spv> --stage …`, optional `--emit-meta` JSON with the
  `serde` feature, `PASS` / `FALLBACK` reporting, and FALLBACK repro bundles under
  `$TMPDIR/metal2vulkan-repros` (override with `METAL2VULKAN_REPRO_DIR`).
- Optional `serde` feature for serializable `ShaderReflection` / metadata dumps.
- Consumer docs: architecture overview and reflection binding layout (`docs/`).

### Notes

- The crate is **alpha** (`0.x`): public API, CLI flags, and SPIR-V output may still change.
- This package does not ship third-party captured shaders; coverage is synthetic fixtures and
  unit tests only.
