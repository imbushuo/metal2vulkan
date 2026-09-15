//! Every capability a module declares must be one the module needs.
//!
//! `OpCapability` is not a note about the module. Vulkan reads it as a demand: the consumer must
//! enable the device feature behind it before the `VkShaderModule` is legal, and a device without
//! that feature cannot run the shader at all. So a capability declared and not used is not spare
//! documentation, it is a requirement the shader does not meet its own end of — and when the
//! capability is one the target platform does not have, it is the difference between a module that
//! loads and one that does not.
//!
//! Four of these were live at once, and each had the same shape: a rule keyed on something coarser
//! than the thing that actually needs the capability.
//!
//! * `Geometry` for `PrimitiveId`, which `Tessellation` also enables — 236 corpus modules, 226 of
//!   them already declaring `Tessellation`, and Metal has no geometry stage.
//! * `GroupNonUniformArithmetic` for an arithmetic group opcode, when a `ClusteredReduce` operand
//!   takes `GroupNonUniformClustered` instead — 244 modules.
//! * `VariablePointers` for a pointer merge in any storage class but `StorageBuffer`, counting
//!   `PhysicalStorageBuffer` addresses that neither variable-pointers capability governs — 30.
//! * `Sampled1D`/`SampledBuffer` for an image of that dimensionality, when only the storage arm
//!   read the `Sampled` operand that separates the two — 9.
//!
//! The check is the oracle that found them: strip one declared capability, reassemble, and ask
//! `spirv-val` whether the module still validates. If it does, nothing in the module needed it.
//!
//! **That oracle is closed, and running it again over the corpus is not worth anyone's time.**
//! Measured 2026-09-06 over all 14,579 sources — not over the 332 *distinct capability sets*, which
//! is not a sweep of the corpus at all, because whether a capability is needed is a fact about a
//! module's INSTRUCTIONS and two modules with identical sets can differ on every member of it. The
//! full run reports 7 modules: six declare `VariablePointersStorageBuffer` beside
//! `VariablePointers`, which implies it, so those are redundant rather than a demand; the seventh
//! declared `VariablePointers` with no pointer merge at all and is fixed in
//! `module_cleanup::drop_unrequired_capabilities`. Give any rerun a positive control — add
//! `OpCapability Int64` to a real module's disassembly and confirm the sweep calls it strippable.

use metal2vulkan::passes::Stage;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Capabilities `spirv-val` is measured NOT to enforce, so stripping one proves nothing.
///
/// `spirv-val` (SPIRV-Tools v2026.3) accepts a module with a `BuiltIn ClipDistance` variable and no
/// `ClipDistance` capability. That is a gap in the validator, not a spare declaration in the module:
/// Vulkan still requires `shaderClipDistance` for the builtin. Anything listed here is exempt from
/// the check below, and the list is evidence about `spirv-val` rather than about this translator.
const NOT_ENFORCED_BY_SPIRV_VAL: &[&str] = &[
    "ClipDistance",
    // For the group-arithmetic opcodes the capability is selected by the GROUP OPERATION operand:
    // `Reduce`/`InclusiveScan`/`ExclusiveScan` take `GroupNonUniformArithmetic`, `ClusteredReduce`
    // takes `GroupNonUniformClustered`. `spirv-val` checks only the opcode-level disjunction
    // ("requires one of these capabilities"), so in a module that uses both operations either one
    // can be stripped and it still validates. The operation-level rule is pinned by name in
    // `native::tests::intrinsics` instead.
    "GroupNonUniformArithmetic",
    "GroupNonUniformClustered",
    "GroupNonUniformPartitionedEXT",
    // `Dim 1D` requires one of `Sampled1D`/`Image1D` and `Dim Buffer` one of
    // `SampledBuffer`/`ImageBuffer`. Which one is right is decided by the type's `Sampled` operand
    // -- 2 means storage -- and the grammar does not encode that, so with both present either
    // strips clean. `every_declared_image_capability_has_an_image_that_needs_it` below checks the
    // half the grammar cannot.
    "Sampled1D",
    "Image1D",
    "SampledBuffer",
    "ImageBuffer",
];

/// The four image-dimension capabilities, as (capability, `Dim` spelling, is-storage).
const IMAGE_CAPABILITIES: &[(&str, &str, bool)] = &[
    ("Sampled1D", "1D", false),
    ("Image1D", "1D", true),
    ("SampledBuffer", "Buffer", false),
    ("ImageBuffer", "Buffer", true),
];

/// `Sampled1D`/`SampledBuffer` cover a SAMPLED image of that dimensionality and
/// `Image1D`/`ImageBuffer` a storage one. `spirv-val` cannot tell them apart, because the `Dim`
/// enumerant asks only for one of the pair. The module can: an `OpTypeImage` whose `Sampled` operand
/// is 2 is the storage one and anything else is the sampled one.
///
/// The storage arm of the rule always read that operand and the sampled arm never did, so a
/// `texture1d<..., access::write>` claimed both -- 9 corpus modules demanded a capability for an
/// image they do not contain.
#[test]
fn every_declared_image_capability_has_an_image_that_needs_it() {
    let mut checked = 0;
    for path in public_fixtures() {
        let source = std::fs::read_to_string(&path).expect("read fixture");
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(stage) = stage_of(&source) else {
            continue;
        };
        let Ok(spirv) = metal2vulkan::translate_sanitized_native(&source, stage, &scratch(&label))
        else {
            continue;
        };
        let text = metal2vulkan::disassemble(&spirv).expect("disassemble");
        let declared = text
            .lines()
            .filter_map(
                |line| match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                    ["OpCapability", name] => Some((*name).to_string()),
                    _ => None,
                },
            )
            .collect::<HashSet<_>>();
        for (capability, dim, storage) in IMAGE_CAPABILITIES {
            if !declared.contains(*capability) {
                continue;
            }
            // `%r = OpTypeImage %type <Dim> <Depth> <Arrayed> <MS> <Sampled> <Format>`
            let present = text.lines().any(|line| {
                match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                    [_, "=", "OpTypeImage", _, image_dim, _, _, _, sampled, ..] => {
                        image_dim == dim && (*sampled == "2") == *storage
                    }
                    _ => false,
                }
            });
            assert!(
                present,
                "{label} declares OpCapability {capability}, but no OpTypeImage in it is a                  {} {dim} image",
                if *storage { "storage" } else { "sampled" }
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 2,
        "only {checked} declared image capabilities were inspected, so this swept almost nothing"
    );
}

/// Capabilities SPIR-V implies from another declared capability, so stripping one changes no demand.
///
/// "Any capability that is a dependency of a declared capability is implicitly declared", and every
/// `GroupNonUniform*` capability depends on `GroupNonUniform`. Declaring it beside them is redundant
/// and costs a consumer nothing, so removing it would be churn rather than a fix.
const IMPLIED_BY_ANOTHER: &[&str] = &["GroupNonUniform"];

#[test]
fn no_public_fixture_declares_a_capability_it_does_not_need() {
    let mut checked = 0;
    let mut stripped = 0;
    for path in public_fixtures() {
        let source = std::fs::read_to_string(&path).expect("read fixture");
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(stage) = stage_of(&source) else {
            continue;
        };
        let scratch = scratch(&label);
        let Ok(spirv) = metal2vulkan::translate_sanitized_native(&source, stage, &scratch) else {
            continue;
        };
        let exempt = NOT_ENFORCED_BY_SPIRV_VAL
            .iter()
            .chain(IMPLIED_BY_ANOTHER)
            .copied()
            .collect::<HashSet<_>>();
        for (index, name) in declared_capabilities(&spirv) {
            // `Shader` is what makes it a Vulkan module at all.
            if name == "Shader" || exempt.contains(name.as_str()) {
                continue;
            }
            let without = without_instruction(&spirv, index);
            if metal2vulkan::tools::spirv_val_bytes(&without, &scratch).is_ok() {
                panic!(
                    "{label} declares OpCapability {name}, but the module still validates without \
                     it. A declared capability is a demand on every consumer -- it must name \
                     something the module actually does."
                );
            }
            stripped += 1;
        }
        checked += 1;
    }
    assert!(
        checked >= 20 && stripped >= 20,
        "only {checked} fixtures carrying {stripped} strippable capabilities were inspected, so \
         this swept almost nothing"
    );
}

/// The group-arithmetic opcodes, whose capability is chosen by their GROUP OPERATION operand.
const GROUP_ARITHMETIC_OPCODES: &[&str] = &[
    "OpGroupNonUniformIAdd",
    "OpGroupNonUniformFAdd",
    "OpGroupNonUniformSMin",
    "OpGroupNonUniformUMin",
    "OpGroupNonUniformFMin",
    "OpGroupNonUniformSMax",
    "OpGroupNonUniformUMax",
    "OpGroupNonUniformFMax",
    "OpGroupNonUniformBitwiseAnd",
    "OpGroupNonUniformBitwiseOr",
    "OpGroupNonUniformBitwiseXor",
];

/// `GroupNonUniformArithmetic` belongs to `Reduce`/`InclusiveScan`/`ExclusiveScan`, not to the
/// opcode. `ClusteredReduce` takes `GroupNonUniformClustered` instead, and every `air.simd_*`
/// whole-simdgroup reduction emits the clustered form -- so keying the capability on the opcode
/// alone demanded arithmetic subgroup support of 244 corpus modules that use none. `spirv-val`
/// checks only the opcode-level disjunction, so this is the half it cannot see.
#[test]
fn group_arithmetic_capability_follows_the_group_operation() {
    let mut checked = 0;
    for (label, text) in translated_fixtures() {
        let declared = capability_names(&text);
        let non_clustered = text.lines().any(|line| {
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let [_, "=", opcode, rest @ ..] = tokens.as_slice() else {
                return false;
            };
            GROUP_ARITHMETIC_OPCODES.contains(opcode) && !rest.contains(&"ClusteredReduce")
        });
        if declared.contains("GroupNonUniformArithmetic") {
            assert!(
                non_clustered,
                "{label} declares OpCapability GroupNonUniformArithmetic, but every group \
                 arithmetic instruction in it is a ClusteredReduce, which takes \
                 GroupNonUniformClustered instead"
            );
            checked += 1;
        } else {
            assert!(
                !non_clustered,
                "{label} performs a non-clustered group arithmetic operation without declaring \
                 GroupNonUniformArithmetic"
            );
        }
    }
    assert!(
        checked >= 1,
        "no fixture declared GroupNonUniformArithmetic, so this swept almost nothing"
    );
}

/// `VariablePointers` and `VariablePointersStorageBuffer` govern LOGICAL pointers. A
/// `PhysicalStorageBuffer` pointer is an address; selecting, phi-ing, or indexing one is what
/// `PhysicalStorageBufferAddresses` is for, and neither variable-pointers capability applies. The
/// rule that classified merges by storage class had one `else` arm for everything that was not
/// `StorageBuffer`, so 30 corpus modules that merge only addresses demanded the strictly stronger
/// `variablePointers` feature.
#[test]
fn variable_pointers_is_not_declared_for_addresses() {
    let mut checked = 0;
    for (label, text) in translated_fixtures() {
        let declared = capability_names(&text);
        if !declared.contains("VariablePointers") {
            checked += usize::from(declared.contains("PhysicalStorageBufferAddresses"));
            continue;
        }
        // `%r = Op... %resultType ...`; the merges that need the capability are the ones whose
        // result type is a pointer in a storage class other than StorageBuffer or
        // PhysicalStorageBuffer.
        let logical = pointer_types(&text);
        let needs = text.lines().any(|line| {
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let [_, "=", opcode, result_type, ..] = tokens.as_slice() else {
                return false;
            };
            matches!(
                *opcode,
                "OpPhi" | "OpSelect" | "OpPtrAccessChain" | "OpInBoundsPtrAccessChain"
            ) && logical.contains(*result_type)
        });
        assert!(
            needs,
            "{label} declares OpCapability VariablePointers, but no pointer merge in it produces a \
             pointer the capability governs"
        );
    }
    assert!(
        checked >= 1,
        "no fixture used PhysicalStorageBufferAddresses without VariablePointers, so this swept \
         almost nothing"
    );
}

/// The ids of pointer types the variable-pointers capabilities actually govern: everything but
/// `StorageBuffer` (covered by `VariablePointersStorageBuffer`) and `PhysicalStorageBuffer`
/// (covered by `PhysicalStorageBufferAddresses`).
fn pointer_types(text: &str) -> HashSet<String> {
    text.lines()
        .filter_map(
            |line| match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                [result, "=", "OpTypePointer", storage, ..]
                    if *storage != "StorageBuffer" && *storage != "PhysicalStorageBuffer" =>
                {
                    Some((*result).to_string())
                }
                _ => None,
            },
        )
        .collect()
}

fn capability_names(text: &str) -> HashSet<String> {
    text.lines()
        .filter_map(
            |line| match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["OpCapability", name] => Some((*name).to_string()),
                _ => None,
            },
        )
        .collect()
}

/// Every public fixture that translates, as (label, disassembly).
fn translated_fixtures() -> Vec<(String, String)> {
    public_fixtures()
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(&path).ok()?;
            let label = path.file_name()?.to_string_lossy().into_owned();
            let stage = stage_of(&source)?;
            let spirv =
                metal2vulkan::translate_sanitized_native(&source, stage, &scratch(&label)).ok()?;
            Some((label, metal2vulkan::disassemble(&spirv).ok()?))
        })
        .collect()
}

/// Every `OpCapability` in the module, as (word offset of the instruction, capability name).
///
/// Read straight out of the binary rather than the disassembly so the instruction can be removed
/// without an assembler in the loop.
fn declared_capabilities(spirv: &[u8]) -> Vec<(usize, String)> {
    let words = words_of(spirv);
    let text = metal2vulkan::disassemble(spirv).expect("disassemble");
    let names = text
        .lines()
        .filter_map(
            |line| match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["OpCapability", name] => Some((*name).to_string()),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    let mut found = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 {
            break;
        }
        if words[offset] & 0xFFFF == 17 {
            found.push(offset);
        }
        offset += count;
    }
    assert_eq!(
        found.len(),
        names.len(),
        "the binary and the disassembly disagree on how many capabilities there are"
    );
    found.into_iter().zip(names).collect()
}

/// The module with the instruction starting at `offset` removed.
fn without_instruction(spirv: &[u8], offset: usize) -> Vec<u8> {
    let words = words_of(spirv);
    let count = (words[offset] >> 16) as usize;
    words
        .iter()
        .enumerate()
        .filter(|(index, _)| *index < offset || *index >= offset + count)
        .flat_map(|(_, word)| word.to_le_bytes())
        .collect()
}

fn words_of(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Scratch for one subject. `spirv-val` writes into it and these tests run concurrently, so each
/// subject gets its own directory.
fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "m2v_declared_capabilities_{}_{}",
        std::process::id(),
        label.replace(['/', '.'], "_")
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

/// The stage the AIR declares. The library's `detect_stage` sanitizes from a file path; these
/// fixtures are already sanitized text, and they name their stage the same way.
fn stage_of(source: &str) -> Option<Stage> {
    if source.contains("!air.vertex =") {
        Some(Stage::Vertex)
    } else if source.contains("!air.fragment =") {
        Some(Stage::Fragment)
    } else if source.contains("!air.kernel =") {
        Some(Stage::Kernel)
    } else {
        None
    }
}

fn public_fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/fixtures/public");
    let mut paths = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}
