//! Nothing this translator emits may ask a consumer for a geometry stage.
//!
//! Declaring a SPIR-V capability is not a note about the module, it is a demand on the consumer:
//! Vulkan requires the corresponding device feature to be enabled before the `VkShaderModule` is
//! legal. `Capability::Geometry` demands `geometryShader`, and Metal has no geometry stage at all,
//! so no Metal-backed Vulkan implementation has one. A module that declares it cannot be loaded on
//! the platform this translator exists to target.
//!
//! Nothing in the source language can produce a geometry shader, so the demand can only ever arrive
//! by accident. It did: `PrimitiveId` is enabled by ANY of `Geometry`, `Tessellation`,
//! `RayTracingKHR` or `MeshShadingEXT`, and the translator picked `Geometry` from that disjunction
//! unconditionally. 236 of the 14,579 corpus sources carried it; 226 of them were
//! tessellation-evaluation entries that already declared `Tessellation`, so the requirement was
//! satisfied before the demand was added and the demand bought exactly nothing.

use metal2vulkan::passes::Stage;
use std::path::{Path, PathBuf};

#[test]
fn no_public_fixture_declares_the_geometry_capability() {
    let mut checked = 0;
    let mut primitive_id = 0;
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
        let asm = metal2vulkan::disassemble(&spirv).expect("disassemble");
        assert!(
            !asm.contains("OpCapability Geometry"),
            "{label} declares Capability Geometry, which demands `geometryShader` -- a feature no \
             Metal-backed Vulkan implementation has, for a stage the source language cannot express"
        );
        if asm.contains("BuiltIn PrimitiveId") {
            primitive_id += 1;
            // A disjunct still has to be declared, and this is the one Metal can grant.
            assert!(
                asm.contains("OpCapability Tessellation"),
                "{label} reads PrimitiveId but declares none of the capabilities that enable it"
            );
        }
        checked += 1;
    }
    // No public fixture reads `PrimitiveId` today, so the `primitive_id` arm above is a guard that
    // will start checking something the day one does. What that builtin's two arms actually do is
    // pinned by name in `native::tests::interface` (a fragment entry, which must ADD
    // `Tessellation`) and `native::tests::intrinsics` (a tessellation-evaluation entry, which must
    // add NOTHING because it already declares it). The sweep here is for the invariant: whatever
    // else changes, no module we emit may demand a geometry stage.
    assert!(
        checked >= 20,
        "only {checked} fixtures were inspected ({primitive_id} reading PrimitiveId), so this \
         swept almost nothing"
    );
}

/// Scratch for one subject. `spirv-val` writes into it and these tests run concurrently, so each
/// subject gets its own directory.
fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "m2v_geometry_capability_{}_{}",
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
