//! In-process shader tools: shared-library LLVM I/O and statically linked SPIRV-Tools.
//!
//! Translation never launches a tool executable. Callers needing hard time/memory isolation must
//! run the entire translation in their own worker process; native calls cannot be safely cancelled.

mod llvm;

use crate::native;
use spirv_tools::assembler::Assembler;
use spirv_tools::val::Validator;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Condvar, Mutex, OnceLock};

/// Bound concurrent memory-heavy validators independently of emission parallelism. Contexts are
/// per-call, not shared across threads. The operational cap does not alter the validation rules.
fn val_gate() -> &'static (Mutex<usize>, Condvar) {
    static GATE: OnceLock<(Mutex<usize>, Condvar)> = OnceLock::new();
    GATE.get_or_init(|| (Mutex::new(0usize), Condvar::new()))
}

fn val_par_limit() -> usize {
    crate::env_vars::val_par()
}

/// RAII permit for a concurrent validator slot — blocks until one of [`val_par_limit`] slots is free,
/// releasing it (and waking a waiter) on drop.
struct ValPermit;
impl ValPermit {
    fn acquire() -> ValPermit {
        let limit = val_par_limit();
        let (lock, cv) = val_gate();
        let mut n = lock.lock().unwrap();
        while *n >= limit {
            n = cv.wait(n).unwrap();
        }
        *n += 1;
        ValPermit
    }
}
impl Drop for ValPermit {
    fn drop(&mut self) {
        let (lock, cv) = val_gate();
        let mut n = lock.lock().unwrap();
        *n = n.saturating_sub(1);
        cv.notify_one();
    }
}

/// Vulkan SPIR-V triple the sanitizer rewrites AIR modules to.
/// Baseline Vulkan environment every emitted module must satisfy. Higher-version paths may be
/// offered separately, but must retain a faithful fallback to this contract.
pub const VULKAN_TARGET_ENV: &str = "vulkan1.2";
pub const VULKAN_TRIPLE: &str = "spirv-unknown-vulkan1.2";

/// `src` (.air bitcode or .ll) -> sanitized `.ll` text ready for the native emitter.
/// Rewrites the target triple to the Vulkan one, preserves the AIR datalayout for executable layout
/// lowering, and drops LLVM metadata/runtime root globals.
pub fn air_to_sanitized_ll(src: &str, tmp: &Path) -> Result<String, String> {
    Ok(air_to_sanitized_ll_with_datalayout(src, tmp)?.0)
}

/// Like [`air_to_sanitized_ll`] but also returns the AIR `target datalayout` string it preserves. The
/// sanitizer preserves that line as the executable source-layout contract and also returns its value
/// for reflection, avoiding a second source read. The sanitized text is byte-identical to
/// [`air_to_sanitized_ll`].
pub fn air_to_sanitized_ll_with_datalayout(
    src: &str,
    _tmp: &Path,
) -> Result<(String, Option<String>), String> {
    let ll_text = if src.ends_with(".ll") {
        std::fs::read_to_string(src).map_err(|e| format!("read {src}: {e}"))?
    } else {
        let bytes = std::fs::read(src).map_err(|e| format!("read {src}: {e}"))?;
        llvm_disassemble(&bytes)?
    };

    Ok(sanitize_ll_text_with_datalayout(&ll_text))
}

/// Disassemble AIR/LLVM bitcode in memory using LLVM's shared library, loaded on first use.
/// Both raw bitcode and the LLVM bitcode wrapper used by AIR are accepted.
pub fn llvm_disassemble(bitcode: &[u8]) -> Result<String, String> {
    llvm::disassemble(bitcode)
}

/// Product source/native-tool declarations and the actual loaded LLVM image.
/// A caller unable to obtain this identity must not reuse persisted AIR results.
pub fn translation_cache_identity() -> Result<(&'static str, &'static str), String> {
    Ok((
        env!("METAL2VULKAN_BUILD_FINGERPRINT"),
        llvm::cache_fingerprint()?,
    ))
}

/// Assemble and verify textual LLVM IR in memory, returning bitcode (the `llvm-as` operation).
/// This is not needed for translating textual AIR through the native emitter.
pub fn llvm_assemble(text: &str) -> Result<Vec<u8>, String> {
    llvm::assemble(text)
}

/// Assemble SPIR-V text for Vulkan 1.2 with the linked SPIRV-Tools assembler.
/// As with `spirv-as`, assembly is not validation; use [`spirv_val_bytes`] for the latter.
pub fn spirv_assemble(text: &str) -> Result<Vec<u8>, String> {
    let assembler = spirv_tools::assembler::compiled::CompiledAssembler::with_env(
        spirv_tools::TargetEnv::Vulkan_1_2,
    );
    let binary = assembler
        .assemble(text, Default::default())
        .map_err(|error| format!("spirv-as failed:\n{error}"))?;
    Ok(binary
        .as_words()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}

/// Sanitize LLVM IR text that is already in `.ll` form.
///
/// This is the text-only core used by [`air_to_sanitized_ll_with_datalayout`]: rewrite the target
/// triple to Vulkan, preserve the AIR datalayout, drop llvm-dis's input-path `ModuleID` comment, and
/// remove LLVM metadata/runtime root globals that are not part of the shader entry body.
pub fn sanitize_ll_text_with_datalayout(ll_text: &str) -> (String, Option<String>) {
    let mut out = String::with_capacity(ll_text.len());
    let mut datalayout = None;
    for line in ll_text.lines() {
        let t = line.trim_start();
        if t.starts_with("target triple") {
            out.push_str(&format!("target triple = \"{VULKAN_TRIPLE}\"\n"));
            continue;
        }
        if t.starts_with("target datalayout") {
            datalayout = datalayout_value(t);
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // llvm-dis derives this comment from its input filename. Validation disassembles through
        // a uniquely named scratch directory, so retaining it makes identical bitcode acquire a
        // different corpus identity on every harvest.
        if t.starts_with("; ModuleID =") {
            continue;
        }
        if t.starts_with("@llvm.global_ctors")
            || t.starts_with("@llvm.global_dtors")
            || t.starts_with("@llvm.used")
            || t.starts_with("@llvm.compiler.used")
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    (out, datalayout)
}

/// The quoted value of a `target datalayout = "..."` line.
fn datalayout_value(line: &str) -> Option<String> {
    let start = line.find('"')?;
    let rest = &line[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Emit Vulkan SPIR-V from sanitized `.ll`, returning the raw SPIR-V binary words.
///
/// This is an in-process native Rust emission boundary. Unsupported IR shapes return explicit
/// errors.
pub fn emit_vulkan_spirv(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv(san_ll)
}

pub(crate) fn emit_vulkan_spirv_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_with_sidecar(san_ll, kern, entry_name, buffer_layouts)
}

pub(crate) fn emit_vulkan_spirv_with_outcome(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, crate::emit_sidecar::EmissionFailure> {
    native::emit_vulkan_spirv_with_outcome(san_ll, kern, entry_name, buffer_layouts)
}

/// Like [`emit_vulkan_spirv`], but models every device/constant buffer param raw. Used by the R4
/// ground-truth retry when the default typed emission produces a mistyped buffer access.
pub fn emit_vulkan_spirv_all_buffers_raw(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
    known_ordinary_plan_rejections: &std::collections::HashSet<String>,
    known_ownership_plan_rejections: &std::collections::HashSet<String>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
        known_ordinary_plan_rejections,
        known_ownership_plan_rejections,
    )
}

/// Like [`emit_vulkan_spirv_all_buffers_raw`], but also models threadgroup (`addrspace(3)`) buffer
/// params raw. This broad form is retained for explicit diagnostics; production infers the exact
/// Workgroup raw scope structurally and never selects this form from a failed candidate.
pub fn emit_vulkan_spirv_all_buffers_raw_with_workgroup(
    san_ll: &str,
    _tmp: &Path,
) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw_with_workgroup(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
    known_ordinary_plan_rejections: &std::collections::HashSet<String>,
    known_ownership_plan_rejections: &std::collections::HashSet<String>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
        known_ordinary_plan_rejections,
        known_ownership_plan_rejections,
    )
}

/// Emit the all-device/constant-buffer raw view with repair skipped for the W2 relooper's exclusive
/// consumption. The caller must rebuild and validate its CFG before adopting any bytes.
pub fn emit_vulkan_spirv_all_buffers_raw_relooper_feed(
    san_ll: &str,
    _tmp: &Path,
) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw_relooper_feed(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
    )
}

/// Like [`emit_vulkan_spirv_all_buffers_raw`], but also enables BDA device-pointer modeling (a device
/// pointer loaded from a buffer becomes its real 64-bit PhysicalStorageBuffer64 address). The honest
/// retry tier for the `raw store for Ptr(1)` BDA frontier class — see
/// [`native::emit_vulkan_spirv_all_buffers_raw_bda`].
pub fn emit_vulkan_spirv_all_buffers_raw_bda(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw_bda(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
    known_ordinary_plan_rejections: &std::collections::HashSet<String>,
    known_ownership_plan_rejections: &std::collections::HashSet<String>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
        known_ordinary_plan_rejections,
        known_ownership_plan_rejections,
    )
}

/// Read and validate a SPIR-V file against Vulkan 1.2 using the linked validator.
pub fn spirv_val(spv_path: &str) -> Result<(), String> {
    let bytes = std::fs::read(spv_path).map_err(|error| format!("read {spv_path}: {error}"))?;
    spirv_val_bytes(&bytes, Path::new(""))
}

/// Validate in-memory SPIR-V against Vulkan 1.2, without scratch files or subprocesses.
/// `tmp` is retained for source compatibility and is not accessed.
pub fn spirv_val_bytes(spv: &[u8], _tmp: &Path) -> Result<(), String> {
    let _permit = ValPermit::acquire();
    if !spv.len().is_multiple_of(4) {
        return Err("spirv-val failed:\nSPIR-V byte length is not a multiple of four".into());
    }
    // A byte slice need not be aligned. Decode explicitly rather than casting it to u32.
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let validator =
        spirv_tools::val::compiled::CompiledValidator::with_env(spirv_tools::TargetEnv::Vulkan_1_2);
    validator
        .validate(&words, None)
        .map_err(|error| format!("spirv-val failed:\n{error}"))
}

/// Best-effort filename stem (drop the final extension).
pub fn strip_ext(p: &str) -> String {
    match p.rfind('.') {
        Some(i) => p[..i].to_string(),
        None => p.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `spirv_val_bytes` is called concurrently -- by parallel tests, by a sweep's worker threads,
    /// and by separate processes pointed at one scratch directory. Its result has to describe the
    /// bytes it was handed and nothing else. A shared scratch file made it describe whichever
    /// module won the race, or none, which surfaces as a validation failure in a module that is
    /// fine; that is a false alarm the reader has no way to tell from a real one.
    #[test]
    fn concurrent_validations_sharing_one_scratch_directory_do_not_collide() {
        let module = crate::translate_sanitized_native(
            r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#,
            crate::passes::Stage::Kernel,
            &std::env::temp_dir().join(format!("m2v_val_race_build_{}", std::process::id())),
        )
        .expect("the probe kernel translates");

        // One directory, deliberately shared by every thread: that is the condition under test.
        let shared = std::env::temp_dir().join(format!("m2v_val_race_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&shared);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let module = &module;
                let shared = &shared;
                scope.spawn(move || {
                    for _ in 0..16 {
                        spirv_val_bytes(module, shared)
                            .expect("a valid module validates, whoever else is validating");
                    }
                });
                assert!(
                    !shared.exists(),
                    "in-memory validation must not create scratch files"
                );
            }
        });
        let _ = std::fs::remove_dir_all(&shared);
    }

    #[test]
    fn sanitizer_captures_and_preserves_target_datalayout() {
        // The same datalayout survives both as a captured reflection string and in the sanitized
        // text handed to executable layout lowering.
        let tmp = std::env::temp_dir();
        let src = tmp.join("m2v_r4_datalayout_test.ll");
        std::fs::write(
            &src,
            "target datalayout = \"e-m:o-i64:64-i128:128-n32:64-S128\"\n\
             target triple = \"air64-apple-macosx\"\n\
             define void @k() {\n  ret void\n}\n",
        )
        .unwrap();
        let (san, datalayout) =
            air_to_sanitized_ll_with_datalayout(src.to_str().unwrap(), &tmp).unwrap();
        assert_eq!(
            datalayout.as_deref(),
            Some("e-m:o-i64:64-i128:128-n32:64-S128")
        );
        assert!(san.contains("target datalayout = \"e-m:o-i64:64-i128:128-n32:64-S128\""));
        assert!(san.contains(&format!("target triple = \"{VULKAN_TRIPLE}\"")));
        // The plain wrapper returns byte-identical sanitized text.
        assert_eq!(
            air_to_sanitized_ll(src.to_str().unwrap(), &tmp).unwrap(),
            san
        );
        std::fs::remove_file(&src).ok();
    }

    #[test]
    fn text_sanitizer_matches_file_sanitizer_rules() {
        let (san, datalayout) = sanitize_ll_text_with_datalayout(
            "; ModuleID = '/tmp/random/case.air'\n\
             target datalayout = \"e-p:64:64\"\n\
             target triple = \"air64-apple-ios\"\n\
             @llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] []\n\
             @llvm.compiler.used = appending global [0 x ptr] [], section \"llvm.metadata\"\n\
             define void @k() {\n  ret void\n}\n",
        );

        assert_eq!(datalayout.as_deref(), Some("e-p:64:64"));
        assert!(san.contains(&format!("target triple = \"{VULKAN_TRIPLE}\"")));
        assert!(san.contains("target datalayout = \"e-p:64:64\""));
        assert!(!san.contains("ModuleID"));
        assert!(!san.contains("@llvm.global_ctors"));
        assert!(!san.contains("@llvm.compiler.used"));
        assert!(san.contains("define void @k()"));
    }

    #[test]
    fn sanitizer_identity_ignores_llvm_dis_scratch_path() {
        let first = sanitize_ll_text_with_datalayout(
            "; ModuleID = '/tmp/first/case.air'\nsource_filename = \"stable\"\ndefine void @k() { ret void }\n",
        )
        .0;
        let second = sanitize_ll_text_with_datalayout(
            "; ModuleID = '/tmp/second/case.air'\nsource_filename = \"stable\"\ndefine void @k() { ret void }\n",
        )
        .0;
        assert_eq!(first, second);
    }

    #[test]
    fn datalayout_value_extracts_quoted_string() {
        assert_eq!(
            datalayout_value("target datalayout = \"e-p:32:32\""),
            Some("e-p:32:32".to_string())
        );
        assert_eq!(datalayout_value("target datalayout = malformed"), None);
    }
}
