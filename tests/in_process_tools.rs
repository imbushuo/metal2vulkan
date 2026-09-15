use metal2vulkan::{passes::Stage, tools};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const KERNEL: &str = r#"
source_filename = "owned-kernel"
target datalayout = "e-p:64:64-i64:64-n8:16:32:64"
target triple = "air64-apple-macosx"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

const SPIRV: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"#;

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "m2v-in-process-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn llvm_roundtrip_preserves_metadata_layout_and_translation() {
    let bitcode = tools::llvm_assemble(KERNEL).unwrap();
    let text = tools::llvm_disassemble(&bitcode).unwrap();
    assert!(text.contains("!air.kernel"));
    assert!(text.contains("target datalayout = \"e-p:64:64-i64:64-n8:16:32:64\""));
    let (sanitized, layout) = tools::sanitize_ll_text_with_datalayout(&text);
    assert!(layout.is_some());
    let original = tools::sanitize_ll_text_with_datalayout(KERNEL).0;
    let tmp = Scratch::new();
    assert_eq!(
        metal2vulkan::translate_sanitized_native(&original, Stage::Kernel, &tmp.0).unwrap(),
        metal2vulkan::translate_sanitized_native(&sanitized, Stage::Kernel, &tmp.0).unwrap()
    );

    // LLVM's writer can already wrap bitcode for an Apple target.
    let raw = if bitcode.starts_with(&0x0b17_c0de_u32.to_le_bytes()) {
        let offset = u32::from_le_bytes(bitcode[8..12].try_into().unwrap()) as usize;
        let length = u32::from_le_bytes(bitcode[12..16].try_into().unwrap()) as usize;
        &bitcode[offset..offset + length]
    } else {
        &bitcode
    };
    let mut wrapped = Vec::new();
    for word in [0x0b17_c0de_u32, 0, 20, raw.len() as u32, u32::MAX] {
        wrapped.extend_from_slice(&word.to_le_bytes());
    }
    wrapped.extend_from_slice(raw);
    assert_eq!(tools::llvm_disassemble(&wrapped).unwrap(), text);
    assert_eq!(tools::llvm_disassemble(raw).unwrap(), text);
}

#[test]
fn llvm_errors_return_without_terminating_the_host() {
    for invalid in [&b""[..], b"not bitcode", b"BC\xc0\xde", b"\xde\xc0\x17\x0b"] {
        assert!(tools::llvm_disassemble(invalid)
            .unwrap_err()
            .contains("LLVM bitcode parse failed"));
    }
    assert!(tools::llvm_assemble("not LLVM IR")
        .unwrap_err()
        .contains("LLVM IR parse failed"));
    assert!(tools::llvm_assemble("define i32 @k() { ret void }").is_err());
    let bitcode = tools::llvm_assemble(KERNEL).unwrap();
    for length in [8, bitcode.len() / 2] {
        assert!(tools::llvm_disassemble(&bitcode[..length]).is_err());
    }
    assert!(tools::llvm_disassemble(&bitcode).is_ok());
}

#[test]
fn llvm_contexts_are_independent_across_threads() {
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..8 {
                    let bitcode = tools::llvm_assemble(KERNEL).unwrap();
                    assert!(tools::llvm_disassemble(&bitcode).unwrap().contains("@k"));
                    assert!(tools::llvm_disassemble(b"invalid").is_err());
                }
            });
        }
    });
}

#[test]
fn spirv_validation_is_strict_and_accepts_unaligned_bytes() {
    let bytes = tools::spirv_assemble(SPIRV).unwrap();
    let scratch = Scratch::new();
    // A file cannot be used as a scratch directory. None is needed by the new boundary.
    let unavailable = scratch.0.join("not-a-directory");
    std::fs::write(&unavailable, b"untouched").unwrap();
    let mut unaligned = vec![0];
    unaligned.extend_from_slice(&bytes);
    tools::spirv_val_bytes(&unaligned[1..], &unavailable).unwrap();
    let mut swapped = Vec::new();
    for word in bytes.chunks_exact(4) {
        swapped.extend(word.iter().rev());
    }
    tools::spirv_val_bytes(&swapped, &unavailable).unwrap();
    for invalid in [&b""[..], &bytes[..bytes.len() - 1], &[0; 20]] {
        assert!(tools::spirv_val_bytes(invalid, &unavailable).is_err());
    }
    let no_local_size = SPIRV.replace("OpExecutionMode %main LocalSize 1 1 1", "");
    let invalid = tools::spirv_assemble(&no_local_size).unwrap();
    let error = tools::spirv_val_bytes(&invalid, &unavailable).unwrap_err();
    assert!(error.starts_with("spirv-val failed:"));
    assert!(error.contains("LocalSize"), "{error}");
    assert!(tools::spirv_assemble("OpNotAnOpcode")
        .unwrap_err()
        .contains("spirv-as failed:"));
    assert_eq!(std::fs::read(&unavailable).unwrap(), b"untouched");
}

fn cli(input: &Path, output: &Path, scratch: &Path, stage: &str) -> std::process::Command {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_metal2vulkan"));
    command.args([input.as_os_str(), output.as_os_str()]);
    command.args(["--stage", stage]);
    command.env("PATH", "");
    command.env("METAL2VULKAN_REPRO_DIR", scratch.join("repros"));
    for tool in ["LLVM_DIS", "LLVM_AS", "SPIRV_AS", "SPIRV_VAL"] {
        command.env(format!("METAL2VULKAN_{tool}"), scratch.join("must-not-run"));
    }
    command
}

#[test]
fn cli_translates_bitcode_and_text_without_external_tools() {
    let scratch = Scratch::new();
    for (extension, contents) in [
        ("ll", KERNEL.as_bytes().to_vec()),
        ("air", tools::llvm_assemble(KERNEL).unwrap()),
    ] {
        let input = scratch.0.join(format!("input.{extension}"));
        let output = scratch.0.join(format!("output.{extension}.spv"));
        std::fs::write(&input, contents).unwrap();
        let mut command = cli(&input, &output, &scratch.0, "auto");
        if extension == "ll" {
            command.env(
                "METAL2VULKAN_LLVM_LIBRARY",
                scratch.0.join("absent-library"),
            );
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        tools::spirv_val_bytes(&std::fs::read(output).unwrap(), &scratch.0).unwrap();
    }
}

#[test]
fn cli_passthrough_needs_neither_llvm_nor_spirv_executables() {
    let scratch = Scratch::new();
    let input = scratch.0.join("fragment.ll");
    std::fs::write(
        &input,
        r#"
define <4 x float> @f(<2 x float> %uv) { ret <4 x float> zeroinitializer }
!air.fragment = !{!0}
!0 = !{ptr @f, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"generated(uv)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#,
    )
    .unwrap();
    let output = scratch.0.join("passthrough.spv");
    let result = cli(&input, &output, &scratch.0, "passthrough")
        .env(
            "METAL2VULKAN_LLVM_LIBRARY",
            scratch.0.join("absent-library"),
        )
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    tools::spirv_val_bytes(&std::fs::read(output).unwrap(), &scratch.0).unwrap();
}

#[test]
fn missing_llvm_library_is_an_explicit_fallback_for_bitcode() {
    let scratch = Scratch::new();
    let input = scratch.0.join("input.air");
    std::fs::write(&input, tools::llvm_assemble(KERNEL).unwrap()).unwrap();
    let result = cli(&input, &scratch.0.join("out.spv"), &scratch.0, "auto")
        .env(
            "METAL2VULKAN_LLVM_LIBRARY",
            scratch.0.join("absent-library"),
        )
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("load LLVM library"));
}
