use metal2vulkan_validation::{source, worker, ScratchDir};
use std::process::Command;
use std::time::{Duration, Instant};

#[test]
fn harvest_uses_its_own_worker_and_removes_scratch_on_success_and_failure() {
    let scratch = ScratchDir::new("in-process-harvest-test").unwrap();
    let root = scratch.path().join("corpus");
    let tmp = scratch.path().join("worker-tmp");
    std::fs::create_dir(&tmp).unwrap();
    let input = scratch.path().join("owned.metallib");
    let bitcode = metal2vulkan::tools::llvm_assemble(include_str!(
        "../fixtures/public/kernel_store_const.ll"
    ))
    .unwrap();
    let raw = if bitcode.starts_with(b"\xde\xc0\x17\x0b") {
        let offset = u32::from_le_bytes(bitcode[8..12].try_into().unwrap()) as usize;
        let length = u32::from_le_bytes(bitcode[12..16].try_into().unwrap()) as usize;
        &bitcode[offset..offset + length]
    } else {
        &bitcode
    };
    for (payload, succeeds) in [(raw, true), (&b"not bitcode"[..], false)] {
        let mut wrapped = Vec::new();
        for word in [0x0b17_c0de_u32, 0, 20, payload.len() as u32, u32::MAX] {
            wrapped.extend_from_slice(&word.to_le_bytes());
        }
        wrapped.extend_from_slice(payload);
        std::fs::write(&input, wrapped).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_corpus-harvest"))
            .args(["--metallib"])
            .arg(&input)
            .arg("--out")
            .arg(&root)
            .env("PATH", "")
            .env("TMPDIR", &tmp)
            .env("METAL2VULKAN_LLVM_DIS", scratch.path().join("must-not-run"))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert_eq!(
            !stderr.contains("FAIL LLVM disassembly"),
            succeeds,
            "{stderr}"
        );
        if succeeds {
            assert!(result.status.success(), "{stderr}");
            assert_eq!(source::read_all_private_sources(&root).unwrap().len(), 1);
        }
        assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0, "{stderr}");
    }
}

#[test]
fn worker_deadline_kills_and_reaps_the_project_process() {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", "worker_child"]);
    command.env("M2V_TEST_WORKER_CHILD", "1");
    worker::configure_process_group(&mut command);
    let mut child = command.spawn().unwrap();
    let expired = Instant::now() - worker::TIME_LIMIT - Duration::from_secs(1);
    let error = worker::wait_bounded(&mut child, expired).unwrap_err();
    assert!(error.contains("timeout"));
    assert!(child.try_wait().unwrap().is_some());
}

#[test]
fn worker_child() {
    if std::env::var_os("M2V_TEST_WORKER_CHILD").is_some() {
        std::thread::sleep(Duration::from_secs(30));
    }
}
