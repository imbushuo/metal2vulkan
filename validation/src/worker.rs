//! Process-boundary guards for this project's isolated native-library workers.

use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant};

pub const TIME_LIMIT: Duration = Duration::from_secs(20);
pub const MEMORY_LIMIT_BYTES: u64 = 500 * 1024 * 1024;

pub fn wait_bounded(child: &mut Child, started: Instant) -> Result<ExitStatus, String> {
    let result = wait_inner(child, started);
    if result.is_err() {
        terminate_child(child);
    }
    result
}

fn wait_inner(child: &mut Child, started: Instant) -> Result<ExitStatus, String> {
    loop {
        if started.elapsed() >= TIME_LIMIT {
            return Err(format!(
                "worker timeout after {} seconds",
                TIME_LIMIT.as_secs()
            ));
        }
        if let Some(resident) = worker_resident_bytes(child.id())? {
            if resident > MEMORY_LIMIT_BYTES {
                return Err(format!(
                    "worker exceeded 500 MiB resident-memory budget (measured {} MiB)",
                    resident / (1024 * 1024)
                ));
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(error) => return Err(format!("poll worker: {error}")),
        }
    }
}

#[cfg(target_os = "macos")]
pub fn worker_resident_bytes(pid: u32) -> Result<Option<u64>, String> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    let result = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            usage.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if result == 0 {
        Ok(Some(unsafe { usage.assume_init() }.ri_resident_size))
    } else if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
        Ok(None)
    } else {
        Err(format!(
            "read worker resident memory: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(target_os = "linux")]
pub fn worker_resident_bytes(pid: u32) -> Result<Option<u64>, String> {
    let path = format!("/proc/{pid}/status");
    let status = match std::fs::read_to_string(&path) {
        Ok(status) => status,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read {path}: {error}")),
    };
    let resident = status.lines().find_map(|line| line.strip_prefix("VmRSS:"));
    // An exited (zombie) worker has no address space and no VmRSS entry.
    resident
        .map(|value| {
            value
                .split_whitespace()
                .next()
                .ok_or("missing VmRSS value")?
                .parse::<u64>()
                .map(|kib| kib * 1024)
                .map_err(|_| "invalid VmRSS value")
        })
        .transpose()
        .map_err(str::to_string)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn worker_resident_bytes(_pid: u32) -> Result<Option<u64>, String> {
    Err("worker memory monitoring requires macOS or Linux".into())
}

#[cfg(unix)]
pub fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt as _;
    command.process_group(0);
}

#[cfg(not(unix))]
pub fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
pub fn terminate_child(child: &mut Child) {
    let group = -(child.id() as i32);
    unsafe {
        let _ = libc::kill(group, libc::SIGTERM);
    }
    let grace_started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if grace_started.elapsed() < Duration::from_millis(100) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => break,
        }
    }
    unsafe {
        let _ = libc::kill(group, libc::SIGKILL);
    }
    let _ = child.wait();
}

#[cfg(not(unix))]
pub fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
