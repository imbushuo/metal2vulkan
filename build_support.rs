use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

pub fn product_hash(root: &Path) -> Sha256 {
    let mut files = vec![
        root.join("Cargo.toml"),
        root.join("build.rs"),
        root.join("build_support.rs"),
    ];
    collect_files(&root.join("src"), &mut files);
    files.sort();
    let mut hash = Sha256::new();
    hash_files(&mut hash, root, &files);
    for name in [
        "TARGET",
        "CARGO_CFG_TARGET_FEATURE",
        "CARGO_CFG_TARGET_ENDIAN",
        "CARGO_CFG_TARGET_POINTER_WIDTH",
        "CARGO_FEATURE_SERDE",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(std::env::var(name).unwrap_or_default().as_bytes());
        hash.update([0]);
    }
    let rustc = std::env::var_os("RUSTC").expect("Cargo supplies RUSTC");
    let version = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("query Rust compiler identity");
    assert!(
        version.status.success(),
        "Rust compiler identity query failed"
    );
    hash.update(version.stdout);
    hash
}

pub fn hash_files(hash: &mut Sha256, root: &Path, files: &[PathBuf]) {
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(root).expect("fingerprint input");
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("read fingerprint input {}: {error}", path.display()));
        hash.update(relative.to_string_lossy().as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
}

pub fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_files(&path, output);
        } else if path.is_file() {
            output.push(path);
        }
    }
}
