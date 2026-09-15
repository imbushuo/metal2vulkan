mod build_support;
use sha2::Digest;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let hash = build_support::product_hash(&root);
    println!(
        "cargo:rustc-env=METAL2VULKAN_BUILD_FINGERPRINT={:x}",
        hash.finalize()
    );
}
