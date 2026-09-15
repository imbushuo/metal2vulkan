#[path = "../build_support.rs"]
mod build_support;
use build_support::{collect_files, hash_files};
use sha2::Digest;
use std::path::PathBuf;

fn main() {
    let product = PathBuf::from("..");
    let product_hash = build_support::product_hash(&product);
    println!(
        "cargo:rustc-env=METAL2VULKAN_PRODUCT_FINGERPRINT={:x}",
        product_hash.clone().finalize()
    );

    // Translation-audit facts also depend on the worker boundary, validation, scheduling, and
    // classification code in this crate. Keep that cache key separate from candidate observations,
    // whose dependency contract is the product translator itself.
    let validation = PathBuf::from(".");
    let mut audit_files = vec![validation.join("Cargo.toml")];
    collect_files(&validation.join("src"), &mut audit_files);
    audit_files.sort();
    let mut audit_hash = product_hash;
    audit_hash.update(b"metal2vulkan-translation-audit\0");
    hash_files(&mut audit_hash, &validation, &audit_files);
    println!(
        "cargo:rustc-env=METAL2VULKAN_TRANSLATION_AUDIT_FINGERPRINT={:x}",
        audit_hash.finalize()
    );
}
