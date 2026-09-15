//! Whole-corpus MoltenVK pipeline-creation oracle.
//!
//! Translates every kernel source in the corpus and asks the Vulkan driver to build a compute
//! pipeline for it against the descriptor layout our own reflection describes. Nothing is bound,
//! nothing is dispatched, nothing is read back. The single question is whether the driver ACCEPTS
//! the module -- which on macOS means MoltenVK compiled it through MSL and Metal accepted the
//! result.
//!
//! That question has no other asker in this repository. `spirv-val` checks the module against the
//! SPIR-V specification; the authored cases put a real pipeline on the device but reach only a
//! fraction of the sources. A module can be valid SPIR-V, translate without error, and still be
//! refused by the driver -- and until this sweep existed, only luck found one.

use metal2vulkan_validation::source::{
    self, for_each_source_shard_analysis, source_shard_path, SourceRow, SHARD_COUNT,
};
use metal2vulkan_validation::ScratchDir;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-pipeline: {error}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    "usage: corpus-pipeline [--corpus DIR] [--limit N] [--after HASH] [--hash HASH]\n\
     \x20      [--attempting PATH]"
        .to_string()
}

fn run() -> Result<(), String> {
    let mut root = source::corpus_root();
    let mut limit = usize::MAX;
    let mut after = None;
    let mut only = None;
    let mut attempting = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(value()?),
            "--limit" => {
                limit = value()?
                    .parse()
                    .map_err(|error| format!("--limit: {error}"))?
            }
            "--after" => after = Some(value()?),
            "--hash" => only = Some(value()?),
            "--attempting" => attempting = Some(PathBuf::from(value()?)),
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unrecognized argument {other:?}\n{}", usage())),
        }
    }

    let scratch = ScratchDir::new("corpus-pipeline")?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut seen = 0usize;
    let mut emitted = 0usize;
    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut untranslated = 0usize;
    // `--after` resumes a sweep the driver killed: shards are walked in a fixed order, so
    // restarting past the last hash reported covers exactly the remainder.
    let mut skipping = after.is_some();

    let mut visit = |row: SourceRow| -> Result<(), String> {
        if emitted >= limit {
            return Ok(());
        }
        if let Some(only) = &only {
            if &row.air_sha256 != only {
                return Ok(());
            }
        }
        seen += 1;
        if skipping {
            if Some(&row.air_sha256) == after.as_ref() {
                skipping = false;
            }
            return Ok(());
        }
        if row.stage != "Kernel" {
            return Ok(());
        }
        emitted += 1;
        // A driver that wedges takes the sweep with it: MoltenVK retries a dead Metal compiler
        // service, and the module it is retrying never reaches the output. Record the hash BEFORE
        // attempting it, so a supervisor that kills a stalled run can resume past the module that
        // stalled instead of walking back into it.
        if let Some(path) = &attempting {
            std::fs::write(path, format!("{}\n", row.air_sha256))
                .map_err(|error| format!("write {}: {error}", path.display()))?;
        }
        let status = match translate(&row, scratch.path()) {
            Err(error) => {
                untranslated += 1;
                format!("UNTRANSLATED\t{}", one_line(&error))
            }
            Ok((spv, reflection)) => {
                match metal2vulkan_validation::candidate::compile_kernel_pipeline(&reflection, &spv)
                {
                    Ok(()) => {
                        ok += 1;
                        "OK\t".to_string()
                    }
                    Err(error) => {
                        refused += 1;
                        format!("REFUSED\t{}", one_line(&error))
                    }
                }
            }
        };
        writeln!(out, "{}\t{}\t{}", row.air_sha256, row.entry, status)
            .map_err(|error| format!("write result: {error}"))?;
        out.flush().map_err(|error| format!("flush: {error}"))?;
        Ok(())
    };

    for shard in 0..SHARD_COUNT {
        let path = source_shard_path(&root, shard);
        if !path.is_file() {
            continue;
        }
        for_each_source_shard_analysis(&path, &mut visit)?;
    }
    for row in source::public_sources()? {
        visit(row)?;
    }

    eprintln!(
        "corpus-pipeline: {seen} sources, {emitted} kernels, {ok} OK, {refused} REFUSED, \
         {untranslated} UNTRANSLATED"
    );
    Ok(())
}

fn translate(
    row: &SourceRow,
    scratch: &std::path::Path,
) -> Result<(Vec<u8>, metal2vulkan::reflect::ShaderReflection), String> {
    metal2vulkan::translate_sanitized_native_reflected(
        &row.air_ll,
        metal2vulkan::passes::Stage::Kernel,
        scratch,
        metal2vulkan::passes::TransformOptions::default(),
    )
}

/// Translator errors chain with `; ` and driver errors do not embed newlines, but a panic message
/// forwarded as a string can. Keep every result on one TSV line.
fn one_line(error: &str) -> String {
    error.replace(['\n', '\r', '\t'], " ")
}
