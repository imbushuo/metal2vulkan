//! Shared harness for the generated-shader differential tests.
//!
//! Both generators do the same three things -- build AIR text, translate and run it, compare
//! against the same computation done in Rust -- and differ only in what they generate. The pieces
//! that are the same for both live here: the PRNG, the operation table, and the dispatch.

use super::*;

/// `(operand pattern, the same operation on a u32)`. Every entry must be TOTAL and WRAPPING: the
/// generated programs run to data-dependent successors, so any input can reach any operation and
/// there is nothing to exclude a value at.
pub(super) type Operation = (&'static str, fn(u32) -> u32);

pub(super) const OPS: [Operation; 8] = [
    ("mul i32 %ACC, 3", |a| a.wrapping_mul(3)),
    ("add i32 %ACC, 7", |a| a.wrapping_add(7)),
    ("xor i32 %ACC, 255", |a| a ^ 255),
    ("shl i32 %ACC, 1", |a| a << 1),
    ("lshr i32 %ACC, 1", |a| a >> 1),
    ("sub i32 %ACC, 13", |a| a.wrapping_sub(13)),
    ("or i32 %ACC, 1024", |a| a | 1024),
    ("and i32 %ACC, 65535", |a| a & 65535),
];

/// Threads per dispatch. Thread `t` is fed `t`, so one dispatch sweeps 32 inputs at once.
pub(super) const THREADS: u32 = 32;

/// Iterations allowed per loop, for a generator whose programs iterate a handful of times. A run
/// that reaches it is already wrong -- see the note on the budget in [`execute_generated_over`].
///
/// This is NOT a safe default for every generator. `loop_budget.rs` falls back to a single
/// function-scope counter when a loop has no identifiable preheader, and that counter then spans
/// EVERY entry to the loop rather than resetting per entry -- so a five-deep nest of four-trip
/// loops reaches 4^5 header visits at the innermost and trips a budget of 256 while being perfectly
/// bounded. A generator that nests must state its own worst case; see
/// [`execute_generated_bounded`].
const FUZZ_LOOP_BUDGET: u32 = 256;

pub(super) struct Xorshift(pub u64);

impl Xorshift {
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(super) fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// What running one generated module produced.
pub(super) struct Run {
    /// Output words, `output_words` per thread, thread-major.
    pub words: Vec<u32>,
    /// Whether the budget had to bound a loop by RETURNING from the function. That shape --
    /// `OpLoopMerge` then a plain `OpBranch` into the body, with every break inside it -- is the
    /// one `src/passes/loop_budget.rs` names as the relooper's; a structured loop is bounded in
    /// place instead. So this is a free, honest signal for which construction ran.
    pub state_machine: bool,
}

/// The AIR metadata tail: buffer 0 written, buffers `1..=inputs` read, `tid` from the grid.
pub(super) fn kernel_metadata(entry: &str, inputs: usize) -> String {
    let mut nodes = String::from(
        "!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"output\"}\n",
    );
    let mut list = String::from("!3");
    for index in 1..=inputs {
        let node = 3 + index;
        list.push_str(&format!(", !{node}"));
        nodes.push_str(&format!(
            "!{node} = !{{i32 {index}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"in{index}\"}}\n"
        ));
    }
    let tid = 4 + inputs;
    list.push_str(&format!(", !{tid}"));
    nodes.push_str(&format!(
        "!{tid} = !{{i32 {inputs_plus}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}\n",
        inputs_plus = inputs + 1
    ));
    format!(
        "!air.kernel = !{{!0}}\n\
         !0 = !{{ptr @{entry}, !1, !2}}\n\
         !1 = !{{}}\n!2 = !{{{list}}}\n{nodes}"
    )
}

/// As [`kernel_metadata`], but buffer 0 is declared `air.read_write` -- a shader that stores into a
/// laid-out buffer and has its untouched bytes checked has to read them back, and a write-only
/// declaration is not that.
pub(super) fn kernel_metadata_written(entry: &str, inputs: usize) -> String {
    kernel_metadata(entry, inputs).replace("!\"air.write\"", "!\"air.read_write\"")
}

/// The data layout the Metal front end emits, verbatim. The vector entries matter: without them a
/// `<3 x i32>` falls back to `next_power_of_two` for its alignment, and a generated module would be
/// laid out by a rule no real module uses.
pub(super) const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
    f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-\
    v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The module header every generated program shares.
pub(super) fn air_header(entry: &str) -> String {
    format!(
        "source_filename = \"{entry}.ll\"\n\
         target datalayout = \"{DATA_LAYOUT}\"\n\
         target triple = \"air64_v28-apple-macosx26.0.0\"\n\n"
    )
}

/// Translate one generated module, bound its loops, run it, and read `output_words` words per
/// thread back.
pub(super) fn execute_generated(
    entry: &str,
    air: &str,
    inputs: &[Vec<u32>],
    output_words: usize,
) -> Result<Run, String> {
    execute_generated_over(
        entry,
        air,
        inputs,
        &vec![0x9c; THREADS as usize * output_words * 4],
    )
}

/// As [`execute_generated`], but with the caller's own loop budget rather than
/// [`FUZZ_LOOP_BUDGET`]. Pass a number DERIVED from the generator's worst case, not a round one --
/// the budget is the only thing standing between a translator bug and a wedged GPU, so raising it
/// on a hunch gives that up, and leaving it too low turns a correct run into a divergence that
/// looks like a translator bug (it cost a shrink of a 5517-line module to find that out).
pub(super) fn execute_generated_bounded(
    entry: &str,
    air: &str,
    inputs: &[Vec<u32>],
    output_words: usize,
    budget: u32,
) -> Result<Run, String> {
    execute_generated_inner(
        entry,
        air,
        inputs,
        &vec![0x9c; THREADS as usize * output_words * 4],
        budget,
    )
}

/// As [`execute_generated`], but the output buffer starts as `output_init` rather than a fill --
/// for a shader that WRITES into a laid-out buffer, where the point of the test is that everything
/// the shader did not name is still what it was.
pub(super) fn execute_generated_over(
    entry: &str,
    air: &str,
    inputs: &[Vec<u32>],
    output_init: &[u8],
) -> Result<Run, String> {
    execute_generated_inner(entry, air, inputs, output_init, FUZZ_LOOP_BUDGET)
}

fn execute_generated_inner(
    entry: &str,
    air: &str,
    inputs: &[Vec<u32>],
    output_init: &[u8],
    budget: u32,
) -> Result<Run, String> {
    let output_bytes = output_init.len();
    let engine = base64::engine::general_purpose::STANDARD;
    let mut buffers = vec![serde_json::json!({
        "binding": 0, "role": "in_out",
        "initial_bytes_b64": engine.encode(output_init)
    })];
    for (index, words) in inputs.iter().enumerate() {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        buffers.push(serde_json::json!({
            "binding": index + 1, "role": "input", "bytes_b64": engine.encode(&bytes)
        }));
    }
    let case: AuthoredCase = serde_json::from_value(serde_json::json!({
        "air_sha256": crate::hash::sha256_bytes(air.as_bytes()),
        "case_id": format!("test-generated-{entry}"),
        "name": format!("generated-{entry}"),
        "entry": entry,
        "stage": "kernel",
        "buffers": buffers,
        "dispatch": {"grid": [THREADS, 1, 1], "threads_per_threadgroup": [THREADS, 1, 1]},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": output_bytes},
        "compare": {"kind": "exact"},
        "execution_safety": "authored_bounded"
    }))
    .map_err(|error| format!("synthetic case: {error}"))?;
    let resources =
        LiteralResources::prepare(&case).map_err(|error| format!("resources: {error}"))?;
    let scratch =
        crate::ScratchDir::new("gpu-fuzz").map_err(|error| format!("scratch: {error}"))?;
    let options = metal2vulkan::passes::TransformOptions {
        kernel_local_size: [THREADS, 1, 1],
        ..metal2vulkan::passes::TransformOptions::default()
    };
    let (spv, reflection) = metal2vulkan::translate_sanitized_native_reflected(
        air,
        metal2vulkan::passes::Stage::Kernel,
        scratch.path(),
        options,
    )
    .map_err(|error| format!("translate: {error}"))?;
    // `platform::execute` is BELOW `bound_loops`, so a test that calls it directly hands the GPU a
    // module whose termination nothing has checked -- and a translator bug can produce exactly
    // that: a loop with no reachable exit. A committed kernel cannot be cancelled, so on macOS
    // that is not a failing test, it is a wedged machine. Apply the product's own budget, and a
    // small one: every generated program here is bounded far below 256, so a run that reaches it
    // is already wrong, and the early return turns it into the divergence it is.
    let (spv, report) = metal2vulkan::instrument_spirv_loop_budget(&spv, budget)
        .map_err(|error| format!("bound loops: {error}"))?;
    let backend = if cfg!(target_os = "macos") {
        Backend::Moltenvk
    } else {
        Backend::Vulkan
    };
    let (output, _) = platform::execute(&case, &resources, &reflection, &spv, None, None, backend)
        .map_err(|error| format!("execute: {error}"))?;
    if output.len() != output_bytes {
        return Err(format!(
            "read back {} bytes, wanted {output_bytes}",
            output.len()
        ));
    }
    Ok(Run {
        words: output
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect(),
        state_machine: report.loops_bounded_via_early_return > 0,
    })
}
