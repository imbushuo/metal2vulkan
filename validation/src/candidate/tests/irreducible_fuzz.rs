//! Differential execution of randomly generated IRREDUCIBLE control flow.
//!
//! Nothing else in the suite reaches `src/native/relooper.rs` with a shape it did not already
//! expect. The authored fixtures cover two-entry and three-entry cycles by hand, but the state
//! machine is general and its spill/restore, phi-edge stores and dispatch encoding are driven by
//! the topology, so the interesting question is what happens on a topology nobody chose.
//!
//! So: generate an automaton, write it out as a CFG whose single cycle is entered at EVERY one of
//! its blocks (irreducible, so `reloop_nest` declines and the state machine takes it), translate
//! it, run it, and compare against the same automaton stepped directly in Rust. The generator is a
//! fixed-seed xorshift, so a failure names a seed that reproduces it.

use super::gpu_fuzz::{air_header, execute_generated, kernel_metadata, Xorshift, OPS, THREADS};

/// A terminating automaton over `blocks` states. Each step applies the block's operation to the
/// accumulator, then moves to one of two successors chosen by a bit OF THE ACCUMULATOR -- so the
/// path is data dependent and the relooper cannot fold the dispatch away.
struct Automaton {
    blocks: usize,
    op: Vec<usize>,
    seed: Vec<u32>,
    succ: Vec<(usize, usize)>,
    bit: Vec<u32>,
}

impl Automaton {
    fn generate(seed: u64, blocks: usize) -> Self {
        let mut rng = Xorshift(seed | 1);
        Automaton {
            blocks,
            op: (0..blocks).map(|_| rng.below(OPS.len())).collect(),
            seed: (0..blocks).map(|_| rng.below(63) as u32 + 1).collect(),
            succ: (0..blocks)
                .map(|_| (rng.below(blocks), rng.below(blocks)))
                .collect(),
            bit: (0..blocks).map(|_| rng.below(5) as u32).collect(),
        }
    }

    /// Step the automaton directly. Returns `(final block, accumulator)` -- both are stored by the
    /// shader, so a wrong PATH is caught even when it happens to produce the right number.
    fn run(&self, steps: u32) -> (u32, u32) {
        let mut block = steps as usize % self.blocks;
        let mut acc = self.seed[block];
        let mut taken = 0u32;
        loop {
            acc = (OPS[self.op[block]].1)(acc);
            taken += 1;
            if taken >= steps {
                return (block as u32, acc);
            }
            let (zero, one) = self.succ[block];
            block = if (acc >> self.bit[block]) & 1 == 1 {
                one
            } else {
                zero
            };
        }
    }

    /// The label a successor edge out of block `p` actually leaves from. A block with two distinct
    /// successors needs a second block to branch in, and THAT is the phi's predecessor -- naming
    /// `%b{p}` there would be a malformed module, and a malformed module would be blamed on the
    /// relooper.
    fn exit_label(&self, p: usize) -> String {
        let (zero, one) = self.succ[p];
        if zero == one {
            format!("b{p}")
        } else {
            format!("s{p}")
        }
    }

    fn air(&self, entry: &str) -> String {
        let mut out = String::new();
        out.push_str(&air_header(entry));
        out.push_str(&format!(
            "define void @{entry}(ptr addrspace(1) %output, ptr addrspace(1) %input, i32 %tid) {{\n"
        ));
        out.push_str("entry:\n");
        out.push_str("  %slot = zext i32 %tid to i64\n");
        out.push_str(
            "  %in_ptr = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %slot\n",
        );
        out.push_str("  %steps = load i32, ptr addrspace(1) %in_ptr, align 4\n");
        out.push_str(&format!("  %pick = urem i32 %steps, {}\n", self.blocks));
        // The switch enters the cycle at EVERY one of its blocks. That is what makes it
        // irreducible: no block dominates the others, so none of them can be a loop header.
        out.push_str("  switch i32 %pick, label %b0 [\n");
        for k in 1..self.blocks {
            out.push_str(&format!("    i32 {k}, label %b{k}\n"));
        }
        out.push_str("  ]\n");

        for k in 0..self.blocks {
            let preds: Vec<usize> = (0..self.blocks)
                .filter(|&p| self.succ[p].0 == k || self.succ[p].1 == k)
                .collect();
            let incoming = |value: &dyn Fn(usize) -> String, initial: String| {
                let mut parts = vec![format!("[ {initial}, %entry ]")];
                for &p in &preds {
                    parts.push(format!("[ {}, %{} ]", value(p), self.exit_label(p)));
                }
                parts.join(", ")
            };
            out.push_str(&format!("\nb{k}:\n"));
            out.push_str(&format!(
                "  %i{k} = phi i32 {}\n",
                incoming(&|p| format!("%i{p}_out"), "0".to_string())
            ));
            out.push_str(&format!(
                "  %a{k} = phi i32 {}\n",
                incoming(&|p| format!("%a{p}_out"), self.seed[k].to_string())
            ));
            out.push_str(&format!(
                "  %a{k}_out = {}\n",
                OPS[self.op[k]].0.replace("%ACC", &format!("%a{k}"))
            ));
            out.push_str(&format!("  %i{k}_out = add i32 %i{k}, 1\n"));
            out.push_str(&format!("  %d{k} = icmp uge i32 %i{k}_out, %steps\n"));
            let (zero, one) = self.succ[k];
            if zero == one {
                out.push_str(&format!("  br i1 %d{k}, label %exit, label %b{zero}\n"));
            } else {
                out.push_str(&format!("  %sh{k} = lshr i32 %a{k}_out, {}\n", self.bit[k]));
                out.push_str(&format!("  %bt{k} = and i32 %sh{k}, 1\n"));
                out.push_str(&format!("  %c{k} = icmp eq i32 %bt{k}, 1\n"));
                out.push_str(&format!("  br i1 %d{k}, label %exit, label %s{k}\n"));
                out.push_str(&format!("\ns{k}:\n"));
                out.push_str(&format!("  br i1 %c{k}, label %b{one}, label %b{zero}\n"));
            }
        }

        out.push_str("\nexit:\n");
        let value: Vec<String> = (0..self.blocks)
            .map(|k| format!("[ %a{k}_out, %b{k} ]"))
            .collect();
        let which: Vec<String> = (0..self.blocks)
            .map(|k| format!("[ {k}, %b{k} ]"))
            .collect();
        out.push_str(&format!("  %vf = phi i32 {}\n", value.join(", ")));
        out.push_str(&format!("  %kf = phi i32 {}\n", which.join(", ")));
        out.push_str("  %w = shl i32 %tid, 1\n");
        out.push_str("  %w64 = zext i32 %w to i64\n");
        out.push_str("  %o0 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %w64\n");
        out.push_str("  store i32 %vf, ptr addrspace(1) %o0, align 4\n");
        out.push_str("  %w1 = add i64 %w64, 1\n");
        out.push_str("  %o1 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %w1\n");
        out.push_str("  store i32 %kf, ptr addrspace(1) %o1, align 4\n");
        out.push_str("  ret void\n}\n\n");
        out.push_str(&kernel_metadata(entry, 1));
        out
    }
}

/// Run one generated automaton and read back the `(final block, accumulator)` each thread reported.
fn execute_automaton(entry: &str, air: &str) -> Result<(bool, Vec<(u32, u32)>), String> {
    let inputs = vec![(0..THREADS).collect::<Vec<u32>>()];
    let run = execute_generated(entry, air, &inputs, 2)?;
    Ok((
        run.state_machine,
        run.words
            .chunks_exact(2)
            .map(|pair| (pair[1], pair[0]))
            .collect(),
    ))
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn generated_irreducible_cycles_step_the_same_automaton_the_state_machine_replaces() {
    let mut failures: Vec<String> = Vec::new();
    let mut diverged: Vec<String> = Vec::new();
    let mut executed = 0usize;
    let mut reached = 0usize;
    let mut structured: Vec<String> = Vec::new();
    for seed in 1u64..=24 {
        for blocks in 2usize..=5 {
            let machine = Automaton::generate(seed.wrapping_mul(0x9e3779b97f4a7c15), blocks);
            let entry = format!("irreducible_s{seed}_b{blocks}");
            let air = machine.air(&entry);
            let (state_machine, observed) = match execute_automaton(&entry, &air) {
                Ok(result) => result,
                Err(error) => {
                    failures.push(format!("seed={seed} blocks={blocks}: {error}"));
                    continue;
                }
            };
            executed += 1;
            if state_machine {
                reached += 1;
            } else {
                structured.push(format!("seed={seed} blocks={blocks}"));
            }
            for (thread, &got) in observed.iter().enumerate() {
                let want = machine.run(thread as u32);
                if got != want {
                    if !diverged.contains(&format!("seed={seed} blocks={blocks}")) {
                        diverged.push(format!("seed={seed} blocks={blocks}"));
                    }
                    failures.push(format!(
                        "seed={seed} blocks={blocks} thread={thread}: \
                         GPU gave block={} acc={}, the automaton gives block={} acc={}",
                        got.0, got.1, want.0, want.1
                    ));
                }
            }
        }
    }
    assert!(executed > 0, "no generated automaton executed at all");
    // Not every generated topology stays irreducible -- a two-block automaton whose successors
    // collapse can still be nested -- so the count is reported rather than required per case. What
    // IS required is that the state machine is still on the path at all: if a future change routed
    // these through the ordinary structurizer, this test would stop testing the relooper silently.
    assert!(
        reached * 2 > executed,
        "only {reached} of {executed} generated automata reached the state machine (the budget \
         bounds a relooper loop by RETURNING, a structured one in place); nested instead: {}",
        structured.join(", ")
    );
    assert!(
        failures.is_empty(),
        "{} of {executed} generated irreducible automata diverged, over {} threads:\n{}",
        diverged.len(),
        failures.len(),
        failures.join("\n")
    );
}
