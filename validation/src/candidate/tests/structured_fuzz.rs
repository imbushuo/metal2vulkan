//! Differential execution of randomly generated STRUCTURED control flow.
//!
//! `irreducible_fuzz.rs` covers the state machine `src/native/relooper.rs` builds when no block
//! dominates the cycle. The other side of that fork -- `src/native/cfg/blocks.rs`, which infers
//! where a loop merges, which selection merge a conditional shares, and when a `switch` arm
//! bypasses one -- is the single largest block of translator code the corpus reaches and no
//! authored case does, and it runs 66.8M times over the corpus. Nothing chose the shapes it sees.
//!
//! So: generate a random STRUCTURED program -- nested loops, two-armed conditionals, multi-way
//! switches, multi-level breaks and continues, early returns -- write it out as AIR, run it, and
//! compare against the same program interpreted in Rust. Being generated from a tree it is
//! reducible by construction, which is exactly what sends it down the structured path instead of
//! the relooper's.
//!
//! Two decisions make the generator honest:
//!
//! * **The accumulator lives in an `alloca`, not in phis.** Threading a phi through an arbitrary
//!   nest of breaks is the generator solving the same problem the translator is being tested on,
//!   and a generator bug there would read as a translator bug. In memory the generated module is
//!   pure control flow, and reconstructing the values is the translator's job.
//! * **Every operation site mixes its own id into the accumulator.** Two different paths through
//!   the same operations then give different answers, so a wrong PATH cannot hide behind a right
//!   number.

use super::gpu_fuzz::{
    air_header, execute_generated_bounded, kernel_metadata, Xorshift, OPS, THREADS,
};

/// Loop trip count -- large enough that a `continue` and a `break` are both reachable more than
/// once, small enough that the deepest nest the generator can build stays bounded by a number this
/// file can state.
const TRIPS: u32 = 4;
/// Deepest nesting the generator is allowed to build.
const MAX_DEPTH: usize = 5;
/// Statements per block of the generated tree.
const FANOUT: usize = 3;

/// One node of the generated program. Every variant is a shape `cfg/blocks.rs` has a rule for.
#[derive(Clone)]
enum Stmt {
    /// `acc = op(acc) ^ site` -- the site id is what makes the PATH observable.
    Op(usize, u32),
    /// Two-armed conditional on a bit of the accumulator: a selection whose merge has to be found.
    If(u32, Vec<Stmt>, Vec<Stmt>),
    /// A counted loop. Its body may leave through the latch, a break, a continue or a return, so
    /// the merge is not simply "the block after the body".
    Loop(Vec<Stmt>),
    /// Leave the `level`-th enclosing loop counting outwards from the innermost.
    Break(usize),
    /// Jump to the `level`-th enclosing loop's latch.
    Continue(usize),
    /// A multi-way switch on `(acc >> bit) % arms` -- some arm bodies fall through to the merge and
    /// some leave the enclosing loop, which is the `rewrite_switch_bypass_merge` shape.
    Switch(u32, Vec<Vec<Stmt>>),
    /// Stop, with whatever the accumulator holds.
    Return,
    /// A conditional whose TRUE arm can jump into the middle of its FALSE arm -- the one shape here
    /// that is reducible but not a tree, so `native::cfg::structured_plan` declines it and the
    /// fall-through tier in `native/cfg/blocks.rs` (`infer_branch_merges`, `infer_loop_merges`,
    /// `infer_switch_merges`) takes over. Without it the generator produces perfect trees and never
    /// leaves the structured planner, which is measurable: an `eprintln` on those three shows only
    /// `infer_switch_merges` firing.
    ///
    /// `CrossArm(a, b, pre, shared, other)` is
    /// `if (a) { pre; if (b) shared } else { other; shared }`, emitted with ONE copy of `shared`,
    /// so its block has two predecessors and is not the selection's merge.
    CrossArm(u32, u32, Vec<Stmt>, Vec<Stmt>, Vec<Stmt>),
}

struct Generator {
    rng: Xorshift,
    sites: u32,
}

impl Generator {
    /// `depth` is how much nesting is still allowed; `loops` how many loops enclose this point (a
    /// `break`/`continue` is only generated where one can legally go).
    fn body(&mut self, depth: usize, loops: usize) -> Vec<Stmt> {
        (0..FANOUT).map(|_| self.stmt(depth, loops)).collect()
    }

    fn stmt(&mut self, depth: usize, loops: usize) -> Stmt {
        // At the leaves, and half the time above them, emit straight-line work. Without that the
        // tree is all control flow and the blocks carry nothing to get wrong.
        if depth == 0 || self.rng.below(2) == 0 {
            let site = self.sites;
            self.sites += 1;
            return Stmt::Op(self.rng.below(OPS.len()), site);
        }
        let choices = if loops > 0 { 7 } else { 4 };
        match self.rng.below(choices) {
            0 => Stmt::If(
                self.rng.below(5) as u32,
                self.body(depth - 1, loops),
                self.body(depth - 1, loops),
            ),
            1 => Stmt::Loop(self.body(depth - 1, loops + 1)),
            2 => {
                let arms = 2 + self.rng.below(2);
                let bit = self.rng.below(3) as u32;
                Stmt::Switch(
                    bit,
                    (0..arms).map(|_| self.body(depth - 1, loops)).collect(),
                )
            }
            3 => Stmt::CrossArm(
                self.rng.below(5) as u32,
                self.rng.below(5) as u32,
                self.body(depth - 1, loops),
                self.body(depth - 1, loops),
                self.body(depth - 1, loops),
            ),
            4 => Stmt::Break(self.rng.below(loops)),
            5 => Stmt::Continue(self.rng.below(loops)),
            _ => Stmt::Return,
        }
    }
}

/// How a body finished. `Break(n)`/`Continue(n)` count OUTWARDS from the innermost enclosing loop,
/// so a loop consumes level 0 and passes anything higher up with one subtracted.
#[derive(PartialEq, Clone, Copy, Debug)]
enum Escape {
    /// Control reaches the statement after this body.
    Fall,
    Break(usize),
    Continue(usize),
    Return,
}

/// Interpret the program exactly as the shader must.
///
/// `budget` bounds the total number of operations, so a generator bug cannot hang the test. It is
/// far above what any generated program reaches; a run that exhausts it is reported rather than
/// silently compared.
fn interpret(body: &[Stmt], acc: &mut u32, budget: &mut u32) -> Escape {
    for stmt in body {
        if *budget == 0 {
            return Escape::Return;
        }
        let escape = match stmt {
            Stmt::Op(op, site) => {
                *budget -= 1;
                *acc = (OPS[*op].1)(*acc) ^ site;
                Escape::Fall
            }
            Stmt::Break(level) => Escape::Break(*level),
            Stmt::Continue(level) => Escape::Continue(*level),
            Stmt::Return => Escape::Return,
            Stmt::If(bit, then_body, else_body) => {
                let taken = if (*acc >> bit) & 1 == 1 {
                    then_body
                } else {
                    else_body
                };
                interpret(taken, acc, budget)
            }
            Stmt::Switch(bit, arms) => {
                let arm = ((*acc >> bit) as usize) % arms.len();
                interpret(&arms[arm], acc, budget)
            }
            Stmt::CrossArm(a, b, pre, shared, other) => {
                if (*acc >> a) & 1 == 1 {
                    match interpret(pre, acc, budget) {
                        Escape::Fall => {
                            if (*acc >> b) & 1 == 1 {
                                interpret(shared, acc, budget)
                            } else {
                                Escape::Fall
                            }
                        }
                        escape => escape,
                    }
                } else {
                    match interpret(other, acc, budget) {
                        Escape::Fall => interpret(shared, acc, budget),
                        escape => escape,
                    }
                }
            }
            Stmt::Loop(inner) => {
                let mut out = Escape::Fall;
                for _ in 0..TRIPS {
                    if *budget == 0 {
                        return Escape::Return;
                    }
                    match interpret(inner, acc, budget) {
                        Escape::Fall | Escape::Continue(0) => {}
                        Escape::Break(0) => break,
                        Escape::Break(level) => {
                            out = Escape::Break(level - 1);
                            break;
                        }
                        Escape::Continue(level) => {
                            out = Escape::Continue(level - 1);
                            break;
                        }
                        Escape::Return => {
                            out = Escape::Return;
                            break;
                        }
                    }
                }
                out
            }
        };
        if escape != Escape::Fall {
            return escape;
        }
    }
    Escape::Fall
}

/// Builds the module one basic block at a time.
///
/// The emitter answers a DIFFERENT question from the interpreter, and conflating the two is what
/// makes a generated module malformed. The interpreter asks "did this execution stop here", which
/// is dynamic. The emitter asks "is the block after this statement still open", which is
/// structural -- and a loop's exit block is always structurally reachable, from the trip counter
/// running out, even when every path through its body leaves for an outer target. So `emit`
/// returns a plain "still open" and a `Loop` always reports open, while `interpret` propagates the
/// escape outwards. They agree on what the shader computes because the paths that carry an escape
/// never reach the loop exit at run time.
struct Builder {
    blocks: Vec<(String, Vec<String>)>,
    next: usize,
    /// Label of the latch and of the exit for each enclosing loop, innermost last.
    loops: Vec<(String, String)>,
}

impl Builder {
    fn fresh(&mut self, tag: &str) -> String {
        self.next += 1;
        format!("{tag}{}", self.next)
    }

    fn open(&mut self, label: &str) {
        self.blocks.push((label.to_string(), Vec::new()));
    }

    fn line(&mut self, text: String) {
        self.blocks
            .last_mut()
            .expect("a block is open")
            .1
            .push(text);
    }

    fn value(&mut self, tag: &str) -> String {
        self.next += 1;
        format!("%{tag}{}", self.next)
    }

    /// Emit `body`. Returns whether the current block is still open for more statements; `false`
    /// means the body terminated it with a branch of its own.
    fn emit(&mut self, body: &[Stmt], acc: &str) -> bool {
        for stmt in body {
            if !self.emit_one(stmt, acc) {
                return false;
            }
        }
        true
    }

    /// `(acc >> bit) & 1 == 1`, as an `i1`.
    fn test_bit(&mut self, acc: &str, bit: u32) -> String {
        let loaded = self.load(acc);
        let shifted = self.value("s");
        self.line(format!("  {shifted} = lshr i32 {loaded}, {bit}"));
        let masked = self.value("a");
        self.line(format!("  {masked} = and i32 {shifted}, 1"));
        let cond = self.value("c");
        self.line(format!("  {cond} = icmp eq i32 {masked}, 1"));
        cond
    }

    fn load(&mut self, acc: &str) -> String {
        let value = self.value("v");
        self.line(format!("  {value} = load i32, ptr {acc}, align 4"));
        value
    }

    fn emit_one(&mut self, stmt: &Stmt, acc: &str) -> bool {
        match stmt {
            Stmt::Op(op, site) => {
                let loaded = self.load(acc);
                let applied = self.value("o");
                self.line(format!(
                    "  {applied} = {}",
                    OPS[*op].0.replace("%ACC", &loaded)
                ));
                let mixed = self.value("m");
                self.line(format!("  {mixed} = xor i32 {applied}, {site}"));
                self.line(format!("  store i32 {mixed}, ptr {acc}, align 4"));
                true
            }
            Stmt::Break(level) => {
                let target = self.loops[self.loops.len() - 1 - level].1.clone();
                self.line(format!("  br label %{target}"));
                false
            }
            Stmt::Continue(level) => {
                let target = self.loops[self.loops.len() - 1 - level].0.clone();
                self.line(format!("  br label %{target}"));
                false
            }
            Stmt::Return => {
                self.line("  br label %exit".to_string());
                false
            }
            Stmt::If(bit, then_body, else_body) => {
                let (t, e, m) = (self.fresh("t"), self.fresh("e"), self.fresh("m"));
                let cond = self.test_bit(acc, *bit);
                self.line(format!("  br i1 {cond}, label %{t}, label %{e}"));
                let mut merged = false;
                for (label, arm) in [(&t, then_body), (&e, else_body)] {
                    self.open(label);
                    if self.emit(arm, acc) {
                        self.line(format!("  br label %{m}"));
                        merged = true;
                    }
                }
                // Both arms left: there is no merge block to open, and an unreachable one is the
                // shape that makes `spirv-opt` choke, so do not emit it.
                if merged {
                    self.open(&m);
                }
                merged
            }
            Stmt::Switch(bit, arms) => {
                let merge = self.fresh("sm");
                let labels: Vec<String> = (0..arms.len()).map(|_| self.fresh("sa")).collect();
                let loaded = self.load(acc);
                let shifted = self.value("s");
                self.line(format!("  {shifted} = lshr i32 {loaded}, {bit}"));
                let selector = self.value("k");
                self.line(format!("  {selector} = urem i32 {shifted}, {}", arms.len()));
                self.line(format!(
                    "  switch i32 {selector}, label %{} [",
                    labels[arms.len() - 1]
                ));
                for (index, label) in labels.iter().enumerate().take(arms.len() - 1) {
                    self.line(format!("    i32 {index}, label %{label}"));
                }
                self.line("  ]".to_string());
                let mut merged = false;
                for (label, arm) in labels.iter().zip(arms) {
                    self.open(label);
                    if self.emit(arm, acc) {
                        self.line(format!("  br label %{merge}"));
                        merged = true;
                    }
                }
                if merged {
                    self.open(&merge);
                }
                merged
            }
            Stmt::CrossArm(a, b, pre, shared, other) => {
                let (t, e, sh, m) = (
                    self.fresh("xt"),
                    self.fresh("xe"),
                    self.fresh("xs"),
                    self.fresh("xm"),
                );
                let cond = self.test_bit(acc, *a);
                self.line(format!("  br i1 {cond}, label %{t}, label %{e}"));

                self.open(&t);
                let true_open = self.emit(pre, acc);
                if true_open {
                    // The true arm's second test is what makes this not a tree: one edge goes into
                    // the shared block the FALSE arm also falls into, the other straight to the
                    // merge, so `%{sh}` has two predecessors and is not `%{m}`.
                    let inner = self.test_bit(acc, *b);
                    self.line(format!("  br i1 {inner}, label %{sh}, label %{m}"));
                }

                self.open(&e);
                let false_open = self.emit(other, acc);
                if false_open {
                    self.line(format!("  br label %{sh}"));
                }

                let mut merge_reached = true_open;
                if true_open || false_open {
                    self.open(&sh);
                    if self.emit(shared, acc) {
                        self.line(format!("  br label %{m}"));
                        merge_reached = true;
                    }
                }
                if merge_reached {
                    self.open(&m);
                }
                merge_reached
            }
            Stmt::Loop(inner) => {
                let (head, body, latch, exit) = (
                    self.fresh("lh"),
                    self.fresh("lb"),
                    self.fresh("ll"),
                    self.fresh("lx"),
                );
                let counter = self.value("n");
                // The trip counter is in memory too, for the same reason the accumulator is: a
                // `continue` from inside a nest would otherwise need the generator to place a phi.
                self.line(format!("  {counter} = alloca i32, align 4"));
                self.line(format!("  store i32 0, ptr {counter}, align 4"));
                self.line(format!("  br label %{head}"));
                self.open(&head);
                let seen = self.load(&counter);
                let done = self.value("c");
                self.line(format!("  {done} = icmp ult i32 {seen}, {TRIPS}"));
                self.line(format!("  br i1 {done}, label %{body}, label %{exit}"));
                self.open(&body);
                self.loops.push((latch.clone(), exit.clone()));
                let body_open = self.emit(inner, acc);
                self.loops.pop();
                if body_open {
                    self.line(format!("  br label %{latch}"));
                }
                self.open(&latch);
                let count = self.load(&counter);
                let stepped = self.value("i");
                self.line(format!("  {stepped} = add i32 {count}, 1"));
                self.line(format!("  store i32 {stepped}, ptr {counter}, align 4"));
                self.line(format!("  br label %{head}"));
                // The exit block is reachable from the HEAD, on the trip counter running out, no
                // matter what the body does -- so it is always opened and the loop always leaves
                // the caller with an open block. Returning "closed" here because every path
                // through the body happened to break to an outer loop is what leaves `%exit`
                // without a terminator, and the translator rejects that module rather than
                // miscompiling it: "block %lxNN role=Normal has no typed carrier".
                self.open(&exit);
                true
            }
        }
    }
}

fn program_air(entry: &str, body: &[Stmt]) -> String {
    let mut builder = Builder {
        blocks: Vec::new(),
        next: 0,
        loops: Vec::new(),
    };
    builder.open("entry");
    builder.line("  %acc = alloca i32, align 4".to_string());
    builder.line("  %slot = zext i32 %tid to i64".to_string());
    builder
        .line("  %in = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %slot".to_string());
    builder.line("  %seed = load i32, ptr addrspace(1) %in, align 4".to_string());
    builder.line("  store i32 %seed, ptr %acc, align 4".to_string());
    if builder.emit(body, "%acc") {
        builder.line("  br label %exit".to_string());
    }
    builder.open("exit");
    builder.line("  %result = load i32, ptr %acc, align 4".to_string());
    builder.line(
        "  %out = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %slot".to_string(),
    );
    builder.line("  store i32 %result, ptr addrspace(1) %out, align 4".to_string());
    builder.line("  ret void".to_string());

    let mut air = air_header(entry);
    air.push_str(&format!(
        "define void @{entry}(ptr addrspace(1) %output, ptr addrspace(1) %input, i32 %tid) {{\n"
    ));
    for (label, lines) in &builder.blocks {
        air.push_str(&format!("{label}:\n"));
        for line in lines {
            air.push_str(line);
            air.push('\n');
        }
    }
    air.push_str("}\n\n");
    air.push_str(&kernel_metadata(entry, 1));
    air
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn generated_structured_control_flow_computes_what_the_program_says() {
    let mut failures: Vec<String> = Vec::new();
    let mut executed = 0usize;
    for seed in 1u64..=24 {
        for depth in 2usize..=MAX_DEPTH {
            let mut generator = Generator {
                rng: Xorshift(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1),
                sites: 1,
            };
            let body = generator.body(depth, 0);
            let seeds: Vec<u32> = (0..THREADS)
                .map(|thread| thread.wrapping_mul(2_654_435_761).wrapping_add(1))
                .collect();
            let mut want = Vec::with_capacity(THREADS as usize);
            let mut starved = false;
            for value in &seeds {
                let mut acc = *value;
                let mut budget = 200_000u32;
                interpret(&body, &mut acc, &mut budget);
                starved |= budget == 0;
                want.push(acc);
            }
            if starved {
                failures.push(format!(
                    "seed={seed} depth={depth}: interpreter budget exhausted"
                ));
                continue;
            }
            let entry = format!("structured_s{seed}_d{depth}");
            // `loop_budget.rs` falls back to a function-scope counter when a loop has no
            // identifiable preheader, and that counter spans every ENTRY to the loop, not each
            // one. So the innermost loop of a `depth`-deep nest can be entered TRIPS^depth times
            // in total, and a budget below that turns a correct run into a divergence. State the
            // worst case and give it four times as much room.
            let worst_case = TRIPS.pow(MAX_DEPTH as u32);
            let run = match execute_generated_bounded(
                &entry,
                &program_air(&entry, &body),
                &[seeds],
                1,
                worst_case * 4,
            ) {
                Ok(run) => run,
                Err(error) => {
                    failures.push(format!("seed={seed} depth={depth}: {error}"));
                    continue;
                }
            };
            executed += 1;
            for (thread, (&got, &expected)) in run
                .words
                .iter()
                .zip(&want)
                .take(THREADS as usize)
                .enumerate()
            {
                if got != expected {
                    failures.push(format!(
                        "seed={seed} depth={depth} thread={thread}: shader gave {got}, the program \
                         says {expected}"
                    ));
                }
            }
        }
    }
    failures.truncate(12);
    assert!(
        failures.is_empty(),
        "{} divergences over {executed} generated programs:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(executed > 0, "no generated program executed at all");
}
