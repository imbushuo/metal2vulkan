//! Differential execution of THREADGROUP memory, barriers and workgroup atomics.
//!
//! Three seams meet here and none is reached by an authored case: `passes/access/workgroup.rs`
//! (7 of 19 functions unverified, 3.2M corpus executions), `passes/workgroup/atomic_loop.rs`
//! (7 of 17) and `native/wg_atomic.rs` (7 of 15). They decide how a `addrspace(3)` array is laid
//! out, what a `air.wg.barrier` orders, and how `air.atomic.local.*` lowers.
//!
//! The generated kernel does what a threadgroup reduction does: every thread writes its own slot,
//! a barrier, every thread reads a PERMUTED slot, and the value it read has to be the one another
//! thread wrote. Then every thread folds one operand into a shared counter with a local atomic and,
//! after another barrier, reads the total -- which is order-independent for every operation used
//! here, so it is exactly predictable even though the order is not.
//!
//! **The sub-word slot types are the sharp part.** A `[32 x i8]` threadgroup array puts four
//! threads in one 32-bit word. Where the emitter has a 16-bit pointer it stores the component; the
//! alternative -- rebuild the whole word and store it back -- loses three writes of every four, and
//! only the slots this thread did NOT write can say so. So the read is permuted: thread `t` checks
//! what thread `perm(t)` wrote, never its own.

use super::gpu_fuzz::{air_header, execute_generated, kernel_metadata, Xorshift, THREADS};

/// One threadgroup slot type. `bits` is what survives a round trip through it.
#[derive(Clone, Copy)]
struct Slot {
    /// The array's LLVM element type.
    elem: &'static str,
    /// Bits a stored value keeps.
    bits: u32,
}

const SLOTS: [Slot; 4] = [
    Slot {
        elem: "i32",
        bits: 32,
    },
    Slot {
        elem: "i16",
        bits: 16,
    },
    // Four threads to a word, which is the point.
    Slot {
        elem: "i8",
        bits: 8,
    },
    Slot {
        elem: "i64",
        bits: 32,
    },
];

/// The atomic folded into the shared counter. Every one is commutative AND associative, so the
/// final value does not depend on the order the threads got there -- which is the only reason a
/// dispatch-wide atomic is checkable at all.
#[derive(Clone, Copy)]
struct Atomic {
    symbol: &'static str,
    /// Starting value, chosen to be the operation's identity.
    identity: u32,
    fold: fn(u32, u32) -> u32,
}

const ATOMICS: [Atomic; 5] = [
    Atomic {
        symbol: "add.u",
        identity: 0,
        fold: |a, b| a.wrapping_add(b),
    },
    Atomic {
        symbol: "or.u",
        identity: 0,
        fold: |a, b| a | b,
    },
    Atomic {
        symbol: "xor.u",
        identity: 0,
        fold: |a, b| a ^ b,
    },
    Atomic {
        symbol: "max.u",
        identity: 0,
        fold: |a, b| a.max(b),
    },
    Atomic {
        symbol: "min.u",
        identity: u32::MAX,
        fold: |a, b| a.min(b),
    },
];

struct Program {
    slot: Slot,
    /// `perm(t) = (t * mul + add) & 31`, with `mul` odd so it is a bijection and every slot is read
    /// exactly once.
    mul: u32,
    add: u32,
    /// Barrier-separated write/read rounds.
    rounds: usize,
    atomic: Atomic,
}

impl Program {
    fn generate(seed: u64) -> Self {
        let mut rng = Xorshift(seed | 1);
        Program {
            slot: SLOTS[rng.below(SLOTS.len())],
            mul: (rng.below(16) as u32) * 2 + 1,
            add: rng.below(THREADS as usize) as u32,
            rounds: 1 + rng.below(3),
            atomic: ATOMICS[rng.below(ATOMICS.len())],
        }
    }

    fn permuted(&self, thread: u32) -> u32 {
        (thread.wrapping_mul(self.mul).wrapping_add(self.add)) & (THREADS - 1)
    }

    /// What a value keeps after a round trip through one slot.
    fn narrowed(&self, value: u32) -> u32 {
        if self.slot.bits >= 32 {
            value
        } else {
            value & ((1u32 << self.slot.bits) - 1)
        }
    }

    /// The same computation, all 32 threads together. Barriers make the rounds lockstep, so a plain
    /// "write every slot, then read every slot" is the exact semantics -- no scheduling to model.
    fn run(&self, seeds: &[u32]) -> Vec<u32> {
        let mut acc: Vec<u32> = seeds.to_vec();
        for round in 0..self.rounds {
            let written: Vec<u32> = (0..THREADS as usize)
                .map(|thread| self.narrowed(acc[thread] ^ (round as u32 + 1)))
                .collect();
            for thread in 0..THREADS as usize {
                let read = written[self.permuted(thread as u32) as usize];
                acc[thread] = acc[thread].wrapping_mul(3) ^ read;
            }
        }
        let total = (0..THREADS as usize).fold(self.atomic.identity, |folded, thread| {
            (self.atomic.fold)(folded, acc[thread] | 1)
        });
        acc.iter().map(|value| value ^ total).collect()
    }

    fn air(&self, entry: &str) -> String {
        let elem = self.slot.elem;
        let mut out = air_header(entry);
        out.push_str(&format!(
            "@scratch = internal unnamed_addr addrspace(3) global [{THREADS} x {elem}] undef, align 8\n"
        ));
        out.push_str("@total = internal unnamed_addr addrspace(3) global i32 undef, align 4\n\n");
        out.push_str("declare void @air.wg.barrier(i32, i32)\n");
        out.push_str(&format!(
            "declare i32 @air.atomic.local.{}.i32(ptr addrspace(3), i32, i32, i32, i1)\n\n",
            self.atomic.symbol
        ));
        out.push_str(&format!(
            "define void @{entry}(ptr addrspace(1) %output, ptr addrspace(1) %input, i32 %tid) {{\nentry:\n"
        ));
        out.push_str("  %t64 = zext i32 %tid to i64\n");
        out.push_str("  %ip = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %t64\n");
        out.push_str("  %acc0 = load i32, ptr addrspace(1) %ip, align 4\n");
        // Thread 0 seeds the shared counter with the operation's identity, before any barrier that
        // another thread's atomic could cross.
        out.push_str("  %isz = icmp eq i32 %tid, 0\n");
        out.push_str("  br i1 %isz, label %seedit, label %seeded\n");
        out.push_str("seedit:\n");
        out.push_str(&format!(
            "  store i32 {}, ptr addrspace(3) @total, align 4\n",
            self.atomic.identity as i32
        ));
        out.push_str("  br label %seeded\n");
        out.push_str("seeded:\n");
        out.push_str("  tail call void @air.wg.barrier(i32 2, i32 1)\n");
        // `perm(t)`, computed once: the multiply and mask are the same every round.
        out.push_str(&format!("  %pm = mul i32 %tid, {}\n", self.mul));
        out.push_str(&format!("  %pa = add i32 %pm, {}\n", self.add));
        out.push_str(&format!("  %pi = and i32 %pa, {}\n", THREADS - 1));
        out.push_str("  %pi64 = zext i32 %pi to i64\n");
        out.push_str(&format!(
            "  %wp = getelementptr inbounds [{THREADS} x {elem}], ptr addrspace(3) @scratch, i64 0, i64 %t64\n"
        ));
        out.push_str(&format!(
            "  %rp = getelementptr inbounds [{THREADS} x {elem}], ptr addrspace(3) @scratch, i64 0, i64 %pi64\n"
        ));
        let mut acc = "%acc0".to_string();
        for round in 0..self.rounds {
            let n = round;
            out.push_str(&format!("  %wv{n} = xor i32 {acc}, {}\n", round + 1));
            let stored = match elem {
                "i32" => format!("%wv{n}"),
                "i64" => {
                    out.push_str(&format!("  %wz{n} = zext i32 %wv{n} to i64\n"));
                    format!("%wz{n}")
                }
                narrow => {
                    out.push_str(&format!("  %wn{n} = trunc i32 %wv{n} to {narrow}\n"));
                    format!("%wn{n}")
                }
            };
            out.push_str(&format!(
                "  store {elem} {stored}, ptr addrspace(3) %wp, align 1\n"
            ));
            out.push_str("  tail call void @air.wg.barrier(i32 2, i32 1)\n");
            out.push_str(&format!(
                "  %rv{n} = load {elem}, ptr addrspace(3) %rp, align 1\n"
            ));
            let read = match elem {
                "i32" => format!("%rv{n}"),
                "i64" => {
                    out.push_str(&format!("  %rt{n} = trunc i64 %rv{n} to i32\n"));
                    format!("%rt{n}")
                }
                narrow => {
                    out.push_str(&format!("  %rz{n} = zext {narrow} %rv{n} to i32\n"));
                    format!("%rz{n}")
                }
            };
            out.push_str(&format!("  %mu{n} = mul i32 {acc}, 3\n"));
            out.push_str(&format!("  %acc{} = xor i32 %mu{n}, {read}\n", n + 1));
            // The next round overwrites the slot this thread's neighbour is about to read, so the
            // read must be fenced from it as well as from the write.
            out.push_str("  tail call void @air.wg.barrier(i32 2, i32 1)\n");
            acc = format!("%acc{}", n + 1);
        }
        // One operand per thread, never zero, so `min` is not trivially the identity.
        out.push_str(&format!("  %op = or i32 {acc}, 1\n"));
        out.push_str(&format!(
            "  %ar = call i32 @air.atomic.local.{}.i32(ptr addrspace(3) @total, i32 %op, i32 0, i32 1, i1 true)\n",
            self.atomic.symbol
        ));
        out.push_str("  tail call void @air.wg.barrier(i32 2, i32 1)\n");
        out.push_str("  %tot = load i32, ptr addrspace(3) @total, align 4\n");
        out.push_str(&format!("  %res = xor i32 {acc}, %tot\n"));
        out.push_str("  %op0 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %t64\n");
        out.push_str("  store i32 %res, ptr addrspace(1) %op0, align 4\n");
        out.push_str("  ret void\n}\n\n");
        out.push_str(&kernel_metadata(entry, 1));
        out
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_threadgroup_slot_holds_what_its_own_thread_wrote() {
    let seeds: Vec<u32> = (0..THREADS)
        .map(|thread| thread.wrapping_mul(0x9e37_79b9).wrapping_add(0x1234_5678))
        .collect();
    let mut failures: Vec<String> = Vec::new();
    let mut executed = 0usize;
    for seed in 1u64..=24 {
        let program = Program::generate(seed.wrapping_mul(0x9e3779b97f4a7c15));
        let want = program.run(&seeds);
        let entry = format!("wg_s{seed}");
        let run = match execute_generated(
            &entry,
            &program.air(&entry),
            std::slice::from_ref(&seeds),
            1,
        ) {
            Ok(run) => run,
            Err(error) => {
                failures.push(format!("seed={seed} slot={}: {error}", program.slot.elem));
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
                    "seed={seed} slot={} rounds={} atomic={} thread={thread} (reads slot {}): \
                     shader gave {got}, the program says {expected}",
                    program.slot.elem,
                    program.rounds,
                    program.atomic.symbol,
                    program.permuted(thread as u32),
                ));
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
