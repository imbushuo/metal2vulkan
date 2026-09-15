//! Two derivations of one address, over generated aggregate layouts.
//!
//! `src/passes/access` is the largest block of translator code no authored case reaches -- 161 of
//! 446 corpus-reached functions -- and its whole job is to reconcile the two ways a datum inside a
//! buffer can be named: a TYPED access chain that walks the aggregate, and a RAW byte offset that
//! jumps straight to it. `access_chain.rs` normalizes strides and over-indexed chains,
//! `byte_aggregate.rs` and `raw_byte.rs` replay the raw path against the typed one,
//! `vector_subword.rs` handles the narrow members. If those two derivations ever disagree, the
//! translator has a bug, and the disagreement is exactly what this test looks for.
//!
//! So each generated kernel reads ONE leaf, twice: once by `getelementptr` through the record type
//! and once by `getelementptr i8` at a byte offset computed in the generator. Both are compared
//! against each other and against the value the layout rules say is there, so the test says which
//! of the three is wrong rather than only that they differ.
//!
//! A third test asks the question the first two cannot: a sub-word store is lowered as a
//! read-modify-write of the 32-bit word around it, and a word in a DEVICE buffer belongs to the
//! whole dispatch. Two threads writing different bytes of one word each rebuild that word from a
//! copy read before the other stored, so one of the two writes disappears -- and, exactly as in the
//! single-thread store test, the byte that was written is still right, so only the OTHER thread's
//! byte accuses it.
//!
//! The two tests here do NOT cover the same code, and that is deliberate. Asking for the byte
//! offset is what puts the buffer on the raw word view, so the LOAD test exercises
//! `layout::raw_size_align` and the offset arithmetic `passes/access` rebuilds from it. The STORE
//! test names its leaf only through the record type, so the buffer keeps its SPIR-V struct and the
//! layout is carried entirely by `OpMemberDecorate Offset` and `OpDecorate ArrayStride` computed by
//! `layout::spirv_size_align`. Mutating either calculator's vector arm fails exactly one of the two
//! tests and leaves the other green -- so both are needed, and neither is redundant.

use super::gpu_fuzz::{
    air_header, execute_generated, execute_generated_over, kernel_metadata,
    kernel_metadata_written, Xorshift, THREADS,
};

/// One record member. Scalars, vectors, arrays and arrays of vectors all appear, because the raw
/// and typed paths diverge most at a member that is not word-sized or not word-aligned -- and a
/// VECTOR is where they diverge for a second, independent reason: `<3 x i32>` stores 12 bytes and
/// OCCUPIES 16, so its store size, its allocation size and its alignment are three different
/// numbers and only the data layout says which is which.
#[derive(Clone, Copy)]
struct Member {
    /// The member's LLVM type as written.
    spelling: &'static str,
    /// The leaf scalar's type, what a load of one lane produces.
    leaf: &'static str,
    /// Bytes per leaf scalar.
    leaf_size: u64,
    /// Elements in the array, or 0 if the member is not an array. Only a non-zero `len` takes a
    /// DYNAMIC index, which is the point: a constant index folds to a word number long before
    /// `passes/access` sees it, and then the two derivations are the same instruction and the test
    /// proves nothing.
    len: u64,
    /// Lanes in one element; 1 when the element is a scalar.
    lanes: u64,
    /// Bytes from one element to the next -- the element's ALLOCATION size, which for `<3 x i32>`
    /// is 16 and not the 12 bytes it stores. This is the number the two derivations must agree on.
    stride: u64,
    /// LLVM alignment of the whole member, from the module's datalayout vector entries.
    align: u64,
}

/// `<N x T>` alignment comes from the `v<bits>` entries of the Metal data layout in
/// [`super::gpu_fuzz::DATA_LAYOUT`], NOT from the element: `<3 x i16>` is 48 bits, so `v48:64:64`
/// aligns it to 8 and pads its 6 stored bytes out to a stride of 8.
const MEMBERS: [Member; 17] = [
    Member {
        spelling: "i32",
        leaf: "i32",
        leaf_size: 4,
        len: 0,
        lanes: 1,
        stride: 4,
        align: 4,
    },
    Member {
        spelling: "i16",
        leaf: "i16",
        leaf_size: 2,
        len: 0,
        lanes: 1,
        stride: 2,
        align: 2,
    },
    Member {
        spelling: "i8",
        leaf: "i8",
        leaf_size: 1,
        len: 0,
        lanes: 1,
        stride: 1,
        align: 1,
    },
    Member {
        spelling: "i64",
        leaf: "i64",
        leaf_size: 8,
        len: 0,
        lanes: 1,
        stride: 8,
        align: 8,
    },
    // Bare vectors. Never the indexed member -- they take no dynamic index -- but their padding is
    // what moves every member after them, and getting that wrong is what the test catches.
    Member {
        spelling: "<2 x i32>",
        leaf: "i32",
        leaf_size: 4,
        len: 0,
        lanes: 2,
        stride: 8,
        align: 8,
    },
    Member {
        spelling: "<3 x i32>",
        leaf: "i32",
        leaf_size: 4,
        len: 0,
        lanes: 3,
        stride: 16,
        align: 16,
    },
    Member {
        spelling: "<3 x i16>",
        leaf: "i16",
        leaf_size: 2,
        len: 0,
        lanes: 3,
        stride: 8,
        align: 8,
    },
    Member {
        spelling: "<4 x i8>",
        leaf: "i8",
        leaf_size: 1,
        len: 0,
        lanes: 4,
        stride: 4,
        align: 4,
    },
    // Arrays of scalars.
    Member {
        spelling: "[3 x i32]",
        leaf: "i32",
        leaf_size: 4,
        len: 3,
        lanes: 1,
        stride: 4,
        align: 4,
    },
    Member {
        spelling: "[5 x i16]",
        leaf: "i16",
        leaf_size: 2,
        len: 5,
        lanes: 1,
        stride: 2,
        align: 2,
    },
    Member {
        spelling: "[7 x i8]",
        leaf: "i8",
        leaf_size: 1,
        len: 7,
        lanes: 1,
        stride: 1,
        align: 1,
    },
    Member {
        spelling: "[2 x i64]",
        leaf: "i64",
        leaf_size: 8,
        len: 2,
        lanes: 1,
        stride: 8,
        align: 8,
    },
    // Arrays of vectors: a dynamic element index over a stride that is not the stored size.
    Member {
        spelling: "[2 x <3 x i32>]",
        leaf: "i32",
        leaf_size: 4,
        len: 2,
        lanes: 3,
        stride: 16,
        align: 16,
    },
    Member {
        spelling: "[3 x <2 x i32>]",
        leaf: "i32",
        leaf_size: 4,
        len: 3,
        lanes: 2,
        stride: 8,
        align: 8,
    },
    Member {
        spelling: "[2 x <3 x i16>]",
        leaf: "i16",
        leaf_size: 2,
        len: 2,
        lanes: 3,
        stride: 8,
        align: 8,
    },
    Member {
        spelling: "[3 x <2 x i16>]",
        leaf: "i16",
        leaf_size: 2,
        len: 3,
        lanes: 2,
        stride: 4,
        align: 4,
    },
    Member {
        spelling: "[2 x <4 x i8>]",
        leaf: "i8",
        leaf_size: 1,
        len: 2,
        lanes: 4,
        stride: 4,
        align: 4,
    },
];

impl Member {
    /// The whole member's size: an array is its stride times its length, anything else is one
    /// element -- and for a vector that element is already padded out to its allocation size.
    fn size(&self) -> u64 {
        self.stride * self.len.max(1)
    }

    /// Whether a runtime element index can reach into this member.
    fn indexed(&self) -> bool {
        self.len > 0
    }
}

/// The record every generated kernel indexes, plus its computed layout.
struct Record {
    members: Vec<Member>,
    offsets: Vec<u64>,
    /// Padded up to the struct's own alignment, which is what makes it the RECORD STRIDE and not
    /// just the sum of the members -- getting that wrong moves every thread but the first.
    size: u64,
}

impl Record {
    /// Lay the members out by the ordinary LLVM rules the module's datalayout string pins: each
    /// member starts at the next multiple of its own alignment, the struct's alignment is the
    /// largest member alignment, and the struct's size is padded up to that.
    fn generate(seed: u64, count: usize) -> Self {
        let mut rng = Xorshift(seed | 1);
        let members: Vec<Member> = (0..count)
            .map(|_| MEMBERS[rng.below(MEMBERS.len())])
            .collect();
        let mut offsets = Vec::with_capacity(count);
        let mut cursor = 0u64;
        let mut align = 1u64;
        for member in &members {
            cursor = cursor.div_ceil(member.align) * member.align;
            offsets.push(cursor);
            cursor += member.size();
            align = align.max(member.align);
        }
        let size = cursor.div_ceil(align) * align;
        Record {
            members,
            offsets,
            size,
        }
    }

    fn spelling(&self) -> String {
        format!(
            "{{ {} }}",
            self.members
                .iter()
                .map(|member| member.spelling)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    /// Byte offset of lane `lane` of element `leaf` of member `member` in record `record`.
    ///
    /// The element steps by the member's STRIDE and the lane by the leaf size, and for a padded
    /// vector those are not the same multiple: in `[2 x <3 x i32>]` element 1 starts at byte 16,
    /// not at byte 12 where element 0's lanes ended.
    fn byte_offset(&self, record: u64, member: usize, leaf: u64, lane: u64) -> u64 {
        let spec = self.members[member];
        record * self.size + self.offsets[member] + leaf * spec.stride + lane * spec.leaf_size
    }
}

/// The buffer's contents: a pattern with no repeating byte within any 8-byte window, so a load
/// that lands one byte off cannot coincidentally read the right value.
fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| ((index as u64).wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

/// What a `size`-byte little-endian load at `offset` must produce, zero-extended to 32 bits.
fn expected(bytes: &[u8], offset: u64, size: u64) -> u32 {
    let mut value = 0u64;
    for index in 0..size {
        value |= (bytes[(offset + index) as usize] as u64) << (8 * index);
    }
    // i64 members are read as their low word; the shader truncates the same way.
    (value & 0xffff_ffff) as u32
}

/// Which member each generated kernel reads. The member index is structural -- you cannot select a
/// struct member at runtime -- so it is fixed per kernel; the RECORD index and the LEAF index are
/// both dynamic, which is what stops the two derivations collapsing into one instruction.
struct Plan {
    record: Record,
    member: usize,
    /// Per thread, which element of that member to read. Supplied through a buffer, so the shader
    /// cannot see it as a constant.
    leaves: Vec<u64>,
    /// Which lane of that element. A vector lane index is structural in the same way a member index
    /// is -- indexing a vector by a runtime value is a different lowering entirely -- so it is one
    /// constant per kernel, chosen by the seed. The chain is still dynamic through `%rec` and
    /// `%leaf`, which is what keeps the two derivations from folding into one instruction.
    lane: u64,
}

impl Plan {
    fn generate(seed: u64, members: usize) -> Option<Self> {
        let record = Record::generate(seed, members);
        let mut rng = Xorshift(seed.wrapping_mul(0x2545f4914f6cdd1d) | 1);
        // Only an array member takes a runtime leaf index. A record that generated none has
        // nothing to test, so it is skipped and counted rather than quietly passing.
        let indexed: Vec<usize> = (0..record.members.len())
            .filter(|&index| record.members[index].indexed())
            .collect();
        if indexed.is_empty() {
            return None;
        }
        let member = indexed[rng.below(indexed.len())];
        let spec = record.members[member];
        let leaves = (0..THREADS as usize)
            .map(|_| rng.below(spec.len as usize) as u64)
            .collect();
        let lane = rng.below(spec.lanes as usize) as u64;
        Some(Plan {
            record,
            member,
            leaves,
            lane,
        })
    }

    /// The typed chain's trailing indices: the element index, always dynamic, then the lane index
    /// when the element is a vector.
    fn typed_tail(&self) -> String {
        let spec = self.record.members[self.member];
        if spec.lanes > 1 {
            format!(", i64 %leaf, i32 {}", self.lane)
        } else {
            ", i64 %leaf".to_string()
        }
    }

    /// The two derivations, both with runtime indices:
    ///
    /// - typed: `getelementptr %Rec, ptr %input, i64 %rec, i32 M, i64 %leaf`
    /// - raw:   `getelementptr i8, ptr %input, i64 (%rec * sizeof(Rec) + off(M) + %leaf * size)`
    ///
    /// The translator has to make those name the same byte, which means deriving the record
    /// stride, the member offset and the element stride twice by two different routes.
    fn air(&self, entry: &str) -> String {
        let spec = self.record.members[self.member];
        let mut out = air_header(entry);
        out.push_str(&format!("%Rec = type {}\n\n", self.record.spelling()));
        out.push_str(&format!(
            "define void @{entry}(ptr addrspace(1) %output, ptr addrspace(1) %input, ptr addrspace(1) %picks, i32 %tid) {{\nentry:\n"
        ));
        out.push_str("  %rec = zext i32 %tid to i64\n");
        out.push_str("  %pp = getelementptr inbounds i32, ptr addrspace(1) %picks, i64 %rec\n");
        out.push_str("  %pick = load i32, ptr addrspace(1) %pp, align 4\n");
        out.push_str("  %leaf = zext i32 %pick to i64\n");
        out.push_str(&format!(
            "  %tp = getelementptr inbounds %Rec, ptr addrspace(1) %input, i64 %rec, i32 {}{}\n",
            self.member,
            self.typed_tail()
        ));
        out.push_str(&format!(
            "  %tv = load {}, ptr addrspace(1) %tp, align {}\n",
            spec.leaf, spec.leaf_size
        ));
        out.push_str(&format!("  %b0 = mul i64 %rec, {}\n", self.record.size));
        out.push_str(&format!(
            "  %b1 = add i64 %b0, {}\n",
            self.record.offsets[self.member]
        ));
        out.push_str(&format!("  %b2 = mul i64 %leaf, {}\n", spec.stride));
        out.push_str(&format!(
            "  %b3 = add i64 %b1, {}\n",
            self.lane * spec.leaf_size
        ));
        out.push_str("  %b4 = add i64 %b3, %b2\n");
        out.push_str("  %rp = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %b4\n");
        out.push_str(&format!(
            "  %rv = load {}, ptr addrspace(1) %rp, align {}\n",
            spec.leaf, spec.leaf_size
        ));
        for (name, value) in [("wt", "%tv"), ("wr", "%rv")] {
            out.push_str(&match spec.leaf {
                "i32" => format!("  %{name} = add i32 {value}, 0\n"),
                "i64" => format!("  %{name} = trunc i64 {value} to i32\n"),
                narrow => format!("  %{name} = zext {narrow} {value} to i32\n"),
            });
        }
        out.push_str("  %slot = shl i32 %tid, 1\n");
        out.push_str("  %slot64 = zext i32 %slot to i64\n");
        out.push_str("  %o0 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %slot64\n");
        out.push_str("  store i32 %wt, ptr addrspace(1) %o0, align 4\n");
        out.push_str("  %o1i = add i64 %slot64, 1\n");
        out.push_str("  %o1 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %o1i\n");
        out.push_str("  store i32 %wr, ptr addrspace(1) %o1, align 4\n");
        out.push_str("  ret void\n}\n\n");
        out.push_str(&kernel_metadata(entry, 2));
        out
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_typed_access_chain_and_a_raw_byte_offset_name_the_same_word() {
    let mut failures: Vec<String> = Vec::new();
    let mut executed = 0usize;
    let mut skipped = 0usize;
    for seed in 1u64..=12 {
        for members in 2usize..=5 {
            let Some(plan) = Plan::generate(seed.wrapping_mul(0x9e3779b97f4a7c15), members) else {
                skipped += 1;
                continue;
            };
            let bytes = pattern(plan.record.size as usize * THREADS as usize + 64);
            let words: Vec<u32> = bytes
                .chunks(4)
                .map(|word| {
                    let mut full = [0u8; 4];
                    full[..word.len()].copy_from_slice(word);
                    u32::from_le_bytes(full)
                })
                .collect();
            let picks: Vec<u32> = plan.leaves.iter().map(|leaf| *leaf as u32).collect();
            let entry = format!("access_s{seed}_m{members}");
            let run = match execute_generated(&entry, &plan.air(&entry), &[words, picks], 2) {
                Ok(run) => run,
                Err(error) => {
                    failures.push(format!(
                        "seed={seed} members={members} ({}): {error}",
                        plan.record.spelling()
                    ));
                    continue;
                }
            };
            executed += 1;
            let spec = plan.record.members[plan.member];
            for thread in 0..THREADS as usize {
                let leaf = plan.leaves[thread];
                let offset = plan
                    .record
                    .byte_offset(thread as u64, plan.member, leaf, plan.lane);
                let want = expected(&bytes, offset, spec.leaf_size);
                let (typed, raw) = (run.words[thread * 2], run.words[thread * 2 + 1]);
                if typed != want || raw != want {
                    failures.push(format!(
                        "seed={seed} members={members} {} thread={thread} member {} ({}) leaf \
                         {leaf} lane {} at byte {offset}: typed chain gave {typed}, raw offset \
                         gave {raw}, the layout says {want}",
                        plan.record.spelling(),
                        plan.member,
                        spec.spelling,
                        plan.lane
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} divergences over {executed} generated layouts:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        executed > 0,
        "no generated layout executed at all ({skipped} had no array member)"
    );
}

/// The store side of the same question, and a sharper one.
///
/// A load that lands at the wrong offset gives a wrong answer, which any comparison catches. A
/// STORE that lands at the wrong offset, or that writes more bytes than the member holds, corrupts
/// a NEIGHBOUR -- and the value it wrote is still right, so nothing about the written leaf reveals
/// it. `vector_subword.rs` lowers a narrow store as a read-modify-write of the surrounding word,
/// which is exactly the operation that clobbers by one byte, and `scalar_store.rs` repairs the
/// pointer carrier. So this test checks the whole buffer, not the written word.
impl Plan {
    /// Each thread writes its own record's chosen leaf and nothing else. The value is derived from
    /// `tid` so no two threads write the same bytes, and it differs from the fill in every byte.
    fn store_air(&self, entry: &str) -> String {
        let spec = self.record.members[self.member];
        let mut out = air_header(entry);
        out.push_str(&format!("%Rec = type {}\n\n", self.record.spelling()));
        out.push_str(&format!(
            "define void @{entry}(ptr addrspace(1) %output, ptr addrspace(1) %picks, i32 %tid) {{\nentry:\n"
        ));
        out.push_str("  %rec = zext i32 %tid to i64\n");
        out.push_str("  %pp = getelementptr inbounds i32, ptr addrspace(1) %picks, i64 %rec\n");
        out.push_str("  %pick = load i32, ptr addrspace(1) %pp, align 4\n");
        out.push_str("  %leaf = zext i32 %pick to i64\n");
        out.push_str("  %v32 = xor i32 %tid, -1515870811\n");
        out.push_str(&match spec.leaf {
            "i32" => "  %val = add i32 %v32, 0\n".to_string(),
            "i64" => "  %val = sext i32 %v32 to i64\n".to_string(),
            narrow => format!("  %val = trunc i32 %v32 to {narrow}\n"),
        });
        out.push_str(&format!(
            "  %tp = getelementptr inbounds %Rec, ptr addrspace(1) %output, i64 %rec, i32 {}{}\n",
            self.member,
            self.typed_tail()
        ));
        out.push_str(&format!(
            "  store {} %val, ptr addrspace(1) %tp, align {}\n",
            spec.leaf, spec.leaf_size
        ));
        out.push_str("  ret void\n}\n\n");
        out.push_str(&kernel_metadata_written(entry, 1));
        out
    }
}

/// The bytes thread `tid` writes for a leaf of `size` bytes, little-endian.
fn stored_bytes(tid: u32, size: u64) -> Vec<u8> {
    let word = tid ^ 0xa5a5_a5a5;
    // i64 members take a sign extension of the same 32-bit value, matching the shader's `sext`.
    let wide = (word as i32) as i64 as u64;
    (0..size).map(|index| (wide >> (8 * index)) as u8).collect()
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_stored_leaf_lands_on_its_own_bytes_and_no_others() {
    let mut failures: Vec<String> = Vec::new();
    let mut executed = 0usize;
    for seed in 1u64..=12 {
        for members in 2usize..=5 {
            let Some(plan) = Plan::generate(seed.wrapping_mul(0x9e3779b97f4a7c15), members) else {
                continue;
            };
            let spec = plan.record.members[plan.member];
            let initial = pattern(plan.record.size as usize * THREADS as usize);
            let mut want = initial.clone();
            for thread in 0..THREADS as usize {
                let offset = plan.record.byte_offset(
                    thread as u64,
                    plan.member,
                    plan.leaves[thread],
                    plan.lane,
                );
                for (index, byte) in stored_bytes(thread as u32, spec.leaf_size)
                    .iter()
                    .enumerate()
                {
                    want[offset as usize + index] = *byte;
                }
            }
            let picks: Vec<u32> = plan.leaves.iter().map(|leaf| *leaf as u32).collect();
            let entry = format!("store_s{seed}_m{members}");
            let run =
                match execute_generated_over(&entry, &plan.store_air(&entry), &[picks], &initial) {
                    Ok(run) => run,
                    Err(error) => {
                        failures.push(format!(
                            "seed={seed} members={members} ({}): {error}",
                            plan.record.spelling()
                        ));
                        continue;
                    }
                };
            executed += 1;
            let got: Vec<u8> = run
                .words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .take(want.len())
                .collect();
            for (offset, (left, right)) in got.iter().zip(want.iter()).enumerate() {
                if left != right {
                    let record = offset / plan.record.size as usize;
                    let within = offset % plan.record.size as usize;
                    let wrote = plan.record.byte_offset(
                        record as u64,
                        plan.member,
                        plan.leaves[record.min(THREADS as usize - 1)],
                        plan.lane,
                    );
                    failures.push(format!(
                        "seed={seed} members={members} {} byte {offset} (record {record} \
                         offset {within}): got {left:#04x}, wanted {right:#04x}; that thread \
                         stored member {} ({}) lane {} at byte {wrote}",
                        plan.record.spelling(),
                        plan.member,
                        spec.spelling,
                        plan.lane
                    ));
                }
            }
        }
    }
    // One byte wrong is one report per byte, and a clobbered neighbour is contiguous; showing the
    // first few is enough to locate it and the count says how far it spread.
    failures.truncate(12);
    assert!(
        failures.is_empty(),
        "{} byte differences over {executed} generated layouts:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(executed > 0, "no generated layout executed at all");
}

/// The cross-thread version of the store question.
///
/// `passes/access/dynamic_reinterpret.rs` rewrites a sub-word view of a raw word buffer into an
/// address on the containing `uint` and a shift. For a STORE that is a read-modify-write, and the
/// word is in a `StorageBuffer` -- shared by every thread in the dispatch. The lowering therefore
/// has to clear and set the lane ATOMICALLY; a plain load/and/or/store loses one of any two writes
/// that land in one word.
///
/// This module is the shape that reaches that rewrite, reduced from the corpus hosts that do (a
/// Unity VFX `particle_data` update): a device base pointer parked in a local struct and read back
/// out inside a helper, a WIDE store through it at a runtime byte offset -- which the raw-byte
/// rewrite decomposes into per-byte stores -- and a wide load of the same buffer, which is what
/// keeps the buffer on the `{ RuntimeArray<uint> }` word view instead of giving it a native `uchar`
/// descriptor. Without any one of those three it lowers some other, already-safe way.
///
/// **The offsets are chosen so the two threads collide at the SAME STEP, and that is the whole
/// difficulty.** A 12-byte store at a 12-byte stride also shares words -- thread `t`'s last two
/// bytes and thread `t+1`'s first two -- but those are byte number 10 and byte number 0 of their
/// respective decompositions. The dispatch is one SIMD group, so thread `t` does not read the word
/// until step 10, by which time thread `t+1`'s step-0 store is already in it, and the lost write
/// hides. A 2-byte store at a 2-byte stride puts thread 0's byte 1 and thread 1's byte 3 in one
/// word at step 0 of both, and then a non-atomic read-modify-write does lose one.
const RACE_BASE: u32 = 1;
const RACE_STRIDE: u32 = 2;
/// 32 threads x `<4 x float>` -- the wide load reads a whole 16-byte element per thread, so the
/// buffer has to be that big for the load to stay in bounds.
const RACE_BYTES: usize = THREADS as usize * 16;
/// The high bits every thread stores, so the value differs from the fill in both of its bytes.
const RACE_TAG: u32 = 0xc000;

fn race_air(entry: &str) -> String {
    let mut out = air_header(entry);
    out.push_str("%S = type { ptr addrspace(1), ptr addrspace(1) }\n\n");
    out.push_str(
        "define internal fastcc void @set_lane(ptr %0, i32 %1, i16 %2) unnamed_addr align 2 {\n\
         \x20 %4 = getelementptr inbounds %S, ptr %0, i64 0, i32 1\n\
         \x20 %5 = load ptr addrspace(1), ptr %4, align 8\n\
         \x20 %6 = sext i32 %1 to i64\n\
         \x20 %7 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %6\n\
         \x20 store i16 %2, ptr addrspace(1) %7, align 2\n\
         \x20 ret void\n}\n\n",
    );
    out.push_str(&format!(
        "define void @{entry}(ptr addrspace(1) %out, ptr addrspace(1) %hdr, i32 %tid) {{\nentry:\n"
    ));
    out.push_str("  %s = alloca %S, align 8\n");
    out.push_str("  %h = getelementptr inbounds %S, ptr %s, i64 0, i32 0\n");
    out.push_str("  store ptr addrspace(1) %hdr, ptr %h, align 8\n");
    out.push_str("  %d = getelementptr inbounds %S, ptr %s, i64 0, i32 1\n");
    out.push_str("  store ptr addrspace(1) %out, ptr %d, align 8\n");
    out.push_str("  %base = load i32, ptr addrspace(1) %hdr, align 4\n");
    out.push_str(&format!("  %step = mul i32 %tid, {RACE_STRIDE}\n"));
    out.push_str("  %off = add i32 %base, %step\n");
    out.push_str("  %li = zext i32 %tid to i64\n");
    out.push_str("  %lp = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %li\n");
    out.push_str("  %lv = load <4 x float>, ptr addrspace(1) %lp, align 16\n");
    out.push_str("  %narrow = trunc i32 %tid to i16\n");
    out.push_str(&format!("  %value = or i16 %narrow, {}\n", RACE_TAG as i16));
    out.push_str("  call fastcc void @set_lane(ptr %s, i32 %off, i16 %value)\n");
    out.push_str("  ret void\n}\n\n");
    out.push_str(&kernel_metadata_written(entry, 1));
    out
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn one_threads_subword_store_does_not_erase_another_threads() {
    let entry = "subword_race";
    let initial = pattern(RACE_BYTES);
    let mut want = initial.clone();
    for thread in 0..THREADS {
        let offset = (RACE_BASE + thread * RACE_STRIDE) as usize;
        let value = ((thread | RACE_TAG) & 0xffff) as u16;
        want[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    let run = execute_generated_over(entry, &race_air(entry), &[vec![RACE_BASE]], &initial)
        .expect("the subword-race module translates and runs");
    let got: Vec<u8> = run
        .words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take(want.len())
        .collect();
    let wrong: Vec<usize> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        wrong.is_empty(),
        "{} of {} bytes differ; first ten at {:?}. A byte that is wrong inside another thread's \
         two is a LOST lane update: the thread that owned it wrote it, and a neighbour's \
         read-modify-write of the shared word put the pre-store value back.",
        wrong.len(),
        want.len(),
        &wrong[..wrong.len().min(10)]
    );
}
