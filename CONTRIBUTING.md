# Contributing

Thanks for interest in `metal2vulkan`. This document covers the day-to-day developer loop for the
standalone crate.

## Prerequisites

- Rust stable (see `rust-version` in `Cargo.toml`)
- A C++17 compiler for Cargo's statically linked SPIRV-Tools build
- LLVM's shared library for bitcode I/O tests
- A Vulkan ICD or Metal-capable macOS host only for explicit machine-specific authored-case runs

## Repository layout

```text
.
├── src/                  # library + CLI
├── tests/                # integration tests
├── examples/             # cargo examples
├── validation/           # authored-case validation tools (workspace, not published)
├── docs/                 # integration, architecture, reflection, and validation guides
├── scripts/              # developer utilities
├── Cargo.toml            # workspace root + metal2vulkan package
├── LICENSE               # LGPL-3.0-or-later
└── README.md
```

| Path | Role |
|---|---|
| `src/` | Library + CLI (`metal2vulkan`) |
| `tests/` | Integration tests |
| `examples/` | Small runnable examples |
| `validation/` | Authored cases, exact evidence, harvest, and A/B tools (not published) |
| `docs/` | User integration and developer guides (`HOWTO.md`, architecture, reflection, validation) |
| `scripts/` | Developer utilities (mtlb extraction, harvest helpers, grammar regen, …) |

## Development

```sh
# format
cargo fmt --all

# clippy (CI denies warnings); --all-features also covers the `serde` cfg the CLI ships with
cargo clippy --workspace --all-targets --all-features -- -D warnings

# public Rustdoc, including feature-gated APIs (CI denies warnings)
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps

# unit + integration tests (Cargo defaults to the available logical CPU count)
cargo test -p metal2vulkan --all-features

# optional validation package
cargo test -p metal2vulkan-validation --all-features
```

Translation and its Rust tests launch no LLVM/SPIR-V executables. On macOS with Homebrew:

```sh
brew install llvm
cargo test -p metal2vulkan --all-features
```

GPU-free byte A/B is provided by `corpus-ab` in the validation crate.

## Build

```sh
cargo build --release
cargo build --release --features serde
```

## Design rules

1. **Structure and semantics over names.** Translate from AIR/LLVM structure (types, storage classes,
   access chains, AIR metadata ABI). Do **not** special-case individual shader, function, or type
   names observed in a particular workload.
2. **Stable ABI symbols are allowed.** Dispatching on documented `air.*` / `llvm.*` intrinsics is
   fine; prefer structural tests when possible.
3. **No third-party captured shaders in tree.** Reduce regressions to synthetic `.ll` tests.
   Optional **private** harvest writes gitignored shard JSONL under
   `validation/corpus/local/sources/` — see
   [`validation/corpus/README.md`](validation/corpus/README.md). Committed authored manifests and
   hash-identified observations never contain AIR source bodies.
4. **Honest FALLBACK.** Unsupported inputs must return `Err` / CLI `FALLBACK`, never emit
   wrong-but-valid SPIR-V.

### Validation while refactoring

The full developer workflow (synthetic tests → exact binary A/B → authored cases → optional GPU
evidence) is in **[`docs/VALIDATION.md`](docs/VALIDATION.md)**.

Quick anchors:

```sh
# always
cargo test -p metal2vulkan --all-features

# before/after a byte-stable refactor
cp target/release/metal2vulkan ./m2v-old
# … edit …
cargo run -p metal2vulkan-validation --release --bin corpus-ab -- \
  --old ./m2v-old --new target/release/metal2vulkan --canary --expect-no-change

# inspect the bounded authoring/evidence queue
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-next -- --limit 1
cargo run -p metal2vulkan-validation --bin corpus-status

# optional private real-AIR bank (macOS; shard JSONL is gitignored)
cargo run -p metal2vulkan-validation --release --bin corpus-harvest
```

See also [AGENTS.md](AGENTS.md) for the full agent operating guide.

## Pull requests

- One focused change per PR when practical
- Include tests for new behavior or bug fixes
- Update `docs/ARCHITECTURE.md` when the pipeline shape changes
- Update `docs/REFLECTION.md` when the consumer binding contract changes
- Update `docs/HOWTO.md` when the recommended CLI or library integration changes
- Update `CHANGELOG.md` under `[Unreleased]`
