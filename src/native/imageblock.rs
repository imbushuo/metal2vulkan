//! AIR imageblock execution-model constants.

/// The most threads a threadgroup can have, which is how many cells a tile has per cell scale.
///
/// A threadgroup holds at most 1024 threads on every Apple GPU -- `maxTotalThreadsPerThreadgroup`.
/// An entry that stages one cell per thread therefore cannot address a 1025th cell, and one that
/// stages a `k`x`k` block per thread (`imageblock_cell_scale`) cannot address past `k * k` times
/// that.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// The threadgroup-memory budget those cells may occupy.
///
/// Vulkan guarantees `maxComputeSharedMemorySize >= 16384`, so this is the largest tile allocation
/// that is portable to every conformant device. A cell wide enough that 1024 of them would not fit
/// gets however many do.
const TILE_BYTE_BUDGET: u32 = 16384;

/// How many coordinate-addressable cells the shared-tile lowering allocates for a `cell_bytes`-wide
/// cell in a tile that is `cell_scale` cells per thread in each axis.
///
/// This used to be a flat 512, a number that answered neither question: it is below the thread count
/// a tile can have, so it refused five corpus modules whose 8-byte cells name a 32x32 block, and it
/// is stated in cells, so what it costs in threadgroup memory depended on a cell width it never
/// looked at. Both bounds here are external facts about the two APIs rather than a chosen size.
///
/// A cell is never zero bytes in practice; treat a degenerate one as a single byte rather than
/// dividing by zero, and always leave room for the one cell the smallest tile has. A scale wide
/// enough to overflow the thread bound is one the byte budget would have cut long before, so
/// saturate rather than refuse here.
pub(super) fn cell_capacity(cell_bytes: u32, cell_scale: u32) -> u32 {
    MAX_THREADS_PER_THREADGROUP
        .saturating_mul(cell_scale.saturating_mul(cell_scale))
        .min(TILE_BYTE_BUDGET / cell_bytes.max(1))
        .max(1)
}
