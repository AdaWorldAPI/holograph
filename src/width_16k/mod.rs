//! 16Kbit (2^14) Vector Width Constants + ANI/NARS/RL Schema Markers
//!
//! The power-of-2 configuration: 16,384-bit vectors in exactly 256 u64 words.
//! Perfect SIMD alignment, zero padding waste, σ = 64 = exactly one word.
//!
//! ## Advantages over 10K
//!
//! - 256 words = 2^8 → all SIMD widths divide evenly (AVX-512, AVX2, NEON)
//! - σ = 64 = one u64 word → integer-exact sigma arithmetic
//! - 16 uniform blocks of 1024 bits → no short last block
//! - Optional 3-block schema sidecar for ANI/NARS/RL markers
//!
//! See `VECTOR_WIDTH.md` for full comparison.

pub mod schema;
pub mod search;
pub mod compat;
pub mod xor_bubble;
#[cfg(test)]
mod demo;

// ============================================================================
// VECTOR DIMENSIONS
// ============================================================================

/// Number of logical bits in the vector (2^14)
pub const VECTOR_BITS: usize = 16_384;

/// Number of u64 words: 16384/64 = 256 (exact, no remainder)
pub const VECTOR_WORDS: usize = VECTOR_BITS / 64; // 256

/// Raw bytes per vector: 256 × 8 = 2,048
pub const VECTOR_BYTES: usize = VECTOR_WORDS * 8; // 2048

/// Padded words — same as raw (already 64-byte aligned: 256 × 8 = 2048 = 32 × 64)
pub const PADDED_VECTOR_WORDS: usize = VECTOR_WORDS; // 256

/// Padded bytes — same as raw (2048 is already a multiple of 64)
pub const PADDED_VECTOR_BYTES: usize = VECTOR_BYTES; // 2048

/// Bits in the last word — all 64 used (16384 / 64 = 256 exactly)
pub const LAST_WORD_BITS: usize = 64;

/// Mask for the last word — all bits (no masking needed)
pub const LAST_WORD_MASK: u64 = u64::MAX;

/// Whether the last word is fully used (true for 16K)
pub const LAST_WORD_FULL: bool = true;

// ============================================================================
// STATISTICAL CONSTANTS (Hamming distribution)
// ============================================================================

/// Expected Hamming distance between two random vectors = n/2
pub const EXPECTED_RANDOM_DISTANCE: f64 = VECTOR_BITS as f64 / 2.0; // 8192.0

/// Standard deviation: σ = √(n/4) = √4096 = 64 (exactly one u64 word!)
pub const HAMMING_STD_DEV: f64 = 64.0;

/// One standard deviation threshold
pub const ONE_SIGMA: u32 = 64;

/// Two standard deviations
pub const TWO_SIGMA: u32 = 128;

/// Three standard deviations (99.7% confidence)
pub const THREE_SIGMA: u32 = 192;

// ============================================================================
// NEURAL TREE BLOCK LAYOUT
// ============================================================================

/// Words per multi-resolution block
pub const WORDS_PER_BLOCK: usize = 16;

/// Number of blocks: 256/16 = 16 (exact, no remainder)
pub const NUM_BLOCKS: usize = VECTOR_WORDS / WORDS_PER_BLOCK; // 16

/// Bits per block (all blocks equal)
pub const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64; // 1024

/// Words in the last block (same as all others: 16)
pub const LAST_BLOCK_WORDS: usize = WORDS_PER_BLOCK; // 16

/// Bits in the last block (same as all others: 1024)
pub const LAST_BLOCK_BITS: usize = BITS_PER_BLOCK; // 1024

/// Blocks per crystal dimension
///
/// With 16 blocks and 5 crystal dimensions:
/// - Semantic blocks: 0..12 (13 blocks = 13,312 bits)
/// - Schema blocks: 13..15 (3 blocks = 3,072 bits)
/// - Crystal mapping: 5D × ~2.6 blocks from semantic region
///
/// Alternatively in all-semantic mode:
/// - 5D × 3 blocks = 15 blocks (leave 1 for global metadata)
/// - 8D × 2 blocks = 16 blocks (higher-dimensional crystal)
pub const BLOCKS_PER_CRYSTAL_DIM: usize = 3;

/// Number of semantic blocks (when using schema sidecar)
pub const SEMANTIC_BLOCKS: usize = 13;

/// First schema block index
pub const SCHEMA_BLOCK_START: usize = 13;

/// Number of schema blocks
pub const SCHEMA_BLOCK_COUNT: usize = 3;

// ============================================================================
// SIMD LAYOUT — All zero remainder!
// ============================================================================

/// AVX-512 registers needed (512 bits = 8 u64): 256/8 = 32 (exact)
pub const AVX512_ITERATIONS: usize = VECTOR_WORDS / 8; // 32
/// AVX-512 remainder words: 0
pub const AVX512_REMAINDER: usize = 0;

/// AVX2 registers needed (256 bits = 4 u64): 256/4 = 64 (exact)
pub const AVX2_ITERATIONS: usize = VECTOR_WORDS / 4; // 64
/// AVX2 remainder words: 0
pub const AVX2_REMAINDER: usize = 0;

/// NEON registers needed (128 bits = 2 u64): 256/2 = 128 (exact)
pub const NEON_ITERATIONS: usize = VECTOR_WORDS / 2; // 128
/// NEON remainder words: 0
pub const NEON_REMAINDER: usize = 0;

// ============================================================================
// BELICHTUNGSMESSER SAMPLE POINTS
// ============================================================================

/// Strategic 7-point sample indices for quick distance estimation.
/// Evenly distributed across 256 words with prime-ish spacing.
pub const SAMPLE_POINTS: [usize; 7] = [0, 37, 73, 127, 163, 211, 251];

// ============================================================================
// SCHEMA SIDECAR OFFSETS (bit positions within the vector)
// ============================================================================

/// Block 13 start bit (node/edge type markers)
pub const SCHEMA_NODE_EDGE_START: usize = SCHEMA_BLOCK_START * BITS_PER_BLOCK; // 13312

/// Block 14 start bit (RL/temporal state)
pub const SCHEMA_RL_STATE_START: usize = (SCHEMA_BLOCK_START + 1) * BITS_PER_BLOCK; // 14336

/// Block 15 start bit (traversal/graph cache)
pub const SCHEMA_GRAPH_CACHE_START: usize = (SCHEMA_BLOCK_START + 2) * BITS_PER_BLOCK; // 15360

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_16k_constants() {
        assert_eq!(VECTOR_BITS, 16_384);
        assert_eq!(VECTOR_WORDS, 256);
        assert_eq!(VECTOR_BYTES, 2048);
        assert_eq!(PADDED_VECTOR_WORDS, 256);
        assert_eq!(PADDED_VECTOR_BYTES, 2048);
        assert_eq!(LAST_WORD_BITS, 64);
        assert!(LAST_WORD_FULL);
        assert_eq!(NUM_BLOCKS, 16);
        assert_eq!(LAST_BLOCK_WORDS, 16);
    }

    #[test]
    fn test_16k_sigma_is_one_word() {
        // The magic property: σ = 64 = exactly one u64 word
        assert_eq!(ONE_SIGMA, 64);
        assert_eq!(ONE_SIGMA as usize, 64); // bits in one word
        assert_eq!(TWO_SIGMA, 128);
        assert_eq!(THREE_SIGMA, 192);
    }

    #[test]
    fn test_16k_perfect_alignment() {
        // All SIMD widths divide evenly
        assert_eq!(VECTOR_WORDS % 8, 0, "AVX-512: 8 words per reg");
        assert_eq!(VECTOR_WORDS % 4, 0, "AVX2: 4 words per reg");
        assert_eq!(VECTOR_WORDS % 2, 0, "NEON: 2 words per reg");

        // Zero remainders
        assert_eq!(AVX512_REMAINDER, 0);
        assert_eq!(AVX2_REMAINDER, 0);
        assert_eq!(NEON_REMAINDER, 0);

        // Byte count is cache-line aligned
        assert_eq!(VECTOR_BYTES % 64, 0);
    }

    #[test]
    fn test_16k_uniform_blocks() {
        assert_eq!(NUM_BLOCKS * WORDS_PER_BLOCK, VECTOR_WORDS);
        assert_eq!(LAST_BLOCK_WORDS, WORDS_PER_BLOCK); // All blocks equal!
        assert_eq!(LAST_BLOCK_BITS, BITS_PER_BLOCK);
    }

    #[test]
    fn test_16k_schema_offsets() {
        assert_eq!(SCHEMA_NODE_EDGE_START, 13312);
        assert_eq!(SCHEMA_RL_STATE_START, 14336);
        assert_eq!(SCHEMA_GRAPH_CACHE_START, 15360);
        // Schema region ends at VECTOR_BITS
        assert_eq!(SCHEMA_GRAPH_CACHE_START + BITS_PER_BLOCK, VECTOR_BITS);
    }

    #[test]
    fn test_16k_sample_points_in_range() {
        for &p in &SAMPLE_POINTS {
            assert!(p < VECTOR_WORDS, "Sample point {} out of range", p);
        }
    }

    #[test]
    fn test_16k_semantic_plus_schema() {
        // 13 semantic + 3 schema = 16 total blocks
        assert_eq!(SEMANTIC_BLOCKS + SCHEMA_BLOCK_COUNT, NUM_BLOCKS);
        // Semantic region covers 13,312 bits
        assert_eq!(SEMANTIC_BLOCKS * BITS_PER_BLOCK, 13312);
        // Schema region covers 3,072 bits
        assert_eq!(SCHEMA_BLOCK_COUNT * BITS_PER_BLOCK, 3072);
    }
}
