//! 10Kbit Vector Width Constants
//!
//! The original configuration: 10,000-bit vectors in 157 u64 words.
//! Compact, cache-friendly, well-suited for memory-constrained environments.
//!
//! See `VECTOR_WIDTH.md` for full comparison with the 16K variant.

// ============================================================================
// VECTOR DIMENSIONS
// ============================================================================

/// Number of logical bits in the vector
pub const VECTOR_BITS: usize = 10_000;

/// Number of u64 words: ceil(10000/64) = 157
pub const VECTOR_WORDS: usize = (VECTOR_BITS + 63) / 64; // 157

/// Raw bytes per vector: 157 × 8 = 1,256
pub const VECTOR_BYTES: usize = VECTOR_WORDS * 8; // 1256

/// Padded words for 64-byte alignment: ceil(157/8)*8 = 160
pub const PADDED_VECTOR_WORDS: usize = (VECTOR_WORDS + 7) & !7; // 160

/// Padded bytes for Arrow FixedSizeBinary: 160 × 8 = 1,280
pub const PADDED_VECTOR_BYTES: usize = PADDED_VECTOR_WORDS * 8; // 1280

/// Bits used in the last word (10000 - 156×64 = 16)
pub const LAST_WORD_BITS: usize = VECTOR_BITS - (VECTOR_WORDS - 1) * 64; // 16

/// Mask for the last word
pub const LAST_WORD_MASK: u64 = (1u64 << LAST_WORD_BITS) - 1;

/// Whether the last word is fully used (false for 10K)
pub const LAST_WORD_FULL: bool = false;

// ============================================================================
// STATISTICAL CONSTANTS (Hamming distribution)
// ============================================================================

/// Expected Hamming distance between two random vectors = n/2
pub const EXPECTED_RANDOM_DISTANCE: f64 = VECTOR_BITS as f64 / 2.0; // 5000.0

/// Standard deviation: σ = √(n/4) = √2500 = 50
pub const HAMMING_STD_DEV: f64 = 50.0;

/// One standard deviation threshold
pub const ONE_SIGMA: u32 = 50;

/// Two standard deviations
pub const TWO_SIGMA: u32 = 100;

/// Three standard deviations (99.7% confidence)
pub const THREE_SIGMA: u32 = 150;

// ============================================================================
// NEURAL TREE BLOCK LAYOUT
// ============================================================================

/// Words per multi-resolution block
pub const WORDS_PER_BLOCK: usize = 16;

/// Number of blocks: ceil(157/16) = 10
pub const NUM_BLOCKS: usize = (VECTOR_WORDS + WORDS_PER_BLOCK - 1) / WORDS_PER_BLOCK; // 10

/// Bits per block (all except possibly last)
pub const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64; // 1024

/// Words in the last block (157 - 9×16 = 13)
pub const LAST_BLOCK_WORDS: usize = VECTOR_WORDS - (NUM_BLOCKS - 1) * WORDS_PER_BLOCK; // 13

/// Bits in the last block (13 × 64 = 832)
pub const LAST_BLOCK_BITS: usize = LAST_BLOCK_WORDS * 64; // 832

/// Blocks per crystal dimension (5D → 2 blocks each)
pub const BLOCKS_PER_CRYSTAL_DIM: usize = 2;

// ============================================================================
// SIMD LAYOUT
// ============================================================================

/// AVX-512 registers needed (512 bits = 8 u64): ceil(157/8) = 20
pub const AVX512_ITERATIONS: usize = VECTOR_WORDS / 8; // 19 full
/// AVX-512 remainder words: 157 - 19×8 = 5
pub const AVX512_REMAINDER: usize = VECTOR_WORDS - AVX512_ITERATIONS * 8; // 5

/// AVX2 registers needed (256 bits = 4 u64): ceil(157/4) = 40
pub const AVX2_ITERATIONS: usize = VECTOR_WORDS / 4; // 39 full
/// AVX2 remainder words: 157 - 39×4 = 1
pub const AVX2_REMAINDER: usize = VECTOR_WORDS - AVX2_ITERATIONS * 4; // 1

/// NEON registers needed (128 bits = 2 u64): ceil(157/2) = 79
pub const NEON_ITERATIONS: usize = VECTOR_WORDS / 2; // 78 full
/// NEON remainder words: 157 - 78×2 = 1
pub const NEON_REMAINDER: usize = VECTOR_WORDS - NEON_ITERATIONS * 2; // 1

// ============================================================================
// BELICHTUNGSMESSER SAMPLE POINTS
// ============================================================================

/// Strategic 7-point sample indices for quick distance estimation.
/// Prime-spaced across 157 words.
pub const SAMPLE_POINTS: [usize; 7] = [0, 23, 47, 78, 101, 131, 155];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_10k_constants() {
        assert_eq!(VECTOR_BITS, 10_000);
        assert_eq!(VECTOR_WORDS, 157);
        assert_eq!(VECTOR_BYTES, 1256);
        assert_eq!(PADDED_VECTOR_WORDS, 160);
        assert_eq!(PADDED_VECTOR_BYTES, 1280);
        assert_eq!(LAST_WORD_BITS, 16);
        assert!(!LAST_WORD_FULL);
        assert_eq!(NUM_BLOCKS, 10);
        assert_eq!(LAST_BLOCK_WORDS, 13);
        assert_eq!(ONE_SIGMA, 50);
        assert_eq!(TWO_SIGMA, 100);
        assert_eq!(THREE_SIGMA, 150);
    }

    #[test]
    fn test_10k_simd_layout() {
        // AVX-512: 19 full iterations + 5 remainder
        assert_eq!(AVX512_ITERATIONS, 19);
        assert_eq!(AVX512_REMAINDER, 5);
        // AVX2: 39 full + 1 remainder
        assert_eq!(AVX2_ITERATIONS, 39);
        assert_eq!(AVX2_REMAINDER, 1);
    }

    #[test]
    fn test_10k_sample_points_in_range() {
        for &p in &SAMPLE_POINTS {
            assert!(p < VECTOR_WORDS, "Sample point {} out of range", p);
        }
    }
}
