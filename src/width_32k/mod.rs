//! 32Kbit (2^15) 3D Holographic Vector — XYZ Superposition Memory
//!
//! Three orthogonal 8K-bit dimensions (X, Y, Z) plus 8K metadata = 512 words.
//! The XOR-bound product space is 8192^3 = 549,755,813,888 (~512 billion)
//! addressable data points in a single 4KB record.
//!
//! ## Dimensions
//!
//! - **X (Content/What)**: Semantic identity — what a concept IS
//! - **Y (Context/Where)**: Situational context — where/when it appears
//! - **Z (Relation/How)**: Relational structure — how it connects (verb/edge)
//! - **M (Metadata)**: 128 words for ANI, NARS, RL, qualia, edges, graph metrics
//!
//! ## Holographic Property
//!
//! ```text
//! Store:   trace = X ⊕ Y ⊕ Z
//! Probe:   X ⊕ Y ⊕ trace = Z    (given content + context, recover relation)
//! Probe:   X ⊕ Z ⊕ trace = Y    (given content + relation, recover context)
//! Probe:   Y ⊕ Z ⊕ trace = X    (given context + relation, recover content)
//! ```
//!
//! ## SIMD Layout
//!
//! 128 words per dimension / 8 = 16 AVX-512 iterations (zero remainder).
//! 512 words total / 8 = 64 AVX-512 iterations for full-vector ops.

pub mod holographic;
pub mod schema;
pub mod compat;

// ============================================================================
// VECTOR DIMENSIONS
// ============================================================================

/// Total bits in the 3D holographic vector (2^15)
pub const VECTOR_BITS: usize = 32_768;

/// Total u64 words: 32768/64 = 512
pub const VECTOR_WORDS: usize = VECTOR_BITS / 64; // 512

/// Raw bytes per vector: 512 * 8 = 4,096 = 4KB
pub const VECTOR_BYTES: usize = VECTOR_WORDS * 8; // 4096

// ============================================================================
// DIMENSION LAYOUT
// ============================================================================

/// Bits per dimension (8K = 8,192 = 2^13)
pub const DIM_BITS: usize = 8_192;

/// Words per dimension: 8192/64 = 128
pub const DIM_WORDS: usize = DIM_BITS / 64; // 128

/// Bytes per dimension: 128 * 8 = 1,024 = 1KB
pub const DIM_BYTES: usize = DIM_WORDS * 8; // 1024

/// X dimension: content/what (words 0-127)
pub const X_START: usize = 0;
pub const X_END: usize = DIM_WORDS; // 128

/// Y dimension: context/where (words 128-255)
pub const Y_START: usize = DIM_WORDS; // 128
pub const Y_END: usize = 2 * DIM_WORDS; // 256

/// Z dimension: relation/how (words 256-383)
pub const Z_START: usize = 2 * DIM_WORDS; // 256
pub const Z_END: usize = 3 * DIM_WORDS; // 384

/// Metadata block (words 384-511)
pub const META_START: usize = 3 * DIM_WORDS; // 384
pub const META_END: usize = VECTOR_WORDS; // 512

/// Words in metadata block
pub const META_WORDS: usize = DIM_WORDS; // 128

// ============================================================================
// STATISTICAL CONSTANTS (per dimension)
// ============================================================================

/// Expected Hamming distance between two random dimension vectors = n/2
pub const DIM_EXPECTED_DISTANCE: f64 = DIM_BITS as f64 / 2.0; // 4096.0

/// Standard deviation per dimension: sigma = sqrt(n/4) = sqrt(2048) ≈ 45.25
pub const DIM_SIGMA: f64 = 45.254833995939045; // sqrt(8192.0 / 4.0)

/// Integer-approximate sigma (rounded)
pub const DIM_SIGMA_APPROX: u32 = 45;

/// Product space size: 8192^3
pub const PRODUCT_SPACE: u128 = (DIM_BITS as u128) * (DIM_BITS as u128) * (DIM_BITS as u128);

/// Holographic capacity: ~sqrt(DIM_BITS) high-fidelity traces per superposition
pub const HOLOGRAPHIC_CAPACITY: usize = 90; // sqrt(8192) ≈ 90.5

// ============================================================================
// SIMD LAYOUT
// ============================================================================

/// AVX-512 iterations per dimension: 128/8 = 16 (exact)
pub const DIM_AVX512_ITERATIONS: usize = DIM_WORDS / 8; // 16

/// AVX-512 iterations for full vector: 512/8 = 64 (exact)
pub const FULL_AVX512_ITERATIONS: usize = VECTOR_WORDS / 8; // 64

/// All SIMD remainders are zero
pub const AVX512_REMAINDER: usize = 0;
pub const AVX2_REMAINDER: usize = 0;
pub const NEON_REMAINDER: usize = 0;

// ============================================================================
// BLOCK LAYOUT (within each dimension)
// ============================================================================

/// Words per block within a dimension
pub const DIM_WORDS_PER_BLOCK: usize = 16;

/// Blocks per dimension: 128/16 = 8
pub const DIM_BLOCKS: usize = DIM_WORDS / DIM_WORDS_PER_BLOCK; // 8

/// Bits per block
pub const DIM_BITS_PER_BLOCK: usize = DIM_WORDS_PER_BLOCK * 64; // 1024

/// Blocks in metadata: 128/16 = 8
pub const META_BLOCKS: usize = META_WORDS / DIM_WORDS_PER_BLOCK; // 8

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_32k_constants() {
        assert_eq!(VECTOR_BITS, 32_768);
        assert_eq!(VECTOR_WORDS, 512);
        assert_eq!(VECTOR_BYTES, 4096);
        assert_eq!(DIM_BITS, 8_192);
        assert_eq!(DIM_WORDS, 128);
        assert_eq!(DIM_BYTES, 1024);
    }

    #[test]
    fn test_32k_is_power_of_two() {
        assert!(VECTOR_BITS.is_power_of_two(), "32768 = 2^15");
        assert!(DIM_BITS.is_power_of_two(), "8192 = 2^13");
        assert!(DIM_WORDS.is_power_of_two(), "128 = 2^7");
        assert!(VECTOR_WORDS.is_power_of_two(), "512 = 2^9");
    }

    #[test]
    fn test_dimension_layout() {
        assert_eq!(X_START, 0);
        assert_eq!(X_END, 128);
        assert_eq!(Y_START, 128);
        assert_eq!(Y_END, 256);
        assert_eq!(Z_START, 256);
        assert_eq!(Z_END, 384);
        assert_eq!(META_START, 384);
        assert_eq!(META_END, 512);
        // No overlap, no gap
        assert_eq!(X_END, Y_START);
        assert_eq!(Y_END, Z_START);
        assert_eq!(Z_END, META_START);
        assert_eq!(META_END, VECTOR_WORDS);
    }

    #[test]
    fn test_product_space() {
        let expected: u128 = 8192 * 8192 * 8192;
        assert_eq!(PRODUCT_SPACE, expected);
        assert_eq!(PRODUCT_SPACE, 549_755_813_888);
    }

    #[test]
    fn test_simd_alignment() {
        // All dimensions align to AVX-512
        assert_eq!(DIM_WORDS % 8, 0);
        assert_eq!(VECTOR_WORDS % 8, 0);
        assert_eq!(META_WORDS % 8, 0);
        // Zero remainders
        assert_eq!(AVX512_REMAINDER, 0);
        assert_eq!(AVX2_REMAINDER, 0);
        assert_eq!(NEON_REMAINDER, 0);
    }

    #[test]
    fn test_4kb_record() {
        assert_eq!(VECTOR_BYTES, 4096, "One record = 4KB = one memory page");
    }

    #[test]
    fn test_blocks_per_dimension() {
        assert_eq!(DIM_BLOCKS, 8);
        assert_eq!(DIM_BLOCKS * DIM_WORDS_PER_BLOCK, DIM_WORDS);
        assert_eq!(META_BLOCKS, 8);
    }

    #[test]
    fn test_sigma_approximate() {
        let exact = (DIM_BITS as f64 / 4.0).sqrt();
        assert!((DIM_SIGMA - exact).abs() < 1e-10);
        assert_eq!(DIM_SIGMA_APPROX, 45);
    }
}
