//! 10K ↔ 16K Compatibility Layer
//!
//! Provides zero-copy-friendly conversions between the two vector widths:
//!
//! - **10K → 16K (zero-extend)**: Pad words 157..255 with zeros.
//!   The semantic content is identical. Schema blocks are blank (all-semantic mode).
//!
//! - **16K → 10K (truncate)**: Drop words 157..255.
//!   Schema is lost but semantic fidelity is preserved for the first 10K bits.
//!
//! - **16K → 10K (fold)**: XOR-fold the extra 6K bits into the base 10K.
//!   This compresses schema and extra semantic info into the 10K space
//!   via hash-like folding. Lossy but preserves more signal than truncation.
//!
//! # Forward/Backward Compatibility
//!
//! The key insight: a 10K vector zero-extended to 16K has distance 0
//! from itself on blocks 0..9 (the original 10K) and distance 0 on
//! blocks 10..15 (all zeros). So 10K vectors can participate in 16K
//! searches with no semantic distortion — they just don't have schema
//! markers or the extra 6K information bits.

use crate::bitpack::{BitpackedVector, VECTOR_WORDS as WORDS_10K};
use super::VECTOR_WORDS as WORDS_16K;
use super::schema::SchemaSidecar;

// ============================================================================
// ZERO-EXTEND: 10K → 16K
// ============================================================================

/// Zero-extend a 10K vector to 16K.
///
/// Words 0..156 are copied. Words 157..255 are zero.
/// The result has identical Hamming distance to any other zero-extended
/// vector on the semantic blocks, and zero distance on the padding blocks.
///
/// This is the recommended way to migrate 10K data into 16K storage.
pub fn zero_extend(v10k: &BitpackedVector) -> [u64; WORDS_16K] {
    let mut words = [0u64; WORDS_16K];
    let src = v10k.words();
    words[..WORDS_10K].copy_from_slice(src);
    words
}

/// Zero-extend and attach schema metadata.
///
/// Same as `zero_extend` but also writes a SchemaSidecar into blocks 13-15.
/// Useful when ingesting 10K vectors into a 16K store and wanting to
/// populate schema fields (e.g., from external metadata).
pub fn zero_extend_with_schema(
    v10k: &BitpackedVector,
    schema: &SchemaSidecar,
) -> [u64; WORDS_16K] {
    let mut words = zero_extend(v10k);
    schema.write_to_words(&mut words);
    words
}

// ============================================================================
// TRUNCATE: 16K → 10K
// ============================================================================

/// Truncate a 16K vector to 10K by dropping words 157..255.
///
/// The first 10,000 bits are preserved exactly. Schema and extra
/// semantic information are discarded.
pub fn truncate(words_16k: &[u64; WORDS_16K]) -> BitpackedVector {
    let mut words_10k = [0u64; WORDS_10K];
    words_10k.copy_from_slice(&words_16k[..WORDS_10K]);
    BitpackedVector::from_words(words_10k)
}

/// Truncate from a slice (e.g., from Arrow buffer).
pub fn truncate_slice(words_16k: &[u64]) -> Option<BitpackedVector> {
    if words_16k.len() < WORDS_16K {
        return None;
    }
    let mut words_10k = [0u64; WORDS_10K];
    words_10k.copy_from_slice(&words_16k[..WORDS_10K]);
    Some(BitpackedVector::from_words(words_10k))
}

// ============================================================================
// XOR-FOLD: 16K → 10K (lossy but preserves more signal)
// ============================================================================

/// XOR-fold a 16K vector into 10K.
///
/// The extra words 157..255 (99 words = 6,336 bits) are folded back
/// into the base via XOR. This is lossy but acts as a hash compression:
/// the folded result encodes both the base semantics and the extra
/// information into the 10K space.
///
/// Preserves more signal than truncation but is not reversible.
pub fn xor_fold(words_16k: &[u64; WORDS_16K]) -> BitpackedVector {
    let mut words_10k = [0u64; WORDS_10K];
    // Start with the base 10K
    words_10k.copy_from_slice(&words_16k[..WORDS_10K]);

    // Fold the extra words back in via XOR
    let extra_start = WORDS_10K;
    let extra_count = WORDS_16K - WORDS_10K; // 99 words
    for i in 0..extra_count {
        words_10k[i % WORDS_10K] ^= words_16k[extra_start + i];
    }

    BitpackedVector::from_words(words_10k)
}

// ============================================================================
// DISTANCE COMPATIBILITY
// ============================================================================

/// Compute semantic distance between a 10K and a 16K vector.
///
/// Only compares the first 157 words (10K bits). The extra 16K words
/// are ignored, so this gives the same result as if both were 10K.
pub fn cross_width_distance(v10k: &BitpackedVector, words_16k: &[u64]) -> u32 {
    let words_a = v10k.words();
    let mut total = 0u32;
    for w in 0..WORDS_10K {
        total += (words_a[w] ^ words_16k[w]).count_ones();
    }
    total
}

/// Compute full 16K distance between two 16K word arrays.
pub fn full_distance_16k(a: &[u64], b: &[u64]) -> u32 {
    debug_assert!(a.len() >= WORDS_16K && b.len() >= WORDS_16K);
    let mut total = 0u32;
    for w in 0..WORDS_16K {
        total += (a[w] ^ b[w]).count_ones();
    }
    total
}

// ============================================================================
// BATCH MIGRATION
// ============================================================================

/// Migrate a batch of 10K vectors to 16K word arrays.
///
/// Returns owned Vec of 16K word arrays. For large batches, consider
/// streaming to Arrow FixedSizeBinary(2048) instead.
pub fn migrate_batch(vectors: &[BitpackedVector]) -> Vec<[u64; WORDS_16K]> {
    vectors.iter().map(|v| zero_extend(v)).collect()
}

/// Migrate with schema: apply the same SchemaSidecar to all vectors.
///
/// Useful for batch-setting a default schema (e.g., all nodes are Entity
/// with default NARS truth values).
pub fn migrate_batch_with_schema(
    vectors: &[BitpackedVector],
    schema: &SchemaSidecar,
) -> Vec<[u64; WORDS_16K]> {
    vectors.iter().map(|v| zero_extend_with_schema(v, schema)).collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schema::NarsTruth;

    #[test]
    fn test_zero_extend_preserves_content() {
        let v = BitpackedVector::random(42);
        let extended = zero_extend(&v);

        // First 157 words match
        for w in 0..WORDS_10K {
            assert_eq!(v.words()[w], extended[w]);
        }
        // Rest is zero
        for w in WORDS_10K..WORDS_16K {
            assert_eq!(extended[w], 0);
        }
    }

    #[test]
    fn test_truncate_roundtrip() {
        let v = BitpackedVector::random(42);
        let extended = zero_extend(&v);
        let truncated = truncate(&extended);
        assert_eq!(v, truncated);
    }

    #[test]
    fn test_xor_fold_different_from_truncate() {
        // Create a 16K vector with non-zero data in the extra words
        let mut words = [0u64; WORDS_16K];
        words[0] = 0xDEADBEEF;
        words[200] = 0xCAFEBABE; // In schema region

        let truncated = truncate(&words);
        let folded = xor_fold(&words);

        // XOR fold should produce different result when extra words are non-zero
        assert_ne!(truncated, folded);
    }

    #[test]
    fn test_xor_fold_identity_when_extra_zero() {
        // When extra words are zero, fold = truncate
        let v = BitpackedVector::random(42);
        let extended = zero_extend(&v);
        let folded = xor_fold(&extended);
        let truncated = truncate(&extended);
        assert_eq!(folded, truncated);
    }

    #[test]
    fn test_cross_width_distance_self() {
        let v = BitpackedVector::random(42);
        let extended = zero_extend(&v);
        assert_eq!(cross_width_distance(&v, &extended), 0);
    }

    #[test]
    fn test_cross_width_distance_symmetry() {
        let a = BitpackedVector::random(1);
        let b = BitpackedVector::random(2);
        let _a16 = zero_extend(&a);
        let b16 = zero_extend(&b);

        // 10K×16K distance should equal 10K×10K distance
        let dist_10k = crate::hamming::hamming_distance_scalar(&a, &b);
        let dist_cross = cross_width_distance(&a, &b16);
        assert_eq!(dist_10k, dist_cross);
    }

    #[test]
    fn test_zero_extend_with_schema() {
        let v = BitpackedVector::random(42);
        let mut schema = SchemaSidecar::default();
        schema.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        schema.metrics.pagerank = 500;

        let extended = zero_extend_with_schema(&v, &schema);

        // Schema should be readable
        let recovered = SchemaSidecar::read_from_words(&extended);
        assert_eq!(recovered.metrics.pagerank, 500);
        assert!((recovered.nars_truth.f() - 0.8).abs() < 0.01);

        // Semantic content preserved
        for w in 0..WORDS_10K {
            assert_eq!(v.words()[w], extended[w]);
        }
    }

    #[test]
    fn test_migrate_batch() {
        let vectors: Vec<BitpackedVector> = (0..10)
            .map(|i| BitpackedVector::random(i as u64))
            .collect();
        let migrated = migrate_batch(&vectors);
        assert_eq!(migrated.len(), 10);

        // Each should truncate back to original
        for (orig, m16k) in vectors.iter().zip(migrated.iter()) {
            assert_eq!(*orig, truncate(m16k));
        }
    }
}
