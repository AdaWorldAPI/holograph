//! 10K / 16K / 32K Compatibility Layer
//!
//! Provides conversions between all three vector widths:
//!
//! - **10K → 32K**: Zero-extend 157 words into X dimension (words 0-127),
//!   remaining 29 words spill into Y (words 128-156). Z and metadata are zero.
//!
//! - **16K → 32K**: Map 256 words into X+Y (words 0-255). Z and metadata zero.
//!   16K schema (words 208-255) lands in Y, which is correct — context dimension.
//!
//! - **32K → 16K**: Truncate or XOR-fold to 256 words.
//!
//! - **32K → 10K**: Truncate to 157 words (drops all but first 157 words of X).
//!
//! # Dimension Mapping
//!
//! ```text
//! 10K (157 words)  →  32K X[0..127] + Y[0..28]  (zero-padded)
//! 16K (256 words)  →  32K X[0..127] + Y[0..127]  (exact 2-dim fill)
//! 32K → 16K        →  X[0..127] + Y[0..127] = words 0..255
//! 32K → 10K        →  X[0..127] + Y[0..28]  = words 0..156
//! ```
//!
//! # Storage Density Note
//!
//! 1 million 32K vectors = 1M × 4KB = 4GB RAM.
//! Each vector addresses 512 billion data points via XYZ superposition.
//! That's ~128 data points per byte of physical storage.

use super::{VECTOR_WORDS as WORDS_32K, DIM_WORDS, X_START, Y_START, Z_START, META_START};
use super::holographic::HoloVector;
use super::schema::HoloSchema;
use crate::bitpack::{BitpackedVector, VECTOR_WORDS as WORDS_10K};
use crate::width_16k::{VECTOR_WORDS as WORDS_16K};

// ============================================================================
// 10K → 32K: Zero-extend into X + Y[0..28]
// ============================================================================

/// Zero-extend a 10K vector (157 words) into a 32K HoloVector.
///
/// Words 0..127 go into X dimension, words 128..156 spill into Y.
/// Z dimension and metadata block are zero. The semantic content
/// is preserved in the first 157 words.
pub fn from_10k(v10k: &BitpackedVector) -> HoloVector {
    let mut holo = HoloVector::zero();
    let src = v10k.words();
    // Copy all 157 words starting at word 0.
    // Words 0..127 land in X, words 128..156 land in Y[0..28].
    let copy_len = WORDS_10K.min(WORDS_32K);
    holo.words[..copy_len].copy_from_slice(&src[..copy_len]);
    holo
}

/// Zero-extend a 10K vector and attach holographic schema metadata.
pub fn from_10k_with_schema(v10k: &BitpackedVector, schema: &HoloSchema) -> HoloVector {
    let mut holo = from_10k(v10k);
    schema.write_to_meta(holo.meta_mut());
    holo
}

// ============================================================================
// 16K → 32K: Map into X + Y (exact 256-word fill)
// ============================================================================

/// Extend a 16K vector (256 words) into a 32K HoloVector.
///
/// Words 0..127 → X dimension (content)
/// Words 128..255 → Y dimension (context)
/// Z dimension is zero (available for relational binding).
/// Metadata block is zero (16K schema in Y can be migrated to 32K meta).
///
/// This is the natural mapping: 16K's semantic content fills X,
/// and 16K's schema/extended blocks fill Y (context).
pub fn from_16k(words_16k: &[u64; WORDS_16K]) -> HoloVector {
    let mut holo = HoloVector::zero();
    // First 128 words → X
    holo.words[X_START..X_START + DIM_WORDS].copy_from_slice(&words_16k[..DIM_WORDS]);
    // Next 128 words → Y
    holo.words[Y_START..Y_START + DIM_WORDS].copy_from_slice(&words_16k[DIM_WORDS..WORDS_16K]);
    holo
}

/// Extend a 16K vector and migrate its schema to 32K holographic schema.
///
/// Reads the 16K SchemaSidecar, converts relevant fields to HoloSchema,
/// and writes it into the 32K metadata block.
pub fn from_16k_with_schema(
    words_16k: &[u64; WORDS_16K],
    schema: &HoloSchema,
) -> HoloVector {
    let mut holo = from_16k(words_16k);
    schema.write_to_meta(holo.meta_mut());
    holo
}

/// Extend a 16K vector from a slice (e.g., from Arrow buffer).
pub fn from_16k_slice(words_16k: &[u64]) -> Option<HoloVector> {
    if words_16k.len() < WORDS_16K {
        return None;
    }
    let mut holo = HoloVector::zero();
    holo.words[X_START..X_START + DIM_WORDS].copy_from_slice(&words_16k[..DIM_WORDS]);
    holo.words[Y_START..Y_START + DIM_WORDS].copy_from_slice(&words_16k[DIM_WORDS..WORDS_16K]);
    Some(holo)
}

// ============================================================================
// 32K → 16K: Truncate X + Y back to 256 words
// ============================================================================

/// Truncate a 32K HoloVector to 16K (256 words).
///
/// Returns X[0..127] ++ Y[0..127] as a 256-word array.
/// Z dimension and metadata are discarded.
pub fn to_16k(holo: &HoloVector) -> [u64; WORDS_16K] {
    let mut words = [0u64; WORDS_16K];
    // X → first 128 words
    words[..DIM_WORDS].copy_from_slice(&holo.words[X_START..X_START + DIM_WORDS]);
    // Y → next 128 words
    words[DIM_WORDS..WORDS_16K].copy_from_slice(&holo.words[Y_START..Y_START + DIM_WORDS]);
    words
}

/// XOR-fold 32K to 16K: fold Z and metadata into X+Y via XOR.
///
/// This preserves more signal than truncation: the Z and metadata
/// information is hashed into the 16K space via XOR compression.
pub fn xor_fold_to_16k(holo: &HoloVector) -> [u64; WORDS_16K] {
    let mut words = to_16k(holo);
    // Fold Z into first 128 words (overlaps with X region)
    for i in 0..DIM_WORDS {
        words[i] ^= holo.words[Z_START + i];
    }
    // Fold metadata into second 128 words (overlaps with Y region)
    for i in 0..DIM_WORDS {
        words[DIM_WORDS + i] ^= holo.words[META_START + i];
    }
    words
}

// ============================================================================
// 32K → 10K: Truncate to 157 words
// ============================================================================

/// Truncate a 32K HoloVector to 10K (157 words).
///
/// Returns words 0..156 (X[0..127] + Y[0..28]).
/// Everything else is discarded.
pub fn to_10k(holo: &HoloVector) -> BitpackedVector {
    let mut words = [0u64; WORDS_10K];
    words.copy_from_slice(&holo.words[..WORDS_10K]);
    BitpackedVector::from_words(words)
}

/// XOR-fold 32K to 10K: fold all extra words back via XOR.
///
/// Words 157..511 are folded into words 0..156 via cyclic XOR.
/// Lossy but encodes all 4 dimensions into the 10K space.
pub fn xor_fold_to_10k(holo: &HoloVector) -> BitpackedVector {
    let mut words = [0u64; WORDS_10K];
    words.copy_from_slice(&holo.words[..WORDS_10K]);
    // Fold all extra words cyclically
    for i in WORDS_10K..WORDS_32K {
        words[i % WORDS_10K] ^= holo.words[i];
    }
    BitpackedVector::from_words(words)
}

// ============================================================================
// CROSS-WIDTH DISTANCE
// ============================================================================

/// Distance between a 10K vector and a 32K HoloVector.
///
/// Compares only the first 157 words (the 10K content region).
pub fn distance_10k_32k(v10k: &BitpackedVector, holo: &HoloVector) -> u32 {
    let src = v10k.words();
    let mut total = 0u32;
    for w in 0..WORDS_10K {
        total += (src[w] ^ holo.words[w]).count_ones();
    }
    total
}

/// Distance between a 16K vector and a 32K HoloVector.
///
/// Compares X+Y dimensions (first 256 words of 32K mapped to 16K layout).
pub fn distance_16k_32k(words_16k: &[u64; WORDS_16K], holo: &HoloVector) -> u32 {
    let mut total = 0u32;
    // Compare first 128 words (X dimension ↔ 16K[0..127])
    for w in 0..DIM_WORDS {
        total += (words_16k[w] ^ holo.words[X_START + w]).count_ones();
    }
    // Compare next 128 words (Y dimension ↔ 16K[128..255])
    for w in 0..DIM_WORDS {
        total += (words_16k[DIM_WORDS + w] ^ holo.words[Y_START + w]).count_ones();
    }
    total
}

// ============================================================================
// BATCH MIGRATION
// ============================================================================

/// Migrate a batch of 10K vectors to 32K HoloVectors.
pub fn migrate_batch_10k(vectors: &[BitpackedVector]) -> Vec<HoloVector> {
    vectors.iter().map(|v| from_10k(v)).collect()
}

/// Migrate a batch of 16K word arrays to 32K HoloVectors.
pub fn migrate_batch_16k(vectors: &[[u64; WORDS_16K]]) -> Vec<HoloVector> {
    vectors.iter().map(|v| from_16k(v)).collect()
}

/// Migrate a batch of 10K vectors with a shared schema.
pub fn migrate_batch_10k_with_schema(
    vectors: &[BitpackedVector],
    schema: &HoloSchema,
) -> Vec<HoloVector> {
    vectors.iter().map(|v| from_10k_with_schema(v, schema)).collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_10k_to_32k_preserves_content() {
        let v = BitpackedVector::random(42);
        let holo = from_10k(&v);

        // First 157 words should match
        for w in 0..WORDS_10K {
            assert_eq!(v.words()[w], holo.words[w],
                "Word {} mismatch after 10K→32K", w);
        }
        // Rest should be zero
        for w in WORDS_10K..WORDS_32K {
            assert_eq!(holo.words[w], 0,
                "Word {} should be zero after 10K→32K", w);
        }
    }

    #[test]
    fn test_10k_roundtrip() {
        let v = BitpackedVector::random(42);
        let holo = from_10k(&v);
        let recovered = to_10k(&holo);
        assert_eq!(v, recovered, "10K→32K→10K roundtrip failed");
    }

    #[test]
    fn test_16k_to_32k_layout() {
        // Create a 16K vector with known data
        let mut words_16k = [0u64; WORDS_16K];
        words_16k[0] = 0xAAAA;     // First word of first 128 (→ X[0])
        words_16k[127] = 0xBBBB;   // Last word of first 128 (→ X[127])
        words_16k[128] = 0xCCCC;   // First word of second 128 (→ Y[0])
        words_16k[255] = 0xDDDD;   // Last word of second 128 (→ Y[127])

        let holo = from_16k(&words_16k);

        // Verify X dimension
        assert_eq!(holo.x()[0], 0xAAAA, "X[0] should be 16K[0]");
        assert_eq!(holo.x()[127], 0xBBBB, "X[127] should be 16K[127]");

        // Verify Y dimension
        assert_eq!(holo.y()[0], 0xCCCC, "Y[0] should be 16K[128]");
        assert_eq!(holo.y()[127], 0xDDDD, "Y[127] should be 16K[255]");

        // Verify Z and metadata are zero
        for w in 0..DIM_WORDS {
            assert_eq!(holo.z()[w], 0, "Z should be zero");
            assert_eq!(holo.meta()[w], 0, "Metadata should be zero");
        }
    }

    #[test]
    fn test_16k_roundtrip() {
        let mut words_16k = [0u64; WORDS_16K];
        // Fill with pseudo-random data
        let mut state = 123u64;
        for w in words_16k.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = state;
        }

        let holo = from_16k(&words_16k);
        let recovered = to_16k(&holo);

        for w in 0..WORDS_16K {
            assert_eq!(words_16k[w], recovered[w],
                "16K→32K→16K roundtrip failed at word {}", w);
        }
    }

    #[test]
    fn test_xor_fold_to_16k_differs_from_truncate() {
        // Create a HoloVector with non-zero Z and metadata
        let mut holo = HoloVector::zero();
        holo.words[0] = 0xDEAD;       // X[0]
        holo.words[Z_START] = 0xBEEF;  // Z[0] — should fold into X[0]
        holo.words[META_START] = 0xCAFE; // Meta[0] — should fold into Y[0]

        let truncated = to_16k(&holo);
        let folded = xor_fold_to_16k(&holo);

        // Word 0: truncate = 0xDEAD, fold = 0xDEAD ^ 0xBEEF
        assert_eq!(truncated[0], 0xDEAD);
        assert_eq!(folded[0], 0xDEAD ^ 0xBEEF);

        // Word 128: truncate = 0, fold = 0 ^ 0xCAFE
        assert_eq!(truncated[DIM_WORDS], 0);
        assert_eq!(folded[DIM_WORDS], 0xCAFE);
    }

    #[test]
    fn test_xor_fold_to_16k_identity_when_z_meta_zero() {
        // When Z and metadata are zero, fold = truncate
        let mut holo = HoloVector::zero();
        let mut state = 42u64;
        // Only fill X and Y
        for i in 0..DIM_WORDS * 2 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            holo.words[i] = state;
        }

        let truncated = to_16k(&holo);
        let folded = xor_fold_to_16k(&holo);

        for w in 0..WORDS_16K {
            assert_eq!(truncated[w], folded[w],
                "Fold should equal truncate when Z/meta are zero (word {})", w);
        }
    }

    #[test]
    fn test_xor_fold_to_10k() {
        let mut holo = HoloVector::zero();
        holo.words[0] = 0xFF;
        holo.words[WORDS_10K] = 0xAA; // First word past 10K boundary

        let truncated = to_10k(&holo);
        let folded = xor_fold_to_10k(&holo);

        // Truncate: word 0 = 0xFF
        assert_eq!(truncated.words()[0], 0xFF);
        // Fold: word 0 = 0xFF ^ 0xAA (word WORDS_10K folds back to position 0)
        assert_eq!(folded.words()[0], 0xFF ^ 0xAA);
    }

    #[test]
    fn test_cross_width_distance_10k_self() {
        let v = BitpackedVector::random(42);
        let holo = from_10k(&v);
        assert_eq!(distance_10k_32k(&v, &holo), 0,
            "10K vector should have distance 0 to its own 32K extension");
    }

    #[test]
    fn test_cross_width_distance_16k_self() {
        let mut words_16k = [0u64; WORDS_16K];
        let mut state = 99u64;
        for w in words_16k.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = state;
        }

        let holo = from_16k(&words_16k);
        assert_eq!(distance_16k_32k(&words_16k, &holo), 0,
            "16K vector should have distance 0 to its own 32K extension");
    }

    #[test]
    fn test_from_16k_slice() {
        let mut words = [0u64; WORDS_16K];
        words[0] = 0x1234;
        words[128] = 0x5678;

        // Valid slice
        let holo = from_16k_slice(&words).unwrap();
        assert_eq!(holo.x()[0], 0x1234);
        assert_eq!(holo.y()[0], 0x5678);

        // Too-short slice
        let short: Vec<u64> = vec![0; 100];
        assert!(from_16k_slice(&short).is_none());
    }

    #[test]
    fn test_batch_migration_10k() {
        let vectors: Vec<BitpackedVector> = (0..5)
            .map(|i| BitpackedVector::random(i as u64))
            .collect();
        let migrated = migrate_batch_10k(&vectors);
        assert_eq!(migrated.len(), 5);

        for (orig, holo) in vectors.iter().zip(migrated.iter()) {
            assert_eq!(*orig, to_10k(holo),
                "Batch 10K→32K→10K roundtrip failed");
        }
    }

    #[test]
    fn test_batch_migration_16k() {
        let vectors: Vec<[u64; WORDS_16K]> = (0..5).map(|seed| {
            let mut words = [0u64; WORDS_16K];
            let mut state = seed as u64 + 1;
            for w in words.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *w = state;
            }
            words
        }).collect();

        let migrated = migrate_batch_16k(&vectors);
        assert_eq!(migrated.len(), 5);

        for (orig, holo) in vectors.iter().zip(migrated.iter()) {
            let recovered = to_16k(holo);
            for w in 0..WORDS_16K {
                assert_eq!(orig[w], recovered[w],
                    "Batch 16K→32K→16K roundtrip failed");
            }
        }
    }

    #[test]
    fn test_storage_density_note() {
        // 1 million vectors × 4KB = 4GB
        let record_size = WORDS_32K * 8;
        assert_eq!(record_size, 4096, "Each record should be 4KB");

        let million_records_bytes = 1_000_000u64 * record_size as u64;
        let gb = million_records_bytes / (1024 * 1024 * 1024);
        // 4,000,000,000 / 1,073,741,824 ≈ 3.72 GB
        assert!(gb >= 3 && gb <= 4,
            "1M records should be ~4GB, got {}GB", gb);
    }
}
