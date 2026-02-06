//! Bitpacked 10Kbit Vector Implementation
//!
//! Core data structure for hyperdimensional computing:
//! - 10,000 bits packed into 157 × u64 words (10,048 bits with 48 padding)
//! - 64-byte aligned for SIMD operations
//! - Efficient bit manipulation primitives

use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not};
use crate::{HdrError, Result};

/// Number of bits in the logical vector (10,000)
pub const VECTOR_BITS: usize = 10_000;

/// Number of u64 words needed: ceil(10000/64) = 157
pub const VECTOR_WORDS: usize = (VECTOR_BITS + 63) / 64;

/// Bytes per vector: 157 × 8 = 1,256 bytes
pub const VECTOR_BYTES: usize = VECTOR_WORDS * 8;

/// Padded words for 64-byte (cache-line) alignment: ceil(157/8)*8 = 160
///
/// In Arrow `FixedSizeBinary(PADDED_VECTOR_BYTES)`, every vector starts at
/// a 64-byte boundary (since 1280 = 20 × 64), enabling zero-copy SIMD loads
/// directly on the Arrow buffer without materializing BitpackedVector.
pub const PADDED_VECTOR_WORDS: usize = (VECTOR_WORDS + 7) & !7; // 160

/// Padded bytes per vector: 160 × 8 = 1,280 bytes = 20 × 64 bytes
///
/// Use this (not VECTOR_BYTES) for Arrow FixedSizeBinary column width.
/// The 3 padding words (157..160) are always zero.
pub const PADDED_VECTOR_BYTES: usize = PADDED_VECTOR_WORDS * 8; // 1280

/// Mask for the last word (only 16 bits used: 10000 - 156×64 = 16)
const LAST_WORD_BITS: usize = VECTOR_BITS - (VECTOR_WORDS - 1) * 64;
const LAST_WORD_MASK: u64 = (1u64 << LAST_WORD_BITS) - 1;

/// A 10,000-bit vector stored as 157 packed u64 words.
///
/// This is the fundamental unit for hyperdimensional computing:
/// - XOR binding for concept composition
/// - Hamming distance for similarity
/// - Majority bundling for prototypes
///
/// # Memory Layout
///
/// ```text
/// ┌────────────────────────────────────────────────────┐
/// │ word[0]   │ word[1]   │ ... │ word[155] │ word[156]│
/// │ bits 0-63 │ bits 64-127│    │           │bits 9984-│
/// │           │           │     │           │    9999  │
/// └────────────────────────────────────────────────────┘
///   64 bits     64 bits          64 bits     16 bits used
/// ```
#[derive(Clone, PartialEq, Eq)]
#[repr(C, align(64))]  // Cache-line aligned for SIMD
pub struct BitpackedVector {
    /// The packed bits
    words: [u64; VECTOR_WORDS],
}

impl Default for BitpackedVector {
    fn default() -> Self {
        Self::zero()
    }
}

impl BitpackedVector {
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    /// Create a zero vector (all bits 0)
    #[inline]
    pub const fn zero() -> Self {
        Self {
            words: [0u64; VECTOR_WORDS],
        }
    }

    /// Create a vector with all bits set to 1
    #[inline]
    pub fn ones() -> Self {
        let mut v = Self { words: [!0u64; VECTOR_WORDS] };
        // Mask the last word to only use valid bits
        v.words[VECTOR_WORDS - 1] &= LAST_WORD_MASK;
        v
    }

    /// Create from raw u64 words
    ///
    /// # Safety
    /// The last word will be masked to ensure only valid bits are set.
    #[inline]
    pub fn from_words(words: [u64; VECTOR_WORDS]) -> Self {
        let mut v = Self { words };
        v.words[VECTOR_WORDS - 1] &= LAST_WORD_MASK;
        v
    }

    /// Create from a slice of u64 words
    pub fn from_slice(slice: &[u64]) -> Result<Self> {
        if slice.len() != VECTOR_WORDS {
            return Err(HdrError::DimensionMismatch {
                expected: VECTOR_WORDS,
                got: slice.len(),
            });
        }
        let mut words = [0u64; VECTOR_WORDS];
        words.copy_from_slice(slice);
        Ok(Self::from_words(words))
    }

    /// Create from bytes (little-endian)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != VECTOR_BYTES {
            return Err(HdrError::DimensionMismatch {
                expected: VECTOR_BYTES,
                got: bytes.len(),
            });
        }
        let mut words = [0u64; VECTOR_WORDS];
        for (i, word) in words.iter_mut().enumerate() {
            let start = i * 8;
            *word = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        }
        Ok(Self::from_words(words))
    }

    /// Create a random vector using a simple PRNG
    /// Uses xorshift128+ for speed
    pub fn random(seed: u64) -> Self {
        let mut s0 = seed;
        let mut s1 = seed.wrapping_mul(0x9E3779B97F4A7C15);

        let mut words = [0u64; VECTOR_WORDS];
        for word in &mut words {
            // xorshift128+
            let mut s = s0;
            s0 = s1;
            s ^= s << 23;
            s ^= s >> 18;
            s ^= s1;
            s ^= s1 >> 5;
            s1 = s;
            *word = s0.wrapping_add(s1);
        }
        Self::from_words(words)
    }

    /// Create from a hash of arbitrary data
    pub fn from_hash(data: &[u8]) -> Self {
        // Simple SipHash-like mixing
        let mut h = 0x736f6d6570736575u64;
        for chunk in data.chunks(8) {
            let mut block = [0u8; 8];
            block[..chunk.len()].copy_from_slice(chunk);
            let k = u64::from_le_bytes(block);
            h ^= k;
            h = h.rotate_left(13);
            h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
        }
        Self::random(h)
    }

    // =========================================================================
    // ACCESSORS
    // =========================================================================

    /// Get the raw words
    #[inline]
    pub fn words(&self) -> &[u64; VECTOR_WORDS] {
        &self.words
    }

    /// Get mutable reference to words
    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64; VECTOR_WORDS] {
        &mut self.words
    }

    /// Convert to bytes (little-endian)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(VECTOR_BYTES);
        for word in &self.words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Get a specific bit (0-indexed)
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index < VECTOR_BITS);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }

    /// Set a specific bit
    #[inline]
    pub fn set_bit(&mut self, index: usize, value: bool) {
        debug_assert!(index < VECTOR_BITS);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.words[word_idx] |= 1u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Toggle a specific bit
    #[inline]
    pub fn toggle_bit(&mut self, index: usize) {
        debug_assert!(index < VECTOR_BITS);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        self.words[word_idx] ^= 1u64 << bit_idx;
    }

    // =========================================================================
    // POPULATION COUNT (Core of Hamming)
    // =========================================================================

    /// Count total set bits (population count)
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Count set bits using stacked popcount (per-word)
    ///
    /// Returns counts for each word - useful for hierarchical filtering
    #[inline]
    pub fn stacked_popcount(&self) -> [u8; VECTOR_WORDS] {
        let mut counts = [0u8; VECTOR_WORDS];
        for (i, word) in self.words.iter().enumerate() {
            counts[i] = word.count_ones() as u8;
        }
        counts
    }

    /// Compute density (fraction of bits set)
    #[inline]
    pub fn density(&self) -> f32 {
        self.popcount() as f32 / VECTOR_BITS as f32
    }

    // =========================================================================
    // BITWISE OPERATIONS (Vector Field Operations)
    // =========================================================================

    /// XOR with another vector (binding operation)
    #[inline]
    pub fn xor(&self, other: &Self) -> Self {
        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = self.words[i] ^ other.words[i];
        }
        Self::from_words(result)
    }

    /// AND with another vector
    #[inline]
    pub fn and(&self, other: &Self) -> Self {
        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = self.words[i] & other.words[i];
        }
        Self::from_words(result)
    }

    /// OR with another vector
    #[inline]
    pub fn or(&self, other: &Self) -> Self {
        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = self.words[i] | other.words[i];
        }
        Self::from_words(result)
    }

    /// NOT (invert all bits)
    #[inline]
    pub fn not(&self) -> Self {
        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = !self.words[i];
        }
        // Mask the last word
        result[VECTOR_WORDS - 1] &= LAST_WORD_MASK;
        Self { words: result }
    }

    /// XOR in-place (for efficiency)
    #[inline]
    pub fn xor_assign(&mut self, other: &Self) {
        for i in 0..VECTOR_WORDS {
            self.words[i] ^= other.words[i];
        }
    }

    // =========================================================================
    // PERMUTATION (Sequence Encoding)
    // =========================================================================

    /// Rotate bits left by n positions within the logical 10,000-bit space
    pub fn rotate_left(&self, n: usize) -> Self {
        let n = n % VECTOR_BITS;
        if n == 0 {
            return self.clone();
        }

        let mut result = Self::zero();

        for i in 0..VECTOR_BITS {
            let src_bit = (i + VECTOR_BITS - n) % VECTOR_BITS;
            if self.get_bit(src_bit) {
                result.set_bit(i, true);
            }
        }

        result
    }

    /// Rotate bits right by n positions
    pub fn rotate_right(&self, n: usize) -> Self {
        self.rotate_left(VECTOR_BITS - (n % VECTOR_BITS))
    }

    /// Fast word-level rotation (64-bit granularity)
    #[inline]
    pub fn rotate_words(&self, n: usize) -> Self {
        let n = n % VECTOR_WORDS;
        if n == 0 {
            return self.clone();
        }

        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = self.words[(i + VECTOR_WORDS - n) % VECTOR_WORDS];
        }
        Self::from_words(result)
    }

    // =========================================================================
    // BIT FLIPPING (for tests and perturbation)
    // =========================================================================

    /// Rotate bits by n positions (alias for rotate_left)
    pub fn rotate_bits(&self, n: usize) -> Self {
        self.rotate_left(n)
    }

    /// Flip n random bits using a seed for deterministic randomness
    pub fn flip_random_bits(&mut self, n: usize, seed: u64) {
        let mut s0 = seed;
        let mut s1 = seed.wrapping_mul(0x9E3779B97F4A7C15);

        for _ in 0..n {
            // xorshift128+
            let mut s = s0;
            s0 = s1;
            s ^= s << 23;
            s ^= s >> 18;
            s ^= s1;
            s ^= s1 >> 5;
            s1 = s;
            let val = s0.wrapping_add(s1);

            let bit_idx = (val as usize) % VECTOR_BITS;
            self.toggle_bit(bit_idx);
        }
    }

    // =========================================================================
    // BUNDLING (Majority Voting)
    // =========================================================================

    /// Bundle multiple vectors using majority voting
    ///
    /// Each bit is set if more than half the input vectors have it set.
    /// Breaks ties randomly using the first vector's bits.
    pub fn bundle(vectors: &[&Self]) -> Self {
        if vectors.is_empty() {
            return Self::zero();
        }
        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let threshold = vectors.len() / 2;
        let tie_breaker = if vectors.len() % 2 == 0 {
            Some(vectors[0])
        } else {
            None
        };

        let mut result = Self::zero();

        // Process word by word for efficiency
        for word_idx in 0..VECTOR_WORDS {
            let mut result_word = 0u64;

            for bit in 0..64 {
                if word_idx == VECTOR_WORDS - 1 && bit >= LAST_WORD_BITS {
                    break;
                }

                let mask = 1u64 << bit;
                let count: usize = vectors
                    .iter()
                    .filter(|v| v.words[word_idx] & mask != 0)
                    .count();

                if count > threshold {
                    result_word |= mask;
                } else if count == threshold {
                    // Tie: use tie-breaker if available
                    if let Some(tb) = tie_breaker {
                        if tb.words[word_idx] & mask != 0 {
                            result_word |= mask;
                        }
                    }
                }
            }

            result.words[word_idx] = result_word;
        }

        result
    }

    /// Bundle with weighted voting
    pub fn bundle_weighted(vectors: &[(&Self, f32)]) -> Self {
        if vectors.is_empty() {
            return Self::zero();
        }

        let total_weight: f32 = vectors.iter().map(|(_, w)| w).sum();
        let threshold = total_weight / 2.0;

        let mut result = Self::zero();

        for word_idx in 0..VECTOR_WORDS {
            let mut result_word = 0u64;

            for bit in 0..64 {
                if word_idx == VECTOR_WORDS - 1 && bit >= LAST_WORD_BITS {
                    break;
                }

                let mask = 1u64 << bit;
                let weight_sum: f32 = vectors
                    .iter()
                    .filter(|(v, _)| v.words[word_idx] & mask != 0)
                    .map(|(_, w)| w)
                    .sum();

                if weight_sum >= threshold {
                    result_word |= mask;
                }
            }

            result.words[word_idx] = result_word;
        }

        result
    }
}

// =========================================================================
// ZERO-COPY VECTOR VIEW
// =========================================================================

/// Trait for anything that can be read as a word-level vector slice.
///
/// This enables all Hamming/Belichtung/StackedPopcount operations to work
/// on both owned `BitpackedVector` and borrowed `VectorSlice` (which points
/// directly into an Arrow buffer with zero copies).
pub trait VectorRef {
    /// Access the underlying u64 words. Always exactly VECTOR_WORDS long.
    fn words(&self) -> &[u64];

    /// Population count (total set bits)
    #[inline]
    fn popcount(&self) -> u32 {
        self.words().iter().map(|w| w.count_ones()).sum()
    }

    /// Density (fraction of bits set)
    #[inline]
    fn density(&self) -> f32 {
        self.popcount() as f32 / VECTOR_BITS as f32
    }

    /// Per-word popcount for hierarchical filtering
    #[inline]
    fn stacked_popcount(&self) -> [u8; VECTOR_WORDS] {
        let mut counts = [0u8; VECTOR_WORDS];
        let w = self.words();
        for i in 0..VECTOR_WORDS {
            counts[i] = w[i].count_ones() as u8;
        }
        counts
    }

    /// Promote to owned BitpackedVector (copies if borrowed)
    fn to_owned_vector(&self) -> BitpackedVector {
        let mut words = [0u64; VECTOR_WORDS];
        words.copy_from_slice(&self.words()[..VECTOR_WORDS]);
        BitpackedVector::from_words(words)
    }
}

impl VectorRef for BitpackedVector {
    #[inline]
    fn words(&self) -> &[u64] {
        &self.words
    }
}

/// A zero-copy borrowed view into vector data stored in an Arrow buffer.
///
/// # Why This Matters
///
/// Without `VectorSlice`, every vector access from Arrow does this:
/// ```text
/// Arrow Buffer → &[u8] → from_bytes() → copy 1256 bytes → BitpackedVector
///                                         ^^^ O(n) memory bloat
/// ```
///
/// With `VectorSlice`, the path is:
/// ```text
/// Arrow Buffer → &[u8] → reinterpret as &[u64] → VectorSlice (zero-copy)
/// ```
///
/// Combined with cascaded Hamming (Belichtungsmesser filters 90% in ~14 cycles),
/// a GQL query touching 1M vectors copies 0 bytes for the 999,000 that fail
/// the cascade.
///
/// # Alignment
///
/// Arrow's FixedSizeBinary column uses `PADDED_VECTOR_BYTES` (1280) per entry.
/// Since 1280 = 20 × 64, every entry is 64-byte (cache-line) aligned when
/// the Arrow buffer itself is 64-byte aligned (which Arrow guarantees).
/// This means SIMD loads work directly on the slice — no memcpy needed.
///
/// # Safety
///
/// The borrowed slice must be at least `VECTOR_WORDS` u64s long and the data
/// must be valid (padding bits in word[156] must be masked). Arrow columns
/// built with `PaddedVectorBuilder` satisfy both invariants.
#[derive(Clone, Copy)]
pub struct VectorSlice<'a> {
    words: &'a [u64],
}

impl<'a> VectorSlice<'a> {
    /// Create from a u64 word slice.
    ///
    /// # Panics
    /// Panics if `words.len() < VECTOR_WORDS`.
    #[inline]
    pub fn from_words(words: &'a [u64]) -> Self {
        debug_assert!(words.len() >= VECTOR_WORDS,
            "VectorSlice requires {} words, got {}", VECTOR_WORDS, words.len());
        Self { words: &words[..VECTOR_WORDS] }
    }

    /// Create from a byte slice (Arrow FixedSizeBinary value).
    ///
    /// # Safety
    /// The byte slice must be at least `VECTOR_BYTES` long and 8-byte aligned.
    /// Arrow's FixedSizeBinary values in a padded column satisfy this because
    /// the buffer is 64-byte aligned and each entry is 1280 bytes (divisible by 8).
    #[inline]
    pub unsafe fn from_bytes_unchecked(bytes: &'a [u8]) -> Self {
        debug_assert!(bytes.len() >= VECTOR_BYTES);
        debug_assert!(bytes.as_ptr() as usize % 8 == 0,
            "VectorSlice requires 8-byte alignment");
        let ptr = bytes.as_ptr() as *const u64;
        let words = unsafe { std::slice::from_raw_parts(ptr, VECTOR_WORDS) };
        Self { words }
    }

    /// Safe creation from bytes — checks alignment and length, falls back to copy.
    ///
    /// Prefers zero-copy reinterpret but will copy if alignment is wrong.
    /// With PADDED_VECTOR_BYTES columns this should never copy.
    pub fn from_bytes_or_copy(bytes: &'a [u8]) -> std::result::Result<Self, BitpackedVector> {
        if bytes.len() < VECTOR_BYTES {
            return Err(BitpackedVector::zero());
        }
        if bytes.as_ptr() as usize % 8 == 0 {
            // Zero-copy path: pointer is already u64-aligned
            Ok(unsafe { Self::from_bytes_unchecked(bytes) })
        } else {
            // Fallback: misaligned, must copy (should never happen with padded Arrow)
            Err(BitpackedVector::from_bytes(bytes).unwrap_or_else(|_| BitpackedVector::zero()))
        }
    }

    /// Get the underlying word slice
    #[inline]
    pub fn as_words(&self) -> &'a [u64] {
        self.words
    }
}

impl<'a> VectorRef for VectorSlice<'a> {
    #[inline]
    fn words(&self) -> &[u64] {
        self.words
    }
}

impl<'a> fmt::Debug for VectorSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorSlice({} words, {} set)",
            VECTOR_WORDS, self.popcount())
    }
}

// =========================================================================
// GENERIC WORD-LEVEL OPERATIONS ON VectorRef
// =========================================================================

/// XOR two VectorRef into a new owned vector
#[inline]
pub fn xor_ref(a: &dyn VectorRef, b: &dyn VectorRef) -> BitpackedVector {
    let aw = a.words();
    let bw = b.words();
    let mut result = [0u64; VECTOR_WORDS];
    for i in 0..VECTOR_WORDS {
        result[i] = aw[i] ^ bw[i];
    }
    BitpackedVector::from_words(result)
}

// =========================================================================
// PADDED BYTE CONVERSION
// =========================================================================

impl BitpackedVector {
    /// Convert to padded bytes for Arrow storage.
    ///
    /// Returns 1280 bytes (160 words) with 3 trailing zero-words.
    /// Use this instead of `to_bytes()` when building Arrow columns
    /// with `FixedSizeBinary(1280)`.
    pub fn to_padded_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PADDED_VECTOR_BYTES);
        for word in &self.words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        // 3 padding words of zeros (157..160)
        bytes.resize(PADDED_VECTOR_BYTES, 0);
        bytes
    }

    /// Create from padded bytes (1280 bytes), ignoring padding words.
    pub fn from_padded_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < VECTOR_BYTES {
            return Err(HdrError::DimensionMismatch {
                expected: PADDED_VECTOR_BYTES,
                got: bytes.len(),
            });
        }
        // Only read the first 157 words, ignore padding
        let mut words = [0u64; VECTOR_WORDS];
        for (i, word) in words.iter_mut().enumerate() {
            let start = i * 8;
            *word = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        }
        Ok(Self::from_words(words))
    }
}

// =========================================================================
// TRAIT IMPLEMENTATIONS
// =========================================================================

impl BitXor for BitpackedVector {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(&rhs)
    }
}

impl BitXor for &BitpackedVector {
    type Output = BitpackedVector;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

impl BitAnd for BitpackedVector {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(&rhs)
    }
}

impl BitOr for BitpackedVector {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(&rhs)
    }
}

impl Not for BitpackedVector {
    type Output = Self;

    fn not(self) -> Self::Output {
        BitpackedVector::not(&self)
    }
}

impl fmt::Debug for BitpackedVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitpackedVector({} bits, {} set, density={:.3})",
            VECTOR_BITS,
            self.popcount(),
            self.density()
        )
    }
}

impl fmt::Display for BitpackedVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show first and last few words in hex
        write!(f, "Vec10K[{:016x}...{:016x}]",
            self.words[0],
            self.words[VECTOR_WORDS - 1]
        )
    }
}

// =========================================================================
// SERDE SUPPORT
// =========================================================================

impl serde::Serialize for BitpackedVector {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.words.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for BitpackedVector {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<u64>::deserialize(deserializer)?;
        if vec.len() != VECTOR_WORDS {
            return Err(serde::de::Error::custom(
                format!("expected {} words, got {}", VECTOR_WORDS, vec.len())
            ));
        }
        let mut words = [0u64; VECTOR_WORDS];
        words.copy_from_slice(&vec);
        Ok(Self::from_words(words))
    }
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_and_ones() {
        let zero = BitpackedVector::zero();
        assert_eq!(zero.popcount(), 0);

        let ones = BitpackedVector::ones();
        assert_eq!(ones.popcount() as usize, VECTOR_BITS);
    }

    #[test]
    fn test_bit_operations() {
        let mut v = BitpackedVector::zero();

        v.set_bit(0, true);
        assert!(v.get_bit(0));
        assert!(!v.get_bit(1));

        v.set_bit(9999, true);
        assert!(v.get_bit(9999));

        v.toggle_bit(0);
        assert!(!v.get_bit(0));

        assert_eq!(v.popcount(), 1);
    }

    #[test]
    fn test_xor_self_inverse() {
        let a = BitpackedVector::random(12345);
        let b = BitpackedVector::random(67890);

        // a ⊕ b ⊕ b = a
        let bound = a.xor(&b);
        let recovered = bound.xor(&b);
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_stacked_popcount() {
        let v = BitpackedVector::random(42);
        let stacked = v.stacked_popcount();

        // Sum of stacked should equal total popcount
        let total: u32 = stacked.iter().map(|&c| c as u32).sum();
        assert_eq!(total, v.popcount());

        // Each word should have at most 64 bits set
        for count in stacked {
            assert!(count <= 64);
        }
    }

    #[test]
    fn test_bundle_majority() {
        // Create 3 vectors, 2 with bit 0 set
        let mut v1 = BitpackedVector::zero();
        let mut v2 = BitpackedVector::zero();
        let v3 = BitpackedVector::zero();

        v1.set_bit(0, true);
        v2.set_bit(0, true);

        let bundled = BitpackedVector::bundle(&[&v1, &v2, &v3]);
        assert!(bundled.get_bit(0)); // Majority says yes
    }

    #[test]
    fn test_random_density() {
        let v = BitpackedVector::random(999);
        let density = v.density();

        // Random vectors should have ~50% density
        assert!(density > 0.4 && density < 0.6,
            "Density {} outside expected range", density);
    }

    #[test]
    fn test_from_bytes_roundtrip() {
        let original = BitpackedVector::random(42);
        let bytes = original.to_bytes();
        let recovered = BitpackedVector::from_bytes(&bytes).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_last_word_mask() {
        let mut v = BitpackedVector::ones();

        // Only VECTOR_BITS should be set
        assert_eq!(v.popcount() as usize, VECTOR_BITS);

        // Bits beyond VECTOR_BITS should be 0
        let last_word = v.words[VECTOR_WORDS - 1];
        assert_eq!(last_word, LAST_WORD_MASK);
    }

    // =====================================================================
    // ZERO-COPY / ALIGNMENT TESTS
    // =====================================================================

    #[test]
    fn test_padded_constants() {
        // 160 words = 20 cache lines
        assert_eq!(PADDED_VECTOR_WORDS, 160);
        // 1280 bytes, divisible by 64
        assert_eq!(PADDED_VECTOR_BYTES, 1280);
        assert_eq!(PADDED_VECTOR_BYTES % 64, 0);
        // Padded > unpadded
        assert!(PADDED_VECTOR_BYTES > VECTOR_BYTES);
        assert_eq!(PADDED_VECTOR_BYTES - VECTOR_BYTES, 24); // 3 words padding
    }

    #[test]
    fn test_padded_bytes_roundtrip() {
        let original = BitpackedVector::random(42);
        let padded = original.to_padded_bytes();
        assert_eq!(padded.len(), PADDED_VECTOR_BYTES);

        // Padding words must be zero
        for byte in &padded[VECTOR_BYTES..] {
            assert_eq!(*byte, 0);
        }

        let recovered = BitpackedVector::from_padded_bytes(&padded).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_vector_slice_from_words() {
        let v = BitpackedVector::random(123);
        let words = v.words();
        let slice = VectorSlice::from_words(words);

        // VectorRef trait: popcount must match
        assert_eq!(VectorRef::popcount(&slice), v.popcount());
        assert_eq!(VectorRef::density(&slice), v.density());

        // to_owned_vector must be identical
        let owned = slice.to_owned_vector();
        assert_eq!(owned, v);
    }

    #[test]
    fn test_vector_slice_from_padded_bytes() {
        let v = BitpackedVector::random(456);
        let padded = v.to_padded_bytes();

        // Simulate what Arrow does: provide aligned byte slice
        // Vec<u8> is 1-byte aligned, but we can check the safe fallback
        match VectorSlice::from_bytes_or_copy(&padded) {
            Ok(slice) => {
                assert_eq!(VectorRef::popcount(&slice), v.popcount());
                assert_eq!(slice.to_owned_vector(), v);
            }
            Err(owned) => {
                // Fallback path: copied but still correct
                assert_eq!(owned, v);
            }
        }
    }

    #[test]
    fn test_xor_ref_matches_xor() {
        let a = BitpackedVector::random(10);
        let b = BitpackedVector::random(20);

        let xor_owned = a.xor(&b);
        let xor_via_ref = xor_ref(&a as &dyn VectorRef, &b as &dyn VectorRef);

        assert_eq!(xor_owned, xor_via_ref);
    }

    #[test]
    fn test_vector_ref_polymorphism() {
        // Owned BitpackedVector and borrowed VectorSlice should give
        // identical results through VectorRef trait
        let v = BitpackedVector::random(789);
        let words = v.words();
        let slice = VectorSlice::from_words(words);

        let owned_pc = VectorRef::popcount(&v);
        let slice_pc = VectorRef::popcount(&slice);
        assert_eq!(owned_pc, slice_pc);

        let owned_stacked = VectorRef::stacked_popcount(&v);
        let slice_stacked = VectorRef::stacked_popcount(&slice);
        assert_eq!(owned_stacked, slice_stacked);
    }
}
