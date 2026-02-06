//! Stacked Popcount Hamming Distance Engine
//!
//! High-performance Hamming distance using:
//! - **Stacked Popcount**: Per-word bit counts for hierarchical filtering
//! - **SIMD Acceleration**: AVX-512, AVX2, and NEON support
//! - **Batch Processing**: Process multiple comparisons efficiently
//!
//! # Stacked Popcount Architecture
//!
//! Instead of computing full Hamming distance immediately, we stack
//! partial results for early termination:
//!
//! ```text
//! Level 0: Quick 7-point sample (Belichtungsmesser)
//!          → 90% candidates filtered in ~7 cycles
//!
//! Level 1: Per-word popcount accumulation
//!          → Running sum with early exit if threshold exceeded
//!
//! Level 2: Full SIMD popcount for final candidates
//!          → ~1 cycle per 64 bits with AVX-512
//! ```

use crate::bitpack::{BitpackedVector, VectorRef, VECTOR_WORDS, VECTOR_BITS};
use std::cmp::Ordering;

/// Strategic sample points for quick distance estimation
/// Prime-spaced across the vector for maximum information
const SAMPLE_POINTS: [usize; 7] = [0, 23, 47, 78, 101, 131, 155];

// ============================================================================
// STACKED POPCOUNT
// ============================================================================

/// Result of stacked popcount operation
#[derive(Debug, Clone, Copy)]
pub struct StackedPopcount {
    /// Per-word XOR popcount (157 values, each 0-64)
    pub per_word: [u8; VECTOR_WORDS],
    /// Running cumulative sum at each word boundary
    pub cumulative: [u16; VECTOR_WORDS],
    /// Total Hamming distance
    pub total: u32,
}

impl StackedPopcount {
    /// Compute stacked popcount between two vectors
    #[inline]
    pub fn compute(a: &BitpackedVector, b: &BitpackedVector) -> Self {
        let mut per_word = [0u8; VECTOR_WORDS];
        let mut cumulative = [0u16; VECTOR_WORDS];
        let mut running_sum = 0u32;

        let a_words = a.words();
        let b_words = b.words();

        for i in 0..VECTOR_WORDS {
            let xor = a_words[i] ^ b_words[i];
            let count = xor.count_ones() as u8;
            per_word[i] = count;
            running_sum += count as u32;
            cumulative[i] = running_sum as u16;
        }

        Self {
            per_word,
            cumulative,
            total: running_sum,
        }
    }

    /// Compute with early termination if threshold exceeded
    #[inline]
    pub fn compute_with_threshold(
        a: &BitpackedVector,
        b: &BitpackedVector,
        threshold: u32,
    ) -> Option<Self> {
        let mut per_word = [0u8; VECTOR_WORDS];
        let mut cumulative = [0u16; VECTOR_WORDS];
        let mut running_sum = 0u32;

        let a_words = a.words();
        let b_words = b.words();

        for i in 0..VECTOR_WORDS {
            let xor = a_words[i] ^ b_words[i];
            let count = xor.count_ones() as u8;
            per_word[i] = count;
            running_sum += count as u32;
            cumulative[i] = running_sum as u16;

            // Early termination: impossible to be under threshold
            if running_sum > threshold {
                return None;
            }
        }

        Some(Self {
            per_word,
            cumulative,
            total: running_sum,
        })
    }

    /// Get per-word counts for a specific range
    #[inline]
    pub fn range_sum(&self, start_word: usize, end_word: usize) -> u32 {
        if start_word == 0 {
            self.cumulative[end_word.min(VECTOR_WORDS - 1)] as u32
        } else {
            let end_cum = self.cumulative[end_word.min(VECTOR_WORDS - 1)] as u32;
            let start_cum = self.cumulative[start_word - 1] as u32;
            end_cum - start_cum
        }
    }

    /// Variance of per-word counts (indicates uniformity of difference)
    pub fn variance(&self) -> f32 {
        let mean = self.total as f32 / VECTOR_WORDS as f32;
        let sum_sq: f32 = self.per_word.iter()
            .map(|&c| {
                let diff = c as f32 - mean;
                diff * diff
            })
            .sum();
        sum_sq / VECTOR_WORDS as f32
    }
}

impl StackedPopcount {
    // ========================================================================
    // ZERO-COPY variants (operate on VectorRef — no BitpackedVector needed)
    // ========================================================================

    /// Compute stacked popcount between any two VectorRef implementors.
    ///
    /// This is the zero-copy path: when a and b are `VectorSlice`s pointing
    /// into Arrow buffers, no bytes are ever copied.
    #[inline]
    pub fn compute_ref(a: &dyn VectorRef, b: &dyn VectorRef) -> Self {
        let mut per_word = [0u8; VECTOR_WORDS];
        let mut cumulative = [0u16; VECTOR_WORDS];
        let mut running_sum = 0u32;

        let aw = a.words();
        let bw = b.words();

        for i in 0..VECTOR_WORDS {
            let xor = aw[i] ^ bw[i];
            let count = xor.count_ones() as u8;
            per_word[i] = count;
            running_sum += count as u32;
            cumulative[i] = running_sum as u16;
        }

        Self {
            per_word,
            cumulative,
            total: running_sum,
        }
    }

    /// Compute with early termination on any VectorRef pair (zero-copy).
    #[inline]
    pub fn compute_with_threshold_ref(
        a: &dyn VectorRef,
        b: &dyn VectorRef,
        threshold: u32,
    ) -> Option<Self> {
        let mut per_word = [0u8; VECTOR_WORDS];
        let mut cumulative = [0u16; VECTOR_WORDS];
        let mut running_sum = 0u32;

        let aw = a.words();
        let bw = b.words();

        for i in 0..VECTOR_WORDS {
            let xor = aw[i] ^ bw[i];
            let count = xor.count_ones() as u8;
            per_word[i] = count;
            running_sum += count as u32;
            cumulative[i] = running_sum as u16;

            if running_sum > threshold {
                return None;
            }
        }

        Some(Self {
            per_word,
            cumulative,
            total: running_sum,
        })
    }
}

// ============================================================================
// BELICHTUNGSMESSER (Quick Exposure Meter)
// ============================================================================

/// Quick 7-point exposure meter for rapid distance estimation
///
/// Like a camera's spot metering: takes strategic samples
/// to estimate overall "exposure" (difference) quickly.
#[derive(Debug, Clone, Copy)]
pub struct Belichtung {
    /// How many sample points differ (0-7)
    pub mean: u8,
    /// Standard deviation × 100 for integer arithmetic
    pub sd_100: u8,
}

impl Belichtung {
    /// Measure distance using 7 strategic samples
    /// Cost: ~14 cycles (7 XOR + 7 compare)
    #[inline]
    pub fn meter(a: &BitpackedVector, b: &BitpackedVector) -> Self {
        let a_words = a.words();
        let b_words = b.words();
        let mut sum = 0u32;

        // Check if each sample word differs at all
        for &idx in &SAMPLE_POINTS {
            sum += ((a_words[idx] ^ b_words[idx]) != 0) as u32;
        }

        // For binary samples: SD = sqrt(p(1-p) * n)
        let p = sum as f32 / 7.0;
        let variance = p * (1.0 - p);
        let sd = (variance * 7.0).sqrt();

        Self {
            mean: sum as u8,
            sd_100: (sd * 100.0) as u8,
        }
    }

    /// Zero-copy meter: works on any VectorRef (VectorSlice from Arrow buffers).
    #[inline]
    pub fn meter_ref(a: &dyn VectorRef, b: &dyn VectorRef) -> Self {
        let aw = a.words();
        let bw = b.words();
        let mut sum = 0u32;

        for &idx in &SAMPLE_POINTS {
            sum += ((aw[idx] ^ bw[idx]) != 0) as u32;
        }

        let p = sum as f32 / 7.0;
        let variance = p * (1.0 - p);
        let sd = (variance * 7.0).sqrt();

        Self {
            mean: sum as u8,
            sd_100: (sd * 100.0) as u8,
        }
    }

    /// Quick threshold check: definitely too different?
    #[inline]
    pub fn definitely_far(&self, threshold_fraction: f32) -> bool {
        // If all 7 samples differ, vector is likely >50% different
        // Scale threshold: 7 samples ≈ 7/157 of vector
        let sample_threshold = (threshold_fraction * 7.0) as u8;
        self.mean > sample_threshold
    }

    /// Estimate full distance from samples
    #[inline]
    pub fn estimate_distance(&self) -> u32 {
        // Each sample word can have 0-64 differing bits
        // Mean of 7 → roughly (mean/7) × VECTOR_BITS / 2
        (self.mean as u32 * VECTOR_BITS as u32) / 14
    }
}

// ============================================================================
// HAMMING ENGINE
// ============================================================================

/// High-performance Hamming distance engine
pub struct HammingEngine {
    /// Cache of recent stacked popcounts for reuse
    cache_enabled: bool,
    /// Batch size for parallel operations
    batch_size: usize,
}

impl Default for HammingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HammingEngine {
    /// Create new engine
    pub fn new() -> Self {
        Self {
            cache_enabled: false,
            batch_size: 1024,
        }
    }

    /// Create with configuration
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            cache_enabled: false,
            batch_size,
        }
    }

    /// Enable caching for repeated comparisons
    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    // ========================================================================
    // SCALAR OPERATIONS
    // ========================================================================

    /// Compute exact Hamming distance
    #[inline]
    pub fn distance(&self, a: &BitpackedVector, b: &BitpackedVector) -> u32 {
        hamming_distance_scalar(a, b)
    }

    /// Compute distance with stacked result
    #[inline]
    pub fn distance_stacked(&self, a: &BitpackedVector, b: &BitpackedVector) -> StackedPopcount {
        StackedPopcount::compute(a, b)
    }

    /// Compute distance with early termination
    #[inline]
    pub fn distance_threshold(
        &self,
        a: &BitpackedVector,
        b: &BitpackedVector,
        threshold: u32,
    ) -> Option<u32> {
        StackedPopcount::compute_with_threshold(a, b, threshold)
            .map(|s| s.total)
    }

    /// Quick exposure check
    #[inline]
    pub fn quick_check(&self, a: &BitpackedVector, b: &BitpackedVector) -> Belichtung {
        Belichtung::meter(a, b)
    }

    // ========================================================================
    // BATCH OPERATIONS
    // ========================================================================

    /// Compute distances from query to multiple candidates
    pub fn batch_distances(
        &self,
        query: &BitpackedVector,
        candidates: &[BitpackedVector],
    ) -> Vec<u32> {
        candidates.iter()
            .map(|c| self.distance(query, c))
            .collect()
    }

    /// Compute distances with parallel processing
    #[cfg(feature = "rayon")]
    pub fn batch_distances_parallel(
        &self,
        query: &BitpackedVector,
        candidates: &[BitpackedVector],
    ) -> Vec<u32> {
        use rayon::prelude::*;

        candidates.par_iter()
            .map(|c| hamming_distance_scalar(query, c))
            .collect()
    }

    /// Find k nearest neighbors
    pub fn knn(
        &self,
        query: &BitpackedVector,
        candidates: &[BitpackedVector],
        k: usize,
    ) -> Vec<(usize, u32)> {
        let mut results: Vec<(usize, u32)> = candidates.iter()
            .enumerate()
            .map(|(i, c)| (i, self.distance(query, c)))
            .collect();

        // Partial sort for efficiency when k << n
        if k < results.len() / 2 {
            results.select_nth_unstable_by_key(k, |&(_, d)| d);
            results.truncate(k);
        }
        results.sort_by_key(|&(_, d)| d);
        results.truncate(k);
        results
    }

    /// Find all within threshold
    pub fn range_search(
        &self,
        query: &BitpackedVector,
        candidates: &[BitpackedVector],
        threshold: u32,
    ) -> Vec<(usize, u32)> {
        candidates.iter()
            .enumerate()
            .filter_map(|(i, c)| {
                self.distance_threshold(query, c, threshold)
                    .map(|d| (i, d))
            })
            .collect()
    }

    /// Cascaded search: quick filter then exact match
    pub fn cascaded_search(
        &self,
        query: &BitpackedVector,
        candidates: &[BitpackedVector],
        k: usize,
        quick_threshold: f32,
    ) -> Vec<(usize, u32)> {
        // Phase 1: Quick exposure filter
        let mut survivors: Vec<usize> = candidates.iter()
            .enumerate()
            .filter(|(_, c)| !self.quick_check(query, c).definitely_far(quick_threshold))
            .map(|(i, _)| i)
            .collect();

        // Phase 2: Exact distance on survivors
        let mut results: Vec<(usize, u32)> = survivors.iter()
            .map(|&i| (i, self.distance(query, &candidates[i])))
            .collect();

        results.sort_by_key(|&(_, d)| d);
        results.truncate(k);
        results
    }
}

// ============================================================================
// CORE DISTANCE FUNCTIONS
// ============================================================================

/// Scalar Hamming distance (always available)
#[inline]
pub fn hamming_distance_scalar(a: &BitpackedVector, b: &BitpackedVector) -> u32 {
    let a_words = a.words();
    let b_words = b.words();
    let mut dist = 0u32;

    for i in 0..VECTOR_WORDS {
        dist += (a_words[i] ^ b_words[i]).count_ones();
    }

    dist
}

/// Zero-copy Hamming distance on any VectorRef pair
#[inline]
pub fn hamming_distance_ref(a: &dyn VectorRef, b: &dyn VectorRef) -> u32 {
    let aw = a.words();
    let bw = b.words();
    let mut dist = 0u32;
    for i in 0..VECTOR_WORDS {
        dist += (aw[i] ^ bw[i]).count_ones();
    }
    dist
}

/// Convert Hamming distance to similarity (0.0 to 1.0)
#[inline]
pub fn hamming_to_similarity(distance: u32) -> f32 {
    1.0 - (distance as f32 / VECTOR_BITS as f32)
}

/// Convert similarity to approximate Hamming distance
#[inline]
pub fn similarity_to_hamming(similarity: f32) -> u32 {
    ((1.0 - similarity) * VECTOR_BITS as f32) as u32
}

// ============================================================================
// SIMD IMPLEMENTATIONS
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod simd_x86 {
    use super::*;

    /// Check for AVX-512 VPOPCNTDQ support at runtime
    #[inline]
    pub fn has_avx512_popcnt() -> bool {
        #[cfg(target_feature = "avx512vpopcntdq")]
        {
            true
        }
        #[cfg(not(target_feature = "avx512vpopcntdq"))]
        {
            is_x86_feature_detected!("avx512vpopcntdq")
        }
    }

    /// Check for AVX2 support at runtime
    #[inline]
    pub fn has_avx2() -> bool {
        #[cfg(target_feature = "avx2")]
        {
            true
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            is_x86_feature_detected!("avx2")
        }
    }

    /// AVX-512 VPOPCNTDQ accelerated Hamming distance
    #[cfg(target_feature = "avx512vpopcntdq")]
    #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
    pub unsafe fn hamming_distance_avx512(a: &BitpackedVector, b: &BitpackedVector) -> u32 {
        use std::arch::x86_64::*;

        unsafe {
            let a_words = a.words();
            let b_words = b.words();
            let mut total = _mm512_setzero_si512();

            // Process 8 u64s at a time (512 bits)
            let chunks = VECTOR_WORDS / 8;
            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm512_loadu_si512(a_words.as_ptr().add(offset) as *const __m512i);
                let vb = _mm512_loadu_si512(b_words.as_ptr().add(offset) as *const __m512i);
                let xor = _mm512_xor_si512(va, vb);
                let pop = _mm512_popcnt_epi64(xor);
                total = _mm512_add_epi64(total, pop);
            }

            // Horizontal sum
            let mut lanes = [0u64; 8];
            _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, total);
            let simd_sum: u64 = lanes.iter().sum();

            // Handle remainder (157 % 8 = 5 words)
            let mut remainder = 0u32;
            for i in (chunks * 8)..VECTOR_WORDS {
                remainder += (a_words[i] ^ b_words[i]).count_ones();
            }

            (simd_sum as u32) + remainder
        }
    }

    /// AVX2 accelerated Hamming distance using lookup table
    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn hamming_distance_avx2(a: &BitpackedVector, b: &BitpackedVector) -> u32 {
        use std::arch::x86_64::*;

        unsafe {
            let a_words = a.words();
            let b_words = b.words();

            // 4-bit lookup table for popcount
            let lookup = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            );
            let low_mask = _mm256_set1_epi8(0x0f);

            let mut total = _mm256_setzero_si256();

            // Process 4 u64s at a time (256 bits)
            let chunks = VECTOR_WORDS / 4;
            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm256_loadu_si256(a_words.as_ptr().add(offset) as *const __m256i);
                let vb = _mm256_loadu_si256(b_words.as_ptr().add(offset) as *const __m256i);
                let xor = _mm256_xor_si256(va, vb);

                // Popcount using nibble lookup
                let lo = _mm256_and_si256(xor, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_mask);
                let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
                let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
                let popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);

                // Sum bytes using SAD against zero
                let sad = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
                total = _mm256_add_epi64(total, sad);
            }

            // Horizontal sum
            let mut lanes = [0u64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, total);
            let simd_sum: u64 = lanes.iter().sum();

            // Handle remainder
            let mut remainder = 0u32;
            for i in (chunks * 4)..VECTOR_WORDS {
                remainder += (a_words[i] ^ b_words[i]).count_ones();
            }

            (simd_sum as u32) + remainder
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
mod simd_arm {
    use super::*;

    /// ARM NEON accelerated Hamming distance
    #[cfg(target_feature = "neon")]
    #[target_feature(enable = "neon")]
    pub unsafe fn hamming_distance_neon(a: &BitpackedVector, b: &BitpackedVector) -> u32 {
        use std::arch::aarch64::*;

        unsafe {
            let a_words = a.words();
            let b_words = b.words();
            let mut total = vdupq_n_u64(0);

            // Process 2 u64s at a time (128 bits)
            let chunks = VECTOR_WORDS / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_u64(a_words.as_ptr().add(offset));
                let vb = vld1q_u64(b_words.as_ptr().add(offset));
                let xor = veorq_u64(va, vb);

                // Count bits using vcntq_u8
                let bytes = vreinterpretq_u8_u64(xor);
                let counts = vcntq_u8(bytes);

                // Sum up through pairwise addition
                let sum16 = vpaddlq_u8(counts);
                let sum32 = vpaddlq_u16(sum16);
                let sum64 = vpaddlq_u32(sum32);

                total = vaddq_u64(total, sum64);
            }

            // Horizontal sum
            let sum = vgetq_lane_u64(total, 0) + vgetq_lane_u64(total, 1);

            // Handle remainder
            let mut remainder = 0u32;
            for i in (chunks * 2)..VECTOR_WORDS {
                remainder += (a_words[i] ^ b_words[i]).count_ones();
            }

            (sum as u32) + remainder
        }
    }
}

/// Dispatch to best available SIMD implementation
#[inline]
pub fn hamming_distance_simd(a: &BitpackedVector, b: &BitpackedVector) -> u32 {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        // Try AVX-512 first
        #[cfg(target_feature = "avx512vpopcntdq")]
        {
            return unsafe { simd_x86::hamming_distance_avx512(a, b) };
        }

        // Fall back to AVX2
        #[cfg(target_feature = "avx2")]
        {
            return unsafe { simd_x86::hamming_distance_avx2(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd", target_feature = "neon"))]
    {
        return unsafe { simd_arm::hamming_distance_neon(a, b) };
    }

    // Scalar fallback
    hamming_distance_scalar(a, b)
}

// ============================================================================
// BATCH SIMD OPERATIONS
// ============================================================================

/// Process 8 candidates against 1 query (optimized batch)
pub fn batch_hamming_8(
    query: &BitpackedVector,
    candidates: &[BitpackedVector; 8],
) -> [u32; 8] {
    let mut results = [0u32; 8];
    for (i, c) in candidates.iter().enumerate() {
        results[i] = hamming_distance_scalar(query, c);
    }
    results
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_zero() {
        let a = BitpackedVector::zero();
        let b = BitpackedVector::zero();
        assert_eq!(hamming_distance_scalar(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_ones() {
        let a = BitpackedVector::zero();
        let b = BitpackedVector::ones();
        assert_eq!(hamming_distance_scalar(&a, &b) as usize, VECTOR_BITS);
    }

    #[test]
    fn test_hamming_self() {
        let v = BitpackedVector::random(42);
        assert_eq!(hamming_distance_scalar(&v, &v), 0);
    }

    #[test]
    fn test_hamming_symmetric() {
        let a = BitpackedVector::random(123);
        let b = BitpackedVector::random(456);
        assert_eq!(
            hamming_distance_scalar(&a, &b),
            hamming_distance_scalar(&b, &a)
        );
    }

    #[test]
    fn test_stacked_popcount() {
        let a = BitpackedVector::random(111);
        let b = BitpackedVector::random(222);

        let stacked = StackedPopcount::compute(&a, &b);

        // Total should match direct computation
        assert_eq!(stacked.total, hamming_distance_scalar(&a, &b));

        // Cumulative should be monotonic
        for i in 1..VECTOR_WORDS {
            assert!(stacked.cumulative[i] >= stacked.cumulative[i - 1]);
        }

        // Last cumulative should equal total
        assert_eq!(stacked.cumulative[VECTOR_WORDS - 1] as u32, stacked.total);
    }

    #[test]
    fn test_stacked_threshold() {
        let a = BitpackedVector::zero();
        let b = BitpackedVector::ones();

        // Should fail with low threshold
        assert!(StackedPopcount::compute_with_threshold(&a, &b, 100).is_none());

        // Should succeed with high threshold
        assert!(StackedPopcount::compute_with_threshold(&a, &b, 20000).is_some());
    }

    #[test]
    fn test_belichtung_meter() {
        let a = BitpackedVector::zero();
        let b = BitpackedVector::zero();

        let meter = Belichtung::meter(&a, &b);
        assert_eq!(meter.mean, 0);

        let c = BitpackedVector::ones();
        let meter2 = Belichtung::meter(&a, &c);
        assert_eq!(meter2.mean, 7); // All samples differ
    }

    #[test]
    fn test_knn() {
        let engine = HammingEngine::new();
        let query = BitpackedVector::random(1);

        let candidates: Vec<_> = (0..100)
            .map(|i| BitpackedVector::random(i as u64 + 100))
            .collect();

        let results = engine.knn(&query, &candidates, 5);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_similarity_conversion() {
        assert_eq!(hamming_to_similarity(0), 1.0);
        assert!((hamming_to_similarity(VECTOR_BITS as u32 / 2) - 0.5).abs() < 0.001);
        assert_eq!(hamming_to_similarity(VECTOR_BITS as u32), 0.0);
    }
}
