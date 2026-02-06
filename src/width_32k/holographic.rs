//! 3D Holographic Memory — XYZ Binding, Probing, and Superposition
//!
//! The core of the 512-word design: three 8K dimensions create a holographic
//! memory with 512 billion addressable data points via XOR superposition.

use super::*;

/// A 512-word 3D holographic vector.
#[repr(align(64))]
#[derive(Clone)]
pub struct HoloVector {
    pub words: [u64; VECTOR_WORDS],
}

/// A single dimension slice (128 words = 8K bits).
pub type DimSlice = [u64; DIM_WORDS];

/// Result of a holographic probe: recovered dimension + noise estimate.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    /// The recovered dimension vector (128 words)
    pub recovered: Vec<u64>,
    /// Estimated signal-to-noise ratio (higher = cleaner recovery)
    pub snr_estimate: f64,
}

/// A holographic trace: the XOR binding of three dimension vectors.
/// Multiple traces can be superposed (bundled) into a single HoloVector.
#[derive(Clone, Debug)]
pub struct HoloTrace {
    /// The XOR-bound trace: X ⊕ Y ⊕ Z (128 words, lives in any one dimension)
    pub binding: Vec<u64>,
}

impl HoloVector {
    /// Create a zero vector.
    pub fn zero() -> Self {
        Self { words: [0u64; VECTOR_WORDS] }
    }

    /// Create from raw words.
    pub fn from_words(words: [u64; VECTOR_WORDS]) -> Self {
        Self { words }
    }

    // ========================================================================
    // DIMENSION ACCESS
    // ========================================================================

    /// Get the X dimension (content/what): words 0-127
    pub fn x(&self) -> &[u64] {
        &self.words[X_START..X_END]
    }

    /// Get the Y dimension (context/where): words 128-255
    pub fn y(&self) -> &[u64] {
        &self.words[Y_START..Y_END]
    }

    /// Get the Z dimension (relation/how): words 256-383
    pub fn z(&self) -> &[u64] {
        &self.words[Z_START..Z_END]
    }

    /// Get the metadata block: words 384-511
    pub fn meta(&self) -> &[u64] {
        &self.words[META_START..META_END]
    }

    /// Mutable X dimension
    pub fn x_mut(&mut self) -> &mut [u64] {
        &mut self.words[X_START..X_END]
    }

    /// Mutable Y dimension
    pub fn y_mut(&mut self) -> &mut [u64] {
        &mut self.words[Y_START..Y_END]
    }

    /// Mutable Z dimension
    pub fn z_mut(&mut self) -> &mut [u64] {
        &mut self.words[Z_START..Z_END]
    }

    /// Mutable metadata
    pub fn meta_mut(&mut self) -> &mut [u64] {
        &mut self.words[META_START..META_END]
    }

    /// Set X dimension from a slice
    pub fn set_x(&mut self, src: &[u64]) {
        let len = src.len().min(DIM_WORDS);
        self.words[X_START..X_START + len].copy_from_slice(&src[..len]);
    }

    /// Set Y dimension from a slice
    pub fn set_y(&mut self, src: &[u64]) {
        let len = src.len().min(DIM_WORDS);
        self.words[Y_START..Y_START + len].copy_from_slice(&src[..len]);
    }

    /// Set Z dimension from a slice
    pub fn set_z(&mut self, src: &[u64]) {
        let len = src.len().min(DIM_WORDS);
        self.words[Z_START..Z_START + len].copy_from_slice(&src[..len]);
    }

    // ========================================================================
    // HOLOGRAPHIC BINDING
    // ========================================================================

    /// Create a holographic trace by XOR-binding X, Y, Z dimensions.
    ///
    /// The trace is a 128-word vector that encodes the association
    /// (content, context, relation). It can be stored in any dimension
    /// slot or superposed with other traces.
    pub fn bind_xyz(&self) -> HoloTrace {
        let mut binding = vec![0u64; DIM_WORDS];
        for i in 0..DIM_WORDS {
            binding[i] = self.words[X_START + i]
                ^ self.words[Y_START + i]
                ^ self.words[Z_START + i];
        }
        HoloTrace { binding }
    }

    /// Bind two specific dimensions (for partial association).
    pub fn bind_xy(&self) -> Vec<u64> {
        let mut result = vec![0u64; DIM_WORDS];
        for i in 0..DIM_WORDS {
            result[i] = self.words[X_START + i] ^ self.words[Y_START + i];
        }
        result
    }

    /// Bind X and Z dimensions.
    pub fn bind_xz(&self) -> Vec<u64> {
        let mut result = vec![0u64; DIM_WORDS];
        for i in 0..DIM_WORDS {
            result[i] = self.words[X_START + i] ^ self.words[Z_START + i];
        }
        result
    }

    /// Bind Y and Z dimensions.
    pub fn bind_yz(&self) -> Vec<u64> {
        let mut result = vec![0u64; DIM_WORDS];
        for i in 0..DIM_WORDS {
            result[i] = self.words[Y_START + i] ^ self.words[Z_START + i];
        }
        result
    }

    // ========================================================================
    // HOLOGRAPHIC PROBING (Recovery)
    // ========================================================================

    /// Probe: given a trace and X + Y, recover Z.
    ///
    /// `recovered_z = trace ⊕ x ⊕ y`
    pub fn probe_for_z(trace: &[u64], x: &[u64], y: &[u64]) -> ProbeResult {
        Self::probe_recover(trace, &[x, y])
    }

    /// Probe: given a trace and X + Z, recover Y.
    pub fn probe_for_y(trace: &[u64], x: &[u64], z: &[u64]) -> ProbeResult {
        Self::probe_recover(trace, &[x, z])
    }

    /// Probe: given a trace and Y + Z, recover X.
    pub fn probe_for_x(trace: &[u64], y: &[u64], z: &[u64]) -> ProbeResult {
        Self::probe_recover(trace, &[y, z])
    }

    /// General probe: XOR the trace with all known dimensions to recover the unknown.
    fn probe_recover(trace: &[u64], known: &[&[u64]]) -> ProbeResult {
        let len = trace.len().min(DIM_WORDS);
        let mut recovered = vec![0u64; DIM_WORDS];
        for i in 0..len {
            let mut val = trace[i];
            for dim in known {
                if i < dim.len() {
                    val ^= dim[i];
                }
            }
            recovered[i] = val;
        }
        // SNR estimate: popcount of recovered / expected random
        // Higher popcount variance from 50% indicates stronger signal
        let total_bits: u32 = recovered.iter().map(|w| w.count_ones()).sum();
        let expected = (DIM_BITS / 2) as f64;
        let deviation = (total_bits as f64 - expected).abs();
        let snr = deviation / DIM_SIGMA;
        ProbeResult {
            recovered,
            snr_estimate: snr,
        }
    }

    // ========================================================================
    // SUPERPOSITION (Bundling Multiple Traces)
    // ========================================================================

    /// Bundle multiple traces via majority vote into a single superposition.
    ///
    /// Each trace is 128 words. The result has bit i set to 1 if more than
    /// half the traces have bit i set. This is the holographic equivalent
    /// of storing multiple associations in one vector.
    pub fn bundle_traces(traces: &[HoloTrace]) -> HoloTrace {
        if traces.is_empty() {
            return HoloTrace { binding: vec![0u64; DIM_WORDS] };
        }
        if traces.len() == 1 {
            return traces[0].clone();
        }

        let threshold = traces.len() / 2;
        let mut result = vec![0u64; DIM_WORDS];

        for word_idx in 0..DIM_WORDS {
            let mut result_word = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count = traces.iter()
                    .filter(|t| t.binding.get(word_idx).copied().unwrap_or(0) & mask != 0)
                    .count();
                if count > threshold {
                    result_word |= mask;
                }
            }
            result[word_idx] = result_word;
        }

        HoloTrace { binding: result }
    }

    // ========================================================================
    // DISTANCE (per dimension and full)
    // ========================================================================

    /// Hamming distance across X dimension only (content similarity).
    pub fn distance_x(&self, other: &Self) -> u32 {
        dim_hamming(&self.words[X_START..X_END], &other.words[X_START..X_END])
    }

    /// Hamming distance across Y dimension only (context similarity).
    pub fn distance_y(&self, other: &Self) -> u32 {
        dim_hamming(&self.words[Y_START..Y_END], &other.words[Y_START..Y_END])
    }

    /// Hamming distance across Z dimension only (relation similarity).
    pub fn distance_z(&self, other: &Self) -> u32 {
        dim_hamming(&self.words[Z_START..Z_END], &other.words[Z_START..Z_END])
    }

    /// Full semantic distance (all three dimensions, excluding metadata).
    pub fn distance_semantic(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..Z_END {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Total Hamming distance (all 512 words including metadata).
    pub fn distance_total(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..VECTOR_WORDS {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Composite distance with per-dimension weights.
    ///
    /// Returns `wx * dist_x + wy * dist_y + wz * dist_z`
    pub fn distance_weighted(&self, other: &Self, wx: f64, wy: f64, wz: f64) -> f64 {
        wx * self.distance_x(other) as f64
            + wy * self.distance_y(other) as f64
            + wz * self.distance_z(other) as f64
    }

    // ========================================================================
    // XOR OPERATIONS (full vector)
    // ========================================================================

    /// XOR bind two HoloVectors (all 512 words).
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = self.words[i] ^ other.words[i];
        }
        Self { words: result }
    }

    /// XOR delta between two vectors (for ConcurrentWriteCache).
    pub fn xor_delta(&self, other: &Self) -> Self {
        self.bind(other)
    }

    /// Apply an XOR delta to produce an updated vector.
    pub fn apply_delta(&self, delta: &Self) -> Self {
        self.bind(delta)
    }

    /// Popcount of entire vector.
    pub fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Popcount of a single dimension.
    pub fn popcount_dim(&self, dim_start: usize) -> u32 {
        self.words[dim_start..dim_start + DIM_WORDS]
            .iter()
            .map(|w| w.count_ones())
            .sum()
    }
}

/// Hamming distance between two dimension slices.
fn dim_hamming(a: &[u64], b: &[u64]) -> u32 {
    let mut dist = 0u32;
    let len = a.len().min(b.len());
    for i in 0..len {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

impl HoloTrace {
    /// Hamming distance between two traces.
    pub fn distance(&self, other: &Self) -> u32 {
        dim_hamming(&self.binding, &other.binding)
    }

    /// Popcount of the trace.
    pub fn popcount(&self) -> u32 {
        self.binding.iter().map(|w| w.count_ones()).sum()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic RNG for tests
    fn test_rng(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            if state == 0 { state = 1; }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// Fill a dimension with pseudo-random data
    fn random_dim(rng: &mut impl FnMut() -> u64) -> Vec<u64> {
        (0..DIM_WORDS).map(|_| rng()).collect()
    }

    /// Fill a full HoloVector with pseudo-random data
    fn random_holo(seed: u64) -> HoloVector {
        let mut rng = test_rng(seed);
        let mut v = HoloVector::zero();
        for w in v.words.iter_mut() {
            *w = rng();
        }
        v
    }

    #[test]
    fn test_zero_vector() {
        let v = HoloVector::zero();
        assert_eq!(v.popcount(), 0);
        assert_eq!(v.distance_total(&v), 0);
    }

    #[test]
    fn test_dimension_access() {
        let mut v = HoloVector::zero();
        // Set first word of each dimension
        v.x_mut()[0] = 0xDEAD;
        v.y_mut()[0] = 0xBEEF;
        v.z_mut()[0] = 0xCAFE;
        v.meta_mut()[0] = 0xF00D;

        assert_eq!(v.x()[0], 0xDEAD);
        assert_eq!(v.y()[0], 0xBEEF);
        assert_eq!(v.z()[0], 0xCAFE);
        assert_eq!(v.meta()[0], 0xF00D);

        // Verify they're in the right word positions
        assert_eq!(v.words[0], 0xDEAD);
        assert_eq!(v.words[128], 0xBEEF);
        assert_eq!(v.words[256], 0xCAFE);
        assert_eq!(v.words[384], 0xF00D);
    }

    #[test]
    fn test_bind_xyz_is_xor() {
        let v = random_holo(42);
        let trace = v.bind_xyz();

        // Verify trace = X ⊕ Y ⊕ Z
        for i in 0..DIM_WORDS {
            let expected = v.x()[i] ^ v.y()[i] ^ v.z()[i];
            assert_eq!(trace.binding[i], expected, "word {} mismatch", i);
        }
    }

    #[test]
    fn test_holographic_recovery_perfect() {
        // With a single trace (no noise), recovery should be EXACT
        let v = random_holo(123);
        let trace = v.bind_xyz();

        // Recover Z given X and Y
        let result = HoloVector::probe_for_z(&trace.binding, v.x(), v.y());
        for i in 0..DIM_WORDS {
            assert_eq!(result.recovered[i], v.z()[i],
                "Perfect recovery of Z failed at word {}", i);
        }

        // Recover Y given X and Z
        let result = HoloVector::probe_for_y(&trace.binding, v.x(), v.z());
        for i in 0..DIM_WORDS {
            assert_eq!(result.recovered[i], v.y()[i],
                "Perfect recovery of Y failed at word {}", i);
        }

        // Recover X given Y and Z
        let result = HoloVector::probe_for_x(&trace.binding, v.y(), v.z());
        for i in 0..DIM_WORDS {
            assert_eq!(result.recovered[i], v.x()[i],
                "Perfect recovery of X failed at word {}", i);
        }
    }

    #[test]
    fn test_superposition_recovery_with_noise() {
        // Store 5 traces via bundling. Recovery is approximate (noisy).
        let mut rng = test_rng(999);
        let traces: Vec<_> = (0..5).map(|i| {
            let v = random_holo(100 + i);
            v.bind_xyz()
        }).collect();

        let superposition = HoloVector::bundle_traces(&traces);

        // Probe with the first vector's X and Y to recover its Z
        let v0 = random_holo(100);
        let result = HoloVector::probe_for_z(
            &superposition.binding, v0.x(), v0.y()
        );

        // With 5 traces in 8K bits, recovery should be noisy but correlated
        let expected_z = v0.z();
        let mut matching_bits = 0u32;
        for i in 0..DIM_WORDS {
            matching_bits += (!(result.recovered[i] ^ expected_z[i])).count_ones();
        }
        let match_rate = matching_bits as f64 / DIM_BITS as f64;
        // With 5 traces, majority vote gives >60% bit accuracy
        assert!(match_rate > 0.55,
            "Superposition recovery too noisy: {:.1}% matching", match_rate * 100.0);
    }

    #[test]
    fn test_per_dimension_distance() {
        let a = random_holo(10);
        let b = random_holo(20);

        let dx = a.distance_x(&b);
        let dy = a.distance_y(&b);
        let dz = a.distance_z(&b);

        // Each dimension distance should be roughly DIM_BITS/2 for random vectors
        let expected = DIM_BITS as u32 / 2;
        let tolerance = 3 * DIM_SIGMA_APPROX; // 3 sigma

        assert!((dx as i64 - expected as i64).unsigned_abs() < tolerance as u64,
            "X distance {} far from expected {}", dx, expected);
        assert!((dy as i64 - expected as i64).unsigned_abs() < tolerance as u64,
            "Y distance {} far from expected {}", dy, expected);
        assert!((dz as i64 - expected as i64).unsigned_abs() < tolerance as u64,
            "Z distance {} far from expected {}", dz, expected);
    }

    #[test]
    fn test_weighted_distance() {
        let a = random_holo(30);
        let b = random_holo(40);

        // Content-only distance
        let content_dist = a.distance_weighted(&b, 1.0, 0.0, 0.0);
        assert_eq!(content_dist, a.distance_x(&b) as f64);

        // Equal weight
        let equal_dist = a.distance_weighted(&b, 1.0, 1.0, 1.0);
        let sum = a.distance_x(&b) as f64 + a.distance_y(&b) as f64 + a.distance_z(&b) as f64;
        assert!((equal_dist - sum).abs() < 1e-10);
    }

    #[test]
    fn test_xor_delta_roundtrip() {
        let original = random_holo(50);
        let modified = random_holo(60);

        let delta = original.xor_delta(&modified);
        let recovered = original.apply_delta(&delta);

        assert_eq!(recovered.distance_total(&modified), 0,
            "XOR delta roundtrip failed");
    }

    #[test]
    fn test_self_distance_is_zero() {
        let v = random_holo(70);
        assert_eq!(v.distance_x(&v), 0);
        assert_eq!(v.distance_y(&v), 0);
        assert_eq!(v.distance_z(&v), 0);
        assert_eq!(v.distance_semantic(&v), 0);
        assert_eq!(v.distance_total(&v), 0);
    }

    #[test]
    fn test_semantic_distance_excludes_metadata() {
        let mut a = random_holo(80);
        let mut b = a.clone();

        // Modify only metadata — semantic distance should stay 0
        b.meta_mut()[0] ^= 0xFFFF_FFFF_FFFF_FFFF;
        b.meta_mut()[50] ^= 0xFFFF_FFFF_FFFF_FFFF;

        assert_eq!(a.distance_semantic(&b), 0,
            "Metadata change affected semantic distance");
        assert!(a.distance_total(&b) > 0,
            "Metadata change should affect total distance");
    }

    #[test]
    fn test_bind_partial_xy_xz_yz() {
        let v = random_holo(90);

        let xy = v.bind_xy();
        let xz = v.bind_xz();
        let yz = v.bind_yz();

        // Verify xy = X ⊕ Y
        for i in 0..DIM_WORDS {
            assert_eq!(xy[i], v.x()[i] ^ v.y()[i]);
            assert_eq!(xz[i], v.x()[i] ^ v.z()[i]);
            assert_eq!(yz[i], v.y()[i] ^ v.z()[i]);
        }
    }

    #[test]
    fn test_analogical_reasoning() {
        // The classic: king - male + female ≈ queen
        // In XYZ: X=entity, Y=context, Z=gender
        let mut king = HoloVector::zero();
        let mut queen = HoloVector::zero();
        let mut rng = test_rng(200);

        // Shared royalty content
        let royalty: Vec<u64> = random_dim(&mut rng);
        king.set_x(&royalty);
        queen.set_x(&royalty);

        // Shared throne context
        let throne: Vec<u64> = random_dim(&mut rng);
        king.set_y(&throne);
        queen.set_y(&throne);

        // Different gender dimension
        let male: Vec<u64> = random_dim(&mut rng);
        let female: Vec<u64> = random_dim(&mut rng);
        king.set_z(&male);
        queen.set_z(&female);

        // Analogy probe: given king's trace, replace male Z with female Z
        // king_trace ⊕ male ⊕ female should ≈ queen's trace
        let king_trace = king.bind_xyz();
        let mut analogy = vec![0u64; DIM_WORDS];
        for i in 0..DIM_WORDS {
            analogy[i] = king_trace.binding[i] ^ male[i] ^ female[i];
        }
        let queen_trace = queen.bind_xyz();

        // The analogy trace should be identical to queen's trace
        // because: (royalty ⊕ throne ⊕ male) ⊕ male ⊕ female
        //        = royalty ⊕ throne ⊕ female
        //        = queen_trace
        for i in 0..DIM_WORDS {
            assert_eq!(analogy[i], queen_trace.binding[i],
                "Analogical reasoning failed at word {}", i);
        }
    }

    #[test]
    fn test_holographic_capacity_bound() {
        // Store N traces and verify recovery degrades gracefully
        let mut match_rates = Vec::new();

        for n in [1, 5, 10, 30, 50, 90] {
            let traces: Vec<_> = (0..n).map(|i| {
                random_holo(1000 + i as u64).bind_xyz()
            }).collect();

            let superposition = HoloVector::bundle_traces(&traces);

            // Probe for the first trace
            let v0 = random_holo(1000);
            let result = HoloVector::probe_for_z(
                &superposition.binding, v0.x(), v0.y()
            );

            let expected_z = v0.z();
            let matching: u32 = (0..DIM_WORDS)
                .map(|i| (!(result.recovered[i] ^ expected_z[i])).count_ones())
                .sum();
            let rate = matching as f64 / DIM_BITS as f64;
            match_rates.push((n, rate));
        }

        // Single trace should be perfect
        assert!(match_rates[0].1 > 0.99, "Single trace recovery should be ~100%");
        // Recovery should degrade as traces increase
        for i in 1..match_rates.len() {
            // Allow some noise but trend should be downward
            if match_rates[i].0 > 10 {
                assert!(match_rates[i].1 < match_rates[0].1,
                    "Recovery should degrade with more traces");
            }
        }
    }

    #[test]
    fn test_set_dimension() {
        let mut v = HoloVector::zero();
        let data = vec![0xAAAA_BBBB_CCCC_DDDDu64; DIM_WORDS];

        v.set_x(&data);
        assert_eq!(v.x()[0], 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(v.x()[127], 0xAAAA_BBBB_CCCC_DDDD);
        // Y should still be zero
        assert_eq!(v.y()[0], 0);
    }
}
