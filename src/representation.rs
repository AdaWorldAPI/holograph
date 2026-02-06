//! Multi-Resolution HDR Representations
//!
//! Beyond binary: stacked bits, counts, and graded representations
//! for higher capacity and precision.
//!
//! # Representation Hierarchy
//!
//! ```text
//! Level 0: Binary (1-bit)     → 10K dimensions, 1.25 KB
//!          Capacity: ~50 concepts per vector
//!
//! Level 1: Ternary (-1,0,+1)  → 10K dimensions, 2.5 KB
//!          Capacity: ~100 concepts (sparse bundling)
//!
//! Level 2: Quaternary (2-bit) → 10K dimensions, 2.5 KB
//!          Values: {-2,-1,+1,+2} or {0,1,2,3}
//!          Capacity: ~200 concepts
//!
//! Level 3: Byte-wise (8-bit)  → 10K dimensions, 10 KB
//!          Full count accumulator for bundling
//!          Capacity: ~1000+ concepts
//!
//! Level 4: Stacked Binary     → N × 10K bits
//!          Multiple "planes" for hierarchical binding
//!          Unlimited capacity via plane selection
//! ```

use crate::bitpack::{BitpackedVector, VECTOR_BITS, VECTOR_WORDS};
use crate::hamming::hamming_distance_scalar;

// ============================================================================
// GRADED VECTOR (Multi-bit per dimension)
// ============================================================================

/// Number of dimensions in graded vectors
pub const GRADED_DIMS: usize = 10_000;

/// Graded vector with configurable bits per dimension
#[derive(Clone, Debug)]
pub struct GradedVector {
    /// Values per dimension (can be -128 to +127 for byte-wise)
    values: Vec<i8>,
    /// Bits per dimension (1, 2, 4, or 8)
    bits_per_dim: u8,
}

impl GradedVector {
    /// Create zero vector
    pub fn zero(bits_per_dim: u8) -> Self {
        Self {
            values: vec![0i8; GRADED_DIMS],
            bits_per_dim,
        }
    }

    /// Create from binary vector (promote 0→-1, 1→+1)
    pub fn from_binary(binary: &BitpackedVector) -> Self {
        let mut values = vec![0i8; GRADED_DIMS];
        for i in 0..GRADED_DIMS {
            values[i] = if binary.get_bit(i) { 1 } else { -1 };
        }
        Self {
            values,
            bits_per_dim: 8, // Full precision for accumulation
        }
    }

    /// Convert back to binary (threshold at 0)
    pub fn to_binary(&self) -> BitpackedVector {
        let mut binary = BitpackedVector::zero();
        for i in 0..GRADED_DIMS {
            if self.values[i] > 0 {
                binary.set_bit(i, true);
            }
        }
        binary
    }

    /// Create random bipolar vector (+1/-1)
    pub fn random_bipolar(seed: u64) -> Self {
        let binary = BitpackedVector::random(seed);
        Self::from_binary(&binary)
    }

    /// Get value at dimension
    #[inline]
    pub fn get(&self, dim: usize) -> i8 {
        self.values[dim]
    }

    /// Set value at dimension
    #[inline]
    pub fn set(&mut self, dim: usize, value: i8) {
        self.values[dim] = self.clamp_value(value);
    }

    /// Clamp value to valid range for bits_per_dim
    fn clamp_value(&self, value: i8) -> i8 {
        match self.bits_per_dim {
            1 => if value >= 0 { 1 } else { -1 },
            2 => value.clamp(-2, 2),
            4 => value.clamp(-8, 7),
            8 => value, // Full range
            _ => value,
        }
    }

    // ========================================================================
    // BINDING (Componentwise Multiply)
    // ========================================================================

    /// Bind two vectors: A ⊗ B = A * B (componentwise)
    /// In bipolar: +1 * +1 = +1, +1 * -1 = -1, etc.
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = Self::zero(self.bits_per_dim);
        for i in 0..GRADED_DIMS {
            // Sign multiplication
            let a_sign = self.values[i].signum();
            let b_sign = other.values[i].signum();
            result.values[i] = a_sign * b_sign;
        }
        result
    }

    /// Unbind (same as bind for bipolar - multiply is self-inverse for signs)
    pub fn unbind(&self, key: &Self) -> Self {
        self.bind(key)
    }

    // ========================================================================
    // BUNDLING (Componentwise Add with optional normalization)
    // ========================================================================

    /// Add another vector (for bundling)
    pub fn add(&mut self, other: &Self) {
        for i in 0..GRADED_DIMS {
            self.values[i] = self.values[i].saturating_add(other.values[i]);
        }
    }

    /// Subtract another vector
    pub fn sub(&mut self, other: &Self) {
        for i in 0..GRADED_DIMS {
            self.values[i] = self.values[i].saturating_sub(other.values[i]);
        }
    }

    /// Bundle multiple vectors with equal weight
    pub fn bundle(vectors: &[&Self]) -> Self {
        if vectors.is_empty() {
            return Self::zero(8);
        }

        let mut result = Self::zero(8);
        for v in vectors {
            result.add(v);
        }
        result
    }

    /// Bundle with weights
    pub fn bundle_weighted(vectors: &[(&Self, i8)]) -> Self {
        let mut result = Self::zero(8);
        for (v, weight) in vectors {
            for i in 0..GRADED_DIMS {
                let contribution = (v.values[i] as i16 * *weight as i16) as i8;
                result.values[i] = result.values[i].saturating_add(contribution);
            }
        }
        result
    }

    /// Normalize to bipolar (+1/-1) based on sign
    pub fn normalize(&mut self) {
        for i in 0..GRADED_DIMS {
            self.values[i] = if self.values[i] >= 0 { 1 } else { -1 };
        }
        self.bits_per_dim = 1;
    }

    /// Threshold to ternary (-1, 0, +1) with dead zone
    pub fn threshold_ternary(&mut self, threshold: i8) {
        for i in 0..GRADED_DIMS {
            self.values[i] = if self.values[i] > threshold {
                1
            } else if self.values[i] < -threshold {
                -1
            } else {
                0
            };
        }
        self.bits_per_dim = 2;
    }

    // ========================================================================
    // SIMILARITY
    // ========================================================================

    /// Dot product (sum of componentwise products)
    pub fn dot(&self, other: &Self) -> i32 {
        let mut sum = 0i32;
        for i in 0..GRADED_DIMS {
            sum += self.values[i] as i32 * other.values[i] as i32;
        }
        sum
    }

    /// Cosine similarity (normalized dot product)
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot(other) as f32;
        let norm_a = self.dot(self) as f32;
        let norm_b = other.dot(other) as f32;

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Hamming-like distance (count of sign disagreements)
    pub fn sign_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..GRADED_DIMS {
            if self.values[i].signum() != other.values[i].signum() {
                dist += 1;
            }
        }
        dist
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Count positive values
    pub fn count_positive(&self) -> usize {
        self.values.iter().filter(|&&v| v > 0).count()
    }

    /// Count negative values
    pub fn count_negative(&self) -> usize {
        self.values.iter().filter(|&&v| v < 0).count()
    }

    /// Count zeros (for sparse/ternary)
    pub fn count_zero(&self) -> usize {
        self.values.iter().filter(|&&v| v == 0).count()
    }

    /// Sparsity (fraction of zeros)
    pub fn sparsity(&self) -> f32 {
        self.count_zero() as f32 / GRADED_DIMS as f32
    }

    /// Sum of absolute values (L1 norm)
    pub fn l1_norm(&self) -> i32 {
        self.values.iter().map(|&v| v.abs() as i32).sum()
    }

    /// Sum of squares (L2 norm squared)
    pub fn l2_norm_sq(&self) -> i32 {
        self.values.iter().map(|&v| (v as i32) * (v as i32)).sum()
    }
}

// ============================================================================
// STACKED BINARY (Multiple Planes)
// ============================================================================

/// Stacked binary vectors - multiple planes for hierarchical representation
///
/// Each plane can represent different "aspects" or resolution levels.
/// Binding across planes enables complex compositional structures.
#[derive(Clone, Debug)]
pub struct StackedBinary {
    /// Multiple binary planes
    planes: Vec<BitpackedVector>,
}

impl StackedBinary {
    /// Create with N planes (all zeros)
    pub fn new(num_planes: usize) -> Self {
        Self {
            planes: vec![BitpackedVector::zero(); num_planes],
        }
    }

    /// Create from single binary vector (1 plane)
    pub fn from_binary(binary: BitpackedVector) -> Self {
        Self {
            planes: vec![binary],
        }
    }

    /// Create random stacked vector
    pub fn random(num_planes: usize, seed: u64) -> Self {
        let planes = (0..num_planes)
            .map(|i| BitpackedVector::random(seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15))))
            .collect();
        Self { planes }
    }

    /// Number of planes
    pub fn num_planes(&self) -> usize {
        self.planes.len()
    }

    /// Get plane by index
    pub fn plane(&self, idx: usize) -> Option<&BitpackedVector> {
        self.planes.get(idx)
    }

    /// Get mutable plane
    pub fn plane_mut(&mut self, idx: usize) -> Option<&mut BitpackedVector> {
        self.planes.get_mut(idx)
    }

    /// XOR bind all planes together
    pub fn collapse(&self) -> BitpackedVector {
        let mut result = BitpackedVector::zero();
        for plane in &self.planes {
            result = result.xor(plane);
        }
        result
    }

    /// Bind two stacked vectors (plane-wise XOR)
    pub fn bind(&self, other: &Self) -> Self {
        let max_planes = self.num_planes().max(other.num_planes());
        let mut planes = Vec::with_capacity(max_planes);

        for i in 0..max_planes {
            let a = self.planes.get(i).cloned().unwrap_or_else(BitpackedVector::zero);
            let b = other.planes.get(i).cloned().unwrap_or_else(BitpackedVector::zero);
            planes.push(a.xor(&b));
        }

        Self { planes }
    }

    /// Bundle stacked vectors (plane-wise majority)
    pub fn bundle(vectors: &[&Self]) -> Self {
        if vectors.is_empty() {
            return Self::new(1);
        }

        let max_planes = vectors.iter().map(|v| v.num_planes()).max().unwrap_or(1);
        let mut planes = Vec::with_capacity(max_planes);

        for plane_idx in 0..max_planes {
            let plane_vecs: Vec<&BitpackedVector> = vectors
                .iter()
                .filter_map(|v| v.planes.get(plane_idx))
                .collect();

            if plane_vecs.is_empty() {
                planes.push(BitpackedVector::zero());
            } else {
                planes.push(BitpackedVector::bundle(&plane_vecs));
            }
        }

        Self { planes }
    }

    /// Hamming distance (sum across planes)
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let max_planes = self.num_planes().max(other.num_planes());
        let mut total = 0u32;

        for i in 0..max_planes {
            let a = self.planes.get(i);
            let b = other.planes.get(i);

            match (a, b) {
                (Some(va), Some(vb)) => {
                    total += hamming_distance_scalar(va, vb);
                }
                (Some(v), None) | (None, Some(v)) => {
                    total += v.popcount();
                }
                (None, None) => {}
            }
        }

        total
    }

    /// Total bits across all planes
    pub fn total_bits(&self) -> usize {
        self.num_planes() * VECTOR_BITS
    }

    /// Total bytes
    pub fn total_bytes(&self) -> usize {
        self.num_planes() * crate::bitpack::VECTOR_BYTES
    }
}

// ============================================================================
// SPARSE HDR (High sparsity for extreme dimensions)
// ============================================================================

/// Sparse HDR vector - only stores non-zero dimensions
///
/// For very high dimensions (100K+) with low density.
#[derive(Clone, Debug)]
pub struct SparseHdr {
    /// Non-zero dimension indices
    indices: Vec<u32>,
    /// Values at those indices (+1 or -1 for bipolar)
    values: Vec<i8>,
    /// Total dimensionality
    dims: u32,
}

impl SparseHdr {
    /// Create empty sparse vector
    pub fn new(dims: u32) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            dims,
        }
    }

    /// Create with capacity
    pub fn with_capacity(dims: u32, nnz: usize) -> Self {
        Self {
            indices: Vec::with_capacity(nnz),
            values: Vec::with_capacity(nnz),
            dims,
        }
    }

    /// Create random sparse vector with given density
    pub fn random_sparse(dims: u32, density: f32, seed: u64) -> Self {
        let nnz = (dims as f32 * density) as usize;
        let mut sparse = Self::with_capacity(dims, nnz);

        // Use simple LCG for reproducibility
        let mut state = seed;
        let a = 6364136223846793005u64;
        let c = 1442695040888963407u64;

        for _ in 0..nnz {
            state = state.wrapping_mul(a).wrapping_add(c);
            let idx = (state % dims as u64) as u32;
            let val = if (state >> 32) & 1 == 0 { 1i8 } else { -1i8 };
            sparse.set(idx, val);
        }

        sparse.sort();
        sparse
    }

    /// Set value at index
    pub fn set(&mut self, idx: u32, value: i8) {
        if value != 0 && idx < self.dims {
            self.indices.push(idx);
            self.values.push(value);
        }
    }

    /// Sort by index (for efficient operations)
    pub fn sort(&mut self) {
        let mut pairs: Vec<_> = self.indices.iter()
            .zip(self.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();
        pairs.sort_by_key(|&(i, _)| i);

        // Deduplicate (keep last value for each index)
        pairs.dedup_by_key(|&mut (i, _)| i);

        self.indices = pairs.iter().map(|&(i, _)| i).collect();
        self.values = pairs.iter().map(|&(_, v)| v).collect();
    }

    /// Number of non-zeros
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Density
    pub fn density(&self) -> f32 {
        self.nnz() as f32 / self.dims as f32
    }

    /// Sparse dot product
    pub fn dot(&self, other: &Self) -> i32 {
        let mut sum = 0i32;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] == other.indices[j] {
                sum += self.values[i] as i32 * other.values[j] as i32;
                i += 1;
                j += 1;
            } else if self.indices[i] < other.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        sum
    }

    /// Convert to dense graded vector
    pub fn to_graded(&self) -> GradedVector {
        let mut graded = GradedVector::zero(8);
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if (idx as usize) < GRADED_DIMS {
                graded.values[idx as usize] = val;
            }
        }
        graded
    }
}

// ============================================================================
// REPRESENTATION INFO
// ============================================================================

/// Summary of representation capabilities
pub fn representation_summary() -> &'static str {
    r#"
HDR Representation Capabilities
===============================

Binary (Default):
  - Dimensions: 10,000 bits
  - Storage: 1,256 bytes
  - Capacity: ~50 bound concepts
  - Operations: XOR bind, majority bundle, Hamming distance
  - Speed: ~1 cycle/64 bits with SIMD

Graded (Multi-bit):
  - Dimensions: 10,000
  - Storage: 10,000 bytes (8-bit), 2,500 bytes (2-bit)
  - Capacity: 100-1000+ concepts
  - Operations: multiply bind, weighted bundle, cosine similarity
  - Precision: Accumulates without saturation

Stacked Binary:
  - Dimensions: N × 10,000 bits
  - Storage: N × 1,256 bytes
  - Capacity: Unlimited (select planes)
  - Operations: per-plane XOR, plane collapse
  - Use case: Hierarchical structures, temporal sequences

Sparse HDR:
  - Dimensions: 100K+ (configurable)
  - Storage: O(nnz) - proportional to density
  - Capacity: Very high for low-density
  - Operations: Sparse dot product, efficient for very sparse
  - Use case: Extreme dimensionality, low overlap scenarios

Key Trade-offs:
  - Binary: Fastest, most compact, limited capacity
  - Graded: Higher capacity, slower, larger storage
  - Stacked: Flexible capacity, multi-aspect encoding
  - Sparse: Highest dimensions, density-dependent performance
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graded_bind() {
        let a = GradedVector::random_bipolar(1);
        let b = GradedVector::random_bipolar(2);

        let bound = a.bind(&b);
        let recovered = bound.unbind(&b);

        // Should recover A (for bipolar, bind is self-inverse)
        assert_eq!(a.dot(&recovered), GRADED_DIMS as i32); // Perfect correlation
    }

    #[test]
    fn test_graded_bundle() {
        let v1 = GradedVector::random_bipolar(1);
        let v2 = GradedVector::random_bipolar(2);
        let v3 = GradedVector::random_bipolar(3);

        let bundled = GradedVector::bundle(&[&v1, &v2, &v3]);

        // Bundled should be closer to all inputs than random
        let random = GradedVector::random_bipolar(999);

        let sim_v1 = bundled.cosine_similarity(&v1);
        let sim_random = bundled.cosine_similarity(&random);

        assert!(sim_v1 > sim_random);
    }

    #[test]
    fn test_stacked_binary() {
        let a = StackedBinary::random(3, 100);
        let b = StackedBinary::random(3, 200);

        assert_eq!(a.num_planes(), 3);

        let bound = a.bind(&b);
        assert_eq!(bound.num_planes(), 3);

        let collapsed = bound.collapse();
        assert!(collapsed.popcount() > 0);
    }

    #[test]
    fn test_sparse_hdr() {
        let a = SparseHdr::random_sparse(100_000, 0.01, 42);
        let b = SparseHdr::random_sparse(100_000, 0.01, 43);

        // ~1% density = ~1000 non-zeros
        assert!(a.nnz() > 500 && a.nnz() < 2000);

        // Dot product of random sparse vectors should be near zero
        let dot = a.dot(&b);
        assert!(dot.abs() < 100); // Low correlation
    }

    #[test]
    fn test_binary_to_graded_roundtrip() {
        let binary = BitpackedVector::random(42);
        let graded = GradedVector::from_binary(&binary);
        let back = graded.to_binary();

        // Should be identical
        assert_eq!(hamming_distance_scalar(&binary, &back), 0);
    }
}
