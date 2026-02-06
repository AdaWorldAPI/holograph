//! GraphBLAS Vector for HDR
//!
//! Sparse vector of HDR scalars with GraphBLAS-compatible operations.

use crate::bitpack::BitpackedVector;
use super::types::{GrBIndex, HdrScalar, GrBType};
use super::sparse::SparseVec;
use super::semiring::{Semiring, HdrSemiring};

/// GraphBLAS Vector
///
/// A sparse vector where each entry is an HDR scalar.
pub struct GrBVector {
    /// Internal sparse storage
    storage: SparseVec,
    /// Element type
    dtype: GrBType,
}

impl GrBVector {
    // ========================================================================
    // CONSTRUCTION
    // ========================================================================

    /// Create empty vector
    pub fn new(len: GrBIndex) -> Self {
        Self {
            storage: SparseVec::new(len),
            dtype: GrBType::HdrVector,
        }
    }

    /// Create with capacity
    pub fn with_capacity(len: GrBIndex, nnz: usize) -> Self {
        Self {
            storage: SparseVec::with_capacity(len, nnz),
            dtype: GrBType::HdrVector,
        }
    }

    /// Create with type
    pub fn new_typed(len: GrBIndex, dtype: GrBType) -> Self {
        Self {
            storage: SparseVec::new(len),
            dtype,
        }
    }

    /// Create from dense array of vectors
    pub fn from_dense(vectors: &[BitpackedVector]) -> Self {
        let len = vectors.len() as GrBIndex;
        let mut v = Self::with_capacity(len, vectors.len());

        for (i, vec) in vectors.iter().enumerate() {
            v.set_vector(i as GrBIndex, vec.clone());
        }

        v
    }

    /// Create from sparse entries
    pub fn from_sparse(len: GrBIndex, entries: &[(GrBIndex, BitpackedVector)]) -> Self {
        let mut v = Self::with_capacity(len, entries.len());

        for (idx, vec) in entries {
            v.set_vector(*idx, vec.clone());
        }

        v
    }

    /// Create all-zeros vector (dense)
    pub fn zeros(len: GrBIndex) -> Self {
        let mut v = Self::with_capacity(len, len as usize);
        for i in 0..len {
            v.set_vector(i, BitpackedVector::zero());
        }
        v
    }

    /// Create all-ones vector (dense with ones vectors)
    pub fn ones(len: GrBIndex) -> Self {
        let mut v = Self::with_capacity(len, len as usize);
        for i in 0..len {
            v.set_vector(i, BitpackedVector::ones());
        }
        v
    }

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    /// Vector length
    pub fn len(&self) -> GrBIndex {
        self.storage.len
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Element type
    pub fn dtype(&self) -> GrBType {
        self.dtype
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.nnz() == 0
    }

    // ========================================================================
    // ELEMENT ACCESS
    // ========================================================================

    /// Get element at index
    pub fn get(&self, idx: GrBIndex) -> Option<&HdrScalar> {
        self.storage.get(idx)
    }

    /// Get as vector (convenience)
    pub fn get_vector(&self, idx: GrBIndex) -> Option<&BitpackedVector> {
        self.get(idx).and_then(|s| s.as_vector())
    }

    /// Set element at index
    pub fn set(&mut self, idx: GrBIndex, value: HdrScalar) {
        if !value.is_empty() {
            self.storage.add(idx, value);
        }
    }

    /// Set vector element
    pub fn set_vector(&mut self, idx: GrBIndex, vec: BitpackedVector) {
        self.set(idx, HdrScalar::Vector(vec));
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.storage = SparseVec::new(self.storage.len);
    }

    // ========================================================================
    // ITERATION
    // ========================================================================

    /// Iterate over (index, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (GrBIndex, &HdrScalar)> {
        self.storage.iter()
    }

    /// Get indices of non-zero elements
    pub fn indices(&self) -> &[GrBIndex] {
        &self.storage.indices
    }

    /// Get values
    pub fn values(&self) -> &[HdrScalar] {
        &self.storage.values
    }

    // ========================================================================
    // OPERATIONS
    // ========================================================================

    /// Apply unary operation
    pub fn apply<F>(&self, op: F) -> GrBVector
    where
        F: Fn(&HdrScalar) -> HdrScalar,
    {
        let mut result = GrBVector::new(self.len());

        for (idx, val) in self.iter() {
            let new_val = op(val);
            if !new_val.is_empty() {
                result.set(idx, new_val);
            }
        }

        result
    }

    /// Element-wise addition
    pub fn ewise_add(&self, other: &GrBVector, semiring: &HdrSemiring) -> GrBVector {
        assert_eq!(self.len(), other.len());

        let mut result = GrBVector::new(self.len());

        // Add entries from self
        for (idx, val) in self.iter() {
            let other_val = other.get(idx);
            let new_val = match other_val {
                Some(ov) => semiring.add(val, ov),
                None => val.clone(),
            };
            if !semiring.is_zero(&new_val) {
                result.set(idx, new_val);
            }
        }

        // Add entries from other not in self
        for (idx, val) in other.iter() {
            if self.get(idx).is_none() && !semiring.is_zero(val) {
                result.set(idx, val.clone());
            }
        }

        result
    }

    /// Element-wise multiplication
    pub fn ewise_mult(&self, other: &GrBVector, semiring: &HdrSemiring) -> GrBVector {
        assert_eq!(self.len(), other.len());

        let mut result = GrBVector::new(self.len());

        // Only entries present in both
        for (idx, val) in self.iter() {
            if let Some(other_val) = other.get(idx) {
                let new_val = semiring.multiply(val, other_val);
                if !semiring.is_zero(&new_val) {
                    result.set(idx, new_val);
                }
            }
        }

        result
    }

    /// Dot product: u · v = Σ(u_i ⊗ v_i) using semiring
    pub fn dot(&self, other: &GrBVector, semiring: &HdrSemiring) -> HdrScalar {
        assert_eq!(self.len(), other.len());

        let mut accum = semiring.zero();

        for (idx, val) in self.iter() {
            if let Some(other_val) = other.get(idx) {
                let product = semiring.multiply(val, other_val);
                accum = semiring.add(&accum, &product);
            }
        }

        accum
    }

    /// Reduce to scalar
    pub fn reduce(&self, semiring: &HdrSemiring) -> HdrScalar {
        let mut accum = semiring.zero();

        for (_, val) in self.iter() {
            accum = semiring.add(&accum, val);
        }

        accum
    }

    /// Select elements matching predicate
    pub fn select<F>(&self, predicate: F) -> GrBVector
    where
        F: Fn(GrBIndex, &HdrScalar) -> bool,
    {
        let mut result = GrBVector::new(self.len());

        for (idx, val) in self.iter() {
            if predicate(idx, val) {
                result.set(idx, val.clone());
            }
        }

        result
    }

    /// Assign to indices (scatter operation)
    pub fn assign(&mut self, indices: &[GrBIndex], values: &[HdrScalar]) {
        assert_eq!(indices.len(), values.len());

        for (&idx, val) in indices.iter().zip(values.iter()) {
            if idx < self.len() {
                self.set(idx, val.clone());
            }
        }
    }

    /// Assign scalar to indices
    pub fn assign_scalar(&mut self, indices: &[GrBIndex], value: &HdrScalar) {
        for &idx in indices {
            if idx < self.len() {
                self.set(idx, value.clone());
            }
        }
    }

    /// Extract elements at indices (gather operation)
    pub fn extract(&self, indices: &[GrBIndex]) -> GrBVector {
        let mut result = GrBVector::new(indices.len() as GrBIndex);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if let Some(val) = self.get(old_idx) {
                result.set(new_idx as GrBIndex, val.clone());
            }
        }

        result
    }

    /// Apply mask: keep only elements where mask is non-zero
    pub fn apply_mask(&self, mask: &GrBVector) -> GrBVector {
        let mut result = GrBVector::new(self.len());

        for (idx, val) in self.iter() {
            if mask.get(idx).map_or(false, |m| m.to_bool()) {
                result.set(idx, val.clone());
            }
        }

        result
    }

    /// Complement mask: keep only elements where mask is zero
    pub fn apply_complement_mask(&self, mask: &GrBVector) -> GrBVector {
        let mut result = GrBVector::new(self.len());

        for (idx, val) in self.iter() {
            if mask.get(idx).is_none() || !mask.get(idx).unwrap().to_bool() {
                result.set(idx, val.clone());
            }
        }

        result
    }

    // ========================================================================
    // HDR-SPECIFIC OPERATIONS
    // ========================================================================

    /// Bundle all vectors (majority voting)
    pub fn bundle_all(&self) -> Option<BitpackedVector> {
        let vecs: Vec<&BitpackedVector> = self.iter()
            .filter_map(|(_, val)| val.as_vector())
            .collect();

        if vecs.is_empty() {
            None
        } else {
            Some(BitpackedVector::bundle(&vecs))
        }
    }

    /// XOR all vectors
    pub fn xor_all(&self) -> BitpackedVector {
        let mut result = BitpackedVector::zero();

        for (_, val) in self.iter() {
            if let Some(vec) = val.as_vector() {
                result = result.xor(vec);
            }
        }

        result
    }

    /// Find index of most similar vector to query
    pub fn find_nearest(&self, query: &BitpackedVector) -> Option<(GrBIndex, u32)> {
        use crate::hamming::hamming_distance_scalar;

        let mut best_idx = 0;
        let mut best_dist = u32::MAX;
        let mut found = false;

        for (idx, val) in self.iter() {
            if let Some(vec) = val.as_vector() {
                let dist = hamming_distance_scalar(query, vec);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                    found = true;
                }
            }
        }

        if found {
            Some((best_idx, best_dist))
        } else {
            None
        }
    }

    /// Find all indices within distance threshold
    pub fn find_within(&self, query: &BitpackedVector, threshold: u32) -> Vec<(GrBIndex, u32)> {
        use crate::hamming::hamming_distance_scalar;

        let mut results = Vec::new();

        for (idx, val) in self.iter() {
            if let Some(vec) = val.as_vector() {
                let dist = hamming_distance_scalar(query, vec);
                if dist <= threshold {
                    results.push((idx, dist));
                }
            }
        }

        results.sort_by_key(|&(_, d)| d);
        results
    }
}

impl Clone for GrBVector {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            dtype: self.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_basic() {
        let mut v = GrBVector::new(10);

        v.set_vector(0, BitpackedVector::random(1));
        v.set_vector(5, BitpackedVector::random(2));
        v.set_vector(9, BitpackedVector::random(3));

        assert_eq!(v.nnz(), 3);
        assert!(v.get(0).is_some());
        assert!(v.get(1).is_none());
        assert!(v.get(5).is_some());
    }

    #[test]
    fn test_ewise_operations() {
        let mut u = GrBVector::new(5);
        let mut v = GrBVector::new(5);

        u.set_vector(0, BitpackedVector::random(1));
        u.set_vector(2, BitpackedVector::random(2));

        v.set_vector(1, BitpackedVector::random(3));
        v.set_vector(2, BitpackedVector::random(4));

        let semiring = HdrSemiring::XorBundle;

        // eWise add: union of indices
        let add_result = u.ewise_add(&v, &semiring);
        assert_eq!(add_result.nnz(), 3); // indices 0, 1, 2

        // eWise mult: intersection of indices
        let mult_result = u.ewise_mult(&v, &semiring);
        assert_eq!(mult_result.nnz(), 1); // only index 2
    }

    #[test]
    fn test_bundle_all() {
        let mut v = GrBVector::new(3);

        v.set_vector(0, BitpackedVector::random(1));
        v.set_vector(1, BitpackedVector::random(2));
        v.set_vector(2, BitpackedVector::random(3));

        let bundled = v.bundle_all();
        assert!(bundled.is_some());

        // Bundled should have ~50% density (random vectors)
        let density = bundled.unwrap().density();
        assert!(density > 0.4 && density < 0.6);
    }

    #[test]
    fn test_find_nearest() {
        let mut v = GrBVector::new(100);

        // Add some vectors
        for i in 0..100 {
            v.set_vector(i, BitpackedVector::random(i as u64 + 100));
        }

        // Query for a specific one
        let query = BitpackedVector::random(150); // Should match index 50

        let (idx, dist) = v.find_nearest(&query).unwrap();
        assert_eq!(idx, 50);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_reduce() {
        let mut v = GrBVector::new(3);

        v.set_vector(0, BitpackedVector::random(1));
        v.set_vector(1, BitpackedVector::random(2));
        v.set_vector(2, BitpackedVector::random(3));

        let semiring = HdrSemiring::XorBundle;
        let reduced = v.reduce(&semiring);

        // Should be a vector (bundled result)
        assert!(matches!(reduced, HdrScalar::Vector(_)));
    }
}
