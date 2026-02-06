//! GraphBLAS Matrix for HDR
//!
//! Sparse matrix of HDR vectors with GraphBLAS-compatible operations.

use std::sync::Arc;
use crate::bitpack::BitpackedVector;
use crate::{HdrError, Result};
use super::types::{GrBIndex, HdrScalar, GrBType, GRB_ALL};
use super::sparse::{CooStorage, CsrStorage, SparseFormat, SparseEntry};
use super::semiring::{Semiring, HdrSemiring};
use super::vector::GrBVector;
use super::descriptor::Descriptor;
use super::GrBInfo;

/// GraphBLAS Matrix
///
/// A sparse matrix where each entry is an HDR scalar (typically a vector).
/// Supports standard GraphBLAS operations mapped to HDR semantics.
pub struct GrBMatrix {
    /// Internal storage (COO for construction, CSR for computation)
    storage: MatrixStorage,
    /// Number of rows
    nrows: GrBIndex,
    /// Number of columns
    ncols: GrBIndex,
    /// Element type
    dtype: GrBType,
}

enum MatrixStorage {
    Coo(CooStorage),
    Csr(CsrStorage),
    Empty,
}

impl GrBMatrix {
    // ========================================================================
    // CONSTRUCTION
    // ========================================================================

    /// Create a new empty matrix
    pub fn new(nrows: GrBIndex, ncols: GrBIndex) -> Self {
        Self {
            storage: MatrixStorage::Coo(CooStorage::new(nrows, ncols)),
            nrows,
            ncols,
            dtype: GrBType::HdrVector,
        }
    }

    /// Create with type
    pub fn new_typed(nrows: GrBIndex, ncols: GrBIndex, dtype: GrBType) -> Self {
        Self {
            storage: MatrixStorage::Coo(CooStorage::new(nrows, ncols)),
            nrows,
            ncols,
            dtype,
        }
    }

    /// Create identity matrix (diagonal of zero vectors)
    pub fn identity(n: GrBIndex) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.set(i, i, HdrScalar::Vector(BitpackedVector::zero()));
        }
        m
    }

    /// Create from adjacency list
    /// Each entry (i, j, v) represents edge from i to j with vector v
    pub fn from_edges(
        nrows: GrBIndex,
        ncols: GrBIndex,
        edges: &[(GrBIndex, GrBIndex, BitpackedVector)],
    ) -> Self {
        let mut coo = CooStorage::with_capacity(nrows, ncols, edges.len());
        for (row, col, vec) in edges {
            coo.add_vector(*row, *col, vec.clone());
        }
        Self {
            storage: MatrixStorage::Coo(coo),
            nrows,
            ncols,
            dtype: GrBType::HdrVector,
        }
    }

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    /// Number of rows
    pub fn nrows(&self) -> GrBIndex {
        self.nrows
    }

    /// Number of columns
    pub fn ncols(&self) -> GrBIndex {
        self.ncols
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        match &self.storage {
            MatrixStorage::Coo(coo) => coo.nnz(),
            MatrixStorage::Csr(csr) => csr.nnz(),
            MatrixStorage::Empty => 0,
        }
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

    /// Get element at (row, col)
    pub fn get(&self, row: GrBIndex, col: GrBIndex) -> Option<&HdrScalar> {
        match &self.storage {
            MatrixStorage::Coo(coo) => coo.get_value(row, col),
            MatrixStorage::Csr(csr) => csr.get(row, col),
            MatrixStorage::Empty => None,
        }
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: GrBIndex, col: GrBIndex, value: HdrScalar) {
        self.ensure_coo();
        if let MatrixStorage::Coo(coo) = &mut self.storage {
            coo.add(row, col, value);
        }
    }

    /// Set vector element
    pub fn set_vector(&mut self, row: GrBIndex, col: GrBIndex, vec: BitpackedVector) {
        self.set(row, col, HdrScalar::Vector(vec));
    }

    /// Remove element (set to empty)
    pub fn remove(&mut self, _row: GrBIndex, _col: GrBIndex) {
        // COO doesn't support removal easily; rebuild without element
        // For now, this is a no-op (sparse matrices ignore missing entries)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.storage = MatrixStorage::Coo(CooStorage::new(self.nrows, self.ncols));
    }

    // ========================================================================
    // FORMAT CONVERSION
    // ========================================================================

    /// Ensure COO format (for modification)
    fn ensure_coo(&mut self) {
        if matches!(self.storage, MatrixStorage::Csr(_)) {
            if let MatrixStorage::Csr(csr) = std::mem::replace(&mut self.storage, MatrixStorage::Empty) {
                self.storage = MatrixStorage::Coo(csr.to_coo());
            }
        }
    }

    /// Ensure CSR format (for computation)
    fn ensure_csr(&mut self) {
        if matches!(self.storage, MatrixStorage::Coo(_)) {
            if let MatrixStorage::Coo(coo) = std::mem::replace(&mut self.storage, MatrixStorage::Empty) {
                self.storage = MatrixStorage::Csr(coo.to_csr());
            }
        }
    }

    /// Get as CSR (converts if needed, returns reference)
    pub fn as_csr(&mut self) -> Option<&CsrStorage> {
        self.ensure_csr();
        match &self.storage {
            MatrixStorage::Csr(csr) => Some(csr),
            _ => None,
        }
    }

    /// Get as COO
    pub fn as_coo(&mut self) -> Option<&CooStorage> {
        self.ensure_coo();
        match &self.storage {
            MatrixStorage::Coo(coo) => Some(coo),
            _ => None,
        }
    }

    // ========================================================================
    // ITERATION
    // ========================================================================

    /// Iterate over non-zero entries
    pub fn iter(&self) -> impl Iterator<Item = SparseEntry> + '_ {
        match &self.storage {
            MatrixStorage::Coo(coo) => IterImpl::Coo(coo.iter()),
            MatrixStorage::Csr(csr) => IterImpl::Csr(CsrIter::new(csr)),
            MatrixStorage::Empty => IterImpl::Empty,
        }
    }

    /// Iterate over row
    pub fn row_iter(&mut self, row: GrBIndex) -> impl Iterator<Item = (GrBIndex, &HdrScalar)> {
        self.ensure_csr();
        match &self.storage {
            MatrixStorage::Csr(csr) => csr.row(row),
            _ => panic!("Should be CSR after ensure_csr"),
        }
    }

    // ========================================================================
    // OPERATIONS (GraphBLAS-style)
    // ========================================================================

    /// Transpose
    pub fn transpose(&mut self) -> GrBMatrix {
        self.ensure_csr();
        if let MatrixStorage::Csr(csr) = &self.storage {
            let transposed_csr = csr.transpose();
            GrBMatrix {
                storage: MatrixStorage::Csr(transposed_csr),
                nrows: self.ncols,
                ncols: self.nrows,
                dtype: self.dtype,
            }
        } else {
            GrBMatrix::new(self.ncols, self.nrows)
        }
    }

    /// Extract submatrix
    pub fn extract(
        &self,
        row_indices: &[GrBIndex],
        col_indices: &[GrBIndex],
    ) -> GrBMatrix {
        let nrows = row_indices.len() as GrBIndex;
        let ncols = col_indices.len() as GrBIndex;
        let mut result = GrBMatrix::new(nrows, ncols);

        for (new_row, &old_row) in row_indices.iter().enumerate() {
            for (new_col, &old_col) in col_indices.iter().enumerate() {
                if let Some(val) = self.get(old_row, old_col) {
                    result.set(new_row as GrBIndex, new_col as GrBIndex, val.clone());
                }
            }
        }

        result
    }

    /// Apply unary operation to all elements
    pub fn apply<F>(&self, op: F) -> GrBMatrix
    where
        F: Fn(&HdrScalar) -> HdrScalar,
    {
        let mut result = GrBMatrix::new(self.nrows, self.ncols);

        for entry in self.iter() {
            let new_val = op(&entry.value);
            if !new_val.is_empty() {
                result.set(entry.row, entry.col, new_val);
            }
        }

        result
    }

    /// Element-wise addition with semiring
    pub fn ewise_add(&self, other: &GrBMatrix, semiring: &HdrSemiring) -> GrBMatrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);

        let mut result = GrBMatrix::new(self.nrows, self.ncols);

        // Add all entries from self
        for entry in self.iter() {
            let other_val = other.get(entry.row, entry.col);
            let new_val = match other_val {
                Some(ov) => semiring.add(&entry.value, ov),
                None => entry.value.clone(),
            };
            if !semiring.is_zero(&new_val) {
                result.set(entry.row, entry.col, new_val);
            }
        }

        // Add entries from other that aren't in self
        for entry in other.iter() {
            if self.get(entry.row, entry.col).is_none() {
                if !semiring.is_zero(&entry.value) {
                    result.set(entry.row, entry.col, entry.value.clone());
                }
            }
        }

        result
    }

    /// Element-wise multiplication with semiring
    pub fn ewise_mult(&self, other: &GrBMatrix, semiring: &HdrSemiring) -> GrBMatrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);

        let mut result = GrBMatrix::new(self.nrows, self.ncols);

        // Only entries present in both matrices
        for entry in self.iter() {
            if let Some(other_val) = other.get(entry.row, entry.col) {
                let new_val = semiring.multiply(&entry.value, other_val);
                if !semiring.is_zero(&new_val) {
                    result.set(entry.row, entry.col, new_val);
                }
            }
        }

        result
    }

    /// Matrix-matrix multiply: C = A ⊕.⊗ B
    pub fn mxm(&mut self, other: &mut GrBMatrix, semiring: &HdrSemiring) -> GrBMatrix {
        assert_eq!(self.ncols, other.nrows);

        self.ensure_csr();
        other.ensure_csr();

        let mut result = GrBMatrix::new(self.nrows, other.ncols);

        if let (MatrixStorage::Csr(a_csr), MatrixStorage::Csr(b_csr)) =
            (&self.storage, &other.storage)
        {
            // For each row in A
            for i in 0..self.nrows {
                // Accumulator for row i of result
                let mut row_accum: std::collections::HashMap<GrBIndex, HdrScalar> =
                    std::collections::HashMap::new();

                // For each non-zero (i, k) in A
                for (k, a_ik) in a_csr.row(i) {
                    // For each non-zero (k, j) in B
                    for (j, b_kj) in b_csr.row(k) {
                        // Multiply: a_ik ⊗ b_kj
                        let product = semiring.multiply(a_ik, b_kj);

                        // Add to accumulator: c_ij ⊕= product
                        row_accum.entry(j)
                            .and_modify(|acc| *acc = semiring.add(acc, &product))
                            .or_insert(product);
                    }
                }

                // Store non-zero results
                for (j, val) in row_accum {
                    if !semiring.is_zero(&val) {
                        result.set(i, j, val);
                    }
                }
            }
        }

        result
    }

    /// Matrix-vector multiply: w = A ⊕.⊗ u
    pub fn mxv(&mut self, u: &GrBVector, semiring: &HdrSemiring) -> GrBVector {
        assert_eq!(self.ncols, u.len());

        self.ensure_csr();

        let mut result = GrBVector::new(self.nrows);

        if let MatrixStorage::Csr(csr) = &self.storage {
            for i in 0..self.nrows {
                let mut accum = semiring.zero();

                for (j, a_ij) in csr.row(i) {
                    if let Some(u_j) = u.get(j) {
                        let product = semiring.multiply(a_ij, u_j);
                        accum = semiring.add(&accum, &product);
                    }
                }

                if !semiring.is_zero(&accum) {
                    result.set(i, accum);
                }
            }
        }

        result
    }

    /// Vector-matrix multiply: w = u ⊕.⊗ A (row vector times matrix)
    pub fn vxm(&mut self, u: &GrBVector, semiring: &HdrSemiring) -> GrBVector {
        assert_eq!(u.len(), self.nrows);

        self.ensure_csr();

        let mut result = GrBVector::new(self.ncols);

        if let MatrixStorage::Csr(csr) = &self.storage {
            // For each non-zero in u
            for (i, u_i) in u.iter() {
                // For each non-zero in row i of A
                for (j, a_ij) in csr.row(i) {
                    let product = semiring.multiply(u_i, a_ij);

                    // Accumulate into result[j]
                    if let Some(existing) = result.get(j) {
                        let new_val = semiring.add(existing, &product);
                        result.set(j, new_val);
                    } else {
                        result.set(j, product);
                    }
                }
            }
        }

        result
    }

    /// Reduce rows to a vector
    pub fn reduce_rows(&self, semiring: &HdrSemiring) -> GrBVector {
        let mut result = GrBVector::new(self.nrows);

        for entry in self.iter() {
            if let Some(existing) = result.get(entry.row) {
                let new_val = semiring.add(existing, &entry.value);
                result.set(entry.row, new_val);
            } else {
                result.set(entry.row, entry.value.clone());
            }
        }

        result
    }

    /// Reduce columns to a vector
    pub fn reduce_cols(&self, semiring: &HdrSemiring) -> GrBVector {
        let mut result = GrBVector::new(self.ncols);

        for entry in self.iter() {
            if let Some(existing) = result.get(entry.col) {
                let new_val = semiring.add(existing, &entry.value);
                result.set(entry.col, new_val);
            } else {
                result.set(entry.col, entry.value.clone());
            }
        }

        result
    }

    /// Reduce entire matrix to a scalar
    pub fn reduce(&self, semiring: &HdrSemiring) -> HdrScalar {
        let mut accum = semiring.zero();

        for entry in self.iter() {
            accum = semiring.add(&accum, &entry.value);
        }

        accum
    }
}

// Iterator implementation helper
enum IterImpl<'a> {
    Coo(Box<dyn Iterator<Item = SparseEntry> + 'a>),
    Csr(CsrIter<'a>),
    Empty,
}

impl<'a> Iterator for IterImpl<'a> {
    type Item = SparseEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IterImpl::Coo(iter) => iter.next(),
            IterImpl::Csr(iter) => iter.next(),
            IterImpl::Empty => None,
        }
    }
}

struct CsrIter<'a> {
    csr: &'a CsrStorage,
    row: GrBIndex,
    col_idx: usize,
}

impl<'a> CsrIter<'a> {
    fn new(csr: &'a CsrStorage) -> Self {
        Self { csr, row: 0, col_idx: 0 }
    }
}

impl<'a> Iterator for CsrIter<'a> {
    type Item = SparseEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let (nrows, _) = self.csr.dims();

        while self.row < nrows {
            let start = self.csr.row_ptr[self.row as usize] as usize;
            let end = self.csr.row_ptr[self.row as usize + 1] as usize;

            if self.col_idx < end - start {
                let global_idx = start + self.col_idx;
                let entry = SparseEntry {
                    row: self.row,
                    col: self.csr.col_idx[global_idx],
                    value: self.csr.values[global_idx].clone(),
                };
                self.col_idx += 1;
                return Some(entry);
            }

            self.row += 1;
            self.col_idx = 0;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamming::hamming_distance_scalar;

    #[test]
    fn test_matrix_basic() {
        let mut m = GrBMatrix::new(3, 3);

        m.set_vector(0, 1, BitpackedVector::random(1));
        m.set_vector(1, 2, BitpackedVector::random(2));

        assert_eq!(m.nnz(), 2);
        assert!(m.get(0, 1).is_some());
        assert!(m.get(0, 0).is_none());
    }

    #[test]
    fn test_mxv() {
        let mut m = GrBMatrix::new(2, 3);
        let mut u = GrBVector::new(3);

        // Set up matrix
        m.set_vector(0, 0, BitpackedVector::random(10));
        m.set_vector(0, 1, BitpackedVector::random(11));
        m.set_vector(1, 1, BitpackedVector::random(12));
        m.set_vector(1, 2, BitpackedVector::random(13));

        // Set up vector
        u.set_vector(0, BitpackedVector::random(20));
        u.set_vector(1, BitpackedVector::random(21));
        u.set_vector(2, BitpackedVector::random(22));

        // Matrix-vector multiply with XOR_BUNDLE semiring
        let semiring = HdrSemiring::XorBundle;
        let w = m.mxv(&u, &semiring);

        // Result should have 2 elements (one per row)
        assert!(w.get(0).is_some());
        assert!(w.get(1).is_some());
    }

    #[test]
    fn test_mxm() {
        let mut a = GrBMatrix::new(2, 2);
        let mut b = GrBMatrix::new(2, 2);

        // Identity-like matrices
        a.set_vector(0, 0, BitpackedVector::zero());
        a.set_vector(1, 1, BitpackedVector::zero());

        b.set_vector(0, 0, BitpackedVector::random(1));
        b.set_vector(1, 1, BitpackedVector::random(2));

        let semiring = HdrSemiring::XorBundle;
        let c = a.mxm(&mut b, &semiring);

        // With zero vectors in A and XOR multiply, should get B back
        assert!(c.get(0, 0).is_some());
        assert!(c.get(1, 1).is_some());
    }
}
