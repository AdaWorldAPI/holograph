//! Sparse Storage Formats with Arrow Backend
//!
//! Provides COO (Coordinate) and CSR (Compressed Sparse Row) storage
//! backed by Arrow arrays for zero-copy interoperability.

use std::sync::Arc;
use arrow::array::{
    UInt64Array, UInt64Builder, FixedSizeBinaryArray, FixedSizeBinaryBuilder,
    ArrayRef, Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::bitpack::{BitpackedVector, VECTOR_BYTES, PADDED_VECTOR_BYTES};
use crate::{HdrError, Result};
use super::types::{GrBIndex, HdrScalar};

/// Sparse storage format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate format (row, col, value triples)
    Coo,
    /// Compressed Sparse Row
    Csr,
    /// Compressed Sparse Column
    Csc,
    /// Hypersparse (for very sparse matrices)
    HyperSparse,
}

/// Entry in sparse storage
#[derive(Clone, Debug)]
pub struct SparseEntry {
    pub row: GrBIndex,
    pub col: GrBIndex,
    pub value: HdrScalar,
}

/// COO (Coordinate) format storage
///
/// Stores triples (row, col, value) for each non-zero entry.
/// Good for: construction, conversion, small matrices
#[derive(Clone)]
pub struct CooStorage {
    /// Row indices
    rows: Vec<GrBIndex>,
    /// Column indices
    cols: Vec<GrBIndex>,
    /// Values as HDR scalars
    values: Vec<HdrScalar>,
    /// Number of rows
    nrows: GrBIndex,
    /// Number of columns
    ncols: GrBIndex,
    /// Is sorted?
    sorted: bool,
}

impl CooStorage {
    /// Create empty COO storage
    pub fn new(nrows: GrBIndex, ncols: GrBIndex) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
            sorted: true,
        }
    }

    /// Create with capacity
    pub fn with_capacity(nrows: GrBIndex, ncols: GrBIndex, nnz: usize) -> Self {
        Self {
            rows: Vec::with_capacity(nnz),
            cols: Vec::with_capacity(nnz),
            values: Vec::with_capacity(nnz),
            nrows,
            ncols,
            sorted: true,
        }
    }

    /// Add an entry
    pub fn add(&mut self, row: GrBIndex, col: GrBIndex, value: HdrScalar) {
        if row >= self.nrows || col >= self.ncols {
            return; // Out of bounds
        }

        // Check if still sorted
        if !self.rows.is_empty() {
            let last_row = *self.rows.last().unwrap();
            let last_col = *self.cols.last().unwrap();
            if row < last_row || (row == last_row && col <= last_col) {
                self.sorted = false;
            }
        }

        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
    }

    /// Add a vector entry
    pub fn add_vector(&mut self, row: GrBIndex, col: GrBIndex, vec: BitpackedVector) {
        self.add(row, col, HdrScalar::Vector(vec));
    }

    /// Number of non-zeros
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Get dimensions
    pub fn dims(&self) -> (GrBIndex, GrBIndex) {
        (self.nrows, self.ncols)
    }

    /// Get entry by index
    pub fn get(&self, idx: usize) -> Option<SparseEntry> {
        if idx >= self.nnz() {
            return None;
        }
        Some(SparseEntry {
            row: self.rows[idx],
            col: self.cols[idx],
            value: self.values[idx].clone(),
        })
    }

    /// Get value at (row, col)
    pub fn get_value(&self, row: GrBIndex, col: GrBIndex) -> Option<&HdrScalar> {
        for i in 0..self.nnz() {
            if self.rows[i] == row && self.cols[i] == col {
                return Some(&self.values[i]);
            }
        }
        None
    }

    /// Sort entries by (row, col)
    pub fn sort(&mut self) {
        if self.sorted {
            return;
        }

        // Create index array
        let mut indices: Vec<usize> = (0..self.nnz()).collect();

        // Sort indices by (row, col)
        indices.sort_by(|&a, &b| {
            match self.rows[a].cmp(&self.rows[b]) {
                std::cmp::Ordering::Equal => self.cols[a].cmp(&self.cols[b]),
                other => other,
            }
        });

        // Reorder arrays
        let rows: Vec<_> = indices.iter().map(|&i| self.rows[i]).collect();
        let cols: Vec<_> = indices.iter().map(|&i| self.cols[i]).collect();
        let values: Vec<_> = indices.iter().map(|&i| self.values[i].clone()).collect();

        self.rows = rows;
        self.cols = cols;
        self.values = values;
        self.sorted = true;
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> CsrStorage {
        let mut csr = CsrStorage::new(self.nrows, self.ncols);

        // Ensure sorted
        let mut sorted = self.clone();
        sorted.sort();

        // Build row pointers
        csr.row_ptr.push(0);
        let mut current_row = 0;

        for i in 0..sorted.nnz() {
            while current_row < sorted.rows[i] {
                csr.row_ptr.push(i as GrBIndex);
                current_row += 1;
            }
            csr.col_idx.push(sorted.cols[i]);
            csr.values.push(sorted.values[i].clone());
        }

        // Fill remaining row pointers
        while current_row < self.nrows {
            csr.row_ptr.push(sorted.nnz() as GrBIndex);
            current_row += 1;
        }
        csr.row_ptr.push(sorted.nnz() as GrBIndex);

        csr
    }

    /// Iterator over entries
    pub fn iter(&self) -> impl Iterator<Item = SparseEntry> + '_ {
        (0..self.nnz()).map(move |i| SparseEntry {
            row: self.rows[i],
            col: self.cols[i],
            value: self.values[i].clone(),
        })
    }

    /// Convert to Arrow RecordBatch (for vector values)
    pub fn to_arrow(&self) -> Result<RecordBatch> {
        let mut row_builder = UInt64Builder::with_capacity(self.nnz());
        let mut col_builder = UInt64Builder::with_capacity(self.nnz());
        let mut val_builder = FixedSizeBinaryBuilder::with_capacity(self.nnz(), PADDED_VECTOR_BYTES as i32);

        for i in 0..self.nnz() {
            row_builder.append_value(self.rows[i]);
            col_builder.append_value(self.cols[i]);

            if let HdrScalar::Vector(v) = &self.values[i] {
                val_builder.append_value(&v.to_padded_bytes())
                    .map_err(|e| HdrError::Storage(e.to_string()))?;
            } else {
                val_builder.append_value(&vec![0u8; PADDED_VECTOR_BYTES])
                    .map_err(|e| HdrError::Storage(e.to_string()))?;
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("row", DataType::UInt64, false),
            Field::new("col", DataType::UInt64, false),
            Field::new("value", DataType::FixedSizeBinary(PADDED_VECTOR_BYTES as i32), false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(row_builder.finish()) as ArrayRef,
                Arc::new(col_builder.finish()) as ArrayRef,
                Arc::new(val_builder.finish()) as ArrayRef,
            ],
        ).map_err(|e| HdrError::Storage(e.to_string()))
    }

    /// Create from Arrow RecordBatch
    pub fn from_arrow(batch: &RecordBatch, nrows: GrBIndex, ncols: GrBIndex) -> Result<Self> {
        let rows = batch.column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| HdrError::Storage("Invalid row column".into()))?;

        let cols = batch.column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| HdrError::Storage("Invalid col column".into()))?;

        let values = batch.column(2)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| HdrError::Storage("Invalid value column".into()))?;

        let mut coo = Self::with_capacity(nrows, ncols, batch.num_rows());

        for i in 0..batch.num_rows() {
            let row = rows.value(i);
            let col = cols.value(i);
            let bytes = values.value(i);
            // Handle both padded (1280) and unpadded (1256) Arrow columns
            let vec = if bytes.len() >= PADDED_VECTOR_BYTES {
                BitpackedVector::from_padded_bytes(bytes)?
            } else {
                BitpackedVector::from_bytes(bytes)?
            };
            coo.add_vector(row, col, vec);
        }

        Ok(coo)
    }
}

/// CSR (Compressed Sparse Row) format storage
///
/// Efficient for row-wise operations and matrix-vector multiply.
#[derive(Clone)]
pub struct CsrStorage {
    /// Row pointers (size nrows + 1)
    pub row_ptr: Vec<GrBIndex>,
    /// Column indices (size nnz)
    pub col_idx: Vec<GrBIndex>,
    /// Values (size nnz)
    pub values: Vec<HdrScalar>,
    /// Number of rows
    nrows: GrBIndex,
    /// Number of columns
    ncols: GrBIndex,
}

impl CsrStorage {
    /// Create empty CSR storage
    pub fn new(nrows: GrBIndex, ncols: GrBIndex) -> Self {
        Self {
            row_ptr: vec![0],
            col_idx: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Number of non-zeros
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// Get dimensions
    pub fn dims(&self) -> (GrBIndex, GrBIndex) {
        (self.nrows, self.ncols)
    }

    /// Get value at (row, col)
    pub fn get(&self, row: GrBIndex, col: GrBIndex) -> Option<&HdrScalar> {
        if row >= self.nrows {
            return None;
        }

        let start = self.row_ptr[row as usize] as usize;
        let end = self.row_ptr[row as usize + 1] as usize;

        // Binary search within row
        let cols = &self.col_idx[start..end];
        match cols.binary_search(&col) {
            Ok(idx) => Some(&self.values[start + idx]),
            Err(_) => None,
        }
    }

    /// Get row as iterator
    pub fn row(&self, row: GrBIndex) -> impl Iterator<Item = (GrBIndex, &HdrScalar)> {
        let start = self.row_ptr.get(row as usize).copied().unwrap_or(0) as usize;
        let end = self.row_ptr.get(row as usize + 1).copied().unwrap_or(0) as usize;

        self.col_idx[start..end].iter()
            .zip(self.values[start..end].iter())
            .map(|(&col, val)| (col, val))
    }

    /// Number of non-zeros in row
    pub fn row_nnz(&self, row: GrBIndex) -> usize {
        if row >= self.nrows {
            return 0;
        }
        let start = self.row_ptr[row as usize];
        let end = self.row_ptr[row as usize + 1];
        (end - start) as usize
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> CooStorage {
        let mut coo = CooStorage::with_capacity(self.nrows, self.ncols, self.nnz());

        for row in 0..self.nrows {
            for (col, val) in self.row(row) {
                coo.add(row, col, val.clone());
            }
        }

        coo.sorted = true;
        coo
    }

    /// Transpose to CSC (returns new CSR of transposed matrix)
    pub fn transpose(&self) -> CsrStorage {
        let coo = self.to_coo();

        // Swap rows and cols
        let mut transposed = CooStorage::with_capacity(self.ncols, self.nrows, self.nnz());
        for entry in coo.iter() {
            transposed.add(entry.col, entry.row, entry.value);
        }

        transposed.to_csr()
    }

    /// Extract diagonal
    pub fn diagonal(&self) -> Vec<HdrScalar> {
        let n = self.nrows.min(self.ncols);
        let mut diag = Vec::with_capacity(n as usize);

        for i in 0..n {
            if let Some(val) = self.get(i, i) {
                diag.push(val.clone());
            } else {
                diag.push(HdrScalar::Empty);
            }
        }

        diag
    }
}

/// Sparse vector storage
#[derive(Clone)]
pub struct SparseVec {
    /// Indices of non-zero elements
    pub indices: Vec<GrBIndex>,
    /// Values
    pub values: Vec<HdrScalar>,
    /// Length
    pub len: GrBIndex,
}

impl SparseVec {
    /// Create empty sparse vector
    pub fn new(len: GrBIndex) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            len,
        }
    }

    /// Create with capacity
    pub fn with_capacity(len: GrBIndex, nnz: usize) -> Self {
        Self {
            indices: Vec::with_capacity(nnz),
            values: Vec::with_capacity(nnz),
            len,
        }
    }

    /// Add element
    pub fn add(&mut self, idx: GrBIndex, value: HdrScalar) {
        if idx < self.len && !value.is_empty() {
            self.indices.push(idx);
            self.values.push(value);
        }
    }

    /// Number of non-zeros
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get value at index
    pub fn get(&self, idx: GrBIndex) -> Option<&HdrScalar> {
        for (i, &idx_i) in self.indices.iter().enumerate() {
            if idx_i == idx {
                return Some(&self.values[i]);
            }
        }
        None
    }

    /// Iterator over (index, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (GrBIndex, &HdrScalar)> {
        self.indices.iter().zip(self.values.iter()).map(|(&i, v)| (i, v))
    }

    /// Sort by index
    pub fn sort(&mut self) {
        let mut pairs: Vec<_> = self.indices.iter()
            .zip(self.values.iter())
            .map(|(&i, v)| (i, v.clone()))
            .collect();

        pairs.sort_by_key(|(i, _)| *i);

        self.indices = pairs.iter().map(|(i, _)| *i).collect();
        self.values = pairs.into_iter().map(|(_, v)| v).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_storage() {
        let mut coo = CooStorage::new(3, 3);

        coo.add_vector(0, 0, BitpackedVector::random(1));
        coo.add_vector(0, 2, BitpackedVector::random(2));
        coo.add_vector(1, 1, BitpackedVector::random(3));
        coo.add_vector(2, 0, BitpackedVector::random(4));

        assert_eq!(coo.nnz(), 4);
        assert!(coo.get_value(0, 0).is_some());
        assert!(coo.get_value(0, 1).is_none());
    }

    #[test]
    fn test_coo_to_csr() {
        let mut coo = CooStorage::new(3, 3);

        coo.add_vector(0, 0, BitpackedVector::random(1));
        coo.add_vector(0, 2, BitpackedVector::random(2));
        coo.add_vector(1, 1, BitpackedVector::random(3));
        coo.add_vector(2, 0, BitpackedVector::random(4));

        let csr = coo.to_csr();

        assert_eq!(csr.nnz(), 4);
        assert!(csr.get(0, 0).is_some());
        assert!(csr.get(0, 1).is_none());
        assert_eq!(csr.row_nnz(0), 2);
        assert_eq!(csr.row_nnz(1), 1);
    }

    #[test]
    fn test_csr_row_iteration() {
        let mut coo = CooStorage::new(3, 4);

        coo.add_vector(1, 0, BitpackedVector::random(1));
        coo.add_vector(1, 2, BitpackedVector::random(2));
        coo.add_vector(1, 3, BitpackedVector::random(3));

        let csr = coo.to_csr();

        let row1: Vec<_> = csr.row(1).collect();
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0].0, 0); // col 0
        assert_eq!(row1[1].0, 2); // col 2
        assert_eq!(row1[2].0, 3); // col 3
    }

    #[test]
    fn test_arrow_roundtrip() {
        let mut coo = CooStorage::new(3, 3);

        let v1 = BitpackedVector::random(100);
        let v2 = BitpackedVector::random(200);

        coo.add_vector(0, 1, v1.clone());
        coo.add_vector(2, 0, v2.clone());

        let batch = coo.to_arrow().unwrap();
        let loaded = CooStorage::from_arrow(&batch, 3, 3).unwrap();

        assert_eq!(loaded.nnz(), 2);

        if let Some(HdrScalar::Vector(loaded_v1)) = loaded.get_value(0, 1) {
            assert_eq!(loaded_v1, &v1);
        } else {
            panic!("Expected vector at (0,1)");
        }
    }
}
