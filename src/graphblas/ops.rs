//! GraphBLAS Operations
//!
//! High-level operations following the GraphBLAS C API specification,
//! adapted for HDR computing with bitpacked vectors.

use crate::bitpack::BitpackedVector;
use super::matrix::GrBMatrix;
use super::vector::GrBVector;
use super::types::{GrBIndex, HdrScalar, GRB_ALL};
use super::semiring::{Semiring, HdrSemiring};
use super::descriptor::Descriptor;
use super::GrBInfo;

// ============================================================================
// MATRIX-MATRIX OPERATIONS
// ============================================================================

/// Matrix-matrix multiply: C = A ⊕.⊗ B
///
/// Using HDR semiring:
/// - Default: C[i,j] = bundle(A[i,k] ⊗ B[k,j] for all k)
///
/// # Arguments
/// * `c` - Output matrix (will be modified)
/// * `mask` - Optional mask matrix
/// * `accum` - Optional accumulator (how to combine with existing C)
/// * `semiring` - The semiring to use
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `desc` - Operation descriptor
pub fn grb_mxm(
    c: &mut GrBMatrix,
    mask: Option<&GrBMatrix>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    a: &mut GrBMatrix,
    b: &mut GrBMatrix,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    // Handle transpose
    let a_work = if desc.is_inp0_transposed() {
        a.transpose()
    } else {
        // Clone would be expensive; for now just use as-is
        // In production, would use a view
        a.transpose().transpose() // Identity
    };

    let b_work = if desc.is_inp1_transposed() {
        b.transpose()
    } else {
        b.transpose().transpose()
    };

    // Perform multiplication
    let mut result = a.mxm(b, semiring);

    // Apply mask
    if let Some(m) = mask {
        result = apply_matrix_mask(&result, m, &desc);
    }

    // Apply accumulator
    if let Some(acc) = accum {
        result = c.ewise_add(&result, acc);
    }

    // Handle output mode
    if desc.should_replace_output() {
        c.clear();
    }

    // Merge result into c
    for entry in result.iter() {
        c.set(entry.row, entry.col, entry.value);
    }

    GrBInfo::Success
}

/// Matrix-vector multiply: w = A ⊕.⊗ u
pub fn grb_mxv(
    w: &mut GrBVector,
    mask: Option<&GrBVector>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    a: &mut GrBMatrix,
    u: &GrBVector,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = a.mxv(u, semiring);

    // Apply mask
    if let Some(m) = mask {
        result = apply_vector_mask(&result, m, &desc);
    }

    // Apply accumulator
    if let Some(acc) = accum {
        result = w.ewise_add(&result, acc);
    }

    // Handle output
    if desc.should_replace_output() {
        w.clear();
    }

    for (idx, val) in result.iter() {
        w.set(idx, val.clone());
    }

    GrBInfo::Success
}

/// Vector-matrix multiply: w = u ⊕.⊗ A
pub fn grb_vxm(
    w: &mut GrBVector,
    mask: Option<&GrBVector>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    u: &GrBVector,
    a: &mut GrBMatrix,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = a.vxm(u, semiring);

    if let Some(m) = mask {
        result = apply_vector_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = w.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        w.clear();
    }

    for (idx, val) in result.iter() {
        w.set(idx, val.clone());
    }

    GrBInfo::Success
}

// ============================================================================
// ELEMENT-WISE OPERATIONS
// ============================================================================

/// Element-wise matrix addition: C = A ⊕ B
pub fn grb_ewise_add_matrix(
    c: &mut GrBMatrix,
    mask: Option<&GrBMatrix>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    a: &GrBMatrix,
    b: &GrBMatrix,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = a.ewise_add(b, semiring);

    if let Some(m) = mask {
        result = apply_matrix_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = c.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        c.clear();
    }

    for entry in result.iter() {
        c.set(entry.row, entry.col, entry.value);
    }

    GrBInfo::Success
}

/// Element-wise matrix multiplication: C = A ⊗ B
pub fn grb_ewise_mult_matrix(
    c: &mut GrBMatrix,
    mask: Option<&GrBMatrix>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    a: &GrBMatrix,
    b: &GrBMatrix,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = a.ewise_mult(b, semiring);

    if let Some(m) = mask {
        result = apply_matrix_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = c.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        c.clear();
    }

    for entry in result.iter() {
        c.set(entry.row, entry.col, entry.value);
    }

    GrBInfo::Success
}

/// Element-wise vector addition: w = u ⊕ v
pub fn grb_ewise_add_vector(
    w: &mut GrBVector,
    mask: Option<&GrBVector>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    u: &GrBVector,
    v: &GrBVector,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = u.ewise_add(v, semiring);

    if let Some(m) = mask {
        result = apply_vector_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = w.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        w.clear();
    }

    for (idx, val) in result.iter() {
        w.set(idx, val.clone());
    }

    GrBInfo::Success
}

/// Element-wise vector multiplication: w = u ⊗ v
pub fn grb_ewise_mult_vector(
    w: &mut GrBVector,
    mask: Option<&GrBVector>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    u: &GrBVector,
    v: &GrBVector,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = u.ewise_mult(v, semiring);

    if let Some(m) = mask {
        result = apply_vector_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = w.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        w.clear();
    }

    for (idx, val) in result.iter() {
        w.set(idx, val.clone());
    }

    GrBInfo::Success
}

// ============================================================================
// REDUCE OPERATIONS
// ============================================================================

/// Reduce matrix to vector (row-wise)
pub fn grb_reduce_to_vector(
    w: &mut GrBVector,
    mask: Option<&GrBVector>,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    a: &GrBMatrix,
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    let mut result = a.reduce_rows(semiring);

    if let Some(m) = mask {
        result = apply_vector_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = w.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        w.clear();
    }

    for (idx, val) in result.iter() {
        w.set(idx, val.clone());
    }

    GrBInfo::Success
}

/// Reduce matrix to scalar
pub fn grb_reduce_to_scalar(
    s: &mut HdrScalar,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    a: &GrBMatrix,
) -> GrBInfo {
    let result = a.reduce(semiring);

    if let Some(acc) = accum {
        *s = acc.add(s, &result);
    } else {
        *s = result;
    }

    GrBInfo::Success
}

/// Reduce vector to scalar
pub fn grb_reduce_vector(
    s: &mut HdrScalar,
    accum: Option<&HdrSemiring>,
    semiring: &HdrSemiring,
    u: &GrBVector,
) -> GrBInfo {
    let result = u.reduce(semiring);

    if let Some(acc) = accum {
        *s = acc.add(s, &result);
    } else {
        *s = result;
    }

    GrBInfo::Success
}

// ============================================================================
// APPLY OPERATIONS
// ============================================================================

/// Apply unary operation to matrix
pub fn grb_apply_matrix<F>(
    c: &mut GrBMatrix,
    mask: Option<&GrBMatrix>,
    accum: Option<&HdrSemiring>,
    op: F,
    a: &GrBMatrix,
    desc: Option<&Descriptor>,
) -> GrBInfo
where
    F: Fn(&HdrScalar) -> HdrScalar,
{
    let desc = desc.cloned().unwrap_or_default();

    let mut result = a.apply(op);

    if let Some(m) = mask {
        result = apply_matrix_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = c.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        c.clear();
    }

    for entry in result.iter() {
        c.set(entry.row, entry.col, entry.value);
    }

    GrBInfo::Success
}

/// Apply unary operation to vector
pub fn grb_apply_vector<F>(
    w: &mut GrBVector,
    mask: Option<&GrBVector>,
    accum: Option<&HdrSemiring>,
    op: F,
    u: &GrBVector,
    desc: Option<&Descriptor>,
) -> GrBInfo
where
    F: Fn(&HdrScalar) -> HdrScalar,
{
    let desc = desc.cloned().unwrap_or_default();

    let mut result = u.apply(op);

    if let Some(m) = mask {
        result = apply_vector_mask(&result, m, &desc);
    }

    if let Some(acc) = accum {
        result = w.ewise_add(&result, acc);
    }

    if desc.should_replace_output() {
        w.clear();
    }

    for (idx, val) in result.iter() {
        w.set(idx, val.clone());
    }

    GrBInfo::Success
}

// ============================================================================
// ASSIGN / EXTRACT OPERATIONS
// ============================================================================

/// Assign to submatrix: C[rows, cols] = A
pub fn grb_assign_matrix(
    c: &mut GrBMatrix,
    mask: Option<&GrBMatrix>,
    accum: Option<&HdrSemiring>,
    a: &GrBMatrix,
    rows: &[GrBIndex],
    cols: &[GrBIndex],
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    for entry in a.iter() {
        if (entry.row as usize) < rows.len() && (entry.col as usize) < cols.len() {
            let target_row = rows[entry.row as usize];
            let target_col = cols[entry.col as usize];

            // Check mask
            if let Some(m) = mask {
                let masked = if desc.is_mask_complemented() {
                    m.get(target_row, target_col).is_none()
                } else {
                    m.get(target_row, target_col).map_or(false, |v| v.to_bool())
                };
                if !masked {
                    continue;
                }
            }

            // Apply accumulator
            let new_val = if let Some(acc) = accum {
                if let Some(existing) = c.get(target_row, target_col) {
                    acc.add(existing, &entry.value)
                } else {
                    entry.value.clone()
                }
            } else {
                entry.value.clone()
            };

            c.set(target_row, target_col, new_val);
        }
    }

    GrBInfo::Success
}

/// Extract submatrix: C = A[rows, cols]
pub fn grb_extract_matrix(
    c: &mut GrBMatrix,
    mask: Option<&GrBMatrix>,
    accum: Option<&HdrSemiring>,
    a: &GrBMatrix,
    rows: &[GrBIndex],
    cols: &[GrBIndex],
    desc: Option<&Descriptor>,
) -> GrBInfo {
    let desc = desc.cloned().unwrap_or_default();

    if desc.should_replace_output() {
        c.clear();
    }

    for (new_row, &old_row) in rows.iter().enumerate() {
        for (new_col, &old_col) in cols.iter().enumerate() {
            if let Some(val) = a.get(old_row, old_col) {
                let new_row_idx = new_row as GrBIndex;
                let new_col_idx = new_col as GrBIndex;

                // Check mask
                if let Some(m) = mask {
                    let masked = if desc.is_mask_complemented() {
                        m.get(new_row_idx, new_col_idx).is_none()
                    } else {
                        m.get(new_row_idx, new_col_idx).map_or(false, |v| v.to_bool())
                    };
                    if !masked {
                        continue;
                    }
                }

                // Apply accumulator
                let new_val = if let Some(acc) = accum {
                    if let Some(existing) = c.get(new_row_idx, new_col_idx) {
                        acc.add(existing, val)
                    } else {
                        val.clone()
                    }
                } else {
                    val.clone()
                };

                c.set(new_row_idx, new_col_idx, new_val);
            }
        }
    }

    GrBInfo::Success
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn apply_matrix_mask(result: &GrBMatrix, mask: &GrBMatrix, desc: &Descriptor) -> GrBMatrix {
    let mut masked = GrBMatrix::new(result.nrows(), result.ncols());

    for entry in result.iter() {
        let mask_val = mask.get(entry.row, entry.col);

        let keep = if desc.is_mask_complemented() {
            mask_val.is_none() || !mask_val.unwrap().to_bool()
        } else {
            mask_val.map_or(false, |v| v.to_bool())
        };

        if keep {
            masked.set(entry.row, entry.col, entry.value);
        }
    }

    masked
}

fn apply_vector_mask(result: &GrBVector, mask: &GrBVector, desc: &Descriptor) -> GrBVector {
    if desc.is_mask_complemented() {
        result.apply_complement_mask(mask)
    } else {
        result.apply_mask(mask)
    }
}

// ============================================================================
// HDR-SPECIFIC GRAPH ALGORITHMS
// ============================================================================

/// BFS traversal using HDR semiring
///
/// Returns vector of bound paths from source to each reachable node.
pub fn hdr_bfs(
    adj: &mut GrBMatrix,
    source: GrBIndex,
    max_depth: usize,
) -> GrBVector {
    let n = adj.nrows();
    let semiring = HdrSemiring::BindFirst;

    // Initialize frontier with source
    let mut frontier = GrBVector::new(n);
    frontier.set_vector(source, BitpackedVector::zero()); // Zero = identity for XOR

    // Visited set (also stores path bindings)
    let mut visited = GrBVector::new(n);
    visited.set_vector(source, BitpackedVector::zero());

    for _depth in 0..max_depth {
        // Next frontier = (frontier × adjacency) AND NOT visited
        let mut next = adj.vxm(&frontier, &semiring);

        // Remove already visited
        next = next.apply_complement_mask(&visited);

        if next.is_empty() {
            break;
        }

        // Add to visited
        for (idx, val) in next.iter() {
            visited.set(idx, val.clone());
        }

        frontier = next;
    }

    visited
}

/// Single-source shortest semantic path
///
/// Uses Hamming distance as edge weight, finds minimum distance paths.
pub fn hdr_sssp(
    adj: &mut GrBMatrix,
    source: GrBIndex,
    max_iters: usize,
) -> GrBVector {
    let n = adj.nrows();
    let semiring = HdrSemiring::HammingMin;

    // Initialize distances
    let mut dist = GrBVector::new(n);
    dist.set(source, HdrScalar::Distance(0));

    for _iter in 0..max_iters {
        let old_nnz = dist.nnz();

        // Relax edges: new_dist = dist × adj (using min-hamming semiring)
        let new_dist = adj.vxm(&dist, &semiring);

        // Merge with existing (keep minimum)
        dist = dist.ewise_add(&new_dist, &semiring);

        // Check for convergence
        if dist.nnz() == old_nnz {
            break;
        }
    }

    dist
}

/// PageRank-style importance using HDR bundling
///
/// Accumulates "influence" vectors through bundling.
pub fn hdr_pagerank(
    adj: &mut GrBMatrix,
    damping: f32,
    max_iters: usize,
) -> GrBVector {
    let n = adj.nrows();
    let semiring = HdrSemiring::XorBundle;

    // Initialize ranks with random vectors
    let mut rank = GrBVector::new(n);
    for i in 0..n {
        rank.set_vector(i, BitpackedVector::random(i as u64));
    }

    // Teleport vector (random background)
    let teleport = BitpackedVector::random(0xDEADBEEF);

    for _iter in 0..max_iters {
        // new_rank = damping * (rank × adj) + (1-damping) * teleport
        let propagated = adj.vxm(&rank, &semiring);

        // Bundle with teleport (simplified: just use propagated for now)
        // In full implementation, would weight the bundling
        rank = propagated;
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs() {
        // Create simple graph: 0 -> 1 -> 2 -> 3
        let mut adj = GrBMatrix::new(4, 4);
        adj.set_vector(0, 1, BitpackedVector::random(10));
        adj.set_vector(1, 2, BitpackedVector::random(20));
        adj.set_vector(2, 3, BitpackedVector::random(30));

        let result = hdr_bfs(&mut adj, 0, 10);

        // Should reach all 4 nodes
        assert_eq!(result.nnz(), 4);
    }

    #[test]
    fn test_mxm() {
        let mut a = GrBMatrix::new(2, 2);
        let mut b = GrBMatrix::new(2, 2);
        let mut c = GrBMatrix::new(2, 2);

        a.set_vector(0, 0, BitpackedVector::random(1));
        a.set_vector(0, 1, BitpackedVector::random(2));
        b.set_vector(0, 0, BitpackedVector::random(3));
        b.set_vector(1, 1, BitpackedVector::random(4));

        let semiring = HdrSemiring::XorBundle;
        grb_mxm(&mut c, None, None, &semiring, &mut a, &mut b, None);

        // C[0,0] = A[0,0] ⊗ B[0,0]
        // C[0,1] = A[0,1] ⊗ B[1,1]
        assert!(c.get(0, 0).is_some());
        assert!(c.get(0, 1).is_some());
    }
}
