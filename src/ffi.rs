//! Foreign Function Interface for C/GraphBLAS Integration
//!
//! Provides C-compatible functions for integrating with RedisGraph's
//! existing C codebase and GraphBLAS library.
//!
//! # Usage from C
//!
//! ```c
//! #include "hdr_hamming.h"
//!
//! // Create a vector
//! HdrVector* vec = hdr_vector_random(12345);
//!
//! // Compute Hamming distance
//! uint32_t dist = hdr_hamming_distance(vec1, vec2);
//!
//! // Bind vectors
//! HdrVector* bound = hdr_vector_bind(vec1, vec2);
//!
//! // Unbind to recover
//! HdrVector* recovered = hdr_vector_unbind(bound, vec2);
//!
//! // Free memory
//! hdr_vector_free(vec);
//! ```
//!
//! # GraphBLAS Integration
//!
//! The module provides sparse matrix operations compatible with GraphBLAS:
//!
//! ```c
//! // Create adjacency matrix from bound edges
//! GrB_Matrix adj;
//! hdr_to_graphblas(edges, n_edges, &adj);
//!
//! // Run BFS using GraphBLAS
//! GrB_Vector result;
//! GrB_vxm(result, NULL, NULL, GxB_ANY_PAIR_BOOL, frontier, adj, NULL);
//!
//! // Convert result back to HDR vectors
//! hdr_from_graphblas(adj, &edges, &n_edges);
//! ```

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::slice;

use crate::bitpack::{BitpackedVector, VECTOR_BYTES, VECTOR_WORDS, VECTOR_BITS, PADDED_VECTOR_BYTES};
use crate::hamming::{hamming_distance_scalar, hamming_to_similarity, StackedPopcount, Belichtung};
use crate::hdr_cascade::{HdrCascade, MexicanHat, SearchResult};
use crate::resonance::{VectorField, Resonator, BoundEdge};

// ============================================================================
// OPAQUE TYPES
// ============================================================================

/// Opaque vector handle for C
#[repr(C)]
pub struct HdrVector {
    inner: BitpackedVector,
}

/// Opaque cascade index handle for C
#[repr(C)]
pub struct HdrCascadeIndex {
    inner: HdrCascade,
}

/// Opaque vector field handle for C
#[repr(C)]
pub struct HdrField {
    inner: VectorField,
}

/// Opaque resonator handle for C
#[repr(C)]
pub struct HdrResonator {
    inner: Resonator,
}

/// Search result for C
#[repr(C)]
pub struct HdrSearchResult {
    pub index: u64,
    pub distance: u32,
    pub similarity: f32,
    pub response: f32,
}

/// Stacked popcount result for C
#[repr(C)]
pub struct HdrStackedPopcount {
    /// Per-word counts (157 bytes)
    pub per_word: [u8; VECTOR_WORDS],
    /// Total Hamming distance
    pub total: u32,
}

/// Belichtung meter result for C
#[repr(C)]
pub struct HdrBelichtung {
    pub mean: u8,
    pub sd_100: u8,
}

// ============================================================================
// VECTOR OPERATIONS
// ============================================================================

/// Create a zero vector
#[no_mangle]
pub extern "C" fn hdr_vector_zero() -> *mut HdrVector {
    Box::into_raw(Box::new(HdrVector {
        inner: BitpackedVector::zero(),
    }))
}

/// Create a random vector
#[no_mangle]
pub extern "C" fn hdr_vector_random(seed: u64) -> *mut HdrVector {
    Box::into_raw(Box::new(HdrVector {
        inner: BitpackedVector::random(seed),
    }))
}

/// Create vector from bytes
#[no_mangle]
pub extern "C" fn hdr_vector_from_bytes(data: *const u8, len: usize) -> *mut HdrVector {
    if data.is_null() || len != VECTOR_BYTES {
        return ptr::null_mut();
    }

    let bytes = unsafe { slice::from_raw_parts(data, len) };
    match BitpackedVector::from_bytes(bytes) {
        Ok(vec) => Box::into_raw(Box::new(HdrVector { inner: vec })),
        Err(_) => ptr::null_mut(),
    }
}

/// Create vector from words
#[no_mangle]
pub extern "C" fn hdr_vector_from_words(words: *const u64, len: usize) -> *mut HdrVector {
    if words.is_null() || len != VECTOR_WORDS {
        return ptr::null_mut();
    }

    let slice = unsafe { slice::from_raw_parts(words, len) };
    let mut arr = [0u64; VECTOR_WORDS];
    arr.copy_from_slice(slice);

    Box::into_raw(Box::new(HdrVector {
        inner: BitpackedVector::from_words(arr),
    }))
}

/// Create vector from hash of data
#[no_mangle]
pub extern "C" fn hdr_vector_from_hash(data: *const u8, len: usize) -> *mut HdrVector {
    if data.is_null() {
        return hdr_vector_random(0);
    }

    let bytes = unsafe { slice::from_raw_parts(data, len) };
    Box::into_raw(Box::new(HdrVector {
        inner: BitpackedVector::from_hash(bytes),
    }))
}

/// Clone a vector
#[no_mangle]
pub extern "C" fn hdr_vector_clone(vec: *const HdrVector) -> *mut HdrVector {
    if vec.is_null() {
        return ptr::null_mut();
    }

    let v = unsafe { &(*vec).inner };
    Box::into_raw(Box::new(HdrVector { inner: v.clone() }))
}

/// Free a vector
#[no_mangle]
pub extern "C" fn hdr_vector_free(vec: *mut HdrVector) {
    if !vec.is_null() {
        unsafe { drop(Box::from_raw(vec)) };
    }
}

/// Get vector bytes
#[no_mangle]
pub extern "C" fn hdr_vector_to_bytes(vec: *const HdrVector, out: *mut u8, out_len: usize) -> i32 {
    if vec.is_null() || out.is_null() || out_len < VECTOR_BYTES {
        return -1;
    }

    let v = unsafe { &(*vec).inner };
    let bytes = v.to_bytes();

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out, VECTOR_BYTES);
    }

    VECTOR_BYTES as i32
}

/// Get vector words
#[no_mangle]
pub extern "C" fn hdr_vector_to_words(vec: *const HdrVector, out: *mut u64, out_len: usize) -> i32 {
    if vec.is_null() || out.is_null() || out_len < VECTOR_WORDS {
        return -1;
    }

    let v = unsafe { &(*vec).inner };
    let words = v.words();

    unsafe {
        ptr::copy_nonoverlapping(words.as_ptr(), out, VECTOR_WORDS);
    }

    VECTOR_WORDS as i32
}

/// Get population count
#[no_mangle]
pub extern "C" fn hdr_vector_popcount(vec: *const HdrVector) -> u32 {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).inner.popcount() }
}

/// Get density (0.0 to 1.0)
#[no_mangle]
pub extern "C" fn hdr_vector_density(vec: *const HdrVector) -> f32 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe { (*vec).inner.density() }
}

// ============================================================================
// BINDING OPERATIONS (Vector Field)
// ============================================================================

/// Bind two vectors: A ⊗ B
#[no_mangle]
pub extern "C" fn hdr_vector_bind(a: *const HdrVector, b: *const HdrVector) -> *mut HdrVector {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let va = unsafe { &(*a).inner };
    let vb = unsafe { &(*b).inner };

    Box::into_raw(Box::new(HdrVector {
        inner: va.xor(vb),
    }))
}

/// Unbind: bound ⊗ key = result (A ⊗ B ⊗ B = A)
#[no_mangle]
pub extern "C" fn hdr_vector_unbind(bound: *const HdrVector, key: *const HdrVector) -> *mut HdrVector {
    // Same as bind (XOR is self-inverse)
    hdr_vector_bind(bound, key)
}

/// Bind three vectors: A ⊗ B ⊗ C
#[no_mangle]
pub extern "C" fn hdr_vector_bind3(
    a: *const HdrVector,
    b: *const HdrVector,
    c: *const HdrVector,
) -> *mut HdrVector {
    if a.is_null() || b.is_null() || c.is_null() {
        return ptr::null_mut();
    }

    let va = unsafe { &(*a).inner };
    let vb = unsafe { &(*b).inner };
    let vc = unsafe { &(*c).inner };

    Box::into_raw(Box::new(HdrVector {
        inner: va.xor(vb).xor(vc),
    }))
}

/// Bundle multiple vectors (majority voting)
#[no_mangle]
pub extern "C" fn hdr_vector_bundle(vecs: *const *const HdrVector, count: usize) -> *mut HdrVector {
    if vecs.is_null() || count == 0 {
        return ptr::null_mut();
    }

    let slice = unsafe { slice::from_raw_parts(vecs, count) };
    let inner_vecs: Vec<&BitpackedVector> = slice.iter()
        .filter_map(|&p| {
            if p.is_null() {
                None
            } else {
                Some(unsafe { &(*p).inner })
            }
        })
        .collect();

    if inner_vecs.is_empty() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(HdrVector {
        inner: BitpackedVector::bundle(&inner_vecs),
    }))
}

/// Permute (rotate) vector
#[no_mangle]
pub extern "C" fn hdr_vector_permute(vec: *const HdrVector, positions: i32) -> *mut HdrVector {
    if vec.is_null() {
        return ptr::null_mut();
    }

    let v = unsafe { &(*vec).inner };
    let rotated = if positions >= 0 {
        v.rotate_left(positions as usize)
    } else {
        v.rotate_right((-positions) as usize)
    };

    Box::into_raw(Box::new(HdrVector { inner: rotated }))
}

// ============================================================================
// HAMMING DISTANCE OPERATIONS
// ============================================================================

/// Compute exact Hamming distance
#[no_mangle]
pub extern "C" fn hdr_hamming_distance(a: *const HdrVector, b: *const HdrVector) -> u32 {
    if a.is_null() || b.is_null() {
        return u32::MAX;
    }

    let va = unsafe { &(*a).inner };
    let vb = unsafe { &(*b).inner };

    hamming_distance_scalar(va, vb)
}

/// Compute similarity (0.0 to 1.0)
#[no_mangle]
pub extern "C" fn hdr_similarity(a: *const HdrVector, b: *const HdrVector) -> f32 {
    let dist = hdr_hamming_distance(a, b);
    if dist == u32::MAX {
        return 0.0;
    }
    hamming_to_similarity(dist)
}

/// Compute stacked popcount
#[no_mangle]
pub extern "C" fn hdr_stacked_popcount(
    a: *const HdrVector,
    b: *const HdrVector,
    out: *mut HdrStackedPopcount,
) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return -1;
    }

    let va = unsafe { &(*a).inner };
    let vb = unsafe { &(*b).inner };

    let stacked = StackedPopcount::compute(va, vb);

    unsafe {
        (*out).per_word = stacked.per_word;
        (*out).total = stacked.total;
    }

    0
}

/// Compute stacked popcount with early exit threshold
#[no_mangle]
pub extern "C" fn hdr_stacked_popcount_threshold(
    a: *const HdrVector,
    b: *const HdrVector,
    threshold: u32,
    out: *mut HdrStackedPopcount,
) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return -1;
    }

    let va = unsafe { &(*a).inner };
    let vb = unsafe { &(*b).inner };

    match StackedPopcount::compute_with_threshold(va, vb, threshold) {
        Some(stacked) => {
            unsafe {
                (*out).per_word = stacked.per_word;
                (*out).total = stacked.total;
            }
            0
        }
        None => 1 // Exceeded threshold
    }
}

/// Quick exposure meter (Belichtungsmesser)
#[no_mangle]
pub extern "C" fn hdr_belichtung_meter(
    a: *const HdrVector,
    b: *const HdrVector,
    out: *mut HdrBelichtung,
) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return -1;
    }

    let va = unsafe { &(*a).inner };
    let vb = unsafe { &(*b).inner };

    let meter = Belichtung::meter(va, vb);

    unsafe {
        (*out).mean = meter.mean;
        (*out).sd_100 = meter.sd_100;
    }

    0
}

// ============================================================================
// CASCADE INDEX OPERATIONS
// ============================================================================

/// Create cascade index
#[no_mangle]
pub extern "C" fn hdr_cascade_create(capacity: usize) -> *mut HdrCascadeIndex {
    Box::into_raw(Box::new(HdrCascadeIndex {
        inner: HdrCascade::with_capacity(capacity),
    }))
}

/// Free cascade index
#[no_mangle]
pub extern "C" fn hdr_cascade_free(cascade: *mut HdrCascadeIndex) {
    if !cascade.is_null() {
        unsafe { drop(Box::from_raw(cascade)) };
    }
}

/// Add vector to cascade index
#[no_mangle]
pub extern "C" fn hdr_cascade_add(cascade: *mut HdrCascadeIndex, vec: *const HdrVector) -> i32 {
    if cascade.is_null() || vec.is_null() {
        return -1;
    }

    let c = unsafe { &mut (*cascade).inner };
    let v = unsafe { &(*vec).inner };

    c.add(v.clone());
    0
}

/// Get cascade index size
#[no_mangle]
pub extern "C" fn hdr_cascade_len(cascade: *const HdrCascadeIndex) -> usize {
    if cascade.is_null() {
        return 0;
    }
    unsafe { (*cascade).inner.len() }
}

/// Search cascade index
#[no_mangle]
pub extern "C" fn hdr_cascade_search(
    cascade: *const HdrCascadeIndex,
    query: *const HdrVector,
    k: usize,
    out: *mut HdrSearchResult,
    out_len: usize,
) -> i32 {
    if cascade.is_null() || query.is_null() || out.is_null() || out_len == 0 {
        return -1;
    }

    let c = unsafe { &(*cascade).inner };
    let q = unsafe { &(*query).inner };

    let results = c.search(q, k.min(out_len));
    let n = results.len();

    let out_slice = unsafe { slice::from_raw_parts_mut(out, out_len) };
    for (i, r) in results.into_iter().enumerate() {
        out_slice[i] = HdrSearchResult {
            index: r.index as u64,
            distance: r.distance,
            similarity: r.similarity,
            response: r.response,
        };
    }

    n as i32
}

/// Set Mexican hat parameters
#[no_mangle]
pub extern "C" fn hdr_cascade_set_mexican_hat(
    cascade: *mut HdrCascadeIndex,
    excite: u32,
    inhibit: u32,
) -> i32 {
    if cascade.is_null() {
        return -1;
    }

    let c = unsafe { &mut (*cascade).inner };
    c.set_mexican_hat(MexicanHat::new(excite, inhibit));
    0
}

// ============================================================================
// RESONATOR OPERATIONS
// ============================================================================

/// Create resonator
#[no_mangle]
pub extern "C" fn hdr_resonator_create(capacity: usize) -> *mut HdrResonator {
    Box::into_raw(Box::new(HdrResonator {
        inner: Resonator::with_capacity(capacity),
    }))
}

/// Free resonator
#[no_mangle]
pub extern "C" fn hdr_resonator_free(resonator: *mut HdrResonator) {
    if !resonator.is_null() {
        unsafe { drop(Box::from_raw(resonator)) };
    }
}

/// Add concept to resonator
#[no_mangle]
pub extern "C" fn hdr_resonator_add(resonator: *mut HdrResonator, vec: *const HdrVector) -> i32 {
    if resonator.is_null() || vec.is_null() {
        return -1;
    }

    let r = unsafe { &mut (*resonator).inner };
    let v = unsafe { &(*vec).inner };

    r.add(v.clone()) as i32
}

/// Add named concept to resonator
#[no_mangle]
pub extern "C" fn hdr_resonator_add_named(
    resonator: *mut HdrResonator,
    name: *const c_char,
    vec: *const HdrVector,
) -> i32 {
    if resonator.is_null() || name.is_null() || vec.is_null() {
        return -1;
    }

    let r = unsafe { &mut (*resonator).inner };
    let v = unsafe { &(*vec).inner };
    let n = unsafe { CStr::from_ptr(name) }.to_string_lossy();

    r.add_named(&n, v.clone()) as i32
}

/// Set resonator threshold
#[no_mangle]
pub extern "C" fn hdr_resonator_set_threshold(resonator: *mut HdrResonator, threshold: u32) -> i32 {
    if resonator.is_null() {
        return -1;
    }

    let r = unsafe { &mut (*resonator).inner };
    r.set_threshold(threshold);
    0
}

/// Find best match (resonate)
#[no_mangle]
pub extern "C" fn hdr_resonator_resonate(
    resonator: *const HdrResonator,
    query: *const HdrVector,
    out_index: *mut usize,
    out_distance: *mut u32,
    out_similarity: *mut f32,
) -> i32 {
    if resonator.is_null() || query.is_null() {
        return -1;
    }

    let r = unsafe { &(*resonator).inner };
    let q = unsafe { &(*query).inner };

    match r.resonate(q) {
        Some(result) => {
            if !out_index.is_null() {
                unsafe { *out_index = result.index };
            }
            if !out_distance.is_null() {
                unsafe { *out_distance = result.distance };
            }
            if !out_similarity.is_null() {
                unsafe { *out_similarity = result.similarity };
            }
            0
        }
        None => 1 // No match found
    }
}

// ============================================================================
// GRAPHBLAS INTEGRATION HELPERS
// ============================================================================

/// Sparse matrix entry for GraphBLAS interop
#[repr(C)]
pub struct HdrSparseEntry {
    pub row: u64,
    pub col: u64,
    pub value: f32, // Similarity or distance
}

/// Convert vector similarities to sparse matrix entries
///
/// This can be used to build a GraphBLAS adjacency matrix from
/// vector search results.
#[no_mangle]
pub extern "C" fn hdr_to_sparse_matrix(
    cascade: *const HdrCascadeIndex,
    queries: *const *const HdrVector,
    n_queries: usize,
    k: usize,
    out: *mut HdrSparseEntry,
    out_capacity: usize,
) -> i32 {
    if cascade.is_null() || queries.is_null() || out.is_null() {
        return -1;
    }

    let c = unsafe { &(*cascade).inner };
    let query_slice = unsafe { slice::from_raw_parts(queries, n_queries) };
    let out_slice = unsafe { slice::from_raw_parts_mut(out, out_capacity) };

    let mut count = 0;

    for (row, &qptr) in query_slice.iter().enumerate() {
        if qptr.is_null() {
            continue;
        }

        let q = unsafe { &(*qptr).inner };
        let results = c.search(q, k);

        for r in results {
            if count >= out_capacity {
                return count as i32;
            }

            out_slice[count] = HdrSparseEntry {
                row: row as u64,
                col: r.index as u64,
                value: r.similarity,
            };
            count += 1;
        }
    }

    count as i32
}

/// Batch process for GraphBLAS integration
///
/// Process multiple edge bindings efficiently for graph construction.
#[no_mangle]
pub extern "C" fn hdr_batch_bind_edges(
    sources: *const *const HdrVector,
    verbs: *const *const HdrVector,
    targets: *const *const HdrVector,
    count: usize,
    out: *mut *mut HdrVector,
) -> i32 {
    if sources.is_null() || verbs.is_null() || targets.is_null() || out.is_null() || count == 0 {
        return -1;
    }

    let src_slice = unsafe { slice::from_raw_parts(sources, count) };
    let verb_slice = unsafe { slice::from_raw_parts(verbs, count) };
    let tgt_slice = unsafe { slice::from_raw_parts(targets, count) };
    let out_slice = unsafe { slice::from_raw_parts_mut(out, count) };

    for i in 0..count {
        let src = src_slice[i];
        let verb = verb_slice[i];
        let tgt = tgt_slice[i];

        if src.is_null() || verb.is_null() || tgt.is_null() {
            out_slice[i] = ptr::null_mut();
            continue;
        }

        let vs = unsafe { &(*src).inner };
        let vv = unsafe { &(*verb).inner };
        let vt = unsafe { &(*tgt).inner };

        out_slice[i] = Box::into_raw(Box::new(HdrVector {
            inner: vs.xor(vv).xor(vt),
        }));
    }

    count as i32
}

// ============================================================================
// VERSION INFO
// ============================================================================

/// Get library version string
#[no_mangle]
pub extern "C" fn hdr_version() -> *const c_char {
    static VERSION: &[u8] = b"hdr-hamming 0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Get vector size in bits
#[no_mangle]
pub extern "C" fn hdr_vector_bits() -> usize {
    VECTOR_BITS
}

/// Get vector size in bytes
#[no_mangle]
pub extern "C" fn hdr_vector_bytes() -> usize {
    VECTOR_BYTES
}

/// Get vector size in words (u64)
#[no_mangle]
pub extern "C" fn hdr_vector_words() -> usize {
    VECTOR_WORDS
}

/// Get padded vector size in bytes (64-byte aligned for Arrow zero-copy)
#[no_mangle]
pub extern "C" fn hdr_vector_padded_bytes() -> usize {
    PADDED_VECTOR_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_vector_lifecycle() {
        let v1 = hdr_vector_random(12345);
        assert!(!v1.is_null());

        let v2 = hdr_vector_clone(v1);
        assert!(!v2.is_null());

        let dist = hdr_hamming_distance(v1, v2);
        assert_eq!(dist, 0); // Clone should be identical

        hdr_vector_free(v1);
        hdr_vector_free(v2);
    }

    #[test]
    fn test_ffi_bind_unbind() {
        let a = hdr_vector_random(1);
        let b = hdr_vector_random(2);

        let bound = hdr_vector_bind(a, b);
        assert!(!bound.is_null());

        let recovered = hdr_vector_unbind(bound, b);
        assert!(!recovered.is_null());

        // recovered should equal a
        let dist = hdr_hamming_distance(a, recovered);
        assert_eq!(dist, 0);

        hdr_vector_free(a);
        hdr_vector_free(b);
        hdr_vector_free(bound);
        hdr_vector_free(recovered);
    }

    #[test]
    fn test_ffi_cascade() {
        let cascade = hdr_cascade_create(100);
        assert!(!cascade.is_null());

        // Add vectors
        for i in 0..50 {
            let v = hdr_vector_random(i as u64 + 100);
            hdr_cascade_add(cascade, v);
            hdr_vector_free(v);
        }

        assert_eq!(hdr_cascade_len(cascade), 50);

        // Search
        let query = hdr_vector_random(125);
        let mut results = [HdrSearchResult {
            index: 0,
            distance: 0,
            similarity: 0.0,
            response: 0.0,
        }; 10];

        let n = hdr_cascade_search(cascade, query, 10, results.as_mut_ptr(), 10);
        assert!(n > 0);

        // First result should be exact match (index 25, seed 125)
        assert_eq!(results[0].index, 25);
        assert_eq!(results[0].distance, 0);

        hdr_vector_free(query);
        hdr_cascade_free(cascade);
    }
}
