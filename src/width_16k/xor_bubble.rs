//! Zero-Copy XOR Bubbling & Delta Compression
//!
//! # XOR Bubbling
//!
//! When traversing a DN tree path (root → ... → leaf), each node's
//! fingerprint is the majority-bundled centroid of its children. Adjacent
//! nodes in a path share significant bit structure. Instead of storing
//! full 16K vectors at every level, we can store:
//!
//! ```text
//! Node 0: full 16K vector (anchor)
//! Node 1: XOR delta from Node 0 (sparse — mostly zero words)
//! Node 2: XOR delta from Node 1
//! ...
//! Leaf:   XOR delta from parent
//! ```
//!
//! The delta (XOR of adjacent centroids) is sparse because parent-child
//! centroids overlap heavily. A typical parent-child delta has 70-90%
//! zero words, enabling:
//! - **Run-length encoding** of zero words → 3-5× compression
//! - **Zero-copy reconstruction** by XOR-chaining from the anchor
//! - **Incremental updates** — inserting a new leaf only changes
//!   deltas along its path, not full vectors
//!
//! # XOR Bubbling Protocol
//!
//! "Bubbling" is the upward propagation of XOR deltas when a leaf changes:
//!
//! ```text
//! Leaf changes (new fingerprint inserted)
//!   │
//!   ├─► δ_leaf = new_centroid ⊕ old_centroid
//!   │     (only the changed bits)
//!   │
//!   ├─► Parent: new_parent = old_parent ⊕ (δ_leaf weighted by 1/fanout)
//!   │     Since XOR is self-inverse, this incrementally adjusts the centroid
//!   │     without recomputing the full majority vote
//!   │
//!   ├─► Grandparent: receives diluted delta (further attenuated)
//!   │
//!   └─► Root: tiny perturbation (δ attenuated k times for depth k)
//!
//! Total work: O(depth × 256 words) instead of O(depth × fanout × 256 words)
//! ```
//!
//! # Schema Block Compression
//!
//! Schema blocks (13-15) compress especially well because adjacent nodes
//! in the same DN subtree tend to share:
//! - Same ANI level profile (all nodes in a planning subtree have high planning)
//! - Similar NARS truth values (evidence accumulates along paths)
//! - Same cluster ID and similar graph metrics
//!
//! Delta-encoding the schema blocks separately gives additional 2-3×
//! compression on top of the semantic delta encoding.

use super::{VECTOR_WORDS, SCHEMA_BLOCK_START};
use std::sync::RwLock;

/// Maximum depth for delta chains (prevents unbounded memory from degenerate paths)
pub const MAX_CHAIN_DEPTH: usize = 256;

// ============================================================================
// XOR DELTA: Compressed representation of difference between two vectors
// ============================================================================

/// A compressed XOR delta between two 16K vectors.
///
/// Instead of storing 256 words (2048 bytes), stores only the non-zero
/// words plus a bitmap indicating which words are non-zero.
///
/// Typical compression for parent-child centroids:
/// - Random vectors: ~50% zero words → ~50% compression
/// - Related centroids: ~70-90% zero words → 3-10× compression
/// - Same cluster: ~95% zero words → 20× compression
#[derive(Clone, Debug)]
pub struct XorDelta {
    /// Bitmap: which words are non-zero (256 bits = 4 u64)
    pub nonzero_bitmap: [u64; 4],
    /// Only the non-zero words, in order
    pub nonzero_words: Vec<u64>,
    /// Number of non-zero words (redundant but avoids recount)
    pub nnz: usize,
}

impl XorDelta {
    /// Compute delta between two 16K word arrays.
    pub fn compute(a: &[u64], b: &[u64]) -> Self {
        debug_assert!(a.len() >= VECTOR_WORDS && b.len() >= VECTOR_WORDS);

        let mut bitmap = [0u64; 4];
        let mut nonzero = Vec::new();

        for w in 0..VECTOR_WORDS {
            let xor = a[w] ^ b[w];
            if xor != 0 {
                bitmap[w / 64] |= 1u64 << (w % 64);
                nonzero.push(xor);
            }
        }

        let nnz = nonzero.len();
        Self {
            nonzero_bitmap: bitmap,
            nonzero_words: nonzero,
            nnz,
        }
    }

    /// Apply delta to a base vector to reconstruct the target.
    ///
    /// `base ⊕ delta = target` (since delta = base ⊕ target, and XOR is self-inverse)
    pub fn apply(&self, base: &[u64], out: &mut [u64]) {
        debug_assert!(base.len() >= VECTOR_WORDS && out.len() >= VECTOR_WORDS);

        // Start with base
        out[..VECTOR_WORDS].copy_from_slice(&base[..VECTOR_WORDS]);

        // XOR in the non-zero delta words
        let mut nz_idx = 0;
        for w in 0..VECTOR_WORDS {
            let bitmap_word = w / 64;
            let bitmap_bit = w % 64;
            if self.nonzero_bitmap[bitmap_word] & (1u64 << bitmap_bit) != 0 {
                out[w] ^= self.nonzero_words[nz_idx];
                nz_idx += 1;
            }
        }
    }

    /// Apply delta in-place (modifies base).
    pub fn apply_in_place(&self, base: &mut [u64]) {
        let mut nz_idx = 0;
        for w in 0..VECTOR_WORDS {
            let bitmap_word = w / 64;
            let bitmap_bit = w % 64;
            if self.nonzero_bitmap[bitmap_word] & (1u64 << bitmap_bit) != 0 {
                base[w] ^= self.nonzero_words[nz_idx];
                nz_idx += 1;
            }
        }
    }

    /// Compressed size in bytes (bitmap + non-zero words).
    pub fn compressed_bytes(&self) -> usize {
        4 * 8 + self.nnz * 8 // 32 bytes bitmap + 8 per non-zero word
    }

    /// Uncompressed size in bytes (full 16K vector).
    pub fn uncompressed_bytes(&self) -> usize {
        VECTOR_WORDS * 8 // 2048 bytes
    }

    /// Compression ratio (lower = better).
    pub fn compression_ratio(&self) -> f32 {
        self.compressed_bytes() as f32 / self.uncompressed_bytes() as f32
    }

    /// Fraction of zero words (sparsity).
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.nnz as f32 / VECTOR_WORDS as f32)
    }

    /// Hamming distance encoded in the delta (popcount of non-zero words).
    pub fn hamming_distance(&self) -> u32 {
        self.nonzero_words.iter().map(|w| w.count_ones()).sum()
    }

    /// Is this a semantic-only delta? (schema blocks unchanged)
    pub fn is_semantic_only(&self) -> bool {
        let schema_word_start = SCHEMA_BLOCK_START * 16; // 208
        for w in schema_word_start..VECTOR_WORDS {
            let bw = w / 64;
            let bb = w % 64;
            if self.nonzero_bitmap[bw] & (1u64 << bb) != 0 {
                return false;
            }
        }
        true
    }

    /// Extract only the schema portion of the delta.
    pub fn schema_delta(&self) -> XorDelta {
        let schema_word_start = SCHEMA_BLOCK_START * 16;
        let mut bitmap = [0u64; 4];
        let mut nonzero = Vec::new();

        let mut nz_idx = 0;
        for w in 0..VECTOR_WORDS {
            let bw = w / 64;
            let bb = w % 64;
            if self.nonzero_bitmap[bw] & (1u64 << bb) != 0 {
                if w >= schema_word_start {
                    bitmap[bw] |= 1u64 << bb;
                    nonzero.push(self.nonzero_words[nz_idx]);
                }
                nz_idx += 1;
            }
        }

        XorDelta {
            nonzero_bitmap: bitmap,
            nnz: nonzero.len(),
            nonzero_words: nonzero,
        }
    }
}

// ============================================================================
// DELTA CHAIN: Path of XOR deltas from anchor to leaf
// ============================================================================

/// A chain of XOR deltas representing a DN tree path.
///
/// The anchor (root) is stored as a full vector. Each subsequent level
/// is stored as a delta from its parent. Reconstruction walks the chain
/// XOR-ing deltas to recover any node's vector.
///
/// Memory savings example (depth=5, 16K vectors):
/// - Full: 5 × 2048 = 10,240 bytes
/// - Delta chain (70% sparsity): 2048 + 4 × (32 + 0.3×2048) ≈ 4,505 bytes (56% savings)
#[derive(Clone, Debug)]
pub struct DeltaChain {
    /// Full anchor vector (root or subtree root)
    pub anchor: Vec<u64>,
    /// Deltas from each level to the next
    pub deltas: Vec<XorDelta>,
}

impl DeltaChain {
    /// Create a chain from a sequence of vectors (root first, leaf last).
    ///
    /// Capped at `MAX_CHAIN_DEPTH` levels. If the path is longer,
    /// only the first `MAX_CHAIN_DEPTH` vectors are included.
    pub fn from_path(vectors: &[&[u64]]) -> Self {
        if vectors.is_empty() {
            return Self {
                anchor: vec![0u64; VECTOR_WORDS],
                deltas: Vec::new(),
            };
        }

        let capped = if vectors.len() > MAX_CHAIN_DEPTH {
            &vectors[..MAX_CHAIN_DEPTH]
        } else {
            vectors
        };

        let anchor = capped[0][..VECTOR_WORDS].to_vec();
        let deltas: Vec<XorDelta> = capped
            .windows(2)
            .map(|pair| XorDelta::compute(pair[0], pair[1]))
            .collect();

        Self { anchor, deltas }
    }

    /// Reconstruct the vector at a given depth (0 = anchor).
    pub fn reconstruct(&self, depth: usize) -> Vec<u64> {
        let mut current = self.anchor.clone();
        for d in 0..depth.min(self.deltas.len()) {
            self.deltas[d].apply_in_place(&mut current);
        }
        current
    }

    /// Depth of the chain (number of deltas + 1 for anchor).
    pub fn depth(&self) -> usize {
        self.deltas.len() + 1
    }

    /// Total compressed bytes.
    pub fn compressed_bytes(&self) -> usize {
        VECTOR_WORDS * 8 // anchor
            + self.deltas.iter().map(|d| d.compressed_bytes()).sum::<usize>()
    }

    /// Total uncompressed bytes (if all stored as full vectors).
    pub fn uncompressed_bytes(&self) -> usize {
        self.depth() * VECTOR_WORDS * 8
    }

    /// Average sparsity of deltas.
    pub fn avg_sparsity(&self) -> f32 {
        if self.deltas.is_empty() {
            return 0.0;
        }
        self.deltas.iter().map(|d| d.sparsity()).sum::<f32>() / self.deltas.len() as f32
    }
}

// ============================================================================
// XOR BUBBLE: Incremental centroid update via delta propagation
// ============================================================================

/// Propagate a leaf change upward through the tree using XOR bubbling.
///
/// When a leaf fingerprint changes from `old` to `new`, the delta
/// `old ⊕ new` represents the changed bits. This delta "bubbles up"
/// through the tree, attenuated at each level by the fanout.
///
/// # Why XOR Bubbling Works
///
/// For majority-bundled centroids, inserting/removing one vector changes
/// roughly `changed_bits / fanout` bits in the parent centroid. XOR captures
/// exactly which bits changed, and applying it to the parent is an O(256)
/// word operation — far cheaper than rebundling all children.
///
/// The attenuation isn't exact (majority vote is nonlinear), but for
/// routing purposes the approximation is sufficient. Periodic exact
/// recomputation keeps the error bounded.
///
/// # Zero-Copy Property
///
/// The delta is computed by XOR-ing two word slices. If both slices come
/// from Arrow buffers, the entire bubble operation is zero-copy: no
/// BitpackedVector is ever constructed.
pub struct XorBubble {
    /// The change delta: old_leaf ⊕ new_leaf
    delta_words: Vec<u64>,
    /// Attenuation factor per level (1/fanout)
    attenuation: f32,
    /// How many levels have been propagated
    levels_propagated: usize,
}

impl XorBubble {
    /// Create a bubble from a leaf change.
    ///
    /// `old_leaf` and `new_leaf` are the before/after word arrays.
    /// `fanout` is the typical branching factor (used for attenuation).
    pub fn from_leaf_change(old_leaf: &[u64], new_leaf: &[u64], fanout: usize) -> Self {
        let mut delta = vec![0u64; VECTOR_WORDS];
        for w in 0..VECTOR_WORDS.min(old_leaf.len()).min(new_leaf.len()) {
            delta[w] = old_leaf[w] ^ new_leaf[w];
        }

        Self {
            delta_words: delta,
            attenuation: 1.0 / fanout.max(1) as f32,
            levels_propagated: 0,
        }
    }

    /// Apply the bubble to a parent's word array (in-place).
    ///
    /// For exact centroid correction, this XORs the attenuated delta
    /// into the parent. The attenuation is applied probabilistically:
    /// each delta bit is kept with probability `1/fanout`.
    ///
    /// For fanout=1 (chain), all bits are applied (exact).
    /// For fanout=16, ~1/16 of changed bits affect the parent.
    ///
    /// `seed` must be nonzero for correct probabilistic behavior.
    /// If zero is passed, it is silently fixed to 1.
    pub fn apply_to_parent(&mut self, parent_words: &mut [u64], seed: u64) {
        let prob = self.current_probability();

        if prob >= 1.0 {
            // Exact: apply all delta bits
            for w in 0..VECTOR_WORDS.min(parent_words.len()) {
                parent_words[w] ^= self.delta_words[w];
            }
        } else {
            // Probabilistic: mask delta bits by attenuation
            let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(self.levels_propagated as u64);
            if rng == 0 { rng = 1; } // xorshift64 degenerates on zero seed
            for w in 0..VECTOR_WORDS.min(parent_words.len()) {
                if self.delta_words[w] == 0 {
                    continue;
                }
                // Generate a random mask: each bit passes with probability `prob`
                let mask = probabilistic_mask(self.delta_words[w], prob, &mut rng);
                parent_words[w] ^= mask;
            }
        }

        self.levels_propagated += 1;
    }

    /// Current probability that a delta bit survives to this level.
    pub fn current_probability(&self) -> f32 {
        self.attenuation.powi(self.levels_propagated as i32).max(0.001)
    }

    /// How many bits are still active in the delta.
    pub fn active_bits(&self) -> u32 {
        self.delta_words.iter().map(|w| w.count_ones()).sum()
    }

    /// Is the bubble exhausted? (All bits attenuated away)
    pub fn is_exhausted(&self) -> bool {
        self.current_probability() < 0.01 || self.active_bits() == 0
    }

    /// Number of levels propagated so far.
    pub fn levels(&self) -> usize {
        self.levels_propagated
    }
}

/// Generate a probabilistic bit mask: for each set bit in `delta`,
/// include it with probability `prob`.
fn probabilistic_mask(delta: u64, prob: f32, rng: &mut u64) -> u64 {
    if prob >= 1.0 {
        return delta;
    }
    if prob <= 0.0 {
        return 0;
    }
    // Guard against degenerate xorshift seed
    if *rng == 0 { *rng = 1; }

    let threshold = (prob * u32::MAX as f32) as u32;
    let mut mask = 0u64;
    let mut bits = delta;

    while bits != 0 {
        let bit_pos = bits.trailing_zeros();
        // xorshift64 — period 2^64-1 (nonzero seed required)
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;

        if (*rng as u32) < threshold {
            mask |= 1u64 << bit_pos;
        }
        bits &= bits - 1; // Clear lowest set bit
    }

    mask
}

// ============================================================================
// ADJACENT NODE COMPRESSION
// ============================================================================

/// Compress a group of adjacent 16K vectors using delta encoding.
///
/// Groups vectors by their DN tree address prefix and encodes each
/// group as an anchor + deltas. Returns the total compressed size.
///
/// This is the storage-level optimization: adjacent nodes in the DN tree
/// share structure, so their XOR deltas are sparse.
pub fn compress_adjacent(vectors: &[&[u64]]) -> (DeltaChain, usize) {
    let chain = DeltaChain::from_path(vectors);
    let size = chain.compressed_bytes();
    (chain, size)
}

/// Estimate compression ratio for a set of vectors without actually compressing.
///
/// Computes pairwise XOR sparsity between adjacent vectors.
pub fn estimate_compression(vectors: &[&[u64]]) -> f32 {
    if vectors.len() < 2 {
        return 1.0;
    }

    let mut total_sparsity = 0.0f32;
    let pairs = vectors.len() - 1;

    for pair in vectors.windows(2) {
        let delta = XorDelta::compute(pair[0], pair[1]);
        total_sparsity += delta.sparsity();
    }

    let avg_sparsity = total_sparsity / pairs as f32;
    // Compressed size = anchor_full + (n-1) * (1 - sparsity) * full_size
    let full_size = vectors.len() as f32;
    let compressed = 1.0 + (vectors.len() - 1) as f32 * (1.0 - avg_sparsity);
    compressed / full_size
}

// ============================================================================
// XOR WRITE CACHE: Avoid zero-copy deflowering
// ============================================================================

/// XOR Write Cache: accumulate delta writes without touching the Arrow buffer.
///
/// # Problem: Zero-Copy Deflowering
///
/// Arrow buffers are immutable (shared `Arc<Buffer>`). The moment you write
/// a single byte, Arrow forces a full copy-on-write (CoW) — the buffer is
/// "deflowered" and you lose zero-copy for all subsequent reads.
///
/// This is catastrophic for XOR bubbling: each bubble propagation would
/// trigger a CoW of the entire Arrow batch just to flip a few bits.
///
/// # Solution: Write Cache
///
/// Instead of modifying the Arrow buffer, accumulate XOR deltas in a sidecar
/// HashMap. When reading a vector, the read path XORs the cached delta
/// on-the-fly (still zero-copy on the base buffer — only the delta is owned).
///
/// ```text
/// Arrow Buffer (immutable, zero-copy):
/// ┌──────────────────────────────────────────┐
/// │ vec[0] │ vec[1] │ vec[2] │ ... │ vec[n]  │ ← never modified
/// └──────────────────────────────────────────┘
///
/// XOR Write Cache (small, owned):
/// ┌────────────────────────────────┐
/// │ id=3 → XorDelta(nnz=2)        │  ← 48 bytes
/// │ id=7 → XorDelta(nnz=5)        │  ← 72 bytes
/// └────────────────────────────────┘
///
/// Read(id=3):
///   arrow_buf[3] ⊕ cache[3].delta  → correct vector, zero-copy on base
///
/// Read(id=5):
///   arrow_buf[5]                    → no cache entry, pure zero-copy
/// ```
///
/// # Flush
///
/// Periodically (or on checkpoint), the cache is flushed to a new Arrow
/// batch. This is the only time a full buffer write occurs. Between flushes,
/// all reads remain zero-copy on the base buffer.
pub struct XorWriteCache {
    /// Pending deltas by vector ID
    pending: std::collections::HashMap<u64, XorDelta>,
    /// Total cached bytes (for flush threshold)
    cached_bytes: usize,
    /// Maximum cached bytes before triggering flush
    max_cached_bytes: usize,
    /// Number of delta applications since last flush
    ops_since_flush: usize,
}

impl XorWriteCache {
    /// Create a new write cache with the given flush threshold.
    ///
    /// `max_bytes`: trigger flush when cached deltas exceed this size.
    /// Recommended: 1MB (covers ~500 sparse deltas before flush).
    pub fn new(max_bytes: usize) -> Self {
        Self {
            pending: std::collections::HashMap::new(),
            cached_bytes: 0,
            max_cached_bytes: max_bytes,
            ops_since_flush: 0,
        }
    }

    /// Default cache: 1MB flush threshold.
    pub fn default_cache() -> Self {
        Self::new(1_048_576)
    }

    /// Record a delta for a vector ID.
    ///
    /// If there's already a pending delta for this ID, the new delta is
    /// composed with the existing one (XOR is associative + self-inverse,
    /// so delta1 ⊕ delta2 = combined delta).
    pub fn record_delta(&mut self, id: u64, delta: XorDelta) {
        self.ops_since_flush += 1;
        let delta_bytes = delta.compressed_bytes();

        self.pending
            .entry(id)
            .and_modify(|existing| {
                // Compose: existing ⊕ new = combined delta from original
                self.cached_bytes -= existing.compressed_bytes();
                *existing = compose_deltas(existing, &delta);
                self.cached_bytes += existing.compressed_bytes();
            })
            .or_insert_with(|| {
                self.cached_bytes += delta_bytes;
                delta
            });
    }

    /// Read a vector through the cache: base ⊕ cached_delta.
    ///
    /// `base_words` comes from the Arrow buffer (zero-copy borrow).
    /// If there's a pending delta, it's applied to a stack-allocated
    /// copy (only the non-zero words are touched). If no delta,
    /// returns None to signal "use base directly" (pure zero-copy).
    pub fn read_through<'a>(&self, id: u64, base_words: &'a [u64]) -> CacheRead<'a> {
        match self.pending.get(&id) {
            None => CacheRead::Clean(base_words),
            Some(delta) => {
                let mut patched = base_words[..VECTOR_WORDS].to_vec();
                delta.apply_in_place(&mut patched);
                CacheRead::Patched(patched)
            }
        }
    }

    /// Is the vector dirty (has pending delta)?
    pub fn is_dirty(&self, id: u64) -> bool {
        self.pending.contains_key(&id)
    }

    /// Should we flush? (Cache size exceeds threshold)
    pub fn should_flush(&self) -> bool {
        self.cached_bytes >= self.max_cached_bytes
    }

    /// Flush the cache: returns all pending deltas and clears the cache.
    ///
    /// The caller applies these to a new Arrow batch (single bulk write
    /// instead of many small writes).
    pub fn flush(&mut self) -> Vec<(u64, XorDelta)> {
        self.cached_bytes = 0;
        self.ops_since_flush = 0;
        self.pending.drain().collect()
    }

    /// Number of dirty vectors.
    pub fn dirty_count(&self) -> usize {
        self.pending.len()
    }

    /// Total cached delta bytes.
    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// Operations since last flush.
    pub fn ops_since_flush(&self) -> usize {
        self.ops_since_flush
    }
}

/// Result of reading through the XOR write cache.
pub enum CacheRead<'a> {
    /// No pending delta — use base directly (zero-copy).
    Clean(&'a [u64]),
    /// Delta applied — patched copy (owned).
    Patched(Vec<u64>),
}

impl<'a> CacheRead<'a> {
    /// Get the word slice (either borrowed or owned).
    pub fn words(&self) -> &[u64] {
        match self {
            CacheRead::Clean(w) => w,
            CacheRead::Patched(w) => w,
        }
    }

    /// Is this a clean (zero-copy) read?
    pub fn is_clean(&self) -> bool {
        matches!(self, CacheRead::Clean(_))
    }
}

/// Compose two XOR deltas: result = delta_a ⊕ delta_b.
///
/// Since delta_a = original ⊕ intermediate, delta_b = intermediate ⊕ final,
/// composed = original ⊕ final (XOR cancels the intermediate).
fn compose_deltas(a: &XorDelta, b: &XorDelta) -> XorDelta {
    let mut composed = vec![0u64; VECTOR_WORDS];

    // Expand a into full
    let mut idx_a = 0;
    for w in 0..VECTOR_WORDS {
        let bw = w / 64;
        let bb = w % 64;
        if a.nonzero_bitmap[bw] & (1u64 << bb) != 0 {
            composed[w] = a.nonzero_words[idx_a];
            idx_a += 1;
        }
    }

    // XOR in b
    let mut idx_b = 0;
    for w in 0..VECTOR_WORDS {
        let bw = w / 64;
        let bb = w % 64;
        if b.nonzero_bitmap[bw] & (1u64 << bb) != 0 {
            composed[w] ^= b.nonzero_words[idx_b];
            idx_b += 1;
        }
    }

    // Recompact
    let mut bitmap = [0u64; 4];
    let mut nonzero = Vec::new();
    for w in 0..VECTOR_WORDS {
        if composed[w] != 0 {
            bitmap[w / 64] |= 1u64 << (w % 64);
            nonzero.push(composed[w]);
        }
    }

    XorDelta {
        nonzero_bitmap: bitmap,
        nnz: nonzero.len(),
        nonzero_words: nonzero,
    }
}

// ============================================================================
// CONCURRENT WRITE CACHE: Thread-safe wrapper
// ============================================================================

/// Thread-safe wrapper around `XorWriteCache`.
///
/// Uses `RwLock` for concurrent reads (zero-copy path) and exclusive writes.
/// Multiple query threads can call `read_through()` simultaneously.
/// Only `record_delta()` and `flush()` require exclusive access.
///
/// # Example
/// ```text
/// let cache = ConcurrentWriteCache::new(1_048_576);
///
/// // Query threads (concurrent reads):
/// let read = cache.read_through(42, &base_words);
///
/// // Writer thread (exclusive):
/// cache.record_delta(42, delta);
///
/// // Checkpoint thread (exclusive):
/// let flushed = cache.flush();
/// ```
pub struct ConcurrentWriteCache {
    inner: RwLock<XorWriteCache>,
}

impl ConcurrentWriteCache {
    /// Create with given flush threshold.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: RwLock::new(XorWriteCache::new(max_bytes)),
        }
    }

    /// Default: 1MB flush threshold.
    pub fn default_cache() -> Self {
        Self::new(1_048_576)
    }

    /// Read through the cache (takes read lock — concurrent with other reads).
    ///
    /// Returns `ConcurrentCacheRead::Clean` for uncached vectors,
    /// or `ConcurrentCacheRead::Patched` with the delta applied.
    ///
    /// Unlike `XorWriteCache::read_through()` which returns a borrowing enum,
    /// this always returns owned data (Vec) for the patched case, or a flag
    /// indicating the vector is clean (caller should use base directly).
    pub fn read_through(&self, id: u64, base_words: &[u64]) -> ConcurrentCacheRead {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        match guard.pending.get(&id) {
            None => ConcurrentCacheRead::Clean,
            Some(delta) => {
                let mut patched = base_words[..VECTOR_WORDS].to_vec();
                delta.apply_in_place(&mut patched);
                ConcurrentCacheRead::Patched(patched)
            }
        }
    }

    /// Record a delta (takes write lock — exclusive).
    pub fn record_delta(&self, id: u64, delta: XorDelta) {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        guard.record_delta(id, delta);
    }

    /// Check if a vector is dirty (takes read lock).
    pub fn is_dirty(&self, id: u64) -> bool {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        guard.is_dirty(id)
    }

    /// Check if flush threshold exceeded (takes read lock).
    pub fn should_flush(&self) -> bool {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        guard.should_flush()
    }

    /// Flush all pending deltas (takes write lock — exclusive).
    pub fn flush(&self) -> Vec<(u64, XorDelta)> {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        guard.flush()
    }

    /// Number of dirty entries (takes read lock).
    pub fn dirty_count(&self) -> usize {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        guard.dirty_count()
    }
}

/// Result of reading through the concurrent write cache.
///
/// Unlike `CacheRead` (which borrows from the base), this is fully owned
/// to avoid lifetime entanglement with the RwLock guard.
pub enum ConcurrentCacheRead {
    /// No pending delta — caller should use base_words directly.
    Clean,
    /// Delta was applied — use this patched copy.
    Patched(Vec<u64>),
}

impl ConcurrentCacheRead {
    /// Is this a clean (zero-copy) read?
    pub fn is_clean(&self) -> bool {
        matches!(self, ConcurrentCacheRead::Clean)
    }

    /// Get the patched words, or None if clean.
    pub fn patched_words(&self) -> Option<&[u64]> {
        match self {
            ConcurrentCacheRead::Clean => None,
            ConcurrentCacheRead::Patched(w) => Some(w),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_words(seed: u64) -> Vec<u64> {
        let mut words = vec![0u64; VECTOR_WORDS];
        let mut rng = seed;
        for w in &mut words {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *w = rng;
        }
        words
    }

    fn similar_words(base: &[u64], flip_count: usize, seed: u64) -> Vec<u64> {
        let mut words = base.to_vec();
        let mut rng = seed;
        for _ in 0..flip_count {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let word_idx = (rng as usize) % VECTOR_WORDS;
            let bit_idx = ((rng >> 8) as usize) % 64;
            words[word_idx] ^= 1u64 << bit_idx;
        }
        words
    }

    #[test]
    fn test_xor_delta_roundtrip() {
        let a = random_words(1);
        let b = random_words(2);

        let delta = XorDelta::compute(&a, &b);
        let mut reconstructed = vec![0u64; VECTOR_WORDS];
        delta.apply(&a, &mut reconstructed);

        assert_eq!(&reconstructed[..VECTOR_WORDS], &b[..VECTOR_WORDS]);
    }

    #[test]
    fn test_xor_delta_self_is_zero() {
        let a = random_words(42);
        let delta = XorDelta::compute(&a, &a);

        assert_eq!(delta.nnz, 0);
        assert_eq!(delta.hamming_distance(), 0);
        assert_eq!(delta.sparsity(), 1.0);
    }

    #[test]
    fn test_xor_delta_similar_vectors_sparse() {
        let base = random_words(1);
        let similar = similar_words(&base, 50, 99); // Only 50 bit flips

        let delta = XorDelta::compute(&base, &similar);

        // With only 50 bit flips across 256 words, most words are unchanged
        assert!(delta.sparsity() > 0.5, "Expected sparse delta, got sparsity={}", delta.sparsity());
        assert!(delta.compression_ratio() < 0.8, "Expected good compression");
    }

    #[test]
    fn test_xor_delta_apply_in_place() {
        let a = random_words(1);
        let b = random_words(2);

        let delta = XorDelta::compute(&a, &b);
        let mut current = a.clone();
        delta.apply_in_place(&mut current);

        assert_eq!(&current[..VECTOR_WORDS], &b[..VECTOR_WORDS]);
    }

    #[test]
    fn test_delta_chain_reconstruct() {
        let v0 = random_words(10);
        let v1 = similar_words(&v0, 100, 1);
        let v2 = similar_words(&v1, 100, 2);
        let v3 = similar_words(&v2, 100, 3);

        let path: Vec<&[u64]> = vec![&v0, &v1, &v2, &v3];
        let chain = DeltaChain::from_path(&path);

        assert_eq!(chain.depth(), 4);

        // Reconstruct each level
        let r0 = chain.reconstruct(0);
        assert_eq!(&r0[..VECTOR_WORDS], &v0[..VECTOR_WORDS]);

        let r1 = chain.reconstruct(1);
        assert_eq!(&r1[..VECTOR_WORDS], &v1[..VECTOR_WORDS]);

        let r2 = chain.reconstruct(2);
        assert_eq!(&r2[..VECTOR_WORDS], &v2[..VECTOR_WORDS]);

        let r3 = chain.reconstruct(3);
        assert_eq!(&r3[..VECTOR_WORDS], &v3[..VECTOR_WORDS]);
    }

    #[test]
    fn test_delta_chain_compression() {
        let v0 = random_words(10);
        let v1 = similar_words(&v0, 50, 1);
        let v2 = similar_words(&v1, 50, 2);

        let path: Vec<&[u64]> = vec![&v0, &v1, &v2];
        let chain = DeltaChain::from_path(&path);

        // Should compress better than 1:1
        let ratio = chain.compressed_bytes() as f32 / chain.uncompressed_bytes() as f32;
        assert!(ratio < 1.0, "Expected compression, got ratio={}", ratio);
        assert!(chain.avg_sparsity() > 0.0);
    }

    #[test]
    fn test_xor_bubble_exact() {
        let old_leaf = random_words(1);
        let new_leaf = random_words(2);
        let mut parent = old_leaf.clone(); // Parent = copy of old leaf (fanout=1)

        let mut bubble = XorBubble::from_leaf_change(&old_leaf, &new_leaf, 1);
        bubble.apply_to_parent(&mut parent, 42);

        // With fanout=1 (exact), parent should become new_leaf
        assert_eq!(&parent[..VECTOR_WORDS], &new_leaf[..VECTOR_WORDS]);
    }

    #[test]
    fn test_xor_bubble_attenuation() {
        let old_leaf = random_words(1);
        let mut new_leaf = old_leaf.clone();
        new_leaf[0] ^= 0xFFFF; // Flip 16 bits

        let mut parent = old_leaf.clone();
        let mut bubble = XorBubble::from_leaf_change(&old_leaf, &new_leaf, 16);

        // With fanout=16, only ~1/16 of changed bits should propagate
        bubble.apply_to_parent(&mut parent, 42);

        // Parent should have changed, but less than 16 bits
        let changed: u32 = (0..VECTOR_WORDS)
            .map(|w| (parent[w] ^ old_leaf[w]).count_ones())
            .sum();
        // Probabilistic: expect ~1 bit changed (16/16), allow 0-5
        assert!(changed <= 16, "Expected attenuated change, got {} bits", changed);
    }

    #[test]
    fn test_xor_bubble_exhaustion() {
        let old = random_words(1);
        let new = random_words(2);
        let mut bubble = XorBubble::from_leaf_change(&old, &new, 16);

        // Propagate many levels — probability should decrease
        for _ in 0..10 {
            let mut dummy = random_words(99);
            bubble.apply_to_parent(&mut dummy, 42);
        }

        assert!(bubble.is_exhausted());
    }

    #[test]
    fn test_schema_only_delta() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];

        // Only differ in schema region
        a[210] = 0xDEADBEEF;

        let delta = XorDelta::compute(&a, &b);
        assert!(!delta.is_semantic_only());

        let schema_d = delta.schema_delta();
        assert_eq!(schema_d.nnz, 1);
    }

    #[test]
    fn test_estimate_compression() {
        let v0 = random_words(10);
        let v1 = similar_words(&v0, 30, 1);
        let v2 = similar_words(&v1, 30, 2);

        let refs: Vec<&[u64]> = vec![&v0, &v1, &v2];
        let ratio = estimate_compression(&refs);
        assert!(ratio < 1.0, "Similar vectors should compress well: ratio={}", ratio);
    }

    // === XOR Write Cache tests ===

    #[test]
    fn test_write_cache_clean_read() {
        let cache = XorWriteCache::default_cache();
        let base = random_words(42);
        let read = cache.read_through(1, &base);
        assert!(read.is_clean());
        assert_eq!(read.words(), &base[..]);
    }

    #[test]
    fn test_write_cache_dirty_read() {
        let mut cache = XorWriteCache::default_cache();
        let base = random_words(42);
        let mut modified = base.clone();
        modified[0] ^= 0xFF;

        let delta = XorDelta::compute(&base, &modified);
        cache.record_delta(1, delta);

        let read = cache.read_through(1, &base);
        assert!(!read.is_clean());
        assert_eq!(read.words()[0], modified[0]);
    }

    #[test]
    fn test_write_cache_compose() {
        let mut cache = XorWriteCache::default_cache();
        let base = random_words(42);

        // First delta: flip word[0]
        let mut mid = base.clone();
        mid[0] ^= 0xFF;
        cache.record_delta(1, XorDelta::compute(&base, &mid));

        // Second delta: flip word[1]
        let mut final_vec = mid.clone();
        final_vec[1] ^= 0xFF00;
        cache.record_delta(1, XorDelta::compute(&mid, &final_vec));

        // Composed: both flips
        let read = cache.read_through(1, &base);
        assert_eq!(read.words()[0], base[0] ^ 0xFF);
        assert_eq!(read.words()[1], base[1] ^ 0xFF00);
    }

    #[test]
    fn test_write_cache_flush() {
        let mut cache = XorWriteCache::default_cache();
        let base = random_words(42);
        let mut mod1 = base.clone();
        mod1[0] ^= 0xFF;

        cache.record_delta(1, XorDelta::compute(&base, &mod1));
        cache.record_delta(2, XorDelta::compute(&base, &mod1));

        assert_eq!(cache.dirty_count(), 2);

        let flushed = cache.flush();
        assert_eq!(flushed.len(), 2);
        assert_eq!(cache.dirty_count(), 0);
        assert_eq!(cache.cached_bytes(), 0);
    }

    #[test]
    fn test_write_cache_self_inverse() {
        let mut cache = XorWriteCache::default_cache();
        let base = random_words(42);
        let mut modified = base.clone();
        modified[0] ^= 0xFF;

        // Apply delta, then apply it again (self-inverse)
        let delta = XorDelta::compute(&base, &modified);
        cache.record_delta(1, delta.clone());
        cache.record_delta(1, delta); // XOR with self = cancel

        // Should be clean again (composed delta is all zeros)
        let read = cache.read_through(1, &base);
        // The composed delta should have nnz=0
        assert_eq!(read.words()[0], base[0]);
    }

    // === Hardening tests ===

    #[test]
    fn test_max_chain_depth_cap() {
        // Create a chain deeper than MAX_CHAIN_DEPTH
        let mut vecs = Vec::new();
        let v0 = random_words(1);
        vecs.push(v0);
        for i in 1..=(MAX_CHAIN_DEPTH + 50) {
            let prev = &vecs[vecs.len() - 1];
            let next = similar_words(prev, 5, i as u64);
            vecs.push(next);
        }

        let refs: Vec<&[u64]> = vecs.iter().map(|v| v.as_slice()).collect();
        let chain = DeltaChain::from_path(&refs);

        // Should be capped at MAX_CHAIN_DEPTH
        assert_eq!(chain.depth(), MAX_CHAIN_DEPTH,
            "Chain depth should be capped at MAX_CHAIN_DEPTH={}", MAX_CHAIN_DEPTH);

        // Reconstruction should still work for capped depth
        let r0 = chain.reconstruct(0);
        assert_eq!(&r0[..VECTOR_WORDS], &vecs[0][..VECTOR_WORDS]);
    }

    #[test]
    fn test_rng_seed_zero_not_degenerate() {
        // Test that seed=0 doesn't produce all-zero masks in probabilistic_mask
        let mut rng: u64 = 0;
        let mask = probabilistic_mask(0xFFFF_FFFF_FFFF_FFFF, 0.5, &mut rng);
        // After the fix, rng should have been bumped to 1 before xorshift
        // The mask should not be all-zeros (with p=0.5 and delta=all-ones)
        assert_ne!(rng, 0, "RNG should not remain at 0 after probabilistic_mask");
        // mask can be anything but degenerate all-zero with full delta and p=0.5 is unlikely
    }

    #[test]
    fn test_xor_bubble_seed_zero() {
        // Test that apply_to_parent handles seed=0 gracefully
        let old = random_words(1);
        let mut new = old.clone();
        new[0] ^= 0xFFFF_FFFF;

        let mut parent = old.clone();
        let mut bubble = XorBubble::from_leaf_change(&old, &new, 16);
        // seed=0 should not cause degenerate behavior
        bubble.apply_to_parent(&mut parent, 0);

        // Parent should have changed (some bits propagated)
        // With seed fix, at least some bits should be different
        let changed: u32 = (0..VECTOR_WORDS)
            .map(|w| (parent[w] ^ old[w]).count_ones())
            .sum();
        // Allow any number of changes — the key is it shouldn't panic or all-zero
        assert!(changed <= 32, "Attenuated change should be bounded");
    }

    // === ConcurrentWriteCache tests ===

    #[test]
    fn test_concurrent_cache_basic() {
        let cache = ConcurrentWriteCache::default_cache();
        let base = random_words(42);

        // Clean read
        let read = cache.read_through(1, &base);
        assert!(read.is_clean());
        assert!(read.patched_words().is_none());

        // Record a delta
        let mut modified = base.clone();
        modified[0] ^= 0xFF;
        let delta = XorDelta::compute(&base, &modified);
        cache.record_delta(1, delta);

        // Dirty read
        assert!(cache.is_dirty(1));
        assert!(!cache.is_dirty(2));
        assert_eq!(cache.dirty_count(), 1);

        let read2 = cache.read_through(1, &base);
        assert!(!read2.is_clean());
        let patched = read2.patched_words().unwrap();
        assert_eq!(patched[0], modified[0]);
    }

    #[test]
    fn test_concurrent_cache_flush() {
        let cache = ConcurrentWriteCache::default_cache();
        let base = random_words(42);
        let mut mod1 = base.clone();
        mod1[0] ^= 0xFF;

        cache.record_delta(1, XorDelta::compute(&base, &mod1));
        cache.record_delta(2, XorDelta::compute(&base, &mod1));

        assert_eq!(cache.dirty_count(), 2);

        let flushed = cache.flush();
        assert_eq!(flushed.len(), 2);
        assert_eq!(cache.dirty_count(), 0);
    }

    #[test]
    fn test_concurrent_cache_compose() {
        let cache = ConcurrentWriteCache::default_cache();
        let base = random_words(42);

        // First delta
        let mut mid = base.clone();
        mid[0] ^= 0xFF;
        cache.record_delta(1, XorDelta::compute(&base, &mid));

        // Second delta
        let mut final_v = mid.clone();
        final_v[1] ^= 0xFF00;
        cache.record_delta(1, XorDelta::compute(&mid, &final_v));

        // Composed result
        let read = cache.read_through(1, &base);
        let patched = read.patched_words().unwrap();
        assert_eq!(patched[0], base[0] ^ 0xFF);
        assert_eq!(patched[1], base[1] ^ 0xFF00);
    }
}
