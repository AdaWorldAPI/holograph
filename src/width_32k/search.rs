//! 3D Holographic Search — Dimensional Cascade, Probe Search, Weighted Distance
//!
//! Three search modes unique to the 32K holographic layout:
//!
//! # 1. Dimensional Cascade Search
//!
//! ```text
//! Candidate pool (n vectors)
//!   │
//!   ├─► Level 0: Schema predicate filter (metadata block, O(1))
//!   │     Read from metadata words, check ANI/NARS/RL/graph predicates
//!   │
//!   ├─► Level 1: Dominant dimension distance (16 AVX-512 iterations)
//!   │     Compute distance on the highest-weighted dimension only
//!   │     Rejects ~80% of survivors at 1/3 the SIMD cost
//!   │
//!   ├─► Level 2: Full semantic distance (48 AVX-512 iterations)
//!   │     All three dimensions: X + Y + Z, weighted
//!   │
//!   └─► Level 3: Top-k selection
//! ```
//!
//! # 2. Holographic Probe Search
//!
//! Given two known dimensions, XOR-probe to recover the third, then rank
//! by closeness to a target. Answers relational queries without graph traversal.
//!
//! # 3. Bloom-Accelerated Dimensional Search
//!
//! Like 16K bloom search, but with per-dimension distance + 512-bit bloom.

use super::holographic::{HoloVector, HoloTrace, ProbeResult};
use super::schema::{HoloSchema, M_ANI_BASE, M_NARS_TRUTH, M_BLOOM_BASE, M_GRAPH_BASE, M_RL_BASE, M_VERSION};
use super::*;

// ============================================================================
// DIMENSION IDENTIFIER
// ============================================================================

/// Which dimension to operate on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dimension {
    X,  // Content / What
    Y,  // Context / Where
    Z,  // Relation / How
}

impl Dimension {
    /// Word range for this dimension.
    pub fn range(&self) -> (usize, usize) {
        match self {
            Dimension::X => (X_START, X_END),
            Dimension::Y => (Y_START, Y_END),
            Dimension::Z => (Z_START, Z_END),
        }
    }
}

// ============================================================================
// DIMENSIONAL WEIGHTS
// ============================================================================

/// Per-dimension weights for search.
///
/// Controls the relative importance of content (X), context (Y), and
/// relation (Z) in distance computation.
#[derive(Clone, Copy, Debug)]
pub struct DimWeights {
    pub wx: f64,
    pub wy: f64,
    pub wz: f64,
}

impl DimWeights {
    /// Content-focused: mostly X, some Y, little Z
    pub const CONTENT: Self = Self { wx: 0.7, wy: 0.2, wz: 0.1 };

    /// Context-focused: mostly Y
    pub const CONTEXT: Self = Self { wx: 0.2, wy: 0.7, wz: 0.1 };

    /// Relation-focused: mostly Z
    pub const RELATION: Self = Self { wx: 0.1, wy: 0.2, wz: 0.7 };

    /// Equal weight across all dimensions
    pub const BALANCED: Self = Self { wx: 1.0 / 3.0, wy: 1.0 / 3.0, wz: 1.0 / 3.0 };

    /// Content + Context (ignore relation)
    pub const SEMANTIC: Self = Self { wx: 0.5, wy: 0.5, wz: 0.0 };

    /// Custom weights.
    pub const fn new(wx: f64, wy: f64, wz: f64) -> Self {
        Self { wx, wy, wz }
    }

    /// The dimension with the highest weight (for dominant-dim-first cascade).
    pub fn dominant(&self) -> Dimension {
        if self.wx >= self.wy && self.wx >= self.wz {
            Dimension::X
        } else if self.wy >= self.wz {
            Dimension::Y
        } else {
            Dimension::Z
        }
    }

    /// Weight for a specific dimension.
    pub fn weight_for(&self, dim: Dimension) -> f64 {
        match dim {
            Dimension::X => self.wx,
            Dimension::Y => self.wy,
            Dimension::Z => self.wz,
        }
    }
}

impl Default for DimWeights {
    fn default() -> Self {
        Self::BALANCED
    }
}

// ============================================================================
// SCHEMA PREDICATES (adapted for 32K metadata block)
// ============================================================================

/// ANI level filter for 32K vectors.
#[derive(Clone, Debug)]
pub struct AniFilter {
    pub min_level: u8,
    pub min_activation: u16,
}

/// NARS truth filter.
#[derive(Clone, Debug)]
pub struct NarsFilter {
    pub min_frequency: Option<f32>,
    pub min_confidence: Option<f32>,
}

/// Graph topology filter.
#[derive(Clone, Debug)]
pub struct GraphFilter {
    pub min_pagerank: Option<u16>,
    pub max_hop: Option<u8>,
    pub cluster_id: Option<u16>,
    pub min_degree: Option<u8>,
}

// ============================================================================
// DIMENSIONAL DISTANCE HELPERS
// ============================================================================

/// Hamming distance on a single dimension (128 words).
#[inline]
fn dim_distance(a: &[u64], b: &[u64], start: usize) -> u32 {
    let mut total = 0u32;
    for i in 0..DIM_WORDS {
        total += (a[start + i] ^ b[start + i]).count_ones();
    }
    total
}

/// Hamming distance on a single dimension with early termination.
#[inline]
fn dim_distance_with_threshold(a: &[u64], b: &[u64], start: usize, threshold: u32) -> Option<u32> {
    let mut total = 0u32;
    // Check in 16-word blocks (1024 bits each, 8 blocks per dim)
    for block in 0..DIM_BLOCKS {
        let block_start = start + block * DIM_WORDS_PER_BLOCK;
        for i in 0..DIM_WORDS_PER_BLOCK {
            total += (a[block_start + i] ^ b[block_start + i]).count_ones();
        }
        if total > threshold {
            return None;
        }
    }
    Some(total)
}

/// Per-dimension distances (X, Y, Z) in one pass.
#[inline]
fn tri_distance(a: &[u64], b: &[u64]) -> (u32, u32, u32) {
    let dx = dim_distance(a, b, X_START);
    let dy = dim_distance(a, b, Y_START);
    let dz = dim_distance(a, b, Z_START);
    (dx, dy, dz)
}

/// Weighted distance from per-dimension distances.
#[inline]
fn weighted_from_tri(dx: u32, dy: u32, dz: u32, w: &DimWeights) -> f64 {
    w.wx * dx as f64 + w.wy * dy as f64 + w.wz * dz as f64
}

// ============================================================================
// DIMENSIONAL SEARCH QUERY
// ============================================================================

/// A dimensional search query for 32K holographic vectors.
///
/// Combines schema predicate filtering with per-dimension weighted distance.
/// The dominant-dimension-first cascade eliminates candidates at 1/3 the cost
/// of a full semantic distance computation.
#[derive(Clone, Debug)]
pub struct DimSearchQuery {
    /// Per-dimension weights
    pub weights: DimWeights,
    /// Schema predicate filters
    pub ani_filter: Option<AniFilter>,
    pub nars_filter: Option<NarsFilter>,
    pub graph_filter: Option<GraphFilter>,
    /// Maximum weighted distance (for early termination)
    pub max_distance: Option<f64>,
}

impl DimSearchQuery {
    pub fn new(weights: DimWeights) -> Self {
        Self {
            weights,
            ani_filter: None,
            nars_filter: None,
            graph_filter: None,
            max_distance: None,
        }
    }

    pub fn with_ani(mut self, filter: AniFilter) -> Self {
        self.ani_filter = Some(filter);
        self
    }

    pub fn with_nars(mut self, filter: NarsFilter) -> Self {
        self.nars_filter = Some(filter);
        self
    }

    pub fn with_graph(mut self, filter: GraphFilter) -> Self {
        self.graph_filter = Some(filter);
        self
    }

    pub fn with_max_distance(mut self, d: f64) -> Self {
        self.max_distance = Some(d);
        self
    }

    /// Check schema predicates against the metadata block.
    ///
    /// Reads from `words[META_START..]` — the 128-word metadata region.
    pub fn passes_predicates(&self, words: &[u64]) -> bool {
        if words.len() < VECTOR_WORDS {
            return false;
        }

        // ANI filter: metadata word M_ANI_BASE (word 384 in absolute terms)
        if let Some(ref ani) = self.ani_filter {
            let w = words[META_START + M_ANI_BASE];
            let level = (w & 0xFF) as u8;
            let activation = ((w >> 16) & 0xFFFF) as u16;
            if level < ani.min_level {
                return false;
            }
            if activation < ani.min_activation {
                return false;
            }
        }

        // NARS filter: metadata word M_NARS_TRUTH
        if let Some(ref nars) = self.nars_filter {
            let w = words[META_START + M_NARS_TRUTH];
            let freq_u16 = (w & 0xFFFF) as u16;
            let conf_u16 = ((w >> 16) & 0xFFFF) as u16;
            let freq = freq_u16 as f32 / 65535.0;
            let conf = conf_u16 as f32 / 65535.0;
            if let Some(min_f) = nars.min_frequency {
                if freq < min_f { return false; }
            }
            if let Some(min_c) = nars.min_confidence {
                if conf < min_c { return false; }
            }
        }

        // Graph filter: metadata words at M_GRAPH_BASE
        if let Some(ref graph) = self.graph_filter {
            let w = words[META_START + M_GRAPH_BASE];
            let pagerank = (w & 0xFFFF) as u16;
            let hop = ((w >> 16) & 0xFF) as u8;
            let cluster = ((w >> 24) & 0xFFFF) as u16;
            let degree = ((w >> 40) & 0xFF) as u8;
            if let Some(min_pr) = graph.min_pagerank {
                if pagerank < min_pr { return false; }
            }
            if let Some(max_h) = graph.max_hop {
                if hop > max_h { return false; }
            }
            if let Some(cid) = graph.cluster_id {
                if cluster != cid { return false; }
            }
            if let Some(min_d) = graph.min_degree {
                if degree < min_d { return false; }
            }
        }

        true
    }

    /// Dominant-dimension-first cascade search.
    ///
    /// The key optimization: compute exact distance on the highest-weighted
    /// dimension first (16 AVX-512 iterations). Use it as a lower bound
    /// to eliminate candidates before computing the remaining two dimensions
    /// (32 more iterations). For asymmetric weights (wx=0.7, wy=0.2, wz=0.1),
    /// this eliminates ~80% of candidates at 1/3 the SIMD cost.
    pub fn search(
        &self,
        candidates: &[&[u64]],
        query: &[u64],
        k: usize,
    ) -> Vec<DimSearchResult> {
        let mut results: Vec<DimSearchResult> = Vec::with_capacity(k + 1);
        let mut threshold = self.max_distance.unwrap_or(f64::MAX);
        let dominant = self.weights.dominant();
        let (dom_start, _) = dominant.range();
        let dom_weight = self.weights.weight_for(dominant);

        for (idx, &candidate) in candidates.iter().enumerate() {
            // Level 0: Schema predicate filter
            if !self.passes_predicates(candidate) {
                continue;
            }

            // Level 1: Dominant dimension distance (1/3 SIMD cost)
            let dom_dist = dim_distance(query, candidate, dom_start);
            let dom_contribution = dom_weight * dom_dist as f64;

            // Lower bound: the dominant dimension alone already exceeds threshold
            if dom_contribution > threshold {
                continue;
            }

            // Level 2: Full per-dimension distance
            let (dx, dy, dz) = tri_distance(query, candidate);
            let weighted = weighted_from_tri(dx, dy, dz, &self.weights);

            if weighted > threshold {
                continue;
            }

            let result = DimSearchResult {
                index: idx,
                distance_x: dx,
                distance_y: dy,
                distance_z: dz,
                weighted_distance: weighted,
            };

            let pos = results.partition_point(|r| r.weighted_distance <= weighted);
            results.insert(pos, result);

            if results.len() > k {
                results.truncate(k);
                threshold = results.last().map(|r| r.weighted_distance).unwrap_or(f64::MAX);
            }
        }

        results
    }
}

impl Default for DimSearchQuery {
    fn default() -> Self {
        Self::new(DimWeights::BALANCED)
    }
}

/// Result from dimensional cascade search.
#[derive(Clone, Debug)]
pub struct DimSearchResult {
    pub index: usize,
    pub distance_x: u32,
    pub distance_y: u32,
    pub distance_z: u32,
    pub weighted_distance: f64,
}

// ============================================================================
// HOLOGRAPHIC PROBE SEARCH
// ============================================================================

/// Which dimension to recover via holographic probing.
#[derive(Clone, Copy, Debug)]
pub enum ProbeTarget {
    /// Given Y + Z, recover X (content)
    RecoverX,
    /// Given X + Z, recover Y (context)
    RecoverY,
    /// Given X + Y, recover Z (relation)
    RecoverZ,
}

/// Holographic probe search: given two known dimensions, XOR-probe each
/// candidate to recover the third dimension, then rank by closeness to
/// a target vector.
///
/// This answers relational queries directly:
/// - RecoverZ: "What relation connects content X to context Y?"
/// - RecoverY: "In what context does content X appear with relation Z?"
/// - RecoverX: "What content has relation Z in context Y?"
pub fn probe_search(
    candidates: &[&[u64]],
    dim_a: &[u64],      // First known dimension (128 words)
    dim_b: &[u64],      // Second known dimension (128 words)
    target: &[u64],     // Target for the recovered dimension (128 words)
    probe: ProbeTarget,
    k: usize,
) -> Vec<ProbeSearchResult> {
    let mut results: Vec<ProbeSearchResult> = Vec::with_capacity(k + 1);

    for (idx, &candidate) in candidates.iter().enumerate() {
        // XOR-probe: bind the two known dimensions with the candidate's trace
        // to recover the unknown dimension.
        let (trace_start, _a_start, _b_start) = match probe {
            ProbeTarget::RecoverX => (X_START, Y_START, Z_START),
            ProbeTarget::RecoverY => (Y_START, X_START, Z_START),
            ProbeTarget::RecoverZ => (Z_START, X_START, Y_START),
        };

        // The candidate's semantic content across all three dims is the "trace"
        // We probe: recovered = candidate_trace_dim ⊕ dim_a ⊕ dim_b
        // Where candidate_trace_dim is the candidate's dimension we want to recover FROM
        // Actually — the holographic trace is: trace = X ⊕ Y ⊕ Z
        // So to recover Z: Z_recovered = trace ⊕ X ⊕ Y = (X⊕Y⊕Z) ⊕ X ⊕ Y = Z
        // The "trace" here is the full XOR binding of the candidate.
        // We reconstruct it from the candidate's three dimensions.

        // Build the candidate's full trace: candidate_X ⊕ candidate_Y ⊕ candidate_Z
        let mut recovered = vec![0u64; DIM_WORDS];
        for i in 0..DIM_WORDS {
            let trace_word = candidate[X_START + i]
                ^ candidate[Y_START + i]
                ^ candidate[Z_START + i];
            // Probe: recovered = trace ⊕ dim_a ⊕ dim_b
            recovered[i] = trace_word ^ dim_a[i.min(dim_a.len() - 1)] ^ dim_b[i.min(dim_b.len() - 1)];
        }

        // Distance between recovered dimension and target
        let dist = dim_hamming(&recovered, target);

        // SNR estimate: how far is the recovered vector from random?
        let popcount: u32 = recovered.iter().map(|w| w.count_ones()).sum();
        let deviation = (popcount as f64 - DIM_EXPECTED_DISTANCE).abs();
        let snr = deviation / DIM_SIGMA;

        let result = ProbeSearchResult {
            index: idx,
            distance: dist,
            snr_estimate: snr,
        };

        let pos = results.partition_point(|r| r.distance <= dist);
        results.insert(pos, result);

        if results.len() > k {
            results.truncate(k);
        }
    }

    results
}

/// Result from holographic probe search.
#[derive(Clone, Debug)]
pub struct ProbeSearchResult {
    pub index: usize,
    /// Hamming distance between recovered dimension and target
    pub distance: u32,
    /// Signal-to-noise ratio of the recovery (higher = cleaner)
    pub snr_estimate: f64,
}

// ============================================================================
// BLOOM-ACCELERATED DIMENSIONAL SEARCH
// ============================================================================

/// Bloom check against 512-bit bloom filter in metadata block.
#[inline]
pub fn bloom_might_be_neighbor(words: &[u64], neighbor_id: u64) -> bool {
    if words.len() < VECTOR_WORDS {
        return false;
    }
    // 512-bit bloom at META_START + M_BLOOM_BASE (8 words)
    let bloom_start = META_START + M_BLOOM_BASE;
    // Hash the neighbor_id to get bloom positions
    let h1 = neighbor_id.wrapping_mul(0x517cc1b727220a95);
    let h2 = neighbor_id.wrapping_mul(0x6c62272e07bb0142);
    let h3 = neighbor_id.wrapping_mul(0x305f39e36ab7be35);

    let check = |h: u64| -> bool {
        let bit_pos = (h % 512) as usize;
        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;
        (words[bloom_start + word_idx] >> bit_idx) & 1 == 1
    };

    check(h1) && check(h2) && check(h3)
}

/// Bloom-accelerated dimensional search.
///
/// Combines per-dimension weighted distance with bloom neighbor bonus.
/// The 512-bit bloom (8 words) supports 3× more neighbors at the same
/// false positive rate as the 256-bit bloom in 16K vectors.
pub fn bloom_dimensional_search(
    candidates: &[&[u64]],
    query: &[u64],
    source_id: u64,
    k: usize,
    neighbor_bonus: f64,
    dim_query: &DimSearchQuery,
) -> Vec<BloomDimResult> {
    let mut results: Vec<BloomDimResult> = Vec::with_capacity(k + 1);

    for (idx, &candidate) in candidates.iter().enumerate() {
        if !dim_query.passes_predicates(candidate) {
            continue;
        }

        let (dx, dy, dz) = tri_distance(query, candidate);
        let weighted = weighted_from_tri(dx, dy, dz, &dim_query.weights);

        if let Some(max) = dim_query.max_distance {
            if weighted > max { continue; }
        }

        let is_neighbor = bloom_might_be_neighbor(candidate, source_id);
        let effective = if is_neighbor {
            weighted * (1.0 - neighbor_bonus)
        } else {
            weighted
        };

        let result = BloomDimResult {
            index: idx,
            distance_x: dx,
            distance_y: dy,
            distance_z: dz,
            raw_distance: weighted,
            effective_distance: effective,
            is_bloom_neighbor: is_neighbor,
        };

        let pos = results.partition_point(|r| r.effective_distance <= effective);
        results.insert(pos, result);

        if results.len() > k {
            results.truncate(k);
        }
    }

    results
}

/// Result from bloom-accelerated dimensional search.
#[derive(Clone, Debug)]
pub struct BloomDimResult {
    pub index: usize,
    pub distance_x: u32,
    pub distance_y: u32,
    pub distance_z: u32,
    pub raw_distance: f64,
    pub effective_distance: f64,
    pub is_bloom_neighbor: bool,
}

// ============================================================================
// DIMENSION-SPECIFIC SEARCH (single dimension, fast path)
// ============================================================================

/// Search on a single dimension only (128 words = 16 AVX-512 iterations).
///
/// Use this when you only care about one axis:
/// - content_search: "find similar content regardless of context"
/// - context_search: "find same context regardless of content"
/// - relation_search: "find same relation type"
pub fn single_dim_search(
    candidates: &[&[u64]],
    query: &[u64],
    dim: Dimension,
    k: usize,
) -> Vec<SingleDimResult> {
    let (start, _end) = dim.range();
    let mut results: Vec<SingleDimResult> = Vec::with_capacity(k + 1);
    let mut threshold = u32::MAX;

    for (idx, &candidate) in candidates.iter().enumerate() {
        if candidate.len() < VECTOR_WORDS || query.len() < VECTOR_WORDS {
            continue;
        }

        let dist = match dim_distance_with_threshold(query, candidate, start, threshold) {
            Some(d) => d,
            None => continue,
        };

        let result = SingleDimResult { index: idx, distance: dist, dimension: dim };

        let pos = results.partition_point(|r| r.distance <= dist);
        results.insert(pos, result);

        if results.len() > k {
            results.truncate(k);
            threshold = results.last().map(|r| r.distance).unwrap_or(u32::MAX);
        }
    }

    results
}

/// Result from single-dimension search.
#[derive(Clone, Debug)]
pub struct SingleDimResult {
    pub index: usize,
    pub distance: u32,
    pub dimension: Dimension,
}

// ============================================================================
// ANALOGICAL SEARCH
// ============================================================================

/// Analogical search: "A is to B as C is to ?"
///
/// Computes the analogical transfer vector `C ⊕ (A ⊕ B)` and finds
/// the k nearest neighbors to this analogy vector. This works natively
/// in the 3D space — the dimensional decomposition means the analogy
/// operates per-axis (content shift, context shift, relation shift).
pub fn analogy_search(
    candidates: &[&[u64]],
    a: &HoloVector,     // Source
    b: &HoloVector,     // Source target
    c: &HoloVector,     // Analogy query
    weights: DimWeights,
    k: usize,
) -> Vec<DimSearchResult> {
    // Analogy vector: c ⊕ (a ⊕ b) = c ⊕ delta(a→b)
    let delta = a.bind(b);    // a ⊕ b = transform
    let analogy = c.bind(&delta); // c ⊕ transform = expected answer

    let query = DimSearchQuery::new(weights);
    query.search(candidates, &analogy.words, k)
}

// ============================================================================
// HELPER
// ============================================================================

/// Hamming distance between two dimension slices.
fn dim_hamming(a: &[u64], b: &[u64]) -> u32 {
    let len = a.len().min(b.len()).min(DIM_WORDS);
    let mut total = 0u32;
    for i in 0..len {
        total += (a[i] ^ b[i]).count_ones();
    }
    total
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::holographic::HoloVector;

    fn random_holo(seed: u64) -> HoloVector {
        let mut v = HoloVector::zero();
        let mut state = seed;
        for w in v.words.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = state;
        }
        v
    }

    fn zero_holo_with_x(pattern: u64) -> HoloVector {
        let mut v = HoloVector::zero();
        for i in X_START..X_END {
            v.words[i] = pattern;
        }
        v
    }

    #[test]
    fn test_dim_weights_dominant() {
        assert_eq!(DimWeights::CONTENT.dominant(), Dimension::X);
        assert_eq!(DimWeights::CONTEXT.dominant(), Dimension::Y);
        assert_eq!(DimWeights::RELATION.dominant(), Dimension::Z);
    }

    #[test]
    fn test_dim_distance_self_is_zero() {
        let v = random_holo(42);
        let (dx, dy, dz) = tri_distance(&v.words, &v.words);
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
        assert_eq!(dz, 0);
    }

    #[test]
    fn test_weighted_distance_respects_weights() {
        let a = random_holo(1);
        let b = random_holo(2);
        let (dx, dy, dz) = tri_distance(&a.words, &b.words);

        // Content-weighted should emphasize X
        let content_dist = weighted_from_tri(dx, dy, dz, &DimWeights::CONTENT);
        // Context-weighted should emphasize Y
        let context_dist = weighted_from_tri(dx, dy, dz, &DimWeights::CONTEXT);

        // Both should be positive
        assert!(content_dist > 0.0);
        assert!(context_dist > 0.0);

        // They should differ (different weighting of same distances)
        // (Unless dx == dy == dz by coincidence, which is astronomically unlikely)
        if dx != dy {
            assert!((content_dist - context_dist).abs() > 0.01,
                "Different weights should produce different distances: content={}, context={}",
                content_dist, context_dist);
        }
    }

    #[test]
    fn test_dimensional_search_finds_nearest() {
        let query = HoloVector::zero();
        let close = {
            let mut v = HoloVector::zero();
            v.words[X_START] = 0xFF; // 8 bits different in X only
            v
        };
        let far = random_holo(99); // ~50% bits different everywhere

        let candidates: Vec<&[u64]> = vec![&far.words, &close.words];

        let search = DimSearchQuery::new(DimWeights::CONTENT);
        let results = search.search(&candidates, &query.words, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 1, "Close vector should rank first");
        assert_eq!(results[0].distance_x, 8);
    }

    #[test]
    fn test_dominant_dim_first_eliminates() {
        // Create a query and candidates where dominant-dim-first filtering helps
        let query = HoloVector::zero();

        // Candidate 0: close in Y (context), far in X (content)
        let mut c0 = HoloVector::zero();
        for i in X_START..X_END {
            c0.words[i] = 0xFFFF_FFFF_FFFF_FFFF; // all bits set in X
        }
        // Y stays zero (close)

        // Candidate 1: close in X (content), moderate in Y
        let mut c1 = HoloVector::zero();
        c1.words[X_START] = 0xFF; // 8 bits in X
        c1.words[Y_START] = 0xFFFF; // 16 bits in Y

        let candidates: Vec<&[u64]> = vec![&c0.words, &c1.words];

        // Content-weighted search should favor c1 (close X)
        let search = DimSearchQuery::new(DimWeights::CONTENT);
        let results = search.search(&candidates, &query.words, 2);

        assert_eq!(results[0].index, 1, "Content-weighted should prefer close-X candidate");
    }

    #[test]
    fn test_schema_predicates_pass() {
        let mut v = HoloVector::zero();
        // Set ANI level and activation in metadata
        let ani_word: u64 = 5 | (300u64 << 16); // level=5, activation=300
        v.words[META_START + M_ANI_BASE] = ani_word;

        // Set NARS truth
        let freq_u16 = (0.8f32 * 65535.0) as u64;
        let conf_u16 = (0.6f32 * 65535.0) as u64;
        v.words[META_START + M_NARS_TRUTH] = freq_u16 | (conf_u16 << 16);

        let query = DimSearchQuery::new(DimWeights::BALANCED)
            .with_ani(AniFilter { min_level: 3, min_activation: 200 })
            .with_nars(NarsFilter {
                min_frequency: Some(0.7),
                min_confidence: Some(0.5),
            });

        assert!(query.passes_predicates(&v.words));
    }

    #[test]
    fn test_schema_predicates_fail_ani() {
        let mut v = HoloVector::zero();
        let ani_word: u64 = 2 | (100u64 << 16); // level=2 (too low)
        v.words[META_START + M_ANI_BASE] = ani_word;

        let query = DimSearchQuery::new(DimWeights::BALANCED)
            .with_ani(AniFilter { min_level: 3, min_activation: 50 });

        assert!(!query.passes_predicates(&v.words));
    }

    #[test]
    fn test_probe_search_perfect_recovery() {
        // Create a vector with known X, Y, Z
        let mut v = HoloVector::zero();
        let mut state = 42u64;
        for i in X_START..Z_END {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.words[i] = state;
        }

        // The target Z is the actual Z dimension
        let target_z: Vec<u64> = v.z().to_vec();

        // Probe with known X and Y to recover Z
        let candidates: Vec<&[u64]> = vec![&v.words];
        let results = probe_search(
            &candidates,
            v.x(),
            v.y(),
            &target_z,
            ProbeTarget::RecoverZ,
            1,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].distance, 0, "Perfect recovery: no noise with single trace");
    }

    #[test]
    fn test_probe_search_ranks_by_closeness() {
        // Create two candidates with different Z dimensions
        let mut v1 = HoloVector::zero();
        let mut v2 = HoloVector::zero();

        // Both share X and Y
        let shared_x = 0xDEADBEEFu64;
        let shared_y = 0xCAFEBABEu64;
        for i in 0..DIM_WORDS {
            v1.words[X_START + i] = shared_x;
            v2.words[X_START + i] = shared_x;
            v1.words[Y_START + i] = shared_y;
            v2.words[Y_START + i] = shared_y;
        }

        // v1 has Z = all zeros
        // v2 has Z = some bits set
        for i in 0..DIM_WORDS {
            v2.words[Z_START + i] = 0xFFFF;
        }

        // Target Z: all zeros (matches v1 perfectly)
        let target_z = vec![0u64; DIM_WORDS];

        let candidates: Vec<&[u64]> = vec![&v1.words, &v2.words];
        let results = probe_search(
            &candidates,
            &[shared_x; DIM_WORDS],
            &[shared_y; DIM_WORDS],
            &target_z,
            ProbeTarget::RecoverZ,
            2,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0, "v1 (Z=0) should match target Z=0 better");
        assert_eq!(results[0].distance, 0);
        assert!(results[1].distance > 0);
    }

    #[test]
    fn test_single_dim_search() {
        let query = HoloVector::zero();
        let mut close = HoloVector::zero();
        close.words[Y_START] = 0xFF; // 8 bits different in Y
        let far = random_holo(77);

        let candidates: Vec<&[u64]> = vec![&far.words, &close.words];
        let results = single_dim_search(&candidates, &query.words, Dimension::Y, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 1, "Close-in-Y should rank first");
        assert_eq!(results[0].distance, 8);
    }

    #[test]
    fn test_bloom_neighbor_check() {
        let mut v = HoloVector::zero();
        let neighbor_id: u64 = 42;

        // Insert into 512-bit bloom
        let h1 = neighbor_id.wrapping_mul(0x517cc1b727220a95);
        let h2 = neighbor_id.wrapping_mul(0x6c62272e07bb0142);
        let h3 = neighbor_id.wrapping_mul(0x305f39e36ab7be35);

        let bloom_start = META_START + M_BLOOM_BASE;
        for h in [h1, h2, h3] {
            let bit_pos = (h % 512) as usize;
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            v.words[bloom_start + word_idx] |= 1u64 << bit_idx;
        }

        assert!(bloom_might_be_neighbor(&v.words, 42));
    }

    #[test]
    fn test_analogy_search() {
        // king - male + female = queen pattern
        let king = random_holo(1);
        let male = random_holo(2);
        let female = random_holo(3);

        // Construct "queen" = king ⊕ male ⊕ female
        let queen = king.bind(&male).bind(&female);

        // Add some other random vectors as distractors
        let d1 = random_holo(10);
        let d2 = random_holo(11);

        let candidates: Vec<&[u64]> = vec![&queen.words, &d1.words, &d2.words];

        let results = analogy_search(
            &candidates,
            &king,
            &male,
            &female,
            DimWeights::BALANCED,
            1,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0, "Queen should be the analogical answer");
        assert_eq!(results[0].weighted_distance, 0.0, "Perfect analogy = zero distance");
    }

    #[test]
    fn test_dim_distance_with_threshold() {
        let a = random_holo(1);
        let b = random_holo(2);

        // Very high threshold should pass
        let result = dim_distance_with_threshold(&a.words, &b.words, X_START, u32::MAX);
        assert!(result.is_some());

        // Very low threshold should fail (random vectors differ by ~4096 bits per dim)
        let result = dim_distance_with_threshold(&a.words, &b.words, X_START, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_search_returns_empty() {
        let query = HoloVector::zero();
        let candidates: Vec<&[u64]> = vec![];

        let search = DimSearchQuery::new(DimWeights::BALANCED);
        let results = search.search(&candidates, &query.words, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_limit() {
        let query = HoloVector::zero();
        let vectors: Vec<HoloVector> = (0..20).map(|i| random_holo(i)).collect();
        let candidates: Vec<&[u64]> = vectors.iter().map(|v| v.words.as_slice()).collect();

        let search = DimSearchQuery::new(DimWeights::BALANCED);
        let results = search.search(&candidates, &query.words, 5);
        assert_eq!(results.len(), 5, "Should return exactly k results");
    }
}
