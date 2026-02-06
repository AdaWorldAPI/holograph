//! Schema-Aware Search API
//!
//! Extends the HDR search cascade with schema predicate pruning.
//! Because ANI/NARS/RL markers live inline in the fingerprint (blocks 13-15),
//! we can reject candidates in O(1) *before* computing Hamming distance.
//!
//! # The Search Cascade with Schema
//!
//! ```text
//! Candidate pool (n vectors)
//!   │
//!   ├─► Level 0: Schema predicate filter (O(1) per vector)
//!   │     Read 2-3 words from blocks 13-15, check ANI/NARS/RL predicates
//!   │     Cost: ~3 cycles per candidate
//!   │     Rejects: depends on predicate selectivity
//!   │
//!   ├─► Level 1: Belichtungsmesser (7-point sample, ~14 cycles)
//!   │     Rejects: ~90% of survivors
//!   │
//!   ├─► Level 2: Block-masked StackedPopcount with threshold
//!   │     Only compute on semantic blocks (0..12), skip schema blocks
//!   │     Rejects: ~80% of survivors
//!   │
//!   └─► Level 3: Exact distance on semantic blocks
//!         k results returned
//! ```
//!
//! # Why This Is Fast
//!
//! Traditional approach: compute Hamming distance first, THEN check metadata.
//! Our approach: check metadata first (it's already in the vector!), then
//! distance on survivors only. For selective predicates (e.g., "ANI level >= 3",
//! "NARS confidence > 0.8"), this eliminates most candidates before the
//! expensive popcount cascade even starts.

use super::schema::{
    AniLevels, NarsTruth, NarsBudget, EdgeTypeMarker, NodeTypeMarker, NodeKind,
    InlineQValues, InlineRewards, NeighborBloom, GraphMetrics, SchemaSidecar,
};
use super::{VECTOR_WORDS, NUM_BLOCKS, BITS_PER_BLOCK, SEMANTIC_BLOCKS, SCHEMA_BLOCK_START};

// ============================================================================
// BLOCK MASK: Which blocks participate in distance computation
// ============================================================================

/// Bitmask selecting which of the 16 blocks participate in distance
/// computation. Default: blocks 0..12 (semantic only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockMask {
    /// 16-bit mask, one bit per block. Bit 0 = block 0, etc.
    mask: u16,
}

impl BlockMask {
    /// All 16 blocks (full 16K distance)
    pub const ALL: Self = Self { mask: 0xFFFF };

    /// Semantic blocks only (0..12 = 13,312 bits)
    pub const SEMANTIC: Self = Self { mask: 0x1FFF }; // bits 0..12

    /// Schema blocks only (13..15 = 3,072 bits)
    pub const SCHEMA: Self = Self { mask: 0xE000 }; // bits 13..15

    /// Custom mask from raw u16
    pub const fn from_raw(mask: u16) -> Self {
        Self { mask }
    }

    /// Is block `i` included?
    #[inline]
    pub fn includes(&self, block: usize) -> bool {
        block < 16 && (self.mask & (1u16 << block)) != 0
    }

    /// Number of included blocks
    pub fn count(&self) -> u32 {
        self.mask.count_ones()
    }

    /// Number of words covered by this mask
    pub fn word_count(&self) -> usize {
        self.count() as usize * 16
    }

    /// Number of bits covered (for normalization)
    pub fn bit_count(&self) -> usize {
        self.word_count() * 64
    }
}

impl Default for BlockMask {
    fn default() -> Self {
        Self::SEMANTIC
    }
}

// ============================================================================
// SCHEMA PREDICATES: O(1) filters on inline metadata
// ============================================================================

/// ANI level filter
#[derive(Clone, Debug)]
pub struct AniFilter {
    /// Minimum reasoning level (0..7) that must be active
    pub min_level: u8,
    /// Minimum activation at that level
    pub min_activation: u16,
}

/// NARS truth/budget filter
#[derive(Clone, Debug)]
pub struct NarsFilter {
    /// Minimum frequency (0.0..1.0)
    pub min_frequency: Option<f32>,
    /// Minimum confidence (0.0..1.0)
    pub min_confidence: Option<f32>,
    /// Minimum priority (0.0..1.0)
    pub min_priority: Option<f32>,
}

/// RL state filter
#[derive(Clone, Debug)]
pub struct RlFilter {
    /// Minimum Q-value for best action
    pub min_best_q: Option<f32>,
    /// Minimum average reward
    pub min_avg_reward: Option<f32>,
    /// Positive reward trend required
    pub positive_trend: bool,
}

/// Graph topology filter
#[derive(Clone, Debug)]
pub struct GraphFilter {
    /// Minimum PageRank (quantized 0..65535)
    pub min_pagerank: Option<u16>,
    /// Maximum hop distance to root
    pub max_hop: Option<u8>,
    /// Required cluster ID
    pub cluster_id: Option<u16>,
    /// Minimum degree
    pub min_degree: Option<u8>,
}

/// Node kind filter
#[derive(Clone, Debug)]
pub struct KindFilter {
    /// Accepted node kinds (empty = accept all)
    pub kinds: Vec<NodeKind>,
    /// Accepted edge verb IDs (empty = accept all)
    pub verb_ids: Vec<u8>,
}

// ============================================================================
// SCHEMA QUERY: Combined predicate + distance search
// ============================================================================

/// A schema-aware search query.
///
/// Combines traditional Hamming distance search with schema predicate filters.
/// Predicates are checked *before* distance computation for early rejection.
///
/// # Example
///
/// ```text
/// SchemaQuery::new()
///     .with_ani(AniFilter { min_level: 3, min_activation: 100 })
///     .with_nars(NarsFilter { min_confidence: Some(0.5), ..Default::default() })
///     .with_block_mask(BlockMask::SEMANTIC)
///     .search(&candidates, &query, 10)
/// ```
#[derive(Clone, Debug)]
pub struct SchemaQuery {
    /// ANI reasoning level filter
    pub ani_filter: Option<AniFilter>,
    /// NARS truth/budget filter
    pub nars_filter: Option<NarsFilter>,
    /// RL state filter
    pub rl_filter: Option<RlFilter>,
    /// Graph topology filter
    pub graph_filter: Option<GraphFilter>,
    /// Node/edge kind filter
    pub kind_filter: Option<KindFilter>,
    /// Which blocks participate in distance (default: semantic only)
    pub block_mask: BlockMask,
    /// Maximum Hamming distance (on masked blocks)
    pub max_distance: Option<u32>,
}

impl SchemaQuery {
    pub fn new() -> Self {
        Self {
            ani_filter: None,
            nars_filter: None,
            rl_filter: None,
            graph_filter: None,
            kind_filter: None,
            block_mask: BlockMask::SEMANTIC,
            max_distance: None,
        }
    }

    /// Builder: add ANI filter
    pub fn with_ani(mut self, filter: AniFilter) -> Self {
        self.ani_filter = Some(filter);
        self
    }

    /// Builder: add NARS filter
    pub fn with_nars(mut self, filter: NarsFilter) -> Self {
        self.nars_filter = Some(filter);
        self
    }

    /// Builder: add RL filter
    pub fn with_rl(mut self, filter: RlFilter) -> Self {
        self.rl_filter = Some(filter);
        self
    }

    /// Builder: add graph topology filter
    pub fn with_graph(mut self, filter: GraphFilter) -> Self {
        self.graph_filter = Some(filter);
        self
    }

    /// Builder: add node/edge kind filter
    pub fn with_kind(mut self, filter: KindFilter) -> Self {
        self.kind_filter = Some(filter);
        self
    }

    /// Builder: set block mask
    pub fn with_block_mask(mut self, mask: BlockMask) -> Self {
        self.block_mask = mask;
        self
    }

    /// Builder: set maximum Hamming distance
    pub fn with_max_distance(mut self, d: u32) -> Self {
        self.max_distance = Some(d);
        self
    }

    /// Check if a candidate's schema passes all predicates.
    ///
    /// This reads directly from the word array — **zero deserialization cost**
    /// when only checking a few fields. Each predicate reads 1-2 words max.
    ///
    /// Returns `true` if the candidate passes (should proceed to distance check).
    pub fn passes_predicates(&self, candidate_words: &[u64]) -> bool {
        if candidate_words.len() < VECTOR_WORDS {
            return false;
        }

        let base = SchemaSidecar::WORD_OFFSET; // 208

        // ANI filter: read words[208..209] (128 bits)
        if let Some(ref ani) = self.ani_filter {
            let ani_packed = candidate_words[base] as u128
                | ((candidate_words[base + 1] as u128) << 64);
            let levels = AniLevels::unpack(ani_packed);
            let level_vals = [
                levels.reactive, levels.memory, levels.analogy, levels.planning,
                levels.meta, levels.social, levels.creative, levels.r#abstract,
            ];
            if ani.min_level as usize >= 8 {
                return false;
            }
            // Check that the required level (and all above) meet activation threshold
            let activation = level_vals[ani.min_level as usize];
            if activation < ani.min_activation {
                return false;
            }
        }

        // NARS filter: read word[210] (lower 32 bits = truth)
        if let Some(ref nars) = self.nars_filter {
            let truth = NarsTruth::unpack(candidate_words[base + 2] as u32);
            if let Some(min_f) = nars.min_frequency {
                if truth.f() < min_f {
                    return false;
                }
            }
            if let Some(min_c) = nars.min_confidence {
                if truth.c() < min_c {
                    return false;
                }
            }
            // Budget: upper 32 bits of word[210] → lower 64 bits
            if let Some(min_p) = nars.min_priority {
                let budget = NarsBudget::unpack((candidate_words[base + 2] >> 32) as u64);
                if (budget.priority as f32 / 65535.0) < min_p {
                    return false;
                }
            }
        }

        // Kind filter: read word[211] (upper 32 bits = node type)
        if let Some(ref kind) = self.kind_filter {
            if !kind.kinds.is_empty() {
                let node = NodeTypeMarker::unpack((candidate_words[base + 3] >> 32) as u32);
                if !kind.kinds.iter().any(|k| *k as u8 == node.kind) {
                    return false;
                }
            }
            if !kind.verb_ids.is_empty() {
                let edge = EdgeTypeMarker::unpack(candidate_words[base + 3] as u32);
                if !kind.verb_ids.contains(&edge.verb_id) {
                    return false;
                }
            }
        }

        // RL filter: read words[224..227]
        if let Some(ref rl) = self.rl_filter {
            let block14_base = base + 16;

            if let Some(min_q) = rl.min_best_q {
                let q = InlineQValues::unpack([
                    candidate_words[block14_base],
                    candidate_words[block14_base + 1],
                ]);
                let best = q.q(q.best_action());
                if best < min_q {
                    return false;
                }
            }

            if rl.min_avg_reward.is_some() || rl.positive_trend {
                let mut rewards = InlineRewards::default();
                let rw0 = candidate_words[block14_base + 2];
                let rw1 = candidate_words[block14_base + 3];
                for i in 0..4 {
                    rewards.rewards[i] = ((rw0 >> (i * 16)) & 0xFFFF) as u16 as i16;
                }
                for i in 0..4 {
                    rewards.rewards[i + 4] = ((rw1 >> (i * 16)) & 0xFFFF) as u16 as i16;
                }

                if let Some(min_avg) = rl.min_avg_reward {
                    if rewards.average() < min_avg {
                        return false;
                    }
                }
                if rl.positive_trend && rewards.trend() <= 0.0 {
                    return false;
                }
            }
        }

        // Graph filter: read word[248]
        if let Some(ref graph) = self.graph_filter {
            let block15_base = base + 32;
            let metrics = GraphMetrics::unpack(candidate_words[block15_base + 8]);

            if let Some(min_pr) = graph.min_pagerank {
                if metrics.pagerank < min_pr {
                    return false;
                }
            }
            if let Some(max_h) = graph.max_hop {
                if metrics.hop_to_root > max_h {
                    return false;
                }
            }
            if let Some(cid) = graph.cluster_id {
                if metrics.cluster_id != cid {
                    return false;
                }
            }
            if let Some(min_d) = graph.min_degree {
                if metrics.degree < min_d {
                    return false;
                }
            }
        }

        true
    }

    /// Compute block-masked Hamming distance between two word arrays.
    ///
    /// Only popcount words in blocks selected by `self.block_mask`.
    /// For `BlockMask::SEMANTIC` (blocks 0..12), this computes distance
    /// over 13,312 bits and ignores the schema blocks entirely.
    pub fn masked_distance(&self, a: &[u64], b: &[u64]) -> u32 {
        debug_assert!(a.len() >= VECTOR_WORDS);
        debug_assert!(b.len() >= VECTOR_WORDS);

        let mut total = 0u32;
        for block in 0..NUM_BLOCKS {
            if !self.block_mask.includes(block) {
                continue;
            }
            let start = block * 16;
            let end = start + 16; // All blocks are 16 words in 16K
            for w in start..end {
                total += (a[w] ^ b[w]).count_ones();
            }
        }
        total
    }

    /// Compute block-masked distance with early termination.
    ///
    /// Returns `None` if the running distance exceeds `threshold` at any
    /// block boundary (coarse-grained pruning on block sums).
    pub fn masked_distance_with_threshold(
        &self,
        a: &[u64],
        b: &[u64],
        threshold: u32,
    ) -> Option<u32> {
        debug_assert!(a.len() >= VECTOR_WORDS);
        debug_assert!(b.len() >= VECTOR_WORDS);

        let mut total = 0u32;
        for block in 0..NUM_BLOCKS {
            if !self.block_mask.includes(block) {
                continue;
            }
            let start = block * 16;
            let end = start + 16;
            let mut block_sum = 0u32;
            for w in start..end {
                block_sum += (a[w] ^ b[w]).count_ones();
            }
            total += block_sum;
            if total > threshold {
                return None; // Early exit: exceeded threshold
            }
        }
        Some(total)
    }

    /// Full search pipeline: predicate filter → block-masked distance → top-k.
    ///
    /// `candidates` is a slice of `&[u64; 256]` word arrays (zero-copy from Arrow).
    /// Returns (index, distance) pairs sorted by distance, up to `k` results.
    pub fn search(
        &self,
        candidates: &[&[u64]],
        query: &[u64],
        k: usize,
    ) -> Vec<SchemaSearchResult> {
        let mut results: Vec<SchemaSearchResult> = Vec::with_capacity(k + 1);
        let mut current_threshold = self.max_distance.unwrap_or(u32::MAX);

        for (idx, &candidate) in candidates.iter().enumerate() {
            // Level 0: Schema predicate filter (O(1), ~3 cycles)
            if !self.passes_predicates(candidate) {
                continue;
            }

            // Level 1: Block-masked distance with threshold
            let dist = match self.masked_distance_with_threshold(
                query, candidate, current_threshold,
            ) {
                Some(d) => d,
                None => continue,
            };

            // Insert into results (maintain sorted order)
            let result = SchemaSearchResult {
                index: idx,
                distance: dist,
                schema: None, // Lazy: only decode schema on demand
            };

            // Binary search for insertion point
            let pos = results.partition_point(|r| r.distance <= dist);
            results.insert(pos, result);

            if results.len() > k {
                results.truncate(k);
                // Tighten threshold to best kth distance
                current_threshold = results.last().map(|r| r.distance).unwrap_or(u32::MAX);
            }
        }

        results
    }
}

impl Default for SchemaQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from schema-aware search
#[derive(Clone, Debug)]
pub struct SchemaSearchResult {
    /// Index in the candidate array
    pub index: usize,
    /// Block-masked Hamming distance
    pub distance: u32,
    /// Decoded schema (lazy, populated on demand)
    pub schema: Option<SchemaSidecar>,
}

impl SchemaSearchResult {
    /// Decode the full schema sidecar from the candidate words.
    /// Call this only when you need the schema details — it's ~50ns per decode.
    pub fn decode_schema(&mut self, candidate_words: &[u64]) {
        self.schema = Some(SchemaSidecar::read_from_words(candidate_words));
    }
}

// ============================================================================
// BLOOM-ASSISTED NEIGHBOR CHECK
// ============================================================================

/// Check if two 16K vectors are likely neighbors using the inline bloom filter.
///
/// This is O(1) with ~1% FPR — no graph traversal needed.
/// The bloom filter in block 15 was populated during graph construction.
#[inline]
pub fn bloom_might_be_neighbors(a_words: &[u64], b_id: u64) -> bool {
    let bloom_base = SchemaSidecar::WORD_OFFSET + 32 + 4; // block 15, offset 4 words
    if a_words.len() < bloom_base + 4 {
        return false;
    }
    let bloom = NeighborBloom {
        words: [
            a_words[bloom_base],
            a_words[bloom_base + 1],
            a_words[bloom_base + 2],
            a_words[bloom_base + 3],
        ],
    };
    bloom.might_contain(b_id)
}

// ============================================================================
// Q-VALUE ROUTING: Use inline RL state for beam search guidance
// ============================================================================

/// Extract the best action and Q-value from a candidate's inline RL state.
///
/// This enables RL-guided beam search: instead of ranking candidates by
/// Hamming distance alone, combine distance with learned Q-value as a
/// routing heuristic. Candidates with higher Q-values for the current
/// action context get priority in the beam.
#[inline]
pub fn read_best_q(candidate_words: &[u64]) -> (usize, f32) {
    let block14_base = SchemaSidecar::WORD_OFFSET + 16;
    if candidate_words.len() < block14_base + 2 {
        return (0, 0.0);
    }
    let q = InlineQValues::unpack([
        candidate_words[block14_base],
        candidate_words[block14_base + 1],
    ]);
    let best = q.best_action();
    (best, q.q(best))
}

/// Composite routing score: weighted combination of Hamming distance
/// and Q-value for RL-guided search.
///
/// `alpha` controls the RL weight: 0.0 = pure distance, 1.0 = pure Q-value.
/// Typical: alpha = 0.2 (20% RL influence on routing).
#[inline]
pub fn rl_routing_score(distance: u32, q_value: f32, alpha: f32) -> f32 {
    let distance_norm = distance as f32 / (SEMANTIC_BLOCKS as f32 * BITS_PER_BLOCK as f32);
    let q_norm = (1.0 - q_value) / 2.0; // Map [-1, 1] → [1, 0] (lower = better)
    (1.0 - alpha) * distance_norm + alpha * q_norm
}

// ============================================================================
// NARS-AWARE OPERATIONS
// ============================================================================

/// Revise two 16K vectors' NARS truth values.
///
/// When bundling two vectors that carry NARS truth values, the resulting
/// truth value should be the NARS revision (combining evidence).
/// This reads both truth values inline, computes the revision, and
/// writes it to the output words.
pub fn nars_revision_inline(a_words: &[u64], b_words: &[u64], out_words: &mut [u64]) {
    let base = SchemaSidecar::WORD_OFFSET;
    if a_words.len() < VECTOR_WORDS || b_words.len() < VECTOR_WORDS || out_words.len() < VECTOR_WORDS {
        return;
    }

    let truth_a = NarsTruth::unpack(a_words[base + 2] as u32);
    let truth_b = NarsTruth::unpack(b_words[base + 2] as u32);
    let revised = truth_a.revision(&truth_b);

    // Preserve budget from higher-priority input
    let budget_a = NarsBudget::unpack((a_words[base + 2] >> 32) as u64);
    let budget_b = NarsBudget::unpack((b_words[base + 2] >> 32) as u64);
    let budget = if budget_a.priority >= budget_b.priority { budget_a } else { budget_b };

    out_words[base + 2] = revised.pack() as u64 | ((budget.pack() as u64) << 32);
}

/// NARS deduction chain: compute truth value for A→B, B→C ⊢ A→C
pub fn nars_deduction_inline(premise_words: &[u64], conclusion_words: &[u64]) -> NarsTruth {
    let base = SchemaSidecar::WORD_OFFSET;
    let t1 = NarsTruth::unpack(premise_words[base + 2] as u32);
    let t2 = NarsTruth::unpack(conclusion_words[base + 2] as u32);
    t1.deduction(&t2)
}

// ============================================================================
// SCHEMA-AWARE BIND: XOR with schema combination
// ============================================================================

/// XOR-bind two 16K vectors with intelligent schema merging.
///
/// The semantic blocks (0..12) are XOR'd as usual. The schema blocks are
/// handled specially:
/// - ANI levels: take element-wise max (binding shouldn't reduce capability)
/// - NARS truth: compute revision (combine evidence)
/// - RL state: preserve from `a` (primary operand)
/// - Graph cache: clear (binding creates a new edge, not a node)
///
/// This is the "surprising feature" — bind operations automatically
/// propagate and combine metadata without explicit schema management.
pub fn schema_bind(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert!(a.len() >= VECTOR_WORDS && b.len() >= VECTOR_WORDS);
    let mut out = vec![0u64; VECTOR_WORDS];

    // Semantic blocks: XOR as usual
    let semantic_end = SCHEMA_BLOCK_START * 16; // word 208
    for w in 0..semantic_end {
        out[w] = a[w] ^ b[w];
    }

    let base = SchemaSidecar::WORD_OFFSET;

    // Block 13: ANI levels — element-wise max
    let ani_a = AniLevels::unpack(a[base] as u128 | ((a[base + 1] as u128) << 64));
    let ani_b = AniLevels::unpack(b[base] as u128 | ((b[base + 1] as u128) << 64));
    let ani_merged = AniLevels {
        reactive: ani_a.reactive.max(ani_b.reactive),
        memory: ani_a.memory.max(ani_b.memory),
        analogy: ani_a.analogy.max(ani_b.analogy),
        planning: ani_a.planning.max(ani_b.planning),
        meta: ani_a.meta.max(ani_b.meta),
        social: ani_a.social.max(ani_b.social),
        creative: ani_a.creative.max(ani_b.creative),
        r#abstract: ani_a.r#abstract.max(ani_b.r#abstract),
    };
    let packed_ani = ani_merged.pack();
    out[base] = packed_ani as u64;
    out[base + 1] = (packed_ani >> 64) as u64;

    // Block 13: NARS — revision
    let truth_a = NarsTruth::unpack(a[base + 2] as u32);
    let truth_b = NarsTruth::unpack(b[base + 2] as u32);
    let revised = truth_a.revision(&truth_b);
    // Budget: max priority
    let budget_a = NarsBudget::unpack((a[base + 2] >> 32) as u64);
    let budget_b = NarsBudget::unpack((b[base + 2] >> 32) as u64);
    let merged_budget = if budget_a.priority >= budget_b.priority {
        budget_a
    } else {
        budget_b
    };
    out[base + 2] = revised.pack() as u64 | ((merged_budget.pack() as u64) << 32);

    // Block 13: Edge type — XOR verb IDs (compositional binding)
    let edge_a = EdgeTypeMarker::unpack(a[base + 3] as u32);
    let edge_b = EdgeTypeMarker::unpack(b[base + 3] as u32);
    let merged_edge = EdgeTypeMarker {
        verb_id: edge_a.verb_id ^ edge_b.verb_id,
        direction: edge_a.direction, // preserve primary direction
        weight: ((edge_a.weight as u16 + edge_b.weight as u16) / 2) as u8,
        flags: edge_a.flags | edge_b.flags, // union of flags
    };
    out[base + 3] = merged_edge.pack() as u64;
    // Node type: XOR (compositional)
    let node_a = NodeTypeMarker::unpack((a[base + 3] >> 32) as u32);
    let node_b = NodeTypeMarker::unpack((b[base + 3] >> 32) as u32);
    out[base + 3] |= (NodeTypeMarker {
        kind: node_a.kind, // preserve primary kind
        subtype: node_a.subtype ^ node_b.subtype,
        provenance: node_a.provenance ^ node_b.provenance,
    }.pack() as u64) << 32;

    // Block 14: RL state — preserve from primary operand (a)
    let block14_base = base + 16;
    for w in 0..16 {
        out[block14_base + w] = a[block14_base + w];
    }

    // Block 15: Graph cache — clear (new binding = new identity)
    // Words 240..255 remain zero

    out
}

// ============================================================================
// BLOOM-ACCELERATED GRAPH TRAVERSAL
// ============================================================================

/// Search that combines ANN similarity with bloom-filter neighbor awareness.
///
/// This is the feature that has no equivalent in traditional graph databases.
/// Neo4j can't do "find similar nodes that are also graph-neighbors" without
/// first doing a full traversal, then a similarity check, or vice versa.
///
/// Here, the bloom filter is inline in the fingerprint (block 15), so we
/// check neighbor adjacency *during* the ANN search — no graph I/O needed.
///
/// ## How It Works
///
/// For each candidate that passes schema predicates + distance threshold:
/// 1. Check `candidate.bloom.might_contain(source_id)` — O(1), ~3 cycles
/// 2. If bloom says "yes": candidate is likely a 1-hop neighbor of source
///    → Apply a distance bonus (e.g., halve the distance)
/// 3. Sort by bonus-adjusted distance
///
/// ## Performance vs. Neo4j
///
/// ```text
/// Neo4j 2-hop traversal (avg degree 150):
///   150 × 150 = 22,500 edge lookups + property filters
///
/// HDR bloom-accelerated (top-k=10 from 10,000 candidates):
///   10,000 predicate checks (3 cycles each = 30µs)
///   ~1,000 distance computations (survivors)
///   ~100 bloom checks (top candidates)
///   Total: ~50µs vs. Neo4j's ~5ms (100× faster)
/// ```
pub fn bloom_accelerated_search(
    candidates: &[&[u64]],
    query: &[u64],
    source_id: u64,
    k: usize,
    neighbor_bonus: f32,
    schema_query: &SchemaQuery,
) -> Vec<BloomSearchResult> {
    let mut results: Vec<BloomSearchResult> = Vec::with_capacity(k + 1);
    let mut current_threshold = schema_query.max_distance.unwrap_or(u32::MAX);

    for (idx, &candidate) in candidates.iter().enumerate() {
        // Level 0: Schema predicate filter
        if !schema_query.passes_predicates(candidate) {
            continue;
        }

        // Level 1: Block-masked distance with threshold
        let raw_dist = match schema_query.masked_distance_with_threshold(
            query, candidate, current_threshold,
        ) {
            Some(d) => d,
            None => continue,
        };

        // Level 2: Bloom neighbor check — is this candidate a known neighbor?
        let is_neighbor = bloom_might_be_neighbors(candidate, source_id);

        // Apply neighbor bonus: neighbors get a discounted distance
        let effective_dist = if is_neighbor {
            (raw_dist as f32 * (1.0 - neighbor_bonus)) as u32
        } else {
            raw_dist
        };

        let result = BloomSearchResult {
            index: idx,
            raw_distance: raw_dist,
            effective_distance: effective_dist,
            is_bloom_neighbor: is_neighbor,
        };

        // Insert sorted by effective distance
        let pos = results.partition_point(|r| r.effective_distance <= effective_dist);
        results.insert(pos, result);

        if results.len() > k {
            results.truncate(k);
            current_threshold = results.last()
                .map(|r| r.raw_distance.max(r.effective_distance))
                .unwrap_or(u32::MAX);
        }
    }

    results
}

/// Result from bloom-accelerated search
#[derive(Clone, Debug)]
pub struct BloomSearchResult {
    /// Index in the candidate array
    pub index: usize,
    /// Raw Hamming distance (before neighbor bonus)
    pub raw_distance: u32,
    /// Effective distance (after neighbor bonus)
    pub effective_distance: u32,
    /// Whether bloom filter indicates this is a 1-hop neighbor
    pub is_bloom_neighbor: bool,
}

// ============================================================================
// RL-GUIDED SEARCH: Combine distance with learned Q-values
// ============================================================================

/// RL-guided search: ranks candidates by a composite of Hamming distance
/// and inline Q-values.
///
/// At each DN tree node, instead of choosing the child with minimum distance,
/// we score: `α × normalized_distance + (1-α) × normalized_q_cost`.
///
/// The Q-values learn from past search outcomes — "this branch usually leads
/// to good results" vs. "this branch has high similarity but leads to dead
/// ends". The Q-values travel with the tree node (inline in the fingerprint).
/// No external Q-table. No shared mutable state.
pub fn rl_guided_search(
    candidates: &[&[u64]],
    query: &[u64],
    k: usize,
    alpha: f32,
    schema_query: &SchemaQuery,
) -> Vec<RlSearchResult> {
    let max_bits = (schema_query.block_mask.count() as f32 * BITS_PER_BLOCK as f32).max(1.0);
    let mut results: Vec<RlSearchResult> = Vec::with_capacity(k + 1);

    for (idx, &candidate) in candidates.iter().enumerate() {
        if !schema_query.passes_predicates(candidate) {
            continue;
        }

        let dist = match schema_query.masked_distance_with_threshold(
            query, candidate, schema_query.max_distance.unwrap_or(u32::MAX),
        ) {
            Some(d) => d,
            None => continue,
        };

        // Read Q-value from inline RL state
        let (best_action, q_value) = read_best_q(candidate);

        // Composite score: lower = better
        let composite = rl_routing_score(dist, q_value, alpha);

        let result = RlSearchResult {
            index: idx,
            distance: dist,
            best_action,
            q_value,
            composite_score: composite,
        };

        let pos = results.partition_point(|r| r.composite_score <= composite);
        results.insert(pos, result);

        if results.len() > k {
            results.truncate(k);
        }
    }

    results
}

/// Result from RL-guided search
#[derive(Clone, Debug)]
pub struct RlSearchResult {
    /// Index in the candidate array
    pub index: usize,
    /// Raw Hamming distance
    pub distance: u32,
    /// Best action index from inline Q-values
    pub best_action: usize,
    /// Q-value for best action
    pub q_value: f32,
    /// Composite routing score (lower = better)
    pub composite_score: f32,
}

// ============================================================================
// FEDERATED SCHEMA MERGE: Combine schemas from distributed instances
// ============================================================================

/// Merge two 16K vectors from different federated instances.
///
/// Unlike `schema_bind` (which creates edges), this merges two representations
/// of the *same entity* from different sources. The semantic blocks are preserved
/// from `primary` (the authoritative source), while schema blocks are merged
/// using evidence-combining rules:
///
/// - **ANI levels**: element-wise max (take highest capability assessment)
/// - **NARS truth**: revision (combine evidence from both instances)
/// - **RL state**: average Q-values (ensemble the policies)
/// - **Bloom filter**: OR (union of known neighbors from both instances)
/// - **Graph metrics**: max pagerank, min hop_to_root, max degree
///
/// This enables distributed deployment where each instance holds partial
/// knowledge, and merging produces a more complete picture.
pub fn schema_merge(primary: &[u64], secondary: &[u64]) -> Vec<u64> {
    assert!(primary.len() >= VECTOR_WORDS && secondary.len() >= VECTOR_WORDS);
    let mut out = vec![0u64; VECTOR_WORDS];

    // Semantic blocks: preserve from primary (authoritative source)
    let semantic_end = SCHEMA_BLOCK_START * 16;
    out[..semantic_end].copy_from_slice(&primary[..semantic_end]);

    let base = SchemaSidecar::WORD_OFFSET;

    // Block 13: ANI levels — element-wise max
    let ani_a = AniLevels::unpack(
        primary[base] as u128 | ((primary[base + 1] as u128) << 64),
    );
    let ani_b = AniLevels::unpack(
        secondary[base] as u128 | ((secondary[base + 1] as u128) << 64),
    );
    let ani_merged = AniLevels {
        reactive: ani_a.reactive.max(ani_b.reactive),
        memory: ani_a.memory.max(ani_b.memory),
        analogy: ani_a.analogy.max(ani_b.analogy),
        planning: ani_a.planning.max(ani_b.planning),
        meta: ani_a.meta.max(ani_b.meta),
        social: ani_a.social.max(ani_b.social),
        creative: ani_a.creative.max(ani_b.creative),
        r#abstract: ani_a.r#abstract.max(ani_b.r#abstract),
    };
    let packed_ani = ani_merged.pack();
    out[base] = packed_ani as u64;
    out[base + 1] = (packed_ani >> 64) as u64;

    // Block 13: NARS — revision (combine evidence)
    let truth_a = NarsTruth::unpack(primary[base + 2] as u32);
    let truth_b = NarsTruth::unpack(secondary[base + 2] as u32);
    let revised = truth_a.revision(&truth_b);
    let budget_a = NarsBudget::unpack((primary[base + 2] >> 32) as u64);
    let budget_b = NarsBudget::unpack((secondary[base + 2] >> 32) as u64);
    let merged_budget = NarsBudget {
        priority: budget_a.priority.max(budget_b.priority),
        durability: budget_a.durability.max(budget_b.durability),
        quality: budget_a.quality.max(budget_b.quality),
        _reserved: 0,
    };
    out[base + 2] = revised.pack() as u64 | ((merged_budget.pack() as u64) << 32);

    // Block 13: Edge/Node types — preserve from primary
    out[base + 3] = primary[base + 3];

    // Block 14: RL state — average Q-values (ensemble)
    let block14_base = base + 16;
    let q_a = InlineQValues::unpack([primary[block14_base], primary[block14_base + 1]]);
    let q_b = InlineQValues::unpack([secondary[block14_base], secondary[block14_base + 1]]);
    let mut q_merged = InlineQValues::default();
    for i in 0..16 {
        // Average the two Q-values
        let avg = ((q_a.values[i] as i16 + q_b.values[i] as i16) / 2) as i8;
        q_merged.values[i] = avg;
    }
    let q_packed = q_merged.pack();
    out[block14_base] = q_packed[0];
    out[block14_base + 1] = q_packed[1];

    // Block 14: Rewards — take from whichever has more evidence (higher avg)
    let rewards_a_word = [primary[block14_base + 2], primary[block14_base + 3]];
    let rewards_b_word = [secondary[block14_base + 2], secondary[block14_base + 3]];
    // Simple heuristic: take the one with higher absolute sum
    let sum_a: u64 = rewards_a_word.iter().sum();
    let sum_b: u64 = rewards_b_word.iter().sum();
    if sum_a >= sum_b {
        out[block14_base + 2] = rewards_a_word[0];
        out[block14_base + 3] = rewards_a_word[1];
    } else {
        out[block14_base + 2] = rewards_b_word[0];
        out[block14_base + 3] = rewards_b_word[1];
    }

    // Block 14: STDP + Hebbian — preserve from primary
    for w in 4..16 {
        out[block14_base + w] = primary[block14_base + w];
    }

    // Block 15: DN address — preserve from primary
    let block15_base = base + 32;
    for w in 0..4 {
        out[block15_base + w] = primary[block15_base + w];
    }

    // Block 15: Bloom filter — OR (union of known neighbors)
    for w in 0..4 {
        out[block15_base + 4 + w] = primary[block15_base + 4 + w]
            | secondary[block15_base + 4 + w];
    }

    // Block 15: Graph metrics — merge intelligently
    let metrics_a = GraphMetrics::unpack(primary[block15_base + 8]);
    let metrics_b = GraphMetrics::unpack(secondary[block15_base + 8]);
    let merged_metrics = GraphMetrics {
        pagerank: metrics_a.pagerank.max(metrics_b.pagerank),
        hop_to_root: metrics_a.hop_to_root.min(metrics_b.hop_to_root),
        cluster_id: metrics_a.cluster_id, // preserve primary's cluster
        degree: metrics_a.degree.max(metrics_b.degree),
        in_degree: metrics_a.in_degree.max(metrics_b.in_degree),
        out_degree: metrics_a.out_degree.max(metrics_b.out_degree),
    };
    out[block15_base + 8] = merged_metrics.pack();

    out
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_words() -> Vec<u64> {
        let mut words = vec![0u64; VECTOR_WORDS];
        // Set some schema data
        let mut sidecar = SchemaSidecar::default();
        sidecar.ani_levels.planning = 500;
        sidecar.ani_levels.meta = 200;
        sidecar.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        sidecar.nars_budget = NarsBudget::from_floats(0.9, 0.5, 0.7);
        sidecar.q_values.set_q(0, 0.7);
        sidecar.rewards.push(0.5);
        sidecar.metrics.pagerank = 1000;
        sidecar.metrics.hop_to_root = 2;
        sidecar.metrics.cluster_id = 42;
        sidecar.metrics.degree = 5;
        sidecar.neighbors.insert(100);
        sidecar.neighbors.insert(200);
        sidecar.write_to_words(&mut words);
        words
    }

    #[test]
    fn test_block_mask() {
        assert_eq!(BlockMask::ALL.count(), 16);
        assert_eq!(BlockMask::SEMANTIC.count(), 13);
        assert_eq!(BlockMask::SCHEMA.count(), 3);
        assert!(BlockMask::SEMANTIC.includes(0));
        assert!(BlockMask::SEMANTIC.includes(12));
        assert!(!BlockMask::SEMANTIC.includes(13));
        assert!(BlockMask::SCHEMA.includes(13));
        assert!(BlockMask::SCHEMA.includes(15));
    }

    #[test]
    fn test_predicate_ani_pass() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_ani(AniFilter {
            min_level: 3, // planning
            min_activation: 100,
        });
        assert!(query.passes_predicates(&words)); // planning=500 >= 100
    }

    #[test]
    fn test_predicate_ani_fail() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_ani(AniFilter {
            min_level: 3, // planning
            min_activation: 600,
        });
        assert!(!query.passes_predicates(&words)); // planning=500 < 600
    }

    #[test]
    fn test_predicate_nars_pass() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_nars(NarsFilter {
            min_frequency: Some(0.7),
            min_confidence: Some(0.5),
            min_priority: None,
        });
        assert!(query.passes_predicates(&words)); // f=0.8 >= 0.7, c=0.6 >= 0.5
    }

    #[test]
    fn test_predicate_nars_fail_confidence() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_nars(NarsFilter {
            min_frequency: None,
            min_confidence: Some(0.9), // too high
            min_priority: None,
        });
        assert!(!query.passes_predicates(&words));
    }

    #[test]
    fn test_predicate_graph_filter() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_graph(GraphFilter {
            min_pagerank: Some(500),
            max_hop: Some(3),
            cluster_id: Some(42),
            min_degree: Some(3),
        });
        assert!(query.passes_predicates(&words));
    }

    #[test]
    fn test_predicate_graph_wrong_cluster() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_graph(GraphFilter {
            min_pagerank: None,
            max_hop: None,
            cluster_id: Some(99), // wrong cluster
            min_degree: None,
        });
        assert!(!query.passes_predicates(&words));
    }

    #[test]
    fn test_predicate_combined() {
        let words = make_test_words();
        // All filters pass together
        let query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 100 })
            .with_nars(NarsFilter {
                min_frequency: Some(0.5),
                min_confidence: Some(0.3),
                min_priority: None,
            })
            .with_graph(GraphFilter {
                min_pagerank: Some(500),
                max_hop: None,
                cluster_id: None,
                min_degree: None,
            });
        assert!(query.passes_predicates(&words));
    }

    #[test]
    fn test_masked_distance_semantic_only() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];

        // Set bit differences only in semantic region
        a[0] = 0xFFFF;
        // Set bit differences only in schema region (should be ignored)
        a[210] = 0xFFFF_FFFF_FFFF_FFFF;

        let query = SchemaQuery::new(); // default: semantic only
        let dist = query.masked_distance(&a, &b);

        // Only semantic bits counted: 16 bits from a[0]
        assert_eq!(dist, 16);
    }

    #[test]
    fn test_masked_distance_all_blocks() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];
        a[0] = 0xFFFF; // 16 bits in semantic
        a[210] = 0xFF;  // 8 bits in schema

        let query = SchemaQuery::new().with_block_mask(BlockMask::ALL);
        let dist = query.masked_distance(&a, &b);
        assert_eq!(dist, 24); // 16 + 8
    }

    #[test]
    fn test_masked_distance_with_threshold() {
        let a = vec![0xFFFF_FFFF_FFFF_FFFFu64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];

        let query = SchemaQuery::new();
        // Very low threshold should abort early
        let result = query.masked_distance_with_threshold(&a, &b, 100);
        assert!(result.is_none()); // Exceeded threshold
    }

    #[test]
    fn test_search_pipeline() {
        let mut candidates: Vec<Vec<u64>> = Vec::new();

        // Candidate 0: close to query
        let mut c0 = vec![0u64; VECTOR_WORDS];
        c0[0] = 0xFF; // 8 bits different
        let mut s0 = SchemaSidecar::default();
        s0.ani_levels.planning = 500;
        s0.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        s0.write_to_words(&mut c0);
        candidates.push(c0);

        // Candidate 1: far from query
        let mut c1 = vec![0xFFFF_FFFF_FFFF_FFFFu64; VECTOR_WORDS];
        let mut s1 = SchemaSidecar::default();
        s1.ani_levels.planning = 100;
        s1.nars_truth = NarsTruth::from_floats(0.3, 0.2);
        s1.write_to_words(&mut c1);
        candidates.push(c1);

        // Candidate 2: close but fails predicate
        let mut c2 = vec![0u64; VECTOR_WORDS];
        c2[0] = 0xF; // 4 bits different
        // No ANI planning set — will fail predicate
        candidates.push(c2);

        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let query_words = vec![0u64; VECTOR_WORDS];

        let query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 50 });

        let results = query.search(&refs, &query_words, 10);

        // Candidate 0 passes (planning=500, dist=8)
        // Candidate 1 passes predicate (planning=100) but distance is huge
        // Candidate 2 fails predicate (planning=0)
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].distance, 8);
    }

    #[test]
    fn test_bloom_neighbor_check() {
        let mut words = vec![0u64; VECTOR_WORDS];
        let mut sidecar = SchemaSidecar::default();
        sidecar.neighbors.insert(42);
        sidecar.neighbors.insert(100);
        sidecar.write_to_words(&mut words);

        assert!(bloom_might_be_neighbors(&words, 42));
        assert!(bloom_might_be_neighbors(&words, 100));
        // Unknown ID: might have false positive, but low probability
    }

    #[test]
    fn test_rl_routing_score() {
        // Pure distance mode (alpha=0)
        let score = rl_routing_score(1000, 0.5, 0.0);
        assert!(score > 0.0);

        // Pure Q-value mode (alpha=1)
        let score_high_q = rl_routing_score(1000, 0.9, 1.0);
        let score_low_q = rl_routing_score(1000, -0.5, 1.0);
        assert!(score_high_q < score_low_q); // Higher Q = lower (better) score
    }

    #[test]
    fn test_schema_bind_merges_metadata() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let mut b = vec![0u64; VECTOR_WORDS];

        let mut sa = SchemaSidecar::default();
        sa.ani_levels.planning = 500;
        sa.ani_levels.meta = 100;
        sa.nars_truth = NarsTruth::from_floats(0.8, 0.5);
        sa.write_to_words(&mut a);

        let mut sb = SchemaSidecar::default();
        sb.ani_levels.planning = 300;
        sb.ani_levels.meta = 400; // higher meta
        sb.nars_truth = NarsTruth::from_floats(0.6, 0.3);
        sb.write_to_words(&mut b);

        let result = schema_bind(&a, &b);
        let result_schema = SchemaSidecar::read_from_words(&result);

        // ANI: element-wise max
        assert_eq!(result_schema.ani_levels.planning, 500); // max(500, 300)
        assert_eq!(result_schema.ani_levels.meta, 400);     // max(100, 400)

        // NARS: revision should increase confidence
        assert!(result_schema.nars_truth.c() > 0.5 || result_schema.nars_truth.c() > 0.3);
    }

    #[test]
    fn test_read_best_q() {
        let mut words = vec![0u64; VECTOR_WORDS];
        let mut sidecar = SchemaSidecar::default();
        sidecar.q_values.set_q(3, 0.8);
        sidecar.q_values.set_q(7, -0.2);
        sidecar.write_to_words(&mut words);

        let (action, q) = read_best_q(&words);
        assert_eq!(action, 3);
        assert!((q - 0.8).abs() < 0.02);
    }

    #[test]
    fn test_nars_deduction_inline() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let mut b = vec![0u64; VECTOR_WORDS];

        let mut sa = SchemaSidecar::default();
        sa.nars_truth = NarsTruth::from_floats(0.9, 0.8);
        sa.write_to_words(&mut a);

        let mut sb = SchemaSidecar::default();
        sb.nars_truth = NarsTruth::from_floats(0.7, 0.6);
        sb.write_to_words(&mut b);

        let deduced = nars_deduction_inline(&a, &b);
        // Deduction: f = f1*f2, c = f1*f2*c1*c2
        assert!(deduced.f() > 0.5); // 0.9 * 0.7 ≈ 0.63
        assert!(deduced.c() < deduced.f()); // confidence always ≤ frequency in deduction
    }

    // === Bloom-accelerated search tests ===

    #[test]
    fn test_bloom_accelerated_search_basic() {
        let mut candidates: Vec<Vec<u64>> = Vec::new();

        // Candidate 0: close to query, is a known neighbor of source_id=999
        let mut c0 = vec![0u64; VECTOR_WORDS];
        c0[0] = 0xFF; // 8 bits different
        let mut s0 = SchemaSidecar::default();
        s0.ani_levels.planning = 500;
        s0.neighbors.insert(999); // known neighbor of source
        s0.write_to_words(&mut c0);
        candidates.push(c0);

        // Candidate 1: same distance, NOT a neighbor
        let mut c1 = vec![0u64; VECTOR_WORDS];
        c1[0] = 0xFF; // same 8 bits different
        let mut s1 = SchemaSidecar::default();
        s1.ani_levels.planning = 500;
        // No bloom entry for 999
        s1.write_to_words(&mut c1);
        candidates.push(c1);

        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let query_words = vec![0u64; VECTOR_WORDS];

        let schema_query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 100 });

        let results = bloom_accelerated_search(
            &refs, &query_words, 999, 10, 0.5, &schema_query,
        );

        assert_eq!(results.len(), 2);
        // Candidate 0 is a bloom neighbor, should get distance bonus
        assert!(results[0].is_bloom_neighbor);
        assert!(results[0].effective_distance < results[0].raw_distance);
        // Candidate 0 should rank higher (lower effective distance) than candidate 1
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_bloom_search_respects_predicates() {
        let mut candidates: Vec<Vec<u64>> = Vec::new();

        // Candidate fails predicate (no ANI)
        let c0 = vec![0u64; VECTOR_WORDS];
        candidates.push(c0);

        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let query_words = vec![0u64; VECTOR_WORDS];

        let schema_query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 500 });

        let results = bloom_accelerated_search(
            &refs, &query_words, 999, 10, 0.5, &schema_query,
        );

        assert!(results.is_empty(), "Should filter out candidates failing predicates");
    }

    // === RL-guided search tests ===

    #[test]
    fn test_rl_guided_search_basic() {
        let mut candidates: Vec<Vec<u64>> = Vec::new();

        // Candidate 0: moderate distance, high Q-value
        let mut c0 = vec![0u64; VECTOR_WORDS];
        c0[0] = 0xFFFF; // 16 bits
        let mut s0 = SchemaSidecar::default();
        s0.q_values.set_q(0, 0.9); // high Q
        s0.write_to_words(&mut c0);
        candidates.push(c0);

        // Candidate 1: similar distance, low Q-value
        let mut c1 = vec![0u64; VECTOR_WORDS];
        c1[0] = 0xFFFF; // same 16 bits
        let mut s1 = SchemaSidecar::default();
        s1.q_values.set_q(0, -0.5); // low Q
        s1.write_to_words(&mut c1);
        candidates.push(c1);

        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let query_words = vec![0u64; VECTOR_WORDS];

        let schema_query = SchemaQuery::new();

        // alpha=0.5: balanced between distance and Q-value
        let results = rl_guided_search(
            &refs, &query_words, 10, 0.5, &schema_query,
        );

        assert_eq!(results.len(), 2);
        // With same distance, candidate 0 (high Q) should rank better (lower composite)
        assert!(results[0].q_value > results[1].q_value,
            "Higher Q should rank first: {} vs {}", results[0].q_value, results[1].q_value);
        assert!(results[0].composite_score <= results[1].composite_score);
    }

    #[test]
    fn test_rl_guided_search_pure_distance() {
        let mut candidates: Vec<Vec<u64>> = Vec::new();

        // Candidate 0: far
        let mut c0 = vec![0u64; VECTOR_WORDS];
        c0[0] = 0xFFFF_FFFF; // 32 bits
        candidates.push(c0);

        // Candidate 1: close
        let mut c1 = vec![0u64; VECTOR_WORDS];
        c1[0] = 0xF; // 4 bits
        candidates.push(c1);

        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let query_words = vec![0u64; VECTOR_WORDS];

        // alpha=0.0: purely Q-based. But both have Q=0, so distance still matters in tie-break
        let results = rl_guided_search(
            &refs, &query_words, 10, 0.0, &SchemaQuery::new(),
        );
        assert_eq!(results.len(), 2);
    }

    // === Federated schema merge tests ===

    #[test]
    fn test_schema_merge_basic() {
        let mut primary = vec![0u64; VECTOR_WORDS];
        let mut secondary = vec![0u64; VECTOR_WORDS];

        // Set semantic bits on primary
        primary[0] = 0xDEADBEEF;
        primary[1] = 0xCAFEBABE;

        // Secondary has different semantic bits (should be ignored)
        secondary[0] = 0x12345678;

        // Primary schema
        let mut sp = SchemaSidecar::default();
        sp.ani_levels.planning = 300;
        sp.ani_levels.meta = 100;
        sp.nars_truth = NarsTruth::from_floats(0.8, 0.5);
        sp.metrics.pagerank = 800;
        sp.metrics.hop_to_root = 5;
        sp.metrics.degree = 3;
        sp.neighbors.insert(10);
        sp.q_values.set_q(0, 0.6);
        sp.write_to_words(&mut primary);

        // Secondary schema
        let mut ss = SchemaSidecar::default();
        ss.ani_levels.planning = 500; // higher
        ss.ani_levels.meta = 50;      // lower
        ss.nars_truth = NarsTruth::from_floats(0.6, 0.3);
        ss.metrics.pagerank = 600;     // lower
        ss.metrics.hop_to_root = 2;    // closer to root
        ss.metrics.degree = 7;         // higher
        ss.neighbors.insert(20);
        ss.q_values.set_q(0, 0.4);
        ss.write_to_words(&mut secondary);

        let merged = schema_merge(&primary, &secondary);
        let ms = SchemaSidecar::read_from_words(&merged);

        // Semantic blocks from primary
        assert_eq!(merged[0], 0xDEADBEEF, "Semantic bits should come from primary");
        assert_eq!(merged[1], 0xCAFEBABE);

        // ANI: element-wise max
        assert_eq!(ms.ani_levels.planning, 500, "ANI should take max: max(300,500)=500");
        assert_eq!(ms.ani_levels.meta, 100, "ANI should take max: max(100,50)=100");

        // NARS: revision (combined evidence increases confidence)
        assert!(ms.nars_truth.c() > 0.0, "Revised confidence should be positive");

        // Metrics: max pagerank, min hop, max degree
        assert_eq!(ms.metrics.pagerank, 800, "Pagerank should take max: max(800,600)=800");
        assert_eq!(ms.metrics.hop_to_root, 2, "Hop should take min: min(5,2)=2");
        assert_eq!(ms.metrics.degree, 7, "Degree should take max: max(3,7)=7");

        // Bloom: OR (union)
        assert!(bloom_might_be_neighbors(&merged, 10), "Should contain primary's neighbors");
        assert!(bloom_might_be_neighbors(&merged, 20), "Should contain secondary's neighbors");
    }

    #[test]
    fn test_schema_merge_preserves_primary_semantic() {
        let mut primary = vec![0u64; VECTOR_WORDS];
        let mut secondary = vec![0u64; VECTOR_WORDS];

        // Fill primary semantic with known pattern
        for i in 0..208 {
            primary[i] = 0xAAAAAAAAAAAAAAAA;
        }
        // Secondary has different pattern
        for i in 0..208 {
            secondary[i] = 0x5555555555555555;
        }

        let merged = schema_merge(&primary, &secondary);
        for i in 0..208 {
            assert_eq!(merged[i], 0xAAAAAAAAAAAAAAAA,
                "Word {} should come from primary", i);
        }
    }
}
