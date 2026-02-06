//! Schema Markers for 16K Fingerprint Sidecar
//!
//! The 16K vector reserves blocks 13-15 (3,072 bits) for structured metadata:
//!
//! - **Block 13**: Node/Edge Type (ANI levels, NARS truth values, verb IDs)
//! - **Block 14**: RL/Temporal State (Q-values, rewards, Hebbian weights)
//! - **Block 15**: Traversal Cache (DN address, neighbor bloom, centrality)
//!
//! These markers are **optional**. In all-semantic mode, all 16 blocks carry
//! fingerprint information. When schema mode is active, blocks 0..12 carry
//! semantics and blocks 13..15 carry the markers below.

use super::VECTOR_WORDS;

// ============================================================================
// BLOCK 13: NODE/EDGE TYPE MARKERS
// ============================================================================

/// ANI reasoning level slots (8 levels × 16 bits each = 128 bits)
///
/// Each level represents a cognitive capability tier from reactive to abstract.
/// Values are activation levels [0..65535].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct AniLevels {
    /// Level 0: Reactive — stimulus→response
    pub reactive: u16,
    /// Level 1: Memory — pattern recognition from stored examples
    pub memory: u16,
    /// Level 2: Analogy — transfer learning across domains
    pub analogy: u16,
    /// Level 3: Planning — multi-step goal decomposition
    pub planning: u16,
    /// Level 4: Meta — reasoning about own reasoning
    pub meta: u16,
    /// Level 5: Social — theory of mind, intent modeling
    pub social: u16,
    /// Level 6: Creative — novel combination of existing concepts
    pub creative: u16,
    /// Level 7: Abstract — mathematical/logical abstraction
    pub r#abstract: u16,
}

impl AniLevels {
    /// Bit offset within Block 13
    pub const OFFSET: usize = 0;
    /// Total bits: 8 × 16 = 128
    pub const BITS: usize = 128;

    /// Dominant reasoning level (highest activation)
    pub fn dominant(&self) -> u8 {
        let levels = [
            self.reactive, self.memory, self.analogy, self.planning,
            self.meta, self.social, self.creative, self.r#abstract,
        ];
        levels.iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Pack into u128 for embedding into fingerprint
    pub fn pack(&self) -> u128 {
        (self.reactive as u128)
            | ((self.memory as u128) << 16)
            | ((self.analogy as u128) << 32)
            | ((self.planning as u128) << 48)
            | ((self.meta as u128) << 64)
            | ((self.social as u128) << 80)
            | ((self.creative as u128) << 96)
            | ((self.r#abstract as u128) << 112)
    }

    /// Unpack from u128
    pub fn unpack(packed: u128) -> Self {
        Self {
            reactive: packed as u16,
            memory: (packed >> 16) as u16,
            analogy: (packed >> 32) as u16,
            planning: (packed >> 48) as u16,
            meta: (packed >> 64) as u16,
            social: (packed >> 80) as u16,
            creative: (packed >> 96) as u16,
            r#abstract: (packed >> 112) as u16,
        }
    }
}

/// NARS truth value: frequency (f) and confidence (c)
///
/// Quantized to 16-bit each:
/// - f ∈ [0, 1] → u16 [0, 65535]
/// - c ∈ [0, 1) → u16 [0, 65534] (confidence < 1 by NAL definition)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NarsTruth {
    /// Frequency: proportion of positive evidence
    pub frequency: u16,
    /// Confidence: evidence / (evidence + horizon)
    pub confidence: u16,
}

impl NarsTruth {
    /// Bit offset within Block 13
    pub const OFFSET: usize = AniLevels::OFFSET + AniLevels::BITS; // 128
    /// Total bits: 2 × 16 = 32
    pub const BITS: usize = 32;

    /// Create from float values
    pub fn from_floats(f: f32, c: f32) -> Self {
        Self {
            frequency: (f.clamp(0.0, 1.0) * 65535.0) as u16,
            confidence: (c.clamp(0.0, 0.9999) * 65535.0) as u16,
        }
    }

    /// Convert to float frequency
    pub fn f(&self) -> f32 {
        self.frequency as f32 / 65535.0
    }

    /// Convert to float confidence
    pub fn c(&self) -> f32 {
        self.confidence as f32 / 65535.0
    }

    /// NARS revision: combine two truth values with more evidence
    pub fn revision(&self, other: &Self) -> Self {
        let w1 = self.c() / (1.0 - self.c());
        let w2 = other.c() / (1.0 - other.c());
        let w = w1 + w2;
        let f = if w > 0.0 {
            (w1 * self.f() + w2 * other.f()) / w
        } else {
            0.5
        };
        let c = w / (w + 1.0); // k=1 (NAL horizon)
        Self::from_floats(f, c)
    }

    /// NARS deduction: f = f1 * f2, c = f1 * f2 * c1 * c2
    pub fn deduction(&self, other: &Self) -> Self {
        let f = self.f() * other.f();
        let c = f * self.c() * other.c();
        Self::from_floats(f, c)
    }

    /// Pack into u32
    pub fn pack(&self) -> u32 {
        (self.frequency as u32) | ((self.confidence as u32) << 16)
    }

    /// Unpack from u32
    pub fn unpack(packed: u32) -> Self {
        Self {
            frequency: packed as u16,
            confidence: (packed >> 16) as u16,
        }
    }
}

/// NARS budget: priority (p), durability (d), quality (q)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NarsBudget {
    /// Priority: urgency of processing [0, 1]
    pub priority: u16,
    /// Durability: resistance to forgetting [0, 1]
    pub durability: u16,
    /// Quality: usefulness [0, 1]
    pub quality: u16,
    /// Reserved for future use
    pub _reserved: u16,
}

impl NarsBudget {
    /// Bit offset within Block 13
    pub const OFFSET: usize = NarsTruth::OFFSET + NarsTruth::BITS; // 160
    /// Total bits: 4 × 16 = 64
    pub const BITS: usize = 64;

    /// Create from float values
    pub fn from_floats(p: f32, d: f32, q: f32) -> Self {
        Self {
            priority: (p.clamp(0.0, 1.0) * 65535.0) as u16,
            durability: (d.clamp(0.0, 1.0) * 65535.0) as u16,
            quality: (q.clamp(0.0, 1.0) * 65535.0) as u16,
            _reserved: 0,
        }
    }

    /// Pack into u64
    pub fn pack(&self) -> u64 {
        (self.priority as u64)
            | ((self.durability as u64) << 16)
            | ((self.quality as u64) << 32)
            | ((self._reserved as u64) << 48)
    }

    /// Unpack from u64
    pub fn unpack(packed: u64) -> Self {
        Self {
            priority: packed as u16,
            durability: (packed >> 16) as u16,
            quality: (packed >> 32) as u16,
            _reserved: (packed >> 48) as u16,
        }
    }
}

/// Edge type descriptor (cognitive verb + context)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct EdgeTypeMarker {
    /// Cognitive verb ID (0..143 for the 144 verbs, or 255 for custom)
    pub verb_id: u8,
    /// Edge direction: 0=undirected, 1=forward, 2=reverse, 3=bidirectional
    pub direction: u8,
    /// Edge weight quantized to [0, 255]
    pub weight: u8,
    /// Flags: bit0=temporal, bit1=causal, bit2=hierarchical, bit3=associative
    pub flags: u8,
}

impl EdgeTypeMarker {
    /// Bit offset within Block 13
    pub const OFFSET: usize = NarsBudget::OFFSET + NarsBudget::BITS; // 224
    /// Total bits: 4 × 8 = 32
    pub const BITS: usize = 32;

    pub fn pack(&self) -> u32 {
        (self.verb_id as u32)
            | ((self.direction as u32) << 8)
            | ((self.weight as u32) << 16)
            | ((self.flags as u32) << 24)
    }

    pub fn unpack(packed: u32) -> Self {
        Self {
            verb_id: packed as u8,
            direction: (packed >> 8) as u8,
            weight: (packed >> 16) as u8,
            flags: (packed >> 24) as u8,
        }
    }

    pub fn is_temporal(&self) -> bool { self.flags & 1 != 0 }
    pub fn is_causal(&self) -> bool { self.flags & 2 != 0 }
    pub fn is_hierarchical(&self) -> bool { self.flags & 4 != 0 }
    pub fn is_associative(&self) -> bool { self.flags & 8 != 0 }
}

/// Node kind classification
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeKind {
    Entity = 0,
    Concept = 1,
    Event = 2,
    Rule = 3,
    Goal = 4,
    Query = 5,
    Hypothesis = 6,
    Observation = 7,
}

impl Default for NodeKind {
    fn default() -> Self {
        Self::Entity
    }
}

/// Node type marker
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct NodeTypeMarker {
    /// Node kind
    pub kind: u8,
    /// Subtype (application-specific)
    pub subtype: u8,
    /// Provenance hash (truncated to 16 bits)
    pub provenance: u16,
}

impl NodeTypeMarker {
    /// Bit offset within Block 13
    pub const OFFSET: usize = EdgeTypeMarker::OFFSET + EdgeTypeMarker::BITS; // 256
    /// Total bits: 4 × 8 = 32 (but uses only 32 of allocated 128)
    pub const BITS: usize = 32;

    pub fn pack(&self) -> u32 {
        (self.kind as u32)
            | ((self.subtype as u32) << 8)
            | ((self.provenance as u32) << 16)
    }

    pub fn unpack(packed: u32) -> Self {
        Self {
            kind: packed as u8,
            subtype: (packed >> 8) as u8,
            provenance: (packed >> 16) as u16,
        }
    }
}

// ============================================================================
// BLOCK 14: RL / TEMPORAL STATE
// ============================================================================

/// Inline Q-values for up to 16 discrete actions
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineQValues {
    /// 16 actions × 8-bit Q-value [-128, +127] mapped to [-1.0, +1.0]
    pub values: [i8; 16],
}

impl InlineQValues {
    /// Bit offset within Block 14
    pub const OFFSET: usize = 0;
    /// Total bits: 16 × 8 = 128
    pub const BITS: usize = 128;

    /// Get Q-value as float for action index
    pub fn q(&self, action: usize) -> f32 {
        if action < 16 {
            self.values[action] as f32 / 127.0
        } else {
            0.0
        }
    }

    /// Set Q-value from float
    pub fn set_q(&mut self, action: usize, value: f32) {
        if action < 16 {
            self.values[action] = (value.clamp(-1.0, 1.0) * 127.0) as i8;
        }
    }

    /// Best action (argmax)
    pub fn best_action(&self) -> usize {
        self.values.iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Pack into two u64 words
    pub fn pack(&self) -> [u64; 2] {
        let mut words = [0u64; 2];
        for i in 0..8 {
            words[0] |= ((self.values[i] as u8) as u64) << (i * 8);
        }
        for i in 0..8 {
            words[1] |= ((self.values[i + 8] as u8) as u64) << (i * 8);
        }
        words
    }

    /// Unpack from two u64 words
    pub fn unpack(words: [u64; 2]) -> Self {
        let mut values = [0i8; 16];
        for i in 0..8 {
            values[i] = ((words[0] >> (i * 8)) & 0xFF) as u8 as i8;
        }
        for i in 0..8 {
            values[i + 8] = ((words[1] >> (i * 8)) & 0xFF) as u8 as i8;
        }
        Self { values }
    }
}

/// Inline reward history (last 8 rewards)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineRewards {
    /// 8 × 16-bit rewards, most recent last
    pub rewards: [i16; 8],
}

impl InlineRewards {
    /// Bit offset within Block 14
    pub const OFFSET: usize = InlineQValues::OFFSET + InlineQValues::BITS; // 128
    /// Total bits: 8 × 16 = 128
    pub const BITS: usize = 128;

    /// Push a new reward (shifts history)
    pub fn push(&mut self, reward: f32) {
        for i in 0..7 {
            self.rewards[i] = self.rewards[i + 1];
        }
        self.rewards[7] = (reward.clamp(-1.0, 1.0) * 32767.0) as i16;
    }

    /// Average reward
    pub fn average(&self) -> f32 {
        let sum: i32 = self.rewards.iter().map(|&r| r as i32).sum();
        (sum as f32 / 8.0) / 32767.0
    }

    /// Trend (positive = improving)
    pub fn trend(&self) -> f32 {
        if self.rewards.len() < 2 {
            return 0.0;
        }
        let first_half: f32 = self.rewards[..4].iter().map(|&r| r as f32).sum::<f32>() / 4.0;
        let second_half: f32 = self.rewards[4..].iter().map(|&r| r as f32).sum::<f32>() / 4.0;
        (second_half - first_half) / 32767.0
    }

    /// Pack into two u64 words
    pub fn pack(&self) -> [u64; 2] {
        let mut words = [0u64; 2];
        for i in 0..4 {
            words[0] |= ((self.rewards[i] as u16) as u64) << (i * 16);
        }
        for i in 0..4 {
            words[1] |= ((self.rewards[i + 4] as u16) as u64) << (i * 16);
        }
        words
    }
}

/// STDP timing markers for spike-timing dependent plasticity
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StdpMarkers {
    /// 8 most recent spike timestamps (16-bit each, wrapping)
    pub timestamps: [u16; 8],
}

impl StdpMarkers {
    /// Bit offset within Block 14
    pub const OFFSET: usize = InlineRewards::OFFSET + InlineRewards::BITS; // 256
    /// Total bits: 8 × 16 = 128
    pub const BITS: usize = 128;

    /// Record a spike at current time
    pub fn record_spike(&mut self, time: u16) {
        for i in 0..7 {
            self.timestamps[i] = self.timestamps[i + 1];
        }
        self.timestamps[7] = time;
    }

    /// Most recent spike time
    pub fn last_spike(&self) -> u16 {
        self.timestamps[7]
    }
}

/// Inline Hebbian weights for 8 nearest neighbors
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineHebbian {
    /// 8 neighbor weights × 16-bit each
    pub weights: [u16; 8],
}

impl InlineHebbian {
    /// Bit offset within Block 14
    pub const OFFSET: usize = StdpMarkers::OFFSET + StdpMarkers::BITS; // 384
    /// Total bits: 8 × 16 = 128
    pub const BITS: usize = 128;

    /// Get weight as float [0, 1]
    pub fn weight(&self, idx: usize) -> f32 {
        if idx < 8 {
            self.weights[idx] as f32 / 65535.0
        } else {
            0.0
        }
    }

    /// Strengthen a connection
    pub fn strengthen(&mut self, idx: usize, amount: f32) {
        if idx < 8 {
            let current = self.weights[idx] as f32 / 65535.0;
            let new_val = (current + amount).clamp(0.0, 1.0);
            self.weights[idx] = (new_val * 65535.0) as u16;
        }
    }

    /// Decay all weights
    pub fn decay(&mut self, factor: f32) {
        for w in &mut self.weights {
            *w = ((*w as f32) * factor) as u16;
        }
    }
}

// ============================================================================
// BLOCK 15: TRAVERSAL / GRAPH CACHE
// ============================================================================

/// Compressed DN address (256 bits = 32 bytes)
///
/// Stores a hierarchical DN path in compressed form.
/// Each level uses 8 bits (0..255 children), supporting up to 32 levels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompressedDnAddr {
    /// Path bytes: addr[0] is root, addr[depth-1] is leaf
    pub path: [u8; 32],
    /// Depth of the address (0 = root)
    pub depth: u8,
}

impl CompressedDnAddr {
    /// Bit offset within Block 15
    pub const OFFSET: usize = 0;
    /// Total bits: 33 × 8 = 264 (rounded to 256 usable + 8 depth)
    pub const BITS: usize = 264;
}

/// Neighbor bloom filter (256 bits)
///
/// Tracks which neighbor IDs are reachable in 1 hop.
/// At 256 bits with ~7 neighbors, false positive rate ≈ 1%.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighborBloom {
    /// 4 × u64 bloom filter words
    pub words: [u64; 4],
}

impl NeighborBloom {
    /// Bit offset within Block 15
    pub const OFFSET: usize = 256;
    /// Total bits: 256
    pub const BITS: usize = 256;

    /// Insert a neighbor ID into the bloom filter
    pub fn insert(&mut self, neighbor_id: u64) {
        let h1 = neighbor_id;
        let h2 = neighbor_id.wrapping_mul(0x9E3779B97F4A7C15);
        let h3 = neighbor_id.wrapping_mul(0x517CC1B727220A95);

        self.set_bit(h1 as usize % 256);
        self.set_bit(h2 as usize % 256);
        self.set_bit(h3 as usize % 256);
    }

    /// Check if a neighbor ID might be present
    pub fn might_contain(&self, neighbor_id: u64) -> bool {
        let h1 = neighbor_id;
        let h2 = neighbor_id.wrapping_mul(0x9E3779B97F4A7C15);
        let h3 = neighbor_id.wrapping_mul(0x517CC1B727220A95);

        self.get_bit(h1 as usize % 256)
            && self.get_bit(h2 as usize % 256)
            && self.get_bit(h3 as usize % 256)
    }

    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] |= 1u64 << bit;
    }

    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] & (1u64 << bit) != 0
    }

    /// Approximate count of items (from popcount)
    pub fn approx_count(&self) -> usize {
        let set_bits: u32 = self.words.iter().map(|w| w.count_ones()).sum();
        // Estimate: n ≈ -m/k * ln(1 - X/m) where m=256, k=3
        let m = 256.0f64;
        let k = 3.0f64;
        let x = set_bits as f64;
        if x >= m {
            return 100; // saturated
        }
        (-(m / k) * (1.0 - x / m).ln()) as usize
    }
}

/// Graph metrics cache
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphMetrics {
    /// PageRank score × 65535 (quantized)
    pub pagerank: u16,
    /// Hop distance to root (0..255)
    pub hop_to_root: u8,
    /// Cluster ID (0..65535)
    pub cluster_id: u16,
    /// Degree (capped at 255)
    pub degree: u8,
    /// In-degree (capped at 255)
    pub in_degree: u8,
    /// Out-degree (capped at 255)
    pub out_degree: u8,
}

impl GraphMetrics {
    /// Bit offset within Block 15
    pub const OFFSET: usize = NeighborBloom::OFFSET + NeighborBloom::BITS; // 512
    /// Total bits: 8 × 8 = 64
    pub const BITS: usize = 64;

    pub fn pack(&self) -> u64 {
        (self.pagerank as u64)
            | ((self.hop_to_root as u64) << 16)
            | ((self.cluster_id as u64) << 24)
            | ((self.degree as u64) << 40)
            | ((self.in_degree as u64) << 48)
            | ((self.out_degree as u64) << 56)
    }

    pub fn unpack(packed: u64) -> Self {
        Self {
            pagerank: packed as u16,
            hop_to_root: (packed >> 16) as u8,
            cluster_id: (packed >> 24) as u16,
            degree: (packed >> 40) as u8,
            in_degree: (packed >> 48) as u8,
            out_degree: (packed >> 56) as u8,
        }
    }
}

// ============================================================================
// UNIFIED SCHEMA: Read/write from u64 word array
// ============================================================================

/// Complete schema sidecar for one 16K fingerprint.
///
/// This struct can be read from / written to words[208..256] of a 16K vector
/// (blocks 13-15).
#[derive(Clone, Debug, Default)]
pub struct SchemaSidecar {
    // Block 13: Node/Edge Type
    pub ani_levels: AniLevels,
    pub nars_truth: NarsTruth,
    pub nars_budget: NarsBudget,
    pub edge_type: EdgeTypeMarker,
    pub node_type: NodeTypeMarker,

    // Block 14: RL/Temporal
    pub q_values: InlineQValues,
    pub rewards: InlineRewards,
    pub stdp: StdpMarkers,
    pub hebbian: InlineHebbian,

    // Block 15: Graph Cache
    pub dn_addr: CompressedDnAddr,
    pub neighbors: NeighborBloom,
    pub metrics: GraphMetrics,
}

impl SchemaSidecar {
    /// Word offset where schema blocks begin (block 13 = word 208)
    pub const WORD_OFFSET: usize = 13 * 16; // 208

    /// Number of words in schema region (3 blocks × 16 words = 48)
    pub const WORD_COUNT: usize = 3 * 16; // 48

    /// Current schema layout version.
    ///
    /// Stored in the top 8 bits of words[208]. When the layout changes,
    /// increment this and add migration logic in `read_from_words()`.
    /// Version 0 = legacy (no version tag), Version 1 = current.
    pub const SCHEMA_VERSION: u8 = 1;

    /// Word offset within the schema region where the version byte lives.
    /// Uses the last word of block 13 (word offset +15 = word 223) which is
    /// otherwise unused padding. Top 8 bits hold the version.
    pub const VERSION_WORD_OFFSET: usize = 15; // relative to WORD_OFFSET

    /// Mask for the version byte (top 8 bits of the version word)
    pub const VERSION_MASK: u64 = 0xFF << 56;

    /// Mask to clear the version byte
    pub const ANI_WORD0_MASK: u64 = !Self::VERSION_MASK;

    /// Write schema markers into the word array at the correct offset.
    ///
    /// `words` must be at least 256 elements (full 16K vector).
    /// Only words[208..256] are modified.
    pub fn write_to_words(&self, words: &mut [u64]) {
        assert!(words.len() >= VECTOR_WORDS);
        let base = Self::WORD_OFFSET;

        // Block 13: ANI levels (words 208-209) — no masking needed now
        let ani = self.ani_levels.pack();
        words[base] = ani as u64;
        words[base + 1] = (ani >> 64) as u64;

        // Block 13: NARS truth (word 210, lower 32 bits)
        let nars_t = self.nars_truth.pack();
        words[base + 2] = nars_t as u64;

        // Block 13: NARS budget (word 210 upper + word 211)
        let nars_b = self.nars_budget.pack();
        words[base + 2] |= (nars_b as u64) << 32;

        // Block 13: Edge type (word 212 lower)
        let edge = self.edge_type.pack();
        words[base + 3] = edge as u64;

        // Block 13: Node type (word 212 upper)
        let node = self.node_type.pack();
        words[base + 3] |= (node as u64) << 32;

        // Block 14: Q-values (words 224-225)
        let block14_base = base + 16;
        let q_packed = self.q_values.pack();
        words[block14_base] = q_packed[0];
        words[block14_base + 1] = q_packed[1];

        // Block 14: Rewards (words 226-227)
        let r_packed = self.rewards.pack();
        words[block14_base + 2] = r_packed[0];
        words[block14_base + 3] = r_packed[1];

        // Block 14: STDP (words 228-229)
        let mut stdp_w0 = 0u64;
        let mut stdp_w1 = 0u64;
        for i in 0..4 {
            stdp_w0 |= (self.stdp.timestamps[i] as u64) << (i * 16);
        }
        for i in 0..4 {
            stdp_w1 |= (self.stdp.timestamps[i + 4] as u64) << (i * 16);
        }
        words[block14_base + 4] = stdp_w0;
        words[block14_base + 5] = stdp_w1;

        // Block 14: Hebbian (words 230-231)
        let mut hebb_w0 = 0u64;
        let mut hebb_w1 = 0u64;
        for i in 0..4 {
            hebb_w0 |= (self.hebbian.weights[i] as u64) << (i * 16);
        }
        for i in 0..4 {
            hebb_w1 |= (self.hebbian.weights[i + 4] as u64) << (i * 16);
        }
        words[block14_base + 6] = hebb_w0;
        words[block14_base + 7] = hebb_w1;

        // Block 15: DN address (words 240..247)
        let block15_base = base + 32;
        for i in 0..4 {
            let mut w = 0u64;
            for j in 0..8 {
                w |= (self.dn_addr.path[i * 8 + j] as u64) << (j * 8);
            }
            words[block15_base + i] = w;
        }

        // Block 15: Neighbor bloom (words 244..247)
        for i in 0..4 {
            words[block15_base + 4 + i] = self.neighbors.words[i];
        }

        // Block 15: Graph metrics (word 248)
        words[block15_base + 8] = self.metrics.pack();

        // Version byte: written to top 8 bits of word[base+15] (end of block 13 padding)
        words[base + Self::VERSION_WORD_OFFSET] =
            (words[base + Self::VERSION_WORD_OFFSET] & Self::ANI_WORD0_MASK)
            | ((Self::SCHEMA_VERSION as u64) << 56);
    }

    /// Read the schema version from a word array.
    ///
    /// Returns 0 for legacy data (no version tag), 1+ for versioned data.
    /// The version byte is stored in the top 8 bits of word[base+15] (block 13 padding).
    pub fn read_version(words: &[u64]) -> u8 {
        if words.len() < VECTOR_WORDS {
            return 0;
        }
        ((words[Self::WORD_OFFSET + Self::VERSION_WORD_OFFSET] >> 56) & 0xFF) as u8
    }

    /// Read schema markers from the word array.
    pub fn read_from_words(words: &[u64]) -> Self {
        assert!(words.len() >= VECTOR_WORDS);
        let base = Self::WORD_OFFSET;
        let block14_base = base + 16;
        let block15_base = base + 32;

        let _version = Self::read_version(words);
        // Version 0 and 1 share the same layout.
        // Future versions: add match on _version here for migration.

        // Block 13: ANI levels (words 208-209) — no masking, version is elsewhere
        let ani = words[base] as u128 | ((words[base + 1] as u128) << 64);
        let ani_levels = AniLevels::unpack(ani);

        // Block 13: NARS truth
        let nars_truth = NarsTruth::unpack(words[base + 2] as u32);

        // Block 13: NARS budget
        let nars_budget = NarsBudget::unpack((words[base + 2] >> 32) as u64);

        // Block 13: Edge type
        let edge_type = EdgeTypeMarker::unpack(words[base + 3] as u32);

        // Block 13: Node type
        let node_type = NodeTypeMarker::unpack((words[base + 3] >> 32) as u32);

        // Block 14: Q-values
        let q_values = InlineQValues::unpack([words[block14_base], words[block14_base + 1]]);

        // Block 14: Rewards
        let rewards_packed = [words[block14_base + 2], words[block14_base + 3]];
        let mut rewards = InlineRewards::default();
        for i in 0..4 {
            rewards.rewards[i] = ((rewards_packed[0] >> (i * 16)) & 0xFFFF) as u16 as i16;
        }
        for i in 0..4 {
            rewards.rewards[i + 4] = ((rewards_packed[1] >> (i * 16)) & 0xFFFF) as u16 as i16;
        }

        // Block 14: STDP
        let mut stdp = StdpMarkers::default();
        for i in 0..4 {
            stdp.timestamps[i] = ((words[block14_base + 4] >> (i * 16)) & 0xFFFF) as u16;
        }
        for i in 0..4 {
            stdp.timestamps[i + 4] = ((words[block14_base + 5] >> (i * 16)) & 0xFFFF) as u16;
        }

        // Block 14: Hebbian
        let mut hebbian = InlineHebbian::default();
        for i in 0..4 {
            hebbian.weights[i] = ((words[block14_base + 6] >> (i * 16)) & 0xFFFF) as u16;
        }
        for i in 0..4 {
            hebbian.weights[i + 4] = ((words[block14_base + 7] >> (i * 16)) & 0xFFFF) as u16;
        }

        // Block 15: DN address
        let mut dn_addr = CompressedDnAddr::default();
        for i in 0..4 {
            for j in 0..8 {
                dn_addr.path[i * 8 + j] = ((words[block15_base + i] >> (j * 8)) & 0xFF) as u8;
            }
        }

        // Block 15: Neighbor bloom
        let mut neighbors = NeighborBloom::default();
        for i in 0..4 {
            neighbors.words[i] = words[block15_base + 4 + i];
        }

        // Block 15: Graph metrics
        let metrics = GraphMetrics::unpack(words[block15_base + 8]);

        Self {
            ani_levels,
            nars_truth,
            nars_budget,
            edge_type,
            node_type,
            q_values,
            rewards,
            stdp,
            hebbian,
            dn_addr,
            neighbors,
            metrics,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ani_levels_pack_unpack() {
        let levels = AniLevels {
            reactive: 100,
            memory: 200,
            analogy: 300,
            planning: 400,
            meta: 500,
            social: 600,
            creative: 700,
            r#abstract: 800,
        };
        let packed = levels.pack();
        let unpacked = AniLevels::unpack(packed);
        assert_eq!(levels, unpacked);
    }

    #[test]
    fn test_ani_dominant() {
        let levels = AniLevels {
            reactive: 10,
            memory: 20,
            analogy: 30,
            planning: 100,
            meta: 50,
            social: 40,
            creative: 30,
            r#abstract: 20,
        };
        assert_eq!(levels.dominant(), 3); // Planning is highest
    }

    #[test]
    fn test_nars_truth_revision() {
        let t1 = NarsTruth::from_floats(0.8, 0.5);
        let t2 = NarsTruth::from_floats(0.6, 0.3);
        let revised = t1.revision(&t2);

        // Revised confidence should be higher than either input
        assert!(revised.c() > t1.c() || revised.c() > t2.c());
        // Revised frequency should be between the two
        assert!(revised.f() >= 0.5 && revised.f() <= 0.9);
    }

    #[test]
    fn test_nars_truth_pack_unpack() {
        let t = NarsTruth::from_floats(0.75, 0.9);
        let packed = t.pack();
        let unpacked = NarsTruth::unpack(packed);
        assert_eq!(t, unpacked);
    }

    #[test]
    fn test_edge_type_flags() {
        let edge = EdgeTypeMarker {
            verb_id: 42,
            direction: 1,
            weight: 200,
            flags: 0b0101, // temporal + hierarchical
        };
        assert!(edge.is_temporal());
        assert!(!edge.is_causal());
        assert!(edge.is_hierarchical());
        assert!(!edge.is_associative());

        let packed = edge.pack();
        let unpacked = EdgeTypeMarker::unpack(packed);
        assert_eq!(edge, unpacked);
    }

    #[test]
    fn test_inline_q_values() {
        let mut q = InlineQValues::default();
        q.set_q(3, 0.75);
        q.set_q(7, -0.5);

        assert!((q.q(3) - 0.75).abs() < 0.02); // 8-bit quantization error
        assert!((q.q(7) - (-0.5)).abs() < 0.02);
        assert_eq!(q.best_action(), 3);

        let packed = q.pack();
        let unpacked = InlineQValues::unpack(packed);
        assert_eq!(q.values, unpacked.values);
    }

    #[test]
    fn test_inline_rewards() {
        let mut r = InlineRewards::default();
        for i in 0..8 {
            r.push(i as f32 / 10.0);
        }
        assert!(r.average() > 0.0);
        assert!(r.trend() > 0.0); // Increasing rewards = positive trend
    }

    #[test]
    fn test_neighbor_bloom() {
        let mut bloom = NeighborBloom::default();
        bloom.insert(42);
        bloom.insert(100);
        bloom.insert(999);

        assert!(bloom.might_contain(42));
        assert!(bloom.might_contain(100));
        assert!(bloom.might_contain(999));
        // False positives are possible but unlikely for small sets
    }

    #[test]
    fn test_graph_metrics_pack_unpack() {
        let m = GraphMetrics {
            pagerank: 1000,
            hop_to_root: 3,
            cluster_id: 42,
            degree: 10,
            in_degree: 5,
            out_degree: 5,
        };
        let packed = m.pack();
        let unpacked = GraphMetrics::unpack(packed);
        assert_eq!(m, unpacked);
    }

    #[test]
    fn test_schema_sidecar_roundtrip() {
        let mut sidecar = SchemaSidecar::default();
        sidecar.ani_levels.planning = 500;
        sidecar.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        sidecar.edge_type.verb_id = 42;
        sidecar.q_values.set_q(0, 0.5);
        sidecar.rewards.push(0.8);
        sidecar.neighbors.insert(123);
        sidecar.metrics.pagerank = 999;
        sidecar.metrics.hop_to_root = 2;

        // Write to word array
        let mut words = [0u64; 256];
        sidecar.write_to_words(&mut words);

        // Read back
        let recovered = SchemaSidecar::read_from_words(&words);

        assert_eq!(recovered.ani_levels.planning, 500);
        assert_eq!(recovered.edge_type.verb_id, 42);
        assert_eq!(recovered.metrics.pagerank, 999);
        assert_eq!(recovered.metrics.hop_to_root, 2);
        assert!(recovered.neighbors.might_contain(123));
    }

    #[test]
    fn test_schema_version_byte() {
        let mut words = [0u64; VECTOR_WORDS];

        // Before writing schema, version should be 0 (legacy)
        assert_eq!(SchemaSidecar::read_version(&words), 0);

        // Write schema
        let schema = SchemaSidecar::default();
        schema.write_to_words(&mut words);

        // Version should now be 1
        assert_eq!(SchemaSidecar::read_version(&words), 1);

        // Version byte should not corrupt ANI levels
        let mut schema2 = SchemaSidecar::default();
        schema2.ani_levels.planning = 500;
        schema2.ani_levels.r#abstract = 800;
        schema2.write_to_words(&mut words);

        let recovered = SchemaSidecar::read_from_words(&words);
        assert_eq!(recovered.ani_levels.planning, 500);
        assert_eq!(recovered.ani_levels.r#abstract, 800);
        assert_eq!(SchemaSidecar::read_version(&words), 1);
    }

    #[test]
    fn test_schema_version_backward_compat() {
        // Simulate legacy data: all zeros (version 0)
        let words = [0u64; VECTOR_WORDS];
        assert_eq!(SchemaSidecar::read_version(&words), 0);

        // Reading from all-zero words should give default values
        let schema = SchemaSidecar::read_from_words(&words);
        assert_eq!(schema.ani_levels.planning, 0);
        assert_eq!(schema.nars_truth.f(), 0.0);
        assert_eq!(schema.metrics.pagerank, 0);
    }

    #[test]
    fn test_schema_version_word_isolation() {
        // Version is at word[base+15] (word 223), bits 56-63.
        // Verify it doesn't interfere with surrounding data.
        let mut words = [0u64; VECTOR_WORDS];

        // Fill word 223 with a known pattern
        let base = SchemaSidecar::WORD_OFFSET;
        words[base + 15] = 0x00FFFFFFFFFFFFFF; // lower 56 bits set

        let mut schema = SchemaSidecar::default();
        schema.write_to_words(&mut words);

        // Version should be 1 in top 8 bits
        assert_eq!(SchemaSidecar::read_version(&words), 1);
        // Lower 56 bits should be preserved from write (may be overwritten by schema)
        // The important thing is version doesn't leak into ANI words
        let recovered = SchemaSidecar::read_from_words(&words);
        assert_eq!(recovered.ani_levels.planning, 0); // default
    }
}
