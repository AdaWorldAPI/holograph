//! 128-Word Metadata Schema for 3D Holographic Vectors
//!
//! With 128 words (8,192 bits = 1KB) of metadata, there is room for
//! everything at full precision: ANI, NARS, RL, qualia (full 18D),
//! 7-layer markers, 64 inline edges, graph metrics, GEL, kernel state.

use super::*;

// ============================================================================
// METADATA WORD OFFSETS (relative to META_START = 384)
// ============================================================================

/// ANI / Consciousness: 4 words (256 bits)
pub const M_ANI_BASE: usize = 0;       // level(u8) + mask(u8) + activation(u16) + L1-L4(4×u8)
pub const M_ANI_EXT: usize = 1;        // L5-L7(3×u8) + cycle(u16) + flags(u8) + tau(u8)
pub const M_NARS_TRUTH: usize = 2;     // freq(u16) + conf(u16) + pos_ev(u16) + neg_ev(u16)
pub const M_NARS_EXT: usize = 3;       // horizon(u16) + expectation(u16) + reserved(u32)

/// Qualia: 2 words — top 8 of 18 channels at u16 precision
pub const M_QUALIA_A: usize = 4;       // valence, arousal, dominance, novelty
pub const M_QUALIA_B: usize = 5;       // certainty, urgency, depth, salience

/// GEL + Kernel: 2 words
pub const M_GEL: usize = 6;            // pc(u16) + stack(u8) + flags(u8) + verb(u8) + phase(u8) + reserved(u16)
pub const M_KERNEL: usize = 7;         // integration(u16) + mode(u8) + epoch(u8) + reserved(u32)

/// DN Tree: 3 words
pub const M_DN_PARENT: usize = 8;      // parent(u16) + depth(u8) + rung(u8) + sigma(u8) + type(u8) + flags(u16)
pub const M_DN_META: usize = 9;        // label_hash(u32) + access_count(u16) + ttl(u16)
pub const M_DN_TIME: usize = 10;       // created(u32) + last_access_delta(u16) + reserved(u16)

/// Inline edges: 16 words = 64 edges (4 packed per word)
pub const M_EDGE_START: usize = 13;    // edges 0-3
pub const M_EDGE_END: usize = 29;      // edges 60-63
pub const M_EDGE_WORDS: usize = 16;
pub const M_EDGES_PER_WORD: usize = 4;
pub const M_MAX_INLINE_EDGES: usize = M_EDGE_WORDS * M_EDGES_PER_WORD; // 64

/// Edge overflow metadata: 2 words
pub const M_EDGE_OVERFLOW: usize = 29; // count(u8) + flag(u8) + table_addr(u16) + version(u16) + reserved(u16)
pub const M_EDGE_DEGREE: usize = 30;   // in_deg(u16) + out_deg(u16) + bidi(u16) + reserved(u16)

/// Schema version
pub const M_VERSION: usize = 31;       // version at bits[56-63], dim_flags at bits[48-55]

/// RL / Decision: 8 words
pub const M_RL_BASE: usize = 32;       // Q-values, rewards, TD error, policy (words 32-39)

/// Bloom filter: 8 words = 512-bit bloom (better FP rate than 256-bit)
pub const M_BLOOM_BASE: usize = 40;    // words 40-47

/// Graph metrics: 16 words (full precision)
pub const M_GRAPH_BASE: usize = 48;    // words 48-63

/// Qualia overflow: full 18D at f32 (9 words)
pub const M_QUALIA_FULL: usize = 64;   // words 64-72 (18 × f32 / 8 bytes per word = 9 words)

/// 7-Layer markers: 16 words (full LayerMarker state)
pub const M_LAYER_BASE: usize = 80;    // words 80-95

/// Rung history: 16 words (condensed shift events)
pub const M_RUNG_HISTORY: usize = 96;  // words 96-111

/// Dimensional flags: which XYZ dimensions are populated
pub const M_DIM_FLAGS: usize = 112;    // word 112: x_active(u8) + y_active(u8) + z_active(u8) + reserved

/// Reserved: words 113-126
pub const M_RESERVED_START: usize = 113;

/// Checksum + version flags: last word
pub const M_CHECKSUM: usize = 127;     // checksum(u32) + reserved(u24) + version_flags(u8)

// ============================================================================
// SCHEMA SIDECAR
// ============================================================================

/// Schema metadata packed in the 128-word metadata block.
#[derive(Clone, Debug, Default)]
pub struct HoloSchema {
    // ANI / Consciousness
    pub ani_level: u8,
    pub layer_mask: u8,
    pub peak_activation: u16,
    pub layer_confidence: [u8; 7],
    pub cycle: u16,
    pub consciousness_flags: u8,
    pub tau: u8,

    // NARS
    pub nars_frequency: u16,
    pub nars_confidence: u16,
    pub nars_pos_evidence: u16,
    pub nars_neg_evidence: u16,

    // Qualia (top 8 channels, u16 quantized)
    pub qualia: [u16; 8],

    // DN Tree
    pub parent_addr: u16,
    pub depth: u8,
    pub rung: u8,
    pub sigma: u8,
    pub node_type: u8,
    pub flags: u16,
    pub label_hash: u32,
    pub access_count: u16,

    // Edge counts
    pub inline_edge_count: u8,
    pub overflow_flag: u8,
    pub in_degree: u16,
    pub out_degree: u16,

    // Version
    pub schema_version: u8,
    pub dim_flags: u8,
}

impl HoloSchema {
    /// Read schema from a metadata word slice (128 words starting at META_START).
    pub fn read_from_meta(meta: &[u64]) -> Self {
        if meta.len() < META_WORDS {
            return Self::default();
        }

        let w0 = meta[M_ANI_BASE];
        let w1 = meta[M_ANI_EXT];
        let w2 = meta[M_NARS_TRUTH];
        let w4 = meta[M_QUALIA_A];
        let w5 = meta[M_QUALIA_B];
        let w8 = meta[M_DN_PARENT];
        let w9 = meta[M_DN_META];
        let w29 = meta[M_EDGE_OVERFLOW];
        let w30 = meta[M_EDGE_DEGREE];
        let w31 = meta[M_VERSION];

        Self {
            ani_level: (w0 & 0xFF) as u8,
            layer_mask: ((w0 >> 8) & 0xFF) as u8,
            peak_activation: ((w0 >> 16) & 0xFFFF) as u16,
            layer_confidence: [
                ((w0 >> 32) & 0xFF) as u8,
                ((w0 >> 40) & 0xFF) as u8,
                ((w0 >> 48) & 0xFF) as u8,
                ((w0 >> 56) & 0xFF) as u8,
                ((w1) & 0xFF) as u8,
                ((w1 >> 8) & 0xFF) as u8,
                ((w1 >> 16) & 0xFF) as u8,
            ],
            cycle: ((w1 >> 24) & 0xFFFF) as u16,
            consciousness_flags: ((w1 >> 40) & 0xFF) as u8,
            tau: ((w1 >> 48) & 0xFF) as u8,

            nars_frequency: (w2 & 0xFFFF) as u16,
            nars_confidence: ((w2 >> 16) & 0xFFFF) as u16,
            nars_pos_evidence: ((w2 >> 32) & 0xFFFF) as u16,
            nars_neg_evidence: ((w2 >> 48) & 0xFFFF) as u16,

            qualia: [
                (w4 & 0xFFFF) as u16,
                ((w4 >> 16) & 0xFFFF) as u16,
                ((w4 >> 32) & 0xFFFF) as u16,
                ((w4 >> 48) & 0xFFFF) as u16,
                (w5 & 0xFFFF) as u16,
                ((w5 >> 16) & 0xFFFF) as u16,
                ((w5 >> 32) & 0xFFFF) as u16,
                ((w5 >> 48) & 0xFFFF) as u16,
            ],

            parent_addr: (w8 & 0xFFFF) as u16,
            depth: ((w8 >> 16) & 0xFF) as u8,
            rung: ((w8 >> 24) & 0xFF) as u8,
            sigma: ((w8 >> 32) & 0xFF) as u8,
            node_type: ((w8 >> 40) & 0xFF) as u8,
            flags: ((w8 >> 48) & 0xFFFF) as u16,
            label_hash: (w9 & 0xFFFF_FFFF) as u32,
            access_count: ((w9 >> 32) & 0xFFFF) as u16,

            inline_edge_count: (w29 & 0xFF) as u8,
            overflow_flag: ((w29 >> 8) & 0xFF) as u8,
            in_degree: (w30 & 0xFFFF) as u16,
            out_degree: ((w30 >> 16) & 0xFFFF) as u16,

            schema_version: ((w31 >> 56) & 0xFF) as u8,
            dim_flags: ((w31 >> 48) & 0xFF) as u8,
        }
    }

    /// Write schema to metadata words.
    pub fn write_to_meta(&self, meta: &mut [u64]) {
        if meta.len() < META_WORDS {
            return;
        }

        // ANI base
        meta[M_ANI_BASE] = (self.ani_level as u64)
            | ((self.layer_mask as u64) << 8)
            | ((self.peak_activation as u64) << 16)
            | ((self.layer_confidence[0] as u64) << 32)
            | ((self.layer_confidence[1] as u64) << 40)
            | ((self.layer_confidence[2] as u64) << 48)
            | ((self.layer_confidence[3] as u64) << 56);

        // ANI ext
        meta[M_ANI_EXT] = (self.layer_confidence[4] as u64)
            | ((self.layer_confidence[5] as u64) << 8)
            | ((self.layer_confidence[6] as u64) << 16)
            | ((self.cycle as u64) << 24)
            | ((self.consciousness_flags as u64) << 40)
            | ((self.tau as u64) << 48);

        // NARS truth
        meta[M_NARS_TRUTH] = (self.nars_frequency as u64)
            | ((self.nars_confidence as u64) << 16)
            | ((self.nars_pos_evidence as u64) << 32)
            | ((self.nars_neg_evidence as u64) << 48);

        // Qualia
        meta[M_QUALIA_A] = (self.qualia[0] as u64)
            | ((self.qualia[1] as u64) << 16)
            | ((self.qualia[2] as u64) << 32)
            | ((self.qualia[3] as u64) << 48);
        meta[M_QUALIA_B] = (self.qualia[4] as u64)
            | ((self.qualia[5] as u64) << 16)
            | ((self.qualia[6] as u64) << 32)
            | ((self.qualia[7] as u64) << 48);

        // DN tree
        meta[M_DN_PARENT] = (self.parent_addr as u64)
            | ((self.depth as u64) << 16)
            | ((self.rung as u64) << 24)
            | ((self.sigma as u64) << 32)
            | ((self.node_type as u64) << 40)
            | ((self.flags as u64) << 48);
        meta[M_DN_META] = (self.label_hash as u64)
            | ((self.access_count as u64) << 32);

        // Edge overflow
        meta[M_EDGE_OVERFLOW] = (self.inline_edge_count as u64)
            | ((self.overflow_flag as u64) << 8);
        meta[M_EDGE_DEGREE] = (self.in_degree as u64)
            | ((self.out_degree as u64) << 16);

        // Version (preserve other bits in the word)
        meta[M_VERSION] = (meta[M_VERSION] & 0x0000_FFFF_FFFF_FFFF)
            | ((self.dim_flags as u64) << 48)
            | ((self.schema_version as u64) << 56);
    }

    /// Read version byte from metadata.
    pub fn read_version(meta: &[u64]) -> u8 {
        if meta.len() <= M_VERSION {
            return 0;
        }
        ((meta[M_VERSION] >> 56) & 0xFF) as u8
    }

    /// Pack an edge: verb(u8) + target(u8) = 16 bits.
    pub fn pack_edge(verb: u8, target: u8) -> u16 {
        ((verb as u16) << 8) | (target as u16)
    }

    /// Unpack an edge from 16 bits.
    pub fn unpack_edge(packed: u16) -> (u8, u8) {
        ((packed >> 8) as u8, (packed & 0xFF) as u8)
    }

    /// Read inline edges from metadata.
    pub fn read_edges(meta: &[u64]) -> Vec<(u8, u8)> {
        if meta.len() < M_EDGE_END {
            return Vec::new();
        }
        let count = (meta[M_EDGE_OVERFLOW] & 0xFF) as usize;
        let count = count.min(M_MAX_INLINE_EDGES);
        let mut edges = Vec::with_capacity(count);

        for edge_idx in 0..count {
            let word_idx = M_EDGE_START + edge_idx / M_EDGES_PER_WORD;
            let slot = edge_idx % M_EDGES_PER_WORD;
            let packed = ((meta[word_idx] >> (slot * 16)) & 0xFFFF) as u16;
            if packed != 0 {
                edges.push(Self::unpack_edge(packed));
            }
        }

        edges
    }

    /// Write an edge at a specific slot index.
    pub fn write_edge(meta: &mut [u64], edge_idx: usize, verb: u8, target: u8) {
        if edge_idx >= M_MAX_INLINE_EDGES {
            return;
        }
        let word_idx = M_EDGE_START + edge_idx / M_EDGES_PER_WORD;
        let slot = edge_idx % M_EDGES_PER_WORD;
        let shift = slot * 16;
        let mask = !(0xFFFFu64 << shift);
        let packed = Self::pack_edge(verb, target) as u64;
        meta[word_idx] = (meta[word_idx] & mask) | (packed << shift);
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_roundtrip() {
        let mut meta = [0u64; META_WORDS];
        let schema = HoloSchema {
            ani_level: 7,
            layer_mask: 0b0111_1111,
            peak_activation: 42000,
            layer_confidence: [200, 180, 160, 140, 120, 100, 80],
            cycle: 12345,
            consciousness_flags: 0xAB,
            tau: 200,
            nars_frequency: 45000,
            nars_confidence: 60000,
            nars_pos_evidence: 100,
            nars_neg_evidence: 50,
            qualia: [10000, 20000, 30000, 40000, 50000, 60000, 5000, 15000],
            parent_addr: 0x8042,
            depth: 3,
            rung: 5,
            sigma: 7,
            node_type: 2,
            flags: 0x0301,
            label_hash: 0xDEADBEEF,
            access_count: 999,
            inline_edge_count: 8,
            overflow_flag: 0,
            in_degree: 12,
            out_degree: 8,
            schema_version: 2,
            dim_flags: 0b0000_0111, // X, Y, Z all active
        };

        schema.write_to_meta(&mut meta);
        let recovered = HoloSchema::read_from_meta(&meta);

        assert_eq!(recovered.ani_level, 7);
        assert_eq!(recovered.layer_mask, 0b0111_1111);
        assert_eq!(recovered.peak_activation, 42000);
        assert_eq!(recovered.layer_confidence, [200, 180, 160, 140, 120, 100, 80]);
        assert_eq!(recovered.cycle, 12345);
        assert_eq!(recovered.consciousness_flags, 0xAB);
        assert_eq!(recovered.tau, 200);
        assert_eq!(recovered.nars_frequency, 45000);
        assert_eq!(recovered.nars_confidence, 60000);
        assert_eq!(recovered.nars_pos_evidence, 100);
        assert_eq!(recovered.nars_neg_evidence, 50);
        assert_eq!(recovered.qualia, [10000, 20000, 30000, 40000, 50000, 60000, 5000, 15000]);
        assert_eq!(recovered.parent_addr, 0x8042);
        assert_eq!(recovered.depth, 3);
        assert_eq!(recovered.rung, 5);
        assert_eq!(recovered.sigma, 7);
        assert_eq!(recovered.node_type, 2);
        assert_eq!(recovered.flags, 0x0301);
        assert_eq!(recovered.label_hash, 0xDEADBEEF);
        assert_eq!(recovered.access_count, 999);
        assert_eq!(recovered.inline_edge_count, 8);
        assert_eq!(recovered.overflow_flag, 0);
        assert_eq!(recovered.in_degree, 12);
        assert_eq!(recovered.out_degree, 8);
        assert_eq!(recovered.schema_version, 2);
        assert_eq!(recovered.dim_flags, 0b0000_0111);
    }

    #[test]
    fn test_version_byte() {
        let mut meta = [0u64; META_WORDS];
        assert_eq!(HoloSchema::read_version(&meta), 0);

        let mut s = HoloSchema::default();
        s.schema_version = 42;
        s.write_to_meta(&mut meta);
        assert_eq!(HoloSchema::read_version(&meta), 42);
    }

    #[test]
    fn test_edge_pack_unpack() {
        let packed = HoloSchema::pack_edge(0x07, 0x42);
        assert_eq!(packed, 0x0742);
        let (verb, target) = HoloSchema::unpack_edge(packed);
        assert_eq!(verb, 0x07);
        assert_eq!(target, 0x42);
    }

    #[test]
    fn test_inline_edges_roundtrip() {
        let mut meta = [0u64; META_WORDS];

        // Write 8 edges
        for i in 0..8 {
            HoloSchema::write_edge(&mut meta, i, (i + 1) as u8, (0x80 + i) as u8);
        }
        // Set edge count
        meta[M_EDGE_OVERFLOW] = (meta[M_EDGE_OVERFLOW] & !0xFF) | 8;

        let edges = HoloSchema::read_edges(&meta);
        assert_eq!(edges.len(), 8);
        for i in 0..8 {
            assert_eq!(edges[i], ((i + 1) as u8, (0x80 + i) as u8),
                "Edge {} mismatch", i);
        }
    }

    #[test]
    fn test_64_inline_edges() {
        let mut meta = [0u64; META_WORDS];

        // Fill all 64 edge slots
        for i in 0..M_MAX_INLINE_EDGES {
            let verb = ((i % 144) + 1) as u8; // 1-144 (Verb range)
            let target = (i % 256) as u8;
            HoloSchema::write_edge(&mut meta, i, verb, target);
        }
        meta[M_EDGE_OVERFLOW] = (meta[M_EDGE_OVERFLOW] & !0xFF) | 64;

        let edges = HoloSchema::read_edges(&meta);
        assert_eq!(edges.len(), 64);

        // Verify first and last
        assert_eq!(edges[0], (1, 0));
        assert_eq!(edges[63], (64, 63));
    }

    #[test]
    fn test_metadata_does_not_overlap_dimensions() {
        // Verify schema writes don't touch dimension words
        let mut v = [0u64; VECTOR_WORDS];

        // Fill dimensions with pattern
        for i in 0..Z_END {
            v[i] = 0xAAAA_BBBB_CCCC_DDDD;
        }

        // Write schema to metadata region
        let mut schema = HoloSchema::default();
        schema.ani_level = 255;
        schema.nars_frequency = 65535;
        schema.write_to_meta(&mut v[META_START..]);

        // Verify dimensions untouched
        for i in 0..Z_END {
            assert_eq!(v[i], 0xAAAA_BBBB_CCCC_DDDD,
                "Dimension word {} corrupted by schema write", i);
        }
    }

    #[test]
    fn test_schema_version_word_isolation() {
        let mut meta = [0u64; META_WORDS];
        // Pre-fill version word with data
        meta[M_VERSION] = 0x0000_1234_5678_9ABC;

        let mut s = HoloSchema::default();
        s.schema_version = 0xFE;
        s.dim_flags = 0x07;
        s.write_to_meta(&mut meta);

        // Version and dim_flags written in top 16 bits
        assert_eq!((meta[M_VERSION] >> 56) & 0xFF, 0xFE);
        assert_eq!((meta[M_VERSION] >> 48) & 0xFF, 0x07);
        // Lower 48 bits preserved
        assert_eq!(meta[M_VERSION] & 0x0000_FFFF_FFFF_FFFF, 0x0000_1234_5678_9ABC);
    }
}
