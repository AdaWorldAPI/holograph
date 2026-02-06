//! Content-Addressable Memory (CAM) Header & Container System
//!
//! Every stored record gets a fixed 512-byte header prefix (64 words):
//!
//! ```text
//!  ┌──────────┬───────────────┬─────────────────────────┐
//!  │ 32 meta  │ 32 fingerprint│  N × 128 content        │
//!  │ offset 0 │ offset 256B   │  offset 512B            │
//!  └──────────┴───────────────┴─────────────────────────┘
//!  ← HEADER: always 512 bytes →← variable quanta       →
//! ```
//!
//! ## Container Types (1 quantum = 128 words = 8,192 bits)
//!
//! ```text
//! MONO:   32 + 32 + 128       = 192 words   1.50 KB   (1 quantum)
//! DENSE:  32 + 32 + 128 + 128 = 320 words   2.50 KB   (2 quanta)
//! HOLO:   32 + 32 + 128×3     = 448 words   3.50 KB   (3 quanta)
//! ```
//!
//! ## Two-Phase Search
//!
//! The 32-word fingerprint (2048 bits) acts as a coarse-grained sketch:
//!
//! ```text
//! Level -1: CAM Fingerprint (32 words, ~32 cycles)
//!   → sketch_distance < threshold → promote to Level 0
//!   → Rejects ~95% of candidates
//!
//! Level 0+: Existing HDR cascade on content region
//! ```
//!
//! ## O(1) Dedup
//!
//! Same content → same fingerprint → hash table lookup → skip insert.
//! No full Hamming comparison needed for dedup.

use crate::bitpack::{BitpackedVector, VECTOR_WORDS};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Words in the metadata block
pub const META_WORDS: usize = 32;

/// Words in the fingerprint block
pub const FINGERPRINT_WORDS: usize = 32;

/// Total header words (meta + fingerprint)
pub const HEADER_WORDS: usize = META_WORDS + FINGERPRINT_WORDS; // 64

/// Header size in bytes
pub const HEADER_BYTES: usize = HEADER_WORDS * 8; // 512

/// Fingerprint bits (2048)
pub const FINGERPRINT_BITS: usize = FINGERPRINT_WORDS * 64; // 2048

/// One content quantum: 128 words = 8,192 bits
pub const QUANTUM_WORDS: usize = 128;

/// Quantum bits
pub const QUANTUM_BITS: usize = QUANTUM_WORDS * 64; // 8192

/// Quantum bytes
pub const QUANTUM_BYTES: usize = QUANTUM_WORDS * 8; // 1024

// --- Container total sizes ---

/// MONO: header + 1 quantum = 192 words
pub const MONO_WORDS: usize = HEADER_WORDS + QUANTUM_WORDS; // 192

/// DENSE: header + 2 quanta = 320 words
pub const DENSE_WORDS: usize = HEADER_WORDS + 2 * QUANTUM_WORDS; // 320

/// HOLO: header + 3 quanta = 448 words
pub const HOLO_WORDS: usize = HEADER_WORDS + 3 * QUANTUM_WORDS; // 448

/// MONO bytes for Arrow FixedSizeBinary
pub const MONO_BYTES: usize = MONO_WORDS * 8; // 1536

/// DENSE bytes
pub const DENSE_BYTES: usize = DENSE_WORDS * 8; // 2560

/// HOLO bytes
pub const HOLO_BYTES: usize = HOLO_WORDS * 8; // 3584

// --- Fingerprint statistics ---

/// Expected Hamming distance between two random fingerprints: n/2
pub const FP_EXPECTED_DISTANCE: f64 = FINGERPRINT_BITS as f64 / 2.0; // 1024.0

/// Standard deviation: sqrt(n/4)
pub const FP_SIGMA: f64 = 22.627416997969522; // sqrt(2048/4) = sqrt(512)

/// Integer sigma (rounded)
pub const FP_SIGMA_APPROX: u32 = 23;

// --- Meta word layout ---

/// Word 0: container kind (bits 0-3), width variant (bits 4-7), version (bits 8-15),
///         flags (bits 16-31), reserved (bits 32-63)
pub const M_DISCRIMINANT: usize = 0;

/// Word 1: DN tree node type (bits 0-7), rung (bits 8-15), depth (bits 16-23),
///         sigma (bits 24-31), reserved (bits 32-63)
pub const M_DN_TYPE: usize = 1;

/// Word 2: DN parent address (bits 0-31), label hash (bits 32-63)
pub const M_DN_ADDR: usize = 2;

/// Word 3: created_epoch (bits 0-31), last_access_delta (bits 32-47),
///         access_count (bits 48-63)
pub const M_TIMESTAMPS: usize = 3;

/// Word 4: TTL (bits 0-15), priority (bits 16-31), reserved (bits 32-63)
pub const M_LIFECYCLE: usize = 4;

/// Words 5-6: ANI consciousness level, layer mask, activation, L1-L7
pub const M_ANI_BASE: usize = 5;

/// Words 7-8: NARS truth (freq, conf, pos_ev, neg_ev, horizon, expectation)
pub const M_NARS_BASE: usize = 7;

/// Words 9-12: Qualia (top-8 channels at u16 each = 2 words, extended 2 words)
pub const M_QUALIA_BASE: usize = 9;

/// Words 13-20: Inline edge slots (32 edges, 4 packed per word)
pub const M_EDGE_BASE: usize = 13;
pub const M_EDGE_WORDS: usize = 8;
pub const M_MAX_INLINE_EDGES: usize = M_EDGE_WORDS * 4; // 32

/// Words 21-24: Graph metrics (pagerank, degree, clustering, betweenness)
pub const M_GRAPH_BASE: usize = 21;

/// Words 25-28: RL state (Q-values, rewards, TD error)
pub const M_RL_BASE: usize = 25;

/// Words 29-30: Bloom filter (128-bit mini-bloom for neighbor adjacency)
pub const M_BLOOM_BASE: usize = 29;

/// Word 31: Checksum (bits 0-31), schema version (bits 32-39), reserved (bits 40-63)
pub const M_CHECKSUM: usize = 31;

// ============================================================================
// CONTAINER KIND
// ============================================================================

/// Discriminant for container types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ContainerKind {
    /// 1 quantum (128 words): pure XOR Hamming content
    Mono = 1,
    /// 2 quanta: XOR content (128) + int8 dense embedding (128 words = 1024 × i8)
    Dense = 2,
    /// 3 quanta: X (128) + Y (128) + Z (128) holographic dimensions
    Holo = 3,
}

/// Width variant tag
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WidthVariant {
    /// 10K-bit legacy (157 words, fits in 128 + 29 padding)
    W10K = 0,
    /// 16K-bit production (256 words = 2 quanta)
    W16K = 1,
    /// 32K-bit holographic (512 words = 4 quanta)
    W32K = 2,
}

impl ContainerKind {
    /// Number of content quanta for this container type
    pub const fn quanta(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Dense => 2,
            Self::Holo => 3,
        }
    }

    /// Total content words (excluding header)
    pub const fn content_words(self) -> usize {
        self.quanta() * QUANTUM_WORDS
    }

    /// Total record words (header + content)
    pub const fn total_words(self) -> usize {
        HEADER_WORDS + self.content_words()
    }

    /// Total record bytes (for Arrow FixedSizeBinary)
    pub const fn total_bytes(self) -> usize {
        self.total_words() * 8
    }

    /// From discriminant byte
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Mono),
            2 => Some(Self::Dense),
            3 => Some(Self::Holo),
            _ => None,
        }
    }
}

// ============================================================================
// CAM HEADER
// ============================================================================

/// The 64-word (512-byte) CAM header.
///
/// Lives at offset 0 of every record. The first 32 words are metadata,
/// the next 32 words are a content-addressable fingerprint sketch.
#[derive(Clone)]
#[repr(align(64))]
pub struct CamHeader {
    /// Metadata block (words 0-31)
    pub meta: [u64; META_WORDS],
    /// Fingerprint sketch (words 32-63)
    pub fingerprint: [u64; FINGERPRINT_WORDS],
}

impl Default for CamHeader {
    fn default() -> Self {
        Self {
            meta: [0u64; META_WORDS],
            fingerprint: [0u64; FINGERPRINT_WORDS],
        }
    }
}

impl CamHeader {
    /// Create a zero header
    pub fn zero() -> Self {
        Self::default()
    }

    // ====================================================================
    // DISCRIMINANT ACCESS
    // ====================================================================

    /// Read container kind from meta word 0 (bits 0-3)
    pub fn container_kind(&self) -> Option<ContainerKind> {
        ContainerKind::from_u8((self.meta[M_DISCRIMINANT] & 0x0F) as u8)
    }

    /// Set container kind in meta word 0
    pub fn set_container_kind(&mut self, kind: ContainerKind) {
        self.meta[M_DISCRIMINANT] =
            (self.meta[M_DISCRIMINANT] & !0x0F) | (kind as u64);
    }

    /// Read width variant from meta word 0 (bits 4-7)
    pub fn width_variant(&self) -> Option<WidthVariant> {
        match ((self.meta[M_DISCRIMINANT] >> 4) & 0x0F) as u8 {
            0 => Some(WidthVariant::W10K),
            1 => Some(WidthVariant::W16K),
            2 => Some(WidthVariant::W32K),
            _ => None,
        }
    }

    /// Set width variant
    pub fn set_width_variant(&mut self, w: WidthVariant) {
        self.meta[M_DISCRIMINANT] =
            (self.meta[M_DISCRIMINANT] & !0xF0) | ((w as u64) << 4);
    }

    /// Read version from meta word 0 (bits 8-15)
    pub fn version(&self) -> u8 {
        ((self.meta[M_DISCRIMINANT] >> 8) & 0xFF) as u8
    }

    /// Set version
    pub fn set_version(&mut self, v: u8) {
        self.meta[M_DISCRIMINANT] =
            (self.meta[M_DISCRIMINANT] & !(0xFF << 8)) | ((v as u64) << 8);
    }

    // ====================================================================
    // FINGERPRINT OPS
    // ====================================================================

    /// Exact fingerprint match (O(1) dedup check).
    ///
    /// If true, the two records are (almost certainly) content-identical.
    /// False positives are astronomically unlikely with 2048-bit sketches.
    #[inline]
    pub fn fingerprint_match(&self, other: &CamHeader) -> bool {
        self.fingerprint == other.fingerprint
    }

    /// Hamming distance on the 2048-bit fingerprint.
    ///
    /// This is the "Level -1" pre-filter: fast, fixed-size, and at a
    /// known offset in every record.
    #[inline]
    pub fn fingerprint_distance(&self, other: &CamHeader) -> u32 {
        fingerprint_distance(&self.fingerprint, &other.fingerprint)
    }

    /// Fingerprint similarity as f64 in [0.0, 1.0]
    pub fn fingerprint_similarity(&self, other: &CamHeader) -> f64 {
        let d = self.fingerprint_distance(other) as f64;
        1.0 - (d / FINGERPRINT_BITS as f64)
    }

    /// Is the fingerprint all zeros? (empty/uninitialized record)
    pub fn fingerprint_is_empty(&self) -> bool {
        self.fingerprint.iter().all(|&w| w == 0)
    }

    /// Fingerprint popcount (density)
    pub fn fingerprint_popcount(&self) -> u32 {
        self.fingerprint.iter().map(|w| w.count_ones()).sum()
    }

    // ====================================================================
    // DN TREE ACCESS
    // ====================================================================

    /// Read rung level from meta word 1 (bits 8-15)
    pub fn rung(&self) -> u8 {
        ((self.meta[M_DN_TYPE] >> 8) & 0xFF) as u8
    }

    /// Set rung level
    pub fn set_rung(&mut self, rung: u8) {
        self.meta[M_DN_TYPE] =
            (self.meta[M_DN_TYPE] & !(0xFF << 8)) | ((rung as u64) << 8);
    }

    /// Read depth from meta word 1 (bits 16-23)
    pub fn depth(&self) -> u8 {
        ((self.meta[M_DN_TYPE] >> 16) & 0xFF) as u8
    }

    /// Set depth
    pub fn set_depth(&mut self, depth: u8) {
        self.meta[M_DN_TYPE] =
            (self.meta[M_DN_TYPE] & !(0xFF << 16)) | ((depth as u64) << 16);
    }

    // ====================================================================
    // SERIALIZATION
    // ====================================================================

    /// Write header to a word slice at offset 0.
    pub fn write_to(&self, buf: &mut [u64]) {
        debug_assert!(buf.len() >= HEADER_WORDS);
        buf[..META_WORDS].copy_from_slice(&self.meta);
        buf[META_WORDS..HEADER_WORDS].copy_from_slice(&self.fingerprint);
    }

    /// Read header from a word slice at offset 0.
    pub fn read_from(buf: &[u64]) -> Self {
        debug_assert!(buf.len() >= HEADER_WORDS);
        let mut h = Self::zero();
        h.meta.copy_from_slice(&buf[..META_WORDS]);
        h.fingerprint.copy_from_slice(&buf[META_WORDS..HEADER_WORDS]);
        h
    }

    /// As contiguous byte slice (512 bytes).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.meta.as_ptr() as *const u8,
                HEADER_BYTES,
            )
        }
    }
}

// ============================================================================
// FINGERPRINT GENERATION
// ============================================================================

/// Generate a 2048-bit CAM fingerprint by XOR-folding content words.
///
/// For a 10K vector (157 words), folds 157 → 32 words.
/// For a 16K vector (256 words), folds 256 → 32 words (exact 8:1 ratio).
/// For a 32K vector (512 words), folds 512 → 32 words (exact 16:1 ratio).
///
/// XOR-folding preserves Hamming properties: if two source vectors
/// are close in Hamming space, their fingerprints are close (with
/// high probability). The converse allows false positives, which is
/// acceptable for a pre-filter.
pub fn xor_fold_fingerprint(content: &[u64]) -> [u64; FINGERPRINT_WORDS] {
    let mut fp = [0u64; FINGERPRINT_WORDS];
    for (i, &word) in content.iter().enumerate() {
        fp[i % FINGERPRINT_WORDS] ^= word;
    }
    fp
}

/// Generate fingerprint from a 10K BitpackedVector.
pub fn fingerprint_from_10k(v: &BitpackedVector) -> [u64; FINGERPRINT_WORDS] {
    xor_fold_fingerprint(v.words())
}

/// Hamming distance between two fingerprints (32 words = 2048 bits).
///
/// ~32 cycles on scalar, ~4 AVX-512 iterations.
#[inline]
pub fn fingerprint_distance(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> u32 {
    let mut total = 0u32;
    for i in 0..FINGERPRINT_WORDS {
        total += (a[i] ^ b[i]).count_ones();
    }
    total
}

/// Fingerprint similarity in [0.0, 1.0].
pub fn fingerprint_similarity(a: &[u64; FINGERPRINT_WORDS], b: &[u64; FINGERPRINT_WORDS]) -> f64 {
    1.0 - (fingerprint_distance(a, b) as f64 / FINGERPRINT_BITS as f64)
}

// ============================================================================
// CAM RECORD
// ============================================================================

/// A complete CAM record: header + variable-length content.
///
/// The content region starts at word 64 and is always a multiple of
/// 128 words (one or more quanta).
#[derive(Clone)]
pub struct CamRecord {
    /// The 64-word header (meta + fingerprint)
    pub header: CamHeader,
    /// Content words (N × 128 words, where N = container quanta)
    pub content: Vec<u64>,
}

impl CamRecord {
    /// Create a MONO record from a 10K BitpackedVector.
    ///
    /// The 157-word vector is placed into a 128-word quantum:
    /// words 0-127 are copied, words 128-156 are XOR-folded back
    /// into the first 29 words to preserve signal in the quantum.
    ///
    /// Alternatively, zero-pad to 128 words (losing 29 words = 1,856 bits).
    /// The `fold` parameter controls this choice.
    pub fn from_10k(v: &BitpackedVector, fold_overflow: bool) -> Self {
        let src = v.words();
        let mut content = [0u64; QUANTUM_WORDS];

        // Copy first 128 words
        let copy_len = QUANTUM_WORDS.min(VECTOR_WORDS);
        content[..copy_len].copy_from_slice(&src[..copy_len]);

        if fold_overflow && VECTOR_WORDS > QUANTUM_WORDS {
            // XOR-fold the remaining 29 words back into the quantum
            for i in QUANTUM_WORDS..VECTOR_WORDS {
                content[i - QUANTUM_WORDS] ^= src[i];
            }
        }

        let fingerprint = xor_fold_fingerprint(&content);

        let mut header = CamHeader::zero();
        header.set_container_kind(ContainerKind::Mono);
        header.set_width_variant(WidthVariant::W10K);
        header.set_version(1);
        header.fingerprint = fingerprint;

        Self {
            header,
            content: content.to_vec(),
        }
    }

    /// Create a MONO record from a 16K word array (256 words → 2 quanta,
    /// but stored as MONO if only semantic content, no dense embedding).
    pub fn mono_from_16k(words: &[u64; crate::width_16k::VECTOR_WORDS]) -> Self {
        // 256 words = 2 quanta of content
        let fingerprint = xor_fold_fingerprint(words.as_slice());

        let mut header = CamHeader::zero();
        header.set_container_kind(ContainerKind::Dense); // 2 quanta
        header.set_width_variant(WidthVariant::W16K);
        header.set_version(1);
        header.fingerprint = fingerprint;

        Self {
            header,
            content: words.to_vec(),
        }
    }

    /// Container kind
    pub fn kind(&self) -> Option<ContainerKind> {
        self.header.container_kind()
    }

    /// Total words in the serialized record
    pub fn total_words(&self) -> usize {
        HEADER_WORDS + self.content.len()
    }

    /// Serialize to a contiguous word buffer.
    pub fn serialize(&self) -> Vec<u64> {
        let mut buf = Vec::with_capacity(self.total_words());
        buf.extend_from_slice(&self.header.meta);
        buf.extend_from_slice(&self.header.fingerprint);
        buf.extend_from_slice(&self.content);
        buf
    }

    /// Deserialize from a contiguous word buffer.
    pub fn deserialize(buf: &[u64]) -> Option<Self> {
        if buf.len() < HEADER_WORDS {
            return None;
        }
        let header = CamHeader::read_from(buf);
        let content = buf[HEADER_WORDS..].to_vec();
        Some(Self { header, content })
    }

    /// Get a reference to the content region (for Hamming operations).
    pub fn content_slice(&self) -> &[u64] {
        &self.content
    }

    /// Regenerate fingerprint from current content.
    /// Call after modifying content to keep the fingerprint consistent.
    pub fn refresh_fingerprint(&mut self) {
        self.header.fingerprint = xor_fold_fingerprint(&self.content);
    }
}

// ============================================================================
// CAM INDEX (O(1) DEDUP)
// ============================================================================

/// A simple CAM index for O(1) content-addressable dedup.
///
/// Maps fingerprint → record offset. For exact dedup, check fingerprint
/// equality. For approximate dedup, threshold on fingerprint distance.
pub struct CamIndex {
    /// Fingerprint hash → record position
    entries: std::collections::HashMap<[u64; FINGERPRINT_WORDS], usize>,
}

impl CamIndex {
    /// Create empty index
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    /// Create with capacity hint
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(cap),
        }
    }

    /// Insert a record. Returns `true` if new, `false` if duplicate.
    pub fn insert(&mut self, fingerprint: [u64; FINGERPRINT_WORDS], offset: usize) -> bool {
        self.entries.insert(fingerprint, offset).is_none()
    }

    /// Exact lookup by fingerprint
    pub fn lookup(&self, fingerprint: &[u64; FINGERPRINT_WORDS]) -> Option<usize> {
        self.entries.get(fingerprint).copied()
    }

    /// Check if content already exists (O(1) dedup)
    pub fn contains(&self, fingerprint: &[u64; FINGERPRINT_WORDS]) -> bool {
        self.entries.contains_key(fingerprint)
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build from a slice of CamRecords
    pub fn build(records: &[CamRecord]) -> Self {
        let mut idx = Self::with_capacity(records.len());
        for (i, rec) in records.iter().enumerate() {
            idx.insert(rec.header.fingerprint, i);
        }
        idx
    }
}

impl Default for CamIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CAM PRE-FILTER FOR HDR CASCADE
// ============================================================================

/// Result from the CAM pre-filter phase.
#[derive(Clone, Debug)]
pub struct CamCandidate {
    /// Index into the record store
    pub offset: usize,
    /// Fingerprint Hamming distance to query
    pub sketch_distance: u32,
}

/// CAM pre-filter: scan fingerprints to produce candidates for the HDR cascade.
///
/// This is "Level -1" in the search pipeline. For each record, we compare
/// only the 32-word fingerprint (at a fixed offset) against the query
/// fingerprint. Records below the threshold are promoted to Level 0.
///
/// # Performance
///
/// 32 words per candidate vs 157+ for full Hamming. ~5× throughput
/// improvement on the scan loop for 10K vectors. For 1M vectors,
/// phase 0 touches 32M words instead of 157M words.
pub fn cam_prefilter(
    query_fp: &[u64; FINGERPRINT_WORDS],
    store_fps: &[[u64; FINGERPRINT_WORDS]],
    threshold: u32,
    max_candidates: usize,
) -> Vec<CamCandidate> {
    let mut candidates: Vec<CamCandidate> = store_fps
        .iter()
        .enumerate()
        .filter_map(|(i, fp)| {
            let d = fingerprint_distance(query_fp, fp);
            if d <= threshold {
                Some(CamCandidate {
                    offset: i,
                    sketch_distance: d,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by sketch distance ascending
    candidates.sort_unstable_by_key(|c| c.sketch_distance);

    // Truncate to max candidates
    candidates.truncate(max_candidates);
    candidates
}

/// Suggest a pre-filter threshold based on a target similarity.
///
/// For 2048-bit fingerprints, the expected distance for `sim` similarity is:
/// `d = (1 - sim) * 2048`
///
/// Add margin of 2σ ≈ 45 for safety.
pub fn suggest_threshold(target_similarity: f64) -> u32 {
    let base = ((1.0 - target_similarity) * FINGERPRINT_BITS as f64) as u32;
    base.saturating_add(2 * FP_SIGMA_APPROX)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitpack::BitpackedVector;

    #[test]
    fn test_constants() {
        assert_eq!(META_WORDS, 32);
        assert_eq!(FINGERPRINT_WORDS, 32);
        assert_eq!(HEADER_WORDS, 64);
        assert_eq!(HEADER_BYTES, 512);
        assert_eq!(FINGERPRINT_BITS, 2048);
        assert_eq!(QUANTUM_WORDS, 128);
        assert_eq!(QUANTUM_BITS, 8192);
        assert_eq!(MONO_WORDS, 192);
        assert_eq!(DENSE_WORDS, 320);
        assert_eq!(HOLO_WORDS, 448);
        assert_eq!(MONO_BYTES, 1536);
        assert_eq!(DENSE_BYTES, 2560);
        assert_eq!(HOLO_BYTES, 3584);
    }

    #[test]
    fn test_container_kind() {
        assert_eq!(ContainerKind::Mono.quanta(), 1);
        assert_eq!(ContainerKind::Dense.quanta(), 2);
        assert_eq!(ContainerKind::Holo.quanta(), 3);
        assert_eq!(ContainerKind::Mono.total_words(), MONO_WORDS);
        assert_eq!(ContainerKind::Dense.total_words(), DENSE_WORDS);
        assert_eq!(ContainerKind::Holo.total_words(), HOLO_WORDS);
    }

    #[test]
    fn test_header_discriminant_roundtrip() {
        let mut h = CamHeader::zero();
        h.set_container_kind(ContainerKind::Dense);
        h.set_width_variant(WidthVariant::W16K);
        h.set_version(42);

        assert_eq!(h.container_kind(), Some(ContainerKind::Dense));
        assert_eq!(h.width_variant(), Some(WidthVariant::W16K));
        assert_eq!(h.version(), 42);
    }

    #[test]
    fn test_header_dn_tree_roundtrip() {
        let mut h = CamHeader::zero();
        h.set_rung(4);
        h.set_depth(7);
        assert_eq!(h.rung(), 4);
        assert_eq!(h.depth(), 7);
    }

    #[test]
    fn test_header_serialize_roundtrip() {
        let mut h = CamHeader::zero();
        h.set_container_kind(ContainerKind::Holo);
        h.set_version(3);
        h.fingerprint[0] = 0xDEADBEEFCAFEBABE;
        h.fingerprint[31] = 0x1234567890ABCDEF;
        h.meta[M_CHECKSUM] = 0xFFFFFFFF;

        let mut buf = [0u64; HEADER_WORDS];
        h.write_to(&mut buf);
        let h2 = CamHeader::read_from(&buf);

        assert_eq!(h2.container_kind(), Some(ContainerKind::Holo));
        assert_eq!(h2.version(), 3);
        assert_eq!(h2.fingerprint[0], 0xDEADBEEFCAFEBABE);
        assert_eq!(h2.fingerprint[31], 0x1234567890ABCDEF);
        assert_eq!(h2.meta[M_CHECKSUM], 0xFFFFFFFF);
    }

    #[test]
    fn test_fingerprint_self_distance() {
        let v = BitpackedVector::random(42);
        let fp = fingerprint_from_10k(&v);
        assert_eq!(fingerprint_distance(&fp, &fp), 0);
    }

    #[test]
    fn test_fingerprint_match_identical() {
        let v = BitpackedVector::random(42);
        let r1 = CamRecord::from_10k(&v, false);
        let r2 = CamRecord::from_10k(&v, false);
        assert!(r1.header.fingerprint_match(&r2.header));
    }

    #[test]
    fn test_fingerprint_different() {
        let a = BitpackedVector::random(1);
        let b = BitpackedVector::random(2);
        let ra = CamRecord::from_10k(&a, false);
        let rb = CamRecord::from_10k(&b, false);
        assert!(!ra.header.fingerprint_match(&rb.header));
        assert!(ra.header.fingerprint_distance(&rb.header) > 0);
    }

    #[test]
    fn test_cam_record_serialize_roundtrip() {
        let v = BitpackedVector::random(42);
        let rec = CamRecord::from_10k(&v, false);

        let buf = rec.serialize();
        assert_eq!(buf.len(), HEADER_WORDS + QUANTUM_WORDS);

        let rec2 = CamRecord::deserialize(&buf).unwrap();
        assert_eq!(rec2.header.container_kind(), Some(ContainerKind::Mono));
        assert!(rec2.header.fingerprint_match(&rec.header));
        assert_eq!(rec2.content, rec.content);
    }

    #[test]
    fn test_cam_index_dedup() {
        let a = BitpackedVector::random(1);
        let b = BitpackedVector::random(2);

        let ra = CamRecord::from_10k(&a, false);
        let rb = CamRecord::from_10k(&b, false);
        let ra_dup = CamRecord::from_10k(&a, false);

        // Verify fingerprints are actually identical
        assert_eq!(ra.header.fingerprint, ra_dup.header.fingerprint,
            "Same input must produce same fingerprint");

        let mut idx = CamIndex::new();
        assert!(idx.insert(ra.header.fingerprint, 0));  // new → true
        assert!(idx.insert(rb.header.fingerprint, 1));  // new → true
        assert!(!idx.insert(ra_dup.header.fingerprint, 2)); // dup → false

        assert_eq!(idx.len(), 2); // still 2 entries
        // contains should find both
        assert!(idx.contains(&ra.header.fingerprint));
        assert!(idx.contains(&rb.header.fingerprint));
    }

    #[test]
    fn test_cam_prefilter() {
        // Create 100 random vectors
        let records: Vec<CamRecord> = (0..100)
            .map(|i| CamRecord::from_10k(&BitpackedVector::random(i), false))
            .collect();

        let fingerprints: Vec<[u64; FINGERPRINT_WORDS]> = records
            .iter()
            .map(|r| r.header.fingerprint)
            .collect();

        // Query for record 42's fingerprint
        let query_fp = &fingerprints[42];

        // Threshold at 0 → exact match only
        let exact = cam_prefilter(query_fp, &fingerprints, 0, 10);
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].offset, 42);
        assert_eq!(exact[0].sketch_distance, 0);

        // Threshold at 2048 → everything matches
        let all = cam_prefilter(query_fp, &fingerprints, 2048, 200);
        assert_eq!(all.len(), 100);
    }

    #[test]
    fn test_suggest_threshold() {
        // 95% similarity → expect ~102 + 46 = ~148
        let t95 = suggest_threshold(0.95);
        assert!(t95 > 100 && t95 < 200);

        // 80% similarity → expect ~410 + 46 = ~456
        let t80 = suggest_threshold(0.80);
        assert!(t80 > 400 && t80 < 500);
    }

    #[test]
    fn test_fold_overflow_preserves_more_signal() {
        let v = BitpackedVector::random(42);

        let no_fold = CamRecord::from_10k(&v, false);
        let with_fold = CamRecord::from_10k(&v, true);

        // Content should differ when overflow words are non-zero
        // (they almost certainly are for a random vector)
        let overflow_nonzero = v.words()[QUANTUM_WORDS..]
            .iter()
            .any(|&w| w != 0);

        if overflow_nonzero {
            assert_ne!(no_fold.content, with_fold.content);
        }

        // Fingerprints should also differ
        if overflow_nonzero {
            assert_ne!(no_fold.header.fingerprint, with_fold.header.fingerprint);
        }
    }

    #[test]
    fn test_fingerprint_statistics() {
        // Random fingerprints should be ~1024 apart (n/2)
        let mut total_dist = 0u64;
        let trials = 100;
        for i in 0..trials {
            let a = BitpackedVector::random(i * 2);
            let b = BitpackedVector::random(i * 2 + 1);
            let fa = fingerprint_from_10k(&a);
            let fb = fingerprint_from_10k(&b);
            total_dist += fingerprint_distance(&fa, &fb) as u64;
        }
        let avg = total_dist as f64 / trials as f64;
        // Should be near 1024 ± ~3σ (≈ 68)
        assert!(
            avg > 900.0 && avg < 1150.0,
            "Average fingerprint distance {avg} should be near 1024"
        );
    }
}
