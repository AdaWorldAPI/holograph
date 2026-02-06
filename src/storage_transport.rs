//! Storage and Transport Formats
//!
//! Two optimized layouts:
//! - **Storage**: Full fidelity, optimized for random access
//! - **Transport**: Compressed, optimized for bandwidth
//!
//! # Storage Format: 32:32:64:128 + Semantic
//!
//! ```text
//! ┌──────┬──────┬────────┬─────────┬─────────────────────┐
//! │ ID   │FLAGS │DN ADDR │META     │ SEMANTIC            │
//! │32bit │32bit │ 64bit  │128 bit  │ 1024-10000 bits     │
//! └──────┴──────┴────────┴─────────┴─────────────────────┘
//!    4B     4B      8B       16B      128-1250 bytes
//! ```
//!
//! # Transport Format: 8:8:48 + XOR Delta
//!
//! ```text
//! ┌────────┬────────┬────────────┬─────────────────────┐
//! │MSG TYPE│VERSION │ ROUTING    │ XOR DELTA PAYLOAD   │
//! │ 8 bit  │ 8 bit  │  48 bits   │ Sparse (10-20%)     │
//! └────────┴────────┴────────────┴─────────────────────┘
//!    1B       1B        6B          Variable
//! ```

use std::io::{Read, Write, Result as IoResult};

// ============================================================================
// STORAGE FORMAT
// ============================================================================

/// Storage header: 32:32:64:128 = 256 bits = 32 bytes
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct StorageHeader {
    /// Unique node/edge ID
    pub id: u32,
    /// Flags and type info
    pub flags: StorageFlags,
    /// DN tree address (depth + branches)
    pub dn_addr: u64,
    /// Metadata block (active items, edge info, etc.)
    pub meta: MetaBlock128,
}

impl StorageHeader {
    pub const BYTES: usize = 32;

    pub fn to_bytes(&self) -> [u8; 32] {
        unsafe { std::mem::transmute_copy(self) }
    }

    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        unsafe { std::mem::transmute_copy(bytes) }
    }
}

/// 32-bit flags
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StorageFlags {
    /// Node type (0-255)
    pub node_type: u8,
    /// Abstraction rung (0-255)
    pub rung: u8,
    /// Semantic tier: 0=none, 1=1024, 2=4096, 3=10000
    pub semantic_tier: u8,
    /// Boolean flags
    pub bits: u8,
}

impl StorageFlags {
    pub const ACTIVE: u8      = 0b0000_0001;
    pub const VERIFIED: u8    = 0b0000_0010;
    pub const LOCKED: u8      = 0b0000_0100;
    pub const COMPRESSED: u8  = 0b0000_1000;
    pub const HAS_EDGE: u8    = 0b0001_0000;
    pub const HAS_CHILDREN: u8= 0b0010_0000;
    pub const TOMBSTONE: u8   = 0b1000_0000;

    pub fn is_active(&self) -> bool { self.bits & Self::ACTIVE != 0 }
    pub fn is_compressed(&self) -> bool { self.bits & Self::COMPRESSED != 0 }
    pub fn has_edge(&self) -> bool { self.bits & Self::HAS_EDGE != 0 }
}

/// 128-bit metadata block
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MetaBlock128 {
    /// Active items (8 × 8-bit indices) or bitfield
    pub active: u64,
    /// Edge info: verb(8) + weight(16) + source(20) + target(20)
    pub edge: u64,
}

impl MetaBlock128 {
    /// Get active items as indices (8 × 8-bit)
    pub fn active_indices(&self) -> [u8; 8] {
        self.active.to_le_bytes()
    }

    /// Set active items
    pub fn set_active_indices(&mut self, items: &[u8; 8]) {
        self.active = u64::from_le_bytes(*items);
    }

    /// Get verb (0-143)
    pub fn verb(&self) -> u8 {
        (self.edge & 0xFF) as u8
    }

    /// Get weight (0-65535 → 0.0-1.0)
    pub fn weight(&self) -> f32 {
        ((self.edge >> 8) & 0xFFFF) as f32 / 65535.0
    }

    /// Get source reference (20 bits → 0-1048575)
    pub fn source(&self) -> u32 {
        ((self.edge >> 24) & 0xFFFFF) as u32
    }

    /// Get target reference (20 bits)
    pub fn target(&self) -> u32 {
        ((self.edge >> 44) & 0xFFFFF) as u32
    }

    /// Pack edge info
    pub fn pack_edge(verb: u8, weight: f32, source: u32, target: u32) -> u64 {
        let w = (weight.clamp(0.0, 1.0) * 65535.0) as u64;
        let s = (source & 0xFFFFF) as u64;
        let t = (target & 0xFFFFF) as u64;
        (verb as u64) | (w << 8) | (s << 24) | (t << 44)
    }
}

/// Semantic tiers for storage
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SemanticTier {
    /// No semantic data (metadata only)
    None = 0,
    /// 1024 bits = 128 bytes (from 1024D transformer)
    Tier1K = 1,
    /// 4096 bits = 512 bytes (rich semantic)
    Tier4K = 2,
    /// 10000 bits = 1250 bytes (full HDR)
    Tier10K = 3,
}

impl SemanticTier {
    pub fn bits(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Tier1K => 1024,
            Self::Tier4K => 4096,
            Self::Tier10K => 10000,
        }
    }

    pub fn bytes(&self) -> usize {
        (self.bits() + 7) / 8
    }

    pub fn words(&self) -> usize {
        (self.bits() + 63) / 64
    }
}

/// Complete storage record
pub struct StorageRecord {
    pub header: StorageHeader,
    pub semantic: Vec<u64>,
}

impl StorageRecord {
    /// Create with header only (no semantic)
    pub fn metadata_only(header: StorageHeader) -> Self {
        Self {
            header,
            semantic: Vec::new(),
        }
    }

    /// Create with 1024-bit semantic
    pub fn with_1k(header: StorageHeader, semantic: [u64; 16]) -> Self {
        Self {
            header,
            semantic: semantic.to_vec(),
        }
    }

    /// Create with 10K-bit semantic
    pub fn with_10k(header: StorageHeader, semantic: [u64; 157]) -> Self {
        Self {
            header,
            semantic: semantic.to_vec(),
        }
    }

    /// Total size in bytes
    pub fn size(&self) -> usize {
        StorageHeader::BYTES + self.semantic.len() * 8
    }

    /// Write to bytes
    pub fn write_to<W: Write>(&self, w: &mut W) -> IoResult<usize> {
        let header_bytes = self.header.to_bytes();
        w.write_all(&header_bytes)?;

        for &word in &self.semantic {
            w.write_all(&word.to_le_bytes())?;
        }

        Ok(self.size())
    }

    /// Read from bytes
    pub fn read_from<R: Read>(r: &mut R, tier: SemanticTier) -> IoResult<Self> {
        let mut header_bytes = [0u8; 32];
        r.read_exact(&mut header_bytes)?;
        let header = StorageHeader::from_bytes(&header_bytes);

        let words = tier.words();
        let mut semantic = vec![0u64; words];
        for word in &mut semantic {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            *word = u64::from_le_bytes(buf);
        }

        Ok(Self { header, semantic })
    }
}

// ============================================================================
// TRANSPORT FORMAT
// ============================================================================

/// Transport header: 8:8:48 = 64 bits = 8 bytes
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct TransportHeader {
    /// Message type
    pub msg_type: MessageType,
    /// Version and compression flags
    pub version: VersionFlags,
    /// Routing info (DN prefix)
    pub routing: [u8; 6],
}

impl TransportHeader {
    pub const BYTES: usize = 8;

    /// Get DN prefix (depth + 5 branches)
    pub fn dn_prefix(&self) -> (u8, [u8; 5]) {
        let depth = self.routing[0];
        let mut branches = [0u8; 5];
        branches.copy_from_slice(&self.routing[1..6]);
        (depth, branches)
    }

    /// Set DN prefix
    pub fn set_dn_prefix(&mut self, depth: u8, branches: &[u8]) {
        self.routing[0] = depth;
        for (i, &b) in branches.iter().take(5).enumerate() {
            self.routing[i + 1] = b;
        }
    }
}

/// Message types for transport
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageType {
    /// Query request
    Query = 0x01,
    /// Query response
    Response = 0x02,
    /// Sync/replicate node
    Sync = 0x03,
    /// Delta update
    Delta = 0x04,
    /// Batch of messages
    Batch = 0x05,
    /// Heartbeat/ping
    Ping = 0x06,
    /// Error
    Error = 0xFF,
}

/// Version and compression flags
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VersionFlags {
    /// Protocol version (0-15) + compression type (0-15)
    bits: u8,
}

impl VersionFlags {
    pub fn new(version: u8, compression: CompressionType) -> Self {
        Self {
            bits: (version & 0x0F) | ((compression as u8) << 4),
        }
    }

    pub fn version(&self) -> u8 {
        self.bits & 0x0F
    }

    pub fn compression(&self) -> CompressionType {
        match self.bits >> 4 {
            0 => CompressionType::None,
            1 => CompressionType::XorDelta,
            2 => CompressionType::Sparse,
            3 => CompressionType::RunLength,
            _ => CompressionType::None,
        }
    }
}

/// Compression types for transport
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression (full fingerprint)
    None = 0,
    /// XOR delta from base
    XorDelta = 1,
    /// Sparse (only active bit indices)
    Sparse = 2,
    /// Run-length encoded
    RunLength = 3,
}

/// XOR delta payload
#[derive(Clone, Debug)]
pub struct XorDeltaPayload {
    /// Base fingerprint ID (reference)
    pub base_id: u32,
    /// Hamming distance to base (for validation)
    pub distance: u16,
    /// Changed bit indices (sparse XOR)
    pub changed_bits: Vec<u16>,
}

impl XorDeltaPayload {
    /// Encode XOR delta from full fingerprints
    pub fn encode(base: &[u64], target: &[u64]) -> Self {
        let mut changed_bits = Vec::new();
        let mut distance = 0u16;

        for (word_idx, (&b, &t)) in base.iter().zip(target.iter()).enumerate() {
            let xor = b ^ t;
            distance += xor.count_ones() as u16;

            // Record changed bit positions
            let mut diff = xor;
            while diff != 0 {
                let bit_pos = diff.trailing_zeros() as u16;
                let global_bit = (word_idx as u16 * 64) + bit_pos;
                changed_bits.push(global_bit);
                diff &= diff - 1; // Clear lowest bit
            }
        }

        Self {
            base_id: 0, // Set by caller
            distance,
            changed_bits,
        }
    }

    /// Decode: apply delta to base
    pub fn decode(&self, base: &[u64]) -> Vec<u64> {
        let mut result = base.to_vec();

        for &bit in &self.changed_bits {
            let word = bit as usize / 64;
            let pos = bit % 64;
            if word < result.len() {
                result[word] ^= 1 << pos;
            }
        }

        result
    }

    /// Compressed size in bytes
    pub fn size(&self) -> usize {
        4 + 2 + 2 + self.changed_bits.len() * 2
    }

    /// Compression ratio vs full fingerprint
    pub fn compression_ratio(&self, full_bits: usize) -> f32 {
        let full_bytes = (full_bits + 7) / 8;
        self.size() as f32 / full_bytes as f32
    }
}

/// Sparse payload (k-hot encoding)
#[derive(Clone, Debug)]
pub struct SparsePayload {
    /// Total dimensionality
    pub dims: u16,
    /// Active bit indices
    pub active: Vec<u16>,
}

impl SparsePayload {
    /// Encode sparse from dense
    pub fn encode(dense: &[u64], max_bits: usize) -> Self {
        let mut active = Vec::new();

        for (word_idx, &word) in dense.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit_pos = w.trailing_zeros() as u16;
                let global_bit = (word_idx as u16 * 64) + bit_pos;
                if (global_bit as usize) < max_bits {
                    active.push(global_bit);
                }
                w &= w - 1;
            }
        }

        Self {
            dims: max_bits as u16,
            active,
        }
    }

    /// Decode to dense
    pub fn decode(&self) -> Vec<u64> {
        let words = (self.dims as usize + 63) / 64;
        let mut result = vec![0u64; words];

        for &bit in &self.active {
            let word = bit as usize / 64;
            let pos = bit % 64;
            if word < result.len() {
                result[word] |= 1 << pos;
            }
        }

        result
    }

    /// Density
    pub fn density(&self) -> f32 {
        self.active.len() as f32 / self.dims as f32
    }
}

/// Complete transport message
#[derive(Clone, Debug)]
pub struct TransportMessage {
    pub header: TransportHeader,
    pub payload: TransportPayload,
}

/// Transport payload variants
#[derive(Clone, Debug)]
pub enum TransportPayload {
    /// Full fingerprint (no compression)
    Full(Vec<u64>),
    /// XOR delta from base
    Delta(XorDeltaPayload),
    /// Sparse encoding
    Sparse(SparsePayload),
    /// Query (just routing, no fingerprint)
    Query { k: u16, threshold: u32 },
    /// Error message
    Error(String),
}

impl TransportMessage {
    /// Estimate wire size
    pub fn wire_size(&self) -> usize {
        TransportHeader::BYTES + match &self.payload {
            TransportPayload::Full(words) => words.len() * 8,
            TransportPayload::Delta(d) => d.size(),
            TransportPayload::Sparse(s) => 2 + s.active.len() * 2,
            TransportPayload::Query { .. } => 6,
            TransportPayload::Error(s) => 2 + s.len(),
        }
    }
}

// ============================================================================
// GRPC / PROTOBUF SCHEMA (for reference)
// ============================================================================

/// Protobuf-style schema for transport
pub const PROTO_SCHEMA: &str = r#"
syntax = "proto3";

package ladybug.transport;

// Transport envelope
message Envelope {
  MessageType type = 1;
  uint32 version = 2;
  bytes routing = 3;  // 6 bytes: DN prefix
  oneof payload {
    FullFingerprint full = 4;
    XorDelta delta = 5;
    SparseFingerprint sparse = 6;
    QueryRequest query = 7;
    QueryResponse response = 8;
    ErrorInfo error = 9;
  }
}

enum MessageType {
  UNKNOWN = 0;
  QUERY = 1;
  RESPONSE = 2;
  SYNC = 3;
  DELTA = 4;
  BATCH = 5;
  PING = 6;
  ERROR = 255;
}

// Full fingerprint (uncompressed)
message FullFingerprint {
  bytes data = 1;  // 128-1250 bytes depending on tier
  uint32 tier = 2; // 1=1K, 2=4K, 3=10K bits
}

// XOR delta encoding
message XorDelta {
  uint32 base_id = 1;      // Reference fingerprint
  uint32 distance = 2;     // Hamming distance (validation)
  repeated uint32 bits = 3; // Changed bit indices
}

// Sparse encoding
message SparseFingerprint {
  uint32 dims = 1;          // Total dimensions
  repeated uint32 active = 2; // Active bit indices
}

// Query request
message QueryRequest {
  bytes fingerprint = 1;    // Query fingerprint
  uint32 k = 2;             // Number of results
  uint32 threshold = 3;     // Max hamming distance
  bytes dn_filter = 4;      // Optional DN prefix filter
}

// Query response
message QueryResponse {
  repeated Result results = 1;
  uint32 total_scanned = 2;
  uint32 time_us = 3;
}

message Result {
  uint32 id = 1;
  uint32 distance = 2;
  bytes metadata = 3;
}

// Error info
message ErrorInfo {
  uint32 code = 1;
  string message = 2;
}
"#;

// ============================================================================
// COMPRESSION DECISION
// ============================================================================

/// Decide best compression for a fingerprint
pub fn choose_compression(
    fingerprint: &[u64],
    base: Option<&[u64]>,
    total_bits: usize,
) -> CompressionType {
    let density = {
        let ones: u32 = fingerprint.iter().map(|w| w.count_ones()).sum();
        ones as f32 / total_bits as f32
    };

    // If very sparse (<10% ones), use sparse encoding
    if density < 0.1 {
        return CompressionType::Sparse;
    }

    // If we have a base and similarity is high, use XOR delta
    if let Some(base) = base {
        let distance: u32 = fingerprint.iter()
            .zip(base.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        let similarity = 1.0 - (distance as f32 / total_bits as f32);

        if similarity > 0.8 {
            // >80% similar = XOR delta is ~20% the size
            return CompressionType::XorDelta;
        }
    }

    // Default: no compression
    CompressionType::None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_header() {
        let header = StorageHeader {
            id: 12345,
            flags: StorageFlags {
                node_type: 1,
                rung: 5,
                semantic_tier: SemanticTier::Tier10K as u8,
                bits: StorageFlags::ACTIVE | StorageFlags::HAS_EDGE,
            },
            dn_addr: 0x0301_0203_0405_0607, // depth=3, branches=[1,2,3,4,5,6,7]
            meta: MetaBlock128 {
                active: 0x0102030405060708,
                edge: MetaBlock128::pack_edge(24, 0.75, 100, 200),
            },
        };

        let bytes = header.to_bytes();
        let restored = StorageHeader::from_bytes(&bytes);

        let restored_id = { restored.id };
        let restored_rung = { restored.flags.rung };
        assert_eq!(restored_id, 12345);
        assert_eq!(restored_rung, 5);
    }

    #[test]
    fn test_xor_delta() {
        let base = [0xFFFF_0000_FFFF_0000u64; 16];
        let mut target = base;
        target[0] ^= 0x0000_00FF; // Change 8 bits

        let delta = XorDeltaPayload::encode(&base, &target);

        assert_eq!(delta.distance, 8);
        assert_eq!(delta.changed_bits.len(), 8);

        let decoded = delta.decode(&base);
        assert_eq!(decoded, target.to_vec());

        println!("Compression ratio: {:.2}%",
            delta.compression_ratio(1024) * 100.0);
    }

    #[test]
    fn test_sparse_encoding() {
        // Create sparse fingerprint (10% density)
        let mut dense = vec![0u64; 16];
        dense[0] = 0x0000_00FF;  // 8 bits
        dense[8] = 0xFF00_0000;  // 8 bits

        let sparse = SparsePayload::encode(&dense, 1024);

        assert_eq!(sparse.active.len(), 16);
        assert!(sparse.density() < 0.02);

        let decoded = sparse.decode();
        assert_eq!(decoded[0], dense[0]);
        assert_eq!(decoded[8], dense[8]);
    }

    #[test]
    fn test_compression_decision() {
        let full = vec![0xFFFF_FFFF_FFFF_FFFFu64; 16];
        let sparse = vec![0x0000_0000_0000_00FFu64; 16];
        let similar = {
            let mut s = full.clone();
            s[0] ^= 0xFF;
            s
        };

        assert_eq!(choose_compression(&full, None, 1024), CompressionType::None);
        assert_eq!(choose_compression(&sparse, None, 1024), CompressionType::Sparse);
        assert_eq!(choose_compression(&similar, Some(&full), 1024), CompressionType::XorDelta);
    }
}
