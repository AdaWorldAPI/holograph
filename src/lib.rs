//! # Holograph — 3D Holographic HDR Bitpacked Vector Search
//!
//! High-performance hyperdimensional computing library with three vector widths:
//!
//! ## Vector Widths
//!
//! | Width | Words | Bits | Size | Sigma | Use Case |
//! |-------|-------|------|------|-------|----------|
//! | **10K** | 157 | 10,000 | 1.25 KB | ~56 | Legacy, compact |
//! | **16K** | 256 | 16,384 | 2 KB | 64 | Production: metadata-in-fingerprint |
//! | **32K** | 512 | 32,768 | 4 KB | 45.25/dim | 3D holographic: XYZ superposition |
//!
//! ## 3D Holographic Memory (32K)
//!
//! The 32K width decomposes into three 8K orthogonal dimensions:
//!
//! ```text
//! X (content/what):    words 0-127   — semantic identity
//! Y (context/where):   words 128-255 — situational context
//! Z (relation/how):    words 256-383 — relational structure
//! Metadata:            words 384-511 — 128 words (ANI, NARS, RL, 64 edges)
//! ```
//!
//! Product space: 8192³ ≈ 512 billion XOR-addressable data points per record.
//! 1M vectors = 4GB RAM. SIMD-clean: 16 AVX-512 iterations per dimension.
//!
//! ## Core Architecture
//!
//! - **Bitpacked vectors** — pure integer operations, no floats
//! - **Stacked Popcount** — hierarchical Hamming distance with SIMD
//! - **Vector Field Resonance** — XOR bind/unbind for O(1) retrieval
//! - **HDR Cascade** — multi-level sketch filtering before exact distance
//! - **DN Tree** — 256-way hierarchical addressing (like LDAP DNs)
//! - **144 Cognitive Verbs** — Go board topology for semantic relations
//! - **GraphBLAS Mindmap** — sparse matrix operations with tree structure
//! - **NN-Tree** — O(log n) nearest neighbor with fingerprint clustering
//! - **Epiphany Engine** — σ-threshold + centroid radius calibration
//! - **Crystal Déjà Vu** — transformer embeddings → 5D crystal → fingerprints
//! - **Déjà Vu RL** — multipass ±3σ overlay for reinforcement patterns
//! - **Truth Markers** — orthogonal superposition cleaning
//! - **DN-Sparse** — DN-addressed O(1) nodes + delta CSR + HDR fingerprints
//!
//! ## XOR Binding
//!
//! ```text
//! Bind:   A ⊗ B = A ⊕ B       (combine concepts)
//! Unbind: A ⊗ B ⊗ B = A       (recover component)
//! Bundle: majority(A, B, C)    (create prototype)
//! ```
//!
//! At 32K, this extends to 3D: `trace = X ⊕ Y ⊕ Z`. Given any two
//! dimensions, XOR recovers the third. This enables holographic probe
//! search — relational queries without graph traversal.

// === Width variants ===
pub mod width_10k;
pub mod width_16k;
pub mod width_32k;

// === Core primitives ===
pub mod bitpack;
pub mod hamming;
pub mod resonance;
pub mod hdr_cascade;

// === Graph foundations ===
pub mod dntree;
pub mod nntree;
pub mod dn_sparse;
pub mod storage_transport;

// === Encoding & representation ===
pub mod representation;
pub mod slot_encoding;

// === AI/ML extensions ===
pub mod epiphany;
pub mod crystal_dejavu;
pub mod neural_tree;
pub mod rl_ops;
pub mod sentence_crystal;

// === Navigator (partially gated) ===
pub mod navigator;

// === DataFusion-gated modules ===
#[cfg(feature = "datafusion-storage")]
pub mod graphblas;
#[cfg(feature = "datafusion-storage")]
pub mod mindmap;
#[cfg(feature = "datafusion-storage")]
pub mod storage;
#[cfg(feature = "datafusion-storage")]
pub mod query;

// === FFI (gated) ===
#[cfg(feature = "ffi")]
pub mod ffi;

// ========================================================================
// Re-exports: Core
// ========================================================================

pub use bitpack::{
    BitpackedVector, VectorRef, VectorSlice,
    VECTOR_BITS, VECTOR_WORDS, VECTOR_BYTES,
    PADDED_VECTOR_BYTES, PADDED_VECTOR_WORDS,
    xor_ref,
};
pub use hamming::{HammingEngine, StackedPopcount, hamming_distance_ref};
pub use resonance::{VectorField, Resonator, BoundEdge};
pub use hdr_cascade::{HdrCascade, MexicanHat, SearchResult};

// ========================================================================
// Re-exports: Graph
// ========================================================================

pub use dntree::{TreeAddr, DnTree, DnNode, DnEdge, CogVerb, VerbCategory};
pub use nntree::{NnTree, NnTreeConfig, SparseNnTree};
pub use dn_sparse::{
    PackedDn, DnGraph, DnNodeStore, DnCsr, DeltaDnMatrix,
    NodeSlot, EdgeDescriptor, hierarchical_fingerprint, xor_bind_fingerprint,
    DnSemiring, BooleanBfs, HdrPathBind, HammingMinPlus, PageRankSemiring, ResonanceMax,
    CascadedHammingMinPlus, CascadedResonanceMax,
};

// ========================================================================
// Re-exports: Encoding
// ========================================================================

pub use representation::{GradedVector, StackedBinary, SparseHdr};
pub use slot_encoding::{SlotEncodedNode, SlotKeys, NodeBuilder, StringEncoder};

// ========================================================================
// Re-exports: AI/ML
// ========================================================================

pub use epiphany::{EpiphanyEngine, EpiphanyZone, CentroidStats, ResonanceCalibrator};
pub use crystal_dejavu::{
    SentenceCrystal, Coord5D, CrystalCell,
    DejaVuRL, DejaVuObservation, SigmaBand,
    TruthMarker, SuperpositionCleaner, CrystalDejaVuTruth,
};
pub use neural_tree::{
    HierarchicalNeuralTree, NeuralTreeNode, NeuralTreeConfig, NeuralProfile,
    NeuralSearchResult, NeuralTreeStats, CrystalAttention, NeuralLayer, NeuralBlock,
    NUM_BLOCKS, WORDS_PER_BLOCK, BITS_PER_BLOCK,
};
pub use rl_ops::{
    RewardSignal, HebbianMatrix, PolicyGradient, RewardTracker, RlEngine, RlStats,
    SearchState, SearchAction, Intervention, Counterfactual, CausalRlAgent, CausalChainLink,
    StdpRule, PlasticityEngine,
};
pub use sentence_crystal::{
    SemanticCrystal, SemanticEncoding, LearningCell, LearningCrystal,
};

// ========================================================================
// Re-exports: Navigator
// ========================================================================

pub use navigator::{Navigator, NavResult, CypherArg, CypherYield};
#[cfg(feature = "datafusion-storage")]
pub use navigator::ZeroCopyCursor;

// ========================================================================
// Re-exports: DataFusion-gated
// ========================================================================

#[cfg(feature = "datafusion-storage")]
pub use graphblas::{GrBMatrix, GrBVector, HdrSemiring, Semiring};
#[cfg(feature = "datafusion-storage")]
pub use mindmap::{GrBMindmap, MindmapBuilder, MindmapNode, NodeType};
#[cfg(feature = "datafusion-storage")]
pub use storage::{ArrowStore, VectorBatch, ArrowBatchSearch, BatchSearchResult};

// ========================================================================
// Error types
// ========================================================================

/// Error types for holographic HDR operations
#[derive(Debug, thiserror::Error)]
pub enum HdrError {
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid vector data: {0}")]
    InvalidData(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, HdrError>;

// ========================================================================
// Configuration
// ========================================================================

/// Global configuration for the holograph engine
pub struct HdrConfig {
    /// Number of bits in vectors (default: 10000)
    pub vector_bits: usize,
    /// Enable SIMD acceleration
    pub use_simd: bool,
    /// Batch size for parallel operations
    pub batch_size: usize,
    /// Number of worker threads
    pub num_threads: usize,
}

impl Default for HdrConfig {
    fn default() -> Self {
        Self {
            vector_bits: 10000,
            use_simd: true,
            batch_size: 1024,
            num_threads: num_cpus::get().max(1),
        }
    }
}

// Inline helper for CPU count when num_cpus isn't available
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HdrConfig::default();
        assert_eq!(config.vector_bits, 10000);
        assert!(config.use_simd);
        assert!(config.num_threads > 0);
    }
}
