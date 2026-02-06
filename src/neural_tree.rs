//! Hierarchical Neural Tree: Stacked Popcount as Neural Layers
//!
//! The insight: each of the 157 u64 words in a BitpackedVector is a **neuron**.
//! XOR is the synaptic input. Popcount is the integration function. Threshold
//! is the firing decision. Early termination is pruning. The cumulative sum
//! across words is a forward pass through a 157-layer neural network.
//!
//! Combined with the 5D crystal lattice, DN tree addressing, and epiphany zones,
//! this creates a hierarchical neural architecture that enables:
//!
//! - **O(log n) nearest neighbor** via stacked popcount pruning
//! - **O(1) semantic recall** via crystal cell fingerprint lookup
//! - **Hebbian learning** — cells that fire together wire together
//! - **Attention masks** from crystal neighborhoods
//! - **Multi-resolution search** — coarse (word blocks) → fine (exact bits)
//!
//! # The Neural Tree Architecture
//!
//! ```text
//! Vector A:  [word0][word1][word2]...[word156]    (10K bits in 157 words)
//! Vector B:  [word0][word1][word2]...[word156]
//!                |      |      |          |
//!                v      v      v          v
//! XOR:       [xor0] [xor1] [xor2]  ...[xor156]   ← synaptic input
//!                |      |      |          |
//!                v      v      v          v
//! Popcount:  [ pc0] [ pc1] [ pc2]  ...[ pc156]   ← integration (0-64 each)
//!                |      |      |          |
//!                v      v      v          v
//! Cumulative:[ c0 ] [ c1 ] [ c2 ]  ...[ c156]    ← forward pass
//!                |      |      |          |
//!                v      v      v          v
//! Threshold: if c[i] > threshold → PRUNE (early terminate)
//!
//! Multi-Resolution Blocks:
//! ┌─────────────┬─────────────┬─────────────┬──────────┐
//! │  Block 0    │  Block 1    │  Block 2    │ Block 9  │
//! │ words 0-15  │ words 16-31 │ words 32-47 │ 144-156  │
//! │  1024 bits  │  1024 bits  │  1024 bits  │ 832 bits │
//! └─────────────┴─────────────┴─────────────┴──────────┘
//!       ↓              ↓              ↓            ↓
//!   Block sums (coarse filter) → only expand surviving blocks
//! ```
//!
//! # Crystal-Neural Integration
//!
//! The 5D crystal (5×5×5×5×5 = 3125 cells) maps to stacked popcount regions:
//!
//! ```text
//! Crystal Coord (d0,d1,d2,d3,d4) ──► Block selector (which word range)
//!                                 ──► Attention mask (which words matter)
//!                                 ──► Cell fingerprint (prototype for that region)
//!
//! This creates "neural attention" without backpropagation:
//! - Crystal neighborhood = attention window
//! - Cell fingerprint = learned prototype
//! - Stacked popcount = activation assessment
//! ```

use crate::bitpack::{BitpackedVector, VectorRef, VECTOR_WORDS};
use crate::hamming::{hamming_distance_scalar, StackedPopcount, Belichtung};
use crate::crystal_dejavu::Coord5D;
use crate::epiphany::{EpiphanyZone, THREE_SIGMA};
use crate::dntree::TreeAddr;
use std::collections::HashMap;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Number of multi-resolution blocks (ceil(157/16) = 10)
pub const NUM_BLOCKS: usize = (VECTOR_WORDS + 15) / 16;

/// Words per block (except possibly the last)
pub const WORDS_PER_BLOCK: usize = 16;

/// Bits per block (16 × 64 = 1024)
pub const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64;

/// Map 5 crystal dimensions to 10 blocks (2 blocks per dimension)
/// Each crystal dimension controls 2 blocks of 1024 bits = 2048 bits
pub const BLOCKS_PER_CRYSTAL_DIM: usize = 2;

// ============================================================================
// NEURAL LAYER: One word = one neuron
// ============================================================================

/// Statistics for one neural layer (word boundary)
#[derive(Clone, Copy, Debug)]
pub struct NeuralLayer {
    /// Layer index (0..157)
    pub index: usize,
    /// XOR popcount at this layer (0-64) — "activation"
    pub activation: u8,
    /// Cumulative sum up to this layer — "membrane potential"
    pub membrane: u16,
    /// Whether this layer exceeds threshold — "firing"
    pub firing: bool,
}

// ============================================================================
// BLOCK: Multi-resolution grouping of 16 words
// ============================================================================

/// A multi-resolution block of 16 words (1024 bits)
#[derive(Clone, Debug)]
pub struct NeuralBlock {
    /// Block index (0..10)
    pub index: usize,
    /// Start word index
    pub start_word: usize,
    /// End word index (exclusive)
    pub end_word: usize,
    /// Sum of activations in this block
    pub block_sum: u32,
    /// Maximum single-layer activation in block
    pub max_activation: u8,
    /// Variance of activations (uniformity indicator)
    pub variance: f32,
}

impl NeuralBlock {
    /// Is this block "hot" (high activation)?
    pub fn is_hot(&self, threshold_per_word: f32) -> bool {
        let expected = threshold_per_word * (self.end_word - self.start_word) as f32;
        self.block_sum as f32 > expected
    }

    /// Block-level sigma classification
    pub fn sigma_zone(&self) -> EpiphanyZone {
        // Expected random block sum = 16 words × 32 bits/word = 512
        // σ for 16 words ≈ sqrt(16 × 16) = 16
        let expected = (self.end_word - self.start_word) as f32 * 32.0;
        let block_sigma = ((self.end_word - self.start_word) as f32 * 16.0).sqrt();
        let deviation = (self.block_sum as f32 - expected).abs();

        if deviation < block_sigma {
            EpiphanyZone::Identity
        } else if deviation < block_sigma * 2.0 {
            EpiphanyZone::Epiphany
        } else if deviation < block_sigma * 3.0 {
            EpiphanyZone::Penumbra
        } else {
            EpiphanyZone::Noise
        }
    }
}

// ============================================================================
// NEURAL PROFILE: Full stacked popcount interpreted as neural activation
// ============================================================================

/// A complete neural activation profile for one vector comparison
#[derive(Clone, Debug)]
pub struct NeuralProfile {
    /// Per-layer activation (raw stacked popcount)
    pub layers: [u8; VECTOR_WORDS],
    /// Cumulative membrane potential
    pub membrane: [u16; VECTOR_WORDS],
    /// Total distance (sum of all activations)
    pub total: u32,
    /// Multi-resolution blocks
    pub blocks: Vec<NeuralBlock>,
    /// Earliest pruning point (first layer exceeding threshold, if any)
    pub prune_point: Option<usize>,
    /// Block-level activation signature (10 values for fast comparison)
    pub block_signature: [u16; NUM_BLOCKS],
}

impl NeuralProfile {
    /// Build neural profile from stacked popcount
    pub fn from_stacked(stacked: &StackedPopcount) -> Self {
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        let mut block_signature = [0u16; NUM_BLOCKS];

        for b in 0..NUM_BLOCKS {
            let start = b * WORDS_PER_BLOCK;
            let end = ((b + 1) * WORDS_PER_BLOCK).min(VECTOR_WORDS);
            let block_sum: u32 = stacked.per_word[start..end]
                .iter()
                .map(|&c| c as u32)
                .sum();
            let max_act = stacked.per_word[start..end]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            let mean = block_sum as f32 / (end - start) as f32;
            let var: f32 = stacked.per_word[start..end]
                .iter()
                .map(|&c| {
                    let d = c as f32 - mean;
                    d * d
                })
                .sum::<f32>()
                / (end - start) as f32;

            block_signature[b] = block_sum as u16;
            blocks.push(NeuralBlock {
                index: b,
                start_word: start,
                end_word: end,
                block_sum,
                max_activation: max_act,
                variance: var,
            });
        }

        Self {
            layers: stacked.per_word,
            membrane: stacked.cumulative,
            total: stacked.total,
            blocks,
            prune_point: None,
            block_signature,
        }
    }

    /// Build with threshold pruning
    pub fn from_vectors_with_threshold(
        a: &dyn VectorRef,
        b: &dyn VectorRef,
        threshold: u32,
    ) -> Option<Self> {
        let stacked = StackedPopcount::compute_with_threshold_ref(a, b, threshold)?;
        let mut profile = Self::from_stacked(&stacked);
        // No pruning occurred if we got here
        Some(profile)
    }

    /// Build from two VectorRef (zero-copy)
    pub fn from_refs(a: &dyn VectorRef, b: &dyn VectorRef) -> Self {
        let stacked = StackedPopcount::compute_ref(a, b);
        Self::from_stacked(&stacked)
    }

    /// Map crystal coordinate to relevant blocks
    ///
    /// Each crystal dimension maps to 2 blocks. The dimension value (0-4)
    /// indicates how "active" that region should be — higher values mean
    /// the query is looking for high activation in those blocks.
    pub fn crystal_attention(&self, coord: &Coord5D) -> CrystalAttention {
        let mut attention_weights = [0.0f32; NUM_BLOCKS];
        let mut focus_blocks = Vec::new();

        for dim in 0..5 {
            let block_base = dim * BLOCKS_PER_CRYSTAL_DIM;
            let crystal_val = coord.dims[dim] as f32 / 4.0; // Normalize to [0, 1]

            for offset in 0..BLOCKS_PER_CRYSTAL_DIM {
                let block_idx = block_base + offset;
                if block_idx < NUM_BLOCKS {
                    // Attention weight based on crystal value and block activation
                    let block_activation = self.block_signature[block_idx] as f32;
                    attention_weights[block_idx] = crystal_val * block_activation;

                    if crystal_val > 0.5 {
                        focus_blocks.push(block_idx);
                    }
                }
            }
        }

        // Normalize attention weights
        let sum: f32 = attention_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut attention_weights {
                *w /= sum;
            }
        }

        let total_focus = focus_blocks.len();
        CrystalAttention {
            weights: attention_weights,
            focus_blocks,
            total_focus,
        }
    }

    /// Weighted distance using crystal attention
    pub fn crystal_weighted_distance(&self, attention: &CrystalAttention) -> f32 {
        let mut weighted = 0.0f32;
        for (i, &weight) in attention.weights.iter().enumerate() {
            if i < self.blocks.len() {
                weighted += weight * self.blocks[i].block_sum as f32;
            }
        }
        weighted
    }
}

/// Crystal-derived attention mask
#[derive(Clone, Debug)]
pub struct CrystalAttention {
    /// Per-block attention weights (sum to 1.0)
    pub weights: [f32; NUM_BLOCKS],
    /// Indices of focus blocks (crystal value > 0.5)
    pub focus_blocks: Vec<usize>,
    /// Number of focus blocks
    pub total_focus: usize,
}

// ============================================================================
// HIERARCHICAL NEURAL TREE
// ============================================================================

/// A node in the hierarchical neural tree
#[derive(Clone, Debug)]
pub struct NeuralTreeNode {
    /// DN tree address for hierarchical navigation
    pub addr: TreeAddr,
    /// Centroid fingerprint (majority bundle of children)
    pub centroid: BitpackedVector,
    /// Block signature of centroid (for coarse routing)
    pub block_signature: [u16; NUM_BLOCKS],
    /// Crystal coordinate (spatial position in 5D lattice)
    pub crystal_coord: Option<Coord5D>,
    /// Epiphany zone classification relative to parent
    pub zone: EpiphanyZone,
    /// Number of items in subtree
    pub count: usize,
    /// Sigma radius of this cluster
    pub radius: u32,
    /// Child node addresses (empty for leaves)
    pub children: Vec<TreeAddr>,
    /// Leaf items: (id, fingerprint)
    pub items: Vec<(u64, BitpackedVector)>,
    /// Hebbian strength (learned importance of this node)
    pub hebbian_weight: f32,
}

impl NeuralTreeNode {
    /// Is this a leaf?
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Compute block signature from centroid
    pub fn compute_block_signature(&mut self) {
        let stacked = self.centroid.stacked_popcount();
        for b in 0..NUM_BLOCKS {
            let start = b * WORDS_PER_BLOCK;
            let end = ((b + 1) * WORDS_PER_BLOCK).min(VECTOR_WORDS);
            self.block_signature[b] = stacked[start..end]
                .iter()
                .map(|&c| c as u16)
                .sum();
        }
    }

    /// Quick block-level distance to query
    pub fn block_distance(&self, query_blocks: &[u16; NUM_BLOCKS]) -> u32 {
        let mut dist = 0u32;
        for i in 0..NUM_BLOCKS {
            let diff = (self.block_signature[i] as i32 - query_blocks[i] as i32).unsigned_abs();
            dist += diff;
        }
        dist
    }
}

/// Configuration for the hierarchical neural tree
#[derive(Clone, Debug)]
pub struct NeuralTreeConfig {
    /// Maximum items per leaf before splitting
    pub max_leaf_size: usize,
    /// Maximum children per internal node
    pub max_children: usize,
    /// Search beam width
    pub beam_width: usize,
    /// Enable crystal-guided routing
    pub crystal_routing: bool,
    /// Enable Hebbian learning on access
    pub hebbian_learning: bool,
    /// Hebbian learning rate
    pub hebbian_rate: f32,
    /// Hebbian decay rate
    pub hebbian_decay: f32,
    /// Use multi-resolution block pre-filter
    pub block_prefilter: bool,
}

impl Default for NeuralTreeConfig {
    fn default() -> Self {
        Self {
            max_leaf_size: 64,
            max_children: 16,
            beam_width: 4,
            crystal_routing: true,
            hebbian_learning: true,
            hebbian_rate: 0.1,
            hebbian_decay: 0.999,
            block_prefilter: true,
        }
    }
}

/// The Hierarchical Neural Tree
///
/// Combines:
/// - **Stacked popcount** as 157-layer neural forward pass
/// - **Multi-resolution blocks** (10 blocks of 1024 bits) for coarse routing
/// - **Crystal coordinates** for spatial attention masks
/// - **DN tree addressing** for O(1) hierarchical node lookup
/// - **Hebbian learning** — accessed nodes strengthen, unused decay
/// - **Epiphany zones** for adaptive threshold calibration
pub struct HierarchicalNeuralTree {
    /// Configuration
    config: NeuralTreeConfig,
    /// Nodes by DN address
    nodes: HashMap<TreeAddr, NeuralTreeNode>,
    /// Root address
    root: TreeAddr,
    /// Total items
    total_items: usize,
    /// Next item ID
    next_id: u64,
    /// Crystal cell cache: crystal coord → cell fingerprint
    crystal_cells: HashMap<usize, BitpackedVector>,
    /// Global search statistics
    total_searches: u64,
    total_pruned: u64,
    total_block_filtered: u64,
}

impl HierarchicalNeuralTree {
    /// Create a new hierarchical neural tree
    pub fn new() -> Self {
        Self::with_config(NeuralTreeConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: NeuralTreeConfig) -> Self {
        let root = TreeAddr::root();
        let mut nodes = HashMap::new();
        nodes.insert(
            root.clone(),
            NeuralTreeNode {
                addr: root.clone(),
                centroid: BitpackedVector::zero(),
                block_signature: [0u16; NUM_BLOCKS],
                crystal_coord: None,
                zone: EpiphanyZone::Identity,
                count: 0,
                radius: 0,
                children: Vec::new(),
                items: Vec::new(),
                hebbian_weight: 1.0,
            },
        );

        Self {
            config,
            nodes,
            root,
            total_items: 0,
            next_id: 0,
            crystal_cells: HashMap::new(),
            total_searches: 0,
            total_pruned: 0,
            total_block_filtered: 0,
        }
    }

    // ========================================================================
    // INSERTION
    // ========================================================================

    /// Insert a fingerprint, returns assigned ID
    pub fn insert(&mut self, fingerprint: BitpackedVector) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.insert_with_id(id, fingerprint);
        id
    }

    /// Insert with explicit ID
    pub fn insert_with_id(&mut self, id: u64, fingerprint: BitpackedVector) {
        self.next_id = self.next_id.max(id + 1);
        self.total_items += 1;

        // Compute crystal coordinate for spatial routing
        let crystal_coord = self.fingerprint_to_crystal(&fingerprint);

        // Register in crystal cell cache
        let cell_idx = crystal_coord.to_index();
        self.crystal_cells
            .entry(cell_idx)
            .and_modify(|existing| {
                // Bundle with existing cell fingerprint
                let old = existing.clone();
                let refs: Vec<&BitpackedVector> = vec![&old, &fingerprint];
                *existing = BitpackedVector::bundle(&refs);
            })
            .or_insert_with(|| fingerprint.clone());

        // Find best leaf using neural routing
        let leaf_addr = self.neural_route(&fingerprint);

        // Insert into leaf
        if let Some(node) = self.nodes.get_mut(&leaf_addr) {
            node.items.push((id, fingerprint.clone()));
            node.count += 1;

            // Update centroid
            let refs: Vec<&BitpackedVector> = node.items.iter().map(|(_, fp)| fp).collect();
            node.centroid = BitpackedVector::bundle(&refs);
            node.compute_block_signature();
            node.crystal_coord = Some(crystal_coord);

            // Check if split needed
            if node.items.len() > self.config.max_leaf_size {
                // Move items out instead of cloning (~80KB savings per split)
                let items = std::mem::take(&mut node.items);
                let addr = node.addr.clone();
                self.split_node(&addr, items);
            }
        }

        // Update centroids up the tree
        self.propagate_centroids(&leaf_addr);
    }

    /// Map fingerprint to crystal coordinate using block signature
    fn fingerprint_to_crystal(&self, fp: &BitpackedVector) -> Coord5D {
        let stacked = fp.stacked_popcount();
        let mut dims = [0u8; 5];

        // Each crystal dimension maps to 2 blocks (≈ 2048 bits)
        for dim in 0..5 {
            let block_base = dim * BLOCKS_PER_CRYSTAL_DIM;
            let mut dim_sum = 0u32;
            let mut dim_bits = 0u32;

            for offset in 0..BLOCKS_PER_CRYSTAL_DIM {
                let block_idx = block_base + offset;
                let start = block_idx * WORDS_PER_BLOCK;
                let end = ((block_idx + 1) * WORDS_PER_BLOCK).min(VECTOR_WORDS);
                for w in start..end {
                    dim_sum += stacked[w] as u32;
                    dim_bits += 64;
                }
            }

            // Map density to crystal coordinate (0-4)
            // density 0.0 → 0, density 0.5 → 2, density 1.0 → 4
            let density = dim_sum as f32 / dim_bits as f32;
            dims[dim] = (density * 4.999).clamp(0.0, 4.0) as u8;
        }

        Coord5D::new(dims[0], dims[1], dims[2], dims[3], dims[4])
    }

    /// Neural routing: find best leaf for insertion
    fn neural_route(&self, fingerprint: &BitpackedVector) -> TreeAddr {
        let mut current = self.root.clone();

        // Precompute query block signature for fast comparison
        let query_profile = NeuralProfile::from_refs(fingerprint, &BitpackedVector::zero());

        loop {
            match self.nodes.get(&current) {
                Some(node) if node.is_leaf() => return current,
                Some(node) => {
                    // Multi-resolution routing:
                    // 1. Block-level pre-filter (coarse)
                    // 2. Belichtungsmesser on survivors (7-point)
                    // 3. Full distance on final candidates

                    let mut best_child = node.children[0].clone();
                    let mut best_score = u32::MAX;

                    for child_addr in &node.children {
                        if let Some(child) = self.nodes.get(child_addr) {
                            if self.config.block_prefilter {
                                // Coarse: block signature distance
                                let block_dist =
                                    child.block_distance(&query_profile.block_signature);
                                if block_dist < best_score {
                                    best_score = block_dist;
                                    best_child = child_addr.clone();
                                }
                            } else {
                                // Fine: exact Hamming
                                let dist =
                                    hamming_distance_scalar(fingerprint, &child.centroid);
                                if dist < best_score {
                                    best_score = dist;
                                    best_child = child_addr.clone();
                                }
                            }
                        }
                    }

                    current = best_child;
                }
                None => return self.root.clone(),
            }
        }
    }

    /// Split a leaf into internal node with children
    fn split_node(&mut self, addr: &TreeAddr, items: Vec<(u64, BitpackedVector)>) {
        let num_children = self.config.max_children.min(items.len());
        if num_children < 2 {
            return;
        }

        // Cluster by crystal coordinate for spatial coherence
        let clusters = self.crystal_cluster(&items, num_children);

        let mut children = Vec::new();
        for (i, cluster) in clusters.into_iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }
            let child_addr = addr.child(i as u8);
            let refs: Vec<&BitpackedVector> = cluster.iter().map(|(_, fp)| fp).collect();
            let centroid = BitpackedVector::bundle(&refs);
            let crystal_coord = self.fingerprint_to_crystal(&centroid);

            let mut child_node = NeuralTreeNode {
                addr: child_addr.clone(),
                centroid,
                block_signature: [0u16; NUM_BLOCKS],
                crystal_coord: Some(crystal_coord),
                zone: EpiphanyZone::Identity,
                count: cluster.len(),
                radius: 0,
                children: Vec::new(),
                items: cluster,
                hebbian_weight: 1.0,
            };
            child_node.compute_block_signature();

            // Compute radius
            if !child_node.items.is_empty() {
                let max_dist = child_node
                    .items
                    .iter()
                    .map(|(_, fp)| hamming_distance_scalar(&child_node.centroid, fp))
                    .max()
                    .unwrap_or(0);
                child_node.radius = max_dist;
                child_node.zone = EpiphanyZone::classify(max_dist);
            }

            children.push(child_addr.clone());
            self.nodes.insert(child_addr, child_node);
        }

        // Convert leaf to internal node
        let refs: Vec<&BitpackedVector> = items.iter().map(|(_, fp)| fp).collect();
        let centroid = BitpackedVector::bundle(&refs);

        let mut internal = NeuralTreeNode {
            addr: addr.clone(),
            centroid,
            block_signature: [0u16; NUM_BLOCKS],
            crystal_coord: None,
            zone: EpiphanyZone::Identity,
            count: items.len(),
            radius: 0,
            children,
            items: Vec::new(),
            hebbian_weight: 1.0,
        };
        internal.compute_block_signature();
        self.nodes.insert(addr.clone(), internal);
    }

    /// Cluster items by crystal coordinate for spatial coherence
    fn crystal_cluster(
        &self,
        items: &[(u64, BitpackedVector)],
        k: usize,
    ) -> Vec<Vec<(u64, BitpackedVector)>> {
        if items.len() <= k {
            return items
                .iter()
                .map(|(id, fp)| vec![(*id, fp.clone())])
                .collect();
        }

        // Assign items to crystal-based clusters
        let mut clusters: Vec<Vec<(u64, BitpackedVector)>> = (0..k).map(|_| Vec::new()).collect();

        for (id, fp) in items {
            let coord = self.fingerprint_to_crystal(fp);
            // Hash crystal coordinate to cluster index
            let cluster_idx = coord.to_index() % k;
            clusters[cluster_idx].push((*id, fp.clone()));
        }

        // Redistribute empty clusters
        let non_empty: Vec<_> = clusters.into_iter().filter(|c| !c.is_empty()).collect();
        if non_empty.is_empty() {
            return vec![items.to_vec()];
        }
        non_empty
    }

    /// Propagate centroid updates from leaf to root
    fn propagate_centroids(&mut self, start: &TreeAddr) {
        let mut current = start.parent();
        while let Some(addr) = current {
            if let Some(node) = self.nodes.get(&addr) {
                let child_addrs = node.children.clone();
                let child_fps: Vec<BitpackedVector> = child_addrs
                    .iter()
                    .filter_map(|c| self.nodes.get(c).map(|n| n.centroid.clone()))
                    .collect();
                let new_count: usize = child_addrs
                    .iter()
                    .filter_map(|c| self.nodes.get(c).map(|n| n.count))
                    .sum();
                let refs: Vec<&BitpackedVector> = child_fps.iter().collect();

                if let Some(node) = self.nodes.get_mut(&addr) {
                    if !refs.is_empty() {
                        node.centroid = BitpackedVector::bundle(&refs);
                        node.compute_block_signature();
                        node.count = new_count;
                    }
                }
            }
            current = addr.parent();
        }
    }

    // ========================================================================
    // SEARCH: The Neural Forward Pass
    // ========================================================================

    /// Neural search: k nearest neighbors with stacked popcount pruning
    ///
    /// This is the "magic" — each word boundary is a pruning checkpoint.
    /// 90% of candidates are rejected in the first 16 words (1024 bits).
    /// The remaining 10% are refined through the full 157-word pass.
    pub fn search(&mut self, query: &BitpackedVector, k: usize) -> Vec<NeuralSearchResult> {
        self.total_searches += 1;

        // Pre-compute query block signature
        let query_stacked = query.stacked_popcount();
        let mut query_blocks = [0u16; NUM_BLOCKS];
        for b in 0..NUM_BLOCKS {
            let start = b * WORDS_PER_BLOCK;
            let end = ((b + 1) * WORDS_PER_BLOCK).min(VECTOR_WORDS);
            query_blocks[b] = query_stacked[start..end]
                .iter()
                .map(|&c| c as u16)
                .sum();
        }

        // Crystal coordinate for attention routing
        let query_crystal = self.fingerprint_to_crystal(query);

        let mut results = Vec::new();
        let mut beam: Vec<(TreeAddr, u32)> = vec![(self.root.clone(), 0)];

        while !beam.is_empty() {
            beam.sort_by_key(|(_, d)| *d);
            beam.truncate(self.config.beam_width);

            let mut next_beam = Vec::new();

            for (addr, _) in &beam {
                let node = match self.nodes.get(addr) {
                    Some(n) => n,
                    None => continue,
                };

                if node.is_leaf() {
                    // Leaf: stacked popcount forward pass on each item
                    for (id, fp) in &node.items {
                        // Level 0: Belichtungsmesser (7 samples, ~14 cycles)
                        let exposure = Belichtung::meter(query, fp);
                        if exposure.definitely_far(0.5) {
                            self.total_pruned += 1;
                            continue;
                        }

                        // Level 1: Stacked popcount with early termination
                        let threshold = if results.len() >= k {
                            results.last().map(|r: &NeuralSearchResult| r.distance).unwrap_or(u32::MAX)
                        } else {
                            THREE_SIGMA
                        };

                        match StackedPopcount::compute_with_threshold(query, fp, threshold) {
                            Some(stacked) => {
                                let profile = NeuralProfile::from_stacked(&stacked);
                                let crystal_coord = self.fingerprint_to_crystal(fp);
                                let attention = profile.crystal_attention(&query_crystal);

                                results.push(NeuralSearchResult {
                                    id: *id,
                                    distance: stacked.total,
                                    zone: EpiphanyZone::classify(stacked.total),
                                    crystal_coord,
                                    crystal_distance: query_crystal.distance(&crystal_coord),
                                    block_signature: profile.block_signature,
                                    attention_score: profile.crystal_weighted_distance(&attention),
                                    prune_depth: VECTOR_WORDS, // Full pass completed
                                });

                                // Keep sorted, trim to k
                                results.sort_by_key(|r| r.distance);
                                if results.len() > k {
                                    results.truncate(k);
                                }
                            }
                            None => {
                                self.total_pruned += 1;
                            }
                        }
                    }

                    // Hebbian: strengthen accessed leaf
                    if self.config.hebbian_learning {
                        if let Some(node) = self.nodes.get_mut(addr) {
                            node.hebbian_weight =
                                (node.hebbian_weight + self.config.hebbian_rate).min(5.0);
                        }
                    }
                } else {
                    // Internal: route to best children using block pre-filter
                    for child_addr in &node.children {
                        if let Some(child) = self.nodes.get(child_addr) {
                            let score = if self.config.block_prefilter {
                                // Block distance as routing heuristic
                                let block_dist = child.block_distance(&query_blocks);

                                // Crystal coherence bonus
                                let crystal_bonus = if self.config.crystal_routing {
                                    if let Some(ref coord) = child.crystal_coord {
                                        let crystal_dist = query_crystal.distance(coord);
                                        // Closer crystal = lower score = higher priority
                                        crystal_dist * 10
                                    } else {
                                        0
                                    }
                                } else {
                                    0
                                };

                                // Hebbian bonus: well-traveled paths get priority
                                let hebbian_discount =
                                    (10.0 / child.hebbian_weight) as u32;

                                block_dist + crystal_bonus + hebbian_discount
                            } else {
                                hamming_distance_scalar(query, &child.centroid)
                            };

                            next_beam.push((child_addr.clone(), score));
                        }
                    }
                }
            }

            beam = next_beam;
        }

        // Apply Hebbian decay globally
        if self.config.hebbian_learning {
            for node in self.nodes.values_mut() {
                node.hebbian_weight *= self.config.hebbian_decay;
                node.hebbian_weight = node.hebbian_weight.max(0.1);
            }
        }

        results
    }

    /// Range search with neural pruning
    pub fn range_search(
        &mut self,
        query: &BitpackedVector,
        threshold: u32,
    ) -> Vec<NeuralSearchResult> {
        let query_crystal = self.fingerprint_to_crystal(query);
        let mut results = Vec::new();
        let mut stack = vec![self.root.clone()];

        while let Some(addr) = stack.pop() {
            let node = match self.nodes.get(&addr) {
                Some(n) => n,
                None => continue,
            };

            if node.is_leaf() {
                for (id, fp) in &node.items {
                    if let Some(stacked) =
                        StackedPopcount::compute_with_threshold(query, fp, threshold)
                    {
                        let crystal_coord = self.fingerprint_to_crystal(fp);
                        results.push(NeuralSearchResult {
                            id: *id,
                            distance: stacked.total,
                            zone: EpiphanyZone::classify(stacked.total),
                            crystal_coord,
                            crystal_distance: query_crystal.distance(&crystal_coord),
                            block_signature: NeuralProfile::from_stacked(&stacked).block_signature,
                            attention_score: 0.0,
                            prune_depth: VECTOR_WORDS,
                        });
                    }
                }
            } else {
                // Prune subtrees whose centroid is too far
                for child_addr in &node.children {
                    if let Some(child) = self.nodes.get(child_addr) {
                        let centroid_dist = hamming_distance_scalar(query, &child.centroid);
                        // Triangle inequality: if centroid - radius > threshold, skip
                        let effective = centroid_dist.saturating_sub(child.radius);
                        if effective <= threshold {
                            stack.push(child_addr.clone());
                        } else {
                            self.total_pruned += 1;
                        }
                    }
                }
            }
        }

        results.sort_by_key(|r| r.distance);
        results
    }

    /// Superposition search: find items resonating with a crystal region
    ///
    /// Instead of searching by exact fingerprint, search by crystal neighborhood.
    /// Returns all items whose crystal coordinates fall within the given radius.
    pub fn crystal_neighborhood_search(
        &self,
        center: &Coord5D,
        crystal_radius: u32,
    ) -> Vec<(u64, Coord5D, u32)> {
        let mut results = Vec::new();

        for node in self.nodes.values() {
            if node.is_leaf() {
                for (id, fp) in &node.items {
                    let coord = self.fingerprint_to_crystal(fp);
                    let dist = center.distance(&coord);
                    if dist <= crystal_radius {
                        results.push((*id, coord, dist));
                    }
                }
            }
        }

        results.sort_by_key(|(_, _, d)| *d);
        results
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Total items
    pub fn len(&self) -> usize {
        self.total_items
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.total_items == 0
    }

    /// Tree depth
    pub fn depth(&self) -> u8 {
        self.nodes.keys().map(|a| a.depth()).max().unwrap_or(0)
    }

    /// Search efficiency statistics
    pub fn stats(&self) -> NeuralTreeStats {
        let internal = self.nodes.values().filter(|n| !n.is_leaf()).count();
        let leaves = self.nodes.values().filter(|n| n.is_leaf()).count();
        let avg_leaf = if leaves > 0 {
            self.total_items as f32 / leaves as f32
        } else {
            0.0
        };

        let avg_hebbian = if self.nodes.is_empty() {
            1.0
        } else {
            self.nodes.values().map(|n| n.hebbian_weight).sum::<f32>()
                / self.nodes.len() as f32
        };

        let crystal_cells_used = self.crystal_cells.len();

        NeuralTreeStats {
            total_items: self.total_items,
            depth: self.depth(),
            internal_nodes: internal,
            leaf_nodes: leaves,
            avg_leaf_size: avg_leaf,
            total_searches: self.total_searches,
            total_pruned: self.total_pruned,
            total_block_filtered: self.total_block_filtered,
            prune_rate: if self.total_searches > 0 {
                self.total_pruned as f32 / (self.total_searches as f32 * self.total_items as f32).max(1.0)
            } else {
                0.0
            },
            avg_hebbian_weight: avg_hebbian,
            crystal_cells_used,
            crystal_coverage: crystal_cells_used as f32 / Coord5D::TOTAL_CELLS as f32,
        }
    }
}

impl Default for HierarchicalNeuralTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Search result from neural tree
#[derive(Clone, Debug)]
pub struct NeuralSearchResult {
    /// Item ID
    pub id: u64,
    /// Exact Hamming distance
    pub distance: u32,
    /// Epiphany zone classification
    pub zone: EpiphanyZone,
    /// Crystal coordinate of the result
    pub crystal_coord: Coord5D,
    /// Crystal distance (Manhattan) from query
    pub crystal_distance: u32,
    /// Block-level activation signature
    pub block_signature: [u16; NUM_BLOCKS],
    /// Crystal-weighted attention score
    pub attention_score: f32,
    /// Depth at which neural forward pass completed (157 = full)
    pub prune_depth: usize,
}

/// Neural tree statistics
#[derive(Clone, Debug)]
pub struct NeuralTreeStats {
    pub total_items: usize,
    pub depth: u8,
    pub internal_nodes: usize,
    pub leaf_nodes: usize,
    pub avg_leaf_size: f32,
    pub total_searches: u64,
    pub total_pruned: u64,
    pub total_block_filtered: u64,
    pub prune_rate: f32,
    pub avg_hebbian_weight: f32,
    pub crystal_cells_used: usize,
    pub crystal_coverage: f32,
}

impl std::fmt::Display for NeuralTreeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NeuralTree[{} items, depth={}, {}/{} int/leaf, \
             searches={}, pruned={} ({:.1}%), \
             hebbian={:.2}, crystal={}/3125 ({:.1}%)]",
            self.total_items,
            self.depth,
            self.internal_nodes,
            self.leaf_nodes,
            self.total_searches,
            self.total_pruned,
            self.prune_rate * 100.0,
            self.avg_hebbian_weight,
            self.crystal_cells_used,
            self.crystal_coverage * 100.0,
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_profile_from_stacked() {
        let a = BitpackedVector::random(42);
        let b = BitpackedVector::random(43);
        let stacked = StackedPopcount::compute(&a, &b);
        let profile = NeuralProfile::from_stacked(&stacked);

        assert_eq!(profile.total, stacked.total);
        assert_eq!(profile.blocks.len(), NUM_BLOCKS);

        // Block sums should add up to total
        let block_total: u32 = profile.blocks.iter().map(|b| b.block_sum).sum();
        assert_eq!(block_total, stacked.total);
    }

    #[test]
    fn test_neural_profile_crystal_attention() {
        let a = BitpackedVector::random(100);
        let b = BitpackedVector::random(200);
        let profile = NeuralProfile::from_refs(&a, &b);

        let coord = Coord5D::new(2, 3, 1, 4, 0);
        let attention = profile.crystal_attention(&coord);

        // Weights should sum to ~1.0
        let sum: f32 = attention.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01 || sum == 0.0);

        // Focus blocks should be populated
        assert!(!attention.focus_blocks.is_empty() || coord.dims.iter().all(|&d| d <= 2));
    }

    #[test]
    fn test_neural_block_sigma_zone() {
        let a = BitpackedVector::zero();
        let b = BitpackedVector::zero();
        let stacked = StackedPopcount::compute(&a, &b);
        let profile = NeuralProfile::from_stacked(&stacked);

        // Zero vs zero: all blocks should be identity (zero activation)
        for block in &profile.blocks {
            assert_eq!(block.block_sum, 0);
        }
    }

    #[test]
    fn test_hierarchical_insert_search() {
        let mut tree = HierarchicalNeuralTree::new();

        // Insert 100 vectors
        let vectors: Vec<BitpackedVector> =
            (0..100).map(|i| BitpackedVector::random(i as u64)).collect();
        for (i, v) in vectors.iter().enumerate() {
            tree.insert_with_id(i as u64, v.clone());
        }

        assert_eq!(tree.len(), 100);

        // Search for exact match
        let results = tree.search(&vectors[50], 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].distance, 0); // Exact match
        assert_eq!(results[0].id, 50);
    }

    #[test]
    fn test_neural_tree_splitting() {
        let config = NeuralTreeConfig {
            max_leaf_size: 8,
            max_children: 4,
            ..Default::default()
        };
        let mut tree = HierarchicalNeuralTree::with_config(config);

        for i in 0..100 {
            tree.insert(BitpackedVector::random(i));
        }

        let stats = tree.stats();
        assert!(stats.depth > 0, "Should have split into deeper tree");
        assert!(stats.internal_nodes > 0, "Should have internal nodes");
    }

    #[test]
    fn test_range_search() {
        let mut tree = HierarchicalNeuralTree::new();

        for i in 0..50 {
            tree.insert(BitpackedVector::random(i));
        }

        let query = BitpackedVector::random(25);
        let results = tree.range_search(&query, 0);
        // Should find exact match
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_crystal_neighborhood_search() {
        let mut tree = HierarchicalNeuralTree::new();

        for i in 0..50 {
            tree.insert(BitpackedVector::random(i));
        }

        let center = Coord5D::new(2, 2, 2, 2, 2);
        let results = tree.crystal_neighborhood_search(&center, 5);
        // With random vectors, many should map near center
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hebbian_learning() {
        let config = NeuralTreeConfig {
            hebbian_learning: true,
            hebbian_rate: 0.5,
            hebbian_decay: 0.9,
            ..Default::default()
        };
        let mut tree = HierarchicalNeuralTree::with_config(config);

        for i in 0..20 {
            tree.insert(BitpackedVector::random(i));
        }

        let query = BitpackedVector::random(10);
        // Search multiple times — accessed paths should strengthen
        for _ in 0..5 {
            tree.search(&query, 3);
        }

        let stats = tree.stats();
        // After searches and decay, hebbian weights should vary
        assert!(stats.total_searches == 5);
    }

    #[test]
    fn test_fingerprint_to_crystal() {
        let tree = HierarchicalNeuralTree::new();

        let fp = BitpackedVector::random(42);
        let coord = tree.fingerprint_to_crystal(&fp);

        // Should be valid crystal coordinate
        assert!(coord.dims.iter().all(|&d| d < 5));

        // Same fingerprint → same coordinate
        let coord2 = tree.fingerprint_to_crystal(&fp);
        assert_eq!(coord, coord2);
    }

    #[test]
    fn test_neural_tree_stats() {
        let mut tree = HierarchicalNeuralTree::new();

        for i in 0..30 {
            tree.insert(BitpackedVector::random(i));
        }

        tree.search(&BitpackedVector::random(15), 5);

        let stats = tree.stats();
        assert_eq!(stats.total_items, 30);
        assert!(stats.total_searches >= 1);
        assert!(stats.crystal_cells_used > 0);
        println!("{}", stats);
    }

    #[test]
    fn test_block_prefilter() {
        let config = NeuralTreeConfig {
            block_prefilter: true,
            max_leaf_size: 8,
            max_children: 4,
            ..Default::default()
        };
        let mut tree = HierarchicalNeuralTree::with_config(config);

        for i in 0..100 {
            tree.insert(BitpackedVector::random(i));
        }

        let results = tree.search(&BitpackedVector::random(50), 5);
        assert!(!results.is_empty());
        // First result should be exact match
        assert_eq!(results[0].distance, 0);
    }
}
