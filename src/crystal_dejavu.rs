//! Sentence Crystal + Déjà Vu RL + Truth Markers
//!
//! A unified system combining:
//! - **Sentence Crystal**: Transformer embeddings → 5D crystal → fingerprints
//! - **Déjà Vu RL**: Multipass ±3σ overlay creates reinforcement patterns
//! - **Truth Markers**: Orthogonal superposition cleaning for interference removal
//!
//! # The Crystal-Déjà Vu-Truth Pipeline
//!
//! ```text
//! Text ──► Transformer ──► 1024D Dense ──► Random Projection ──► 5D Crystal
//!              │                                                      │
//!              │           ┌──────────────────────────────────────────┘
//!              │           │
//!              │           ▼
//!              │      Crystal Cell (5×5×5×5×5 = 3125 cells)
//!              │           │
//!              │           ▼
//!              │      Fingerprint (10Kbit)
//!              │           │
//!              │           ├──► Déjà Vu RL (multipass ±3σ overlay)
//!              │           │         │
//!              │           │         ▼
//!              │           │    Reinforcement Pattern
//!              │           │         │
//!              │           └──► Truth Marker Cleaning
//!              │                     │
//!              │                     ▼
//!              └──────────────► Clean Signal
//! ```
//!
//! # Déjà Vu Effect
//!
//! When the same concept appears across multiple passes at different σ levels,
//! it creates a "déjà vu" reinforcement - the feeling that you've seen this before.
//! This is captured as accumulated evidence across the ±3σ range.

use crate::bitpack::{BitpackedVector, VECTOR_BITS};
use crate::hamming::hamming_distance_scalar;
use crate::epiphany::{EpiphanyZone, ONE_SIGMA, TWO_SIGMA, THREE_SIGMA, HAMMING_STD_DEV};
use std::collections::HashMap;

// ============================================================================
// SENTENCE CRYSTAL: Transformer → Fingerprint Bridge
// ============================================================================

/// 5D Crystal coordinate (each dimension 0-4)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Coord5D {
    pub dims: [u8; 5],
}

impl Coord5D {
    pub const LATTICE_SIZE: usize = 5;
    pub const TOTAL_CELLS: usize = 5 * 5 * 5 * 5 * 5; // 3125

    /// Create from dimensions
    pub fn new(d0: u8, d1: u8, d2: u8, d3: u8, d4: u8) -> Self {
        Self {
            dims: [
                d0 % 5, d1 % 5, d2 % 5, d3 % 5, d4 % 5
            ],
        }
    }

    /// Create from linear index
    pub fn from_index(mut idx: usize) -> Self {
        let mut dims = [0u8; 5];
        for i in (0..5).rev() {
            dims[i] = (idx % 5) as u8;
            idx /= 5;
        }
        Self { dims }
    }

    /// Convert to linear index
    pub fn to_index(&self) -> usize {
        let mut idx = 0;
        for &d in &self.dims {
            idx = idx * 5 + d as usize;
        }
        idx
    }

    /// Manhattan distance to another coordinate
    pub fn distance(&self, other: &Self) -> u32 {
        self.dims.iter()
            .zip(other.dims.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum()
    }

    /// Get all coordinates within Manhattan radius
    pub fn neighborhood(&self, radius: u32) -> Vec<Coord5D> {
        let mut neighbors = Vec::new();
        for idx in 0..Self::TOTAL_CELLS {
            let coord = Coord5D::from_index(idx);
            if self.distance(&coord) <= radius {
                neighbors.push(coord);
            }
        }
        neighbors
    }

    /// Convert to deterministic fingerprint
    pub fn to_fingerprint(&self) -> BitpackedVector {
        let seed = (self.to_index() as u64).wrapping_mul(0x9E3779B97F4A7C15);
        BitpackedVector::random(seed)
    }
}

/// Random projection matrix (Johnson-Lindenstrauss)
/// Projects from dense embedding space to 5D crystal
pub struct ProjectionMatrix {
    /// Projection weights [5][input_dim]
    weights: Vec<Vec<f32>>,
    /// Input dimensionality
    input_dim: usize,
    /// Bias terms for each output dimension
    bias: [f32; 5],
}

impl ProjectionMatrix {
    /// Create random projection matrix
    pub fn new(input_dim: usize, seed: u64) -> Self {
        // Use LFSR-based PRNG for reproducibility
        let mut state = seed;
        let mut weights = Vec::with_capacity(5);

        for _ in 0..5 {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                // LFSR step
                state = state.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);

                // Map to [-1, 1] range
                let val = ((state >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                // Scale by sqrt(1/input_dim) for variance preservation
                row.push(val / (input_dim as f32).sqrt());
            }
            weights.push(row);
        }

        Self {
            weights,
            input_dim,
            bias: [0.0; 5],
        }
    }

    /// Project dense embedding to 5D, then quantize to crystal coordinate
    pub fn project(&self, embedding: &[f32]) -> Coord5D {
        assert_eq!(embedding.len(), self.input_dim);

        let mut coords = [0u8; 5];

        for (dim, row) in self.weights.iter().enumerate() {
            // Dot product
            let sum: f32 = embedding.iter()
                .zip(row.iter())
                .map(|(e, w)| e * w)
                .sum();

            // Tanh normalization to [-1, 1], then map to [0, 5)
            let normalized = (sum + self.bias[dim]).tanh();
            let quantized = ((normalized + 1.0) * 2.5).clamp(0.0, 4.999) as u8;
            coords[dim] = quantized;
        }

        Coord5D { dims: coords }
    }
}

/// Maximum entries kept per cell to prevent memory bloat.
/// Beyond this, we keep only the bundled prototype and a count.
const MAX_CELL_ENTRIES: usize = 128;

/// Crystal cell containing bundled fingerprints
#[derive(Clone, Debug)]
pub struct CrystalCell {
    /// Coordinate in 5D lattice
    pub coord: Coord5D,
    /// Bundled fingerprint (majority of all entries)
    pub fingerprint: BitpackedVector,
    /// Ring buffer of recent entry fingerprints (capped at MAX_CELL_ENTRIES)
    entries: Vec<BitpackedVector>,
    /// Total entry count (including evicted)
    pub count: usize,
    /// Average qualia (emotional signature)
    pub qualia: [f32; 8],
    /// Truth marker (confidence)
    pub truth: f32,
}

impl CrystalCell {
    pub fn new(coord: Coord5D) -> Self {
        Self {
            coord,
            fingerprint: coord.to_fingerprint(),
            entries: Vec::new(),
            count: 0,
            qualia: [0.0; 8],
            truth: 0.5, // Neutral truth
        }
    }

    /// Add fingerprint to cell
    pub fn add(&mut self, fp: BitpackedVector, qualia: Option<[f32; 8]>) {
        self.count += 1;

        // Evict oldest entry if at capacity (ring buffer behavior)
        if self.entries.len() >= MAX_CELL_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push(fp);

        // Update bundled fingerprint via majority of retained entries
        if self.entries.len() > 1 {
            let refs: Vec<&BitpackedVector> = self.entries.iter().collect();
            self.fingerprint = BitpackedVector::bundle(&refs);
        } else {
            self.fingerprint = self.entries[0].clone();
        }

        // Update qualia (running average)
        if let Some(q) = qualia {
            for i in 0..8 {
                self.qualia[i] = (self.qualia[i] * (self.count - 1) as f32 + q[i])
                    / self.count as f32;
            }
        }
    }

    /// Get similarity to query
    pub fn similarity(&self, query: &BitpackedVector) -> f32 {
        let dist = hamming_distance_scalar(&self.fingerprint, query);
        1.0 - (dist as f32 / VECTOR_BITS as f32)
    }
}

/// Maximum embedding cache entries to prevent unbounded memory growth.
/// At 1024-dim embeddings × 4 bytes = 4KB per entry, 10K entries = ~40MB.
const MAX_EMBEDDING_CACHE: usize = 10_000;

/// Sentence Crystal: transforms text embeddings to fingerprints
pub struct SentenceCrystal {
    /// Projection matrix
    projection: ProjectionMatrix,
    /// Crystal cells
    cells: HashMap<usize, CrystalCell>,
    /// Embedding cache (for expensive transformer calls), bounded
    embedding_cache: HashMap<String, Vec<f32>>,
    /// Insertion order for FIFO eviction of embedding_cache
    cache_order: Vec<String>,
    /// Embedding dimension (default: 1024 for Jina v3)
    embedding_dim: usize,
}

impl SentenceCrystal {
    /// Create new crystal with given embedding dimension
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            projection: ProjectionMatrix::new(embedding_dim, 0xC4157A15EED00001),
            cells: HashMap::new(),
            embedding_cache: HashMap::new(),
            cache_order: Vec::new(),
            embedding_dim,
        }
    }

    /// Create with Jina v3 dimensions (1024)
    pub fn jina_v3() -> Self {
        Self::new(1024)
    }

    /// Store embedding with fingerprint
    pub fn store(&mut self, text: &str, embedding: Vec<f32>) -> Coord5D {
        // Cache embedding with FIFO eviction
        if !self.embedding_cache.contains_key(text) {
            if self.embedding_cache.len() >= MAX_EMBEDDING_CACHE {
                // Evict oldest
                if let Some(oldest) = self.cache_order.first().cloned() {
                    self.embedding_cache.remove(&oldest);
                    self.cache_order.remove(0);
                }
            }
            self.cache_order.push(text.to_string());
        }
        self.embedding_cache.insert(text.to_string(), embedding.clone());

        // Project to crystal coordinate
        let coord = self.projection.project(&embedding);
        let idx = coord.to_index();

        // Create fingerprint from embedding (before mutable borrow of cells)
        let fp = self.embedding_to_fingerprint(&embedding);

        // Create or update cell
        let cell = self.cells.entry(idx).or_insert_with(|| CrystalCell::new(coord));
        cell.add(fp, None);

        coord
    }

    /// Store with qualia (emotional signature)
    pub fn store_with_qualia(
        &mut self,
        text: &str,
        embedding: Vec<f32>,
        qualia: [f32; 8],
    ) -> Coord5D {
        // Cache embedding with FIFO eviction
        if !self.embedding_cache.contains_key(text) {
            if self.embedding_cache.len() >= MAX_EMBEDDING_CACHE {
                if let Some(oldest) = self.cache_order.first().cloned() {
                    self.embedding_cache.remove(&oldest);
                    self.cache_order.remove(0);
                }
            }
            self.cache_order.push(text.to_string());
        }
        self.embedding_cache.insert(text.to_string(), embedding.clone());

        let coord = self.projection.project(&embedding);
        let idx = coord.to_index();

        let fp = self.embedding_to_fingerprint(&embedding);
        let cell = self.cells.entry(idx).or_insert_with(|| CrystalCell::new(coord));
        cell.add(fp, Some(qualia));

        coord
    }

    /// Convert dense embedding to fingerprint via thresholding
    fn embedding_to_fingerprint(&self, embedding: &[f32]) -> BitpackedVector {
        let mut fp = BitpackedVector::zero();

        // Use embedding values to set bits
        // Each embedding dimension contributes to multiple bits
        let bits_per_dim = VECTOR_BITS / embedding.len();

        for (i, &val) in embedding.iter().enumerate() {
            let base_bit = i * bits_per_dim;

            // Threshold at different levels
            if val > 0.0 {
                fp.set_bit(base_bit, true);
            }
            if val > 0.5 {
                fp.set_bit(base_bit + 1, true);
            }
            if val > 1.0 {
                fp.set_bit(base_bit + 2, true);
            }
            if val < -0.5 {
                fp.set_bit(base_bit + 3, true);
            }
        }

        fp
    }

    /// Query for similar entries
    pub fn query(&self, embedding: &[f32], radius: u32) -> Vec<(&CrystalCell, f32)> {
        let coord = self.projection.project(embedding);
        let query_fp = self.embedding_to_fingerprint(embedding);

        let neighbors = coord.neighborhood(radius);

        let mut results: Vec<_> = neighbors.iter()
            .filter_map(|c| self.cells.get(&c.to_index()))
            .map(|cell| (cell, cell.similarity(&query_fp)))
            .filter(|(_, sim)| *sim > 0.5)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Get cell at coordinate
    pub fn get_cell(&self, coord: &Coord5D) -> Option<&CrystalCell> {
        self.cells.get(&coord.to_index())
    }

    /// Number of populated cells
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }
}

// ============================================================================
// DÉJÀ VU REINFORCEMENT LEARNING: Multipass ±3σ Overlay
// ============================================================================

/// Sigma band for multipass overlay
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SigmaBand {
    /// Within 1σ: strong signal
    Inner,     // 0 - 50
    /// 1σ to 2σ: moderate signal
    Middle,    // 50 - 100
    /// 2σ to 3σ: weak signal
    Outer,     // 100 - 150
    /// Beyond 3σ: noise (still tracked for anti-patterns)
    Beyond,    // > 150
}

impl SigmaBand {
    pub fn from_distance(dist: u32) -> Self {
        match dist {
            d if d <= ONE_SIGMA => SigmaBand::Inner,
            d if d <= TWO_SIGMA => SigmaBand::Middle,
            d if d <= THREE_SIGMA => SigmaBand::Outer,
            _ => SigmaBand::Beyond,
        }
    }

    pub fn weight(&self) -> f32 {
        match self {
            SigmaBand::Inner => 1.0,
            SigmaBand::Middle => 0.5,
            SigmaBand::Outer => 0.25,
            SigmaBand::Beyond => 0.0,
        }
    }
}

/// Déjà Vu observation across sigma bands
#[derive(Clone, Debug)]
pub struct DejaVuObservation {
    /// Item ID
    pub id: u64,
    /// Observations per sigma band
    pub band_counts: [u32; 4], // Inner, Middle, Outer, Beyond
    /// Total weighted score
    pub score: f32,
    /// First seen pass
    pub first_pass: usize,
    /// Last seen pass
    pub last_pass: usize,
    /// The "déjà vu strength" - how strongly we feel we've seen this
    pub deja_vu_strength: f32,
}

impl DejaVuObservation {
    pub fn new(id: u64, pass: usize) -> Self {
        Self {
            id,
            band_counts: [0; 4],
            score: 0.0,
            first_pass: pass,
            last_pass: pass,
            deja_vu_strength: 0.0,
        }
    }

    /// Record observation in a sigma band
    pub fn observe(&mut self, band: SigmaBand, pass: usize) {
        let idx = match band {
            SigmaBand::Inner => 0,
            SigmaBand::Middle => 1,
            SigmaBand::Outer => 2,
            SigmaBand::Beyond => 3,
        };
        self.band_counts[idx] += 1;
        self.score += band.weight();
        self.last_pass = pass;

        // Déjà vu strengthens with:
        // 1. Observations across multiple bands (breadth)
        // 2. Multiple observations in same band (depth)
        // 3. Spread across passes (temporal distribution)
        let breadth = self.band_counts.iter().filter(|&&c| c > 0).count() as f32;
        let depth = self.band_counts.iter().sum::<u32>() as f32;
        let temporal = (self.last_pass - self.first_pass + 1) as f32;

        self.deja_vu_strength = (breadth * depth.sqrt() * temporal.sqrt()) / 10.0;
    }

    /// Is this a strong déjà vu candidate?
    pub fn is_strong_deja_vu(&self) -> bool {
        // Strong if: seen in multiple bands AND multiple passes
        let multi_band = self.band_counts.iter().filter(|&&c| c > 0).count() >= 2;
        let multi_pass = self.last_pass > self.first_pass;
        let significant_score = self.score >= 1.5;

        multi_band && multi_pass && significant_score
    }
}

/// Déjà Vu Reinforcement Learning Engine
pub struct DejaVuRL {
    /// Observations by item ID
    observations: HashMap<u64, DejaVuObservation>,
    /// Current pass number
    current_pass: usize,
    /// Learning rate
    learning_rate: f32,
    /// Discount factor (how much past observations matter)
    gamma: f32,
    /// Q-values for (state, action) pairs
    /// State = sigma band, Action = accept/reject
    q_table: HashMap<(SigmaBand, bool), f32>,
    /// Running reward average and count (replaces unbounded Vec<f32>)
    reward_sum: f64,
    reward_count: u64,
}

impl DejaVuRL {
    pub fn new(learning_rate: f32, gamma: f32) -> Self {
        Self {
            observations: HashMap::new(),
            current_pass: 0,
            learning_rate,
            gamma,
            q_table: HashMap::new(),
            reward_sum: 0.0,
            reward_count: 0,
        }
    }

    /// Start a new pass
    pub fn begin_pass(&mut self) {
        self.current_pass += 1;
    }

    /// Observe an item at a given distance
    pub fn observe(&mut self, id: u64, distance: u32) {
        let band = SigmaBand::from_distance(distance);

        let obs = self.observations
            .entry(id)
            .or_insert_with(|| DejaVuObservation::new(id, self.current_pass));
        obs.observe(band, self.current_pass);
    }

    /// Run a complete multipass search
    pub fn multipass_search(
        &mut self,
        query: &BitpackedVector,
        candidates: &[(u64, BitpackedVector)],
        num_passes: usize,
    ) -> Vec<(u64, f32)> {
        // Clear previous observations
        self.observations.clear();

        for pass in 0..num_passes {
            self.current_pass = pass;

            // Rotate query slightly for each pass (different perspective)
            let rotated = query.rotate_bits(pass * 7);

            for (id, fp) in candidates {
                let dist = hamming_distance_scalar(&rotated, fp);
                let band = SigmaBand::from_distance(dist);

                // Only observe if within 3σ
                if !matches!(band, SigmaBand::Beyond) {
                    self.observe(*id, dist);
                }
            }
        }

        // Collect and rank by déjà vu strength
        let mut results: Vec<_> = self.observations.iter()
            .map(|(&id, obs)| (id, obs.deja_vu_strength))
            .filter(|(_, strength)| *strength > 0.1)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Get strong déjà vu candidates
    pub fn strong_deja_vu(&self) -> Vec<&DejaVuObservation> {
        self.observations.values()
            .filter(|obs| obs.is_strong_deja_vu())
            .collect()
    }

    /// Provide reward feedback for learning
    pub fn reward(&mut self, id: u64, was_correct: bool) {
        let reward = if was_correct { 1.0 } else { -1.0 };
        self.reward_sum += reward as f64;
        self.reward_count += 1;

        // Update Q-values based on which band this item was primarily in
        if let Some(obs) = self.observations.get(&id) {
            let primary_band = if obs.band_counts[0] > 0 {
                SigmaBand::Inner
            } else if obs.band_counts[1] > 0 {
                SigmaBand::Middle
            } else {
                SigmaBand::Outer
            };

            // Q-learning update: Q(s,a) += α * (r + γ * max_a' Q(s',a') - Q(s,a))
            let key = (primary_band, was_correct);
            let old_q = *self.q_table.get(&key).unwrap_or(&0.0);
            let new_q = old_q + self.learning_rate * (reward - old_q);
            self.q_table.insert(key, new_q);
        }
    }

    /// Get learned threshold for accepting items in each band
    pub fn learned_policy(&self) -> HashMap<SigmaBand, f32> {
        let mut policy = HashMap::new();

        for band in [SigmaBand::Inner, SigmaBand::Middle, SigmaBand::Outer] {
            let accept_q = *self.q_table.get(&(band, true)).unwrap_or(&0.0);
            let reject_q = *self.q_table.get(&(band, false)).unwrap_or(&0.0);

            // Probability of accepting = softmax
            let prob = 1.0 / (1.0 + (-accept_q + reject_q).exp());
            policy.insert(band, prob);
        }

        policy
    }

    /// Get observation statistics
    pub fn stats(&self) -> DejaVuStats {
        let total_obs = self.observations.len();
        let strong_count = self.observations.values()
            .filter(|o| o.is_strong_deja_vu())
            .count();

        let avg_strength = if total_obs > 0 {
            self.observations.values()
                .map(|o| o.deja_vu_strength)
                .sum::<f32>() / total_obs as f32
        } else {
            0.0
        };

        DejaVuStats {
            total_observations: total_obs,
            strong_deja_vu_count: strong_count,
            average_strength: avg_strength,
            passes_completed: self.current_pass + 1,
            total_rewards: self.reward_count as usize,
            average_reward: if self.reward_count == 0 {
                0.0
            } else {
                (self.reward_sum / self.reward_count as f64) as f32
            },
        }
    }
}

/// Déjà Vu statistics
#[derive(Clone, Debug)]
pub struct DejaVuStats {
    pub total_observations: usize,
    pub strong_deja_vu_count: usize,
    pub average_strength: f32,
    pub passes_completed: usize,
    pub total_rewards: usize,
    pub average_reward: f32,
}

// ============================================================================
// TRUTH MARKERS + ORTHOGONAL SUPERPOSITION CLEANING
// ============================================================================

/// Truth marker with confidence
#[derive(Clone, Debug)]
pub struct TruthMarker {
    /// Fingerprint being marked
    pub fingerprint: BitpackedVector,
    /// Truth value (0.0 = false, 1.0 = true)
    pub truth: f32,
    /// Confidence in this truth value
    pub confidence: f32,
    /// Count of supporting evidence (no need to store full vectors)
    pub evidence_for_count: usize,
    /// Count of counter-evidence
    pub evidence_against_count: usize,
    /// Bundled support fingerprint (majority of all support evidence)
    pub support_bundle: BitpackedVector,
    /// Bundled counter fingerprint (majority of all counter evidence)
    pub counter_bundle: BitpackedVector,
}

impl TruthMarker {
    pub fn new(fingerprint: BitpackedVector) -> Self {
        Self {
            fingerprint: fingerprint.clone(),
            truth: 0.5, // Unknown
            confidence: 0.0,
            evidence_for_count: 0,
            evidence_against_count: 0,
            support_bundle: BitpackedVector::zero(),
            counter_bundle: BitpackedVector::zero(),
        }
    }

    /// Add supporting evidence
    pub fn add_support(&mut self, evidence: BitpackedVector) {
        self.evidence_for_count += 1;
        // Incremental bundle: weighted merge with existing bundle
        if self.evidence_for_count == 1 {
            self.support_bundle = evidence;
        } else {
            let refs = vec![&self.support_bundle, &evidence];
            self.support_bundle = BitpackedVector::bundle(&refs);
        }
        self.update_truth();
    }

    /// Add counter-evidence
    pub fn add_counter(&mut self, evidence: BitpackedVector) {
        self.evidence_against_count += 1;
        if self.evidence_against_count == 1 {
            self.counter_bundle = evidence;
        } else {
            let refs = vec![&self.counter_bundle, &evidence];
            self.counter_bundle = BitpackedVector::bundle(&refs);
        }
        self.update_truth();
    }

    /// Update truth value based on evidence
    fn update_truth(&mut self) {
        let support = self.evidence_for_count as f32;
        let counter = self.evidence_against_count as f32;
        let total = support + counter;

        if total > 0.0 {
            self.truth = support / total;
            // Confidence increases with more evidence
            self.confidence = (total / 10.0).min(1.0);
        }
    }

    /// Is this considered true?
    pub fn is_true(&self) -> bool {
        self.truth > 0.5 && self.confidence > 0.3
    }

    /// Is this considered false?
    pub fn is_false(&self) -> bool {
        self.truth < 0.5 && self.confidence > 0.3
    }

    /// Is this uncertain?
    pub fn is_uncertain(&self) -> bool {
        self.confidence <= 0.3
    }
}

/// Orthogonal interference patterns for cleaning
pub struct OrthogonalBasis {
    /// Basis vectors (orthogonal or near-orthogonal)
    basis: Vec<BitpackedVector>,
    /// Interference threshold (distance to consider as interference)
    interference_threshold: u32,
}

impl OrthogonalBasis {
    /// Create basis with n orthogonal vectors
    pub fn new(n: usize) -> Self {
        // Generate pseudo-orthogonal vectors using golden ratio seeding
        let golden = 0x9E3779B97F4A7C15u64;
        let basis: Vec<_> = (0..n)
            .map(|i| BitpackedVector::random(golden.wrapping_mul(i as u64 + 1)))
            .collect();

        Self {
            basis,
            interference_threshold: TWO_SIGMA,
        }
    }

    /// Create from known interference patterns
    pub fn from_interference(patterns: Vec<BitpackedVector>) -> Self {
        Self {
            basis: patterns,
            interference_threshold: TWO_SIGMA,
        }
    }

    /// Project signal onto basis and identify interference components
    pub fn decompose(&self, signal: &BitpackedVector) -> Decomposition {
        let mut components = Vec::new();
        let mut interference = Vec::new();

        for (i, basis_vec) in self.basis.iter().enumerate() {
            let dist = hamming_distance_scalar(signal, basis_vec);
            let similarity = 1.0 - (dist as f32 / VECTOR_BITS as f32);

            components.push((i, similarity));

            // If strongly correlated with a basis vector, it might be interference
            if dist < self.interference_threshold {
                interference.push(i);
            }
        }

        Decomposition {
            components,
            interference_indices: interference,
        }
    }
}

/// Decomposition result
#[derive(Clone, Debug)]
pub struct Decomposition {
    /// Similarity to each basis vector
    pub components: Vec<(usize, f32)>,
    /// Indices of interfering basis vectors
    pub interference_indices: Vec<usize>,
}

/// Superposition cleaner: removes interference from bundled signals
pub struct SuperpositionCleaner {
    /// Known interference patterns
    interference_basis: OrthogonalBasis,
    /// Truth markers for validation
    truth_markers: HashMap<u64, TruthMarker>,
    /// Cleaning strength (0.0 = no cleaning, 1.0 = aggressive)
    strength: f32,
}

impl SuperpositionCleaner {
    pub fn new(strength: f32) -> Self {
        Self {
            interference_basis: OrthogonalBasis::new(0),
            truth_markers: HashMap::new(),
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Register known interference pattern
    /// Maximum interference patterns to prevent O(n) per clean and unbounded growth
    const MAX_INTERFERENCE: usize = 64;

    pub fn register_interference(&mut self, pattern: BitpackedVector) {
        if self.interference_basis.basis.len() >= Self::MAX_INTERFERENCE {
            // Evict oldest pattern
            self.interference_basis.basis.remove(0);
        }
        self.interference_basis.basis.push(pattern);
    }

    /// Register truth marker
    pub fn register_truth(&mut self, id: u64, marker: TruthMarker) {
        self.truth_markers.insert(id, marker);
    }

    /// Clean a signal by removing interference
    pub fn clean(&self, signal: &BitpackedVector) -> CleanedSignal {
        if self.interference_basis.basis.is_empty() {
            return CleanedSignal {
                original: signal.clone(),
                cleaned: signal.clone(),
                removed_interference: Vec::new(),
                cleaning_delta: 0,
            };
        }

        let decomp = self.interference_basis.decompose(signal);
        let mut cleaned = signal.clone();
        let mut removed = Vec::new();

        // Remove interference by XORing with interfering basis vectors
        // (XOR is self-inverse, so this "subtracts" the interference)
        for &idx in &decomp.interference_indices {
            let interference_vec = &self.interference_basis.basis[idx];
            let similarity = decomp.components[idx].1;

            // Only remove if similarity exceeds threshold
            if similarity > 0.5 + (self.strength * 0.3) {
                cleaned = cleaned.xor(interference_vec);
                removed.push(idx);
            }
        }

        let delta = hamming_distance_scalar(signal, &cleaned);

        CleanedSignal {
            original: signal.clone(),
            cleaned,
            removed_interference: removed,
            cleaning_delta: delta,
        }
    }

    /// Clean multiple signals and return consistent components
    pub fn clean_bundle(&self, signals: &[&BitpackedVector]) -> BitpackedVector {
        // Clean each signal individually
        let cleaned: Vec<BitpackedVector> = signals.iter()
            .map(|s| self.clean(s).cleaned)
            .collect();

        // Bundle the cleaned signals
        let refs: Vec<&BitpackedVector> = cleaned.iter().collect();
        BitpackedVector::bundle(&refs)
    }

    /// Validate signal against truth markers
    pub fn validate(&self, signal: &BitpackedVector, expected_id: u64) -> ValidationResult {
        let marker = self.truth_markers.get(&expected_id);

        if let Some(m) = marker {
            let similarity = 1.0 - (hamming_distance_scalar(signal, &m.fingerprint) as f32
                / VECTOR_BITS as f32);

            ValidationResult {
                is_valid: similarity > 0.8 && m.is_true(),
                truth_value: m.truth,
                confidence: m.confidence,
                similarity,
            }
        } else {
            ValidationResult {
                is_valid: false,
                truth_value: 0.5,
                confidence: 0.0,
                similarity: 0.0,
            }
        }
    }
}

/// Result of signal cleaning
#[derive(Clone, Debug)]
pub struct CleanedSignal {
    pub original: BitpackedVector,
    pub cleaned: BitpackedVector,
    pub removed_interference: Vec<usize>,
    pub cleaning_delta: u32,
}

/// Result of truth validation
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub truth_value: f32,
    pub confidence: f32,
    pub similarity: f32,
}

// ============================================================================
// UNIFIED CRYSTAL-DEJAVU-TRUTH PIPELINE
// ============================================================================

/// Unified pipeline combining all three systems
pub struct CrystalDejaVuTruth {
    /// Sentence crystal for embedding → fingerprint
    pub crystal: SentenceCrystal,
    /// Déjà vu RL for multipass discovery
    pub deja_vu: DejaVuRL,
    /// Superposition cleaner for truth validation
    pub cleaner: SuperpositionCleaner,
}

impl CrystalDejaVuTruth {
    pub fn new() -> Self {
        Self {
            crystal: SentenceCrystal::jina_v3(),
            deja_vu: DejaVuRL::new(0.1, 0.95),
            cleaner: SuperpositionCleaner::new(0.5),
        }
    }

    /// Full pipeline: embed → crystalize → multipass search → clean → validate
    pub fn process(
        &mut self,
        query_embedding: &[f32],
        candidates: &[(u64, Vec<f32>)],
        num_passes: usize,
    ) -> Vec<PipelineResult> {
        // Convert query to fingerprint
        let query_coord = self.crystal.projection.project(query_embedding);
        let query_fp = self.crystal.embedding_to_fingerprint(query_embedding);

        // Convert candidates to fingerprints
        let candidate_fps: Vec<(u64, BitpackedVector)> = candidates.iter()
            .map(|(id, emb)| (*id, self.crystal.embedding_to_fingerprint(emb)))
            .collect();

        // Run multipass déjà vu search
        let deja_vu_results = self.deja_vu.multipass_search(&query_fp, &candidate_fps, num_passes);

        // Clean and validate results
        let mut results = Vec::new();

        for (id, strength) in deja_vu_results {
            if let Some((_, fp)) = candidate_fps.iter().find(|(cid, _)| *cid == id) {
                let cleaned = self.cleaner.clean(fp);
                let validation = self.cleaner.validate(&cleaned.cleaned, id);

                results.push(PipelineResult {
                    id,
                    deja_vu_strength: strength,
                    cleaning_delta: cleaned.cleaning_delta,
                    truth_value: validation.truth_value,
                    confidence: validation.confidence,
                    final_score: strength * validation.confidence.max(0.1),
                });
            }
        }

        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
        results
    }
}

impl Default for CrystalDejaVuTruth {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from the unified pipeline
#[derive(Clone, Debug)]
pub struct PipelineResult {
    pub id: u64,
    pub deja_vu_strength: f32,
    pub cleaning_delta: u32,
    pub truth_value: f32,
    pub confidence: f32,
    pub final_score: f32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord5d() {
        let coord = Coord5D::new(1, 2, 3, 4, 0);
        let idx = coord.to_index();
        let back = Coord5D::from_index(idx);
        assert_eq!(coord, back);

        let neighbor = Coord5D::new(1, 2, 3, 3, 0);
        assert_eq!(coord.distance(&neighbor), 1);
    }

    #[test]
    fn test_projection() {
        let proj = ProjectionMatrix::new(1024, 42);

        let embedding: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) - 0.5).collect();
        let coord = proj.project(&embedding);

        // Should be valid coordinate
        assert!(coord.dims.iter().all(|&d| d < 5));
    }

    #[test]
    fn test_deja_vu_rl() {
        let mut rl = DejaVuRL::new(0.1, 0.95);

        // Create test query and candidates
        let query = BitpackedVector::random(42);

        let candidates: Vec<(u64, BitpackedVector)> = (0..20)
            .map(|i| {
                let mut fp = query.clone();
                fp.flip_random_bits(i * 10, i as u64); // Varying distances
                (i as u64, fp)
            })
            .collect();

        let results = rl.multipass_search(&query, &candidates, 5);

        // Closest candidates should have highest déjà vu strength
        assert!(!results.is_empty());
        println!("Déjà vu results: {:?}", &results[..3.min(results.len())]);
    }

    #[test]
    fn test_truth_marker() {
        let mut marker = TruthMarker::new(BitpackedVector::random(1));

        // Add evidence
        for i in 0..5 {
            marker.add_support(BitpackedVector::random(100 + i));
        }
        for i in 0..2 {
            marker.add_counter(BitpackedVector::random(200 + i));
        }

        assert!(marker.is_true()); // 5:2 in favor
        println!("Truth: {}, Confidence: {}", marker.truth, marker.confidence);
    }

    #[test]
    fn test_superposition_cleaning() {
        let mut cleaner = SuperpositionCleaner::new(0.7);

        // Register some interference patterns
        let interference1 = BitpackedVector::random(999);
        let interference2 = BitpackedVector::random(888);
        cleaner.register_interference(interference1.clone());
        cleaner.register_interference(interference2.clone());

        // Create signal with interference
        let pure_signal = BitpackedVector::random(42);
        let noisy_signal = pure_signal.xor(&interference1); // Add interference

        let cleaned = cleaner.clean(&noisy_signal);

        // Cleaning should have removed interference
        assert!(cleaned.cleaning_delta > 0);
        println!("Cleaning delta: {}", cleaned.cleaning_delta);
    }

    #[test]
    fn test_sigma_bands() {
        assert_eq!(SigmaBand::from_distance(30), SigmaBand::Inner);
        assert_eq!(SigmaBand::from_distance(75), SigmaBand::Middle);
        assert_eq!(SigmaBand::from_distance(120), SigmaBand::Outer);
        assert_eq!(SigmaBand::from_distance(200), SigmaBand::Beyond);
    }
}
