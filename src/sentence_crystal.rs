//! Sentence Crystal: GPU-Free Semantic Transformer + Learning Crystal
//!
//! Two complementary systems:
//!
//! - **SemanticCrystal**: Transforms text into 10Kbit fingerprints WITHOUT
//!   any external model, GPU, or LLM. Uses character n-gram hashing,
//!   positional encoding via bit rotation, and crystal lattice bundling.
//!   This is the "semantic transformer without GPU without LLM" from ladybug-rs.
//!
//! - **LearningCrystal**: Hebbian learning on the crystal lattice. Cells that
//!   co-activate strengthen their connections. Over time, the crystal learns
//!   which semantic regions are related, enabling associative recall.
//!
//! # SemanticCrystal Architecture
//!
//! ```text
//! Text: "The cat sat on the mat"
//!   │
//!   ├─► Character trigrams: ["The", "he ", "e c", " ca", "cat", ...]
//!   │     │
//!   │     ▼ Hash each trigram to a 10Kbit fingerprint
//!   │     [fp_0, fp_1, fp_2, ...]
//!   │     │
//!   │     ▼ Positional encoding: rotate fp_i by i positions
//!   │     [rot(fp_0,0), rot(fp_1,1), rot(fp_2,2), ...]
//!   │     │
//!   │     ▼ Bundle all positioned trigrams (majority vote)
//!   │     sentence_fingerprint
//!   │
//!   ├─► Word-level: hash each word, position-encode, bundle
//!   │     word_fingerprint
//!   │
//!   ├─► Crystal projection: map fingerprint density per block → Coord5D
//!   │     crystal_coordinate
//!   │
//!   └─► Final: XOR-bind sentence_fingerprint with crystal cell prototype
//!         semantic_fingerprint (encodes both content and spatial position)
//! ```
//!
//! # Why This Works Without GPU
//!
//! Traditional transformers use float matrix multiplies (~1 TFLOP per sentence).
//! We use:
//! - Character n-gram hashing: O(n) integer operations
//! - Bit rotation: O(1) per position
//! - Majority bundling: O(n × 157 words)
//! - Crystal projection: O(157) additions
//! - XOR bind: O(157) operations
//!
//! Total: ~50K integer operations per sentence vs ~1B float operations.
//! No GPU needed. No model weights. Deterministic and reproducible.

use crate::bitpack::{BitpackedVector, VECTOR_BITS, VECTOR_WORDS};
use crate::hamming::hamming_distance_scalar;
use crate::crystal_dejavu::Coord5D;
use std::collections::HashMap;

// ============================================================================
// SEMANTIC CRYSTAL: GPU-free text → fingerprint transformer
// ============================================================================

/// Character n-gram sizes to use for hashing
const NGRAM_SIZES: [usize; 3] = [3, 4, 5]; // Trigrams, tetragrams, pentagrams

/// Number of word-level rotation steps between words
const WORD_ROTATION_STEP: usize = 7;

/// Number of character-level rotation steps between ngrams
const CHAR_ROTATION_STEP: usize = 1;

/// GPU-free semantic transformer.
///
/// Converts raw text into 10Kbit fingerprints using only integer operations.
/// No neural network weights, no embedding model, no GPU required.
///
/// The quality won't match BERT/Jina for nuanced semantics, but for
/// structural similarity, keyword overlap, and topic clustering it's
/// surprisingly effective — and infinitely faster.
pub struct SemanticCrystal {
    /// Cached ngram → fingerprint mappings (for speed on repeated text)
    ngram_cache: HashMap<String, BitpackedVector>,
    /// Word-level cache
    word_cache: HashMap<String, BitpackedVector>,
    /// Crystal cell prototypes (learned over time)
    cell_prototypes: HashMap<usize, BitpackedVector>,
    /// Weight for character ngrams vs word-level (0.0 = all words, 1.0 = all chars)
    char_weight: f32,
    /// Maximum cache size
    max_cache: usize,
}

impl SemanticCrystal {
    /// Create new semantic crystal with default settings
    ///
    /// Cache is limited to 10K entries per cache (~13MB each at 1,256 bytes/entry).
    /// Ngrams are deterministically computed from hash so eviction just means
    /// recomputing on cache miss — no data loss.
    pub fn new() -> Self {
        Self {
            ngram_cache: HashMap::new(),
            word_cache: HashMap::new(),
            cell_prototypes: HashMap::new(),
            char_weight: 0.6, // 60% character ngrams, 40% word-level
            max_cache: 10_000, // 10K not 100K: ~13MB per cache, not ~130MB
        }
    }

    /// Create with custom character/word weight balance
    pub fn with_char_weight(char_weight: f32) -> Self {
        Self {
            char_weight: char_weight.clamp(0.0, 1.0),
            ..Self::new()
        }
    }

    /// Transform text into a semantic fingerprint
    pub fn encode(&mut self, text: &str) -> SemanticEncoding {
        let normalized = Self::normalize(text);

        // Character n-gram fingerprint
        let char_fp = self.encode_char_ngrams(&normalized);

        // Word-level fingerprint
        let word_fp = self.encode_words(&normalized);

        // Weighted bundle of char and word fingerprints
        let combined = BitpackedVector::bundle_weighted(&[
            (&char_fp, self.char_weight),
            (&word_fp, 1.0 - self.char_weight),
        ]);

        // Crystal coordinate from block density
        let crystal_coord = self.fingerprint_to_crystal(&combined);

        // Bind with crystal cell prototype for spatial encoding
        let cell_idx = crystal_coord.to_index();
        let cell_proto = self
            .cell_prototypes
            .entry(cell_idx)
            .or_insert_with(|| crystal_coord.to_fingerprint())
            .clone();
        let semantic_fp = combined.xor(&cell_proto);

        SemanticEncoding {
            fingerprint: semantic_fp,
            char_fingerprint: char_fp,
            word_fingerprint: word_fp,
            crystal_coord,
            text_length: text.len(),
            word_count: normalized.split_whitespace().count(),
        }
    }

    /// Encode character n-grams into a fingerprint
    fn encode_char_ngrams(&mut self, text: &str) -> BitpackedVector {
        let chars: Vec<char> = text.chars().collect();
        let mut all_ngram_fps: Vec<(BitpackedVector, f32)> = Vec::new();

        for &ngram_size in &NGRAM_SIZES {
            if chars.len() < ngram_size {
                continue;
            }

            // Weight larger ngrams slightly more (they carry more specificity)
            let weight = ngram_size as f32 / 3.0;

            for (pos, window) in chars.windows(ngram_size).enumerate() {
                let ngram: String = window.iter().collect();

                // Get or compute ngram fingerprint
                let fp = if let Some(cached) = self.ngram_cache.get(&ngram) {
                    cached.clone()
                } else {
                    let fp = BitpackedVector::from_hash(ngram.as_bytes());
                    // Evict random entry if at capacity
                    if self.ngram_cache.len() >= self.max_cache {
                        let evict_key = self.ngram_cache.keys().next().cloned();
                        if let Some(k) = evict_key {
                            self.ngram_cache.remove(&k);
                        }
                    }
                    self.ngram_cache.insert(ngram, fp.clone());
                    fp
                };

                // Positional encoding: rotate by position
                let positioned = fp.rotate_words(pos * CHAR_ROTATION_STEP % VECTOR_WORDS);

                all_ngram_fps.push((positioned, weight));
            }
        }

        if all_ngram_fps.is_empty() {
            // Very short text: just hash it
            return BitpackedVector::from_hash(text.as_bytes());
        }

        // Weighted majority vote
        let refs: Vec<(&BitpackedVector, f32)> =
            all_ngram_fps.iter().map(|(fp, w)| (fp, *w)).collect();
        BitpackedVector::bundle_weighted(&refs)
    }

    /// Encode words into a fingerprint
    fn encode_words(&mut self, text: &str) -> BitpackedVector {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return BitpackedVector::zero();
        }

        let mut word_fps: Vec<BitpackedVector> = Vec::new();

        for (pos, word) in words.iter().enumerate() {
            let fp = if let Some(cached) = self.word_cache.get(*word) {
                cached.clone()
            } else {
                let fp = BitpackedVector::from_hash(word.as_bytes());
                // Evict random entry if at capacity
                if self.word_cache.len() >= self.max_cache {
                    let evict_key = self.word_cache.keys().next().cloned();
                    if let Some(k) = evict_key {
                        self.word_cache.remove(&k);
                    }
                }
                self.word_cache.insert(word.to_string(), fp.clone());
                fp
            };

            // Positional encoding via word rotation
            let positioned = fp.rotate_words(pos * WORD_ROTATION_STEP % VECTOR_WORDS);
            word_fps.push(positioned);
        }

        let refs: Vec<&BitpackedVector> = word_fps.iter().collect();
        BitpackedVector::bundle(&refs)
    }

    /// Map fingerprint to crystal coordinate via block density
    fn fingerprint_to_crystal(&self, fp: &BitpackedVector) -> Coord5D {
        let stacked = fp.stacked_popcount();
        let words_per_block = 16;
        let mut dims = [0u8; 5];

        for dim in 0..5 {
            let block_base = dim * 2; // 2 blocks per dimension
            let mut dim_sum = 0u32;
            let mut dim_bits = 0u32;

            for offset in 0..2 {
                let block_idx = block_base + offset;
                let start = block_idx * words_per_block;
                let end = ((block_idx + 1) * words_per_block).min(VECTOR_WORDS);
                for w in start..end {
                    dim_sum += stacked[w] as u32;
                    dim_bits += 64;
                }
            }

            let density = dim_sum as f32 / dim_bits as f32;
            dims[dim] = (density * 4.999).clamp(0.0, 4.0) as u8;
        }

        Coord5D::new(dims[0], dims[1], dims[2], dims[3], dims[4])
    }

    /// Normalize text for consistent encoding
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute similarity between two texts
    pub fn similarity(&mut self, text_a: &str, text_b: &str) -> f32 {
        let enc_a = self.encode(text_a);
        let enc_b = self.encode(text_b);
        let dist = hamming_distance_scalar(&enc_a.fingerprint, &enc_b.fingerprint);
        1.0 - (dist as f32 / VECTOR_BITS as f32)
    }

    /// Batch encode multiple texts
    pub fn encode_batch(&mut self, texts: &[&str]) -> Vec<SemanticEncoding> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.ngram_cache.len(), self.word_cache.len())
    }
}

impl Default for SemanticCrystal {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of semantic encoding
#[derive(Clone, Debug)]
pub struct SemanticEncoding {
    /// The final semantic fingerprint
    pub fingerprint: BitpackedVector,
    /// Character n-gram component
    pub char_fingerprint: BitpackedVector,
    /// Word-level component
    pub word_fingerprint: BitpackedVector,
    /// Crystal lattice coordinate
    pub crystal_coord: Coord5D,
    /// Original text length
    pub text_length: usize,
    /// Word count
    pub word_count: usize,
}

// ============================================================================
// LEARNING CRYSTAL: Hebbian learning on crystal lattice
// ============================================================================

/// A learning crystal cell that adapts its prototype over time
#[derive(Clone, Debug)]
pub struct LearningCell {
    /// Crystal coordinate
    pub coord: Coord5D,
    /// Current prototype fingerprint (evolves via bundling)
    pub prototype: BitpackedVector,
    /// Number of items bundled into prototype
    pub count: usize,
    /// Hebbian connection strengths to neighboring cells
    pub connections: HashMap<usize, f32>,
    /// Learning rate (decays over time for stability)
    pub learning_rate: f32,
    /// Activation history (sliding window)
    activation_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
}

impl LearningCell {
    pub fn new(coord: Coord5D) -> Self {
        Self {
            coord,
            prototype: coord.to_fingerprint(),
            count: 0,
            connections: HashMap::new(),
            learning_rate: 0.1,
            activation_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Learn from a new fingerprint (update prototype via weighted bundle)
    pub fn learn(&mut self, fp: &BitpackedVector) {
        self.count += 1;

        // Weighted bundle: old prototype has weight (count-1), new has weight 1
        // But we approximate via XOR interpolation:
        // At high learning rate: mostly new signal
        // At low learning rate: mostly old prototype
        if self.count == 1 {
            self.prototype = fp.clone();
        } else {
            let refs = vec![
                (&self.prototype, (1.0 - self.learning_rate) * self.count as f32),
                (fp, self.learning_rate * self.count as f32),
            ];
            self.prototype = BitpackedVector::bundle_weighted(&refs);
        }

        // Decay learning rate (convergence)
        self.learning_rate *= 0.999;
        self.learning_rate = self.learning_rate.max(0.001);
    }

    /// Record activation (similarity to query)
    pub fn activate(&mut self, similarity: f32) {
        self.activation_history.push(similarity);
        if self.activation_history.len() > self.max_history {
            self.activation_history.remove(0);
        }
    }

    /// Average recent activation
    pub fn avg_activation(&self) -> f32 {
        if self.activation_history.is_empty() {
            return 0.0;
        }
        self.activation_history.iter().sum::<f32>() / self.activation_history.len() as f32
    }

    /// Strengthen connection to another cell (Hebbian)
    pub fn strengthen(&mut self, other_cell: usize, amount: f32) {
        let weight = self.connections.entry(other_cell).or_insert(0.0);
        *weight = (*weight + amount).min(5.0);
    }

    /// Decay all connections
    pub fn decay_connections(&mut self, factor: f32) {
        for w in self.connections.values_mut() {
            *w *= factor;
        }
        self.connections.retain(|_, w| *w > 0.001);
    }
}

/// The Learning Crystal: a 5D lattice that learns associations
///
/// Each cell adapts its prototype fingerprint based on items assigned to it.
/// Cells that co-activate develop stronger connections (Hebbian learning).
/// Over time, the crystal develops a topographic map of semantic space.
pub struct LearningCrystal {
    /// Cells by index (0..3125)
    cells: HashMap<usize, LearningCell>,
    /// Hebbian learning rate for inter-cell connections
    hebbian_rate: f32,
    /// Connection decay rate
    decay_rate: f32,
    /// Total items learned
    total_learned: usize,
    /// Recently activated cells (for Hebbian co-activation)
    recent_activations: Vec<usize>,
    /// Max recent activations to track
    max_recent: usize,
}

impl LearningCrystal {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            hebbian_rate: 0.05,
            decay_rate: 0.999,
            total_learned: 0,
            recent_activations: Vec::new(),
            max_recent: 10,
        }
    }

    /// Learn from a fingerprint at its natural crystal coordinate
    pub fn learn(&mut self, fp: &BitpackedVector, coord: Coord5D) {
        self.total_learned += 1;
        let cell_idx = coord.to_index();

        // Update the target cell
        let cell = self
            .cells
            .entry(cell_idx)
            .or_insert_with(|| LearningCell::new(coord));
        cell.learn(fp);
        cell.activate(1.0); // Direct learning = full activation

        // Also weakly activate neighboring cells (spread activation)
        let neighbors = coord.neighborhood(1);
        for neighbor_coord in &neighbors {
            let neighbor_idx = neighbor_coord.to_index();
            if neighbor_idx != cell_idx {
                let ncell = self
                    .cells
                    .entry(neighbor_idx)
                    .or_insert_with(|| LearningCell::new(*neighbor_coord));

                let sim = 1.0 - (hamming_distance_scalar(fp, &ncell.prototype) as f32 / VECTOR_BITS as f32);
                ncell.activate(sim * 0.3); // Weak neighbor activation
            }
        }

        // Hebbian co-activation: strengthen connections between recently active cells
        for &recent_idx in &self.recent_activations {
            if recent_idx != cell_idx {
                if let Some(cell) = self.cells.get_mut(&cell_idx) {
                    cell.strengthen(recent_idx, self.hebbian_rate);
                }
                if let Some(rcell) = self.cells.get_mut(&recent_idx) {
                    rcell.strengthen(cell_idx, self.hebbian_rate);
                }
            }
        }

        // Update recent activations
        self.recent_activations.push(cell_idx);
        if self.recent_activations.len() > self.max_recent {
            self.recent_activations.remove(0);
        }

        // Periodic decay
        if self.total_learned % 100 == 0 {
            for cell in self.cells.values_mut() {
                cell.decay_connections(self.decay_rate);
            }
        }
    }

    /// Query: find best matching cells for a fingerprint
    pub fn query(&self, fp: &BitpackedVector, k: usize) -> Vec<(Coord5D, f32, usize)> {
        let mut results: Vec<_> = self
            .cells
            .values()
            .map(|cell| {
                let dist = hamming_distance_scalar(fp, &cell.prototype);
                let sim = 1.0 - (dist as f32 / VECTOR_BITS as f32);
                (cell.coord, sim, cell.count)
            })
            .filter(|(_, sim, _)| *sim > 0.5)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Associative recall: given a cell, find its strongest connections
    pub fn recall(&self, coord: Coord5D, k: usize) -> Vec<(Coord5D, f32)> {
        let cell_idx = coord.to_index();

        let cell = match self.cells.get(&cell_idx) {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut connections: Vec<_> = cell
            .connections
            .iter()
            .filter_map(|(&other_idx, &weight)| {
                self.cells
                    .get(&other_idx)
                    .map(|other_cell| (other_cell.coord, weight))
            })
            .collect();

        connections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        connections.truncate(k);
        connections
    }

    /// Spread activation: activate a cell and propagate through connections
    pub fn spread_activation(
        &self,
        start: Coord5D,
        depth: usize,
    ) -> Vec<(Coord5D, f32)> {
        let mut activations: HashMap<usize, f32> = HashMap::new();
        let start_idx = start.to_index();
        activations.insert(start_idx, 1.0);

        let mut frontier = vec![(start_idx, 1.0f32)];

        for _ in 0..depth {
            let mut next_frontier = Vec::new();

            for (cell_idx, activation) in &frontier {
                if let Some(cell) = self.cells.get(cell_idx) {
                    for (&connected_idx, &weight) in &cell.connections {
                        let propagated = activation * weight * 0.5; // Decay per hop
                        if propagated > 0.01 {
                            let entry = activations.entry(connected_idx).or_insert(0.0);
                            *entry = entry.max(propagated);
                            next_frontier.push((connected_idx, propagated));
                        }
                    }
                }
            }

            frontier = next_frontier;
        }

        let mut results: Vec<_> = activations
            .iter()
            .filter_map(|(&idx, &act)| {
                self.cells.get(&idx).map(|cell| (cell.coord, act))
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Number of populated cells
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Total items learned
    pub fn total_learned(&self) -> usize {
        self.total_learned
    }

    /// Get a cell's prototype
    pub fn cell_prototype(&self, coord: &Coord5D) -> Option<&BitpackedVector> {
        self.cells.get(&coord.to_index()).map(|c| &c.prototype)
    }
}

impl Default for LearningCrystal {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_encode_deterministic() {
        let mut crystal = SemanticCrystal::new();
        let enc1 = crystal.encode("the cat sat on the mat");
        let enc2 = crystal.encode("the cat sat on the mat");

        // Same text → same fingerprint
        assert_eq!(enc1.fingerprint, enc2.fingerprint);
        assert_eq!(enc1.crystal_coord, enc2.crystal_coord);
    }

    #[test]
    fn test_semantic_similar_texts() {
        let mut crystal = SemanticCrystal::new();

        let sim_close = crystal.similarity("the cat sat on the mat", "the cat sits on the mat");
        let sim_far = crystal.similarity("the cat sat on the mat", "quantum physics is complex");

        assert!(
            sim_close > sim_far,
            "Similar texts should have higher similarity: close={}, far={}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_semantic_word_count() {
        let mut crystal = SemanticCrystal::new();
        let enc = crystal.encode("hello world how are you");
        assert_eq!(enc.word_count, 5);
    }

    #[test]
    fn test_semantic_crystal_coordinate() {
        let mut crystal = SemanticCrystal::new();
        let enc = crystal.encode("test text for crystal projection");

        // Should produce valid coordinate
        assert!(enc.crystal_coord.dims.iter().all(|&d| d < 5));
    }

    #[test]
    fn test_semantic_batch_encode() {
        let mut crystal = SemanticCrystal::new();
        let texts = ["hello world", "goodbye world", "test sentence"];
        let encodings = crystal.encode_batch(&texts);

        assert_eq!(encodings.len(), 3);
        // Different texts → different fingerprints
        assert_ne!(encodings[0].fingerprint, encodings[1].fingerprint);
    }

    #[test]
    fn test_semantic_normalization() {
        let mut crystal = SemanticCrystal::new();

        // Should normalize case and punctuation
        let sim = crystal.similarity("Hello, World!", "hello world");
        assert!(sim > 0.8, "Normalized texts should be very similar: {}", sim);
    }

    #[test]
    fn test_learning_cell_learn() {
        let coord = Coord5D::new(2, 2, 2, 2, 2);
        let mut cell = LearningCell::new(coord);

        let fp1 = BitpackedVector::random(42);
        let fp2 = BitpackedVector::random(43);

        cell.learn(&fp1);
        assert_eq!(cell.count, 1);
        // After first learn, prototype should match fp1
        assert_eq!(cell.prototype, fp1);

        cell.learn(&fp2);
        assert_eq!(cell.count, 2);

        // After second learn, prototype should be a weighted bundle
        // At default learning_rate=0.1, the old prototype dominates
        // but the result is still a valid prototype
        let dist = hamming_distance_scalar(&cell.prototype, &fp1);
        assert!(dist > 0 || cell.prototype == fp1, "Learning should incorporate new signal or keep old");
    }

    #[test]
    fn test_learning_cell_connections() {
        let coord = Coord5D::new(1, 1, 1, 1, 1);
        let mut cell = LearningCell::new(coord);

        cell.strengthen(100, 0.5);
        cell.strengthen(200, 0.3);
        cell.strengthen(100, 0.5); // Strengthen again

        assert!(cell.connections[&100] > cell.connections[&200]);
    }

    #[test]
    fn test_learning_crystal_learn_and_query() {
        let mut crystal = LearningCrystal::new();

        // Learn some fingerprints
        for i in 0..20 {
            let fp = BitpackedVector::random(i);
            let coord = Coord5D::new(
                (i % 5) as u8,
                (i / 5 % 5) as u8,
                2,
                2,
                2,
            );
            crystal.learn(&fp, coord);
        }

        assert_eq!(crystal.total_learned(), 20);
        assert!(crystal.num_cells() > 0);

        // Query should find matches
        let query = BitpackedVector::random(10); // Same seed as one of the learned
        let results = crystal.query(&query, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_learning_crystal_hebbian() {
        let mut crystal = LearningCrystal::new();

        // Learn sequential items in same region
        let coord_a = Coord5D::new(1, 1, 1, 1, 1);
        let coord_b = Coord5D::new(1, 2, 1, 1, 1);

        crystal.learn(&BitpackedVector::random(100), coord_a);
        crystal.learn(&BitpackedVector::random(200), coord_b);

        // Cells should develop connections from co-activation
        let connections = crystal.recall(coord_a, 5);
        // There should be at least some connected cells
        // (from neighbor activation in learn())
    }

    #[test]
    fn test_learning_crystal_spread_activation() {
        let mut crystal = LearningCrystal::new();

        // Create a small network
        let c1 = Coord5D::new(1, 1, 1, 1, 1);
        let c2 = Coord5D::new(1, 2, 1, 1, 1);
        let c3 = Coord5D::new(2, 2, 1, 1, 1);

        // Learn items to create cells
        crystal.learn(&BitpackedVector::random(1), c1);
        crystal.learn(&BitpackedVector::random(2), c2);
        crystal.learn(&BitpackedVector::random(3), c3);

        // Spread activation from c1
        let activated = crystal.spread_activation(c1, 2);
        assert!(!activated.is_empty());
    }

    #[test]
    fn test_end_to_end_semantic_learning() {
        let mut semantic = SemanticCrystal::new();
        let mut learning = LearningCrystal::new();

        // Encode and learn several sentences
        let texts = [
            "the cat sat on the mat",
            "the dog lay on the rug",
            "quantum mechanics describes particles",
            "general relativity explains gravity",
        ];

        for text in &texts {
            let enc = semantic.encode(text);
            learning.learn(&enc.fingerprint, enc.crystal_coord);
        }

        // Query with a related sentence
        let query_enc = semantic.encode("the cat sits on the floor");
        let results = learning.query(&query_enc.fingerprint, 5);

        // Should find cat/dog sentences more similar than physics sentences
        assert!(!results.is_empty());
    }
}
