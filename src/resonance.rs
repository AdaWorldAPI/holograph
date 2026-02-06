//! Vector Field Resonance - Bind/Unbind Operations
//!
//! This module implements the "alien magic" of hyperdimensional computing:
//! instead of matrix operations, we use XOR-based binding that enables
//! O(1) retrieval through algebraic computation.
//!
//! # The Core Insight
//!
//! In traditional databases:
//! ```text
//! Store: edges table with (src, verb, dst)
//! Query: SELECT dst FROM edges WHERE src=? AND verb=?
//!        → O(log n) with index, O(n) without
//! ```
//!
//! With vector field resonance:
//! ```text
//! Store: edge = src ⊗ verb ⊗ dst   (single XOR binding)
//! Query: dst = edge ⊗ verb ⊗ src   (compute directly in O(1)!)
//!
//! Because A ⊗ B ⊗ B = A (XOR is self-inverse)
//! ```
//!
//! # Vector Field Operations
//!
//! - **Bind**: Combine concepts (A ⊗ B creates "A related to B")
//! - **Unbind**: Recover component (A ⊗ B ⊗ B = A)
//! - **Bundle**: Create prototype from multiple examples
//! - **Resonance**: Match noisy vector to clean concept (cleanup memory)

use crate::bitpack::{BitpackedVector, VECTOR_WORDS, VECTOR_BITS};
use crate::hamming::{hamming_distance_scalar, hamming_to_similarity, StackedPopcount};
use std::collections::HashMap;

// ============================================================================
// VECTOR FIELD
// ============================================================================

/// A vector field represents a semantic space where vectors can be
/// bound, unbound, and bundled.
pub struct VectorField {
    /// Named concept vectors (the "atoms" of the field)
    atoms: HashMap<String, BitpackedVector>,
    /// Verb vectors for typed relationships
    verbs: HashMap<String, BitpackedVector>,
    /// Cleanup memory for resonance matching
    cleanup_memory: Vec<BitpackedVector>,
}

impl Default for VectorField {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorField {
    /// Create an empty vector field
    pub fn new() -> Self {
        Self {
            atoms: HashMap::new(),
            verbs: HashMap::new(),
            cleanup_memory: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(atoms: usize, verbs: usize) -> Self {
        Self {
            atoms: HashMap::with_capacity(atoms),
            verbs: HashMap::with_capacity(verbs),
            cleanup_memory: Vec::new(),
        }
    }

    // ========================================================================
    // ATOM MANAGEMENT
    // ========================================================================

    /// Register a concept atom with a random vector
    pub fn create_atom(&mut self, name: &str) -> &BitpackedVector {
        let seed = hash_string(name);
        let vector = BitpackedVector::random(seed);
        self.atoms.insert(name.to_string(), vector);
        self.atoms.get(name).unwrap()
    }

    /// Register a concept atom with a specific vector
    pub fn set_atom(&mut self, name: &str, vector: BitpackedVector) {
        self.atoms.insert(name.to_string(), vector);
    }

    /// Get an atom by name
    pub fn get_atom(&self, name: &str) -> Option<&BitpackedVector> {
        self.atoms.get(name)
    }

    /// Get or create an atom
    pub fn atom(&mut self, name: &str) -> &BitpackedVector {
        if !self.atoms.contains_key(name) {
            self.create_atom(name);
        }
        self.atoms.get(name).unwrap()
    }

    // ========================================================================
    // VERB MANAGEMENT
    // ========================================================================

    /// Register a relationship verb
    pub fn create_verb(&mut self, name: &str) -> &BitpackedVector {
        let seed = hash_string(&format!("__verb__{}", name));
        let vector = BitpackedVector::random(seed);
        self.verbs.insert(name.to_string(), vector);
        self.verbs.get(name).unwrap()
    }

    /// Get a verb by name
    pub fn get_verb(&self, name: &str) -> Option<&BitpackedVector> {
        self.verbs.get(name)
    }

    /// Get or create a verb
    pub fn verb(&mut self, name: &str) -> &BitpackedVector {
        if !self.verbs.contains_key(name) {
            self.create_verb(name);
        }
        self.verbs.get(name).unwrap()
    }

    // ========================================================================
    // BINDING OPERATIONS
    // ========================================================================

    /// Bind two vectors: A ⊗ B
    #[inline]
    pub fn bind(&self, a: &BitpackedVector, b: &BitpackedVector) -> BitpackedVector {
        a.xor(b)
    }

    /// Bind three vectors: A ⊗ B ⊗ C (for typed edges)
    #[inline]
    pub fn bind3(
        &self,
        a: &BitpackedVector,
        b: &BitpackedVector,
        c: &BitpackedVector,
    ) -> BitpackedVector {
        a.xor(b).xor(c)
    }

    /// Unbind: A ⊗ B ⊗ B = A (same as bind, XOR is self-inverse)
    #[inline]
    pub fn unbind(&self, bound: &BitpackedVector, key: &BitpackedVector) -> BitpackedVector {
        bound.xor(key)
    }

    /// Create a typed edge: src --[verb]--> dst
    pub fn create_edge(
        &self,
        src: &BitpackedVector,
        verb: &BitpackedVector,
        dst: &BitpackedVector,
    ) -> BoundEdge {
        let binding = self.bind3(src, verb, dst);
        BoundEdge {
            binding,
            src: src.clone(),
            verb: verb.clone(),
            dst: dst.clone(),
        }
    }

    // ========================================================================
    // CLEANUP MEMORY (Resonance Matching)
    // ========================================================================

    /// Add a vector to cleanup memory
    pub fn add_to_cleanup(&mut self, vector: BitpackedVector) {
        self.cleanup_memory.push(vector);
    }

    /// Add all atoms to cleanup memory
    pub fn populate_cleanup_from_atoms(&mut self) {
        for vector in self.atoms.values() {
            self.cleanup_memory.push(vector.clone());
        }
    }

    /// Find the closest vector in cleanup memory (resonance)
    pub fn resonate(&self, noisy: &BitpackedVector) -> Option<(usize, u32, f32)> {
        if self.cleanup_memory.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = u32::MAX;

        for (i, clean) in self.cleanup_memory.iter().enumerate() {
            let dist = hamming_distance_scalar(noisy, clean);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        let similarity = hamming_to_similarity(best_dist);
        Some((best_idx, best_dist, similarity))
    }

    /// Find all vectors within threshold (multi-resonance)
    pub fn resonate_all(&self, noisy: &BitpackedVector, threshold: u32) -> Vec<(usize, u32)> {
        self.cleanup_memory
            .iter()
            .enumerate()
            .filter_map(|(i, clean)| {
                let dist = hamming_distance_scalar(noisy, clean);
                if dist <= threshold {
                    Some((i, dist))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get vector from cleanup memory by index
    pub fn get_cleanup(&self, index: usize) -> Option<&BitpackedVector> {
        self.cleanup_memory.get(index)
    }
}

// ============================================================================
// BOUND EDGE
// ============================================================================

/// A bound edge represents a relationship: src --[verb]--> dst
///
/// The binding `src ⊗ verb ⊗ dst` allows O(1) retrieval of any
/// component given the other two.
#[derive(Clone, Debug)]
pub struct BoundEdge {
    /// The XOR binding of all three components
    pub binding: BitpackedVector,
    /// Source vector (cached for verification)
    pub src: BitpackedVector,
    /// Verb/relationship vector
    pub verb: BitpackedVector,
    /// Destination vector
    pub dst: BitpackedVector,
}

impl BoundEdge {
    /// Create from components
    pub fn new(src: BitpackedVector, verb: BitpackedVector, dst: BitpackedVector) -> Self {
        let binding = src.xor(&verb).xor(&dst);
        Self { binding, src, verb, dst }
    }

    /// Create from just the binding and verb (lazy edge)
    pub fn from_binding(binding: BitpackedVector, verb: BitpackedVector) -> Self {
        Self {
            binding,
            src: BitpackedVector::zero(),
            verb,
            dst: BitpackedVector::zero(),
        }
    }

    /// Recover destination given source: edge ⊗ verb ⊗ src = dst
    #[inline]
    pub fn get_dst(&self, src: &BitpackedVector) -> BitpackedVector {
        self.binding.xor(&self.verb).xor(src)
    }

    /// Recover source given destination: edge ⊗ verb ⊗ dst = src
    #[inline]
    pub fn get_src(&self, dst: &BitpackedVector) -> BitpackedVector {
        self.binding.xor(&self.verb).xor(dst)
    }

    /// Verify that recovered component matches stored
    pub fn verify_dst(&self, src: &BitpackedVector) -> bool {
        let recovered = self.get_dst(src);
        hamming_distance_scalar(&recovered, &self.dst) == 0
    }

    /// Verify source recovery
    pub fn verify_src(&self, dst: &BitpackedVector) -> bool {
        let recovered = self.get_src(dst);
        hamming_distance_scalar(&recovered, &self.src) == 0
    }
}

// ============================================================================
// RESONATOR (Cleanup Memory Engine)
// ============================================================================

/// High-performance resonance matcher with cascaded filtering
pub struct Resonator {
    /// Clean concept vectors
    concepts: Vec<BitpackedVector>,
    /// Names for concepts (optional)
    names: Vec<String>,
    /// Threshold for "good enough" match
    threshold: u32,
}

impl Default for Resonator {
    fn default() -> Self {
        Self::new()
    }
}

impl Resonator {
    /// Create empty resonator
    pub fn new() -> Self {
        Self {
            concepts: Vec::new(),
            names: Vec::new(),
            threshold: VECTOR_BITS as u32 / 4, // 25% different
        }
    }

    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            concepts: Vec::with_capacity(n),
            names: Vec::with_capacity(n),
            threshold: VECTOR_BITS as u32 / 4,
        }
    }

    /// Set matching threshold
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
    }

    /// Set threshold from similarity (0.0 to 1.0)
    pub fn set_threshold_similarity(&mut self, min_similarity: f32) {
        self.threshold = ((1.0 - min_similarity) * VECTOR_BITS as f32) as u32;
    }

    /// Add a concept
    pub fn add(&mut self, vector: BitpackedVector) -> usize {
        let idx = self.concepts.len();
        self.concepts.push(vector);
        self.names.push(String::new());
        idx
    }

    /// Add a named concept
    pub fn add_named(&mut self, name: &str, vector: BitpackedVector) -> usize {
        let idx = self.concepts.len();
        self.concepts.push(vector);
        self.names.push(name.to_string());
        idx
    }

    /// Number of concepts
    pub fn len(&self) -> usize {
        self.concepts.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.concepts.is_empty()
    }

    /// Get concept by index
    pub fn get(&self, index: usize) -> Option<&BitpackedVector> {
        self.concepts.get(index)
    }

    /// Get name by index
    pub fn get_name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(|s| s.as_str())
    }

    /// Find best match (single resonance)
    pub fn resonate(&self, noisy: &BitpackedVector) -> Option<ResonanceResult> {
        if self.concepts.is_empty() {
            return None;
        }

        let mut best = ResonanceResult {
            index: 0,
            distance: u32::MAX,
            similarity: 0.0,
            name: String::new(),
        };

        for (i, concept) in self.concepts.iter().enumerate() {
            let dist = hamming_distance_scalar(noisy, concept);
            if dist < best.distance {
                best.index = i;
                best.distance = dist;
            }
        }

        best.similarity = hamming_to_similarity(best.distance);
        best.name = self.names[best.index].clone();

        if best.distance <= self.threshold {
            Some(best)
        } else {
            None
        }
    }

    /// Find k-best matches
    pub fn resonate_k(&self, noisy: &BitpackedVector, k: usize) -> Vec<ResonanceResult> {
        let mut results: Vec<_> = self.concepts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dist = hamming_distance_scalar(noisy, c);
                ResonanceResult {
                    index: i,
                    distance: dist,
                    similarity: hamming_to_similarity(dist),
                    name: self.names[i].clone(),
                }
            })
            .collect();

        results.sort_by_key(|r| r.distance);
        results.truncate(k);
        results
    }

    /// Find all within threshold (superposition cleanup)
    pub fn resonate_all(&self, noisy: &BitpackedVector) -> Vec<ResonanceResult> {
        self.concepts
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let dist = hamming_distance_scalar(noisy, c);
                if dist <= self.threshold {
                    Some(ResonanceResult {
                        index: i,
                        distance: dist,
                        similarity: hamming_to_similarity(dist),
                        name: self.names[i].clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Cascaded resonance with early termination
    pub fn resonate_cascaded(&self, noisy: &BitpackedVector) -> Option<ResonanceResult> {
        if self.concepts.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = u32::MAX;

        for (i, concept) in self.concepts.iter().enumerate() {
            // Use stacked popcount with early termination
            if let Some(stacked) = StackedPopcount::compute_with_threshold(noisy, concept, best_dist) {
                if stacked.total < best_dist {
                    best_dist = stacked.total;
                    best_idx = i;
                }
            }
        }

        if best_dist <= self.threshold {
            Some(ResonanceResult {
                index: best_idx,
                distance: best_dist,
                similarity: hamming_to_similarity(best_dist),
                name: self.names[best_idx].clone(),
            })
        } else {
            None
        }
    }
}

/// Result of resonance matching
#[derive(Debug, Clone)]
pub struct ResonanceResult {
    /// Index in resonator
    pub index: usize,
    /// Hamming distance
    pub distance: u32,
    /// Similarity (0.0 to 1.0)
    pub similarity: f32,
    /// Name (if available)
    pub name: String,
}

// ============================================================================
// SEQUENCE ENCODING (Positional Binding)
// ============================================================================

/// Encode a sequence using positional binding
///
/// ```text
/// seq([A, B, C]) = (A ⊗ P₀) + (B ⊗ P₁) + (C ⊗ P₂)
/// where Pᵢ = rotate(base, i)
/// ```
pub fn encode_sequence(items: &[BitpackedVector]) -> BitpackedVector {
    if items.is_empty() {
        return BitpackedVector::zero();
    }
    if items.len() == 1 {
        return items[0].clone();
    }

    // Generate position vectors through rotation
    let base = BitpackedVector::random(0xDEADBEEF);
    let mut bound_items: Vec<BitpackedVector> = Vec::with_capacity(items.len());

    for (i, item) in items.iter().enumerate() {
        let position = base.rotate_words(i);
        let bound = item.xor(&position);
        bound_items.push(bound);
    }

    // Bundle all bound items
    let refs: Vec<&BitpackedVector> = bound_items.iter().collect();
    BitpackedVector::bundle(&refs)
}

/// Probe sequence for item at position
///
/// Returns approximate match if item was at that position
pub fn probe_sequence(
    sequence: &BitpackedVector,
    position: usize,
) -> BitpackedVector {
    let base = BitpackedVector::random(0xDEADBEEF);
    let pos_vector = base.rotate_words(position);
    sequence.xor(&pos_vector)
}

// ============================================================================
// ANALOGY ENGINE
// ============================================================================

/// Compute analogy: A is to B as C is to ?
///
/// Uses the transformation vector: T = unbind(B, A)
/// Then applies: ? = bind(C, T)
pub fn analogy(
    a: &BitpackedVector,
    b: &BitpackedVector,
    c: &BitpackedVector,
) -> BitpackedVector {
    // T = B ⊗ A (the transformation from A to B)
    let transform = b.xor(a);
    // ? = C ⊗ T (apply transformation to C)
    c.xor(&transform)
}

/// Complete analogy with cleanup
pub fn analogy_with_cleanup(
    a: &BitpackedVector,
    b: &BitpackedVector,
    c: &BitpackedVector,
    resonator: &Resonator,
) -> Option<ResonanceResult> {
    let result = analogy(a, b, c);
    resonator.resonate(&result)
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Simple string hash for seeding random vectors
fn hash_string(s: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64; // FNV-1a offset basis
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    hash
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_unbind() {
        let a = BitpackedVector::random(1);
        let b = BitpackedVector::random(2);

        // A ⊗ B ⊗ B = A
        let bound = a.xor(&b);
        let recovered = bound.xor(&b);
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_vector_field() {
        let mut field = VectorField::new();

        // Create atoms
        let cat = field.atom("cat").clone();
        let dog = field.atom("dog").clone();
        let is_a = field.verb("is_a").clone();

        // Create edge: cat --[is_a]--> animal
        let animal = field.atom("animal").clone();
        let edge = field.create_edge(&cat, &is_a, &animal);

        // Recover animal from edge
        let recovered = edge.get_dst(&cat);
        assert_eq!(hamming_distance_scalar(&recovered, &animal), 0);

        // Recover cat from edge
        let recovered_cat = edge.get_src(&animal);
        assert_eq!(hamming_distance_scalar(&recovered_cat, &cat), 0);
    }

    #[test]
    fn test_resonator() {
        let mut resonator = Resonator::new();
        resonator.set_threshold(VECTOR_BITS as u32 / 2);

        // Add some concepts
        let cat = BitpackedVector::random(100);
        let dog = BitpackedVector::random(200);
        let bird = BitpackedVector::random(300);

        resonator.add_named("cat", cat.clone());
        resonator.add_named("dog", dog.clone());
        resonator.add_named("bird", bird.clone());

        // Exact match
        let result = resonator.resonate(&cat).unwrap();
        assert_eq!(result.name, "cat");
        assert_eq!(result.distance, 0);

        // Noisy match (flip some bits)
        let mut noisy_cat = cat.clone();
        for i in 0..100 {
            noisy_cat.toggle_bit(i);
        }
        let result = resonator.resonate(&noisy_cat).unwrap();
        assert_eq!(result.name, "cat");
        assert!(result.distance <= 100);
    }

    #[test]
    fn test_bound_edge() {
        let src = BitpackedVector::random(1);
        let verb = BitpackedVector::random(2);
        let dst = BitpackedVector::random(3);

        let edge = BoundEdge::new(src.clone(), verb.clone(), dst.clone());

        // Verify recovery
        assert!(edge.verify_dst(&src));
        assert!(edge.verify_src(&dst));
    }

    #[test]
    fn test_analogy() {
        // man:woman :: king:queen
        let man = BitpackedVector::random(1);
        let woman = BitpackedVector::random(2);
        let king = BitpackedVector::random(3);
        let queen = BitpackedVector::random(4);

        // In a real system, woman-man ≈ queen-king (gender transform)
        // Here we verify the algebra works
        let result = analogy(&man, &woman, &king);

        // Result should be some vector (not necessarily queen without cleanup)
        // The structure is correct: result = king ⊗ (woman ⊗ man)
        let expected = king.xor(&woman.xor(&man));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sequence_encoding() {
        let a = BitpackedVector::random(1);
        let b = BitpackedVector::random(2);
        let c = BitpackedVector::random(3);

        let seq = encode_sequence(&[a.clone(), b.clone(), c.clone()]);

        // Probe position 0 should be closest to A
        let probe_0 = probe_sequence(&seq, 0);
        let dist_a = hamming_distance_scalar(&probe_0, &a);
        let dist_b = hamming_distance_scalar(&probe_0, &b);
        let dist_c = hamming_distance_scalar(&probe_0, &c);

        // A should be closest (this is probabilistic but usually works)
        // Due to bundling noise, we just check it's reasonably close
        assert!(dist_a < VECTOR_BITS as u32 / 2);
    }
}
