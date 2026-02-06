//! Slot-Based Node Encoding
//!
//! Encodes node attributes INTO the fingerprint with recoverability.
//!
//! # The Encoding Problem
//!
//! ```text
//! Option 1: Metadata separate (current)
//!   Node = { fingerprint, name, type, value, ... }
//!   ✓ Easy to read attributes
//!   ✗ Can't do similarity search on attributes
//!   ✗ Attributes not part of the "signal"
//!
//! Option 2: Naive binding (lossy)
//!   Node = Base ⊕ Name ⊕ Type ⊕ Value
//!   ✓ Attributes affect similarity
//!   ✗ Can't recover individual attributes
//!   ✗ Order-dependent
//!
//! Option 3: Slot binding (this module) ✓
//!   Node = Base ⊕ (Slot₁ ⊕ Val₁) ⊕ (Slot₂ ⊕ Val₂) ⊕ ...
//!   ✓ Attributes affect similarity
//!   ✓ Individual attributes recoverable
//!   ✓ Order-independent (XOR is commutative)
//! ```
//!
//! # Slot Recovery Formula
//!
//! ```text
//! Given:
//!   Encoded = Base ⊕ (Slot_name ⊕ Val_name) ⊕ (Slot_type ⊕ Val_type)
//!
//! To recover Val_name:
//!   Val_name = Encoded ⊕ Base ⊕ Slot_name ⊕ (Slot_type ⊕ Val_type)
//!
//! If we don't know Val_type, we need to isolate:
//!   Residual = Encoded ⊕ Base ⊕ Slot_name
//!            = Val_name ⊕ (Slot_type ⊕ Val_type)
//!
//! Then search for Val_name among candidates.
//! ```

use crate::bitpack::BitpackedVector;
use crate::hamming::hamming_distance_scalar;
use crate::dntree::TreeAddr;
use std::collections::HashMap;

// ============================================================================
// SLOT KEYS (Orthogonal Vectors for Attribute Binding)
// ============================================================================

/// Well-known slot keys for common attributes
pub struct SlotKeys {
    slots: HashMap<String, BitpackedVector>,
}

impl SlotKeys {
    /// Create standard slot keys
    pub fn standard() -> Self {
        let mut slots = HashMap::new();

        // Generate orthogonal-ish slot keys from reserved seeds
        let slot_names = [
            "name", "type", "label", "description",
            "created", "modified", "author", "version",
            "rung", "qualia", "truth", "confidence",
            "parent", "children", "source", "target",
            "weight", "count", "score", "rank",
            "slot_0", "slot_1", "slot_2", "slot_3",
            "slot_4", "slot_5", "slot_6", "slot_7",
        ];

        for (i, name) in slot_names.iter().enumerate() {
            // Use golden ratio multiplier for good distribution
            let seed = 0x510714E7BA5E0000_u64.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            slots.insert(name.to_string(), BitpackedVector::random(seed));
        }

        Self { slots }
    }

    /// Get slot key by name
    pub fn get(&self, name: &str) -> Option<&BitpackedVector> {
        self.slots.get(name)
    }

    /// Create custom slot key
    pub fn create(&mut self, name: &str) -> &BitpackedVector {
        self.slots.entry(name.to_string()).or_insert_with(|| {
            // Hash name to create deterministic key
            let mut seed = 0u64;
            for (i, b) in name.bytes().enumerate() {
                seed = seed.wrapping_add((b as u64) << ((i % 8) * 8));
            }
            seed = seed.wrapping_mul(0x9E3779B97F4A7C15);
            BitpackedVector::random(seed)
        })
    }

    /// List all slot names
    pub fn names(&self) -> Vec<&str> {
        self.slots.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for SlotKeys {
    fn default() -> Self {
        Self::standard()
    }
}

// ============================================================================
// SLOT-ENCODED NODE
// ============================================================================

/// A node with attributes encoded into the fingerprint
#[derive(Clone, Debug)]
pub struct SlotEncodedNode {
    /// Tree address (identity)
    pub addr: TreeAddr,

    /// Base fingerprint (from addr alone)
    pub base: BitpackedVector,

    /// Fully encoded fingerprint (base + all slots)
    pub encoded: BitpackedVector,

    /// Attribute values (for recovery verification)
    attributes: HashMap<String, BitpackedVector>,

    /// Slot keys used (reference)
    slot_names: Vec<String>,
}

impl SlotEncodedNode {
    /// Create from tree address with no attributes
    pub fn new(addr: TreeAddr) -> Self {
        let base = addr.to_fingerprint();
        Self {
            addr,
            base: base.clone(),
            encoded: base,
            attributes: HashMap::new(),
            slot_names: Vec::new(),
        }
    }

    /// Create with attributes
    pub fn with_attributes(
        addr: TreeAddr,
        attributes: &[(&str, BitpackedVector)],
        slot_keys: &SlotKeys,
    ) -> Self {
        let base = addr.to_fingerprint();
        let mut encoded = base.clone();
        let mut attr_map = HashMap::new();
        let mut slot_names = Vec::new();

        for (slot_name, value) in attributes {
            if let Some(slot_key) = slot_keys.get(slot_name) {
                // Bind: Encoded = Encoded ⊕ (Slot ⊕ Value)
                let bound = slot_key.xor(value);
                encoded = encoded.xor(&bound);

                attr_map.insert(slot_name.to_string(), value.clone());
                slot_names.push(slot_name.to_string());
            }
        }

        Self {
            addr,
            base,
            encoded,
            attributes: attr_map,
            slot_names,
        }
    }

    /// Add/update an attribute
    pub fn set_attribute(
        &mut self,
        slot_name: &str,
        value: BitpackedVector,
        slot_keys: &SlotKeys,
    ) {
        if let Some(slot_key) = slot_keys.get(slot_name) {
            // Remove old value if exists
            if let Some(old_value) = self.attributes.get(slot_name) {
                let old_bound = slot_key.xor(old_value);
                self.encoded = self.encoded.xor(&old_bound);
            }

            // Add new value
            let new_bound = slot_key.xor(&value);
            self.encoded = self.encoded.xor(&new_bound);

            self.attributes.insert(slot_name.to_string(), value);
            if !self.slot_names.contains(&slot_name.to_string()) {
                self.slot_names.push(slot_name.to_string());
            }
        }
    }

    /// Remove an attribute
    pub fn remove_attribute(&mut self, slot_name: &str, slot_keys: &SlotKeys) {
        if let Some(slot_key) = slot_keys.get(slot_name) {
            if let Some(old_value) = self.attributes.remove(slot_name) {
                // XOR out the bound value
                let bound = slot_key.xor(&old_value);
                self.encoded = self.encoded.xor(&bound);

                self.slot_names.retain(|n| n != slot_name);
            }
        }
    }

    /// Recover an attribute value (if we know all other attributes)
    pub fn recover_attribute(
        &self,
        slot_name: &str,
        slot_keys: &SlotKeys,
    ) -> Option<BitpackedVector> {
        let slot_key = slot_keys.get(slot_name)?;

        // Start with: Encoded ⊕ Base ⊕ SlotKey
        let mut residual = self.encoded.xor(&self.base).xor(slot_key);

        // XOR out all OTHER slot bindings
        for (other_name, other_value) in &self.attributes {
            if other_name != slot_name {
                if let Some(other_slot) = slot_keys.get(other_name) {
                    let other_bound = other_slot.xor(other_value);
                    residual = residual.xor(&other_bound);
                }
            }
        }

        // Residual should now be the value
        Some(residual)
    }

    /// Probe for attribute value (search among candidates)
    pub fn probe_attribute(
        &self,
        slot_name: &str,
        candidates: &[(&str, BitpackedVector)],
        slot_keys: &SlotKeys,
        threshold: u32,
    ) -> Option<(String, u32)> {
        let slot_key = slot_keys.get(slot_name)?;

        // Compute residual (may contain noise from unknown slots)
        let residual = self.encoded.xor(&self.base).xor(slot_key);

        // Find best matching candidate
        let mut best: Option<(String, u32)> = None;

        for (name, value) in candidates {
            // If this candidate were the value, the residual ⊕ value
            // should leave only the other slot bindings (low popcount if correct)
            let test = residual.xor(value);
            let dist = test.popcount();

            // For a correct match with no other slots, dist should be 0
            // With N other slots, dist should be ~N * expected_slot_noise
            if dist < threshold {
                if best.is_none() || dist < best.as_ref().unwrap().1 {
                    best = Some((name.to_string(), dist));
                }
            }
        }

        best
    }

    /// Get stored attribute (from local cache)
    pub fn get_attribute(&self, slot_name: &str) -> Option<&BitpackedVector> {
        self.attributes.get(slot_name)
    }

    /// List attribute names
    pub fn attribute_names(&self) -> &[String] {
        &self.slot_names
    }

    /// Number of encoded attributes
    pub fn num_attributes(&self) -> usize {
        self.attributes.len()
    }
}

// ============================================================================
// STRING VALUE ENCODING
// ============================================================================

/// Encode string values as fingerprints
pub struct StringEncoder {
    /// Cached string → fingerprint mappings
    cache: HashMap<String, BitpackedVector>,
}

impl StringEncoder {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Encode string to fingerprint
    pub fn encode(&mut self, s: &str) -> BitpackedVector {
        if let Some(fp) = self.cache.get(s) {
            return fp.clone();
        }

        // Hash string to seed
        let mut seed = 0u64;
        for (i, b) in s.bytes().enumerate() {
            seed = seed.wrapping_mul(31).wrapping_add(b as u64);
            seed = seed.wrapping_add((i as u64) << 40);
        }
        seed = seed.wrapping_mul(0x9E3779B97F4A7C15);

        let fp = BitpackedVector::random(seed);
        self.cache.insert(s.to_string(), fp.clone());
        fp
    }

    /// Find closest string match
    pub fn decode(&self, fp: &BitpackedVector, threshold: u32) -> Option<&str> {
        let mut best: Option<(&str, u32)> = None;

        for (s, cached_fp) in &self.cache {
            let dist = hamming_distance_scalar(fp, cached_fp);
            if dist <= threshold {
                if best.is_none() || dist < best.unwrap().1 {
                    best = Some((s.as_str(), dist));
                }
            }
        }

        best.map(|(s, _)| s)
    }

    /// Register known string (for decoding)
    pub fn register(&mut self, s: &str) {
        self.encode(s);
    }

    /// Number of cached strings
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for StringEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NUMERIC VALUE ENCODING
// ============================================================================

/// Encode numeric values with locality preservation
pub struct NumericEncoder {
    /// Resolution (values within this range share some bits)
    resolution: f64,
    /// Scale factor
    scale: f64,
}

impl NumericEncoder {
    pub fn new(resolution: f64) -> Self {
        Self {
            resolution,
            scale: 1.0 / resolution,
        }
    }

    /// Encode f64 to fingerprint (locality-sensitive)
    pub fn encode(&self, value: f64) -> BitpackedVector {
        // Quantize to resolution
        let quantized = (value * self.scale).round() as i64;

        // Generate fingerprint from quantized value
        // Use thermometer encoding for locality: similar values share bits
        let mut fp = BitpackedVector::zero();

        // Base fingerprint from value
        let base_seed = quantized as u64;
        let base = BitpackedVector::random(base_seed.wrapping_mul(0x9E3779B97F4A7C15));

        // Add "blur" from nearby values for soft boundaries
        let blur1 = BitpackedVector::random(((quantized - 1) as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let blur2 = BitpackedVector::random(((quantized + 1) as u64).wrapping_mul(0x9E3779B97F4A7C15));

        // Combine: base dominates, neighbors add similarity
        let refs = [&base, &base, &base, &blur1, &blur2];
        fp = BitpackedVector::bundle(&refs);

        fp
    }

    /// Encode integer
    pub fn encode_int(&self, value: i64) -> BitpackedVector {
        self.encode(value as f64)
    }

    /// Estimate value from fingerprint (approximate)
    pub fn decode_approx(&self, fp: &BitpackedVector, search_range: (f64, f64), step: f64) -> f64 {
        let mut best_value = search_range.0;
        let mut best_dist = u32::MAX;

        let mut v = search_range.0;
        while v <= search_range.1 {
            let candidate_fp = self.encode(v);
            let dist = hamming_distance_scalar(fp, &candidate_fp);
            if dist < best_dist {
                best_dist = dist;
                best_value = v;
            }
            v += step;
        }

        best_value
    }
}

// ============================================================================
// COMPOSITE NODE BUILDER
// ============================================================================

/// Builder for nodes with multiple encoded attributes
pub struct NodeBuilder {
    addr: TreeAddr,
    attributes: Vec<(String, BitpackedVector)>,
    slot_keys: SlotKeys,
    string_encoder: StringEncoder,
    numeric_encoder: NumericEncoder,
}

impl NodeBuilder {
    pub fn new(addr: TreeAddr) -> Self {
        Self {
            addr,
            attributes: Vec::new(),
            slot_keys: SlotKeys::standard(),
            string_encoder: StringEncoder::new(),
            numeric_encoder: NumericEncoder::new(0.01),
        }
    }

    /// Add string attribute
    pub fn with_string(mut self, slot: &str, value: &str) -> Self {
        let fp = self.string_encoder.encode(value);
        self.attributes.push((slot.to_string(), fp));
        self
    }

    /// Add numeric attribute
    pub fn with_number(mut self, slot: &str, value: f64) -> Self {
        let fp = self.numeric_encoder.encode(value);
        self.attributes.push((slot.to_string(), fp));
        self
    }

    /// Add fingerprint attribute directly
    pub fn with_fingerprint(mut self, slot: &str, value: BitpackedVector) -> Self {
        self.attributes.push((slot.to_string(), value));
        self
    }

    /// Add boolean attribute
    pub fn with_bool(mut self, slot: &str, value: bool) -> Self {
        // True/False as distinct fingerprints
        let seed = if value { 0x74AE5EED00000001 } else { 0xFA15E5EED0000000 };
        let fp = BitpackedVector::random(seed);
        self.attributes.push((slot.to_string(), fp));
        self
    }

    /// Build the node
    pub fn build(self) -> SlotEncodedNode {
        let attrs: Vec<(&str, BitpackedVector)> = self.attributes
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        SlotEncodedNode::with_attributes(self.addr, &attrs, &self.slot_keys)
    }
}

// ============================================================================
// COMPARISON: INTERNAL VS EXTERNAL ENCODING
// ============================================================================

/// Demonstrates the two approaches
pub mod comparison {
    use super::*;

    /// External encoding (current approach)
    #[derive(Clone, Debug)]
    pub struct ExternalNode {
        pub addr: TreeAddr,
        pub fingerprint: BitpackedVector,  // From addr only
        // Metadata stored separately:
        pub name: String,
        pub node_type: String,
        pub weight: f32,
    }

    /// Internal encoding (slot-based)
    #[derive(Clone, Debug)]
    pub struct InternalNode {
        pub addr: TreeAddr,
        pub fingerprint: BitpackedVector,  // Includes all attributes!
        // No separate fields - everything is in the fingerprint
    }

    /// Comparison results
    pub fn compare_approaches() -> &'static str {
        r#"
EXTERNAL ENCODING (Metadata Separate)
=====================================
Pros:
  + Fast attribute access (direct field read)
  + No decoding overhead
  + Exact values preserved
  + Simple implementation

Cons:
  - Similarity search ignores attributes
  - More memory (fingerprint + fields)
  - Schema is rigid
  - Can't query "find nodes with name similar to X"

INTERNAL ENCODING (Slot-Based)
==============================
Pros:
  + Similarity search includes attributes
  + Single unified representation
  + Schema-free (any attributes)
  + Can find "nodes with similar name"
  + Composable (node = attributes)

Cons:
  - Decoding is approximate
  - Capacity limits (~50 attributes)
  - More complex implementation
  - Some information loss

RECOMMENDATION
==============
Use HYBRID approach:
  - Internal encoding for SEARCHABLE attributes
  - External storage for EXACT values needed

Example:
  InternalNode {
    fingerprint: encode(addr, name, type, tags),  // Searchable
  }
  + ExternalMetadata {
    exact_name: "Alice",     // For display
    exact_weight: 0.7532,    // For computation
  }
"#
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_encoding() {
        let slot_keys = SlotKeys::standard();
        let mut string_enc = StringEncoder::new();

        let addr = TreeAddr::from_string("/test/node");

        // Encode with attributes
        let name_fp = string_enc.encode("Alice");
        let type_fp = string_enc.encode("Person");

        let node = SlotEncodedNode::with_attributes(
            addr,
            &[("name", name_fp.clone()), ("type", type_fp.clone())],
            &slot_keys,
        );

        // Recover attribute
        let recovered = node.recover_attribute("name", &slot_keys).unwrap();

        // Should match original
        assert_eq!(hamming_distance_scalar(&recovered, &name_fp), 0);
    }

    #[test]
    fn test_string_encoder() {
        let mut enc = StringEncoder::new();

        let fp1 = enc.encode("hello");
        let fp2 = enc.encode("hello");  // Same string
        let fp3 = enc.encode("world");  // Different string

        // Same string = same fingerprint
        assert_eq!(hamming_distance_scalar(&fp1, &fp2), 0);

        // Different strings = different fingerprints
        assert!(hamming_distance_scalar(&fp1, &fp3) > 1000);
    }

    #[test]
    fn test_numeric_encoder() {
        let enc = NumericEncoder::new(0.1);

        let fp1 = enc.encode(1.0);
        let fp2 = enc.encode(1.05);  // Close
        let fp3 = enc.encode(100.0); // Far

        // Close values should have lower distance
        let d_close = hamming_distance_scalar(&fp1, &fp2);
        let d_far = hamming_distance_scalar(&fp1, &fp3);

        assert!(d_close < d_far);
    }

    #[test]
    fn test_node_builder() {
        let addr = TreeAddr::from_string("/people/alice");

        let node = NodeBuilder::new(addr)
            .with_string("name", "Alice")
            .with_string("type", "Person")
            .with_number("age", 30.0)
            .with_bool("active", true)
            .build();

        assert_eq!(node.num_attributes(), 4);
    }

    #[test]
    fn test_attribute_modification() {
        let slot_keys = SlotKeys::standard();
        let mut string_enc = StringEncoder::new();

        let addr = TreeAddr::from_string("/test");
        let mut node = SlotEncodedNode::new(addr);

        // Add attribute
        let name1 = string_enc.encode("Alice");
        node.set_attribute("name", name1.clone(), &slot_keys);

        // Update attribute
        let name2 = string_enc.encode("Bob");
        node.set_attribute("name", name2.clone(), &slot_keys);

        // Recover should give new value
        let recovered = node.recover_attribute("name", &slot_keys).unwrap();
        assert_eq!(hamming_distance_scalar(&recovered, &name2), 0);
    }
}
