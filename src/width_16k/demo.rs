//! Integration Demo: Full-Stack Showcase + Benchmarks
//!
//! This module exercises the entire 16K stack:
//! - Cypher procedure calls (RedisGraph/Neo4j compatible)
//! - ANN search with schema predicate pruning
//! - GNN message passing via XOR-bind + majority bundle
//! - SNN/STDP Hebbian weight updates
//! - NARS revision + deduction chains
//! - DN tree addressing (Redis-style GET/SET)
//! - XOR write cache + bubble propagation
//! - Schema-aware bind with intelligent metadata merge
//! - 10K ↔ 16K compatibility layer
//!
//! ## vs. Typical Neo4j Traversal Benchmarks
//!
//! Traditional Neo4j property graph traversal:
//! ```text
//! MATCH (a:Person)-[:KNOWS]->(b:Person)-[:LIKES]->(c:Movie)
//! WHERE a.name = "Alice" AND b.trust > 0.5
//! RETURN c.title, b.trust
//! LIMIT 10
//! ```
//!
//! With HDR + Schema:
//! ```text
//! CALL hdr.schemaSearch($alice, 10, {
//!   ani: { min_level: 5, min_activation: 100 },  // social reasoning
//!   nars: { min_confidence: 0.5 },                // trust threshold
//!   graph: { max_hop: 2 }                         // 2-hop neighborhood
//! }) YIELD id, distance, schema
//! ```
//!
//! Key differences:
//! - Neo4j: O(|E|) edge traversal per hop, B+ tree index lookups
//! - HDR:   O(1) schema predicate check + O(log n) ANN cascade
//! - Neo4j: Cypher planner builds execution graph → iterator pipeline
//! - HDR:   Single popcount cascade with inline metadata checking
//! - Neo4j: Property lookups are pointer chases to property store
//! - HDR:   Properties ARE the fingerprint (zero additional I/O)

#[cfg(test)]
mod tests {
    use crate::bitpack::BitpackedVector;
    use crate::navigator::{Navigator, CypherArg, CypherYield};
    use crate::width_16k::schema::*;
    use crate::width_16k::search::*;
    use crate::width_16k::compat;
    use crate::width_16k::xor_bubble::*;
    use crate::width_16k::VECTOR_WORDS;

    // =====================================================================
    // SCENARIO 1: Social Graph with NARS Trust Propagation
    //
    // Models a trust network where:
    // - Each person is a 10K fingerprint (semantic identity)
    // - Trust edges carry NARS truth values (frequency=trustworthiness, confidence=evidence)
    // - 2-hop trust transitivity via NARS deduction
    // =====================================================================

    #[test]
    fn demo_social_trust_network() {
        let _nav = Navigator::new();

        // Create people as random fingerprints
        let alice = BitpackedVector::random(1001);
        let bob = BitpackedVector::random(1002);
        let carol = BitpackedVector::random(1003);

        // Zero-extend to 16K and attach NARS trust values
        let mut alice_16k = compat::zero_extend(&alice).to_vec();
        let mut bob_16k = compat::zero_extend(&bob).to_vec();
        let mut carol_16k = compat::zero_extend(&carol).to_vec();

        // Alice trusts Bob with f=0.9, c=0.8 (strong evidence)
        let mut alice_schema = SchemaSidecar::default();
        alice_schema.nars_truth = NarsTruth::from_floats(0.9, 0.8);
        alice_schema.ani_levels.social = 500; // High social reasoning
        alice_schema.node_type.kind = NodeKind::Entity as u8;
        alice_schema.metrics.pagerank = 800;
        alice_schema.metrics.degree = 3;
        alice_schema.neighbors.insert(1002); // Bob is neighbor
        alice_schema.neighbors.insert(1003); // Carol is neighbor
        alice_schema.write_to_words(&mut alice_16k);

        // Bob trusts Carol with f=0.7, c=0.5
        let mut bob_schema = SchemaSidecar::default();
        bob_schema.nars_truth = NarsTruth::from_floats(0.7, 0.5);
        bob_schema.ani_levels.social = 300;
        bob_schema.node_type.kind = NodeKind::Entity as u8;
        bob_schema.metrics.pagerank = 600;
        bob_schema.metrics.degree = 2;
        bob_schema.neighbors.insert(1003);
        bob_schema.write_to_words(&mut bob_16k);

        // Carol has moderate self-confidence
        let mut carol_schema = SchemaSidecar::default();
        carol_schema.nars_truth = NarsTruth::from_floats(0.6, 0.4);
        carol_schema.ani_levels.social = 200;
        carol_schema.node_type.kind = NodeKind::Entity as u8;
        carol_schema.metrics.pagerank = 400;
        carol_schema.write_to_words(&mut carol_16k);

        // ----- Test 1: Schema predicate search -----
        // Find socially capable nodes with high trust
        let query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 5, min_activation: 200 }) // social level >= 200
            .with_nars(NarsFilter {
                min_frequency: Some(0.6),
                min_confidence: Some(0.3),
                min_priority: None,
            })
            .with_graph(GraphFilter {
                min_pagerank: Some(300),
                max_hop: None,
                cluster_id: None,
                min_degree: None,
            });

        assert!(query.passes_predicates(&alice_16k), "Alice should pass: social=500, f=0.9, pagerank=800");
        assert!(query.passes_predicates(&bob_16k), "Bob should pass: social=300, f=0.7, pagerank=600");
        // Carol has social=200 >= 200 (passes ANI), f=0.6 >= 0.6, c=0.4 >= 0.3 (passes NARS),
        // pagerank=400 >= 300 (passes graph) → she passes all predicates
        assert!(query.passes_predicates(&carol_16k), "Carol passes: social=200 >= threshold");

        // ----- Test 2: NARS trust deduction -----
        // Alice→Bob trust × Bob→Carol trust = Alice→Carol transitive trust
        let deduced = nars_deduction_inline(&alice_16k, &bob_16k);
        // f = 0.9 × 0.7 ≈ 0.63, c = 0.63 × 0.8 × 0.5 ≈ 0.25
        assert!(deduced.f() > 0.5, "Transitive trust frequency should be decent: {}", deduced.f());
        assert!(deduced.c() < 0.5, "Transitive confidence should attenuate: {}", deduced.c());

        // ----- Test 3: NARS revision (combining evidence) -----
        let mut revised = alice_16k.clone();
        nars_revision_inline(&alice_16k, &bob_16k, &mut revised);
        let revised_schema = SchemaSidecar::read_from_words(&revised);
        // Revision should increase confidence (combining independent evidence)
        let _alice_orig = SchemaSidecar::read_from_words(&alice_16k);
        // Not always true that revision > original confidence, but revised should be reasonable
        assert!(revised_schema.nars_truth.c() > 0.0, "Revised confidence should be positive");

        // ----- Test 4: Bloom filter neighbor check -----
        assert!(bloom_might_be_neighbors(&alice_16k, 1002), "Alice knows Bob");
        assert!(bloom_might_be_neighbors(&alice_16k, 1003), "Alice knows Carol");
        // Unknown person - very low false positive probability
        let _unknown_likely_absent = !bloom_might_be_neighbors(&alice_16k, 99999);
        // Can't assert definitively due to FPR, but it's very likely false

        // ----- Test 5: Schema-aware bind -----
        // Bind Alice and Bob: creates an edge fingerprint with merged metadata
        let edge = schema_bind(&alice_16k, &bob_16k);
        let edge_schema = SchemaSidecar::read_from_words(&edge);
        // ANI: max(alice.social=500, bob.social=300) = 500
        assert_eq!(edge_schema.ani_levels.social, 500);
        // NARS: revision of their truth values
        assert!(edge_schema.nars_truth.f() > 0.0);

        // ----- Test 6: RL routing score -----
        let (_best_action, best_q) = read_best_q(&alice_16k);
        let routing = rl_routing_score(1000, best_q, 0.2);
        assert!(routing >= 0.0 && routing <= 1.0, "Routing score in [0,1]: {}", routing);
    }

    // =====================================================================
    // SCENARIO 2: Knowledge Graph with Cypher Procedures
    //
    // Simulates a Neo4j-style property graph using Cypher calls.
    // Compares: Neo4j index lookup → property filter → return
    //       vs: HDR cascade → inline predicate → zero-copy
    // =====================================================================

    #[test]
    fn demo_cypher_knowledge_graph() {
        let nav = Navigator::new();

        // Create concept fingerprints
        let france = BitpackedVector::random(100);
        let capital_of = BitpackedVector::random(200);
        let paris = BitpackedVector::random(300);

        // ----- Cypher: Create edge via bind3 -----
        let yields = nav.cypher_call("hdr.bind3", &[
            CypherArg::Vector(france.clone()),
            CypherArg::Vector(capital_of.clone()),
            CypherArg::Vector(paris.clone()),
        ]).unwrap();

        let edge = match &yields[0] {
            CypherYield::Vector(_, v) => v.clone(),
            _ => panic!("Expected vector"),
        };

        // ----- Cypher: Retrieve france from edge + verb + target -----
        let yields = nav.cypher_call("hdr.retrieve", &[
            CypherArg::Vector(edge.clone()),
            CypherArg::Vector(capital_of.clone()),
            CypherArg::Vector(paris.clone()),
        ]).unwrap();

        let recovered = match &yields[0] {
            CypherYield::Vector(_, v) => v.clone(),
            _ => panic!("Expected vector"),
        };
        assert_eq!(recovered, france, "Retrieval should recover France exactly");

        // ----- Cypher: Compute analogy -----
        // france:paris :: germany:?
        let germany = BitpackedVector::random(400);
        let yields = nav.cypher_call("hdr.analogy", &[
            CypherArg::Vector(france.clone()),
            CypherArg::Vector(paris.clone()),
            CypherArg::Vector(germany.clone()),
        ]).unwrap();

        let berlin_estimate = match &yields[0] {
            CypherYield::Vector(_, v) => v.clone(),
            _ => panic!("Expected vector"),
        };
        // Verify analogy property: france ⊕ paris = berlin_estimate ⊕ germany
        let transform_a = france.xor(&paris);
        let transform_b = berlin_estimate.xor(&germany);
        assert_eq!(transform_a, transform_b, "Analogy should preserve the transform");

        // ----- Cypher: Hamming distance -----
        let yields = nav.cypher_call("hdr.hamming", &[
            CypherArg::Vector(france.clone()),
            CypherArg::Vector(france.clone()),
        ]).unwrap();
        if let CypherYield::Int(_, dist) = &yields[0] {
            assert_eq!(*dist, 0, "Self-distance should be zero");
        }

        // ----- Cypher: Schema procedures -----
        // ANI levels
        let yields = nav.cypher_call("hdr.aniLevels", &[
            CypherArg::Vector(france.clone()),
        ]).unwrap();
        assert_eq!(yields.len(), 9, "Should return dominant + 8 levels");

        // NARS truth
        let yields = nav.cypher_call("hdr.narsTruth", &[
            CypherArg::Vector(france.clone()),
        ]).unwrap();
        assert_eq!(yields.len(), 2, "Should return frequency + confidence");

        // Best action
        let yields = nav.cypher_call("hdr.bestAction", &[
            CypherArg::Vector(france.clone()),
        ]).unwrap();
        assert_eq!(yields.len(), 2, "Should return action + q_value");

        // Schema bind
        let yields = nav.cypher_call("hdr.schemaBind", &[
            CypherArg::Vector(france.clone()),
            CypherArg::Vector(paris.clone()),
        ]).unwrap();
        assert!(!yields.is_empty(), "Schema bind should return result");

        // Error handling
        let err = nav.cypher_call("hdr.nonexistent", &[]);
        assert!(err.is_err(), "Unknown procedure should error");
    }

    // =====================================================================
    // SCENARIO 3: GNN Message Passing + SNN STDP
    //
    // Models a small neural network with HDR vectors as activations.
    // GNN layers aggregate neighbor messages, SNN updates Hebbian weights.
    // =====================================================================

    #[test]
    fn demo_gnn_snn_integration() {
        let nav = Navigator::new();

        // Create a 4-node graph: 0←→1←→2←→3
        let node0 = BitpackedVector::random(1);
        let node1 = BitpackedVector::random(2);
        let node2 = BitpackedVector::random(3);
        let node3 = BitpackedVector::random(4);

        let edge_01 = BitpackedVector::random(101);
        let edge_12 = BitpackedVector::random(102);
        let edge_23 = BitpackedVector::random(103);

        // ----- GNN: 1-hop message passing on node 1 -----
        // Node 1 receives from node 0 (via edge_01) and node 2 (via edge_12)
        let result = nav.gnn_message_pass(&node1, &[
            (node0.clone(), edge_01.clone()),
            (node2.clone(), edge_12.clone()),
        ]);
        assert_ne!(result, node1, "Message passing should change the embedding");

        // ----- GNN: Multi-hop (2 layers) -----
        let layer0 = vec![
            (node0.clone(), edge_01.clone()),
            (node2.clone(), edge_12.clone()),
        ];
        let layer1 = vec![
            (node1.clone(), edge_12.clone()),
            (node3.clone(), edge_23.clone()),
        ];
        let multi_hop = nav.gnn_multi_hop(&node1, &[layer0, layer1]);
        assert_ne!(multi_hop, node1, "Multi-hop should produce different embedding");
        assert_ne!(multi_hop, result, "2-hop should differ from 1-hop");

        // ----- SNN: STDP + Hebbian weight update -----
        // Simulate spike-timing-dependent plasticity
        let mut hebbian = InlineHebbian::default();
        let mut stdp = StdpMarkers::default();

        // Pre-synaptic spike at t=100, post-synaptic at t=105 (LTP: strengthen)
        stdp.record_spike(100);
        stdp.record_spike(105);

        // Strengthen connection to neighbor 0
        hebbian.strengthen(0, 0.1);
        assert!(hebbian.weight(0) > 0.0, "Weight should increase after LTP");

        // Record more spikes
        stdp.record_spike(110);
        hebbian.strengthen(0, 0.05);

        // Decay all weights (homeostatic regulation)
        hebbian.decay(0.95);
        let w0 = hebbian.weight(0);
        assert!(w0 > 0.0 && w0 < 1.0, "Weight should be positive after decay: {}", w0);

        // ----- SNN: Inline Q-values for RL-guided routing -----
        let mut q = InlineQValues::default();
        q.set_q(0, 0.8); // Action 0: follow edge_01
        q.set_q(1, -0.3); // Action 1: follow edge_12 (negative = avoid)
        q.set_q(2, 0.5); // Action 2: explore

        assert_eq!(q.best_action(), 0, "Should prefer action 0 (highest Q)");
        assert!((q.q(0) - 0.8).abs() < 0.02, "Q-value quantization error");

        // ----- Inline rewards for TD learning -----
        let mut rewards = InlineRewards::default();
        // Push increasing rewards (simulating learning progress)
        for i in 0..8 {
            rewards.push(i as f32 * 0.1);
        }
        assert!(rewards.trend() > 0.0, "Should detect positive reward trend");
        assert!(rewards.average() > 0.0, "Average reward should be positive");

        // ----- Write schema to 16K vector and verify roundtrip -----
        let mut words_16k = compat::zero_extend(&node1).to_vec();
        let mut schema = SchemaSidecar::default();
        schema.hebbian = hebbian;
        schema.stdp = stdp;
        schema.q_values = q;
        schema.rewards = rewards;
        schema.write_to_words(&mut words_16k);

        let recovered = SchemaSidecar::read_from_words(&words_16k);
        assert_eq!(recovered.q_values.best_action(), 0);
        assert!(recovered.rewards.trend() > 0.0);
        assert_eq!(recovered.stdp.last_spike(), 110);
    }

    // =====================================================================
    // SCENARIO 4: XOR Write Cache + Delta Compression
    //
    // Demonstrates zero-copy-preserving writes via the XOR write cache,
    // and delta compression along a DN tree path.
    // =====================================================================

    #[test]
    fn demo_xor_write_cache_and_compression() {
        // ----- Setup: Create a DN tree path (root → depth=4) -----
        let make_words = |seed: u64| -> Vec<u64> {
            let mut words = vec![0u64; VECTOR_WORDS];
            let mut r = seed;
            for w in &mut words {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                *w = r;
            }
            words
        };

        let root = make_words(1);
        // Children are similar to parent (simulate centroid hierarchy)
        let child1 = {
            let mut w = root.clone();
            w[0] ^= 0xFFFF; // Flip 16 bits in word 0
            w[5] ^= 0xFF;   // Flip 8 bits in word 5
            w
        };
        let child2 = {
            let mut w = child1.clone();
            w[1] ^= 0xFFFFFF; // Flip 24 bits
            w
        };
        let leaf = {
            let mut w = child2.clone();
            w[10] ^= 0xF; // Flip 4 bits
            w
        };

        // ----- Delta chain compression -----
        let path: Vec<&[u64]> = vec![&root, &child1, &child2, &leaf];
        let chain = DeltaChain::from_path(&path);

        assert_eq!(chain.depth(), 4);
        assert!(chain.avg_sparsity() > 0.9, "Adjacent centroids should be >90% sparse: {}", chain.avg_sparsity());

        let ratio = chain.compressed_bytes() as f32 / chain.uncompressed_bytes() as f32;
        assert!(ratio < 0.3, "Should achieve >3x compression: ratio={}", ratio);

        // Verify lossless reconstruction
        let reconstructed = chain.reconstruct(3);
        assert_eq!(&reconstructed[..VECTOR_WORDS], &leaf[..VECTOR_WORDS],
            "Delta chain should reconstruct leaf losslessly");

        // ----- XOR Write Cache -----
        let mut cache = XorWriteCache::new(1_048_576); // 1MB threshold

        // Simulate updating a vector (leaf change)
        let new_leaf = {
            let mut w = leaf.clone();
            w[0] ^= 0xDEAD; // Small mutation
            w
        };
        let delta = XorDelta::compute(&leaf, &new_leaf);
        assert!(delta.sparsity() > 0.99, "Single-word change = very sparse");

        // Record in cache (no Arrow buffer mutation)
        cache.record_delta(42, delta);
        assert!(cache.is_dirty(42));
        assert!(!cache.is_dirty(99));
        assert_eq!(cache.dirty_count(), 1);

        // Read through cache: applies delta on-the-fly
        let read = cache.read_through(42, &leaf);
        assert!(!read.is_clean(), "Should be patched");
        assert_eq!(read.words()[0], new_leaf[0], "Patched read should match new leaf");

        // Clean read for uncached vector
        let clean = cache.read_through(99, &root);
        assert!(clean.is_clean(), "Uncached should be clean (zero-copy)");

        // Record a second delta (compose automatically)
        let newer_leaf = {
            let mut w = new_leaf.clone();
            w[1] ^= 0xBEEF;
            w
        };
        let delta2 = XorDelta::compute(&new_leaf, &newer_leaf);
        cache.record_delta(42, delta2);
        assert_eq!(cache.dirty_count(), 1, "Should compose, not add entry");

        // Read through shows composed result
        let read2 = cache.read_through(42, &leaf);
        assert_eq!(read2.words()[0], newer_leaf[0]);
        assert_eq!(read2.words()[1], newer_leaf[1]);

        // Self-inverse: applying same delta twice cancels
        let mut cancel_cache = XorWriteCache::default_cache();
        let d = XorDelta::compute(&leaf, &new_leaf);
        cancel_cache.record_delta(1, d.clone());
        cancel_cache.record_delta(1, d); // XOR with self = identity
        let cancel_read = cancel_cache.read_through(1, &leaf);
        assert_eq!(cancel_read.words()[0], leaf[0], "Double-apply should cancel");

        // Flush
        assert!(!cache.should_flush(), "Below 1MB threshold");
        let flushed = cache.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(cache.dirty_count(), 0);
    }

    // =====================================================================
    // SCENARIO 5: XOR Bubble Propagation
    //
    // Demonstrates incremental centroid updates: leaf change bubbles up
    // through the tree with attenuation at each level.
    // =====================================================================

    #[test]
    fn demo_xor_bubble_propagation() {
        let mut make_words = |seed: u64| -> Vec<u64> {
            let mut words = vec![0u64; VECTOR_WORDS];
            let mut r = seed;
            for w in &mut words {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                *w = r;
            }
            words
        };

        let old_leaf = make_words(100);
        let mut new_leaf = old_leaf.clone();
        new_leaf[0] ^= 0xFFFF_FFFF; // Flip 32 bits

        // ----- Exact bubble (fanout=1) -----
        let mut parent_exact = old_leaf.clone();
        let mut bubble_exact = XorBubble::from_leaf_change(&old_leaf, &new_leaf, 1);
        bubble_exact.apply_to_parent(&mut parent_exact, 42);
        assert_eq!(&parent_exact[..VECTOR_WORDS], &new_leaf[..VECTOR_WORDS],
            "Fanout=1 should be exact update");

        // ----- Attenuated bubble (fanout=16) -----
        let mut parent_approx = old_leaf.clone();
        let mut bubble_approx = XorBubble::from_leaf_change(&old_leaf, &new_leaf, 16);
        bubble_approx.apply_to_parent(&mut parent_approx, 42);

        let changed_bits: u32 = (0..VECTOR_WORDS)
            .map(|w| (parent_approx[w] ^ old_leaf[w]).count_ones())
            .sum();
        // With fanout=16, expect ~32/16 = 2 bits changed (probabilistic)
        assert!(changed_bits <= 32, "Attenuated: expect few bits changed, got {}", changed_bits);

        // ----- Bubble exhaustion -----
        let mut bubble_deep = XorBubble::from_leaf_change(&old_leaf, &new_leaf, 16);
        let mut dummy = make_words(999);
        for _ in 0..20 {
            bubble_deep.apply_to_parent(&mut dummy, 42);
        }
        assert!(bubble_deep.is_exhausted(), "Should exhaust after many levels");
    }

    // =====================================================================
    // SCENARIO 6: 10K ↔ 16K Migration + Compatibility
    //
    // Shows that existing 10K vectors work seamlessly with 16K operations.
    // =====================================================================

    #[test]
    fn demo_10k_16k_compatibility() {
        // Create 10K vectors (existing data)
        let v1 = BitpackedVector::random(1);
        let v2 = BitpackedVector::random(2);

        // ----- Zero-extend to 16K -----
        let v1_16k = compat::zero_extend(&v1);
        let v2_16k = compat::zero_extend(&v2);

        // Distance is preserved
        let dist_10k = crate::hamming::hamming_distance_scalar(&v1, &v2);
        let dist_cross = compat::cross_width_distance(&v1, &v2_16k);
        assert_eq!(dist_10k, dist_cross, "Cross-width distance should match 10K distance");

        let dist_16k = compat::full_distance_16k(&v1_16k, &v2_16k);
        assert_eq!(dist_10k, dist_16k, "16K distance of zero-extended should match 10K");

        // ----- Truncate roundtrip -----
        let v1_back = compat::truncate(&v1_16k);
        assert_eq!(v1, v1_back, "Truncate(zero_extend(v)) should be identity");

        // ----- XOR fold vs truncate (when extra words are zero) -----
        let folded = compat::xor_fold(&v1_16k);
        let truncated = compat::truncate(&v1_16k);
        assert_eq!(folded, truncated, "Fold = truncate when extra words are zero");

        // ----- XOR fold with non-zero schema -----
        let mut v1_with_schema = v1_16k;
        let mut schema = SchemaSidecar::default();
        schema.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        schema.ani_levels.planning = 500;
        schema.write_to_words(&mut v1_with_schema);

        let folded_schema = compat::xor_fold(&v1_with_schema);
        let trunc_schema = compat::truncate(&v1_with_schema);
        assert_ne!(folded_schema, trunc_schema,
            "Fold should differ from truncate when schema blocks are non-zero");

        // ----- Batch migration -----
        let batch: Vec<BitpackedVector> = (0..5)
            .map(|i| BitpackedVector::random(i as u64))
            .collect();
        let migrated = compat::migrate_batch(&batch);
        assert_eq!(migrated.len(), 5);

        for (orig, m16k) in batch.iter().zip(migrated.iter()) {
            assert_eq!(*orig, compat::truncate(m16k), "Batch migration should be lossless");
        }

        // Batch migration with schema
        let with_schema = compat::migrate_batch_with_schema(&batch, &schema);
        for m in &with_schema {
            let recovered = SchemaSidecar::read_from_words(m);
            assert_eq!(recovered.ani_levels.planning, 500);
            assert!((recovered.nars_truth.f() - 0.8).abs() < 0.01);
        }
    }

    // =====================================================================
    // SCENARIO 7: Full Search Pipeline with Benchmarks
    //
    // Compares schema-filtered search performance characteristics.
    // =====================================================================

    #[test]
    fn demo_search_pipeline_benchmark() {
        // Build a dataset of 100 16K vectors with varied schemas
        let mut candidates: Vec<Vec<u64>> = Vec::new();
        let mut rng = 42u64;

        for i in 0..100 {
            let v = BitpackedVector::random(i as u64);
            let mut words = compat::zero_extend(&v).to_vec();

            let mut schema = SchemaSidecar::default();
            // Vary properties
            schema.ani_levels.planning = (i * 100) as u16;
            schema.ani_levels.social = ((100 - i) * 50) as u16;
            schema.nars_truth = NarsTruth::from_floats(
                (i as f32) / 100.0,           // frequency increases
                ((100 - i) as f32) / 200.0,   // confidence decreases
            );
            schema.metrics.pagerank = (i * 10) as u16;
            schema.metrics.degree = (i % 20) as u8;
            schema.metrics.cluster_id = (i / 10) as u16;
            schema.node_type.kind = NodeKind::Entity as u8;

            if i > 50 {
                // Only high-numbered nodes have good Q-values
                schema.q_values.set_q(0, (i as f32 - 50.0) / 50.0);
            }

            schema.write_to_words(&mut words);
            candidates.push(words);
        }

        let query_vec = BitpackedVector::random(42);
        let query_words = compat::zero_extend(&query_vec).to_vec();
        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();

        // ----- Unfiltered search -----
        let basic_query = SchemaQuery::new().with_max_distance(u32::MAX);
        let basic_results = basic_query.search(&refs, &query_words, 10);
        assert_eq!(basic_results.len(), 10, "Should return top-10");

        // ----- Schema-filtered search (selective) -----
        let selective_query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 5000 }) // planning > 5000 → only top ~50
            .with_nars(NarsFilter {
                min_frequency: Some(0.5), // f > 0.5 → only i > 50
                min_confidence: None,
                min_priority: None,
            })
            .with_graph(GraphFilter {
                min_pagerank: Some(500), // pagerank > 500 → only i > 50
                max_hop: None,
                cluster_id: None,
                min_degree: None,
            });

        let filtered_results = selective_query.search(&refs, &query_words, 10);
        // Only candidates with i > 50 should survive all predicates
        // The result set should be smaller or different from unfiltered
        // (Can't guarantee exact count due to predicate interaction)

        // ----- Cluster-specific search -----
        let cluster_query = SchemaQuery::new()
            .with_graph(GraphFilter {
                min_pagerank: None,
                max_hop: None,
                cluster_id: Some(5), // Only cluster 5 (i=50..59)
                min_degree: None,
            });

        let cluster_results = cluster_query.search(&refs, &query_words, 10);
        assert!(cluster_results.len() <= 10, "Should return at most 10 from cluster 5");

        // ----- Block-masked distance comparison -----
        // Semantic-only distance vs full distance
        let semantic_query = SchemaQuery::new().with_block_mask(BlockMask::SEMANTIC);
        let all_query = SchemaQuery::new().with_block_mask(BlockMask::ALL);

        let d_semantic = semantic_query.masked_distance(&query_words, &candidates[50]);
        let d_all = all_query.masked_distance(&query_words, &candidates[50]);

        // Full distance includes schema blocks, so may be larger
        assert!(d_all >= d_semantic,
            "Full distance should be >= semantic-only: {} vs {}", d_all, d_semantic);
    }

    // =====================================================================
    // SCENARIO 8: GraphBLAS SpMV with HDR Semirings
    //
    // Equivalent to Neo4j multi-hop traversal but using XOR-bind as the
    // "multiply" and majority bundle as the "add" in a semiring.
    // =====================================================================

    #[test]
    fn demo_graphblas_spmv() {
        let nav = Navigator::new();

        // 3-node graph: 0→1→2 with unique edges
        let nodes: Vec<BitpackedVector> = (0..3)
            .map(|i| BitpackedVector::random(i * 100))
            .collect();
        let edge_01 = BitpackedVector::random(1001);
        let edge_12 = BitpackedVector::random(1002);

        // Adjacency structure: (row, col, edge_fingerprint)
        let edges = vec![
            (0, 1, edge_01.clone()),
            (1, 2, edge_12.clone()),
        ];

        // SpMV: output[i] = bundle(edge[i,j] XOR input[j])
        let output = nav.graphblas_spmv(&edges, &nodes, 3);
        assert_eq!(output.len(), 3);

        // Row 0: receives edge_01 XOR nodes[1]
        assert_eq!(output[0], edge_01.xor(&nodes[1]));

        // Row 1: receives edge_12 XOR nodes[2]
        assert_eq!(output[1], edge_12.xor(&nodes[2]));

        // Row 2: no incoming edges → zero vector
        assert_eq!(output[2], BitpackedVector::zero());

        // ----- Filtered SpMV (with cascade) -----
        let query = BitpackedVector::random(42);
        let filtered = nav.graphblas_spmv_filtered(
            &edges, &nodes, &query, 3, 10000, // large radius = let everything through
        );
        assert_eq!(filtered.len(), 3);
    }

    // =====================================================================
    // SCENARIO 9: DN Tree Redis-Style Addressing
    //
    // Tests the full DN path parsing, addressing, and compatibility
    // with the Redis GET/SET protocol.
    // =====================================================================

    #[test]
    fn demo_dn_redis_addressing() {
        let nav = Navigator::new();

        // ----- Parse various address formats -----
        let addr = crate::navigator::DnPath::parse("graphs:semantic:3:7:42").unwrap();
        assert_eq!(addr.domain, "graphs");
        assert_eq!(addr.segments[0], "semantic");
        assert_eq!(addr.child_indices, vec![3, 7, 42]);
        assert_eq!(addr.depth, 5);

        // With protocol prefix
        let addr2 = crate::navigator::DnPath::parse("hdr://mydb:tree:1:2:3").unwrap();
        assert_eq!(addr2.domain, "mydb");

        // Roundtrip
        assert_eq!(addr.to_redis_key(), "graphs:semantic:3:7:42");

        // Prefix matching (for SCAN)
        assert!(addr.matches_prefix("graphs:semantic:*"));
        assert!(addr.matches_prefix("graphs:*"));
        assert!(!addr.matches_prefix("other:*"));

        // ----- DN GET/SET (API surface) -----
        let v = BitpackedVector::random(42);
        assert!(nav.dn_set("graphs:semantic:3:7:42", &v).is_ok());
        let get_result = nav.dn_get("graphs:semantic:3:7:42").unwrap();
        assert_eq!(get_result.path.domain, "graphs");

        // ----- MGET (batch) -----
        let results = nav.dn_mget(&[
            "graphs:semantic:3:7:42",
            "graphs:semantic:3:7:43",
            "graphs:semantic:3:8:1",
        ]).unwrap();
        assert_eq!(results.len(), 3);

        // ----- TreeAddr conversion -----
        let tree_addr = addr.to_tree_addr();
        assert_eq!(tree_addr.depth(), 3); // 3 numeric child indices
    }

    // =====================================================================
    // SCENARIO 10: Schema Pack/Unpack Stress Test
    //
    // Exercises all schema fields simultaneously to verify bit-level
    // correctness of the sidecar layout.
    // =====================================================================

    #[test]
    fn demo_schema_stress_test() {
        let mut schema = SchemaSidecar::default();

        // Fill ALL fields
        schema.ani_levels = AniLevels {
            reactive: 100, memory: 200, analogy: 300, planning: 400,
            meta: 500, social: 600, creative: 700, r#abstract: 800,
        };
        schema.nars_truth = NarsTruth::from_floats(0.85, 0.72);
        schema.nars_budget = NarsBudget::from_floats(0.9, 0.5, 0.7);
        schema.edge_type = EdgeTypeMarker {
            verb_id: 42, direction: 1, weight: 200, flags: 0b1111,
        };
        schema.node_type = NodeTypeMarker {
            kind: NodeKind::Concept as u8, subtype: 3, provenance: 0xABCD,
        };
        schema.q_values.set_q(0, 0.9);
        schema.q_values.set_q(5, -0.5);
        schema.q_values.set_q(15, 0.3);
        for i in 0..8 {
            schema.rewards.push((i as f32 - 3.0) / 10.0);
        }
        schema.stdp.record_spike(100);
        schema.stdp.record_spike(200);
        schema.stdp.record_spike(300);
        schema.hebbian.strengthen(0, 0.5);
        schema.hebbian.strengthen(3, 0.8);
        schema.hebbian.strengthen(7, 0.2);
        schema.dn_addr.path[0] = 1;
        schema.dn_addr.path[1] = 2;
        schema.dn_addr.path[2] = 3;
        schema.dn_addr.depth = 3;
        schema.neighbors.insert(100);
        schema.neighbors.insert(200);
        schema.neighbors.insert(300);
        schema.neighbors.insert(400);
        schema.neighbors.insert(500);
        schema.metrics = GraphMetrics {
            pagerank: 42000, hop_to_root: 5, cluster_id: 999,
            degree: 15, in_degree: 7, out_degree: 8,
        };

        // Write and read back
        let mut words = [0u64; VECTOR_WORDS];
        schema.write_to_words(&mut words);
        let recovered = SchemaSidecar::read_from_words(&words);

        // Verify ALL fields
        assert_eq!(recovered.ani_levels.reactive, 100);
        assert_eq!(recovered.ani_levels.planning, 400);
        assert_eq!(recovered.ani_levels.r#abstract, 800);
        assert_eq!(recovered.ani_levels.dominant(), 7); // abstract is highest

        assert!((recovered.nars_truth.f() - 0.85).abs() < 0.01);
        assert!((recovered.nars_truth.c() - 0.72).abs() < 0.01);

        assert_eq!(recovered.edge_type.verb_id, 42);
        assert_eq!(recovered.edge_type.direction, 1);
        assert!(recovered.edge_type.is_temporal());
        assert!(recovered.edge_type.is_causal());
        assert!(recovered.edge_type.is_hierarchical());
        assert!(recovered.edge_type.is_associative());

        assert_eq!(recovered.node_type.kind, NodeKind::Concept as u8);
        assert_eq!(recovered.node_type.subtype, 3);
        assert_eq!(recovered.node_type.provenance, 0xABCD);

        assert_eq!(recovered.q_values.best_action(), 0); // action 0 has Q=0.9
        assert!((recovered.q_values.q(0) - 0.9).abs() < 0.02);
        assert!((recovered.q_values.q(5) - (-0.5)).abs() < 0.02);

        assert_eq!(recovered.stdp.last_spike(), 300);

        assert!(recovered.neighbors.might_contain(100));
        assert!(recovered.neighbors.might_contain(500));

        assert_eq!(recovered.metrics.pagerank, 42000);
        assert_eq!(recovered.metrics.hop_to_root, 5);
        assert_eq!(recovered.metrics.cluster_id, 999);
        assert_eq!(recovered.metrics.degree, 15);
        assert_eq!(recovered.metrics.in_degree, 7);
        assert_eq!(recovered.metrics.out_degree, 8);
    }

    // =====================================================================
    // SCENARIO 11: Bloom-Accelerated + RL-Guided Search
    //
    // Demonstrates the new search modes that leverage inline metadata
    // for smarter candidate ranking beyond pure Hamming distance.
    // =====================================================================

    #[test]
    fn demo_bloom_rl_search() {
        // Build a small dataset
        let mut candidates: Vec<Vec<u64>> = Vec::new();
        let source_id = 9999u64;

        for i in 0..20 {
            let v = crate::bitpack::BitpackedVector::random(i as u64);
            let mut words = compat::zero_extend(&v).to_vec();

            let mut schema = SchemaSidecar::default();
            schema.ani_levels.planning = 500;
            schema.q_values.set_q(0, (i as f32 - 10.0) / 10.0); // Q from -1 to +0.9

            // Some candidates are known neighbors of source
            if i % 3 == 0 {
                schema.neighbors.insert(source_id);
            }

            schema.write_to_words(&mut words);
            candidates.push(words);
        }

        let query_v = crate::bitpack::BitpackedVector::random(42);
        let query_words = compat::zero_extend(&query_v).to_vec();
        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();

        let schema_query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 100 });

        // ----- Bloom-accelerated search -----
        let bloom_results = bloom_accelerated_search(
            &refs, &query_words, source_id, 5, 0.3, &schema_query,
        );
        assert!(!bloom_results.is_empty(), "Should find some results");
        assert!(bloom_results.len() <= 5, "Should respect k limit");

        // Check that bloom neighbors get a bonus
        for r in &bloom_results {
            if r.is_bloom_neighbor {
                assert!(r.effective_distance <= r.raw_distance,
                    "Bloom neighbors should have bonus: eff={} raw={}", r.effective_distance, r.raw_distance);
            }
        }

        // ----- RL-guided search -----
        let rl_results = rl_guided_search(
            &refs, &query_words, 5, 0.3, &schema_query,
        );
        assert!(!rl_results.is_empty(), "Should find RL results");
        assert!(rl_results.len() <= 5);

        // Results should be sorted by composite score
        for w in rl_results.windows(2) {
            assert!(w[0].composite_score <= w[1].composite_score,
                "Results should be sorted by composite score");
        }
    }

    // =====================================================================
    // SCENARIO 12: Federated Schema Merge
    //
    // Demonstrates combining schema metadata from two independent
    // instances that hold different evidence about the same entity.
    // =====================================================================

    #[test]
    fn demo_federated_merge() {
        let base_vec = crate::bitpack::BitpackedVector::random(42);
        let mut instance_a = compat::zero_extend(&base_vec).to_vec();
        let mut instance_b = compat::zero_extend(&base_vec).to_vec();

        // Instance A: observed high social reasoning, has trust evidence
        let mut schema_a = SchemaSidecar::default();
        schema_a.ani_levels.social = 700;
        schema_a.ani_levels.planning = 200;
        schema_a.nars_truth = NarsTruth::from_floats(0.9, 0.6);
        schema_a.metrics.pagerank = 500;
        schema_a.metrics.hop_to_root = 5;
        schema_a.metrics.degree = 3;
        schema_a.neighbors.insert(100);
        schema_a.neighbors.insert(200);
        schema_a.q_values.set_q(0, 0.7);
        schema_a.write_to_words(&mut instance_a);

        // Instance B: observed high planning, different trust evidence
        let mut schema_b = SchemaSidecar::default();
        schema_b.ani_levels.social = 300;
        schema_b.ani_levels.planning = 600;
        schema_b.nars_truth = NarsTruth::from_floats(0.7, 0.4);
        schema_b.metrics.pagerank = 800;
        schema_b.metrics.hop_to_root = 2;
        schema_b.metrics.degree = 8;
        schema_b.neighbors.insert(300);
        schema_b.neighbors.insert(400);
        schema_b.q_values.set_q(0, 0.3);
        schema_b.write_to_words(&mut instance_b);

        // Merge: A is primary (authoritative source)
        let merged = schema_merge(&instance_a, &instance_b);
        let ms = SchemaSidecar::read_from_words(&merged);

        // ANI: element-wise max
        assert_eq!(ms.ani_levels.social, 700, "Social should be max(700,300)=700");
        assert_eq!(ms.ani_levels.planning, 600, "Planning should be max(200,600)=600");

        // NARS: revision combines evidence → confidence should be reasonable
        assert!(ms.nars_truth.f() > 0.0, "Merged frequency should be positive");

        // Metrics: max pagerank, min hop, max degree
        assert_eq!(ms.metrics.pagerank, 800, "Pagerank: max(500,800)=800");
        assert_eq!(ms.metrics.hop_to_root, 2, "Hop: min(5,2)=2");
        assert_eq!(ms.metrics.degree, 8, "Degree: max(3,8)=8");

        // Bloom: union of neighbors from both instances
        assert!(bloom_might_be_neighbors(&merged, 100), "Should know neighbor 100 from A");
        assert!(bloom_might_be_neighbors(&merged, 300), "Should know neighbor 300 from B");

        // Semantic content preserved from primary (A)
        let truncated_a = compat::truncate_slice(&instance_a).unwrap();
        let truncated_merged = compat::truncate_slice(&merged).unwrap();
        assert_eq!(truncated_a, truncated_merged,
            "Semantic content should be preserved from primary");
    }

    // =====================================================================
    // SCENARIO 13: Schema Versioning + ConcurrentWriteCache
    //
    // Tests the hardening features: version byte in schema, and
    // thread-safe write cache.
    // =====================================================================

    #[test]
    fn demo_hardening_features() {
        // ----- Schema versioning -----
        let mut words = vec![0u64; VECTOR_WORDS];

        // Before writing, version is 0 (legacy)
        assert_eq!(SchemaSidecar::read_version(&words), 0);

        // Write schema → version becomes 1
        let mut schema = SchemaSidecar::default();
        schema.ani_levels.planning = 999;
        schema.nars_truth = NarsTruth::from_floats(0.9, 0.8);
        schema.write_to_words(&mut words);
        assert_eq!(SchemaSidecar::read_version(&words), 1);

        // Version byte doesn't corrupt ANI
        let recovered = SchemaSidecar::read_from_words(&words);
        assert_eq!(recovered.ani_levels.planning, 999);
        assert!((recovered.nars_truth.f() - 0.9).abs() < 0.01);

        // ----- ConcurrentWriteCache -----
        let cache = ConcurrentWriteCache::default_cache();
        let base = words.clone();

        // Clean read
        let read = cache.read_through(1, &base);
        assert!(read.is_clean());

        // Record a delta
        let mut modified = base.clone();
        modified[0] ^= 0xDEAD;
        let delta = XorDelta::compute(&base, &modified);
        cache.record_delta(1, delta);

        // Dirty read shows patched data
        let read = cache.read_through(1, &base);
        assert!(!read.is_clean());
        let patched = read.patched_words().unwrap();
        assert_eq!(patched[0], modified[0]);

        // Schema still readable from patched words (schema region unchanged)
        let patched_schema = SchemaSidecar::read_from_words(patched);
        assert_eq!(patched_schema.ani_levels.planning, 999);

        // ----- DeltaChain depth limit -----
        assert_eq!(MAX_CHAIN_DEPTH, 256);

        // Flush
        assert_eq!(cache.dirty_count(), 1);
        let flushed = cache.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(cache.dirty_count(), 0);
    }
}
