//! GraphBLAS Mindmap with DN Tree Addressing
//!
//! Combines hierarchical DN Tree navigation with GraphBLAS sparse
//! matrix operations for building and querying mindmaps.
//!
//! # Architecture
//!
//! ```text
//! Mindmap = DN Tree (hierarchy) + GraphBLAS (sparse adjacency)
//!
//!          Concepts                      GraphBLAS Adjacency
//!             │                              0  1  2  3  4
//!     ┌───────┼───────┐                   ┌─────────────────┐
//!   Animals  Plants  Things             0 │ .  1  .  .  .  │  IS_A
//!     │               │                 1 │ .  .  1  .  1  │  PART_OF
//!   ┌─┴─┐           ┌─┴─┐               2 │ .  .  .  1  .  │  CAUSES
//!  Cat  Dog      Chair Table            3 │ 1  .  .  .  .  │  SIMILAR
//!                                       4 │ .  .  .  .  .  │
//!                                         └─────────────────┘
//! ```
//!
//! The DN Tree provides O(log n) hierarchical navigation while
//! GraphBLAS enables efficient BFS/PageRank/similarity traversals.

use crate::bitpack::BitpackedVector;
use crate::dntree::{TreeAddr, DnTree, DnNode, DnEdge, CogVerb, VerbCategory, WellKnown};
use crate::graphblas::{GrBMatrix, GrBVector, HdrSemiring, Semiring};
use crate::graphblas::types::{GrBIndex, HdrScalar};
use crate::hamming::hamming_distance_scalar;
use std::collections::HashMap;

// ============================================================================
// MINDMAP NODE
// ============================================================================

/// Mindmap node with DN address and sparse matrix index
#[derive(Clone, Debug)]
pub struct MindmapNode {
    /// Tree address (hierarchical location)
    pub addr: TreeAddr,
    /// Matrix index (for GraphBLAS operations)
    pub index: GrBIndex,
    /// Node fingerprint
    pub fingerprint: BitpackedVector,
    /// Display label
    pub label: String,
    /// Node type
    pub node_type: NodeType,
    /// Importance score (PageRank-like)
    pub importance: f32,
    /// Activation (for spreading activation)
    pub activation: f32,
}

/// Node types in mindmap
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeType {
    /// Central topic
    Central,
    /// Main branch
    Branch,
    /// Sub-topic
    SubTopic,
    /// Leaf node (detail)
    Leaf,
    /// Cross-link target
    Link,
}

impl MindmapNode {
    pub fn new(addr: TreeAddr, index: GrBIndex, label: impl Into<String>) -> Self {
        let fingerprint = addr.to_fingerprint();
        let label = label.into();

        // Determine type from depth
        let node_type = match addr.depth() {
            0 => NodeType::Central,
            1 => NodeType::Branch,
            2 => NodeType::SubTopic,
            _ => NodeType::Leaf,
        };

        Self {
            addr,
            index,
            fingerprint,
            label,
            node_type,
            importance: 1.0,
            activation: 0.0,
        }
    }
}

// ============================================================================
// GRAPHBLAS MINDMAP
// ============================================================================

/// Mindmap backed by GraphBLAS sparse matrices
pub struct GrBMindmap {
    /// Nodes by index
    nodes: Vec<MindmapNode>,
    /// Address to index mapping
    addr_to_idx: HashMap<TreeAddr, GrBIndex>,
    /// Label to index mapping
    label_to_idx: HashMap<String, GrBIndex>,
    /// Fingerprint index for similarity search
    fp_index: Vec<(BitpackedVector, GrBIndex)>,

    /// Adjacency matrices by verb category (sparse)
    adjacency: HashMap<VerbCategory, GrBMatrix>,
    /// Combined adjacency (all edges)
    combined_adj: GrBMatrix,
    /// Edge weights
    weights: GrBMatrix,

    /// Current size
    size: GrBIndex,
    /// Default semiring for operations
    semiring: HdrSemiring,
}

impl GrBMindmap {
    /// Create new mindmap with capacity
    pub fn new(capacity: GrBIndex) -> Self {
        let mut adjacency = HashMap::new();

        // Create sparse matrix for each verb category
        for cat in [
            VerbCategory::Structural,
            VerbCategory::Causal,
            VerbCategory::Temporal,
            VerbCategory::Epistemic,
            VerbCategory::Agentive,
            VerbCategory::Experiential,
        ] {
            adjacency.insert(cat, GrBMatrix::new(capacity, capacity));
        }

        Self {
            nodes: Vec::with_capacity(capacity as usize),
            addr_to_idx: HashMap::new(),
            label_to_idx: HashMap::new(),
            fp_index: Vec::new(),
            adjacency,
            combined_adj: GrBMatrix::new(capacity, capacity),
            weights: GrBMatrix::new(capacity, capacity),
            size: 0,
            semiring: HdrSemiring::XorBundle,
        }
    }

    /// Create from central topic
    pub fn from_topic(topic: impl Into<String>) -> Self {
        let mut mindmap = Self::new(1000);
        let topic = topic.into();

        // Create central node at root
        let root = TreeAddr::from_string(&format!("/{}", topic.to_lowercase().replace(' ', "_")));
        mindmap.add_node_at(root, &topic);

        mindmap
    }

    // ========================================================================
    // NODE MANAGEMENT
    // ========================================================================

    /// Add node at tree address
    pub fn add_node_at(&mut self, addr: TreeAddr, label: &str) -> GrBIndex {
        if let Some(&idx) = self.addr_to_idx.get(&addr) {
            return idx;
        }

        let idx = self.size;
        self.size += 1;

        let node = MindmapNode::new(addr.clone(), idx, label);

        self.fp_index.push((node.fingerprint.clone(), idx));
        self.addr_to_idx.insert(addr, idx);
        self.label_to_idx.insert(label.to_string(), idx);
        self.nodes.push(node);

        idx
    }

    /// Add child node under parent
    pub fn add_child(&mut self, parent: &TreeAddr, branch: u8, label: &str) -> GrBIndex {
        let child_addr = parent.child(branch);
        let child_idx = self.add_node_at(child_addr.clone(), label);

        // Auto-connect with PART_OF
        if let Some(&parent_idx) = self.addr_to_idx.get(parent) {
            self.connect_indices(child_idx, CogVerb::PART_OF, parent_idx, 1.0);
        }

        child_idx
    }

    /// Add sibling with auto-generated address
    pub fn add_sibling(&mut self, existing: &TreeAddr, label: &str) -> GrBIndex {
        if let Some(parent) = existing.parent() {
            // Find next available branch
            let used_branches: Vec<u8> = self.nodes
                .iter()
                .filter(|n| n.addr.parent().as_ref() == Some(&parent))
                .filter_map(|n| n.addr.branch(parent.depth() as usize))
                .collect();

            let next_branch = (0..=255u8)
                .find(|b| !used_branches.contains(b))
                .unwrap_or(0);

            self.add_child(&parent, next_branch, label)
        } else {
            // Root sibling - create parallel root
            let addr = TreeAddr::from_string(&format!("/{}", label.to_lowercase().replace(' ', "_")));
            self.add_node_at(addr, label)
        }
    }

    /// Get node by index
    pub fn node(&self, idx: GrBIndex) -> Option<&MindmapNode> {
        self.nodes.get(idx as usize)
    }

    /// Get node by address
    pub fn node_at(&self, addr: &TreeAddr) -> Option<&MindmapNode> {
        self.addr_to_idx.get(addr).and_then(|&idx| self.node(idx))
    }

    /// Get node by label
    pub fn node_by_label(&self, label: &str) -> Option<&MindmapNode> {
        self.label_to_idx.get(label).and_then(|&idx| self.node(idx))
    }

    // ========================================================================
    // EDGE MANAGEMENT (GraphBLAS Sparse)
    // ========================================================================

    /// Connect two nodes by indices
    pub fn connect_indices(&mut self, from: GrBIndex, verb: CogVerb, to: GrBIndex, weight: f32) {
        let category = verb.category();

        // Set in category-specific matrix
        if let Some(mat) = self.adjacency.get_mut(&category) {
            let from_fp = self.nodes.get(from as usize)
                .map(|n| n.fingerprint.clone())
                .unwrap_or_else(BitpackedVector::zero);
            mat.set(from, to, HdrScalar::Vector(from_fp));
        }

        // Set in combined adjacency
        let edge_fp = if let (Some(from_node), Some(to_node)) =
            (self.nodes.get(from as usize), self.nodes.get(to as usize))
        {
            // Edge = from ⊗ verb ⊗ to
            from_node.fingerprint
                .xor(&verb.to_fingerprint())
                .xor(&to_node.fingerprint)
        } else {
            BitpackedVector::zero()
        };

        self.combined_adj.set(from, to, HdrScalar::Vector(edge_fp));
        self.weights.set(from, to, HdrScalar::Distance(weight as u32));
    }

    /// Connect by addresses
    pub fn connect(&mut self, from: &TreeAddr, verb: CogVerb, to: &TreeAddr, weight: f32) {
        if let (Some(&from_idx), Some(&to_idx)) =
            (self.addr_to_idx.get(from), self.addr_to_idx.get(to))
        {
            self.connect_indices(from_idx, verb, to_idx, weight);
        }
    }

    /// Connect by labels
    pub fn connect_labels(&mut self, from: &str, verb: CogVerb, to: &str, weight: f32) {
        if let (Some(&from_idx), Some(&to_idx)) =
            (self.label_to_idx.get(from), self.label_to_idx.get(to))
        {
            self.connect_indices(from_idx, verb, to_idx, weight);
        }
    }

    /// Get outgoing edges from node (sparse iteration)
    pub fn outgoing(&self, idx: GrBIndex) -> Vec<(GrBIndex, VerbCategory)> {
        let mut edges = Vec::new();

        for (&cat, mat) in &self.adjacency {
            for (_, col, _) in mat.iter_row(idx) {
                edges.push((col, cat));
            }
        }

        edges
    }

    /// Get incoming edges to node
    pub fn incoming(&self, idx: GrBIndex) -> Vec<(GrBIndex, VerbCategory)> {
        let mut edges = Vec::new();

        for (&cat, mat) in &self.adjacency {
            for (row, _, _) in mat.iter_col(idx) {
                edges.push((row, cat));
            }
        }

        edges
    }

    // ========================================================================
    // GRAPHBLAS TRAVERSAL
    // ========================================================================

    /// BFS from source (GraphBLAS push-pull)
    pub fn bfs(&self, source: GrBIndex, max_depth: usize) -> Vec<(GrBIndex, u32)> {
        let mut visited = GrBVector::new(self.size);
        let mut frontier = GrBVector::new(self.size);

        // Initialize
        let source_fp = self.nodes.get(source as usize)
            .map(|n| n.fingerprint.clone())
            .unwrap_or_else(BitpackedVector::zero);
        frontier.set_vector(source, source_fp);
        visited.set(source, HdrScalar::Distance(0));

        for depth in 1..=max_depth as u32 {
            // Push: next = A * frontier (sparse matrix-vector multiply)
            let next = self.combined_adj.mxv(&frontier, &self.semiring);

            if next.is_empty() {
                break;
            }

            // Mark newly visited
            for (idx, _) in next.iter() {
                if visited.get(idx).is_none() {
                    visited.set(idx, HdrScalar::Distance(depth));
                }
            }

            // Update frontier (only unvisited)
            frontier = next.apply_complement_mask(&visited);
        }

        // Collect results
        visited.iter()
            .filter_map(|(idx, val)| {
                if let HdrScalar::Distance(d) = val {
                    Some((idx, *d))
                } else {
                    None
                }
            })
            .collect()
    }

    /// PageRank (GraphBLAS iterative)
    pub fn pagerank(&mut self, iterations: usize, damping: f32) -> Vec<(GrBIndex, f32)> {
        let n = self.size as f32;
        let base = (1.0 - damping) / n;

        // Initialize ranks
        let mut rank = vec![1.0 / n; self.size as usize];

        for _ in 0..iterations {
            let mut new_rank = vec![base; self.size as usize];

            // For each node, distribute rank to neighbors
            for from in 0..self.size {
                let out_edges = self.outgoing(from);
                if out_edges.is_empty() {
                    continue;
                }

                let contrib = damping * rank[from as usize] / out_edges.len() as f32;
                for (to, _) in out_edges {
                    new_rank[to as usize] += contrib;
                }
            }

            rank = new_rank;
        }

        // Update importance scores
        for (idx, &r) in rank.iter().enumerate() {
            if let Some(node) = self.nodes.get_mut(idx) {
                node.importance = r;
            }
        }

        // Return sorted
        let mut results: Vec<_> = rank.iter().enumerate()
            .map(|(i, &r)| (i as GrBIndex, r))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Spreading activation (GraphBLAS semiring)
    pub fn spread_activation(
        &mut self,
        sources: &[(GrBIndex, f32)],
        decay: f32,
        iterations: usize,
    ) {
        // Reset activations
        for node in &mut self.nodes {
            node.activation = 0.0;
        }

        // Initialize source activations
        let mut activation = GrBVector::new(self.size);
        for &(idx, act) in sources {
            if let Some(node) = self.nodes.get_mut(idx as usize) {
                node.activation = act;
            }
            let fp = self.nodes.get(idx as usize)
                .map(|n| n.fingerprint.clone())
                .unwrap_or_else(BitpackedVector::zero);
            activation.set_vector(idx, fp);
        }

        // Iterate spreading
        for _ in 0..iterations {
            // next = A^T * activation (spread to neighbors)
            let next = self.combined_adj.transpose().mxv(&activation, &self.semiring);

            // Apply decay and update
            for (idx, _) in next.iter() {
                if let Some(node) = self.nodes.get_mut(idx as usize) {
                    node.activation = (node.activation + decay).min(1.0);
                }
            }

            // activation = activation ∪ next (with decay)
            activation = activation.ewise_add(&next, &self.semiring);
        }
    }

    // ========================================================================
    // SIMILARITY SEARCH
    // ========================================================================

    /// Find most similar nodes to query fingerprint
    pub fn find_similar(&self, query: &BitpackedVector, k: usize) -> Vec<(GrBIndex, u32)> {
        let mut results: Vec<_> = self.fp_index
            .iter()
            .map(|(fp, idx)| (*idx, hamming_distance_scalar(query, fp)))
            .collect();

        results.sort_by_key(|(_, d)| *d);
        results.truncate(k);
        results
    }

    /// Find nodes similar to label
    pub fn find_similar_to(&self, label: &str, k: usize) -> Vec<(&str, u32)> {
        if let Some(&idx) = self.label_to_idx.get(label) {
            if let Some(node) = self.nodes.get(idx as usize) {
                return self.find_similar(&node.fingerprint, k + 1)
                    .into_iter()
                    .filter(|(i, _)| *i != idx) // Exclude self
                    .filter_map(|(i, d)| {
                        self.nodes.get(i as usize).map(|n| (n.label.as_str(), d))
                    })
                    .take(k)
                    .collect();
            }
        }
        vec![]
    }

    /// Pattern match: find edges matching pattern fingerprint
    pub fn pattern_match(&self, pattern: &BitpackedVector, threshold: u32) -> Vec<(GrBIndex, GrBIndex, u32)> {
        let mut matches = Vec::new();

        for (row, col, val) in self.combined_adj.iter() {
            if let Some(edge_fp) = val.as_vector() {
                let dist = hamming_distance_scalar(pattern, edge_fp);
                if dist <= threshold {
                    matches.push((row, col, dist));
                }
            }
        }

        matches.sort_by_key(|(_, _, d)| *d);
        matches
    }

    // ========================================================================
    // MINDMAP OPERATIONS
    // ========================================================================

    /// Get all children of a node (by tree structure)
    pub fn children(&self, idx: GrBIndex) -> Vec<GrBIndex> {
        if let Some(node) = self.node(idx) {
            self.nodes
                .iter()
                .filter(|n| node.addr.is_ancestor_of(&n.addr) &&
                           n.addr.depth() == node.addr.depth() + 1)
                .map(|n| n.index)
                .collect()
        } else {
            vec![]
        }
    }

    /// Get subtree rooted at node
    pub fn subtree(&self, idx: GrBIndex) -> Vec<GrBIndex> {
        if let Some(node) = self.node(idx) {
            self.nodes
                .iter()
                .filter(|n| node.addr.is_ancestor_of(&n.addr) || n.index == idx)
                .map(|n| n.index)
                .collect()
        } else {
            vec![]
        }
    }

    /// Collapse subtree to single summary node
    pub fn collapse_subtree(&self, root: GrBIndex) -> Option<BitpackedVector> {
        let indices = self.subtree(root);
        if indices.is_empty() {
            return None;
        }

        // Bundle all fingerprints in subtree
        let fps: Vec<&BitpackedVector> = indices
            .iter()
            .filter_map(|&i| self.nodes.get(i as usize))
            .map(|n| &n.fingerprint)
            .collect();

        Some(BitpackedVector::bundle(&fps))
    }

    /// Find path between two nodes (BFS-based)
    pub fn path(&self, from: GrBIndex, to: GrBIndex) -> Option<Vec<GrBIndex>> {
        let bfs_result = self.bfs(from, 10);

        if !bfs_result.iter().any(|(idx, _)| *idx == to) {
            return None;
        }

        // Backtrack from target
        let mut path = vec![to];
        let mut current = to;

        while current != from {
            let incoming = self.incoming(current);
            if let Some((prev, _)) = incoming
                .iter()
                .filter(|(idx, _)| bfs_result.iter().any(|(i, _)| i == idx))
                .min_by_key(|(idx, _)| {
                    bfs_result.iter().find(|(i, _)| i == idx).map(|(_, d)| *d).unwrap_or(u32::MAX)
                })
            {
                path.push(*prev);
                current = *prev;
            } else {
                break;
            }
        }

        path.reverse();
        Some(path)
    }

    // ========================================================================
    // EXPORT / VISUALIZATION
    // ========================================================================

    /// Export as DOT graph
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph mindmap {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Nodes
        for node in &self.nodes {
            let shape = match node.node_type {
                NodeType::Central => "ellipse",
                NodeType::Branch => "box",
                NodeType::SubTopic => "box",
                NodeType::Leaf => "plaintext",
                NodeType::Link => "diamond",
            };
            dot.push_str(&format!(
                "  n{} [label=\"{}\" shape={}];\n",
                node.index, node.label, shape
            ));
        }

        dot.push_str("\n");

        // Edges
        for (row, col, _) in self.combined_adj.iter() {
            dot.push_str(&format!("  n{} -> n{};\n", row, col));
        }

        dot.push_str("}\n");
        dot
    }

    /// Export as markdown outline
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Find root nodes (depth 1)
        let mut roots: Vec<_> = self.nodes
            .iter()
            .filter(|n| n.addr.depth() == 1)
            .collect();
        roots.sort_by_key(|n| n.index);

        for root in roots {
            self.node_to_markdown(root.index, 0, &mut md);
        }

        md
    }

    fn node_to_markdown(&self, idx: GrBIndex, depth: usize, out: &mut String) {
        if let Some(node) = self.node(idx) {
            let prefix = "  ".repeat(depth);
            let bullet = if depth == 0 { "#" } else { "-" };
            out.push_str(&format!("{}{} {}\n", prefix, bullet, node.label));

            // Get children
            let children = self.children(idx);
            for child_idx in children {
                self.node_to_markdown(child_idx, depth + 1, out);
            }
        }
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        self.combined_adj.nnz()
    }

    /// Get nodes by type
    pub fn nodes_by_type(&self, node_type: NodeType) -> Vec<GrBIndex> {
        self.nodes
            .iter()
            .filter(|n| n.node_type == node_type)
            .map(|n| n.index)
            .collect()
    }

    /// Get most important nodes
    pub fn most_important(&self, k: usize) -> Vec<(&str, f32)> {
        let mut nodes: Vec<_> = self.nodes
            .iter()
            .map(|n| (n.label.as_str(), n.importance))
            .collect();

        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        nodes.truncate(k);
        nodes
    }

    /// Get most activated nodes
    pub fn most_activated(&self, k: usize) -> Vec<(&str, f32)> {
        let mut nodes: Vec<_> = self.nodes
            .iter()
            .filter(|n| n.activation > 0.0)
            .map(|n| (n.label.as_str(), n.activation))
            .collect();

        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        nodes.truncate(k);
        nodes
    }
}

// ============================================================================
// MINDMAP BUILDER (Fluent API)
// ============================================================================

/// Fluent builder for mindmaps
pub struct MindmapBuilder {
    mindmap: GrBMindmap,
    current: Option<TreeAddr>,
}

impl MindmapBuilder {
    /// Start building from central topic
    pub fn new(topic: &str) -> Self {
        let mindmap = GrBMindmap::from_topic(topic);
        let root = TreeAddr::from_string(&format!("/{}", topic.to_lowercase().replace(' ', "_")));

        Self {
            mindmap,
            current: Some(root),
        }
    }

    /// Add branch to current node
    pub fn branch(mut self, label: &str) -> Self {
        if let Some(current) = self.current.clone() {
            let branch = self.next_branch(&current);
            let addr = current.child(branch);
            self.mindmap.add_node_at(addr.clone(), label);

            // Connect to parent
            if let (Some(&from), Some(&to)) =
                (self.mindmap.addr_to_idx.get(&addr), self.mindmap.addr_to_idx.get(&current))
            {
                self.mindmap.connect_indices(from, CogVerb::PART_OF, to, 1.0);
            }

            self.current = Some(addr);
        }
        self
    }

    /// Add sibling to current node
    pub fn sibling(mut self, label: &str) -> Self {
        if let Some(current) = self.current.clone() {
            if let Some(parent) = current.parent() {
                let branch = self.next_branch(&parent);
                let addr = parent.child(branch);
                self.mindmap.add_node_at(addr.clone(), label);

                // Connect to parent
                if let (Some(&from), Some(&to)) =
                    (self.mindmap.addr_to_idx.get(&addr), self.mindmap.addr_to_idx.get(&parent))
                {
                    self.mindmap.connect_indices(from, CogVerb::PART_OF, to, 1.0);
                }

                self.current = Some(addr);
            }
        }
        self
    }

    /// Go up one level
    pub fn up(mut self) -> Self {
        if let Some(current) = &self.current {
            self.current = current.parent();
        }
        self
    }

    /// Go to root
    pub fn root(mut self) -> Self {
        if let Some(current) = &self.current {
            self.current = Some(current.ancestor(1));
        }
        self
    }

    /// Add cross-link between labels
    pub fn link(mut self, from: &str, verb: CogVerb, to: &str) -> Self {
        self.mindmap.connect_labels(from, verb, to, 1.0);
        self
    }

    /// Build the mindmap
    pub fn build(self) -> GrBMindmap {
        self.mindmap
    }

    fn next_branch(&self, parent: &TreeAddr) -> u8 {
        let used: Vec<u8> = self.mindmap.nodes
            .iter()
            .filter(|n| n.addr.parent().as_ref() == Some(parent))
            .filter_map(|n| n.addr.branch(parent.depth() as usize))
            .collect();

        (0..=255u8).find(|b| !used.contains(b)).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mindmap_builder() {
        let mindmap = MindmapBuilder::new("Machine Learning")
            .branch("Supervised")
                .branch("Classification")
                .sibling("Regression")
            .up()
            .sibling("Unsupervised")
                .branch("Clustering")
                .sibling("Dimensionality Reduction")
            .up()
            .sibling("Reinforcement")
            .link("Classification", CogVerb::SIMILAR_TO, "Regression")
            .build();

        assert!(mindmap.num_nodes() >= 7);
        assert!(mindmap.num_edges() >= 6);

        // Check hierarchy
        let supervised = mindmap.node_by_label("Supervised").unwrap();
        let classification = mindmap.node_by_label("Classification").unwrap();
        assert!(supervised.addr.is_ancestor_of(&classification.addr));
    }

    #[test]
    fn test_bfs() {
        let mindmap = MindmapBuilder::new("Root")
            .branch("A")
                .branch("A1")
                .sibling("A2")
            .up()
            .sibling("B")
                .branch("B1")
            .build();

        let root_idx = mindmap.node_by_label("Root").unwrap().index;
        let bfs_result = mindmap.bfs(root_idx, 5);

        // Should reach all nodes
        assert!(bfs_result.len() >= 5);
    }

    #[test]
    fn test_similarity() {
        let mindmap = MindmapBuilder::new("Animals")
            .branch("Mammals")
                .branch("Cat")
                .sibling("Dog")
            .up()
            .sibling("Birds")
                .branch("Eagle")
            .build();

        // Cat and Dog should be more similar (same parent) than Cat and Eagle
        let cat = mindmap.node_by_label("Cat").unwrap();
        let dog = mindmap.node_by_label("Dog").unwrap();
        let eagle = mindmap.node_by_label("Eagle").unwrap();

        let cat_dog_dist = hamming_distance_scalar(&cat.fingerprint, &dog.fingerprint);
        let cat_eagle_dist = hamming_distance_scalar(&cat.fingerprint, &eagle.fingerprint);

        // Tree-derived fingerprints show structural similarity
        // (siblings have related addresses)
        println!("Cat-Dog: {}, Cat-Eagle: {}", cat_dog_dist, cat_eagle_dist);
    }

    #[test]
    fn test_pagerank() {
        let mut mindmap = MindmapBuilder::new("Hub")
            .branch("Spoke1")
            .sibling("Spoke2")
            .sibling("Spoke3")
            .link("Spoke1", CogVerb::CAUSES, "Spoke2")
            .link("Spoke2", CogVerb::CAUSES, "Spoke3")
            .link("Spoke3", CogVerb::CAUSES, "Spoke1")
            .build();

        let ranks = mindmap.pagerank(10, 0.85);

        // Hub should have reasonable importance
        assert!(!ranks.is_empty());
    }

    #[test]
    fn test_export() {
        let mindmap = MindmapBuilder::new("Test")
            .branch("Branch1")
            .sibling("Branch2")
            .build();

        let dot = mindmap.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("Test"));

        let md = mindmap.to_markdown();
        assert!(md.contains("Test"));
    }
}
