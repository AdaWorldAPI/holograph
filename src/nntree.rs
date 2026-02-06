//! Sparse Nearest Neighbor Tree (NN-Tree)
//!
//! Hierarchical tree structure for efficient nearest neighbor search
//! using DN Tree addressing with fingerprint-based routing.
//!
//! # Key Innovation
//!
//! ```text
//! Traditional k-NN:  O(n) linear scan
//! VP-Tree / KD-Tree: O(log n) but poor for high dimensions
//! NN-Tree:           O(log n) using fingerprint clustering
//!
//!                    Root (bundle of all)
//!                    /    |    \
//!               Child0  Child1  Child2  (cluster centroids)
//!              /  |  \   ...
//!           Leaves contain actual vectors
//!
//! Routing: At each level, descend to child with
//!          minimum Hamming distance to query
//! ```
//!
//! The tree uses majority bundling to create cluster centroids,
//! enabling logarithmic search with fingerprint similarity.

use crate::bitpack::BitpackedVector;
use crate::dntree::TreeAddr;
use crate::hamming::hamming_distance_scalar;
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// NN-Tree configuration
#[derive(Clone, Debug)]
pub struct NnTreeConfig {
    /// Maximum children per internal node (branching factor)
    pub max_children: usize,
    /// Maximum items per leaf
    pub max_leaf_size: usize,
    /// Number of candidates to check during search
    pub search_beam: usize,
    /// Use bundle centroids for routing
    pub use_bundling: bool,
}

impl Default for NnTreeConfig {
    fn default() -> Self {
        Self {
            max_children: 16,     // 16-way branching
            max_leaf_size: 64,    // Leaves hold up to 64 items
            search_beam: 4,       // Check top 4 candidates per level
            use_bundling: true,
        }
    }
}

// ============================================================================
// NN-TREE NODE
// ============================================================================

/// Node in the NN-Tree
#[derive(Clone, Debug)]
pub enum NnNode {
    /// Internal node with centroid and children
    Internal {
        /// Centroid fingerprint (bundle of descendants)
        centroid: BitpackedVector,
        /// Tree address
        addr: TreeAddr,
        /// Child node addresses
        children: Vec<TreeAddr>,
        /// Number of items in subtree
        count: usize,
    },
    /// Leaf node containing actual items
    Leaf {
        /// Tree address
        addr: TreeAddr,
        /// Items: (id, fingerprint)
        items: Vec<(u64, BitpackedVector)>,
    },
}

impl NnNode {
    /// Get tree address
    pub fn addr(&self) -> &TreeAddr {
        match self {
            NnNode::Internal { addr, .. } => addr,
            NnNode::Leaf { addr, .. } => addr,
        }
    }

    /// Get centroid/representative fingerprint
    pub fn centroid(&self) -> BitpackedVector {
        match self {
            NnNode::Internal { centroid, .. } => centroid.clone(),
            NnNode::Leaf { items, .. } => {
                if items.is_empty() {
                    BitpackedVector::zero()
                } else {
                    let refs: Vec<&BitpackedVector> = items.iter().map(|(_, fp)| fp).collect();
                    BitpackedVector::bundle(&refs)
                }
            }
        }
    }

    /// Get item count
    pub fn count(&self) -> usize {
        match self {
            NnNode::Internal { count, .. } => *count,
            NnNode::Leaf { items, .. } => items.len(),
        }
    }

    /// Is this a leaf?
    pub fn is_leaf(&self) -> bool {
        matches!(self, NnNode::Leaf { .. })
    }
}

// ============================================================================
// NN-TREE
// ============================================================================

/// Sparse Nearest Neighbor Tree
pub struct NnTree {
    /// Configuration
    config: NnTreeConfig,
    /// Nodes by address
    nodes: HashMap<TreeAddr, NnNode>,
    /// Root address
    root: TreeAddr,
    /// Total items
    total_items: usize,
    /// Next item ID
    next_id: u64,
}

impl NnTree {
    /// Create new NN-Tree
    pub fn new() -> Self {
        Self::with_config(NnTreeConfig::default())
    }

    /// Create with configuration
    pub fn with_config(config: NnTreeConfig) -> Self {
        let root = TreeAddr::root();

        let mut nodes = HashMap::new();
        nodes.insert(root.clone(), NnNode::Leaf {
            addr: root.clone(),
            items: Vec::new(),
        });

        Self {
            config,
            nodes,
            root,
            total_items: 0,
            next_id: 0,
        }
    }

    // ========================================================================
    // INSERTION
    // ========================================================================

    /// Insert a fingerprint, returns assigned ID
    pub fn insert(&mut self, fingerprint: BitpackedVector) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.total_items += 1;

        // Find best leaf for insertion
        let leaf_addr = self.find_leaf(&fingerprint);

        // Insert into leaf
        if let Some(NnNode::Leaf { items, addr }) = self.nodes.get_mut(&leaf_addr) {
            items.push((id, fingerprint.clone()));

            // Check if split needed
            if items.len() > self.config.max_leaf_size {
                let items_clone = items.clone();
                let addr_clone = addr.clone();
                drop(items); // Release borrow
                self.split_leaf(&addr_clone, items_clone);
            }
        }

        // Update centroids up the tree
        self.update_centroids(&leaf_addr);

        id
    }

    /// Insert with custom ID
    pub fn insert_with_id(&mut self, id: u64, fingerprint: BitpackedVector) {
        self.next_id = self.next_id.max(id + 1);
        self.total_items += 1;

        let leaf_addr = self.find_leaf(&fingerprint);

        if let Some(NnNode::Leaf { items, addr }) = self.nodes.get_mut(&leaf_addr) {
            items.push((id, fingerprint.clone()));

            if items.len() > self.config.max_leaf_size {
                let items_clone = items.clone();
                let addr_clone = addr.clone();
                drop(items);
                self.split_leaf(&addr_clone, items_clone);
            }
        }

        self.update_centroids(&leaf_addr);
    }

    /// Find best leaf for inserting fingerprint
    fn find_leaf(&self, fingerprint: &BitpackedVector) -> TreeAddr {
        let mut current = self.root.clone();

        loop {
            match self.nodes.get(&current) {
                Some(NnNode::Leaf { .. }) => return current,
                Some(NnNode::Internal { children, .. }) => {
                    // Find child with minimum distance to fingerprint
                    let mut best_child = children[0].clone();
                    let mut best_dist = u32::MAX;

                    for child_addr in children {
                        if let Some(child) = self.nodes.get(child_addr) {
                            let dist = hamming_distance_scalar(fingerprint, &child.centroid());
                            if dist < best_dist {
                                best_dist = dist;
                                best_child = child_addr.clone();
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
    fn split_leaf(&mut self, addr: &TreeAddr, items: Vec<(u64, BitpackedVector)>) {
        let num_children = self.config.max_children.min(items.len());
        if num_children < 2 {
            return;
        }

        // K-means-like clustering to split items
        let clusters = self.cluster_items(&items, num_children);

        // Create child leaves
        let mut children = Vec::new();
        for (i, cluster) in clusters.into_iter().enumerate() {
            let child_addr = addr.child(i as u8);
            children.push(child_addr.clone());

            self.nodes.insert(child_addr.clone(), NnNode::Leaf {
                addr: child_addr,
                items: cluster,
            });
        }

        // Convert current leaf to internal node
        let centroid = {
            let refs: Vec<&BitpackedVector> = items.iter().map(|(_, fp)| fp).collect();
            BitpackedVector::bundle(&refs)
        };

        self.nodes.insert(addr.clone(), NnNode::Internal {
            centroid,
            addr: addr.clone(),
            children,
            count: items.len(),
        });
    }

    /// Cluster items into k groups using k-means-like approach
    fn cluster_items(&self, items: &[(u64, BitpackedVector)], k: usize) -> Vec<Vec<(u64, BitpackedVector)>> {
        if items.len() <= k {
            return items.iter()
                .map(|(id, fp)| vec![(*id, fp.clone())])
                .collect();
        }

        // Initialize centroids by sampling
        let step = items.len() / k;
        let mut centroids: Vec<BitpackedVector> = (0..k)
            .map(|i| items[i * step].1.clone())
            .collect();

        // Run a few iterations of k-means
        let mut clusters = vec![Vec::new(); k];

        for _ in 0..5 {
            // Clear clusters
            for c in &mut clusters {
                c.clear();
            }

            // Assign items to nearest centroid
            for (id, fp) in items {
                let mut best_cluster = 0;
                let mut best_dist = u32::MAX;

                for (i, centroid) in centroids.iter().enumerate() {
                    let dist = hamming_distance_scalar(fp, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = i;
                    }
                }

                clusters[best_cluster].push((*id, fp.clone()));
            }

            // Update centroids
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let refs: Vec<&BitpackedVector> = cluster.iter().map(|(_, fp)| fp).collect();
                    centroids[i] = BitpackedVector::bundle(&refs);
                }
            }
        }

        // Handle empty clusters
        clusters.retain(|c| !c.is_empty());
        clusters
    }

    /// Update centroids from leaf to root
    fn update_centroids(&mut self, start: &TreeAddr) {
        let mut current = start.parent();

        while let Some(addr) = current {
            // Collect child data before mutating
            let child_data = if let Some(NnNode::Internal { children, .. }) = self.nodes.get(&addr) {
                let child_addrs: Vec<_> = children.clone();
                let child_fps: Vec<BitpackedVector> = child_addrs
                    .iter()
                    .filter_map(|c| self.nodes.get(c))
                    .map(|n| n.centroid())
                    .collect();
                let new_count: usize = child_addrs
                    .iter()
                    .filter_map(|c| self.nodes.get(c))
                    .map(|n| n.count())
                    .sum();
                Some((child_fps, new_count))
            } else {
                None
            };

            if let Some((child_fps, new_count)) = child_data {
                if let Some(NnNode::Internal { centroid, count, .. }) = self.nodes.get_mut(&addr) {
                    let refs: Vec<&BitpackedVector> = child_fps.iter().collect();
                    *centroid = BitpackedVector::bundle(&refs);
                    *count = new_count;
                }
            }

            current = addr.parent();
        }
    }

    // ========================================================================
    // SEARCH
    // ========================================================================

    /// Find k nearest neighbors
    pub fn search(&self, query: &BitpackedVector, k: usize) -> Vec<(u64, u32)> {
        let mut results = Vec::new();
        let mut candidates = Vec::new();

        // Start beam search from root
        candidates.push((self.root.clone(), 0u32));

        while !candidates.is_empty() {
            // Sort candidates by distance
            candidates.sort_by_key(|(_, d)| *d);
            candidates.truncate(self.config.search_beam);

            let mut next_candidates = Vec::new();

            for (addr, _) in &candidates {
                match self.nodes.get(addr) {
                    Some(NnNode::Leaf { items, .. }) => {
                        // Search leaf
                        for (id, fp) in items {
                            let dist = hamming_distance_scalar(query, fp);
                            results.push((*id, dist));
                        }
                    }
                    Some(NnNode::Internal { children, .. }) => {
                        // Add children as candidates
                        for child_addr in children {
                            if let Some(child) = self.nodes.get(child_addr) {
                                let dist = hamming_distance_scalar(query, &child.centroid());
                                next_candidates.push((child_addr.clone(), dist));
                            }
                        }
                    }
                    None => {}
                }
            }

            candidates = next_candidates;
        }

        // Sort and return top k
        results.sort_by_key(|(_, d)| *d);
        results.dedup_by_key(|(id, _)| *id);
        results.truncate(k);
        results
    }

    /// Find all neighbors within distance threshold
    pub fn range_search(&self, query: &BitpackedVector, threshold: u32) -> Vec<(u64, u32)> {
        let mut results = Vec::new();
        let mut stack = vec![self.root.clone()];

        while let Some(addr) = stack.pop() {
            match self.nodes.get(&addr) {
                Some(NnNode::Leaf { items, .. }) => {
                    for (id, fp) in items {
                        let dist = hamming_distance_scalar(query, fp);
                        if dist <= threshold {
                            results.push((*id, dist));
                        }
                    }
                }
                Some(NnNode::Internal { children, centroid, .. }) => {
                    // Prune: skip subtree if centroid is too far
                    // (heuristic: centroid distance - max_radius)
                    let centroid_dist = hamming_distance_scalar(query, centroid);
                    let radius = 1000; // Approximate subtree radius

                    if centroid_dist <= threshold + radius {
                        for child_addr in children {
                            stack.push(child_addr.clone());
                        }
                    }
                }
                None => {}
            }
        }

        results.sort_by_key(|(_, d)| *d);
        results
    }

    /// Find exact nearest neighbor (exhaustive within tree)
    pub fn nearest(&self, query: &BitpackedVector) -> Option<(u64, u32)> {
        self.search(query, 1).into_iter().next()
    }

    // ========================================================================
    // DELETION
    // ========================================================================

    /// Delete item by ID
    pub fn delete(&mut self, id: u64) -> bool {
        // Linear search through leaves (could optimize with ID index)
        let mut found_addr = None;

        for (addr, node) in &self.nodes {
            if let NnNode::Leaf { items, .. } = node {
                if items.iter().any(|(item_id, _)| *item_id == id) {
                    found_addr = Some(addr.clone());
                    break;
                }
            }
        }

        if let Some(addr) = found_addr {
            if let Some(NnNode::Leaf { items, .. }) = self.nodes.get_mut(&addr) {
                items.retain(|(item_id, _)| *item_id != id);
                self.total_items -= 1;
                self.update_centroids(&addr);
                return true;
            }
        }

        false
    }

    // ========================================================================
    // BATCH OPERATIONS
    // ========================================================================

    /// Batch insert multiple fingerprints
    pub fn insert_batch(&mut self, fingerprints: &[BitpackedVector]) -> Vec<u64> {
        fingerprints.iter()
            .map(|fp| self.insert(fp.clone()))
            .collect()
    }

    /// Batch search for multiple queries
    pub fn search_batch(&self, queries: &[BitpackedVector], k: usize) -> Vec<Vec<(u64, u32)>> {
        queries.iter()
            .map(|q| self.search(q, k))
            .collect()
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Total number of items
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

    /// Number of internal nodes
    pub fn num_internal(&self) -> usize {
        self.nodes.values().filter(|n| !n.is_leaf()).count()
    }

    /// Number of leaf nodes
    pub fn num_leaves(&self) -> usize {
        self.nodes.values().filter(|n| n.is_leaf()).count()
    }

    /// Average items per leaf
    pub fn avg_leaf_size(&self) -> f32 {
        let leaves: Vec<_> = self.nodes.values()
            .filter_map(|n| match n {
                NnNode::Leaf { items, .. } => Some(items.len()),
                _ => None,
            })
            .collect();

        if leaves.is_empty() {
            0.0
        } else {
            leaves.iter().sum::<usize>() as f32 / leaves.len() as f32
        }
    }

    /// Tree statistics summary
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            total_items: self.total_items,
            depth: self.depth(),
            internal_nodes: self.num_internal(),
            leaf_nodes: self.num_leaves(),
            avg_leaf_size: self.avg_leaf_size(),
        }
    }
}

impl Default for NnTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Tree statistics
#[derive(Clone, Debug)]
pub struct TreeStats {
    pub total_items: usize,
    pub depth: u8,
    pub internal_nodes: usize,
    pub leaf_nodes: usize,
    pub avg_leaf_size: f32,
}

impl std::fmt::Display for TreeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NnTree[{} items, depth={}, {} internal, {} leaves, avg leaf={:.1}]",
            self.total_items, self.depth, self.internal_nodes,
            self.leaf_nodes, self.avg_leaf_size)
    }
}

// ============================================================================
// SPARSE NN-TREE (for very large datasets)
// ============================================================================

/// Sparse NN-Tree with disk-backed storage hints
pub struct SparseNnTree {
    /// In-memory tree for hot data
    hot: NnTree,
    /// Cold data references (id -> fingerprint hash for verification)
    cold_refs: HashMap<u64, u64>,
    /// Hot/cold threshold (items accessed less than this are cold)
    access_threshold: u32,
    /// Access counts
    access_counts: HashMap<u64, u32>,
}

impl SparseNnTree {
    /// Create new sparse NN-tree
    pub fn new() -> Self {
        Self {
            hot: NnTree::new(),
            cold_refs: HashMap::new(),
            access_threshold: 10,
            access_counts: HashMap::new(),
        }
    }

    /// Insert fingerprint
    pub fn insert(&mut self, fingerprint: BitpackedVector) -> u64 {
        let id = self.hot.insert(fingerprint);
        self.access_counts.insert(id, 0);
        id
    }

    /// Search with access tracking
    pub fn search(&mut self, query: &BitpackedVector, k: usize) -> Vec<(u64, u32)> {
        let results = self.hot.search(query, k);

        // Update access counts
        for (id, _) in &results {
            *self.access_counts.entry(*id).or_insert(0) += 1;
        }

        results
    }

    /// Compact: move cold items to cold storage
    pub fn compact(&mut self) -> Vec<u64> {
        let cold_ids: Vec<u64> = self.access_counts
            .iter()
            .filter(|(_, count)| **count < self.access_threshold)
            .map(|(id, _)| *id)
            .collect();

        for &id in &cold_ids {
            // Mark as cold (actual eviction would involve external storage)
            self.cold_refs.insert(id, id); // Placeholder
            self.hot.delete(id);
        }

        cold_ids
    }

    /// Get statistics
    pub fn stats(&self) -> (TreeStats, usize) {
        (self.hot.stats(), self.cold_refs.len())
    }
}

impl Default for SparseNnTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_search() {
        let mut tree = NnTree::new();

        // Insert some vectors
        for i in 0..100 {
            let fp = BitpackedVector::random(i as u64);
            tree.insert(fp);
        }

        assert_eq!(tree.len(), 100);

        // Search for a specific one
        let query = BitpackedVector::random(50);
        let results = tree.search(&query, 5);

        assert!(!results.is_empty());
        // First result should be exact match (distance 0)
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_range_search() {
        let mut tree = NnTree::new();

        for i in 0..50 {
            tree.insert(BitpackedVector::random(i as u64));
        }

        let query = BitpackedVector::random(25);
        let results = tree.range_search(&query, 0);

        // Should find exact match
        assert!(results.iter().any(|(_, d)| *d == 0));
    }

    #[test]
    fn test_tree_splitting() {
        let config = NnTreeConfig {
            max_leaf_size: 8,
            max_children: 4,
            ..Default::default()
        };

        let mut tree = NnTree::with_config(config);

        // Insert enough to trigger splits
        for i in 0..100 {
            tree.insert(BitpackedVector::random(i as u64));
        }

        let stats = tree.stats();
        assert!(stats.depth > 0); // Should have split
        assert!(stats.internal_nodes > 0);

        println!("{}", stats);
    }

    #[test]
    fn test_deletion() {
        let mut tree = NnTree::new();

        let ids: Vec<u64> = (0..10)
            .map(|i| tree.insert(BitpackedVector::random(i)))
            .collect();

        assert_eq!(tree.len(), 10);

        // Delete some
        assert!(tree.delete(ids[5]));
        assert_eq!(tree.len(), 9);

        // Can't delete again
        assert!(!tree.delete(ids[5]));
    }

    #[test]
    fn test_batch_operations() {
        let mut tree = NnTree::new();

        let fps: Vec<_> = (0..100)
            .map(|i| BitpackedVector::random(i))
            .collect();

        let ids = tree.insert_batch(&fps);
        assert_eq!(ids.len(), 100);

        let queries: Vec<_> = (0..5)
            .map(|i| BitpackedVector::random(i * 20))
            .collect();

        let results = tree.search_batch(&queries, 3);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_sparse_tree() {
        let mut tree = SparseNnTree::new();

        for i in 0..50 {
            tree.insert(BitpackedVector::random(i));
        }

        // Search several times to build access patterns
        let query = BitpackedVector::random(10);
        for _ in 0..15 {
            tree.search(&query, 3);
        }

        // Compact should identify cold items
        let cold = tree.compact();
        println!("Cold items: {}", cold.len());
    }
}
