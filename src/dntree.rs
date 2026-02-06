//! DN Tree - Distinguished Name Tree Addressing
//!
//! 256-way hierarchical navigation like LDAP Distinguished Names,
//! integrated with HDR fingerprints and GraphBLAS sparse operations.
//!
//! # Architecture
//!
//! ```text
//! TreeAddr: [depth][b0][b1][b2]...[bn]
//!           └─────┬──────────────────┘
//!           max 255 levels × 256 branches = massive address space
//!
//! Example: /concepts/animals/mammals/cat
//!          TreeAddr([4, 0x01, 0x10, 0x15, hash("cat")])
//! ```
//!
//! The DN Tree provides O(log n) navigation with fingerprint-based
//! similarity search at leaf nodes.

use crate::bitpack::BitpackedVector;
use crate::hamming::hamming_distance_scalar;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ============================================================================
// TREE ADDRESS
// ============================================================================

/// Maximum tree depth (255 levels)
pub const MAX_DEPTH: usize = 255;

/// Tree Address - 256-way hierarchical path
///
/// Format: [depth, b0, b1, ..., bn] where each b_i is 0-255
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TreeAddr {
    /// Path bytes: first byte is depth, rest are branch indices
    path: Vec<u8>,
}

impl TreeAddr {
    /// Create root address
    pub fn root() -> Self {
        Self { path: vec![0] }
    }

    /// Create from path components
    pub fn from_path(components: &[u8]) -> Self {
        let depth = components.len().min(MAX_DEPTH) as u8;
        let mut path = vec![depth];
        path.extend_from_slice(&components[..depth as usize]);
        Self { path }
    }

    /// Create from string path (like "/concepts/animals/cat")
    pub fn from_string(s: &str) -> Self {
        let components: Vec<u8> = s
            .split('/')
            .filter(|c| !c.is_empty())
            .map(|c| Self::hash_component(c))
            .collect();
        Self::from_path(&components)
    }

    /// Hash a string component to u8
    fn hash_component(s: &str) -> u8 {
        let mut hash = 0u64;
        for (i, b) in s.bytes().enumerate() {
            hash = hash.wrapping_add((b as u64).wrapping_mul(31u64.pow(i as u32)));
        }
        (hash % 256) as u8
    }

    /// Get tree depth
    pub fn depth(&self) -> u8 {
        self.path.get(0).copied().unwrap_or(0)
    }

    /// Get branch at level (0-indexed from root)
    pub fn branch(&self, level: usize) -> Option<u8> {
        if level < self.depth() as usize {
            self.path.get(level + 1).copied()
        } else {
            None
        }
    }

    /// Get all branches as slice
    pub fn branches(&self) -> &[u8] {
        if self.path.len() > 1 {
            &self.path[1..]
        } else {
            &[]
        }
    }

    /// Navigate to child branch
    pub fn child(&self, branch: u8) -> Self {
        if self.depth() >= MAX_DEPTH as u8 {
            return self.clone(); // Max depth reached
        }
        let mut new_path = self.path.clone();
        new_path[0] += 1; // Increment depth
        new_path.push(branch);
        Self { path: new_path }
    }

    /// Navigate to parent
    pub fn parent(&self) -> Option<Self> {
        if self.depth() == 0 {
            return None;
        }
        let mut new_path = self.path.clone();
        new_path[0] -= 1;
        new_path.pop();
        Some(Self { path: new_path })
    }

    /// Get ancestor at specific level
    pub fn ancestor(&self, level: u8) -> Self {
        if level >= self.depth() {
            return self.clone();
        }
        let mut new_path = vec![level];
        new_path.extend_from_slice(&self.path[1..=level as usize]);
        Self { path: new_path }
    }

    /// Check if this is ancestor of other
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        if self.depth() >= other.depth() {
            return false;
        }
        self.branches() == &other.branches()[..self.depth() as usize]
    }

    /// Find common ancestor with another address
    pub fn common_ancestor(&self, other: &Self) -> Self {
        let min_depth = self.depth().min(other.depth()) as usize;
        let mut common_depth = 0;

        for i in 0..min_depth {
            if self.path[i + 1] == other.path[i + 1] {
                common_depth = i + 1;
            } else {
                break;
            }
        }

        self.ancestor(common_depth as u8)
    }

    /// Convert to fingerprint (deterministic mapping)
    pub fn to_fingerprint(&self) -> BitpackedVector {
        // Use path bytes as seed for deterministic fingerprint
        let mut seed = 0u64;
        for (i, &b) in self.path.iter().enumerate() {
            seed = seed.wrapping_mul(256).wrapping_add(b as u64);
            seed = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i as u64);
        }
        BitpackedVector::random(seed)
    }

    /// Encode to u64 (for shallow trees, depth ≤ 7)
    pub fn to_u64(&self) -> Option<u64> {
        if self.depth() > 7 {
            return None;
        }
        let mut val = 0u64;
        for &b in &self.path {
            val = (val << 8) | (b as u64);
        }
        Some(val)
    }

    /// Decode from u64
    pub fn from_u64(val: u64) -> Self {
        let depth = (val >> 56) as u8;
        let mut path = vec![depth];
        for i in (0..depth).rev() {
            path.push(((val >> (i * 8)) & 0xFF) as u8);
        }
        Self { path }
    }

    /// Distance between addresses (tree distance)
    pub fn distance(&self, other: &Self) -> u32 {
        let common = self.common_ancestor(other);
        let up = self.depth() - common.depth();
        let down = other.depth() - common.depth();
        (up + down) as u32
    }
}

impl std::fmt::Display for TreeAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "/")?;
        for (i, &b) in self.branches().iter().enumerate() {
            if i > 0 {
                write!(f, "/")?;
            }
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

// ============================================================================
// WELL-KNOWN BRANCHES (Like LDAP OUs)
// ============================================================================

/// Well-known tree branches (namespace constants)
pub mod WellKnown {
    /// Root namespaces (0x00-0x0F)
    pub const CONCEPTS: u8 = 0x01;
    pub const ENTITIES: u8 = 0x02;
    pub const EVENTS: u8 = 0x03;
    pub const RELATIONS: u8 = 0x04;
    pub const TEMPLATES: u8 = 0x05;
    pub const MEMORIES: u8 = 0x06;
    pub const GOALS: u8 = 0x07;
    pub const ACTIONS: u8 = 0x08;

    /// NSM Primes (0x10-0x4F) - Natural Semantic Metalanguage
    pub const I: u8 = 0x10;
    pub const YOU: u8 = 0x11;
    pub const SOMEONE: u8 = 0x12;
    pub const SOMETHING: u8 = 0x13;
    pub const PEOPLE: u8 = 0x14;
    pub const BODY: u8 = 0x15;
    pub const KIND: u8 = 0x16;
    pub const PART: u8 = 0x17;
    pub const THIS: u8 = 0x18;
    pub const THE_SAME: u8 = 0x19;
    pub const OTHER: u8 = 0x1A;
    pub const ONE: u8 = 0x1B;
    pub const TWO: u8 = 0x1C;
    pub const SOME: u8 = 0x1D;
    pub const ALL: u8 = 0x1E;
    pub const MUCH: u8 = 0x1F;
    pub const LITTLE: u8 = 0x20;
    pub const GOOD: u8 = 0x21;
    pub const BAD: u8 = 0x22;
    pub const BIG: u8 = 0x23;
    pub const SMALL: u8 = 0x24;

    /// Cognitive frameworks (0x80-0x8F)
    pub const NARS: u8 = 0x80;
    pub const ACT_R: u8 = 0x81;
    pub const REINFORCEMENT: u8 = 0x82;
    pub const CAUSALITY: u8 = 0x83;
    pub const COUNTERFACTUAL: u8 = 0x84;
    pub const ABDUCTION: u8 = 0x85;

    /// User-defined (0xF0-0xFF)
    pub const USER_0: u8 = 0xF0;
    pub const USER_1: u8 = 0xF1;
    pub const USER_2: u8 = 0xF2;
    pub const USER_3: u8 = 0xF3;
}

// ============================================================================
// 144 COGNITIVE VERBS (Go Board Topology)
// ============================================================================

/// Verb category (6 categories × 24 verbs = 144 total)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VerbCategory {
    /// Structural: IS_A, PART_OF, CONTAINS, etc. (0-23)
    Structural = 0,
    /// Causal: CAUSES, ENABLES, PREVENTS, etc. (24-47)
    Causal = 1,
    /// Temporal: BEFORE, DURING, AFTER, etc. (48-71)
    Temporal = 2,
    /// Epistemic: KNOWS, BELIEVES, INFERS, etc. (72-95)
    Epistemic = 3,
    /// Agentive: DOES, CHOOSES, INTENDS, etc. (96-119)
    Agentive = 4,
    /// Experiential: SEES, FEELS, ENJOYS, etc. (120-143)
    Experiential = 5,
}

impl VerbCategory {
    pub fn from_verb(verb: u8) -> Self {
        match verb / 24 {
            0 => VerbCategory::Structural,
            1 => VerbCategory::Causal,
            2 => VerbCategory::Temporal,
            3 => VerbCategory::Epistemic,
            4 => VerbCategory::Agentive,
            _ => VerbCategory::Experiential,
        }
    }
}

/// Cognitive verb (0-143)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CogVerb(pub u8);

impl CogVerb {
    // Structural verbs (0-23)
    pub const IS_A: Self = Self(0);
    pub const PART_OF: Self = Self(1);
    pub const CONTAINS: Self = Self(2);
    pub const HAS_PROPERTY: Self = Self(3);
    pub const INSTANCE_OF: Self = Self(4);
    pub const SUBCLASS_OF: Self = Self(5);
    pub const SIMILAR_TO: Self = Self(6);
    pub const OPPOSITE_OF: Self = Self(7);
    pub const DERIVED_FROM: Self = Self(8);
    pub const COMPOSED_OF: Self = Self(9);
    pub const MEMBER_OF: Self = Self(10);
    pub const LOCATED_IN: Self = Self(11);
    pub const ADJACENT_TO: Self = Self(12);
    pub const CONNECTED_TO: Self = Self(13);
    pub const OVERLAPS: Self = Self(14);
    pub const DISJOINT: Self = Self(15);
    pub const EXEMPLAR_OF: Self = Self(16);
    pub const PROTOTYPE: Self = Self(17);
    pub const BOUNDARY_OF: Self = Self(18);
    pub const INTERIOR_OF: Self = Self(19);
    pub const EXTERIOR_OF: Self = Self(20);
    pub const SURROUNDS: Self = Self(21);
    pub const INTERSECTS: Self = Self(22);
    pub const DEFINES: Self = Self(23);

    // Causal verbs (24-47)
    pub const CAUSES: Self = Self(24);
    pub const ENABLES: Self = Self(25);
    pub const PREVENTS: Self = Self(26);
    pub const TRANSFORMS: Self = Self(27);
    pub const TRIGGERS: Self = Self(28);
    pub const INHIBITS: Self = Self(29);
    pub const CATALYZES: Self = Self(30);
    pub const REQUIRES: Self = Self(31);
    pub const PRODUCES: Self = Self(32);
    pub const CONSUMES: Self = Self(33);
    pub const MAINTAINS: Self = Self(34);
    pub const DESTROYS: Self = Self(35);
    pub const CREATES: Self = Self(36);
    pub const MODIFIES: Self = Self(37);
    pub const AMPLIFIES: Self = Self(38);
    pub const ATTENUATES: Self = Self(39);
    pub const REGULATES: Self = Self(40);
    pub const COMPENSATES: Self = Self(41);
    pub const BLOCKS: Self = Self(42);
    pub const UNBLOCKS: Self = Self(43);
    pub const INITIATES: Self = Self(44);
    pub const TERMINATES: Self = Self(45);
    pub const SUSTAINS: Self = Self(46);
    pub const DISRUPTS: Self = Self(47);

    // Temporal verbs (48-71) - Allen interval algebra
    pub const BEFORE: Self = Self(48);
    pub const AFTER: Self = Self(49);
    pub const MEETS: Self = Self(50);
    pub const MET_BY: Self = Self(51);
    pub const OVERLAPS_T: Self = Self(52);
    pub const OVERLAPPED_BY: Self = Self(53);
    pub const STARTS: Self = Self(54);
    pub const STARTED_BY: Self = Self(55);
    pub const DURING: Self = Self(56);
    pub const CONTAINS_T: Self = Self(57);
    pub const FINISHES: Self = Self(58);
    pub const FINISHED_BY: Self = Self(59);
    pub const EQUALS_T: Self = Self(60);
    pub const PRECEDES: Self = Self(61);
    pub const SUCCEEDS: Self = Self(62);
    pub const CONCURRENT: Self = Self(63);
    pub const GRADUAL: Self = Self(64);
    pub const SUDDEN: Self = Self(65);
    pub const PERIODIC: Self = Self(66);
    pub const CONTINUOUS: Self = Self(67);
    pub const INTERMITTENT: Self = Self(68);
    pub const ACCELERATES: Self = Self(69);
    pub const DECELERATES: Self = Self(70);
    pub const REVERSES: Self = Self(71);

    // Epistemic verbs (72-95)
    pub const KNOWS: Self = Self(72);
    pub const BELIEVES: Self = Self(73);
    pub const INFERS: Self = Self(74);
    pub const LEARNS: Self = Self(75);
    pub const FORGETS: Self = Self(76);
    pub const REMEMBERS: Self = Self(77);
    pub const DOUBTS: Self = Self(78);
    pub const CONFIRMS: Self = Self(79);
    pub const REFUTES: Self = Self(80);
    pub const HYPOTHESIZES: Self = Self(81);
    pub const DEDUCES: Self = Self(82);
    pub const INDUCES: Self = Self(83);
    pub const ABDUCES: Self = Self(84);
    pub const ASSUMES: Self = Self(85);
    pub const QUESTIONS: Self = Self(86);
    pub const ANSWERS: Self = Self(87);
    pub const EXPLAINS: Self = Self(88);
    pub const PREDICTS: Self = Self(89);
    pub const EXPECTS: Self = Self(90);
    pub const SURPRISES: Self = Self(91);
    pub const UNDERSTANDS: Self = Self(92);
    pub const MISUNDERSTANDS: Self = Self(93);
    pub const RECOGNIZES: Self = Self(94);
    pub const IDENTIFIES: Self = Self(95);

    // Agentive verbs (96-119)
    pub const DOES: Self = Self(96);
    pub const INTENDS: Self = Self(97);
    pub const CHOOSES: Self = Self(98);
    pub const DECIDES: Self = Self(99);
    pub const PLANS: Self = Self(100);
    pub const EXECUTES: Self = Self(101);
    pub const ATTEMPTS: Self = Self(102);
    pub const SUCCEEDS_AT: Self = Self(103);
    pub const FAILS: Self = Self(104);
    pub const COOPERATES: Self = Self(105);
    pub const COMPETES: Self = Self(106);
    pub const NEGOTIATES: Self = Self(107);
    pub const COMMANDS: Self = Self(108);
    pub const OBEYS: Self = Self(109);
    pub const RESISTS: Self = Self(110);
    pub const PERMITS: Self = Self(111);
    pub const FORBIDS: Self = Self(112);
    pub const REQUESTS: Self = Self(113);
    pub const OFFERS: Self = Self(114);
    pub const ACCEPTS: Self = Self(115);
    pub const REJECTS: Self = Self(116);
    pub const PROMISES: Self = Self(117);
    pub const THREATENS: Self = Self(118);
    pub const WARNS: Self = Self(119);

    // Experiential verbs (120-143)
    pub const SEES: Self = Self(120);
    pub const HEARS: Self = Self(121);
    pub const TOUCHES: Self = Self(122);
    pub const TASTES: Self = Self(123);
    pub const SMELLS: Self = Self(124);
    pub const FEELS: Self = Self(125);
    pub const ENJOYS: Self = Self(126);
    pub const DISLIKES: Self = Self(127);
    pub const FEARS: Self = Self(128);
    pub const HOPES: Self = Self(129);
    pub const LOVES: Self = Self(130);
    pub const HATES: Self = Self(131);
    pub const DESIRES: Self = Self(132);
    pub const AVOIDS: Self = Self(133);
    pub const APPROACHES: Self = Self(134);
    pub const WITHDRAWS: Self = Self(135);
    pub const ATTENDS: Self = Self(136);
    pub const IGNORES: Self = Self(137);
    pub const FOCUSES: Self = Self(138);
    pub const DISTRACTS: Self = Self(139);
    pub const IMAGINES: Self = Self(140);
    pub const DREAMS: Self = Self(141);
    pub const PERCEIVES: Self = Self(142);
    pub const SENSES: Self = Self(143);

    /// Get category
    pub fn category(&self) -> VerbCategory {
        VerbCategory::from_verb(self.0)
    }

    /// Get verb fingerprint (deterministic)
    pub fn to_fingerprint(&self) -> BitpackedVector {
        // Each verb gets a unique, reproducible fingerprint
        let seed = 0xBE4B5EED00000000 + self.0 as u64;
        BitpackedVector::random(seed)
    }

    /// Create verb from index
    pub fn from_index(idx: u8) -> Self {
        Self(idx % 144)
    }

    /// Get verb name
    pub fn name(&self) -> &'static str {
        match self.0 {
            0 => "IS_A", 1 => "PART_OF", 2 => "CONTAINS", 3 => "HAS_PROPERTY",
            4 => "INSTANCE_OF", 5 => "SUBCLASS_OF", 6 => "SIMILAR_TO", 7 => "OPPOSITE_OF",
            24 => "CAUSES", 25 => "ENABLES", 26 => "PREVENTS", 27 => "TRANSFORMS",
            48 => "BEFORE", 49 => "AFTER", 56 => "DURING",
            72 => "KNOWS", 73 => "BELIEVES", 74 => "INFERS",
            96 => "DOES", 97 => "INTENDS", 98 => "CHOOSES",
            120 => "SEES", 125 => "FEELS", 126 => "ENJOYS",
            _ => "VERB",
        }
    }
}

// ============================================================================
// DN TREE NODE
// ============================================================================

/// Node in the DN Tree
#[derive(Clone, Debug)]
pub struct DnNode {
    /// Tree address
    pub addr: TreeAddr,
    /// Node fingerprint
    pub fingerprint: BitpackedVector,
    /// Optional name
    pub name: Option<String>,
    /// Abstraction level (0 = concrete, higher = more abstract)
    pub rung: u8,
    /// Activation level (for spreading activation)
    pub activation: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl DnNode {
    /// Create new node
    pub fn new(addr: TreeAddr) -> Self {
        let fingerprint = addr.to_fingerprint();
        Self {
            addr,
            fingerprint,
            name: None,
            rung: 0,
            activation: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Create with name
    pub fn with_name(addr: TreeAddr, name: impl Into<String>) -> Self {
        let mut node = Self::new(addr);
        node.name = Some(name.into());
        node
    }

    /// Create with fingerprint
    pub fn with_fingerprint(addr: TreeAddr, fingerprint: BitpackedVector) -> Self {
        Self {
            addr,
            fingerprint,
            name: None,
            rung: 0,
            activation: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Get unique ID (based on address)
    pub fn id(&self) -> u64 {
        self.addr.to_u64().unwrap_or_else(|| {
            // Hash for deep addresses
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.addr.hash(&mut h);
            h.finish()
        })
    }
}

// ============================================================================
// DN TREE EDGE
// ============================================================================

/// Edge in the DN Tree (bound representation)
#[derive(Clone, Debug)]
pub struct DnEdge {
    /// Source node address
    pub from: TreeAddr,
    /// Target node address
    pub to: TreeAddr,
    /// Relationship verb
    pub verb: CogVerb,
    /// Edge fingerprint: from ⊗ verb ⊗ to
    pub fingerprint: BitpackedVector,
    /// Edge weight
    pub weight: f32,
}

impl DnEdge {
    /// Create edge with automatic fingerprint binding
    pub fn new(from: TreeAddr, verb: CogVerb, to: TreeAddr) -> Self {
        let from_fp = from.to_fingerprint();
        let verb_fp = verb.to_fingerprint();
        let to_fp = to.to_fingerprint();

        // Bind: from ⊗ verb ⊗ to
        let fingerprint = from_fp.xor(&verb_fp).xor(&to_fp);

        Self {
            from,
            to,
            verb,
            fingerprint,
            weight: 1.0,
        }
    }

    /// Create with weight
    pub fn with_weight(from: TreeAddr, verb: CogVerb, to: TreeAddr, weight: f32) -> Self {
        let mut edge = Self::new(from, verb, to);
        edge.weight = weight;
        edge
    }

    /// Recover 'to' from edge, verb, and from
    pub fn recover_to(edge_fp: &BitpackedVector, from: &TreeAddr, verb: &CogVerb) -> BitpackedVector {
        // to = edge ⊗ from ⊗ verb (XOR is self-inverse)
        let from_fp = from.to_fingerprint();
        let verb_fp = verb.to_fingerprint();
        edge_fp.xor(&from_fp).xor(&verb_fp)
    }

    /// Recover 'from' from edge, verb, and to
    pub fn recover_from(edge_fp: &BitpackedVector, verb: &CogVerb, to: &TreeAddr) -> BitpackedVector {
        let verb_fp = verb.to_fingerprint();
        let to_fp = to.to_fingerprint();
        edge_fp.xor(&verb_fp).xor(&to_fp)
    }
}

// ============================================================================
// DN TREE (Main Structure)
// ============================================================================

/// Distinguished Name Tree with GraphBLAS-compatible sparse storage
pub struct DnTree {
    /// Nodes indexed by address
    nodes: HashMap<TreeAddr, DnNode>,
    /// Forward adjacency: from -> [(verb, to, weight)]
    forward: HashMap<TreeAddr, Vec<(CogVerb, TreeAddr, f32)>>,
    /// Reverse adjacency: to -> [(verb, from, weight)]
    reverse: HashMap<TreeAddr, Vec<(CogVerb, TreeAddr, f32)>>,
    /// Edge fingerprints for similarity search
    edge_fingerprints: Vec<(BitpackedVector, TreeAddr, CogVerb, TreeAddr)>,
    /// Node index for nearest neighbor search
    node_index: Vec<(BitpackedVector, TreeAddr)>,
}

impl DnTree {
    /// Create empty tree
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            forward: HashMap::new(),
            reverse: HashMap::new(),
            edge_fingerprints: Vec::new(),
            node_index: Vec::new(),
        }
    }

    /// Add node
    pub fn add_node(&mut self, node: DnNode) {
        self.node_index.push((node.fingerprint.clone(), node.addr.clone()));
        self.nodes.insert(node.addr.clone(), node);
    }

    /// Add node from address
    pub fn add_addr(&mut self, addr: TreeAddr) -> &mut DnNode {
        if !self.nodes.contains_key(&addr) {
            let node = DnNode::new(addr.clone());
            self.node_index.push((node.fingerprint.clone(), addr.clone()));
            self.nodes.insert(addr.clone(), node);
        }
        self.nodes.get_mut(&addr).unwrap()
    }

    /// Get node
    pub fn get_node(&self, addr: &TreeAddr) -> Option<&DnNode> {
        self.nodes.get(addr)
    }

    /// Get node mut
    pub fn get_node_mut(&mut self, addr: &TreeAddr) -> Option<&mut DnNode> {
        self.nodes.get_mut(addr)
    }

    /// Add edge
    pub fn add_edge(&mut self, edge: DnEdge) {
        // Ensure nodes exist
        self.add_addr(edge.from.clone());
        self.add_addr(edge.to.clone());

        // Store in adjacency
        self.forward
            .entry(edge.from.clone())
            .or_default()
            .push((edge.verb, edge.to.clone(), edge.weight));

        self.reverse
            .entry(edge.to.clone())
            .or_default()
            .push((edge.verb, edge.from.clone(), edge.weight));

        // Store fingerprint for search
        self.edge_fingerprints.push((
            edge.fingerprint,
            edge.from,
            edge.verb,
            edge.to,
        ));
    }

    /// Connect two addresses with verb
    pub fn connect(&mut self, from: &TreeAddr, verb: CogVerb, to: &TreeAddr) {
        let edge = DnEdge::new(from.clone(), verb, to.clone());
        self.add_edge(edge);
    }

    /// Get outgoing edges from address
    pub fn outgoing(&self, addr: &TreeAddr) -> &[(CogVerb, TreeAddr, f32)] {
        self.forward.get(addr).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get incoming edges to address
    pub fn incoming(&self, addr: &TreeAddr) -> &[(CogVerb, TreeAddr, f32)] {
        self.reverse.get(addr).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get edges filtered by verb
    pub fn edges_by_verb(&self, addr: &TreeAddr, verb: CogVerb) -> Vec<&TreeAddr> {
        self.outgoing(addr)
            .iter()
            .filter(|(v, _, _)| *v == verb)
            .map(|(_, to, _)| to)
            .collect()
    }

    /// Get edges filtered by verb category
    pub fn edges_by_category(&self, addr: &TreeAddr, category: VerbCategory) -> Vec<(&CogVerb, &TreeAddr)> {
        self.outgoing(addr)
            .iter()
            .filter(|(v, _, _)| v.category() == category)
            .map(|(v, to, _)| (v, to))
            .collect()
    }

    // ========================================================================
    // TREE NAVIGATION
    // ========================================================================

    /// Get all children of address
    pub fn children(&self, addr: &TreeAddr) -> Vec<&TreeAddr> {
        self.nodes
            .keys()
            .filter(|a| addr.is_ancestor_of(a) && a.depth() == addr.depth() + 1)
            .collect()
    }

    /// Get all descendants
    pub fn descendants(&self, addr: &TreeAddr) -> Vec<&TreeAddr> {
        self.nodes
            .keys()
            .filter(|a| addr.is_ancestor_of(a))
            .collect()
    }

    /// Get siblings
    pub fn siblings(&self, addr: &TreeAddr) -> Vec<&TreeAddr> {
        if let Some(parent) = addr.parent() {
            self.nodes
                .keys()
                .filter(|a| {
                    *a != addr
                        && a.depth() == addr.depth()
                        && a.parent().as_ref() == Some(&parent)
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Get path from root to address
    pub fn path_to_root(&self, addr: &TreeAddr) -> Vec<TreeAddr> {
        let mut path = vec![addr.clone()];
        let mut current = addr.clone();
        while let Some(parent) = current.parent() {
            path.push(parent.clone());
            current = parent;
        }
        path.reverse();
        path
    }

    // ========================================================================
    // NEAREST NEIGHBOR SEARCH
    // ========================================================================

    /// Find nearest node by fingerprint
    pub fn find_nearest(&self, query: &BitpackedVector) -> Option<(&TreeAddr, u32)> {
        let mut best = None;
        let mut best_dist = u32::MAX;

        for (fp, addr) in &self.node_index {
            let dist = hamming_distance_scalar(query, fp);
            if dist < best_dist {
                best_dist = dist;
                best = Some(addr);
            }
        }

        best.map(|addr| (addr, best_dist))
    }

    /// Find K nearest nodes
    pub fn find_k_nearest(&self, query: &BitpackedVector, k: usize) -> Vec<(&TreeAddr, u32)> {
        let mut results: Vec<_> = self.node_index
            .iter()
            .map(|(fp, addr)| (addr, hamming_distance_scalar(query, fp)))
            .collect();

        results.sort_by_key(|(_, d)| *d);
        results.truncate(k);
        results
    }

    /// Find nodes within distance threshold
    pub fn find_within(&self, query: &BitpackedVector, threshold: u32) -> Vec<(&TreeAddr, u32)> {
        self.node_index
            .iter()
            .filter_map(|(fp, addr)| {
                let dist = hamming_distance_scalar(query, fp);
                if dist <= threshold {
                    Some((addr, dist))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find nearest edge by fingerprint
    pub fn find_nearest_edge(&self, query: &BitpackedVector) -> Option<(&TreeAddr, &CogVerb, &TreeAddr, u32)> {
        let mut best = None;
        let mut best_dist = u32::MAX;

        for (fp, from, verb, to) in &self.edge_fingerprints {
            let dist = hamming_distance_scalar(query, fp);
            if dist < best_dist {
                best_dist = dist;
                best = Some((from, verb, to));
            }
        }

        best.map(|(from, verb, to)| (from, verb, to, best_dist))
    }

    // ========================================================================
    // SPREADING ACTIVATION
    // ========================================================================

    /// Spread activation from source
    pub fn spread_activation(
        &mut self,
        source: &TreeAddr,
        initial: f32,
        decay: f32,
        max_depth: usize,
    ) {
        if let Some(node) = self.nodes.get_mut(source) {
            node.activation = initial;
        }

        let mut frontier = vec![(source.clone(), initial)];
        let mut visited = std::collections::HashSet::new();
        visited.insert(source.clone());

        for _ in 0..max_depth {
            let mut next_frontier = Vec::new();

            for (addr, act) in frontier {
                let next_act = act * decay;
                if next_act < 0.01 {
                    continue;
                }

                for (_, neighbor, weight) in self.outgoing(&addr).to_vec() {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor.clone());
                        let neighbor_act = next_act * weight;
                        if let Some(node) = self.nodes.get_mut(&neighbor) {
                            node.activation = node.activation.max(neighbor_act);
                        }
                        next_frontier.push((neighbor, neighbor_act));
                    }
                }
            }

            frontier = next_frontier;
        }
    }

    /// Get most activated nodes
    pub fn most_activated(&self, k: usize) -> Vec<(&TreeAddr, f32)> {
        let mut activated: Vec<_> = self.nodes
            .iter()
            .filter(|(_, n)| n.activation > 0.0)
            .map(|(addr, n)| (addr, n.activation))
            .collect();

        activated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        activated.truncate(k);
        activated
    }

    /// Reset all activations
    pub fn reset_activation(&mut self) {
        for node in self.nodes.values_mut() {
            node.activation = 0.0;
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
        self.edge_fingerprints.len()
    }

    /// Maximum depth
    pub fn max_depth(&self) -> u8 {
        self.nodes.keys().map(|a| a.depth()).max().unwrap_or(0)
    }

    /// Get all verbs used
    pub fn verb_histogram(&self) -> HashMap<CogVerb, usize> {
        let mut hist = HashMap::new();
        for (verb, _, _) in self.forward.values().flatten() {
            *hist.entry(*verb).or_insert(0) += 1;
        }
        hist
    }
}

impl Default for DnTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_addr() {
        let root = TreeAddr::root();
        assert_eq!(root.depth(), 0);

        let child = root.child(0x01);
        assert_eq!(child.depth(), 1);
        assert_eq!(child.branch(0), Some(0x01));

        let grandchild = child.child(0x10);
        assert_eq!(grandchild.depth(), 2);
        assert!(child.is_ancestor_of(&grandchild));

        let parent = grandchild.parent().unwrap();
        assert_eq!(parent, child);
    }

    #[test]
    fn test_tree_addr_from_string() {
        let addr = TreeAddr::from_string("/concepts/animals/mammals");
        assert_eq!(addr.depth(), 3);

        let addr2 = TreeAddr::from_string("/concepts/animals/birds");
        let common = addr.common_ancestor(&addr2);
        assert_eq!(common.depth(), 2); // /concepts/animals
    }

    #[test]
    fn test_dn_tree() {
        let mut tree = DnTree::new();

        let concepts = TreeAddr::from_string("/concepts");
        let animals = TreeAddr::from_string("/concepts/animals");
        let mammals = TreeAddr::from_string("/concepts/animals/mammals");
        let cat = TreeAddr::from_string("/concepts/animals/mammals/cat");
        let dog = TreeAddr::from_string("/concepts/animals/mammals/dog");

        tree.add_addr(concepts.clone());
        tree.add_addr(animals.clone());
        tree.add_addr(mammals.clone());
        tree.add_addr(cat.clone());
        tree.add_addr(dog.clone());

        tree.connect(&cat, CogVerb::IS_A, &mammals);
        tree.connect(&dog, CogVerb::IS_A, &mammals);
        tree.connect(&mammals, CogVerb::PART_OF, &animals);
        tree.connect(&cat, CogVerb::SIMILAR_TO, &dog);

        assert_eq!(tree.num_nodes(), 5);
        assert_eq!(tree.num_edges(), 4);

        // Test traversal
        let is_a_edges = tree.edges_by_verb(&cat, CogVerb::IS_A);
        assert_eq!(is_a_edges.len(), 1);
        assert_eq!(is_a_edges[0], &mammals);
    }

    #[test]
    fn test_edge_recovery() {
        let from = TreeAddr::from_string("/concepts/cat");
        let to = TreeAddr::from_string("/concepts/mammal");
        let verb = CogVerb::IS_A;

        let edge = DnEdge::new(from.clone(), verb, to.clone());

        // Recover 'to' from edge
        let recovered = DnEdge::recover_to(&edge.fingerprint, &from, &verb);
        let expected = to.to_fingerprint();

        // Should be identical
        assert_eq!(hamming_distance_scalar(&recovered, &expected), 0);
    }

    #[test]
    fn test_nearest_neighbor() {
        let mut tree = DnTree::new();

        for i in 0..100 {
            let addr = TreeAddr::from_path(&[0x01, i as u8]);
            tree.add_addr(addr);
        }

        // Query for specific fingerprint
        let target = TreeAddr::from_path(&[0x01, 50]);
        let query = target.to_fingerprint();

        let (found, dist) = tree.find_nearest(&query).unwrap();
        assert_eq!(found, &target);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_spreading_activation() {
        let mut tree = DnTree::new();

        let a = TreeAddr::from_path(&[1]);
        let b = TreeAddr::from_path(&[2]);
        let c = TreeAddr::from_path(&[3]);

        tree.add_addr(a.clone());
        tree.add_addr(b.clone());
        tree.add_addr(c.clone());

        tree.connect(&a, CogVerb::CAUSES, &b);
        tree.connect(&b, CogVerb::CAUSES, &c);

        tree.spread_activation(&a, 1.0, 0.5, 3);

        let activated = tree.most_activated(3);
        assert!(!activated.is_empty());
        assert_eq!(activated[0].0, &a);
    }
}
