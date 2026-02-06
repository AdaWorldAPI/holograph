# Metadata Architecture: Why Ladybug-RS Needs Properties-in-Fingerprint

> The metadata problem in ladybug-rs is not a missing feature — it's the root
> cause of three cascading failures: the "one value blocks all" storage
> problem, the inability to filter during search, and the 4096 CAM fitting
> confusion. Fixing metadata fixes all three.

---

## The Core Difference

### RedisGraph: Metadata IN the Fingerprint

In the RedisGraph HDR engine, metadata lives inside the 256-word fingerprint
vector as bit-packed fields in designated blocks:

```
Words 0-207   (blocks 0-12):  Semantic content (13,312 bits)
Words 208-211 (block 13):     ANI levels, consciousness tier markers
Word  210:                     NARS truth: frequency(u16) + confidence(u16) + evidence(u32)
Words 212-215:                Qualia (18D quantized) + Sigma/Rung + GEL + Kernel
Words 216-223:                DN tree: parent(u16) + depth(u8) + rung(u8) + flags
Word  223 bits[56-63]:        Schema version byte (v0=legacy, v1=current)
Words 224-231 (block 14-15):  RL Q-values, reward history, action indices
Words 232-243:                Inline edge slots (16-32 sparse edges)
Words 244-247:                Neighbor bloom filter (256-bit, 4 u64s)
Words 248-255:                Graph metrics (degree, PageRank, cluster, centrality)
```

**Key property**: Reading metadata is reading words from the same array.
No extra column. No separate lookup. No deserialization. Just bit shifts
on u64 words that are already in the CPU cache line because you loaded the
fingerprint for distance computation.

### What Ladybug-RS Needs That RedisGraph Doesn't

RedisGraph is a fingerprint engine. Ladybug-rs is a fingerprint engine
PLUS a graph database PLUS GEL (Graph Execution Language) PLUS NARS
PLUS RL PLUS a semantic kernel PLUS 7-layer consciousness PLUS qualia.
The metadata block must carry all of this. The full proposed layout for
ladybug-rs at 256 words:

```
SEMANTIC CONTENT (words 0-207, 13,312 bits)
├── 10K semantic bits from the original fingerprint
├── Remaining 3,312 bits: zero-extended or filled by upscaling membrane
└── Distance computation uses ONLY these words (blocks 0-12)

ANI / CONSCIOUSNESS (words 208-211, 256 bits)
├── Word 208: ANI level(u8) + active layer mask(u8) + peak activation(u16)
│             + L1-L4 condensed confidence (4×u8)
├── Word 209: L5-L7 condensed confidence (3×u8) + cycle(u16)
│             + consciousness flags(u8) + tau(u8, quantized)
├── Word 210: NARS truth: frequency(u16) + confidence(u16)
│             + pos_evidence(u16) + neg_evidence(u16)
└── Word 211: Membrane sigma(u16) + processing mode(u8) + reserved(u8)
              + membrane tau_hash(u32, condensed temporal context)

QUALIA / KERNEL / GEL (words 212-215, 256 bits)
├── Word 212: Qualia 18D → top 4 channels (4×u16): valence, arousal, dominance, novelty
├── Word 213: Qualia next 4 channels (4×u16): certainty, urgency, depth, salience
│             (remaining 10 dimensions stored in overflow or derived)
├── Word 214: GEL state: program_counter(u16) + stack_depth(u8) + exec_flags(u8)
│             + current_verb(u8) + gel_phase(u8) + reserved(u16)
└── Word 215: Semantic kernel: integration_state(u16) + kernel_mode(u8)
              + kernel_epoch(u8) + reserved(u32)

DN TREE STRUCTURE (words 216-223, 512 bits)
├── Word 216: parent_addr(u16) + depth(u8) + rung(u8)
│             + sigma(u8) + node_type(u8) + flags(u16)
├── Word 217: label_hash(u32) + access_count(u16) + ttl_remaining(u16)
├── Word 218: created_timestamp(u32) + last_access_delta(u16) + reserved(u16)
├── Word 219: verb_slots[0-3]: 4 × packed verb(u8)+target_addr(u8)
│             (first 4 edges — immediate children/relations)
├── Word 220: verb_slots[4-7]: next 4 edges
├── Word 221: verb_slots[8-11]: next 4 edges
├── Word 222: verb_slots[12-15]: next 4 edges (16 inline edges total)
└── Word 223: overflow_count(u8) + overflow_ptr(u16) + reserved(29 bits)
              + version_byte(u8) at bits[56-63]

RL / DECISION (words 224-231, 512 bits)
├── Word 224: Q-values for 4 actions (4×u16)
├── Word 225: Q-values for 4 more actions (4×u16)
├── Word 226: Reward history ring (4×u16, last 4 rewards)
├── Word 227: Reward trend(u16) + action_count(u16) + epsilon(u16) + reserved(u16)
├── Word 228: Policy fingerprint condensed hash (u64)
├── Word 229: State-action binding cache (u64)
├── Word 230: TD error accumulator (u32) + discount factor(u16) + alpha(u16)
└── Word 231: RL routing score cache (u32) + reserved(u32)

INLINE EDGE SLOTS (words 232-243, 768 bits = up to 32 edges)
├── Words 232-235: edges 16-19 (4 × packed edge: verb(u8)+addr(u8) = 16 bits each)
│                  4 edges per word × 4 words = 16 more edges
├── Words 236-239: edges 20-27 (another 16 edges packed same way — OPTIONAL)
│                  Only used if node has >20 edges. Otherwise reserved.
├── Words 240-243: edge overflow metadata:
│                  Word 240: inline_edge_count(u8) + overflow_flag(u8)
│                            + overflow_table_addr(u16) + edge_version(u16) + reserved(u16)
│                  Word 241: in_degree(u16) + out_degree(u16) + bidirectional_count(u16) + reserved(u16)
│                  Word 242: reserved for CSR offset pointer
│                  Word 243: reserved for CSR offset pointer
└── OVERFLOW RULE: nodes with >32 edges set overflow_flag=1 and store
    remaining edges in database table (Lance/external). This covers 95%+
    of real graphs where most nodes have <32 edges. Hub nodes overflow.

BLOOM FILTER (words 244-247, 256 bits)
├── 256-bit neighbor bloom filter
├── Hashes of 1-hop neighbor fingerprints
├── Used for bloom_accelerated_search() (neighbor bonus)
└── False positive rate ~1% at 20 neighbors

GRAPH METRICS (words 248-255, 512 bits)
├── Word 248: degree(u16) + in_degree(u16) + out_degree(u16) + reserved(u16)
├── Word 249: PageRank(u32, fixed-point) + HITS authority(u16) + hub(u16)
├── Word 250: cluster_id(u16) + community(u16) + betweenness(u16) + closeness(u16)
├── Word 251: local_clustering_coeff(u16) + triangle_count(u16) + reserved(u32)
├── Word 252: eccentricity(u16) + katz_centrality(u16) + reserved(u32)
├── Word 253: temporal_degree(u16, recent edges) + growth_rate(u16) + reserved(u32)
├── Word 254: reserved for application-specific graph metrics
└── Word 255: checksum(u32) + reserved(u24) + version_flags(u8)
```

### Ladybug-RS: Metadata BESIDE the Fingerprint

In ladybug-rs, metadata lives in native Rust struct fields alongside the
fingerprint array:

```rust
pub struct BindNode {
    pub fingerprint: [u64; 156],     // The vector
    pub label: Option<String>,        // Heap-allocated string
    pub qidx: u8,                     // Qualia index
    pub access_count: u32,            // LRU tracking
    pub payload: Option<Vec<u8>>,     // Heap-allocated blob
    pub parent: Option<Addr>,         // Tree pointer
    pub depth: u8,                    // Tree depth
    pub rung: u8,                     // Access rung
    pub sigma: u8,                    // Reasoning depth
}

pub struct CogValue {
    pub fingerprint: [u64; 156],      // The vector
    pub qualia: QualiaVector,         // Separate struct
    pub truth: TruthValue,            // 2 × f32 (IEEE 754)
    pub access_count: u32,            // LRU tracking
    pub last_access: Instant,         // Timestamp
    pub ttl: Option<Duration>,        // Expiry
    pub created: Instant,             // Timestamp
    pub label: Option<String>,        // Heap-allocated string
}
```

This is correct for a Rust application. It's idiomatic. It's fast for
single-record access. But it creates three problems that compound into
architectural deadlocks.

---

## Problem 1: "One Value Blocks All"

When `write_at()` is called, the ENTIRE BindNode is replaced:

```rust
pub fn write_at(&mut self, addr: Addr, fingerprint: [u64; FINGERPRINT_WORDS]) -> bool {
    let node = BindNode::new(fingerprint);  // Fresh node, all metadata zeroed
    // ... tier checking ...
    c[slot] = Some(node);  // OVERWRITES label, qidx, access_count, parent, depth, rung, sigma
    true
}
```

The fingerprint overwrites everything. If you had a label, it's gone. If you
had a parent pointer, it's gone. If you had access_count=47, it resets to 0.

**This is why CAM operations can't safely write results back to addresses.**
A CAM operation that computes `A ⊕ B` and writes the result to address C
destroys all metadata at C. The "one value blocks all" phenomenon.

### Mitigations in the current codebase:

1. **Bundle instead of replace**: Majority voting preserves some bits, but
   metadata fields (label, parent, depth) are not bit-voteable
2. **Touch for access tracking**: Separate `touch()` method, but it requires
   a read-modify-write cycle with no atomicity guarantee
3. **Layer isolation**: SevenLayerNode keeps markers separate from VSA core,
   but markers don't survive a write_at()

### How RedisGraph solves this:

The `ConcurrentWriteCache` never overwrites the base record. Instead:

1. **XOR delta**: Compute `old ⊕ new = delta`. Store only the delta.
2. **Read-through**: On read, apply `base ⊕ delta = current`. O(1) per word.
3. **Schema blocks preserved**: Delta only touches semantic blocks (0-12).
   Metadata in blocks 13-15 is orthogonal — a semantic update doesn't touch
   ANI, NARS, RL, or bloom metadata unless explicitly requested.
4. **Partial update**: To update ONLY the NARS truth value, write a delta
   that is zero everywhere except word 210. Everything else untouched.

**The key insight**: When metadata is IN the fingerprint, you get partial
updates for free via XOR delta. When metadata is BESIDE the fingerprint in
struct fields, partial updates require field-by-field read-modify-write with
locking.

---

## Problem 2: Search Has No Inline Predicate Filtering

The HDR cascade search in ladybug-rs is pure distance:

```rust
pub fn search(&self, query: &[u64; WORDS], k: usize) -> Vec<(usize, u32)> {
    for (idx, fp) in self.fingerprints.iter().enumerate() {
        // Level 0: 1-bit sketch filter
        // Level 1: 4-bit sketch filter
        // Level 2: 8-bit sketch filter
        // Level 3: exact distance
        let exact = hamming_distance(query, fp);
        candidates.push((idx, exact));
    }
    candidates.sort_by_key(|&(_, d)| d);
    candidates.truncate(k);
    candidates
}
```

There is no way to say "give me the 10 nearest nodes with ANI level ≥ 5
and NARS confidence > 0.7". You get the 10 nearest by raw distance, then
post-filter in application code. This means:

- **Wasted computation**: You compute exact distances for nodes that will be
  filtered out
- **Top-k pollution**: If 8 of the top 10 don't pass your predicates, you
  get 2 useful results instead of 10
- **Two-pass penalty**: Post-filtering requires loading metadata from separate
  struct fields (cache miss) after computing distance (which only touches the
  fingerprint array)

### How RedisGraph solves this:

```rust
pub fn passes_predicates(&self, query: &SchemaQuery) -> bool {
    // Check ANI level — word 208, bits 0-7
    if let Some(min_level) = query.ani_filter.as_ref().map(|a| a.min_level) {
        if self.ani.level < min_level { return false; }
    }
    // Check NARS confidence — word 210, bits 32-47
    if let Some(min_conf) = query.nars_filter.as_ref().map(|n| n.min_confidence) {
        if self.nars.confidence < min_conf { return false; }
    }
    // ... more predicates, all from the same cache line
    true
}
```

Predicates are checked DURING the cascade, between sketch levels. A node
that fails an ANI predicate is eliminated at O(1) cost before the expensive
exact distance computation. The metadata is in the same words as the
fingerprint — no separate struct field access, no cache miss.

---

## Problem 3: NARS Truth Values Are IEEE 754 Floats

Ladybug-rs stores NARS truth as two `f32` values:

```rust
pub struct TruthValue {
    pub frequency: f32,    // 0.0 - 1.0
    pub confidence: f32,   // 0.0 - 1.0
}
```

This is mathematically precise but architecturally expensive:

1. **8 bytes per truth value** (64 bits) vs. RedisGraph's **4 bytes** (32 bits:
   frequency u16 + confidence u16)
2. **Not XOR-composable**: Float XOR is meaningless. You can't delta-compress
   truth values alongside the fingerprint.
3. **Not bit-comparable**: Checking `confidence > 0.7` requires float comparison.
   Checking `confidence_u16 > 45875` is an integer compare — 1 cycle vs 3-5.
4. **Precision overkill**: NARS truth with confidence resolution of 1/65536
   (0.0000153) is more than sufficient. The inference rules (revision,
   deduction, induction) introduce far more noise than quantization.

### RedisGraph's bit-packed NARS truth:

```rust
pub struct NarsTruth {
    pub frequency: u16,     // 0-65535 → 0.0-1.0 (precision: 0.0000153)
    pub confidence: u16,    // 0-65535 → 0.0-1.0
    pub pos_evidence: u16,  // Positive evidence count
    pub neg_evidence: u16,  // Negative evidence count
}
// Total: 8 bytes = 1 u64 word
// Stored at: word 210 of the 256-word fingerprint
```

One u64 word carries frequency, confidence, AND evidence counts. It sits
inline in the fingerprint. It XOR-deltas like any other word. It compares
with integer operations during predicate filtering.

---

## Problem 4: The Seven-Layer Model Has No Fingerprint Representation

The SevenLayerNode is a beautiful model but it's structurally divorced from
the fingerprint:

```rust
pub struct SevenLayerNode {
    pub path: String,
    pub vsa_core: Fingerprint,           // Shared 10K-bit core
    markers: [LayerMarker; 7],           // SEPARATE from fingerprint
}

pub struct LayerMarker {
    pub active: bool,
    pub timestamp: Instant,
    pub value: f32,
    pub confidence: f32,
    pub cycle: u64,
    pub flags: u32,
}
```

Each LayerMarker is 25+ bytes. Seven layers = 175+ bytes of metadata that
cannot be stored in the fingerprint, cannot be filtered during search, and
cannot travel with the fingerprint when it's passed to another function,
serialized to disk, or sent over the network.

### RedisGraph's approach:

The ANI (consciousness tier) metadata is packed into block 13:

```
Word 208, bits 0-7:    ANI level (0-255)
Word 208, bits 8-15:   Active layer mask (7 bits = 7 layers)
Word 208, bits 16-31:  Peak activation (u16, quantized from f32)
Word 209, bits 0-15:   Layer confidence composite (weighted average)
Word 209, bits 16-31:  Processing cycle (truncated to u16)
```

10 bytes capture the essential state of all 7 layers. Not every field —
`timestamp` and `flags` don't travel with the fingerprint. But the fields
that matter for search (level, activation, confidence) do.

---

## The 4096 CAM Clarification: Transport Protocol, Not Storage

The user's critical insight:

> "The commandlets are not a storage issue, they belong into classes and
> methods and the transport has the 4096 in order to reach those methods"

The 4096 CAM is a **transport protocol**. It's an addressing scheme for
reaching operations, like HTTP method + path reaches a REST endpoint.
The CAM opcode dispatches to a class and method. The operation itself lives
in Rust code (classes, methods, trait implementations).

**What should remain in the CAM**: GEL (Graph Execution Language) — the
mechanism for compiling programs into graph execution sequences. GEL is
inherently a transport/dispatch concern: "compile this program, route the
steps to the right operations, execute in order."

**What should NOT be in the CAM**: Individual operation implementations.
NARS inference, RL Q-updates, Hamming distance — these are methods on types.
They belong in `impl TruthValue`, `impl QTable`, `impl Fingerprint16K`.
The CAM routes TO them, it doesn't contain them.

### The current cam_ops.rs problem:

4,661 lines of match arms mixing routing with implementation:

```rust
// This is routing AND implementation mixed together
0x410 => {
    if args.len() < 3 {
        return OpResult::Error("Deduction requires M, P, S".to_string());
    }
    let conclusion = args[2].bind(&args[1]);  // Implementation inline
    OpResult::One(conclusion)
}
```

### The fix:

```rust
// CAM is routing only — reaches the method
0x410 => TruthValue::deduction(&args[0], &args[1], &args[2]),

// Implementation lives in impl TruthValue (separate file)
impl TruthValue {
    pub fn deduction(m: &Fingerprint16K, p: &Fingerprint16K, s: &Fingerprint16K)
        -> CamResult { ... }
}
```

The 4,661 lines shrink to ~200 lines of pure routing. Everything else moves
to where it belongs: `impl` blocks, trait implementations, method bodies.

---

## The Surplus Problem: Ladybug-RS Is More Than a Fingerprint Engine

The 10,000-to-16,384 bit surplus (6,384 bits = ~99 u64 words) seems generous
until you count what ladybug-rs actually needs to store:

| System | What It Needs in the Fingerprint | Words | Bits |
|--------|----------------------------------|-------|------|
| Core VSA | 10,000 semantic content bits | 0-207 | 13,312 |
| ANI/7-Layer | level, mask, activation, confidence, cycle, tau | 208-211 | 256 |
| NARS | frequency, confidence, evidence counts | 210 | 64 |
| Qualia | 18D → top 8 channels quantized to u16 | 212-213 | 128 |
| GEL | program counter, stack depth, exec flags, verb | 214 | 64 |
| Semantic Kernel | integration state, kernel mode, epoch | 215 | 64 |
| DN Tree | parent, depth, rung, sigma, type, flags, label hash | 216-218 | 192 |
| Inline Edges | 16-32 sparse edges (verb+addr packed) | 219-222, 232-239 | 512-1024 |
| Edge Overflow | inline count, overflow flag, degree tracking | 240-243 | 256 |
| Schema version | layout version byte | 223 | 8 |
| RL / Decision | Q-values, rewards, TD error, policy, routing | 224-231 | 512 |
| Bloom | 256-bit neighbor bloom filter | 244-247 | 256 |
| Graph Metrics | degree, PageRank, cluster, centrality, etc. | 248-255 | 512 |
| **Total** | | **256 words** | **16,384** |

**Every word accounted for.** No surplus left unassigned. The point:
ladybug-rs is a fingerprint engine PLUS GEL PLUS semantic kernel PLUS
NARS PLUS 7-layer consciousness PLUS RL PLUS qualia PLUS a graph
database — and the fingerprint must carry metadata for ALL of them.

At 156 words (9,984 bits), there is literally NO room for metadata. The
semantic content alone consumes 100% of the vector. This is why metadata
lives in struct fields today — there's nowhere else to put it.

At 256 words (16,384 bits), there's room for everything. The metadata
moves from struct fields into the vector. And that move enables:

1. **Partial updates** via XOR delta (no more "one value blocks all")
2. **Inline predicate filtering** during search (no more post-filter)
3. **Self-describing vectors** that carry their metadata wherever they go
4. **Integer comparison** for predicates (no more float conversion)
5. **Network-portable records** (one array = complete record, no serialization)
6. **Inline graph traversal** (edges in the fingerprint = no separate edge table lookup)

---

## Problem 5: Nodes and Edges Are Separate From the Fingerprint

Ladybug-rs stores graph structure in parallel data structures:

```rust
// BindSpace holds nodes as Option<BindNode> in chunked arrays
// Edges live in SEPARATE arrays:
pub struct BindSpace {
    edges: Vec<BindEdge>,           // Edge list
    edge_out: Vec<Vec<usize>>,      // from.0 → edge indices
    edge_in: Vec<Vec<usize>>,       // to.0 → edge indices
    csr: Option<BitpackedCsr>,      // Compressed sparse row
}

pub struct BindEdge {
    pub from: Addr,
    pub to: Addr,
    pub verb: Addr,
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    pub weight: f32,
}
```

And the CogGraph duplicates this with HashMaps:

```rust
pub struct CogGraph {
    nodes: HashMap<u64, CogNode>,
    edges: HashMap<u64, CogEdge>,
    adjacency: HashMap<u64, Vec<(Verb, u64)>>,
    reverse_adj: HashMap<u64, Vec<(Verb, u64)>>,
}
```

This means traversing from a node to its neighbors requires:
1. Read the node fingerprint from BindSpace (cache line 1)
2. Look up edges in edge_out or CSR (cache line 2, different array)
3. For each neighbor: read its fingerprint (cache line 3+)

Three separate memory regions per hop. At scale, this is cache-hostile.

### How 256 words with inline edges solves this:

```rust
// Words 219-222: first 16 edges packed inline
// Each edge = verb(u8) + target_addr(u8) = 16 bits
// 4 edges per u64 word × 4 words = 16 edges

fn inline_edges(words: &[u64; 256]) -> impl Iterator<Item = (u8, Addr)> {
    (219..=222).flat_map(move |w| {
        (0..4).map(move |slot| {
            let packed = (words[w] >> (slot * 16)) & 0xFFFF;
            let verb = (packed >> 8) as u8;
            let addr = Addr(packed as u16 & 0xFF);
            (verb, addr)
        })
    })
}
```

Read the node. Its edges are right there in the same 2KB cache block.
No separate lookup. For 95%+ of nodes in real graphs (degree < 16),
the entire adjacency is inline.

Nodes with more edges use words 232-239 for another 16 edges (total 32).
Nodes exceeding 32 edges (hub nodes, <5% of most graphs) set the
overflow flag and store remaining edges in a Lance table. The overflow
table is accessed via DataFusion — no custom code needed.

---

## XOR Parent-Child Compression for Graph Storage

The same XOR delta trick that solves the "one value blocks all" problem
also compresses graph storage. In a DN tree like `Ada:A:soul:identity`:

```
Node "Ada"                 → stored as full 256-word fingerprint (base)
Node "Ada:A"               → stored as Ada ⊕ delta_A (only differences)
Node "Ada:A:soul"          → stored as Ada:A ⊕ delta_soul
Node "Ada:A:soul:identity" → stored as Ada:A:soul ⊕ delta_identity
```

Each child shares most of its semantic content with its parent. The XOR
delta between parent and child is sparse — most words are zero. Storing
only the non-zero delta words compresses the tree dramatically.

### Integration with DataFusion layer:

The XOR write cache in the DataFusion persistence layer can serve double
duty:

1. **Version deltas** (original purpose): `current = base ⊕ delta_v1 ⊕ delta_v2`
2. **Tree deltas** (graph compression): `child = parent ⊕ delta_child`

Both use the same `ConcurrentWriteCache`. Both apply `XorDelta` on read.
The DataFusion TableProvider resolves the chain transparently:

```rust
// TableProvider resolves XOR chains on read
fn read_node(&self, addr: Addr) -> [u64; 256] {
    let base = self.base_store.read(addr);
    // Apply version delta (if dirty in write cache)
    let versioned = self.write_cache.read_through(addr, &base);
    // Apply tree delta (if node is stored as parent⊕delta)
    let resolved = self.tree_cache.resolve_chain(addr, &versioned);
    resolved
}
```

Nodes are pulled into the cache by fingerprint hash. Hot paths through
the tree stay cached. Cold branches stay compressed on disk. The
DataFusion query optimizer can push predicates through the XOR chain
because `(parent ⊕ delta)[word_210]` gives you the child's NARS truth
directly — XOR is word-independent.

**The edge limit (16-32 inline, overflow to table) is the same principle**:
keep the common case fast (inline), handle the long tail via database
tables accessed through DataFusion. The fingerprint is the fast path.
The database is the overflow path. Both use the same query interface.

---

## Migration: What Changes in Ladybug-RS

### Phase A: Define the bit layout

Create a `SchemaSidecar` equivalent that maps all metadata to specific words:

```rust
// src/width_16k/schema.rs (new file, don't modify existing code)

// ANI / Consciousness (block 13)
pub const WORD_ANI_BASE: usize = 208;      // ANI level + layer mask + activation
pub const WORD_ANI_EXT: usize = 209;       // L5-L7 confidence + cycle + tau
pub const WORD_NARS_TRUTH: usize = 210;    // NARS frequency + confidence + evidence
pub const WORD_MEMBRANE: usize = 211;      // Sigma + processing mode + tau hash

// Qualia / Kernel / GEL
pub const WORD_QUALIA_A: usize = 212;      // Qualia channels 0-3 (4×u16)
pub const WORD_QUALIA_B: usize = 213;      // Qualia channels 4-7 (4×u16)
pub const WORD_GEL_STATE: usize = 214;     // GEL execution state
pub const WORD_KERNEL: usize = 215;        // Semantic kernel state

// DN Tree structure
pub const WORD_DN_PARENT: usize = 216;     // parent(u16) + depth(u8) + rung(u8) + sigma(u8) + type(u8) + flags(u16)
pub const WORD_DN_META: usize = 217;       // label_hash(u32) + access_count(u16) + ttl(u16)
pub const WORD_DN_TIME: usize = 218;       // created(u32) + last_access_delta(u16) + reserved(u16)

// Inline edges (16 edges: 4 per word)
pub const WORD_EDGE_INLINE_0: usize = 219; // edges 0-3
pub const WORD_EDGE_INLINE_1: usize = 220; // edges 4-7
pub const WORD_EDGE_INLINE_2: usize = 221; // edges 8-11
pub const WORD_EDGE_INLINE_3: usize = 222; // edges 12-15

// Schema version
pub const WORD_VERSION: usize = 223;       // overflow(u8) + overflow_ptr(u16) + reserved + version(u8)@[56-63]

// RL / Decision (block 14-15)
pub const WORD_RL_BASE: usize = 224;       // Q-values, rewards, TD error

// Extended edge slots (words 232-239: 16 more edges if needed)
pub const WORD_EDGE_EXT_0: usize = 232;    // edges 16-19
// ... through WORD_EDGE_EXT_3 = 235       // edges 28-31
pub const WORD_EDGE_OVERFLOW: usize = 240; // inline_count + overflow_flag + table addr
pub const WORD_EDGE_DEGREE: usize = 241;   // in_degree + out_degree + bidi count

// Bloom filter
pub const WORD_BLOOM_BASE: usize = 244;    // 256-bit neighbor bloom (4 words)

// Graph metrics
pub const WORD_GRAPH_BASE: usize = 248;    // degree, PageRank, cluster, centrality (8 words)
```

### Phase B: Add quantization functions

```rust
// TruthValue → u64 (lossless at u16 precision)
pub fn truth_to_word(tv: &TruthValue) -> u64 {
    let freq = (tv.frequency * 65535.0) as u64;
    let conf = (tv.confidence * 65535.0) as u64;
    let (pos, neg) = tv.to_evidence();
    let pos_u16 = (pos.min(65535.0)) as u64;
    let neg_u16 = (neg.min(65535.0)) as u64;
    freq | (conf << 16) | (pos_u16 << 32) | (neg_u16 << 48)
}

// u64 → TruthValue
pub fn word_to_truth(w: u64) -> TruthValue {
    TruthValue {
        frequency: (w & 0xFFFF) as f32 / 65535.0,
        confidence: ((w >> 16) & 0xFFFF) as f32 / 65535.0,
    }
}

// QualiaField 18D → 2 × u64 (top 8 channels at u16 precision)
pub fn qualia_to_words(q: &QualiaField) -> (u64, u64) {
    let w0 = (q.valence()    * 65535.0) as u64
           | ((q.arousal()   * 65535.0) as u64) << 16
           | ((q.dominance() * 65535.0) as u64) << 32
           | ((q.novelty()   * 65535.0) as u64) << 48;
    let w1 = (q.certainty()  * 65535.0) as u64
           | ((q.urgency()   * 65535.0) as u64) << 16
           | ((q.depth()     * 65535.0) as u64) << 32
           | ((q.salience()  * 65535.0) as u64) << 48;
    (w0, w1)
}

// Inline edge packing: verb(u8) + target_addr(u8) = 16 bits, 4 per word
pub fn pack_edge(verb: u8, target: Addr) -> u16 {
    ((verb as u16) << 8) | (target.0 & 0xFF)
}

pub fn pack_edge_word(edges: &[(u8, Addr); 4]) -> u64 {
    edges.iter().enumerate().fold(0u64, |word, (i, (verb, addr))| {
        word | ((pack_edge(*verb, *addr) as u64) << (i * 16))
    })
}
```

### Phase C: Add inline predicate filtering to HDR cascade

```rust
// Add between Level 2 (8-bit sketch) and Level 3 (exact distance)
if let Some(ref predicates) = query.predicates {
    let schema = SchemaSidecar::read_from_words(fp);
    if !schema.passes_predicates(predicates) {
        continue;  // Skip this candidate, no exact distance needed
    }
}
```

### Phase D: Replace write_at() overwrite with delta recording

```rust
// Instead of: c[slot] = Some(BindNode::new(fingerprint));
// Do:
let old = c[slot].as_ref().map(|n| n.fingerprint).unwrap_or([0u64; 256]);
let delta = xor_delta(&old, &fingerprint);
write_cache.record_delta(addr, delta);
// Metadata blocks in old are preserved because delta is zero there
```

### Phase E: Wire inline edges into graph traversal

```rust
// Replace CSR lookup with inline-first, overflow-second pattern
pub fn neighbors(&self, addr: Addr) -> Vec<(u8, Addr)> {
    let fp = self.read(addr);
    let mut result = Vec::new();

    // 1. Read inline edges (words 219-222, always present)
    let inline_count = (fp[240] & 0xFF) as usize;
    for w in 219..=222 {
        for slot in 0..4 {
            let packed = ((fp[w] >> (slot * 16)) & 0xFFFF) as u16;
            if packed != 0 {
                let verb = (packed >> 8) as u8;
                let target = Addr(packed & 0xFF);
                result.push((verb, target));
            }
        }
        if result.len() >= inline_count { break; }
    }

    // 2. Read extended slots if needed (words 232-239)
    if inline_count > 16 {
        for w in 232..=239 {
            for slot in 0..4 {
                let packed = ((fp[w] >> (slot * 16)) & 0xFFFF) as u16;
                if packed != 0 {
                    let verb = (packed >> 8) as u8;
                    let target = Addr(packed & 0xFF);
                    result.push((verb, target));
                }
            }
            if result.len() >= inline_count { break; }
        }
    }

    // 3. Overflow to database table if flag set
    let overflow_flag = ((fp[240] >> 8) & 0xFF) as u8;
    if overflow_flag != 0 {
        let overflow_edges = self.lance_table.read_edges(addr);
        result.extend(overflow_edges);
    }

    result
}
```

---

## Summary Table: Ladybug-RS vs RedisGraph Metadata

| Aspect | Ladybug-RS (Current) | 256-Word Target |
|--------|---------------------|----------------------|
| **Where metadata lives** | Rust struct fields | Bit-packed in fingerprint words |
| **NARS truth storage** | 2 × f32 (8 bytes, float) | 1 × u64 (8 bytes, integer, XOR-able) |
| **ANI/layer state** | 7 × LayerMarker (~175 bytes) | 4 words (32 bytes, essential state) |
| **Qualia** | QualiaField with 18 × f32 (72 bytes) | 2 words (16 bytes, top 8 channels u16) |
| **DN tree** | Struct fields (parent, depth, rung, sigma) | 3 words (parent + depth + rung + sigma + type + flags + label hash + timestamps) |
| **Graph edges** | Separate Vec + CSR (different memory) | 16-32 inline edge slots + overflow to Lance |
| **Search predicate filtering** | Post-filter only | Inline during cascade |
| **Partial update** | Impossible (full overwrite) | Natural via XOR delta |
| **Graph compression** | None (full copy per node) | XOR parent-child delta chains |
| **Network serialization** | Struct → JSON/protobuf | Copy 256 u64s (zero-ser) |
| **GEL state** | Not in fingerprint | Word 214 (execution markers) |
| **Semantic kernel** | Not in fingerprint | Word 215 (integration state) |
| **RL / Decision** | Not in fingerprint | 8 words (Q-values, rewards, TD error, policy) |
| **Graph metrics** | Not tracked | 8 words (degree, PageRank, cluster, centrality) |
| **Self-describing** | No (needs struct context) | Yes (schema version byte) |
| **CAM interaction** | Operations need struct refs | Operations read/write words directly |
| **Edge traversal** | 3+ cache misses per hop | 1 cache miss (edges inline in same 2KB block) |

---

## The Connection to 4096

The 4096 CAM as **transport** is correct: an opcode dispatches to a method.
The problem arises when that method needs to read or write metadata:

- **Current**: Method receives `[u64; 156]`. Metadata is in struct fields
  the method can't see. To read NARS truth, the method needs a reference
  to the entire BindNode or CogValue. The fingerprint alone is insufficient.

- **At 256 words**: Method receives `[u64; 256]`. NARS truth is at word 210.
  The method reads `words[210]`, applies the inference rule, writes the
  result back to `words[210]`. The fingerprint IS the complete record. The
  transport carries everything.

This is why 4096 works for "one shot one command" transport — the command
arrives, dispatches to a method, and the method operates on the fingerprint.
But it fails for storage when the fingerprint is only 156 words, because
the method can't update metadata that isn't IN the fingerprint.

At 256 words, transport and storage align: the CAM command dispatches to
a method, the method operates on the 256-word fingerprint (which contains
all metadata), and the result can be stored (with or without delta
compression) without losing anything.

**The 4096 CAM is the transport. The 256-word fingerprint is the storage.
GEL compiles programs into sequences of CAM-dispatched operations on
256-word fingerprints. Everything fits when the fingerprint is the record.**

---

## Alternative Design: Vertical + Horizontal with XOR Coupling

There's a more aggressive approach worth considering. Instead of packing
all metadata into the 256-word fingerprint (vertical/columnar), split the
architecture:

### Vertical (Columnar): The 16K Fingerprint Store

Use the RedisGraph 16K codebase as-is. The 256-word fingerprint carries
semantic content (blocks 0-12) plus the RedisGraph schema sidecar
(ANI, NARS, RL, bloom, graph metrics — blocks 13-15). This is the proven,
tested, 259-passing-tests implementation.

### Horizontal (Row): Cognitive Metadata Table

Store cognitive-specific metadata (the full 18D qualia, 7-layer markers
with timestamps, GEL execution state, semantic kernel state, rung history,
surplus edges beyond 32) in a row-oriented table. One row per node address.

```
┌─────────────────────────────────────────────────────────────────┐
│  VERTICAL: [u64; 256] columnar fingerprint store (Lance)        │
│  ├── Words 0-207:   Semantic content                            │
│  ├── Words 208-215: ANI + NARS + bloom (RedisGraph schema)      │
│  ├── Words 216-231: RL + graph metrics                          │
│  └── Words 232-255: Reserved / inline edges (16 sparse)         │
├─────────────────────────────────────────────────────────────────┤
│  HORIZONTAL: Row table per node (Lance or SQLite)               │
│  ├── addr:          Addr (u16, primary key)                     │
│  ├── qualia_18d:    [f32; 18] (full precision, all 18 channels) │
│  ├── layer_markers: [LayerMarker; 7] (with timestamps, flags)   │
│  ├── gel_state:     GelExecutionContext (full program state)     │
│  ├── kernel_state:  SemanticKernelState (full kernel context)    │
│  ├── rung_history:  VecDeque<RungShiftEvent> (decision log)     │
│  ├── surplus_edges: Vec<(Verb, Addr)> (overflow beyond 16-32)   │
│  ├── label:         String (human-readable)                     │
│  ├── payload:       Vec<u8> (arbitrary data)                    │
│  └── timestamps:    (created, last_access, last_modified)       │
└─────────────────────────────────────────────────────────────────┘
```

### XOR Coupling Between Vertical and Horizontal

The key trick: the horizontal row's identity IS the vertical fingerprint.
The address (u16) is the primary key for both. But you can go further:

1. **Fingerprint hash as row key**: `hash(words[0..208])` gives a
   content-addressable key for the horizontal row. Same semantic content
   = same row = shared metadata.

2. **XOR delta between row versions**: When the horizontal row changes
   (qualia shift, rung elevation, new edges), compute the XOR delta of
   the row's serialized form. Store the delta in the same XOR write cache
   that handles fingerprint deltas. One cache, two dimensions.

3. **Cross-dimensional query**: DataFusion can JOIN the vertical fingerprint
   table with the horizontal metadata table on `addr`. The query optimizer
   pushes predicates from the horizontal table into the vertical scan
   (inline ANI/NARS predicates) and vice versa.

```sql
-- DataFusion query that spans both dimensions
SELECT v.addr, v.fingerprint, h.qualia_18d, h.gel_state
FROM fingerprints v
JOIN node_metadata h ON v.addr = h.addr
WHERE hamming_distance(v.fingerprint, $query) < 500
  AND schema_passes(v.fingerprint, '{"ani": {"min_level": 5}}')
  AND h.qualia_18d[0] > 0.7  -- valence filter from horizontal table
ORDER BY hamming_distance(v.fingerprint, $query)
LIMIT 10;
```

### When This Design Makes Sense

- **Full qualia precision matters**: 18 dimensions at f32 precision for
  research/visualization, not just the top 8 at u16 for search filtering
- **Execution state is large**: GEL programs with deep stacks, branching
  history, and checkpoint state don't fit in 1-2 words
- **Audit trail needed**: Rung shift history, timestamp logs, decision
  provenance — data that grows over time
- **Hub nodes**: Nodes with hundreds of edges where 32 inline slots aren't
  enough and overflow is the common case, not the exception

### When This Design Is Overkill

- **Search-only workloads**: If you're just doing ANN search with predicate
  filtering, the vertical store alone (with quantized metadata in-fingerprint)
  is simpler and faster
- **Low-metadata nodes**: Most graph nodes in practice carry minimal cognitive
  state — the full horizontal row is mostly empty

### The Pragmatic Middle Ground

Start with the vertical-only approach (all metadata in 256 words). Add the
horizontal table ONLY for the fields that genuinely don't fit:

- Surplus edges (overflow from 32 inline slots) → horizontal
- Full 18D qualia (when 8-channel u16 isn't enough) → horizontal
- Rung shift history (unbounded log) → horizontal
- Labels and payloads (heap-allocated strings/blobs) → horizontal

Everything else stays in the fingerprint. The XOR write cache serves both
dimensions. DataFusion queries both through the same TableProvider.

---

## Alternative Design: 3D Holographic Memory (32K = 2^15)

The most radical alternative. Instead of a 1D 256-word vector, use three
8K-bit vectors (X, Y, Z) that create a 3-dimensional holographic memory
through XOR superposition.

### The Structure: 512 Words (32,768 bits = 2^15)

```
┌────────────────────────────────────────────────────────────────────────┐
│  512 words = 32K bits = 4KB per fingerprint                           │
├────────────────────────────────────────────────────────────────────────┤
│  X dimension: words 0-127    (8K bits, 128 words) — CONTENT / WHAT    │
│  Y dimension: words 128-255  (8K bits, 128 words) — CONTEXT / WHERE   │
│  Z dimension: words 256-383  (8K bits, 128 words) — RELATION / HOW    │
│  Metadata:    words 384-511  (8K bits, 128 words) — everything else   │
└────────────────────────────────────────────────────────────────────────┘
```

### Why This Is Holographic

In VSA/HDR algebra, `bind(a, b) = a ⊕ b` creates a compound representation.
With three 8K vectors, the XOR-bound product space is:

```
8,192 × 8,192 × 8,192 = 549,755,813,888 ≈ 512 billion
```

A single 32K-bit vector encodes a holographic memory with **512 billion
addressable data points**. You don't store 512 billion records — you ENCODE
them through the combinatorial power of XOR binding across three orthogonal
dimensions.

### How It Works: XYZ Superposition

```rust
// Store: bind content × context × relation into a single holographic trace
let trace = x_content ^ y_context ^ z_relation;

// Retrieve: probe with any two dimensions to recover the third
let recovered_relation = trace ^ x_content ^ y_context;
// recovered_relation ≈ z_relation (with noise from other stored traces)

let recovered_content = trace ^ y_context ^ z_relation;
// recovered_content ≈ x_content

let recovered_context = trace ^ x_content ^ z_relation;
// recovered_context ≈ y_context
```

This is the holographic property: given any two of three components,
XOR recovers the third. Multiple traces can be superposed (majority-vote
bundled) and individual associations recovered by probing.

### What Each Dimension Carries

**X (Content/What)**: The semantic identity of the concept. What it IS.
Equivalent to the current 10K fingerprint's semantic content, but at 8K.

**Y (Context/Where)**: The situational context. Where/when it appears.
Enables queries like "what concepts appear in THIS context?" by probing
`Y_context ⊕ stored_trace` to recover X.

**Z (Relation/How)**: The relational structure. How it connects.
Encodes the verb/edge type. Probing `X_subject ⊕ Z_verb` recovers Y
(the object in the relation subject→verb→object).

### The 128-Word Metadata Block

With 128 words (8,192 bits = 1,024 bytes) for metadata, there is
abundant room:

```
METADATA BLOCK (words 384-511, 8,192 bits)
├── Words 384-387:  ANI/consciousness (4 words = 256 bits, full 7-layer state)
├── Words 388-389:  NARS truth (2 words = frequency + confidence + evidence + horizon)
├── Words 390-391:  Qualia (2 words = top 8 channels at u16)
├── Word  392:      GEL execution state
├── Word  393:      Semantic kernel state
├── Words 394-396:  DN tree (parent, depth, rung, sigma, type, flags, timestamps)
├── Words 397-412:  Inline edges: 64 edges (4 per word × 16 words)
├── Words 413-414:  Edge overflow metadata (count, flag, table addr, degrees)
├── Word  415:      Schema version + dimensional flags
├── Words 416-423:  RL/Decision (8 words, same as 256-word layout)
├── Words 424-431:  Bloom filter (8 words = 512-bit bloom, better FP rate)
├── Words 432-447:  Graph metrics (16 words, room for all metrics at full precision)
├── Words 448-463:  Qualia overflow (full 18D at f32: 18 × 32 bits = 9 words)
├── Words 464-479:  7-Layer markers (full LayerMarker state, 16 words)
├── Words 480-495:  Rung history (last 8 shift events condensed)
├── Words 496-510:  Reserved for future use
└── Word  511:      Checksum + version flags
```

**64 inline edges** (vs 16-32 at 256 words). The overflow threshold
moves from "hub nodes with >32 edges" to "only extreme hubs with >64 edges."
The metadata block alone is larger than the entire current BindNode struct.

### The Trade-offs

**Gains**:
- 512 billion XOR-addressable data points in one 4KB record
- Per-dimension queries: similar content in different context, or same
  relation applied to different content
- 128 words of metadata: room for EVERYTHING at full precision, no
  quantization compromises, 64 inline edges
- Holographic retrieval: given 2 of 3 dimensions, recover the third
- XOR delta works per-dimension: update content without touching context

**Costs**:
- 4KB per fingerprint instead of 2KB (256 words) or 1.25KB (156 words)
- 65K addresses × 4KB = 256MB base store (vs 128MB at 256 words)
- sigma per 8K dimension = sqrt(8192/4) = 45.25 (not a clean integer;
  sigma=64 at 16K is cleaner for threshold math)
- SIMD: 128 words / 8 = 16 AVX-512 iterations per dimension (clean,
  zero remainder, but 48 iterations total vs 32 for 256 words)
- Existing HDR cascade search needs adaptation for per-dimension distance
- The XOR holographic encoding has noise that scales with the number of
  stored traces — capacity is O(sqrt(8192)) ≈ 90 high-fidelity traces
  per superposition before retrieval degrades

### When 3D Holographic Makes Sense

- **Relational reasoning**: "what relates to X the way Y relates to Z?"
  is a single XOR probe, not a graph traversal
- **Analogical transfer**: `king ⊕ male ⊕ female ≈ queen` works natively
  in XYZ space — content dimension shifts while relation holds
- **Context switching**: Same concept in different contexts creates different
  traces. Probing by context recovers context-appropriate meaning.
- **Massive implicit storage**: 512 billion data points in 4KB is a
  compression ratio that no columnar store can match for holographic data

### When 256 Words Is Better

- **Pure ANN search**: You just need distance and predicates, not
  dimensional decomposition
- **Memory-constrained**: 2KB per fingerprint is half the cost
- **Clean sigma**: sigma=64 is more elegant for threshold computation
- **Simpler implementation**: No dimensional algebra, just flat word array
- **Tested and proven**: The 256-word layout has 259 passing tests today

### The Path Between: Start 256, Graduate to 512

The 256-word implementation is the foundation. The 512-word 3D layout is
the evolution. The migration path:

1. Build and ship on 256 words (docs 01-06 cover this completely)
2. Validate the metadata layout and inline edges in production
3. When relational reasoning demands it, extend to 512 words:
   - Words 0-207 become X dimension (content)
   - Add Y dimension (context) at words 128-255 of the new layout
   - Add Z dimension (relation) at words 256-383
   - Move metadata to words 384-511 (128 words, 3× more room)
4. The compat layer (zero-extend 256→512) is the same pattern as 156→256
