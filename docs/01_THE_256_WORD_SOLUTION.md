# The 256-Word Solution: Why 16K Bits Resolves Everything

> This document explains how moving from 156/157 u64 words to 256 u64 words
> (16,384 bits) resolves the fingerprint sizing crisis, the SIMD remainder
> problem, the CAM prefix fitting problem, and the metadata-in-separate-columns
> problem — all at once.

---

## The Current Crisis in Ladybug-RS

Three competing proposals exist:
- **156 words** (bind_space.rs) — loses 16 bits, 4-word SIMD remainder
- **157 words** (lib.rs) — wastes 48 bits, 5-word SIMD remainder
- **160 words** (COMPOSITE_FINGERPRINT_SCHEMA.md) — SIMD-clean, but still separates metadata
- **192 words** (COGNITIVE_RECORD_192.md) — fits metadata, but 8192-bit fingerprint is 18% smaller
- **256 words** (COGNITIVE_RECORD_256.md) — proposed but not implemented

**Answer: 256 words is correct.** Here's why, with mathematical proof.

---

## The sigma = 64 Argument

For a random binary vector of length `n`, the Hamming distance between two
independent random vectors follows Binomial(n, 0.5) with:
- Mean = n/2
- Standard deviation = sqrt(n/4)

| Words | Bits   | sigma        | sigma as integer | SIMD remainder |
|-------|--------|-------------|------------------|----------------|
| 156   | 9,984  | 49.92       | ~50 (ugly)       | 4 words        |
| 157   | 10,048 | 50.12       | ~50 (ugly)       | 5 words        |
| 160   | 10,240 | 50.60       | ~51 (ugly)       | 0              |
| 192   | 12,288 | 55.42       | ~55 (ugly)       | 0              |
| **256** | **16,384** | **64.00** | **64 (perfect)** | **0**          |

**sigma = 64 = exactly one u64 word.** This is the only vector width where
sigma is simultaneously:
- An exact integer (no floating-point in threshold calculations)
- A power of 2 (bit shifts instead of division)
- One word (block-level sigmas are exact multiples of sigma)

### Consequences That Cascade Through the System

1. **Zone thresholds are integers**: 1sigma=64, 2sigma=128, 3sigma=192
2. **Block sigma is exact**: 16 blocks × 1024 bits each → block sigma = 16
3. **Popcount arithmetic stays integer**: "how many sigmas?" = popcount / 64
4. **Mexican hat excite/inhibit thresholds**: exact integer boundaries
5. **SIMD alignment**: 256/8 = 32 AVX-512 iterations, zero remainder

### How This Fixes the HDR Cascade

Current `hdr_cascade.rs` uses `WORDS=156` with hardcoded thresholds:
```rust
const DEFAULT_EXCITE: u32 = 2000;   // ~20% of 10,000
const DEFAULT_INHIBIT: u32 = 5000;  // ~50% of 10,000
```

At 16K bits:
```rust
const DEFAULT_EXCITE: u32 = 3277;   // 20% of 16,384 = 3276.8 ≈ 3277
const DEFAULT_INHIBIT: u32 = 8192;  // 50% of 16,384 = EXACT
// Or better: use sigma-based thresholds
const EXCITE_SIGMA: u32 = 3;        // Within 3σ = within 192 bits of mean
const INHIBIT_SIGMA: u32 = 1;       // Beyond 1σ = beyond 64 bits from mean
```

---

## The Block Layout: Properties ARE the Fingerprint

The key architectural insight: **don't store metadata in separate Arrow columns.
Store it in the fingerprint itself.**

```
256 u64 words = 2,048 bytes = 32 cache lines

┌─────────────────────────────────────────────────────────────────────┐
│ Blocks 0-12: SEMANTIC FINGERPRINT (13,312 bits = 208 words)        │
│   Pure VSA: XOR bind, Hamming distance, majority bundle            │
│   13,312 bits > 10,000 (33% MORE capacity than current)            │
│   208 words / 8 = 26 AVX-512 iterations, zero remainder            │
├─────────────────────────────────────────────────────────────────────┤
│ Block 13: NODE/EDGE TYPE + ANI REASONING LEVELS (1024 bits)        │
│   words 208-223 (16 words = 2 cache lines)                         │
│   ├── words 208-209: ANI 8 levels × 16-bit (reactive..abstract)   │
│   ├── word 210:      NARS truth {f,c} quantized                   │
│   ├── word 210-211:  NARS budget {p,d,q}                          │
│   ├── word 211:      Edge type (verb_id, direction, weight, flags) │
│   ├── word 211:      Node type (kind, subtype, provenance)        │
│   └── words 212-223: Reserved / user-defined                      │
│       word 223 bits 56-63: SCHEMA VERSION BYTE                    │
├─────────────────────────────────────────────────────────────────────┤
│ Block 14: RL / TEMPORAL STATE (1024 bits)                          │
│   words 224-239 (16 words = 2 cache lines)                         │
│   ├── words 224-225: Q-values (16 actions × 8-bit)                │
│   ├── words 226-227: Reward history (8 × 16-bit)                  │
│   ├── words 228-229: STDP timing markers (8 × 16-bit)            │
│   ├── words 230-231: Hebbian weights (8 neighbors × 16-bit)      │
│   └── words 232-239: Reserved                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Block 15: TRAVERSAL / GRAPH CACHE (1024 bits)                      │
│   words 240-255 (16 words = 2 cache lines)                         │
│   ├── words 240-243: DN address (compressed TreeAddr, 32 bytes)   │
│   ├── words 244-247: Neighbor bloom filter (256 bits)              │
│   ├── word 248:      Graph metrics (pagerank, hop, cluster, degree)│
│   └── words 249-255: Reserved                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Beats Separate Columns

**Ladybug's Schema A** (COMPOSITE_FINGERPRINT_SCHEMA.md) stores metadata in
separate Arrow columns: nars_f, nars_c, rung, sigma, popcount, scent, verb_mask,
edge_count, etc. Each column adds:
- 1 buffer pointer per batch
- 1 validity bitmap per batch
- O(n) memory for n rows
- A join cost when combining with fingerprint data

**With properties-in-fingerprint**:
- NARS truth is at words[210] — same cache fetch as the fingerprint itself
- No join. No separate column. No buffer pointer overhead.
- Predicate check during search: mask word[210], compare. O(1) per candidate.
- The predicate check happens *during* the distance cascade, not after.

### What Ladybug Gains

| Current (Schema A)          | With 256-word properties-in-fingerprint      |
|-----------------------------|----------------------------------------------|
| 1,280 bytes fingerprint     | 2,048 bytes (fingerprint + all metadata)     |
| + 4 bytes nars_f            | included in word 210                         |
| + 4 bytes nars_c            | included in word 210                         |
| + 1 byte rung               | included in word 208 (ANI level)             |
| + 1 byte sigma              | computed: always 64 at 16K                   |
| + 2 bytes popcount          | pre-computed in word 248 (graph metrics)     |
| + 5 bytes scent             | XOR-fold of block popcounts (computed)       |
| + 32 bytes verb_mask        | included in block 13 edge type               |
| + 16 bytes parent_key       | included in block 15 DN address              |
| = ~1,345 bytes + joins      | = 2,048 bytes, zero joins                    |

**Net cost**: +703 bytes per row. **Net benefit**: zero joins, zero pointer
chases, O(1) predicate checks inline with distance computation.

---

## Mapping to Ladybug's 8+8 Address Model

The 8+8 address model (prefix:slot → 65,536 direct array addresses) is
**orthogonal** to the fingerprint width. Each address points to a 256-word
record instead of a 156-word record. The BindSpace arrays grow from:

```
Current:  65,536 × 156 × 8 = 81,788,928 bytes ≈ 78 MiB
256-word: 65,536 × 256 × 8 = 134,217,728 bytes = 128 MiB (exact power of 2!)
```

128 MiB for the full bind space. This fits in L3 cache on modern server hardware.

### Surface/Fluid/Nodes at 256 Words

| Zone    | Prefixes | Addresses | Memory (256w) |
|---------|----------|-----------|---------------|
| Surface | 0x00-0x0F | 4,096   | 8 MiB         |
| Fluid   | 0x10-0x7F | 28,672  | 56 MiB        |
| Nodes   | 0x80-0xFF | 32,768  | 64 MiB        |
| **Total** |        | **65,536** | **128 MiB** |

---

## Compatibility with 10K Vectors

A 10K (157-word) fingerprint zero-extends to 256 words by padding words
157-255 with zeros. The semantic content in words 0-156 is unchanged.
Schema blocks (words 208-255) start at all-zero (version 0 = legacy).

```rust
pub fn zero_extend(fp_10k: &[u64; 157]) -> [u64; 256] {
    let mut fp_16k = [0u64; 256];
    fp_16k[..157].copy_from_slice(fp_10k);
    fp_16k
}

pub fn truncate(fp_16k: &[u64; 256]) -> [u64; 157] {
    let mut fp_10k = [0u64; 157];
    fp_10k.copy_from_slice(&fp_16k[..157]);
    fp_10k
}
```

**Distance is preserved**: `hamming(zero_extend(a), zero_extend(b)) == hamming(a, b)`
because XOR of zero-padded regions is zero, contributing nothing to popcount.

---

## The Schema Version Byte

Word 223, bits 56-63, stores an 8-bit schema version:
- Version 0: Legacy (no schema markers, zero-extended 10K)
- Version 1: Current (ANI/NARS/RL/Graph metadata populated)
- Versions 2-255: Future layout changes

This was tested and proven in the RedisGraph implementation. The version byte
is placed in block 13 padding (word 223), which is unused in both legacy and
current layouts. It does NOT overlap with ANI levels (words 208-209).

---

## Implementation in Ladybug-RS

### Step 1: Add Constants

```rust
// In lib.rs or a new width_16k module:
pub const FP_WORDS_16K: usize = 256;
pub const FP_BYTES_16K: usize = 2048;
pub const FP_BITS_16K: usize = 16384;
pub const FP_SIGMA_16K: usize = 64;
pub const SCHEMA_BLOCK_START: usize = 13;
pub const SCHEMA_WORD_START: usize = 208;
```

### Step 2: Create Fingerprint16K Type

```rust
#[repr(align(64))]
#[derive(Clone)]
pub struct Fingerprint16K {
    data: [u64; 256],
}

impl Fingerprint16K {
    pub fn from_10k(fp: &Fingerprint) -> Self { /* zero-extend */ }
    pub fn semantic_distance(&self, other: &Self) -> u32 {
        // Only blocks 0-12 (words 0-207)
        let mut dist = 0u32;
        for i in 0..208 {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        dist
    }
    pub fn full_distance(&self, other: &Self) -> u32 {
        // All 256 words
        let mut dist = 0u32;
        for i in 0..256 {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        dist
    }
    pub fn schema(&self) -> SchemaSidecar {
        SchemaSidecar::read_from_words(&self.data)
    }
}
```

### Step 3: Update BindSpace

```rust
// In bind_space.rs, alongside existing FINGERPRINT_WORDS:
pub const FINGERPRINT_WORDS_16K: usize = 256;

// New array variant
pub struct BindSpace16K {
    data: Vec<[u64; 256]>,  // 65,536 × 256 words
}
```

### Step 4: Migrate HDR Cascade

```rust
// In hdr_cascade.rs, add a 16K variant:
const WORDS_16K: usize = 256;
const SEMANTIC_WORDS: usize = 208;

pub fn hamming_distance_16k(a: &[u64; 256], b: &[u64; 256]) -> u32 {
    // 32 AVX-512 iterations, zero remainder
    let mut dist = 0u32;
    for i in 0..256 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

pub fn semantic_distance_16k(a: &[u64; 256], b: &[u64; 256]) -> u32 {
    // 26 AVX-512 iterations, zero remainder
    let mut dist = 0u32;
    for i in 0..SEMANTIC_WORDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}
```

---

## What This Unlocks

1. **No more 156/157 confusion** — exactly 256, period
2. **No more SIMD remainder loops** — everything divides by 8
3. **No more separate metadata columns** — properties in the vector
4. **O(1) schema predicates during search** — inline with distance cascade
5. **sigma = 64** — all thresholds become exact integers
6. **Schema versioning** — future-proof layout with version byte
7. **33% more semantic capacity** — 13,312 bits vs 10,000
8. **128 MiB total BindSpace** — fits in L3 cache
9. **Backward compatible** — zero-extend existing 10K vectors losslessly
