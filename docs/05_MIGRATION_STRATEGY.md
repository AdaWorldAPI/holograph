# Migration Strategy: 156-Word to 256-Word Without Breaking Anything

> Additive changes only. No overwrites. No breaking existing tests.
> The 256-word system coexists with the current 156-word system until
> all tests pass on both, then the old path is deprecated.

---

## Phase 0: Preparation (No Code Changes)

### Read these files in ladybug-rs:
- `CLAUDE.md` — understand what works and what doesn't
- `docs/STORAGE_CONTRACTS.md` — the 9 race conditions
- `docs/COMPOSITE_FINGERPRINT_SCHEMA.md` — the 160-word proposal
- `docs/COGNITIVE_RECORD_256.md` — the 256-word proposal

### Read these files in docs/redisgraph/:
- `01_THE_256_WORD_SOLUTION.md` — why 256
- `02_DATAFUSION_NOT_LANCEDB.md` — where to invest
- `03_CAM_PREFIX_SOLUTION.md` — how CAM fits
- `04_RACE_CONDITION_PATTERNS.md` — fix templates

---

## Phase 1: Add 16K Module (New Files Only)

Create `src/width_16k/` alongside existing code. **Do not modify any existing
files yet.**

### New files:

```
src/width_16k/
├── mod.rs          # Constants: VECTOR_WORDS=256, VECTOR_BITS=16384, SIGMA=64
├── schema.rs       # SchemaSidecar: ANI, NARS, RL, bloom, graph metrics
│                   # write_to_words(), read_from_words(), read_version()
├── search.rs       # SchemaQuery, passes_predicates(), masked_distance()
│                   # bloom_accelerated_search(), rl_guided_search()
│                   # schema_merge(), schema_bind()
├── compat.rs       # zero_extend(), truncate(), cross_width_distance()
│                   # migrate_batch(), migrate_batch_with_schema()
└── xor_bubble.rs   # XorDelta, DeltaChain, XorBubble, XorWriteCache
                    # ConcurrentWriteCache (with RwLock)
```

### Source: Copy from RedisGraph

The RedisGraph implementation has all of these files tested and passing.
Copy them, adjusting:
- Module paths (`crate::width_16k::` → ladybug path)
- Import paths (`crate::bitpack::BitpackedVector` → `crate::core::Fingerprint`)
- Constants (`VECTOR_WORDS` → match ladybug naming conventions)

### What to verify:

```bash
# Existing tests still pass
cargo test

# New module compiles
cargo test --lib width_16k
```

---

## Phase 2: Wire Compatibility Layer

### Modify: `src/lib.rs`

Add the new module declaration alongside existing ones:

```rust
pub mod width_16k;  // Add this line, don't remove anything

// Keep existing constants — they're still used by existing code
pub const FINGERPRINT_BITS: usize = 10_000;
pub const FINGERPRINT_U64: usize = 157;

// Add new constants
pub const FP_BITS_16K: usize = 16_384;
pub const FP_WORDS_16K: usize = 256;
pub const FP_BYTES_16K: usize = 2_048;
pub const FP_SIGMA_16K: usize = 64;
```

### Add: `src/core/fingerprint_16k.rs`

New type that wraps the 256-word array:

```rust
use crate::width_16k::{VECTOR_WORDS, schema::SchemaSidecar};
use crate::core::Fingerprint;

#[repr(align(64))]
#[derive(Clone)]
pub struct Fingerprint16K {
    data: [u64; VECTOR_WORDS],
}

impl Fingerprint16K {
    /// Zero-extend a 10K fingerprint to 16K
    pub fn from_10k(fp: &Fingerprint) -> Self {
        let mut data = [0u64; VECTOR_WORDS];
        data[..157].copy_from_slice(fp.as_words());
        Self { data }
    }

    /// Truncate back to 10K (lossless if schema blocks are zero)
    pub fn to_10k(&self) -> Fingerprint {
        let mut words = [0u64; 157];
        words.copy_from_slice(&self.data[..157]);
        Fingerprint::from_raw(words)
    }

    /// Read schema metadata
    pub fn schema(&self) -> SchemaSidecar {
        SchemaSidecar::read_from_words(&self.data)
    }

    /// Write schema metadata
    pub fn set_schema(&mut self, schema: &SchemaSidecar) {
        schema.write_to_words(&mut self.data);
    }

    /// Semantic distance (blocks 0-12 only)
    pub fn semantic_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..208 {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        dist
    }

    /// XOR bind
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            result[i] = self.data[i] ^ other.data[i];
        }
        Self { data: result }
    }

    pub fn as_words(&self) -> &[u64; VECTOR_WORDS] { &self.data }
    pub fn as_words_mut(&mut self) -> &mut [u64; VECTOR_WORDS] { &mut self.data }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, 2048) }
    }
}
```

### Verify:

```bash
cargo test  # All existing tests still pass
cargo test width_16k  # New tests pass
cargo test fingerprint_16k  # Compat tests pass
```

---

## Phase 3: Add DataFusion Extensions

### New files:

```
src/query/
├── bind_space_provider.rs  # BindSpaceTableProvider
├── hdr_udfs.rs             # hamming_distance, xor_bind, schema_passes UDFs
└── hdr_optimizer.rs        # HdrCascadePushdown optimizer rule
```

### Modify: `src/query/datafusion.rs`

Add registration of new UDFs and table provider:

```rust
impl SqlEngine {
    pub async fn new_16k(bind_space: Arc<BindSpace16K>) -> Self {
        let mut engine = Self::new().await;
        // Register 16K table provider
        engine.ctx.register_table("nodes", Arc::new(
            BindSpaceTable::new(bind_space.clone(), Zone::Nodes)
        ));
        engine.ctx.register_table("surface", Arc::new(
            BindSpaceTable::new(bind_space.clone(), Zone::Surface)
        ));
        // Register HDR UDFs
        register_hdr_udfs(&engine.ctx);
        // Register optimizer
        engine.ctx.add_optimizer_rule(Arc::new(HdrCascadePushdown));
        engine
    }
}
```

### Verify:

```sql
-- These should work after Phase 3:
SELECT addr, hamming_distance(fingerprint, $query) as dist
FROM nodes
ORDER BY dist ASC
LIMIT 10;

SELECT addr, semantic_distance(fingerprint, $query) as dist
FROM nodes
WHERE schema_passes(fingerprint, '{"ani": {"min_level": 5, "min_activation": 300}}')
ORDER BY dist ASC
LIMIT 10;
```

---

## Phase 4: Wire CAM Operations

### Modify: `src/learning/cam_ops.rs`

Replace stubs with schema-block dispatches. Don't delete the existing
match arms — add 16K variants alongside them:

```rust
match op {
    0x300 => {
        if args.len() == 2 {
            let fp_a = Fingerprint16K::from_10k(&args[0]);
            let fp_b = Fingerprint16K::from_10k(&args[1]);
            OpResult::Scalar(fp_a.semantic_distance(&fp_b) as f64)
        } else {
            OpResult::Error("HAMMING.DISTANCE requires 2 args".into())
        }
    }
    // ... existing arms unchanged
}
```

### Verify:

```bash
cargo test cam_ops  # Existing CAM tests pass
cargo test cam_16k  # New CAM-on-16K tests pass
```

---

## Phase 5: Fix Race Conditions

Apply the fixes from `04_RACE_CONDITION_PATTERNS.md` to:
1. `hardening.rs` — WAL + LruTracker
2. `resilient.rs` — WriteBuffer
3. `xor_dag.rs` — Parity TOCTOU
4. `temporal.rs` — Serializable conflict

Each fix is ~50 lines. All follow the same pattern: merge two locks into one.

### Verify:

```bash
cargo test storage  # All storage tests pass
cargo test --release storage  # Race conditions don't manifest under optimization
```

---

## Phase 6: Deprecate 156-Word Path

Only after ALL tests pass on the 256-word path:

1. Mark `FINGERPRINT_WORDS = 156` as `#[deprecated]`
2. Mark `Fingerprint` (157 words) as `#[deprecated]`
3. Update `BindSpace` to use `[u64; 256]` arrays
4. Update `hdr_cascade.rs` to use `WORDS = 256`
5. Remove SIMD remainder loops

This is the last step, not the first.

---

## File Change Summary

| Phase | New Files | Modified Files | Risk |
|-------|-----------|----------------|------|
| 1 | 5 (width_16k/) | 0 | Zero |
| 2 | 1 (fingerprint_16k.rs) | 1 (lib.rs: add mod) | Minimal |
| 3 | 3 (query extensions) | 1 (datafusion.rs: add registrations) | Low |
| 4 | 0 | 1 (cam_ops.rs: add match arms) | Low |
| 5 | 0 | 4 (storage files: fix locks) | Medium |
| 6 | 0 | ~10 (deprecate old path) | Medium |

Total new files: **9**
Total modified files: **~17** (spread across 6 phases)
Lines of code: **~2000 new, ~200 modified**

---

## What Success Looks Like

```bash
# All 408 existing tests pass (none broken)
cargo test
# test result: ok. 408 passed; 10 failed; 0 ignored

# Plus ~100 new tests for 16K functionality
cargo test width_16k
# test result: ok. ~100 passed; 0 failed

# Plus DataFusion integration tests
cargo test query::bind_space_provider
# test result: ok. ~20 passed; 0 failed

# Schema predicates work in SQL
cargo test query::hdr_udfs
# test result: ok. ~15 passed; 0 failed
```

The 10 pre-existing failures are unrelated and should be tracked separately.
