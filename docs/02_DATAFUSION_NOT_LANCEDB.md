# Stop Reimplementing LanceDB Features — Extend DataFusion + Vendor-Import Lance

> The ladybug-rs codebase hardcoded XOR backup, caching, and similar features
> directly into the BindSpace↔Arrow layer (`lance_zero_copy/`, `unified_engine.rs`)
> instead of vendor-importing LanceDB and adding those features as extensions
> to Lance itself. This document explains the better path: vendor-import Lance,
> add XOR-delta extensions there, and use DataFusion extensions for query.

---

## The Actual Problem

### What Happened

The ladybug-rs team wanted features that LanceDB doesn't natively provide:
- **XOR delta backup** — store sparse diffs instead of full snapshots
- **XOR write cache** — avoid mutating Arrow columns (copy-on-write avoidance)
- **Schema predicate filtering** — inline metadata checks during search
- **Bloom-accelerated search** — neighbor bonus during ANN

Instead of vendor-importing LanceDB and adding these features *inside* the
Lance codebase, the team built them from scratch in `lance_zero_copy/` and
`unified_engine.rs` — reimplementing parts of what Lance already does
(Arrow buffer management, column scanning, batch I/O) while adding the
XOR-specific features on top.

This is why `lance_zero_copy/` exists: it's a parallel Arrow integration
layer that doesn't depend on the Lance crate. It works, but it means
maintaining two storage paths (Lance for persistence, ArrowZeroCopy for
runtime), and the XOR features aren't available in the persistence layer.

### The Vendor Directory Already Has Lance 2.1

```
vendor/
├── lance/           # Lance 2.1.0-beta.0 source code
│   └── rust/lance/  # The actual Rust crate
└── lancedb/         # LanceDB source code
```

The source is right there. The API mismatch exists because `Cargo.toml`
pulls from crates.io (`lance = "1.0"`) instead of the vendor directory.

---

## The Correct Architecture: Three Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: QUERY (DataFusion extensions)                                  │
│                                                                         │
│   Custom TableProvider → reads from BindSpace + Lance                   │
│   HDR UDFs → hamming_distance, xor_bind, schema_passes                  │
│   Optimizer rule → pushes schema predicates below sort                  │
│   Cypher transpiler → maps MATCH patterns to SQL with UDFs              │
│                                                                         │
│   ↑ This is where ladybug-rs should invest query logic                  │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: RUNTIME (BindSpace + XOR write cache)                          │
│                                                                         │
│   BindSpace: 65,536 × 256 u64 arrays (128 MiB, direct addressing)      │
│   ConcurrentWriteCache: RwLock<XorWriteCache> for delta accumulation    │
│   Reads: cache.read_through(addr, base_words) — zero-copy or patched   │
│   Writes: cache.record_delta(addr, delta) — no Arrow mutation           │
│   Flush: batch-apply deltas to Lance, clear cache                       │
│                                                                         │
│   ↑ This is what ArrowZeroCopy partially does — unify with Lance        │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: PERSISTENCE (Vendor-imported Lance with XOR extensions)        │
│                                                                         │
│   Standard Lance: Parquet storage, IVF-PQ index, versioned datasets     │
│   XOR Delta Extension: store sparse diffs as a Lance column             │
│   XOR Backup Extension: incremental backup via delta chains             │
│   Schema Column: FixedSizeBinary(2048) with inline metadata             │
│                                                                         │
│   ↑ Add these features TO Lance, not around it                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Vendor-Import Lance

### Fix Cargo.toml

```toml
# Replace:
# lance = { version = "1.0", optional = true }

# With:
[patch.crates-io]
lance = { path = "vendor/lance/rust/lance" }

# And update the dependency:
[dependencies]
lance = { version = "2.1.0-beta.0", optional = true }
```

### Update lance.rs API Calls

The vendor has Lance 2.1. Key API changes from 1.0:
- `Dataset::query()` → renamed/restructured
- `Schema` types moved to `lance::datatypes::Schema`
- `RecordBatchReader` trait requirements changed

These are mechanical fixes — the vendor source code is the documentation.

---

## Step 2: Add XOR Extensions to Vendor Lance

Instead of reimplementing Arrow buffer management in `lance_zero_copy/`,
add XOR delta support *inside* the vendor Lance codebase.

### Extension A: XOR Delta Column Type

Add a new column type to Lance that stores sparse XOR deltas:

```rust
// In vendor/lance/rust/lance/src/xor_delta.rs (NEW FILE)

/// A sparse XOR delta: bitmap + non-zero words
/// Stored as FixedSizeBinary in Lance
pub struct XorDeltaColumn {
    /// 4 u64 words = 256-bit bitmap indicating which words changed
    bitmap: [u64; 4],
    /// Only the non-zero words (typically <10 out of 256)
    nonzero: Vec<u64>,
}

impl XorDeltaColumn {
    pub fn compute(old: &[u64; 256], new: &[u64; 256]) -> Self {
        let mut bitmap = [0u64; 4];
        let mut nonzero = Vec::new();
        for w in 0..256 {
            let diff = old[w] ^ new[w];
            if diff != 0 {
                bitmap[w / 64] |= 1u64 << (w % 64);
                nonzero.push(diff);
            }
        }
        Self { bitmap, nonzero }
    }

    pub fn apply(&self, base: &mut [u64; 256]) {
        let mut nz_idx = 0;
        for w in 0..256 {
            if self.bitmap[w / 64] & (1u64 << (w % 64)) != 0 {
                base[w] ^= self.nonzero[nz_idx];
                nz_idx += 1;
            }
        }
    }

    /// Compression ratio (typically >3x for parent-child pairs)
    pub fn compression_ratio(&self) -> f32 {
        let compressed = 32 + self.nonzero.len() * 8; // bitmap + data
        let uncompressed = 256 * 8; // full fingerprint
        compressed as f32 / uncompressed as f32
    }
}
```

### Extension B: XOR Incremental Backup

Add to Lance's versioning system:

```rust
// In vendor/lance/rust/lance/src/xor_backup.rs (NEW FILE)

/// Incremental backup: store only XOR deltas between versions
pub struct XorBackup {
    base_version: u64,
    deltas: Vec<(u64, XorDeltaColumn)>, // (addr, delta)
}

impl XorBackup {
    /// Create backup from version N to version N+1
    pub fn from_versions(old: &Dataset, new: &Dataset) -> Self {
        // Read fingerprint columns, compute per-row deltas
        // Only store non-zero deltas (unchanged rows → no entry)
    }

    /// Apply backup to restore version N+1 from version N
    pub fn apply(&self, base: &mut Dataset) {
        // Apply each delta to the corresponding row
    }
}
```

### Extension C: Schema-Filtered Scan

Add to Lance's scan builder:

```rust
// In vendor/lance/rust/lance/src/schema_scan.rs (NEW FILE)

/// Custom scan that evaluates schema predicates inline during scan
pub struct SchemaFilteredScan {
    inner: Scan,
    predicates: Vec<SchemaPredicate>,
}

/// Predicate that operates on the fingerprint's schema blocks
pub enum SchemaPredicate {
    AniLevel { level: u8, min_activation: u16 },
    NarsConfidence { min: f32 },
    GraphCluster { id: u16 },
    BloomNeighbor { node_id: u64 },
}

impl SchemaFilteredScan {
    pub fn next_batch(&mut self) -> Option<RecordBatch> {
        loop {
            let batch = self.inner.next_batch()?;
            let fp_col = batch.column_by_name("fingerprint")?;
            // Filter rows where schema predicates pass
            let mask = evaluate_predicates(fp_col, &self.predicates);
            let filtered = filter_record_batch(&batch, &mask);
            if filtered.num_rows() > 0 {
                return Some(filtered);
            }
        }
    }
}
```

---

## Step 3: DataFusion Extensions (Same as Before)

The DataFusion extension layer is still the right place for query logic.
This doesn't change from the original document:

### BindSpaceTableProvider

Makes BindSpace look like a SQL table to DataFusion:

```rust
pub struct BindSpaceTable {
    bind_space: Arc<BindSpace16K>,
    zone: Zone,
}

impl TableProvider for BindSpaceTable {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("addr", DataType::UInt16, false),
            Field::new("fingerprint", DataType::FixedSizeBinary(2048), false),
            // Virtual columns extracted from fingerprint schema blocks:
            Field::new("popcount", DataType::UInt16, false),
            Field::new("nars_f", DataType::Float32, true),
            Field::new("nars_c", DataType::Float32, true),
            Field::new("ani_dominant", DataType::UInt8, true),
            Field::new("schema_version", DataType::UInt8, false),
        ]))
    }

    async fn scan(&self, ...) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(BindSpaceScan::new(
            self.bind_space.clone(),
            self.zone,
            projection.cloned(),
            filters.to_vec(),
        )))
    }
}
```

### HDR UDFs

```rust
pub fn register_hdr_udfs(ctx: &SessionContext) {
    // hamming_distance(a, b) → UInt32
    // xor_bind(a, b) → FixedSizeBinary(2048)
    // schema_passes(fp, predicate_json) → Boolean
    // semantic_distance(a, b) → UInt32 (blocks 0-12 only)
    // ani_level(fp, level_index) → UInt16
    // nars_truth(fp) → Struct{f, c}
}
```

### HdrCascadePushdown Optimizer Rule

Rewrites `SortExec(FilterExec(Scan))` into `TopKExec(HdrCascadeScan)` when
the sort key is hamming_distance and the filter uses schema_passes.

---

## What This Changes

### Old approach (hardcoded in lance_zero_copy/):
```
BindSpace → ArrowZeroCopy (custom Arrow management)
    ↓ (no Lance features: no versioning, no IVF index, no S3)
  Parquet (manual write)
```

### New approach (vendor-extend Lance):
```
BindSpace → ConcurrentWriteCache (XOR deltas in memory)
    ↓ flush
  Lance (vendor-imported, with XOR extensions)
    ↓ (gets: versioning, IVF index, S3, delta backup for free)
  Parquet / S3 / local storage
```

### Benefits of vendor-importing:
1. **Lance's IVF-PQ index** works on FixedSizeBinary(2048) out of the box
2. **Lance's versioning** gives time-travel for free
3. **Lance's S3 support** gives cloud persistence for free
4. **XOR delta backup** is ~3x compression (proven in RedisGraph tests)
5. **Schema-filtered scan** prunes during I/O, not after
6. `lance_zero_copy/` can be deprecated (its features move into Lance)

### Effort estimate:
- Fix Cargo.toml patch: 5 minutes
- Update lance.rs API calls: 1-2 hours (mechanical)
- XOR delta column extension: ~200 lines
- XOR backup extension: ~150 lines
- Schema-filtered scan: ~200 lines
- DataFusion table provider: ~200 lines
- DataFusion UDFs: ~150 lines
- DataFusion optimizer rule: ~150 lines

Total: **~1050 lines** of new code, plus ~100 lines of lance.rs fixes.

---

## Proven in RedisGraph

The XOR delta, write cache, schema predicate filtering, bloom-accelerated
search, and RL-guided search are all proven with 259 passing tests in the
RedisGraph HDR engine. The code can be copied directly into the vendor
Lance extensions with path adjustments.

Key source files to reference:
- `width_16k/xor_bubble.rs` — XorDelta, XorWriteCache, ConcurrentWriteCache
- `width_16k/search.rs` — SchemaQuery, passes_predicates, bloom/RL search
- `width_16k/schema.rs` — SchemaSidecar pack/unpack, version byte
- `navigator.rs` — Cypher procedure mapping, DN addressing
