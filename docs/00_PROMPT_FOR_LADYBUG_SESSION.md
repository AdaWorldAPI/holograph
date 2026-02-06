# Prompt for Ladybug-RS Claude Code Session

> Copy-paste this into a fresh Claude Code session working on the ladybug-rs
> repository. It transfers the learning curve from the RedisGraph HDR
> fingerprint engine review so the session starts at full understanding
> instead of cold.

---

## Context Prompt

```
I need you to help refactor ladybug-rs using architectural insights from a
parallel Rust codebase (RedisGraph HDR fingerprint engine) that solved the
same core problems ladybug-rs is struggling with. The insights are documented
in docs/redisgraph/ — read ALL files there before making any changes.

Key problems to solve, in priority order:

### 1. The 156/157 Word Bug — Use 256 Words (16K Bits)

The codebase has FINGERPRINT_WORDS=156 in bind_space.rs and FINGERPRINT_U64=157
in lib.rs. Neither is correct. The RedisGraph engine proved that 256 words
(16,384 bits = 2^14) is the right choice because:
- sigma = sqrt(16384/4) = 64 = exactly one u64 word
- 256 / 8 = 32 AVX-512 iterations with ZERO remainder
- 16 uniform blocks of 1024 bits each (no short last block)
- Blocks 0-12 carry semantic content (13,312 bits > 10K requirement)
- Blocks 13-15 carry structured metadata (ANI, NARS, RL, graph metrics)

Read docs/redisgraph/01_THE_256_WORD_SOLUTION.md for the complete analysis.

### 2. Stop Reimplementing LanceDB — Vendor-Import and Extend

The codebase hardcoded XOR backup, caching, and similar features into the
BindSpace-Arrow layer (lance_zero_copy/, unified_engine.rs) instead of
vendor-importing LanceDB and adding those features as Lance extensions.
The vendor directory already has Lance 2.1 source. The right path is:
- Fix Cargo.toml to use vendor path (the source is already there)
- Update lance.rs API calls for 2.1 (mechanical changes)
- Add XOR delta column, XOR backup, schema-filtered scan TO vendor Lance
- Use DataFusion extensions for query (TableProvider + UDFs + optimizer)
  instead of reimplementing query capabilities in application code

Read docs/redisgraph/02_DATAFUSION_NOT_LANCEDB.md for the 3-layer architecture.

### 3. The 4096 CAM Is Transport, Not Storage — Keep Only GEL

The 4096 CAM commandlets are NOT a storage problem. They belong in classes and
methods (impl TruthValue, impl QTable, impl Fingerprint16K). The 4096 transport
protocol reaches those methods like HTTP reaches REST endpoints. Remove all
commandlet implementations from cam_ops.rs (4,661 → ~200 lines of pure routing).
Only GEL (Graph Execution Language) stays in the CAM as a first-class concept —
it compiles programs into graph execution sequences.

Read docs/redisgraph/03_CAM_PREFIX_SOLUTION.md for the full architecture.

### 4. Race Conditions Have Known Fixes

All 9 documented race conditions follow the same pattern: lock released between
check and commit. The ConcurrentWriteCache pattern from RedisGraph (RwLock with
owned return values) solves most of them.

Read docs/redisgraph/04_RACE_CONDITION_PATTERNS.md for the fix templates.

### 5. Metadata Must Move INTO the Fingerprint

BindNode and CogValue store metadata as native Rust struct fields (label,
qualia, truth, access_count, parent, depth, rung, sigma). At 256 words, ALL
of this moves into the fingerprint as bit-packed words. This enables:
- Partial updates via XOR delta (no more "one value blocks all")
- Inline predicate filtering during HDR cascade search
- 16-32 inline edge slots per node (sparse adjacency in-fingerprint)
- XOR parent-child compression for DN tree storage
- Overflow to Lance tables for hub nodes with >32 edges

Read docs/redisgraph/06_METADATA_REVIEW.md for the complete bit layout.

### 6. Don't Overwrite Anything — Additive Changes Only

Create new files alongside existing ones. The migration from 156-word to 256-word
should be a separate module (width_16k/) that coexists with the current code.
Wire it in gradually, test both paths, then deprecate the old one.

Read docs/redisgraph/05_MIGRATION_STRATEGY.md for the step-by-step plan.

IMPORTANT: Read ALL docs/redisgraph/*.md files before starting. They contain
proven, tested solutions from a working implementation — not speculation.
```

---

## What This Prompt Does

1. **Transfers the learning curve** — The receiving session understands
   *why* 256 words, not just *that* 256 words
2. **Prevents the LanceDB trap** — Explicitly redirects to DataFusion
   extensions, which is where the leverage actually is
3. **Solves the CAM confusion** — CAM is transport (routing to methods),
   not storage. Only GEL stays in the CAM. cam_ops.rs shrinks from 4,661
   to ~260 lines.
4. **Provides fix templates** — Not just "fix the race conditions" but
   exact code patterns proven in another codebase
5. **Protects existing work** — Additive migration, no overwrites
6. **Maps every metadata field** — Complete bit layout for DN tree, edges,
   NARS, RL, qualia, GEL, kernel, bloom, graph metrics at 256 words

## Prerequisite

The docs/redisgraph/ directory must exist in the ladybug-rs repo. Copy it
from the RedisGraph repo or ensure both repos are accessible.
