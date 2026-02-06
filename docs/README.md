# RedisGraph HDR → Ladybug-RS Knowledge Transfer

> These documents transfer architectural insights from the RedisGraph HDR
> fingerprint engine (this repo) to the ladybug-rs codebase. They are the
> result of a deep review that identified why ladybug-rs hit roadblocks
> and how proven solutions from the HDR engine resolve them.

---

## Documents (Read in Order)

| # | File | What It Solves |
|---|------|----------------|
| 00 | [PROMPT_FOR_LADYBUG_SESSION.md](00_PROMPT_FOR_LADYBUG_SESSION.md) | Copy-paste prompt to bootstrap a Claude Code session on ladybug-rs |
| 01 | [THE_256_WORD_SOLUTION.md](01_THE_256_WORD_SOLUTION.md) | The 156/157 word bug, SIMD remainder, metadata fitting, sigma=64 |
| 02 | [DATAFUSION_NOT_LANCEDB.md](02_DATAFUSION_NOT_LANCEDB.md) | Why extending DataFusion beats rewriting LanceDB |
| 03 | [CAM_PREFIX_SOLUTION.md](03_CAM_PREFIX_SOLUTION.md) | CAM is transport/GEL only; commandlets → classes and methods |
| 04 | [RACE_CONDITION_PATTERNS.md](04_RACE_CONDITION_PATTERNS.md) | Fix templates for all 9 documented race conditions |
| 05 | [MIGRATION_STRATEGY.md](05_MIGRATION_STRATEGY.md) | 6-phase additive migration, no breaking changes |
| 06 | [METADATA_REVIEW.md](06_METADATA_REVIEW.md) | Complete metadata bit layout, DN tree, inline edges, XOR coupling |
| 07 | [COMPRESSION_AND_RESONANCE.md](07_COMPRESSION_AND_RESONANCE.md) | Dimensional sparsity theorem, per-stripe SIMD, holographic probe search |

## Origin

These insights come from reviewing and implementing the HDR fingerprint
engine in `src/fingerprint/rust/`:

- `width_16k/schema.rs` — Schema sidecar with version byte, ANI/NARS/RL metadata
- `width_16k/search.rs` — Schema-filtered search, bloom-accelerated, RL-guided
- `width_16k/xor_bubble.rs` — Delta compression, write cache, ConcurrentWriteCache
- `width_16k/compat.rs` — 10K↔16K conversion, batch migration
- `navigator.rs` — Cypher procedures, DN addressing, GNN/GraphBLAS
- `ARCHITECTURAL_INSIGHTS.md` — The "why it clicks" document

**Test results**: 301 tests passing (259 original + 42 width_32k), all functionality verified.

## How to Use

1. Copy this `docs/redisgraph/` directory into the ladybug-rs repository
2. Open a Claude Code session on ladybug-rs
3. Paste the prompt from `00_PROMPT_FOR_LADYBUG_SESSION.md`
4. Follow the migration phases in `05_MIGRATION_STRATEGY.md`

Or: point the ladybug-rs session at this repo and have it read the docs directly.
