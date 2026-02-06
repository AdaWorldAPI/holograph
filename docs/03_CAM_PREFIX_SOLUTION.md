# The 4096 CAM Is a Transport Protocol — Not Storage

> The 4096 CAM (Content-Addressable Methods) is ladybug-rs's most innovative
> idea. The confusion was treating it as a storage problem. It's not. The
> 4096 is a transport protocol: an opcode that reaches a class and method.
> The commandlets belong in classes and methods. Only GEL (Graph Execution
> Language) — the ability to compile programs into graph execution sequences
> — stays in the CAM as a first-class concern.

---

## The Clarification

The CAM dictionary defines 4096 operations across 16 categories:

```
0x000-0x0FF: LanceDB Core
0x100-0x1FF: SQL
0x200-0x2FF: Cypher
0x300-0x3FF: Hamming/VSA
0x400-0x4FF: NARS
0x500-0x5FF: Search
0x600-0x6FF: Crystal/Temporal
0x700-0x7FF: NSM Semantic
0x800-0x8FF: ACT-R Cognitive
0x900-0x9FF: RL/Decision
0xA00-0xAFF: Causality
0xB00-0xBFF: Qualia/Affect
0xC00-0xCFF: Rung/Abstraction
0xD00-0xDFF: Meta/Reflection
0xE00-0xEFF: Learning
0xF00-0xFFF: User-Defined/Extension
```

The original design struggled to fit 4096 entries into the surplus bits
between 10,000 and 16,384. This was the wrong question entirely.

**The commandlets are not a storage issue.** They belong in classes and
methods — `impl TruthValue`, `impl QTable`, `impl Fingerprint16K`,
`impl CogGraph`. The 4096 CAM transport protocol reaches those methods
the same way HTTP reaches REST endpoints.

---

## What Stays in the CAM: GEL Only

**GEL (Graph Execution Language)** is the compiler that turns programs into
sequences of graph operations. GEL is inherently a dispatch concern:

```
GEL program: "find similar concepts with high confidence, then propagate activation"
      ↓ compiles to
Step 1: CAM 0x501 SEARCH.SCHEMA (args: query_fp, predicates: {nars_confidence > 0.7})
Step 2: CAM 0x410 NARS.DEDUCTION (args: result[0], result[1], query)
Step 3: CAM 0x900 RL.BEST_ACTION (args: state_fp)
Step 4: CAM 0x302 HAMMING.BIND (args: action, state)
      ↓ executes as
4 method calls on 256-word fingerprints, each reading/writing metadata in-place
```

GEL stays in the CAM because it IS routing — it compiles a program into
a sequence of CAM opcodes that reach the right methods in the right order.

Everything else — the NARS inference rules, the RL Q-update math, the
Hamming distance computation — those are **methods on types**, not CAM
operations. They get called BY the CAM, they don't live IN the CAM.

---

## What Changes: Commandlets → Classes and Methods

### Before (cam_ops.rs today, ~4,661 lines):

```rust
fn execute(&self, op: u16, args: Vec<Fingerprint>) -> OpResult {
    match op {
        0x410 => {
            // ROUTING + IMPLEMENTATION mixed in one match arm
            if args.len() < 3 {
                return OpResult::Error("Deduction requires M, P, S".to_string());
            }
            let conclusion = args[2].bind(&args[1]);  // Implementation inline
            OpResult::One(conclusion)
        }
        0x430 => {
            // More implementation inline
            let revised = bundle_fingerprints(&[args[0].clone(), args[1].clone()]);
            OpResult::One(revised)
        }
        // ... 4000+ lines of this
    }
}
```

### After: CAM is pure routing (~200 lines)

```rust
// cam_ops.rs: ONLY routing, no implementation
fn execute(&self, op: u16, bs: &BindSpace16K, args: &[Addr]) -> CamResult {
    let category = (op >> 8) as u8;
    let operation = (op & 0xFF) as u8;
    match category {
        0x03 => Fingerprint16K::cam_dispatch(operation, bs, args),
        0x04 => NarsTruth::cam_dispatch(operation, bs, args),
        0x05 => SchemaSearch::cam_dispatch(operation, bs, args),
        0x09 => RlPolicy::cam_dispatch(operation, bs, args),
        0x0B => QualiaField::cam_dispatch(operation, bs, args),
        0x0C => RungLevel::cam_dispatch(operation, bs, args),
        0x0E => GelCompiler::cam_dispatch(operation, bs, args),
        _    => CamResult::Error(format!("Unknown category: 0x{:02X}", category)),
    }
}
```

### Implementation lives in `impl` blocks (separate files)

```rust
// src/nars/truth_16k.rs
impl NarsTruth {
    /// Read truth from word 210 of a 256-word fingerprint
    pub fn from_word(w: u64) -> Self { ... }
    pub fn to_word(&self) -> u64 { ... }

    pub fn deduction(fp_m: &[u64; 256], fp_p: &[u64; 256], fp_s: &[u64; 256])
        -> [u64; 256]
    {
        let truth_m = Self::from_word(fp_m[210]);
        let truth_p = Self::from_word(fp_p[210]);
        let f = truth_m.frequency * truth_p.frequency;
        let c = f * truth_m.confidence * truth_p.confidence;
        let mut result = fp_s.clone();  // Start from subject
        // Semantic: result = S ⊗ P
        for i in 0..208 {
            result[i] = fp_s[i] ^ fp_p[i];
        }
        // Metadata: write computed truth to word 210
        result[210] = Self { frequency: f, confidence: c, ..Default::default() }.to_word();
        result
    }

    /// CAM dispatch for category 0x04
    pub fn cam_dispatch(op: u8, bs: &BindSpace16K, args: &[Addr]) -> CamResult {
        match op {
            0x10 => { // DEDUCTION
                let (m, p, s) = (bs.read(args[0]), bs.read(args[1]), bs.read(args[2]));
                CamResult::Fingerprint(Self::deduction(&m, &p, &s))
            }
            0x30 => { // REVISION
                let (a, b) = (bs.read(args[0]), bs.read(args[1]));
                CamResult::Fingerprint(Self::revision(&a, &b))
            }
            _ => CamResult::Error(format!("Unknown NARS op: 0x{:02X}", op)),
        }
    }
}
```

```rust
// src/graph/gel.rs — GEL IS the CAM-native component
impl GelCompiler {
    /// Compile a program description into a GEL execution plan
    pub fn compile(program: &str) -> GelPlan {
        // Parse program → sequence of CAM opcodes with argument bindings
        // This IS the CAM's native function — compiling programs into
        // graph execution sequences
    }

    /// Execute a compiled GEL plan
    pub fn execute(plan: &GelPlan, bs: &mut BindSpace16K) -> Vec<CamResult> {
        plan.steps.iter().map(|step| {
            // Each step is a CAM opcode + addresses
            // GEL manages: sequencing, branching, loops, error handling
            cam_execute(step.op, bs, &step.args)
        }).collect()
    }

    pub fn cam_dispatch(op: u8, bs: &BindSpace16K, args: &[Addr]) -> CamResult {
        match op {
            0x00 => { /* GEL.COMPILE */ }
            0x01 => { /* GEL.EXECUTE */ }
            0x02 => { /* GEL.STEP */ }
            0x10 => { /* GEL.BRANCH_IF */ }
            0x11 => { /* GEL.LOOP */ }
            0x20 => { /* GEL.BIND_RESULT */ }
            _ => CamResult::Error(format!("Unknown GEL op: 0x{:02X}", op)),
        }
    }
}
```

---

## The CAM Operates ON Metadata, Not WITH Metadata

At 256 words, each method called by the CAM reads and writes metadata
directly in the fingerprint's word array:

| CAM Category | Method Target | Reads Words | Writes Words |
|-------------|---------------|-------------|--------------|
| 0x03 Hamming | `Fingerprint16K` | 0-207 (semantic) | 0-207 |
| 0x04 NARS | `NarsTruth` | 210 (truth) | 210 |
| 0x05 Search | `SchemaSearch` | 208-255 (predicates) | — (read-only) |
| 0x09 RL | `RlPolicy` | 224-231 (Q-values) | 224-231 |
| 0x0B Qualia | `QualiaField` | 212-213 | 212-213 |
| 0x0C Rung | `RungLevel` | 216 bits[24-31] | 216 bits[24-31] |
| 0x0E GEL | `GelCompiler` | 214 (exec state) | 214 |

The method receives the 256-word fingerprint. It reads the words it needs.
It writes the words it changes. It returns the fingerprint. The CAM never
touches the fingerprint — it just routes to the method that does.

**This is why the commandlets don't belong in the CAM.** `NARS.DEDUCTION`
is not a CAM operation — it's `NarsTruth::deduction()`. The CAM routes
opcode 0x410 to that method. The method reads word 210, does arithmetic,
writes word 210. The CAM is the phone system. The methods are the people
who answer.

---

## What cam_ops.rs Becomes

| Before | After |
|--------|-------|
| 4,661 lines | ~200 lines routing + ~60 lines GEL compiler |
| 16 match arm blocks with inline implementations | 16 one-line dispatches to `::cam_dispatch()` |
| `OpResult` enum with 8 variants | `CamResult` enum with 3 variants (Fingerprint, Scalar, Error) |
| Operations compute results inline | Methods on types compute results |
| Fingerprints passed by value | Addresses passed, BindSpace provides fingerprints |
| Stubs for unimplemented operations | No stubs needed — method doesn't exist yet = no route |

The remaining ~4,400 lines move to where they belong:

- `src/nars/truth_16k.rs` — NARS inference on word 210
- `src/rl/policy_16k.rs` — RL operations on words 224-231
- `src/search/schema_search.rs` — Schema-predicate search on words 208-255
- `src/cognitive/qualia_16k.rs` — Qualia operations on words 212-213
- `src/cognitive/rung_16k.rs` — Rung operations on word 216
- `src/graph/gel.rs` — GEL compilation and execution (stays CAM-native)
- `src/graph/edges_16k.rs` — Inline edge operations on words 219-243

---

## The Key Insight

The CAM prefix was never a fitting problem. It was a separation-of-concerns
problem. The 4096 opcodes are an addressing scheme — a transport protocol
that reaches methods. The methods are implementations on types. GEL is the
one CAM-native concept: it compiles programs INTO CAM opcode sequences.

Remove the commandlet implementations from cam_ops.rs. Move them to `impl`
blocks. Keep the routing. Keep GEL. The 4,661-line file becomes 260 lines
and every operation gains access to the full 256-word fingerprint with all
its metadata.
