# Race Condition Fix Patterns from RedisGraph

> Ladybug-rs documents 9 race conditions in STORAGE_CONTRACTS.md. All 9
> follow the same pattern: a lock is released between a check and a commit.
> The RedisGraph HDR engine solved the equivalent problem with
> `ConcurrentWriteCache`. Here are the fix templates.

---

## The Universal Pattern

Every race condition in ladybug-rs has this shape:

```rust
// BROKEN: check-then-act with lock gap
let data = self.lock.read();    // Read lock
let valid = check(&data);       // Check under read lock
drop(data);                     // RELEASE LOCK
// ← GAP: another thread mutates here
let mut data = self.lock.write(); // Write lock
commit(&mut data);              // Act based on stale check
```

The fix is always the same: **hold the lock across check and commit**.

```rust
// FIXED: check-and-act under single lock
let mut data = self.lock.write(); // Write lock
let valid = check(&data);         // Check under write lock
if valid {
    commit(&mut data);            // Act under same lock
}
// Lock released here, after both check and commit
```

---

## Fix 1: WAL Write-Behind → Write-Ahead

**Location**: `src/storage/hardening.rs:WriteAheadLog`
**Severity**: CRITICAL

```rust
// CURRENT (write-behind):
self.bind_space.write_at(addr, fp);  // Memory first
self.wal.append(entry)?;              // Disk second - CRASH = LOST

// FIXED (write-ahead):
self.wal.append(entry)?;              // Disk first
self.wal.sync()?;                     // fsync (crucial!)
self.bind_space.write_at(addr, fp);  // Memory second

// Or with the ConcurrentWriteCache pattern from RedisGraph:
// 1. Append to WAL on disk (durable)
// 2. Record XOR delta in ConcurrentWriteCache (in-memory)
// 3. Reads go through cache (applies delta to base data)
// 4. On flush: batch-apply deltas to BindSpace, truncate WAL
```

The `ConcurrentWriteCache` approach is superior because:
- Writes to WAL are sequential (fast)
- BindSpace is never mutated during normal operations (zero-copy reads work)
- Flush is batched and amortized
- Crash recovery: replay WAL into fresh cache

### Implementation (from RedisGraph xor_bubble.rs):

```rust
pub struct ConcurrentWriteCache {
    inner: RwLock<XorWriteCache>,
}

impl ConcurrentWriteCache {
    /// Read: applies cached delta on-the-fly. Read lock (concurrent).
    pub fn read_through(&self, id: u64, base_words: &[u64]) -> ConcurrentCacheRead {
        let cache = self.inner.read().unwrap();
        match cache.get(id) {
            None => ConcurrentCacheRead::Clean,
            Some(delta) => {
                let mut patched = base_words.to_vec();
                delta.apply(&mut patched);
                ConcurrentCacheRead::Patched(patched)
            }
        }
    }

    /// Write: records delta. Write lock (exclusive).
    pub fn record_delta(&self, id: u64, delta: XorDelta) {
        let mut cache = self.inner.write().unwrap();
        cache.record_delta(id, delta);
    }

    /// Flush: returns all dirty entries, clears cache. Write lock.
    pub fn flush(&self) -> Vec<(u64, XorDelta)> {
        let mut cache = self.inner.write().unwrap();
        cache.flush()
    }
}

// IMPORTANT: ConcurrentCacheRead is OWNED (no lifetime borrowing).
// This avoids the "lock guard lifetime" problem.
pub enum ConcurrentCacheRead {
    Clean,
    Patched(Vec<u64>),
}
```

---

## Fix 2: LruTracker Duplicate Entries

**Location**: `src/storage/hardening.rs:LruTracker`
**Severity**: HIGH

```rust
// BROKEN: two separate locks
fn touch(&self, addr: Addr) {
    let mut times = self.access_times.write();
    times.insert(addr, Instant::now());
    drop(times);  // ← GAP
    let mut order = self.order.write();
    order.push(addr);  // Duplicate if another thread touched same addr
}

// FIXED: single lock, dedup
struct LruTracker {
    inner: RwLock<LruInner>,
}

struct LruInner {
    access_times: HashMap<Addr, Instant>,
    order: VecDeque<Addr>,
}

fn touch(&self, addr: Addr) {
    let mut inner = self.inner.write().unwrap();
    inner.access_times.insert(addr, Instant::now());
    // Remove old position, push to back (no duplicates)
    inner.order.retain(|a| *a != addr);
    inner.order.push_back(addr);
}
```

Or use `parking_lot::RwLock` (already in Cargo.toml) which is non-poisoning
and faster.

---

## Fix 3: WriteBuffer ID Gap

**Location**: `src/storage/resilient.rs:WriteBuffer`
**Severity**: HIGH

```rust
// BROKEN: ID allocated before buffer insertion
fn write(&self, entry: Entry) -> u64 {
    let id = self.next_id.fetch_add(1, Ordering::SeqCst);
    // ← GAP: flusher sees incremented count but entry not yet in buffer
    let mut buffer = self.buffer.write();
    buffer.insert(id, entry);
    id
}

// FIXED: allocate ID under buffer lock
fn write(&self, entry: Entry) -> u64 {
    let mut buffer = self.buffer.write().unwrap();
    let id = self.next_id.fetch_add(1, Ordering::SeqCst);
    buffer.insert(id, entry);
    id
}
```

---

## Fix 4: XorDag Parity TOCTOU

**Location**: `src/storage/xor_dag.rs:commit`
**Severity**: HIGH

```rust
// BROKEN: parity computed after lock release
fn commit(&self, txn: Transaction) -> Result<()> {
    let mut space = self.bind_space.write();
    for (addr, fp) in &txn.writes {
        space.write_at(*addr, fp);
    }
    drop(space);  // ← GAP: parity is now stale
    self.update_parity_blocks(&txn)?;  // Uses stale data if concurrent write
    Ok(())
}

// FIXED: hold lock through parity update
fn commit(&self, txn: Transaction) -> Result<()> {
    let mut space = self.bind_space.write().unwrap();
    for (addr, fp) in &txn.writes {
        space.write_at(*addr, fp);
    }
    // Parity computed under same write lock — no gap
    self.update_parity_blocks_locked(&mut space, &txn)?;
    Ok(())
    // Lock released here
}
```

---

## Fix 5: Temporal Serializable Conflict

**Location**: `src/storage/temporal.rs:check_conflicts`
**Severity**: HIGH

```rust
// BROKEN: conflict check under read lock, commit under separate write lock
fn commit_txn(&self, txn: &Transaction) -> Result<()> {
    let entries = self.entries.read();
    self.check_conflicts(&entries, txn)?;
    drop(entries);  // ← GAP: another txn commits here
    let mut entries = self.entries.write();
    let version = self.versions.advance();
    entries.apply(txn, version);
    Ok(())
}

// FIXED: write lock for entire commit
fn commit_txn(&self, txn: &Transaction) -> Result<()> {
    let mut entries = self.entries.write().unwrap();
    self.check_conflicts(&entries, txn)?;
    let version = self.versions.advance();
    entries.apply(txn, version);
    Ok(())
}
```

---

## The ConcurrentCacheRead Pattern

The most subtle fix in RedisGraph was the `ConcurrentCacheRead` enum.
The naive approach returns a borrowed reference:

```rust
// WON'T COMPILE: lifetime of guard leaks into return value
fn read_through<'a>(&'a self, id: u64, base: &'a [u64]) -> &'a [u64] {
    let cache = self.inner.read().unwrap();
    match cache.get(id) {
        Some(delta) => {
            let mut patched = base.to_vec();
            delta.apply(&mut patched);
            &patched  // ← ERROR: patched is local, can't return reference
        }
        None => base,
    }
}
```

The fix: return an owned enum that either says "use the base directly" or
"here's the patched copy":

```rust
pub enum ConcurrentCacheRead {
    Clean,              // Caller uses base_words directly
    Patched(Vec<u64>),  // Caller uses these owned words
}

impl ConcurrentCacheRead {
    pub fn is_clean(&self) -> bool { matches!(self, Self::Clean) }
    pub fn patched_words(&self) -> Option<&[u64]> {
        match self {
            Self::Patched(w) => Some(w),
            Self::Clean => None,
        }
    }
}
```

This pattern applies to ladybug-rs everywhere a cached read returns data:
- `XorDag::read_with_parity()`
- `TemporalStore::read_at_version()`
- `UnifiedEngine::read()`

---

## Priority Order

1. **WAL write-behind** (CRITICAL) — data loss on crash
2. **XorDag parity TOCTOU** (HIGH) — corruption on recovery
3. **Temporal conflict detection** (HIGH) — lost updates under serializable
4. **LruTracker duplicates** (HIGH) — wrong evictions
5. **WriteBuffer ID gap** (HIGH) — orphaned writes

Fixes 1-5 are all the same pattern: merge two locks into one. Total code
change is ~50 lines per fix.

The remaining 4 race conditions (MEDIUM/LOW) follow the same pattern and
can be fixed the same way. See docs/REWIRING_GUIDE.md in ladybug-rs for
copy-paste ready fixes.
