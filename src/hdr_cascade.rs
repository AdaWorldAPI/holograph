//! HDR Cascade Search Engine
//!
//! Hierarchical Distance Resolution with Mexican hat discrimination,
//! Belichtungsmesser adaptive thresholds, and Voyager deep field search.
//!
//! # The Cascade Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Query                                                          │
//! │    │                                                            │
//! │    ▼                                                            │
//! │  ┌─────────────────────┐                                        │
//! │  │ Level 0: Belichtung │  7-point sample (~14 cycles)           │
//! │  │   90% filtered      │  "Is this worth looking at?"           │
//! │  └──────────┬──────────┘                                        │
//! │             │ survivors                                         │
//! │             ▼                                                   │
//! │  ┌─────────────────────┐                                        │
//! │  │ Level 1: 1-bit scan │  Which words differ? (~157 cycles)     │
//! │  │   80% filtered      │  "Where are the differences?"          │
//! │  └──────────┬──────────┘                                        │
//! │             │ survivors                                         │
//! │             ▼                                                   │
//! │  ┌─────────────────────┐                                        │
//! │  │ Level 2: Stacked    │  Per-word popcount with threshold      │
//! │  │   Popcount          │  Early exit if impossible              │
//! │  └──────────┬──────────┘                                        │
//! │             │ candidates                                        │
//! │             ▼                                                   │
//! │  ┌─────────────────────┐                                        │
//! │  │ Level 3: Mexican    │  Discrimination filter                 │
//! │  │   Hat Response      │  Excitation + Inhibition               │
//! │  └──────────┬──────────┘                                        │
//! │             │ results                                           │
//! │             ▼                                                   │
//! │  ┌─────────────────────┐                                        │
//! │  │ Voyager Deep Field  │  Optional: stack weak signals          │
//! │  │   (if no results)   │  Find faint stars in noise             │
//! │  └─────────────────────┘                                        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::bitpack::{BitpackedVector, VECTOR_WORDS, VECTOR_BITS};
use crate::hamming::{
    hamming_distance_scalar, hamming_to_similarity, Belichtung, StackedPopcount,
};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default Mexican hat excitation threshold (~20% different)
const DEFAULT_EXCITE: u32 = 2000;

/// Default Mexican hat inhibition threshold (~50% different)
const DEFAULT_INHIBIT: u32 = 5000;

/// Sample points for Belichtungsmesser (prime-spaced)
const METER_POINTS: [usize; 7] = [0, 23, 47, 78, 101, 131, 155];

// ============================================================================
// MEXICAN HAT RESPONSE
// ============================================================================

/// Mexican hat (difference of Gaussians) response curve
///
/// ```text
///   response
///      │
///   1.0┤    ╭───╮
///      │   ╱     ╲
///   0.0┤──╱───────╲──────────
///      │ ╱         ╲
///  -0.5┤╱           ╲___╱
///      └────────────────────→ distance
///         excite  inhibit
/// ```
///
/// - **Center (excitation)**: Strong match, high positive response
/// - **Ring (inhibition)**: Too similar, suppress (negative response)
/// - **Far**: Irrelevant (zero response)
#[derive(Debug, Clone, Copy)]
pub struct MexicanHat {
    /// Excitation threshold (center of receptive field)
    pub excite: u32,
    /// Inhibition threshold (edge of surround)
    pub inhibit: u32,
    /// Inhibition strength (0.0 to 1.0)
    pub inhibit_strength: f32,
}

impl Default for MexicanHat {
    fn default() -> Self {
        Self {
            excite: DEFAULT_EXCITE,
            inhibit: DEFAULT_INHIBIT,
            inhibit_strength: 0.5,
        }
    }
}

impl MexicanHat {
    /// Create with custom thresholds
    pub fn new(excite: u32, inhibit: u32) -> Self {
        Self {
            excite,
            inhibit,
            inhibit_strength: 0.5,
        }
    }

    /// Create from similarity thresholds (0.0 to 1.0)
    pub fn from_similarity(excite_sim: f32, inhibit_sim: f32) -> Self {
        Self {
            excite: ((1.0 - excite_sim) * VECTOR_BITS as f32) as u32,
            inhibit: ((1.0 - inhibit_sim) * VECTOR_BITS as f32) as u32,
            inhibit_strength: 0.5,
        }
    }

    /// Set inhibition strength
    pub fn with_inhibition(mut self, strength: f32) -> Self {
        self.inhibit_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Compute response for a given distance
    #[inline]
    pub fn response(&self, distance: u32) -> f32 {
        if distance < self.excite {
            // Excitation: linear ramp from 1.0 to 0.0
            1.0 - (distance as f32 / self.excite as f32)
        } else if distance < self.inhibit {
            // Inhibition: negative response
            let t = (distance - self.excite) as f32 / (self.inhibit - self.excite) as f32;
            -self.inhibit_strength * (1.0 - t)
        } else {
            // Beyond range
            0.0
        }
    }

    /// Check if distance is in excitation zone
    #[inline]
    pub fn is_excited(&self, distance: u32) -> bool {
        distance < self.excite
    }

    /// Check if distance is in inhibition zone
    #[inline]
    pub fn is_inhibited(&self, distance: u32) -> bool {
        distance >= self.excite && distance < self.inhibit
    }
}

// ============================================================================
// QUALITY TRACKER (Rubicon Control)
// ============================================================================

/// Tracks search quality for adaptive threshold adjustment
#[derive(Clone, Debug)]
pub struct QualityTracker {
    /// Exponential moving average of quality
    pub ema: f32,
    /// SD trajectory history (last 4 readings)
    pub sd_history: [u8; 4],
    /// History index
    pub sd_idx: usize,
    /// Current dynamic threshold
    pub threshold: u16,
    /// Base threshold (learned sweet spot)
    pub base_threshold: u16,
}

impl Default for QualityTracker {
    fn default() -> Self {
        Self::new(2000)
    }
}

impl QualityTracker {
    /// Create with initial threshold
    pub fn new(base_threshold: u16) -> Self {
        Self {
            ema: 0.5,
            sd_history: [50; 4],
            sd_idx: 0,
            threshold: base_threshold,
            base_threshold,
        }
    }

    /// Record a meter reading and update trajectory
    pub fn record_meter(&mut self, _mean: u8, sd: u8) {
        self.sd_history[self.sd_idx % 4] = sd;
        self.sd_idx += 1;
    }

    /// Calculate optimal threshold from meter reading
    pub fn calculate_sweet_spot(&self, mean: u8, sd: u8) -> u16 {
        let base = match mean {
            0..=1 => self.base_threshold / 2,
            2..=3 => (self.base_threshold * 3) / 4,
            4..=5 => self.base_threshold,
            6..=7 => (self.base_threshold * 3) / 2,
            _ => self.base_threshold,
        };

        let sd_factor = 1.0 + (sd as f32 / 150.0);
        (base as f32 * sd_factor) as u16
    }

    /// Infer trajectory and pre-adjust threshold
    pub fn infer_trajectory(&mut self) -> i16 {
        if self.sd_idx < 4 {
            return 0;
        }

        let h = &self.sd_history;
        let slope = (h[3] as i16 - h[0] as i16) / 3;

        if slope > 10 {
            self.threshold = (self.threshold as i32 + slope as i32 * 20).min(5000) as u16;
        } else if slope < -10 {
            self.threshold = (self.threshold as i32 + slope as i32 * 15).max(500) as u16;
        }

        slope
    }

    /// Update quality EMA after batch
    pub fn update_quality(&mut self, batch_quality: f32) {
        self.ema = 0.85 * self.ema + 0.15 * batch_quality;
    }

    /// Check if we should retreat from Rubicon
    pub fn should_retreat(&self, current_quality: f32) -> bool {
        current_quality < self.ema * 0.6
    }
}

// ============================================================================
// SEARCH RESULT
// ============================================================================

/// Comprehensive search result with multiple representations
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index in corpus
    pub index: usize,
    /// Hamming distance (0 to ~10K)
    pub distance: u32,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Mexican hat response (-0.5 to 1.0)
    pub response: f32,
}

impl SearchResult {
    /// Create from distance
    pub fn new(index: usize, distance: u32) -> Self {
        Self {
            index,
            distance,
            similarity: hamming_to_similarity(distance),
            response: 0.0,
        }
    }

    /// Create with Mexican hat response
    pub fn with_hat(index: usize, distance: u32, hat: &MexicanHat) -> Self {
        Self {
            index,
            distance,
            similarity: hamming_to_similarity(distance),
            response: hat.response(distance),
        }
    }
}

// ============================================================================
// HDR CASCADE INDEX
// ============================================================================

/// Hierarchical Distance Resolution index for fast similarity search
pub struct HdrCascade {
    /// Stored fingerprints
    fingerprints: Vec<BitpackedVector>,
    /// Mexican hat parameters
    hat: MexicanHat,
    /// Quality tracker for adaptive search
    tracker: QualityTracker,
    /// Cascade thresholds
    threshold_l0: f32,   // Belichtung: max fraction
    threshold_l1: u32,   // 1-bit: max differing words
    threshold_l2: u32,   // Stacked: max distance
    /// Batch size for Rubicon processing
    batch_size: usize,
}

impl Default for HdrCascade {
    fn default() -> Self {
        Self::new()
    }
}

impl HdrCascade {
    /// Create empty index
    pub fn new() -> Self {
        Self {
            fingerprints: Vec::new(),
            hat: MexicanHat::default(),
            tracker: QualityTracker::default(),
            threshold_l0: 0.8,
            threshold_l1: 130,
            threshold_l2: 3000,
            batch_size: 64,
        }
    }

    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            fingerprints: Vec::with_capacity(n),
            ..Self::new()
        }
    }

    /// Set cascade thresholds
    pub fn set_thresholds(&mut self, l0: f32, l1: u32, l2: u32) {
        self.threshold_l0 = l0;
        self.threshold_l1 = l1;
        self.threshold_l2 = l2;
    }

    /// Set Mexican hat parameters
    pub fn set_mexican_hat(&mut self, hat: MexicanHat) {
        self.hat = hat;
    }

    /// Add a fingerprint to the index
    pub fn add(&mut self, fp: BitpackedVector) {
        self.fingerprints.push(fp);
    }

    /// Add multiple fingerprints
    pub fn add_batch(&mut self, fps: &[BitpackedVector]) {
        self.fingerprints.extend_from_slice(fps);
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Get fingerprint by index
    pub fn get(&self, index: usize) -> Option<&BitpackedVector> {
        self.fingerprints.get(index)
    }

    // ========================================================================
    // SEARCH METHODS
    // ========================================================================

    /// Full HDR cascade search
    pub fn search(&self, query: &BitpackedVector, k: usize) -> Vec<SearchResult> {
        let mut candidates = Vec::with_capacity(k * 2);

        for (idx, fp) in self.fingerprints.iter().enumerate() {
            // Level 0: Belichtungsmesser
            let meter = Belichtung::meter(query, fp);
            if meter.definitely_far(self.threshold_l0) {
                continue;
            }

            // Level 1: 1-bit scan (how many words differ?)
            let differing_words = count_differing_words(query, fp);
            if differing_words > self.threshold_l1 {
                continue;
            }

            // Level 2: Stacked popcount with threshold
            if let Some(stacked) = StackedPopcount::compute_with_threshold(
                query, fp, self.threshold_l2,
            ) {
                candidates.push(SearchResult::with_hat(idx, stacked.total, &self.hat));
            }
        }

        // Sort by distance and take top k
        candidates.sort_by_key(|r| r.distance);
        candidates.truncate(k);
        candidates
    }

    /// Search with adaptive Rubicon thresholds
    pub fn search_adaptive(&mut self, query: &BitpackedVector, k: usize) -> Vec<SearchResult> {
        let mut results = Vec::with_capacity(k * 2);
        let mut batch_start = 0;

        while batch_start < self.fingerprints.len() && results.len() < k * 10 {
            // Phase 1: Belichtungsmesser on first item of batch
            let meter = Belichtung::meter(query, &self.fingerprints[batch_start]);
            self.tracker.record_meter(meter.mean, meter.sd_100);

            // Phase 2: Calculate dynamic threshold
            let dynamic_threshold = self.tracker.calculate_sweet_spot(meter.mean, meter.sd_100);

            // Phase 3: Process batch
            let batch_end = (batch_start + self.batch_size).min(self.fingerprints.len());
            let mut batch_quality_sum = 0.0f32;
            let mut batch_count = 0u32;

            for i in batch_start..batch_end {
                let dist = hamming_distance_scalar(query, &self.fingerprints[i]);

                if dist <= dynamic_threshold as u32 {
                    results.push(SearchResult::with_hat(i, dist, &self.hat));
                    batch_quality_sum += 1.0 - (dist as f32 / 10000.0);
                    batch_count += 1;
                }

                // Phase 4: Quality monitoring
                if batch_count >= 8 && i > batch_start + 16 {
                    let current_q = batch_quality_sum / batch_count as f32;
                    if self.tracker.should_retreat(current_q) {
                        break;
                    }
                }
            }

            // Update tracker
            if batch_count > 0 {
                let batch_q = batch_quality_sum / batch_count as f32;
                self.tracker.update_quality(batch_q);
            }

            // Phase 5: Infer trajectory for next batch
            self.tracker.infer_trajectory();

            batch_start = batch_end;
        }

        results.sort_by_key(|r| r.distance);
        results.truncate(k);
        results
    }

    /// Search with Mexican hat discrimination
    pub fn search_discriminate(&self, query: &BitpackedVector, k: usize) -> Vec<SearchResult> {
        let mut results = Vec::new();

        for (idx, fp) in self.fingerprints.iter().enumerate() {
            let dist = hamming_distance_scalar(query, fp);
            let response = self.hat.response(dist);

            // Only keep positive responses (excited, not inhibited)
            if response > 0.0 {
                results.push(SearchResult::with_hat(idx, dist, &self.hat));
            }
        }

        // Sort by response (highest first)
        results.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        results.truncate(k);
        results
    }

    /// Voyager deep field search: find faint signals in noise
    pub fn voyager_deep_field(
        &self,
        query: &BitpackedVector,
        radius: u32,
        stack_size: usize,
    ) -> Option<VoyagerResult> {
        if self.fingerprints.is_empty() {
            return None;
        }

        // Gather weak candidates at the edge of detection
        let mut weak_candidates = Vec::with_capacity(stack_size);
        let radius_min = radius.saturating_sub(500);
        let radius_max = radius + 500;

        for fp in &self.fingerprints {
            let dist = hamming_distance_scalar(query, fp);
            if dist >= radius_min && dist <= radius_max {
                weak_candidates.push(fp.clone());
                if weak_candidates.len() >= stack_size {
                    break;
                }
            }
        }

        if weak_candidates.len() < 3 {
            return None;
        }

        // Stack exposures using superposition
        let star = superposition_clean(query, &weak_candidates)?;

        // Measure the cleaned signal
        let cleaned_dist = hamming_distance_scalar(query, &star);
        let signal_strength = 1.0 - (cleaned_dist as f32 / 10000.0);
        let noise_reduction = radius as f32 / cleaned_dist.max(1) as f32;

        // Did we find a star? (signal improved by at least 1.5x)
        if noise_reduction > 1.5 {
            Some(VoyagerResult {
                star,
                original_radius: radius,
                cleaned_distance: cleaned_dist,
                signal_strength,
                noise_reduction,
                stack_count: weak_candidates.len(),
            })
        } else {
            None
        }
    }

    /// Get current quality EMA
    pub fn quality(&self) -> f32 {
        self.tracker.ema
    }

    /// Get current dynamic threshold
    pub fn threshold(&self) -> u16 {
        self.tracker.threshold
    }
}

// ============================================================================
// VOYAGER DEEP FIELD
// ============================================================================

/// Result from Voyager deep field search
#[derive(Debug, Clone)]
pub struct VoyagerResult {
    /// The cleaned "star" fingerprint
    pub star: BitpackedVector,
    /// Original search radius
    pub original_radius: u32,
    /// Distance after cleaning
    pub cleaned_distance: u32,
    /// Signal strength (0.0-1.0)
    pub signal_strength: f32,
    /// Noise reduction factor
    pub noise_reduction: f32,
    /// Number of exposures stacked
    pub stack_count: usize,
}

/// Orthogonal superposition noise cleaning
///
/// Like stacking astronomical photos:
/// - Noise is random → cancels in majority vote
/// - Signal is consistent → survives the vote
pub fn superposition_clean(
    query: &BitpackedVector,
    weak_candidates: &[BitpackedVector],
) -> Option<BitpackedVector> {
    if weak_candidates.len() < 3 {
        return None;
    }

    let n = weak_candidates.len();
    let threshold = n / 2;

    // XOR each candidate with query to get the "difference signal"
    let deltas: Vec<_> = weak_candidates
        .iter()
        .map(|c| query.xor(c))
        .collect();

    // Componentwise majority vote (VSA bundle)
    let mut cleaned_delta = BitpackedVector::zero();

    for word_idx in 0..VECTOR_WORDS {
        let mut result_word = 0u64;

        for bit in 0..64 {
            let mask = 1u64 << bit;

            // Count votes for this bit
            let votes: usize = deltas
                .iter()
                .filter(|d| d.words()[word_idx] & mask != 0)
                .count();

            // Majority vote
            if votes > threshold {
                result_word |= mask;
            }
        }

        cleaned_delta.words_mut()[word_idx] = result_word;
    }

    // Apply cleaned delta back to query to get the "star"
    let star = query.xor(&cleaned_delta);
    Some(star)
}

// ============================================================================
// SIGNAL CLASSIFICATION
// ============================================================================

/// Signal classification based on Belichtungsmesser
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalClass {
    /// Clean hit - high confidence match
    Strong,
    /// Normal search territory
    Moderate,
    /// Noisy but potentially stackable for deep field
    WeakButStackable,
    /// Pure noise - reject
    Noise,
}

/// Classify a comparison result for routing
pub fn classify_signal(mean: u8, sd: u8, distance: u32) -> SignalClass {
    match (mean, sd, distance) {
        (0..=2, 0..=30, 0..=2000) => SignalClass::Strong,
        (0..=4, 0..=60, 0..=4000) => SignalClass::Moderate,
        (_, 50.., 4000..=8000) => SignalClass::WeakButStackable,
        _ => SignalClass::Noise,
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Count how many 64-bit words differ at all
#[inline]
fn count_differing_words(a: &BitpackedVector, b: &BitpackedVector) -> u32 {
    let a_words = a.words();
    let b_words = b.words();
    let mut count = 0u32;

    for i in 0..VECTOR_WORDS {
        if a_words[i] ^ b_words[i] != 0 {
            count += 1;
        }
    }

    count
}

// ============================================================================
// UNIFIED SEARCH API (The Alien Magic Interface)
// ============================================================================

/// Unified search engine that looks like float vector search
///
/// This is THE alien magic API. User sees similarity scores.
/// Underneath it's HDR cascade + Mexican hat + rolling σ.
pub struct AlienSearch {
    /// HDR cascade index
    cascade: HdrCascade,
    /// Rolling window for coherence detection
    window: RollingWindow,
}

impl Default for AlienSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl AlienSearch {
    /// Create new search engine
    pub fn new() -> Self {
        Self {
            cascade: HdrCascade::new(),
            window: RollingWindow::new(100),
        }
    }

    /// Create with capacity
    pub fn with_capacity(n: usize) -> Self {
        Self {
            cascade: HdrCascade::with_capacity(n),
            window: RollingWindow::new(100),
        }
    }

    /// Set Mexican hat parameters
    pub fn set_mexican_hat(&mut self, excite: u32, inhibit: u32) {
        self.cascade.set_mexican_hat(MexicanHat::new(excite, inhibit));
    }

    /// Add fingerprint to index
    pub fn add(&mut self, fp: BitpackedVector) {
        self.cascade.add(fp);
    }

    /// Add multiple fingerprints
    pub fn add_batch(&mut self, fps: &[BitpackedVector]) {
        self.cascade.add_batch(fps);
    }

    /// Number of indexed fingerprints
    pub fn len(&self) -> usize {
        self.cascade.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.cascade.is_empty()
    }

    /// Search - returns results that look like float vector search
    pub fn search(&mut self, query: &BitpackedVector, k: usize) -> Vec<SearchResult> {
        let results = self.cascade.search(query, k);

        // Update rolling window with distances
        for r in &results {
            self.window.push(r.distance);
        }

        results
    }

    /// Search returning only similarity scores (float-like API)
    pub fn search_similarity(
        &mut self,
        query: &BitpackedVector,
        k: usize,
    ) -> Vec<(usize, f32)> {
        self.search(query, k)
            .into_iter()
            .map(|r| (r.index, r.similarity))
            .collect()
    }

    /// Search with Mexican hat discrimination
    pub fn search_discriminate(
        &mut self,
        query: &BitpackedVector,
        k: usize,
    ) -> Vec<(usize, f32)> {
        self.cascade
            .search_discriminate(query, k)
            .into_iter()
            .filter(|r| r.response > 0.0)
            .map(|r| (r.index, r.response))
            .collect()
    }

    /// Get coherence stats for recent searches
    pub fn coherence(&self) -> (f32, f32) {
        self.window.stats()
    }

    /// Is recent search pattern coherent?
    pub fn is_coherent(&self) -> bool {
        self.window.is_coherent(0.3)
    }
}

// ============================================================================
// ROLLING WINDOW STATISTICS
// ============================================================================

/// Rolling window statistics for coherence detection
pub struct RollingWindow {
    size: usize,
    distances: Vec<u32>,
    pos: usize,
    sum: u64,
    sum_sq: u64,
    count: usize,
}

impl RollingWindow {
    /// Create a new rolling window
    pub fn new(size: usize) -> Self {
        Self {
            size,
            distances: vec![0; size],
            pos: 0,
            sum: 0,
            sum_sq: 0,
            count: 0,
        }
    }

    /// Add a distance to the window
    pub fn push(&mut self, distance: u32) {
        let d = distance as u64;

        if self.count >= self.size {
            let old = self.distances[self.pos] as u64;
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }

        self.distances[self.pos] = distance;
        self.sum += d;
        self.sum_sq += d * d;

        self.pos = (self.pos + 1) % self.size;
    }

    /// Get mean distance
    #[inline]
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum as f32 / self.count as f32
    }

    /// Get standard deviation
    #[inline]
    pub fn stddev(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f32;
        let mean = self.sum as f32 / n;
        let variance = (self.sum_sq as f32 / n) - (mean * mean);
        variance.max(0.0).sqrt()
    }

    /// Get mean and stddev together
    #[inline]
    pub fn stats(&self) -> (f32, f32) {
        (self.mean(), self.stddev())
    }

    /// Get coefficient of variation (σ/μ)
    #[inline]
    pub fn cv(&self) -> f32 {
        let μ = self.mean();
        if μ < 1.0 {
            return 0.0;
        }
        self.stddev() / μ
    }

    /// Is the window showing coherent pattern?
    pub fn is_coherent(&self, threshold: f32) -> bool {
        self.cv() < threshold
    }

    /// Clear the window
    pub fn clear(&mut self) {
        self.distances.fill(0);
        self.pos = 0;
        self.sum = 0;
        self.sum_sq = 0;
        self.count = 0;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_fp(seed: u64) -> BitpackedVector {
        BitpackedVector::random(seed)
    }

    #[test]
    fn test_mexican_hat() {
        let hat = MexicanHat::new(2000, 5000);

        // Center: strong positive
        assert!(hat.response(0) > 0.9);
        assert!(hat.response(1000) > 0.0);

        // Ring: negative
        assert!(hat.response(3000) < 0.0);

        // Far: zero
        assert_eq!(hat.response(6000), 0.0);
    }

    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(5);

        for d in [100, 110, 105, 108, 103] {
            window.push(d);
        }

        let (μ, σ) = window.stats();
        assert!((μ - 105.2).abs() < 1.0);
        assert!(σ > 0.0 && σ < 10.0);
    }

    #[test]
    fn test_hdr_cascade() {
        let mut cascade = HdrCascade::with_capacity(100);

        let fps: Vec<_> = (0..100).map(|i| random_fp(i as u64 + 100)).collect();
        for fp in &fps {
            cascade.add(fp.clone());
        }

        let results = cascade.search(&fps[42], 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 42);
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_alien_search_api() {
        let mut search = AlienSearch::with_capacity(100);

        let fps: Vec<_> = (0..100).map(|i| random_fp(i as u64 + 100)).collect();
        search.add_batch(&fps);

        let results = search.search_similarity(&fps[0], 5);
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_signal_classification() {
        assert_eq!(classify_signal(1, 20, 1000), SignalClass::Strong);
        assert_eq!(classify_signal(3, 50, 3000), SignalClass::Moderate);
        assert_eq!(classify_signal(5, 70, 6000), SignalClass::WeakButStackable);
        assert_eq!(classify_signal(7, 100, 9000), SignalClass::Noise);
    }

    #[test]
    fn test_quality_tracker() {
        let mut tracker = QualityTracker::new(2000);

        // Simulate decreasing SD
        tracker.record_meter(3, 80);
        tracker.record_meter(3, 60);
        tracker.record_meter(3, 45);
        tracker.record_meter(3, 30);

        let slope = tracker.infer_trajectory();
        assert!(slope < 0);
        assert!(tracker.threshold < 2000);
    }
}
