//! Epiphany Engine: SD Threshold + Centroid Radius Calibration
//!
//! The "sweet spot" emerges when statistical distance thresholds align
//! with geometric centroid radii. This creates natural resonance zones
//! where related concepts cluster and insights emerge.
//!
//! # The Epiphany Zone
//!
//! ```text
//!                     Statistical View (Hamming Distribution)
//!
//!     P(d)
//!      │
//!      │                        ┌──────┐
//!      │                       ╱        ╲
//!      │                      ╱    μ     ╲
//!      │            ┌────────╱      │      ╲────────┐
//!      │           ╱   -2σ  │   -1σ │ +1σ  │  +2σ   ╲
//!      │──────────╱─────────┼───────┼──────┼─────────╲──────
//!      └──────────┴─────────┴───────┴──────┴─────────┴──────→ d
//!                 │         │       │      │         │
//!               NOISE    INHIBIT  EXCITE INHIBIT   NOISE
//!                 │         │   ◄──┬──►   │         │
//!                 │         │      │      │         │
//!                 │         │  EPIPHANY   │         │
//!                 │         │    ZONE     │         │
//!
//!                     Geometric View (Centroid Radii)
//!
//!                          ●──────● centroid
//!                         ╱│╲    ╱
//!                        ╱ │ ╲  ╱
//!                       ╱  │  ╲╱
//!              radius →●───┼───● ← child vectors
//!                       ╲  │  ╱
//!                        ╲ │ ╱
//!                         ╲│╱
//!                          ●
//!
//! When radius ≈ σ, the centroid naturally captures one SD of variance,
//! creating optimal clustering for semantic similarity.
//! ```

use crate::bitpack::{BitpackedVector, VECTOR_BITS};
use crate::hamming::hamming_distance_scalar;
use crate::nntree::{NnTree, NnTreeConfig};
use std::collections::HashMap;

// ============================================================================
// STATISTICAL CONSTANTS FOR 10K-BIT VECTORS
// ============================================================================

/// Expected Hamming distance between random vectors = n/2
pub const EXPECTED_RANDOM_DISTANCE: f64 = VECTOR_BITS as f64 / 2.0;

/// Standard deviation of Hamming distance = sqrt(n/4)
pub const HAMMING_STD_DEV: f64 = 50.0; // sqrt(10000/4) = 50

/// One standard deviation threshold
pub const ONE_SIGMA: u32 = 50;

/// Two standard deviations
pub const TWO_SIGMA: u32 = 100;

/// Three standard deviations (99.7% confidence)
pub const THREE_SIGMA: u32 = 150;

// ============================================================================
// EPIPHANY ZONES
// ============================================================================

/// Zone classification based on distance
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EpiphanyZone {
    /// Perfect match (d < μ - 3σ from random)
    /// Distance < ~4850 for random baseline
    /// But for SIMILAR vectors, this is d < 1σ from zero = 50
    Identity,

    /// Strong resonance (1σ - 2σ from target)
    /// The "aha!" zone where related concepts live
    Epiphany,

    /// Weak resonance (2σ - 3σ)
    /// Tangentially related, worth exploring
    Penumbra,

    /// Statistical noise (> 3σ)
    /// Indistinguishable from random
    Noise,

    /// Anti-correlation (closer to max distance)
    /// Potentially interesting as opposites
    Antipode,
}

impl EpiphanyZone {
    /// Classify a distance into zones
    pub fn classify(distance: u32) -> Self {
        // For similar vectors, distances cluster near 0
        // Zone boundaries based on σ = 50 for 10K bits
        match distance {
            d if d <= ONE_SIGMA => EpiphanyZone::Identity,
            d if d <= TWO_SIGMA => EpiphanyZone::Epiphany,
            d if d <= THREE_SIGMA => EpiphanyZone::Penumbra,
            d if d >= VECTOR_BITS as u32 - THREE_SIGMA as u32 => EpiphanyZone::Antipode,
            _ => EpiphanyZone::Noise,
        }
    }

    /// Get activation multiplier for this zone
    pub fn activation(&self) -> f32 {
        match self {
            EpiphanyZone::Identity => 1.0,
            EpiphanyZone::Epiphany => 0.7,  // Strong but not overwhelming
            EpiphanyZone::Penumbra => 0.3,  // Worth noting
            EpiphanyZone::Noise => 0.0,
            EpiphanyZone::Antipode => -0.5, // Negative correlation interesting
        }
    }

    /// Is this zone worth exploring?
    pub fn is_significant(&self) -> bool {
        !matches!(self, EpiphanyZone::Noise)
    }
}

// ============================================================================
// CENTROID RADIUS CALCULATOR
// ============================================================================

/// Statistics about a centroid and its children
#[derive(Clone, Debug)]
pub struct CentroidStats {
    /// The centroid fingerprint (majority bundle)
    pub centroid: BitpackedVector,
    /// Number of vectors bundled
    pub count: usize,
    /// Mean distance from centroid to children
    pub mean_radius: f32,
    /// Standard deviation of distances
    pub radius_std: f32,
    /// Maximum distance (worst child)
    pub max_radius: u32,
    /// Minimum distance (best child)
    pub min_radius: u32,
    /// Ratio of radius to expected σ
    pub sigma_ratio: f32,
}

impl CentroidStats {
    /// Compute statistics for a set of vectors
    pub fn compute(vectors: &[&BitpackedVector]) -> Self {
        if vectors.is_empty() {
            return Self {
                centroid: BitpackedVector::zero(),
                count: 0,
                mean_radius: 0.0,
                radius_std: 0.0,
                max_radius: 0,
                min_radius: 0,
                sigma_ratio: 0.0,
            };
        }

        // Compute centroid via majority bundling
        let centroid = BitpackedVector::bundle(vectors);

        // Compute distances to centroid
        let distances: Vec<u32> = vectors
            .iter()
            .map(|v| hamming_distance_scalar(&centroid, v))
            .collect();

        let count = vectors.len();
        let sum: u32 = distances.iter().sum();
        let mean_radius = sum as f32 / count as f32;

        // Compute standard deviation
        let variance: f32 = distances
            .iter()
            .map(|&d| {
                let diff = d as f32 - mean_radius;
                diff * diff
            })
            .sum::<f32>() / count as f32;
        let radius_std = variance.sqrt();

        let max_radius = distances.iter().copied().max().unwrap_or(0);
        let min_radius = distances.iter().copied().min().unwrap_or(0);

        // How does our radius compare to theoretical σ?
        let sigma_ratio = mean_radius / HAMMING_STD_DEV as f32;

        Self {
            centroid,
            count,
            mean_radius,
            radius_std,
            max_radius,
            min_radius,
            sigma_ratio,
        }
    }

    /// Is this a tight cluster (radius < 1σ)?
    pub fn is_tight(&self) -> bool {
        self.sigma_ratio < 1.0
    }

    /// Is this cluster in the epiphany zone?
    pub fn is_epiphany_cluster(&self) -> bool {
        self.sigma_ratio >= 0.5 && self.sigma_ratio <= 2.0
    }

    /// Suggested search radius for this cluster
    pub fn suggested_search_radius(&self) -> u32 {
        // Use mean + 2*std to capture ~95% of cluster
        (self.mean_radius + 2.0 * self.radius_std) as u32
    }
}

// ============================================================================
// ADAPTIVE THRESHOLD ENGINE
// ============================================================================

/// Adaptive threshold that learns optimal cutoffs
#[derive(Clone, Debug)]
pub struct AdaptiveThreshold {
    /// Running statistics of "good" matches (user-confirmed)
    good_distances: Vec<u32>,
    /// Running statistics of "bad" matches (user-rejected)
    bad_distances: Vec<u32>,
    /// Current optimal threshold
    threshold: u32,
    /// Confidence in current threshold
    confidence: f32,
}

impl AdaptiveThreshold {
    pub fn new() -> Self {
        Self {
            good_distances: Vec::new(),
            bad_distances: Vec::new(),
            threshold: TWO_SIGMA, // Start at 2σ
            confidence: 0.5,
        }
    }

    /// Record a confirmed good match
    pub fn record_good(&mut self, distance: u32) {
        self.good_distances.push(distance);
        self.recalibrate();
    }

    /// Record a rejected match
    pub fn record_bad(&mut self, distance: u32) {
        self.bad_distances.push(distance);
        self.recalibrate();
    }

    /// Recalibrate threshold based on feedback
    fn recalibrate(&mut self) {
        if self.good_distances.is_empty() {
            return;
        }

        // Find threshold that maximizes separation
        let good_max = self.good_distances.iter().copied().max().unwrap_or(0);
        let good_mean: f32 = self.good_distances.iter().sum::<u32>() as f32
            / self.good_distances.len() as f32;

        let bad_min = self.bad_distances.iter().copied().min()
            .unwrap_or(VECTOR_BITS as u32);

        // Optimal threshold is midpoint between good_max and bad_min
        if bad_min > good_max {
            self.threshold = (good_max + bad_min) / 2;
            self.confidence = (bad_min - good_max) as f32 / HAMMING_STD_DEV as f32;
        } else {
            // Overlap exists, use good_mean + 1σ
            self.threshold = (good_mean + HAMMING_STD_DEV as f32) as u32;
            self.confidence = 0.3;
        }
    }

    /// Get current threshold
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Get confidence (0-1)
    pub fn confidence(&self) -> f32 {
        self.confidence.min(1.0)
    }

    /// Is this distance likely good?
    pub fn is_likely_good(&self, distance: u32) -> bool {
        distance <= self.threshold
    }
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EPIPHANY ENGINE
// ============================================================================

/// The Epiphany Engine: combines statistical and geometric calibration
/// to find the "sweet spot" for semantic discovery
pub struct EpiphanyEngine {
    /// Adaptive threshold for similarity
    pub threshold: AdaptiveThreshold,
    /// Cluster statistics cache
    cluster_stats: HashMap<u64, CentroidStats>,
    /// Configuration
    config: EpiphanyConfig,
    /// Discovery history
    discoveries: Vec<Discovery>,
}

/// Configuration for epiphany detection
#[derive(Clone, Debug)]
pub struct EpiphanyConfig {
    /// Minimum sigma ratio for epiphany zone
    pub min_sigma_ratio: f32,
    /// Maximum sigma ratio for epiphany zone
    pub max_sigma_ratio: f32,
    /// Weight for statistical component
    pub statistical_weight: f32,
    /// Weight for geometric component
    pub geometric_weight: f32,
    /// Minimum confidence for reporting
    pub min_confidence: f32,
}

impl Default for EpiphanyConfig {
    fn default() -> Self {
        Self {
            min_sigma_ratio: 0.5,
            max_sigma_ratio: 2.0,
            statistical_weight: 0.6,
            geometric_weight: 0.4,
            min_confidence: 0.3,
        }
    }
}

/// A discovered insight
#[derive(Clone, Debug)]
pub struct Discovery {
    /// Query that led to discovery
    pub query_id: u64,
    /// Discovered item
    pub found_id: u64,
    /// Distance
    pub distance: u32,
    /// Zone classification
    pub zone: EpiphanyZone,
    /// Confidence score
    pub confidence: f32,
    /// Path through clusters (if applicable)
    pub path: Vec<u64>,
}

impl EpiphanyEngine {
    pub fn new() -> Self {
        Self::with_config(EpiphanyConfig::default())
    }

    pub fn with_config(config: EpiphanyConfig) -> Self {
        Self {
            threshold: AdaptiveThreshold::new(),
            cluster_stats: HashMap::new(),
            config,
            discoveries: Vec::new(),
        }
    }

    /// Analyze a potential match and compute epiphany score
    pub fn analyze(
        &self,
        query: &BitpackedVector,
        candidate: &BitpackedVector,
        candidate_id: u64,
    ) -> Option<Discovery> {
        let distance = hamming_distance_scalar(query, candidate);
        let zone = EpiphanyZone::classify(distance);

        if !zone.is_significant() {
            return None;
        }

        // Statistical component: how many σ from expected random?
        let z_score = (EXPECTED_RANDOM_DISTANCE - distance as f64) / HAMMING_STD_DEV;
        let statistical_confidence = (z_score / 3.0).min(1.0).max(0.0) as f32;

        // Geometric component: is distance in the "sweet spot"?
        let sigma_ratio = distance as f32 / HAMMING_STD_DEV as f32;
        let geometric_confidence = if sigma_ratio >= self.config.min_sigma_ratio
            && sigma_ratio <= self.config.max_sigma_ratio
        {
            1.0 - ((sigma_ratio - 1.0).abs() / 1.0) // Peak at 1σ
        } else {
            0.2
        };

        // Combined confidence
        let confidence = self.config.statistical_weight * statistical_confidence
            + self.config.geometric_weight * geometric_confidence;

        if confidence < self.config.min_confidence {
            return None;
        }

        Some(Discovery {
            query_id: 0, // Caller should set
            found_id: candidate_id,
            distance,
            zone,
            confidence,
            path: Vec::new(),
        })
    }

    /// Search with epiphany awareness
    pub fn search(
        &mut self,
        query: &BitpackedVector,
        candidates: &[(u64, BitpackedVector)],
        max_results: usize,
    ) -> Vec<Discovery> {
        let mut discoveries: Vec<_> = candidates
            .iter()
            .filter_map(|(id, fp)| self.analyze(query, fp, *id))
            .collect();

        // Sort by confidence, then by zone significance
        discoveries.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        discoveries.truncate(max_results);

        // Record discoveries
        self.discoveries.extend(discoveries.clone());

        discoveries
    }

    /// Register a cluster for geometric calibration
    pub fn register_cluster(&mut self, cluster_id: u64, vectors: &[&BitpackedVector]) {
        let stats = CentroidStats::compute(vectors);
        self.cluster_stats.insert(cluster_id, stats);
    }

    /// Get optimal search radius for a cluster
    pub fn optimal_radius(&self, cluster_id: u64) -> u32 {
        self.cluster_stats
            .get(&cluster_id)
            .map(|s| s.suggested_search_radius())
            .unwrap_or(TWO_SIGMA)
    }

    /// Feedback: user confirmed this was a good match
    pub fn confirm_good(&mut self, distance: u32) {
        self.threshold.record_good(distance);
    }

    /// Feedback: user rejected this match
    pub fn confirm_bad(&mut self, distance: u32) {
        self.threshold.record_bad(distance);
    }

    /// Get the current "epiphany zone" boundaries
    pub fn zone_boundaries(&self) -> ZoneBoundaries {
        ZoneBoundaries {
            identity_max: ONE_SIGMA,
            epiphany_max: self.threshold.threshold(),
            penumbra_max: THREE_SIGMA,
            antipode_min: VECTOR_BITS as u32 - THREE_SIGMA,
        }
    }

    /// Analyze cluster health
    pub fn cluster_health(&self, cluster_id: u64) -> Option<ClusterHealth> {
        self.cluster_stats.get(&cluster_id).map(|stats| {
            ClusterHealth {
                is_tight: stats.is_tight(),
                is_epiphany_zone: stats.is_epiphany_cluster(),
                sigma_ratio: stats.sigma_ratio,
                suggested_split: stats.sigma_ratio > 2.5,
                suggested_merge: stats.sigma_ratio < 0.3,
            }
        })
    }

    /// Get discovery statistics
    pub fn stats(&self) -> EpiphanyStats {
        let zone_counts: HashMap<EpiphanyZone, usize> = self.discoveries
            .iter()
            .fold(HashMap::new(), |mut acc, d| {
                *acc.entry(d.zone).or_insert(0) += 1;
                acc
            });

        let avg_confidence = if self.discoveries.is_empty() {
            0.0
        } else {
            self.discoveries.iter().map(|d| d.confidence).sum::<f32>()
                / self.discoveries.len() as f32
        };

        EpiphanyStats {
            total_discoveries: self.discoveries.len(),
            identity_count: zone_counts.get(&EpiphanyZone::Identity).copied().unwrap_or(0),
            epiphany_count: zone_counts.get(&EpiphanyZone::Epiphany).copied().unwrap_or(0),
            penumbra_count: zone_counts.get(&EpiphanyZone::Penumbra).copied().unwrap_or(0),
            antipode_count: zone_counts.get(&EpiphanyZone::Antipode).copied().unwrap_or(0),
            average_confidence: avg_confidence,
            current_threshold: self.threshold.threshold(),
            threshold_confidence: self.threshold.confidence(),
        }
    }
}

impl Default for EpiphanyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Zone boundaries (distances)
#[derive(Clone, Debug)]
pub struct ZoneBoundaries {
    pub identity_max: u32,
    pub epiphany_max: u32,
    pub penumbra_max: u32,
    pub antipode_min: u32,
}

/// Cluster health assessment
#[derive(Clone, Debug)]
pub struct ClusterHealth {
    pub is_tight: bool,
    pub is_epiphany_zone: bool,
    pub sigma_ratio: f32,
    pub suggested_split: bool,
    pub suggested_merge: bool,
}

/// Discovery statistics
#[derive(Clone, Debug)]
pub struct EpiphanyStats {
    pub total_discoveries: usize,
    pub identity_count: usize,
    pub epiphany_count: usize,
    pub penumbra_count: usize,
    pub antipode_count: usize,
    pub average_confidence: f32,
    pub current_threshold: u32,
    pub threshold_confidence: f32,
}

// ============================================================================
// RESONANCE CALIBRATOR
// ============================================================================

/// Calibrates NN-Tree based on epiphany zones
pub struct ResonanceCalibrator {
    /// Target sigma ratio for clusters
    target_sigma: f32,
    /// Samples for calibration
    samples: Vec<(BitpackedVector, u64)>,
    /// Computed optimal config
    optimal_config: Option<NnTreeConfig>,
}

impl ResonanceCalibrator {
    pub fn new(target_sigma: f32) -> Self {
        Self {
            target_sigma,
            samples: Vec::new(),
            optimal_config: None,
        }
    }

    /// Add sample for calibration
    pub fn add_sample(&mut self, fingerprint: BitpackedVector, id: u64) {
        self.samples.push((fingerprint, id));
    }

    /// Calibrate based on samples
    pub fn calibrate(&mut self) -> NnTreeConfig {
        if self.samples.len() < 10 {
            return NnTreeConfig::default();
        }

        // Compute pairwise distances to estimate data distribution
        let mut distances = Vec::new();
        let sample_size = self.samples.len().min(100);

        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                let d = hamming_distance_scalar(&self.samples[i].0, &self.samples[j].0);
                distances.push(d);
            }
        }

        if distances.is_empty() {
            return NnTreeConfig::default();
        }

        // Compute statistics
        let mean: f32 = distances.iter().sum::<u32>() as f32 / distances.len() as f32;
        let variance: f32 = distances
            .iter()
            .map(|&d| (d as f32 - mean).powi(2))
            .sum::<f32>() / distances.len() as f32;
        let data_std = variance.sqrt();

        // The "sweet spot": leaf size where intra-cluster variance ≈ target_sigma
        // Larger leaves = more variance, smaller = less
        // Empirical formula: leaf_size ≈ (data_std / target_sigma)^2 * base_size
        let variance_ratio = data_std / (self.target_sigma * HAMMING_STD_DEV as f32);
        let optimal_leaf_size = ((variance_ratio * variance_ratio) * 32.0)
            .clamp(8.0, 256.0) as usize;

        // Branching factor: balance between depth and breadth
        // Higher variance data needs more branches for discrimination
        let optimal_branches = if data_std > HAMMING_STD_DEV as f32 * 1.5 {
            32 // High variance: more branches
        } else if data_std < HAMMING_STD_DEV as f32 * 0.5 {
            8  // Low variance: fewer branches
        } else {
            16 // Normal
        };

        // Search beam: wider for high variance
        let optimal_beam = (variance_ratio * 4.0).clamp(2.0, 16.0) as usize;

        let config = NnTreeConfig {
            max_children: optimal_branches,
            max_leaf_size: optimal_leaf_size,
            search_beam: optimal_beam,
            use_bundling: true,
        };

        self.optimal_config = Some(config.clone());
        config
    }

    /// Get calibrated config
    pub fn config(&self) -> NnTreeConfig {
        self.optimal_config.clone().unwrap_or_default()
    }

    /// Build calibrated tree
    pub fn build_tree(&mut self) -> NnTree {
        let config = self.calibrate();
        let mut tree = NnTree::with_config(config);

        for (fp, id) in &self.samples {
            tree.insert_with_id(*id, fp.clone());
        }

        tree
    }
}

// ============================================================================
// INSIGHT AMPLIFIER
// ============================================================================

/// Amplifies weak signals in the penumbra zone
pub struct InsightAmplifier {
    /// Accumulator for weak signals
    accumulators: HashMap<u64, f32>,
    /// Decay rate per round
    decay: f32,
    /// Threshold for promotion to epiphany
    promotion_threshold: f32,
}

impl InsightAmplifier {
    pub fn new(decay: f32, promotion_threshold: f32) -> Self {
        Self {
            accumulators: HashMap::new(),
            decay,
            promotion_threshold,
        }
    }

    /// Observe a weak signal
    pub fn observe(&mut self, id: u64, confidence: f32) {
        let entry = self.accumulators.entry(id).or_insert(0.0);
        *entry = (*entry * self.decay) + confidence;
    }

    /// Check for promoted insights
    pub fn promoted(&self) -> Vec<(u64, f32)> {
        self.accumulators
            .iter()
            .filter(|(_, acc)| **acc >= self.promotion_threshold)
            .map(|(id, acc)| (*id, *acc))
            .collect()
    }

    /// Decay all accumulators
    pub fn tick(&mut self) {
        for acc in self.accumulators.values_mut() {
            *acc *= self.decay;
        }
        // Remove dead signals
        self.accumulators.retain(|_, &mut acc| acc > 0.01);
    }

    /// Clear all
    pub fn clear(&mut self) {
        self.accumulators.clear();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_classification() {
        assert_eq!(EpiphanyZone::classify(30), EpiphanyZone::Identity);
        assert_eq!(EpiphanyZone::classify(75), EpiphanyZone::Epiphany);
        assert_eq!(EpiphanyZone::classify(120), EpiphanyZone::Penumbra);
        assert_eq!(EpiphanyZone::classify(1000), EpiphanyZone::Noise);
        assert_eq!(EpiphanyZone::classify(9900), EpiphanyZone::Antipode);
    }

    #[test]
    fn test_centroid_stats() {
        // Create similar vectors
        let base = BitpackedVector::random(42);
        let v1 = base.clone();
        let mut v2 = base.clone();
        v2.flip_random_bits(30, 100); // Small perturbation
        let mut v3 = base.clone();
        v3.flip_random_bits(40, 200);

        let stats = CentroidStats::compute(&[&v1, &v2, &v3]);

        assert!(stats.is_tight()); // Should be tight cluster
        assert!(stats.mean_radius < ONE_SIGMA as f32);
        println!("Centroid stats: {:?}", stats);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut threshold = AdaptiveThreshold::new();

        // Record some good matches
        threshold.record_good(30);
        threshold.record_good(45);
        threshold.record_good(55);

        // Record some bad matches
        threshold.record_bad(150);
        threshold.record_bad(200);

        // Threshold should be between good_max (55) and bad_min (150)
        assert!(threshold.threshold() > 55);
        assert!(threshold.threshold() < 150);
        println!("Adaptive threshold: {}", threshold.threshold());
    }

    #[test]
    fn test_epiphany_engine() {
        let mut engine = EpiphanyEngine::new();

        let query = BitpackedVector::random(1);

        // Create candidates at various distances
        let mut similar = query.clone();
        similar.flip_random_bits(30, 42); // Close

        let mut related = query.clone();
        related.flip_random_bits(80, 43); // In epiphany zone

        let random = BitpackedVector::random(999); // Far

        let candidates = vec![
            (1, similar),
            (2, related),
            (3, random),
        ];

        let discoveries = engine.search(&query, &candidates, 10);

        // Should find the similar and related, not the random
        assert!(discoveries.iter().any(|d| d.found_id == 1));
        assert!(discoveries.iter().any(|d| d.found_id == 2));
        println!("Discoveries: {:?}", discoveries);
    }

    #[test]
    fn test_resonance_calibrator() {
        let mut calibrator = ResonanceCalibrator::new(1.0); // Target 1σ

        // Add some samples
        for i in 0..50 {
            calibrator.add_sample(BitpackedVector::random(i), i);
        }

        let config = calibrator.calibrate();
        println!("Calibrated config: {:?}", config);

        // Should have reasonable values
        assert!(config.max_leaf_size >= 8);
        assert!(config.max_children >= 4);
    }

    #[test]
    fn test_insight_amplifier() {
        let mut amplifier = InsightAmplifier::new(0.9, 2.0);

        // Observe weak signal multiple times
        for _ in 0..10 {
            amplifier.observe(42, 0.3);
        }

        let promoted = amplifier.promoted();
        assert!(promoted.iter().any(|(id, _)| *id == 42));
    }
}
