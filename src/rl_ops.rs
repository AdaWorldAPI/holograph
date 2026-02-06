//! Reinforcement Learning Operations for HDR Neural Trees
//!
//! Combines Déjà Vu multipass RL with Hebbian crystal learning and neural tree
//! reward propagation. This is the "learning" layer that makes the hierarchical
//! neural tree adapt over time — without backpropagation, without GPU.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                       RL Operations                              │
//! │                                                                  │
//! │  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
//! │  │ RewardTracker │  │ HebbianMatrix │  │ PolicyGradient      │  │
//! │  │ per-node      │  │ cell→cell     │  │ (state,action)→Q    │  │
//! │  │ rewards       │  │ co-activation │  │                     │  │
//! │  └──────┬───────┘  └───────┬───────┘  └──────────┬──────────┘  │
//! │         │                  │                      │             │
//! │         ▼                  ▼                      ▼             │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │                Neural Tree + Crystal                      │  │
//! │  │  - Hebbian weights adjust routing priority                │  │
//! │  │  - Crystal cells learn via bundled reinforcement          │  │
//! │  │  - Sigma bands adapt per-path based on reward             │  │
//! │  │  - Block attention weights shift toward rewarded regions  │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Insight: XOR as Backpropagation
//!
//! In traditional neural networks, error backpropagates via chain rule.
//! In HDR, the XOR binding IS the backward pass:
//!
//! ```text
//! Forward:  query ⊕ path_fingerprint → similarity score
//! Backward: reward_signal ⊕ path_fingerprint → credit assignment
//! ```
//!
//! The XOR-bound reward creates a fingerprint that, when unbound at any
//! node along the path, reveals how much credit that node deserves.

use crate::bitpack::{BitpackedVector, VECTOR_BITS, VECTOR_WORDS};
use crate::hamming::hamming_distance_scalar;
use crate::crystal_dejavu::{Coord5D, SigmaBand};
use crate::epiphany::{EpiphanyZone, TWO_SIGMA, THREE_SIGMA};
use std::collections::HashMap;

// ============================================================================
// REWARD SIGNAL: HDR-native reward encoding
// ============================================================================

/// A reward encoded as a fingerprint perturbation.
///
/// Positive reward: small Hamming distance from target → reinforce
/// Negative reward: flip bits away from target → weaken
/// Magnitude: number of bits flipped ∝ |reward|
#[derive(Clone, Debug)]
pub struct RewardSignal {
    /// The reward fingerprint (XOR mask to apply)
    pub mask: BitpackedVector,
    /// Scalar reward value (-1.0 to 1.0)
    pub value: f32,
    /// Which sigma band this reward targets
    pub band: SigmaBand,
    /// Path through tree (DN addresses as strings)
    pub path: Vec<String>,
}

impl RewardSignal {
    /// Create reward signal from scalar value
    ///
    /// Positive rewards create a mask that, when XORed with the target,
    /// produces a vector closer to the query (reinforcing the association).
    /// Negative rewards produce a vector farther from query (weakening it).
    pub fn from_scalar(
        query: &BitpackedVector,
        target: &BitpackedVector,
        reward: f32,
    ) -> Self {
        let reward = reward.clamp(-1.0, 1.0);
        let distance = hamming_distance_scalar(query, target);
        let band = SigmaBand::from_distance(distance);

        // Number of bits to flip proportional to |reward|
        let flip_count = (reward.abs() * 100.0) as usize;
        let xor = query.xor(target);

        // Create mask: for positive reward, flip bits that REDUCE distance
        // (i.e., bits where query and target already agree → set those in mask)
        // For negative reward, flip bits that INCREASE distance
        let mut mask = BitpackedVector::zero();
        let mut flipped = 0;

        if reward > 0.0 {
            // Positive: reinforce by flipping XOR-1 bits to 0 (bring closer)
            let xor_words = xor.words();
            for word_idx in 0..VECTOR_WORDS {
                if flipped >= flip_count {
                    break;
                }
                let word = xor_words[word_idx];
                for bit in 0..64 {
                    if flipped >= flip_count {
                        break;
                    }
                    if (word >> bit) & 1 == 1 {
                        mask.set_bit(word_idx * 64 + bit, true);
                        flipped += 1;
                    }
                }
            }
        } else {
            // Negative: weaken by flipping agreement bits to disagreement
            let xor_words = xor.words();
            for word_idx in 0..VECTOR_WORDS {
                if flipped >= flip_count {
                    break;
                }
                let word = xor_words[word_idx];
                for bit in 0..64 {
                    if flipped >= flip_count {
                        break;
                    }
                    if (word >> bit) & 1 == 0 {
                        mask.set_bit(word_idx * 64 + bit, true);
                        flipped += 1;
                    }
                }
            }
        }

        Self {
            mask,
            value: reward,
            band,
            path: Vec::new(),
        }
    }

    /// Apply reward to a fingerprint (returns modified fingerprint)
    pub fn apply(&self, target: &BitpackedVector) -> BitpackedVector {
        target.xor(&self.mask)
    }
}

// ============================================================================
// HEBBIAN MATRIX: Crystal cell co-activation tracking
// ============================================================================

/// Tracks co-activation between crystal cells.
///
/// When two cells fire together (both match a query within threshold),
/// the connection between them strengthens. This is Hebb's rule:
/// "Cells that fire together wire together."
///
/// The matrix is sparse — only active connections are stored.
pub struct HebbianMatrix {
    /// Co-activation weights: (cell_a, cell_b) → weight
    weights: HashMap<(usize, usize), f32>,
    /// Per-cell activation count
    activations: HashMap<usize, u32>,
    /// Learning rate
    eta: f32,
    /// Decay rate per timestep
    decay: f32,
    /// Minimum weight before pruning
    prune_threshold: f32,
}

impl HebbianMatrix {
    /// Create new Hebbian matrix
    pub fn new(eta: f32, decay: f32) -> Self {
        Self {
            weights: HashMap::new(),
            activations: HashMap::new(),
            eta,
            decay,
            prune_threshold: 0.001,
        }
    }

    /// Record co-activation between two crystal cells
    pub fn fire_together(&mut self, cell_a: usize, cell_b: usize) {
        let key = if cell_a <= cell_b {
            (cell_a, cell_b)
        } else {
            (cell_b, cell_a)
        };

        *self.weights.entry(key).or_insert(0.0) += self.eta;
        *self.activations.entry(cell_a).or_insert(0) += 1;
        *self.activations.entry(cell_b).or_insert(0) += 1;
    }

    /// Record activation of a set of cells (all pairs fire together)
    pub fn fire_set(&mut self, cells: &[usize]) {
        for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                self.fire_together(cells[i], cells[j]);
            }
        }
    }

    /// Get connection strength between two cells
    pub fn strength(&self, cell_a: usize, cell_b: usize) -> f32 {
        let key = if cell_a <= cell_b {
            (cell_a, cell_b)
        } else {
            (cell_b, cell_a)
        };
        *self.weights.get(&key).unwrap_or(&0.0)
    }

    /// Get strongest connections for a cell
    pub fn strongest_connections(&self, cell: usize, k: usize) -> Vec<(usize, f32)> {
        let mut connections: Vec<(usize, f32)> = self
            .weights
            .iter()
            .filter_map(|(&(a, b), &w)| {
                if a == cell {
                    Some((b, w))
                } else if b == cell {
                    Some((a, w))
                } else {
                    None
                }
            })
            .collect();

        connections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        connections.truncate(k);
        connections
    }

    /// Apply decay and prune dead connections
    pub fn decay_step(&mut self) {
        for weight in self.weights.values_mut() {
            *weight *= self.decay;
        }
        self.weights
            .retain(|_, w| *w > self.prune_threshold);
    }

    /// Total number of active connections
    pub fn num_connections(&self) -> usize {
        self.weights.len()
    }

    /// Most activated cells (hub nodes in the Hebbian graph)
    pub fn hub_cells(&self, k: usize) -> Vec<(usize, u32)> {
        let mut cells: Vec<(usize, u32)> = self.activations.iter().map(|(&c, &n)| (c, n)).collect();
        cells.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        cells.truncate(k);
        cells
    }
}

// ============================================================================
// POLICY GRADIENT: State-Action Q-values for search routing
// ============================================================================

/// State in the neural tree search
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SearchState {
    /// At a specific tree depth with observed block signature pattern
    AtDepth {
        depth: u8,
        /// Dominant block pattern (top 3 hottest blocks)
        hot_blocks: [u8; 3],
    },
    /// In a specific sigma band
    InBand(SigmaBand),
    /// In a specific epiphany zone
    InZone(EpiphanyZone),
}

/// Action in the neural tree search
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SearchAction {
    /// Explore this subtree
    Explore,
    /// Prune this subtree
    Prune,
    /// Widen beam
    Widen,
    /// Narrow beam
    Narrow,
    /// Switch to crystal routing
    CrystalRoute,
    /// Switch to block prefilter
    BlockFilter,
}

/// Maximum Q-table entries to prevent unbounded growth.
/// With 256 depths × top-3-blocks patterns × 6 actions, this caps at ~50K.
const MAX_Q_TABLE_SIZE: usize = 50_000;

/// Maximum episode reward history (ring buffer)
const MAX_EPISODE_HISTORY: usize = 10_000;

/// Q-learning policy for search routing decisions
pub struct PolicyGradient {
    /// Q-table: (state, action) → value, bounded
    q_table: HashMap<(SearchState, SearchAction), f32>,
    /// Learning rate
    alpha: f32,
    /// Discount factor
    gamma: f32,
    /// Exploration rate (epsilon-greedy)
    epsilon: f32,
    /// Episode rewards for tracking (bounded ring buffer)
    episode_rewards: std::collections::VecDeque<f32>,
    /// Action history for current episode
    current_episode: Vec<(SearchState, SearchAction)>,
}

impl PolicyGradient {
    /// Create new policy with default parameters
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 0.1,
            episode_rewards: std::collections::VecDeque::new(),
            current_episode: Vec::new(),
        }
    }

    /// Create with custom parameters
    pub fn with_params(alpha: f32, gamma: f32, epsilon: f32) -> Self {
        Self {
            q_table: HashMap::new(),
            alpha,
            gamma,
            epsilon,
            episode_rewards: std::collections::VecDeque::new(),
            current_episode: Vec::new(),
        }
    }

    /// Get best action for a state (or explore with epsilon probability)
    pub fn select_action(&self, state: &SearchState, seed: u64) -> SearchAction {
        // Epsilon-greedy: explore with probability epsilon
        let explore = (seed % 1000) < (self.epsilon * 1000.0) as u64;
        if explore {
            // Random action
            match seed % 6 {
                0 => SearchAction::Explore,
                1 => SearchAction::Prune,
                2 => SearchAction::Widen,
                3 => SearchAction::Narrow,
                4 => SearchAction::CrystalRoute,
                _ => SearchAction::BlockFilter,
            }
        } else {
            // Greedy: best known action
            self.best_action(state)
        }
    }

    /// Get best action for a state (exploit)
    pub fn best_action(&self, state: &SearchState) -> SearchAction {
        let actions = [
            SearchAction::Explore,
            SearchAction::Prune,
            SearchAction::Widen,
            SearchAction::Narrow,
            SearchAction::CrystalRoute,
            SearchAction::BlockFilter,
        ];

        let mut best = SearchAction::Explore;
        let mut best_q = f32::NEG_INFINITY;

        for action in &actions {
            let q = *self
                .q_table
                .get(&(state.clone(), action.clone()))
                .unwrap_or(&0.0);
            if q > best_q {
                best_q = q;
                best = action.clone();
            }
        }

        best
    }

    /// Record state-action pair for current episode
    pub fn record(&mut self, state: SearchState, action: SearchAction) {
        self.current_episode.push((state, action));
    }

    /// End episode with reward, update Q-values via temporal difference
    pub fn end_episode(&mut self, final_reward: f32) {
        self.episode_rewards.push_back(final_reward);
        // Cap episode history (ring buffer)
        while self.episode_rewards.len() > MAX_EPISODE_HISTORY {
            self.episode_rewards.pop_front();
        }

        // Backward TD update through the episode
        let mut future_q = 0.0f32;

        for (state, action) in self.current_episode.iter().rev() {
            let key = (state.clone(), action.clone());
            let current_q = *self.q_table.get(&key).unwrap_or(&0.0);

            // TD update: Q(s,a) += α * (r + γ * max_a' Q(s',a') - Q(s,a))
            let td_target = final_reward + self.gamma * future_q;
            let new_q = current_q + self.alpha * (td_target - current_q);
            self.q_table.insert(key, new_q);

            future_q = new_q;
        }

        self.current_episode.clear();

        // Cap Q-table: evict entries with smallest absolute Q-values
        if self.q_table.len() > MAX_Q_TABLE_SIZE {
            let mut entries: Vec<_> = self.q_table.iter()
                .map(|(k, &v)| (k.clone(), v.abs()))
                .collect();
            entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let evict_count = self.q_table.len() - MAX_Q_TABLE_SIZE + MAX_Q_TABLE_SIZE / 10;
            for (key, _) in entries.iter().take(evict_count) {
                self.q_table.remove(key);
            }
        }
    }

    /// Get average reward over recent episodes
    pub fn avg_reward(&self, window: usize) -> f32 {
        if self.episode_rewards.is_empty() {
            return 0.0;
        }
        let n = self.episode_rewards.len();
        let start = n.saturating_sub(window);
        let sum: f32 = self.episode_rewards.iter().skip(start).sum();
        sum / (n - start) as f32
    }

    /// Decay exploration rate (anneal epsilon)
    pub fn anneal(&mut self, factor: f32) {
        self.epsilon = (self.epsilon * factor).max(0.01);
    }

    /// Number of learned state-action pairs
    pub fn policy_size(&self) -> usize {
        self.q_table.len()
    }
}

impl Default for PolicyGradient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// REWARD TRACKER: Per-path reward accumulation
// ============================================================================

/// Tracks rewards along tree paths for credit assignment
pub struct RewardTracker {
    /// Per-node accumulated reward
    node_rewards: HashMap<String, f32>,
    /// Per-node visit count
    node_visits: HashMap<String, u32>,
    /// Reward decay for temporal credit
    temporal_decay: f32,
    /// Total episodes
    total_episodes: u64,
}

impl RewardTracker {
    /// Create new tracker
    pub fn new(temporal_decay: f32) -> Self {
        Self {
            node_rewards: HashMap::new(),
            node_visits: HashMap::new(),
            temporal_decay,
            total_episodes: 0,
        }
    }

    /// Propagate reward along a path (credit assignment)
    ///
    /// Uses temporal decay: nodes closer to the reward receive more credit.
    /// This is analogous to backpropagation through the tree.
    pub fn propagate_reward(&mut self, path: &[String], reward: f32) {
        self.total_episodes += 1;
        let path_len = path.len();

        for (i, node_id) in path.iter().enumerate() {
            // Temporal credit: nodes closer to reward get more
            let temporal_factor = self.temporal_decay.powi((path_len - i - 1) as i32);
            let credit = reward * temporal_factor;

            *self.node_rewards.entry(node_id.clone()).or_insert(0.0) += credit;
            *self.node_visits.entry(node_id.clone()).or_insert(0) += 1;
        }
    }

    /// Get average reward for a node
    pub fn avg_reward(&self, node_id: &str) -> f32 {
        let reward = *self.node_rewards.get(node_id).unwrap_or(&0.0);
        let visits = *self.node_visits.get(node_id).unwrap_or(&1) as f32;
        reward / visits
    }

    /// Get UCB1 score (Upper Confidence Bound) for exploration/exploitation
    pub fn ucb1(&self, node_id: &str, exploration_constant: f32) -> f32 {
        let avg = self.avg_reward(node_id);
        let visits = *self.node_visits.get(node_id).unwrap_or(&1) as f32;
        let total = self.total_episodes.max(1) as f32;

        avg + exploration_constant * (2.0 * total.ln() / visits).sqrt()
    }

    /// Top rewarded nodes
    pub fn top_nodes(&self, k: usize) -> Vec<(String, f32)> {
        let mut nodes: Vec<_> = self
            .node_rewards
            .iter()
            .map(|(id, &r)| {
                let visits = *self.node_visits.get(id).unwrap_or(&1) as f32;
                (id.clone(), r / visits)
            })
            .collect();
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        nodes.truncate(k);
        nodes
    }

    /// Decay all rewards and prune dead entries (for non-stationarity)
    pub fn decay_all(&mut self, factor: f32) {
        for r in self.node_rewards.values_mut() {
            *r *= factor;
        }
        // Prune near-zero entries to prevent unbounded growth
        self.node_rewards.retain(|_, r| r.abs() > 0.001);
        // Also prune visit counts for removed nodes
        self.node_visits.retain(|k, _| self.node_rewards.contains_key(k));
    }
}

// ============================================================================
// UNIFIED RL ENGINE
// ============================================================================

/// Unified reinforcement learning engine for the neural tree + crystal system
pub struct RlEngine {
    /// Hebbian co-activation matrix
    pub hebbian: HebbianMatrix,
    /// Search policy
    pub policy: PolicyGradient,
    /// Path reward tracker
    pub tracker: RewardTracker,
    /// Crystal cell reward accumulator
    crystal_rewards: HashMap<usize, f32>,
    /// Block attention adjustments (learned per-block weights)
    pub block_weights: [f32; 10],
}

impl RlEngine {
    /// Create new RL engine with default parameters
    pub fn new() -> Self {
        Self {
            hebbian: HebbianMatrix::new(0.1, 0.999),
            policy: PolicyGradient::new(),
            tracker: RewardTracker::new(0.9),
            crystal_rewards: HashMap::new(),
            block_weights: [1.0; 10],
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        hebbian_eta: f32,
        hebbian_decay: f32,
        policy_alpha: f32,
        policy_gamma: f32,
        policy_epsilon: f32,
        temporal_decay: f32,
    ) -> Self {
        Self {
            hebbian: HebbianMatrix::new(hebbian_eta, hebbian_decay),
            policy: PolicyGradient::with_params(policy_alpha, policy_gamma, policy_epsilon),
            tracker: RewardTracker::new(temporal_decay),
            crystal_rewards: HashMap::new(),
            block_weights: [1.0; 10],
        }
    }

    /// Process a search result with reward feedback
    ///
    /// This is the main entry point for RL updates:
    /// 1. Generate reward signal from query/result pair
    /// 2. Propagate reward along the search path
    /// 3. Update Hebbian co-activation for crystal cells
    /// 4. Update policy Q-values
    /// 5. Adjust block attention weights
    pub fn reward_search(
        &mut self,
        query: &BitpackedVector,
        result: &BitpackedVector,
        reward: f32,
        path: &[String],
        query_crystal: Coord5D,
        result_crystal: Coord5D,
        block_signature: &[u16; 10],
    ) {
        // 1. Generate reward signal
        let signal = RewardSignal::from_scalar(query, result, reward);

        // 2. Propagate along path
        self.tracker.propagate_reward(path, reward);

        // 3. Hebbian: if reward positive, fire crystal cells together
        if reward > 0.0 {
            let query_cell = query_crystal.to_index();
            let result_cell = result_crystal.to_index();
            self.hebbian.fire_together(query_cell, result_cell);

            // Also fire with neighboring cells
            let neighbors = query_crystal.neighborhood(1);
            let neighbor_cells: Vec<usize> = neighbors.iter().map(|c| c.to_index()).collect();
            self.hebbian.fire_set(&neighbor_cells);
        }

        // 4. Update crystal cell rewards
        let result_cell = result_crystal.to_index();
        *self.crystal_rewards.entry(result_cell).or_insert(0.0) += reward;

        // 5. Adjust block weights based on which blocks contributed to match
        for (i, &sig) in block_signature.iter().enumerate() {
            if i < 10 {
                // Blocks with high activation in successful matches get boosted
                let block_contribution = sig as f32 / 1000.0; // normalize
                self.block_weights[i] += reward * block_contribution * 0.01;
                self.block_weights[i] = self.block_weights[i].clamp(0.1, 5.0);
            }
        }

        // 6. Decay
        self.hebbian.decay_step();
    }

    /// Get adjusted block weights for search
    pub fn adjusted_block_weights(&self) -> [f32; 10] {
        let mut normalized = self.block_weights;
        let sum: f32 = normalized.iter().sum();
        if sum > 0.0 {
            for w in &mut normalized {
                *w /= sum;
            }
        }
        normalized
    }

    /// Get crystal cell with highest accumulated reward
    pub fn best_crystal_cells(&self, k: usize) -> Vec<(usize, f32)> {
        let mut cells: Vec<(usize, f32)> = self.crystal_rewards.iter().map(|(&c, &r)| (c, r)).collect();
        cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cells.truncate(k);
        cells
    }

    /// Get Hebbian hub cells (most connected in co-activation graph)
    pub fn hub_cells(&self, k: usize) -> Vec<(usize, u32)> {
        self.hebbian.hub_cells(k)
    }

    /// Summary statistics
    pub fn stats(&self) -> RlStats {
        RlStats {
            hebbian_connections: self.hebbian.num_connections(),
            policy_states: self.policy.policy_size(),
            tracked_nodes: self.tracker.node_rewards.len(),
            rewarded_crystals: self.crystal_rewards.len(),
            avg_reward: self.policy.avg_reward(100),
            exploration_rate: self.policy.epsilon,
            block_weights: self.block_weights,
        }
    }
}

impl Default for RlEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// RL engine statistics
#[derive(Clone, Debug)]
pub struct RlStats {
    pub hebbian_connections: usize,
    pub policy_states: usize,
    pub tracked_nodes: usize,
    pub rewarded_crystals: usize,
    pub avg_reward: f32,
    pub exploration_rate: f32,
    pub block_weights: [f32; 10],
}

impl std::fmt::Display for RlStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RL[hebb={} conn, policy={} states, tracked={} nodes, \
             crystals={}, avg_r={:.3}, ε={:.3}]",
            self.hebbian_connections,
            self.policy_states,
            self.tracked_nodes,
            self.rewarded_crystals,
            self.avg_reward,
            self.exploration_rate,
        )
    }
}

// ============================================================================
// CAUSAL RL: Intervention-based learning (harvested from ladybug-rs)
// ============================================================================

/// An intervention record: "In state S, action A caused outcome O with reward R"
///
/// Adapted from ladybug-rs CausalRlAgent. Uses BitpackedVector instead of
/// raw [u64; 156], enabling XOR-bind for causal relationship encoding.
#[derive(Clone, Debug)]
pub struct Intervention {
    /// State fingerprint
    pub state: BitpackedVector,
    /// Action fingerprint (e.g., XOR-bound edge traversal)
    pub action: BitpackedVector,
    /// Observed outcome fingerprint
    pub outcome: BitpackedVector,
    /// Scalar reward
    pub reward: f32,
    /// Causal binding: state ⊕ action (for fast lookup)
    pub causal_bind: BitpackedVector,
}

impl Intervention {
    pub fn new(
        state: BitpackedVector,
        action: BitpackedVector,
        outcome: BitpackedVector,
        reward: f32,
    ) -> Self {
        let causal_bind = state.xor(&action);
        Self {
            state,
            action,
            outcome,
            reward,
            causal_bind,
        }
    }
}

/// A counterfactual: "What would have happened if I had done A' instead?"
#[derive(Clone, Debug)]
pub struct Counterfactual {
    /// Original state
    pub state: BitpackedVector,
    /// Alternative action (not taken)
    pub alt_action: BitpackedVector,
    /// Hypothesized outcome
    pub alt_outcome: BitpackedVector,
    /// Estimated reward of alternative
    pub alt_reward: f32,
    /// Regret: actual_reward - alt_reward (negative = we should have taken alt)
    pub regret: f32,
}

/// Causal RL agent that learns from interventions and counterfactuals.
///
/// Three rungs of causal reasoning (Pearl's ladder):
/// 1. **Association**: P(outcome | state, action) — standard Q-learning
/// 2. **Intervention**: P(outcome | state, do(action)) — causal Q-value
/// 3. **Counterfactual**: P(outcome_cf | state, do(alt_action)) — regret/credit
///
/// All encoded in BitpackedVector space: state⊕action is the causal binding,
/// and Hamming distance measures causal proximity.
pub struct CausalRlAgent {
    /// Stored interventions (Rung 2), bounded FIFO
    interventions: std::collections::VecDeque<Intervention>,
    /// Stored counterfactuals (Rung 3), bounded FIFO
    counterfactuals: std::collections::VecDeque<Counterfactual>,
    /// Q-value cache: hash(state⊕action) → estimated value
    q_cache: HashMap<u64, f32>,
    /// Discount factor
    gamma: f32,
    /// Learning rate
    alpha: f32,
    /// Exploration rate
    epsilon: f32,
    /// Curiosity bonus: 1/(1 + visit_count) for unseen state-action pairs
    visit_counts: HashMap<u64, u32>,
    /// Maximum stored interventions
    max_interventions: usize,
    /// Maximum stored counterfactuals (prevents unbounded growth)
    max_counterfactuals: usize,
    /// Maximum Q-cache entries
    max_q_cache: usize,
}

impl CausalRlAgent {
    pub fn new(gamma: f32, alpha: f32, epsilon: f32) -> Self {
        Self {
            interventions: std::collections::VecDeque::new(),
            counterfactuals: std::collections::VecDeque::new(),
            q_cache: HashMap::new(),
            gamma,
            alpha,
            epsilon,
            visit_counts: HashMap::new(),
            max_interventions: 10_000,
            max_counterfactuals: 5_000,
            max_q_cache: 50_000,
        }
    }

    /// Hash a state-action pair for Q-table lookup
    fn hash_sa(state: &BitpackedVector, action: &BitpackedVector) -> u64 {
        let sw = state.words();
        let aw = action.words();
        sw[0] ^ aw[0] ^ sw[1].rotate_left(32) ^ aw[1].rotate_left(32)
            ^ sw[78].rotate_left(16) ^ aw[78].rotate_left(48)
    }

    /// Store an intervention (Rung 2: "I did A in state S and got O")
    pub fn store_intervention(
        &mut self,
        state: BitpackedVector,
        action: BitpackedVector,
        outcome: BitpackedVector,
        reward: f32,
    ) {
        // Update visit counts
        let hash = Self::hash_sa(&state, &action);
        *self.visit_counts.entry(hash).or_insert(0) += 1;

        // Update Q-value via TD(0)
        let old_q = *self.q_cache.get(&hash).unwrap_or(&0.0);
        let new_q = old_q + self.alpha * (reward + self.gamma * old_q - old_q);
        self.q_cache.insert(hash, new_q);

        // Store intervention
        self.interventions
            .push_back(Intervention::new(state, action, outcome, reward));

        // Evict oldest if over capacity (O(1) with VecDeque)
        while self.interventions.len() > self.max_interventions {
            self.interventions.pop_front();
        }

        // Cap Q-cache to prevent unbounded growth
        if self.q_cache.len() > self.max_q_cache {
            // Evict ~10% of entries (those with lowest visit counts)
            let evict_count = self.max_q_cache / 10;
            let mut entries: Vec<_> = self.visit_counts.iter().map(|(&k, &v)| (k, v)).collect();
            entries.sort_by_key(|&(_, v)| v);
            for (key, _) in entries.iter().take(evict_count) {
                self.q_cache.remove(key);
                self.visit_counts.remove(key);
            }
        }
    }

    /// Store a counterfactual (Rung 3: "If I had done A' instead...")
    pub fn store_counterfactual(
        &mut self,
        state: BitpackedVector,
        alt_action: BitpackedVector,
        alt_outcome: BitpackedVector,
        alt_reward: f32,
        actual_reward: f32,
    ) {
        self.counterfactuals.push_back(Counterfactual {
            state,
            alt_action,
            alt_outcome,
            alt_reward,
            regret: actual_reward - alt_reward,
        });

        // Evict oldest if over capacity (O(1) with VecDeque)
        while self.counterfactuals.len() > self.max_counterfactuals {
            self.counterfactuals.pop_front();
        }
    }

    /// Causal Q-value: E[reward | state, do(action)]
    ///
    /// Unlike standard Q-learning which uses correlation,
    /// this queries only from interventional data.
    pub fn q_value_causal(
        &self,
        state: &BitpackedVector,
        action: &BitpackedVector,
    ) -> f32 {
        let hash = Self::hash_sa(state, action);
        if let Some(&cached) = self.q_cache.get(&hash) {
            return cached;
        }

        // Find similar interventions via Hamming distance
        let query_bind = state.xor(action);
        let mut total_weight = 0.0f32;
        let mut weighted_reward = 0.0f32;

        for interv in &self.interventions {
            let dist = hamming_distance_scalar(&query_bind, &interv.causal_bind);
            if dist < TWO_SIGMA {
                let weight = 1.0 / (1.0 + dist as f32);
                total_weight += weight;
                weighted_reward += weight * interv.reward;
            }
        }

        if total_weight > 0.0 {
            weighted_reward / total_weight
        } else {
            0.0
        }
    }

    /// Query outcomes from interventional data
    pub fn query_outcomes(
        &self,
        state: &BitpackedVector,
        action: &BitpackedVector,
        k: usize,
    ) -> Vec<(&Intervention, u32)> {
        let query_bind = state.xor(action);
        let mut results: Vec<_> = self
            .interventions
            .iter()
            .map(|interv| {
                let dist = hamming_distance_scalar(&query_bind, &interv.causal_bind);
                (interv, dist)
            })
            .filter(|(_, d)| *d < THREE_SIGMA)
            .collect();

        results.sort_by_key(|(_, d)| *d);
        results.truncate(k);
        results
    }

    /// Curiosity-driven action selection
    ///
    /// Combines Q-value with novelty bonus: less-visited state-action pairs
    /// get an exploration boost (intrinsic motivation from ladybug-rs).
    pub fn select_action_curious(
        &self,
        state: &BitpackedVector,
        actions: &[BitpackedVector],
        curiosity_weight: f32,
    ) -> Option<usize> {
        if actions.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, action) in actions.iter().enumerate() {
            let q = self.q_value_causal(state, action);
            let hash = Self::hash_sa(state, action);
            let visits = *self.visit_counts.get(&hash).unwrap_or(&0);

            // Curiosity bonus: 1/(1 + visits) — unvisited pairs get max bonus
            let curiosity = curiosity_weight / (1.0 + visits as f32);
            let score = q + curiosity;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        Some(best_idx)
    }

    /// Compute regret for a past decision
    pub fn compute_regret(
        &self,
        state: &BitpackedVector,
        actual_action: &BitpackedVector,
        actual_reward: f32,
        alt_action: &BitpackedVector,
    ) -> f32 {
        let alt_q = self.q_value_causal(state, alt_action);
        actual_reward - alt_q
    }

    /// Trace causal chain: follow interventions forward from initial state
    pub fn trace_causal_chain(
        &self,
        initial_state: &BitpackedVector,
        max_depth: usize,
    ) -> Vec<CausalChainLink> {
        let mut chain = Vec::new();
        let mut current_state = initial_state.clone();

        for _ in 0..max_depth {
            // Find best intervention from current state
            let mut best_interv: Option<&Intervention> = None;
            let mut best_dist = u32::MAX;

            for interv in &self.interventions {
                let dist = hamming_distance_scalar(&current_state, &interv.state);
                if dist < best_dist {
                    best_dist = dist;
                    best_interv = Some(interv);
                }
            }

            match best_interv {
                Some(interv) if best_dist < THREE_SIGMA => {
                    chain.push(CausalChainLink {
                        state: current_state.clone(),
                        action: interv.action.clone(),
                        outcome: interv.outcome.clone(),
                        reward: interv.reward,
                        confidence: 1.0 - (best_dist as f32 / VECTOR_BITS as f32),
                    });
                    current_state = interv.outcome.clone();
                }
                _ => break,
            }
        }

        chain
    }

    /// Number of stored interventions
    pub fn num_interventions(&self) -> usize {
        self.interventions.len()
    }

    /// Number of stored counterfactuals
    pub fn num_counterfactuals(&self) -> usize {
        self.counterfactuals.len()
    }
}

impl Default for CausalRlAgent {
    fn default() -> Self {
        Self::new(0.99, 0.1, 0.1)
    }
}

/// One link in a causal chain
#[derive(Clone, Debug)]
pub struct CausalChainLink {
    /// State at this step
    pub state: BitpackedVector,
    /// Action taken
    pub action: BitpackedVector,
    /// Resulting outcome
    pub outcome: BitpackedVector,
    /// Reward received
    pub reward: f32,
    /// Confidence in this link (similarity-based)
    pub confidence: f32,
}

// ============================================================================
// NEURAL PLASTICITY: Synaptic-style weight updates on crystal cells
// ============================================================================

/// Spike-Timing Dependent Plasticity (STDP) for crystal cells.
///
/// When cell A fires before cell B within a time window, strengthen A→B.
/// When cell B fires before cell A, weaken A→B.
/// This creates directional associations in the crystal lattice.
pub struct StdpRule {
    /// Time window for potentiation (positive Δt)
    pub potentiation_window: u32,
    /// Time window for depression (negative Δt)
    pub depression_window: u32,
    /// Potentiation learning rate (A+)
    pub a_plus: f32,
    /// Depression learning rate (A-)
    pub a_minus: f32,
    /// Time constant for exponential decay
    pub tau: f32,
}

impl StdpRule {
    pub fn new() -> Self {
        Self {
            potentiation_window: 20,
            depression_window: 20,
            a_plus: 0.01,
            a_minus: 0.012, // Slightly stronger depression (homeostasis)
            tau: 10.0,
        }
    }

    /// Compute weight change given spike timing difference (Δt = t_post - t_pre)
    pub fn weight_change(&self, delta_t: i32) -> f32 {
        if delta_t > 0 && (delta_t as u32) < self.potentiation_window {
            // Pre before post: potentiate (LTP)
            self.a_plus * (-delta_t as f32 / self.tau).exp()
        } else if delta_t < 0 && ((-delta_t) as u32) < self.depression_window {
            // Post before pre: depress (LTD)
            -self.a_minus * (delta_t as f32 / self.tau).exp()
        } else {
            0.0
        }
    }
}

impl Default for StdpRule {
    fn default() -> Self {
        Self::new()
    }
}

/// Neural plasticity engine: combines STDP, Hebbian, and homeostatic plasticity
pub struct PlasticityEngine {
    /// Hebbian co-activation matrix
    pub hebbian: HebbianMatrix,
    /// STDP rule for directional learning
    pub stdp: StdpRule,
    /// Per-cell firing timestamps (last fire time)
    fire_times: HashMap<usize, u32>,
    /// Global timestep
    timestep: u32,
    /// Homeostatic target: desired average firing rate
    target_rate: f32,
    /// Homeostatic scaling factor per cell
    scaling: HashMap<usize, f32>,
}

impl PlasticityEngine {
    pub fn new() -> Self {
        Self {
            hebbian: HebbianMatrix::new(0.05, 0.999),
            stdp: StdpRule::new(),
            fire_times: HashMap::new(),
            timestep: 0,
            target_rate: 0.1,
            scaling: HashMap::new(),
        }
    }

    /// Record a cell firing event
    pub fn fire(&mut self, cell: usize) {
        let now = self.timestep;

        // STDP: update weights with all recently-fired cells
        for (&other_cell, &fire_time) in &self.fire_times {
            if other_cell == cell {
                continue;
            }
            let delta_t = now as i32 - fire_time as i32;
            let dw = self.stdp.weight_change(delta_t);
            if dw.abs() > 0.0001 {
                // Directed: cell fired after other_cell → potentiate other→cell
                if dw > 0.0 {
                    self.hebbian.fire_together(other_cell, cell);
                }
                // TODO: directional hebbian (asymmetric matrix) for depression
            }
        }

        self.fire_times.insert(cell, now);

        // Homeostatic scaling: track firing rate
        let scale = self.scaling.entry(cell).or_insert(1.0);
        *scale *= 0.99; // Decay toward target rate
        *scale += self.target_rate * 0.01;
    }

    /// Advance timestep
    pub fn tick(&mut self) {
        self.timestep += 1;

        // Prune old fire times
        let cutoff = self.timestep.saturating_sub(100);
        self.fire_times.retain(|_, &mut t| t >= cutoff);

        // Periodic Hebbian decay
        if self.timestep % 100 == 0 {
            self.hebbian.decay_step();
        }
    }

    /// Get homeostatic scaling factor for a cell
    pub fn scale(&self, cell: usize) -> f32 {
        *self.scaling.get(&cell).unwrap_or(&1.0)
    }

    /// Get STDP-modified connection strength between cells
    pub fn connection(&self, pre: usize, post: usize) -> f32 {
        let base = self.hebbian.strength(pre, post);
        let pre_scale = self.scale(pre);
        let post_scale = self.scale(post);
        base * pre_scale * post_scale
    }
}

impl Default for PlasticityEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_signal_positive() {
        let query = BitpackedVector::random(42);
        let target = BitpackedVector::random(43);
        let original_dist = hamming_distance_scalar(&query, &target);

        let signal = RewardSignal::from_scalar(&query, &target, 0.5);
        assert!(signal.value > 0.0);

        // Applying positive reward should bring target closer to query
        let modified = signal.apply(&target);
        let new_dist = hamming_distance_scalar(&query, &modified);
        assert!(
            new_dist <= original_dist,
            "Positive reward should reduce distance: {} -> {}",
            original_dist,
            new_dist
        );
    }

    #[test]
    fn test_reward_signal_negative() {
        let query = BitpackedVector::random(42);
        let target = BitpackedVector::random(43);
        let original_dist = hamming_distance_scalar(&query, &target);

        let signal = RewardSignal::from_scalar(&query, &target, -0.5);
        assert!(signal.value < 0.0);

        // Applying negative reward should push target away from query
        let modified = signal.apply(&target);
        let new_dist = hamming_distance_scalar(&query, &modified);
        assert!(
            new_dist >= original_dist,
            "Negative reward should increase distance: {} -> {}",
            original_dist,
            new_dist
        );
    }

    #[test]
    fn test_hebbian_matrix() {
        let mut hebb = HebbianMatrix::new(0.1, 0.99);

        // Fire cells together
        hebb.fire_together(10, 20);
        hebb.fire_together(10, 20);
        hebb.fire_together(10, 30);

        // 10-20 should be stronger than 10-30
        assert!(hebb.strength(10, 20) > hebb.strength(10, 30));

        // Strongest connections for cell 10
        let conns = hebb.strongest_connections(10, 5);
        assert_eq!(conns.len(), 2);
        assert_eq!(conns[0].0, 20); // Strongest first
    }

    #[test]
    fn test_hebbian_fire_set() {
        let mut hebb = HebbianMatrix::new(0.1, 0.99);

        hebb.fire_set(&[1, 2, 3, 4]);
        // All pairs should be connected
        assert!(hebb.strength(1, 2) > 0.0);
        assert!(hebb.strength(1, 4) > 0.0);
        assert!(hebb.strength(2, 3) > 0.0);
        assert!(hebb.strength(3, 4) > 0.0);
        assert_eq!(hebb.num_connections(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_hebbian_decay() {
        let mut hebb = HebbianMatrix::new(1.0, 0.5);
        hebb.fire_together(1, 2);

        let before = hebb.strength(1, 2);
        hebb.decay_step();
        let after = hebb.strength(1, 2);

        assert!(after < before);
    }

    #[test]
    fn test_policy_gradient() {
        let mut policy = PolicyGradient::with_params(0.1, 0.9, 0.0); // No exploration

        // Record some experience
        let state = SearchState::InBand(SigmaBand::Inner);
        policy.record(state.clone(), SearchAction::Explore);
        policy.end_episode(1.0);

        // Should prefer Explore for Inner band now
        let action = policy.best_action(&state);
        assert!(matches!(action, SearchAction::Explore));
    }

    #[test]
    fn test_policy_anneal() {
        let mut policy = PolicyGradient::with_params(0.1, 0.9, 0.5);
        assert_eq!(policy.epsilon, 0.5);

        policy.anneal(0.5);
        assert!((policy.epsilon - 0.25).abs() < 0.001);

        // Should not go below minimum
        for _ in 0..100 {
            policy.anneal(0.5);
        }
        assert!(policy.epsilon >= 0.01);
    }

    #[test]
    fn test_reward_tracker() {
        let mut tracker = RewardTracker::new(0.9);

        // Propagate reward along path
        let path = vec!["root".into(), "child1".into(), "leaf3".into()];
        tracker.propagate_reward(&path, 1.0);

        // Leaf (closest to reward) should get most credit
        let leaf_reward = tracker.avg_reward("leaf3");
        let root_reward = tracker.avg_reward("root");
        assert!(leaf_reward > root_reward);
    }

    #[test]
    fn test_ucb1() {
        let mut tracker = RewardTracker::new(0.9);

        // Visited node
        tracker.propagate_reward(&["node_a".into()], 0.5);
        tracker.propagate_reward(&["node_a".into()], 0.5);
        tracker.propagate_reward(&["node_a".into()], 0.5);

        // Unvisited node
        let ucb_visited = tracker.ucb1("node_a", 1.0);
        let ucb_unvisited = tracker.ucb1("node_b", 1.0);

        // UCB1 should favor unvisited (high exploration bonus)
        assert!(
            ucb_unvisited > ucb_visited,
            "UCB1 should favor unvisited: {} vs {}",
            ucb_unvisited,
            ucb_visited
        );
    }

    #[test]
    fn test_rl_engine_unified() {
        let mut engine = RlEngine::new();

        let query = BitpackedVector::random(42);
        let result = BitpackedVector::random(43);
        let path = vec!["root".into(), "child".into(), "leaf".into()];
        let query_crystal = Coord5D::new(2, 2, 2, 2, 2);
        let result_crystal = Coord5D::new(2, 3, 2, 2, 2);
        let block_sig = [100u16; 10];

        // Positive reward
        engine.reward_search(
            &query,
            &result,
            1.0,
            &path,
            query_crystal,
            result_crystal,
            &block_sig,
        );

        // Check Hebbian connections were created
        assert!(engine.hebbian.num_connections() > 0);

        // Check crystal rewards
        let best_cells = engine.best_crystal_cells(5);
        assert!(!best_cells.is_empty());

        let stats = engine.stats();
        assert!(stats.hebbian_connections > 0);
        assert!(stats.rewarded_crystals > 0);
        println!("{}", stats);
    }

    #[test]
    fn test_block_weight_adjustment() {
        let mut engine = RlEngine::new();

        let query = BitpackedVector::random(1);
        let result = BitpackedVector::random(2);
        let path = vec!["root".into()];

        // High activation in block 3
        let mut block_sig = [50u16; 10];
        block_sig[3] = 500;

        // Positive reward should boost block 3
        let before = engine.block_weights[3];
        engine.reward_search(
            &query,
            &result,
            1.0,
            &path,
            Coord5D::new(2, 2, 2, 2, 2),
            Coord5D::new(2, 2, 2, 2, 2),
            &block_sig,
        );
        let after = engine.block_weights[3];

        assert!(after > before, "Block 3 weight should increase: {} -> {}", before, after);
    }

    // ================================================================
    // Causal RL tests
    // ================================================================

    #[test]
    fn test_causal_intervention() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.1);

        let state = BitpackedVector::random(42);
        let action = BitpackedVector::random(43);
        let outcome = BitpackedVector::random(44);

        agent.store_intervention(state.clone(), action.clone(), outcome, 1.0);
        assert_eq!(agent.num_interventions(), 1);

        // Q-value should be non-zero now
        let q = agent.q_value_causal(&state, &action);
        assert!(q > 0.0, "Q-value should be positive after positive intervention");
    }

    #[test]
    fn test_causal_counterfactual() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.1);

        let state = BitpackedVector::random(1);
        let actual_action = BitpackedVector::random(2);
        let alt_action = BitpackedVector::random(3);
        let alt_outcome = BitpackedVector::random(4);

        agent.store_counterfactual(state, alt_action, alt_outcome, 0.5, 1.0);
        assert_eq!(agent.num_counterfactuals(), 1);
    }

    #[test]
    fn test_curiosity_selection() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.0); // No epsilon

        let state = BitpackedVector::random(10);
        let visited_action = BitpackedVector::random(20);
        let novel_action = BitpackedVector::random(30);

        // Visit one action many times
        for _ in 0..10 {
            agent.store_intervention(
                state.clone(),
                visited_action.clone(),
                BitpackedVector::random(99),
                0.5,
            );
        }

        let actions = vec![visited_action.clone(), novel_action.clone()];
        let selected = agent.select_action_curious(&state, &actions, 1.0);

        // With curiosity weight, should prefer novel action
        assert!(selected.is_some());
        // Novel action (index 1) should be preferred due to curiosity bonus
        assert_eq!(selected.unwrap(), 1);
    }

    #[test]
    fn test_causal_chain_trace() {
        let mut agent = CausalRlAgent::new(0.99, 0.1, 0.1);

        // Create a chain: S0 --A0--> S1 --A1--> S2
        let s0 = BitpackedVector::random(100);
        let a0 = BitpackedVector::random(200);
        let s1 = BitpackedVector::random(300);
        let a1 = BitpackedVector::random(400);
        let s2 = BitpackedVector::random(500);

        agent.store_intervention(s0.clone(), a0, s1.clone(), 0.5);
        agent.store_intervention(s1.clone(), a1, s2, 1.0);

        let chain = agent.trace_causal_chain(&s0, 5);
        assert!(!chain.is_empty());
        assert!(chain[0].confidence > 0.5);
    }

    // ================================================================
    // STDP and Plasticity tests
    // ================================================================

    #[test]
    fn test_stdp_potentiation() {
        let stdp = StdpRule::new();

        // Pre fires before post (delta_t > 0): should potentiate
        let dw = stdp.weight_change(5);
        assert!(dw > 0.0, "Pre-before-post should potentiate: dw={}", dw);
    }

    #[test]
    fn test_stdp_depression() {
        let stdp = StdpRule::new();

        // Post fires before pre (delta_t < 0): should depress
        let dw = stdp.weight_change(-5);
        assert!(dw < 0.0, "Post-before-pre should depress: dw={}", dw);
    }

    #[test]
    fn test_stdp_decay() {
        let stdp = StdpRule::new();

        // Larger timing difference → smaller weight change
        let close = stdp.weight_change(2).abs();
        let far = stdp.weight_change(15).abs();
        assert!(close > far, "Closer timing should give larger change");
    }

    #[test]
    fn test_plasticity_engine() {
        let mut engine = PlasticityEngine::new();

        // Fire cells in sequence
        engine.fire(10);
        engine.tick();
        engine.fire(20);
        engine.tick();
        engine.fire(30);

        // 10 fired before 20: should have Hebbian connection
        let strength = engine.hebbian.strength(10, 20);
        assert!(strength > 0.0, "Sequential firing should create connection");

        // Homeostatic scaling should be near 1.0
        let scale = engine.scale(10);
        assert!((scale - 1.0).abs() < 0.5);
    }
}
