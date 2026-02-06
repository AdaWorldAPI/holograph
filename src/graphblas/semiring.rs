//! GraphBLAS Semirings for HDR Computing
//!
//! A semiring (⊕, ⊗) provides:
//! - ⊕ (add): Associative, commutative, with identity 0
//! - ⊗ (multiply): Associative, distributes over ⊕, with identity 1
//! - 0 annihilates: a ⊗ 0 = 0 ⊗ a = 0
//!
//! ## HDR Semirings
//!
//! | Semiring | ⊕ (Add) | ⊗ (Multiply) | Identity | Zero | Use Case |
//! |----------|---------|--------------|----------|------|----------|
//! | XOR_BUNDLE | Bundle | XOR | zero_vec | - | Path composition |
//! | BIND_FIRST | First | XOR | empty | empty | Single traversal |
//! | HAMMING_MIN | Min | Hamming | ∞ | - | Shortest path |
//! | SIMILARITY_MAX | Max | Similarity | 0.0 | - | Best match |
//! | RESONANCE | BestMatch | Bind | empty | empty | Query expansion |

use crate::bitpack::BitpackedVector;
use crate::hamming::{hamming_distance_scalar, hamming_to_similarity};
use super::types::{HdrScalar, GrBMonoid, GrBBinaryOp};

/// A semiring defines the algebraic operations for matrix computation
pub trait Semiring: Clone + Send + Sync {
    /// The element type
    type Element: Clone + Send + Sync;

    /// Additive identity (0)
    fn zero(&self) -> Self::Element;

    /// Multiplicative identity (1)
    fn one(&self) -> Self::Element;

    /// Addition operation (⊕)
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Multiplication operation (⊗)
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Check if element is zero
    fn is_zero(&self, a: &Self::Element) -> bool;

    /// Name of this semiring
    fn name(&self) -> &'static str;
}

/// HDR-specific semiring implementations
#[derive(Clone, Debug)]
pub enum HdrSemiring {
    /// XOR multiply, Bundle add
    /// Good for: path composition, multi-hop queries
    XorBundle,

    /// XOR multiply, First non-empty add
    /// Good for: BFS traversal, single path finding
    BindFirst,

    /// Hamming distance multiply, Min add
    /// Good for: shortest semantic path
    HammingMin,

    /// Similarity multiply, Max add
    /// Good for: best match finding
    SimilarityMax,

    /// Bind multiply, Best resonance add
    /// Good for: query expansion with cleanup
    Resonance {
        threshold: f32,
    },

    /// AND multiply, OR add (traditional boolean)
    /// Good for: reachability queries
    BooleanAndOr,

    /// XOR multiply, XOR add (field arithmetic)
    /// Good for: algebraic path counting mod 2
    XorXor,

    /// Custom semiring with user-defined operations
    Custom {
        name: String,
        add_op: GrBBinaryOp,
        mult_op: GrBBinaryOp,
    },
}

impl Default for HdrSemiring {
    fn default() -> Self {
        HdrSemiring::XorBundle
    }
}

impl Semiring for HdrSemiring {
    type Element = HdrScalar;

    fn zero(&self) -> HdrScalar {
        match self {
            HdrSemiring::XorBundle => HdrScalar::Vector(BitpackedVector::zero()),
            HdrSemiring::BindFirst => HdrScalar::Empty,
            HdrSemiring::HammingMin => HdrScalar::Distance(u32::MAX),
            HdrSemiring::SimilarityMax => HdrScalar::Similarity(0.0),
            HdrSemiring::Resonance { .. } => HdrScalar::Empty,
            HdrSemiring::BooleanAndOr => HdrScalar::Bool(false),
            HdrSemiring::XorXor => HdrScalar::Vector(BitpackedVector::zero()),
            HdrSemiring::Custom { .. } => HdrScalar::Empty,
        }
    }

    fn one(&self) -> HdrScalar {
        match self {
            HdrSemiring::XorBundle => HdrScalar::Vector(BitpackedVector::zero()), // XOR identity
            HdrSemiring::BindFirst => HdrScalar::Vector(BitpackedVector::zero()),
            HdrSemiring::HammingMin => HdrScalar::Distance(0),
            HdrSemiring::SimilarityMax => HdrScalar::Similarity(1.0),
            HdrSemiring::Resonance { .. } => HdrScalar::Vector(BitpackedVector::zero()),
            HdrSemiring::BooleanAndOr => HdrScalar::Bool(true),
            HdrSemiring::XorXor => HdrScalar::Vector(BitpackedVector::zero()),
            HdrSemiring::Custom { .. } => HdrScalar::Empty,
        }
    }

    fn add(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
        match self {
            HdrSemiring::XorBundle => {
                // Bundle: majority voting over vectors
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                        HdrScalar::Vector(BitpackedVector::bundle(&[va, vb]))
                    }
                    (HdrScalar::Vector(v), HdrScalar::Empty) |
                    (HdrScalar::Empty, HdrScalar::Vector(v)) => {
                        HdrScalar::Vector(v.clone())
                    }
                    _ => HdrScalar::Empty,
                }
            }

            HdrSemiring::BindFirst => {
                // First non-empty
                if !a.is_empty() { a.clone() } else { b.clone() }
            }

            HdrSemiring::HammingMin => {
                // Minimum distance
                match (a, b) {
                    (HdrScalar::Distance(da), HdrScalar::Distance(db)) => {
                        HdrScalar::Distance((*da).min(*db))
                    }
                    (HdrScalar::Distance(d), _) | (_, HdrScalar::Distance(d)) => {
                        HdrScalar::Distance(*d)
                    }
                    _ => HdrScalar::Distance(u32::MAX),
                }
            }

            HdrSemiring::SimilarityMax => {
                // Maximum similarity
                match (a, b) {
                    (HdrScalar::Similarity(sa), HdrScalar::Similarity(sb)) => {
                        HdrScalar::Similarity(sa.max(*sb))
                    }
                    (HdrScalar::Similarity(s), _) | (_, HdrScalar::Similarity(s)) => {
                        HdrScalar::Similarity(*s)
                    }
                    _ => HdrScalar::Similarity(0.0),
                }
            }

            HdrSemiring::Resonance { threshold } => {
                // Best matching vector above threshold
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                        // In real use, would compare to query
                        // Here just keep the denser vector as proxy for "better"
                        if va.density() >= vb.density() {
                            HdrScalar::Vector(va.clone())
                        } else {
                            HdrScalar::Vector(vb.clone())
                        }
                    }
                    (HdrScalar::Vector(v), _) | (_, HdrScalar::Vector(v)) => {
                        HdrScalar::Vector(v.clone())
                    }
                    _ => HdrScalar::Empty,
                }
            }

            HdrSemiring::BooleanAndOr => {
                // Logical OR
                HdrScalar::Bool(a.to_bool() || b.to_bool())
            }

            HdrSemiring::XorXor => {
                // XOR add (field arithmetic)
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                        HdrScalar::Vector(va.xor(vb))
                    }
                    (HdrScalar::Vector(v), _) | (_, HdrScalar::Vector(v)) => {
                        HdrScalar::Vector(v.clone())
                    }
                    _ => HdrScalar::Vector(BitpackedVector::zero()),
                }
            }

            HdrSemiring::Custom { add_op, .. } => {
                apply_binary_op(*add_op, a, b)
            }
        }
    }

    fn multiply(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
        match self {
            HdrSemiring::XorBundle | HdrSemiring::BindFirst |
            HdrSemiring::XorXor | HdrSemiring::Resonance { .. } => {
                // XOR binding
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                        HdrScalar::Vector(va.xor(vb))
                    }
                    _ => HdrScalar::Empty,
                }
            }

            HdrSemiring::HammingMin => {
                // Hamming distance
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                        HdrScalar::Distance(hamming_distance_scalar(va, vb))
                    }
                    _ => HdrScalar::Distance(u32::MAX),
                }
            }

            HdrSemiring::SimilarityMax => {
                // Similarity score
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                        let dist = hamming_distance_scalar(va, vb);
                        HdrScalar::Similarity(hamming_to_similarity(dist))
                    }
                    _ => HdrScalar::Similarity(0.0),
                }
            }

            HdrSemiring::BooleanAndOr => {
                // Logical AND
                HdrScalar::Bool(a.to_bool() && b.to_bool())
            }

            HdrSemiring::Custom { mult_op, .. } => {
                apply_binary_op(*mult_op, a, b)
            }
        }
    }

    fn is_zero(&self, a: &HdrScalar) -> bool {
        match self {
            HdrSemiring::XorBundle | HdrSemiring::XorXor => {
                match a {
                    HdrScalar::Vector(v) => v.popcount() == 0,
                    HdrScalar::Empty => true,
                    _ => false,
                }
            }
            HdrSemiring::BindFirst | HdrSemiring::Resonance { .. } => {
                a.is_empty()
            }
            HdrSemiring::HammingMin => {
                matches!(a, HdrScalar::Distance(d) if *d == u32::MAX)
            }
            HdrSemiring::SimilarityMax => {
                matches!(a, HdrScalar::Similarity(s) if *s == 0.0)
            }
            HdrSemiring::BooleanAndOr => {
                !a.to_bool()
            }
            HdrSemiring::Custom { .. } => {
                a.is_empty()
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            HdrSemiring::XorBundle => "XOR_BUNDLE",
            HdrSemiring::BindFirst => "BIND_FIRST",
            HdrSemiring::HammingMin => "HAMMING_MIN",
            HdrSemiring::SimilarityMax => "SIMILARITY_MAX",
            HdrSemiring::Resonance { .. } => "RESONANCE",
            HdrSemiring::BooleanAndOr => "BOOLEAN_AND_OR",
            HdrSemiring::XorXor => "XOR_XOR",
            HdrSemiring::Custom { .. } => "CUSTOM",
        }
    }
}

/// Apply a binary operator
fn apply_binary_op(op: GrBBinaryOp, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
    match op {
        GrBBinaryOp::First => a.clone(),
        GrBBinaryOp::Second => b.clone(),

        GrBBinaryOp::HdrBind => {
            match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    HdrScalar::Vector(va.xor(vb))
                }
                _ => HdrScalar::Empty,
            }
        }

        GrBBinaryOp::HdrBundle => {
            match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    HdrScalar::Vector(BitpackedVector::bundle(&[va, vb]))
                }
                (HdrScalar::Vector(v), _) | (_, HdrScalar::Vector(v)) => {
                    HdrScalar::Vector(v.clone())
                }
                _ => HdrScalar::Empty,
            }
        }

        GrBBinaryOp::HdrHamming => {
            match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    HdrScalar::Distance(hamming_distance_scalar(va, vb))
                }
                _ => HdrScalar::Distance(u32::MAX),
            }
        }

        GrBBinaryOp::HdrSimilarity => {
            match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    let dist = hamming_distance_scalar(va, vb);
                    HdrScalar::Similarity(hamming_to_similarity(dist))
                }
                _ => HdrScalar::Similarity(0.0),
            }
        }

        GrBBinaryOp::Min => {
            match (a, b) {
                (HdrScalar::Distance(da), HdrScalar::Distance(db)) => {
                    HdrScalar::Distance((*da).min(*db))
                }
                (HdrScalar::Int(ia), HdrScalar::Int(ib)) => {
                    HdrScalar::Int((*ia).min(*ib))
                }
                (HdrScalar::Float(fa), HdrScalar::Float(fb)) => {
                    HdrScalar::Float(fa.min(*fb))
                }
                _ => a.clone(),
            }
        }

        GrBBinaryOp::Max => {
            match (a, b) {
                (HdrScalar::Similarity(sa), HdrScalar::Similarity(sb)) => {
                    HdrScalar::Similarity(sa.max(*sb))
                }
                (HdrScalar::Int(ia), HdrScalar::Int(ib)) => {
                    HdrScalar::Int((*ia).max(*ib))
                }
                (HdrScalar::Float(fa), HdrScalar::Float(fb)) => {
                    HdrScalar::Float(fa.max(*fb))
                }
                _ => a.clone(),
            }
        }

        GrBBinaryOp::Plus => {
            match (a, b) {
                (HdrScalar::Int(ia), HdrScalar::Int(ib)) => {
                    HdrScalar::Int(ia.wrapping_add(*ib))
                }
                (HdrScalar::Float(fa), HdrScalar::Float(fb)) => {
                    HdrScalar::Float(fa + fb)
                }
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    // Plus on vectors = bundle
                    HdrScalar::Vector(BitpackedVector::bundle(&[va, vb]))
                }
                _ => a.clone(),
            }
        }

        GrBBinaryOp::Times => {
            match (a, b) {
                (HdrScalar::Int(ia), HdrScalar::Int(ib)) => {
                    HdrScalar::Int(ia.wrapping_mul(*ib))
                }
                (HdrScalar::Float(fa), HdrScalar::Float(fb)) => {
                    HdrScalar::Float(fa * fb)
                }
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    // Times on vectors = AND
                    HdrScalar::Vector(va.and(vb))
                }
                _ => a.clone(),
            }
        }

        GrBBinaryOp::LOr => {
            HdrScalar::Bool(a.to_bool() || b.to_bool())
        }

        GrBBinaryOp::LAnd => {
            HdrScalar::Bool(a.to_bool() && b.to_bool())
        }

        GrBBinaryOp::LXor => {
            HdrScalar::Bool(a.to_bool() ^ b.to_bool())
        }

        GrBBinaryOp::Eq => {
            HdrScalar::Bool(a == b)
        }

        GrBBinaryOp::Ne => {
            HdrScalar::Bool(a != b)
        }

        _ => HdrScalar::Empty,
    }
}

/// Built-in semiring instances
pub mod semirings {
    use super::*;

    /// Standard path composition: XOR bind, Bundle add
    pub fn xor_bundle() -> HdrSemiring {
        HdrSemiring::XorBundle
    }

    /// BFS traversal: XOR bind, First add
    pub fn bind_first() -> HdrSemiring {
        HdrSemiring::BindFirst
    }

    /// Shortest semantic path: Hamming multiply, Min add
    pub fn hamming_min() -> HdrSemiring {
        HdrSemiring::HammingMin
    }

    /// Best match: Similarity multiply, Max add
    pub fn similarity_max() -> HdrSemiring {
        HdrSemiring::SimilarityMax
    }

    /// Query expansion with cleanup
    pub fn resonance(threshold: f32) -> HdrSemiring {
        HdrSemiring::Resonance { threshold }
    }

    /// Boolean reachability
    pub fn boolean() -> HdrSemiring {
        HdrSemiring::BooleanAndOr
    }

    /// GF(2) field arithmetic
    pub fn xor_field() -> HdrSemiring {
        HdrSemiring::XorXor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_bundle_semiring() {
        let sr = HdrSemiring::XorBundle;

        let a = HdrScalar::Vector(BitpackedVector::random(1));
        let b = HdrScalar::Vector(BitpackedVector::random(2));

        // Multiply = XOR
        let product = sr.multiply(&a, &b);
        assert!(matches!(product, HdrScalar::Vector(_)));

        // XOR is self-inverse
        if let HdrScalar::Vector(va) = &a {
            if let HdrScalar::Vector(prod) = &product {
                if let HdrScalar::Vector(vb) = &b {
                    let recovered = prod.xor(vb);
                    assert_eq!(va, &recovered);
                }
            }
        }

        // Zero is identity for add
        let zero = sr.zero();
        let sum = sr.add(&a, &zero);
        if let (HdrScalar::Vector(va), HdrScalar::Vector(vs)) = (&a, &sum) {
            // Bundle of [a, zero] ≈ a (majority wins)
            assert!(hamming_distance_scalar(va, vs) < 1000);
        }
    }

    #[test]
    fn test_hamming_min_semiring() {
        let sr = HdrSemiring::HammingMin;

        let a = HdrScalar::Vector(BitpackedVector::random(1));
        let b = HdrScalar::Vector(BitpackedVector::random(2));

        // Multiply = Hamming distance
        let dist = sr.multiply(&a, &b);
        assert!(matches!(dist, HdrScalar::Distance(_)));

        // Add = minimum
        let d1 = HdrScalar::Distance(100);
        let d2 = HdrScalar::Distance(200);
        let min = sr.add(&d1, &d2);
        assert_eq!(min, HdrScalar::Distance(100));
    }

    #[test]
    fn test_semiring_identity() {
        let sr = HdrSemiring::XorBundle;

        let a = HdrScalar::Vector(BitpackedVector::random(42));
        let one = sr.one();

        // a ⊗ 1 = a (for XOR, 1 = zero vector)
        let product = sr.multiply(&a, &one);
        if let (HdrScalar::Vector(va), HdrScalar::Vector(vp)) = (&a, &product) {
            assert_eq!(va, vp);
        }
    }
}
