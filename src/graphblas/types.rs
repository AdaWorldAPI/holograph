//! GraphBLAS Type Definitions for HDR
//!
//! Maps GraphBLAS types to HDR vector representations.

use crate::bitpack::BitpackedVector;
use std::any::TypeId;

/// GraphBLAS index type
pub type GrBIndex = u64;

/// Marker for "all indices"
pub const GRB_ALL: GrBIndex = u64::MAX;

/// HDR scalar type - our "numbers" are vectors
#[derive(Clone, Debug, PartialEq)]
pub enum HdrScalar {
    /// Empty/null value
    Empty,
    /// A bitpacked vector (the fundamental type)
    Vector(BitpackedVector),
    /// Hamming distance (result of comparison)
    Distance(u32),
    /// Similarity score (0.0 to 1.0)
    Similarity(f32),
    /// Boolean (for masks)
    Bool(bool),
    /// Integer (for counts, indices)
    Int(i64),
    /// Float (for weights, scores)
    Float(f64),
}

impl Default for HdrScalar {
    fn default() -> Self {
        HdrScalar::Empty
    }
}

impl HdrScalar {
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        matches!(self, HdrScalar::Empty)
    }

    /// Try to get as vector
    pub fn as_vector(&self) -> Option<&BitpackedVector> {
        match self {
            HdrScalar::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as distance
    pub fn as_distance(&self) -> Option<u32> {
        match self {
            HdrScalar::Distance(d) => Some(*d),
            _ => None,
        }
    }

    /// Try to get as similarity
    pub fn as_similarity(&self) -> Option<f32> {
        match self {
            HdrScalar::Similarity(s) => Some(*s),
            _ => None,
        }
    }

    /// Convert to boolean (for masks)
    pub fn to_bool(&self) -> bool {
        match self {
            HdrScalar::Empty => false,
            HdrScalar::Vector(_) => true,
            HdrScalar::Distance(d) => *d > 0,
            HdrScalar::Similarity(s) => *s > 0.0,
            HdrScalar::Bool(b) => *b,
            HdrScalar::Int(i) => *i != 0,
            HdrScalar::Float(f) => *f != 0.0,
        }
    }
}

impl From<BitpackedVector> for HdrScalar {
    fn from(v: BitpackedVector) -> Self {
        HdrScalar::Vector(v)
    }
}

impl From<u32> for HdrScalar {
    fn from(d: u32) -> Self {
        HdrScalar::Distance(d)
    }
}

impl From<f32> for HdrScalar {
    fn from(s: f32) -> Self {
        HdrScalar::Similarity(s)
    }
}

impl From<bool> for HdrScalar {
    fn from(b: bool) -> Self {
        HdrScalar::Bool(b)
    }
}

/// GraphBLAS type descriptor
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrBType {
    /// Boolean
    Bool,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// HDR bitpacked vector (our primary type)
    HdrVector,
    /// Hamming distance
    HdrDistance,
    /// Similarity score
    HdrSimilarity,
    /// User-defined type
    UserDefined(u64),
}

impl GrBType {
    /// Size in bytes (for traditional types)
    pub fn size(&self) -> usize {
        match self {
            GrBType::Bool => 1,
            GrBType::Int8 | GrBType::UInt8 => 1,
            GrBType::Int16 | GrBType::UInt16 => 2,
            GrBType::Int32 | GrBType::UInt32 | GrBType::Float32 => 4,
            GrBType::Int64 | GrBType::UInt64 | GrBType::Float64 => 8,
            GrBType::HdrVector => crate::bitpack::VECTOR_BYTES,
            GrBType::HdrDistance => 4,
            GrBType::HdrSimilarity => 4,
            GrBType::UserDefined(_) => 0, // Unknown
        }
    }

    /// Check if this is an HDR type
    pub fn is_hdr(&self) -> bool {
        matches!(self, GrBType::HdrVector | GrBType::HdrDistance | GrBType::HdrSimilarity)
    }
}

/// Unary operator types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrBUnaryOp {
    /// Identity
    Identity,
    /// Additive inverse (for vectors: NOT)
    AInv,
    /// Multiplicative inverse (for vectors: NOT)
    MInv,
    /// Logical NOT
    LNot,
    /// Absolute value (for vectors: popcount)
    Abs,
    /// One (for vectors: ones vector)
    One,
    /// HDR: Compute density
    HdrDensity,
    /// HDR: Normalize to unit vector
    HdrNormalize,
    /// HDR: Permute left by 1
    HdrPermute,
}

/// Binary operator types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrBBinaryOp {
    /// First argument
    First,
    /// Second argument
    Second,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Addition (for vectors: bundle/OR)
    Plus,
    /// Subtraction (for vectors: XOR with NOT)
    Minus,
    /// Multiplication (for vectors: AND)
    Times,
    /// Division (not applicable for vectors)
    Div,
    /// Logical OR
    LOr,
    /// Logical AND
    LAnd,
    /// Logical XOR
    LXor,
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Less than
    Lt,
    /// Greater or equal
    Ge,
    /// Less or equal
    Le,
    /// HDR: XOR bind
    HdrBind,
    /// HDR: Majority bundle
    HdrBundle,
    /// HDR: Hamming distance
    HdrHamming,
    /// HDR: Similarity
    HdrSimilarity,
    /// HDR: Resonance (best match)
    HdrResonance,
}

/// Monoid types (associative binary op with identity)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrBMonoid {
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Addition (for vectors: bundle)
    Plus,
    /// Multiplication (for vectors: AND)
    Times,
    /// Logical OR
    LOr,
    /// Logical AND
    LAnd,
    /// Logical XOR
    LXor,
    /// HDR: Bundle monoid (majority voting)
    HdrBundle,
    /// HDR: First non-empty
    HdrFirst,
    /// HDR: Best similarity
    HdrBestMatch,
}

impl GrBMonoid {
    /// Get identity element for this monoid
    pub fn identity(&self) -> HdrScalar {
        match self {
            GrBMonoid::Min => HdrScalar::Int(i64::MAX),
            GrBMonoid::Max => HdrScalar::Int(i64::MIN),
            GrBMonoid::Plus => HdrScalar::Int(0),
            GrBMonoid::Times => HdrScalar::Int(1),
            GrBMonoid::LOr => HdrScalar::Bool(false),
            GrBMonoid::LAnd => HdrScalar::Bool(true),
            GrBMonoid::LXor => HdrScalar::Bool(false),
            GrBMonoid::HdrBundle => HdrScalar::Empty,
            GrBMonoid::HdrFirst => HdrScalar::Empty,
            GrBMonoid::HdrBestMatch => HdrScalar::Similarity(0.0),
        }
    }
}

/// Select operator for thresholding
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrBSelectOp {
    /// Select entries equal to threshold
    Tril,
    /// Upper triangular
    Triu,
    /// Diagonal
    Diag,
    /// Off-diagonal
    OffDiag,
    /// Non-zero
    NonZero,
    /// Equal to value
    EqValue,
    /// Not equal to value
    NeValue,
    /// Greater than value
    GtValue,
    /// Greater or equal
    GeValue,
    /// Less than value
    LtValue,
    /// Less or equal
    LeValue,
    /// HDR: Similarity above threshold
    HdrSimilarTo,
    /// HDR: Distance below threshold
    HdrCloserThan,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr_scalar() {
        let v = BitpackedVector::random(42);
        let scalar = HdrScalar::from(v.clone());

        assert!(!scalar.is_empty());
        assert!(scalar.as_vector().is_some());
        assert!(scalar.to_bool());
    }

    #[test]
    fn test_grb_type_size() {
        assert_eq!(GrBType::Bool.size(), 1);
        assert_eq!(GrBType::Int64.size(), 8);
        assert_eq!(GrBType::HdrVector.size(), crate::bitpack::VECTOR_BYTES);
    }
}
