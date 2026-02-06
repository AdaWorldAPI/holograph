//! GraphBLAS Descriptor
//!
//! Controls operation modifiers like transpose, complement mask, etc.

/// GraphBLAS Descriptor
///
/// Modifies how operations are performed:
/// - Transpose input/output
/// - Complement mask
/// - Replace vs merge output
/// - Structural mask (only pattern, not values)
#[derive(Clone, Debug, Default)]
pub struct Descriptor {
    /// Transpose first input matrix
    pub inp0: DescField,
    /// Transpose second input matrix
    pub inp1: DescField,
    /// Mask handling
    pub mask: DescField,
    /// Output handling
    pub outp: DescField,
}

/// Descriptor field value
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DescField {
    /// Default (no modification)
    #[default]
    Default,
    /// Transpose (for matrices)
    Transpose,
    /// Complement (for masks)
    Complement,
    /// Replace output (clear before write)
    Replace,
    /// Structural only (for masks)
    Structure,
}

impl Descriptor {
    /// Create default descriptor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set first input to transpose
    pub fn transpose_inp0(mut self) -> Self {
        self.inp0 = DescField::Transpose;
        self
    }

    /// Set second input to transpose
    pub fn transpose_inp1(mut self) -> Self {
        self.inp1 = DescField::Transpose;
        self
    }

    /// Set mask to complement
    pub fn complement_mask(mut self) -> Self {
        self.mask = DescField::Complement;
        self
    }

    /// Set mask to structural
    pub fn structural_mask(mut self) -> Self {
        self.mask = DescField::Structure;
        self
    }

    /// Set output to replace mode
    pub fn replace_output(mut self) -> Self {
        self.outp = DescField::Replace;
        self
    }

    /// Check if inp0 should be transposed
    pub fn is_inp0_transposed(&self) -> bool {
        self.inp0 == DescField::Transpose
    }

    /// Check if inp1 should be transposed
    pub fn is_inp1_transposed(&self) -> bool {
        self.inp1 == DescField::Transpose
    }

    /// Check if mask should be complemented
    pub fn is_mask_complemented(&self) -> bool {
        self.mask == DescField::Complement
    }

    /// Check if mask is structural
    pub fn is_mask_structural(&self) -> bool {
        self.mask == DescField::Structure
    }

    /// Check if output should be replaced
    pub fn should_replace_output(&self) -> bool {
        self.outp == DescField::Replace
    }
}

/// Common descriptor presets
pub mod GrBDesc {
    use super::*;

    /// Default descriptor
    pub fn default() -> Descriptor {
        Descriptor::new()
    }

    /// Transpose first input
    pub fn t0() -> Descriptor {
        Descriptor::new().transpose_inp0()
    }

    /// Transpose second input
    pub fn t1() -> Descriptor {
        Descriptor::new().transpose_inp1()
    }

    /// Transpose both inputs
    pub fn t0t1() -> Descriptor {
        Descriptor::new().transpose_inp0().transpose_inp1()
    }

    /// Complement mask
    pub fn c() -> Descriptor {
        Descriptor::new().complement_mask()
    }

    /// Replace output
    pub fn r() -> Descriptor {
        Descriptor::new().replace_output()
    }

    /// Structural mask
    pub fn s() -> Descriptor {
        Descriptor::new().structural_mask()
    }

    /// Replace output and complement mask
    pub fn rc() -> Descriptor {
        Descriptor::new().replace_output().complement_mask()
    }

    /// Replace output and structural mask
    pub fn rs() -> Descriptor {
        Descriptor::new().replace_output().structural_mask()
    }

    /// Replace, structural, complement
    pub fn rsc() -> Descriptor {
        Descriptor::new()
            .replace_output()
            .structural_mask()
            .complement_mask()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor() {
        let desc = Descriptor::new()
            .transpose_inp0()
            .complement_mask()
            .replace_output();

        assert!(desc.is_inp0_transposed());
        assert!(!desc.is_inp1_transposed());
        assert!(desc.is_mask_complemented());
        assert!(desc.should_replace_output());
    }

    #[test]
    fn test_presets() {
        let t0 = GrBDesc::t0();
        assert!(t0.is_inp0_transposed());

        let rc = GrBDesc::rc();
        assert!(rc.should_replace_output());
        assert!(rc.is_mask_complemented());
    }
}
