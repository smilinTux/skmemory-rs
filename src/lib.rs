//! skmem: the Rust engine for skmemory (strangler plan).
//!
//! Bootstrap only. The engine moves here in later cards; this crate currently
//! exposes its own version so the build, test and wheel pipeline can be proven.

/// The crate version, as declared in `Cargo.toml`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    /// Return the engine version.
    #[pyfunction]
    fn version() -> &'static str {
        super::version()
    }

    #[pymodule]
    #[pyo3(name = "_skmem_core")]
    fn skmem_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add("__version__", super::version())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
        assert!(!version().is_empty());
    }
}
