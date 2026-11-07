//! skmem: the Rust engine for skmemory (strangler plan).
//!
//! Storage core: the schema v2 [`model::Memory`] and the flat JSON
//! [`flat::Store`], byte-compatible with the Python implementation.

pub mod audience;
pub mod bench;
pub mod decay;
pub mod embed;
pub mod env;
pub mod flat;
pub mod fusion;
pub mod index;
pub mod intentions;
pub mod ledger;
pub mod model;
#[cfg(feature = "pg")]
pub mod pg;
mod printable_table;
pub mod procedures;
pub mod pyjson;
pub mod pytime;
pub mod query_sanitizer;
pub mod rerank;
pub mod search;
mod sys;
pub mod tiers;
mod util;
pub mod validity;
mod wordchars_table;

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
