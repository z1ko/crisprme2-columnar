//! Optional extension modules, gated behind cargo features.

/// Zero-copy Python integration via the PEP 3118 buffer protocol.
#[cfg(feature = "pyo3")]
pub mod pyo3;