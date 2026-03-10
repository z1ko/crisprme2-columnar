//! Zero-copy, cache-friendly Structure-of-Arrays (SoA) columnar buffers.
//!
//! This crate provides a typed columnar buffer ([`buffer::ColumnarBuffer`])
//! that stores rows of data in column-major order for optimal SIMD
//! vectorisation, cache efficiency, and zero-copy interop with external
//! systems (Arrow, Polars, CUDA, Python/NumPy).
//!
//! # Crate layout
//!
//! | Module | Purpose |
//! |---|---|
//! | [`buffer`] | Core types: [`Schema`](buffer::Schema), [`ColumnarBuffer`](buffer::ColumnarBuffer), [`ColumnIdx`](buffer::ColumnIdx), [`ColumnGroupIdx`](buffer::ColumnGroupIdx) |
//! | [`ring`] | Memory pooling ([`Pool`](ring::Pool)) and batch passing ([`BatchMut`](ring::BatchMut), [`BatchRef`](ring::BatchRef), [`connector`](ring::connector)) |
//! | [`pipeline`] | Multi-stage processing pipeline built on top of `ring` |
//! | [`ext`] | Optional extensions (currently: PyO3 buffer-protocol integration) |
//! | [`macros`] | Re-exports `#[derive(Columnar)]` from `columnar-derive` |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use columnar::{macros::Columnar, buffer::*};
//!
//! #[repr(C)]
//! #[derive(Columnar)]
//! pub struct Sequence {
//!     pub id:    u64,
//!     pub score: f32,
//!     #[columnar(group)]
//!     pub elements: [u8; 32],
//! }
//!
//! let mut buf: ColumnarBuffer<SequenceSchema, AlignedBox> =
//!     AlignedBox::new(1024 * SequenceSchema::LAYOUT.stride).columnar();
//!
//! buf.push(Sequence { id: 1, score: 0.95, elements: [0u8; 32] });
//! let seq: Sequence = buf.get(0).unwrap();
//! ```

#[cfg(test)]
extern crate self as columnar;
#[cfg(test)]
mod tests;

/// Re-exports `#[derive(Columnar)]` from the `columnar-derive` proc-macro crate.
pub mod macros {
    pub use columnar_derive::*;
}

pub mod buffer;
pub mod ring;
pub mod pipeline;

/// Optional extension modules (feature-gated).
pub mod ext;
