//! Derive macro for the `columnar` crate.
//!
//! Provides `#[derive(Columnar)]` which generates a cache-friendly
//! Structure-of-Arrays (SoA) columnar layout for any `#[repr(C)]` struct
//! whose fields implement `bytemuck::Pod`.
//!
//! # Pipeline
//!
//! The macro is split into three phases:
//!
//! 1. **Parse** (`parse`) — extract field names, types, and attributes from
//!    the `DeriveInput`.
//! 2. **Lower** (`repr`) — build an intermediate representation (`SchemaIR`)
//!    that normalises group vs. simple fields.
//! 3. **Codegen** (`codegen`) — emit the final token stream (schema struct,
//!    `SoARead`/`SoAWrite` impls, column accessors, optional pyclass wrapper).

use syn::{DeriveInput, parse_macro_input};
use proc_macro::TokenStream;

mod parse;
mod repr;
mod codegen;

/// Derive a columnar SoA schema for a `#[repr(C)]` struct.
///
/// # Generated items
///
/// For a struct named `Foo`, the macro emits:
///
/// | Item | Description |
/// |---|---|
/// | `FooSchema` | A zero-sized type implementing `columnar::buffer::Schema` |
/// | `mod schema { ... }` | `pub const` column accessors (`ColumnIdx` / `ColumnGroupIdx`) |
/// | `impl SoAWrite for Foo` | Scatter a `Foo` into the columnar buffer |
/// | `impl SoARead for Foo` | Gather a `Foo` from the columnar buffer |
///
/// # Field attributes
///
/// | Attribute | Effect |
/// |---|---|
/// | `#[columnar(group)]` | Expand a `[T; N]` field into N separate sub-columns |
/// | `#[columnar(skip_py)]` | Exclude this field from the generated Python batch wrapper |
///
/// # Struct attributes
///
/// | Attribute | Effect |
/// |---|---|
/// | `#[columnar(pyclass = "Name")]` | Generate a `#[pyclass]` batch wrapper with `__getitem__` access |
///
/// # Example
///
/// ```rust,ignore
/// #[repr(C)]
/// #[derive(Columnar)]
/// pub struct Sequence {
///     pub id:    u64,
///     pub score: f32,
///     #[columnar(group)]
///     pub elements: [u8; 32],
/// }
/// ```
#[proc_macro_derive(Columnar, attributes(columnar))]
pub fn derive_columnar_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let parsed = parse::parse(input)?;
    let ir = repr::lower(&parsed)?;
    Ok(codegen::generate(&ir))
}
