//! Third phase of the derive pipeline: token generation.
//!
//! Consumes a [`SchemaIR`] and emits the complete token stream containing the
//! schema struct, `SoARead`/`SoAWrite` impls, column accessors, and the
//! optional pyclass wrapper.

mod schema;
mod soa;
mod pyclass;

use proc_macro2::TokenStream;
use quote::quote;

use crate::repr::SchemaIR;

/// Top-level code generation entry point.
///
/// Combines the output of [`schema::generate`], [`soa::generate`], and
/// [`pyclass::generate`] into a single token stream.
pub fn generate(ir: &SchemaIR) -> TokenStream {
    let schema_tokens = schema::generate(ir);
    let soa_tokens = soa::generate(ir);
    let pyclass_tokens = pyclass::generate(ir);

    quote! {
        #schema_tokens
        #soa_tokens
        #pyclass_tokens
    }
}
