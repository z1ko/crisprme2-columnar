use proc_macro2::TokenStream;
use quote::quote;

use crate::repr::SchemaIR;

/// Generate the `__pybatch_impl!` invocation if `#[columnar(pyclass = "...")]` is present.
pub fn generate(ir: &SchemaIR) -> TokenStream {
    let Some(pyclass_name) = &ir.pyclass else {
        return quote! {};
    };

    let schema_name = &ir.schema_name;

    let mut col_entries = Vec::new();
    let mut group_entries = Vec::new();

    for f in &ir.fields {
        if f.skip_py { continue; }
        let name = f.name;
        if let Some(array_len) = f.array_len {
            group_entries.push(quote! { #name [ #array_len ] => schema::#name });
        } else {
            col_entries.push(quote! { #name => schema::#name });
        }
    }

    quote! {
        ::columnar::__pybatch_impl!(#pyclass_name, #schema_name,
            cols: [ #( #col_entries ),* ],
            groups: [ #( #group_entries ),* ]
        );
    }
}
