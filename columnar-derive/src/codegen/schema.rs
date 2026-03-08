use proc_macro2::TokenStream;
use quote::quote;

use crate::repr::SchemaIR;

/// Generate the Schema struct, Layout computation, and column accessor constants.
pub fn generate(ir: &SchemaIR) -> TokenStream {
    let schema_name = &ir.schema_name;
    let fcnt = ir.fields.len();

    // Total column count as a const expression: sum of counts per field
    let count_parts: Vec<TokenStream> = ir.fields.iter().map(|f| {
        match f.array_len {
            Some(len) => quote! { #len },
            None => quote! { 1usize },
        }
    }).collect();
    let ccnt_expr = quote! { #( #count_parts )+* };

    // Generate LayoutUnit array entries
    let units: Vec<TokenStream> = ir.fields.iter().enumerate().map(|(i, f)| {
        let elem_ty = f.elem_ty;
        let count: TokenStream = match f.array_len {
            Some(len) => quote! { #len },
            None => quote! { 1usize },
        };
        quote! {
            ::columnar::buffer::LayoutUnit {
                field: #i,
                align: ::std::mem::align_of::<#elem_ty>(),
                size: ::std::mem::size_of::<#elem_ty>(),
                count: #count
            }
        }
    }).collect();

    // Generate column accessor constants in schema module
    let columns: Vec<TokenStream> = ir.fields.iter().enumerate().map(|(i, f)| {
        let name = f.name;
        let elem_ty = f.elem_ty;
        if f.is_group() {
            quote! {
                pub const #name: ::columnar::buffer::ColumnGroupIdx<
                    #schema_name,
                    { #schema_name::LAYOUT.fields[#i] },
                    { #schema_name::LAYOUT.counts[#i] },
                    #elem_ty
                > = ::columnar::buffer::ColumnGroupIdx::NEW;
            }
        } else {
            quote! {
                pub const #name: ::columnar::buffer::ColumnIdx<
                    #schema_name,
                    { #schema_name::LAYOUT.fields[#i] },
                    #elem_ty
                > = ::columnar::buffer::ColumnIdx::NEW;
            }
        }
    }).collect();

    quote! {
        #[derive(Clone, Copy)]
        pub struct #schema_name;
        impl #schema_name {
            pub const COLUMN_COUNT: usize = #ccnt_expr;
            pub const LAYOUT: ::columnar::buffer::Layout<{ Self::COLUMN_COUNT }, #fcnt> =
                ::columnar::buffer::Layout::compute([
                    #( #units ),*
                ]);
        }

        impl ::columnar::buffer::Schema for #schema_name {
            #[inline]
            fn stride() -> usize { Self::LAYOUT.stride }

            #[inline]
            fn offset(col_index: usize, row_capacity: usize) -> usize {
                Self::LAYOUT.offsets[col_index] * row_capacity
            }
        }

        pub mod schema {
            #![allow(non_camel_case_types)]
            use super::*;
            #( #columns )*
        }
    }
}
