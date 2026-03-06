use heck::ToSnakeCase;
use syn::{Data, DeriveInput, Fields, Ident, parse_macro_input, spanned::Spanned};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;

#[proc_macro_derive(Columnar)]
pub fn derive_columnar_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(token_stream) => token_stream.into(),
        Err(e) => e.to_compile_error().into()
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {

    // #[repr(C)] must be present for bytemuck
    let has_repr_c = input.attrs.iter().any(|attr| {
        attr.path().is_ident("repr") && 
        attr.parse_args::<Ident>()
            .map(|v| v == "C").unwrap_or(false)
    });

    if !has_repr_c {
        return Err(syn::Error::new(input.span(),
            "Columnar requires #[repr(C)] to guarantee stable field offsets",
        ));
    }

    // Get all fields of the struct
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => return Err(syn::Error::new(input.span(), 
                "Columnar requires a struct with named fields")
            )
        },
        _ => return Err(syn::Error::new(input.span(),
            "Columnar can only be derived for structs",
        )),
    };

    let n = fields.len();

    let field_types: Vec<&syn::Type> = fields.iter()
        .map(|f| &f.ty)
        .collect();

    let field_names: Vec<&Ident> = fields.iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect();

    // Name of the schema struct
    let struct_name = &input.ident;
    let schema_name = Ident::new(
        &format!("{}Schema", struct_name),
        Span::call_site(),
    );

    // Column indices in order of declaration
    let col_indices: Vec<proc_macro2::Literal> = (0..field_names.len())
        .map(proc_macro2::Literal::usize_suffixed)
        .collect();

    // This is the fun part :)
    Ok(quote! {

        /// Schema unit type for this struct
        pub struct #schema_name;
        impl #schema_name {

            /// Element byte sizes in declaration order.
            pub const ELEM_SIZES: [usize; #n] = [
                #( ::std::mem::size_of::<#field_types>() ),*
            ];

            /// Field alignments in declaration order.
            pub const FIELD_ALIGNS: [usize; #n] = [
                #( ::std::mem::align_of::<#field_types>() ),*
            ];

            /// Bytes occupied by a single AoS type
            pub const STRIDE: usize = {
                let mut stride = 0;
                let sizes = Self::ELEM_SIZES;
                let mut i = 0usize;
                while i < #n {
                    stride += sizes[i];
                    i += 1;
                }
                stride
            };

            /// BLOCK_ORDER[k] = declaration index of the field stored in column block k.
            /// Fields are sorted by descending alignment so that every column block is
            /// naturally aligned for any row_capacity (stable sort preserves decl order
            /// for ties).
            pub const BLOCK_ORDER: [usize; #n] = {
                let aligns = Self::FIELD_ALIGNS;
                let mut order = [0usize; #n];
                // Initialise to identity permutation
                let mut i = 0usize;
                while i < #n { order[i] = i; i += 1; }
                // Stable bubble sort descending by alignment (#n passes suffice)
                let mut pass = 0usize;
                while pass < #n {
                    let mut i = 0usize;
                    while i + 1 < #n {
                        if aligns[order[i]] < aligns[order[i + 1]] {
                            let tmp = order[i];
                            order[i] = order[i + 1];
                            order[i + 1] = tmp;
                        }
                        i += 1;
                    }
                    pass += 1;
                }
                order
            };

            /// FIELD_TO_BLOCK[i] = block index for the i-th declared field
            /// (inverse permutation of BLOCK_ORDER).
            pub const FIELD_TO_BLOCK: [usize; #n] = {
                let order = Self::BLOCK_ORDER;
                let mut inv = [0usize; #n];
                let mut k = 0usize;
                while k < #n { inv[order[k]] = k; k += 1; }
                inv
            };

            /// Byte offsets of column blocks in block order (alignment-descending).
            /// BLOCK_OFFSETS[k] * row_capacity gives the byte start of block k.
            pub const BLOCK_OFFSETS: [usize; #n] = {
                let sizes = Self::ELEM_SIZES;
                let order = Self::BLOCK_ORDER;
                let mut prefix = [0usize; #n];
                let mut i = 1usize;
                while i < #n {
                    prefix[i] = prefix[i - 1] + sizes[order[i - 1]];
                    i += 1;
                }
                prefix
            };
        }

        impl ::columnar::Schema for #schema_name {

            #[inline]
            fn stride() -> usize { Self::STRIDE }

            #[inline]
            fn offset(col_index: usize, row_capacity: usize) -> usize {
                Self::BLOCK_OFFSETS[col_index] * row_capacity
            }
        }

        /// SoAWrite: scatter all fields into the correct column blocks.
        /// Works for any field type that implements bytemuck::Pod,
        /// including primitives, [u8; N], and other fixed-size arrays.
        impl ::columnar::SoAWrite for #struct_name {
            type Schema = #schema_name;

            #[inline]
            fn write_into(self, data: &mut [u8], row: usize, row_capacity: usize) {
                #({
                    let block      = #schema_name::FIELD_TO_BLOCK[#col_indices];
                    let col_offset = #schema_name::offset(block, row_capacity);
                    let elem_size  = ::std::mem::size_of::<#field_types>();
                    let start      = col_offset + row * elem_size;
                    data[start..start + elem_size]
                        .copy_from_slice(::bytemuck::bytes_of(&self.#field_names));
                })*
            }
        }

        /// SoARead: gather all fields from the correct column blocks.
        impl ::columnar::SoARead for #struct_name {
            type Schema = #schema_name;

            #[inline]
            fn read_from(data: &[u8], row: usize, row_capacity: usize) -> Self {
                #struct_name {
                    #(
                        #field_names: {
                            let block      = #schema_name::FIELD_TO_BLOCK[#col_indices];
                            let col_offset = #schema_name::offset(block, row_capacity);
                            let elem_size  = ::std::mem::size_of::<#field_types>();
                            let start      = col_offset + row * elem_size;
                            *::bytemuck::from_bytes(&data[start..start + elem_size])
                        },
                    )*
                }
            }
        }

        /// Column ZST abstraction
        pub mod schema {
            #![allow(non_camel_case_types)]
            use super::*;
            #(
                #[derive(Clone, Copy)]
                pub struct #field_names;

                impl ::columnar::ColumnType for #field_names {
                    type Schema = #schema_name;
                    type Value  = #field_types;

                    #[inline]
                    fn col_index(self) -> usize {
                        #schema_name::FIELD_TO_BLOCK[#col_indices]
                    }

                    #[inline]
                    fn offset(self, row_capacity: usize) -> usize {
                        #schema_name::offset(self.col_index(), row_capacity)
                    }
                }
            )*
        }
    })
}