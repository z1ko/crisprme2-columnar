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

    /*
    let col_indices: Vec<proc_macro2::Literal> = (0..field_names.len())
        .map(proc_macro2::Literal::usize_suffixed)
        .collect();
     */

    // Lowercase module name from struct name
    let mod_name = Ident::new(
        &struct_name.to_string().to_lowercase(),
        Span::call_site(),
    );

    // This is the fun part :)
    Ok(quote! {
        
        /// Schema unit type for this struct
        pub struct #schema_name;
        impl ::columnar::Schema for #schema_name {
            
            #[inline]
            fn stride() -> usize {
                ::std::mem::size_of::<#struct_name>()
            }

            #[inline]
            fn offset(col_index: usize, row_capacity: usize) -> usize {
                // Byte sizes of each column element, in field declaration order
                const ELEM_SIZES: &[usize] = &[ #( ::std::mem::size_of::<#field_types>() ),* ];
                // Prefix sum: sum of all blocks before col_index
                let mut offset = 0;
                let mut i = 0;
                while i < col_index {
                    offset += ELEM_SIZES[i] * row_capacity;
                    i += 1;
                }
                offset
            }
        }

        /*
        /// Column ZST abstraction
        pub mod schema {
            pub mod #mod_name {
                #![allow(non_camel_case_types)]
                use super::super::*;
                #(
                    #[derive(Clone, Copy)]
                    pub struct #field_names;

                    impl ::columnar::ColumnType for #field_names {
                        type Schema = #schema_name;
                        type Value  = #field_types;

                        #[inline]
                        fn col_index(self) -> usize { 0 }  // ← now expands to 0, 1, 2...

                        #[inline]
                        fn offset(self, row_capacity: usize) -> usize {
                            #schema_name::offset(0, row_capacity)
                        }
                    }
                )*
            }
        }
        */
    })
}