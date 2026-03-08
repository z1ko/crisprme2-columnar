use proc_macro2::TokenStream;
use quote::quote;

use crate::repr::SchemaIR;

/// Generate SoAWrite and SoARead implementations.
pub fn generate(ir: &SchemaIR) -> TokenStream {
    let struct_name = ir.struct_name;
    let schema_name = &ir.schema_name;

    let write_impl = gen_write(ir);
    let read_impl = gen_read(ir);

    quote! {
        impl ::columnar::buffer::SoAWrite for #struct_name {
            type Schema = #schema_name;

            #[inline]
            fn write_into(self, data: &mut [u8], row: usize, row_capacity: usize) {
                #write_impl
            }
        }

        impl ::columnar::buffer::SoARead for #struct_name {
            type Schema = #schema_name;

            #[inline]
            fn read_from(data: &[u8], row: usize, row_capacity: usize) -> Self {
                #struct_name { #read_impl }
            }
        }
    }
}

/// Emit the body of `SoAWrite::write_into`: scatter each field into its
/// column block(s) via `bytemuck::bytes_of` + `copy_from_slice`.
fn gen_write(ir: &SchemaIR) -> TokenStream {
    let schema_name = &ir.schema_name;

    let write_fields: Vec<TokenStream> = ir.fields.iter().enumerate().map(|(i, f)| {
        let name = f.name;
        let elem_ty = f.elem_ty;

        if let Some(array_len) = f.array_len {
            quote! {
                {
                    let first = #schema_name::LAYOUT.fields[#i];
                    let elem_size = ::std::mem::size_of::<#elem_ty>();
                    let mut k = 0usize;
                    while k < #array_len {
                        let col_offset = #schema_name::offset(first + k, row_capacity);
                        let start = col_offset + row * elem_size;
                        data[start..start + elem_size]
                            .copy_from_slice(::bytemuck::bytes_of(&self.#name[k]));
                        k += 1;
                    }
                }
            }
        } else {
            quote! {
                {
                    let col_offset = #schema_name::offset(#schema_name::LAYOUT.fields[#i], row_capacity);
                    let elem_size = ::std::mem::size_of::<#elem_ty>();
                    let start = col_offset + row * elem_size;
                    data[start..start + elem_size]
                        .copy_from_slice(::bytemuck::bytes_of(&self.#name));
                }
            }
        }
    }).collect();

    quote! { #( #write_fields )* }
}

/// Emit the body of `SoARead::read_from`: gather each field from its
/// column block(s) via `bytemuck::from_bytes`.
fn gen_read(ir: &SchemaIR) -> TokenStream {
    let schema_name = &ir.schema_name;

    let read_fields: Vec<TokenStream> = ir.fields.iter().enumerate().map(|(i, f)| {
        let name = f.name;
        let elem_ty = f.elem_ty;

        if let Some(array_len) = f.array_len {
            quote! {
                #name: {
                    let first = #schema_name::LAYOUT.fields[#i];
                    let elem_size = ::std::mem::size_of::<#elem_ty>();
                    let mut arr: [#elem_ty; #array_len] = [<#elem_ty as ::bytemuck::Zeroable>::zeroed(); #array_len];
                    let mut k = 0usize;
                    while k < #array_len {
                        let col_offset = #schema_name::offset(first + k, row_capacity);
                        let start = col_offset + row * elem_size;
                        arr[k] = *::bytemuck::from_bytes(&data[start..start + elem_size]);
                        k += 1;
                    }
                    arr
                }
            }
        } else {
            quote! {
                #name: {
                    let col_offset = #schema_name::offset(#schema_name::LAYOUT.fields[#i], row_capacity);
                    let elem_size = ::std::mem::size_of::<#elem_ty>();
                    let start = col_offset + row * elem_size;
                    *::bytemuck::from_bytes(&data[start..start + elem_size])
                }
            }
        }
    }).collect();

    quote! { #( #read_fields, )* }
}
