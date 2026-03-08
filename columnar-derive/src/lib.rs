use syn::{Data, DeriveInput, Fields, Ident, parse_macro_input, spanned::Spanned};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;

#[proc_macro_derive(Columnar, attributes(columnar))]
pub fn derive_columnar_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(token_stream) => token_stream.into(),
        Err(e) => e.to_compile_error().into()
    }
}

/// Information about a single field, possibly expanded into multiple blocks.
struct FieldInfo<'a> {

    name: &'a Ident,
    ty: &'a syn::Type,
    is_group: bool,
    /// For group fields: the element type (T in [T; N])
    elem_type: Option<&'a syn::Type>,
    /// For group fields: the array length expression
    array_len: Option<&'a syn::Expr>,
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {

    /*
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
     */

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

    // Parse field info, detecting #[columnar(group)]
    let mut field_infos: Vec<FieldInfo> = Vec::new();

    for field in fields.iter() {
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        let is_group = field.attrs.iter().any(|attr| {
            attr.path().is_ident("columnar") &&
            attr.parse_args::<Ident>()
                .map(|v| v == "group")
                .unwrap_or(false)
        });

        if is_group {
            // Extract T and N from [T; N]
            let (elem_type, array_len) = match ty {
                syn::Type::Array(arr) => (&*arr.elem, &arr.len),
                _ => return Err(syn::Error::new(
                    field.span(),
                    "#[columnar(group)] requires a fixed-size array type [T; N]"
                )),
            };

            // We don't know N at macro time as a usize, but we emit it as a const expr.
            // For the schema consts, we need N as a literal. We'll emit const-evaluable code.
            field_infos.push(FieldInfo {
                name,
                ty,
                is_group: true,
                elem_type: Some(elem_type),
                array_len: Some(array_len),
            });
        } else {
            field_infos.push(FieldInfo {
                name,
                ty,
                is_group: false,
                elem_type: None,
                array_len: None,
            });
        }
    }

    let has_groups = field_infos.iter().any(|f| f.is_group);

    // Name of the schema struct
    let struct_name = &input.ident;
    let schema_name = Ident::new(
        &format!("{}Schema", struct_name),
        Span::call_site(),
    );

    if has_groups {
        expand_with_groups(struct_name, &schema_name, &field_infos)
    } else {
        expand_simple(struct_name, &schema_name, &field_infos)
    }
}

/// Original expansion path for structs with no group fields.
fn expand_simple(
    struct_name: &Ident,
    schema_name: &Ident,
    field_infos: &[FieldInfo],
) -> syn::Result<proc_macro2::TokenStream> {
    let n = field_infos.len();
    let field_types: Vec<&syn::Type> = field_infos.iter().map(|f| f.ty).collect();
    let field_names: Vec<&Ident> = field_infos.iter().map(|f| f.name).collect();
    let col_indices: Vec<proc_macro2::Literal> = (0..n)
        .map(proc_macro2::Literal::usize_suffixed)
        .collect();

    Ok(quote! {
        pub struct #schema_name;
        impl #schema_name {
            pub const ELEM_SIZES: [usize; #n] = [
                #( ::std::mem::size_of::<#field_types>() ),*
            ];

            pub const FIELD_ALIGNS: [usize; #n] = [
                #( ::std::mem::align_of::<#field_types>() ),*
            ];

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

            pub const BLOCK_ORDER: [usize; #n] = {
                let aligns = Self::FIELD_ALIGNS;
                let mut order = [0usize; #n];
                let mut i = 0usize;
                while i < #n { order[i] = i; i += 1; }
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

            pub const FIELD_TO_BLOCK: [usize; #n] = {
                let order = Self::BLOCK_ORDER;
                let mut inv = [0usize; #n];
                let mut k = 0usize;
                while k < #n { inv[order[k]] = k; k += 1; }
                inv
            };

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

        impl ::columnar::buffer::Schema for #schema_name {
            #[inline]
            fn stride() -> usize { Self::STRIDE }

            #[inline]
            fn offset(col_index: usize, row_capacity: usize) -> usize {
                Self::BLOCK_OFFSETS[col_index] * row_capacity
            }
        }

        impl ::columnar::buffer::SoAWrite for #struct_name {
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

        impl ::columnar::buffer::SoARead for #struct_name {
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

        pub mod schema {
            #![allow(non_camel_case_types)]
            use super::*;
            #(
                #[derive(Clone, Copy)]
                pub struct #field_names;

                impl ::columnar::buffer::ColumnType for #field_names {
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

                impl ::columnar::buffer::ColumnSelector for #field_names {
                    type Schema = #schema_name;
                    type Ref<'a> = &'a [#field_types];
                    type Mut<'a> = &'a mut [#field_types];

                    #[inline]
                    fn collect_col_indices(&self, out: &mut Vec<usize>) {
                        out.push(#schema_name::FIELD_TO_BLOCK[#col_indices]);
                    }

                    #[inline]
                    fn get_ref<'a>(self, data: &'a [u8], row_count: usize, row_capacity: usize) -> Self::Ref<'a> {
                        let off = #schema_name::offset(#schema_name::FIELD_TO_BLOCK[#col_indices], row_capacity);
                        let elem_size = ::std::mem::size_of::<#field_types>();
                        ::bytemuck::cast_slice(&data[off..off + row_count * elem_size])
                    }

                    #[inline]
                    unsafe fn get_mut<'a>(self, data: *mut u8, row_count: usize, row_capacity: usize) -> Self::Mut<'a> {
                        let off = #schema_name::offset(#schema_name::FIELD_TO_BLOCK[#col_indices], row_capacity);
                        let elem_size = ::std::mem::size_of::<#field_types>();
                        ::bytemuck::cast_slice_mut(
                            ::std::slice::from_raw_parts_mut(data.add(off), row_count * elem_size)
                        )
                    }
                }
            )*
        }
    })
}

/// Expansion path for structs that have at least one #[columnar(group)] field.
fn expand_with_groups(
    struct_name: &Ident,
    schema_name: &Ident,
    field_infos: &[FieldInfo],
) -> syn::Result<proc_macro2::TokenStream> {
    // Build the expanded block list at codegen time.
    // For each field, emit entries into ELEM_SIZES and FIELD_ALIGNS.
    // Group fields emit N entries; regular fields emit 1.

    // We need to compute total_blocks as a const expression.
    // Emit: const TOTAL_BLOCKS: usize = 1 + N1 + 1 + ...;
    let mut total_blocks_tokens = Vec::new();
    for fi in field_infos.iter() {
        if fi.is_group {
            let len = fi.array_len.unwrap();
            total_blocks_tokens.push(quote! { #len });
        } else {
            total_blocks_tokens.push(quote! { 1usize });
        }
    }
    let total_blocks_expr = quote! { #( #total_blocks_tokens )+* };

    // Build ELEM_SIZES and FIELD_ALIGNS as const fn since we need loops for groups
    // We'll generate a const fn that builds the arrays.

    // For each field, generate code to fill ELEM_SIZES and FIELD_ALIGNS
    let mut fill_sizes = Vec::new();
    let mut stride_parts = Vec::new();

    for fi in field_infos.iter() {
        if fi.is_group {
            let elem_type = fi.elem_type.unwrap();
            let array_len = fi.array_len.unwrap();
            fill_sizes.push(quote! {
                {
                    let mut k = 0usize;
                    while k < #array_len {
                        sizes[idx] = ::std::mem::size_of::<#elem_type>();
                        aligns[idx] = ::std::mem::align_of::<#elem_type>();
                        idx += 1;
                        k += 1;
                    }
                }
            });
            stride_parts.push(quote! { #array_len * ::std::mem::size_of::<#elem_type>() });
        } else {
            let ty = fi.ty;
            fill_sizes.push(quote! {
                {
                    sizes[idx] = ::std::mem::size_of::<#ty>();
                    aligns[idx] = ::std::mem::align_of::<#ty>();
                    idx += 1;
                }
            });
            stride_parts.push(quote! { ::std::mem::size_of::<#ty>() });
        }
    }

    // Compute FIELD_FIRST_EXPANDED as const expressions for each field
    // field 0 starts at 0, field 1 starts at field0.block_count, etc.
    let mut first_expanded_parts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut running = quote! { 0usize };
    for fi in field_infos.iter() {
        first_expanded_parts.push(running.clone());
        if fi.is_group {
            let len = fi.array_len.unwrap();
            running = quote! { #running + #len };
        } else {
            running = quote! { #running + 1usize };
        }
    }

    // Generate SoAWrite body
    let mut write_fields = Vec::new();
    for (i, fi) in field_infos.iter().enumerate() {
        let name = fi.name;
        let first = &first_expanded_parts[i];
        if fi.is_group {
            let elem_type = fi.elem_type.unwrap();
            let array_len = fi.array_len.unwrap();
            write_fields.push(quote! {
                {
                    let first_expanded = #first;
                    let elem_size = ::std::mem::size_of::<#elem_type>();
                    let mut k = 0usize;
                    while k < #array_len {
                        let block = #schema_name::FIELD_TO_BLOCK[first_expanded + k];
                        let col_offset = #schema_name::offset(block, row_capacity);
                        let start = col_offset + row * elem_size;
                        data[start..start + elem_size]
                            .copy_from_slice(::bytemuck::bytes_of(&self.#name[k]));
                        k += 1;
                    }
                }
            });
        } else {
            let ty = fi.ty;
            write_fields.push(quote! {
                {
                    let block = #schema_name::FIELD_TO_BLOCK[#first];
                    let col_offset = #schema_name::offset(block, row_capacity);
                    let elem_size = ::std::mem::size_of::<#ty>();
                    let start = col_offset + row * elem_size;
                    data[start..start + elem_size]
                        .copy_from_slice(::bytemuck::bytes_of(&self.#name));
                }
            });
        }
    }

    // Generate SoARead body
    let mut read_fields = Vec::new();
    for (i, fi) in field_infos.iter().enumerate() {
        let name = fi.name;
        let first = &first_expanded_parts[i];
        if fi.is_group {
            let elem_type = fi.elem_type.unwrap();
            let array_len = fi.array_len.unwrap();
            read_fields.push(quote! {
                #name: {
                    let first_expanded = #first;
                    let elem_size = ::std::mem::size_of::<#elem_type>();
                    let mut arr: [#elem_type; #array_len] = [<#elem_type as ::bytemuck::Zeroable>::zeroed(); #array_len];
                    let mut k = 0usize;
                    while k < #array_len {
                        let block = #schema_name::FIELD_TO_BLOCK[first_expanded + k];
                        let col_offset = #schema_name::offset(block, row_capacity);
                        let start = col_offset + row * elem_size;
                        arr[k] = *::bytemuck::from_bytes(&data[start..start + elem_size]);
                        k += 1;
                    }
                    arr
                }
            });
        } else {
            let ty = fi.ty;
            read_fields.push(quote! {
                #name: {
                    let block = #schema_name::FIELD_TO_BLOCK[#first];
                    let col_offset = #schema_name::offset(block, row_capacity);
                    let elem_size = ::std::mem::size_of::<#ty>();
                    let start = col_offset + row * elem_size;
                    *::bytemuck::from_bytes(&data[start..start + elem_size])
                }
            });
        }
    }

    // Generate schema module entries
    let mut schema_entries = Vec::new();
    for (i, fi) in field_infos.iter().enumerate() {
        let name = fi.name;
        let first = &first_expanded_parts[i];
        if fi.is_group {
            let elem_type = fi.elem_type.unwrap();
            let array_len = fi.array_len.unwrap();
            schema_entries.push(quote! {
                #[derive(Clone, Copy)]
                pub struct #name;

                impl ::columnar::buffer::GroupColumnType<{ #array_len }> for #name {
                    type Schema = #schema_name;
                    type Value = #elem_type;

                    #[inline]
                    fn col_index(self, k: usize) -> usize {
                        #schema_name::FIELD_TO_BLOCK[#first + k]
                    }

                    #[inline]
                    fn offset(self, k: usize, row_capacity: usize) -> usize {
                        #schema_name::offset(self.col_index(k), row_capacity)
                    }
                }

                impl ::columnar::buffer::ColumnSelector for #name {
                    type Schema = #schema_name;
                    type Ref<'a> = [&'a [#elem_type]; #array_len];
                    type Mut<'a> = [&'a mut [#elem_type]; #array_len];

                    #[inline]
                    fn collect_col_indices(&self, out: &mut Vec<usize>) {
                        let first_expanded = #first;
                        let mut k = 0usize;
                        while k < #array_len {
                            out.push(#schema_name::FIELD_TO_BLOCK[first_expanded + k]);
                            k += 1;
                        }
                    }

                    #[inline]
                    fn get_ref<'a>(self, data: &'a [u8], row_count: usize, row_capacity: usize) -> Self::Ref<'a> {
                        let first_expanded = #first;
                        let elem_size = ::std::mem::size_of::<#elem_type>();
                        ::std::array::from_fn(|k| {
                            let off = #schema_name::offset(
                                #schema_name::FIELD_TO_BLOCK[first_expanded + k],
                                row_capacity,
                            );
                            ::bytemuck::cast_slice(&data[off..off + row_count * elem_size])
                        })
                    }

                    #[inline]
                    unsafe fn get_mut<'a>(self, data: *mut u8, row_count: usize, row_capacity: usize) -> Self::Mut<'a> {
                        let first_expanded = #first;
                        let elem_size = ::std::mem::size_of::<#elem_type>();
                        let byte_count = row_count * elem_size;
                        let mut result: [::std::mem::MaybeUninit<&'a mut [#elem_type]>; #array_len] =
                            unsafe { ::std::mem::MaybeUninit::uninit().assume_init() };
                        let mut k = 0usize;
                        while k < #array_len {
                            let off = #schema_name::offset(
                                #schema_name::FIELD_TO_BLOCK[first_expanded + k],
                                row_capacity,
                            );
                            result[k] = ::std::mem::MaybeUninit::new(
                                ::bytemuck::cast_slice_mut(
                                    ::std::slice::from_raw_parts_mut(data.add(off), byte_count)
                                )
                            );
                            k += 1;
                        }
                        unsafe {
                            let ptr = &result as *const [::std::mem::MaybeUninit<&'a mut [#elem_type]>; #array_len]
                                as *const [&'a mut [#elem_type]; #array_len];
                            ::std::ptr::read(ptr)
                        }
                    }
                }
            });
        } else {
            let ty = fi.ty;
            schema_entries.push(quote! {
                #[derive(Clone, Copy)]
                pub struct #name;

                impl ::columnar::buffer::ColumnType for #name {
                    type Schema = #schema_name;
                    type Value  = #ty;

                    #[inline]
                    fn col_index(self) -> usize {
                        #schema_name::FIELD_TO_BLOCK[#first]
                    }

                    #[inline]
                    fn offset(self, row_capacity: usize) -> usize {
                        #schema_name::offset(self.col_index(), row_capacity)
                    }
                }

                impl ::columnar::buffer::ColumnSelector for #name {
                    type Schema = #schema_name;
                    type Ref<'a> = &'a [#ty];
                    type Mut<'a> = &'a mut [#ty];

                    #[inline]
                    fn collect_col_indices(&self, out: &mut Vec<usize>) {
                        out.push(#schema_name::FIELD_TO_BLOCK[#first]);
                    }

                    #[inline]
                    fn get_ref<'a>(self, data: &'a [u8], row_count: usize, row_capacity: usize) -> Self::Ref<'a> {
                        let off = #schema_name::offset(#schema_name::FIELD_TO_BLOCK[#first], row_capacity);
                        let elem_size = ::std::mem::size_of::<#ty>();
                        ::bytemuck::cast_slice(&data[off..off + row_count * elem_size])
                    }

                    #[inline]
                    unsafe fn get_mut<'a>(self, data: *mut u8, row_count: usize, row_capacity: usize) -> Self::Mut<'a> {
                        let off = #schema_name::offset(#schema_name::FIELD_TO_BLOCK[#first], row_capacity);
                        let elem_size = ::std::mem::size_of::<#ty>();
                        ::bytemuck::cast_slice_mut(
                            ::std::slice::from_raw_parts_mut(data.add(off), row_count * elem_size)
                        )
                    }
                }
            });
        }
    }

    Ok(quote! {
        pub struct #schema_name;
        impl #schema_name {
            /// Total number of expanded column blocks.
            pub const TOTAL_BLOCKS: usize = #total_blocks_expr;

            /// Element byte sizes for each expanded block.
            pub const ELEM_SIZES: [usize; Self::TOTAL_BLOCKS] = {
                let mut sizes = [0usize; Self::TOTAL_BLOCKS];
                let mut aligns = [0usize; Self::TOTAL_BLOCKS];
                let mut idx = 0usize;
                #( #fill_sizes )*
                let _ = aligns; // suppress unused warning
                sizes
            };

            /// Alignments for each expanded block.
            pub const FIELD_ALIGNS: [usize; Self::TOTAL_BLOCKS] = {
                let mut sizes = [0usize; Self::TOTAL_BLOCKS];
                let mut aligns = [0usize; Self::TOTAL_BLOCKS];
                let mut idx = 0usize;
                #( #fill_sizes )*
                let _ = sizes; // suppress unused warning
                aligns
            };

            /// Bytes occupied by a single logical row.
            pub const STRIDE: usize = #( #stride_parts )+*;

            /// BLOCK_ORDER[k] = expanded index of the block stored at sorted position k.
            pub const BLOCK_ORDER: [usize; Self::TOTAL_BLOCKS] = {
                let n = Self::TOTAL_BLOCKS;
                let aligns = Self::FIELD_ALIGNS;
                let mut order = [0usize; Self::TOTAL_BLOCKS];
                let mut i = 0usize;
                while i < n { order[i] = i; i += 1; }
                let mut pass = 0usize;
                while pass < n {
                    let mut i = 0usize;
                    while i + 1 < n {
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

            /// FIELD_TO_BLOCK[expanded_index] = sorted block position.
            pub const FIELD_TO_BLOCK: [usize; Self::TOTAL_BLOCKS] = {
                let n = Self::TOTAL_BLOCKS;
                let order = Self::BLOCK_ORDER;
                let mut inv = [0usize; Self::TOTAL_BLOCKS];
                let mut k = 0usize;
                while k < n { inv[order[k]] = k; k += 1; }
                inv
            };

            /// Byte offsets of column blocks in sorted order.
            pub const BLOCK_OFFSETS: [usize; Self::TOTAL_BLOCKS] = {
                let n = Self::TOTAL_BLOCKS;
                let sizes = Self::ELEM_SIZES;
                let order = Self::BLOCK_ORDER;
                let mut prefix = [0usize; Self::TOTAL_BLOCKS];
                let mut i = 1usize;
                while i < n {
                    prefix[i] = prefix[i - 1] + sizes[order[i - 1]];
                    i += 1;
                }
                prefix
            };
        }

        impl ::columnar::buffer::Schema for #schema_name {
            #[inline]
            fn stride() -> usize { Self::STRIDE }

            #[inline]
            fn offset(col_index: usize, row_capacity: usize) -> usize {
                Self::BLOCK_OFFSETS[col_index] * row_capacity
            }
        }

        impl ::columnar::buffer::SoAWrite for #struct_name {
            type Schema = #schema_name;

            #[inline]
            fn write_into(self, data: &mut [u8], row: usize, row_capacity: usize) {
                #( #write_fields )*
            }
        }

        impl ::columnar::buffer::SoARead for #struct_name {
            type Schema = #schema_name;

            #[inline]
            fn read_from(data: &[u8], row: usize, row_capacity: usize) -> Self {
                #struct_name {
                    #( #read_fields, )*
                }
            }
        }

        pub mod schema {
            #![allow(non_camel_case_types)]
            use super::*;
            #( #schema_entries )*
        }
    })
}
