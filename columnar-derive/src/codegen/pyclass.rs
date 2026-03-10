use std::{collections::HashMap, ffi::CStr};

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Token, token::Token};

use crate::repr::{FieldKindIR, SchemaIR};

/// Generate the `__pybatch_impl!` invocation if `#[columnar(pyclass = "...")]` is present.
pub fn generate(ir: &SchemaIR) -> TokenStream {
    let Some(pyclass_name) = &ir.pyclass else {
        return quote! {};
    };

    let schema = &ir.schema_name;

    // Create view for simple fields
    let fields_simpl: Vec<_> = ir.fields.iter()
        .filter_map(|f| {
            if let FieldKindIR::Simple = f.kind {
                let name = f.name;
                let ty = f.ty;
                return Some(quote! {
                    stringify!(#name) => {
                        let col = schema::#name;
                        let offset = col.offset(capacity);
                        let item_size = col.elem_size() as isize;
                        let ptr = unsafe { buffer.storage.as_bytes().as_ptr().add(offset) as *mut u8 };
                        let format = <#ty as ::columnar::ext::pyo3::PyBufferFormat>::FORMAT;
                        let view = unsafe {
                            ::columnar::ext::pyo3::PyColumnView::new_raw(
                                ptr, len, item_size, format, writable,
                                slf.clone_ref(py).into_any(),
                            )
                        };
                        Ok(::pyo3::Py::new(py, view)?)
                    }
                })
            }
            None
        }).collect();

    let fields_array: Vec<_> = ir.fields.iter()
        .filter_map(|f| {
            if let FieldKindIR::Array { elem_ty, len } = f.kind {
                let name = f.name;
                let n = len;
                return Some(quote! {
                    stringify!(#name) => {
                        let col = schema::#name;
                        let offset = col.offset(capacity);
                        let item_size = ::std::mem::size_of::<#elem_ty>() as isize;
                        let cols = #n as isize;
                        let ptr = unsafe { buffer.storage.as_bytes().as_ptr().add(offset) as *mut u8 };
                        let format = <#elem_ty as ::columnar::ext::pyo3::PyBufferFormat>::FORMAT;
                        // Shape (rows, N), C-contiguous strides
                        let view = unsafe {
                            ::columnar::ext::pyo3::PyColumnView::new_raw_2d(
                                ptr, [len, cols], [cols * item_size, item_size], item_size, format, writable,
                                slf.clone_ref(py).into_any(),
                            )
                        };
                        Ok(::pyo3::Py::new(py, view)?)
                    }
                })
            }
            None
        }).collect();

    let fields_group: Vec<_> = ir.fields.iter()
        .filter_map(|f| {
            if let FieldKindIR::Group { len } = f.kind {
                let name = f.name;
                let ty = f.ty;
                return Some(quote! {
                    stringify!(#name) => {
                        let col = schema::#name;
                        let item_size = col.elem_size() as isize;
                        let format = <#ty as ::columnar::ext::pyo3::PyBufferFormat>::FORMAT;
                        // First sub-column start; sub-columns are contiguous in layout
                        let offset = col.offset(0, capacity);
                        let ptr = unsafe { buffer.storage.as_bytes().as_ptr().add(offset) as *mut u8 };
                        // Shape (N, rows), stride between sub-columns = item_size * row_capacity
                        let view = unsafe {
                            ::columnar::ext::pyo3::PyColumnView::new_raw_2d(
                                ptr, [#len as isize, len], [item_size * capacity as isize, item_size], item_size, format, writable,
                                slf.clone_ref(py).into_any(),
                            )
                        };
                        Ok(::pyo3::Py::new(py, view)?)
                    }
                })
            }
            None
        }).collect();


    quote! { 

        #[::pyo3::pyclass(str = "{batch:?}")]
        pub struct #pyclass_name {
            pub batch: ::std::option::Option<::columnar::ring::BatchMut<#schema, ()>>,
        }

        #[::pyo3::pymethods]
        impl #pyclass_name {
            fn __getitem__(slf: ::pyo3::Py<Self>, py: ::pyo3::Python<'_>, key: &str)
                -> ::pyo3::PyResult<::pyo3::Py<::columnar::ext::pyo3::PyColumnView>>
            {
                let this = slf.borrow(py);
                let batch = this.batch.as_ref()
                    .ok_or(::pyo3::exceptions::PyRuntimeError::new_err("batch consumed"))?;

                let writable = true;
                let buffer = &**batch;
                let capacity = buffer.capacity();
                let len = buffer.len() as isize;

                use ::columnar::buffer::ByteBuffer as _;
                match key {
                    #( #fields_simpl, )*
                    #( #fields_array, )*
                    #( #fields_group, )*
                    _ => Err(::pyo3::exceptions::PyKeyError::new_err(
                        format!("unknown column: '{key}'")
                    )),
                }
            }
        }
    }
}
