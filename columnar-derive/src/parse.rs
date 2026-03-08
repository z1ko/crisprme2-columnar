//! First phase of the derive pipeline: attribute extraction.
//!
//! Walks the [`syn::DeriveInput`] and collects field-level and struct-level
//! `#[columnar(...)]` attributes into plain data structures ([`ParsedField`],
//! [`ParsedStruct`]) without performing any validation or lowering.

use syn::{DeriveInput, Data, Fields, Ident, spanned::Spanned};

/// Raw parsed information for a single struct field.
pub struct ParsedField<'a> {
    /// The field's identifier (e.g. `id`, `score`).
    pub name: &'a Ident,
    /// The field's declared type, as-is from the source.
    pub ty: &'a syn::Type,
    /// `true` when `#[columnar(group)]` is present — the field will be
    /// expanded into N separate sub-columns.
    pub group: bool,
}

/// Raw parsed information for the whole struct being derived.
pub struct ParsedStruct<'a> {
    /// The struct's identifier (e.g. `Sequence`).
    pub struct_name: &'a Ident,
    /// All named fields, in declaration order.
    pub fields: Vec<ParsedField<'a>>,
    /// The name given in `#[columnar(pyclass = "...")]`, if present.
    pub pyclass: Option<Ident>,
}

/// Extract field and struct attributes from a [`DeriveInput`].
///
/// Returns an error if the input is not a struct with named fields.
pub fn parse(input: &DeriveInput) -> syn::Result<ParsedStruct<'_>> {
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => return Err(syn::Error::new(input.span(),
                "Columnar requires a struct with named fields")),
        },
        _ => return Err(syn::Error::new(input.span(),
            "Columnar can only be derived for structs")),
    };

    let mut parsed_fields = Vec::new();
    for field in fields.iter() {
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        let mut group = false;
        for attr in &field.attrs {
            if attr.path().is_ident("columnar") {
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("group") { group = true; }
                    Ok(())
                });
            }
        }

        parsed_fields.push(ParsedField { name, ty, group });
    }

    // Parse struct-level #[columnar(pyclass = "Name")]
    let mut pyclass = None;
    for attr in &input.attrs {
        if attr.path().is_ident("columnar") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("pyclass") {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    pyclass = Some(Ident::new(&lit.value(), lit.span()));
                }
                Ok(())
            })?;
        }
    }

    Ok(ParsedStruct {
        struct_name: &input.ident,
        fields: parsed_fields,
        pyclass,
    })
}
