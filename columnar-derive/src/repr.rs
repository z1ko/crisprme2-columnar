//! Second phase of the derive pipeline: lowering to an intermediate representation.
//!
//! Transforms the raw [`ParsedStruct`] into a [`SchemaIR`] that normalises
//! simple and group fields into a uniform representation ready for code
//! generation. Group fields (`[T; N]`) are decomposed into their element type
//! `T` and length expression `N`; simple fields get `array_len = None`.

use proc_macro2::Span;
use syn::Ident;

use crate::parse::ParsedStruct;

/// Intermediate representation for a single field.
///
/// Both simple fields and `#[columnar(group)]` fields share this type.
/// Group fields store the element type and array length separately;
/// simple fields have `array_len = None`.
pub struct FieldIR<'a> {
    /// The field's identifier.
    pub name: &'a Ident,
    /// The per-element type: `T` from `[T; N]` for groups, or the field type
    /// itself for simple fields.
    pub elem_ty: &'a syn::Type,
    /// `Some(N)` for group fields (the array length expression), `None` for
    /// simple fields.
    pub array_len: Option<&'a syn::Expr>,
    /// Whether this field is excluded from the Python batch wrapper.
    pub skip_py: bool,
}

impl FieldIR<'_> {
    /// Returns `true` if this field is a group column (`[T; N]`).
    pub fn is_group(&self) -> bool {
        self.array_len.is_some()
    }
}

/// Intermediate representation for the whole schema.
pub struct SchemaIR<'a> {
    /// The original struct's identifier (e.g. `Sequence`).
    pub struct_name: &'a Ident,
    /// The generated schema type name (e.g. `SequenceSchema`).
    pub schema_name: Ident,
    /// All fields, in declaration order.
    pub fields: Vec<FieldIR<'a>>,
    /// The Python batch wrapper name, if `#[columnar(pyclass = "...")]` was given.
    pub pyclass: Option<Ident>,
}

/// Lower a [`ParsedStruct`] into a [`SchemaIR`].
///
/// Validates that `#[columnar(group)]` fields have a fixed-size array type
/// `[T; N]` and decomposes them into element type + length expression.
/// Returns a compile error if validation fails.
pub fn lower<'a>(parsed: &ParsedStruct<'a>) -> syn::Result<SchemaIR<'a>> {
    let schema_name = Ident::new(
        &format!("{}Schema", parsed.struct_name),
        Span::call_site(),
    );

    let mut fields = Vec::new();
    for f in &parsed.fields {
        if f.group {
            // Extract T and N from [T; N]
            let (elem_ty, array_len) = match f.ty {
                syn::Type::Array(arr) => (&*arr.elem, &arr.len),
                _ => return Err(syn::Error::new_spanned(
                    f.ty,
                    "#[columnar(group)] requires a fixed-size array type [T; N]",
                )),
            };
            fields.push(FieldIR {
                name: f.name,
                elem_ty,
                array_len: Some(array_len),
                skip_py: f.skip_py,
            });
        } else {
            fields.push(FieldIR {
                name: f.name,
                elem_ty: f.ty,
                array_len: None,
                skip_py: f.skip_py,
            });
        }
    }

    Ok(SchemaIR {
        struct_name: parsed.struct_name,
        schema_name,
        fields,
        pyclass: parsed.pyclass.clone(),
    })
}
