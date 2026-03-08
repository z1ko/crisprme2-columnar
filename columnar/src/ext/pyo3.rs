//! Zero-copy Python integration via the PEP 3118 buffer protocol.
//!
//! This module provides `PyColumnView`, a `#[pyclass]` that exposes a single
//! column (or sub-column) of a [`ColumnarBuffer`](crate::buffer::ColumnarBuffer)
//! to Python as a `memoryview`-compatible object — no copies involved.
//!
//! On the Python side the returned object supports `memoryview()` and can be
//! wrapped with `numpy.frombuffer()` for zero-copy array access.

use std::ffi::CStr;
use std::os::raw::c_int;

use pyo3::ffi;
use pyo3::prelude::*;

use crate::buffer::{ColumnIdx, ColumnGroupIdx, Schema};

// =============================================================================
// PyBufferFormat
// =============================================================================

/// Maps a Rust [`bytemuck::Pod`] type to its Python struct format character
/// (PEP 3118).
///
/// Implemented for all common numeric primitives. If you need a custom type,
/// implement this trait and provide the correct format string (must be
/// null-terminated).
pub trait PyBufferFormat: bytemuck::Pod {
    /// Null-terminated PEP 3118 format string, e.g. `b"B\0"` for `u8`.
    const FORMAT: &'static CStr;
}

macro_rules! impl_format {
    ($ty:ty, $fmt:literal) => {
        impl PyBufferFormat for $ty {
            const FORMAT: &'static CStr = unsafe {
                CStr::from_bytes_with_nul_unchecked(concat!($fmt, "\0").as_bytes())
            };
        }
    };
}

impl_format!(u8,  "B");
impl_format!(u16, "H");
impl_format!(u32, "I");
impl_format!(u64, "Q");
impl_format!(i8,  "b");
impl_format!(i16, "h");
impl_format!(i32, "i");
impl_format!(i64, "q");
impl_format!(f32, "f");
impl_format!(f64, "d");

// =============================================================================
// PyColumnView
// =============================================================================

/// A zero-copy view into a single column of a `ColumnarBuffer`.
///
/// Implements the PEP 3118 buffer protocol so that Python code can obtain a
/// `memoryview` (or a numpy array via `numpy.frombuffer`) backed directly by
/// the Rust-owned memory — no data is copied.
///
/// # Preventing garbage collection
///
/// `PyColumnView` stores a `PyObject` reference to the owning Python wrapper
/// (e.g. your `#[pyclass]` around `ColumnarBuffer`). As long as the
/// `PyColumnView` is alive, Python's GC will not collect the owner, keeping
/// the backing memory valid.
///
/// # Safety contract
///
/// The caller must ensure that the `ColumnarBuffer` is not reallocated or
/// dropped while any `PyColumnView` exists. In practice this means the Python
/// wrapper should **not** expose `push` or other capacity-changing operations
/// while column views are outstanding.
#[pyclass(unsendable)]
pub struct PyColumnView {
    /// Prevent GC of the parent Python object that owns the Columnar buffer.
    _owner: Py<PyAny>,
    /// Raw pointer to the first element of this column's data.
    ptr: *mut u8,
    /// Number of elements (rows) visible through this view.
    len: isize,
    /// Size of a single element in bytes.
    item_size: isize,
    /// PEP 3118 format string (null-terminated).
    format: &'static CStr,
    /// Whether this view allows mutation.
    writable: bool,

    // Buffer protocol requires stable addresses for shape/strides, so we
    // store them inline and hand out pointers to these fields.
    shape: [isize; 1],
    strides: [isize; 1],
}

impl PyColumnView {

    /// Helper to get the PEP 3118 format string for a column token.
    #[inline]
    pub fn format_for_column<S: Schema, const IDX: usize, T: PyBufferFormat>(
        _col: ColumnIdx<S, IDX, T>,
    ) -> &'static CStr {
        T::FORMAT
    }

    /// Helper to get the PEP 3118 format string for a group column token.
    #[inline]
    pub fn format_for_group_column<S: Schema, const IDX: usize, const N: usize, T: PyBufferFormat>(
        _col: ColumnGroupIdx<S, IDX, N, T>,
    ) -> &'static CStr {
        T::FORMAT
    }

    /// Create a `ColumnView` from pre-computed raw values.
    ///
    /// This is the low-level constructor used by the `pybatch!` macro.
    /// The caller is responsible for computing the correct pointer, length,
    /// and format from the column token and buffer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to valid memory for `len * item_size` bytes, and must
    /// remain valid for the lifetime of `owner`.
    pub unsafe fn new_raw(
        ptr: *mut u8,
        len: isize,
        item_size: isize,
        format: &'static CStr,
        writable: bool,
        owner: Py<PyAny>,
    ) -> Self {
        Self {
            _owner: owner,
            ptr,
            len,
            item_size,
            format,
            writable,
            shape: [len],
            strides: [item_size],
        }
    }
}

/// Declare a `#[pyclass]` batch wrapper with dict-style column access.
///
/// Columns are accessed via `batch["col_name"]` in Python. Writability is
/// automatically determined: if the batch is the sole owner of the underlying
/// buffer, the view is writable; otherwise it is read-only.
///
/// # Example
///
/// ```rust,ignore
/// columnar::pybatch!(PySequenceBatch, sequence::SequenceSchema, {
///     id => sequence::schema::id,
/// });
/// ```
///
/// Python usage:
/// ```python
/// ids = np.asarray(batch["id"])  # auto read-only or writable
/// ```
/// Internal macro that generates the `__getitem__` body. Not public API.
#[macro_export]
#[doc(hidden)]
macro_rules! __pybatch_impl {
    ($name:ident, $schema:ty,
     cols: [ $( $method:ident => $col:expr ),* ],
     groups: [ $( $gmethod:ident [ $gn:expr ] => $gcol:expr ),* ]
    ) => {

        #[::pyo3::pyclass(str = "{batch:?}")]
        pub struct $name {
            pub batch: ::std::option::Option<$crate::ring::Batch<$schema, ()>>,
        }

        #[::pyo3::pymethods]
        impl $name {
            fn __getitem__(slf: ::pyo3::Py<Self>, py: ::pyo3::Python<'_>, key: &::pyo3::Bound<'_, ::pyo3::PyAny>)
                -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>>
            {
                let this = slf.borrow(py);
                let batch = this.batch.as_ref()
                    .ok_or(::pyo3::exceptions::PyRuntimeError::new_err("batch consumed"))?;

                let writable = batch.is_exclusive();
                let buf = batch.as_ref();
                let capacity = buf.capacity();
                let len = buf.len() as isize;

                use $crate::buffer::ByteBuffer as _;

                let (col_name, sub_index): (String, Option<usize>) = if let Ok(tuple) = key.cast::<::pyo3::types::PyTuple>() {
                    if tuple.len() != 2 {
                        return Err(::pyo3::exceptions::PyKeyError::new_err("expected (name, index) tuple"));
                    }
                    let name: String = tuple.get_item(0)?.extract()?;
                    let idx: usize = tuple.get_item(1)?.extract()?;
                    (name, Some(idx))
                } else {
                    (key.extract::<String>()?, None)
                };

                match (col_name.as_str(), sub_index) {
                    $(
                        (stringify!($method), None) => {
                            let col = $col;
                            let offset = col.offset(capacity);
                            let item_size = col.elem_size() as isize;
                            let ptr = unsafe { buf.storage.as_bytes().as_ptr().add(offset) as *mut u8 };
                            let format = $crate::ext::pyo3::PyColumnView::format_for_column(col);
                            let view = unsafe {
                                $crate::ext::pyo3::PyColumnView::new_raw(
                                    ptr, len, item_size, format, writable,
                                    slf.clone_ref(py).into_any(),
                                )
                            };
                            Ok(::pyo3::Py::new(py, view)?.into_any())
                        }
                    )*
                    $(
                        (stringify!($gmethod), Some(k)) => {
                            if k >= $gn {
                                return Err(::pyo3::exceptions::PyIndexError::new_err(
                                    format!("sub-column index {} out of range (0..{})", k, $gn)
                                ));
                            }
                            let col = $gcol;
                            let offset = col.offset(k, capacity);
                            let item_size = col.elem_size() as isize;
                            let ptr = unsafe { buf.storage.as_bytes().as_ptr().add(offset) as *mut u8 };
                            let format = $crate::ext::pyo3::PyColumnView::format_for_group_column(col);
                            let view = unsafe {
                                $crate::ext::pyo3::PyColumnView::new_raw(
                                    ptr, len, item_size, format, writable,
                                    slf.clone_ref(py).into_any(),
                                )
                            };
                            Ok(::pyo3::Py::new(py, view)?.into_any())
                        }
                        (stringify!($gmethod), None) => {
                            let col = $gcol;
                            let item_size = col.elem_size() as isize;
                            let format = $crate::ext::pyo3::PyColumnView::format_for_group_column(col);
                            let mut views = Vec::with_capacity($gn);
                            for k in 0..$gn {
                                let offset = col.offset(k, capacity);
                                let ptr = unsafe { buf.storage.as_bytes().as_ptr().add(offset) as *mut u8 };
                                let view = unsafe {
                                    $crate::ext::pyo3::PyColumnView::new_raw(
                                        ptr, len, item_size, format, writable,
                                        slf.clone_ref(py).into_any(),
                                    )
                                };
                                views.push(::pyo3::Py::new(py, view)?);
                            }
                            Ok(::pyo3::types::PyList::new(py, views)?.into_any().unbind())
                        }
                    )*
                    _ => Err(::pyo3::exceptions::PyKeyError::new_err(
                        format!("unknown column: '{col_name}'")
                    )),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! pybatch {
    // With groups
    ($name:ident, $schema:ty,
     { $( $method:ident => $col:expr ),* $(,)? },
     groups: { $( $gmethod:ident [ $gn:expr ] => $gcol:expr ),* $(,)? }
    ) => {
        $crate::__pybatch_impl!($name, $schema,
            cols: [ $( $method => $col ),* ],
            groups: [ $( $gmethod [ $gn ] => $gcol ),* ]
        );
    };
    // Without groups
    ($name:ident, $schema:ty, { $( $method:ident => $col:expr ),* $(,)? }) => {
        $crate::__pybatch_impl!($name, $schema,
            cols: [ $( $method => $col ),* ],
            groups: []
        );
    };
}

#[pymethods]
impl PyColumnView {

    unsafe fn __getbuffer__(&self,view: *mut ffi::Py_buffer, flags: c_int) 
        -> PyResult<()> 
    {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "null Py_buffer pointer",
            ));
        }

        // If writable access is requested but we're read-only, reject.
        if !self.writable && (flags & ffi::PyBUF_WRITABLE) != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "this column view is read-only",
            ));
        }

        let v = unsafe { &mut *view };
        v.buf = self.ptr as *mut std::ffi::c_void;
        v.len = self.len * self.item_size;
        v.itemsize = self.item_size;
        v.readonly = if self.writable { 0 } else { 1 };
        v.ndim = 1;
        v.format = self.format.as_ptr() as *mut _;
        v.shape = self.shape.as_ptr() as *mut _;
        v.strides = self.strides.as_ptr() as *mut _;
        v.suboffsets = std::ptr::null_mut();
        v.internal = std::ptr::null_mut();

        // Prevent GC of this ColumnView while the buffer is exported.
        // We increment the refcount of the ColumnView's Python object.
        // The obj field is set by the caller (Python runtime) in most cases,
        // but we must set it for safety.
        // NOTE: pyo3 handles obj automatically when using __getbuffer__.

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // No-op: we don't allocate anything extra in __getbuffer__.
    }
}
