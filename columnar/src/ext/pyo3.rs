//! Zero-copy Python integration via the PEP 3118 buffer protocol.
//!
//! This module provides [`ColumnView`], a `#[pyclass]` that exposes a single
//! column (or sub-column) of a [`Columnar`] buffer to Python as a
//! `memoryview`-compatible object — no copies involved.
//!
//! # Usage
//!
//! You write a schema-specific `#[pyclass]` wrapper around your `Columnar`
//! buffer and expose column accessors that return [`ColumnView`] instances:
//!
//! ```rust,ignore
//! use columnar::pyo3_support::ColumnView;
//! use pyo3::prelude::*;
//!
//! #[pyclass]
//! struct PyMyBuffer {
//!     inner: Columnar<MySchema, Vec<u8>>,
//! }
//!
//! #[pymethods]
//! impl PyMyBuffer {
//!     fn column_score(slf: Py<Self>, py: Python<'_>) -> PyResult<ColumnView> {
//!         let this = slf.borrow(py);
//!         ColumnView::from_column(&this.inner, my_schema::score, slf.into_any(), false)
//!     }
//! }
//! ```
//!
//! On the Python side the returned object supports `memoryview()` and can be
//! wrapped with `numpy.frombuffer()` for zero-copy array access.

use std::ffi::CStr;
use std::os::raw::c_int;

use pyo3::ffi;
use pyo3::prelude::*;

use crate::{ByteBuffer, ColumnType, ColumnarBuffer, GroupColumnType, Schema};

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
// ColumnView
// =============================================================================

/// A zero-copy view into a single column of a [`Columnar`] buffer.
///
/// Implements the PEP 3118 buffer protocol so that Python code can obtain a
/// `memoryview` (or a numpy array via `numpy.frombuffer`) backed directly by
/// the Rust-owned memory — no data is copied.
///
/// # Preventing garbage collection
///
/// `ColumnView` stores a [`PyObject`] reference to the owning Python wrapper
/// (e.g. your `#[pyclass]` around `Columnar`). As long as the `ColumnView`
/// is alive, Python's GC will not collect the owner, keeping the backing
/// memory valid.
///
/// # Safety contract
///
/// The caller must ensure that the `Columnar` buffer is not reallocated or
/// dropped while any `ColumnView` exists. In practice this means the Python
/// wrapper should **not** expose `push` or other capacity-changing operations
/// while column views are outstanding.
#[pyclass(unsendable)]
pub struct ColumnView {
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

impl ColumnView {

    /// Create a `ColumnView` for a regular (non-group) column.
    ///
    /// # Arguments
    ///
    /// * `buf`      — the columnar buffer to view into
    /// * `col`      — the column token (ZST)
    /// * `owner`    — a `PyObject` referencing the Python wrapper that owns `buf`
    /// * `writable` — if `true`, Python may write through this view
    ///
    /// # Safety (caller obligations)
    ///
    /// The `owner` must prevent `buf` from being deallocated or reallocated for
    /// the entire lifetime of the returned `ColumnView`.
    pub fn from_column<S, B, C>(
        buf: &mut ColumnarBuffer<S, B>,
        col: C,
        owner: Py<PyAny>,
        writable: bool,
    ) -> Self
    where
        S: Schema,
        B: ByteBuffer,
        C: ColumnType<Schema = S>,
        C::Value: PyBufferFormat,
    {
        let offset = col.offset(buf.capacity());
        let ptr = unsafe { buf.storage.as_bytes_mut().as_mut_ptr().add(offset) };
        let len = buf.len() as isize;
        let item_size = col.elem_size() as isize;

        Self {
            _owner: owner,
            ptr,
            len,
            item_size,
            format: C::Value::FORMAT,
            writable,
            shape: [len],
            strides: [item_size],
        }
    }

    /// Create `N` `ColumnView`s for a group column (one per sub-column).
    ///
    /// Returns a `Vec` of length `N`. Each view covers one sub-column.
    pub fn from_group_column<S, B, G, const N: usize>(
        buf: &mut ColumnarBuffer<S, B>,
        col: G,
        owner: Py<PyAny>,
        writable: bool,
        py: Python<'_>,
    ) -> Vec<Self>
    where
        S: Schema,
        B: ByteBuffer,
        G: GroupColumnType<N, Schema = S>,
        G::Value: PyBufferFormat,
    {
        let len = buf.len() as isize;
        let item_size = col.elem_size() as isize;
        let base_ptr = buf.storage.as_bytes_mut().as_mut_ptr();

        (0..N)
            .map(|k| {
                let offset = col.offset(k, buf.capacity());
                let ptr = unsafe { base_ptr.add(offset) };
                Self {
                    _owner: owner.clone_ref(py),
                    ptr,
                    len,
                    item_size,
                    format: G::Value::FORMAT,
                    writable,
                    shape: [len],
                    strides: [item_size],
                }
            })
            .collect()
    }
}

#[pymethods]
impl ColumnView {

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
