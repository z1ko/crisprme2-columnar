//! # Columnar
//!
//! A zero-copy, cache-friendly Structure-of-Arrays (SoA) columnar buffer designed
//! for high-throughput data processing pipelines such as genomic analysis, physics
//! simulations, or any workload that benefits from operating on one field across
//! many rows at a time.
//!
//! ## Layout
//!
//! Data is stored as contiguous column blocks rather than interleaved rows:
//!
//! ```text
//! AoS (Array of Structs) — typical Rust struct layout:
//! [ id | score | elem ][ id | score | elem ][ id | score | elem ] ...
//!
//! SoA (Struct of Arrays) — this crate's layout:
//! [ id0 | id1 | id2 | ... ][ score0 | score1 | score2 | ... ][ elem0 | elem1 | ... ]
//! ```
//!
//! This layout is optimal for:
//! - **SIMD vectorisation**: iterating a single column keeps the CPU's vector units fed
//! - **Cache efficiency**: filtering on `score` never loads `id` or `elem` bytes
//! - **Zero-copy export**: column slices can be handed to Arrow/Polars/CUDA directly
//!
//! ## Quick start
//!
//! ```rust,ignore,no_run
//! #[repr(C)]
//! #[derive(Columnar)]
//! pub struct Sequence {
//!     pub id:       u64,
//!     pub score:    f32,
//!     pub elements: [u8; 32],
//! }
//!
//! // Allocate a buffer for 1024 rows
//! let mut buf: ColumnarBuffer<SequenceSchema, AlignedBox> =
//!     AlignedBox::new(1024 * SequenceSchema::STRIDE).columnar();
//!
//! // Push a full row at once
//! buf.push(Sequence { id: 1, score: 0.95, elements: [0u8; 32] });
//!
//! // Push with a closure for selective column writes
//! buf.push_with((sequence_schema::id, sequence_schema::score), |row, (ids, scores)| {
//!     ids[row]    = 42;
//!     scores[row] = 0.88;
//! });
//!
//! // Read a full row back
//! let seq: Option<Sequence> = buf.get(0);
//!
//! // Read individual columns — zero copy
//! let (ids, scores) = buf.columns((sequence_schema::id, sequence_schema::score));
//! ```

// =============================================================================
// ByteBuffer
// =============================================================================

use std::marker::PhantomData;

use bytemuck::Pod;

/// Trait for types that can serve as the raw byte backing store of a [`ColumnarBuffer`]
/// buffer.
///
/// Implement this for any contiguous, byte-addressable allocation you want to use
/// as columnar storage — a plain heap `Vec<u8>`, a GPU-pinned allocation, a
/// memory-mapped file, a borrowed slice, etc.
///
/// # Requirements
///
/// The byte slice returned by `as_bytes` and `as_bytes_mut` must be the **same**
/// contiguous region on every call, with a length that does not change after the
/// buffer is handed to [`ColumnarBuffer::new`].
pub trait ByteBuffer {
    /// Return a shared view of the raw storage.
    fn as_bytes(&self) -> &[u8];

    /// Return an exclusive view of the raw storage.
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

impl ByteBuffer for Vec<u8> {
    fn as_bytes(&self) -> &[u8] { self }
    fn as_bytes_mut(&mut self) -> &mut [u8] { self }
}

// =============================================================================
// AlignedBox
// =============================================================================

/// A heap-allocated byte buffer with guaranteed 8-byte alignment.
///
/// Uses a custom [`Layout`](std::alloc::Layout) to ensure the backing
/// allocation is aligned to 8 bytes, so all standard primitive column types
/// (up to `u64` / `f64`) can be safely reinterpreted via [`bytemuck`] without
/// alignment issues.
pub struct AlignedBox {
    layout: std::alloc::Layout,
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

// SAFETY: AlignedBox owns its allocation exclusively, like Box<[u8]>.
unsafe impl Send for AlignedBox {}
unsafe impl Sync for AlignedBox {}

impl AlignedBox {
    /// The alignment guaranteed for the backing allocation.
    pub const ALIGN: usize = 8;

    /// Allocate `len` zero-initialized bytes with 8-byte alignment.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0` (zero-sized allocations are not supported).
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "AlignedBox: zero-sized allocation");
        let layout = std::alloc::Layout::from_size_align(len, Self::ALIGN).unwrap();
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(ptr).expect("allocation failed");
        Self { layout, ptr, len }
    }

    pub fn len(&self) -> usize { self.len }
}

impl Drop for AlignedBox {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout); }
    }
}

impl ByteBuffer for AlignedBox {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes and we have shared access.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for len bytes and we have exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl AlignedBox {
    /// Wrap this slab in a typed columnar buffer. Takes ownership of `self`.
    pub fn columnar<S: Schema>(self) -> ColumnarBuffer<S, AlignedBox> {
        ColumnarBuffer::new(self)
    }
}

// =============================================================================
// Layout
// =============================================================================

/// Describes one logical field for the layout computation.
///
/// Simple fields have `count = 1`; group fields (`[T; N]`) have `count = N`
/// so that each array element becomes a separate column block.
#[derive(Clone, Copy)]
pub struct LayoutUnit {
    /// Index of the originating field in declaration order.
    pub field: usize,
    /// Required alignment of the element type (`align_of::<T>()`).
    pub align: usize,
    /// Byte size of a single element (`size_of::<T>()`).
    pub size:  usize,
    /// Number of columns this field expands into (1 for simple, N for groups).
    pub count: usize,
}

/// Pre-computed, alignment-sorted column layout for a schema.
///
/// `C` is the total number of columns (sum of all field counts) and `F` is
/// the number of logical fields. All arrays are computed at compile time via
/// [`Layout::compute`] and used by the generated `Schema` impl for zero-cost
/// offset lookups.
pub struct Layout<const C: usize, const F: usize> {
    /// Byte offset of each column within a single row's stride.
    /// `offsets[col]` is the prefix-sum position of that column in sorted
    /// order, so that `offsets[col] * row_capacity` gives the absolute byte
    /// offset in the buffer.
    pub offsets: [usize; C],
    /// Maps each field index to the first column index it occupies after
    /// alignment sorting.
    pub fields: [usize; F],
    /// Number of columns each field occupies (1 for simple, N for groups).
    pub counts: [usize; F],
    /// Total bytes per logical row (sum of all element sizes).
    pub stride: usize,
}

impl<const C: usize, const F: usize> Layout<C, F> {

    /// Compute the column layout from an array of [`LayoutUnit`]s.
    ///
    /// Fields are sorted by alignment (descending) so that higher-aligned
    /// columns appear first in memory, ensuring natural alignment for any
    /// row capacity. Group fields keep their sub-columns contiguous.
    pub const fn compute(mut units: [LayoutUnit; F]) -> Self {

        // sort units by alignment
        units = Layout::<C, F>::sort_by_align(units);

        let mut offsets = [0usize; C];
        let mut fields  = [usize::MAX; F];
        let mut counts  = [0usize; F];

        let mut stride = 0;
        let mut col = 0;

        let mut i = 0;
        while i < F {
            let unit = units[i];

            // store first column of this field
            if fields[unit.field] == usize::MAX {
                fields[unit.field] = col;
                counts[unit.field] = unit.count;
            }

            let mut k = 0;
            while k < unit.count {
                offsets[col] = stride;
                stride += unit.size;
                col += 1;
                k += 1;
            }

            i += 1;
        }

        Layout { offsets, fields, counts, stride }
    }

    const fn sort_by_align(mut units: [LayoutUnit; F]) -> [LayoutUnit; F] {
        let mut i = 1;
        while i < F {
            let key = units[i];
            let mut j = i;

            while j > 0 && units[j - 1].align < key.align {
                units[j] = units[j - 1];
                j -= 1;
            }

            units[j] = key;
            i += 1;
        }
        units
    }
}

// =============================================================================
// Column and ColumnGroup
// =============================================================================

/// A typed, zero-sized token addressing a single column within a schema.
///
/// Generated by `#[derive(Columnar)]` as `pub const` items inside the
/// `schema` module. The const-generic `IDX` encodes the column index at
/// compile time, providing zero-cost offset computation.
///
/// # Type parameters
///
/// - `S` — the schema this column belongs to (prevents cross-schema misuse)
/// - `IDX` — the column index within the sorted layout
/// - `T` — the element type stored in this column ([`bytemuck::Pod`])
#[derive(Clone, Copy)]
pub struct ColumnIdx<S: Schema, const IDX: usize, T: Pod>(PhantomData<(S, T)>);
impl<S: Schema, const IDX: usize, T: Pod> ColumnIdx<S, IDX, T> {
    /// Const constructor used by the derive macro.
    pub const NEW: Self = Self(PhantomData);

    /// The column's index within the schema's sorted layout.
    #[inline]
    pub fn col_index(self) -> usize { IDX }

    /// Byte offset of this column's block for a buffer with `row_capacity` rows.
    #[inline]
    pub fn offset(self, row_capacity: usize) -> usize {
        S::offset(IDX, row_capacity)
    }

    /// Size in bytes of a single element (`size_of::<T>()`).
    #[inline]
    pub fn elem_size(self) -> usize {
        std::mem::size_of::<T>()
    }
}

/// A typed, zero-sized token addressing a group of N contiguous sub-columns.
///
/// Generated by `#[derive(Columnar)]` for fields annotated with
/// `#[columnar(group)]`. A `[T; N]` field is expanded into N separate
/// sub-columns so that all rows' `element[k]` values are contiguous in
/// memory — ideal for SIMD and GPU coalesced access.
///
/// # Type parameters
///
/// - `S` — the schema this group belongs to
/// - `IDX` — the first sub-column's index in the sorted layout
/// - `N` — the number of sub-columns (array length)
/// - `T` — the element type stored in each sub-column
#[derive(Clone, Copy)]
pub struct ColumnGroupIdx<S: Schema, const IDX: usize, const N: usize, T: Pod>(PhantomData<(S, T)>);
impl<S: Schema, const IDX: usize, const N: usize, T: Pod> ColumnGroupIdx<S, IDX, N, T> {
    /// Const constructor used by the derive macro.
    pub const NEW: Self = Self(PhantomData);

    /// The column index of the k-th sub-column (`0 <= k < N`).
    #[inline]
    pub fn col_index(self, k: usize) -> usize { IDX + k }

    /// Byte offset of the k-th sub-column's block for a buffer with
    /// `row_capacity` rows.
    #[inline]
    pub fn offset(self, k: usize, row_capacity: usize) -> usize {
        S::offset(IDX + k, row_capacity)
    }

    /// Size in bytes of a single element (`size_of::<T>()`).
    #[inline]
    pub fn elem_size(self) -> usize {
        std::mem::size_of::<T>()
    }

    /// Number of sub-columns in this group (`N`).
    #[inline]
    pub fn len(self) -> usize { N }
}

/// [`ColumnSelector`] impl for single columns — returns `&[T]` / `&mut [T]`.
impl<S, const IDX: usize, T> ColumnSelector for ColumnIdx<S, IDX, T>
where
    S: Schema + Copy,
    T: Pod
{
    type Schema = S;
    type Ref<'a> = &'a [T];
    type Mut<'a> = &'a mut [T];

    fn collect_col_indices(&self, out: &mut Vec<usize>) {
        out.push(IDX);
    }

    fn get_ref<'a>(self, data: &'a [u8], row_count: usize, row_capacity: usize) -> Self::Ref<'a> {
        let offset = S::offset(IDX, row_capacity);
        let elem_size = std::mem::size_of::<T>();
        bytemuck::cast_slice(
            &data[offset..offset + elem_size * row_count])
    }

    unsafe fn get_mut<'a>(self, data: *mut u8, row_count: usize, row_capacity: usize) -> Self::Mut<'a> {
        let offset = S::offset(IDX, row_capacity);
        let elem_size = std::mem::size_of::<T>();
        unsafe {
            bytemuck::cast_slice_mut(
                core::slice::from_raw_parts_mut(
                    data.add(offset), row_count * elem_size))
        }
    }
}

/// [`ColumnSelector`] impl for group columns — returns `[&[T]; N]` / `[&mut [T]; N]`.
impl<S, const IDX: usize, const N: usize, T> ColumnSelector for ColumnGroupIdx<S, IDX, N, T>
where
    S: Schema + Copy,
    T: bytemuck::Pod
{
    type Schema = S;
    type Ref<'a> = [&'a [T]; N];
    type Mut<'a> = [&'a mut [T]; N];

    fn collect_col_indices(&self, out: &mut Vec<usize>) {
        let mut k = 0;
        while k < N {
            out.push(IDX + k);
            k += 1;
        }
    }

    fn get_ref<'a>(self, data: &'a [u8], row_count: usize, row_capacity: usize) -> Self::Ref<'a> {
        let elem_size = std::mem::size_of::<T>();
        core::array::from_fn(|k| {
            let offset = S::offset(IDX + k, row_capacity);
            bytemuck::cast_slice(
                &data[offset..offset + elem_size * row_count])
        })
    }

    unsafe fn get_mut<'a>(self, data: *mut u8, row_count: usize, row_capacity: usize) -> Self::Mut<'a> {
        let elem_size = std::mem::size_of::<T>();
        core::array::from_fn(|k| {
            let offset = S::offset(IDX + k, row_capacity);
            unsafe {
                bytemuck::cast_slice_mut(
                    core::slice::from_raw_parts_mut(
                        data.add(offset), row_count * elem_size))
            }
        })
    }
}

// =============================================================================
// Schema
// =============================================================================

/// Describes the memory layout of a SoA columnar buffer for a specific struct.
///
/// A `Schema` encodes two pieces of information:
/// - [`stride`](Schema::stride): how many bytes one logical row occupies in
///   total (sum of all field sizes).
/// - [`offset`](Schema::offset): where a given column's contiguous block starts
///   within the flat byte buffer, given the buffer's row capacity.
///
/// You do not implement this trait manually. The `#[derive(Columnar)]` macro
/// emits a `<StructName>Schema` type that implements it, along with precomputed
/// `const` arrays for zero-cost offset lookups.
///
/// ## Buffer layout
///
/// For a struct with fields `[A, B, C]` and `row_capacity = N`:
///
/// ```text
/// Byte offset 0                          N*size_of(A)          N*(size_of(A)+size_of(B))
/// |<-- col A: N elements of type A -->|<-- col B -->|<-- col C -->|
/// ```
pub trait Schema: Sized {

    /// Total bytes per logical row, the sum of `size_of` for every field.
    ///
    /// Used to calculate `row_capacity` from a raw byte buffer length.
    fn stride() -> usize;

    /// Byte offset at which the block for column `col_index` starts, given
    /// that the buffer holds `row_capacity` rows.
    ///
    /// Computed as: `sum of elem_sizes[0..col_index] * row_capacity`.
    fn offset(col_index: usize, row_capacity: usize) -> usize;
}

// =============================================================================
// ColumnSelector
// =============================================================================

/// Unified column access trait for single and group columns.
/// Implemented by `ColumnIdx` and `ColumnGroupIdx`.
pub trait ColumnSelector: Copy {
    /// The schema this selector belongs to.
    type Schema: Schema;

    /// Shared slice(s) returned for read access.
    /// - For a regular column: `&'a [T]`
    /// - For a group column `[T; N]`: `[&'a [T]; N]`
    type Ref<'a>;

    /// Mutable slice(s) returned for write access.
    /// - For a regular column: `&'a mut [T]`
    /// - For a group column `[T; N]`: `[&'a mut [T]; N]`
    type Mut<'a>;

    /// Append all col_indices occupied by this selector to `out`.
    /// Used for duplicate detection in mutable access.
    fn collect_col_indices(&self, out: &mut Vec<usize>);

    /// Extract shared typed slice(s) from raw buffer bytes.
    fn get_ref<'a>(self, data: &'a [u8], row_count: usize, row_capacity: usize) -> Self::Ref<'a>;

    /// Extract mutable typed slice(s) from a raw buffer pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the byte ranges covered by this selector
    /// do not overlap with any other concurrently borrowed ranges.
    unsafe fn get_mut<'a>(self, data: *mut u8, row_count: usize, row_capacity: usize) -> Self::Mut<'a>;
}

// =============================================================================
// SoAWrite
// =============================================================================

/// Scatter a struct's fields into the correct column blocks of a flat byte buffer.
///
/// This trait is the write half of the SoA serialisation contract. It is
/// automatically derived by `#[derive(Columnar)]` and works by calling
/// [`bytemuck::bytes_of`] on each field, meaning it supports any field type
/// that is [`bytemuck::Pod`], including primitives, fixed-size arrays like
/// `[u8; N]`, and compound `#[repr(C)]` structs.
///
/// # Safety
///
/// The implementation writes directly into the raw byte buffer at calculated
/// offsets. Correctness depends on the [`Schema`] producing accurate offsets.
/// The derive macro guarantees this via `const` prefix-sum arrays.
pub trait SoAWrite {

    /// The schema that describes the column layout this type writes into.
    type Schema: Schema;

    /// Scatter `self` into `data` at row index `row`, assuming the
    /// buffer holds `row_capacity` rows.
    fn write_into(self, data: &mut [u8], row: usize, row_capacity: usize);
}

// =============================================================================
// SoARead
// =============================================================================

/// Gather a struct's fields from the correct column blocks of a flat byte buffer.
///
/// This trait is the read half of the SoA serialisation contract and the dual
/// of [`SoAWrite`]. It is automatically derived by `#[derive(Columnar)]`.
///
/// Each field is reconstructed by reinterpreting the relevant byte slice as
/// the field's type via [`bytemuck::from_bytes`], then dereferencing to produce
/// an owned copy. This is sound for any [`bytemuck::Pod`] type.
///
pub trait SoARead {

    /// The schema that describes the column layout this type reads from.
    type Schema: Schema;

    /// Reconstruct `Self` from `data` at row index `row`, assuming the
    /// buffer holds `row_capacity` rows.
    fn read_from(data: &[u8], row: usize, row_capacity: usize) -> Self;
}

// =============================================================================
// Columns trait + macro
// =============================================================================

/// Enables retrieving multiple typed column slices from a [`ColumnarBuffer`]
/// in a single call, with compile-time type safety and runtime non-overlap
/// checking for mutable access.
///
/// Implemented for tuples of [`ColumnSelector`] values up to arity 8 via the
/// `impl_columns!` macro. You never implement this trait manually.
///
/// # Output types
///
/// For a tuple `(ColA, ColB)`:
/// - `Output`    = `(&[ColA::Value], &[ColB::Value])`
/// - `OutputMut` = `(&mut [ColA::Value], &mut [ColB::Value])`
///
/// All returned slices cover only the *valid* rows (i.e. `0..row_count`),
/// not the full allocated capacity.
///
/// # Mutable aliasing safety
///
/// [`get_mut`](Columns::get_mut) asserts at runtime that no two requested
/// columns map to the same byte range. Passing the same column token twice
/// (e.g. `(col::id, col::id)`) will panic rather than produce aliased
/// mutable references, which would be undefined behaviour.
pub trait Columns<'buffer, S: Schema, B: ByteBuffer> {

    /// The type returned by [`get_mut`](Columns::get_mut),
    /// a tuple of mutable slices, one per requested column.
    type OutputMut;

    /// The type returned by [`get`](Columns::get),
    /// a tuple of shared slices, one per requested column.
    type Output;

    /// Borrow multiple columns mutably from `buf`.
    ///
    /// # Panics
    ///
    /// Panics if any two requested columns refer to the same byte range
    /// (i.e. duplicate column tokens), as this would create aliased mutable
    /// references.
    fn get_mut(self, buf: &'buffer mut ColumnarBuffer<S, B>) -> Self::OutputMut;

    /// Borrow multiple columns immutably from `buf`.
    fn get(self, buf: &'buffer ColumnarBuffer<S, B>) -> Self::Output;
}

macro_rules! impl_columns {
    ( $( ($idx:tt, $C:ident) ),+ ) => {
        impl<'buffer, S, B, $($C),+> Columns<'buffer, S, B> for ($($C,)+)
        where
            $( $C: ColumnSelector<Schema = S>, )+
            S: Schema,
            B: ByteBuffer,
        {
            type OutputMut = ( $( $C::Mut<'buffer>, )+ );
            type Output    = ( $( $C::Ref<'buffer>, )+ );

            fn get_mut(self, buf: &'buffer mut ColumnarBuffer<S, B>) -> Self::OutputMut {
                // Collect all col_indices for duplicate detection.
                let mut all_indices: Vec<usize> = Vec::new();
                $( self.$idx.collect_col_indices(&mut all_indices); )+

                // Panic on duplicate columns — would produce aliased &mut refs (UB).
                let mut i = 0;
                while i < all_indices.len() {
                    let mut j = i + 1;
                    while j < all_indices.len() {
                        if all_indices[i] == all_indices[j] {
                            panic!("duplicate columns requested, would alias mutable references");
                        }
                        j += 1;
                    }
                    i += 1;
                }

                // SAFETY: col_indices are guaranteed non-overlapping by the check above.
                // Each selector constructs slices over distinct, non-overlapping byte
                // ranges, so no aliasing occurs.
                let data = buf.storage.as_bytes_mut().as_mut_ptr();
                let row_count = buf.row_count;
                let row_capacity = buf.row_capacity;
                unsafe {
                    ($(
                        self.$idx.get_mut(data, row_count, row_capacity),
                    )+)
                }
            }

            fn get(self, buf: &'buffer ColumnarBuffer<S, B>) -> Self::Output {
                let data = buf.storage.as_bytes();
                let row_count = buf.row_count;
                let row_capacity = buf.row_capacity;
                ($(
                    self.$idx.get_ref(data, row_count, row_capacity),
                )+)
            }
        }
    };
}

// Generate Columns impls for tuple arities 1 through 8.
impl_columns!((0, C0));
impl_columns!((0, C0), (1, C1));
impl_columns!((0, C0), (1, C1), (2, C2));
impl_columns!((0, C0), (1, C1), (2, C2), (3, C3));
impl_columns!((0, C0), (1, C1), (2, C2), (3, C3), (4, C4));
impl_columns!((0, C0), (1, C1), (2, C2), (3, C3), (4, C4), (5, C5));
impl_columns!((0, C0), (1, C1), (2, C2), (3, C3), (4, C4), (5, C5), (6, C6));
impl_columns!((0, C0), (1, C1), (2, C2), (3, C3), (4, C4), (5, C5), (6, C6), (7, C7));

// =============================================================================
// Columnar
// =============================================================================

/// A typed, cache-friendly SoA buffer backed by any [`ByteBuffer`].
///
/// `Columnar<S, B>` stores rows of data in column-major order as described
/// by schema `S`, using `B` as the raw byte storage. All columns share a
/// single contiguous allocation, partitioned into blocks: one block per field,
/// each block holding `row_capacity` elements.
///
/// # Type parameters
///
/// - `S` must implement [`Schema`]. In practice this is always a type emitted by
///   `#[derive(Columnar)]`, e.g. `Columnar<SequenceSchema, _>`.
/// - `B` must implement [`ByteBuffer`]. Use [`AlignedBox`] for plain heap storage,
///   or supply your own type for GPU-pinned memory, memory-mapped files, etc.
///
/// # Capacity vs. length
///
/// - `row_capacity`: total rows the buffer can hold, fixed at construction.
/// - `row_count`: number of rows currently written. Starts at 0, incremented
///   by [`push`](ColumnarBuffer::push) and [`push_with`](ColumnarBuffer::push_with).
///
/// Column accessors (`columns`, `mutate`) return slices of length `row_count`,
/// not `row_capacity`, so unwritten rows are never exposed.
pub struct ColumnarBuffer<S: Schema, B: ByteBuffer> {

    /// Phantom marker tying this buffer to its schema type.
    _schema: std::marker::PhantomData<S>,

    /// The underlying raw byte storage.
    pub storage: B,

    /// Maximum number of rows this buffer can hold.
    row_capacity: usize,

    /// Number of rows that have been written so far.
    row_count: usize,
}

impl<S: Schema, B: ByteBuffer> ColumnarBuffer<S, B> {

    /// Create a new columnar buffer wrapping `buffer`.
    ///
    /// `row_capacity` is computed as `buffer.as_bytes().len() / S::stride()`.
    /// The entire byte range must already be allocated and zeroed.
    pub fn new(buffer: B) -> Self {
        let row_capacity = buffer.as_bytes().len() / S::stride();
        Self {
            _schema: std::marker::PhantomData,
            row_capacity,
            storage: buffer,
            row_count: 0,
        }
    }

    /// Create a new columnar buffer wrapping `buffer`, 
    /// with rows set to capacity for in-place writing.
    ///
    /// `row_capacity` is computed as `buffer.as_bytes().len() / S::stride()`.
    /// The entire byte range must already be allocated and zeroed.
    pub fn new_complete(buffer: B) -> Self {
        let row_capacity = buffer.as_bytes().len() / S::stride();
        Self {
            _schema: std::marker::PhantomData,
            row_count: row_capacity,
            row_capacity,
            storage: buffer,
        }
    }

    // ── Ownership ─────────────────────────────────────────────────────────────

    /// Consume this buffer and return the underlying [`ByteBuffer`], discarding
    /// all type and schema information.
    ///
    /// Useful for returning the slot to a pool or passing raw bytes to an
    /// external system (e.g. a network writer or memory-mapped file).
    pub fn detach(self) -> B {
        self.storage
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// Maximum number of rows this buffer can hold.
    pub fn capacity(&self) -> usize { self.row_capacity }

    /// Returns `true` if `row_count == row_capacity` (no more rows can be pushed).
    pub fn is_full(&self) -> bool { self.row_count >= self.row_capacity }

    /// Returns `true` if no rows have been written yet (`row_count == 0`).
    pub fn is_empty(&self) -> bool { self.row_count == 0 }

    /// Number of rows that have been written to this buffer.
    pub fn len(&self) -> usize { self.row_count }

    /// Manually set the number of valid rows.
    ///
    /// Useful when external code (e.g. a CUDA kernel) writes directly into
    /// the buffer's column blocks. The caller must ensure that all column
    /// data for `0..rows` is properly initialised.
    ///
    /// # Panics
    ///
    /// Panics if `rows > row_capacity`.
    pub fn set_len(&mut self, rows: usize) {
        assert!(rows <= self.row_capacity);
        self.row_count = rows;
    }

    // ── Range helpers ─────────────────────────────────────────────────────────

    /// Return the byte range `[start, end)` covering the **valid rows**
    /// (`0..row_count`) of a column, given its byte offset and element size.
    ///
    /// Obtain `col_offset` and `elem_size` from a [`ColumnIdx`] token:
    /// ```rust,ignore
    /// let range = buf.column_content_range(col.offset(buf.capacity()), col.elem_size());
    /// ```
    pub fn column_content_range(&self, col_offset: usize, elem_size: usize) -> std::ops::Range<usize> {
        let beg = col_offset;
        beg..beg + self.row_count * elem_size
    }

    /// Return the byte range `[start, end)` covering the **full allocated
    /// block** (`0..row_capacity`) of a column.
    ///
    /// Includes unwritten rows. Useful when handing the buffer to external
    /// code (e.g. CUDA kernels) that will fill the entire block directly.
    pub fn column_capacity_range(&self, col_offset: usize, elem_size: usize) -> std::ops::Range<usize> {
        let beg = col_offset;
        beg..beg + self.row_capacity * elem_size
    }

    // ── Read access ───────────────────────────────────────────────────────────

    /// Borrow one or more columns as typed slices covering the valid rows.
    ///
    /// `cols` is a tuple of column tokens, e.g. `(sequence::schema::id,)` or
    /// `(sequence::schema::id, sequence::schema::score)`. Returns a matching
    /// tuple of `&[T]` slices.
    ///
    /// # Example
    /// ```rust,ignore
    /// let (ids, scores) = buf.columns((sequence::schema::id, sequence::schema::score));
    /// for (id, score) in ids.iter().zip(scores.iter()) {
    ///     println!("{id}: {score}");
    /// }
    /// ```
    pub fn columns<'s, C>(&'s self, cols: C) -> C::Output
    where
        C: Columns<'s, S, B>
    {
        cols.get(self)
    }

    /// Reconstruct a full row as an owned struct at index `row`.
    ///
    /// Returns `None` if `row >= row_count`. Each field is gathered from its
    /// column block via [`SoARead::read_from`].
    pub fn get<T>(&self, row: usize) -> Option<T>
    where
        T: SoARead<Schema = S>
    {
        if row >= self.row_count { return None; }
        Some(T::read_from(self.storage.as_bytes(), row, self.row_capacity))
    }

    // ── Write access ──────────────────────────────────────────────────────────

    /// Borrow one or more columns as mutable typed slices, then apply a
    /// mutation closure.
    ///
    /// The closure receives a tuple of `&mut [T]` slices, one per requested
    /// column. The `&mut self` borrow is held for the duration of the closure,
    /// ensuring safe exclusive access without lifetime gymnastics at the call site.
    ///
    /// # Panics
    ///
    /// Panics if the same column token appears more than once in `cols`, as
    /// that would create aliased mutable references.
    ///
    /// # Example
    /// ```rust,ignore
    /// buf.mutate((sequence::schema::score,), |(scores,)| {
    ///     for i in 0..scores.len() {
    ///         scores[i] *= 1.1;  // boost all scores in-place
    ///     }
    /// });
    /// ```
    pub fn mutate<C, F>(&mut self, cols: C, mutation: F)
    where
        F: for<'s> FnOnce(<C as Columns<'s, S, B>>::OutputMut),
        C: for<'s> Columns<'s, S, B>,
    {
        let cols_mut = cols.get_mut(self);
        mutation(cols_mut);
    }

    /// Append a new row by filling selected columns via a closure.
    ///
    /// `row_count` is incremented **before** the closure is called, so the
    /// column slices passed to the closure already include the new (zeroed)
    /// slot at the last index. The closure receives the index of the new row
    /// as its first argument for convenient addressing.
    ///
    /// Columns not included in `cols` retain their zero-initialised value for
    /// the new row.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is full (`row_count == row_capacity`).
    ///
    /// # Example
    /// ```rust,ignore
    /// buf.push_with(
    ///     (sequence_schema::id, sequence_schema::score),
    ///     |row, (ids, scores)| {
    ///         ids[row]    = 99;
    ///         scores[row] = 0.75;
    ///     }
    /// );
    /// ```
    pub fn push_with<C, F>(&mut self, cols: C, fill: F)
    where
        F: for<'s> FnOnce(usize, <C as Columns<'s, S, B>>::OutputMut),
        C: for<'s> Columns<'s, S, B>,
    {
        assert!(!self.is_full(), "Columnar: buffer full");
        let write_idx = self.row_count;
        self.row_count += 1;

        let cols_mut = cols.get_mut(self);
        fill(write_idx, cols_mut);
    }

    /// Append a fully constructed struct as a new row.
    ///
    /// Each field is scattered into its column block via [`SoAWrite::write_into`].
    /// This is the preferred API when you have all fields available at once.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is full (`row_count == row_capacity`).
    ///
    /// # Example
    /// ```rust,ignore
    /// buf.push(Sequence { id: 1, score: 0.95, elements: [0u8; 32] });
    /// ```
    pub fn push<T>(&mut self, value: T)
    where
        T: SoAWrite<Schema = S>
    {
        assert!(!self.is_full(), "Columnar: buffer full");
        value.write_into(self.storage.as_bytes_mut(), self.row_count, self.row_capacity);
        self.row_count += 1;
    }
}
