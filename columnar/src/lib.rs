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
//! let mut buf: Columnar<SequenceSchema, RingSlot> =
//!     RingSlot::new(1024 * SequenceSchema::STRIDE).columnar();
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

#[cfg(test)]
extern crate self as columnar;

#[cfg(test)]
mod tests;

// =============================================================================
// ByteBuffer
// =============================================================================

/// Trait for types that can serve as the raw byte backing store of a [`Columnar`] buffer.
///
/// Implement this for any contiguous, byte-addressable allocation you want to use
/// as columnar storage — a plain heap `Vec<u8>`, a GPU-pinned allocation, a
/// memory-mapped file, a borrowed slice, etc.
///
/// # Requirements
///
/// The byte slice returned by `as_bytes` and `as_bytes_mut` must be the **same**
/// contiguous region on every call, with a length that does not change after the
/// buffer is handed to [`Columnar::new`].
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
// RingSlot
// =============================================================================

/// A plain heap-allocated byte slab intended to be used as a slot in a ring
/// buffer pool, but usable as a standalone backing store for [`Columnar`].
pub struct RingSlot {
    pub data: Vec<u8>
}

impl RingSlot {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0u8; capacity]
        }
    }

    /// Wrap this slot in a typed columnar buffer. Takes ownership of `self`.
    pub fn columnar<S: Schema>(self) -> Columnar<S, RingSlot> {
        Columnar::new(self)
    }
}

impl ByteBuffer for RingSlot {
    fn as_bytes(&self) -> &[u8] { &self.data }
    fn as_bytes_mut(&mut self) -> &mut [u8] { &mut self.data }
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
// ColumnType
// =============================================================================

/// A zero-sized type (ZST) token representing a single typed column within a
/// [`Schema`].
///
/// The `#[derive(Columnar)]` macro emits one ZST per struct field inside the
/// `schema` module of the module where the struct is defined.
/// These tokens are used as keys to address columns inside a [`Columnar`]
/// without any runtime cost.
///
/// # Type safety
///
/// Because `Schema` is an associated type, passing a column token from one
/// schema to a buffer of a different schema is a **compile-time error**:
///
/// ```rust,ignore
/// // ✓ correct — token matches the buffer's schema
/// sequences.columns((sequence::schema::id,));
///
/// // ✗ compile error — token belongs to a different schema
/// sequences.columns((alignment::schema::score,));
/// ```
pub trait ColumnType: Copy {

    /// The schema this column token belongs to. Used by the compiler to
    /// prevent mixing column tokens from different schemas.
    type Schema: Schema;

    /// The Rust element type stored in this column. Must implement
    /// [`bytemuck::Pod`] for safe byte-level reinterpretation.
    type Value: bytemuck::Pod;

    /// Zero-based index of this column among all columns in the schema,
    /// in field declaration order.
    fn col_index(self) -> usize;

    /// Byte offset of this column's contiguous block within a buffer of
    /// `row_capacity` rows. Delegates to [`Schema::offset`].
    fn offset(self, row_capacity: usize) -> usize;

    /// Size in bytes of a single element in this column.
    fn elem_size(self) -> usize {
        std::mem::size_of::<Self::Value>()
    }
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

/// Enables retrieving multiple typed column slices from a [`Columnar`]
/// in a single call, with compile-time type safety and runtime non-overlap
/// checking for mutable access.
///
/// Implemented for tuples of [`ColumnType`] values up to arity 8 via the
/// [`impl_columns!`] macro. You never implement this trait manually.
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
    fn get_mut(self, buf: &'buffer mut Columnar<S, B>) -> Self::OutputMut;

    /// Borrow multiple columns immutably from `buf`.
    fn get(self, buf: &'buffer Columnar<S, B>) -> Self::Output;
}

macro_rules! impl_columns {
    ( $( ($idx:tt, $C:ident) ),+ ) => {
        impl<'buffer, S, B, $($C),+> Columns<'buffer, S, B> for ($($C,)+)
        where
            $( $C: ColumnType<Schema = S>, )+
            S: Schema,
            B: ByteBuffer,
        {
            type OutputMut = ( $( &'buffer mut [$C::Value], )+ );
            type Output    = ( $( &'buffer     [$C::Value], )+ );

            fn get_mut(self, buf: &'buffer mut Columnar<S, B>) -> Self::OutputMut {
                let ranges: &[std::ops::Range<usize>] = &[
                    $( buf.column_content_range(&self.$idx), )+
                ];

                // Panic on duplicate columns — would produce aliased &mut refs (UB).
                // O(n²) over the tuple arity, which is at most 8 — negligible cost.
                let mut i = 0;
                while i < ranges.len() {
                    let mut j = i + 1;
                    while j < ranges.len() {
                        if ranges[i] == ranges[j] {
                            panic!("duplicate columns requested, would alias mutable references");
                        }
                        j += 1;
                    }
                    i += 1;
                }

                // SAFETY: ranges are guaranteed non-overlapping by the check above.
                // We obtain a raw pointer to the allocation and construct independent
                // sub-slices from it. Each sub-slice covers a distinct, non-overlapping
                // byte range, so no aliasing occurs.
                let data = buf.buffer.as_bytes_mut().as_mut_ptr();
                ($(
                    bytemuck::cast_slice_mut(unsafe {
                        std::slice::from_raw_parts_mut(
                            data.add(ranges[$idx].start),
                            ranges[$idx].len(),
                        )
                    }),
                )+)
            }

            fn get(self, buf: &'buffer Columnar<S, B>) -> Self::Output {
                let ranges: &[std::ops::Range<usize>] = &[
                    $( buf.column_content_range(&self.$idx), )+
                ];
                ($(
                    bytemuck::cast_slice(
                        &buf.buffer.as_bytes()[ranges[$idx].clone()]
                    ),
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
/// - `B` must implement [`ByteBuffer`]. Use [`RingSlot`] for plain heap storage,
///   or supply your own type for GPU-pinned memory, memory-mapped files, etc.
///
/// # Capacity vs. length
///
/// - `row_capacity`: total rows the buffer can hold, fixed at construction.
/// - `row_count`: number of rows currently written. Starts at 0, incremented
///   by [`push`](Columnar::push) and [`push_with`](Columnar::push_with).
///
/// Column accessors (`columns`, `mutate`) return slices of length `row_count`,
/// not `row_capacity`, so unwritten rows are never exposed.
pub struct Columnar<S: Schema, B: ByteBuffer> {

    /// Phantom marker tying this buffer to its schema type.
    _schema: std::marker::PhantomData<S>,

    /// The underlying raw byte storage.
    pub buffer: B,

    /// Maximum number of rows this buffer can hold.
    row_capacity: usize,

    /// Number of rows that have been written so far.
    row_count: usize,
}

impl<S: Schema, B: ByteBuffer> Columnar<S, B> {

    /// Create a new columnar buffer wrapping `buffer`.
    ///
    /// `row_capacity` is computed as `buffer.as_bytes().len() / S::stride()`.
    /// The entire byte range must already be allocated and zeroed.
    pub fn new(buffer: B) -> Self {
        let row_capacity = buffer.as_bytes().len() / S::stride();
        Self {
            _schema: std::marker::PhantomData,
            row_capacity,
            buffer,
            row_count: 0,
        }
    }

    // ── Ownership ─────────────────────────────────────────────────────────────

    /// Consume this buffer and return the underlying [`ByteBuffer`], discarding
    /// all type and schema information.
    ///
    /// Useful for returning the slot to a pool or passing raw bytes to an
    /// external system (e.g. a network writer or memory-mapped file).
    pub fn detach(self) -> B {
        self.buffer
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

    // ── Range helpers ─────────────────────────────────────────────────────────

    /// Return the byte range `[start, end)` covering the **valid elements**
    /// (rows `0..row_count`) of column `col` within the raw buffer.
    ///
    /// This is the range used by [`columns`](Columnar::columns) and
    /// [`mutate`](Columnar::mutate) to construct column slices.
    pub fn column_content_range<C: ColumnType>(&self, col: &C) -> std::ops::Range<usize> {
        let beg = col.offset(self.row_capacity);
        beg..beg + self.row_count * col.elem_size()
    }

    /// Return the byte range `[start, end)` covering the **full allocated block**
    /// (rows `0..row_capacity`) of column `col` within the raw buffer.
    ///
    /// Includes unwritten rows. Useful when handing the buffer to external
    /// code (e.g. CUDA kernels) that will fill the entire block directly.
    pub fn column_capacity_range<C: ColumnType>(&self, col: &C) -> std::ops::Range<usize> {
        let beg = col.offset(self.row_capacity);
        beg..beg + self.row_capacity * col.elem_size()
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
        Some(T::read_from(self.buffer.as_bytes(), row, self.row_capacity))
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
        value.write_into(self.buffer.as_bytes_mut(), self.row_count, self.row_capacity);
        self.row_count += 1;
    }
}
