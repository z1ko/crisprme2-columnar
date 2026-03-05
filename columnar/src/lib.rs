
// Underlying buffer
pub struct RingSlot {
    pub data: Vec<u8>
}

impl RingSlot {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0u8; capacity]
        }
    }

    /// Create a columnar buffer out of this buffer,
    /// takes ownership of self
    pub fn columnar<S: Schema>(self) -> ColumnarBuffer<S> {
        ColumnarBuffer { 
            _schema: std::marker::PhantomData,
            row_capacity: self.data.len() / S::stride(),
            ring_slot: self,
            row_count: 0,
        }
    }
}

// =========================================================================

/// Describes the layout of a SoA columnar buffer.
/// Columns are stored as contiguous blocks:
/// [ col0[0..n] | col1[0..n] | col2[0..n] | ... ]
pub trait Schema: Sized {

    /// Total bytes per row, summed across all column types.
    fn stride() -> usize;
    /// Byte offset where a column block starts, given the buffer row capacity.
    /// Sum of sizes of all previous columns * row_capacity
    fn offset(col_index: usize, row_capacity: usize) -> usize;
}

/// A typed column token. Each field of a struct gets one ZST that implements
/// this, carrying the field's type as `Value` and its schema as `Schema`.
pub trait ColumnType: Copy {

    /// The schema this column belongs to.
    type Schema: Schema;
    /// The Rust type of this column's elements.
    type Value: bytemuck::Pod;

    /// Index of this column among all columns in the schema.
    fn col_index(self) -> usize;
    /// Byte offset of this field within one row.
    fn offset(self, row_capacity: usize) -> usize;
    /// Size in bytes of the type in this column
    fn elem_size(self) -> usize {
        std::mem::size_of::<Self::Value>()
    }
}

pub trait SoAWrite {
    type Schema: Schema;

    /// Scatters each field into the correct column block.
    fn write_into(self, data: &mut [u8], row: usize, row_capacity: usize);
}

#[derive(Debug, Clone, Copy)]
pub struct ColumnDescriptor {
    /// Byte offset where this column's contiguous block starts.
    pub block_start: usize,
    /// Size in bytes of one element.
    pub elem_size: usize,
}

/// Allow the user to extract multiple mutable columns
pub trait ColumnsMut<'a, S: Schema> {
    type Output;

    /// # Safety
    /// Caller must guarantee `data` is valid for the entire buffer,
    /// and that no other references to the same regions exist.
    unsafe fn columns_mut(self, buffer: &mut ColumnarBuffer<S>) -> Self::Output;
}

impl<'a, S, C0> ColumnsMut<'a, S> for (C0,)
where
    C0: ColumnType<Schema = S>,
    S: Schema,
{
    type Output = (&'a mut [C0::Value],);

    unsafe fn columns_mut(self, buffer: &mut ColumnarBuffer<S>) -> Self::Output {
        let (b0, e0) = buffer.column_block_range(&self.0);
        let data = buffer.ring_slot.data.as_mut_ptr();
        (
            bytemuck::cast_slice_mut(
                unsafe { std::slice::from_raw_parts_mut(data.add(b0), e0 - b0) }),
        )
    }
}

impl<'a, S, C0, C1> ColumnsMut<'a, S> for (C0, C1)
where
    C0: ColumnType<Schema = S>,
    C1: ColumnType<Schema = S>,
    S: Schema,
{
    type Output = (&'a mut [C0::Value], &'a mut [C1::Value]);

    unsafe fn columns_mut(self, buffer: &mut ColumnarBuffer<S>) -> Self::Output {
        
        assert_ne!(self.0.col_index(), self.1.col_index(), "duplicate column");

        let (b0, e0) = buffer.column_block_range(&self.0);
        let (b1, e1) = buffer.column_block_range(&self.1);

        let data = buffer.ring_slot.data.as_mut_ptr();
        (
            bytemuck::cast_slice_mut(
                unsafe { std::slice::from_raw_parts_mut(data.add(b0), e0 - b0) }),
            bytemuck::cast_slice_mut(
                unsafe { std::slice::from_raw_parts_mut(data.add(b1), e1 - b1) }),
        )
    }
}

impl<'a, S, C0, C1, C2> ColumnsMut<'a, S> for (C0, C1, C2)
where
    C0: ColumnType<Schema = S>,
    C1: ColumnType<Schema = S>,
    C2: ColumnType<Schema = S>,
    S: Schema,
{
    type Output = (&'a mut [C0::Value], &'a mut [C1::Value], &'a mut [C2::Value]);

    unsafe fn columns_mut(self, buffer: &mut ColumnarBuffer<S>) -> Self::Output {
        
        assert_ne!(self.0.col_index(), self.1.col_index(), "duplicate column");
        assert_ne!(self.1.col_index(), self.2.col_index(), "duplicate column");
        assert_ne!(self.2.col_index(), self.0.col_index(), "duplicate column");

        let (b0, e0) = buffer.column_block_range(&self.0);
        let (b1, e1) = buffer.column_block_range(&self.1);
        let (b2, e2) = buffer.column_block_range(&self.2);

        let data = buffer.ring_slot.data.as_mut_ptr();
        (
            bytemuck::cast_slice_mut(
                unsafe { std::slice::from_raw_parts_mut(data.add(b0), e0 - b0) }),
            bytemuck::cast_slice_mut(
                unsafe { std::slice::from_raw_parts_mut(data.add(b1), e1 - b1) }),
            bytemuck::cast_slice_mut(
                unsafe { std::slice::from_raw_parts_mut(data.add(b2), e2 - b2) }),
        )
    }
}

/// Allow the user to extract multiple columns
pub trait Columns<'a, S: Schema> {
    type Output;

    /// # Safety
    /// Caller must guarantee `data` is valid for the entire buffer,
    /// and that no other references to the same regions exist.
    unsafe fn columns(self, buffer: &ColumnarBuffer<S>) -> Self::Output;
}

impl<'a, S, C0> Columns<'a, S> for (C0,)
where
    C0: ColumnType<Schema = S>,
    S: Schema,
{
    type Output = (&'a [C0::Value],);

    unsafe fn columns(self, buffer: &ColumnarBuffer<S>) -> Self::Output {

        let (b0, e0) = buffer.column_block_range(&self.0);

        let data = buffer.ring_slot.data.as_ptr();
        (
            bytemuck::cast_slice(
                unsafe { std::slice::from_raw_parts(data.add(b0), e0 - b0) }),
        )
    }
}

/*
impl<'a, S, C0, C1> Columns<'a, S> for (C0, C1)
where
    C0: ColumnType<Schema = S>,
    C1: ColumnType<Schema = S>,
    S: Schema,
{
    type Output = (&'a [C0::Value], &'a [C1::Value]);

    unsafe fn columns(self, buffer: &ColumnarBuffer<S>) -> Self::Output {

        let (b0, e0) = buffer.column_block_range(&self.0);
        let (b1, e1) = buffer.column_block_range(&self.1);

        let data = buffer.ring_slot.data.as_ptr();
        (
            bytemuck::cast_slice(
                unsafe { std::slice::from_raw_parts(data.add(b0), e0 - b0) }),
            bytemuck::cast_slice(
                unsafe { std::slice::from_raw_parts(data.add(b1), e1 - b1) }),
        )
    }
}
*/

/// View a buffer like columnar data
pub struct ColumnarBuffer<S: Schema> {
    _schema: std::marker::PhantomData<S>,
    ring_slot: RingSlot,
    row_capacity: usize,
    row_count: usize,
}

impl<S: Schema> ColumnarBuffer<S> {
    
    /// Take ownership of the inner ring slot
    pub fn detach(self) -> RingSlot {
        self.ring_slot
    }

    pub fn capacity(&self) -> usize { self.row_capacity }
    pub fn is_full(&self)  -> bool  { self.row_count >= self.row_capacity }
    pub fn is_empty(&self) -> bool  { self.row_count == 0 }
    pub fn len(&self)      -> usize { self.row_count }

    /// Calculate the range for a column
    pub fn column_block_range<C: ColumnType>(&self, col: &C) -> (usize, usize) {
        let beg = col.offset(self.row_capacity);
        let end = beg + self.row_count * col.elem_size();
        (beg, end)
    }

    /// Get columns
    pub fn columns<'s, C>(&'s self, cols: C) -> C::Output
    where
        C: Columns<'s, S> 
    {
        // SAFETY: Some pointer manipulation to simplify the logic
        unsafe { cols.columns(self) }
    }

    /// Get mutable columns
    pub fn columns_mut<'s, C>(&'s mut self, cols: C) -> C::Output
    where 
        C: ColumnsMut<'s, S> 
    {
        // SAFETY: We have asserts in ColumnsMut, the columns must be different 
        unsafe { cols.columns_mut(self) }
    }

    pub fn mutate<C, F>(&mut self, cols: C, mutation: F) 
    where 
        C: for<'s> ColumnsMut<'s, S>, 
        F: for<'s> FnOnce(<C as ColumnsMut<'s, S>>::Output),
    {
        let cols_mut = self.columns_mut(cols);
        mutation(cols_mut);
    }

    /// Get single column element
    pub fn get<C>(&self, row: usize, col: C) -> Option<C::Value>
    where
        C: ColumnType<Schema = S>
    {
        self.columns((col,))
            .0.get(row)
            .cloned()
    }

    /// Add a single structure
    pub fn push<T>(&mut self, value: T) 
    where
        T: SoAWrite<Schema = S> 
    {
        assert!(!self.is_full(), "ColumnarBuffer: buffer full");
        value.write_into(&mut self.ring_slot.data, self.row_count, self.row_capacity);
        self.row_count += 1;
    }
}