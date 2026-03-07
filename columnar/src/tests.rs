// Each struct that derives Columnar emits a `pub mod schema { ... }` in the
// surrounding module, so multiple derived structs must live in separate modules
// to avoid the name conflict.

use crate::buffer::{ByteBuffer, ColumnarBuffer, RingSlot, Schema};
use crate::macros::Columnar;

mod point {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Columnar)]
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }
}

mod record {
    use super::*;
    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Columnar)]
    pub struct Record {
        pub id:    u32,
        pub score: f32,
        pub tag:   [u8; 4],
    }
}

mod single {
    use super::*;
    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Columnar)]
    pub struct Single {
        pub value: u64,
    }
}

mod wide {
    use super::*;
    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Columnar)]
    pub struct Wide {
        pub a: u8, pub b: u8, pub c: u8, pub d: u8,
        pub e: u8, pub f: u8, pub g: u8, pub h: u8,
    }
}

// Declaration order: a(u8, align 1), b(u32, align 4), c(u8, align 1)
// Expected block order: b first (align 4), then a and c (align 1, stable)
mod scrambled {
    use super::*;
    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Columnar)]
    pub struct Scrambled {
        pub a: u8,
        pub b: u32,
        pub c: u8,
    }
}

use point::{Point, PointSchema, schema as ps};
use record::{Record, RecordSchema, schema as rs};
use single::{Single, SingleSchema, schema as ss};
use wide::{Wide, WideSchema, schema as ws};
use scrambled::{Scrambled, ScrambledSchema, schema as scr};

fn point_buf(rows: usize) -> ColumnarBuffer<PointSchema, RingSlot> {
    RingSlot::new(rows * PointSchema::stride()).columnar()
}

fn record_buf(rows: usize) -> ColumnarBuffer<RecordSchema, RingSlot> {
    RingSlot::new(rows * RecordSchema::stride()).columnar()
}

// =============================================================================
// Schema constants
// =============================================================================

#[test]
fn schema_elem_sizes_point() {
    assert_eq!(PointSchema::ELEM_SIZES, [4, 4]);
}

#[test]
fn schema_stride_point() {
    assert_eq!(PointSchema::STRIDE, 8);
}

#[test]
fn schema_block_offsets_point() {
    // prefix-sum: [0, 4]
    assert_eq!(PointSchema::BLOCK_OFFSETS, [0, 4]);
}

#[test]
fn schema_elem_sizes_record() {
    assert_eq!(RecordSchema::ELEM_SIZES, [4, 4, 4]);
}

#[test]
fn schema_stride_record() {
    assert_eq!(RecordSchema::STRIDE, 12);
}

#[test]
fn schema_block_offsets_record() {
    assert_eq!(RecordSchema::BLOCK_OFFSETS, [0, 4, 8]);
}

#[test]
fn schema_single_field() {
    assert_eq!(SingleSchema::ELEM_SIZES, [8]);
    assert_eq!(SingleSchema::STRIDE, 8);
    assert_eq!(SingleSchema::BLOCK_OFFSETS, [0]);
}

// =============================================================================
// Construction and metadata
// =============================================================================

#[test]
fn new_capacity_from_byte_len() {
    let buf = point_buf(16);
    assert_eq!(buf.capacity(), 16);
}

#[test]
fn new_vec_backing() {
    let raw = vec![0u8; 10 * PointSchema::STRIDE];
    let buf: ColumnarBuffer<PointSchema, Vec<u8>> = ColumnarBuffer::new(raw);
    assert_eq!(buf.capacity(), 10);
}

#[test]
fn initial_state_empty() {
    let buf = point_buf(8);
    assert!(buf.is_empty());
    assert!(!buf.is_full());
    assert_eq!(buf.len(), 0);
}

#[test]
fn capacity_unchanged_after_push() {
    let mut buf = point_buf(4);
    buf.push(Point { x: 1.0, y: 2.0 });
    assert_eq!(buf.capacity(), 4);
}

// =============================================================================
// push + get round-trip
// =============================================================================

#[test]
fn push_single_get_round_trip() {
    let mut buf = point_buf(8);
    buf.push(Point { x: 1.5, y: -3.0 });
    assert_eq!(buf.get::<Point>(0).unwrap(), Point { x: 1.5, y: -3.0 });
}

#[test]
fn push_multiple_get_each() {
    let mut buf = point_buf(8);
    let pts = [
        Point { x: 0.0, y: 0.0 },
        Point { x: 1.0, y: 2.0 },
        Point { x: -1.0, y: 100.0 },
    ];
    for p in &pts { buf.push(p.clone()); }
    for (i, expected) in pts.iter().enumerate() {
        assert_eq!(&buf.get::<Point>(i).unwrap(), expected);
    }
}

#[test]
fn push_record_round_trip() {
    let mut buf = record_buf(4);
    let r = Record { id: 42, score: 0.99, tag: [1, 2, 3, 4] };
    buf.push(r.clone());
    assert_eq!(buf.get::<Record>(0).unwrap(), r);
}

#[test]
fn get_returns_none_on_empty() {
    let buf = point_buf(8);
    assert!(buf.get::<Point>(0).is_none());
}

#[test]
fn get_returns_none_out_of_bounds() {
    let mut buf = point_buf(8);
    buf.push(Point { x: 1.0, y: 1.0 });
    assert!(buf.get::<Point>(1).is_none());
    assert!(buf.get::<Point>(100).is_none());
}

#[test]
fn get_last_row_valid() {
    let mut buf = point_buf(4);
    for i in 0..4u32 { buf.push(Point { x: i as f32, y: 0.0 }); }
    assert!(buf.get::<Point>(3).is_some());
    assert!(buf.get::<Point>(4).is_none());
}

// =============================================================================
// len / is_empty / is_full
// =============================================================================

#[test]
fn len_increments_on_push() {
    let mut buf = point_buf(4);
    assert_eq!(buf.len(), 0);
    buf.push(Point { x: 0.0, y: 0.0 });
    assert_eq!(buf.len(), 1);
    buf.push(Point { x: 0.0, y: 0.0 });
    assert_eq!(buf.len(), 2);
}

#[test]
fn is_full_after_filling() {
    let mut buf = point_buf(3);
    for _ in 0..3 { buf.push(Point { x: 0.0, y: 0.0 }); }
    assert!(buf.is_full());
    assert!(!buf.is_empty());
}

// =============================================================================
// Push panics when full
// =============================================================================

#[test]
#[should_panic(expected = "buffer full")]
fn push_when_full_panics() {
    let mut buf = point_buf(2);
    for _ in 0..3 { buf.push(Point { x: 0.0, y: 0.0 }); }
}

#[test]
#[should_panic(expected = "buffer full")]
fn push_with_when_full_panics() {
    let mut buf = point_buf(1);
    buf.push(Point { x: 0.0, y: 0.0 });
    buf.push_with((ps::x,), |row, (xs,)| { xs[row] = 1.0; });
}

// =============================================================================
// push_with
// =============================================================================

#[test]
fn push_with_partial_columns_rest_zero() {
    let mut buf = point_buf(4);
    buf.push_with((ps::x,), |row, (xs,)| { xs[row] = 99.0; });
    let p: Point = buf.get(0).unwrap();
    assert_eq!(p.x, 99.0);
    assert_eq!(p.y, 0.0); // never written — stays zeroed
}

#[test]
fn push_with_all_columns() {
    let mut buf = point_buf(4);
    buf.push_with((ps::x, ps::y), |row, (xs, ys)| {
        xs[row] = 3.0;
        ys[row] = 4.0;
    });
    assert_eq!(buf.get::<Point>(0).unwrap(), Point { x: 3.0, y: 4.0 });
}

#[test]
fn push_with_receives_correct_row_index() {
    let mut buf = point_buf(4);
    buf.push(Point { x: 0.0, y: 0.0 });
    buf.push(Point { x: 0.0, y: 0.0 });

    let mut captured_row = usize::MAX;
    buf.push_with((ps::x,), |row, (xs,)| {
        captured_row = row;
        xs[row] = 7.0;
    });
    assert_eq!(captured_row, 2);
    assert_eq!(buf.get::<Point>(2).unwrap().x, 7.0);
}

// =============================================================================
// columns (shared slices)
// =============================================================================

#[test]
fn columns_single_returns_correct_values() {
    let mut buf = point_buf(4);
    for i in 0..4u32 { buf.push(Point { x: i as f32, y: -(i as f32) }); }
    let (xs,) = buf.columns((ps::x,));
    assert_eq!(xs, &[0.0f32, 1.0, 2.0, 3.0]);
}

#[test]
fn columns_two_correct_values() {
    let mut buf = point_buf(3);
    buf.push(Point { x: 1.0, y: 10.0 });
    buf.push(Point { x: 2.0, y: 20.0 });
    buf.push(Point { x: 3.0, y: 30.0 });

    let (xs, ys) = buf.columns((ps::x, ps::y));
    assert_eq!(xs, &[1.0f32, 2.0, 3.0]);
    assert_eq!(ys, &[10.0f32, 20.0, 30.0]);
}

#[test]
fn columns_length_matches_row_count() {
    let mut buf = point_buf(8);
    for _ in 0..5 { buf.push(Point { x: 0.0, y: 0.0 }); }
    let (xs,) = buf.columns((ps::x,));
    assert_eq!(xs.len(), 5);
}

#[test]
fn columns_on_empty_returns_empty_slice() {
    let buf = point_buf(8);
    let (xs,) = buf.columns((ps::x,));
    assert!(xs.is_empty());
}

#[test]
fn columns_record_three_fields() {
    let mut buf = record_buf(2);
    buf.push(Record { id: 1, score: 0.5, tag: [9, 8, 7, 6] });
    buf.push(Record { id: 2, score: 1.0, tag: [1, 2, 3, 4] });

    let (ids, scores, tags) = buf.columns((rs::id, rs::score, rs::tag));
    assert_eq!(ids,    &[1u32, 2]);
    assert_eq!(scores, &[0.5f32, 1.0]);
    assert_eq!(tags,   &[[9u8, 8, 7, 6], [1, 2, 3, 4]]);
}

// =============================================================================
// mutate
// =============================================================================

#[test]
fn mutate_modifies_values_in_place() {
    let mut buf = point_buf(4);
    for i in 0..4u32 { buf.push(Point { x: i as f32, y: 0.0 }); }
    buf.mutate((ps::x,), |(xs,)| {
        for x in xs.iter_mut() { *x *= 2.0; }
    });
    let (xs,) = buf.columns((ps::x,));
    assert_eq!(xs, &[0.0f32, 2.0, 4.0, 6.0]);
}

#[test]
fn mutate_two_columns_independently() {
    let mut buf = point_buf(2);
    buf.push(Point { x: 1.0, y: 1.0 });
    buf.push(Point { x: 2.0, y: 2.0 });

    buf.mutate((ps::x, ps::y), |(xs, ys)| {
        xs[0] = 10.0;
        ys[1] = 20.0;
    });

    assert_eq!(buf.get::<Point>(0).unwrap(), Point { x: 10.0, y: 1.0 });
    assert_eq!(buf.get::<Point>(1).unwrap(), Point { x: 2.0,  y: 20.0 });
}

#[test]
fn mutate_only_exposes_written_rows() {
    let mut buf = point_buf(8);
    buf.push(Point { x: 5.0, y: 0.0 });
    buf.push(Point { x: 5.0, y: 0.0 });
    buf.mutate((ps::x,), |(xs,)| {
        assert_eq!(xs.len(), 2); // only written rows visible, not capacity
    });
}

// =============================================================================
// Duplicate column panics
// =============================================================================

#[test]
#[should_panic(expected = "duplicate columns")]
fn mutate_duplicate_column_panics() {
    let mut buf = point_buf(4);
    buf.push(Point { x: 1.0, y: 1.0 });
    buf.mutate((ps::x, ps::x), |_| {});
}

#[test]
#[should_panic(expected = "duplicate columns")]
fn push_with_duplicate_column_panics() {
    let mut buf = point_buf(4);
    buf.push_with((ps::x, ps::x), |_, _| {});
}

// =============================================================================
// SoA memory layout verification
// =============================================================================

#[test]
fn soa_layout_two_columns_are_contiguous() {
    // 4-row Point layout:
    // bytes  0..16 = [x0, x1, x2, x3]  (4 * 4 bytes)
    // bytes 16..32 = [y0, y1, y2, y3]
    let mut buf = point_buf(4);
    for i in 0..4u32 {
        buf.push(Point { x: i as f32, y: (i * 10) as f32 });
    }
    let raw = buf.storage.as_bytes();
    let xs: &[f32] = bytemuck::cast_slice(&raw[0..16]);
    let ys: &[f32] = bytemuck::cast_slice(&raw[16..32]);
    assert_eq!(xs, &[0.0f32, 1.0, 2.0, 3.0]);
    assert_eq!(ys, &[0.0f32, 10.0, 20.0, 30.0]);
}

#[test]
fn soa_layout_three_columns() {
    // Record layout for 2 rows (stride = 12):
    // bytes  0..8  = [id0, id1]       (2 * 4)
    // bytes  8..16 = [score0, score1] (2 * 4)
    // bytes 16..24 = [tag0, tag1]     (2 * 4)
    let mut buf = record_buf(2);
    buf.push(Record { id: 10, score: 0.1, tag: [1, 2, 3, 4] });
    buf.push(Record { id: 20, score: 0.2, tag: [5, 6, 7, 8] });

    let raw = buf.storage.as_bytes();
    let ids:    &[u32]    = bytemuck::cast_slice(&raw[0..8]);
    let scores: &[f32]    = bytemuck::cast_slice(&raw[8..16]);
    let tags:   &[[u8;4]] = bytemuck::cast_slice(&raw[16..24]);
    assert_eq!(ids,    &[10u32, 20]);
    assert_eq!(scores, &[0.1f32, 0.2]);
    assert_eq!(tags,   &[[1u8,2,3,4], [5,6,7,8]]);
}

// =============================================================================
// column_content_range / column_capacity_range
// =============================================================================

#[test]
fn content_range_empty() {
    let buf = point_buf(8);
    let range = buf.column_content_range(&ps::x);
    assert_eq!(range.len(), 0);
    // start is still at the column block offset, just 0-length
    assert_eq!(range.start, PointSchema::BLOCK_OFFSETS[0] * 8);
}

#[test]
fn content_range_grows_with_rows() {
    let mut buf = point_buf(8);
    buf.push(Point { x: 0.0, y: 0.0 });
    assert_eq!(buf.column_content_range(&ps::x).len(), 4);   // 1 * sizeof(f32)
    buf.push(Point { x: 0.0, y: 0.0 });
    assert_eq!(buf.column_content_range(&ps::x).len(), 8);   // 2 * sizeof(f32)
}

#[test]
fn capacity_range_is_always_full_block() {
    let mut buf = point_buf(8);
    let before = buf.column_capacity_range(&ps::x);
    buf.push(Point { x: 0.0, y: 0.0 });
    let after = buf.column_capacity_range(&ps::x);
    assert_eq!(before, after);
    assert_eq!(before.len(), 8 * 4); // capacity * sizeof(f32)
}

// =============================================================================
// detach / re-wrap
// =============================================================================

#[test]
fn detach_and_rewrap_preserves_bytes() {
    let mut buf = point_buf(4);
    buf.push(Point { x: 1.0, y: 2.0 });

    let slot: RingSlot = buf.detach();
    // Rewrap — row_count resets, but bytes are intact
    let buf2: ColumnarBuffer<PointSchema, _> = ColumnarBuffer::new(slot);
    let xs: &[f32] = bytemuck::cast_slice(&buf2.storage.as_bytes()[0..16]);
    assert_eq!(xs[0], 1.0);
}

// =============================================================================
// Vec<u8> as ByteBuffer
// =============================================================================

#[test]
fn vec_backing_push_and_get() {
    let raw = vec![0u8; 8 * PointSchema::STRIDE];
    let mut buf: ColumnarBuffer<PointSchema, Vec<u8>> = ColumnarBuffer::new(raw);
    buf.push(Point { x: 7.0, y: -7.0 });
    assert_eq!(buf.get::<Point>(0).unwrap(), Point { x: 7.0, y: -7.0 });
}

#[test]
fn vec_backing_columns_correct() {
    let raw = vec![0u8; 4 * PointSchema::STRIDE];
    let mut buf: ColumnarBuffer<PointSchema, Vec<u8>> = ColumnarBuffer::new(raw);
    for i in 0..4u32 { buf.push(Point { x: i as f32, y: 0.0 }); }
    let (xs,) = buf.columns((ps::x,));
    assert_eq!(xs, &[0.0f32, 1.0, 2.0, 3.0]);
}

// =============================================================================
// Push ordering
// =============================================================================

#[test]
fn push_order_preserved_in_columns() {
    let raw = vec![0u8; 5 * SingleSchema::STRIDE];
    let mut buf: ColumnarBuffer<SingleSchema, Vec<u8>> = ColumnarBuffer::new(raw);
    for v in [10u64, 20, 30, 40, 50] { buf.push(Single { value: v }); }
    let (vals,) = buf.columns((ss::value,));
    assert_eq!(vals, &[10u64, 20, 30, 40, 50]);
}

// =============================================================================
// High-arity column access (8 columns)
// =============================================================================

#[test]
fn eight_columns_access() {
    let raw = vec![0u8; WideSchema::STRIDE];
    let mut buf: ColumnarBuffer<WideSchema, Vec<u8>> = ColumnarBuffer::new(raw);
    buf.push(Wide { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8 });

    let (a, b, c, d, e, f, g, h) = buf.columns((
        ws::a, ws::b, ws::c, ws::d,
        ws::e, ws::f, ws::g, ws::h,
    ));
    assert_eq!(
        (a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0]),
        (1u8, 2, 3, 4, 5, 6, 7, 8)
    );
}

// =============================================================================
// Push → mutate → get consistency
// =============================================================================

#[test]
fn push_mutate_get_consistent() {
    let mut buf = record_buf(4);
    for i in 0..4u32 {
        buf.push(Record { id: i, score: i as f32 * 0.1, tag: [0; 4] });
    }

    buf.mutate((rs::score,), |(scores,)| {
        for s in scores.iter_mut() { *s *= 2.0; }
    });

    for i in 0..4u32 {
        let r: Record = buf.get(i as usize).unwrap();
        assert!((r.score - i as f32 * 0.2).abs() < 1e-6,
            "row {i}: expected {}, got {}", i as f32 * 0.2, r.score);
        assert_eq!(r.id, i);
    }
}

// =============================================================================
// Alignment-sorted layout verification
// =============================================================================

// Scrambled { a: u8 (align 1), b: u32 (align 4), c: u8 (align 1) }
// Expected BLOCK_ORDER   = [1, 0, 2]  (b first, then a, then c)
// Expected FIELD_TO_BLOCK = [1, 0, 2]  (field a→block 1, b→block 0, c→block 2)
// Expected BLOCK_OFFSETS  = [0, 4, 5]  (b at 0, a at 4*cap, c at 5*cap)

#[test]
fn scrambled_field_to_block() {
    assert_eq!(ScrambledSchema::BLOCK_ORDER,    [1, 0, 2]);
    assert_eq!(ScrambledSchema::FIELD_TO_BLOCK, [1, 0, 2]);
}

#[test]
fn scrambled_block_offsets() {
    // block 0 (b, u32, size 4): prefix[0] = 0
    // block 1 (a, u8,  size 1): prefix[1] = 0 + ELEM_SIZES[BLOCK_ORDER[0]] = 0 + 4 = 4
    // block 2 (c, u8,  size 1): prefix[2] = 4 + ELEM_SIZES[BLOCK_ORDER[1]] = 4 + 1 = 5
    assert_eq!(ScrambledSchema::BLOCK_OFFSETS, [0, 4, 5]);
    assert_eq!(ScrambledSchema::STRIDE, 6);
}

#[test]
fn scrambled_block_offsets_aligned_for_any_capacity() {
    // Block 0 (b, u32, align 4): offset = 0 * cap → always aligned to 4.
    // Block 1 (a, u8,  align 1): offset = 4 * cap → always aligned to 1.
    // Block 2 (c, u8,  align 1): offset = 5 * cap → always aligned to 1.
    for cap in [1usize, 2, 3, 4, 7, 8, 13, 16, 100] {
        let off_b = ScrambledSchema::offset(0, cap);
        let off_a = ScrambledSchema::offset(1, cap);
        let off_c = ScrambledSchema::offset(2, cap);
        assert_eq!(off_b % 4, 0, "b block misaligned at cap={cap}");
        assert_eq!(off_a % 1, 0, "a block misaligned at cap={cap}");
        assert_eq!(off_c % 1, 0, "c block misaligned at cap={cap}");
    }
}

#[test]
fn scrambled_raw_layout() {
    // 4-row Scrambled buffer (stride=6, total=24 bytes rounded to 24)
    // block 0 (b): bytes  0..16 = [b0, b1, b2, b3] (4 * 4)
    // block 1 (a): bytes 16..20 = [a0, a1, a2, a3] (4 * 1)
    // block 2 (c): bytes 20..24 = [c0, c1, c2, c3] (4 * 1)
    let rows = 4usize;
    let mut buf: ColumnarBuffer<ScrambledSchema, RingSlot> =
        RingSlot::new(rows * ScrambledSchema::STRIDE).columnar();
    buf.push(Scrambled { a: 10, b: 100, c: 200 });
    buf.push(Scrambled { a: 11, b: 101, c: 201 });
    buf.push(Scrambled { a: 12, b: 102, c: 202 });
    buf.push(Scrambled { a: 13, b: 103, c: 203 });

    let raw = buf.storage.as_bytes();
    let bs: &[u32] = bytemuck::cast_slice(&raw[0..16]);
    let a_block = &raw[16..20];
    let c_block = &raw[20..24];
    assert_eq!(bs,      &[100u32, 101, 102, 103]);
    assert_eq!(a_block, &[10u8,   11,  12,  13]);
    assert_eq!(c_block, &[200u8,  201, 202, 203]);
}

#[test]
fn scrambled_push_get_round_trip() {
    let mut buf: ColumnarBuffer<ScrambledSchema, RingSlot> =
        RingSlot::new(4 * ScrambledSchema::STRIDE).columnar();
    let rows = [
        Scrambled { a: 1, b: 1000, c: 100 },
        Scrambled { a: 2, b: 2000, c: 200 },
        Scrambled { a: 3, b: 3000, c: 255 },
    ];
    for r in &rows { buf.push(r.clone()); }
    for (i, expected) in rows.iter().enumerate() {
        assert_eq!(&buf.get::<Scrambled>(i).unwrap(), expected);
    }
}

#[test]
fn scrambled_columns_access() {
    let mut buf: ColumnarBuffer<ScrambledSchema, RingSlot> =
        RingSlot::new(3 * ScrambledSchema::STRIDE).columnar();
    buf.push(Scrambled { a: 7, b: 77, c: 17 });
    buf.push(Scrambled { a: 8, b: 88, c: 18 });
    buf.push(Scrambled { a: 9, b: 99, c: 19 });

    let (a_col, b_col, c_col) = buf.columns((scr::a, scr::b, scr::c));
    assert_eq!(a_col, &[7u8,   8,  9]);
    assert_eq!(b_col, &[77u32, 88, 99]);
    assert_eq!(c_col, &[17u8,  18, 19]);
}

// =============================================================================
// Group columns tests
// =============================================================================

mod grouped {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Columnar)]
    pub struct Grouped {
        pub id: u32,
        #[columnar(group)]
        pub elements: [u8; 4],
    }
}

use grouped::*;
use grouped::schema as gs;

#[test]
fn group_push_get_round_trip() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });
    buf.push(Grouped { id: 2, elements: [11, 21, 31, 41] });

    let row0: Grouped = buf.get(0).unwrap();
    assert_eq!(row0.id, 1);
    assert_eq!(row0.elements, [10, 20, 30, 40]);

    let row1: Grouped = buf.get(1).unwrap();
    assert_eq!(row1.id, 2);
    assert_eq!(row1.elements, [11, 21, 31, 41]);
}

#[test]
fn group_columns_access() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });
    buf.push(Grouped { id: 2, elements: [11, 21, 31, 41] });
    buf.push(Grouped { id: 3, elements: [12, 22, 32, 42] });

    let (ids,) = buf.columns((gs::id,));
    assert_eq!(ids, &[1u32, 2, 3]);

    let (elems,) = buf.columns((gs::elements,));
    // elems[element_index][row_index]
    assert_eq!(elems[0], &[10u8, 11, 12]); // all element[0] across rows
    assert_eq!(elems[1], &[20u8, 21, 22]); // all element[1] across rows
    assert_eq!(elems[2], &[30u8, 31, 32]); // all element[2] across rows
    assert_eq!(elems[3], &[40u8, 41, 42]); // all element[3] across rows
}

#[test]
fn group_mixed_columns_access() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });
    buf.push(Grouped { id: 2, elements: [11, 21, 31, 41] });

    let (ids, elems) = buf.columns((gs::id, gs::elements));
    assert_eq!(ids, &[1u32, 2]);
    assert_eq!(elems[0], &[10u8, 11]);
    assert_eq!(elems[1], &[20u8, 21]);
    assert_eq!(elems[2], &[30u8, 31]);
    assert_eq!(elems[3], &[40u8, 41]);
}

#[test]
fn group_mutate() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });
    buf.push(Grouped { id: 2, elements: [11, 21, 31, 41] });

    buf.mutate((gs::elements,), |(elems,)| {
        // Double all element[0] values
        for v in elems[0].iter_mut() { *v *= 2; }
    });

    let row0: Grouped = buf.get(0).unwrap();
    assert_eq!(row0.elements, [20, 20, 30, 40]);
    let row1: Grouped = buf.get(1).unwrap();
    assert_eq!(row1.elements, [22, 21, 31, 41]);
}

#[test]
fn group_mutate_mixed() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });

    buf.mutate((gs::id, gs::elements), |(ids, mut elems)| {
        ids[0] = 99;
        elems[2][0] = 77;
    });

    let row0: Grouped = buf.get(0).unwrap();
    assert_eq!(row0.id, 99);
    assert_eq!(row0.elements, [10, 20, 77, 40]);
}

#[test]
fn group_raw_layout() {
    // Verify that sub-columns are stored transposed in memory
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });
    buf.push(Grouped { id: 2, elements: [11, 21, 31, 41] });
    buf.push(Grouped { id: 3, elements: [12, 22, 32, 42] });
    buf.push(Grouped { id: 4, elements: [13, 23, 33, 43] });

    let (elems,) = buf.columns((gs::elements,));
    let ptr0 = elems[0].as_ptr();

    assert_eq!(elems[0], &[10, 11, 12, 13]);
    assert_eq!(elems[1], &[20, 21, 22, 23]);
    assert_eq!(elems[2], &[30, 31, 32, 33]);
    assert_eq!(elems[3], &[40, 41, 42, 43]);

    // Verify contiguity within each sub-column
    assert_eq!(unsafe { *ptr0.add(0) }, 10);
    assert_eq!(unsafe { *ptr0.add(1) }, 11);
    assert_eq!(unsafe { *ptr0.add(2) }, 12);
    assert_eq!(unsafe { *ptr0.add(3) }, 13);
}

#[test]
fn group_schema_stride() {
    // id: u32 (4 bytes) + elements: [u8; 4] (4 bytes) = 8 bytes stride
    assert_eq!(GroupedSchema::STRIDE, 8);
}

#[test]
fn group_schema_total_blocks() {
    // id: 1 block + elements: 4 blocks = 5 total
    assert_eq!(GroupedSchema::TOTAL_BLOCKS, 5);
}

#[test]
#[should_panic(expected = "duplicate columns")]
fn group_duplicate_panics() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push(Grouped { id: 1, elements: [10, 20, 30, 40] });
    buf.mutate((gs::elements, gs::elements), |_| {});
}

#[test]
fn group_push_with() {
    let mut buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    buf.push_with((gs::id, gs::elements), |row, (ids, mut elems)| {
        ids[row] = 42;
        elems[0][row] = 10;
        elems[1][row] = 20;
        elems[2][row] = 30;
        elems[3][row] = 40;
    });

    let row0: Grouped = buf.get(0).unwrap();
    assert_eq!(row0.id, 42);
    assert_eq!(row0.elements, [10, 20, 30, 40]);
}

#[test]
fn group_empty_columns() {
    let buf: ColumnarBuffer<GroupedSchema, RingSlot> =
        RingSlot::new(4 * GroupedSchema::STRIDE).columnar();
    let (elems,) = buf.columns((gs::elements,));
    assert_eq!(elems[0].len(), 0);
    assert_eq!(elems[1].len(), 0);
    assert_eq!(elems[2].len(), 0);
    assert_eq!(elems[3].len(), 0);
}