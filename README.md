# columnar

Zero-copy, cache-friendly Structure-of-Arrays (SoA) columnar buffers for Rust.

Data is stored as contiguous column blocks rather than interleaved rows, optimised for SIMD vectorisation, cache efficiency, and zero-copy interop with Arrow, Polars, CUDA, and Python/NumPy.

```text
AoS (typical Rust struct layout):
[ id | score | elem ][ id | score | elem ][ id | score | elem ] ...

SoA (this crate's layout):
[ id0 | id1 | id2 | ... ][ score0 | score1 | score2 | ... ][ elem0 | elem1 | ... ]
```

## Workspace crates

| Crate | Description |
|---|---|
| `columnar` | Core library: `ColumnarBuffer`, `Schema`, memory pooling (`Pool`/`Batch`), and multi-stage `Pipeline` |
| `columnar-derive` | `#[derive(Columnar)]` proc macro for automatic SoA schema generation |
| `columnar-python` | PyO3 cdylib exposing columnar buffers to Python via PEP 3118 buffer protocol |

## Quick start

```rust
use columnar::{macros::Columnar, buffer::*};

#[repr(C)]
#[derive(Columnar)]
pub struct Sequence {
    pub id:    u64,
    pub score: f32,
    #[columnar(group)]
    pub elements: [u8; 32],
}

// Allocate a buffer for 1024 rows
let mut buf: ColumnarBuffer<SequenceSchema, AlignedBox> =
    AlignedBox::new(1024 * SequenceSchema::LAYOUT.stride).columnar();

// Push a full row
buf.push(Sequence { id: 1, score: 0.95, elements: [0u8; 32] });

// Read a full row back
let seq: Sequence = buf.get(0).unwrap();

// Zero-copy column access
let (ids,): (&[u64],) = buf.columns((schema::id,));
```

## Derive macro

`#[derive(Columnar)]` on a `#[repr(C)]` struct generates:

- A schema type (`FooSchema`) implementing `columnar::buffer::Schema`
- A `mod schema` with `const` column accessors (`ColumnIdx` / `ColumnGroupIdx`)
- `SoAWrite` and `SoARead` impls for row-level push/get

### Field attributes

| Attribute | Effect |
|---|---|
| `#[columnar(group)]` | Expand a `[T; N]` field into N separate sub-columns |
| `#[columnar(skip_py)]` | Exclude this field from the generated Python batch wrapper |

### Struct attributes

| Attribute | Effect |
|---|---|
| `#[columnar(pyclass = "Name")]` | Generate a `#[pyclass]` batch wrapper with `__getitem__` access |

## Pipeline

The `ring` and `pipeline` modules provide a multi-stage processing framework:

```rust
use columnar::ring::{Pool, connector};
use columnar::pipeline::Pipeline;

let pool = Arc::new(Pool::<MySchema>::new(8, 1024));
let (tx, rx) = connector::<MySchema, ()>(4);

let mut pipeline = Pipeline::new();
pipeline.stage("process", rx, 4, move |batch, _ctx| {
    // process batch...
});
```

- **Pool** — fixed-size pre-allocated memory slots with automatic return-on-drop
- **Batch** — `Arc`-wrapped columnar buffer, cheaply clonable for fan-out
- **connector** — bounded channels for passing batches between stages
- **Pipeline** — spawns worker threads per stage, supports fan-in/fan-out topologies

## Building

```sh
cargo build
cargo test
```

### With Python bindings

```sh
cd columnar-python
maturin develop  # or: maturin build --release
```

## License

See repository for license details.
