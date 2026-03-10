//! Memory pooling and typed batch passing for multi-stage pipelines.
//!
//! This module provides three main abstractions:
//!
//! - [`Pool`] — a fixed-size pool of pre-allocated [`AlignedBox`]
//!   memory slots, preventing allocation churn in hot loops.
//! - [`BatchMut`] / [`BatchRef`] — mutable and shared columnar buffers
//!   with optional metadata. `BatchMut` owns the buffer exclusively;
//!   `BatchRef` wraps it in an `Arc` for cheap cloning and fan-out.
//! - [`connector`] — bounded channel pairs ([`ConnectorTx`] / [`ConnectorRx`])
//!   for passing batches between threads.

use std::{marker::PhantomData, ops::{Deref, DerefMut}, sync::Arc, time::Duration};

use crossbeam::channel::{Receiver, RecvError, RecvTimeoutError, SendError, Sender, bounded};

use crate::buffer::{AlignedBox, ByteBuffer, ColumnarBuffer, Schema};
use crate::pipeline::{Emit, EmitError, Recv};

/// Type alias for the raw memory slot managed by a [`Pool`].
pub type PoolSlot = AlignedBox;

/// A leased memory slot that automatically returns to the pool on drop.
///
/// Implements [`ByteBuffer`] so it can back a [`ColumnarBuffer`] directly.
/// When dropped, the underlying [`PoolSlot`] is sent back to the pool's
/// free-list via the stored channel sender.
pub struct PoolSlotLease {
    /// The leased memory; `None` only after `drop` has returned it.
    memory: Option<PoolSlot>,
    /// Channel sender back to the pool's free-list.
    ret: Sender<PoolSlot>,
}

impl Drop for PoolSlotLease {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            let _ = self.ret.send(memory);
        }
    }
}

/// NOTE: `Option<PoolSlot>` is empty only after drop.
impl ByteBuffer for PoolSlotLease {

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.memory.as_mut().unwrap().as_bytes_mut()
    }

    fn as_bytes(&self) -> &[u8] {
        self.memory.as_ref().unwrap().as_bytes()
    }
}

// =============================================================================
// BatchMut
// =============================================================================

/// A mutable batch with exclusive ownership of the underlying buffer.
///
/// Acquired from a [`Pool`]. Supports both read and write access to columns.
/// Call [`freeze`](BatchMut::freeze) to convert into a shared [`BatchRef`].
pub struct BatchMut<S: Schema, M> {
    buffer: ColumnarBuffer<S, PoolSlotLease>,
    metadata: M,
}

impl<S: Schema, M: std::fmt::Debug> std::fmt::Debug for BatchMut<S, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchMut")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl<S: Schema, M> Deref for BatchMut<S, M> {
    type Target = ColumnarBuffer<S, PoolSlotLease>;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<S: Schema, M> DerefMut for BatchMut<S, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl<S: Schema, M> BatchMut<S, M> {

    /// Get a reference to the metadata.
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Replace the metadata, keeping the same buffer.
    pub fn with_metadata<M2>(self, metadata: M2) -> BatchMut<S, M2> {
        BatchMut {
            buffer: self.buffer,
            metadata,
        }
    }

    /// Convert into a shared, read-only [`BatchRef`] by wrapping in an `Arc`.
    pub fn freeze(self) -> BatchRef<S, M> {
        BatchRef {
            buffer: Arc::new(self.buffer),
            metadata: self.metadata,
        }
    }
}

// =============================================================================
// BatchRef
// =============================================================================

/// A shared, read-only batch backed by an `Arc`.
///
/// Cheaply clonable for fan-out between pipeline stages.
/// Call [`try_into_mut`](BatchRef::try_into_mut) to attempt recovering
/// exclusive ownership as a [`BatchMut`].
pub struct BatchRef<S: Schema, M> {
    buffer: Arc<ColumnarBuffer<S, PoolSlotLease>>,
    metadata: M,
}

impl<S: Schema, M: std::fmt::Debug> std::fmt::Debug for BatchRef<S, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchRef")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl<S: Schema, M: Clone> Clone for BatchRef<S, M> {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            buffer: self.buffer.clone(),
        }
    }
}

impl<S: Schema, M> Deref for BatchRef<S, M> {
    type Target = ColumnarBuffer<S, PoolSlotLease>;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<S: Schema, M> BatchRef<S, M> {

    /// Get a reference to the metadata.
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Replace the metadata, keeping the same buffer.
    pub fn with_metadata<M2>(self, metadata: M2) -> BatchRef<S, M2> {
        BatchRef {
            buffer: self.buffer,
            metadata,
        }
    }

    /// Attempt to recover exclusive ownership. Returns `Err(self)` if
    /// other clones of this batch exist.
    pub fn try_into_mut(self) -> Result<BatchMut<S, M>, BatchRef<S, M>> {
        match Arc::try_unwrap(self.buffer) {
            Ok(buffer) => Ok(BatchMut {
                buffer,
                metadata: self.metadata,
            }),
            Err(buffer) => Err(BatchRef {
                buffer,
                metadata: self.metadata,
            }),
        }
    }
}

// =============================================================================
// Pool
// =============================================================================

/// A typed memory pool for a specific schema.
///
/// Each pool allocates fixed-size slots sized in **elements** (rows) rather
/// than raw bytes, using `S::stride()` to compute the per-slot byte size.
pub struct Pool<S: Schema> {
    free_rx: Receiver<PoolSlot>,
    free_tx: Sender<PoolSlot>,
    _marker: PhantomData<S>,
}

impl<S: Schema> Pool<S> {

    /// Allocate a new pool with `count` slots, each holding `elements` rows.
    pub fn new(count: usize, elements: usize) -> Self {
        let slot_bytes = S::stride() * elements;
        let (free_tx, free_rx) = bounded(count);
        for _ in 0..count {
            free_tx.send(PoolSlot::new(slot_bytes)).unwrap();
        }
        Pool { free_rx, free_tx, _marker: PhantomData }
    }

    /// Number of currently available (unleased) slots.
    pub fn get_available_slots(&self) -> usize {
        self.free_rx.len()
    }

    /// Acquire a mutable batch from the pool. Times out after 1 second to
    /// detect pool starvation.
    pub fn acquire(&self) -> Result<BatchMut<S, ()>, RecvTimeoutError> {
        let slot = self.free_rx.recv_timeout(Duration::from_secs(1))?;
        let lease = PoolSlotLease {
            memory: Some(slot),
            ret: self.free_tx.clone(),
        };
        let buffer = ColumnarBuffer::new_complete(lease);
        Ok(BatchMut {
            buffer,
            metadata: (),
        })
    }
}

// =============================================================================
// Connectors
// =============================================================================

/// Create a bounded connector channel pair.
pub fn connector<B>(cap: usize) -> (ConnectorTx<B>, ConnectorRx<B>) {
    let (tx, rx) = bounded(cap);
    (ConnectorTx(tx), ConnectorRx(rx))
}

/// Create a bounded connector for shared read-only batches.
pub fn connector_ref<S: Schema, M>(cap: usize) -> (ConnectorTx<BatchRef<S, M>>, ConnectorRx<BatchRef<S, M>>) {
    connector(cap)
}

/// Create a bounded connector for exclusively owned mutable batches.
pub fn connector_mut<S: Schema, M>(cap: usize) -> (ConnectorTx<BatchMut<S, M>>, ConnectorRx<BatchMut<S, M>>) {
    connector(cap)
}

/// Sending half of a connector. Clone to fan-in from multiple producers.
/// When all `ConnectorTx` clones are dropped, receivers get `RecvError`.
pub struct ConnectorTx<B>(Sender<B>);

impl<B> Clone for ConnectorTx<B> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<B> ConnectorTx<B> {
    /// Send a batch into the channel. Blocks if the channel is full.
    pub fn send(&self, batch: B) -> Result<(), SendError<B>> {
        self.0.send(batch)
    }
}

impl<B> Emit<B> for ConnectorTx<B> {
    fn emit(&mut self, item: B) -> Result<(), EmitError> {
        self.send(item).map_err(|_| EmitError)
    }
}

/// Receiving half of a connector. Clone to have multiple workers consume from the same channel.
/// Returns `RecvError` when all senders are dropped.
pub struct ConnectorRx<B>(Receiver<B>);

impl<B> Clone for ConnectorRx<B> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<B> ConnectorRx<B> {
    /// Block until a batch is available, or return `RecvError` if all senders
    /// have been dropped.
    pub fn recv(&self) -> Result<B, RecvError> {
        self.0.recv()
    }
}

impl<B> Recv<B> for ConnectorRx<B> {
    fn recv(&mut self) -> Option<B> {
        self.0.recv().ok()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod test {
    use columnar::buffer::Schema;
    use super::*;

    mod src {
        use columnar_derive::Columnar;
        use super::*;

        #[derive(Debug, Columnar)]
        pub struct Input {
            pub id: u32
        }
    }

    mod dst {
        use columnar_derive::Columnar;
        use super::*;

        #[derive(Debug, Columnar)]
        pub struct Output {
            pub pos: u32
        }
    }

    pub struct OutputMetadata {
        pub source: BatchRef<src::InputSchema, ()>
    }

    #[test]
    fn simple() {
        let pool = Pool::<src::InputSchema>::new(2, 128);
        let dst_pool = Pool::<dst::OutputSchema>::new(2, 128);

        let (src_tx, src_rx) = connector_mut::<src::InputSchema, ()>(2);
        let (dst_tx, dst_rx) = connector_ref::<dst::OutputSchema, OutputMetadata>(1);
        {
            let mut batch = pool.acquire().unwrap();
            batch.mutate((src::schema::id,), |(ids,)| {
                for id in ids {
                    *id = 9;
                }
            });

            src_tx.send(batch).unwrap();
        }
        {
            let input = src_rx.recv().unwrap();
            let mut output = dst_pool.acquire().unwrap();

            output.mutate((dst::schema::pos,), |(pos,): (&mut [u32],)| {
                let (ids,): (&[u32],) = input.columns((src::schema::id,));
                for (id, p) in ids.iter().zip(pos) {
                    *p = id * 2;
                }
            });

            let input = input.freeze();
            let result = output.with_metadata(OutputMetadata { source: input }).freeze();
            dst_tx.send(result).unwrap();
        }

        let result = dst_rx.recv().unwrap();

        let metadata = result.metadata();
        let (pos,) = result.columns((dst::schema::pos,));
        let (ids,) = metadata.source.columns((src::schema::id,));
        for (p, id) in pos.iter().zip(ids) {
            assert_eq!(*p, id * 2);
        }

        assert_eq!(pool.get_available_slots(), 1);
        assert_eq!(dst_pool.get_available_slots(), 1);
        drop(result);
        assert_eq!(pool.get_available_slots(), 2);
        assert_eq!(dst_pool.get_available_slots(), 2);
    }

    #[test]
    fn sole_owner_can_mutate() {
        let pool = Pool::<src::InputSchema>::new(1, 128);
        let mut batch = pool.acquire().unwrap();
        batch.mutate((src::schema::id,), |(ids,)| {
            ids[0] = 42;
        });

        let batch = batch.with_metadata(7u32);
        assert_eq!(*batch.metadata(), 7u32);

        // Data survived metadata change
        let (ids,): (&[u32],) = batch.columns((src::schema::id,));
        assert_eq!(ids[0], 42);

        // Can get back to mutable
        let mut batch = batch.with_metadata(());
        batch.mutate((src::schema::id,), |(ids,)| {
            ids[0] = 99;
        });
        let (ids,): (&[u32],) = batch.columns((src::schema::id,));
        assert_eq!(ids[0], 99);
    }

    #[test]
    fn freeze_and_try_into_mut() {
        let pool = Pool::<src::InputSchema>::new(1, 128);
        let mut batch = pool.acquire().unwrap();
        batch.mutate((src::schema::id,), |(ids,)| {
            ids[0] = 42;
        });

        // Freeze into shared ref
        let shared = batch.freeze();
        let (ids,): (&[u32],) = shared.columns((src::schema::id,));
        assert_eq!(ids[0], 42);

        // Sole owner can recover mutability
        let mut batch = shared.try_into_mut().unwrap();
        batch.mutate((src::schema::id,), |(ids,)| {
            ids[0] = 99;
        });
        let (ids,): (&[u32],) = batch.columns((src::schema::id,));
        assert_eq!(ids[0], 99);
    }

    #[test]
    fn cloned_ref_cannot_become_mut() {
        let pool = Pool::<src::InputSchema>::new(1, 128);
        let batch = pool.acquire().unwrap();
        let shared = batch.freeze();
        let _clone = shared.clone();

        // try_into_mut should fail because a clone exists
        assert!(shared.try_into_mut().is_err());
    }
}
