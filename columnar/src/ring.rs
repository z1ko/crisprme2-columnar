use std::{marker::PhantomData, ops::Deref, sync::Arc, time::Duration};

use crossbeam::channel::{Receiver, RecvError, RecvTimeoutError, SendError, Sender, bounded};

use crate::buffer::{AlignedBox, ByteBuffer, ColumnarBuffer, Schema};

pub type PoolSlot = AlignedBox;

/// A leased memory slot that automatically returns to the pool on drop.
pub struct PoolSlotLease {
    memory: Option<PoolSlot>,
    ret: Sender<PoolSlot>,
}

impl Drop for PoolSlotLease {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            let _ = self.ret.send(memory);
        }
    }
}

/// NOTE: Option<PoolSlot> is empty only after drop
impl ByteBuffer for PoolSlotLease {

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.memory.as_mut().unwrap().as_bytes_mut()
    }

    fn as_bytes(&self) -> &[u8] {
        self.memory.as_ref().unwrap().as_bytes()
    }
}

// =============================================================================
// Batch
// =============================================================================

/// A unified batch type wrapping a columnar buffer with optional metadata.
///
/// Batches are `Arc`-wrapped so they can be cheaply cloned for fan-out.
/// Mutable access requires sole ownership (`Arc::get_mut`), which panics
/// if clones exist — indicating a pipeline misconfiguration.
pub struct Batch<S: Schema, M> {
    buffer: Arc<ColumnarBuffer<S, PoolSlotLease>>,
    metadata: M,
}

impl<S: Schema, M: std::fmt::Debug> std::fmt::Debug for Batch<S, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl<S: Schema, M: Clone> Clone for Batch<S, M> {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            buffer: self.buffer.clone(),
        }
    }
}

impl<S: Schema, M> Deref for Batch<S, M> {
    type Target = ColumnarBuffer<S, PoolSlotLease>;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<S: Schema, M> Batch<S, M> {

    /// Get mutable access to the underlying buffer.
    ///
    /// # Panics
    ///
    /// Panics if other clones of this batch exist, indicating a pipeline
    /// misconfiguration (multiple mutable references to the same buffer).
    pub fn as_mut(&mut self) -> &mut ColumnarBuffer<S, PoolSlotLease> {
        Arc::get_mut(&mut self.buffer)
            .expect("pipeline error: multiple owners of mutable batch")
    }

    /// Get shared access to the underlying buffer.
    pub fn as_ref(&self) -> &ColumnarBuffer<S, PoolSlotLease> {
        &self.buffer
    }

    /// Get a reference to the metadata.
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Replace the metadata, keeping the same buffer.
    pub fn with_metadata<M2>(self, metadata: M2) -> Batch<S, M2> {
        Batch {
            buffer: self.buffer,
            metadata,
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

    /// Acquire a batch from the pool. Times out after 1 second to detect
    /// pool starvation.
    pub fn acquire(&self) -> Result<Batch<S, ()>, RecvTimeoutError> {
        let slot = self.free_rx.recv_timeout(Duration::from_secs(1))?;
        let lease = PoolSlotLease {
            memory: Some(slot),
            ret: self.free_tx.clone(),
        };
        let buffer = ColumnarBuffer::new_complete(lease);
        Ok(Batch {
            buffer: Arc::new(buffer),
            metadata: (),
        })
    }
}

// =============================================================================
// Connectors
// =============================================================================

/// Create a bounded connector channel pair for passing batches between pipeline stages.
pub fn connector<S: Schema, M>(cap: usize) -> (ConnectorTx<S, M>, ConnectorRx<S, M>) {
    let (tx, rx) = bounded(cap);
    (ConnectorTx(tx), ConnectorRx(rx))
}

/// Sending half of a connector. Clone to fan-in from multiple producers.
/// When all `ConnectorTx` clones are dropped, receivers get `RecvError`.
pub struct ConnectorTx<S: Schema, M>(Sender<Batch<S, M>>);

impl<S: Schema, M> Clone for ConnectorTx<S, M> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<S: Schema, M> ConnectorTx<S, M> {
    pub fn send(&self, batch: Batch<S, M>) -> Result<(), SendError<Batch<S, M>>> {
        self.0.send(batch)
    }
}

/// Receiving half of a connector. Clone to have multiple workers consume from the same channel.
/// Returns `RecvError` when all senders are dropped.
pub struct ConnectorRx<S: Schema, M>(Receiver<Batch<S, M>>);

impl<S: Schema, M> Clone for ConnectorRx<S, M> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<S: Schema, M> ConnectorRx<S, M> {
    pub fn recv(&self) -> Result<Batch<S, M>, RecvError> {
        self.0.recv()
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
        pub source: Batch<src::InputSchema, ()>
    }

    #[test]
    fn simple() {
        let pool = Pool::<src::InputSchema>::new(2, 128);
        let dst_pool = Pool::<dst::OutputSchema>::new(2, 128);

        let (src_tx, src_rx) = connector::<src::InputSchema, ()>(2);
        let (dst_tx, dst_rx) = connector::<dst::OutputSchema, OutputMetadata>(1);
        {
            let mut batch = pool.acquire().unwrap();
            batch.as_mut().mutate((src::schema::id,), |(ids,)| {
                for id in ids {
                    *id = 9;
                }
            });

            src_tx.send(batch).unwrap();
        }
        {
            let input = src_rx.recv().unwrap();
            let mut output = dst_pool.acquire().unwrap();

            output.as_mut().mutate((dst::schema::pos,), |(pos,): (&mut [u32],)| {
                let (ids,): (&[u32],) = input.columns((src::schema::id,));
                for (id, p) in ids.iter().zip(pos) {
                    *p = id * 2;
                }
            });

            let result = output.with_metadata(OutputMetadata { source: input });
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
        batch.as_mut().mutate((src::schema::id,), |(ids,)| {
            ids[0] = 42;
        });

        let batch = batch.with_metadata(7u32);
        assert_eq!(*batch.metadata(), 7u32);

        // Data survived metadata change
        let (ids,): (&[u32],) = batch.columns((src::schema::id,));
        assert_eq!(ids[0], 42);

        // Can get back to mutable
        let mut batch = batch.with_metadata(());
        batch.as_mut().mutate((src::schema::id,), |(ids,)| {
            ids[0] = 99;
        });
        let (ids,): (&[u32],) = batch.columns((src::schema::id,));
        assert_eq!(ids[0], 99);
    }

    #[test]
    #[should_panic(expected = "pipeline error")]
    fn cloned_batch_panics_on_mut() {
        let pool = Pool::<src::InputSchema>::new(1, 128);
        let mut batch = pool.acquire().unwrap();
        let _clone = batch.clone();
        batch.as_mut(); // should panic
    }
}
