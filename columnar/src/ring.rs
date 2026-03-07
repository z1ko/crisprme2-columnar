use std::{ops::Deref, sync::Arc, time::{Duration, Instant}};

use crossbeam::channel::{Receiver, RecvError, RecvTimeoutError, SendError, Sender, bounded};

use crate::buffer::{AlignedBox, ColumnarBuffer, Schema};

pub type PoolSlot = AlignedBox;

/// Container for memory chunks
pub struct Pool {
    free_rx: Receiver<PoolSlot>,
    free_tx: Sender<PoolSlot>,
}

impl Pool {

    /// Allocate a new pool with fixed amount of slots
    pub fn new(count: usize, slot_bytes: usize) -> Self {
        let (free_tx, free_rx) = bounded(count);
        for _ in 0..count {
            free_tx.send(PoolSlot::new(slot_bytes))
                .unwrap();
        }
        Pool { free_rx, free_tx }
    }

    pub fn get_available_slots(&self) -> usize {
        self.free_rx.len()
    }

    pub fn acquire<S: Schema>(&self) -> Result<WriteChunk<S>, RecvTimeoutError> {
        // Timeout used to detect pool starvation
        let slot = self.free_rx.recv_timeout(Duration::from_secs(1))?;
        let buffer: ColumnarBuffer<S, PoolSlot> = ColumnarBuffer::new_complete(slot);
        Ok(WriteChunk {
            lease: PoolSlotLease { 
                buffer: Some(buffer), 
                ret: self.free_tx.clone(),
                received_at: Instant::now()
            }
        })
    }
}

pub struct PoolSlotLease<S: Schema> {
    buffer: Option<ColumnarBuffer<S, PoolSlot>>,
    ret: Sender<PoolSlot>,
    received_at: Instant,
}

impl<S: Schema> Drop for PoolSlotLease<S> {
    fn drop(&mut self) {
        println!("Lease released after {} ms", self.received_at.elapsed().as_millis());
        if let Some(buffer) = self.buffer.take() {
            let slot = buffer.detach();
            let _ = self.ret.send(slot);
        }
    }
}

pub struct WriteChunk<S: Schema> {
    lease: PoolSlotLease<S>,
}

impl<S: Schema> WriteChunk<S> {
    
    pub fn buffer_mut(&mut self) -> &mut ColumnarBuffer<S, PoolSlot> { 
        // NOTE: Optional is used only for the custom drop
        self.lease.buffer.as_mut().unwrap()
    }

    pub fn buffer(&self) -> &ColumnarBuffer<S, PoolSlot> { 
        // NOTE: Optional is used only for the custom drop
        self.lease.buffer.as_ref().unwrap() 
    }

    /// Make this read only
    pub fn publish_with_metadata<M>(self, metadata: M) -> ReadChunk<S, M> {
        let lease = self.lease;
        ReadChunk {
            inner: Arc::new(ReadChunkInner {
                metadata,
                lease
            })
        }
    }

    /// Make this read only, with no metadata
    pub fn publish(self) -> ReadChunk<S, ()> {
        self.publish_with_metadata(())
    }
}

pub struct ReadChunkInner<S: Schema, M> {
    lease: PoolSlotLease<S>,
    metadata: M,
}

#[derive(Clone)]
pub struct ReadChunk<S: Schema, M> {
    inner: Arc<ReadChunkInner<S, M>>
}

/// The main point of a ReadChunk is to access the inner content
impl<S: Schema, M> Deref for ReadChunk<S, M> {
    type Target = ColumnarBuffer<S, PoolSlot>;
    fn deref(&self) -> &Self::Target {
        // SAFETY: Optional is used only for the custom drop
        self.inner.lease.buffer.as_ref().unwrap()
    }
}

impl<S: Schema, M> ReadChunk<S, M> {
    pub fn metadata(&self) -> &M {
        &self.inner.metadata
    }
}

/// Create a bounded connector channel pair for passing chunks between pipeline stages.
pub fn connector<S: Schema, M>(cap: usize) -> (ConnectorTx<S, M>, ConnectorRx<S, M>) {
    let (tx, rx) = bounded(cap);
    (ConnectorTx(tx), ConnectorRx(rx))
}

/// Sending half of a connector. Clone to fan-in from multiple producers.
/// When all `ConnectorTx` clones are dropped, receivers get `RecvError`.
pub struct ConnectorTx<S: Schema, M>(Sender<ReadChunk<S, M>>);

impl<S: Schema, M> Clone for ConnectorTx<S, M> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<S: Schema, M> ConnectorTx<S, M> {
    pub fn send(&self, chunk: ReadChunk<S, M>) -> Result<(), SendError<ReadChunk<S, M>>> {
        self.0.send(chunk)
    }
}

/// Receiving half of a connector. Clone to have multiple workers consume from the same channel.
/// Returns `RecvError` when all senders are dropped.
pub struct ConnectorRx<S: Schema, M>(Receiver<ReadChunk<S, M>>);

impl<S: Schema, M> Clone for ConnectorRx<S, M> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<S: Schema, M> ConnectorRx<S, M> {
    pub fn recv(&self) -> Result<ReadChunk<S, M>, RecvError> {
        self.0.recv()
    }
}

#[cfg(test)]
mod test {
    use columnar::buffer::Schema;
    use super::*;

    pub struct OutputMetadata {
        pub source: ReadChunk<src::InputSchema, ()>
    }

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

    #[test]
    fn simple() {

        let pool = Pool::new(2, 1024);

        let (src_tx, src_rx) = connector::<src::InputSchema, ()>(2);
        let (dst_tx, dst_rx) = connector::<dst::OutputSchema, OutputMetadata>(1);
        {
            let mut writer = pool.acquire::<src::InputSchema>().unwrap();
            writer.buffer_mut().mutate((src::schema::id,), |(ids,)| {
                for id in ids {
                    *id = 9;
                }
            });

            let reader = writer.publish();
            src_tx.send(reader).unwrap();
        }
        {
            let input = src_rx.recv().unwrap();
            let mut output = pool.acquire::<dst::OutputSchema>().unwrap();

            output.buffer_mut().mutate((dst::schema::pos,), |(pos,) : (&mut [u32],)| {
                let (ids,): (&[u32],) = input.columns((src::schema::id,));
                for (id, p) in ids.iter().zip(pos) {
                    *p = id * 2;
                }
            });

            let result = output.publish_with_metadata(OutputMetadata { source: input });
            dst_tx.send(result).unwrap();
        }

        let result = dst_rx.recv().unwrap();

        let metadata = result.metadata();
        let (pos,) = result.columns((dst::schema::pos,));
        let (ids,) = metadata.source.columns((src::schema::id,));
        for (p, id) in pos.iter().zip(ids) {
            assert_eq!(*p, id * 2);
        }

        assert_eq!(pool.get_available_slots(), 0);
        drop(result);
        assert_eq!(pool.get_available_slots(), 2);

    }

}