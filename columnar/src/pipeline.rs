use std::{sync::Arc, thread::JoinHandle};

use crate::{
    buffer::Schema,
    ring::{ConnectorRx, Pool, ReadChunk},
};

/// Shared resources available to stage handler functions.
pub struct StageContext {
    pool: Arc<Pool>,
}

impl StageContext {
    pub fn pool(&self) -> &Pool {
        &self.pool
    }
}

/// Orchestrates a multi-stage processing pipeline.
///
/// Each stage is a set of worker threads consuming from a `ConnectorRx`.
/// Output connectors are captured by the handler closure, giving full
/// flexibility for fan-out, fan-in, and arbitrary topologies.
pub struct Pipeline {
    handles: Vec<JoinHandle<()>>,
    pool: Arc<Pool>,
}

impl Pipeline {

    pub fn new(pool_slots: usize, slot_bytes: usize) -> Self {
        Self::new_with_pool(
            Arc::new(
                Pool::new(pool_slots, slot_bytes)))
    }

    pub fn new_with_pool(pool: Arc<Pool>) -> Self {
        Self {
            handles: vec![],
            pool,
        }
    }

    /// Spawn `workers` threads that consume chunks from `input` and call `handler`.
    ///
    /// The handler closure should capture any output `ConnectorTx` it needs.
    /// Workers exit when all corresponding `ConnectorTx` senders are dropped
    /// (i.e. `recv()` returns `Err`).
    ///
    /// **Note**: With multiple workers, chunk processing order is not guaranteed.
    pub fn stage<S, M, F>(
        &mut self,
        name: &str,
        input: ConnectorRx<S, M>,
        workers: usize,
        handler: F,
    ) where
        S: Schema + Send + Sync + 'static,
        M: Send + Sync + 'static,
        F: Fn(ReadChunk<S, M>, &StageContext) + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        for i in 0..workers {
            let input = input.clone();
            let handler = handler.clone();
            let ctx = StageContext {
                pool: self.pool.clone(),
            };
            let thread_name = format!("{name}-{i}");
            let handle = std::thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    while let Ok(chunk) = input.recv() {
                        handler(chunk, &ctx);
                    }
                })
                .expect("failed to spawn stage thread");
            self.handles.push(handle);
        }
    }

    /// Wait for all stage worker threads to finish.
    pub fn join(self) {
        for handle in self.handles {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::connector;

    mod data {
        use columnar_derive::Columnar;
        use crate::buffer::Schema;

        #[derive(Debug, Columnar)]
        pub struct Record {
            pub value: u32,
        }
    }

    #[test]
    fn two_stage_pipeline() {
        let pool = Arc::new(Pool::new(4, 1024));

        // source -> stage_a -> stage_b -> output
        let (src_tx, src_rx) = connector::<data::RecordSchema, ()>(2);
        let (mid_tx, mid_rx) = connector::<data::RecordSchema, ()>(2);
        let (out_tx, out_rx) = connector::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new_with_pool(pool.clone());

        // Stage A: multiply values by 2
        let mid_tx_clone = mid_tx.clone();
        pipeline.stage("multiply", src_rx, 1, move |chunk, ctx| {
            let mut writer = ctx.pool().acquire::<data::RecordSchema>().unwrap();
            writer.buffer_mut().mutate(
                (data::schema::value,),
                |(out_vals,): (&mut [u32],)| {
                    let (in_vals,): (&[u32],) = chunk.columns((data::schema::value,));
                    for (o, i) in out_vals.iter_mut().zip(in_vals) {
                        *o = i * 2;
                    }
                },
            );
            mid_tx_clone.send(writer.publish()).unwrap();
        });

        // Stage B: add 1
        let out_tx_clone = out_tx.clone();
        pipeline.stage("add-one", mid_rx, 2, move |chunk, ctx| {
            let mut writer = ctx.pool().acquire::<data::RecordSchema>().unwrap();
            writer.buffer_mut().mutate(
                (data::schema::value,),
                |(out_vals,): (&mut [u32],)| {
                    let (in_vals,): (&[u32],) = chunk.columns((data::schema::value,));
                    for (o, i) in out_vals.iter_mut().zip(in_vals) {
                        *o = i + 1;
                    }
                },
            );
            out_tx_clone.send(writer.publish()).unwrap();
        });

        // Feed input
        let mut writer = pool.acquire::<data::RecordSchema>().unwrap();
        writer.buffer_mut().mutate(
            (data::schema::value,),
            |(vals,): (&mut [u32],)| {
                for (i, v) in vals.iter_mut().enumerate() {
                    *v = i as u32 + 1; // 1, 2, 3, ...
                }
            },
        );
        src_tx.send(writer.publish()).unwrap();

        // Signal shutdown: drop all senders so workers exit after processing
        drop(src_tx);
        drop(mid_tx);
        drop(out_tx);

        // Collect output
        let result = out_rx.recv().unwrap();
        let (vals,): (&[u32],) = result.columns((data::schema::value,));
        for (i, v) in vals.iter().enumerate() {
            // (i+1) * 2 + 1
            assert_eq!(*v, (i as u32 + 1) * 2 + 1);
        }

        pipeline.join();
    }
}
