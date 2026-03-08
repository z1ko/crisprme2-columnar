use std::{fmt::{Debug, Display}, sync::Arc, thread::JoinHandle};

use crate::{
    buffer::Schema,
    ring::{Batch, ConnectorRx, Pool},
};

/// Shared resources available to stage handler functions.
pub struct StageContext {
}

/// Describes a stage
pub struct Stage {
    pub name: String,
    pub handles: Vec<JoinHandle<()>>
}

impl Debug for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// Orchestrates a multi-stage processing pipeline.
///
/// Each stage is a set of worker threads consuming from a `ConnectorRx`.
/// Output connectors are captured by the handler closure, giving full
/// flexibility for fan-out, fan-in, and arbitrary topologies.
#[derive(Debug)]
pub struct Pipeline {
    stages: Vec<Stage>,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        println!("shutting down pipeline");

        // Wait for all threads
        for mut stage in self.stages.drain(..) {
            for handle in stage.handles.drain(..) {
                if let Err(e) = handle.join() {
                println!("error dropping thread for stage {}: {:?}", 
                    stage.name, e);
                }
            }
        }

        println!("all threads stopped");
    }
}

impl Pipeline {

    pub fn new() -> Self {
        Self {
            stages: vec![],
        }
    }

    /// Spawn `workers` threads that consume batches from `input` and call `handler`.
    ///
    /// The handler closure should capture any output `ConnectorTx` and pools it needs.
    /// Workers exit when all corresponding `ConnectorTx` senders are dropped
    /// (i.e. `recv()` returns `Err`).
    ///
    /// **Note**: With multiple workers, batch processing order is not guaranteed.
    pub fn stage<S, M, F>(
        &mut self,
        name: &str,
        input: ConnectorRx<S, M>,
        workers: usize,
        handler: F,
    ) where
        S: Schema + Send + Sync + 'static,
        M: Send + Sync + 'static,
        F: Fn(Batch<S, M>, &StageContext) + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        let mut stage = Stage {
            name: name.into(),
            handles: vec![]
        };

        for i in 0..workers {
            let input = input.clone();
            let handler = handler.clone();
            let ctx = StageContext {};
            let thread_name = format!("{name}-{i}");
            let handle = std::thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    while let Ok(batch) = input.recv() {
                        handler(batch, &ctx);
                    }
                })
                .expect("failed to spawn stage thread");

            stage.handles.push(handle);
        }
        self.stages.push(stage);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::{connector, Pool};

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
        let pool = Arc::new(Pool::<data::RecordSchema>::new(4, 256));

        // source -> stage_a -> stage_b -> output
        let (src_tx, src_rx) = connector::<data::RecordSchema, ()>(2);
        let (mid_tx, mid_rx) = connector::<data::RecordSchema, ()>(2);
        let (out_tx, out_rx) = connector::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new();

        // Stage A: multiply values by 2
        let mid_tx_clone = mid_tx.clone();
        let stage_pool = pool.clone();
        pipeline.stage("multiply", src_rx, 1, move |batch, _ctx| {
            let mut writer = stage_pool.acquire().unwrap();
            writer.as_mut().mutate(
                (data::schema::value,),
                |(out_vals,): (&mut [u32],)| {
                    let (in_vals,): (&[u32],) = batch.columns((data::schema::value,));
                    for (o, i) in out_vals.iter_mut().zip(in_vals) {
                        *o = i * 2;
                    }
                },
            );
            mid_tx_clone.send(writer).unwrap();
        });

        // Stage B: add 1
        let out_tx_clone = out_tx.clone();
        let stage_pool = pool.clone();
        pipeline.stage("add-one", mid_rx, 1, move |batch, _ctx| {
            let mut writer = stage_pool.acquire().unwrap();
            writer.as_mut().mutate(
                (data::schema::value,),
                |(out_vals,): (&mut [u32],)| {
                    let (in_vals,): (&[u32],) = batch.columns((data::schema::value,));
                    for (o, i) in out_vals.iter_mut().zip(in_vals) {
                        *o = i + 1;
                    }
                },
            );
            out_tx_clone.send(writer).unwrap();
        });

        // Feed input
        let mut batch = pool.acquire().unwrap();
        batch.as_mut().mutate(
            (data::schema::value,),
            |(vals,): (&mut [u32],)| {
                for (i, v) in vals.iter_mut().enumerate() {
                    *v = i as u32 + 1; // 1, 2, 3, ...
                }
            },
        );
        src_tx.send(batch).unwrap();

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

        //pipeline.join();
    }

    /// Fan-out: one stage sends to two downstream consumers
    #[test]
    fn fan_out() {
        let pool = Arc::new(Pool::<data::RecordSchema>::new(6, 256));

        let (src_tx, src_rx) = connector::<data::RecordSchema, ()>(2);
        let (a_tx, a_rx) = connector::<data::RecordSchema, ()>(2);
        let (b_tx, b_rx) = connector::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new();

        let a = a_tx.clone();
        let b = b_tx.clone();
        pipeline.stage("fanout", src_rx, 1, move |batch, _ctx| {
            a.send(batch.clone()).unwrap();
            b.send(batch).unwrap();
        });

        let mut batch = pool.acquire().unwrap();
        batch.as_mut().mutate(
            (data::schema::value,),
            |(vals,): (&mut [u32],)| {
                for (i, v) in vals.iter_mut().enumerate() {
                    *v = i as u32 + 10;
                }
            },
        );
        src_tx.send(batch).unwrap();
        drop(src_tx);
        drop(a_tx);
        drop(b_tx);

        let ra = a_rx.recv().unwrap();
        let rb = b_rx.recv().unwrap();
        let (va,): (&[u32],) = ra.columns((data::schema::value,));
        let (vb,): (&[u32],) = rb.columns((data::schema::value,));
        assert_eq!(va, vb);
        for (i, v) in va.iter().enumerate() {
            assert_eq!(*v, i as u32 + 10);
        }

        //pipeline.join();
    }

    /// Fan-in: two producers feed into one consumer stage
    #[test]
    fn fan_in() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let pool = Arc::new(Pool::<data::RecordSchema>::new(6, 256));

        let (merge_tx, merge_rx) = connector::<data::RecordSchema, ()>(4);
        let (out_tx, out_rx) = connector::<data::RecordSchema, ()>(4);

        let mut pipeline = Pipeline::new();

        let counter = Arc::new(AtomicU32::new(0));
        let cnt = counter.clone();
        let out = out_tx.clone();
        pipeline.stage("sink", merge_rx, 1, move |batch, _ctx| {
            cnt.fetch_add(1, Ordering::Relaxed);
            out.send(batch).unwrap();
        });

        let tx1 = merge_tx.clone();
        let tx2 = merge_tx.clone();
        drop(merge_tx);

        for tx in [tx1, tx2] {
            let mut batch = pool.acquire().unwrap();
            batch.as_mut().mutate(
                (data::schema::value,),
                |(vals,): (&mut [u32],)| { vals[0] = 42; },
            );
            tx.send(batch).unwrap();
        }
        drop(out_tx);

        let _r1 = out_rx.recv().unwrap();
        let _r2 = out_rx.recv().unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);

        //pipeline.join();
    }

    /// Multiple chunks through a multi-worker stage
    #[test]
    fn multi_worker() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let pool = Arc::new(Pool::<data::RecordSchema>::new(16, 256));

        let (src_tx, src_rx) = connector::<data::RecordSchema, ()>(8);
        let (out_tx, out_rx) = connector::<data::RecordSchema, ()>(8);

        let mut pipeline = Pipeline::new();

        let processed = Arc::new(AtomicU32::new(0));
        let cnt = processed.clone();
        let out = out_tx.clone();
        pipeline.stage("workers", src_rx, 4, move |batch, _ctx| {
            cnt.fetch_add(1, Ordering::Relaxed);
            out.send(batch).unwrap();
        });

        for i in 0..8u32 {
            let mut batch = pool.acquire().unwrap();
            batch.as_mut().mutate(
                (data::schema::value,),
                |(vals,): (&mut [u32],)| { vals[0] = i; },
            );
            src_tx.send(batch).unwrap();
        }
        drop(src_tx);
        drop(out_tx);

        let mut received = Vec::new();
        while let Ok(batch) = out_rx.recv() {
            let (vals,): (&[u32],) = batch.columns((data::schema::value,));
            received.push(vals[0]);
        }
        received.sort();
        assert_eq!(received, (0..8).collect::<Vec<u32>>());
        assert_eq!(processed.load(Ordering::Relaxed), 8);

        //pipeline.join();
    }

    /// Slots return to the pool after pipeline drains
    #[test]
    fn pool_slots_reclaimed() {
        let pool = Arc::new(Pool::<data::RecordSchema>::new(4, 256));

        let (src_tx, src_rx) = connector::<data::RecordSchema, ()>(2);
        let (out_tx, out_rx) = connector::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new();

        let out = out_tx.clone();
        pipeline.stage("passthrough", src_rx, 1, move |batch, _ctx| {
            out.send(batch).unwrap();
        });

        let mut batch = pool.acquire().unwrap();
        batch.as_mut().mutate(
            (data::schema::value,),
            |(vals,): (&mut [u32],)| { vals[0] = 99; },
        );
        src_tx.send(batch).unwrap();
        drop(src_tx);
        drop(out_tx);

        let result = out_rx.recv().unwrap();
        assert_eq!(pool.get_available_slots(), 3);
        drop(result);
        assert_eq!(pool.get_available_slots(), 4);

        //pipeline.join();
    }
}
