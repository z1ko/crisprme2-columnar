//! Multi-stage processing pipeline with typed stages and shared context.
//!
//! A [`Pipeline`] manages a set of named stages, each backed by one or more
//! worker threads. Stages communicate via [`Emit`] / [`Recv`] abstractions
//! (implemented by [`ConnectorTx`](crate::ring::ConnectorTx) and
//! [`ConnectorRx`](crate::ring::ConnectorRx)).

use std::{sync::Arc, thread::JoinHandle};

// =============================================================================
// Emit / Recv traits
// =============================================================================

/// Emission error: the downstream channel is closed.
#[derive(Debug)]
pub struct EmitError;

/// Object that emits T
pub trait Emit<T> {
    fn emit(&mut self, item: T) -> Result<(), EmitError>;
}

/// How to compose two emitters
impl<A, B, E1, E2> Emit<(A, B)> for (E1, E2)
where
    E1: Emit<A>,
    E2: Emit<B>,
{
    fn emit(&mut self, (a, b): (A, B)) -> Result<(), EmitError> {
        self.0.emit(a)?;
        self.1.emit(b)?;
        Ok(())
    }
}

/// Object that receives T
pub trait Recv<T> {
    fn recv(&mut self) -> Option<T>;
}

/// How to compose two receivers
impl<A, B, E1, E2> Recv<(A, B)> for (E1, E2)
where
    E1: Recv<A>,
    E2: Recv<B>,
{
    fn recv(&mut self) -> Option<(A, B)> {
        let a = self.0.recv()?;
        let b = self.1.recv()?;
        Some((a, b))
    }
}

// =============================================================================
// Stage trait
// =============================================================================

/// A typed processing stage in a pipeline.
pub trait Stage {

    type Input;
    type Output;

    /// Process a single input element, emitting results downstream.
    fn process<E>(&mut self, input: Self::Input, emitter: &mut E) -> Result<(), EmitError>
    where
        E: Emit<Self::Output>;
}

// =============================================================================
// Pipeline
// =============================================================================

struct StageDescriptor {
    handles: Vec<JoinHandle<()>>,
    name: String,
}

/// A multi-stage processing pipeline with shared context.
///
/// The context `C` is wrapped in an `Arc` and passed to each stage's
/// factory function, allowing workers to access shared state like
/// metrics, loggers, or configuration.
pub struct Pipeline<C> {
    ctx: Arc<C>,
    stages: Vec<StageDescriptor>,
}

impl<C> std::fmt::Debug for Pipeline<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("stages", &self.stages.len())
            .finish()
    }
}

impl<C: Send + Sync + 'static> Pipeline<C> {

    /// Create a new pipeline with shared context.
    pub fn new(ctx: C) -> Self {
        Self {
            ctx: Arc::new(ctx),
            stages: vec![],
        }
    }

    /// Create a new named stage using a [`Stage`] trait impl.
    ///
    /// The factory function receives an `Arc<C>` clone so each worker
    /// can embed shared context into its stage instance.
    pub fn stage<S, E, R, F>(&mut self, name: &str, workers: usize, recv: R, emit: E, stage: F)
    where
        S: Stage           + Send + 'static,
        E: Emit<S::Output> + Send + Clone + 'static,
        R: Recv<S::Input>  + Send + Clone + 'static,
        F: Fn(Arc<C>) -> S + Send + Sync  + 'static,
    {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {

            let mut worker_emit = emit.clone();
            let mut worker_recv = recv.clone();

            let mut worker_stage = stage(self.ctx.clone());
            handles.push(std::thread::spawn(move || {
                while let Some(input) = worker_recv.recv() {
                    if worker_stage.process(input, &mut worker_emit).is_err() {
                        break;
                    }
                }
            }));
        }

        self.stages.push(
            StageDescriptor {
                name: name.to_owned(),
                handles
            }
        );
    }

    /// Convenience: create a stage from a closure instead of a [`Stage`] impl.
    ///
    /// The closure receives each input item and a mutable reference to the
    /// emitter. Useful for simple stages that don't need per-worker state.
    pub fn stage_fn<I, O, E, R, F>(&mut self, name: &str, workers: usize, recv: R, emit: E, handler: F)
    where
        F: Fn(I, &mut E) -> Result<(), EmitError> + Send + Sync + 'static,
        E: Emit<O> + Send + Clone + 'static,
        R: Recv<I> + Send + Clone + 'static,
        I: Send + 'static,
        O: 'static,
    {
        let handler = Arc::new(handler);
        let mut handles = Vec::with_capacity(workers);

        for _ in 0..workers {

            let mut worker_emit = emit.clone();
            let mut worker_recv = recv.clone();
            
            let handler = handler.clone();
            handles.push(std::thread::spawn(move || {
                while let Some(input) = worker_recv.recv() {
                    if handler(input, &mut worker_emit).is_err() {
                        break;
                    }
                }
            }));
        }

        self.stages.push(
            StageDescriptor {
                name: name.to_owned(),
                handles
            }
        );
    }

    /// Wait for all stages to exit.
    pub fn wait(&mut self) {
        for mut stage in self.stages.drain(..) {
            for h in stage.handles.drain(..) {
                let _ = h.join();
            }
        }
    }
}

impl<C> Drop for Pipeline<C> {
    fn drop(&mut self) {
        if !self.stages.is_empty() {
            eprintln!("pipeline dropped without calling wait");
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::{connector_mut, connector_ref, Pool};

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

        // source -> multiply -> add_one -> output
        let (src_tx, src_rx) = connector_mut::<data::RecordSchema, ()>(2);
        let (mid_tx, mid_rx) = connector_mut::<data::RecordSchema, ()>(2);
        let (out_tx, out_rx) = connector_mut::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new(());

        let stage_pool = pool.clone();
        pipeline.stage_fn("multiply", 1, src_rx, mid_tx.clone(), move |batch, emit| {
            let mut writer = stage_pool.acquire().unwrap();
            writer.mutate(
                (data::schema::value,),
                |(out_vals,): (&mut [u32],)| {
                    let (in_vals,): (&[u32],) = batch.columns((data::schema::value,));
                    for (o, i) in out_vals.iter_mut().zip(in_vals) {
                        *o = i * 2;
                    }
                },
            );
            emit.emit(writer)
        });

        let stage_pool = pool.clone();
        pipeline.stage_fn("add-one", 1, mid_rx, out_tx.clone(), move |batch, emit| {
            let mut writer = stage_pool.acquire().unwrap();
            writer.mutate(
                (data::schema::value,),
                |(out_vals,): (&mut [u32],)| {
                    let (in_vals,): (&[u32],) = batch.columns((data::schema::value,));
                    for (o, i) in out_vals.iter_mut().zip(in_vals) {
                        *o = i + 1;
                    }
                },
            );
            emit.emit(writer)
        });

        // Feed input
        let mut batch = pool.acquire().unwrap();
        batch.mutate(
            (data::schema::value,),
            |(vals,): (&mut [u32],)| {
                for (i, v) in vals.iter_mut().enumerate() {
                    *v = i as u32 + 1;
                }
            },
        );
        src_tx.send(batch).unwrap();

        drop(src_tx);
        drop(mid_tx);
        drop(out_tx);

        let result = out_rx.recv().unwrap();
        let (vals,): (&[u32],) = result.columns((data::schema::value,));
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(*v, (i as u32 + 1) * 2 + 1);
        }
    }

    /// Fan-out: one stage sends to two downstream consumers
    #[test]
    fn fan_out() {
        let pool = Arc::new(Pool::<data::RecordSchema>::new(6, 256));

        let (src_tx, src_rx) = connector_mut::<data::RecordSchema, ()>(2);
        let (a_tx, a_rx) = connector_ref::<data::RecordSchema, ()>(2);
        let (b_tx, b_rx) = connector_ref::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new(());

        let a = a_tx.clone();
        let b = b_tx.clone();
        pipeline.stage_fn("fanout", 1, src_rx, a_tx.clone(), move |batch, _emit| {
            let shared = batch.freeze();
            a.send(shared.clone()).unwrap();
            b.send(shared).unwrap();
            Ok(())
        });

        let mut batch = pool.acquire().unwrap();
        batch.mutate(
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
    }

    /// Fan-in: two producers feed into one consumer stage
    #[test]
    fn fan_in() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let pool = Arc::new(Pool::<data::RecordSchema>::new(6, 256));

        let (merge_tx, merge_rx) = connector_mut::<data::RecordSchema, ()>(4);
        let (out_tx, out_rx) = connector_mut::<data::RecordSchema, ()>(4);

        let mut pipeline = Pipeline::new(());

        let counter = Arc::new(AtomicU32::new(0));
        let cnt = counter.clone();
        pipeline.stage_fn("sink", 1, merge_rx, out_tx.clone(), move |batch, emit| {
            cnt.fetch_add(1, Ordering::Relaxed);
            emit.emit(batch)
        });

        let tx1 = merge_tx.clone();
        let tx2 = merge_tx.clone();
        drop(merge_tx);

        for tx in [tx1, tx2] {
            let mut batch = pool.acquire().unwrap();
            batch.mutate(
                (data::schema::value,),
                |(vals,): (&mut [u32],)| { vals[0] = 42; },
            );
            tx.send(batch).unwrap();
        }
        drop(out_tx);

        let _r1 = out_rx.recv().unwrap();
        let _r2 = out_rx.recv().unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    /// Multiple chunks through a multi-worker stage
    #[test]
    fn multi_worker() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let pool = Arc::new(Pool::<data::RecordSchema>::new(16, 256));

        let (src_tx, src_rx) = connector_mut::<data::RecordSchema, ()>(8);
        let (out_tx, out_rx) = connector_mut::<data::RecordSchema, ()>(8);

        let mut pipeline = Pipeline::new(());

        let processed = Arc::new(AtomicU32::new(0));
        let cnt = processed.clone();
        pipeline.stage_fn("workers", 4, src_rx, out_tx.clone(), move |batch, emit| {
            cnt.fetch_add(1, Ordering::Relaxed);
            emit.emit(batch)
        });

        for i in 0..8u32 {
            let mut batch = pool.acquire().unwrap();
            batch.mutate(
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
    }

    /// Slots return to the pool after pipeline drains
    #[test]
    fn pool_slots_reclaimed() {
        let pool = Arc::new(Pool::<data::RecordSchema>::new(4, 256));

        let (src_tx, src_rx) = connector_mut::<data::RecordSchema, ()>(2);
        let (out_tx, out_rx) = connector_mut::<data::RecordSchema, ()>(2);

        let mut pipeline = Pipeline::new(());

        pipeline.stage_fn("passthrough", 1, src_rx, out_tx.clone(), |batch, emit| {
            emit.emit(batch)
        });

        let mut batch = pool.acquire().unwrap();
        batch.mutate(
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
    }
}
