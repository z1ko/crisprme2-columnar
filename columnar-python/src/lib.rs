use std::sync::Arc;

use columnar::pipeline::Pipeline;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use columnar::macros::Columnar;
use columnar::buffer::Schema;
use columnar::ring::{Batch, ConnectorRx, ConnectorTx, Pool};
use columnar::ext::pyo3::ColumnView;

const MAX_STR_LEN: usize = 32;
const MAX_ANNOTATIONS_LEN: usize = 10;
const MAX_SCORES_LEN: usize = 4;

type StrBytes = [u8; MAX_STR_LEN];

pub mod alignment {
    use super::*;

    #[derive(Debug, Clone, Columnar)]
    pub struct Alignment {

        /// position
        pub occurence: u32,
        pub strand: u8,

        /// row-major byte buffers
        pub rguide: StrBytes,
        pub rseq: StrBytes,

        /// alignment results
        pub mism: u8,
        pub bdna: u8,
        pub brna: u8,

        /// column-major scores
        #[columnar(group)]
        pub score: [f32; MAX_SCORES_LEN],

        /// column-major annotations
        #[columnar(group)]
        pub annotations: [u32; MAX_ANNOTATIONS_LEN]
    }
}

pub mod sequence {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Columnar)]
    pub struct Sequence {
        pub id: u32,
    }
}

// =============================================================================
// Batch wrappers — schema-specific Python-visible batch handles
// =============================================================================

/// Batch of sequences
#[pyclass(str = "{batch:?}")]
pub struct PySequenceBatch {
    batch: Option<Batch<sequence::SequenceSchema, ()>>,
}

#[pymethods]
impl PySequenceBatch {
    fn ids(slf: Py<Self>, py: Python<'_>) -> PyResult<ColumnView> {
        let mut this = slf.borrow_mut(py);
        let batch = this.batch.as_mut()
            .ok_or(PyRuntimeError::new_err("batch consumed"))?;
        Ok(ColumnView::from_batch(
            batch, sequence::schema::id, slf.clone_ref(py).into_any(), true))
    }
}

/// Batch of alignments
#[pyclass(str = "{batch:?}")]
pub struct PyAlignmentBatch {
    batch: Option<Batch<alignment::AlignmentSchema, ()>>,
}

#[pymethods]
impl PyAlignmentBatch {
    fn occurences(slf: Py<Self>, py: Python<'_>) -> PyResult<ColumnView> {
        let mut this = slf.borrow_mut(py);
        let batch = this.batch.as_mut()
            .ok_or(PyRuntimeError::new_err("batch consumed"))?;
        Ok(ColumnView::from_batch(
            batch, alignment::schema::occurence, slf.clone_ref(py).into_any(), true))
    }
}

// =============================================================================
// Source stage — Python produces sequences, Rust pipeline transforms them
// =============================================================================

/// Source handle: Python acquires batches, fills them, and publishes.
#[pyclass(unsendable)]
pub struct PySequenceSource {
    tx: Option<ConnectorTx<sequence::SequenceSchema, ()>>,
    pool: Arc<Pool<sequence::SequenceSchema>>,
}

#[pymethods]
impl PySequenceSource {
    fn acquire(&self) -> PyResult<PySequenceBatch> {
        let batch = self.pool.acquire()
            .map_err(|e| PyRuntimeError::new_err(format!("pool acquire failed: {e}")))?;
        Ok(PySequenceBatch { batch: Some(batch) })
    }

    fn publish(&self, py_batch: &mut PySequenceBatch, len: usize) -> PyResult<()> {
        let mut batch = py_batch.batch.take()
            .ok_or(PyRuntimeError::new_err("batch already consumed"))?;
        batch.as_mut().set_len(len);

        let tx = self.tx.as_ref()
            .ok_or(PyRuntimeError::new_err("source is closed"))?;
        tx.send(batch)
            .map_err(|e| PyRuntimeError::new_err(format!("downstream disconnected: {e}")))?;
        Ok(())
    }

    fn close(&mut self) { self.tx.take(); }
}

// =============================================================================
// Sink stage — Python consumes alignment results
// =============================================================================

/// Sink handle: Python receives read-only alignment batches.
#[pyclass]
pub struct PyAlignmentSink {
    rx: columnar::ring::ConnectorRx<alignment::AlignmentSchema, ()>,
}

#[pymethods]
impl PyAlignmentSink {
    fn recv(&self) -> Option<PyAlignmentBatch> {
        self.rx.recv().ok().map(|batch| PyAlignmentBatch { batch: Some(batch) })
    }
}

// =============================================================================
// Pipeline factory — wires up stages with Python callbacks
// =============================================================================

#[pyclass(str = "{pipeline:?}")]
pub struct PyPipeline {

    pipeline: Pipeline,
    seq_pool: Arc<Pool<sequence::SequenceSchema>>,

    src_tx: ConnectorTx<sequence::SequenceSchema, ()>,
    out_rx: ConnectorRx<alignment::AlignmentSchema, ()>
}

#[pymethods]
impl PyPipeline {

    // Example submit
    fn submit(&mut self, py: Python<'_>) {
        println!("Submitting new sequence batch...");

        let mut input = self.seq_pool.acquire().unwrap();
        input.as_mut().mutate(
            (sequence::schema::id,),
            |(ids,)| {
                for (i, id) in ids.iter_mut().enumerate() {
                    *id = i as u32;
                }
            }
        );

        // Release GIL so stage threads can acquire it
        py.detach(|| {
            self.src_tx.send(input).unwrap();
        });

        println!("Submission done!")
    }

    // Example receive
    fn receive(&mut self, py: Python<'_>) -> PyResult<PyAlignmentBatch> {
        println!("Receiving alignment batch...");

        // Release GIL so stage threads can acquire it while we block
        let result = py.detach(|| self.out_rx.recv());

        match result {
            Ok(batch) => {
                print!("Received!");
                Ok(PyAlignmentBatch { batch: Some(batch) })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("recv failed: {e}")))
        }
    }
}

#[pymodule]
mod columnar_python {
    use columnar::{ext::pyo3::ColumnView, ring::connector};
    use pyo3::prelude::*;
    use crate::*;

    /// Create a pipeline: source (Python) → transform (Python callback on Rust thread) → sink (Python).
    ///
    /// The `transform` callback receives a `PyAlignmentWriteBatch` for in-place modification.
    /// After the callback returns, the batch is auto-published downstream.
    #[pyfunction]
    fn create_pipeline(transform: Py<PyAny>,) -> PyPipeline {

        let (seq_pool, align_pool) = (
            Arc::new(Pool::<sequence::SequenceSchema>::new(2, 128)),
            Arc::new(Pool::<alignment::AlignmentSchema>::new(2, 128))
        );

        let mut pipeline = Pipeline::new();

        let (src_tx, src_rx) = connector::<sequence::SequenceSchema, ()>(2);
        let (out_tx, out_rx) = connector::<alignment::AlignmentSchema, ()>(2);

        // Middle stage: Rust thread receives sequences, produces alignment batches,
        // calls Python callback for in-place transformation, then forwards downstream.
        pipeline.stage("py-transform", src_rx, 1, move |input, _ctx| {
            println!("running py-transform...");

            let result = align_pool.acquire().unwrap();
            let alignments = PyAlignmentBatch { batch: Some(result) };
            let sequences = PySequenceBatch { batch: Some(input) };

            Python::try_attach(|py| {

                let py_input = Py::new(py, sequences).unwrap();
                let py_batch = Py::new(py, alignments).unwrap();

                transform.call1(py, (&py_input, &py_batch,)).unwrap();

                // Take the batch back after callback returns
                let mut inner = py_batch.borrow_mut(py);
                if let Some(batch) = inner.batch.take() {
                    out_tx.send(batch).unwrap();
                }
            });

            println!("run!");
        });

        PyPipeline { pipeline, seq_pool, src_tx, out_rx }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<ColumnView>()?;
        m.add_class::<PySequenceBatch>()?;
        m.add_class::<PyAlignmentBatch>()?;
        m.add_class::<PySequenceSource>()?;
        m.add_class::<PyAlignmentSink>()?;
        Ok(())
    }
}
