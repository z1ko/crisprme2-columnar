use pyo3::prelude::*;
use columnar::macros::Columnar;
use columnar::{ColumnarBuffer, RingSlot, Schema};
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

#[pyclass]
pub struct PyAlignmentBatch {
    buffer: ColumnarBuffer<
        alignment::AlignmentSchema, 
        RingSlot>,
}

#[pymethods]
impl PyAlignmentBatch {

    #[new]
    fn new(elements: usize) -> Self { // Only for debug

        let bytes = elements * alignment::AlignmentSchema::STRIDE;
        let mut buffer = RingSlot::new(bytes)
            .columnar();

        println!("stride: {}", alignment::AlignmentSchema::STRIDE);
        println!("capacity: {}", buffer.capacity());

        for i in 0..elements {

            let guide = format!("ACGTACGTACGT{:04}", i);
            let seq = format!("ACGTACGTACGG{:04}", i);

            let mut rguide = [0u8; MAX_STR_LEN];
            rguide[..guide.len()].copy_from_slice(guide.as_bytes());

            let mut rseq = [0u8; MAX_STR_LEN];
            rseq[..seq.len()].copy_from_slice(seq.as_bytes());

            buffer.push(alignment::Alignment {
                occurence: i as u32,
                strand: if i % 2 == 0 { b'+' } else { b'-' },
                rguide,
                rseq,
                mism: (i % 5) as u8,
                bdna: (i % 3) as u8,
                brna: (i % 4) as u8,
                score: [0.1 * i as f32, 0.5 * i as f32, 0.9 * i as f32, 1.0 / (i + 1) as f32],
                annotations: [i as u32; MAX_ANNOTATIONS_LEN],
            });
        }

        Self { buffer }
    }

    /// Get all occurrences
    fn occurences(slf: Py<Self>, py: Python<'_>) -> PyResult<ColumnView> {
        let view = {
            let mut this = slf.borrow_mut(py);
            ColumnView::from_column(
                &mut this.buffer, 
                alignment::schema::occurence, 
                slf.clone_ref(py).into_any(), 
                false)
        };
        Ok(view)
    }

    /// Get all scores
    fn scores(slf: Py<Self>, py: Python<'_>) -> PyResult<Vec<ColumnView>> {
        let views = {
            let mut this = slf.borrow_mut(py);
            ColumnView::from_group_column(
                &mut this.buffer, 
                alignment::schema::score, 
                slf.clone_ref(py).into_any(), 
                false, 
                py)
        };
        Ok(views)
    }
    
}

/// A Python module implemented in Rust.
#[pymodule]
mod columnar_python {
    use columnar::ext::pyo3::ColumnView;
    use pyo3::prelude::*;

    use crate::PyAlignmentBatch;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {

        m.add_class::<PyAlignmentBatch>()?;
        m.add_class::<ColumnView>()?;

        Ok(())
    }
}
