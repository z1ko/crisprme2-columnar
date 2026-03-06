use columnar::{ColumnarBuffer, RingSlot, Schema};
use columnar::macros::Columnar;

use crate::alignment::Alignment;

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

fn main() {

    let mut alignments = RingSlot::new(1024)
        .columnar::<alignment::AlignmentSchema>();

    alignments.push(Alignment {
        occurence: 0,
        strand: 1,
        rguide: [0u8; MAX_STR_LEN],
        rseq: [0u8; MAX_STR_LEN],
        mism: 1,
        bdna: 1,
        brna: 2,
        score: [0.5, 0.2, 0.1, 0.6],
        annotations: [0u32; 10],
    });

    let (rguides,): (&[StrBytes],) = alignments.columns((alignment::schema::rguide,));
    for rguide in rguides {
        println!("{:?}", rguide);
    }

    let (scores,): ([&[f32]; 4],) = alignments.columns((alignment::schema::score,));
    for (a, b) in scores[0].iter().zip(scores[1]) {
        println!("score a: {}, score b: {}", a, b);
    }

    println!("Hello, world!");
}
