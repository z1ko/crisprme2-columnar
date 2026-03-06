use columnar_derive::Columnar;
use columnar::{Columnar, Schema, SoAWrite};

#[repr(C)]
#[derive(Debug, Columnar)]
pub struct Sequence {
    pub id: u32,
    pub other: [u8; 4],
}