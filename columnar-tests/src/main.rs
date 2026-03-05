use bytemuck::{Pod, Zeroable};
use columnar::{RingSlot, Schema};
use columnar_derive::Columnar;

#[repr(C)]
#[derive(Debug, Clone, Columnar)]
pub struct Tick {
    pub timestamp: u64,
    pub price:     f64,
    pub quantity:  u32,
}

pub mod schema {
    pub mod tick {
        #![allow(non_camel_case_types)]
        use columnar::ColumnType;

        use super::super::*;

        #[derive(Clone, Copy)] pub struct timestamp;
        #[derive(Clone, Copy)] pub struct price;
        #[derive(Clone, Copy)] pub struct quantity;

        impl ColumnType for timestamp {
            type Schema = TickSchema;
            type Value  = u64;
            fn col_index(self) -> usize { 0 }
            
            fn offset(self, row_capacity: usize) -> usize {
                todo!()
            }
        }
        impl ColumnType for price {
            type Schema = TickSchema;
            type Value  = f64;
            fn col_index(self) -> usize { 1 }

            fn offset(self, row_capacity: usize) -> usize {
                todo!()
            }
        }
        impl ColumnType for quantity {
            type Schema = TickSchema;
            type Value  = u32;
            fn col_index(self) -> usize { 2 }

            fn offset(self, row_capacity: usize) -> usize {
                todo!()
            }
        }
    }
}

fn main() {
    println!("Hello, world!");

    let mut buffer = RingSlot::new(1024);
    let mut ticks = buffer.columnar::<TickSchema>();
    println!("ticks count: {}", ticks.len());

    let mut a = 0;

    ticks.mutate((schema::tick::timestamp,), 
    |(timestamps,)| {
            timestamps[0] = 123456789;
            a += 1;
        }
    );
}
