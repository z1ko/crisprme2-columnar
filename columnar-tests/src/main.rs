use columnar::{RingSlot, Schema};
use columnar_derive::Columnar;

mod sequence;

fn main() {
    println!("Hello, world!");

    let buffer = RingSlot::new(1024);
    let mut frame = buffer.columnar::<sequence::SequenceSchema>();

    for i in 0..10 {
        frame.push_with((sequence::schema::id, sequence::schema::other),
            |idx, (ids, others,)| {
                others[idx] = [i, i + 1, i + 2, i + 3];
                ids[idx] = i as u32;
            }
        );
    }

    for i in 10..20 {
        frame.push(sequence::Sequence {
            other: [i, i + 1, i + 2, i + 3], id: i as u32
        });
    }

    println!("ELEM_SIZES: {:?}", sequence::SequenceSchema::ELEM_SIZES);
    println!("STRIDE: {:?}",     sequence::SequenceSchema::STRIDE);
    println!("OFFSET: {:?}",     sequence::SequenceSchema::BLOCK_OFFSETS);
    println!("CAPACITY: {:?}",   frame.capacity());
    println!("COUNT: {}",        frame.len());

    frame.mutate((sequence::schema::id, sequence::schema::other), 
    |(ids, others)| {
            for i in 0..ids.len() {
                println!("id    : {}", ids[i]);
                println!("other : {:?}", others[i]);
            }
        }
    );

    let s5: sequence::Sequence = frame.get(5).unwrap();
    println!("{:?}", s5);

}
