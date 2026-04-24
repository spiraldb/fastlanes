use arrayref::{array_mut_ref, array_ref};
use divan::Bencher;
use std::hint::black_box;
use std::mem::size_of;

use fastlanes::{BitPacking, Delta, FastLanes, Transpose};

mod shared;

fn main() {
    divan::main();
}

#[divan::bench]
fn delta_u16_fused(bencher: Bencher) {
    const W: usize = 9;
    const B: usize = 1024 * W / <u16 as FastLanes>::T;
    const LANES: usize = u16::LANES;

    let mut values: [u16; 1024] = [0; 1024];
    for i in 0..1024 {
        values[i] = (i / 8) as u16;
    }

    let mut transposed = [0; 1024];
    Transpose::transpose(&values, &mut transposed);

    let mut deltas = [0; 1024];
    Delta::delta(&transposed, &[0; 64], &mut deltas);

    let mut packed = [0; 128 * W / size_of::<u16>()];
    BitPacking::pack::<W, B>(&deltas, &mut packed);

    with_counter!(bencher, values.len() * std::mem::size_of::<u16>()).bench_local(|| {
        let mut unpacked = [0; 1024];
        Delta::undelta_pack::<LANES, W, B>(black_box(&packed), black_box(&[0; 64]), &mut unpacked);
        unpacked
    });
}

#[divan::bench]
fn delta_u16_unfused(bencher: Bencher) {
    const W: usize = 9;
    const B: usize = 1024 * W / <u16 as FastLanes>::T;

    let mut values: [u16; 1024] = [0; 1024];
    for i in 0..1024 {
        values[i] = (i / 8) as u16;
    }

    let mut transposed = [0; 1024];
    Transpose::transpose(&values, &mut transposed);

    let mut deltas = [0; 1024];
    Delta::delta(&transposed, &[0; 64], &mut deltas);

    let mut packed = [0; 128 * W / size_of::<u16>()];
    BitPacking::pack::<W, B>(&deltas, &mut packed);

    with_counter!(bencher, values.len() * std::mem::size_of::<u16>()).bench_local(|| {
        let mut unpacked = [0; 1024];
        BitPacking::unpack::<W, B>(black_box(&packed), &mut unpacked);
        let mut undelta = [0; 1024];
        Delta::undelta(black_box(&unpacked), black_box(&[0; 64]), &mut undelta);
        undelta
    });
}

#[divan::bench]
fn delta_throughput_compress(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();

    let values: Vec<u16> = (0..N).map(|i| (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    with_counter!(bencher, values.len() * std::mem::size_of::<u16>()).bench_local(|| {
        for i in 0..NUM_BATCHES {
            BitPacking::pack::<WIDTH, B>(
                black_box(array_ref![values, i * 1024, 1024]),
                array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
            );
        }
        black_box(&packed);
    });
}

#[divan::bench]
fn delta_throughput_decompress(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();

    let mut values: Vec<u16> = (0..N).map(|i| (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    for i in 0..NUM_BATCHES {
        BitPacking::pack::<WIDTH, B>(
            array_ref![values, i * 1024, 1024],
            array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
        );
    }

    with_counter!(bencher, values.len() * std::mem::size_of::<u16>()).bench_local(|| {
        for i in 0..NUM_BATCHES {
            BitPacking::unpack::<WIDTH, B>(
                black_box(array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE]),
                array_mut_ref![values, i * 1024, 1024],
            );
        }
        black_box(&values);
    });
}
