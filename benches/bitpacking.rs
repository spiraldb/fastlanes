use std::mem::size_of;

use arrayref::{array_mut_ref, array_ref};
use divan::Bencher;
use fastlanes::{BitPacking, FastLanes};
use std::hint::black_box;

mod shared;

use shared::Aligned;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 10000)]
fn pack_16_to_3_heap(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = vec![3u16; 1024];
    let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];

    bencher.bench_local(|| {
        BitPacking::pack::<WIDTH, B>(
            black_box(array_ref![values, 0, 1024]),
            array_mut_ref![packed, 0, 128 * WIDTH / size_of::<u16>()],
        );
        black_box(&packed);
    });
}

#[divan::bench(sample_count = 10000)]
fn pack_16_to_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = Aligned([3u16; 1024]);
    let mut packed = Aligned([0; 128 * WIDTH / size_of::<u16>()]);

    bencher.bench_local(|| {
        BitPacking::pack::<WIDTH, B>(black_box(&values.0), &mut packed.0);
        black_box(&packed.0);
    });
}

#[divan::bench(sample_count = 10000)]
fn unpack_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = Aligned([3u16; 1024]);
    let mut packed = Aligned([0; 128 * WIDTH / size_of::<u16>()]);
    BitPacking::pack::<WIDTH, B>(&values.0, &mut packed.0);

    let mut unpacked = Aligned([0u16; 1024]);

    bencher.bench_local(|| {
        BitPacking::unpack::<WIDTH, B>(black_box(&packed.0), &mut unpacked.0);
        black_box(&unpacked.0);
    });
}

#[divan::bench(sample_count = 10000)]
fn unchecked_unpack_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = Aligned([3u16; 1024]);
    let mut packed = Aligned([0; 128 * WIDTH / size_of::<u16>()]);
    BitPacking::pack::<WIDTH, B>(&values.0, &mut packed.0);

    let mut unpacked = Aligned([0u16; 1024]);

    bencher.bench_local(|| {
        unsafe { BitPacking::unchecked_unpack(WIDTH, black_box(&packed.0), &mut unpacked.0) };
        black_box(&unpacked.0);
    });
}

#[divan::bench(sample_count = 10000)]
fn unpack_single_16_from_3(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = vec![3u16; 1024];
    let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];
    BitPacking::pack::<WIDTH, B>(array_ref![values, 0, 1024], array_mut_ref![packed, 0, 192]);

    bencher.bench_local(|| {
        for i in 0..1024 {
            black_box::<u16>(BitPacking::unpack_single::<WIDTH, B>(
                black_box(array_ref![packed, 0, 192]),
                black_box(i),
            ));
        }
    });
}

#[divan::bench(sample_count = 10000)]
fn throughput_compress(bencher: Bencher) {
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

#[divan::bench(sample_count = 10000)]
fn throughput_decompress(bencher: Bencher) {
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
