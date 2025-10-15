use std::mem::size_of;

use arrayref::{array_mut_ref, array_ref};
use divan::Bencher;
use fastlanes::{BitPacking, FastLanes, FoR};
use std::hint::black_box;

mod shared;

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 10000)]
fn for_pack_16_to_3_heap(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = vec![13u16; 1024];
    let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];
    let reference = 10u16;

    bencher.bench_local(|| {
        FoR::for_pack::<WIDTH, B>(
            black_box(array_ref![values, 0, 1024]),
            reference,
            array_mut_ref![packed, 0, 128 * WIDTH / size_of::<u16>()],
        );
        black_box(&packed);
    });
}

#[divan::bench(sample_count = 10000)]
fn for_pack_16_to_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [13u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    let reference = 10u16;

    bencher.bench_local(|| {
        FoR::for_pack::<WIDTH, B>(black_box(&values), reference, &mut packed);
        black_box(packed);
    });
}

#[divan::bench(sample_count = 10000)]
fn unfor_pack_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [13u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    let reference = 10u16;
    FoR::for_pack::<WIDTH, B>(&values, reference, &mut packed);

    let mut unpacked = [0u16; 1024];

    bencher.bench_local(|| {
        FoR::unfor_pack::<WIDTH, B>(&black_box(packed), reference, &mut unpacked);
        black_box(unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn unchecked_unfor_pack_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [13u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    let reference = 10u16;
    FoR::for_pack::<WIDTH, B>(&values, reference, &mut packed);

    let mut unpacked = [0u16; 1024];

    bencher.bench_local(|| {
        unsafe { FoR::unchecked_unfor_pack(WIDTH, &black_box(packed), reference, &mut unpacked) };
        black_box(unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn throughput_compress(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();
    const REFERENCE: u16 = 1000;

    let values: Vec<u16> = (0..N).map(|i| REFERENCE + (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    with_counter!(bencher, values.len() * std::mem::size_of::<u16>()).bench_local(|| {
        for i in 0..NUM_BATCHES {
            FoR::for_pack::<WIDTH, B>(
                black_box(array_ref![values, i * 1024, 1024]),
                REFERENCE,
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
    const REFERENCE: u16 = 1000;

    let values: Vec<u16> = (0..N).map(|i| REFERENCE + (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    for i in 0..NUM_BATCHES {
        FoR::for_pack::<WIDTH, B>(
            array_ref![values, i * 1024, 1024],
            REFERENCE,
            array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
        );
    }

    let mut unpacked = vec![0u16; N];

    with_counter!(bencher, unpacked.len() * std::mem::size_of::<u16>()).bench_local(|| {
        for i in 0..NUM_BATCHES {
            FoR::unfor_pack::<WIDTH, B>(
                black_box(array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE]),
                REFERENCE,
                array_mut_ref![unpacked, i * 1024, 1024],
            );
        }
        black_box(&unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn throughput_decompress_unchecked(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();
    const REFERENCE: u16 = 1000;

    let values: Vec<u16> = (0..N).map(|i| REFERENCE + (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    for i in 0..NUM_BATCHES {
        FoR::for_pack::<WIDTH, B>(
            array_ref![values, i * 1024, 1024],
            REFERENCE,
            array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
        );
    }

    let mut unpacked = vec![0u16; N];

    with_counter!(bencher, unpacked.len() * std::mem::size_of::<u16>()).bench_local(|| {
        for i in 0..NUM_BATCHES {
            unsafe {
                FoR::unchecked_unfor_pack(
                    WIDTH,
                    black_box(array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE]),
                    REFERENCE,
                    array_mut_ref![unpacked, i * 1024, 1024],
                )
            };
        }
        black_box(&unpacked);
    });
}

// Benchmarks that separate bitpacking unpack from reference value application

#[divan::bench(sample_count = 10000)]
fn unpack_then_add_reference_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [13u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    let reference = 10u16;
    FoR::for_pack::<WIDTH, B>(&values, reference, &mut packed);

    let mut unpacked = [0u16; 1024];

    bencher.bench_local(|| {
        // First, unpack using bitpacking kernel
        BitPacking::unpack::<WIDTH, B>(&black_box(packed), &mut unpacked);
        // Then, apply reference values in a separate loop
        for i in 0..1024 {
            unpacked[i] = unpacked[i].wrapping_add(reference);
        }
        black_box(unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn unchecked_unpack_then_add_reference_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [13u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    let reference = 10u16;
    FoR::for_pack::<WIDTH, B>(&values, reference, &mut packed);

    let mut unpacked = [0u16; 1024];

    bencher.bench_local(|| {
        // First, unpack using unchecked bitpacking kernel
        unsafe { BitPacking::unchecked_unpack(WIDTH, &black_box(packed), &mut unpacked) };
        // Then, apply reference values in a separate loop
        for i in 0..1024 {
            unpacked[i] = unpacked[i].wrapping_add(reference);
        }
        black_box(unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn throughput_decompress_separate_reference(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();
    const REFERENCE: u16 = 1000;

    let values: Vec<u16> = (0..N).map(|i| REFERENCE + (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    for i in 0..NUM_BATCHES {
        FoR::for_pack::<WIDTH, B>(
            array_ref![values, i * 1024, 1024],
            REFERENCE,
            array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
        );
    }

    let mut unpacked = vec![0u16; N];

    with_counter!(bencher, unpacked.len() * std::mem::size_of::<u16>()).bench_local(|| {
        // First pass: unpack all batches using bitpacking kernel
        for i in 0..NUM_BATCHES {
            BitPacking::unpack::<WIDTH, B>(
                black_box(array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE]),
                array_mut_ref![unpacked, i * 1024, 1024],
            );
        }
        // Second pass: apply reference values
        for val in unpacked.iter_mut() {
            *val = val.wrapping_add(REFERENCE);
        }
        black_box(&unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn throughput_decompress_unchecked_separate_reference(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();
    const REFERENCE: u16 = 1000;

    let values: Vec<u16> = (0..N).map(|i| REFERENCE + (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    for i in 0..NUM_BATCHES {
        FoR::for_pack::<WIDTH, B>(
            array_ref![values, i * 1024, 1024],
            REFERENCE,
            array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
        );
    }

    let mut unpacked = vec![0u16; N];

    with_counter!(bencher, unpacked.len() * std::mem::size_of::<u16>()).bench_local(|| {
        // First pass: unpack all batches using unchecked bitpacking kernel
        for i in 0..NUM_BATCHES {
            unsafe {
                BitPacking::unchecked_unpack(
                    WIDTH,
                    black_box(array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE]),
                    array_mut_ref![unpacked, i * 1024, 1024],
                )
            };
        }
        // Second pass: apply reference values
        for val in unpacked.iter_mut() {
            *val = val.wrapping_add(REFERENCE);
        }
        black_box(&unpacked);
    });
}
