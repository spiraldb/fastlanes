use std::mem::MaybeUninit;
use std::mem::size_of;

use arrayref::{array_mut_ref, array_ref};
use divan::Bencher;
use fastlanes::{BitPacking, FastLanes};
use std::hint::black_box;

mod shared;

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
    let values = [3u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];

    bencher.bench_local(|| {
        BitPacking::pack::<WIDTH, B>(black_box(&values), &mut packed);
        black_box(&packed);
    });
}

#[divan::bench(sample_count = 10000)]
fn unpack_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [3u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    BitPacking::pack::<WIDTH, B>(&values, &mut packed);

    let mut unpacked = [0u16; 1024];

    bencher.bench_local(|| {
        BitPacking::unpack::<WIDTH, B>(black_box(&packed), &mut unpacked);
        black_box(&unpacked);
    });
}

#[divan::bench(sample_count = 10000)]
fn unchecked_unpack_16_from_3_stack(bencher: Bencher) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    let values = [3u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    BitPacking::pack::<WIDTH, B>(&values, &mut packed);

    let mut unpacked = [0u16; 1024];

    bencher.bench_local(|| {
        unsafe { BitPacking::unchecked_unpack(WIDTH, black_box(&packed), &mut unpacked) };
        black_box(&unpacked);
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

const MAX_BENCHMARK_INDICES: usize = 128;

#[derive(Clone, Copy)]
enum IndexDistribution {
    Uniform,
    Clustered,
    UnorderedWithDuplicates,
}

fn benchmark_indices(
    distribution: IndexDistribution,
    num_indices: usize,
) -> [usize; MAX_BENCHMARK_INDICES] {
    assert!(num_indices <= MAX_BENCHMARK_INDICES);
    let mut indices = [0; MAX_BENCHMARK_INDICES];

    match distribution {
        IndexDistribution::Uniform => {
            for (position, index) in indices[..num_indices].iter_mut().enumerate() {
                *index = position * 1024 / num_indices;
            }
        }
        IndexDistribution::Clustered => {
            for (position, index) in indices[..num_indices].iter_mut().enumerate() {
                *index = 448 + position;
            }
        }
        IndexDistribution::UnorderedWithDuplicates => {
            for position in 0..num_indices {
                indices[position] = if position > 0 && position % 4 == 0 {
                    indices[position - 1]
                } else {
                    (position * 541 + 17) % 1024
                };
            }
        }
    }

    indices
}

macro_rules! unpack_indices_benchmarks {
    ($module:ident, $type:ty, $width:expr, $distribution:expr) => {
        mod $module {
            use super::*;

            const WIDTH: usize = $width;
            const PACKED_LENGTH: usize = 1024 * WIDTH / <$type>::T;

            fn fixture(
                num_indices: usize,
            ) -> ([$type; PACKED_LENGTH], [usize; MAX_BENCHMARK_INDICES]) {
                let mask = ((1_u128 << WIDTH) - 1) as $type;
                let values =
                    std::array::from_fn(|index| ((index as $type).wrapping_mul(17)) & mask);
                let mut packed = [0; PACKED_LENGTH];
                BitPacking::pack::<WIDTH, PACKED_LENGTH>(&values, &mut packed);
                let indices = benchmark_indices($distribution, num_indices);
                (packed, indices)
            }

            #[divan::bench(args = [1, 8, 32, 128], sample_size = 10_000)]
            fn batched(bencher: Bencher, num_indices: usize) {
                let (packed, indices) = fixture(num_indices);
                let mut output = [MaybeUninit::<$type>::uninit(); MAX_BENCHMARK_INDICES];

                bencher.bench_local(|| {
                    let packed = black_box(&packed);
                    let indices = black_box(&indices[..num_indices]);
                    let output = black_box(&mut output[..num_indices]);
                    // SAFETY: `packed` contains exactly one packed FastLanes block.
                    unsafe {
                        BitPacking::unchecked_unpack_indices(WIDTH, packed, indices, output);
                    }
                    black_box(&*output);
                });
            }

            #[divan::bench(args = [1, 8, 32, 128], sample_size = 10_000)]
            fn repeated_single(bencher: Bencher, num_indices: usize) {
                let (packed, indices) = fixture(num_indices);
                let mut output = [MaybeUninit::<$type>::uninit(); MAX_BENCHMARK_INDICES];

                bencher.bench_local(|| {
                    let packed = black_box(&packed);
                    let indices = black_box(&indices[..num_indices]);
                    let output = black_box(&mut output[..num_indices]);
                    for (&index, value) in indices.iter().zip(output.iter_mut()) {
                        // SAFETY: `packed` contains exactly one packed FastLanes block.
                        value.write(unsafe {
                            BitPacking::unchecked_unpack_single(WIDTH, packed, index)
                        });
                    }
                    black_box(&*output);
                });
            }

            #[divan::bench(args = [1, 8, 32, 128], sample_size = 10_000)]
            fn full_unpack_then_gather(bencher: Bencher, num_indices: usize) {
                let (packed, indices) = fixture(num_indices);
                let mut unpacked = [0; 1024];
                let mut output = [MaybeUninit::<$type>::uninit(); MAX_BENCHMARK_INDICES];

                bencher.bench_local(|| {
                    let packed = black_box(&packed);
                    let indices = black_box(&indices[..num_indices]);
                    let unpacked = black_box(&mut unpacked);
                    let output = black_box(&mut output[..num_indices]);
                    // SAFETY: both buffers have the required lengths for `WIDTH`.
                    unsafe { BitPacking::unchecked_unpack(WIDTH, packed, unpacked) };
                    for (&index, value) in indices.iter().zip(output.iter_mut()) {
                        value.write(unpacked[index]);
                    }
                    black_box(&*output);
                });
            }
        }
    };
}

unpack_indices_benchmarks!(
    unpack_indices_u16_width3_uniform,
    u16,
    3,
    IndexDistribution::Uniform
);
unpack_indices_benchmarks!(
    unpack_indices_u32_width16_clustered,
    u32,
    16,
    IndexDistribution::Clustered
);
unpack_indices_benchmarks!(
    unpack_indices_u64_width63_unordered_duplicates,
    u64,
    63,
    IndexDistribution::UnorderedWithDuplicates
);

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
