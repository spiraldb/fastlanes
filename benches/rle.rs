use arrayref::array_ref;
use divan::Bencher;
use fastlanes::RLE;
use std::hint::black_box;

mod shared;

fn main() {
    divan::main();
}

#[divan::bench]
fn rle_encode_u32(bencher: Bencher) {
    let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
    let mut rle_vals = [0u32; 1024];
    let mut rle_idxs = [0u16; 1024];

    bencher.bench_local(|| {
        // SAFETY: all arguments are 1024-element arrays.
        let unique_count =
            unsafe { u32::encode_unchecked(black_box(&input), &mut rle_vals, &mut rle_idxs) };
        black_box(unique_count);
    });
}

#[divan::bench]
fn rle_decode_u32(bencher: Bencher) {
    let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
    let mut rle_vals = [0u32; 1024];
    let mut rle_idxs = [0u16; 1024];
    // SAFETY: all arguments are 1024-element arrays.
    let unique_count = unsafe { u32::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

    let mut output = [0u32; 1024];

    bencher.bench_local(|| {
        // SAFETY: `encode_unchecked` only writes indices below the returned `unique_count`.
        unsafe {
            u32::decode_unchecked(
                black_box(&rle_vals[..unique_count]),
                black_box(&rle_idxs),
                &mut output,
            );
        }
        black_box(&output);
    });
}

#[divan::bench]
fn rle_throughput_encode_32(bencher: Bencher) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();

    with_counter!(bencher, input_data.len() * std::mem::size_of::<u32>()).bench_local(|| {
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];
        for batch in 0..NUM_BATCHES {
            let batch_start = batch * 1024;
            let input_batch = array_ref![input_data, batch_start, 1024];

            // SAFETY: all arguments are 1024-element arrays.
            let unique_count = unsafe {
                u32::encode_unchecked(black_box(input_batch), &mut rle_vals, &mut rle_idxs)
            };
            black_box(unique_count);
        }
    });
}

#[divan::bench]
fn rle_throughput_decode_32(bencher: Bencher) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();

    // Pre-encode all batches
    let mut encoded_batches = Vec::new();
    for batch in 0..NUM_BATCHES {
        let batch_start = batch * 1024;
        let input_batch: [u32; 1024] = std::array::from_fn(|i| input_data[batch_start + i]);

        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];
        // SAFETY: all arguments are 1024-element arrays.
        let unique_count =
            unsafe { u32::encode_unchecked(&input_batch, &mut rle_vals, &mut rle_idxs) };
        encoded_batches.push((rle_vals, rle_idxs, unique_count));
    }

    with_counter!(bencher, input_data.len() * std::mem::size_of::<u32>()).bench_local(|| {
        let mut output = [0u32; 1024];
        for (rle_vals, rle_idxs, unique_count) in &encoded_batches {
            // SAFETY: `encode_unchecked` only writes indices below the returned `unique_count`.
            unsafe {
                u32::decode_unchecked(
                    black_box(&rle_vals[..*unique_count]),
                    black_box(rle_idxs),
                    &mut output,
                );
            }
            black_box(&output);
        }
    });
}

/// Decompression of 1M `u32` values with random run lengths averaging `MEAN_RUN`, comparing the
/// RLE index representations against run-end encoding.
mod decode_vs_runend {
    use super::{Bencher, RLE, black_box, with_counter};
    use fastlanes::{BitPacking, Delta, Transpose, transpose};

    const NUM_CHUNKS: usize = 1024;
    const N: usize = 1024 * NUM_CHUNKS;
    const MEAN_RUNS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 256, 1024];

    /// Random run lengths uniform in `1..2 * mean_run`, with random values.
    fn generate(mean_run: usize) -> Vec<u32> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut data = Vec::with_capacity(N);
        while data.len() < N {
            let run = 1 + (next() as usize % (2 * mean_run - 1));
            let value = next() as u32;
            data.extend(std::iter::repeat_n(value, run.min(N - data.len())));
        }
        data
    }

    struct Chunk {
        values: [u32; 1024],
        num_values: usize,
        indices: [u16; 1024],
        index_width: usize,
        packed_indices: [u16; 1024],
        run_starts: [u16; 64],
        bases: [u16; 64],
    }

    fn encode_chunks(data: &[u32]) -> Vec<Chunk> {
        data.as_chunks::<1024>()
            .0
            .iter()
            .map(|input| {
                let mut chunk = Chunk {
                    values: [0; 1024],
                    num_values: 0,
                    indices: [0; 1024],
                    index_width: 0,
                    packed_indices: [0; 1024],
                    run_starts: [0; 64],
                    bases: [0; 64],
                };
                chunk.num_values =
                    unsafe { u32::encode_unchecked(input, &mut chunk.values, &mut chunk.indices) };
                chunk.index_width = (16 - (chunk.num_values as u16 - 1).leading_zeros()) as usize;
                let packed_len = 64 * chunk.index_width;
                unsafe {
                    BitPacking::unchecked_pack(
                        chunk.index_width,
                        &chunk.indices,
                        &mut chunk.packed_indices[..packed_len],
                    );
                }
                let mut values = [0; 1024];
                unsafe {
                    u32::encode_run_bitmap_unchecked(input, &mut values, &mut chunk.run_starts)
                };
                chunk.bases = std::array::from_fn(|lane| {
                    let start = transpose(lane);
                    if start == 0 {
                        0
                    } else {
                        chunk.indices[start - 1]
                    }
                });

                // Sanity check that the bases plus the bitmap are Delta(BitPack<1>).
                let mut transposed = [0u16; 1024];
                Transpose::transpose(&chunk.indices, &mut transposed);
                let mut deltas = [0u16; 1024];
                Delta::delta(&transposed, &chunk.bases, &mut deltas);
                let mut packed = [0u16; 64];
                BitPacking::pack::<1, 64>(&deltas, &mut packed);
                assert_eq!(packed, chunk.run_starts);
                chunk
            })
            .collect()
    }

    /// Current path with canonical `u16` indices: gather only.
    #[divan::bench(consts = MEAN_RUNS)]
    fn rle_gather<const MEAN_RUN: usize>(bencher: Bencher) {
        let chunks = encode_chunks(&generate(MEAN_RUN));
        let mut output = vec![0u32; N];
        with_counter!(bencher, N * size_of::<u32>()).bench_local(|| {
            for (chunk, out) in chunks.iter().zip(output.as_chunks_mut::<1024>().0) {
                unsafe {
                    u32::decode_unchecked(
                        black_box(&chunk.values[..chunk.num_values]),
                        black_box(&chunk.indices),
                        out,
                    );
                }
            }
            black_box(&output);
        });
    }

    /// Current path with bit-packed indices: unpack to `u16`, then gather.
    #[divan::bench(consts = MEAN_RUNS)]
    fn rle_unpack_gather<const MEAN_RUN: usize>(bencher: Bencher) {
        let chunks = encode_chunks(&generate(MEAN_RUN));
        let mut output = vec![0u32; N];
        with_counter!(bencher, N * size_of::<u32>()).bench_local(|| {
            let mut indices = [0u16; 1024];
            for (chunk, out) in chunks.iter().zip(output.as_chunks_mut::<1024>().0) {
                unsafe {
                    BitPacking::unchecked_unpack(
                        chunk.index_width,
                        black_box(&chunk.packed_indices[..64 * chunk.index_width]),
                        &mut indices,
                    );
                    u32::decode_unchecked(&chunk.values[..chunk.num_values], &indices, out);
                }
            }
            black_box(&output);
        });
    }

    /// Fused 1-bit unpack + undelta (stored bases) + gather.
    #[divan::bench(consts = MEAN_RUNS)]
    fn rle_delta_packed_fused<const MEAN_RUN: usize>(bencher: Bencher) {
        let chunks = encode_chunks(&generate(MEAN_RUN));
        let mut output = vec![0u32; N];
        with_counter!(bencher, N * size_of::<u32>()).bench_local(|| {
            for (chunk, out) in chunks.iter().zip(output.as_chunks_mut::<1024>().0) {
                unsafe {
                    u32::decode_delta_packed_unchecked::<1, 64>(
                        black_box(&chunk.values[..chunk.num_values]),
                        black_box(&chunk.run_starts),
                        black_box(&chunk.bases),
                        out,
                    );
                }
            }
            black_box(&output);
        });
    }

    /// Run-start bitmap: the same bits, with bases derived by popcount instead of stored.
    #[divan::bench(consts = MEAN_RUNS)]
    fn rle_run_bitmap<const MEAN_RUN: usize>(bencher: Bencher) {
        let chunks = encode_chunks(&generate(MEAN_RUN));
        let mut output = vec![0u32; N];
        with_counter!(bencher, N * size_of::<u32>()).bench_local(|| {
            for (chunk, out) in chunks.iter().zip(output.as_chunks_mut::<1024>().0) {
                unsafe {
                    u32::decode_run_bitmap_unchecked(
                        black_box(&chunk.values[..chunk.num_values]),
                        black_box(&chunk.run_starts),
                        out,
                    );
                }
            }
            black_box(&output);
        });
    }

    /// Run-end decode over the whole array with canonical `u32` ends, mirroring Vortex's
    /// `RunWriter`: short runs write a whole fixed-size block, the next run overwrites the excess.
    #[divan::bench(consts = MEAN_RUNS)]
    fn runend<const MEAN_RUN: usize>(bencher: Bencher) {
        const BLOCK: usize = 8;
        let data = generate(MEAN_RUN);
        let mut ends = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in data.iter().enumerate() {
            if i > 0 && v != data[i - 1] {
                ends.push(i as u32);
                values.push(data[i - 1]);
            }
        }
        ends.push(N as u32);
        values.push(data[N - 1]);

        let mut output = vec![0u32; N + BLOCK];
        with_counter!(bencher, N * size_of::<u32>()).bench_local(|| {
            let mut start = 0usize;
            let dst = output.as_mut_ptr();
            for (&end, &value) in black_box(&ends).iter().zip(black_box(&values)) {
                let end = end as usize;
                assert!(start <= end && end <= N);
                unsafe {
                    let dst = dst.add(start);
                    let run_len = end - start;
                    if run_len <= BLOCK {
                        for i in 0..BLOCK {
                            dst.add(i).write(value);
                        }
                    } else {
                        for i in 0..run_len {
                            dst.add(i).write(value);
                        }
                    }
                }
                start = end;
            }
            black_box(&output);
        });
    }
}
