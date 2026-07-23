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
