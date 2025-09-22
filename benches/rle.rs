use divan::{counter::BytesCount, Bencher};
use fastlanes::RLE;
use std::hint::black_box;
use std::mem::size_of;

fn main() {
    divan::main();
}

#[divan::bench]
fn rle_encode_u32(bencher: Bencher) {
    let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);

    bencher
        .with_inputs(|| {
            let rle_vals = [0u32; 1024];
            let rle_idxs = [0u16; 1024];
            (rle_vals, rle_idxs)
        })
        .bench_local_refs(|(rle_vals, rle_idxs)| {
            let unique_count = u32::encode(black_box(&input), rle_vals, rle_idxs);
            black_box(unique_count);
        });
}

#[divan::bench]
fn rle_decode_u32(bencher: Bencher) {
    let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
    let mut rle_vals = [0u32; 1024];
    let mut rle_idxs = [0u16; 1024];
    let unique_count = u32::encode(&input, &mut rle_vals, &mut rle_idxs);

    bencher
        .with_inputs(|| [0u32; 1024])
        .bench_local_refs(|output| {
            u32::decode(
                black_box(&rle_vals[..unique_count]),
                black_box(&rle_idxs),
                output,
            );
            black_box(*output);
        });
}

#[divan::bench]
fn rle_throughput_encode_32(bencher: Bencher) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    bencher
        .counter(BytesCount::new(N * size_of::<u32>()))
        .with_inputs(|| {
            let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();
            input_data
        })
        .bench_local_refs(|input_data| {
            for batch in 0..NUM_BATCHES {
                let batch_start = batch * 1024;
                let input_batch: [u32; 1024] = std::array::from_fn(|i| input_data[batch_start + i]);

                let mut rle_vals = [0u32; 1024];
                let mut rle_idxs = [0u16; 1024];

                let unique_count =
                    u32::encode(black_box(&input_batch), &mut rle_vals, &mut rle_idxs);
                black_box(unique_count);
            }
        });
}

#[divan::bench]
fn rle_throughput_decode_32(bencher: Bencher) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    bencher
        .counter(BytesCount::new(N * size_of::<u32>()))
        .with_inputs(|| {
            let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();
            let mut encoded_batches = Vec::new();
            for batch in 0..NUM_BATCHES {
                let batch_start = batch * 1024;
                let input_batch: [u32; 1024] = std::array::from_fn(|i| input_data[batch_start + i]);

                let mut rle_vals = [0u32; 1024];
                let mut rle_idxs = [0u16; 1024];
                let unique_count = u32::encode(&input_batch, &mut rle_vals, &mut rle_idxs);
                encoded_batches.push((rle_vals, rle_idxs, unique_count));
            }
            encoded_batches
        })
        .bench_local_refs(|encoded_batches| {
            for (rle_vals, rle_idxs, unique_count) in encoded_batches {
                let mut output = [0u32; 1024];
                u32::decode(
                    black_box(&rle_vals[..*unique_count]),
                    black_box(rle_idxs),
                    &mut output,
                );
                black_box(output);
            }
        });
}
