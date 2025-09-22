use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fastlanes::RLE;
use std::hint::black_box;
use std::mem::size_of;

fn rle(c: &mut Criterion) {
    let mut group = c.benchmark_group("rle");

    group.bench_function("rle encode u32", |b| {
        let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];

        b.iter(|| {
            let unique_count = u32::encode(black_box(&input), &mut rle_vals, &mut rle_idxs);
            black_box(unique_count);
        });
    });

    group.bench_function("rle decode u32", |b| {
        let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];
        let unique_count = u32::encode(&input, &mut rle_vals, &mut rle_idxs);

        let mut output = [0u32; 1024];

        b.iter(|| {
            u32::decode(
                black_box(&rle_vals[..unique_count]),
                black_box(&rle_idxs),
                &mut output,
            );
            black_box(&output);
        });
    });
}

fn throughput(c: &mut Criterion) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    let mut group = c.benchmark_group("rle_throughput");
    group.throughput(Throughput::Bytes(N as u64 * size_of::<u32>() as u64));

    let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();

    group.bench_function("rle encode 32", |b| {
        b.iter(|| {
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
    });

    let mut encoded_batches = Vec::new();
    for batch in 0..NUM_BATCHES {
        let batch_start = batch * 1024;
        let input_batch: [u32; 1024] = std::array::from_fn(|i| input_data[batch_start + i]);

        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];
        let unique_count = u32::encode(&input_batch, &mut rle_vals, &mut rle_idxs);
        encoded_batches.push((rle_vals, rle_idxs, unique_count));
    }

    group.bench_function("rle decode 32", |b| {
        b.iter(|| {
            for (rle_vals, rle_idxs, unique_count) in &encoded_batches {
                let mut output = [0u32; 1024];
                u32::decode(
                    black_box(&rle_vals[..*unique_count]),
                    black_box(rle_idxs),
                    &mut output,
                );
                black_box(&output);
            }
        });
    });
}

criterion_group!(benches, rle, throughput);
criterion_main!(benches);
