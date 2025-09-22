use arrayref::{array_mut_ref, array_ref};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use std::mem::size_of;

use fastlanes::{BitPacking, Delta, FastLanes, Transpose};

fn delta(c: &mut Criterion) {
    const W: usize = 9;
    const B: usize = 1024 * W / <u16 as FastLanes>::T;
    const LANES: usize = u16::LANES;

    let mut group = c.benchmark_group("delta");
    group.throughput(Throughput::Bytes(1024 * size_of::<u16>() as u64));
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

    group.bench_function("delta u16 fused", |b| {
        b.iter(|| {
            let mut unpacked = [0; 1024];
            Delta::undelta_pack::<LANES, W, B>(
                black_box(&packed),
                black_box(&[0; 64]),
                &mut unpacked,
            );
            black_box(unpacked);
        });
    });

    group.bench_function("delta u16 unfused", |b| {
        b.iter(|| {
            let mut unpacked = [0; 1024];
            BitPacking::unpack::<W, B>(black_box(&packed), &mut unpacked);
            let mut undelta = [0; 1024];
            Delta::undelta(black_box(&unpacked), black_box(&[0; 64]), &mut undelta);
            black_box(undelta);
        });
    });
}

fn throughput(c: &mut Criterion) {
    const WIDTH: usize = 3;
    const B: usize = 1024 * WIDTH / u16::T;
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;
    const OUTPUT_BATCH_SIZE: usize = 128 * WIDTH / size_of::<u16>();

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Bytes(N as u64 * size_of::<u16>() as u64));
    let mut values: Vec<u16> = (0..N).map(|i| (i % 8) as u16).collect();
    let mut packed = vec![0u16; NUM_BATCHES * OUTPUT_BATCH_SIZE];

    group.bench_function("compress", |b| {
        b.iter(|| {
            for i in 0..NUM_BATCHES {
                BitPacking::pack::<WIDTH, B>(
                    black_box(array_ref![values, i * 1024, 1024]),
                    array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
                );
            }
            black_box(&packed);
        });
    });

    group.bench_function("decompress", |b| {
        b.iter(|| {
            for i in 0..NUM_BATCHES {
                BitPacking::unpack::<WIDTH, B>(
                    black_box(array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE]),
                    array_mut_ref![values, i * 1024, 1024],
                );
            }
            black_box(&values);
        });
    });
}

criterion_group!(benches, delta, throughput);
criterion_main!(benches);
