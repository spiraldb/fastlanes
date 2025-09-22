use std::mem::size_of;

use arrayref::{array_mut_ref, array_ref};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fastlanes::{BitPacking, FastLanes};
use std::hint::black_box;
//

fn pack(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("pack");
        group.bench_function("pack 16 -> 3 heap", |b| {
            const WIDTH: usize = 3;
            const B: usize = 1024 * WIDTH / u16::T;
            let values = vec![3u16; 1024];
            let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];

            b.iter(|| {
                BitPacking::pack::<WIDTH, B>(
                    black_box(array_ref![values, 0, 1024]),
                    array_mut_ref![packed, 0, 192],
                );
                black_box(&packed);
            });
        });

        group.bench_function("pack 16 -> 3 stack", |b| {
            const WIDTH: usize = 3;
            const B: usize = 1024 * WIDTH / u16::T;
            let values = [3u16; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
            b.iter(|| {
                BitPacking::pack::<WIDTH, B>(black_box(&values), &mut packed);
                black_box(packed);
            });
        });
    }

    {
        let mut group = c.benchmark_group("unpack");
        group.bench_function("unpack 16 <- 3 stack", |b| {
            const WIDTH: usize = 3;
            const B: usize = 1024 * WIDTH / u16::T;
            let values = [3u16; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
            BitPacking::pack::<WIDTH, B>(&values, &mut packed);

            let mut unpacked = [0u16; 1024];
            b.iter(|| {
                BitPacking::unpack::<WIDTH, B>(&black_box(packed), &mut unpacked);
                black_box(unpacked);
            });
        });
    }

    {
        let mut group = c.benchmark_group("unpack");
        group.bench_function("unchecked unpack 16 <- 3 stack", |b| {
            const WIDTH: usize = 3;
            const B: usize = 1024 * WIDTH / u16::T;
            let values = [3u16; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
            BitPacking::pack::<WIDTH, B>(&values, &mut packed);

            let mut unpacked = [0u16; 1024];
            b.iter(|| {
                unsafe { BitPacking::unchecked_unpack(WIDTH, &black_box(packed), &mut unpacked) };
                black_box(unpacked);
            });
        });
    }

    {
        let mut group = c.benchmark_group("unpack-single");
        group.bench_function("unpack single 16 <- 3", |b| {
            const WIDTH: usize = 3;
            const B: usize = 1024 * WIDTH / u16::T;
            let values = vec![3u16; 1024];
            let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];
            BitPacking::pack::<WIDTH, B>(
                array_ref![values, 0, 1024],
                array_mut_ref![packed, 0, 192],
            );

            b.iter(|| {
                for i in 0..1024 {
                    black_box::<u16>(BitPacking::unpack_single::<WIDTH, B>(
                        black_box(array_ref![packed, 0, 192]),
                        black_box(i),
                    ));
                }
            });
        });
    }
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

criterion_group!(benches, pack, throughput);
criterion_main!(benches);
