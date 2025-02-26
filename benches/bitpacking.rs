#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::mem::size_of;

use arrayref::{array_mut_ref, array_ref};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fastlanes::{BitPacking, BitPackingCompare};

#[allow(clippy::too_many_lines)]
fn pack(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("pack");
        group.bench_function("pack 16 -> 3 heap", |b| {
            const WIDTH: usize = 3;
            let values = vec![3u16; 1024];
            let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];

            b.iter(|| {
                BitPacking::pack::<WIDTH>(
                    array_ref![values, 0, 1024],
                    array_mut_ref![packed, 0, 192],
                );
            });
        });

        group.bench_function("pack 16 -> 3 stack", |b| {
            const WIDTH: usize = 3;
            let values = [3u16; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
            b.iter(|| BitPacking::pack::<WIDTH>(&values, &mut packed));
        });
    }

    {
        let mut group = c.benchmark_group("unpack");
        group.bench_function("unpack 16 <- 3 stack", |b| {
            const WIDTH: usize = 3;
            let values = [3u64; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u64>()];
            BitPacking::pack::<WIDTH>(&values, &mut packed);

            let mut unpacked = [0u64; 1024];
            b.iter(|| BitPacking::unpack::<WIDTH>(&packed, &mut unpacked));
        });
    }

    {
        let mut group = c.benchmark_group("unpack-single");
        group.bench_function("unpack single 16 <- 3", |b| {
            const WIDTH: usize = 3;
            let values = vec![3u16; 1024];
            let mut packed = vec![0; 128 * WIDTH / size_of::<u16>()];
            BitPacking::pack::<WIDTH>(array_ref![values, 0, 1024], array_mut_ref![packed, 0, 192]);

            b.iter(|| {
                for i in 0..1024 {
                    black_box::<u16>(BitPacking::unpack_single::<WIDTH>(
                        array_ref![packed, 0, 192],
                        i,
                    ));
                }
            });
        });
    }

    {
        let mut group = c.benchmark_group("unpack_eq_unpack");
        group.bench_function("16 <- 3 stack", |b| {
            const WIDTH: usize = 20;
            let values = [4u32; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u32>()];
            BitPacking::pack::<WIDTH>(&values, &mut packed);

            let mut unpacked = [0u32; 1024];
            b.iter(|| {
                BitPacking::unpack::<WIDTH>(&packed, &mut unpacked);
                black_box(());
            });
        });
    }

    {
        let mut group = c.benchmark_group("unpack_eq_fused");
        group.bench_function("16 <- 3 stack", |b| {
            type BitPackingT = u32;
            const WIDTH: usize = 3;
            let values = [4; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<BitPackingT>()];
            BitPackingT::pack::<WIDTH>(&values, &mut packed);

            let mut unpacked = [0u64; 16];
            b.iter(|| {
                BitPackingT::unpack_cmp::<WIDTH, _>(&packed, &mut unpacked, |a, b| a == b, 1);
                black_box(());
            });
        });
    }

    {
        let mut group = c.benchmark_group("unpack_eq_collect");
        group.bench_function("16 <- 3 stack", |b| {
            const WIDTH: usize = 20;
            let values = [4u32; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u32>()];
            BitPacking::pack::<WIDTH>(&values, &mut packed);

            let mut unpacked = [0u32; 1024];
            BitPacking::unpack::<WIDTH>(&packed, &mut unpacked);
            black_box(());
            b.iter(|| black_box(collect_bool_cmp(unpacked, 1)));
        });
    }

    {
        let mut group = c.benchmark_group("unpack_eq_unpack_collect");
        group.bench_function("16 <- 3 stack", |b| {
            const WIDTH: usize = 20;
            let values = [4u32; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u32>()];
            BitPacking::pack::<WIDTH>(&values, &mut packed);

            let mut unpacked = [0u32; 1024];
            b.iter(|| {
                BitPacking::unpack::<WIDTH>(&packed, &mut unpacked);
                black_box(collect_bool_cmp(unpacked, 1));
            });
        });
    }
}

#[inline(never)]
#[must_use]
pub fn collect_bool_cmp(unpacked: [u32; 1024], cmp: u32) -> Vec<u64> {
    collect_bool(unpacked.len(), |idx| unpacked[idx] == cmp)
}

#[inline]
#[must_use]
pub fn ceil(value: usize, divisor: usize) -> usize {
    // Rewrite as `value.div_ceil(&divisor)` after
    // https://github.com/rust-lang/rust/issues/88581 is merged.
    value / divisor + usize::from(0 != value % divisor)
}

#[inline]
pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, mut f: F) -> Vec<u64> {
    let mut buffer = Vec::with_capacity(ceil(len, 64) * 8);

    let chunks = len / 64;
    let remainder = len % 64;
    for chunk in 0..chunks {
        let mut packed = 0;
        for bit_idx in 0..64 {
            let i = bit_idx + chunk * 64;
            packed |= u64::from(f(i)) << bit_idx;
        }

        // SAFETY: Already allocated sufficient capacity
        buffer.push(packed);
    }

    if remainder != 0 {
        let mut packed = 0;
        for bit_idx in 0..remainder {
            let i = bit_idx + chunks * 64;
            packed |= u64::from(f(i)) << bit_idx;
        }

        // SAFETY: Already allocated sufficient capacity
        buffer.push(packed);
    }

    buffer.truncate(ceil(len, 8));
    buffer
}

fn throughput(c: &mut Criterion) {
    const WIDTH: usize = 3;
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
                BitPacking::pack::<WIDTH>(
                    array_ref![values, i * 1024, 1024],
                    array_mut_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
                );
            }
        });
    });

    group.bench_function("decompress", |b| {
        b.iter(|| {
            for i in 0..NUM_BATCHES {
                BitPacking::unpack::<WIDTH>(
                    array_ref![packed, i * OUTPUT_BATCH_SIZE, OUTPUT_BATCH_SIZE],
                    array_mut_ref![values, i * 1024, 1024],
                );
            }
        });
    });
}

criterion_group!(benches, pack, throughput);
criterion_main!(benches);
