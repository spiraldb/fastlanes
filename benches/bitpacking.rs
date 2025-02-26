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
            let values = [3u16; 1024];
            let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
            BitPacking::pack::<WIDTH>(&values, &mut packed);

            let mut unpacked = [0u16; 1024];
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
