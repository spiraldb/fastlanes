#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::mem::size_of;
use std::time::Duration;

use arrayref::{array_mut_ref, array_ref};
use criterion::{criterion_group, criterion_main, Criterion};
use fastlanes::BitPacking;

fn compute(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("compute");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(100);

        // Pick an n such that n * u32 > L1 cache.
        let n = 1024 * 16;

        const W: usize = 9;
        const PACKED: usize = 128 * W / size_of::<u32>();

        // Generate some u32 values that fit into W - 1 bits (so we have room to add a delta)
        let values = (0..(1024 * n))
            .map(|x| (x % (1 << (W - 1))) as u32)
            .collect::<Vec<_>>();

        // Pack them into W bits.
        let mut packed = vec![0u32; n * PACKED];
        for i in 0..n {
            BitPacking::pack::<W>(
                array_ref![values[i * 1024..], 0, 1024],
                array_mut_ref![
                    packed[i * 128 * W / size_of::<u32>()..][..PACKED],
                    0,
                    PACKED
                ],
            );
        }

        group.bench_function("unpack + add", |b| {
            let mut result = vec![0u32; values.len()];
            b.iter(|| {
                let mut buffer = [0u32; 1024];
                for i in 0..n {
                    BitPacking::unpack::<W>(
                        array_ref![&packed[i * PACKED..], 0, PACKED],
                        &mut buffer,
                    );
                    let result_buf = &mut result[i * 1024..];
                    for j in 0..1024 {
                        result_buf[j] = buffer[j] + 10;
                    }
                }
            });
        });

        group.bench_function("unpack + add + pack", |b| {
            let mut result = vec![0u32; packed.len()];
            b.iter(|| {
                let mut buffer = [0u32; 1024];
                for i in 0..n {
                    BitPacking::unpack::<W>(
                        array_ref![&packed[i * PACKED..], 0, PACKED],
                        &mut buffer,
                    );
                    buffer.iter_mut().for_each(|x| *x += 10);
                    BitPacking::pack::<W>(&buffer, array_mut_ref![result[i * PACKED..], 0, PACKED]);
                }
            });
        });
    }
}

criterion_group!(benches, compute);
criterion_main!(benches);
