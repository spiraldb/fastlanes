#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use criterion::{criterion_group, criterion_main, Criterion};
use std::mem::size_of;

use fastlanes::{BitPacking, Delta, Transpose};

fn delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta");

    const W: usize = 9;
    let mut values: [u16; 1024] = [0; 1024];
    for i in 0..1024 {
        values[i] = (i / 8) as u16;
    }

    let mut transposed = [0; 1024];
    Transpose::transpose(&values, &mut transposed);

    group.bench_function("delta u16 fused", |b| {
        b.iter(|| {
            let mut packed = [0; 128 * W / size_of::<u16>()];
            Delta::delta::<W>(&transposed, &[0; 64], &mut packed)
        });
    });

    group.bench_function("delta u16 unfused", |b| {
        b.iter(|| {
            let mut delta = [0; 1024];
            // Using width == 16 does not bit-packing
            Delta::delta::<16>(&transposed, &[0; 64], &mut delta);

            let mut packed = [0; 128 * W / size_of::<u16>()];
            BitPacking::bitpack::<W>(&delta, &mut packed);
        });
    });
}

criterion_group!(benches, delta);
criterion_main!(benches);
