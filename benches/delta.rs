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

    let mut deltas = [0; 1024];
    Delta::delta(&transposed, &[0; 64], &mut deltas);

    let mut packed = [0; 128 * W / size_of::<u16>()];
    BitPacking::bitpack::<W>(&deltas, &mut packed);

    group.bench_function("delta u16 fused", |b| {
        b.iter(|| {
            let mut unpacked = [0; 1024];
            Delta::undelta_bitpack::<W>(&packed, &[0; 64], &mut unpacked)
        });
    });

    group.bench_function("delta u16 unfused", |b| {
        b.iter(|| {
            let mut unpacked = [0; 1024];
            BitPacking::bitunpack::<W>(&packed, &mut unpacked);
            let mut undelta = [0; 1024];
            Delta::undelta(&unpacked, &[0; 64], &mut undelta);
        });
    });
}

criterion_group!(benches, delta);
criterion_main!(benches);
