#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::mem::size_of;

use criterion::{criterion_group, criterion_main, Criterion};

use fastlanes::{BitPacking, Transpose};

fn transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    group.bench_function("transpose u16", |b| {
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % u16::MAX as usize) as u16;
        }

        let mut transposed = [0; 1024];
        b.iter(|| Transpose::transpose(&values, &mut transposed));
    });
}

criterion_group!(benches, transpose);
criterion_main!(benches);
