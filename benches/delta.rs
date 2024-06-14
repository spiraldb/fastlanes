#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use arrayref::array_ref;
use std::any::type_name;
use std::fmt::Debug;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};
use num_traits::Bounded;

use fastlanes::{Delta, FastLanes};

fn delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta");
    delta_typed::<u8>(&mut group);
    delta_typed::<u16>(&mut group);
    delta_typed::<u32>(&mut group);
    delta_typed::<u64>(&mut group);
}

fn delta_typed<T: Delta + Bounded + Debug>(group: &mut BenchmarkGroup<WallTime>)
where
    [(); T::LANES]:,
{
    // We heap allocate the values to avoid Rust evaluating the Delta functions at compile time!
    let mut values = Vec::with_capacity(1024);
    for i in 0..1024 {
        values.push(T::from(i % <T>::max_value().to_usize().unwrap()).unwrap());
    }
    let base: [T; <T as FastLanes>::LANES] = [T::zero(); <T as FastLanes>::LANES];
    let mut output: [T; 1024] = [T::zero(); 1024];

    group.bench_function(format!("delta encode {}", type_name::<T>()), |b| {
        b.iter(|| Delta::encode(array_ref![values, 0, 1024], &base, &mut output));
    });

    group.bench_function(format!("delta decode {}", type_name::<T>()), |b| {
        b.iter(|| Delta::decode(array_ref![values, 0, 1024], &base, &mut output));
    });
}

criterion_group!(benches, delta);
criterion_main!(benches);
