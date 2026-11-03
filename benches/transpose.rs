use divan::Bencher;
use std::hint::black_box;

use fastlanes::{FastLanes, Transpose};

fn main() {
    divan::main();
}

fn values<T: FastLanes + TryFrom<usize>>() -> [T; 1024] {
    core::array::from_fn(|i| T::try_from(i % 256).ok().unwrap_or_else(T::zero))
}

#[divan::bench(types = [u8, u16, u32, u64])]
fn transpose<T: Transpose + TryFrom<usize>>(bencher: Bencher) {
    let values = values::<T>();
    let mut output = [T::zero(); 1024];

    bencher.bench_local(|| {
        T::transpose(black_box(&values), &mut output);
        black_box(&output);
    });
}

#[divan::bench(types = [u8, u16, u32, u64])]
fn untranspose<T: Transpose + TryFrom<usize>>(bencher: Bencher) {
    let values = values::<T>();
    let mut output = [T::zero(); 1024];

    bencher.bench_local(|| {
        T::untranspose(black_box(&values), &mut output);
        black_box(&output);
    });
}
