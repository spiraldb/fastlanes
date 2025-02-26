#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use divan::Bencher;
use fastlanes::{BitPacking, BitPackingCompare};
use std::hint::black_box;

fn main() {
    divan::main();
}

#[divan::bench]
fn bitpacking_cmp_u64_w3_fused(bencher: Bencher) {
    const W: usize = 3;
    type T = u64;
    let values = [2; 1024];
    let mut packed = [0; 128 * W / size_of::<T>()];
    T::pack::<W>(&values, &mut packed);

    let mut unpacked = [0u64; 16];
    bencher.bench_local(|| {
        T::unpack_cmp::<W, _>(&packed, &mut unpacked, |a, b| a == b, 1);
        black_box(());
    });
}

#[divan::bench]
fn bitpacking_cmp_u64_w3_seq(bencher: Bencher) {
    const W: usize = 3;
    type T = u64;
    let values = [2; 1024];
    let mut packed = [0; 128 * W / size_of::<T>()];
    T::pack::<W>(&values, &mut packed);

    let mut unpacked = [0; 1024];

    bencher.bench_local(|| {
        T::unpack::<W>(&packed, &mut unpacked);
        criterion::black_box(collect_bool_cmp(unpacked, 1));
    });
}

#[divan::bench]
fn bitpacking_cmp_u64_w15_fused(bencher: Bencher) {
    const W: usize = 15;
    type T = u64;
    let values = [2; 1024];
    let mut packed = [0; 128 * W / size_of::<T>()];
    T::pack::<W>(&values, &mut packed);

    let mut unpacked = [0u64; 16];
    bencher.bench_local(|| {
        T::unpack_cmp::<W, _>(&packed, &mut unpacked, |a, b| a == b, 1);
        black_box(());
    });
}

#[divan::bench]
fn bitpacking_cmp_u64_w15_seq(bencher: Bencher) {
    const W: usize = 15;
    type T = u64;
    let values = [2; 1024];
    let mut packed = [0; 128 * W / size_of::<T>()];
    T::pack::<W>(&values, &mut packed);

    let mut unpacked = [0; 1024];

    bencher.bench_local(|| {
        T::unpack::<W>(&packed, &mut unpacked);
        criterion::black_box(collect_bool_cmp(unpacked, 1));
    });
}

#[divan::bench]
fn bitpacking_cmp_u32_w3_fused(bencher: Bencher) {
    const W: usize = 3;
    type T = u32;
    let values = [2; 1024];
    let mut packed = [0; 128 * W / size_of::<T>()];
    T::pack::<W>(&values, &mut packed);

    let mut unpacked = [0u64; 16];
    bencher.bench_local(|| {
        T::unpack_cmp::<W, _>(&packed, &mut unpacked, |a, b| a == b, 1);
        black_box(());
    });
}

#[divan::bench]
fn bitpacking_cmp_u32_w3_seq(bencher: Bencher) {
    const W: usize = 3;
    type T = u16;
    let values = [2; 1024];
    let mut packed = [0; 128 * W / size_of::<T>()];
    T::pack::<W>(&values, &mut packed);

    let mut unpacked = [0; 1024];

    bencher.bench_local(|| {
        T::unpack::<W>(&packed, &mut unpacked);
        criterion::black_box(collect_bool_cmp(unpacked, 1));
    });
}

#[allow(clippy::needless_pass_by_value)]
#[inline(never)]
pub fn collect_bool_cmp<T: PartialEq>(unpacked: [T; 1024], cmp: T) -> Vec<u64> {
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
