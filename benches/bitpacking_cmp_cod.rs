#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// TODO(joe): remove this once codspeed supports const generics.

use divan::Bencher;
use fastlanes::{BitPacking, BitPackingCompare};
use num_traits::FromPrimitive;

fn main() {
    divan::main();
}

#[divan::bench(types=[u16, u32, u64])]
fn bitpacking_cmp_fused<T: BitPacking + BitPackingCompare + FromPrimitive + Copy>(bencher: Bencher)
where
    [(); 128 * 3 / size_of::<T>()]:,
{
    const W: usize = 3;
    let value = T::from_usize(1).expect("");
    let values = [T::from_usize(2).expect(""); 1024];
    let mut packed = [T::zero(); 128 * 3 / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values, &mut packed) };

    let mut unpacked = [false; 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack_cmp(W, &packed, &mut unpacked, |a, b| a == b, value) };
    });
}

#[divan::bench(types=[u16, u32, u64])]
fn bitpacking_cmp_seq<T: BitPacking + FromPrimitive + Copy>(bencher: Bencher)
where
    [(); 128 * 3 / size_of::<T>()]:,
{
    const W: usize = 3;
    let value = T::from_usize(1).expect("");
    let values = [T::from_usize(2).expect(""); 1024];
    let mut packed = [T::zero(); 128 * 3 / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values, &mut packed) };

    let mut unpacked = [T::zero(); 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack(W, &packed, &mut unpacked) };
        collect_bool_cmp(&unpacked, &value)
    });
}

#[divan::bench(types=[u16, u32, u64])]
fn bitpacking_cmp_unpack<T: BitPacking + FromPrimitive + Copy>(bencher: Bencher)
where
    [(); 128 * 3 / size_of::<T>()]:,
{
    const W: usize = 3;
    let values = [T::from_usize(2).expect(""); 1024];
    let mut packed = [T::zero(); 128 * W / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values, &mut packed) };

    let mut unpacked = [T::zero(); 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack(W, &packed, &mut unpacked) };
    });
}

#[inline(never)]
#[must_use]
pub fn collect_bool_cmp<T: PartialEq + Copy>(unpacked: &[T; 1024], cmp: &T) -> Vec<u64> {
    collect_bool(unpacked.len(), |idx| unpacked[idx] == *cmp)
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
