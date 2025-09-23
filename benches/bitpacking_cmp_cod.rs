// TODO(joe): remove this once codspeed supports const generics.

use divan::Bencher;
use fastlanes::{BitPacking, BitPackingCompare, FastLanesComparable};
use num_traits::FromPrimitive;
use std::hint::black_box;

fn main() {
    divan::main();
}

#[divan::bench(types=[u16, u32, u64])]
fn bitpacking_cmp_fused<T>(bencher: Bencher)
where
    T: BitPacking + FastLanesComparable<Bitpacked = T> + FromPrimitive + Copy,
    T::Bitpacked: BitPacking + BitPackingCompare + Copy,
{
    const W: usize = 3;
    let value = T::from_usize(1).expect("");
    let values = [T::from_usize(2).expect(""); 1024];
    let mut packed = vec![T::zero(); 128 * 3 / size_of::<T>()];

    unsafe { BitPacking::unchecked_pack(W, &values, &mut packed) };

    let mut unpacked = [false; 1024];

    bencher.bench_local(|| {
        unsafe {
            BitPackingCompare::unchecked_unpack_cmp(
                W,
                black_box(&packed),
                &mut unpacked,
                |a, b| a == b,
                black_box(value),
            );
            black_box(unpacked);
        };
    });
}

#[divan::bench(types=[u16, u32, u64], sample_count = 10000)]
fn bitpacking_cmp_seq<T: BitPacking + FromPrimitive + Copy>(bencher: Bencher) {
    const W: usize = 3;
    let value = T::from_usize(1).expect("");
    let values = [T::from_usize(2).expect(""); 1024];
    let mut packed = vec![T::zero(); 128 * 3 / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values, &mut packed) };

    let mut unpacked = [T::zero(); 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack(W, black_box(&packed), &mut unpacked) };
        black_box(collect_bool_cmp(&unpacked, black_box(&value)))
    });
}

#[divan::bench(types=[u16, u32, u64], sample_count = 10000)]
fn bitpacking_cmp_unpack<T: BitPacking + FromPrimitive + Copy>(bencher: Bencher) {
    const W: usize = 3;
    let values = [T::from_usize(2).expect(""); 1024];
    let mut packed = vec![T::zero(); 128 * W / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values, &mut packed) };

    let mut unpacked = [T::zero(); 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack(W, black_box(&packed), &mut unpacked) };
        black_box(unpacked);
    });
}

#[inline(never)]
#[must_use]
pub fn collect_bool_cmp<T: PartialEq + Copy>(unpacked: &[T; 1024], cmp: &T) -> Vec<u64> {
    collect_bool(unpacked.len(), |idx| unpacked[idx] == *cmp)
}

#[inline]
pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, mut f: F) -> Vec<u64> {
    let mut buffer = Vec::with_capacity(len.div_ceil(64) * 8);

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

    buffer.truncate(len.div_ceil(8));
    buffer
}
