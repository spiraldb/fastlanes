#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![recursion_limit = "10000"]

use divan::Bencher;
use fastlanes::test::collect_bool_cmp;
use fastlanes::{collect_byte_to_bit_16, collect_byte_to_bit_4, BitPacking, BitPackingCompare};
use num_traits::FromPrimitive;

fn main() {
    divan::main();
}

const BENCH_W: [usize; 4] = [2, 3, 5, 7];

#[divan::bench(types=[u16, u32, u64], consts = BENCH_W)]
fn bitpacking_cmp_fused<T: BitPacking + BitPackingCompare + FromPrimitive + Copy, const W: usize>(
    bencher: Bencher,
) where
    [(); 128 * W / size_of::<T>()]:,
{
    let value = T::from_usize(1).unwrap();
    let values = [T::from_usize(2).unwrap(); 1024];
    let mut packed = [T::zero(); 128 * W / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values.as_slice(), &mut packed.as_mut_slice()) };

    let mut unpacked = [false; 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack_cmp(W, &packed, &mut unpacked, |a, b| a == b, value) };
    });
}

#[divan::bench(types=[u16, u32, u64], consts = BENCH_W)]
fn bitpacking_cmp_seq<T: BitPacking + FromPrimitive + Copy, const W: usize>(bencher: Bencher)
where
    [(); 128 * W / size_of::<T>()]:,
{
    let value = T::from_usize(1).unwrap();
    let values = [T::from_usize(2).unwrap(); 1024];
    let mut packed = [T::zero(); 128 * W / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values.as_slice(), &mut packed.as_mut_slice()) };

    let mut unpacked = [T::zero(); 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack(W, &packed.as_slice(), &mut unpacked.as_mut_slice()) };
        collect_bool_cmp(&unpacked, &value)
    });
}

#[divan::bench(types=[u16, u32, u64], consts = BENCH_W)]
fn bitpacking_cmp_unpack<T: BitPacking + FromPrimitive + Copy, const W: usize>(bencher: Bencher)
where
    [(); 128 * W / size_of::<T>()]:,
{
    let values = [T::from_usize(2).unwrap(); 1024];
    let mut packed = [T::zero(); 128 * W / size_of::<T>()];

    unsafe { T::unchecked_pack(W, &values.as_slice(), &mut packed.as_mut_slice()) };

    let mut unpacked = [T::zero(); 1024];

    bencher.bench_local(|| {
        unsafe { T::unchecked_unpack(W, &packed.as_slice(), &mut unpacked.as_mut_slice()) };
    });
}
