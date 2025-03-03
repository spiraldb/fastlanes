#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![allow(unexpected_cfgs)]

fn main() {
    divan::main();
}

#[cfg(not(codspeed))]
mod bench {
    use divan::Bencher;
    use fastlanes::{BitPacking, BitPackingCompare, FastLanesComparable};
    use num_traits::FromPrimitive;
    use std::hint::black_box;

    const BENCH_W: [usize; 6] = [2, 3, 5, 7, 15, 31];

    #[divan::bench(types=[u32, u64], consts = BENCH_W)]
    fn bitpacking_cmp_fused<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FastLanesComparable<Bitpacked = T> + FromPrimitive + Copy,
        T: BitPacking + BitPackingCompare + Copy,
        [(); 128 * W / size_of::<T>()]:,
    {
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = [T::zero(); 128 * W / size_of::<T>()];

        unsafe { BitPacking::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [false; 1024];

        bencher.bench_local(|| {
            unsafe {
                BitPackingCompare::unchecked_unpack_cmp(
                    W,
                    &packed,
                    &mut unpacked,
                    |a, b| a == b,
                    black_box(value),
                );
                black_box(unpacked)
            };
        });
    }

    #[divan::bench(types=[u32, u64], consts = BENCH_W)]
    fn bitpacking_cmp_seq<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FromPrimitive + Copy,
        [(); 128 * W / size_of::<T>()]:,
    {
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = [T::zero(); 128 * W / size_of::<T>()];

        unsafe { T::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [T::zero(); 1024];
        let mut bools = [0u64; 16];

        bencher.bench_local(|| {
            unsafe { T::unchecked_unpack(W, &packed, &mut unpacked) };
            collect_bool_cmp(&unpacked, &black_box(value), black_box(&mut bools));
            black_box(bools)
        });
    }

    #[divan::bench(types=[u32, u64], consts = BENCH_W)]
    fn bitpacking_cmp_unpack<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FromPrimitive + Copy,
        [(); 128 * W / size_of::<T>()]:,
    {
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = [T::zero(); 128 * W / size_of::<T>()];

        unsafe { T::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [T::zero(); 1024];

        bencher.bench_local(|| {
            unsafe { T::unchecked_unpack(W, &packed, &mut unpacked) };
        });
    }

    pub fn collect_bool_cmp<T: PartialEq + Copy>(
        unpacked: &[T; 1024],
        cmp: &T,
        output: &mut [u64; 16],
    ) {
        collect_bool(|idx| unpacked[idx] == *cmp, output);
    }

    #[inline]
    pub fn collect_bool<F: FnMut(usize) -> bool>(mut f: F, output: &mut [u64; 16]) {
        for chunk in 0..16 {
            let mut packed = 0;
            for bit_idx in 0..64 {
                let i = bit_idx + chunk * 64;
                packed |= u64::from(f(i)) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            output[chunk] = packed;
        }
    }
}
