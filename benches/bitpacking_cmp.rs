#![allow(unexpected_cfgs)]

mod shared;

fn main() {
    divan::main();
}

mod bench {
    use divan::Bencher;
    use fastlanes::{BitPacking, BitPackingCompare, FastLanesComparable};
    use num_traits::FromPrimitive;
    use std::hint::black_box;

    const BENCH_W: [usize; 4] = [2, 3, 5, 7];

    const ALL_WIDTHS: [usize; 62] = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];

    #[divan::bench(name = crate::variant!("bitpacking_cmp_fused"), types=[u8, u16, u32, u64], args=ALL_WIDTHS)]
    fn bitpacking_cmp_fused<T>(bencher: Bencher, width: usize)
    where
        T: BitPacking + FastLanesComparable<Bitpacked = T> + FromPrimitive + Copy,
        T: BitPacking + BitPackingCompare + Copy,
    {
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = vec![T::zero(); 128 * width / size_of::<T>()];

        if width >= T::T {
            return;
        }

        unsafe { BitPacking::unchecked_pack(width, &values, &mut packed) };

        let mut unpacked = [false; 1024];

        bencher.bench_local(|| {
            unsafe {
                BitPackingCompare::unchecked_unpack_cmp(
                    black_box(width),
                    black_box(&packed),
                    &mut unpacked,
                    |a, b| a == b,
                    black_box(value),
                );
                black_box(&unpacked);
            };
        });
    }

    #[divan::bench(name = crate::variant!("bitpacking_cmp_seq"), types=[u8, u16, u32, u64], consts = BENCH_W, sample_count = 10000)]
    fn bitpacking_cmp_seq<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FromPrimitive + Copy,
    {
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = vec![T::zero(); 128 * W / size_of::<T>()];

        unsafe { T::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [T::zero(); 1024];
        let mut bools = [0u64; 16];

        bencher.bench_local(|| {
            unsafe { T::unchecked_unpack(black_box(W), black_box(&packed), &mut unpacked) };
            collect_bool_cmp(&unpacked, black_box(&value), black_box(&mut bools));
            black_box(&bools);
        });
    }

    #[divan::bench(name = crate::variant!("bitpacking_cmp_unpack"), types=[u8, u16, u32, u64], consts = BENCH_W, sample_count = 10000)]
    fn bitpacking_cmp_unpack<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FromPrimitive + Copy,
    {
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = vec![T::zero(); 128 * W / size_of::<T>()];

        unsafe { T::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [T::zero(); 1024];

        bencher.bench_local(|| {
            unsafe { T::unchecked_unpack(black_box(W), black_box(&packed), &mut unpacked) };
            black_box(&unpacked);
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
