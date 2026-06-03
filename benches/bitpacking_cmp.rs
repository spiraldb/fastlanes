#![allow(unexpected_cfgs)]

fn main() {
    divan::main();
}

mod bench {
    use divan::Bencher;
    use fastlanes::{
        untranspose_bits, BitPacking, BitPackingCompare, FastLanes, FastLanesComparable,
    };
    use num_traits::FromPrimitive;
    use std::hint::black_box;

    const ALL_WIDTHS: [usize; 63] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];

    /// Baseline: unpack into a `[T; 1024]` scratch buffer, then compare into a `[u64; 16]` bitmask
    /// in logical row order (arrow-rs `collect_bool` style).
    #[divan::bench(types=[u8, u16, u32, u64], args=ALL_WIDTHS, sample_count = 2000)]
    fn cmp_seq<T>(bencher: Bencher, width: usize)
    where
        T: BitPacking + FromPrimitive + Copy + PartialEq,
    {
        if width >= T::T {
            return;
        }
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = vec![T::zero(); 128 * width / size_of::<T>()];
        unsafe { T::unchecked_pack(width, &values, &mut packed) };

        let mut unpacked = [T::zero(); 1024];
        let mut bools = [0u64; 16];

        bencher.bench_local(|| {
            unsafe { T::unchecked_unpack(black_box(width), black_box(&packed), &mut unpacked) };
            collect_bool_cmp(&unpacked, black_box(&value), black_box(&mut bools));
            black_box(&bools);
        });
    }

    /// Fused unpack+compare straight into a transposed `[u64; 16]` bitmask (`FastLanes` order). No
    /// intermediate buffer, no untranspose.
    #[divan::bench(types=[u8, u16, u32, u64], args=ALL_WIDTHS, sample_count = 2000)]
    fn cmp_fused_transposed<T>(bencher: Bencher, width: usize)
    where
        T: BitPacking
            + BitPackingCompare
            + FastLanesComparable<Bitpacked = T>
            + FromPrimitive
            + Copy,
    {
        if width >= T::T {
            return;
        }
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = vec![T::zero(); 128 * width / size_of::<T>()];
        unsafe { T::unchecked_pack(width, &values, &mut packed) };

        let mut output = [0u64; 16];

        bencher.bench_local(|| {
            unsafe {
                T::unchecked_unpack_cmp(
                    black_box(width),
                    black_box(&packed),
                    &mut output,
                    |a, b| a == b,
                    black_box(value),
                );
            }
            black_box(&output);
        });
    }

    /// Full drop-in for `cmp_seq`: fused compare into a transposed mask, then bit-untranspose into
    /// logical row order. Identical output to unpack-then-`collect_bool`.
    #[divan::bench(types=[u8, u16, u32, u64], args=ALL_WIDTHS, sample_count = 2000)]
    fn cmp_fused_untranspose<T>(bencher: Bencher, width: usize)
    where
        T: BitPacking
            + BitPackingCompare
            + FastLanes
            + FastLanesComparable<Bitpacked = T>
            + FromPrimitive
            + Copy,
    {
        if width >= T::T {
            return;
        }
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = vec![T::zero(); 128 * width / size_of::<T>()];
        unsafe { T::unchecked_pack(width, &values, &mut packed) };

        let mut transposed = [0u64; 16];
        let mut logical = [0u64; 16];

        bencher.bench_local(|| {
            unsafe {
                T::unchecked_unpack_cmp(
                    black_box(width),
                    black_box(&packed),
                    &mut transposed,
                    |a, b| a == b,
                    black_box(value),
                );
            }
            untranspose_bits::<T>(black_box(&transposed), &mut logical);
            black_box(&logical);
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
            output[chunk] = packed;
        }
    }
}
