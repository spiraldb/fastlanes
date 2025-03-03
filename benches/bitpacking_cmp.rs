#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![allow(unexpected_cfgs)]

fn main() {
    divan::main();
}

#[cfg(not(codspeed))]
mod bench {
    use divan::Bencher;
    use fastlanes::{BitPacking, BitPackingCompare};
    use num_traits::{FromPrimitive, Zero};

    const BENCH_W: [usize; 4] = [2, 3, 5, 7];

    #[divan::bench(types=[u16, u32, u64], consts = BENCH_W)]
    fn bitpacking_cmp_fused<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + BitPackingCompare + FromPrimitive + Copy,
        T::Bitpacked: BitPacking + BitPackingCompare + FromPrimitive + Copy,
        [(); 128 * W / size_of::<T>()]:,
    {
        let value = T::from_usize(1).expect("");
        let values = [T::Bitpacked::from_usize(2).expect(""); 1024];
        let mut packed = [T::Bitpacked::zero(); 128 * W / size_of::<T>()];

        unsafe { BitPacking::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [false; 1024];

        bencher.bench_local(|| {
            unsafe {
                BitPackingCompare::unchecked_unpack_cmp(
                    W,
                    &packed,
                    &mut unpacked,
                    |a, b| a == b,
                    value,
                )
            };
        });
    }

    #[divan::bench(types=[u16, u32, u64], consts = BENCH_W)]
    fn bitpacking_cmp_seq<T: BitPacking + FromPrimitive + Copy, const W: usize>(bencher: Bencher)
    where
        [(); 128 * W / size_of::<T>()]:,
    {
        let value = T::from_usize(1).expect("");
        let values = [T::from_usize(2).expect(""); 1024];
        let mut packed = [T::zero(); 128 * W / size_of::<T>()];

        unsafe { T::unchecked_pack(W, &values, &mut packed) };

        let mut unpacked = [T::zero(); 1024];

        bencher.bench_local(|| {
            unsafe { T::unchecked_unpack(W, &packed, &mut unpacked) };
            collect_bool_cmp(&unpacked, &value)
        });
    }

    #[divan::bench(types=[u16, u32, u64], consts = BENCH_W)]
    fn bitpacking_cmp_unpack<T: BitPacking + FromPrimitive + Copy, const W: usize>(bencher: Bencher)
    where
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

    pub fn collect_bool_cmp<T: PartialEq + Copy>(unpacked: &[T; 1024], cmp: &T) -> Vec<u64> {
        collect_bool(unpacked.len(), |idx| unpacked[idx] == *cmp)
    }

    #[inline]
    pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, mut f: F) -> Vec<u64> {
        let mut buffer = Vec::with_capacity(len.div_ceil(4) * 8);

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
}
