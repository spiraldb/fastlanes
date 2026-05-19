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

    #[divan::bench(types=[u8, u16, u32, u64], args=ALL_WIDTHS)]
    fn bitpacking_cmp_fused<T>(bencher: Bencher, width: usize)
    where
        T: BitPacking + FastLanesComparable<Bitpacked = T> + FromPrimitive + Copy,
        T: BitPacking + BitPackingCompare + Copy,
    {
        if width >= T::T {
            return;
        }

        bencher
            .with_inputs(|| {
                let value = T::from_usize(1).expect("");
                let values = [T::from_usize(2).expect(""); 1024];
                let mut packed = vec![T::zero(); 128 * width / size_of::<T>()];

                unsafe { BitPacking::unchecked_pack(width, &values, &mut packed) };

                FusedInput {
                    value,
                    packed,
                    output: [0u64; 16],
                }
            })
            .bench_refs(|input| {
                unsafe {
                    BitPackingCompare::unchecked_unpack_cmp(
                        black_box(width),
                        black_box(input.packed.as_slice()),
                        &mut input.output,
                        |a, b| a == b,
                        black_box(input.value),
                    );
                }
                black_box(&input.output);
            });
    }

    // Baseline for the same logical operation as `bitpacking_cmp_fused`, but not
    // a single crate API. This combines the exposed `BitPacking::unchecked_unpack`
    // entry point with a benchmark-local helper that compares unpacked values and
    // packs the results into the same `[u64; 16]` mask format as the fused API.
    #[divan::bench(types=[u8, u16, u32, u64], consts = BENCH_W, sample_count = 10000)]
    fn bitpacking_cmp_seq<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FromPrimitive + Copy,
    {
        bencher
            .with_inputs(|| {
                let value = T::from_usize(1).expect("");
                let values = [T::from_usize(2).expect(""); 1024];
                let mut packed = vec![T::zero(); 128 * W / size_of::<T>()];

                unsafe { T::unchecked_pack(W, &values, &mut packed) };

                SeqInput {
                    value,
                    packed,
                    unpacked: [T::zero(); 1024],
                    bools: [0u64; 16],
                }
            })
            .bench_refs(|input| {
                unsafe {
                    T::unchecked_unpack(
                        black_box(W),
                        black_box(input.packed.as_slice()),
                        &mut input.unpacked,
                    )
                };
                collect_bool_cmp(
                    &input.unpacked,
                    &black_box(input.value),
                    black_box(&mut input.bools),
                );
                black_box(&input.bools);
            });
    }

    // Measures the unpack API exposed by the crate:
    // `BitPacking::unchecked_unpack`.
    #[divan::bench(types=[u8, u16, u32, u64], consts = BENCH_W, sample_count = 10000)]
    fn bitpacking_cmp_unpack<T, const W: usize>(bencher: Bencher)
    where
        T: BitPacking + FromPrimitive + Copy,
    {
        bencher
            .with_inputs(|| {
                let values = [T::from_usize(2).expect(""); 1024];
                let mut packed = vec![T::zero(); 128 * W / size_of::<T>()];

                unsafe { T::unchecked_pack(W, &values, &mut packed) };

                UnpackInput {
                    packed,
                    unpacked: [T::zero(); 1024],
                }
            })
            .bench_refs(|input| {
                unsafe {
                    T::unchecked_unpack(
                        black_box(W),
                        black_box(input.packed.as_slice()),
                        &mut input.unpacked,
                    )
                };
                black_box(&input.unpacked);
            });
    }

    struct FusedInput<T> {
        value: T,
        packed: Vec<T>,
        output: [u64; 16],
    }

    struct SeqInput<T> {
        value: T,
        packed: Vec<T>,
        unpacked: [T; 1024],
        bools: [u64; 16],
    }

    struct UnpackInput<T> {
        packed: Vec<T>,
        unpacked: [T; 1024],
    }

    // Benchmark-only helper for `bitpacking_cmp_seq`.
    // This stays out of the crate because there is no public API for "compare an
    // already-unpacked block and pack the predicates into a mask"; its purpose is
    // to provide a staged baseline against `bitpacking_cmp_fused`.
    pub fn collect_bool_cmp<T: PartialEq + Copy>(
        unpacked: &[T; 1024],
        cmp: &T,
        output: &mut [u64; 16],
    ) {
        output.fill(0);

        for idx in 0..1024 {
            output[idx / 64] |= u64::from(unpacked[idx] == *cmp) << (idx % 64);
        }
    }
}
