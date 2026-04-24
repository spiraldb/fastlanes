use crate::seq_t;
use crate::unpack;
use crate::{supported_bit_width, FastLanes, FastLanesComparable};
use paste::paste;

pub trait BitPackingCompare: FastLanes {
    /// A fused unpack (see `BitPacking::unpack`) and compare into a 1024-bit mask.
    /// This will compare using the comparison function all the packed values with a constant value,
    /// the values are of type `Self`, whereas the comparison is on the type `V` (where `V::Bitpacked` = `Self`).
    /// This allows for comparison between signed values which are bitpacked as unsigned ones.
    fn unpack_cmp<const W: usize, const B: usize, V, F>(
        input: &[Self; B],
        output: &mut [u64; 16],
        comparison: F,
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;

    /// A fused unpack (see `BitPacking::unpack`) and compare into a 1024-bit mask.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length `[u64; 16]` (`1024` bits).
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp<V, F>(
        width: usize,
        input: &[Self],
        output: &mut [u64; 16],
        comparison: F,
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;
}

macro_rules! impl_packing_compare {
    ($T:ty) => {
        impl BitPackingCompare for $T {
            #[inline(never)]
            fn unpack_cmp<const W: usize, const B: usize, V, F>(
                input: &[Self; B],
                output: &mut [u64; 16],
                f: F,
                other: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self>,
                F: Fn(V, V) -> bool
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                if <$T>::T == 8 {
                    // u8 (LANES == 128). Split 128 lanes into two halves of 64
                    // so `output[2*row + half]` is loop-invariant per half —
                    // the condition LLVM's vectorizer needs to widen the
                    // 64-lane reduction.
                    let mut output_local = [0u64; 16];

                    for half in 0..2usize {
                        let mut row_words = [0u64; 8];

                        for lane_in_half in 0..64usize {
                            let lane = half * 64 + lane_in_half;
                            unpack!($T, W, input, lane, |$idx, $elem| {
                                let row = $idx / 128;
                                let pred = u64::from(f(V::as_unpacked($elem), other));
                                row_words[row] |= pred << lane_in_half;
                            });
                        }

                        for row in 0..8 {
                            output_local[2 * row + half] = row_words[row];
                        }
                    }

                    *output = output_local;
                } else {
                    output.fill(0);

                    for lane in 0..Self::LANES {
                        unpack!($T, W, input, lane, |$idx, $elem| {
                            unsafe { core::hint::assert_unchecked($idx < 1024); }
                            output[$idx / 64] |=
                                u64::from(f(V::as_unpacked($elem), other)) << ($idx % 64);
                        });
                    }
                }
            }

            unsafe fn unchecked_unpack_cmp<V, F>(
                 width: usize,
                 input: &[Self],
                 output: &mut [u64; 16],
                 comparison: F,
                 value: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self>,
                F: Fn(V, V) -> bool
            {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack_cmp::<W, B, V, F>(
                                unsafe { &*input.as_ptr().cast::<[Self; B]>() },
                                output,
                                comparison,
                                value
                            )
                        },)*
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }
        }
    };
}

impl_packing_compare!(u8);
impl_packing_compare!(u16);
impl_packing_compare!(u32);
impl_packing_compare!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitPacking;
    use core::array;
    use num_traits::FromPrimitive;

    fn assert_unpack_eq<T, const W: usize, const B: usize>()
    where
        T: BitPackingCompare + BitPacking + FastLanesComparable<Bitpacked = T> + FromPrimitive,
    {
        let values = array::from_fn(|i| T::from_usize(i % (1 << W)).unwrap());

        let mut packed = [T::zero(); B];
        T::pack::<W, B>(&values, &mut packed);

        // Values cycle through [0, min(1024, 1 << W)); cap the search range so
        // the test stays fast for larger W while still covering both match and
        // no-match cases.
        let search_limit = core::cmp::min(1usize << W, 2048);
        for v in 0..search_limit {
            let expected_value = T::from_usize(v).unwrap();
            let cmp = {
                let mut output = [0u64; 16];
                T::unpack_cmp::<W, B, _, _>(&packed, &mut output, |a, b| a == b, expected_value);
                output
            };

            let mut expected = [0u64; 16];
            for (idx, &value) in values.iter().enumerate() {
                expected[idx / 64] |= u64::from(value == expected_value) << (idx % 64);
            }

            assert_eq!(cmp, expected, "Failed == {v}");
        }
    }

    #[test]
    fn test_unpack_eq_u8() {
        type T = u8;
        const W: usize = 5;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
    }

    #[test]
    fn test_unpack_eq_u8_w7() {
        type T = u8;
        const W: usize = 7;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
    }

    #[test]
    fn test_unpack_eq_u32() {
        type T = u32;
        const W: usize = 10;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
    }

    #[test]
    fn test_unpack_eq_u16_w11() {
        type T = u16;
        const W: usize = 11;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
    }

    #[test]
    fn test_unpack_eq_u32_w31() {
        type T = u32;
        const W: usize = 31;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
    }

    #[test]
    fn test_unpack_eq_u64_w33() {
        type T = u64;
        const W: usize = 33;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
    }
}
