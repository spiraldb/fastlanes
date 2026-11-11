#![allow(unused_assignments)]

use crate::{BitPacking, FastLanes, iterate, seq_t, supported_bit_width, unpack};
use core::mem::size_of;
use pastey::paste;

pub trait Delta: BitPacking {
    fn delta<const LANES: usize>(
        input: &[Self; 1024],
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    fn undelta<const LANES: usize>(
        input: &[Self; 1024],
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    fn undelta_pack<const LANES: usize, const W: usize, const B: usize>(
        input: &[Self; B],
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    /// Fused unpack and undelta of 1024 elements packed into `W` bits each, where `W` is
    /// runtime-known instead of compile-time known.
    ///
    /// The output is in the transposed order the deltas were computed in, like
    /// [`Delta::undelta`].
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    ///
    /// # Panics
    /// Panics if `width` is greater than the bit-width of `Self`.
    unsafe fn unchecked_undelta_pack<const LANES: usize>(
        width: usize,
        input: &[Self],
        base: &[Self; LANES],
        output: &mut [Self],
    );
}

macro_rules! impl_delta {
    ($T:ty) => {
        impl Delta for $T {
            #[inline(never)]
            fn delta<const LANES: usize>(
                input: &[Self; 1024],
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                }

                for lane in 0..Self::LANES {
                    let mut prev = base[lane];
                    iterate!($T, lane, |$idx| {
                        let next = input[$idx];
                        output[$idx] = next.wrapping_sub(prev);
                        prev = next;
                    });
                }
            }

            #[inline(never)]
            fn undelta<const LANES: usize>(
                input: &[Self; 1024],
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                }
                for lane in 0..LANES {
                    let mut prev = base[lane];
                    iterate!($T, lane, |$idx| {
                        let next = input[$idx].wrapping_add(prev);
                        output[$idx] = next;
                        prev = next;
                    });
                }
            }

            #[inline(never)]
            fn undelta_pack<const LANES: usize, const W: usize, const B: usize>(
                input: &[Self; B],
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in 0..Self::LANES {
                    let mut prev = base[lane];
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let next = $elem.wrapping_add(prev);
                        output[$idx] = next;
                        prev = next;
                    });
                }
            }

            unsafe fn unchecked_undelta_pack<const LANES: usize>(
                width: usize,
                input: &[Self],
                base: &[Self; LANES],
                output: &mut [Self],
            ) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(
                    input.len(),
                    packed_len,
                    "Input buffer must be of size 1024 * W / T"
                );
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::undelta_pack::<LANES, W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                base,
                                unsafe { crate::as_array_mut_unchecked(output) },
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::undelta_pack::<LANES, W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                base,
                                unsafe { crate::as_array_mut_unchecked(output) },
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }
        }
    };
}

impl_delta!(u8);
impl_delta!(u16);
impl_delta!(u32);
impl_delta!(u64);

#[cfg(test)]
mod test {
    use super::*;
    use crate::Transpose;
    use alloc::{format, string::ToString, vec};
    use core::fmt::Debug;
    use core::mem::size_of;
    use hegel::generators as gs;
    use hegel::generators::Integer;
    use hegel::{PrettyPrintable, TestCase};
    use num_traits::WrappingSub;
    use pastey::paste;

    #[test]
    fn test_delta() {
        const LANES: usize = u16::LANES;
        const W: usize = 15;
        const B: usize = 1024 * W / u16::T;

        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i / 8) as u16;
        }

        let mut transposed = [0; 1024];
        Transpose::transpose(&values, &mut transposed);

        let mut deltas = [0; 1024];
        Delta::delta(&transposed, &[0; 64], &mut deltas);

        let mut packed = [0; 128 * W / size_of::<u16>()];
        BitPacking::pack::<W, B>(&deltas, &mut packed);

        // Fused kernel
        let mut unpacked = [0; 1024];
        Delta::undelta_pack::<LANES, W, B>(&packed, &[0; 64], &mut unpacked);
        assert_eq!(transposed, unpacked);

        // Unfused kernel
        BitPacking::unpack::<W, B>(&packed, &mut unpacked);
        let mut undelta = [0; 1024];
        Delta::undelta(&unpacked, &[0; 64], &mut undelta);
        assert_eq!(transposed, undelta);
    }

    fn assert_delta_matches_reference<T, const LANES: usize>(tc: &TestCase)
    where
        T: Delta + Debug + Integer + PrettyPrintable + Send + Sync + WrappingSub + 'static,
    {
        let input: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let base: [T; LANES] = tc.draw(gs::arrays(gs::integers::<T>()));
        let mut expected = [T::max_value(); 1024];

        for lane in 0..LANES {
            let mut previous = base[lane];
            for row in 0..T::T {
                let order = row / 8;
                let sub_row = row % 8;
                let index = (crate::FL_ORDER[order] * 16) + (sub_row * 128) + lane;
                expected[index] = input[index].wrapping_sub(&previous);
                previous = input[index];
            }
        }

        let mut actual = [T::max_value(); 1024];
        T::delta::<LANES>(&input, &base, &mut actual);

        assert_eq!(actual, expected);
    }

    fn assert_delta_roundtrip<T, const LANES: usize>(tc: &TestCase)
    where
        T: Delta + Debug + Integer + PrettyPrintable + Send + Sync + 'static,
    {
        let input: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let base: [T; LANES] = tc.draw(gs::arrays(gs::integers::<T>()));
        let mut deltas = [T::max_value(); 1024];
        let mut output = [T::max_value(); 1024];

        T::delta::<LANES>(&input, &base, &mut deltas);
        T::undelta::<LANES>(&deltas, &base, &mut output);

        assert_eq!(output, input);
    }

    fn assert_undelta_pack_matches_unfused<T, const LANES: usize>(tc: &TestCase)
    where
        T: Delta + Debug + Integer + PrettyPrintable + Send + Sync + 'static,
    {
        let deltas: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let base: [T; LANES] = tc.draw(gs::arrays(gs::integers::<T>()));

        for width in 0..=T::T {
            let packed_len = 1024 * width / T::T;
            let mut packed = vec![T::max_value(); packed_len];
            unsafe { T::unchecked_pack(width, &deltas, &mut packed) };

            let mut unpacked = [T::max_value(); 1024];
            let mut expected = [T::max_value(); 1024];
            unsafe { T::unchecked_unpack(width, &packed, &mut unpacked) };
            T::undelta::<LANES>(&unpacked, &base, &mut expected);

            let mut actual = [T::max_value(); 1024];
            unsafe { T::unchecked_undelta_pack(width, &packed, &base, &mut actual) };
            assert_eq!(actual, expected);
        }
    }

    macro_rules! delta_property_tests {
        ($T:ident, $lanes:expr) => {
            paste! {
                #[hegel::test]
                fn [<test_delta_matches_reference_ $T>](tc: TestCase) {
                    assert_delta_matches_reference::<$T, $lanes>(&tc);
                }

                #[hegel::test]
                fn [<test_delta_roundtrip_ $T>](tc: TestCase) {
                    assert_delta_roundtrip::<$T, $lanes>(&tc);
                }

                #[hegel::test(test_cases = 10)]
                fn [<test_undelta_pack_matches_unfused_ $T>](tc: TestCase) {
                    assert_undelta_pack_matches_unfused::<$T, $lanes>(&tc);
                }
            }
        };
    }

    delta_property_tests!(u8, 128);
    delta_property_tests!(u16, 64);
    delta_property_tests!(u32, 32);
    delta_property_tests!(u64, 16);
}
