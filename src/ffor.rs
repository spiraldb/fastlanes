use crate::{BitPacking, FastLanes, pack, supported_bit_width, unpack, width_dispatch};
use pastey::paste;

pub trait FoR: BitPacking {
    fn for_pack<const W: usize, const B: usize>(
        input: &[Self; 1024],
        reference: Self,
        output: &mut [Self; B],
    );

    fn unfor_pack<const W: usize, const B: usize>(
        input: &[Self; B],
        reference: Self,
        output: &mut [Self; 1024],
    );

    /// Fused unpack and wrapping add a `FoR` reference value.
    /// Unpacks 1024 elements from `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    ///
    /// # Panics
    /// Panics if `width` is greater than the bit-width of `Self`.
    unsafe fn unchecked_unfor_pack(
        width: usize,
        input: &[Self],
        reference: Self,
        output: &mut [Self],
    );
}

macro_rules! impl_for {
    ($T:ident) => {
        paste! {
        /// # Safety
        /// `input` must hold `1024` elements and `output` `B == 1024 * W / T`.
        #[inline(always)]
        unsafe fn [<for_pack_ $T>]<const W: usize, const B: usize>(
            input: &[$T],
            reference: $T,
            output: &mut [$T],
        ) {
            const {
                assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                assert!(B == 1024 * W / <$T>::T);
            }
            // SAFETY: guaranteed by the caller.
            let input: &[$T; 1024] = unsafe { crate::as_array_unchecked(input) };
            let output: &mut [$T; B] = unsafe { crate::as_array_mut_unchecked(output) };

            for lane in 0..<$T>::LANES {
                pack!($T, W, output, lane, |$idx| {
                    input[$idx].wrapping_sub(reference)
                });
            }
        }

        /// # Safety
        /// `input` must hold `B == 1024 * W / T` elements and `output` `1024`.
        #[inline(always)]
        unsafe fn [<unfor_pack_ $T>]<const W: usize, const B: usize>(
            input: &[$T],
            reference: $T,
            output: &mut [$T],
        ) {
            const {
                assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                assert!(B == 1024 * W / <$T>::T);
            }
            // SAFETY: guaranteed by the caller.
            let input: &[$T; B] = unsafe { crate::as_array_unchecked(input) };
            let output: &mut [$T; 1024] = unsafe { crate::as_array_mut_unchecked(output) };

            for lane in 0..<$T>::LANES {
                unpack!($T, W, input, lane, |$idx, $elem| {
                    output[$idx] = $elem.wrapping_add(reference)
                });
            }
        }

        width_dispatch!(
            $T,
            [<unfor_pack_width_ $T>],
            [<unfor_pack_ $T>](input: &[$T], reference: $T, output: &mut [$T])
        );

        impl FoR for $T {
            fn for_pack<const W: usize, const B: usize>(
                input: &[Self; 1024],
                reference: Self,
                output: &mut [Self; B],
            ) {
                // SAFETY: the lengths are those of the arrays.
                unsafe { [<for_pack_ $T>]::<W, B>(input, reference, output) }
            }

            #[inline(never)]
            fn unfor_pack<const W: usize, const B: usize>(
                input: &[Self; B],
                reference: Self,
                output: &mut [Self; 1024],
            ) {
                // SAFETY: the lengths are those of the arrays.
                unsafe { [<unfor_pack_ $T>]::<W, B>(input, reference, output) }
            }

            unsafe fn unchecked_unfor_pack(width: usize, input: &[Self], reference: Self, output: &mut [Self]) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                // SAFETY: forwarded to the caller.
                unsafe { [<unfor_pack_width_ $T>]::dispatch(width, input, reference, output) }
            }
        }
        }
    };
}

impl_for!(u8);
impl_for!(u16);
impl_for!(u32);
impl_for!(u64);

#[cfg(test)]
mod test {
    use super::*;
    use alloc::{format, string::ToString, vec};
    use core::fmt::Debug;
    use core::mem::size_of;
    use hegel::generators as gs;
    use hegel::generators::Integer;
    use hegel::{PrettyPrintable, TestCase};
    use pastey::paste;

    #[test]
    fn test_ffor() {
        const W: usize = 15;
        const B: usize = 1024 * W / u16::T;

        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << W)) as u16 + 10;
        }

        let mut packed = [0; 128 * W / size_of::<u16>()];
        FoR::for_pack::<W, B>(&values, 10, &mut packed);

        let mut unpacked = [0; 1024];
        BitPacking::unpack::<W, B>(&packed, &mut unpacked);

        for (i, (a, b)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                // Check that the unpacked array is 10 less than the original (modulo 2^15)
                a.wrapping_sub(10) & ((1 << W) - 1),
                *b,
                "Mismatch at index {i}"
            );
        }
    }

    trait RuntimeForPack: FoR {
        fn for_pack_for_width(
            width: usize,
            input: &[Self; 1024],
            reference: Self,
            output: &mut [Self],
        );
    }

    macro_rules! impl_runtime_for_pack {
        ($T:ident) => {
            impl RuntimeForPack for $T {
                fn for_pack_for_width(
                    width: usize,
                    input: &[Self; 1024],
                    reference: Self,
                    output: &mut [Self],
                ) {
                    macro_rules! pack_width {
                        ($W:expr) => {{
                            const B: usize = 1024 * $W / <$T>::T;
                            // SAFETY: the caller allocates `output` for `width`, and this branch is
                            // selected only when `width == W`.
                            let output =
                                unsafe { crate::as_array_mut_unchecked::<$T, B>(output) };
                            <$T>::for_pack::<$W, B>(input, reference, output);
                        }};
                    }

                    paste!(crate::seq_t!(W in $T {
                        match width {
                            #(W => pack_width!(W),)*
                            <$T>::T => pack_width!({ <$T>::T }),
                            _ => unreachable!("unsupported width {width}"),
                        }
                    }));
                }
            }
        };
    }

    impl_runtime_for_pack!(u8);
    impl_runtime_for_pack!(u16);
    impl_runtime_for_pack!(u32);
    impl_runtime_for_pack!(u64);

    fn assert_ffor_matches_wrapping_model<T>(tc: &TestCase)
    where
        T: Debug
            + Integer
            + PrettyPrintable
            + RuntimeForPack
            + Send
            + Sync
            + num_traits::WrappingAdd
            + num_traits::WrappingSub
            + 'static,
    {
        let input: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let reference = tc.draw(gs::integers::<T>());

        for width in 0..=T::T {
            let packed_len = 1024 * width / T::T;
            let mut packed = vec![T::max_value(); packed_len];
            let mut actual = [T::max_value(); 1024];

            T::for_pack_for_width(width, &input, reference, &mut packed);
            unsafe { T::unchecked_unfor_pack(width, &packed, reference, &mut actual) };

            let mask = if width == 0 {
                T::zero()
            } else if width == T::T {
                T::max_value()
            } else {
                (T::one() << width) - T::one()
            };
            let expected = core::array::from_fn(|i| {
                let delta = num_traits::WrappingSub::wrapping_sub(&input[i], &reference);
                num_traits::WrappingAdd::wrapping_add(&reference, &(delta & mask))
            });
            assert_eq!(actual, expected);
        }
    }

    fn assert_unchecked_unfor_pack_matches_unfused<T>(tc: &TestCase)
    where
        T: Debug
            + FoR
            + Integer
            + PrettyPrintable
            + Send
            + Sync
            + num_traits::WrappingAdd
            + 'static,
    {
        let packed_source: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let reference = tc.draw(gs::integers::<T>());

        for width in 0..=T::T {
            let packed_len = 1024 * width / T::T;
            let mut packed = vec![T::max_value(); packed_len];
            unsafe {
                T::unchecked_pack(width, &packed_source, &mut packed);
            }

            let mut unpacked = [T::max_value(); 1024];
            unsafe { T::unchecked_unpack(width, &packed, &mut unpacked) };
            let expected =
                unpacked.map(|value| num_traits::WrappingAdd::wrapping_add(&value, &reference));

            let mut actual = [T::max_value(); 1024];
            unsafe {
                T::unchecked_unfor_pack(width, &packed, reference, &mut actual);
            }

            assert_eq!(actual, expected);
        }
    }

    macro_rules! ffor_property_tests {
        ($T:ident) => {
            paste! {
                #[hegel::test(test_cases = 10)]
                fn [<test_ffor_matches_wrapping_model_ $T>](tc: TestCase) {
                    assert_ffor_matches_wrapping_model::<$T>(&tc);
                }

                #[hegel::test(test_cases = 10)]
                fn [<test_unchecked_unfor_pack_matches_unfused_ $T>](tc: TestCase) {
                    assert_unchecked_unfor_pack_matches_unfused::<$T>(&tc);
                }
            }
        };
    }

    ffor_property_tests!(u8);
    ffor_property_tests!(u16);
    ffor_property_tests!(u32);
    ffor_property_tests!(u64);
}
