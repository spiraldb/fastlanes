#![allow(unused_assignments)]

use crate::{BitPacking, FastLanes, iterate, supported_bit_width, unpack};

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
    use core::mem::size_of;
    use hegel::TestCase;
    use hegel::generators as gs;
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

    macro_rules! delta_property_tests {
        ($T:ident, $lanes:expr) => {
            paste! {
                #[hegel::test]
                fn [<test_delta_matches_reference_ $T>](tc: TestCase) {
                    let input: [$T; 1024] =
                        tc.draw(gs::arrays(gs::integers::<$T>()));
                    let base: [$T; $lanes] =
                        tc.draw(gs::arrays(gs::integers::<$T>()));
                    let mut expected = [<$T>::MAX; 1024];

                    for lane in 0..$lanes {
                        let mut previous = base[lane];
                        for row in 0..<$T>::T {
                            let order = row / 8;
                            let sub_row = row % 8;
                            let index =
                                (crate::FL_ORDER[order] * 16) + (sub_row * 128) + lane;
                            expected[index] = input[index].wrapping_sub(previous);
                            previous = input[index];
                        }
                    }

                    let mut actual = [<$T>::MAX; 1024];
                    <$T>::delta::<$lanes>(&input, &base, &mut actual);

                    assert_eq!(actual, expected);
                }

                #[hegel::test]
                fn [<test_delta_roundtrip_ $T>](tc: TestCase) {
                    let input: [$T; 1024] =
                        tc.draw(gs::arrays(gs::integers::<$T>()));
                    let base: [$T; $lanes] =
                        tc.draw(gs::arrays(gs::integers::<$T>()));
                    let mut deltas = [<$T>::MAX; 1024];
                    let mut output = [<$T>::MAX; 1024];

                    <$T>::delta::<$lanes>(&input, &base, &mut deltas);
                    <$T>::undelta::<$lanes>(&deltas, &base, &mut output);

                    assert_eq!(output, input);
                }

                #[hegel::test(test_cases = 10)]
                fn [<test_undelta_pack_matches_unfused_ $T>](tc: TestCase) {
                    let deltas: [$T; 1024] =
                        tc.draw(gs::arrays(gs::integers::<$T>()));
                    let base: [$T; $lanes] =
                        tc.draw(gs::arrays(gs::integers::<$T>()));

                    for width in 0..=<$T>::T {
                        let packed_len = 1024 * width / <$T>::T;
                        let mut packed = vec![<$T>::MAX; packed_len];
                        unsafe { <$T>::unchecked_pack(width, &deltas, &mut packed) };

                        let mut unpacked = [<$T>::MAX; 1024];
                        let mut expected = [<$T>::MAX; 1024];
                        unsafe { <$T>::unchecked_unpack(width, &packed, &mut unpacked) };
                        <$T>::undelta::<$lanes>(&unpacked, &base, &mut expected);

                        macro_rules! check_width {
                            ($W:expr) => {{
                                const B: usize = 1024 * $W / <$T>::T;
                                let mut actual = [<$T>::MAX; 1024];
                                // SAFETY: `packed_len` is exactly `1024 * width / T`, and this
                                // branch is selected only when `width == W`.
                                let packed_array =
                                    unsafe { crate::as_array_unchecked::<$T, B>(&packed) };
                                <$T>::undelta_pack::<$lanes, $W, B>(
                                    packed_array,
                                    &base,
                                    &mut actual,
                                );
                                assert_eq!(actual, expected);
                            }};
                        }

                        paste!(crate::seq_t!(W in $T {
                            match width {
                                #(W => check_width!(W),)*
                                <$T>::T => check_width!({ <$T>::T }),
                                _ => unreachable!("unsupported width {width}"),
                            }
                        }));
                    }
                }
            }
        };
    }

    delta_property_tests!(u8, 128);
    delta_property_tests!(u16, 64);
    delta_property_tests!(u32, 32);
    delta_property_tests!(u64, 16);
}
