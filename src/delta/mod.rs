#![allow(unused_assignments)]

use crate::{
    BitPacking, FL_ORDER, FastLanes, iterate, supported_bit_width, unpack, width_dispatch,
};
use core::mem::size_of;
use pastey::paste;

// The per-type bodies of `Delta::undelta_pack_single`: one small copy for every width by default,
// or an unrolled copy for each width with the `unroll` feature.
#[cfg_attr(feature = "unroll", path = "single_unrolled.rs")]
mod single;

/// The lane and row of position `index` in a delta-encoded vector of `T`.
///
/// Delta encodes each lane against its own base, row by row, so the element at `index` of
/// [`Delta::undelta`]'s output depends on its lane's base and that lane's deltas at rows `0..=row`,
/// found at [`delta_index`]. Positions count in the transposed order the deltas were computed in;
/// [`untranspose`](crate::untranspose) gives the position of an untransposed element.
///
/// # Panics
///
/// Panics if `index` is not less than 1024.
#[inline]
#[must_use]
pub fn delta_lane_row<T: FastLanes>(index: usize) -> (usize, usize) {
    assert!(index < 1024, "index {index} out of bounds");
    crate::lane_row::<T>(index)
}

/// The position in a delta-encoded vector of `T` of row `row` of lane `lane`: the inverse of
/// [`delta_lane_row`].
///
/// # Panics
///
/// Panics if `lane` is not less than `T::LANES` or `row` is not less than `T::T`.
#[inline]
#[must_use]
pub fn delta_index<T: FastLanes>(lane: usize, row: usize) -> usize {
    assert!(lane < T::LANES, "lane {lane} out of bounds");
    assert!(row < T::T, "row {row} out of bounds");
    FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane
}

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

    /// Element `index` of [`Delta::undelta`]'s output, reading only the deltas of its lane up to
    /// its row: at most `T` of the 1024.
    ///
    /// `base` is the base of the element's lane, `base[delta_lane_row::<Self>(index).0]` in
    /// [`Delta::undelta`]'s terms.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not less than 1024.
    fn undelta_single(input: &[Self; 1024], base: Self, index: usize) -> Self;

    /// Element `index` of the output of [`Delta::unchecked_undelta_pack`] at width `width`,
    /// unpacking only the deltas of its lane up to its row: at most `T` of the 1024.
    ///
    /// `base` is the base of the element's lane, as for [`Delta::undelta_single`].
    ///
    /// By default one small copy of this function serves every width. The `unroll` feature
    /// instead compiles an unrolled copy for each width, which is faster, especially for wide
    /// deltas, at the cost of much more code.
    ///
    /// # Panics
    ///
    /// Panics if `width` is greater than the bit-width of `Self`, if `input` is not of length
    /// `1024 * width / T`, or if `index` is not less than 1024.
    fn undelta_pack_single(width: usize, input: &[Self], base: Self, index: usize) -> Self;
}

macro_rules! impl_undelta_pack {
    ($T:ident) => {
        paste! {
            /// # Safety
            /// `input` must hold `B == 1024 * W / T` elements and `output` `1024`.
            #[inline(always)]
            unsafe fn [<undelta_pack_ $T>]<const W: usize, const B: usize>(
                input: &[$T],
                base: &[$T; <$T>::LANES],
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
                    let mut prev = base[lane];
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let next = $elem.wrapping_add(prev);
                        output[$idx] = next;
                        prev = next;
                    });
                }
            }

            width_dispatch!(
                $T,
                [<undelta_pack_width_ $T>],
                [<undelta_pack_ $T>](input: &[$T], base: &[$T; <$T>::LANES], output: &mut [$T])
            );
        }
    };
}

impl_undelta_pack!(u8);
impl_undelta_pack!(u16);
impl_undelta_pack!(u32);
impl_undelta_pack!(u64);

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
                }
                // SAFETY: `LANES == Self::LANES`, asserted above.
                let base = unsafe { &*base.as_ptr().cast::<[Self; <$T>::LANES]>() };
                // SAFETY: the lengths are those of the arrays.
                paste!(unsafe { [<undelta_pack_ $T>]::<W, B>(input, base, output) })
            }

            unsafe fn unchecked_undelta_pack<const LANES: usize>(
                width: usize,
                input: &[Self],
                base: &[Self; LANES],
                output: &mut [Self],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                }
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(
                    input.len(),
                    packed_len,
                    "Input buffer must be of size 1024 * W / T"
                );
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                // SAFETY: `LANES == Self::LANES`, asserted above. The caller guarantees the
                // lengths of `input` and `output`.
                unsafe {
                    let base = &*base.as_ptr().cast::<[Self; <$T>::LANES]>();
                    paste!([<undelta_pack_width_ $T>]::dispatch(width, input, base, output))
                }
            }

            #[inline]
            fn undelta_single(input: &[Self; 1024], base: Self, index: usize) -> Self {
                let (lane, row) = delta_lane_row::<Self>(index);
                let mut value = base;
                for row in 0..=row {
                    value = value.wrapping_add(input[delta_index::<Self>(lane, row)]);
                }
                value
            }

            fn undelta_pack_single(
                width: usize,
                input: &[Self],
                base: Self,
                index: usize,
            ) -> Self {
                assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);
                assert_eq!(
                    input.len(),
                    1024 * width / Self::T,
                    "Input buffer must be of size 1024 * W / T"
                );
                let (lane, row) = delta_lane_row::<Self>(index);
                if width == 0 {
                    return base;
                }

                paste!(single::[<undelta_pack_single_ $T>](width, input, base, lane, row))
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

    fn assert_lane_row_inverts_index<T: FastLanes>() {
        let mut seen = [false; 1024];
        for lane in 0..T::LANES {
            for row in 0..T::T {
                let index = delta_index::<T>(lane, row);
                assert!(!seen[index], "position {index} repeats");
                seen[index] = true;
                assert_eq!(delta_lane_row::<T>(index), (lane, row));
            }
        }
    }

    fn assert_undelta_single_matches_undelta<T, const LANES: usize>(tc: &TestCase)
    where
        T: Delta + Debug + Integer + PrettyPrintable + Send + Sync + 'static,
    {
        let deltas: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let base: [T; LANES] = tc.draw(gs::arrays(gs::integers::<T>()));
        let mut expected = [T::max_value(); 1024];
        T::undelta::<LANES>(&deltas, &base, &mut expected);

        for (index, &expected) in expected.iter().enumerate() {
            let lane_base = base[delta_lane_row::<T>(index).0];
            assert_eq!(T::undelta_single(&deltas, lane_base, index), expected);
        }
    }

    fn assert_undelta_pack_single_matches_undelta_pack<T, const LANES: usize>(
        deltas: &[T; 1024],
        base: &[T; LANES],
    ) where
        T: Delta + Debug,
    {
        for width in 0..=T::T {
            let packed_len = 1024 * width / T::T;
            let mut packed = vec![T::max_value(); packed_len];
            unsafe { T::unchecked_pack(width, deltas, &mut packed) };

            let mut expected = [T::max_value(); 1024];
            unsafe { T::unchecked_undelta_pack(width, &packed, base, &mut expected) };
            for (index, &expected) in expected.iter().enumerate() {
                let lane_base = base[delta_lane_row::<T>(index).0];
                let actual = T::undelta_pack_single(width, &packed, lane_base, index);
                assert_eq!(actual, expected, "width {width}, index {index}");
            }
        }
    }

    /// The largest deltas are the ones that could overflow the sums `undelta_pack_single` keeps.
    #[test]
    fn test_undelta_pack_single_of_largest_deltas() {
        assert_undelta_pack_single_matches_undelta_pack(&[u8::MAX; 1024], &[1; 128]);
        assert_undelta_pack_single_matches_undelta_pack(&[u16::MAX; 1024], &[1; 64]);
        assert_undelta_pack_single_matches_undelta_pack(&[u32::MAX; 1024], &[1; 32]);
        assert_undelta_pack_single_matches_undelta_pack(&[u64::MAX; 1024], &[1; 16]);
    }

    #[test]
    fn test_delta_lane_row_inverts_delta_index() {
        assert_lane_row_inverts_index::<u8>();
        assert_lane_row_inverts_index::<u16>();
        assert_lane_row_inverts_index::<u32>();
        assert_lane_row_inverts_index::<u64>();
    }

    #[test]
    fn test_undelta_single_of_transposed_element() {
        let values: [u32; 1024] = core::array::from_fn(|i| (i * 3 + i % 7) as u32);
        let mut transposed = [0; 1024];
        Transpose::transpose(&values, &mut transposed);
        let base: [u32; 32] = core::array::from_fn(|lane| transposed[lane]);
        let mut deltas = [0; 1024];
        Delta::delta(&transposed, &base, &mut deltas);

        for (i, &value) in values.iter().enumerate() {
            let index = crate::untranspose(i);
            let lane_base = base[delta_lane_row::<u32>(index).0];
            assert_eq!(u32::undelta_single(&deltas, lane_base, index), value);
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

                #[hegel::test]
                fn [<test_undelta_single_matches_undelta_ $T>](tc: TestCase) {
                    assert_undelta_single_matches_undelta::<$T, $lanes>(&tc);
                }

                #[hegel::test(test_cases = 10)]
                fn [<test_undelta_pack_single_matches_undelta_pack_ $T>](tc: TestCase) {
                    let deltas: [$T; 1024] = tc.draw(gs::arrays(gs::integers::<$T>()));
                    let base: [$T; $lanes] = tc.draw(gs::arrays(gs::integers::<$T>()));
                    assert_undelta_pack_single_matches_undelta_pack(&deltas, &base);
                }
            }
        };
    }

    delta_property_tests!(u8, 128);
    delta_property_tests!(u16, 64);
    delta_property_tests!(u32, 32);
    delta_property_tests!(u64, 16);
}
