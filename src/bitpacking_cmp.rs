use crate::seq_t;
use crate::unpack;
use crate::{FastLanes, FastLanesComparable, supported_bit_width};
use pastey::paste;

pub trait BitPackingCompare: FastLanes {
    /// A fused unpack (see `BitPacking::unpack`) and compare, packing the boolean results into a
    /// bitmask of `1024` bits (`16 x u64`).
    ///
    /// This compares, using the comparison function, all of the packed values against a constant
    /// `value`. The values are of type `Self`, whereas the comparison is on the type `V` (where
    /// `V::Bitpacked` = `Self`). This allows for comparison between signed values which are
    /// bit-packed as unsigned ones.
    ///
    /// The output is a bitmask in **lane-major order**, not logical row order. The `1024` bits
    /// are `Self::LANES` words of `Self::T` bits, one word per lane laid out contiguously
    /// (little-endian) in the `[u64; 16]`. Within a lane's word the comparison results are packed
    /// LSB-first: row `r` (for `r` in `0..Self::T`) lands at bit `r`, holding the comparison for
    /// the value at logical index `index(row, lane)` (see the `unpack!` macro). This is the
    /// cheapest order to produce: it needs no cross-lane shuffles, just a per-lane accumulator
    /// that the compiler keeps in a (vectorized) register.
    ///
    /// For `u64` this lane-major order is the bit-level [`crate::Transpose::untranspose`] of the
    /// logical mask. To recover logical row order (e.g. an Arrow-style boolean buffer), pass the
    /// result through [`crate::transpose_bits::<Self>`](crate::transpose_bits).
    fn unpack_cmp<const W: usize, const B: usize, V, F>(
        input: &[Self; B],
        output: &mut [u64; 16],
        comparison: F,
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;

    /// A fused unpack (see `BitPacking::unpack`) and compare, packing the boolean results into a
    /// bitmask of `1024` bits (`16 x u64`). See [`BitPackingCompare::unpack_cmp`] for the output
    /// bit ordering.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output is exactly `[u64; 16]` (`1024` bits).
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    ///
    /// # Panics
    /// Panics if `width` is greater than the bit-width of `Self`.
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

                // The output is 1024 bits laid out as `Self::LANES` words of `Self::T` bits each
                // (which is always 128 bytes == `[u64; 16]`). Each lane owns one contiguous word
                // holding that lane's `Self::T` comparison results, LSB-first: row `r` lands at bit
                // `r`. Per-lane ownership means the accumulator stays in a register and the store is
                // a single contiguous (vectorizable) write per lane -- no `[bool; 1024]`
                // (or `[Self; 1024]`) materialization, no cross-lane shuffles.
                //
                // For `u64` (`Self::LANES == 16`) this LSB-first ordering coincides with the
                // canonical FastLanes transpose; for narrower widths it is the per-width packing
                // that [`crate::bit_transpose::untranspose_bits::<Self>`] inverts. Either way that
                // is what [`untranspose_cmp_mask`] uses to recover logical row order.
                //
                // SAFETY: `[u64; 16]` and `[Self; LANES]` are both exactly 128 bytes, and `u64`'s
                // alignment (8) is >= `Self`'s alignment, so the reinterpret is sound.
                let words: &mut [$T; <$T>::LANES] =
                    unsafe { &mut *output.as_mut_ptr().cast::<[$T; <$T>::LANES]>() };

                for lane in 0..Self::LANES {
                    let mut word: $T = 0;
                    let mut bit: usize = 0;
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let _ = $idx;
                        word |= <$T>::from(f(V::as_unpacked($elem), other)) << bit;
                        #[allow(unused_assignments)]
                        { bit += 1; }
                    });
                    words[lane] = word;
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
                // `width > Self::T` falls through to the `unreachable!` arm below in every build.
                debug_assert!(input.len() == 128 * width / size_of::<Self>());

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack_cmp::<W, B, V, F>(
                                unsafe { crate::as_array_unchecked(input) },
                                output,
                                comparison,
                                value
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                           const W: usize = <$T>::T;
                           Self::unpack_cmp::<W, 1024, V, F>(
                               unsafe { crate::as_array_unchecked(input) },
                               output,
                               comparison,
                               value
                           )
                        },
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
    use crate::{BitPacking, transpose_bits};
    use alloc::{format, string::ToString, vec};
    use core::array;
    use core::fmt::Debug;
    use hegel::TestCase;
    use hegel::generators as gs;
    use hegel::generators::Integer;
    use pastey::paste;

    /// Reference bitmask in the same `FastLanes` (LSB-first, per-lane) order produced by
    /// `unpack_cmp`:
    /// fully unpack, then for each lane set bit `row` from the comparison of the value at the
    /// logical index `index(row, lane)`.
    fn reference_mask<T, V, F>(packed_unpacked: &[T; 1024], f: F, other: V) -> [u64; 16]
    where
        T: FastLanes,
        V: FastLanesComparable<Bitpacked = T>,
        F: Fn(V, V) -> bool,
    {
        let mut out = [0u64; 16];
        for lane in 0..T::LANES {
            for row in 0..T::T {
                // `index(row, lane)` from the unpack macro.
                let o = row / 8;
                let s = row % 8;
                let idx = (crate::FL_ORDER[o] * 16) + (s * 128) + lane;
                if f(V::as_unpacked(packed_unpacked[idx]), other) {
                    // LSB-first within each lane word: row `r` lands at bit `r`.
                    let bit = lane * T::T + row;
                    out[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
        out
    }

    fn comparison<V: PartialOrd>(operation: u8) -> fn(V, V) -> bool {
        match operation {
            0 => |a, b| a == b,
            1 => |a, b| a != b,
            2 => |a, b| a < b,
            3 => |a, b| a <= b,
            4 => |a, b| a > b,
            5 => |a, b| a >= b,
            _ => unreachable!("unsupported comparison {operation}"),
        }
    }

    #[test]
    fn test_unpack_eq() {
        type T = u32;
        const W: usize = 10;
        const B: usize = 1024 * W / T::T;

        let values = array::from_fn(|i| i as T % (1 << W));

        let mut packed = [0; (128 * W) / size_of::<T>()];
        T::pack::<W, B>(&values, &mut packed);

        let mut unpacked = [0u32; 1024];
        T::unpack::<W, B>(&packed, &mut unpacked);

        // Check equality against every value of the vector.
        for v in 0..1024 {
            let cmp = {
                let mut output = [0u64; 16];
                T::unpack_cmp::<W, B, _, _>(&packed, &mut output, |a, b| a == b, v);
                output
            };

            let expected = reference_mask(&unpacked, |a, b| a == b, v);
            assert_eq!(cmp, expected, "Failed == {v}");
        }
    }

    fn assert_unpack_cmp_matches_reference<T, V>(tc: &TestCase)
    where
        T: BitPacking + BitPackingCompare + Debug + Integer + Send + Sync + 'static,
        V: Debug + FastLanesComparable<Bitpacked = T> + Integer + PartialOrd + 'static,
    {
        let values: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let other = tc.draw(gs::integers::<V>());

        for width in 0..=T::T {
            let packed_len = 1024 * width / T::T;
            let mut packed = vec![T::max_value(); packed_len];
            unsafe { T::unchecked_pack(width, &values, &mut packed) };
            let mask = if width == 0 {
                T::zero()
            } else if width == T::T {
                T::max_value()
            } else {
                (T::one() << width) - T::one()
            };
            let unpacked = values.map(|value| value & mask);

            for operation in 0..=5 {
                let f = comparison::<V>(operation);
                let expected = reference_mask(&unpacked, f, other);
                let mut actual = [u64::MAX; 16];
                unsafe {
                    T::unchecked_unpack_cmp(width, &packed, &mut actual, f, other);
                }

                assert_eq!(
                    actual,
                    expected,
                    "bitpacked={} comparable={} width={width} operation={operation}",
                    core::any::type_name::<T>(),
                    core::any::type_name::<V>(),
                );
            }
        }
    }

    fn assert_unpack_cmp_transposes_to_logical<T, V>(tc: &TestCase)
    where
        T: BitPacking + BitPackingCompare + Debug + Integer + Send + Sync + 'static,
        V: Debug + FastLanesComparable<Bitpacked = T> + Integer + PartialOrd + 'static,
    {
        let values: [T; 1024] = tc.draw(gs::arrays(gs::integers::<T>()));
        let other = tc.draw(gs::integers::<V>());

        for width in 0..=T::T {
            let packed_len = 1024 * width / T::T;
            let mut packed = vec![T::max_value(); packed_len];
            unsafe { T::unchecked_pack(width, &values, &mut packed) };
            let mask = if width == 0 {
                T::zero()
            } else if width == T::T {
                T::max_value()
            } else {
                (T::one() << width) - T::one()
            };
            let unpacked = values.map(|value| value & mask);

            for operation in 0..=5 {
                let f = comparison::<V>(operation);
                let mut expected = [0u64; 16];
                for (index, value) in unpacked.iter().copied().enumerate() {
                    if f(V::as_unpacked(value), other) {
                        expected[index / 64] |= 1u64 << (index % 64);
                    }
                }

                let mut lane_major = [u64::MAX; 16];
                unsafe {
                    T::unchecked_unpack_cmp(width, &packed, &mut lane_major, f, other);
                }
                let mut actual = [u64::MAX; 16];
                transpose_bits::<T>(&lane_major, &mut actual);

                assert_eq!(
                    actual,
                    expected,
                    "bitpacked={} comparable={} width={width} operation={operation}",
                    core::any::type_name::<T>(),
                    core::any::type_name::<V>(),
                );
            }
        }
    }

    macro_rules! comparison_property_tests {
        ($T:ident, $V:ident) => {
            paste! {
                #[hegel::test(test_cases = 10)]
                fn [<test_unpack_cmp_matches_reference_ $T _ $V>](tc: TestCase) {
                    assert_unpack_cmp_matches_reference::<$T, $V>(&tc);
                }

                #[hegel::test(test_cases = 10)]
                fn [<test_unpack_cmp_transposes_to_logical_ $T _ $V>](tc: TestCase) {
                    assert_unpack_cmp_transposes_to_logical::<$T, $V>(&tc);
                }
            }
        };
    }

    comparison_property_tests!(u8, u8);
    comparison_property_tests!(u8, i8);
    comparison_property_tests!(u16, u16);
    comparison_property_tests!(u16, i16);
    comparison_property_tests!(u32, u32);
    comparison_property_tests!(u32, i32);
    comparison_property_tests!(u64, u64);
    comparison_property_tests!(u64, i64);
}
