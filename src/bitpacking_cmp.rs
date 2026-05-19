use crate::seq_t;
use crate::unpack;
use crate::{supported_bit_width, BitPacking, FastLanes, FastLanesComparable};
use paste::paste;

fn invert(output: &mut [bool; 1024]) {
    for value in output {
        *value = !*value;
    }
}

fn or_assign(lhs: &mut [bool; 1024], rhs: &[bool; 1024]) {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs |= *rhs;
    }
}

fn not_or_assign(lhs: &mut [bool; 1024], rhs: &[bool; 1024]) {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs = !(*lhs || *rhs);
    }
}

pub trait BitPackingCompare: BitPacking {
    /// Fused `BitPacking::unpack` and equality comparison against a constant value.
    ///
    /// The packed values are stored as `Self`, while the comparison is on `V` where
    /// `V::Bitpacked = Self`. This allows signed values to be compared after unpacking from
    /// their unsigned bitpacked representation.
    fn unpack_cmp_eq<const W: usize, const B: usize, V>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Eq;

    /// Fused `BitPacking::unpack` and less-than comparison against a constant value.
    ///
    /// The packed values are stored as `Self`, while the comparison is on `V` where
    /// `V::Bitpacked = Self`. This allows signed values to be compared after unpacking from
    /// their unsigned bitpacked representation.
    fn unpack_cmp_lt<const W: usize, const B: usize, V>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord;

    /// Fused `BitPacking::unpack` and inequality comparison against a constant value.
    fn unpack_cmp_ne<const W: usize, const B: usize, V>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Eq,
    {
        Self::unpack_cmp_eq::<W, B, V>(input, output, value);
        invert(output);
    }

    /// Fused `BitPacking::unpack` and less-than-or-equal comparison against a constant value.
    fn unpack_cmp_le<const W: usize, const B: usize, V>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        Self::unpack_cmp_lt::<W, B, V>(input, output, value);

        let mut eq = [false; 1024];
        Self::unpack_cmp_eq::<W, B, V>(input, &mut eq, value);
        or_assign(output, &eq);
    }

    /// Fused `BitPacking::unpack` and greater-than comparison against a constant value.
    fn unpack_cmp_gt<const W: usize, const B: usize, V>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        Self::unpack_cmp_lt::<W, B, V>(input, output, value);

        let mut eq = [false; 1024];
        Self::unpack_cmp_eq::<W, B, V>(input, &mut eq, value);
        not_or_assign(output, &eq);
    }

    /// Fused `BitPacking::unpack` and greater-than-or-equal comparison against a constant value.
    fn unpack_cmp_ge<const W: usize, const B: usize, V>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        Self::unpack_cmp_lt::<W, B, V>(input, output, value);
        invert(output);
    }

    /// Runtime-width fused `BitPacking::unpack` and equality comparison against a constant value.
    ///
    /// # Safety
    ///
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output array must contain exactly 1024 bools.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp_eq<V>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Eq,
    {
        let mut unpacked = [Self::zero(); 1024];
        // SAFETY: This method has the same input and width safety contract as `unchecked_unpack`.
        unsafe { Self::unchecked_unpack(width, input, &mut unpacked) };

        for (output, unpacked) in output.iter_mut().zip(unpacked) {
            *output = V::as_unpacked(unpacked) == value;
        }
    }

    /// Runtime-width fused `BitPacking::unpack` and less-than comparison against a constant value.
    ///
    /// # Safety
    ///
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output array must contain exactly 1024 bools.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp_lt<V>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        let mut unpacked = [Self::zero(); 1024];
        // SAFETY: This method has the same input and width safety contract as `unchecked_unpack`.
        unsafe { Self::unchecked_unpack(width, input, &mut unpacked) };

        for (output, unpacked) in output.iter_mut().zip(unpacked) {
            *output = V::as_unpacked(unpacked) < value;
        }
    }

    /// Runtime-width fused `BitPacking::unpack` and inequality comparison against a constant value.
    ///
    /// # Safety
    ///
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output array must contain exactly 1024 bools.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp_ne<V>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Eq,
    {
        // SAFETY: This method has the same input and width safety contract as `unchecked_unpack_cmp_eq`.
        unsafe { Self::unchecked_unpack_cmp_eq(width, input, output, value) };
        invert(output);
    }

    /// Runtime-width fused `BitPacking::unpack` and less-than-or-equal comparison against a constant value.
    ///
    /// # Safety
    ///
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output array must contain exactly 1024 bools.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp_le<V>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        // SAFETY: This method has the same input and width safety contract as `unchecked_unpack_cmp_lt`.
        unsafe { Self::unchecked_unpack_cmp_lt(width, input, output, value) };

        let mut eq = [false; 1024];
        // SAFETY: Same arguments and safety contract as above.
        unsafe { Self::unchecked_unpack_cmp_eq(width, input, &mut eq, value) };
        or_assign(output, &eq);
    }

    /// Runtime-width fused `BitPacking::unpack` and greater-than comparison against a constant value.
    ///
    /// # Safety
    ///
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output array must contain exactly 1024 bools.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp_gt<V>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        // SAFETY: This method has the same input and width safety contract as `unchecked_unpack_cmp_lt`.
        unsafe { Self::unchecked_unpack_cmp_lt(width, input, output, value) };

        let mut eq = [false; 1024];
        // SAFETY: Same arguments and safety contract as above.
        unsafe { Self::unchecked_unpack_cmp_eq(width, input, &mut eq, value) };
        not_or_assign(output, &eq);
    }

    /// Runtime-width fused `BitPacking::unpack` and greater-than-or-equal comparison against a constant value.
    ///
    /// # Safety
    ///
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output array must contain exactly 1024 bools.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp_ge<V>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self> + Ord,
    {
        // SAFETY: This method has the same input and width safety contract as `unchecked_unpack_cmp_lt`.
        unsafe { Self::unchecked_unpack_cmp_lt(width, input, output, value) };
        invert(output);
    }
}

macro_rules! impl_packing_compare {
    ($T:ty) => {
        impl BitPackingCompare for $T {
            #[inline(never)]
            fn unpack_cmp_eq<const W: usize, const B: usize, V>(
                input: &[Self; B],
                output: &mut [bool; 1024],
                other: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self> + Eq
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in (0..Self::LANES) {
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        output[$idx] = V::as_unpacked($elem) == other;
                    });
                }
            }

            #[inline(never)]
            fn unpack_cmp_lt<const W: usize, const B: usize, V>(
                input: &[Self; B],
                output: &mut [bool; 1024],
                other: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self> + Ord
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in (0..Self::LANES) {
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        output[$idx] = V::as_unpacked($elem) < other;
                    });
                }
            }

            unsafe fn unchecked_unpack_cmp_eq<V>(
                 width: usize,
                 input: &[Self],
                 output: &mut [bool; 1024],
                 value: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self> + Eq
            {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack_cmp_eq::<W, B, V>(
                                arrayref::array_ref![input, 0, 1024 * W / <$T>::T],
                                output,
                                value
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::unpack_cmp_eq::<W, B, V>(
                                arrayref::array_ref![input, 0, B],
                                output,
                                value
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }

            unsafe fn unchecked_unpack_cmp_lt<V>(
                 width: usize,
                 input: &[Self],
                 output: &mut [bool; 1024],
                 value: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self> + Ord
            {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack_cmp_lt::<W, B, V>(
                                arrayref::array_ref![input, 0, 1024 * W / <$T>::T],
                                output,
                                value
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::unpack_cmp_lt::<W, B, V>(
                                arrayref::array_ref![input, 0, B],
                                output,
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
    use crate::BitPacking;
    use core::array;

    #[test]
    fn test_unpack_cmp_eq() {
        type T = u32;
        const W: usize = 10;
        const B: usize = 1024 * W / T::T;

        let values = array::from_fn(|i| i as T % (1 << W));

        let mut packed = [0; (128 * W) / size_of::<T>()];
        T::pack::<W, B>(&values, &mut packed);

        // Check equality against every value of the vector
        for value in 0u32..1024 {
            let cmp = {
                let mut output = [false; 1024];
                T::unpack_cmp_eq::<W, B, T>(&packed, &mut output, value);
                output
            };

            let expected = array::from_fn(|i| values[i] == value);

            assert_eq!(cmp, expected, "Failed == {value}");
        }
    }

    #[test]
    fn const_width_named_comparisons_match_unpacked_values() {
        type T = u32;
        const W: usize = 10;
        const B: usize = 1024 * W / T::T;

        let values = array::from_fn(|i| (i as T) % 17);
        let mut packed = [0; B];
        T::pack::<W, B>(&values, &mut packed);

        let value = 8;

        let mut output = [false; 1024];

        T::unpack_cmp_eq::<W, B, T>(&packed, &mut output, value);
        let expected = array::from_fn(|i| values[i] == value);
        assert_eq!(output, expected, "eq");

        T::unpack_cmp_ne::<W, B, T>(&packed, &mut output, value);
        let expected = array::from_fn(|i| values[i] != value);
        assert_eq!(output, expected, "ne");

        T::unpack_cmp_lt::<W, B, T>(&packed, &mut output, value);
        let expected = array::from_fn(|i| values[i] < value);
        assert_eq!(output, expected, "lt");

        T::unpack_cmp_le::<W, B, T>(&packed, &mut output, value);
        let expected = array::from_fn(|i| values[i] <= value);
        assert_eq!(output, expected, "le");

        T::unpack_cmp_gt::<W, B, T>(&packed, &mut output, value);
        let expected = array::from_fn(|i| values[i] > value);
        assert_eq!(output, expected, "gt");

        T::unpack_cmp_ge::<W, B, T>(&packed, &mut output, value);
        let expected = array::from_fn(|i| values[i] >= value);
        assert_eq!(output, expected, "ge");
    }

    #[test]
    fn runtime_width_named_comparisons_match_unpacked_values() {
        type T = u16;
        const W: usize = 5;
        const B: usize = 1024 * W / T::T;

        let values = array::from_fn(|i| (i as T) % 31);
        let mut packed = [0; B];
        T::pack::<W, B>(&values, &mut packed);

        let value = 13;

        let mut output = [false; 1024];

        unsafe { T::unchecked_unpack_cmp_eq(W, &packed, &mut output, value) };
        let expected = array::from_fn(|i| values[i] == value);
        assert_eq!(output, expected, "eq");

        unsafe { T::unchecked_unpack_cmp_ne(W, &packed, &mut output, value) };
        let expected = array::from_fn(|i| values[i] != value);
        assert_eq!(output, expected, "ne");

        unsafe { T::unchecked_unpack_cmp_lt(W, &packed, &mut output, value) };
        let expected = array::from_fn(|i| values[i] < value);
        assert_eq!(output, expected, "lt");

        unsafe { T::unchecked_unpack_cmp_le(W, &packed, &mut output, value) };
        let expected = array::from_fn(|i| values[i] <= value);
        assert_eq!(output, expected, "le");

        unsafe { T::unchecked_unpack_cmp_gt(W, &packed, &mut output, value) };
        let expected = array::from_fn(|i| values[i] > value);
        assert_eq!(output, expected, "gt");

        unsafe { T::unchecked_unpack_cmp_ge(W, &packed, &mut output, value) };
        let expected = array::from_fn(|i| values[i] >= value);
        assert_eq!(output, expected, "ge");
    }

    #[test]
    fn runtime_width_accepts_full_type_width() {
        type T = u8;
        const W: usize = T::T;
        const B: usize = 1024 * W / T::T;

        let values = array::from_fn(|i| i as T);
        let mut packed = [0; B];
        T::pack::<W, B>(&values, &mut packed);

        let mut output = [false; 1024];
        unsafe { T::unchecked_unpack_cmp_eq(W, &packed, &mut output, 255u8) };

        let expected = array::from_fn(|i| values[i] == 255);
        assert_eq!(output, expected);
    }

    #[test]
    fn signed_less_than_compares_unpacked_signed_values() {
        type T = u8;
        const W: usize = T::T;
        const B: usize = 1024 * W / T::T;

        let signed_values = array::from_fn(|i| {
            let value = (i % 128) as u8;
            i8::from_ne_bytes([value]) - 64
        });
        let values = signed_values.map(|value| T::from_ne_bytes(value.to_ne_bytes()));
        let mut packed = [0; B];
        T::pack::<W, B>(&values, &mut packed);

        let mut output = [false; 1024];
        T::unpack_cmp_lt::<W, B, i8>(&packed, &mut output, -10);

        let expected = array::from_fn(|i| signed_values[i] < -10);
        assert_eq!(output, expected);
    }
}
