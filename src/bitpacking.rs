use const_for::const_for;
use core::mem::MaybeUninit;
use core::mem::size_of;
use pastey::paste;

use crate::{FL_ORDER, FastLanes, pack, seq_t, supported_bit_width, unpack};

/// `BitPack` into a compile-time known bit-width.
pub trait BitPacking: FastLanes {
    /// Packs 1024 elements into `W` bits each.
    ///
    /// The output is given as `Self` to ensure correct alignment.
    fn pack<const W: usize, const B: usize>(input: &[Self; 1024], output: &mut [Self; B]);

    /// Packs 1024 elements into `W` bits each, where `W` is runtime-known instead of compile-time
    /// known.
    ///
    /// # Safety
    ///
    /// - The input slice must be of exactly length 1024.
    /// - The output slice must be of length `1024 * W / T`, where `T` is the (unpacked) bit-width
    ///   of `Self` and `W` is the packed bit-width.
    /// - The `width` must be less than or equal to the (unpacked) bit-width of `Self`.
    ///
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks 1024 elements from `W` bits each.
    fn unpack<const W: usize, const B: usize>(input: &[Self; B], output: &mut [Self; 1024]);

    /// Unpacks 1024 elements from `W` bits each, where `W` is runtime-known instead of compile-time
    /// known.
    ///
    /// # Safety
    ///
    /// - The input slice must be of length `1024 * W / T`, where `T` is the (unpacked) bit-width
    ///   of `Self` and `W` is the packed bit-width.
    /// - The output slice must be of exactly length 1024.
    /// - The `width` must be less than or equal to the (unpacked) bit-width of `Self`.
    ///
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not less than 1024.
    fn unpack_single<const W: usize, const B: usize>(packed: &[Self; B], index: usize) -> Self;

    /// Unpacks selected elements from a packed array of 1024 `W` bit elements.
    ///
    /// # Panics
    ///
    /// Panics if the output length differs from the index length or an index is not less than 1024.
    fn unpack_indices<const W: usize, const B: usize>(
        packed: &[Self; B],
        indices: &[usize],
        output: &mut [MaybeUninit<Self>],
    ) {
        assert_eq!(
            indices.len(),
            output.len(),
            "Output length must equal index length"
        );
        for (&index, value) in indices.iter().zip(output) {
            value.write(Self::unpack_single::<W, B>(packed, index));
        }
    }

    /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements,
    /// where `W` is runtime-known instead of compile-time known.
    ///
    /// # Safety
    ///
    /// The input slice must contain at least `1024 * W / T` elements, where `T` is the unpacked
    /// bit width of `Self` and `W` is `width`.
    ///
    /// # Panics
    ///
    /// Panics if `width` exceeds the bit width of `Self` or `index` is not less than 1024. Debug
    /// builds also panic unless the input length equals `1024 * W / T`.
    unsafe fn unchecked_unpack_single(width: usize, input: &[Self], index: usize) -> Self;

    /// Unpacks selected elements where `W` is known only at runtime.
    ///
    /// This method dispatches on `width` once for the complete index batch.
    ///
    /// # Safety
    ///
    /// The input slice must contain at least `1024 * W / T` elements, where `T` is the unpacked
    /// bit width of `Self` and `W` is `width`.
    ///
    /// # Panics
    ///
    /// Panics if `width` exceeds the bit width of `Self`, the output length differs from the index
    /// length, or an index is not less than 1024. Debug builds also panic unless the input length
    /// equals `1024 * W / T`.
    unsafe fn unchecked_unpack_indices(
        width: usize,
        input: &[Self],
        indices: &[usize],
        output: &mut [MaybeUninit<Self>],
    ) {
        assert!(
            width <= Self::T,
            "Width must be less than or equal to {}",
            Self::T
        );
        assert_eq!(
            indices.len(),
            output.len(),
            "Output length must equal index length"
        );
        for (&index, value) in indices.iter().zip(output) {
            // SAFETY: the caller guarantees that `input` contains the packed representation.
            value.write(unsafe { Self::unchecked_unpack_single(width, input, index) });
        }
    }
}

macro_rules! impl_packing {
    ($T:ty) => {
        impl BitPacking for $T {
            #[inline(never)]
            fn pack<const W: usize, const B: usize>(
                input: &[Self; 1024],
                output: &mut [Self; B],
            ) {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }


                for lane in 0..Self::LANES {
                    pack!($T, W, output, lane, |$idx| {
                        input[$idx]
                    });
                }
            }

            unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
                debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::pack::<W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                unsafe { crate::as_array_mut_unchecked(output) },
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            // How large is the target buffer size?
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::pack::<W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                unsafe { crate::as_array_mut_unchecked(output) },
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }

            #[inline(never)]
            fn unpack<const W: usize, const B: usize>(
                input: &[Self; B],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }


                for lane in 0..Self::LANES {
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        output[$idx] = $elem
                    });
                }
            }

            unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack::<W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                unsafe { crate::as_array_mut_unchecked(output) },
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::unpack::<W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                unsafe { crate::as_array_mut_unchecked(output) },
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }

            /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements.
            fn unpack_single<const W: usize, const B: usize>(packed: &[Self; B], index: usize) -> Self
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                assert!(index < 1024, "Index must be less than 1024, got {}", index);

                if W == 0 {
                    // Special case for W=0, we just need to zero the output.
                    return 0 as $T;
                }

                // We can think of the input array as effectively a row-major, left-to-right
                // 2-D array of with `Self::LANES` columns and `Self::T` rows.
                //
                // Meanwhile, we can think of the packed array as either:
                //      1. `Self::T` rows of W-bit elements, with `Self::LANES` columns
                //      2. `W` rows of `Self::T`-bit words, with `Self::LANES` columns
                //
                // Bitpacking involves a transposition of the input array ordering, such that
                // decompression can be fused efficiently with encodings like delta and RLE.
                //
                // First step, we need to get the lane and row for interpretation #1 above.
                let (lane, row): (usize, usize) = {
                    const LANES: [u8; 1024] = lanes_by_index::<$T>();
                    const ROWS: [u8; 1024] = rows_by_index::<$T>();
                    (LANES[index] as usize, ROWS[index] as usize)
                };

                if W == <$T>::T {
                    // Special case for W==T, we can just read the value directly
                    return packed[<$T>::LANES * row + lane];
                }

                let mask: $T = (1 << (W % <$T>::T)) - 1;
                let start_bit = row * W;
                let start_word = start_bit / <$T>::T;
                let lo_shift = start_bit % <$T>::T;
                let remaining_bits = <$T>::T - lo_shift;

                let lo = packed[<$T>::LANES * start_word + lane] >> lo_shift;
                return if remaining_bits >= W {
                    // in this case we will mask out all bits of hi word
                    lo & mask
                } else {
                    // guaranteed that lo_shift > 0 and thus remaining_bits < T
                    let hi = packed[<$T>::LANES * (start_word + 1) + lane] << remaining_bits;
                    (lo | hi) & mask
                };
            }

            unsafe fn unchecked_unpack_single(width: usize, packed: &[Self], index: usize) -> Self {
                const T: usize = <$T>::T;

                assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);
                assert!(index < 1024, "Index must be less than 1024, got {}", index);
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(packed.len(), packed_len, "Input buffer must be of size {}", packed_len);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / T;
                            return <$T>::unpack_single::<W, B>(unsafe { crate::as_array_unchecked(packed) }, index);
                        },)*
                        // seq_t has exclusive upper bound
                        T => {
                            const W: usize = T;
                            const B: usize = 1024;
                            return <$T>::unpack_single::<W, B>(unsafe { crate::as_array_unchecked(packed) }, index);
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }

            unsafe fn unchecked_unpack_indices(
                width: usize,
                packed: &[Self],
                indices: &[usize],
                output: &mut [MaybeUninit<Self>],
            ) {
                const T: usize = <$T>::T;

                assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);
                assert_eq!(indices.len(), output.len(), "Output length must equal index length");
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(packed.len(), packed_len, "Input buffer must be of size {}", packed_len);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / T;
                            return <$T>::unpack_indices::<W, B>(
                                unsafe { crate::as_array_unchecked(packed) },
                                indices,
                                output,
                            );
                        },)*
                        T => {
                            const W: usize = T;
                            const B: usize = 1024;
                            return <$T>::unpack_indices::<W, B>(
                                unsafe { crate::as_array_unchecked(packed) },
                                indices,
                                output,
                            );
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }
        }
    };
}

// helper function executed at compile-time to speed up unpack_single at runtime
const fn lanes_by_index<T: FastLanes>() -> [u8; 1024] {
    let mut lanes = [0u8; 1024];
    const_for!(i in 0..1024 => {
        lanes[i] = (i % T::LANES) as u8;
    });
    lanes
}

// helper function executed at compile-time to speed up unpack_single at runtime
const fn rows_by_index<T: FastLanes>() -> [u8; 1024] {
    let mut rows = [0u8; 1024];
    const_for!(i in 0..1024 => {
        // This is the inverse of the `index` function from the pack/unpack macros:
        //     fn index(row: usize, lane: usize) -> usize {
        //         let o = row / 8;
        //         let s = row % 8;
        //         (FL_ORDER[o] * 16) + (s * 128) + lane
        //     }
        let lane = i % T::LANES;
        let s = i / 128; // because `(FL_ORDER[o] * 16) + lane` is always < 128
        let fl_order = (i - s * 128 - lane) / 16; // value of FL_ORDER[o]
        let o = FL_ORDER[fl_order]; // because this transposition is invertible!
        rows[i] = (o * 8 + s) as u8;
    });
    rows
}

impl_packing!(u8);
impl_packing!(u16);
impl_packing!(u32);
impl_packing!(u64);

#[cfg(test)]
mod test {
    use core::array;
    use core::fmt::Debug;

    use super::*;
    use alloc::{format, string::ToString, vec, vec::Vec};
    use hegel::TestCase;
    use hegel::generators as gs;
    use hegel::generators::Integer;
    use pastey::paste;

    const BUFFER_SIZE: usize = 1024;

    #[test]
    fn test_unpack_single() {
        let values = array::from_fn(|i| i as u32);
        let mut packed = [0; 512];
        BitPacking::pack::<16, 512>(&values, &mut packed);

        for i in 0..1024 {
            assert_eq!(BitPacking::unpack_single::<16, 512>(&packed, i), values[i]);
            assert_eq!(
                unsafe { BitPacking::unchecked_unpack_single(16, &packed, i) },
                values[i]
            );
        }
    }

    fn assume_initialized<T: Copy>(output: &[MaybeUninit<T>]) -> Vec<T> {
        output
            .iter()
            .map(|value| {
                // SAFETY: callers use this helper only after an unpack method initializes every
                // output element.
                unsafe { value.assume_init() }
            })
            .collect()
    }

    fn assert_u32_unpack_indices(indices: &[usize]) {
        const WIDTH: usize = 13;
        const PACKED_LENGTH: usize = 1024 * WIDTH / u32::T;

        let values = array::from_fn(|index| ((index as u32).wrapping_mul(17)) & 0x1fff);
        let mut packed = [0; PACKED_LENGTH];
        BitPacking::pack::<WIDTH, PACKED_LENGTH>(&values, &mut packed);
        let expected = indices
            .iter()
            .map(|&index| values[index])
            .collect::<Vec<_>>();

        let mut output = vec![MaybeUninit::uninit(); indices.len()];
        BitPacking::unpack_indices::<WIDTH, PACKED_LENGTH>(&packed, indices, &mut output);
        assert_eq!(assume_initialized(&output), expected);

        let mut output = vec![MaybeUninit::uninit(); indices.len()];
        // SAFETY: `packed` contains exactly one packed FastLanes block.
        unsafe {
            BitPacking::unchecked_unpack_indices(WIDTH, &packed, indices, &mut output);
        }
        assert_eq!(assume_initialized(&output), expected);
    }

    #[test]
    fn test_unpack_indices_empty() {
        assert_u32_unpack_indices(&[]);
    }

    #[test]
    fn test_unpack_indices_single() {
        assert_u32_unpack_indices(&[731]);
    }

    #[test]
    fn test_unpack_indices_duplicates() {
        assert_u32_unpack_indices(&[17, 17, 511, 17, 511]);
    }

    #[test]
    fn test_unpack_indices_unordered() {
        assert_u32_unpack_indices(&[1023, 0, 700, 17, 512, 511, 1]);
    }

    #[test]
    fn test_unpack_indices_full_block() {
        assert_u32_unpack_indices(&(0..1024).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "Output length must equal index length")]
    fn test_unpack_indices_rejects_output_length_mismatch() {
        let packed = [0_u32; 32];
        let mut output = [MaybeUninit::uninit(); 1];
        BitPacking::unpack_indices::<1, 32>(&packed, &[], &mut output);
    }

    #[test]
    #[should_panic(expected = "Output length must equal index length")]
    fn test_unchecked_unpack_indices_rejects_output_length_mismatch() {
        let packed = [0_u32; 32];
        let mut output = [MaybeUninit::uninit(); 1];
        // SAFETY: the packed input has the required length. The mismatched output is a documented
        // panic condition, not a safety requirement.
        unsafe { BitPacking::unchecked_unpack_indices(1, &packed, &[], &mut output) };
    }

    #[test]
    #[should_panic(expected = "Index must be less than 1024")]
    fn test_unpack_single_rejects_invalid_index_at_zero_width() {
        u32::unpack_single::<0, 0>(&[], 1024);
    }

    #[test]
    #[should_panic(expected = "Index must be less than 1024")]
    fn test_unchecked_unpack_single_rejects_invalid_index_at_zero_width() {
        // SAFETY: the zero-width packed representation contains no elements. The invalid index is
        // a documented panic condition, not a safety requirement.
        unsafe { u32::unchecked_unpack_single(0, &[], 1024) };
    }

    #[test]
    #[should_panic(expected = "Index must be less than 1024")]
    fn test_unchecked_unpack_indices_rejects_invalid_index_at_zero_width() {
        let mut output = [MaybeUninit::<u32>::uninit()];
        // SAFETY: the zero-width packed representation contains no elements. The invalid index is
        // a documented panic condition, not a safety requirement.
        unsafe { BitPacking::unchecked_unpack_indices(0, &[], &[1024], &mut output) };
    }

    #[test]
    #[should_panic(expected = "Width must be less than or equal to 32")]
    fn test_unchecked_unpack_indices_rejects_invalid_width() {
        // SAFETY: an invalid width is a documented panic condition, not a safety requirement.
        unsafe { u32::unchecked_unpack_indices(33, &[], &[], &mut []) };
    }

    fn assert_bitpack_roundtrip<T>(tc: &TestCase)
    where
        T: BitPacking + Debug + Integer + 'static,
    {
        let input = tc.draw(
            gs::vecs(gs::integers::<T>())
                .min_size(BUFFER_SIZE)
                .max_size(BUFFER_SIZE),
        );

        for width in 0..=T::T {
            let mut packed_output = vec![T::one(); (BUFFER_SIZE * width) / T::T];
            let mut unpacked_output = vec![T::one(); BUFFER_SIZE];
            unsafe { T::unchecked_pack(width, &input, &mut packed_output) };
            unsafe { T::unchecked_unpack(width, &packed_output, &mut unpacked_output) };

            let mask = if width == 0 {
                T::zero()
            } else if width == T::T {
                T::max_value()
            } else {
                (T::one() << width) - T::one()
            };
            let expected = input
                .iter()
                .copied()
                .map(|value| value & mask)
                .collect::<Vec<_>>();

            assert_eq!(
                expected,
                unpacked_output,
                "roundtrip failed for type={} width={width}",
                core::any::type_name::<T>(),
            );
        }
    }

    fn assert_bitpack_repack_roundtrip<T>(tc: &TestCase)
    where
        T: BitPacking + Debug + Integer + 'static,
    {
        let packed_source = tc.draw(
            gs::vecs(gs::integers::<T>())
                .min_size(BUFFER_SIZE)
                .max_size(BUFFER_SIZE),
        );

        for width in 0..=T::T {
            let packed_length = (BUFFER_SIZE * width) / T::T;
            let packed_input = &packed_source[..packed_length];
            let mut unpacked_output = vec![T::one(); BUFFER_SIZE];
            let mut repacked_output = vec![T::one(); packed_length];
            unsafe {
                T::unchecked_unpack(width, packed_input, &mut unpacked_output);
                T::unchecked_pack(width, &unpacked_output, &mut repacked_output);
            }

            assert_eq!(
                packed_input,
                repacked_output,
                "repack roundtrip failed for type={} width={width}",
                core::any::type_name::<T>(),
            );
        }
    }

    fn assert_bitpack_unpack_single_matches_bulk<T>(tc: &TestCase)
    where
        T: BitPacking + Debug + Integer + 'static,
    {
        let packed_source = tc.draw(
            gs::vecs(gs::integers::<T>())
                .min_size(BUFFER_SIZE)
                .max_size(BUFFER_SIZE),
        );
        let index = tc.draw(
            gs::integers::<usize>()
                .min_value(0)
                .max_value(BUFFER_SIZE - 1),
        );

        for width in 0..=T::T {
            let packed_length = (BUFFER_SIZE * width) / T::T;
            let packed_input = &packed_source[..packed_length];
            let mut unpacked_output = vec![T::one(); BUFFER_SIZE];
            unsafe { T::unchecked_unpack(width, packed_input, &mut unpacked_output) };

            assert_eq!(
                unsafe { T::unchecked_unpack_single(width, packed_input, index) },
                unpacked_output[index],
                "single unpack failed for type={} width={width} index={index}",
                core::any::type_name::<T>(),
            );
        }
    }

    fn assert_bitpack_unpack_indices_matches_bulk<T>(tc: &TestCase)
    where
        T: BitPacking + Debug + Integer + 'static,
    {
        let packed_source = tc.draw(
            gs::vecs(gs::integers::<T>())
                .min_size(BUFFER_SIZE)
                .max_size(BUFFER_SIZE),
        );
        let indices = tc.draw(
            gs::vecs(
                gs::integers::<usize>()
                    .min_value(0)
                    .max_value(BUFFER_SIZE - 1),
            )
            .min_size(0)
            .max_size(128),
        );

        for width in 0..=T::T {
            let packed_length = (BUFFER_SIZE * width) / T::T;
            let packed_input = &packed_source[..packed_length];
            let mut unpacked = vec![T::zero(); BUFFER_SIZE];
            // SAFETY: both buffers have the required lengths for `width`.
            unsafe { T::unchecked_unpack(width, packed_input, &mut unpacked) };
            let expected = indices
                .iter()
                .map(|&index| unpacked[index])
                .collect::<Vec<_>>();
            let mut output = vec![MaybeUninit::uninit(); indices.len()];

            // SAFETY: `packed_input` contains exactly one packed FastLanes block.
            unsafe {
                T::unchecked_unpack_indices(width, packed_input, &indices, &mut output);
            }

            assert_eq!(
                assume_initialized(&output),
                expected,
                "indexed unpack failed for type={} width={width} indices={indices:?}",
                core::any::type_name::<T>(),
            );
        }
    }

    fn reference_pack<T>(width: usize, input: &[T]) -> Vec<T>
    where
        T: BitPacking,
    {
        let mut packed = vec![T::zero(); (BUFFER_SIZE * width) / T::T];

        for lane in 0..T::LANES {
            for row in 0..T::T {
                let order = row / 8;
                let sub_row = row % 8;
                let input_idx = (FL_ORDER[order] * 16) + (sub_row * 128) + lane;

                for bit in 0..width {
                    if ((input[input_idx] >> bit) & T::one()) != T::zero() {
                        let packed_bit = row * width + bit;
                        let word = packed_bit / T::T;
                        let word_bit = packed_bit % T::T;
                        packed[word * T::LANES + lane] =
                            packed[word * T::LANES + lane] | (T::one() << word_bit);
                    }
                }
            }
        }

        packed
    }

    fn assert_bitpack_matches_reference<T>(tc: &TestCase)
    where
        T: BitPacking + Debug + Integer + 'static,
    {
        let input = tc.draw(
            gs::vecs(gs::integers::<T>())
                .min_size(BUFFER_SIZE)
                .max_size(BUFFER_SIZE),
        );

        for width in 0..=T::T {
            let mut packed = vec![T::one(); (BUFFER_SIZE * width) / T::T];
            unsafe { T::unchecked_pack(width, &input, &mut packed) };

            assert_eq!(packed, reference_pack(width, &input));
        }
    }

    macro_rules! bitpack_property_tests {
        ($property:ident, $test_cases:literal for $($type:ident),+ $(,)?) => {
            paste! {
                $(
                    #[hegel::test(test_cases = $test_cases)]
                    fn [<test_ $property _ $type>](tc: TestCase) {
                        [<assert_ $property>]::<$type>(&tc);
                    }
                )+
            }
        };
    }

    bitpack_property_tests!(bitpack_roundtrip, 10 for u8, u16, u32, u64);
    bitpack_property_tests!(bitpack_repack_roundtrip, 10 for u8, u16, u32, u64);
    bitpack_property_tests!(bitpack_unpack_single_matches_bulk, 10 for u8, u16, u32, u64);
    bitpack_property_tests!(bitpack_unpack_indices_matches_bulk, 10 for u8, u16, u32, u64);
    bitpack_property_tests!(bitpack_matches_reference, 10 for u8, u16, u32, u64);
}
