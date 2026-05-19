use crate::seq_t;
use crate::{supported_bit_width, FastLanes, FastLanesComparable, FL_ORDER};
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

#[inline]
fn output_byte_index(row: usize, lane: usize) -> usize {
    let o = row / 8;
    let s = row % 8;
    let idx = (FL_ORDER[o] * 16) + (s * 128) + lane;
    let word = idx / 64;
    let byte = (idx % 64) / 8;

    #[cfg(target_endian = "little")]
    return word * 8 + byte;
    #[cfg(not(target_endian = "little"))]
    return word * 8 + (7 - byte);
}

#[inline]
fn pack_8_predicates(chunk: u64) -> u64 {
    // Pack the low bit of each byte into the low byte.
    ((chunk & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) & 0xff
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
                #[inline(always)]
                fn input_word<const B: usize>(input: &[$T; B], word: usize, lane: usize) -> $T {
                    unsafe { *input.get_unchecked(<$T>::LANES * word + lane) }
                }

                #[inline(always)]
                fn low_mask(width: usize) -> $T {
                    if width == 0 {
                        0
                    } else {
                        <$T>::MAX >> (<$T>::T - width)
                    }
                }

                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                let output_bytes = output.as_mut_ptr().cast::<u8>();

                for lane_base in (0..<$T>::LANES).step_by(8) {
                    if W == 0 {
                        let byte = if f(V::as_unpacked(0), other) { 0xff } else { 0 };
                        for row in 0..<$T>::T {
                            unsafe {
                                *output_bytes.add(output_byte_index(row, lane_base)) = byte;
                            }
                        }
                        continue;
                    }

                    let mut srcs = [0 as $T; 8];
                    for bit in 0..8 {
                        srcs[bit] = input_word(input, 0, lane_base + bit);
                    }

                    for row in 0..<$T>::T {
                        let mut predicates = 0u64;

                        if W == <$T>::T {
                            for bit in 0..8 {
                                let lane = lane_base + bit;
                                let elem = input_word(input, row, lane);
                                predicates |=
                                    u64::from(f(V::as_unpacked(elem), other)) << (bit * 8);
                            }
                        } else {
                            let mask = low_mask(W);
                            let curr_word = (row * W) / <$T>::T;
                            let next_word = ((row + 1) * W) / <$T>::T;
                            let shift = (row * W) % <$T>::T;

                            for bit in 0..8 {
                                let src = srcs[bit];
                                let mut elem = src >> shift;

                                if next_word > curr_word {
                                    let remaining_bits = ((row + 1) * W) % <$T>::T;
                                    let current_bits = W - remaining_bits;
                                    elem &= low_mask(current_bits);

                                    if next_word < W {
                                        let lane = lane_base + bit;
                                        let next_src = input_word(input, next_word, lane);
                                        srcs[bit] = next_src;
                                        elem |=
                                            (next_src & low_mask(remaining_bits)) << current_bits;
                                    }
                                } else {
                                    elem &= mask;
                                }

                                predicates |=
                                    u64::from(f(V::as_unpacked(elem), other)) << (bit * 8);
                            }
                        }

                        unsafe {
                            *output_bytes.add(output_byte_index(row, lane_base)) =
                                pack_8_predicates(predicates) as u8;
                        }
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

    #[test]
    fn test_pack_8_predicates() {
        for mask in 0u64..=0xff {
            let mut chunk = 0u64;
            for bit in 0..8 {
                chunk |= ((mask >> bit) & 1) << (bit * 8);
            }

            assert_eq!(pack_8_predicates(chunk), mask);
        }
    }

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
    fn test_unpack_eq_u8_w0() {
        type T = u8;
        const W: usize = 0;
        const B: usize = 1024 * W / T::T;

        assert_unpack_eq::<T, W, B>();
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
