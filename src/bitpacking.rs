use core::mem::size_of;

use arrayref::{array_mut_ref, array_ref};
use paste::paste;

use crate::{pack, seq_t, unpack, FastLanes, Pred, Satisfied, FL_ORDER};

pub struct BitPackWidth<const W: usize>;
pub trait SupportedBitPackWidth<T> {}
impl<const W: usize, T> SupportedBitPackWidth<T> for BitPackWidth<W> where
    Pred<{ W <= 8 * size_of::<T>() }>: Satisfied
{
}

/// `BitPack` into a compile-time known bit-width.
pub trait BitPacking: FastLanes {
    /// Packs 1024 elements into W bits each.
    /// The output is given as Self to ensure correct alignment.
    fn pack<const W: usize>(input: &[Self; 1024], output: &mut [Self; 1024 * W / Self::T])
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    /// Packs 1024 elements into `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of exactly length 1024. The output slice must be of length
    /// `1024 * W / T`, where `T` is the bit-width of Self and `W` is the packed width.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks 1024 elements from `W` bits each.
    fn unpack<const W: usize>(input: &[Self; 1024 * W / Self::T], output: &mut [Self; 1024])
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    /// Unpacks 1024 elements from `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements.
    fn unpack_single<const W: usize>(packed: &[Self; 1024 * W / Self::T], index: usize) -> Self
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements,
    /// where `W` is runtime-known instead of compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_single(width: usize, input: &[Self], index: usize) -> Self;
}

macro_rules! impl_packing {
    ($T:ty) => {
        paste! {
            impl BitPacking for $T {
                #[inline(never)] // Makes it easier to disassemble and validate ASM.
                fn pack<const W: usize>(
                    input: &[Self; 1024],
                    output: &mut [Self; 1024 * W / Self::T],
                ) where BitPackWidth<W>: SupportedBitPackWidth<Self> {
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

                    seq_t!(W in $T {
                        match width {
                            #(W => Self::pack::<W>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024 * W / <$T>::T],
                            ),)*
                            // seq_t has exclusive upper bound
                            Self::T => Self::pack::<{ Self::T }>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024],
                            ),
                            _ => unreachable!("Unsupported width: {}", width)
                        }
                    })
                }

                #[inline(never)]
                fn unpack<const W: usize>(
                    input: &[Self; 1024 * W / Self::T],
                    output: &mut [Self; 1024],
                ) where BitPackWidth<W>: SupportedBitPackWidth<Self> {
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

                    seq_t!(W in $T {
                        match width {
                            #(W => Self::unpack::<W>(
                                array_ref![input, 0, 1024 * W / <$T>::T],
                                array_mut_ref![output, 0, 1024],
                            ),)*
                            // seq_t has exclusive upper bound
                            Self::T => Self::unpack::<{ Self::T }>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024],
                            ),
                            _ => unreachable!("Unsupported width: {}", width)
                        }
                    })
                }

                fn unpack_single<const W: usize>(packed: &[Self; 1024 * W / Self::T], index: usize) -> Self
                where
                    BitPackWidth<W>: SupportedBitPackWidth<Self>
                {
                    // We can think of the input array as effectively a row-major 2-D array of with
                    // Self::LANES columns and Self::T rows.
                    // Meanwhile, the packed array is (logically) a *column-major* 2-D
                    // array of 128 columns and 8 rows, each of which has W-bits.
                    // The ordering of the elements in the packed array is transposed to match
                    // the required layout for delta and other more complex encodings.
                    //
                    // First step, we need to get the transposed index
                    let row = index / Self::LANES;
                    let lane = index % Self::LANES;
                    let transposed_index = {
                        let o = row / 8;
                        let s = row % 8;
                        (FL_ORDER[o] * 16) + (s * 128) + lane
                    };

                    // From the transposed index, we can get the correct start bit within the packed array
                    let start_bit = transposed_index * W;

                    // we read one or two T-bit words from the lane, depending on how our target
                    // W-bit value overlaps with the T-bit words
                    let start_word = start_bit / Self::T;
                    let end_word_inclusive = (start_bit + W - 1) / Self::T;

                    // shift and mask the correct bits from the T-bit words
                    let lo_shift = start_bit % Self::T;
                    let lo = packed[start_word] >> lo_shift;

                    let hi_shift = (Self::T - lo_shift) % Self::T;
                    let hi = packed[end_word_inclusive] << hi_shift;

                    let mask: Self = ((1 as Self) << W) - 1;
                    (lo | hi) & mask
                }

                unsafe fn unchecked_unpack_single(width: usize, input: &[Self], index: usize) -> Self {
                    let packed_len = 128 * width / size_of::<Self>();
                    debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size {}", packed_len);
                    debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);
                    debug_assert!(index <= 1024, "index must be less than or equal to 1024");

                    seq_t!(W in $T {
                        match width {
                            #(W => Self::unpack_single::<W>(
                                array_ref![input, 0, 1024 * W / <$T>::T],
                                index,
                            ),)*
                            // seq_t has exclusive upper bound
                            Self::T => Self::unpack_single::<{ Self::T }>(
                                array_ref![input, 0, 1024],
                                index,
                            ),
                            _ => unreachable!("Unsupported width: {}", width)
                        }
                    })
                }
            }
        }
    };
}

impl_packing!(u8);
impl_packing!(u16);
impl_packing!(u32);
impl_packing!(u64);

#[cfg(test)]
mod test {
    use core::array;
    use core::fmt::Debug;
    use core::mem::size_of;

    use seq_macro::seq;

    use super::*;

    #[test]
    fn test_unchecked_pack() {
        let input = array::from_fn(|i| i as u32);
        let mut packed = [0; 320];
        unsafe { BitPacking::unchecked_pack(10, &input, &mut packed) };
        let mut output = [0; 1024];
        unsafe { BitPacking::unchecked_unpack(10, &packed, &mut output) };
        assert_eq!(input, output);
    }

    #[test]
    fn test_unpack_single() {
        let values = array::from_fn(|i| i as u32);
        let mut packed = [0; 512];
        BitPacking::pack::<16>(&values, &mut packed);

        for i in 0..1024 {
            assert_eq!(BitPacking::unpack_single::<16>(&packed, i), values[i]);
            assert_eq!(
                unsafe { BitPacking::unchecked_unpack_single(16, &packed, i) },
                values[i]
            );
        }
    }

    fn try_round_trip<T: BitPacking + Debug, const W: usize>()
    where
        BitPackWidth<W>: SupportedBitPackWidth<T>,
        [(); 128 * W / size_of::<T>()]:,
        [(); 1024 * W / T::T]:,
    {
        let mut values: [T; 1024] = [T::zero(); 1024];
        for i in 0..1024 {
            values[i] = T::from(i % (1 << (W % T::T))).unwrap();
        }

        let mut packed = [T::zero(); 1024 * W / T::T];
        BitPacking::pack::<W>(&values, &mut packed);

        let mut unpacked = [T::zero(); 1024];
        BitPacking::unpack::<W>(&packed, &mut unpacked);

        assert_eq!(&unpacked, &values);
    }

    macro_rules! impl_try_round_trip {
        ($T:ty, $W:expr) => {
            paste! {
                #[test]
                fn [<test_round_trip_ $T _ $W>]() {
                    try_round_trip::<$T, $W>();
                }
            }
        };
    }

    seq!(W in 0..=8 { impl_try_round_trip!(u8, W); });
    seq!(W in 0..=16 { impl_try_round_trip!(u16, W); });
    seq!(W in 0..=32 { impl_try_round_trip!(u32, W); });
    seq!(W in 0..=64 { impl_try_round_trip!(u64, W); });
}
