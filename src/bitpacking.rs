use crate::{bitpack, bitunpack, seq_t, FastLanes, Pred, Satisfied};
use arrayref::{array_mut_ref, array_ref};
use num_traits::One;
use paste::paste;
use std::mem::size_of;

pub struct BitPackWidth<const W: usize>;
pub trait SupportedBitPackWidth<T> {}
impl<const W: usize, T> SupportedBitPackWidth<T> for BitPackWidth<W> where
    Pred<{ W <= 8 * size_of::<T>() }>: Satisfied
{
}

/// BitPack into a compile-time known bit-width.
pub trait BitPacking: FastLanes {
    /// Packs 1024 elements into W bits each.
    /// The output is given as Self to ensure correct alignment.
    fn bitpack<const W: usize>(input: &[Self; 1024], output: &mut [Self; 1024 * W / Self::T])
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    unsafe fn unchecked_bitpack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks W-bit elements into 1024 elements.
    fn bitunpack<const W: usize>(input: &[Self; 1024 * W / Self::T], output: &mut [Self; 1024])
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    unsafe fn unchecked_bitunpack(width: usize, input: &[Self], output: &mut [Self]);

    fn bitunpack_single<const W: usize>(packed: &[Self; 1024 * W / Self::T], index: usize) -> Self
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>,
        Self: One,
    {
        // TODO(ngates): implement this function to not unpack the world.
        let mut output = [Self::zero(); 1024];
        Self::bitunpack::<W>(packed, &mut output);
        output[index]
    }
}

macro_rules! impl_bitpacking {
    ($T:ty) => {
        paste! {
            impl BitPacking for $T {
                #[inline(never)] // Makes it easier to disassemble and validate ASM.
                fn bitpack<const W: usize>(
                    input: &[Self; 1024],
                    output: &mut [Self; 1024 * W / Self::T],
                ) where BitPackWidth<W>: SupportedBitPackWidth<Self> {
                    for lane in 0..Self::LANES {
                        bitpack!($T, W, output, lane, |$idx| {
                            input[$idx]
                        });
                    }
                }

                unsafe fn unchecked_bitpack(width: usize, input: &[Self], output: &mut [Self]) {
                    let packed_len = 128 * width / size_of::<Self>();
                    debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
                    debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
                    debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                    seq_t!(W in $T {
                        match width {
                            #(W => Self::bitpack::<W>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024 * W / <$T>::T],
                            ),)*
                            // seq_t has exclusive upper bound
                            Self::T => Self::bitpack::<{ Self::T }>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024],
                            ),
                            _ => unreachable!("Unsupported width: {}", width)
                        }
                    })
                }

                #[inline(never)]
                fn bitunpack<const W: usize>(
                    input: &[Self; 1024 * W / Self::T],
                    output: &mut [Self; 1024],
                ) where BitPackWidth<W>: SupportedBitPackWidth<Self> {
                    for lane in 0..Self::LANES {
                        bitunpack!($T, W, input, lane, |$idx, $elem| {
                            output[$idx] = $elem
                        });
                    }
                }

                unsafe fn unchecked_bitunpack(width: usize, input: &[Self], output: &mut [Self]) {
                    let packed_len = 128 * width / size_of::<Self>();
                    debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                    debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                    debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                    seq_t!(W in $T {
                        match width {
                            #(W => Self::bitunpack::<W>(
                                array_ref![input, 0, 1024 * W / <$T>::T],
                                array_mut_ref![output, 0, 1024],
                            ),)*
                            // seq_t has exclusive upper bound
                            Self::T => Self::bitunpack::<{ Self::T }>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024],
                            ),
                            _ => unreachable!("Unsupported width: {}", width)
                        }
                    })
                }
            }
        }
    };
}

impl_bitpacking!(u8);
impl_bitpacking!(u16);
impl_bitpacking!(u32);
impl_bitpacking!(u64);

#[cfg(test)]
mod test {
    use super::*;
    use seq_macro::seq;
    use std::array;
    use std::fmt::Debug;
    use std::mem::size_of;

    #[test]
    fn test_unchecked_bitpack() {
        let input = (0u32..1024).collect::<Vec<_>>();
        let mut packed = [0; 320];
        unsafe { BitPacking::unchecked_bitpack(10, &input, &mut packed) };
        let mut output = [0; 1024];
        unsafe { BitPacking::unchecked_bitunpack(10, &packed, &mut output) };
        assert_eq!(input, output);
    }

    #[test]
    fn test_bitunpack_single() {
        let values = array::from_fn(|i| i as u32);
        let mut packed = [0; 512];
        BitPacking::bitpack::<16>(&values, &mut packed);

        for i in 0..1024 {
            let unpacked = BitPacking::bitunpack_single::<16>(&packed, i);
            println!("P {} {}", i, unpacked);
            //assert_eq!(unpacked, values[i]);
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
        BitPacking::bitpack::<W>(&values, &mut packed);

        let mut unpacked = [T::zero(); 1024];
        BitPacking::bitunpack::<W>(&packed, &mut unpacked);

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
