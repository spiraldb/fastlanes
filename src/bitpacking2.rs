use paste::paste;

use crate::{bitpack, bitunpack, BitPackWidth, FastLanes, SupportedBitPackWidth};

/// BitPack into a compile-time known bit-width.
pub trait BitPacking2: FastLanes {
    /// Packs 1024 elements into W bits each.
    /// The output is given as Self to ensure correct alignment.
    fn bitpack<const W: usize>(input: &[Self; 1024], output: &mut [Self; 1024 * W / Self::T])
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    /// Unpacks W-bit elements into 1024 elements.
    fn bitunpack<const W: usize>(input: &[Self; 1024 * W / Self::T], output: &mut [Self; 1024])
    where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;
}

// We need to use a macro instead of generic impl since we have to know the bit-width of T ahead
// of time.
macro_rules! impl_bitpacking {
    ($T:ty) => {
        paste! {
            impl BitPacking2 for $T {
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
    use std::fmt::Debug;
    use std::mem::size_of;

    fn try_round_trip<T: BitPacking2 + Debug, const W: usize>()
    where
        BitPackWidth<W>: SupportedBitPackWidth<T>,
        [(); 128 * W / size_of::<T>()]:,
        [(); 1024 * W / T::T]:,
    {
        let mut values: [T; 1024] = [T::zero(); 1024];
        for i in 0..1024 {
            values[i] = T::from(i % (1 << W)).unwrap();
        }

        let mut packed = [T::zero(); 1024 * W / T::T];
        BitPacking2::bitpack::<W>(&values, &mut packed);

        let mut unpacked = [T::zero(); 1024];
        BitPacking2::bitunpack::<W>(&packed, &mut unpacked);

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

    seq!(W in 0..8 { impl_try_round_trip!(u8, W); });
    seq!(W in 0..16 { impl_try_round_trip!(u16, W); });
    seq!(W in 0..32 { impl_try_round_trip!(u32, W); });
    seq!(W in 0..64 { impl_try_round_trip!(u64, W); });
}
