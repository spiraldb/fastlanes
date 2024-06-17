use crate::{bitpack, bitunpack, BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};
use paste::paste;

pub trait FoR: BitPacking {
    fn for_bitpack<const W: usize>(
        input: &[Self; 1024],
        reference: Self,
        output: &mut [Self; 1024 * W / Self::T],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    fn unfor_bitpack<const W: usize>(
        input: &[Self; 1024 * W / Self::T],
        reference: Self,
        output: &mut [Self; 1024],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;
}

macro_rules! impl_for {
    ($T:ty) => {
        paste! {
            impl FoR for $T {
                fn for_bitpack<const W: usize>(
                    input: &[Self; 1024],
                    reference: Self,
                    output: &mut [Self; 1024 * W / Self::T],
                ) where
                    BitPackWidth<W>: SupportedBitPackWidth<Self>,
                {
                    for lane in 0..Self::LANES {
                        bitpack!($T, W, output, lane, |$idx| {
                            input[$idx].wrapping_sub(reference)
                        });
                    }
                }

                fn unfor_bitpack<const W: usize>(
                    input: &[Self; 1024 * W / Self::T],
                    reference: Self,
                    output: &mut [Self; 1024],
                ) where
                    BitPackWidth<W>: SupportedBitPackWidth<Self>,
                {
                    for lane in 0..Self::LANES {
                        bitunpack!($T, W, input, lane, |$idx, $elem| {
                            output[$idx] = $elem.wrapping_add(reference)
                        });
                    }
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
    use core::mem::size_of;

    #[test]
    fn test_ffor() {
        const W: usize = 15;
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << W)) as u16;
        }

        let mut packed = [0; 128 * W / size_of::<u16>()];
        FoR::for_bitpack::<W>(&values, 10, &mut packed);

        let mut unpacked = [0; 1024];
        BitPacking::bitunpack::<W>(&packed, &mut unpacked);

        for (i, (a, b)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                // Check that the unpacked array is 10 less than the original (modulo 2^15)
                a.wrapping_sub(10) & ((1 << W) - 1),
                *b,
                "Mismatch at index {}",
                i
            );
        }
    }
}
