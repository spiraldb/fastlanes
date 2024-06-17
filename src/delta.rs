#![allow(unused_assignments)]
use crate::{bitpack, bitunpack, BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};
use paste::paste;

pub trait Delta: BitPacking {
    fn delta<const W: usize>(
        input: &[Self; 1024],
        base: &[Self; Self::LANES],
        output: &mut [Self; 1024 * W / Self::T],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    fn undelta<const W: usize>(
        input: &[Self; 1024 * W / Self::T],
        base: &[Self; Self::LANES],
        output: &mut [Self; 1024],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;
}

macro_rules! impl_delta {
    ($T:ty) => {
        paste! {
            impl Delta for $T {
                #[inline(never)]
                fn delta<const W: usize>(
                    input: &[Self; 1024],
                    base: &[Self; Self::LANES],
                    output: &mut [Self; 1024 * W / Self::T],
                ) where
                    BitPackWidth<W>: SupportedBitPackWidth<Self>,
                {
                    for lane in 0..Self::LANES {
                        let mut prev = base[lane];
                        bitpack!($T, W, output, lane, |$idx| {
                            let next = input[$idx];
                            let out = next.saturating_sub(prev);
                            prev = next;
                            out
                        });
                    }
                }

                #[inline(never)]
                fn undelta<const W: usize>(
                    input: &[Self; 1024 * W / Self::T],
                    base: &[Self; Self::LANES],
                    output: &mut [Self; 1024],
                ) where
                    BitPackWidth<W>: SupportedBitPackWidth<Self>,
                {
                    for lane in 0..Self::LANES {
                        let mut prev = base[lane];
                        bitunpack!($T, W, input, lane, |$idx, $elem| {
                            let next = $elem.saturating_add(prev);
                            output[$idx] = next;
                            prev = next;
                        });
                    }
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
    use core::mem::size_of;

    #[test]
    fn test_delta() {
        const W: usize = 15;
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i / 8) as u16;
        }

        let mut transposed = [0; 1024];
        Transpose::transpose(&values, &mut transposed);

        let mut packed = [0; 128 * W / size_of::<u16>()];
        Delta::delta::<W>(&transposed, &[0; 64], &mut packed);

        let mut unpacked = [0; 1024];
        Delta::undelta::<W>(&packed, &[0; 64], &mut unpacked);

        let mut untransposed = [0; 1024];
        Transpose::untranspose(&unpacked, &mut untransposed);

        assert_eq!(values, untransposed);
    }
}
