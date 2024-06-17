#![allow(unused_assignments)]

use crate::{iterate, unpack, BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};
use paste::paste;

pub trait Delta: BitPacking {
    fn delta(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]);

    fn undelta(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]);

    fn undelta_pack<const W: usize>(
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
                fn delta(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]) {
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
                fn undelta(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]) {
                    for lane in 0..Self::LANES {
                        let mut prev = base[lane];
                        iterate!($T, lane, |$idx| {
                            let next = input[$idx].wrapping_add(prev);
                            output[$idx] = next;
                            prev = next;
                        });
                    }
                }

                #[inline(never)]
                fn undelta_pack<const W: usize>(
                    input: &[Self; 1024 * W / Self::T],
                    base: &[Self; Self::LANES],
                    output: &mut [Self; 1024],
                ) where
                    BitPackWidth<W>: SupportedBitPackWidth<Self>,
                {
                    for lane in 0..Self::LANES {
                        let mut prev = base[lane];
                        unpack!($T, W, input, lane, |$idx, $elem| {
                            let next = $elem.wrapping_add(prev);
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

        let mut deltas = [0; 1024];
        Delta::delta(&transposed, &[0; 64], &mut deltas);

        let mut packed = [0; 128 * W / size_of::<u16>()];
        BitPacking::pack::<W>(&deltas, &mut packed);

        // Fused kernel
        let mut unpacked = [0; 1024];
        Delta::undelta_pack::<W>(&packed, &[0; 64], &mut unpacked);
        assert_eq!(transposed, unpacked);

        // Unfused kernel
        BitPacking::unpack::<W>(&packed, &mut unpacked);
        let mut undelta = [0; 1024];
        Delta::undelta(&unpacked, &[0; 64], &mut undelta);
        assert_eq!(transposed, undelta);
    }
}
