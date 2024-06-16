#![allow(unused_assignments)]
use crate::{bitpack, bitunpack, BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};

pub trait FusedDelta: BitPacking {
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

// While we experiment with the FusedDelta, we will only implement it for u16.
impl FusedDelta for u16 {
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
            bitpack!(u16, W, output, lane, |$idx| {
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
            bitunpack!(u16, W, input, lane, |$idx, $elem| {
                println!("{} {}", $idx, $elem);
                let next = $elem.saturating_add(prev);
                output[$idx] = next;
                prev = next;
            });
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Transpose;
    use std::mem::size_of;

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
        FusedDelta::delta::<W>(&transposed, &[0; 64], &mut packed);

        let mut unpacked = [0; 1024];
        FusedDelta::undelta::<W>(&packed, &[0; 64], &mut unpacked);

        let mut untransposed = [0; 1024];
        Transpose::untranspose(&unpacked, &mut untransposed);

        assert_eq!(values, untransposed);
    }
}
