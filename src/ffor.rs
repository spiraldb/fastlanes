use crate::bitpacking::bitpack;
use crate::{BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};
use seq_macro::seq;

pub trait FusedFOR: BitPacking {
    fn ffor<const W: usize>(
        input: &[Self; 1024],
        reference: Self,
        output: &mut [Self; 1024 * W / Self::T],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    fn unffor<const W: usize>(
        input: &[Self; 1024 * W / Self::T],
        reference: Self,
        output: &mut [Self; 1024],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;
}

impl FusedFOR for u16 {
    fn ffor<const W: usize>(
        input: &[Self; 1024],
        reference: Self,
        output: &mut [Self; 1024 * W / Self::T],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>,
    {
        for lane in 0..Self::LANES {
            bitpack!(u16, W, output, lane, |$pos| {
                input[$pos].wrapping_sub(reference)
            });
        }
    }

    fn unffor<const W: usize>(
        input: &[Self; 1024 * W / Self::T],
        reference: Self,
        output: &mut [Self; 1024],
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>,
    {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::BitPacking;
    use std::mem::size_of;

    #[test]
    fn test_ffor() {
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << 15)) as u16;
        }

        let mut packed = [0; 128 * 15 / size_of::<u16>()];
        FusedFOR::ffor::<15>(&values, 10, &mut packed);

        let mut unpacked = [0; 1024];
        BitPacking::bitunpack::<15>(&packed, &mut unpacked);

        println!("{:?}", &values.iter().zip(&unpacked).collect::<Vec<_>>());
    }
}
