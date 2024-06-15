use crate::bitpacking::bitpack;
use crate::{BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};
use seq_macro::seq;

pub trait FusedFOR<const W: usize>: BitPacking<W>
where
    BitPackWidth<W>: SupportedBitPackWidth<Self>,
{
    fn ffor(input: &[Self; 1024], reference: Self, output: &mut [Self; 1024 * W / Self::T]);
    fn unffor(input: &[Self; 1024 * W / Self::T], reference: Self, output: &mut [Self; 1024]);
}

impl<const W: usize> FusedFOR<W> for u16
where
    BitPackWidth<W>: SupportedBitPackWidth<Self>,
{
    fn ffor(input: &[Self; 1024], reference: Self, output: &mut [Self; 1024 * W / Self::T]) {
        for lane in 0..Self::LANES {
            bitpack!(u16, W, input, output, lane, |$src| {
                $src.wrapping_sub(reference)
            });
        }
    }

    fn unffor(_input: &[Self; 1024 * W / Self::T], _reference: Self, _output: &mut [Self; 1024]) {
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
        FusedFOR::<15>::ffor(&values, 10, &mut packed);

        let mut unpacked = [0; 1024];
        BitPacking::<15>::bitunpack(&packed, &mut unpacked);

        println!("{:?}", &values.iter().zip(&unpacked).collect::<Vec<_>>());
    }
}
