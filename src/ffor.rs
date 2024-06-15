use crate::{bitpack, bitunpack, BitPackWidth, BitPacking, FastLanes, SupportedBitPackWidth};

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
            bitpack!(u16, W, output, lane, |$idx| {
                input[$idx].wrapping_sub(reference)
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
        for lane in 0..Self::LANES {
            bitunpack!(u16, W, input, lane, |$idx, $elem| {
                output[$idx] = $elem.wrapping_add(reference)
            });
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_ffor() {
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << 15)) as u16;
        }

        let mut packed = [0; 128 * 15 / size_of::<u16>()];
        FusedFOR::ffor::<15>(&values, 10, &mut packed);

        println!("PACKED: {:?}", packed);

        let mut unpacked = [0; 1024];
        for lane in 0..u16::LANES {
            bitunpack!(u16, 15, packed, lane, |$idx, $elem| {
                unpacked[$idx] = $elem;
            });
        }

        for (i, (a, b)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                // Check that the unpacked array is 10 less than the original (modulo 2^15)
                a.wrapping_sub(10) & ((1 << 15) - 1),
                *b,
                "Mismatch at index {}",
                i
            );
        }
    }
}
