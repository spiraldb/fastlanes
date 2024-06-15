/// This module contains an alternative macro-based implementation of bitpacking.
///
/// Warning: it is NOT wire compatible with the original FastLanes implementation.
///
/// It differs in that it iterates over the elements respecting the transposed ordering.
/// While this doesn't make a difference for bit-packing, it means this same implementation can
/// be used to easily generated fused kernels with transposed encodings such as delta.
///
/// Essentially this means: BitPack(Delta(Transpose(V))) == Delta+BitPack(Transpose(V))
use crate::FastLanes;

#[macro_export]
macro_rules! bitpack {
    ($T:ty, $W:expr, $packed:expr, $lane:expr, | $_1:tt $idx:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER};
            use paste::paste;

            let mask = (1 << $W) - 1;

            // First we loop over each lane in the virtual 1024 bit word.
            let mut tmp: $T = 0;

            // Loop over each of the rows of the lane.
            // Inlining this loop means all branches are known at compile time and
            // the code is auto-vectorized for SIMD execution.
            paste!(seq_t!(row in $T {
                let o = row / 8;
                let s = row % 8;
                let src_idx = (FL_ORDER[o] * 16) + (s * 128) + $lane;

                let src = __kernel__!(src_idx);
                let src = src & mask;

                // Shift the src bits into their position in the tmp output variable.
                if row == 0 {
                    tmp = src;
                } else {
                    tmp |= src << (row * $W) % <$T>::T;
                }

                // If the next packed position is after our current one, then we have filled
                // the current output and we can write the packed value.
                let curr_pos: usize = (row * $W) / <$T>::T;
                let next_pos: usize = ((row + 1) * $W) / <$T>::T;

                #[allow(unused_assignments)]
                if next_pos > curr_pos {
                    $packed[<$T>::LANES * curr_pos + $lane] = tmp;

                    let remaining_bits: usize = ((row + 1) * $W) % <$T>::T;
                    tmp = src >> $W - remaining_bits;
                }
            }));
        }
    };
}

#[macro_export]
macro_rules! bitunpack {
    ($T:ty, $W:expr, $packed:expr, $lane:expr, | $_1:tt $idx:ident, $_2:tt $elem:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident, $_2 $elem:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER};
            use paste::paste;

            let mut src = $packed[$lane];
            let mut tmp: $T;

            #[inline]
            fn mask(width: usize) -> $T {
                (1 << width) - 1
            }

            paste!(seq_t!(row in $T {
                // Figure out the packed positions
                let curr_pos: usize = (row * $W) / <$T>::T;
                let next_pos = ((row + 1) * $W) / <$T>::T;

                let shift = (row * $W) % <$T>::T;

                if next_pos > curr_pos {
                    // Consume some bits from the curr packed input, the remainder are in the next
                    // packed input value
                    let remaining_bits = ((row + 1) * $W) % <$T>::T;
                    let current_bits = $W - remaining_bits;
                    tmp = (src >> shift) & mask(current_bits);

                    if next_pos < $W {
                        // Load the next packed value
                        src = $packed[<$T>::LANES * next_pos + $lane];
                        // Consume the remaining bits from the next input value.
                        tmp |= (src & mask(remaining_bits)) << current_bits;
                    }
                } else {
                    // Otherwise, just grab W bits from the src value
                    tmp = (src >> shift) & mask($W);
                }

                // Write out the unpacked value
                let o = row / 8;
                let s = row % 8;
                let idx = (FL_ORDER[o] * 16) + (s * 128) + $lane;
                __kernel__!(idx, tmp);
            }));
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::BitPacking;

    #[test]
    fn test_pack() {
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << 15)) as u16;
        }

        let mut packed: [u16; 960] = [0; 960];
        for lane in 0..u16::LANES {
            // Always loop over lanes first. This is what the compiler vectorizes.
            bitpack!(u16, 15, packed, lane, |$pos| {
                values[$pos]
            });
        }

        let mut packed_orig: [u16; 960] = [0; 960];
        BitPacking::bitpack::<15>(&values, &mut packed_orig);
        println!("{:?}", packed_orig);
        println!("{:?}", packed);

        let mut unpacked: [u16; 1024] = [0; 1024];
        for lane in 0..u16::LANES {
            // Always loop over lanes first. This is what the compiler vectorizes.
            bitunpack!(u16, 15, packed, lane, |$idx, $elem| {
                unpacked[$idx] = $elem;
            });
        }

        assert_eq!(values, unpacked);
    }
}
