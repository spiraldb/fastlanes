/// This module contains an alternative macro-based implementation of packing.
///
/// Warning: it is NOT wire compatible with the original `FastLanes` implementation.
///
/// It differs in that it iterates over the elements respecting the transposed ordering.
/// While this doesn't make a difference for bit-packing, it means this same implementation can
/// be used to easily generated fused kernels with transposed encodings such as delta.
///
/// Essentially this means: BitPack(Delta(Transpose(V))) == Delta+BitPack(Transpose(V))

#[macro_export]
macro_rules! iterate {
    ($T:ty, $lane: expr, | $_1:tt $idx:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER};
            use paste::paste;

            #[inline(always)]
            fn index(row: usize, lane: usize) -> usize {
                let o = row / 8;
                let s = row % 8;
                (FL_ORDER[o] * 16) + (s * 128) + lane
            }

            paste!(seq_t!(row in $T {
                let idx = index(row, $lane);
                __kernel__!(idx);
            }));
        }
    }
}

#[macro_export]
macro_rules! pack {
    ($T:ty, $W:expr, $packed:expr, $lane:expr, | $_1:tt $idx:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER};
            use paste::paste;

            // The number of bits of T.
            const T: usize = <$T>::T;

            #[inline(always)]
            fn index(row: usize, lane: usize) -> usize {
                let o = row / 8;
                let s = row % 8;
                (FL_ORDER[o] * 16) + (s * 128) + lane
            }

            if $W == 0 {
                // Nothing to do if W is 0, since the packed array is zero bytes.
            } else if $W == T {
                // Special case for W=T, we can just copy the input value directly to the packed value.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    $packed[<$T>::LANES * row + $lane] = __kernel__!(idx);
                }));
            } else {
                // A mask of W bits.
                let mask: $T = (1 << $W) - 1;

                // First we loop over each lane in the virtual 1024 bit word.
                let mut tmp: $T = 0;

                // Loop over each of the rows of the lane.
                // Inlining this loop means all branches are known at compile time and
                // the code is auto-vectorized for SIMD execution.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    let src = __kernel__!(idx);
                    let src = src & mask;

                    // Shift the src bits into their position in the tmp output variable.
                    if row == 0 {
                        tmp = src;
                    } else {
                        tmp |= src << (row * $W) % T;
                    }

                    // If the next packed position is after our current one, then we have filled
                    // the current output and we can write the packed value.
                    let curr_word: usize = (row * $W) / T;
                    let next_word: usize = ((row + 1) * $W) / T;

                    #[allow(unused_assignments)]
                    if next_word > curr_word {
                        $packed[<$T>::LANES * curr_word + $lane] = tmp;
                        let remaining_bits: usize = ((row + 1) * $W) % T;
                        // Keep the remaining bits for the next packed value.
                        tmp = src >> $W - remaining_bits;
                    }
                }));
            }
        }
    };
}

#[macro_export]
macro_rules! unpack {
    ($T:ty, $W:expr, $packed:expr, $lane:expr, | $_1:tt $idx:ident, $_2:tt $elem:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident, $_2 $elem:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER};
            use paste::paste;

            // The number of bits of T.
            const T: usize = <$T>::T;

            #[inline(always)]
            fn index(row: usize, lane: usize) -> usize {
                let o = row / 8;
                let s = row % 8;
                (FL_ORDER[o] * 16) + (s * 128) + lane
            }

            if $W == 0 {
                // Special case for W=0, we just need to zero the output.
                // We'll still respect the iteration order in case the kernel has side effects.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    let zero: $T = 0;
                    __kernel__!(idx, zero);
                }));
            } else if $W == T {
                // Special case for W=T, we can just copy the packed value directly to the output.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    let src = $packed[<$T>::LANES * row + $lane];
                    __kernel__!(idx, src);
                }));
            } else {
                #[inline]
                fn mask(width: usize) -> $T {
                    if width == T { <$T>::MAX } else { (1 << (width % T)) - 1 }
                }

                let mut src: $T = $packed[$lane];
                let mut tmp: $T;

                paste!(seq_t!(row in $T {
                    // Figure out the packed positions
                    let curr_word: usize = (row * $W) / T;
                    let next_word = ((row + 1) * $W) / T;

                    let shift = (row * $W) % T;

                    if next_word > curr_word {
                        // Consume some bits from the curr packed input, the remainder are in the next
                        // packed input value
                        let remaining_bits = ((row + 1) * $W) % T;
                        let current_bits = $W - remaining_bits;
                        tmp = (src >> shift) & mask(current_bits);

                        if next_word < $W {
                            // Load the next packed value
                            src = $packed[<$T>::LANES * next_word + $lane];
                            // Consume the remaining bits from the next input value.
                            tmp |= (src & mask(remaining_bits)) << current_bits;
                        }
                    } else {
                        // Otherwise, just grab W bits from the src value
                        tmp = (src >> shift) & mask($W);
                    }

                    // Write out the unpacked value
                    let idx = index(row, $lane);
                    __kernel__!(idx, tmp);
                }));
            }
        }
    };
}

#[macro_export]
macro_rules! unpack_single {
    ($T:ty, $W:expr, $packed:expr, $index:expr, | $_1:tt $elem:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $elem:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER, FastLanes};
            use paste::paste;

            // The number of bits of T.
            const T: usize = <$T>::T;

            // This calculation of (lane, row) is the inverse of the `index` function from the
            // pack/unpack macros
            #[inline(always)]
            const fn lane_and_row<const INDEX: usize>() -> (usize, usize) {
                const lane: usize = INDEX % <$T>::LANES;
                const row: usize = {
                    let s = INDEX / 128; // because `(FL_ORDER[o] * 16) + lane` is always < 128
                    let fl_order = (INDEX - s * 128 - lane) / 16; // value of FL_ORDER[o]
                    let o = FL_ORDER[fl_order]; // because this transposition is invertible!
                    o * 8 + s
                };
                (lane, row)
            }

            fn unpack_single_const_helper<const START_BIT: usize, const ONE_WORD: bool>(
                packed: &[$T], lane: usize, mask: Self) -> Self
            where
                Pred< { START_BIT < T * T }> : Satisfied
            {
                let start_word = START_BIT / Self::T;
                let lo_shift = START_BIT % Self::T;
                let lo = packed[Self::LANES * start_word + lane] >> lo_shift;
                if ONE_WORD {
                    lo & mask
                } else {
                    let hi_shift = Self::T - lo_shift; // guaranteed that lo_shift > 0 if ONE_WORD == false
                    let hi = packed[Self::LANES * (start_word + 1) + lane] << hi_shift;
                    (lo | hi) & mask
                }
            }

            if $W == 0 {
                // Special case for W=0, we just need to zero the output.
                // We'll still respect the iteration order in case the kernel has side effects.
                let zero: $T = 0;
                __kernel__!(zero);
            } else {
                let (lane, row): (usize, usize) = seq!(I in 0..1024 {
                        match index {
                            #(I =>
                                lane_and_row::<I>(),
                            )*
                            _ => unreachable!("Unsupported index: {}", index)
                        }
                    });

                // Special case for W=T, we can just copy the packed value directly to the output.
                if $W == T {
                    let val = $packed[<$T>::LANES * row + lane];
                    __kernel__!(val);
                } else {
                    const mask: usize = (1 << ($W % T)) - 1;
                    paste!(seq_t!(ROW in $T {
                        match row {
                            #(ROW => {
                                const START_BIT: usize = ROW * $W;
                                const REMAINING_BITS: usize = T - (START_BIT % T);
                                const ONE_WORD: bool = REMAINING_BITS <= $W;
                                let val = unpack_single_const_helper::<START_BIT, ONE_WORD>($packed, lane, mask);
                                __kernel__!(val);
                            },)*
                            _ => unreachable!("Unsupported row: {}", row)
                        }
                    }))
                }
            }
        }
    };
}

#[cfg(test)]
mod test {
    use crate::{BitPacking, FastLanes};

    #[test]
    fn test_pack() {
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << 15)) as u16;
        }

        let mut packed: [u16; 960] = [0; 960];
        for lane in 0..u16::LANES {
            // Always loop over lanes first. This is what the compiler vectorizes.
            pack!(u16, 15, packed, lane, |$pos| {
                values[$pos]
            });
        }

        let mut packed_orig: [u16; 960] = [0; 960];
        BitPacking::pack::<15>(&values, &mut packed_orig);

        let mut unpacked: [u16; 1024] = [0; 1024];
        for lane in 0..u16::LANES {
            // Always loop over lanes first. This is what the compiler vectorizes.
            unpack!(u16, 15, packed, lane, |$idx, $elem| {
                unpacked[$idx] = $elem;
            });
        }

        assert_eq!(values, unpacked);
    }
}
