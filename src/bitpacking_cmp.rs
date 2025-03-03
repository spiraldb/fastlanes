#![allow(unused_variables)]
#![allow(dead_code)]
use crate::FastLanes;
use crate::{BitPackWidth, BitPacking, SupportedBitPackWidth};

pub trait BitPackingCompare: BitPacking {
    fn unpack_cmp_impl<const W: usize, F: Fn(Self, Self) -> bool>(
        input: &[Self; 1024 * W / Self::T],
        output: &mut [bool; 1024],
        f: F,
        eq_value: Self,
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    #[inline(never)]
    fn unpack_cmp<const W: usize, F: Fn(Self, Self) -> bool>(
        input: &[Self; 1024 * W / Self::T],
        output: &mut [bool; 1024],
        comparison: F,
        value: Self,
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>,
        [(); 1024 / Self::T]:,
    {
        // The number of bits in the output == number of bits in the new_output.
        assert_eq!(
            16 * u64::BITS as usize,
            1024 / Self::T * size_of::<Self>() * 8_usize
        );
        //let new_output =
        //            unsafe { &mut *(ptr::from_mut::<[u64; 16]>(output)).cast::<[Self; 1024 / Self::T]>() };
        Self::unpack_cmp_impl(input, output, comparison, value);
    }

    /// A fused unpack (see `BitPacking::unpack`) compare and pack into bit bools.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length `[u64; 16]` (`1024` bits).
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp<F: Fn(Self, Self) -> bool>(
        width: usize,
        input: &[Self],
        output: &mut [bool; 1024],
        comparison: F,
        value: Self,
    );
}

macro_rules! impl_packing_compare {
    ($T:ty) => {
        paste::paste! {
            impl BitPackingCompare for $T {
               #[inline(always)]
                fn unpack_cmp_impl<const W: usize, F: Fn(Self, Self) -> bool>(
                    input: &[Self; 1024 * W / Self::T],
                    output: &mut [bool; 1024],
                    f: F,
                    other: Self,
                ) where BitPackWidth<W>: SupportedBitPackWidth<Self> {
                   for lane in (0..Self::LANES) {
                       $crate::unpack!($T, W, input, lane, |$idx, $elem| {
                           output[$idx] = f($elem, other);
                       });
                   }
                }

               unsafe fn unchecked_unpack_cmp<F: Fn(Self, Self) -> bool>(
                    width: usize,
                    input: &[Self],
                    output: &mut [bool; 1024],
                    comparison: F,
                    value: Self,
               )
               {
                   let packed_len = 128 * width / size_of::<Self>();
                   debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                   debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                   $crate::seq_t!(W in $T {
                        match width {
                            #(W => Self::unpack_cmp::<W, F>(
                                arrayref::array_ref![input, 0, 1024 * W / <$T>::T],
                                output,
                                comparison,
                                value
                            ),)*
                            _ => unreachable!("Unsupported width: {}", width)
                        }
                    })
                }
            }
        }
    };
}

impl_packing_compare!(u8);
impl_packing_compare!(u16);
impl_packing_compare!(u32);
impl_packing_compare!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitPacking;
    use alloc::vec::Vec;
    use core::array;

    #[test]
    fn test_unpack_eq() {
        type T = u32;
        const W: usize = 10;

        let values = array::from_fn(|i| i as T % (1 << W));

        let mut packed = [0; (128 * W) / size_of::<T>()];
        T::pack::<W>(&values, &mut packed);

        // Check equality against every value of the vector
        for v in 0..1024 {
            let cmp = {
                let mut output = [false; 1024];
                T::unpack_cmp::<W, _>(&packed, &mut output, |a, b| a == b, v);
                output
            };

            let expected = values.iter().map(|&x| x == v).collect::<Vec<_>>();

            assert_eq!(cmp.as_slice(), expected.as_slice(), "Failed == {}", v);
        }
    }
}
