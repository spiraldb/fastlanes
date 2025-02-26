use crate::FastLanes;
use crate::{BitPackWidth, BitPacking, SupportedBitPackWidth};

pub trait BitPackingCompare: BitPacking {
    fn unpack_cmp_impl<const W: usize, F: Fn(Self, Self) -> bool>(
        input: &[Self; 1024 * W / Self::T],
        output: &mut [Self; 1024 / Self::T],
        f: F,
        eq_value: Self,
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>;

    #[inline(never)]
    fn unpack_cmp<const W: usize, F: Fn(Self, Self) -> bool>(
        input: &[Self; 1024 * W / Self::T],
        output: &mut [u64; 16],
        comparison: F,
        value: Self,
    ) where
        BitPackWidth<W>: SupportedBitPackWidth<Self>,
        [(); 1024 / Self::T]:, // [(); 1024 * W / Self::T]:,
    {
        // The number of bits in the output == number of bits in the new_output.
        assert_eq!(
            16 * u64::BITS as usize,
            1024 / Self::T * size_of::<Self>() * 8 as usize
        );
        let new_output = unsafe {
            // &mut *core::ptr::from_mut::<[u64; 16]>(output).cast::<[Self; 1024 / Self::T]>()
            core::mem::transmute::<&mut [u64; 16], &mut [Self; 1024 / Self::T]>(output)
        };
        Self::unpack_cmp_impl(input, new_output, comparison, value);
    }

    unsafe fn unchecked_unpack_cmp<F: Fn(Self, Self) -> bool>(
        width: usize,
        input: &[Self],
        output: &mut [u64; 16],
        comparison: F,
        value: Self,
    ) where
        [(); 1024 / Self::T]:;
}

macro_rules! impl_packing_compare {
    ($T:ty) => {
        paste::paste! {
            impl BitPackingCompare for $T {
               #[inline(always)]
                fn unpack_cmp_impl<const W: usize, F: Fn(Self, Self) -> bool>(
                    input: &[Self; 1024 * W / Self::T],
                    output: &mut [Self; 1024 / Self::T],
                    f: F,
                    other: Self,
                ) where BitPackWidth<W>: SupportedBitPackWidth<Self> {
                    for lane in (0..Self::LANES){
                        $crate::unpack!($T, W, input, lane, |$idx, $elem| {
                            let bool_idx = $idx / Self::T;
                            let bool_bit = $idx % Self::T;
                            let value = f($elem, other);
                            output[bool_idx] |= (num_traits::AsPrimitive::<Self>::as_(value)) << bool_bit;
                        });
                    }
                }

               unsafe fn unchecked_unpack_cmp<F: Fn(Self, Self) -> bool>(
                    width: usize,
                    input: &[Self],
                    output: &mut [u64; 16],
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

// TODO(joe): fix this.
// Do not impl this for u8/u16 as its currently slower.
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
        const W: usize = 3;

        let values = array::from_fn(|i| i as T % 32);

        let mut packed = [0; (128 * W) / size_of::<T>()];
        T::pack::<W>(&values, &mut packed);

        let cmp = {
            let mut output = [0u64; 16];
            T::unpack_cmp::<W, _>(&packed, &mut output, |a, b| a == b, 4);
            output
        };

        let cmp_unchecked = {
            let mut output = [0u64; 16];
            unsafe {
                T::unchecked_unpack_cmp::<_>(W, &packed, &mut output, |a, b| a == b, 4);
            }
            output
        };

        let bools = {
            let mut unpacked = [0; 1024];
            T::unpack::<W>(&packed, &mut unpacked);
            collect_bool(unpacked.len(), |idx| unpacked[idx] == 4)
        };

        assert_eq!(cmp.as_slice(), bools.as_slice());
        assert_eq!(cmp_unchecked.as_slice(), bools.as_slice());
    }

    #[inline]
    pub fn ceil(value: usize, divisor: usize) -> usize {
        // Rewrite as `value.div_ceil(&divisor)` after
        // https://github.com/rust-lang/rust/issues/88581 is merged.
        value / divisor + usize::from(0 != value % divisor)
    }

    #[inline]
    pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, mut f: F) -> Vec<u64> {
        let mut buffer = Vec::with_capacity(ceil(len, 64) * 8);

        let chunks = len / 64;
        let remainder = len % 64;
        for chunk in 0..chunks {
            let mut packed = 0;
            for bit_idx in 0..64 {
                let i = bit_idx + chunk * 64;
                packed |= u64::from(f(i)) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            buffer.push(packed);
        }

        if remainder != 0 {
            let mut packed = 0;
            for bit_idx in 0..remainder {
                let i = bit_idx + chunks * 64;
                packed |= u64::from(f(i)) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            buffer.push(packed);
        }

        buffer.truncate(ceil(len, 8));
        buffer
    }
}
