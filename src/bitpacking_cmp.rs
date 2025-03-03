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
        [(); 1024 / Self::T]:, // [(); 1024 * W / Self::T]:,
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

#[inline]
pub fn collect_bits_dumb(buffer: &[bool; 1024]) -> [u64; 16] {
    let mut packed = [0; 16];
    for (i, &bit) in buffer.iter().enumerate() {
        packed[i / 64] |= (bit as u64) << (i % 64);
    }
    packed
}

#[inline]
pub fn collect_bits(bools: &[bool; 1024]) -> [u64; 16] {
    let mut result = [0u64; 16];

    // Process in larger chunks
    for chunk in 0..16 {
        let chunk_base = chunk * 64;
        let mut value = 0u64;

        for i in 0..64 {
            value |= (bools[chunk_base + i] as u64) << i;
        }

        result[chunk] = value;
    }

    result
}

#[inline]
pub fn collect_bits_v2(bools: &[bool; 1024]) -> [u64; 16] {
    let mut result = [0u64; 16];

    for u32_idx in 0..32 {
        let mut u32_val: u32 = 0;
        let base_idx = u32_idx * 32;

        // Process a 32-bit chunk - good SIMD target
        for bit_idx in 0..32 {
            u32_val |= (bools[base_idx + bit_idx] as u32) << bit_idx;
        }

        // Place the 32-bit value into the appropriate 64-bit slot
        let u64_idx = u32_idx / 2;
        let shift = (u32_idx % 2) * 32;
        result[u64_idx] |= (u32_val as u64) << shift;
    }

    result
}

#[inline]
pub fn collect_bits_v3(bools: &[bool; 1024]) -> [u64; 16] {
    let mut result = [0u64; 16];

    // Process the entire array in one pass with linear indexing
    // This pattern is more likely to be recognized for SIMD optimization
    for i in 0..1024 {
        let chunk_idx = i / 64;
        let bit_pos = i % 64;

        // Use a branchless operation that's more SIMD-friendly
        // Convert bool to u64 (0 or 1) and shift to position
        let bit_value = bools[i] as u64;
        result[chunk_idx] |= bit_value << bit_pos;
    }

    result
}

// TODO(joe): fix this.
// Do not impl this for u8/u16 as its currently slower.
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
