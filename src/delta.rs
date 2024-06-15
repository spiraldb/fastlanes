use crate::{seq_s, FastLanes, FL_ORDER};
use paste::paste;
use seq_macro::seq;

pub trait Delta: FastLanes {
    /// Delta-encode the input array. The input array should already be transposed.
    fn delta(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]);

    /// Delta-decode the input array. The output array will be in transposed order.
    fn undelta(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]);
}

macro_rules! impl_delta {
    ($T:ty) => {
        paste! {
            impl Delta for $T {
                #[allow(unused_assignments)]
                #[inline(never)]
                fn encode(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]) {
                    for i in 0..Self::LANES {
                        let mut prev = base[i];

                        seq_s!(o in $T {
                             seq!(row in 0..8 {
                                // NOTE(ngates): 128 elems in 8 x 8x16 blocks.
                                let pos = (FL_ORDER[o] * 16) + (128 * row) + i;
                                let next = input[pos];
                                output[pos] = next.wrapping_sub(prev);
                                prev = next;
                            });
                        });
                    }
                }

                #[allow(unused_assignments)]
                #[inline(never)]
                fn decode(input: &[Self; 1024], base: &[Self; Self::LANES], output: &mut [Self; 1024]) {
                    for i in 0..Self::LANES {
                        let mut prev = base[i];

                        seq_s!(o in $T {
                             seq!(row in 0..8 {
                                let pos = (FL_ORDER[o] * 16) + (128 * row) + i;
                                let next = input[pos].wrapping_add(prev);
                                output[pos] = next;
                                prev = next;
                            });
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
    use crate::test::round_robin_values;
    use crate::Transpose;
    use std::fmt::Debug;

    #[test]
    fn test_delta() {
        test_delta_typed::<u8>();
        test_delta_typed::<u16>();
        test_delta_typed::<u32>();
        test_delta_typed::<u64>();
    }

    #[inline(never)]
    fn test_delta_typed<T: Delta + Debug>()
    where
        [(); T::LANES]:,
    {
        let values: [T; 1024] = round_robin_values();

        let mut input = [T::zero(); 1024];
        Transpose::transpose(&values, &mut input);

        // Encode with base of zeros
        let base = [T::zero(); T::LANES];
        let mut encoded = [T::zero(); 1024];
        Delta::encode(&input, &base, &mut encoded);

        let mut decoded = [T::zero(); 1024];
        Delta::decode(&encoded, &base, &mut decoded);
        assert_eq!(input, decoded);

        let mut decoded_values = [T::zero(); 1024];
        Transpose::untranspose(&decoded, &mut decoded_values);
        assert_eq!(values, decoded_values);
    }

    #[test]
    fn delta_non_increasing() {
        test_delta_non_increasing::<u8>();
        test_delta_non_increasing::<u16>();
        test_delta_non_increasing::<u32>();
        test_delta_non_increasing::<u64>();
    }

    fn test_delta_non_increasing<T: Delta + Debug>()
    where
        [(); T::LANES]:,
    {
        // We skip transposing such that the values aren't necessarily increasing
        let values: [T; 1024] = round_robin_values::<T>();

        // Encode with base of zeros
        let base = [T::zero(); T::LANES];
        let mut encoded = [T::zero(); 1024];
        Delta::encode(&values, &base, &mut encoded);

        let mut decoded = [T::zero(); 1024];
        Delta::decode(&encoded, &base, &mut decoded);
        assert_eq!(values, decoded);
    }
}
