#![allow(unused_assignments)]

#[cfg(feature = "delta_for_bitpacking")]
use arrayref::{array_mut_ref, array_ref};
#[cfg(feature = "delta_for_bitpacking")]
use paste::paste;

#[cfg(feature = "delta_for_bitpacking")]
use crate::seq_t;
use crate::{iterate, supported_bit_width, unpack, BitPacking, FastLanes};

pub trait Delta: BitPacking {
    fn delta<const LANES: usize>(
        input: &[Self; 1024],
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    fn undelta<const LANES: usize>(
        input: &[Self; 1024],
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    fn undelta_pack<const LANES: usize, const W: usize, const B: usize>(
        input: &[Self; B],
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    /// Fused decode of a `delta(for(bitpacking))` stack.
    ///
    /// In a single pass over the `W`-bit packed `input`, this unpacks each lane, wrapping-adds the
    /// frame-of-reference `reference` (inverting `FoR`), then accumulates the result against the
    /// running per-lane `base` (inverting delta encoding). This fuses what would otherwise be three
    /// passes — `unpack`, `unfor`, and `undelta` — into one, avoiding two intermediate buffers.
    ///
    /// Available only with the `delta_for_bitpacking` feature; the API is not yet covered by semver guarantees.
    #[cfg(feature = "delta_for_bitpacking")]
    fn unfor_undelta_pack<const LANES: usize, const W: usize, const B: usize>(
        input: &[Self; B],
        reference: Self,
        base: &[Self; LANES],
        output: &mut [Self; 1024],
    );

    /// Fused decode of a `delta(for(bitpacking))` stack where the packed width `W` is only known at
    /// runtime. Dispatches to [`Delta::unfor_undelta_pack`] for the matching compile-time width.
    ///
    /// # Safety
    /// `input` must have length `1024 * width / T` (where `T` is the bit-width of `Self`), `base`
    /// must have length `Self::LANES`, and `output` must have length exactly 1024. These lengths are
    /// checked only with `debug_assert` (i.e., not checked on release builds).
    ///
    /// Available only with the `delta_for_bitpacking` feature; the API is not yet covered by semver guarantees.
    #[cfg(feature = "delta_for_bitpacking")]
    unsafe fn unchecked_unfor_undelta_pack(
        width: usize,
        input: &[Self],
        reference: Self,
        base: &[Self],
        output: &mut [Self],
    );
}

macro_rules! impl_delta {
    ($T:ty) => {
        impl Delta for $T {
            #[inline(never)]
            fn delta<const LANES: usize>(
                input: &[Self; 1024],
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                }

                for lane in 0..Self::LANES {
                    let mut prev = base[lane];
                    iterate!($T, lane, |$idx| {
                        let next = input[$idx];
                        output[$idx] = next.wrapping_sub(prev);
                        prev = next;
                    });
                }
            }

            #[inline(never)]
            fn undelta<const LANES: usize>(
                input: &[Self; 1024],
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                }
                for lane in 0..LANES {
                    let mut prev = base[lane];
                    iterate!($T, lane, |$idx| {
                        let next = input[$idx].wrapping_add(prev);
                        output[$idx] = next;
                        prev = next;
                    });
                }
            }

            #[inline(never)]
            fn undelta_pack<const LANES: usize, const W: usize, const B: usize>(
                input: &[Self; B],
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in 0..Self::LANES {
                    let mut prev = base[lane];
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let next = $elem.wrapping_add(prev);
                        output[$idx] = next;
                        prev = next;
                    });
                }
            }

            #[cfg(feature = "delta_for_bitpacking")]
            #[inline(never)]
            fn unfor_undelta_pack<const LANES: usize, const W: usize, const B: usize>(
                input: &[Self; B],
                reference: Self,
                base: &[Self; LANES],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(LANES == Self::LANES);
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in 0..Self::LANES {
                    let mut prev = base[lane];
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let next = $elem.wrapping_add(reference).wrapping_add(prev);
                        output[$idx] = next;
                        prev = next;
                    });
                }
            }

            #[cfg(feature = "delta_for_bitpacking")]
            unsafe fn unchecked_unfor_undelta_pack(
                width: usize,
                input: &[Self],
                reference: Self,
                base: &[Self],
                output: &mut [Self],
            ) {
                const LANES: usize = <$T>::LANES;
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert_eq!(base.len(), LANES, "Base buffer must be of size LANES");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                let base = array_ref![base, 0, LANES];
                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unfor_undelta_pack::<LANES, W, B>(
                                array_ref![input, 0, B],
                                reference,
                                base,
                                array_mut_ref![output, 0, 1024],
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::unfor_undelta_pack::<LANES, W, B>(
                                array_ref![input, 0, 1024],
                                reference,
                                base,
                                array_mut_ref![output, 0, 1024],
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
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
    use crate::Transpose;
    use core::mem::size_of;

    #[test]
    fn test_delta() {
        const LANES: usize = u16::LANES;
        const W: usize = 15;
        const B: usize = 1024 * W / u16::T;

        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i / 8) as u16;
        }

        let mut transposed = [0; 1024];
        Transpose::transpose(&values, &mut transposed);

        let mut deltas = [0; 1024];
        Delta::delta(&transposed, &[0; 64], &mut deltas);

        let mut packed = [0; 128 * W / size_of::<u16>()];
        BitPacking::pack::<W, B>(&deltas, &mut packed);

        // Fused kernel
        let mut unpacked = [0; 1024];
        Delta::undelta_pack::<LANES, W, B>(&packed, &[0; 64], &mut unpacked);
        assert_eq!(transposed, unpacked);

        // Unfused kernel
        BitPacking::unpack::<W, B>(&packed, &mut unpacked);
        let mut undelta = [0; 1024];
        Delta::undelta(&unpacked, &[0; 64], &mut undelta);
        assert_eq!(transposed, undelta);
    }

    #[cfg(feature = "delta_for_bitpacking")]
    #[test]
    fn test_unfor_undelta() {
        const LANES: usize = u16::LANES;
        const W: usize = 9;
        const B: usize = 1024 * W / u16::T;
        const REFERENCE: u16 = 5;

        // Arbitrary FoR-encoded deltas that fit in W bits.
        let mut for_deltas: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            for_deltas[i] = (i % (1 << W)) as u16;
        }

        let base = [11u16; LANES];

        // Independently reconstruct the expected output: FoR-decode (add reference) then undelta.
        let mut deltas = [0u16; 1024];
        for i in 0..1024 {
            deltas[i] = for_deltas[i].wrapping_add(REFERENCE);
        }
        let mut expected = [0u16; 1024];
        Delta::undelta::<LANES>(&deltas, &base, &mut expected);

        // Bit-pack the FoR-encoded deltas and decode them with the fused kernel.
        let mut packed = [0; 128 * W / size_of::<u16>()];
        BitPacking::pack::<W, B>(&for_deltas, &mut packed);

        let mut unpacked = [0u16; 1024];
        Delta::unfor_undelta_pack::<LANES, W, B>(&packed, REFERENCE, &base, &mut unpacked);
        assert_eq!(expected, unpacked);

        // The runtime-width dispatch must agree with the const-generic kernel.
        let mut unpacked_dyn = [0u16; 1024];
        unsafe {
            Delta::unchecked_unfor_undelta_pack(W, &packed, REFERENCE, &base, &mut unpacked_dyn);
        }
        assert_eq!(expected, unpacked_dyn);
    }
}
