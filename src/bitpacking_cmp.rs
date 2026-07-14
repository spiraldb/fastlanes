use crate::seq_t;
use crate::unpack;
use crate::{supported_bit_width, FastLanes, FastLanesComparable};
use paste::paste;

pub trait BitPackingCompare: FastLanes {
    /// Benchmark/reference variant of [`BitPackingCompare::unpack_cmp`] that materializes one
    /// byte-sized `bool` per logical value instead of producing a packed mask.
    ///
    /// This preserves the pre-packed-mask implementation so benchmarks can compare the two
    /// output strategies using the same decoder and compiler revision.
    #[doc(hidden)]
    fn unpack_cmp_byte<const W: usize, const B: usize, V, F>(
        input: &[Self; B],
        output: &mut [bool; 1024],
        comparison: F,
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;

    /// A fused unpack (see `BitPacking::unpack`) and compare, packing the boolean results into a
    /// bitmask of `1024` bits (`16 x u64`).
    ///
    /// This compares, using the comparison function, all of the packed values against a constant
    /// `value`. The values are of type `Self`, whereas the comparison is on the type `V` (where
    /// `V::Bitpacked` = `Self`). This allows for comparison between signed values which are
    /// bit-packed as unsigned ones.
    ///
    /// The output is a bitmask in **`FastLanes` (transposed) order**, not logical row order. The
    /// `1024` bits are `Self::LANES` words of `Self::T` bits, one word per lane laid out
    /// contiguously (little-endian) in the `[u64; 16]`. Within a lane's word the comparison
    /// results are packed LSB-first: row `r` (for `r` in `0..Self::T`) lands at bit `r`, holding
    /// the comparison for the value at logical index `index(row, lane)` (see the `unpack!` macro).
    /// This is the cheapest order to produce: it needs no cross-lane shuffles, just a per-lane
    /// accumulator that the compiler keeps in a (vectorized) register.
    ///
    /// To recover logical row order (e.g. an Arrow-style boolean buffer), pass the result through
    /// [`untranspose_cmp_mask`].
    fn unpack_cmp<const W: usize, const B: usize, V, F>(
        input: &[Self; B],
        output: &mut [u64; 16],
        comparison: F,
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;

    /// A fused unpack (see `BitPacking::unpack`) and compare, packing the boolean results into a
    /// bitmask of `1024` bits (`16 x u64`). See [`BitPackingCompare::unpack_cmp`] for the output
    /// bit ordering.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output is exactly `[u64; 16]` (`1024` bits).
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_cmp<V, F>(
        width: usize,
        input: &[Self],
        output: &mut [u64; 16],
        comparison: F,
        value: V,
    ) where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;
}

macro_rules! impl_packing_compare {
    ($T:ty) => {
        impl BitPackingCompare for $T {
            #[inline(never)]
            fn unpack_cmp_byte<const W: usize, const B: usize, V, F>(
                input: &[Self; B],
                output: &mut [bool; 1024],
                f: F,
                other: V,
            ) where
                V: FastLanesComparable<Bitpacked = Self>,
                F: Fn(V, V) -> bool,
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in 0..Self::LANES {
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        output[$idx] = f(V::as_unpacked($elem), other);
                    });
                }
            }

            #[inline(never)]
            fn unpack_cmp<const W: usize, const B: usize, V, F>(
                input: &[Self; B],
                output: &mut [u64; 16],
                f: F,
                other: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self>,
                F: Fn(V, V) -> bool
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                // The output is 1024 bits laid out as `Self::LANES` words of `Self::T` bits each
                // (which is always 128 bytes == `[u64; 16]`). Each lane owns one contiguous word
                // holding that lane's `Self::T` comparison results, LSB-first: row `r` lands at bit
                // `r`. Per-lane ownership means the accumulator stays in a register and the store is
                // a single contiguous (vectorizable) write per lane -- no `[bool; 1024]`
                // (or `[Self; 1024]`) materialization, no cross-lane shuffles.
                //
                // For `u64` (`Self::LANES == 16`) this LSB-first ordering coincides with the
                // canonical FastLanes transpose; for narrower widths it is the per-width packing
                // that [`crate::bit_transpose::untranspose_bits::<Self>`] inverts. Either way that
                // is what [`untranspose_cmp_mask`] uses to recover logical row order.
                //
                // SAFETY: `[u64; 16]` and `[Self; LANES]` are both exactly 128 bytes, and `u64`'s
                // alignment (8) is >= `Self`'s alignment, so the reinterpret is sound.
                let words: &mut [$T; <$T>::LANES] =
                    unsafe { &mut *output.as_mut_ptr().cast::<[$T; <$T>::LANES]>() };

                for lane in 0..Self::LANES {
                    let mut word: $T = 0;
                    let mut bit: usize = 0;
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let _ = $idx;
                        word |= <$T>::from(f(V::as_unpacked($elem), other)) << bit;
                        #[allow(unused_assignments)]
                        { bit += 1; }
                    });
                    words[lane] = word;
                }
            }

            unsafe fn unchecked_unpack_cmp<V, F>(
                 width: usize,
                 input: &[Self],
                 output: &mut [u64; 16],
                 comparison: F,
                 value: V,
            )
            where
                V: FastLanesComparable<Bitpacked = Self>,
                F: Fn(V, V) -> bool
            {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack_cmp::<W, B, V, F>(
                                unsafe { crate::as_array_unchecked(input) },
                                output,
                                comparison,
                                value
                            )
                        },)*
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
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
    use crate::{untranspose_bits, BitPacking};
    use alloc::vec::Vec;
    use core::array;

    /// Reference bitmask in the same `FastLanes` (LSB-first, per-lane) order produced by
    /// `unpack_cmp`:
    /// fully unpack, then for each lane set bit `row` from the comparison of the value at the
    /// logical index `index(row, lane)`.
    fn reference_mask<T, V, F>(packed_unpacked: &[T; 1024], f: F, other: V) -> [u64; 16]
    where
        T: FastLanes,
        V: FastLanesComparable<Bitpacked = T>,
        F: Fn(V, V) -> bool,
    {
        let mut out = [0u64; 16];
        for lane in 0..T::LANES {
            for row in 0..T::T {
                // `index(row, lane)` from the unpack macro.
                let o = row / 8;
                let s = row % 8;
                let idx = (crate::FL_ORDER[o] * 16) + (s * 128) + lane;
                if f(V::as_unpacked(packed_unpacked[idx]), other) {
                    // LSB-first within each lane word: row `r` lands at bit `r`.
                    let bit = lane * T::T + row;
                    out[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
        out
    }

    #[test]
    fn test_unpack_eq() {
        type T = u32;
        const W: usize = 10;
        const B: usize = 1024 * W / T::T;

        let values = array::from_fn(|i| i as T % (1 << W));

        let mut packed = [0; (128 * W) / size_of::<T>()];
        T::pack::<W, B>(&values, &mut packed);

        let mut unpacked = [0u32; 1024];
        T::unpack::<W, B>(&packed, &mut unpacked);

        // Check equality against every value of the vector.
        for v in 0..1024 {
            let cmp = {
                let mut output = [0u64; 16];
                T::unpack_cmp::<W, B, _, _>(&packed, &mut output, |a, b| a == b, v);
                output
            };

            let expected = reference_mask(&unpacked, |a, b| a == b, v);
            assert_eq!(cmp, expected, "Failed == {v}");
        }
    }

    #[test]
    fn test_unpack_cmp_all_widths_and_ops() {
        fn check<T>()
        where
            T: BitPacking + BitPackingCompare + FastLanesComparable<Bitpacked = T>,
        {
            for width in 1..T::T {
                let mask: u64 = if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };
                let values: [T; 1024] = array::from_fn(|i| {
                    T::from((i as u64).wrapping_mul(2_654_435_761) & mask).unwrap()
                });

                let mut packed = Vec::new();
                packed.resize(128 * width / size_of::<T>(), T::zero());
                unsafe { T::unchecked_pack(width, &values, &mut packed) };

                let mut unpacked = [T::zero(); 1024];
                unsafe { T::unchecked_unpack(width, &packed, &mut unpacked) };

                let other = T::from(7u64 & mask).unwrap();
                for (name, f) in [
                    ("eq", (|a: T, b: T| a == b) as fn(T, T) -> bool),
                    ("ne", |a, b| a != b),
                    ("lt", |a, b| a < b),
                    ("le", |a, b| a <= b),
                    ("gt", |a, b| a > b),
                    ("ge", |a, b| a >= b),
                ] {
                    let mut output = [0u64; 16];
                    unsafe {
                        T::unchecked_unpack_cmp(width, &packed, &mut output, f, other);
                    }
                    let expected = reference_mask(&unpacked, f, other);
                    assert_eq!(
                        output,
                        expected,
                        "type={} width={width} op={name}",
                        core::any::type_name::<T>()
                    );

                    // Untransposing the mask must yield logical row order: bit `i` is the
                    // comparison for logical value `i` (i.e. `collect_bool` semantics).
                    let mut logical = [0u64; 16];
                    untranspose_bits::<T>(&output, &mut logical);
                    let mut expected_logical = [0u64; 16];
                    for i in 0..1024 {
                        if f(T::as_unpacked(unpacked[i]), other) {
                            expected_logical[i / 64] |= 1u64 << (i % 64);
                        }
                    }
                    assert_eq!(
                        logical,
                        expected_logical,
                        "untranspose type={} width={width} op={name}",
                        core::any::type_name::<T>()
                    );
                }
            }
        }

        check::<u8>();
        check::<u16>();
        check::<u32>();
        check::<u64>();
    }
}
