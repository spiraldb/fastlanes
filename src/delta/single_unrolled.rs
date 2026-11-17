//! The bodies of [`Delta::undelta_pack_single`](crate::Delta::undelta_pack_single) with the
//! `unroll` feature: an unrolled copy for each width, with every shift and mask a constant. Faster,
//! especially for wide deltas, but one copy of the code per width.

use crate::{FastLanes, seq_t, unpack};
use pastey::paste;

/// Defines `undelta_pack_single_<T>` for `width` in `1..=T`, with `input` of length
/// `1024 * width / T`.
macro_rules! undelta_pack_single_impl {
    ($T:ty) => {
        paste! {
            /// [`Delta::undelta_pack_single`](crate::Delta::undelta_pack_single) with an unrolled copy for each width: every shift
            /// and mask is a constant, which is faster, especially for wide deltas, but takes one
            /// copy of the code per width.
            // `#[inline]` lets the trait method in another codegen unit inline this; without
            // it the unrolled version runs up to about 50% more instructions.
            #[inline]
            pub(super) fn [<undelta_pack_single_ $T>](
                width: usize,
                input: &[$T],
                base: $T,
                lane: usize,
                row: usize,
            ) -> $T {
                /// Undeltas the element at `row` of `lane` from deltas packed at `W` bits.
                #[inline(always)]
                fn unrolled<const W: usize, const B: usize>(
                    input: &[$T; B],
                    base: $T,
                    lane: usize,
                    row: usize,
                ) -> $T {
                    let mut value = base;
                    let mut current = 0;
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        let _ = $idx;
                        if current <= row {
                            value = value.wrapping_add($elem);
                        }
                        current += 1;
                    });
                    value
                }

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            // SAFETY: the length of `input` was checked to be `1024 * W / T`.
                            unrolled::<W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                base,
                                lane,
                                row,
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        <$T>::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            // SAFETY: as above.
                            unrolled::<W, B>(
                                unsafe { crate::as_array_unchecked(input) },
                                base,
                                lane,
                                row,
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }
        }
    };
}

undelta_pack_single_impl!(u8);
undelta_pack_single_impl!(u16);
undelta_pack_single_impl!(u32);
undelta_pack_single_impl!(u64);
