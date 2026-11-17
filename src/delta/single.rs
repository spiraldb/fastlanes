//! The default bodies of [`Delta::undelta_pack_single`](crate::Delta::undelta_pack_single): one
//! small copy for every width, which adds the lane's packed deltas 64 bits at a time.

use crate::FastLanes;
use pastey::paste;

/// `EVEN_DIGITS[width]` has the bits of the even-numbered `width`-bit digits of a `u64` set: bits
/// `0..width`, `2 * width..3 * width`, and so on.
const EVEN_DIGITS: [u64; 64] = {
    let mut even = [0; 64];
    let mut width = 1;
    while width < 64 {
        let mut bit = 0;
        while bit < 64 {
            even[width] |= (u64::MAX >> (64 - width)) << bit;
            bit += 2 * width;
        }
        width += 1;
    }
    even
};

/// `REM_64[width]` is `64 % width`. Dividing takes about as long as adding a lane's deltas does.
const REM_64: [u8; 64] = {
    let mut rem = [0; 64];
    let mut width = 1;
    while width < 64 {
        rem[width] = (64 % width) as u8;
        width += 1;
    }
    rem
};

/// The sum, modulo `2^64`, of the slots of `2 * width` bits that `slots` is made of, where
/// `width` is in `1..64`. No slot may start at or above bit 64.
fn sum_slots(slots: u128, width: usize) -> u64 {
    let mut slot = 2 * width;
    if slot >= 64 {
        return slots as u64;
    }
    // The slots that end at or below bit `whole`, and the one slot that starts there if any.
    let whole = 64 - usize::from(REM_64[slot]);
    let high = (slots >> whole) as u64;
    let mut low = slots as u64 & (u64::MAX >> (64 - whole));
    // Adding each odd-numbered slot to the even-numbered one below it makes slots twice as wide
    // that hold the same total, until one slot holds all of it.
    while slot < 64 {
        low = (low & EVEN_DIGITS[slot]) + ((low >> slot) & EVEN_DIGITS[slot]);
        slot *= 2;
    }
    low.wrapping_add(high)
}

/// Defines `undelta_pack_single_<T>` for `width` in `1..=T`, with `input` of length
/// `1024 * width / T`.
macro_rules! undelta_pack_single_impl {
    ($T:ty) => {
        paste! {
            /// [`Delta::undelta_pack_single`](crate::Delta::undelta_pack_single) with one copy for every width. It adds the lane's
            /// packed deltas 64 bits at a time.
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
                // The words of a lane that hold 64 bits of its packed deltas.
                const WORDS: usize = 64 / <$T>::T;

                let mut value = base;
                if width == <$T>::T {
                    for row in 0..=row {
                        value = value.wrapping_add(input[<$T>::LANES * row + lane]);
                    }
                    return value;
                }

                // Deltas wider than 32 bits fill a 64-bit chunk with at most one pair, so adding
                // them one at a time is as fast.
                if width > 32 {
                    let mask: $T = (1 << width) - 1;
                    for row in 0..=row {
                        let bit = row * width;
                        let (word, shift) = (bit / <$T>::T, bit % <$T>::T);
                        let mut delta = input[<$T>::LANES * word + lane] >> shift;
                        if shift + width > <$T>::T {
                            delta |= input[<$T>::LANES * (word + 1) + lane] << (<$T>::T - shift);
                        }
                        value = value.wrapping_add(delta & mask);
                    }
                    return value;
                }

                // Add the deltas 64 bits of the lane at a time, rather than one at a time. A
                // chunk of 64 bits starts with `head` bits that end a delta (all of it, when
                // `head` is `width`), which go to `ends`. The deltas above those start at bit 0
                // once shifted down, so adding the odd-numbered ones to the even-numbered ones
                // leaves a sum of two deltas in each slot of `2 * width` bits. The slots of all
                // the chunks add up in `slots`, and `sum_slots` adds the slots to one another.
                //
                // A slot does not overflow: a chunk adds less than `2 << width` to it, and the
                // at most `T * width / 64 <= width` chunks of the lane add less than
                // `2 * width << width <= 1 << (2 * width)`. A slot cut short by the top of the
                // chunk carries on into the high half of `slots`.
                let even = EVEN_DIGITS[width];
                let mask = u64::MAX >> (64 - width);
                // A chunk starts `64 % width` bits further into a delta than the one before it.
                let step = usize::from(REM_64[width]);
                let mut ends = 0u64;
                let mut slots = 0u128;
                let mut head = width;
                let mut add = |chunk: u64| {
                    ends = ends.wrapping_add((chunk << (width - head)) & mask);
                    let deltas = chunk >> head;
                    slots += u128::from((deltas & even) + ((deltas >> width) & even));
                    head = if head > step { head - step } else { head + width - step };
                };

                let bits = (row + 1) * width;
                let mut word = lane;
                for _ in 0..bits / 64 {
                    let mut chunk = 0;
                    for part in 0..WORDS {
                        chunk |= u64::from(input[word]) << (part * <$T>::T);
                        word += <$T>::LANES;
                    }
                    add(chunk);
                }
                let rest = bits % 64;
                if rest != 0 {
                    let mut chunk = 0;
                    for part in 0..rest.div_ceil(<$T>::T) {
                        chunk |= u64::from(input[word]) << (part * <$T>::T);
                        word += <$T>::LANES;
                    }
                    add(chunk & ((1 << rest) - 1));
                }
                value.wrapping_add(ends.wrapping_add(sum_slots(slots, width)) as $T)
            }
        }
    };
}

undelta_pack_single_impl!(u8);
undelta_pack_single_impl!(u16);
undelta_pack_single_impl!(u32);
undelta_pack_single_impl!(u64);
