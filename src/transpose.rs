use crate::{FastLanes, FL_ORDER};
use seq_macro::seq;

pub trait Transpose: FastLanes {
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]);
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]);
    fn transpose_inplace(data: &mut [Self; 1024]);
    fn untranspose_inplace(data: &mut [Self; 1024]);
}

impl<T: FastLanes> Transpose for T {
    #[inline(never)]
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        seq!(i in 0..1024 {
            output[i] = input[transpose(i)];
        });
    }

    #[inline(never)]
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        seq!(i in 0..1024 {
            output[transpose(i)] = input[i];
        });
    }

    #[inline(never)]
    fn transpose_inplace(data: &mut [Self; 1024]) {
        apply_permutation_inplace(data, transpose);
    }

    #[inline(never)]
    fn untranspose_inplace(data: &mut [Self; 1024]) {
        apply_permutation_inplace(data, inverse_transpose);
    }
}

/// Apply a permutation in-place using cycle-following with O(1) temporary space.
///
/// After this function completes, `data[i] == original_data[perm(i)]` for all `i`.
///
/// Each index belongs to exactly one cycle of the permutation. We process each cycle
/// exactly once by only starting a cycle walk when `start` is the minimum index in
/// that cycle.
#[allow(clippy::inline_always)]
#[inline(always)]
fn apply_permutation_inplace<T: Copy, F: Fn(usize) -> usize>(data: &mut [T; 1024], perm: F) {
    seq!(start in 0..1024 {{
        let first_dest = perm(start);

        // Quick reject: if perm(start) <= start, either it's a fixed point
        // (first_dest == start) or the cycle contains a smaller index and
        // was already processed.
        if first_dest > start {
            // Walk the rest of the cycle to verify `start` is the minimum.
            let mut j = perm(first_dest);
            let mut is_min = true;
            while j != start {
                #[allow(clippy::absurd_extreme_comparisons)]
                #[allow(unused_comparisons)]
                if j < start {
                    is_min = false;
                    break;
                }
                j = perm(j);
            }

            // Follow the cycle: data[pos] ← data[perm(pos)] ← ... ← tmp
            if is_min {
                let tmp = data[start];
                let mut pos = start;
                let mut next = first_dest;
                while next != start {
                    data[pos] = data[next];
                    pos = next;
                    next = perm(next);
                }
                data[pos] = tmp;
            }
        }
    }});
}

/// Return the corresponding index in a transposed `FastLanes` vector.
#[allow(clippy::inline_always)]
#[inline(always)]
#[must_use]
pub const fn transpose(idx: usize) -> usize {
    // Row * 8, ORDER * 8, lane * 16.
    let lane = idx % 16;
    let order = (idx / 16) % 8;
    let row = idx / 128;

    (lane * 64) + (FL_ORDER[order] * 8) + row
}

/// Return the inverse of [`transpose`]: `inverse_transpose(transpose(i)) == i` for all valid `i`.
///
/// This exploits the fact that `FL_ORDER` is its own inverse (an involution).
#[allow(clippy::inline_always)]
#[inline(always)]
#[must_use]
pub const fn inverse_transpose(idx: usize) -> usize {
    let lane = idx / 64;
    let remainder = idx % 64;
    let fl_order_val = remainder / 8;
    let row = remainder % 8;
    let order = FL_ORDER[fl_order_val];

    (row * 128) + (order * 16) + lane
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inverse_transpose_is_inverse() {
        for i in 0..1024 {
            assert_eq!(inverse_transpose(transpose(i)), i);
            assert_eq!(transpose(inverse_transpose(i)), i);
        }
    }

    #[test]
    fn test_transpose_inplace_matches_out_of_place() {
        let mut input: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            input[i] = i as u16;
        }

        let mut expected = [0u16; 1024];
        Transpose::transpose(&input, &mut expected);

        let mut inplace = input;
        Transpose::transpose_inplace(&mut inplace);
        assert_eq!(expected, inplace);
    }

    #[test]
    fn test_untranspose_inplace_matches_out_of_place() {
        // Start from transposed data
        let mut input: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            input[i] = i as u16;
        }
        let mut transposed = [0u16; 1024];
        Transpose::transpose(&input, &mut transposed);

        let mut expected = [0u16; 1024];
        Transpose::untranspose(&transposed, &mut expected);

        let mut inplace = transposed;
        Transpose::untranspose_inplace(&mut inplace);
        assert_eq!(expected, inplace);
    }

    #[test]
    fn test_transpose_untranspose_inplace_roundtrip() {
        let mut values: [u32; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i * 37 + 13) as u32;
        }
        let original = values;

        Transpose::transpose_inplace(&mut values);
        assert_ne!(values, original); // sanity: transposing actually changes the data
        Transpose::untranspose_inplace(&mut values);
        assert_eq!(values, original);
    }
}
