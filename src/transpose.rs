use crate::{FL_ORDER, FastLanes};
use const_for::const_for;

pub trait Transpose: FastLanes {
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]);
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]);
}

impl<T: FastLanes> Transpose for T {
    #[inline(never)]
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        const_for!(i in 0..1024 => {
            output[i] = input[transpose(i)];
        });
    }

    #[inline(never)]
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        const_for!(i in 0..1024 => {
            output[transpose(i)] = input[i];
        });
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString};
    use hegel::TestCase;
    use hegel::generators as gs;

    #[hegel::test]
    fn test_transpose_roundtrip(tc: TestCase) {
        let input: [u64; 1024] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut transposed = [u64::MAX; 1024];
        let mut actual = [u64::MAX; 1024];

        u64::transpose(&input, &mut transposed);
        u64::untranspose(&transposed, &mut actual);

        assert_eq!(actual, input);
    }

    #[test]
    fn test_transpose_indices_are_a_permutation() {
        let mut seen = [false; 1024];

        for input in 0..1024 {
            let output = transpose(input);
            assert!(output < 1024, "transpose({input}) returned {output}");
            assert!(!seen[output], "transpose produced duplicate index {output}");
            seen[output] = true;
        }

        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn test_transpose_known_indices() {
        for (input, expected) in [
            (0, 0),
            (1, 64),
            (16, 32),
            (32, 16),
            (48, 48),
            (64, 8),
            (128, 1),
            (1023, 1023),
        ] {
            assert_eq!(transpose(input), expected, "transpose({input})");
        }
    }
}
