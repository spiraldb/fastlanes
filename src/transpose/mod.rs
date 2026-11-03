use crate::{FL_ORDER, FastLanes};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use fearless_simd::Level;

pub trait Transpose: FastLanes {
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]);
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]);
}

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod tiles;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
pub use aarch64::{transpose_neon, untranspose_neon};
#[cfg(target_arch = "x86_64")]
pub use x86::{transpose_avx2, transpose_sse2, untranspose_avx2, untranspose_sse2};

impl<T: FastLanes> Transpose for T {
    #[inline(never)]
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        #[cfg(target_arch = "x86_64")]
        {
            let level = level();
            if let Some(avx2) = level.as_avx2() {
                return transpose_avx2(avx2, input, output);
            }
            if let Some(sse2) = level.as_sse2() {
                return transpose_sse2(sse2, input, output);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(neon) = level().as_neon() {
            return transpose_neon(neon, input, output);
        }

        transpose_scalar(input, output);
    }

    #[inline(never)]
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        #[cfg(target_arch = "x86_64")]
        {
            let level = level();
            if let Some(avx2) = level.as_avx2() {
                return untranspose_avx2(avx2, input, output);
            }
            if let Some(sse2) = level.as_sse2() {
                return untranspose_sse2(sse2, input, output);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(neon) = level().as_neon() {
            return untranspose_neon(neon, input, output);
        }

        untranspose_scalar(input, output);
    }
}

/// Give the SIMD level for the dispatch: detected at runtime with the `runtime` feature, and
/// otherwise from the enabled target features.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn level() -> Level {
    #[cfg(feature = "runtime")]
    {
        Level::new()
    }
    #[cfg(not(feature = "runtime"))]
    {
        Level::baseline()
    }
}

/// Portable [`Transpose::transpose`]: writes the output in order and gathers from the input.
pub fn transpose_scalar<T: Copy>(input: &[T; 1024], output: &mut [T; 1024]) {
    for row in 0..8 {
        for (order, &fl_order) in FL_ORDER.iter().enumerate() {
            for lane in 0..16 {
                output[row * 128 + order * 16 + lane] = input[lane * 64 + fl_order * 8 + row];
            }
        }
    }
}

/// Portable [`Transpose::untranspose`].
///
/// Writes the output in order and gathers from the input, which vectorizes far better than
/// scattering writes to `output[transpose(i)]`. `transpose` maps index
/// `row * 128 + order * 16 + lane` to `lane * 64 + FL_ORDER[order] * 8 + row`, and `FL_ORDER` is
/// its own inverse.
pub fn untranspose_scalar<T: Copy>(input: &[T; 1024], output: &mut [T; 1024]) {
    for lane in 0..16 {
        for (fl_order, &order) in FL_ORDER.iter().enumerate() {
            for row in 0..8 {
                output[lane * 64 + fl_order * 8 + row] = input[row * 128 + order * 16 + lane];
            }
        }
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
    use core::fmt::Debug;
    use hegel::TestCase;
    use hegel::generators as gs;
    use pastey::paste;

    /// Check a transpose variant against [`transpose`], and its inverse.
    fn check_variant<T: FastLanes + Debug>(
        input: &[T; 1024],
        forward: impl Fn(&[T; 1024], &mut [T; 1024]),
        inverse: impl Fn(&[T; 1024], &mut [T; 1024]),
    ) {
        let mut transposed = [T::zero(); 1024];
        let mut untransposed = [T::zero(); 1024];
        forward(input, &mut transposed);
        inverse(&transposed, &mut untransposed);
        for (i, value) in transposed.iter().enumerate() {
            assert_eq!(*value, input[transpose(i)], "transposed index {i}");
        }
        assert_eq!(&untransposed, input);
    }

    macro_rules! variants_match_permutation {
        ($T:ident) => {
            paste! {
                #[hegel::test]
                fn [<test_transpose_variants_ $T>](tc: TestCase) {
                    let input: [$T; 1024] = tc.draw(gs::arrays(gs::integers::<$T>()));
                    check_variant::<$T>(&input, transpose_scalar, untranspose_scalar);
                    check_variant::<$T>(&input, $T::transpose, $T::untranspose);

                    #[cfg(target_arch = "x86_64")]
                    {
                        if let Some(sse2) = level().as_sse2() {
                            check_variant::<$T>(
                                &input,
                                |i, o| transpose_sse2(sse2, i, o),
                                |i, o| untranspose_sse2(sse2, i, o),
                            );
                        }
                        if let Some(avx2) = level().as_avx2() {
                            check_variant::<$T>(
                                &input,
                                |i, o| transpose_avx2(avx2, i, o),
                                |i, o| untranspose_avx2(avx2, i, o),
                            );
                        }
                    }

                    #[cfg(target_arch = "aarch64")]
                    if let Some(neon) = level().as_neon() {
                        check_variant::<$T>(
                            &input,
                            |i, o| transpose_neon(neon, i, o),
                            |i, o| untranspose_neon(neon, i, o),
                        );
                    }
                }
            }
        };
    }

    variants_match_permutation!(u8);
    variants_match_permutation!(u16);
    variants_match_permutation!(u32);
    variants_match_permutation!(u64);

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
