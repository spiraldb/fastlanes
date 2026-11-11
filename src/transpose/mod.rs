use crate::{FL_ORDER, FastLanes};

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

#[cfg(all(target_arch = "x86_64", feature = "runtime"))]
cpufeatures::new!(cpuid_avx2, "avx2");

impl<T: FastLanes> Transpose for T {
    #[inline(never)]
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        dispatch::<T, true>(input, output);
    }

    #[inline(never)]
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        dispatch::<T, false>(input, output);
    }
}

/// Transpose with the fastest variant that the CPU supports, or untranspose if `FORWARD` is
/// false.
#[inline]
fn dispatch<T: FastLanes, const FORWARD: bool>(input: &[T; 1024], output: &mut [T; 1024]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: the CPU supports AVX2.
            unsafe { x86::avx2::<T, FORWARD>(input, output) }
        } else {
            // SAFETY: every x86-64 CPU supports SSE2.
            unsafe { x86::sse2::<T, FORWARD>(input, output) }
        }
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: every aarch64 CPU supports NEON.
    unsafe {
        aarch64::neon::<T, FORWARD>(input, output);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        if FORWARD {
            transpose_scalar(input, output);
        } else {
            untranspose_scalar(input, output);
        }
    }
}

/// Give if the CPU supports AVX2: detected at runtime with the `runtime` feature, and otherwise
/// from the enabled target features.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    #[cfg(feature = "runtime")]
    {
        cpuid_avx2::get()
    }

    #[cfg(not(feature = "runtime"))]
    {
        cfg!(target_feature = "avx2")
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

    /// A buffer with 64-byte alignment and one spare element.
    #[repr(C, align(64))]
    struct Aligned<T>([T; 1025]);

    impl<T: FastLanes> Aligned<T> {
        fn new() -> Self {
            Self([T::zero(); 1025])
        }

        /// Give the 1024 elements that start at element `offset`.
        fn at(&mut self, offset: usize) -> &mut [T; 1024] {
            (&mut self.0[offset..offset + 1024]).try_into().unwrap()
        }
    }

    /// Check a transpose variant against [`transpose`], and its inverse.
    ///
    /// The buffers start at a 64-byte boundary, and then one element after it, so that the
    /// kernels also get buffers with only the alignment of `T`.
    fn check_variant<T: FastLanes + Debug>(
        input: &[T; 1024],
        forward: impl Fn(&[T; 1024], &mut [T; 1024]),
        inverse: impl Fn(&[T; 1024], &mut [T; 1024]),
    ) {
        for offset in 0..2 {
            let (mut source, mut transposed, mut untransposed) =
                (Aligned::new(), Aligned::new(), Aligned::new());
            let source = source.at(offset);
            *source = *input;

            let transposed = transposed.at(offset);
            forward(source, transposed);
            for (i, value) in transposed.iter().enumerate() {
                assert_eq!(
                    *value,
                    input[transpose(i)],
                    "offset {offset}, transposed index {i}"
                );
            }

            let untransposed = untransposed.at(offset);
            inverse(transposed, untransposed);
            assert_eq!(untransposed, input, "offset {offset}");
        }
    }

    /// Check every variant that the CPU supports.
    fn check_variants<T: FastLanes + Debug>(input: &[T; 1024]) {
        check_variant::<T>(input, transpose_scalar, untranspose_scalar);
        check_variant::<T>(input, T::transpose, T::untranspose);

        // SAFETY: every x86-64 CPU supports SSE2, and the AVX2 variant runs only if the CPU
        // supports AVX2.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            check_variant::<T>(
                input,
                |i, o| transpose_sse2(i, o),
                |i, o| untranspose_sse2(i, o),
            );

            if has_avx2() {
                check_variant::<T>(
                    input,
                    |i, o| transpose_avx2(i, o),
                    |i, o| untranspose_avx2(i, o),
                );
            }
        }

        // SAFETY: every aarch64 CPU supports NEON.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            check_variant::<T>(
                input,
                |i, o| transpose_neon(i, o),
                |i, o| untranspose_neon(i, o),
            );
        }
    }

    macro_rules! variants_match_permutation {
        ($T:ident) => {
            paste! {
                #[hegel::test]
                fn [<test_transpose_variants_ $T>](tc: TestCase) {
                    let input: [$T; 1024] = tc.draw(gs::arrays(gs::integers::<$T>()));
                    check_variants::<$T>(&input);
                }

                /// A fixed input, so that Miri can run the test. Miri checks the alignment
                /// of every access.
                #[test]
                fn [<test_transpose_variants_fixed_ $T>]() {
                    #[allow(clippy::cast_possible_truncation)]
                    let input: [$T; 1024] = core::array::from_fn(|i| (i * 7 + 3) as $T);
                    check_variants::<$T>(&input);
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
