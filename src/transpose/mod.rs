use crate::{FL_ORDER, FastLanes};

pub trait Transpose: FastLanes {
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]);
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]);
}

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod tiles;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
pub use aarch64::{transpose_neon, untranspose_neon};
#[cfg(target_arch = "x86_64")]
pub use avx512::{transpose_avx512, untranspose_avx512};
#[cfg(target_arch = "x86_64")]
pub use x86::{transpose_avx2, transpose_sse2, untranspose_avx2, untranspose_sse2};

#[cfg(all(target_arch = "x86_64", feature = "runtime"))]
cpufeatures::new!(cpuid_avx2, "avx2");
#[cfg(all(target_arch = "x86_64", feature = "runtime"))]
cpufeatures::new!(cpuid_avx512, "avx512f", "avx512bw", "avx512vbmi");

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
        if has_avx512() && !avx512_is_slower(size_of::<T>()) {
            // SAFETY: the CPU supports AVX-512 F, BW and VBMI.
            unsafe { avx512::avx512::<T, FORWARD>(input, output) }
        } else if has_avx2() {
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

/// Give if the CPU supports AVX-512 F, BW and VBMI: detected at runtime with the `runtime`
/// feature, and otherwise from the enabled target features.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512() -> bool {
    #[cfg(feature = "runtime")]
    {
        cpuid_avx512::get()
    }

    #[cfg(not(feature = "runtime"))]
    {
        cfg!(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        ))
    }
}

/// Give if the AVX-512 variant is slower than the AVX2 variant for elements of `size` bytes on
/// this CPU.
///
/// Zen 4 runs 512-bit shuffles at half rate, so there `u32` and `u64` are about 25% slower with
/// AVX-512.
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx512_is_slower(size: usize) -> bool {
    size >= 4 && is_zen4()
}

/// Give if the CPU is a Zen 4 with AVX-512. Only the `runtime` feature detects the CPU model.
/// Without it, the dispatch uses AVX-512 whenever the target features enable it.
#[cfg(target_arch = "x86_64")]
#[inline]
fn is_zen4() -> bool {
    #[cfg(feature = "runtime")]
    {
        zen4::get()
    }

    #[cfg(not(feature = "runtime"))]
    {
        false
    }
}

/// Detection of AMD Zen 4 CPUs.
#[cfg(all(target_arch = "x86_64", feature = "runtime"))]
mod zen4 {
    use core::arch::x86_64::__cpuid;
    use core::sync::atomic::{AtomicU8, Ordering};

    /// Give if the CPU is a Zen 4 with AVX-512. The first call reads `cpuid`, and the other calls
    /// read the cached result.
    #[inline]
    pub(super) fn get() -> bool {
        // 0: not detected yet, 1: not Zen 4, 2: Zen 4.
        static CACHE: AtomicU8 = AtomicU8::new(0);

        match CACHE.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let zen4 = detect();
                CACHE.store(1 + u8::from(zen4), Ordering::Relaxed);
                zen4
            }
        }
    }

    /// Read `cpuid`. Call only on a CPU with AVX-512.
    fn detect() -> bool {
        // `__cpuid` is unsafe on the minimum supported Rust version and safe on newer versions.
        // SAFETY: every x86-64 CPU has `cpuid`, with leaves 0 and 1.
        #[allow(unused_unsafe)]
        let (vendor, signature) = unsafe { (__cpuid(0), __cpuid(1).eax) };

        let amd = [vendor.ebx, vendor.edx, vendor.ecx]
            == [
                u32::from_le_bytes(*b"Auth"),
                u32::from_le_bytes(*b"enti"),
                u32::from_le_bytes(*b"cAMD"),
            ];

        amd && is_zen4_family(signature)
    }

    /// Give if the processor signature `eax` of `cpuid` leaf 1 is AMD family 0x19.
    ///
    /// Of the family 0x19 CPUs, only Zen 4 has AVX-512: Zen 3 does not. Zen 5 is family 0x1A.
    pub(super) const fn is_zen4_family(eax: u32) -> bool {
        let base = (eax >> 8) & 0xF;
        let family = if base == 0xF {
            base + ((eax >> 20) & 0xFF)
        } else {
            base
        };

        family == 0x19
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

        // SAFETY: every x86-64 CPU supports SSE2, and the AVX2 and AVX-512 variants run only if
        // the CPU supports them.
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

            if has_avx512() {
                check_variant::<T>(
                    input,
                    |i, o| transpose_avx512(i, o),
                    |i, o| untranspose_avx512(i, o),
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

    #[cfg(all(target_arch = "x86_64", feature = "runtime"))]
    #[test]
    fn test_zen4_family() {
        // The `cpuid` leaf 1 signatures of a Ryzen 9 7950X (Zen 4), a Ryzen 9 9950X (Zen 5), a
        // Ryzen 9 5950X (Zen 3) and a Xeon Platinum 8488C (Sapphire Rapids).
        assert!(zen4::is_zen4_family(0x00A6_0F12));
        assert!(!zen4::is_zen4_family(0x00B4_0F40));
        assert!(zen4::is_zen4_family(0x00A2_0F10));
        assert!(!zen4::is_zen4_family(0x0008_06F8));
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
