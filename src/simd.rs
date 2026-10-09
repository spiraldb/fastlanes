//! Runtime SIMD dispatch for the auto-vectorized kernels.
//!
//! The `FastLanes` kernels are written as plain scalar Rust that LLVM auto-vectorizes. Which
//! instructions LLVM may use is decided by the target features enabled for the function being
//! compiled, so on its own a kernel only uses the features of the compile-time target.
//!
//! The compile-time target is the baseline: we expect x86 builds to target at least AVX2
//! (`-C target-cpu=x86-64-v3`). With the `runtime` feature, [`vectorize`] additionally uses
//! [`fearless_simd`]'s CPU feature detection to run the kernel inside a function compiled for
//! AVX-512 when the CPU supports it. This keeps at most two copies of each kernel. Without the
//! feature, or when the compile-time target already has AVX-512, the kernel runs with the
//! compile-time target features only.

/// Run `f` compiled for AVX-512 if the CPU supports it, otherwise for the compile-time target.
///
/// `f` must be `#[inline(always)]` so that its body is compiled inside the level-specific
/// `#[target_feature]` function rather than as a separate baseline function.
#[cfg(all(
    feature = "runtime",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512f")
))]
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn vectorize<R>(f: impl FnOnce() -> R) -> R {
    use fearless_simd::{Level, Simd};

    if let Some(avx512) = Level::new().as_avx512() {
        return avx512.vectorize(f);
    }
    f()
}

/// Run `f` compiled for the compile-time target features.
///
/// Used when runtime dispatch is disabled, when the compile-time target already includes
/// AVX-512, and on non-x86 targets, where the compile-time baseline (e.g. NEON on aarch64) is
/// the best level `fearless_simd` can detect.
#[cfg(not(all(
    feature = "runtime",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512f")
)))]
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn vectorize<R>(f: impl FnOnce() -> R) -> R {
    f()
}
