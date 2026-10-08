//! Runtime SIMD dispatch for the auto-vectorized kernels.
//!
//! The `FastLanes` kernels are written as plain scalar Rust that LLVM auto-vectorizes. Which
//! instructions LLVM may use is decided by the target features enabled for the function being
//! compiled, so on its own a kernel only uses the features of the compile-time target (e.g. SSE2
//! on a default `x86_64` build).
//!
//! With the `runtime` feature, [`vectorize`] uses [`fearless_simd`]'s CPU feature detection to
//! pick the best available level and runs the kernel inside a function compiled with that level's
//! target features, letting LLVM auto-vectorize it for e.g. AVX2 or AVX-512. Without the feature,
//! the kernel runs with the compile-time target features only.

/// Run `f` compiled for the best SIMD level available on the current CPU.
///
/// `f` must be `#[inline(always)]` so that its body is compiled inside the level-specific
/// `#[target_feature]` function rather than as a separate baseline function.
#[cfg(feature = "runtime")]
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn vectorize<R>(f: impl FnOnce() -> R) -> R {
    // fearless_simd only performs runtime detection on x86; elsewhere the best level is the
    // compile-time baseline (e.g. NEON on aarch64), which is what `f()` already uses.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use fearless_simd::{Level, Simd};

        let level = Level::new();
        if let Some(avx512) = level.as_avx512() {
            return avx512.vectorize(f);
        }
        if let Some(avx2) = level.as_avx2() {
            return avx2.vectorize(f);
        }
    }
    f()
}

/// Run `f` compiled for the compile-time target features.
#[cfg(not(feature = "runtime"))]
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn vectorize<R>(f: impl FnOnce() -> R) -> R {
    f()
}
