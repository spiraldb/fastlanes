//! x86-64 SIMD implementations of [`Transpose`](crate::Transpose).
//!
//! Both directions of the transpose move square tiles: `untranspose` reads tile rows 128 elements
//! apart and writes them 64 apart, `transpose` the other way around. So each element width needs
//! one tile kernel, `tile(src, src_stride, dst, dst_stride)`, setting `dst[i][j] = src[j][i]`.
//!
//! The SSE2 kernels use 8x8 tiles for `u8` and `u16`, 4x4 for `u32` and 2x2 for `u64`. The AVX2
//! kernels use 8x8 tiles for `u32` and 4x4 for `u64`, where each register holds half rows of two
//! tile rows, `N / 2` rows apart, so that the tile transpose needs no shuffles across 128-bit
//! lanes. AVX2 keeps the SSE2 tiles for `u8` and `u16`.

// The tile kernels must inline into the `target_feature` functions to be compiled with them.
#![allow(clippy::inline_always)]

use super::tiles::{Tile, transpose_tiles, untranspose_tiles};
use crate::FastLanes;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_castsi128_pd, _mm_loadl_epi64, _mm_loadu_si128, _mm_storeh_pd,
    _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
    _mm256_loadu2_m128i, _mm256_storeu_si256, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};
use core::mem::size_of;

#[cfg(feature = "runtime")]
cpufeatures::new!(avx2, "avx2");

/// Check if AVX2 is available, at runtime with the `runtime` feature and otherwise from the
/// enabled target features.
#[inline]
pub(super) fn has_avx2() -> bool {
    #[cfg(feature = "runtime")]
    {
        avx2::get()
    }
    #[cfg(not(feature = "runtime"))]
    {
        cfg!(target_feature = "avx2")
    }
}

/// [`Transpose::transpose`](crate::Transpose::transpose) with SSE2.
///
/// # Safety
/// The CPU must support SSE2, which every x86-64 CPU does.
#[target_feature(enable = "sse2")]
pub unsafe fn transpose_sse2<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: both arrays hold 1024 elements of the width matched below, and the transpose only
    // moves elements, so reading them as unsigned integers of that width is exact. SSE2 is
    // enabled on this function.
    unsafe {
        match size_of::<T>() {
            1 => transpose_tiles::<U8x8Sse2>(input.cast(), output.cast()),
            2 => transpose_tiles::<U16x8Sse2>(input.cast(), output.cast()),
            4 => transpose_tiles::<U32x4Sse2>(input.cast(), output.cast()),
            _ => transpose_tiles::<U64x2Sse2>(input.cast(), output.cast()),
        }
    }
}

/// [`Transpose::untranspose`](crate::Transpose::untranspose) with SSE2.
///
/// # Safety
/// The CPU must support SSE2, which every x86-64 CPU does.
#[target_feature(enable = "sse2")]
pub unsafe fn untranspose_sse2<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: as in `transpose_sse2`.
    unsafe {
        match size_of::<T>() {
            1 => untranspose_tiles::<U8x8Sse2>(input.cast(), output.cast()),
            2 => untranspose_tiles::<U16x8Sse2>(input.cast(), output.cast()),
            4 => untranspose_tiles::<U32x4Sse2>(input.cast(), output.cast()),
            _ => untranspose_tiles::<U64x2Sse2>(input.cast(), output.cast()),
        }
    }
}

/// [`Transpose::transpose`](crate::Transpose::transpose) with AVX2.
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_avx2<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: as in `transpose_sse2`, with AVX2 enabled on this function.
    unsafe {
        match size_of::<T>() {
            1 => transpose_tiles::<U8x8Sse2>(input.cast(), output.cast()),
            2 => transpose_tiles::<U16x8Sse2>(input.cast(), output.cast()),
            4 => transpose_tiles::<U32x8Avx2>(input.cast(), output.cast()),
            _ => transpose_tiles::<U64x4Avx2>(input.cast(), output.cast()),
        }
    }
}

/// [`Transpose::untranspose`](crate::Transpose::untranspose) with AVX2.
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn untranspose_avx2<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: as in `transpose_avx2`.
    unsafe {
        match size_of::<T>() {
            1 => untranspose_tiles::<U8x8Sse2>(input.cast(), output.cast()),
            2 => untranspose_tiles::<U16x8Sse2>(input.cast(), output.cast()),
            4 => untranspose_tiles::<U32x8Avx2>(input.cast(), output.cast()),
            _ => untranspose_tiles::<U64x4Avx2>(input.cast(), output.cast()),
        }
    }
}

/// Transpose an 8x8 tile of bytes.
struct U8x8Sse2;

impl Tile for U8x8Sse2 {
    type Elem = u8;
    const N: usize = 8;

    #[inline(always)]
    unsafe fn tile(src: *const u8, src_stride: usize, dst: *mut u8, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows, and SSE2 is part of x86-64.
        unsafe {
            let row = |i: usize| _mm_loadl_epi64(src.add(i * src_stride).cast());
            let p01 = _mm_unpacklo_epi8(row(0), row(1));
            let p23 = _mm_unpacklo_epi8(row(2), row(3));
            let p45 = _mm_unpacklo_epi8(row(4), row(5));
            let p67 = _mm_unpacklo_epi8(row(6), row(7));
            // Columns 0..4 and 4..8 of rows 0..4, then of rows 4..8.
            let q0 = _mm_unpacklo_epi16(p01, p23);
            let q1 = _mm_unpackhi_epi16(p01, p23);
            let q2 = _mm_unpacklo_epi16(p45, p67);
            let q3 = _mm_unpackhi_epi16(p45, p67);
            // Each register holds two columns of all 8 rows.
            let cols = [
                _mm_unpacklo_epi32(q0, q2),
                _mm_unpackhi_epi32(q0, q2),
                _mm_unpacklo_epi32(q1, q3),
                _mm_unpackhi_epi32(q1, q3),
            ];
            for (i, col) in cols.into_iter().enumerate() {
                _mm_storel_epi64(dst.add(2 * i * dst_stride).cast(), col);
                _mm_storeh_pd(
                    dst.add((2 * i + 1) * dst_stride).cast(),
                    _mm_castsi128_pd(col),
                );
            }
        }
    }
}

/// Transpose an 8x8 tile of `u16`.
struct U16x8Sse2;

impl Tile for U16x8Sse2 {
    type Elem = u16;
    const N: usize = 8;

    #[inline(always)]
    unsafe fn tile(src: *const u16, src_stride: usize, dst: *mut u16, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows, and SSE2 is part of x86-64.
        unsafe {
            let r: [__m128i; 8] =
                core::array::from_fn(|i| _mm_loadu_si128(src.add(i * src_stride).cast()));
            let a0 = _mm_unpacklo_epi16(r[0], r[1]);
            let a1 = _mm_unpackhi_epi16(r[0], r[1]);
            let a2 = _mm_unpacklo_epi16(r[2], r[3]);
            let a3 = _mm_unpackhi_epi16(r[2], r[3]);
            let a4 = _mm_unpacklo_epi16(r[4], r[5]);
            let a5 = _mm_unpackhi_epi16(r[4], r[5]);
            let a6 = _mm_unpacklo_epi16(r[6], r[7]);
            let a7 = _mm_unpackhi_epi16(r[6], r[7]);
            let b0 = _mm_unpacklo_epi32(a0, a2);
            let b1 = _mm_unpackhi_epi32(a0, a2);
            let b2 = _mm_unpacklo_epi32(a1, a3);
            let b3 = _mm_unpackhi_epi32(a1, a3);
            let b4 = _mm_unpacklo_epi32(a4, a6);
            let b5 = _mm_unpackhi_epi32(a4, a6);
            let b6 = _mm_unpacklo_epi32(a5, a7);
            let b7 = _mm_unpackhi_epi32(a5, a7);
            let cols = [
                _mm_unpacklo_epi64(b0, b4),
                _mm_unpackhi_epi64(b0, b4),
                _mm_unpacklo_epi64(b1, b5),
                _mm_unpackhi_epi64(b1, b5),
                _mm_unpacklo_epi64(b2, b6),
                _mm_unpackhi_epi64(b2, b6),
                _mm_unpacklo_epi64(b3, b7),
                _mm_unpackhi_epi64(b3, b7),
            ];
            for (i, col) in cols.into_iter().enumerate() {
                _mm_storeu_si128(dst.add(i * dst_stride).cast(), col);
            }
        }
    }
}

/// Transpose an 8x8 tile of `u32`.
struct U32x8Avx2;

impl Tile for U32x8Avx2 {
    type Elem = u32;
    const N: usize = 8;

    #[inline(always)]
    unsafe fn tile(src: *const u32, src_stride: usize, dst: *mut u32, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows and AVX2.
        unsafe {
            // Register `k` holds columns `4 * (k / 4)..` of rows `k % 4` and `k % 4 + 4`.
            let r: [__m256i; 8] = core::array::from_fn(|k| {
                let (row, col) = (k % 4, 4 * (k / 4));
                _mm256_loadu2_m128i(
                    src.add((row + 4) * src_stride + col).cast(),
                    src.add(row * src_stride + col).cast(),
                )
            });
            for half in 0..2 {
                let r = &r[4 * half..];
                let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
                let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
                let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
                let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
                let cols = [
                    _mm256_unpacklo_epi64(t0, t2),
                    _mm256_unpackhi_epi64(t0, t2),
                    _mm256_unpacklo_epi64(t1, t3),
                    _mm256_unpackhi_epi64(t1, t3),
                ];
                for (i, col) in cols.into_iter().enumerate() {
                    _mm256_storeu_si256(dst.add((4 * half + i) * dst_stride).cast(), col);
                }
            }
        }
    }
}

/// Transpose a 4x4 tile of `u64`.
struct U64x4Avx2;

impl Tile for U64x4Avx2 {
    type Elem = u64;
    const N: usize = 4;

    #[inline(always)]
    unsafe fn tile(src: *const u64, src_stride: usize, dst: *mut u64, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows and AVX2.
        unsafe {
            // Register `k` holds columns `2 * (k / 2)..` of rows `k % 2` and `k % 2 + 2`.
            let r: [__m256i; 4] = core::array::from_fn(|k| {
                let (row, col) = (k % 2, 2 * (k / 2));
                _mm256_loadu2_m128i(
                    src.add((row + 2) * src_stride + col).cast(),
                    src.add(row * src_stride + col).cast(),
                )
            });
            for half in 0..2 {
                let (r0, r1) = (r[2 * half], r[2 * half + 1]);
                _mm256_storeu_si256(
                    dst.add(2 * half * dst_stride).cast(),
                    _mm256_unpacklo_epi64(r0, r1),
                );
                _mm256_storeu_si256(
                    dst.add((2 * half + 1) * dst_stride).cast(),
                    _mm256_unpackhi_epi64(r0, r1),
                );
            }
        }
    }
}

/// Transpose a 4x4 tile of `u32`.
struct U32x4Sse2;

impl Tile for U32x4Sse2 {
    type Elem = u32;
    const N: usize = 4;

    #[inline(always)]
    unsafe fn tile(src: *const u32, src_stride: usize, dst: *mut u32, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows, and SSE2 is part of x86-64.
        unsafe {
            let r: [__m128i; 4] =
                core::array::from_fn(|i| _mm_loadu_si128(src.add(i * src_stride).cast()));
            let t0 = _mm_unpacklo_epi32(r[0], r[1]);
            let t1 = _mm_unpackhi_epi32(r[0], r[1]);
            let t2 = _mm_unpacklo_epi32(r[2], r[3]);
            let t3 = _mm_unpackhi_epi32(r[2], r[3]);
            let cols = [
                _mm_unpacklo_epi64(t0, t2),
                _mm_unpackhi_epi64(t0, t2),
                _mm_unpacklo_epi64(t1, t3),
                _mm_unpackhi_epi64(t1, t3),
            ];
            for (i, col) in cols.into_iter().enumerate() {
                _mm_storeu_si128(dst.add(i * dst_stride).cast(), col);
            }
        }
    }
}

/// Transpose a 2x2 tile of `u64`.
struct U64x2Sse2;

impl Tile for U64x2Sse2 {
    type Elem = u64;
    const N: usize = 2;

    #[inline(always)]
    unsafe fn tile(src: *const u64, src_stride: usize, dst: *mut u64, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows, and SSE2 is part of x86-64.
        unsafe {
            let r0 = _mm_loadu_si128(src.cast());
            let r1 = _mm_loadu_si128(src.add(src_stride).cast());
            _mm_storeu_si128(dst.cast(), _mm_unpacklo_epi64(r0, r1));
            _mm_storeu_si128(dst.add(dst_stride).cast(), _mm_unpackhi_epi64(r0, r1));
        }
    }
}
