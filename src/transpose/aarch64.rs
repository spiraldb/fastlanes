//! aarch64 NEON implementations of [`Transpose`](crate::Transpose).
//!
//! These use the tile walks of [`super::tiles`], with 8x8 tiles for `u8` and `u16`, 4x4 for `u32`
//! and 2x2 for `u64`, transposed with `TRN1`/`TRN2` at doubling element widths.

// The tile kernels must inline into the `target_feature` functions to be compiled with them.
#![allow(clippy::inline_always)]

use super::tiles::{Tile, transpose_tiles, untranspose_tiles};
use crate::FastLanes;
use core::arch::aarch64::{
    uint16x8_t, uint32x4_t, uint64x2_t, vld1_u8, vld1q_u16, vld1q_u32, vld1q_u64,
    vreinterpret_u8_u32, vreinterpret_u16_u8, vreinterpret_u32_u16, vreinterpretq_u16_u64,
    vreinterpretq_u32_u16, vreinterpretq_u32_u64, vreinterpretq_u64_u32, vst1_u8, vst1q_u16,
    vst1q_u32, vst1q_u64, vtrn1_u8, vtrn1_u16, vtrn1_u32, vtrn1q_u16, vtrn1q_u32, vtrn1q_u64,
    vtrn2_u8, vtrn2_u16, vtrn2_u32, vtrn2q_u16, vtrn2q_u32, vtrn2q_u64,
};
use core::mem::size_of;

/// [`Transpose::transpose`](crate::Transpose::transpose) with NEON.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub unsafe fn transpose_neon<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: both arrays hold 1024 elements of the width matched below, and the transpose only
    // moves elements, so reading them as unsigned integers of that width is exact. NEON is
    // enabled on this function.
    unsafe {
        match size_of::<T>() {
            1 => transpose_tiles::<U8x8Neon>(input.cast(), output.cast()),
            2 => transpose_tiles::<U16x8Neon>(input.cast(), output.cast()),
            4 => transpose_tiles::<U32x4Neon>(input.cast(), output.cast()),
            _ => transpose_tiles::<U64x2Neon>(input.cast(), output.cast()),
        }
    }
}

/// [`Transpose::untranspose`](crate::Transpose::untranspose) with NEON.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub unsafe fn untranspose_neon<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: as in `transpose_neon`.
    unsafe {
        match size_of::<T>() {
            1 => untranspose_tiles::<U8x8Neon>(input.cast(), output.cast()),
            2 => untranspose_tiles::<U16x8Neon>(input.cast(), output.cast()),
            4 => untranspose_tiles::<U32x4Neon>(input.cast(), output.cast()),
            _ => untranspose_tiles::<U64x2Neon>(input.cast(), output.cast()),
        }
    }
}

/// Transpose an 8x8 tile of bytes.
struct U8x8Neon;

impl Tile for U8x8Neon {
    type Elem = u8;
    const N: usize = 8;

    #[inline(always)]
    unsafe fn tile(src: *const u8, src_stride: usize, dst: *mut u8, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows and NEON.
        unsafe {
            let rows: [_; 8] = core::array::from_fn(|i| vld1_u8(src.add(i * src_stride)));
            // `pairs[k]` interleaves rows `k & !1` and `(k & !1) + 1`: their even columns for even
            // `k`, their odd columns for odd `k`.
            let pairs: [_; 8] = core::array::from_fn(|k| {
                let row = k & !1;
                if k % 2 == 0 {
                    vtrn1_u8(rows[row], rows[row + 1])
                } else {
                    vtrn2_u8(rows[row], rows[row + 1])
                }
            });
            // `quads[k]` holds column `k % 4` of rows `4 * (k / 4)..` in its low 32 bits and
            // column `k % 4 + 4` in its high 32 bits.
            let quads: [_; 8] = core::array::from_fn(|k| {
                let first = 4 * (k / 4) + k % 2;
                let lo = vreinterpret_u16_u8(pairs[first]);
                let hi = vreinterpret_u16_u8(pairs[first + 2]);
                vreinterpret_u32_u16(if k % 4 < 2 {
                    vtrn1_u16(lo, hi)
                } else {
                    vtrn2_u16(lo, hi)
                })
            });
            for col in 0..8 {
                let (top, bottom) = (quads[col % 4], quads[4 + col % 4]);
                let column = if col < 4 {
                    vtrn1_u32(top, bottom)
                } else {
                    vtrn2_u32(top, bottom)
                };
                vst1_u8(dst.add(col * dst_stride), vreinterpret_u8_u32(column));
            }
        }
    }
}

/// Transpose an 8x8 tile of `u16`.
struct U16x8Neon;

impl Tile for U16x8Neon {
    type Elem = u16;
    const N: usize = 8;

    #[inline(always)]
    unsafe fn tile(src: *const u16, src_stride: usize, dst: *mut u16, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows and NEON.
        unsafe {
            let rows: [uint16x8_t; 8] =
                core::array::from_fn(|i| vld1q_u16(src.add(i * src_stride)));
            // As in `U8x8Neon`, at twice the element width.
            let pairs: [uint16x8_t; 8] = core::array::from_fn(|k| {
                let row = k & !1;
                if k % 2 == 0 {
                    vtrn1q_u16(rows[row], rows[row + 1])
                } else {
                    vtrn2q_u16(rows[row], rows[row + 1])
                }
            });
            let quads: [uint64x2_t; 8] = core::array::from_fn(|k| {
                let first = 4 * (k / 4) + k % 2;
                let lo = vreinterpretq_u32_u16(pairs[first]);
                let hi = vreinterpretq_u32_u16(pairs[first + 2]);
                vreinterpretq_u64_u32(if k % 4 < 2 {
                    vtrn1q_u32(lo, hi)
                } else {
                    vtrn2q_u32(lo, hi)
                })
            });
            for col in 0..8 {
                let (top, bottom) = (quads[col % 4], quads[4 + col % 4]);
                let column = if col < 4 {
                    vtrn1q_u64(top, bottom)
                } else {
                    vtrn2q_u64(top, bottom)
                };
                vst1q_u16(dst.add(col * dst_stride), vreinterpretq_u16_u64(column));
            }
        }
    }
}

/// Transpose a 4x4 tile of `u32`.
struct U32x4Neon;

impl Tile for U32x4Neon {
    type Elem = u32;
    const N: usize = 4;

    #[inline(always)]
    unsafe fn tile(src: *const u32, src_stride: usize, dst: *mut u32, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows and NEON.
        unsafe {
            let r: [uint32x4_t; 4] = core::array::from_fn(|i| vld1q_u32(src.add(i * src_stride)));
            let t = [
                vreinterpretq_u64_u32(vtrn1q_u32(r[0], r[1])),
                vreinterpretq_u64_u32(vtrn2q_u32(r[0], r[1])),
                vreinterpretq_u64_u32(vtrn1q_u32(r[2], r[3])),
                vreinterpretq_u64_u32(vtrn2q_u32(r[2], r[3])),
            ];
            let cols = [
                vtrn1q_u64(t[0], t[2]),
                vtrn1q_u64(t[1], t[3]),
                vtrn2q_u64(t[0], t[2]),
                vtrn2q_u64(t[1], t[3]),
            ];
            for (i, col) in cols.into_iter().enumerate() {
                vst1q_u32(dst.add(i * dst_stride), vreinterpretq_u32_u64(col));
            }
        }
    }
}

/// Transpose a 2x2 tile of `u64`.
struct U64x2Neon;

impl Tile for U64x2Neon {
    type Elem = u64;
    const N: usize = 2;

    #[inline(always)]
    unsafe fn tile(src: *const u64, src_stride: usize, dst: *mut u64, dst_stride: usize) {
        // SAFETY: the caller guarantees the rows and NEON.
        unsafe {
            let r0 = vld1q_u64(src);
            let r1 = vld1q_u64(src.add(src_stride));
            vst1q_u64(dst, vtrn1q_u64(r0, r1));
            vst1q_u64(dst.add(dst_stride), vtrn2q_u64(r0, r1));
        }
    }
}
