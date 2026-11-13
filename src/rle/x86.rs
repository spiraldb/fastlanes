// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 window-permute kernels for RLE decoding.

use core::any::TypeId;

use core::arch::x86_64::__m128i;
use core::arch::x86_64::_mm_add_epi8;
use core::arch::x86_64::_mm_cmpeq_epi8;
use core::arch::x86_64::_mm_cmpeq_epi16;
use core::arch::x86_64::_mm_loadl_epi64;
use core::arch::x86_64::_mm_loadu_si128;
use core::arch::x86_64::_mm_max_epu8;
use core::arch::x86_64::_mm_max_epu16;
use core::arch::x86_64::_mm_movemask_epi8;
use core::arch::x86_64::_mm_packus_epi16;
use core::arch::x86_64::_mm_set1_epi8;
use core::arch::x86_64::_mm_set1_epi16;
use core::arch::x86_64::_mm_shuffle_epi8;
use core::arch::x86_64::_mm_storel_epi64;
use core::arch::x86_64::_mm_storeu_si128;
use core::arch::x86_64::_mm_sub_epi8;
use core::arch::x86_64::_mm_sub_epi16;
use core::arch::x86_64::_mm_unpacklo_epi8;
use core::arch::x86_64::_mm256_cvtepu8_epi32;
use core::arch::x86_64::_mm256_loadu_si256;
use core::arch::x86_64::_mm256_permutevar8x32_epi32;
use core::arch::x86_64::_mm256_storeu_si256;
use core::ptr;

use super::DELTA_LUT;
use super::GROUPS;
use super::U8_LANES;
use super::u8_lane_start;

/// Decode one chunk with AVX2. Returns `false`, writing nothing, for unsupported value widths
/// and for index types other than `u8` and `u16`.
///
/// # Safety
///
/// The CPU must support AVX2, and every index must be less than `rle_vals.len()`.
pub(super) unsafe fn decode<V: Copy, I: Copy + Into<usize> + 'static>(
    rle_vals: &[V],
    rle_idxs: &[I; 1024],
    output: &mut [V; 1024],
) -> bool {
    let vals = rle_vals.as_ptr().cast::<u8>();
    let len = rle_vals.len();
    let dst = output.as_mut_ptr().cast::<u8>();
    let idx = rle_idxs.as_ptr();
    let index = TypeId::of::<I>();
    // SAFETY: the caller guarantees AVX2, and the indices are bounded per this function's
    // contract; `I` is
    // read as `u8` or `u16` only when it is that type.
    unsafe {
        if index == TypeId::of::<u8>() {
            match size_of::<V>() {
                1 => chunk_u8::<1>(vals, len, idx.cast(), dst),
                2 => chunk_u8::<2>(vals, len, idx.cast(), dst),
                4 => chunk_u8::<4>(vals, len, idx.cast(), dst),
                _ => return false,
            }
        } else if index == TypeId::of::<u16>() {
            match size_of::<V>() {
                1 => chunk_u16::<1>(vals, len, idx.cast(), dst),
                2 => chunk_u16::<2>(vals, len, idx.cast(), dst),
                4 => chunk_u16::<4>(vals, len, idx.cast(), dst),
                _ => return false,
            }
        } else {
            return false;
        }
    }
    true
}

/// Decode one chunk from `u8` Delta lanes with AVX2. Returns `false`, writing nothing, for
/// unsupported widths.
///
/// # Safety
///
/// The CPU must support AVX2, and each lane's last index must be less than `rle_vals.len()`.
pub(super) unsafe fn decode_delta<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; U8_LANES],
    deltas: Option<&[u8; U8_LANES]>,
    output: &mut [V; 1024],
) -> bool {
    let vals = rle_vals.as_ptr().cast::<u8>();
    let len = rle_vals.len();
    let deltas = deltas.map_or(ptr::null(), |d| d.as_ptr());
    let dst = output.as_mut_ptr().cast::<u8>();
    // SAFETY: the caller guarantees AVX2, and every lane is bounded per this function's
    // contract.
    unsafe {
        match size_of::<V>() {
            1 => delta_lanes::<1>(vals, len, bases.as_ptr(), deltas, dst),
            2 => delta_lanes::<2>(vals, len, bases.as_ptr(), deltas, dst),
            4 => delta_lanes::<4>(vals, len, bases.as_ptr(), deltas, dst),
            _ => return false,
        }
    }
    true
}

/// Write the 8 values of `S` bytes at `window[rel[r]]` to `out`, where `rel` holds 8 byte offsets
/// below 8 in its low half.
///
/// # Safety
///
/// `window` must be readable for `8 * S` bytes and `out` writable for `8 * S` bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn permute8<const S: usize>(window: *const u8, rel: __m128i, out: *mut u8) {
    // SAFETY: the caller guarantees `8 * S` readable and writable bytes; the loads and stores
    // below touch exactly that many.
    unsafe {
        match S {
            1 => {
                let w = _mm_loadl_epi64(window.cast());
                _mm_storel_epi64(out.cast(), _mm_shuffle_epi8(w, rel));
            }
            2 => {
                let w = _mm_loadu_si128(window.cast());
                let lo = _mm_add_epi8(rel, rel);
                let idx = _mm_unpacklo_epi8(lo, _mm_add_epi8(lo, _mm_set1_epi8(1)));
                _mm_storeu_si128(out.cast(), _mm_shuffle_epi8(w, idx));
            }
            4 => {
                let w = _mm256_loadu_si256(window.cast());
                let idx = _mm256_cvtepu8_epi32(rel);
                _mm256_storeu_si256(out.cast(), _mm256_permutevar8x32_epi32(w, idx));
            }
            _ => unreachable!("unsupported value width {S}"),
        }
    }
}

/// Copy the `S`-byte values at `indices` to `out`, one at a time.
///
/// # Safety
///
/// Every index must be in bounds of `vals`, and `out` writable for `8 * S` bytes.
#[inline]
unsafe fn scalar_group<const S: usize>(
    vals: *const u8,
    indices: impl Iterator<Item = usize>,
    out: *mut u8,
) {
    for (row, index) in indices.enumerate() {
        // SAFETY: the caller bounds every index and the 8 output slots.
        unsafe { ptr::copy_nonoverlapping(vals.add(index * S), out.add(row * S), S) };
    }
}

/// # Safety
///
/// AVX2 must be available, `vals` must hold `len` values of `S` bytes, `idx` must point to 1024
/// indices below `len`, and `out` must be writable for 1024 values.
#[target_feature(enable = "avx2")]
unsafe fn chunk_u8<const S: usize>(vals: *const u8, len: usize, idx: *const u8, out: *mut u8) {
    let seven = _mm_set1_epi8(7);
    for group in 0..GROUPS {
        // SAFETY: `idx` holds 1024 indices, so each group's 8 are in bounds; `out` has room for
        // every group; the window is read only when it ends within `len` values.
        unsafe {
            let group_idx = idx.add(group * 8);
            let first = *group_idx;
            let rel = _mm_sub_epi8(
                _mm_loadl_epi64(group_idx.cast()),
                _mm_set1_epi8(first.cast_signed()),
            );
            let in_window = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_max_epu8(rel, seven), seven));
            let group_out = out.add(group * 8 * S);
            if in_window & 0xFF == 0xFF && usize::from(first) + 8 <= len {
                permute8::<S>(vals.add(usize::from(first) * S), rel, group_out);
            } else {
                let indices = (0..8).map(|r| usize::from(*group_idx.add(r)));
                scalar_group::<S>(vals, indices, group_out);
            }
        }
    }
}

/// # Safety
///
/// As [`chunk_u8`], with `u16` indices.
#[target_feature(enable = "avx2")]
unsafe fn chunk_u16<const S: usize>(vals: *const u8, len: usize, idx: *const u16, out: *mut u8) {
    let seven = _mm_set1_epi16(7);
    for group in 0..GROUPS {
        // SAFETY: as in `chunk_u8`.
        unsafe {
            let group_idx = idx.add(group * 8);
            let first = *group_idx;
            let rel16 = _mm_sub_epi16(
                _mm_loadu_si128(group_idx.cast()),
                _mm_set1_epi16(first.cast_signed()),
            );
            let in_window = _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_max_epu16(rel16, seven), seven));
            let group_out = out.add(group * 8 * S);
            if in_window == 0xFFFF && usize::from(first) + 8 <= len {
                let rel = _mm_packus_epi16(rel16, rel16);
                permute8::<S>(vals.add(usize::from(first) * S), rel, group_out);
            } else {
                let indices = (0..8).map(|r| usize::from(*group_idx.add(r)));
                scalar_group::<S>(vals, indices, group_out);
            }
        }
    }
}

/// # Safety
///
/// AVX2 must be available, `vals` must hold `len` values of `S` bytes, `bases` must point to 128
/// bases, `deltas` to 128 delta bytes or be null, each lane's last index must be below `len`, and
/// `out` must be writable for 1024 values.
#[target_feature(enable = "avx2")]
unsafe fn delta_lanes<const S: usize>(
    vals: *const u8,
    len: usize,
    bases: *const u8,
    deltas: *const u8,
    out: *mut u8,
) {
    for lane in 0..U8_LANES {
        // SAFETY: `bases` and `deltas` hold 128 bytes; every lane writes 8 values inside the
        // chunk; the window is read only when it ends within `len` values, and the caller bounds
        // every index the lane uses.
        unsafe {
            let bits = if deltas.is_null() {
                0
            } else {
                *deltas.add(lane)
            };
            let first = usize::from(*bases.add(lane)) + usize::from(bits & 1);
            let rel = DELTA_LUT[usize::from(bits)];
            let lane_out = out.add(u8_lane_start(lane) * S);
            if first + 8 <= len {
                permute8::<S>(
                    vals.add(first * S),
                    _mm_loadl_epi64(rel.as_ptr().cast()),
                    lane_out,
                );
            } else {
                let indices = rel.iter().map(|&r| first + usize::from(r));
                scalar_group::<S>(vals, indices, lane_out);
            }
        }
    }
}
