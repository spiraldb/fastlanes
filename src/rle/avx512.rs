//! AVX-512 window-permute kernels for RLE decoding.
//!
//! One step decodes `W = 64 / S` outputs of `S` bytes: the indices of `W` consecutive outputs
//! span at most `W` values, so the step is one masked load of the `W`-value window at the first
//! index and one `vpermb` (`S = 1`), `vpermw` (`S = 2`) or `vpermd` (`S = 4`). Near the end of the
//! values the window load is masked to the values that exist, so no padding past the values is
//! needed.

use core::any::TypeId;
use core::arch::x86_64::__m512i;
use core::arch::x86_64::_mm512_castsi256_si512;
use core::arch::x86_64::_mm512_castsi512_si128;
use core::arch::x86_64::_mm512_castsi512_si256;
use core::arch::x86_64::_mm512_cmpgt_epu8_mask;
use core::arch::x86_64::_mm512_cmpgt_epu16_mask;
use core::arch::x86_64::_mm512_cvtepi16_epi8;
use core::arch::x86_64::_mm512_cvtepu8_epi16;
use core::arch::x86_64::_mm512_cvtepu8_epi32;
use core::arch::x86_64::_mm512_inserti64x4;
use core::arch::x86_64::_mm512_loadu_si512;
use core::arch::x86_64::_mm512_maskz_loadu_epi8;
use core::arch::x86_64::_mm512_maskz_loadu_epi16;
use core::arch::x86_64::_mm512_maskz_loadu_epi32;
use core::arch::x86_64::_mm512_permutexvar_epi8;
use core::arch::x86_64::_mm512_permutexvar_epi16;
use core::arch::x86_64::_mm512_permutexvar_epi32;
use core::arch::x86_64::_mm512_set1_epi8;
use core::arch::x86_64::_mm512_set1_epi16;
use core::arch::x86_64::_mm512_setzero_si512;
use core::arch::x86_64::_mm512_storeu_si512;
use core::arch::x86_64::_mm512_sub_epi8;
use core::arch::x86_64::_mm512_sub_epi16;
use core::ptr;

/// Decode one chunk with AVX-512. Returns `false`, writing nothing, for unsupported value widths
/// and for index types other than `u8` and `u16`.
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VBMI, and every index must be less than
/// `rle_vals.len()`.
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
    // SAFETY: the caller guarantees AVX-512, and the indices are bounded per this function's
    // contract; `I` is read as `u8` or `u16` only when it is that type.
    unsafe {
        if index == TypeId::of::<u8>() {
            match size_of::<V>() {
                1 => chunk::<1, false>(vals, len, idx.cast(), dst),
                2 => chunk::<2, false>(vals, len, idx.cast(), dst),
                4 => chunk::<4, false>(vals, len, idx.cast(), dst),
                _ => return false,
            }
        } else if index == TypeId::of::<u16>() {
            match size_of::<V>() {
                1 => chunk::<1, true>(vals, len, idx.cast(), dst),
                2 => chunk::<2, true>(vals, len, idx.cast(), dst),
                4 => chunk::<4, true>(vals, len, idx.cast(), dst),
                _ => return false,
            }
        } else {
            return false;
        }
    }
    true
}

/// A mask of the low `n` lanes, for `n` up to 64.
#[inline]
const fn low_lanes(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1 << n) - 1 }
}

/// The offsets of the `w` indices at `idx` from the first one, as bytes in the low lanes, and
/// whether every offset is below `w`.
///
/// # Safety
///
/// AVX-512 must be available, and `idx` must point to `w` readable `u8` indices (`U16 = false`)
/// or `u16` indices (`U16 = true`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "with `U16`, `idx` points into a `[u16; 1024]`, so it is aligned for `u16`"
)]
unsafe fn offsets<const U16: bool>(idx: *const u8, w: usize) -> (usize, __m512i, bool) {
    // SAFETY: the caller guarantees `w` readable indices; the masked loads read only those.
    unsafe {
        if U16 {
            let idx = idx.cast::<u16>();
            let first = *idx;
            let limit = _mm512_set1_epi16(((w - 1) as u16).cast_signed());
            let mut rel = _mm512_setzero_si512();
            let mut fits = true;
            // 32 `u16` indices per half; `w` is at most 64.
            for half in 0..w.div_ceil(32) {
                let lanes = (w - half * 32).min(32);
                let mask = low_lanes(lanes) as u32;
                let v = _mm512_maskz_loadu_epi16(mask, idx.add(half * 32).cast());
                let d = _mm512_sub_epi16(v, _mm512_set1_epi16(first.cast_signed()));
                fits &= _mm512_cmpgt_epu16_mask(d, limit) & mask == 0;
                let bytes = _mm512_cvtepi16_epi8(d);
                rel = if half == 0 {
                    _mm512_castsi256_si512(bytes)
                } else {
                    _mm512_inserti64x4::<1>(rel, bytes)
                };
            }
            (usize::from(first), rel, fits)
        } else {
            let first = *idx;
            let mask = low_lanes(w);
            let v = _mm512_maskz_loadu_epi8(mask, idx.cast());
            let rel = _mm512_sub_epi8(v, _mm512_set1_epi8(first.cast_signed()));
            let limit = _mm512_set1_epi8(((w - 1) as u8).cast_signed());
            let fits = _mm512_cmpgt_epu8_mask(rel, limit) & mask == 0;
            (usize::from(first), rel, fits)
        }
    }
}

/// # Safety
///
/// AVX-512 must be available, `vals` must hold `len` values of `S` bytes, `idx` must point to
/// 1024 indices below `len`, and `out` must be writable for 1024 values.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "with `U16`, `idx` points into a `[u16; 1024]`, so it is aligned for `u16`"
)]
unsafe fn chunk<const S: usize, const U16: bool>(
    vals: *const u8,
    len: usize,
    idx: *const u8,
    out: *mut u8,
) {
    let w = 64 / S;
    let index_size = if U16 { 2 } else { 1 };
    for step in 0..1024 / w {
        // SAFETY: `idx` holds 1024 indices, so each step's `w` are in bounds; `out` has room for
        // every step; the window load reads only the values below `len`, and every offset that
        // passes the check selects one of them, as every index is below `len`.
        unsafe {
            let step_idx = idx.add(step * w * index_size);
            let step_out = out.add(step * w * S);
            let (first, rel, fits) = offsets::<U16>(step_idx, w);
            if !fits {
                for row in 0..w {
                    let index = if U16 {
                        usize::from(*step_idx.cast::<u16>().add(row))
                    } else {
                        usize::from(*step_idx.add(row))
                    };
                    ptr::copy_nonoverlapping(vals.add(index * S), step_out.add(row * S), S);
                }
                continue;
            }
            let base = vals.add(first * S);
            // Load the window whole when it lies within the values, which is every step but
            // those near the end, and masked to the values that exist otherwise.
            let window = if first + w <= len {
                _mm512_loadu_si512(base.cast())
            } else {
                let lanes = low_lanes(len - first);
                match S {
                    1 => _mm512_maskz_loadu_epi8(lanes, base.cast()),
                    2 => _mm512_maskz_loadu_epi16(lanes as u32, base.cast()),
                    _ => _mm512_maskz_loadu_epi32(lanes as u16, base.cast()),
                }
            };
            let v = match S {
                1 => _mm512_permutexvar_epi8(rel, window),
                2 => _mm512_permutexvar_epi16(
                    _mm512_cvtepu8_epi16(_mm512_castsi512_si256(rel)),
                    window,
                ),
                4 => _mm512_permutexvar_epi32(
                    _mm512_cvtepu8_epi32(_mm512_castsi512_si128(rel)),
                    window,
                ),
                _ => unreachable!("unsupported value width {S}"),
            };
            _mm512_storeu_si512(step_out.cast(), v);
        }
    }
}
