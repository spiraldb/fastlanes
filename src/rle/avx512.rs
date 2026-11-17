//! AVX-512 window-permute kernels for RLE decoding.
//!
//! One step decodes `W = 64 / S` outputs of `S` bytes: one load of the `W`-value window at the
//! step's first index and one `vpermb`, `vpermw` or `vpermd`. A window that would pass the end of
//! the values is loaded masked, so the values need no padding.

use core::any::TypeId;
use core::arch::x86_64::__m512i;
use core::arch::x86_64::_mm512_add_epi8;
use core::arch::x86_64::_mm512_castsi256_si512;
use core::arch::x86_64::_mm512_castsi512_si128;
use core::arch::x86_64::_mm512_castsi512_si256;
use core::arch::x86_64::_mm512_cmpgt_epu8_mask;
use core::arch::x86_64::_mm512_cmpgt_epu16_mask;
use core::arch::x86_64::_mm512_cmplt_epu8_mask;
use core::arch::x86_64::_mm512_cvtepi16_epi8;
use core::arch::x86_64::_mm512_cvtepu8_epi16;
use core::arch::x86_64::_mm512_cvtepu8_epi32;
use core::arch::x86_64::_mm512_extracti32x4_epi32;
use core::arch::x86_64::_mm512_extracti64x4_epi64;
use core::arch::x86_64::_mm512_inserti64x4;
use core::arch::x86_64::_mm512_loadu_si512;
use core::arch::x86_64::_mm512_maskz_loadu_epi8;
use core::arch::x86_64::_mm512_maskz_loadu_epi16;
use core::arch::x86_64::_mm512_maskz_loadu_epi32;
use core::arch::x86_64::_mm512_maskz_mov_epi8;
use core::arch::x86_64::_mm512_permutex2var_epi8;
use core::arch::x86_64::_mm512_permutexvar_epi8;
use core::arch::x86_64::_mm512_permutexvar_epi16;
use core::arch::x86_64::_mm512_permutexvar_epi32;
use core::arch::x86_64::_mm512_set1_epi8;
use core::arch::x86_64::_mm512_set1_epi16;
use core::arch::x86_64::_mm512_set1_epi64;
use core::arch::x86_64::_mm512_setzero_si512;
use core::arch::x86_64::_mm512_slli_epi64;
use core::arch::x86_64::_mm512_storeu_si512;
use core::arch::x86_64::_mm512_sub_epi8;
use core::arch::x86_64::_mm512_sub_epi16;
use core::ptr;

use crate::FL_ORDER;

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
    // SAFETY: the caller guarantees AVX-512 and bounds the indices; `I` is read as `u8` or
    // `u16` only when it is that type.
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
        // SAFETY: each step's `w` indices and outputs are in bounds; the window load reads only
        // values below `len`, and offsets that pass the check select only those.
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
            // Mask the load only when the window would pass the end of the values.
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

/// Decode one chunk from `u8` Delta indices with AVX-512. Returns `false`, writing nothing, for
/// unsupported value widths.
///
/// Lane `FL_ORDER[g] * 16 + l` covers the 8 outputs at `l * 64 + g * 8`, so 8 lanes cover a block
/// of 64 consecutive outputs. Their delta bytes form a 64-bit mask with one bit per output, and
/// an output's index is its lane's base plus the prefix sum of the mask within the lane. Each
/// block then takes `S` window permutes, as in [`decode`].
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VBMI, and every lane's last index must be less
/// than `rle_vals.len()`.
pub(super) unsafe fn decode_delta<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; 128],
    deltas: Option<&[u8; 128]>,
    output: &mut [V; 1024],
) -> bool {
    let vals = rle_vals.as_ptr().cast::<u8>();
    let len = rle_vals.len();
    let out = output.as_mut_ptr().cast::<u8>();
    // SAFETY: the caller guarantees AVX-512 and bounds every lane's last index.
    unsafe {
        match size_of::<V>() {
            1 => delta_chunk::<1>(vals, len, bases, deltas, out),
            2 => delta_chunk::<2>(vals, len, bases, deltas, out),
            4 => delta_chunk::<4>(vals, len, bases, deltas, out),
            _ => return false,
        }
    }
    true
}

/// Permute indices for [`blocks`], for blocks 0-7 and 8-15.
static BLOCK_GATHER: [[u8; 64]; 2] = {
    let mut idx = [[0u8; 64]; 2];
    let mut h = 0;
    while h < 2 {
        let mut j = 0;
        while j < 64 {
            let (lb, g) = (j / 8, j % 8);
            idx[h][j] = (FL_ORDER[g] * 16 + h * 8 + lb) as u8;
            j += 1;
        }
        h += 1;
    }
    idx
};

/// Byte `j` selects byte `j / 8`, repeating each of the 8 low bytes 8 times.
static SPREAD_BYTES: [u8; 64] = {
    let mut idx = [0u8; 64];
    let mut j = 0;
    while j < 64 {
        idx[j] = (j / 8) as u8;
        j += 1;
    }
    idx
};

/// One `u64` per block of 64 outputs from 128 per-lane bytes: byte `g` of block `l` is the byte
/// of lane `FL_ORDER[g] * 16 + l`.
///
/// # Safety
///
/// AVX-512 must be available, and `lanes` must point to 128 readable bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn blocks(lanes: *const u8) -> [u64; 16] {
    let mut blocks = [0u64; 16];
    // SAFETY: the loads read the 128 bytes of `lanes`, and each store writes 8 of the 16 `u64`s.
    unsafe {
        let lo = _mm512_loadu_si512(lanes.cast());
        let hi = _mm512_loadu_si512(lanes.add(64).cast());
        for (half, gather) in BLOCK_GATHER.iter().enumerate() {
            let idx = _mm512_loadu_si512(gather.as_ptr().cast());
            let v = _mm512_permutex2var_epi8(lo, idx, hi);
            _mm512_storeu_si512(blocks.as_mut_ptr().add(half * 8).cast(), v);
        }
    }
    blocks
}

/// Write the 8 outputs of one lane, one value at a time.
///
/// # Safety
///
/// `vals` must hold the lane's indices, from `base` to `base` plus the set bits of `bits`, and
/// `out` must be writable for 8 values of `S` bytes.
#[inline]
unsafe fn scalar_lane<const S: usize>(vals: *const u8, base: usize, bits: u8, out: *mut u8) {
    let mut index = base;
    for row in 0..8 {
        index += usize::from((bits >> row) & 1);
        // SAFETY: the caller bounds the lane's indices and its 8 outputs.
        unsafe { ptr::copy_nonoverlapping(vals.add(index * S), out.add(row * S), S) };
    }
}

/// For steps of 64, 32 and 16 outputs: byte `j` selects the first byte of its step.
static STEP_FIRST: [[u8; 64]; 3] = {
    let mut idx = [[0u8; 64]; 3];
    let mut i = 0;
    while i < 3 {
        let w = 64 >> i;
        let mut j = 0;
        while j < 64 {
            idx[i][j] = (j / w * w) as u8;
            j += 1;
        }
        i += 1;
    }
    idx
};

/// The offsets of step `step`'s outputs, widened to `S`-byte permute indices.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn step_offsets<const S: usize>(rel: __m512i, step: usize) -> __m512i {
    match (S, step) {
        (1, _) => rel,
        (2, 0) => _mm512_cvtepu8_epi16(_mm512_castsi512_si256(rel)),
        (2, _) => _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(rel)),
        (4, 0) => _mm512_cvtepu8_epi32(_mm512_castsi512_si128(rel)),
        (4, 1) => _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<1>(rel)),
        (4, 2) => _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<2>(rel)),
        (4, _) => _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<3>(rel)),
        _ => unreachable!("unsupported value width {S}"),
    }
}

/// # Safety
///
/// AVX-512 must be available, `vals` must hold `len` values of `S` bytes, every lane's last index
/// must be below `len`, and `out` must be writable for 1024 values.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn delta_chunk<const S: usize>(
    vals: *const u8,
    len: usize,
    bases: &[u8; 128],
    deltas: Option<&[u8; 128]>,
    out: *mut u8,
) {
    // Outputs per step; a block of 64 outputs takes `S` steps.
    let w = 64 / S;
    // SAFETY: `bases` and `deltas` hold 128 bytes each.
    let (base_blocks, bit_blocks) = unsafe {
        (
            blocks(bases.as_ptr()),
            deltas.map_or([0; 16], |d| blocks(d.as_ptr())),
        )
    };
    // SAFETY: the tables hold 64 bytes each.
    let (spread, step_first) = unsafe {
        (
            _mm512_loadu_si512(SPREAD_BYTES.as_ptr().cast()),
            _mm512_loadu_si512(STEP_FIRST[S.trailing_zeros() as usize].as_ptr().cast()),
        )
    };
    let limit = _mm512_set1_epi8(((w - 1) as u8).cast_signed());

    for (l, (&base_block, &bits)) in base_blocks.iter().zip(&bit_blocks).enumerate() {
        // SAFETY: stores stay within the block's 64 outputs; window loads read only values below
        // `len`, as a step's first index is an index of the chunk; permutes select only offsets
        // the check below places in the window.
        unsafe {
            let block_out = out.add(l * 64 * S);
            // Byte `j`: the index of output `l * 64 + j`, modulo 256.
            let ones = _mm512_maskz_mov_epi8(bits, _mm512_set1_epi8(1));
            let mut prefix = _mm512_add_epi8(ones, _mm512_slli_epi64::<8>(ones));
            prefix = _mm512_add_epi8(prefix, _mm512_slli_epi64::<16>(prefix));
            prefix = _mm512_add_epi8(prefix, _mm512_slli_epi64::<32>(prefix));
            let lane_bases =
                _mm512_permutexvar_epi8(spread, _mm512_set1_epi64(base_block.cast_signed()));
            let index = _mm512_add_epi8(lane_bases, prefix);
            // Each output's offset from the first index of its step.
            let firsts = _mm512_permutexvar_epi8(step_first, index);
            let rel = _mm512_sub_epi8(index, firsts);

            // A block with an index past 255 or outside its step's window (never for RLE indices)
            // takes the scalar gather. Otherwise `index` is exact and `rel` its window offsets.
            let wrapped = _mm512_cmplt_epu8_mask(index, lane_bases);
            let outside =
                _mm512_cmplt_epu8_mask(index, firsts) | _mm512_cmpgt_epu8_mask(rel, limit);
            if wrapped | outside != 0 {
                for (g, &order) in FL_ORDER.iter().enumerate() {
                    let lane = order * 16 + l;
                    let bits = deltas.map_or(0, |d| d[lane]);
                    scalar_lane::<S>(
                        vals,
                        usize::from(bases[lane]),
                        bits,
                        block_out.add(g * 8 * S),
                    );
                }
                continue;
            }

            for step in 0..S {
                let first_group = step * 8 / S;
                let byte = |word: u64| ((word >> (8 * first_group)) & 0xFF) as usize;
                let first = byte(base_block) + (byte(bits) & 1);
                let base = vals.add(first * S);
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
                let offsets = step_offsets::<S>(rel, step);
                let v = match S {
                    1 => _mm512_permutexvar_epi8(offsets, window),
                    2 => _mm512_permutexvar_epi16(offsets, window),
                    4 => _mm512_permutexvar_epi32(offsets, window),
                    _ => unreachable!("unsupported value width {S}"),
                };
                _mm512_storeu_si512(block_out.add(step * w * S).cast(), v);
            }
        }
    }
}
