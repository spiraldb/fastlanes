//! x86-64 transpose implementations: BMI2 (`PEXT`/`PDEP`) and AVX-512 VBMI.
#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::__m512i;
use core::arch::x86_64::_mm512_and_si512;
use core::arch::x86_64::_mm512_loadu_si512;
use core::arch::x86_64::_mm512_permutex2var_epi8;
use core::arch::x86_64::_mm512_permutexvar_epi8;
use core::arch::x86_64::_mm512_set1_epi64;
use core::arch::x86_64::_mm512_slli_epi64;
use core::arch::x86_64::_mm512_srli_epi64;
use core::arch::x86_64::_mm512_storeu_si512;
use core::arch::x86_64::_mm512_xor_si512;
use core::arch::x86_64::_pdep_u64;
use core::arch::x86_64::_pext_u64;

use crate::FL_ORDER;
use crate::FastLanes;
use crate::bit_transpose::BASE_PATTERN_FIRST;
use crate::bit_transpose::BASE_PATTERN_SECOND;
use crate::bit_transpose::TRANSPOSE_2X2;
use crate::bit_transpose::TRANSPOSE_4X4;
use crate::bit_transpose::TRANSPOSE_8X8;
use crate::bit_transpose::as_byte_array;
use crate::bit_transpose::as_byte_array_mut;
use crate::bit_transpose::group_perm::group_tables;

#[cfg(feature = "runtime")]
cpufeatures::new!(bmi2, "bmi2");
#[cfg(feature = "runtime")]
cpufeatures::new!(vbmi, "avx512f", "avx512bw", "avx512vbmi");

/// Check if BMI2 is available (requires the `runtime` feature).
#[cfg(feature = "runtime")]
#[inline]
#[must_use]
pub fn has_bmi2() -> bool {
    bmi2::get()
}

/// Check if AVX-512 VBMI (+ F + BW) is available (requires the `runtime` feature).
#[cfg(feature = "runtime")]
#[inline]
#[must_use]
pub fn has_vbmi() -> bool {
    vbmi::get()
}

/// Per-bit-position masks selecting one bit out of every byte of a `u64`.
const BIT_MASKS: [u64; 8] = [
    0x0101_0101_0101_0101,
    0x0202_0202_0202_0202,
    0x0404_0404_0404_0404,
    0x0808_0808_0808_0808,
    0x1010_1010_1010_1010,
    0x2020_2020_2020_2020,
    0x4040_4040_4040_4040,
    0x8080_8080_8080_8080,
];

/// Transpose 1024 bits using BMI2 `PEXT`.
///
/// `PEXT` extracts bits at positions specified by a mask into contiguous low bits.
///
/// # Safety
/// Requires BMI2 support. Check with [`has_bmi2`] before calling.
#[target_feature(enable = "bmi2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn transpose_bits_bmi2(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);

    for (half, groups) in [BASE_PATTERN_FIRST, BASE_PATTERN_SECOND].iter().enumerate() {
        for (group, &base) in groups.iter().enumerate() {
            let g = gather(input, base);
            for (bit, &mask) in BIT_MASKS.iter().enumerate() {
                output[half * 64 + bit * 8 + group] = _pext_u64(g, mask) as u8;
            }
        }
    }
}

/// Untranspose a `T`-width comparison mask (1024 bits) using BMI2 `PDEP`.
///
/// Regardless of width the 128 bytes factor into 16 groups of 8 bytes, each an independent 8x8
/// bit transpose (see [`crate::bit_transpose::scalar::untranspose_bits`]). The width only changes
/// the gather stride and the scatter base. For each group we gather its 8 bytes (at stride
/// `T::T / 8`) and use `PDEP` to deposit byte `llo` into bit-position `llo` of all 8 packed
/// output bytes — which is exactly the 8x8 bit transpose — then scatter the group at stride 16.
/// For `T = u64` this is the canonical `FastLanes` bit untranspose.
///
/// # Safety
/// Requires BMI2 support. Check with [`has_bmi2`] before calling.
#[target_feature(enable = "bmi2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn untranspose_bits_bmi2<T: FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);

    let bytes = T::T / 8;
    let lhi_count = 128 / T::T;
    for lhi in 0..lhi_count {
        for hi in 0..bytes {
            let gather_base = lhi * T::T + hi;
            let mut v = 0u64;
            for (llo, &mask) in BIT_MASKS.iter().enumerate() {
                v |= _pdep_u64(u64::from(input[gather_base + llo * bytes]), mask);
            }
            let scatter_base = FL_ORDER[hi] * 2 + lhi;
            for row in 0..8 {
                output[scatter_base + row * 16] = (v >> (row * 8)) as u8;
            }
        }
    }
}

/// Gather 8 bytes at stride 16 into a `u64`.
#[inline]
fn gather(input: &[u8; 128], base: usize) -> u64 {
    let mut result = 0u64;
    for row in 0..8 {
        result |= u64::from(input[base + row * 16]) << (row * 8);
    }
    result
}

// Static permutation tables for VBMI gather operations.
// Gather bytes at stride 16 for the 8 groups of each half (bases from BASE_PATTERN_*).
static GATHER_FIRST: [u8; 64] = [
    0, 16, 32, 48, 64, 80, 96, 112, // base 0
    8, 24, 40, 56, 72, 88, 104, 120, // base 8
    4, 20, 36, 52, 68, 84, 100, 116, // base 4
    12, 28, 44, 60, 76, 92, 108, 124, // base 12
    2, 18, 34, 50, 66, 82, 98, 114, // base 2
    10, 26, 42, 58, 74, 90, 106, 122, // base 10
    6, 22, 38, 54, 70, 86, 102, 118, // base 6
    14, 30, 46, 62, 78, 94, 110, 126, // base 14
];

static GATHER_SECOND: [u8; 64] = [
    1, 17, 33, 49, 65, 81, 97, 113, // base 1
    9, 25, 41, 57, 73, 89, 105, 121, // base 9
    5, 21, 37, 53, 69, 85, 101, 117, // base 5
    13, 29, 45, 61, 77, 93, 109, 125, // base 13
    3, 19, 35, 51, 67, 83, 99, 115, // base 3
    11, 27, 43, 59, 75, 91, 107, 123, // base 11
    7, 23, 39, 55, 71, 87, 103, 119, // base 7
    15, 31, 47, 63, 79, 95, 111, 127, // base 15
];

// 8x8 byte transpose permutation for the scatter phase.
// Input:  [g0b0..g0b7, g1b0..g1b7, ..., g7b0..g7b7] (8 groups of 8 bytes)
// Output: [g0b0,g1b0,..,g7b0, g0b1,g1b1,..,g7b1, ...] (8 rows of 8 bytes)
static SCATTER_8X8: [u8; 64] = [
    0, 8, 16, 24, 32, 40, 48, 56, // byte 0 from each group
    1, 9, 17, 25, 33, 41, 49, 57, // byte 1 from each group
    2, 10, 18, 26, 34, 42, 50, 58, // byte 2 from each group
    3, 11, 19, 27, 35, 43, 51, 59, // byte 3 from each group
    4, 12, 20, 28, 36, 44, 52, 60, // byte 4 from each group
    5, 13, 21, 29, 37, 45, 53, 61, // byte 5 from each group
    6, 14, 22, 30, 38, 46, 54, 62, // byte 6 from each group
    7, 15, 23, 31, 39, 47, 55, 63, // byte 7 from each group
];

/// Transpose 8x8 bit blocks of all 8 lanes of a ZMM register in parallel.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
#[allow(clippy::cast_possible_wrap, unsafe_op_in_unsafe_fn)]
unsafe fn bit_transpose_8x8_zmm(mut v: __m512i) -> __m512i {
    let mask1 = _mm512_set1_epi64(TRANSPOSE_2X2 as i64);
    let mask2 = _mm512_set1_epi64(TRANSPOSE_4X4 as i64);
    let mask3 = _mm512_set1_epi64(TRANSPOSE_8X8 as i64);

    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
    _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t))
}

/// Transpose 1024 bits using AVX-512 VBMI for vectorized gather and scatter.
///
/// Uses `vpermi2b` to gather bytes from stride-16 positions in parallel,
/// and `vpermb` for the final 8x8 byte transpose to output format.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VBMI support. Check with [`has_vbmi`].
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
#[inline]
#[allow(clippy::cast_ptr_alignment, unsafe_op_in_unsafe_fn)]
pub unsafe fn transpose_bits_vbmi(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);

    let in_lo = _mm512_loadu_si512(input.as_ptr().cast::<__m512i>());
    let in_hi = _mm512_loadu_si512(input.as_ptr().add(64).cast::<__m512i>());

    let idx_scatter = _mm512_loadu_si512(SCATTER_8X8.as_ptr().cast::<__m512i>());

    for (half, gather_idx) in [&GATHER_FIRST, &GATHER_SECOND].iter().enumerate() {
        let idx = _mm512_loadu_si512(gather_idx.as_ptr().cast::<__m512i>());
        let gathered = _mm512_permutex2var_epi8(in_lo, idx, in_hi);
        let transposed = bit_transpose_8x8_zmm(gathered);
        let scattered = _mm512_permutexvar_epi8(idx_scatter, transposed);
        _mm512_storeu_si512(
            output.as_mut_ptr().add(half * 64).cast::<__m512i>(),
            scattered,
        );
    }
}

/// Untranspose a `T`-width comparison mask (1024 bits) using AVX-512 VBMI.
///
/// Regardless of width the 128 bytes factor into 16 groups of 8 bytes, each an independent 8x8
/// bit transpose. The `T = 64` block keeps the original kernel (gather within each 64-byte half
/// with `vpermb`, then scatter); narrower widths use the width-generic [`untranspose_bits_vbmi_lt64`]
/// kernel whose groups span both halves. For `T = u64` this is the canonical `FastLanes` bit
/// untranspose.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VBMI support. Check with [`has_vbmi`].
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
#[inline]
#[allow(clippy::cast_ptr_alignment, unsafe_op_in_unsafe_fn)]
pub unsafe fn untranspose_bits_vbmi<T: FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    // `T::T` is a compile-time constant, so this branch is resolved at monomorphization and the
    // unused arm is eliminated. The `u64` arm is kept byte-identical to the original kernel.
    if T::T != 64 {
        untranspose_bits_vbmi_lt64::<T>(input, output);
        return;
    }

    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);

    // In the transposed layout the 8 bytes of a group are consecutive; this gather
    // collects them into the eight u64 lanes of a ZMM register.
    let gather_indices: [u8; 64] = SCATTER_8X8;
    let idx = _mm512_loadu_si512(gather_indices.as_ptr().cast::<__m512i>());

    for (half, groups) in [BASE_PATTERN_FIRST, BASE_PATTERN_SECOND].iter().enumerate() {
        let in_half = _mm512_loadu_si512(input.as_ptr().add(half * 64).cast::<__m512i>());
        let gathered = _mm512_permutexvar_epi8(idx, in_half);
        let transposed = bit_transpose_8x8_zmm(gathered);

        let mut lanes = [0u64; 8];
        _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), transposed);
        for (group, &base) in groups.iter().enumerate() {
            for row in 0..8 {
                output[base + row * 16] = (lanes[group] >> (row * 8)) as u8;
            }
        }
    }
}

/// Width-generic (`T::T < 64`) VBMI untranspose for the narrow comparison-mask widths.
///
/// Mirrors the width-generic NEON kernel: `vpermi2b` gathers the 16 eight-byte groups into
/// group-major order across two ZMM registers (each 64-bit lane is one group), every lane is
/// 8x8 bit-transposed in parallel, then a second pair of `vpermi2b` scatters the transposed
/// groups to their logical byte positions. For these widths a group spans both 64-byte halves,
/// so the two-source `vpermi2b` is required (unlike the single-source `vpermb` in the `u64` path).
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VBMI support. Check with [`has_vbmi`].
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
#[inline]
#[allow(clippy::cast_ptr_alignment, unsafe_op_in_unsafe_fn)]
unsafe fn untranspose_bits_vbmi_lt64<T: FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);
    let (gather_idx, scatter_idx) = group_tables::<T>();

    let in_lo = _mm512_loadu_si512(input.as_ptr().cast::<__m512i>());
    let in_hi = _mm512_loadu_si512(input.as_ptr().add(64).cast::<__m512i>());

    // Gather the 16 groups (8 bytes each) into group-major order: groups 0..7 in `grp0`,
    // groups 8..15 in `grp1`. Each 64-bit lane holds one group's 8 bytes.
    let g_lo = _mm512_loadu_si512(gather_idx.as_ptr().cast::<__m512i>());
    let g_hi = _mm512_loadu_si512(gather_idx.as_ptr().add(64).cast::<__m512i>());
    let grp0 = _mm512_permutex2var_epi8(in_lo, g_lo, in_hi);
    let grp1 = _mm512_permutex2var_epi8(in_lo, g_hi, in_hi);

    // 8x8 bit-transpose every group (all 8 lanes of each register in parallel).
    let t0 = bit_transpose_8x8_zmm(grp0);
    let t1 = bit_transpose_8x8_zmm(grp1);

    // Scatter the transposed groups to their logical byte positions.
    let s_lo = _mm512_loadu_si512(scatter_idx.as_ptr().cast::<__m512i>());
    let s_hi = _mm512_loadu_si512(scatter_idx.as_ptr().add(64).cast::<__m512i>());
    _mm512_storeu_si512(
        output.as_mut_ptr().cast::<__m512i>(),
        _mm512_permutex2var_epi8(t0, s_lo, t1),
    );
    _mm512_storeu_si512(
        output.as_mut_ptr().add(64).cast::<__m512i>(),
        _mm512_permutex2var_epi8(t0, s_hi, t1),
    );
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "runtime")]
    use super::*;
    #[cfg(feature = "runtime")]
    use crate::bit_transpose::transpose_bits_baseline;
    #[cfg(feature = "runtime")]
    use crate::bit_transpose::untranspose_bits_baseline;
    #[cfg(feature = "runtime")]
    use alloc::{format, string::ToString};
    #[cfg(feature = "runtime")]
    use hegel::TestCase;
    #[cfg(feature = "runtime")]
    use hegel::generators as gs;

    #[cfg(feature = "runtime")]
    #[hegel::test]
    fn test_bmi2_matches_baseline(tc: TestCase) {
        if !has_bmi2() {
            return;
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut baseline_out = [u64::MAX; 16];
        let mut bmi2_out = [u64::MAX; 16];

        transpose_bits_baseline(&input, &mut baseline_out);
        // SAFETY: guarded by `has_bmi2`.
        unsafe { transpose_bits_bmi2(&input, &mut bmi2_out) };

        assert_eq!(baseline_out, bmi2_out);
    }

    #[cfg(feature = "runtime")]
    #[hegel::test]
    fn test_bmi2_roundtrip(tc: TestCase) {
        if !has_bmi2() {
            return;
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut transposed = [u64::MAX; 16];
        let mut roundtrip = [u64::MAX; 16];

        // SAFETY: guarded by `has_bmi2`.
        unsafe {
            transpose_bits_bmi2(&input, &mut transposed);
            untranspose_bits_bmi2::<u64>(&transposed, &mut roundtrip);
        }

        assert_eq!(input, roundtrip);
    }

    /// The width-generic BMI2 untranspose must match the width-parameterized baseline for every
    /// element width.
    #[cfg(feature = "runtime")]
    #[hegel::test]
    fn test_bmi2_untranspose_all_widths_match_baseline(tc: TestCase) {
        fn check<T: FastLanes>(input: &[u64; 16]) {
            let mut baseline_out = [u64::MAX; 16];
            let mut bmi2_out = [u64::MAX; 16];

            untranspose_bits_baseline::<T>(input, &mut baseline_out);
            // SAFETY: guarded by `has_bmi2`.
            unsafe { untranspose_bits_bmi2::<T>(input, &mut bmi2_out) };

            assert_eq!(
                baseline_out,
                bmi2_out,
                "BMI2 untranspose != baseline for type={}",
                core::any::type_name::<T>()
            );
        }
        if !has_bmi2() {
            return;
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        check::<u8>(&input);
        check::<u16>(&input);
        check::<u32>(&input);
        check::<u64>(&input);
    }

    #[cfg(feature = "runtime")]
    #[hegel::test]
    fn test_vbmi_matches_baseline(tc: TestCase) {
        if !has_vbmi() {
            return;
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut baseline_out = [u64::MAX; 16];
        let mut vbmi_out = [u64::MAX; 16];

        transpose_bits_baseline(&input, &mut baseline_out);
        // SAFETY: guarded by `has_vbmi`.
        unsafe { transpose_bits_vbmi(&input, &mut vbmi_out) };

        assert_eq!(baseline_out, vbmi_out);
    }

    #[cfg(feature = "runtime")]
    #[hegel::test]
    fn test_vbmi_roundtrip(tc: TestCase) {
        if !has_vbmi() {
            return;
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut transposed = [u64::MAX; 16];
        let mut roundtrip = [u64::MAX; 16];

        // SAFETY: guarded by `has_vbmi`.
        unsafe {
            transpose_bits_vbmi(&input, &mut transposed);
            untranspose_bits_vbmi::<u64>(&transposed, &mut roundtrip);
        }

        assert_eq!(input, roundtrip);
    }

    /// The width-generic VBMI untranspose must match the width-parameterized baseline for every
    /// element width.
    #[cfg(feature = "runtime")]
    #[hegel::test]
    fn test_vbmi_untranspose_all_widths_match_baseline(tc: TestCase) {
        fn check<T: FastLanes>(input: &[u64; 16]) {
            let mut baseline_out = [u64::MAX; 16];
            let mut vbmi_out = [u64::MAX; 16];

            untranspose_bits_baseline::<T>(input, &mut baseline_out);
            // SAFETY: guarded by `has_vbmi`.
            unsafe { untranspose_bits_vbmi::<T>(input, &mut vbmi_out) };

            assert_eq!(
                baseline_out,
                vbmi_out,
                "VBMI untranspose != baseline for type={}",
                core::any::type_name::<T>()
            );
        }
        if !has_vbmi() {
            return;
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        check::<u8>(&input);
        check::<u16>(&input);
        check::<u32>(&input);
        check::<u64>(&input);
    }
}
