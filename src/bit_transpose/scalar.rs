//! Portable scalar transpose using a 64-bit gather and the classic 8x8 bit-matrix
//! transpose. Used as the fallback when no SIMD implementation is available.
//!
//! Conceptually this mirrors the SIMD kernels: each of the 16 byte-groups is gathered
//! into a `u64`, run through an 8x8 bit transpose, and scattered back out. The SIMD
//! kernels do eight groups at once in a single register; the scalar code does them one
//! at a time and relies on the compiler to unroll the fixed-length loops.

use crate::FL_ORDER;
use crate::FastLanes;
use crate::bit_transpose::BASE_PATTERN_FIRST;
use crate::bit_transpose::BASE_PATTERN_SECOND;
use crate::bit_transpose::TRANSPOSE_2X2;
use crate::bit_transpose::TRANSPOSE_4X4;
use crate::bit_transpose::TRANSPOSE_8X8;
use crate::bit_transpose::as_byte_array;
use crate::bit_transpose::as_byte_array_mut;

/// The two halves of the `FastLanes` byte-group ordering.
const HALVES: [[usize; 8]; 2] = [BASE_PATTERN_FIRST, BASE_PATTERN_SECOND];

/// Transpose 8x8 bit blocks within a `u64` (each byte is a row).
#[inline]
fn transpose_8x8(mut x: u64) -> u64 {
    // Step 1: Transpose 2x2 bit blocks
    let t = (x ^ (x >> 7)) & TRANSPOSE_2X2;
    x = x ^ t ^ (t << 7);
    // Step 2: Transpose 4x4 bit blocks
    let t = (x ^ (x >> 14)) & TRANSPOSE_4X4;
    x = x ^ t ^ (t << 14);
    // Step 3: Transpose 8x8 bit blocks
    let t = (x ^ (x >> 28)) & TRANSPOSE_8X8;
    x ^ t ^ (t << 28)
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

/// Scatter a `u64` to 8 output bytes at stride 16.
#[inline]
fn scatter(output: &mut [u8; 128], base: usize, val: u64) {
    for row in 0..8 {
        output[base + row * 16] = (val >> (row * 8)) as u8;
    }
}

/// Untranspose one 1024-bit block using the scalar implementation.
///
/// The bit-level equivalent of [`crate::Transpose::untranspose`]; the inverse of
/// [`transpose_bits::<u64>`](transpose_bits).
#[inline]
pub fn untranspose_bits(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);
    for (half, groups) in HALVES.iter().enumerate() {
        for (group, &base) in groups.iter().enumerate() {
            let row = transpose_8x8(gather(input, base));
            for bit in 0..8 {
                output[half * 64 + bit * 8 + group] = (row >> (bit * 8)) as u8;
            }
        }
    }
}

/// Transpose one 1024-bit block into `FastLanes` layout, scalar implementation.
///
/// For `T = u64` this is the canonical `FastLanes` bit transpose, the bit-level equivalent of
/// [`crate::Transpose::transpose`] (`output[i] = input[transpose(i)]`).
///
/// Narrower `T` bring a `T`-width comparison mask into logical row order. The mask produced by
/// `unpack_cmp` for an element width of `T::T` bits is laid out as `T::LANES` words of `T::T`
/// bits (LSB-first per lane); the bit at position `lane * T::T + row` holds the comparison for
/// logical index `index(row, lane)` (see the `unpack!` macro). This moves each mask bit to that
/// logical index.
///
/// Regardless of width the 128 bytes always factor into exactly 16 groups of 8 bytes, each
/// group being an independent 8x8 bit transpose. The width only changes which input bytes form
/// a group (the gather stride) and where the transposed bytes land (the scatter base).
#[inline]
pub fn transpose_bits<T: FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);

    // `bytes` = bytes per lane word (T::T / 8); `lhi_count` = lanes / 8 (= 128 / T::T). Their
    // product is always 16 groups.
    let bytes = T::T / 8;
    let lhi_count = 128 / T::T;
    for lhi in 0..lhi_count {
        for hi in 0..bytes {
            // Gather the 8 bytes of group `(lhi, hi)`: byte `llo` of the lane word for the lanes
            // sharing this group sits at stride `bytes`.
            let gather_base = lhi * T::T + hi;
            let mut packed = 0u64;
            for llo in 0..8 {
                packed |= u64::from(input[gather_base + llo * bytes]) << (llo * 8);
            }
            // After the 8x8 transpose the byte index is the comparison row's low bits, scattered
            // at stride 16; the FL_ORDER permutation of `hi` picks the base byte.
            let scatter_base = FL_ORDER[hi] * 2 + lhi;
            scatter(output, scatter_base, transpose_8x8(packed));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_transpose::transpose_bits_baseline;
    use crate::bit_transpose::untranspose_bits_baseline;
    use alloc::{format, string::ToString};
    use hegel::TestCase;
    use hegel::generators as gs;

    #[hegel::test]
    fn test_scalar_untranspose_matches_baseline(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut baseline_out = [u64::MAX; 16];
        let mut scalar_out = [u64::MAX; 16];

        untranspose_bits_baseline(&input, &mut baseline_out);
        untranspose_bits(&input, &mut scalar_out);

        assert_eq!(baseline_out, scalar_out);
    }

    #[hegel::test]
    fn test_scalar_roundtrip(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut transposed = [u64::MAX; 16];
        let mut roundtrip = [u64::MAX; 16];

        transpose_bits::<u64>(&input, &mut transposed);
        untranspose_bits(&transposed, &mut roundtrip);

        assert_eq!(input, roundtrip);
    }

    #[test]
    fn test_all_zeros() {
        let input = [0u64; 16];
        let mut output = [u64::MAX; 16];

        transpose_bits::<u64>(&input, &mut output);
        assert_eq!(output, [0u64; 16]);

        untranspose_bits(&input, &mut output);
        assert_eq!(output, [0u64; 16]);
    }

    #[test]
    fn test_all_ones() {
        let input = [u64::MAX; 16];
        let mut output = [0u64; 16];

        transpose_bits::<u64>(&input, &mut output);
        assert_eq!(output, [u64::MAX; 16]);

        untranspose_bits(&input, &mut output);
        assert_eq!(output, [u64::MAX; 16]);
    }

    /// The generic transpose must match the width-parameterized baseline for every element
    /// width, not just `u64`.
    #[hegel::test]
    fn test_transpose_all_widths_match_baseline(tc: TestCase) {
        fn check<T: FastLanes>(input: &[u64; 16]) {
            let mut baseline_out = [u64::MAX; 16];
            let mut scalar_out = [u64::MAX; 16];

            transpose_bits_baseline::<T>(input, &mut baseline_out);
            transpose_bits::<T>(input, &mut scalar_out);

            assert_eq!(
                baseline_out,
                scalar_out,
                "scalar transpose != baseline for type={}",
                core::any::type_name::<T>()
            );
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        check::<u8>(&input);
        check::<u16>(&input);
        check::<u32>(&input);
        check::<u64>(&input);
    }
}
