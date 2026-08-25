//! Fast implementations of the `FastLanes` 1024-bit transpose.
//!
//! The `FastLanes` transpose is a fixed permutation of 1024 bits (16 × `u64`,
//! i.e. 128 bytes) that enables SIMD parallelism for encodings like delta and
//! RLE, and for transposing validity bitmaps. This module provides architecture
//! specific implementations of both the transpose and its inverse.
//!
//! The bit-level permutation is the same one that [`crate::Transpose`] applies to
//! elements: bit `i` of the output of [`transpose_bits`] is bit
//! [`crate::transpose`]`(i)` of the input, exactly as `Transpose::transpose` sets
//! `output[i] = input[transpose(i)]`. [`untranspose_bits`] is the inverse, matching
//! `Transpose::untranspose`.
//!
//! The key insight is that each output byte is formed by extracting the SAME bit
//! position from 8 different input bytes at stride 16. The input byte groups follow
//! the [`crate::FL_ORDER`] permutation pattern.
//!
//! # Choosing an implementation
//!
//! [`transpose_bits`] / [`untranspose_bits`] dispatch to the fastest available
//! implementation. When the crate is built with the `runtime` feature this
//! dispatch is performed at runtime via `no_std` CPU feature detection; otherwise
//! it is resolved at compile time from the enabled `target_feature`s, falling back
//! to the portable scalar implementation.
//!
//! Every implementation is also exposed directly so that a downstream crate can
//! select one explicitly even in a `no_std` build: [`scalar`], [`x86`]
//! (`bmi2` / `avx512vbmi`), and [`aarch64`] (`neon`).
//!
//! All entry points operate over `[u64; 16]` (one 1024-bit block); bytes are
//! interpreted in little-endian order.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

/// Base indices for the first 64 output bytes (lanes 0-7).
/// Each entry indicates the starting input byte index for that output byte group.
/// Pattern: `[0*2, 4*2, 2*2, 6*2, 1*2, 5*2, 3*2, 7*2]` = `[0, 8, 4, 12, 2, 10, 6, 14]`
pub(crate) const BASE_PATTERN_FIRST: [usize; 8] = [0, 8, 4, 12, 2, 10, 6, 14];

/// Base indices for the second 64 output bytes (lanes 8-15).
/// Pattern: first pattern + 1 = `[1, 9, 5, 13, 3, 11, 7, 15]`
pub(crate) const BASE_PATTERN_SECOND: [usize; 8] = [1, 9, 5, 13, 3, 11, 7, 15];

/// Masks for transposing 8x8 bit blocks.
pub(crate) const TRANSPOSE_2X2: u64 = 0x00AA_00AA_00AA_00AA;
pub(crate) const TRANSPOSE_4X4: u64 = 0x0000_CCCC_0000_CCCC;
pub(crate) const TRANSPOSE_8X8: u64 = 0x0000_0000_F0F0_F0F0;

/// Group-major gather/scatter permutation tables shared by the SIMD width-generic transpose
/// kernels (NEON and AVX-512 VBMI). Only built for the architectures that have such a kernel.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod group_perm {
    /// Byte-gather indices that pull the 8 bytes of every group into contiguous group-major order
    /// for an element width of `tb` bits. Group `g = lhi * (tb/8) + hi` collects byte `llo` of its
    /// lane words from input byte `lhi*tb + hi + llo*(tb/8)`. See
    /// [`crate::bit_transpose::scalar::transpose_bits`].
    const fn gather_indices(tb: usize) -> [u8; 128] {
        let bytes = tb / 8;
        let mut idx = [0u8; 128];
        let mut g = 0;
        while g < 16 {
            let lhi = g / bytes;
            let hi = g % bytes;
            let gather_base = lhi * tb + hi;
            let mut llo = 0;
            while llo < 8 {
                idx[g * 8 + llo] = (gather_base + llo * bytes) as u8;
                llo += 1;
            }
            g += 1;
        }
        idx
    }

    /// Byte-scatter indices applied after the per-group 8x8 transpose: transposed byte `g*8 + lo`
    /// lands at logical byte `FL_ORDER[hi]*2 + lhi + lo*16`. Expressed as a gather table:
    /// `idx[logical_byte] = g*8 + lo`.
    const fn scatter_indices(tb: usize) -> [u8; 128] {
        let bytes = tb / 8;
        let mut idx = [0u8; 128];
        let mut g = 0;
        while g < 16 {
            let lhi = g / bytes;
            let hi = g % bytes;
            let scatter_base = crate::FL_ORDER[hi] * 2 + lhi;
            let mut lo = 0;
            while lo < 8 {
                idx[scatter_base + lo * 16] = (g * 8 + lo) as u8;
                lo += 1;
            }
            g += 1;
        }
        idx
    }

    static GATHER_8: [u8; 128] = gather_indices(8);
    static GATHER_16: [u8; 128] = gather_indices(16);
    static GATHER_32: [u8; 128] = gather_indices(32);
    static GATHER_64: [u8; 128] = gather_indices(64);

    static SCATTER_8: [u8; 128] = scatter_indices(8);
    static SCATTER_16: [u8; 128] = scatter_indices(16);
    static SCATTER_32: [u8; 128] = scatter_indices(32);
    static SCATTER_64: [u8; 128] = scatter_indices(64);

    /// Select the `(gather, scatter)` group-major permutation tables for an element width.
    #[inline]
    pub(crate) fn group_tables<T: crate::FastLanes>() -> (&'static [u8; 128], &'static [u8; 128]) {
        match T::T {
            8 => (&GATHER_8, &SCATTER_8),
            16 => (&GATHER_16, &SCATTER_16),
            32 => (&GATHER_32, &SCATTER_32),
            _ => (&GATHER_64, &SCATTER_64),
        }
    }
}

/// Reinterpret a 1024-bit block (`[u64; 16]`) as its 128 little-endian bytes.
#[inline]
#[must_use]
pub(crate) fn as_byte_array(block: &[u64; 16]) -> &[u8; 128] {
    // SAFETY: `[u64; 16]` and `[u8; 128]` have identical size (128 bytes). Every bit
    // pattern is a valid `u8`, and the source is over-aligned for a `u8` array.
    unsafe { &*block.as_ptr().cast::<[u8; 128]>() }
}

/// Reinterpret a mutable 1024-bit block (`[u64; 16]`) as its 128 little-endian bytes.
#[inline]
#[must_use]
pub(crate) fn as_byte_array_mut(block: &mut [u64; 16]) -> &mut [u8; 128] {
    // SAFETY: `[u64; 16]` and `[u8; 128]` have identical size (128 bytes). Every bit
    // pattern is a valid `u8`, and the source is over-aligned for a `u8` array.
    unsafe { &mut *block.as_mut_ptr().cast::<[u8; 128]>() }
}

/// Whether the AVX-512 VBMI implementation should be used.
///
/// Resolved at runtime with the `runtime` feature, otherwise at compile time.
#[cfg(target_arch = "x86_64")]
#[inline]
fn detect_vbmi() -> bool {
    #[cfg(feature = "runtime")]
    {
        x86::has_vbmi()
    }
    #[cfg(not(feature = "runtime"))]
    {
        cfg!(all(
            target_feature = "avx512vbmi",
            target_feature = "avx512bw",
            target_feature = "avx512f"
        ))
    }
}

/// Whether the BMI2 implementation should be used.
///
/// Resolved at runtime with the `runtime` feature, otherwise at compile time.
#[cfg(target_arch = "x86_64")]
#[inline]
fn detect_bmi2() -> bool {
    #[cfg(feature = "runtime")]
    {
        x86::has_bmi2()
    }
    #[cfg(not(feature = "runtime"))]
    {
        cfg!(target_feature = "bmi2")
    }
}

/// Untranspose 1024 bits out of `FastLanes` layout, dispatching to the best implementation.
///
/// This is the bit-level equivalent of [`crate::Transpose::untranspose`]: the inverse of
/// [`transpose_bits::<u64>`](transpose_bits).
#[inline]
pub fn untranspose_bits(input: &[u64; 16], output: &mut [u64; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if detect_vbmi() {
            // SAFETY: guarded by `detect_vbmi`.
            unsafe { x86::untranspose_bits_vbmi(input, output) }
        } else if detect_bmi2() {
            // SAFETY: guarded by `detect_bmi2`.
            unsafe { x86::untranspose_bits_bmi2(input, output) }
        } else {
            scalar::untranspose_bits(input, output);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is always available on aarch64.
    unsafe {
        aarch64::untranspose_bits_neon(input, output);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::untranspose_bits(input, output);
}

/// Transpose 1024 bits into `FastLanes` layout, dispatching to the best implementation.
///
/// For `T = u64` this is the canonical `FastLanes` bit transpose, the bit-level equivalent of
/// [`crate::Transpose::transpose`]. Narrower `T` apply the per-width permutation that brings the
/// `T`-width comparison mask produced by `unpack_cmp` into logical row order (see
/// [`crate::BitPackingCompare::unpack_cmp`]).
#[inline]
pub fn transpose_bits<T: crate::FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if detect_vbmi() {
            // SAFETY: guarded by `detect_vbmi`.
            unsafe { x86::transpose_bits_vbmi::<T>(input, output) }
        } else if detect_bmi2() {
            // SAFETY: guarded by `detect_bmi2`.
            unsafe { x86::transpose_bits_bmi2::<T>(input, output) }
        } else {
            scalar::transpose_bits::<T>(input, output);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is always available on aarch64.
    unsafe {
        aarch64::transpose_bits_neon::<T>(input, output);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::transpose_bits::<T>(input, output);
}

/// Reference untranspose built directly on top of [`crate::transpose`], one bit at a time.
///
/// Mirrors [`crate::Transpose::untranspose`]: `output[transpose(i)] = input[i]`.
#[cfg(test)]
pub(crate) fn untranspose_bits_baseline(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);
    *output = [0u8; 128];
    for in_bit in 0..1024 {
        let out_bit = crate::transpose(in_bit);
        let bit_val = (input[in_bit / 8] >> (in_bit % 8)) & 1;
        output[out_bit / 8] |= bit_val << (out_bit % 8);
    }
}

/// Reference transpose for a `T`-width comparison mask, one bit at a time.
///
/// Mask bit `b = lane * T::T + row` holds the comparison for logical index `index(row, lane)`
/// (the `unpack!` macro's formula), so we scatter each mask bit to its logical position. For
/// `T = u64` this is exactly [`crate::Transpose::transpose`] applied to bits
/// (`output[i] = input[transpose(i)]`), and the inverse of [`untranspose_bits_baseline`].
#[cfg(test)]
pub(crate) fn transpose_bits_baseline<T: crate::FastLanes>(
    input: &[u64; 16],
    output: &mut [u64; 16],
) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);
    *output = [0u8; 128];
    for b in 0..1024 {
        let lane = b / T::T;
        let row = b % T::T;
        let logical = crate::FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane;
        let bit_val = (input[b / 8] >> (b % 8)) & 1;
        output[logical / 8] |= bit_val << (logical % 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString};
    use hegel::TestCase;
    use hegel::generators as gs;

    /// Expand a 1024-bit block into 1024 elements, one bit per element.
    fn bits_to_elements(bits: &[u64; 16]) -> [u64; 1024] {
        let mut elements = [0u64; 1024];
        for (i, element) in elements.iter_mut().enumerate() {
            *element = (bits[i / 64] >> (i % 64)) & 1;
        }
        elements
    }

    /// Pack 1024 one-bit elements back into a 1024-bit block.
    fn elements_to_bits(elements: &[u64; 1024]) -> [u64; 16] {
        let mut bits = [0u64; 16];
        for (i, &element) in elements.iter().enumerate() {
            bits[i / 64] |= element << (i % 64);
        }
        bits
    }

    /// The bit-level transpose must apply the same permutation as the element-level
    /// [`crate::Transpose::transpose`].
    #[hegel::test]
    fn test_transpose_bits_matches_element_transpose(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));

        let mut expected_elements = [u64::MAX; 1024];
        crate::Transpose::transpose(&bits_to_elements(&input), &mut expected_elements);

        let mut out = [u64::MAX; 16];
        transpose_bits::<u64>(&input, &mut out);

        assert_eq!(out, elements_to_bits(&expected_elements));
    }

    /// The bit-level untranspose must apply the same permutation as the element-level
    /// [`crate::Transpose::untranspose`].
    #[hegel::test]
    fn test_untranspose_bits_matches_element_untranspose(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));

        let mut expected_elements = [u64::MAX; 1024];
        crate::Transpose::untranspose(&bits_to_elements(&input), &mut expected_elements);

        let mut out = [u64::MAX; 16];
        untranspose_bits(&input, &mut out);

        assert_eq!(out, elements_to_bits(&expected_elements));
    }

    /// Exhaustive one-hot check: input bit `transpose(i)` must land on output bit `i`, exactly
    /// as `Transpose::transpose` sets `output[i] = input[transpose(i)]`.
    #[test]
    fn test_transpose_bits_one_hot_matches_transpose_index() {
        for i in 0..1024 {
            let src = crate::transpose(i);
            let mut input = [0u64; 16];
            input[src / 64] |= 1u64 << (src % 64);

            let mut out = [u64::MAX; 16];
            transpose_bits::<u64>(&input, &mut out);

            let mut expected = [0u64; 16];
            expected[i / 64] |= 1u64 << (i % 64);
            assert_eq!(out, expected, "input bit {src} must land on output bit {i}");
        }
    }

    #[hegel::test]
    fn test_baseline_roundtrip(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut transposed = [u64::MAX; 16];
        let mut roundtrip = [u64::MAX; 16];

        transpose_bits_baseline::<u64>(&input, &mut transposed);
        untranspose_bits_baseline(&transposed, &mut roundtrip);

        assert_eq!(input, roundtrip);
    }

    #[hegel::test]
    fn test_untranspose_dispatch_matches_baseline(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut baseline_out = [u64::MAX; 16];
        let mut out = [u64::MAX; 16];

        untranspose_bits_baseline(&input, &mut baseline_out);
        untranspose_bits(&input, &mut out);

        assert_eq!(baseline_out, out);
    }

    #[hegel::test]
    fn test_transpose_dispatch_matches_baseline(tc: TestCase) {
        fn check<T: crate::FastLanes>(input: &[u64; 16]) {
            let mut baseline_out = [u64::MAX; 16];
            let mut out = [u64::MAX; 16];

            transpose_bits_baseline::<T>(input, &mut baseline_out);
            transpose_bits::<T>(input, &mut out);

            assert_eq!(
                baseline_out,
                out,
                "transpose dispatch doesn't match baseline for type={}",
                core::any::type_name::<T>()
            );
        }
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        check::<u8>(&input);
        check::<u16>(&input);
        check::<u32>(&input);
        check::<u64>(&input);
    }

    #[hegel::test]
    fn test_dispatch_roundtrip(tc: TestCase) {
        let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
        let mut transposed = [u64::MAX; 16];
        let mut roundtrip = [u64::MAX; 16];

        transpose_bits::<u64>(&input, &mut transposed);
        untranspose_bits(&transposed, &mut roundtrip);

        assert_eq!(input, roundtrip);
    }

    /// The shared `group_perm` gather/scatter tables are consumed only by the SIMD kernels
    /// (NEON on `aarch64`, VBMI on `x86_64`) — so on a given host most of them are never executed
    /// (e.g. on an x86 CI runner without AVX-512VBMI nothing touches them, since BMI2 computes
    /// its indices inline). These two tests exercise the tables directly, on every architecture,
    /// independent of any SIMD support.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    mod group_perm_tables {
        use super::*;

        /// Portable scalar re-implementation of the SIMD width-generic transpose, driven purely
        /// by the shared tables: gather the 16 byte-groups into group-major order, 8x8-transpose
        /// each group, then scatter back. Equivalent to the NEON/VBMI kernels but in plain Rust,
        /// so it validates the table *data* against the bit-level baseline on any host.
        fn transpose_via_tables<T: crate::FastLanes>(input: &[u64; 16]) -> [u64; 16] {
            fn transpose_8x8(mut x: u64) -> u64 {
                let t = (x ^ (x >> 7)) & TRANSPOSE_2X2;
                x = x ^ t ^ (t << 7);
                let t = (x ^ (x >> 14)) & TRANSPOSE_4X4;
                x = x ^ t ^ (t << 14);
                let t = (x ^ (x >> 28)) & TRANSPOSE_8X8;
                x ^ t ^ (t << 28)
            }

            let (gather, scatter) = group_perm::group_tables::<T>();
            let src = as_byte_array(input);

            // Gather: group-major order, `grouped[k] = src[gather[k]]`.
            let mut grouped = [0u8; 128];
            for k in 0..128 {
                grouped[k] = src[gather[k] as usize];
            }

            // 8x8 bit-transpose each of the 16 groups (8 bytes each).
            let mut transposed = [0u8; 128];
            for g in 0..16 {
                let mut word = 0u64;
                for b in 0..8 {
                    word |= u64::from(grouped[g * 8 + b]) << (b * 8);
                }
                let w = transpose_8x8(word);
                for b in 0..8 {
                    transposed[g * 8 + b] = (w >> (b * 8)) as u8;
                }
            }

            // Scatter: `out[k] = transposed[scatter[k]]`.
            let mut out = [0u64; 16];
            let dst = as_byte_array_mut(&mut out);
            for k in 0..128 {
                dst[k] = transposed[scatter[k] as usize];
            }
            out
        }

        #[hegel::test]
        fn tables_match_baseline_all_widths(tc: TestCase) {
            fn check<T: crate::FastLanes>(input: &[u64; 16]) {
                let mut baseline = [u64::MAX; 16];
                transpose_bits_baseline::<T>(input, &mut baseline);
                assert_eq!(
                    transpose_via_tables::<T>(input),
                    baseline,
                    "group_perm tables != baseline for type={}",
                    core::any::type_name::<T>()
                );
            }
            let input: [u64; 16] = tc.draw(gs::arrays(gs::integers::<u64>()));
            check::<u8>(&input);
            check::<u16>(&input);
            check::<u32>(&input);
            check::<u64>(&input);
        }

        /// Both tables must be permutations of `0..128` — every input byte is read exactly once
        /// (gather) and every output byte is written exactly once (scatter). A duplicated or
        /// dropped index would silently corrupt data, so assert bijectivity for every width.
        #[test]
        fn tables_are_permutations() {
            fn is_permutation(t: &[u8; 128]) -> bool {
                let mut seen = [false; 128];
                for &i in t {
                    if i as usize >= 128 || seen[i as usize] {
                        return false;
                    }
                    seen[i as usize] = true;
                }
                seen.iter().all(|&b| b)
            }
            fn check<T: crate::FastLanes>() {
                let (gather, scatter) = group_perm::group_tables::<T>();
                assert!(
                    is_permutation(gather),
                    "gather table for {} is not a permutation",
                    core::any::type_name::<T>()
                );
                assert!(
                    is_permutation(scatter),
                    "scatter table for {} is not a permutation",
                    core::any::type_name::<T>()
                );
            }
            check::<u8>();
            check::<u16>();
            check::<u32>();
            check::<u64>();
        }
    }
}
