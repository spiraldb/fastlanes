//! Fast implementations of the `FastLanes` 1024-bit transpose.
//!
//! The `FastLanes` transpose is a fixed permutation of 1024 bits (16 × `u64`,
//! i.e. 128 bytes) that enables SIMD parallelism for encodings like delta and
//! RLE, and for transposing validity bitmaps. This module provides architecture
//! specific implementations of both the transpose and its inverse.
//!
//! The key insight is that each output byte is formed by extracting the SAME bit
//! position from 8 different input bytes at stride 16. The input byte groups follow
//! the [`crate::FL_ORDER`] permutation pattern.
//!
//! # Choosing an implementation
//!
//! [`transpose_bits`] / [`untranspose_bits`] dispatch to the fastest available
//! implementation. When the crate is built with the `std` feature this dispatch
//! is performed at runtime via CPU feature detection; otherwise it is resolved at
//! compile time from the enabled `target_feature`s, falling back to the portable
//! scalar implementation.
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

/// Group-major gather/scatter permutation tables shared by the SIMD width-generic untranspose
/// kernels (NEON and AVX-512 VBMI). Only built for the architectures that have such a kernel.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod group_perm {
    /// Byte-gather indices that pull the 8 bytes of every group into contiguous group-major order
    /// for an element width of `tb` bits. Group `g = lhi * (tb/8) + hi` collects byte `llo` of its
    /// lane words from input byte `lhi*tb + hi + llo*(tb/8)`. See
    /// [`crate::bit_transpose::scalar::untranspose_bits`].
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
/// Resolved at runtime with the `std` feature, otherwise at compile time.
#[cfg(target_arch = "x86_64")]
#[inline]
fn detect_vbmi() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(feature = "std"))]
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
/// Resolved at runtime with the `std` feature, otherwise at compile time.
#[cfg(target_arch = "x86_64")]
#[inline]
fn detect_bmi2() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "bmi2")
    }
}

/// Transpose 1024 bits into `FastLanes` layout, dispatching to the best implementation.
#[inline]
pub fn transpose_bits(input: &[u64; 16], output: &mut [u64; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if detect_vbmi() {
            // SAFETY: guarded by `detect_vbmi`.
            unsafe { x86::transpose_bits_vbmi(input, output) }
        } else if detect_bmi2() {
            // SAFETY: guarded by `detect_bmi2`.
            unsafe { x86::transpose_bits_bmi2(input, output) }
        } else {
            scalar::transpose_bits(input, output);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is always available on aarch64.
    unsafe {
        aarch64::transpose_bits_neon(input, output);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::transpose_bits(input, output);
}

/// Untranspose a `T`-width comparison mask (1024 bits) from `FastLanes` layout into logical row
/// order, dispatching to the best implementation. For `T = u64` this is the canonical `FastLanes`
/// bit untranspose; narrower `T` undo the per-lane packing produced by `unpack_cmp` for that width.
#[inline]
pub fn untranspose_bits<T: crate::FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if detect_vbmi() {
            // SAFETY: guarded by `detect_vbmi`.
            unsafe { x86::untranspose_bits_vbmi::<T>(input, output) }
        } else if detect_bmi2() {
            // SAFETY: guarded by `detect_bmi2`.
            unsafe { x86::untranspose_bits_bmi2::<T>(input, output) }
        } else {
            scalar::untranspose_bits::<T>(input, output);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is always available on aarch64.
    unsafe {
        aarch64::untranspose_bits_neon::<T>(input, output);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::untranspose_bits::<T>(input, output);
}

#[cfg(test)]
pub(crate) fn generate_test_data(seed: u8) -> [u64; 16] {
    let mut data = [0u64; 16];
    for (i, byte) in as_byte_array_mut(&mut data).iter_mut().enumerate() {
        *byte = seed.wrapping_mul(17).wrapping_add(i as u8).wrapping_mul(31);
    }
    data
}

/// Reference transpose built directly on top of [`crate::transpose`], one bit at a time.
#[cfg(test)]
pub(crate) fn transpose_bits_baseline(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);
    *output = [0u8; 128];
    for in_bit in 0..1024 {
        let out_bit = crate::transpose(in_bit);
        let bit_val = (input[in_bit / 8] >> (in_bit % 8)) & 1;
        output[out_bit / 8] |= bit_val << (out_bit % 8);
    }
}

/// Reference untranspose for a `T`-width comparison mask, one bit at a time.
///
/// Mask bit `b = lane * T::T + row` holds the comparison for logical index `index(row, lane)`
/// (the `unpack!` macro's formula), so we scatter each mask bit to its logical position. For
/// `T = u64` this is exactly the inverse permutation of [`transpose_bits_baseline`].
#[cfg(test)]
pub(crate) fn untranspose_bits_baseline<T: crate::FastLanes>(
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

    #[test]
    fn test_baseline_roundtrip() {
        let input = generate_test_data(42);
        let mut transposed = [0u64; 16];
        let mut roundtrip = [0u64; 16];

        transpose_bits_baseline(&input, &mut transposed);
        untranspose_bits_baseline::<u64>(&transposed, &mut roundtrip);

        assert_eq!(input, roundtrip);
    }

    #[test]
    fn test_dispatch_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u64; 16];
            let mut out = [0u64; 16];

            transpose_bits_baseline(&input, &mut baseline_out);
            transpose_bits(&input, &mut out);

            assert_eq!(
                baseline_out, out,
                "transpose dispatch doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_untranspose_dispatch_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u64; 16];
            let mut out = [0u64; 16];

            untranspose_bits_baseline::<u64>(&input, &mut baseline_out);
            untranspose_bits::<u64>(&input, &mut out);

            assert_eq!(
                baseline_out, out,
                "untranspose dispatch doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_dispatch_roundtrip() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u64; 16];
            let mut roundtrip = [0u64; 16];

            transpose_bits(&input, &mut transposed);
            untranspose_bits::<u64>(&transposed, &mut roundtrip);

            assert_eq!(
                input, roundtrip,
                "dispatch roundtrip failed for seed {seed}"
            );
        }
    }
}
