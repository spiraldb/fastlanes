//! `AArch64` NEON transpose implementation using TBL-based gather/scatter.

use core::arch::aarch64::uint64x2_t;
use core::arch::aarch64::vandq_u64;
use core::arch::aarch64::vdupq_n_u8;
use core::arch::aarch64::vdupq_n_u64;
use core::arch::aarch64::veorq_u64;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vld1q_u8_x4;
use core::arch::aarch64::vorrq_u8;
use core::arch::aarch64::vqtbl4q_u8;
use core::arch::aarch64::vreinterpretq_u8_u64;
use core::arch::aarch64::vreinterpretq_u64_u8;
use core::arch::aarch64::vshlq_n_u64;
use core::arch::aarch64::vshrq_n_u64;
use core::arch::aarch64::vst1q_u8;
use core::arch::aarch64::vsubq_u8;

use crate::FastLanes;
use crate::bit_transpose::TRANSPOSE_2X2;
use crate::bit_transpose::TRANSPOSE_4X4;
use crate::bit_transpose::TRANSPOSE_8X8;
use crate::bit_transpose::as_byte_array;
use crate::bit_transpose::as_byte_array_mut;
use crate::bit_transpose::group_perm::group_tables;

/// Gather indices for the first half from input[0..64] (low 4 bytes of each group).
static GATHER_FIRST_LO: [[u8; 16]; 4] = [
    [
        0, 16, 32, 48, 0xFF, 0xFF, 0xFF, 0xFF, 8, 24, 40, 56, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        4, 20, 36, 52, 0xFF, 0xFF, 0xFF, 0xFF, 12, 28, 44, 60, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        2, 18, 34, 50, 0xFF, 0xFF, 0xFF, 0xFF, 10, 26, 42, 58, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        6, 22, 38, 54, 0xFF, 0xFF, 0xFF, 0xFF, 14, 30, 46, 62, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
];

/// Gather indices for the first half from input[64..128] (high 4 bytes of each group).
static GATHER_FIRST_HI: [[u8; 16]; 4] = [
    [
        0xFF, 0xFF, 0xFF, 0xFF, 0, 16, 32, 48, 0xFF, 0xFF, 0xFF, 0xFF, 8, 24, 40, 56,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 4, 20, 36, 52, 0xFF, 0xFF, 0xFF, 0xFF, 12, 28, 44, 60,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 2, 18, 34, 50, 0xFF, 0xFF, 0xFF, 0xFF, 10, 26, 42, 58,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 6, 22, 38, 54, 0xFF, 0xFF, 0xFF, 0xFF, 14, 30, 46, 62,
    ],
];

/// Gather indices for the second half from input[0..64].
static GATHER_SECOND_LO: [[u8; 16]; 4] = [
    [
        1, 17, 33, 49, 0xFF, 0xFF, 0xFF, 0xFF, 9, 25, 41, 57, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        5, 21, 37, 53, 0xFF, 0xFF, 0xFF, 0xFF, 13, 29, 45, 61, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        3, 19, 35, 51, 0xFF, 0xFF, 0xFF, 0xFF, 11, 27, 43, 59, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        7, 23, 39, 55, 0xFF, 0xFF, 0xFF, 0xFF, 15, 31, 47, 63, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
];

/// Gather indices for the second half from input[64..128].
static GATHER_SECOND_HI: [[u8; 16]; 4] = [
    [
        0xFF, 0xFF, 0xFF, 0xFF, 1, 17, 33, 49, 0xFF, 0xFF, 0xFF, 0xFF, 9, 25, 41, 57,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 5, 21, 37, 53, 0xFF, 0xFF, 0xFF, 0xFF, 13, 29, 45, 61,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 3, 19, 35, 51, 0xFF, 0xFF, 0xFF, 0xFF, 11, 27, 43, 59,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 7, 23, 39, 55, 0xFF, 0xFF, 0xFF, 0xFF, 15, 31, 47, 63,
    ],
];

/// 8x8 byte transpose (scatter) permutation split into 4 × 16-byte chunks for NEON TBL.
/// Same permutation as x86 `SCATTER_8X8`, split for 16-byte NEON registers.
static SCATTER_8X8_NEON: [[u8; 16]; 4] = [
    [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57],
    [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59],
    [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61],
    [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63],
];

/// Perform 8x8 bit transpose on two u64s packed in a `uint64x2_t`.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn bit_transpose_8x8_neon(mut v: uint64x2_t) -> uint64x2_t {
    let mask1 = vdupq_n_u64(TRANSPOSE_2X2);
    let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<7>(v)), mask1);
    v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<7>(t));

    let mask2 = vdupq_n_u64(TRANSPOSE_4X4);
    let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<14>(v)), mask2);
    v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<14>(t));

    let mask3 = vdupq_n_u64(TRANSPOSE_8X8);
    let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<28>(v)), mask3);
    veorq_u64(veorq_u64(v, t), vshlq_n_u64::<28>(t))
}

/// Transpose one 1024-bit block using NEON with TBL-based gather and scatter.
///
/// # Safety
/// Requires `AArch64` with NEON (always available on `AArch64`).
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn transpose_bits_neon(input: &[u64; 16], output: &mut [u64; 16]) {
    let input = as_byte_array(input);
    let output = as_byte_array_mut(output);

    let tbl_lo = vld1q_u8_x4(input.as_ptr());
    let tbl_hi = vld1q_u8_x4(input.as_ptr().add(64));

    let scatter0 = vld1q_u8(SCATTER_8X8_NEON[0].as_ptr());
    let scatter1 = vld1q_u8(SCATTER_8X8_NEON[1].as_ptr());
    let scatter2 = vld1q_u8(SCATTER_8X8_NEON[2].as_ptr());
    let scatter3 = vld1q_u8(SCATTER_8X8_NEON[3].as_ptr());

    let mut buf = [0u8; 64];
    for (i, (gather_lo, gather_high)) in [
        (GATHER_FIRST_LO, GATHER_FIRST_HI),
        (GATHER_SECOND_LO, GATHER_SECOND_HI),
    ]
    .iter()
    .enumerate()
    {
        for pair in 0..4 {
            let idx_lo = vld1q_u8(gather_lo[pair].as_ptr());
            let idx_hi = vld1q_u8(gather_high[pair].as_ptr());

            let from_lo = vqtbl4q_u8(tbl_lo, idx_lo);
            let from_hi = vqtbl4q_u8(tbl_hi, idx_hi);
            let gathered = vorrq_u8(from_lo, from_hi);

            let v = bit_transpose_8x8_neon(vreinterpretq_u64_u8(gathered));
            vst1q_u8(buf.as_mut_ptr().add(pair * 16), vreinterpretq_u8_u64(v));
        }

        let result_tbl = vld1q_u8_x4(buf.as_ptr());
        vst1q_u8(
            output.as_mut_ptr().add(i * 64),
            vqtbl4q_u8(result_tbl, scatter0),
        );
        vst1q_u8(
            output.as_mut_ptr().add(i * 64 + 16),
            vqtbl4q_u8(result_tbl, scatter1),
        );
        vst1q_u8(
            output.as_mut_ptr().add(i * 64 + 32),
            vqtbl4q_u8(result_tbl, scatter2),
        );
        vst1q_u8(
            output.as_mut_ptr().add(i * 64 + 48),
            vqtbl4q_u8(result_tbl, scatter3),
        );
    }
}

/// Apply an arbitrary 128-byte gather permutation (`out[k] = src[idx[k]]`) using NEON TBL.
///
/// `vqtbl4q_u8` only addresses 64 source bytes, returning 0 for indices `>= 64`. We split `src`
/// into its low and high 64 bytes and OR the two table lookups: high indices select from `hi`
/// (after subtracting 64), low indices wrap past 64 in the `hi` lookup and so contribute 0.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn permute_128(src: &[u8; 128], idx: &[u8; 128]) -> [u8; 128] {
    let lo = vld1q_u8_x4(src.as_ptr());
    let hi = vld1q_u8_x4(src.as_ptr().add(64));
    let bias = vdupq_n_u8(64);

    let mut out = [0u8; 128];
    for k in 0..8 {
        let want = vld1q_u8(idx.as_ptr().add(k * 16));
        let want_hi = vsubq_u8(want, bias);
        let from_lo = vqtbl4q_u8(lo, want);
        let from_hi = vqtbl4q_u8(hi, want_hi);
        vst1q_u8(out.as_mut_ptr().add(k * 16), vorrq_u8(from_lo, from_hi));
    }
    out
}

/// Untranspose a `T`-width comparison mask into logical row order using NEON.
///
/// Regardless of width the 128 bytes factor into 16 groups of 8 bytes, each an independent 8x8
/// bit transpose (see [`crate::scalar::untranspose_bits`]). We TBL-gather the groups into
/// group-major order, run the per-group 8x8 transpose, then TBL-scatter to logical positions.
///
/// # Safety
/// Requires `AArch64` with NEON (always available on `AArch64`).
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn untranspose_bits_neon<T: FastLanes>(input: &[u64; 16], output: &mut [u64; 16]) {
    let (gather_idx, scatter_idx) = group_tables::<T>();

    // Gather the 16 groups into contiguous group-major order.
    let gathered = permute_128(as_byte_array(input), gather_idx);

    // 8x8 bit-transpose each group; a `uint64x2_t` covers two consecutive groups (16 bytes).
    let mut transposed = [0u8; 128];
    for r in 0..8 {
        let v = bit_transpose_8x8_neon(vreinterpretq_u64_u8(vld1q_u8(
            gathered.as_ptr().add(r * 16),
        )));
        vst1q_u8(transposed.as_mut_ptr().add(r * 16), vreinterpretq_u8_u64(v));
    }

    // Scatter the transposed groups to their logical byte positions.
    *as_byte_array_mut(output) = permute_128(&transposed, scatter_idx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_transpose::generate_test_data;
    use crate::bit_transpose::transpose_bits_baseline;
    use crate::bit_transpose::untranspose_bits_baseline;

    #[test]
    fn test_neon_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u64; 16];
            let mut tbl_out = [0u64; 16];

            transpose_bits_baseline(&input, &mut baseline_out);
            // SAFETY: NEON is always available on aarch64.
            unsafe { transpose_bits_neon(&input, &mut tbl_out) };

            assert_eq!(
                baseline_out, tbl_out,
                "NEON transpose doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_neon_roundtrip() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u64; 16];
            let mut roundtrip = [0u64; 16];

            // SAFETY: NEON is always available on aarch64.
            unsafe {
                transpose_bits_neon(&input, &mut transposed);
                untranspose_bits_neon::<u64>(&transposed, &mut roundtrip);
            }

            assert_eq!(input, roundtrip, "NEON roundtrip failed for seed {seed}");
        }
    }

    #[test]
    fn test_untranspose_neon_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u64; 16];
            let mut tbl_out = [0u64; 16];

            untranspose_bits_baseline::<u64>(&input, &mut baseline_out);
            // SAFETY: NEON is always available on aarch64.
            unsafe { untranspose_bits_neon::<u64>(&input, &mut tbl_out) };

            assert_eq!(
                baseline_out, tbl_out,
                "NEON untranspose doesn't match baseline for seed {seed}"
            );
        }
    }

    /// The generic NEON untranspose must match the width-parameterized baseline for every
    /// element width.
    #[test]
    fn test_untranspose_neon_all_widths_match_baseline() {
        fn check<T: FastLanes>() {
            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u64; 16];
                let mut tbl_out = [0u64; 16];

                untranspose_bits_baseline::<T>(&input, &mut baseline_out);
                // SAFETY: NEON is always available on aarch64.
                unsafe { untranspose_bits_neon::<T>(&input, &mut tbl_out) };

                assert_eq!(
                    baseline_out,
                    tbl_out,
                    "NEON untranspose != baseline for type={} seed={seed}",
                    core::any::type_name::<T>()
                );
            }
        }
        check::<u8>();
        check::<u16>();
        check::<u32>();
        check::<u64>();
    }
}
