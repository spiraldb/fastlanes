//! Stack-only comparison of the byte-bool and packed-mask FastLanes comparison paths.
//!
//! The timed region operates on 1024 `u16` values packed at width 3. All input, scratch, and
//! output buffers are fixed-size stack arrays. Each timed sample executes `BATCH` blocks so the
//! sub-100ns kernels are measured well above the timer resolution.
//!
//! For native code generation use:
//! `RUSTFLAGS="-C target-cpu=native" cargo bench --bench bitpacking_cmp_stack`.
//! For the optional dot-product path use nightly and add `--features nightly-dotprod`.

#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly-dotprod"),
    feature(stdarch_neon_dotprod)
)]

fn main() {
    divan::main();
}

use fastlanes::{untranspose_bits, BitPacking, BitPackingCompare};
use std::hint::black_box;

const W: usize = 3;
const B: usize = 1024 * W / u16::BITS as usize;
const BATCH: usize = 256;
const COMPARE_VALUE: u16 = 3;

fn packed_input() -> [u16; B] {
    let values = std::array::from_fn(|i| (i as u16).wrapping_mul(17) & 7);
    let mut packed = [0u16; B];
    <u16 as BitPacking>::pack::<W, B>(&values, &mut packed);
    packed
}

fn expected_bytes() -> [bool; 1024] {
    std::array::from_fn(|i| ((i as u16).wrapping_mul(17) & 7) == COMPARE_VALUE)
}

fn expected_packed() -> [u64; 16] {
    let bytes = expected_bytes();
    let mut output = [0u64; 16];
    pack_bools_collect(&bytes, &mut output);
    output
}

/// The inner loop used by both Vortex `BitBufferMut::collect_bool` and Arrow
/// `MutableBuffer::collect_bool`, adapted to an existing stack output.
#[inline]
fn pack_bools_collect(input: &[bool; 1024], output: &mut [u64; 16]) {
    for (chunk, word) in output.iter_mut().enumerate() {
        let mut packed = 0u64;
        for bit in 0..64 {
            packed |= u64::from(input[chunk * 64 + bit]) << bit;
        }
        *word = packed;
    }
}

/// The inner strategy used by Arrow `BooleanBufferBuilder::append_slice` and Vortex's
/// `FromIterator<bool>`: zero the packed storage and set only true bits.
#[inline]
fn pack_bools_set_true(input: &[bool; 1024], output: &mut [u64; 16]) {
    output.fill(0);
    let output = output.as_mut_ptr().cast::<u8>();
    for (index, value) in input.iter().copied().enumerate() {
        if value {
            // SAFETY: `index < 1024`, hence `index / 8 < 128`, the output size in bytes.
            unsafe {
                *output.add(index / 8) |= 1 << (index % 8);
            }
        }
    }
}

/// AArch64 NEON equivalent of a movemask for byte booleans. NEON has no direct byte movemask,
/// so each group of 16 bytes is shifted by repeating bit positions and pairwise-added down to two
/// bytes. Four groups are narrowed into eight bytes and written with one vector store.
#[cfg(target_arch = "aarch64")]
#[inline]
fn pack_bools_collect_neon(input: &[bool; 1024], output: &mut [u64; 16]) {
    use core::arch::aarch64::{
        int8x16_t, uint64x2_t, vcombine_u16, vcombine_u32, vld1q_s8, vld1q_u8, vmovn_u16,
        vmovn_u32, vmovn_u64, vpaddlq_u16, vpaddlq_u32, vpaddlq_u8, vshlq_u8, vst1_u8,
    };

    #[inline(always)]
    unsafe fn pack_16(input: *const u8, shifts: int8x16_t) -> uint64x2_t {
        // SAFETY: the caller provides a readable 16-byte region.
        unsafe {
            let bytes = vld1q_u8(input);
            let weighted = vshlq_u8(bytes, shifts);
            let sums_16 = vpaddlq_u8(weighted);
            let sums_32 = vpaddlq_u16(sums_16);
            vpaddlq_u32(sums_32)
        }
    }

    const SHIFTS: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
    let input = input.as_ptr().cast::<u8>();
    let output = output.as_mut_ptr().cast::<u8>();

    // SAFETY: the loop reads all 1024 input bytes in 64-byte groups and writes all 128 output
    // bytes in eight-byte groups. Both pointers remain within their respective stack arrays.
    unsafe {
        let shifts = vld1q_s8(SHIFTS.as_ptr());
        for group in 0..16 {
            let base = input.add(group * 64);
            let masks_0 = pack_16(base, shifts);
            let masks_1 = pack_16(base.add(16), shifts);
            let masks_2 = pack_16(base.add(32), shifts);
            let masks_3 = pack_16(base.add(48), shifts);

            let masks_01 = vmovn_u32(vcombine_u32(vmovn_u64(masks_0), vmovn_u64(masks_1)));
            let masks_23 = vmovn_u32(vcombine_u32(vmovn_u64(masks_2), vmovn_u64(masks_3)));
            let masks = vmovn_u16(vcombine_u16(masks_01, masks_23));
            vst1_u8(output.add(group * 8), masks);
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "nightly-dotprod"))]
#[target_feature(enable = "dotprod")]
unsafe fn pack_bools_collect_neon_dotprod(input: &[bool; 1024], output: &mut [u64; 16]) {
    use core::arch::aarch64::{
        uint64x2_t, uint8x16_t, vcombine_u16, vcombine_u32, vdotq_u32, vdupq_n_u32, vld1q_u8,
        vmovn_u16, vmovn_u32, vmovn_u64, vpaddlq_u32, vst1_u8,
    };

    #[inline(always)]
    unsafe fn pack_16(input: *const u8, weights: uint8x16_t) -> uint64x2_t {
        // SAFETY: the caller provides a readable 16-byte region.
        unsafe {
            let bytes = vld1q_u8(input);
            vpaddlq_u32(vdotq_u32(vdupq_n_u32(0), bytes, weights))
        }
    }

    const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let input = input.as_ptr().cast::<u8>();
    let output = output.as_mut_ptr().cast::<u8>();

    // SAFETY: the loop reads all 1024 input bytes and writes all 128 output bytes in bounds.
    unsafe {
        let weights = vld1q_u8(WEIGHTS.as_ptr());
        for group in 0..16 {
            let base = input.add(group * 64);
            let masks_0 = pack_16(base, weights);
            let masks_1 = pack_16(base.add(16), weights);
            let masks_2 = pack_16(base.add(32), weights);
            let masks_3 = pack_16(base.add(48), weights);

            let masks_01 = vmovn_u32(vcombine_u32(vmovn_u64(masks_0), vmovn_u64(masks_1)));
            let masks_23 = vmovn_u32(vcombine_u32(vmovn_u64(masks_2), vmovn_u64(masks_3)));
            let masks = vmovn_u16(vcombine_u16(masks_01, masks_23));
            vst1_u8(output.add(group * 8), masks);
        }
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn unpack_and_cmp_kernel(
    input: &[u16; B],
    unpacked: &mut [u16; 1024],
    output: &mut [u64; 16],
    value: u16,
) {
    <u16 as BitPacking>::unpack::<W, B>(input, unpacked);
    for (chunk, word) in output.iter_mut().enumerate() {
        let mut packed = 0u64;
        for bit in 0..64 {
            packed |= u64::from(unpacked[chunk * 64 + bit] == value) << bit;
        }
        *word = packed;
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn cmp_byte_kernel(input: &[u16; B], output: &mut [bool; 1024], value: u16) {
    <u16 as BitPackingCompare>::unpack_cmp_byte::<W, B, _, _>(input, output, |a, b| a == b, value);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn cmp_then_pack_kernel(
    input: &[u16; B],
    byte_bools: &mut [bool; 1024],
    output: &mut [u64; 16],
    value: u16,
) {
    cmp_byte_kernel(input, byte_bools, value);
    pack_bools_collect(byte_bools, output);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn cmp_then_set_true_kernel(
    input: &[u16; B],
    byte_bools: &mut [bool; 1024],
    output: &mut [u64; 16],
    value: u16,
) {
    cmp_byte_kernel(input, byte_bools, value);
    pack_bools_set_true(byte_bools, output);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn collect_bool_kernel(input: &[bool; 1024], output: &mut [u64; 16]) {
    pack_bools_collect(input, output);
}

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn collect_bool_neon_kernel(input: &[bool; 1024], output: &mut [u64; 16]) {
    pack_bools_collect_neon(input, output);
}

#[cfg(all(target_arch = "aarch64", feature = "nightly-dotprod"))]
#[target_feature(enable = "dotprod")]
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe fn collect_bool_neon_dotprod_kernel(input: &[bool; 1024], output: &mut [u64; 16]) {
    // SAFETY: this function requires dot-product support from its caller.
    unsafe { pack_bools_collect_neon_dotprod(input, output) }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn cmp_packed_raw_kernel(input: &[u16; B], output: &mut [u64; 16], value: u16) {
    <u16 as BitPackingCompare>::unpack_cmp::<W, B, _, _>(input, output, |a, b| a == b, value);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn cmp_packed_kernel(
    input: &[u16; B],
    transposed: &mut [u64; 16],
    logical: &mut [u64; 16],
    value: u16,
) {
    cmp_packed_raw_kernel(input, transposed, value);
    untranspose_bits::<u16>(transposed, logical);
}

#[divan::bench(sample_count = 10000)]
fn unpack_and_cmp(bencher: divan::Bencher) {
    let packed = packed_input();
    let mut unpacked = [0u16; 1024];
    let mut output = [0u64; 16];
    unpack_and_cmp_kernel(&packed, &mut unpacked, &mut output, COMPARE_VALUE);
    assert_eq!(output, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            unpack_and_cmp_kernel(
                black_box(&packed),
                black_box(&mut unpacked),
                black_box(&mut output),
                black_box(COMPARE_VALUE),
            );
        }
        black_box(&output);
    });
}

#[divan::bench(sample_count = 10000)]
fn cmp_byte(bencher: divan::Bencher) {
    let packed = packed_input();
    let mut output = [false; 1024];
    cmp_byte_kernel(&packed, &mut output, COMPARE_VALUE);
    assert_eq!(output, expected_bytes());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            cmp_byte_kernel(
                black_box(&packed),
                black_box(&mut output),
                black_box(COMPARE_VALUE),
            );
        }
        black_box(&output);
    });
}

#[divan::bench(sample_count = 10000)]
fn cmp_then_pack(bencher: divan::Bencher) {
    let packed = packed_input();
    let mut byte_bools = [false; 1024];
    let mut output = [0u64; 16];
    cmp_then_pack_kernel(&packed, &mut byte_bools, &mut output, COMPARE_VALUE);
    assert_eq!(output, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            cmp_then_pack_kernel(
                black_box(&packed),
                black_box(&mut byte_bools),
                black_box(&mut output),
                black_box(COMPARE_VALUE),
            );
        }
        black_box(&output);
    });
}

#[divan::bench(sample_count = 10000)]
fn cmp_then_set_true(bencher: divan::Bencher) {
    let packed = packed_input();
    let mut byte_bools = [false; 1024];
    let mut output = [0u64; 16];
    cmp_then_set_true_kernel(&packed, &mut byte_bools, &mut output, COMPARE_VALUE);
    assert_eq!(output, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            cmp_then_set_true_kernel(
                black_box(&packed),
                black_box(&mut byte_bools),
                black_box(&mut output),
                black_box(COMPARE_VALUE),
            );
        }
        black_box(&output);
    });
}

#[divan::bench(sample_count = 10000)]
fn collect_bool(bencher: divan::Bencher) {
    let input = expected_bytes();
    let mut output = [0u64; 16];
    collect_bool_kernel(&input, &mut output);
    assert_eq!(output, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            collect_bool_kernel(black_box(&input), black_box(&mut output));
        }
        black_box(&output);
    });
}

#[cfg(target_arch = "aarch64")]
#[divan::bench(sample_count = 10000)]
fn collect_bool_neon(bencher: divan::Bencher) {
    let input = expected_bytes();
    let mut output = [0u64; 16];
    collect_bool_neon_kernel(&input, &mut output);
    assert_eq!(output, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            collect_bool_neon_kernel(black_box(&input), black_box(&mut output));
        }
        black_box(&output);
    });
}

#[cfg(all(target_arch = "aarch64", feature = "nightly-dotprod"))]
#[divan::bench(sample_count = 10000)]
fn collect_bool_neon_dotprod(bencher: divan::Bencher) {
    assert!(std::arch::is_aarch64_feature_detected!("dotprod"));
    let input = expected_bytes();
    let mut output = [0u64; 16];
    // SAFETY: dot-product support was checked above.
    unsafe { collect_bool_neon_dotprod_kernel(&input, &mut output) };
    assert_eq!(output, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            // SAFETY: dot-product support was checked before entering the timed loop.
            unsafe { collect_bool_neon_dotprod_kernel(black_box(&input), black_box(&mut output)) };
        }
        black_box(&output);
    });
}

#[divan::bench(sample_count = 10000)]
fn cmp_packed_raw(bencher: divan::Bencher) {
    let packed = packed_input();
    let mut output = [0u64; 16];
    let mut logical = [0u64; 16];
    cmp_packed_raw_kernel(&packed, &mut output, COMPARE_VALUE);
    untranspose_bits::<u16>(&output, &mut logical);
    assert_eq!(logical, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            cmp_packed_raw_kernel(
                black_box(&packed),
                black_box(&mut output),
                black_box(COMPARE_VALUE),
            );
        }
        black_box(&output);
    });
}

#[divan::bench(sample_count = 10000)]
fn cmp_packed(bencher: divan::Bencher) {
    let packed = packed_input();
    let mut transposed = [0u64; 16];
    let mut logical = [0u64; 16];
    cmp_packed_kernel(&packed, &mut transposed, &mut logical, COMPARE_VALUE);
    assert_eq!(logical, expected_packed());

    bencher.bench_local(|| {
        for _ in 0..BATCH {
            cmp_packed_kernel(
                black_box(&packed),
                black_box(&mut transposed),
                black_box(&mut logical),
                black_box(COMPARE_VALUE),
            );
        }
        black_box(&logical);
    });
}
