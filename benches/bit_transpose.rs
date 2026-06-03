use std::hint::black_box;

use arrayref::array_mut_ref;
use arrayref::array_ref;
use bench_macros::bench;
use divan::counter::BytesCount;
use divan::Bencher;

fn main() {
    divan::main();
}

/// Number of 1024-bit blocks processed per benchmark iteration.
/// 1024 blocks × 128 bytes = 128 KiB of input.
const BLOCKS: usize = 1024;
const U64S: usize = BLOCKS * 16;

fn make_input() -> Vec<u64> {
    (0..U64S as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect()
}

/// Shared driver: run `op` over every 1024-bit block of the buffer.
fn bench_blocks(bencher: Bencher, op: impl Fn(&[u64; 16], &mut [u64; 16])) {
    let input = make_input();
    let mut output = vec![0u64; U64S];
    bencher.counter(BytesCount::new(U64S * 8)).bench_local(|| {
        for blk in 0..BLOCKS {
            op(
                array_ref!(input, blk * 16, 16),
                array_mut_ref!(output, blk * 16, 16),
            );
        }
        black_box(&output);
    });
}

// `#[bench(<tier>)]` (from the `bench-macros` dev-dependency) gates each benchmark
// on its Intel feature tier, so it compiles — and therefore runs — only in the CI
// matrix entry built with that `-C target-feature` flag. The tiers are mutually
// exclusive, so every benchmark has exactly one home runner:
//   * baseline (scalar / dispatch) → the `walltime-baseline` runner
//   * bmi2                         → the `walltime-bmi2` runner
//   * avx512 vbmi                  → the `walltime-avx512` runner
// Locally, `cargo bench` builds the baseline tier; pass the matching
// `RUSTFLAGS="-C target-feature=+…"` (or `-C target-cpu=native`) to run a SIMD tier.

#[bench(baseline)]
#[divan::bench]
fn scalar_transpose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::scalar::transpose_bits);
}

/// Untranspose is generic over the element width `T`; benchmark each width separately. The mask
/// always factors into 16 groups of 8 bytes regardless of `T`, so per-arch the widths should be
/// within noise of one another (only the gather/scatter index tables differ).
#[bench(baseline)]
#[divan::bench(types = [u8, u16, u32, u64])]
fn scalar_untranspose<T: fastlanes::FastLanes>(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::scalar::untranspose_bits::<T>);
}

#[bench(baseline)]
#[divan::bench]
fn dispatch_transpose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::transpose_bits);
}

#[bench(baseline)]
#[divan::bench(types = [u8, u16, u32, u64])]
fn dispatch_untranspose<T: fastlanes::FastLanes>(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::untranspose_bits::<T>);
}

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "bmi2", target_feature = "avx512vbmi")
))]
mod x86 {
    use bench_macros::bench;
    use fastlanes::x86;

    use super::{bench_blocks, Bencher};

    #[bench(bmi2)]
    #[divan::bench]
    fn bmi2_transpose(bencher: Bencher) {
        // SAFETY: this benchmark only compiles with `+bmi2`.
        bench_blocks(bencher, |i, o| unsafe { x86::transpose_bits_bmi2(i, o) });
    }

    #[bench(bmi2)]
    #[divan::bench]
    fn bmi2_untranspose(bencher: Bencher) {
        // SAFETY: this benchmark only compiles with `+bmi2`.
        bench_blocks(bencher, |i, o| unsafe { x86::untranspose_bits_bmi2(i, o) });
    }

    #[bench(avx512)]
    #[divan::bench]
    fn vbmi_transpose(bencher: Bencher) {
        // SAFETY: this benchmark only compiles with `+avx512vbmi`.
        bench_blocks(bencher, |i, o| unsafe { x86::transpose_bits_vbmi(i, o) });
    }

    #[bench(avx512)]
    #[divan::bench]
    fn vbmi_untranspose(bencher: Bencher) {
        // SAFETY: this benchmark only compiles with `+avx512vbmi`.
        bench_blocks(bencher, |i, o| unsafe { x86::untranspose_bits_vbmi(i, o) });
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use fastlanes::aarch64;

    use super::{bench_blocks, Bencher};

    #[divan::bench]
    fn neon_transpose(bencher: Bencher) {
        // SAFETY: NEON is always available on aarch64.
        bench_blocks(bencher, |i, o| unsafe {
            aarch64::transpose_bits_neon(i, o)
        });
    }

    #[divan::bench(types = [u8, u16, u32, u64])]
    fn neon_untranspose<T: fastlanes::FastLanes>(bencher: Bencher) {
        // SAFETY: NEON is always available on aarch64.
        bench_blocks(bencher, |i, o| unsafe {
            aarch64::untranspose_bits_neon::<T>(i, o)
        });
    }
}
