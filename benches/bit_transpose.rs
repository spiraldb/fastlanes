use std::hint::black_box;

use arrayref::array_mut_ref;
use arrayref::array_ref;
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

/// Drive an unsafe, runtime-gated implementation: skip the benchmark when the
/// CPU feature is unavailable, otherwise hand the call to [`bench_blocks`]. This
/// is the only thing the per-tier benchmarks don't share, so it's all the macro
/// hides — the `#[divan::bench]` functions themselves stay spelled out below.
macro_rules! gated_bench {
    ($bencher:expr, $supported:expr, $op:path) => {
        if $supported {
            // SAFETY: guarded by the `$supported` check above.
            bench_blocks($bencher, |i, o| unsafe { $op(i, o) });
        }
    };
}

#[divan::bench]
fn scalar_transpose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::scalar::transpose_bits);
}

#[divan::bench]
fn scalar_untranspose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::scalar::untranspose_bits);
}

#[divan::bench]
fn dispatch_transpose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::transpose_bits);
}

#[divan::bench]
fn dispatch_untranspose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::untranspose_bits);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{bench_blocks, Bencher};
    use fastlanes::x86;

    #[divan::bench]
    fn bmi2_transpose(bencher: Bencher) {
        gated_bench!(bencher, x86::has_bmi2(), x86::transpose_bits_bmi2);
    }

    #[divan::bench]
    fn bmi2_untranspose(bencher: Bencher) {
        gated_bench!(bencher, x86::has_bmi2(), x86::untranspose_bits_bmi2);
    }

    #[divan::bench]
    fn vbmi_transpose(bencher: Bencher) {
        gated_bench!(bencher, x86::has_vbmi(), x86::transpose_bits_vbmi);
    }

    #[divan::bench]
    fn vbmi_untranspose(bencher: Bencher) {
        gated_bench!(bencher, x86::has_vbmi(), x86::untranspose_bits_vbmi);
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{bench_blocks, Bencher};
    use fastlanes::aarch64;

    #[divan::bench]
    fn neon_transpose(bencher: Bencher) {
        // SAFETY: NEON is always available on aarch64.
        bench_blocks(bencher, |i, o| unsafe {
            aarch64::transpose_bits_neon(i, o)
        });
    }

    #[divan::bench]
    fn neon_untranspose(bencher: Bencher) {
        // SAFETY: NEON is always available on aarch64.
        bench_blocks(bencher, |i, o| unsafe {
            aarch64::untranspose_bits_neon(i, o)
        });
    }
}
