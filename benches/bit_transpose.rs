use std::hint::black_box;

use arrayref::array_mut_ref;
use arrayref::array_ref;
use divan::Bencher;
use divan::counter::BytesCount;
use fastlanes::FastLanes;

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

/// Run `op` over every 1024-bit block of the buffer.
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

/// Transpose is generic over the element width `T`; benchmark each width separately. The mask
/// always factors into 16 groups of 8 bytes regardless of `T`, so per-arch the widths should be
/// within noise of one another (only the gather/scatter index tables differ).
#[divan::bench(types = [u8, u16, u32, u64])]
fn scalar_transpose<T: FastLanes>(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::scalar::transpose_bits::<T>);
}

#[divan::bench]
fn scalar_untranspose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::scalar::untranspose_bits);
}

#[divan::bench(types = [u8, u16, u32, u64])]
fn dispatch_transpose<T: FastLanes>(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::transpose_bits::<T>);
}

#[divan::bench]
fn dispatch_untranspose(bencher: Bencher) {
    bench_blocks(bencher, fastlanes::untranspose_bits);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Bencher, bench_blocks};
    use fastlanes::FastLanes;
    use fastlanes::x86;

    #[divan::bench(types = [u8, u16, u32, u64])]
    fn bmi2_transpose<T: FastLanes>(bencher: Bencher) {
        if !x86::has_bmi2() {
            return;
        }
        // SAFETY: guarded by `has_bmi2`.
        bench_blocks(bencher, |i, o| unsafe {
            x86::transpose_bits_bmi2::<T>(i, o)
        });
    }

    #[divan::bench]
    fn bmi2_untranspose(bencher: Bencher) {
        if !x86::has_bmi2() {
            return;
        }
        // SAFETY: guarded by `has_bmi2`.
        bench_blocks(bencher, |i, o| unsafe {
            x86::untranspose_bits_bmi2(i, o);
        });
    }

    #[divan::bench(types = [u8, u16, u32, u64])]
    fn vbmi_transpose<T: FastLanes>(bencher: Bencher) {
        if !x86::has_vbmi() {
            return;
        }
        // SAFETY: guarded by `has_vbmi`.
        bench_blocks(bencher, |i, o| unsafe {
            x86::transpose_bits_vbmi::<T>(i, o)
        });
    }

    #[divan::bench]
    fn vbmi_untranspose(bencher: Bencher) {
        if !x86::has_vbmi() {
            return;
        }
        // SAFETY: guarded by `has_vbmi`.
        bench_blocks(bencher, |i, o| unsafe {
            x86::untranspose_bits_vbmi(i, o);
        });
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{Bencher, bench_blocks};
    use fastlanes::FastLanes;
    use fastlanes::aarch64;

    #[divan::bench(types = [u8, u16, u32, u64])]
    fn neon_transpose<T: FastLanes>(bencher: Bencher) {
        // SAFETY: NEON is always available on aarch64.
        bench_blocks(bencher, |i, o| unsafe {
            aarch64::transpose_bits_neon::<T>(i, o);
        });
    }

    #[divan::bench]
    fn neon_untranspose(bencher: Bencher) {
        // SAFETY: NEON is always available on aarch64.
        bench_blocks(bencher, |i, o| unsafe {
            aarch64::untranspose_bits_neon(i, o);
        });
    }
}
