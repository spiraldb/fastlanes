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

/// Generate a `<feature>_transpose` / `<feature>_untranspose` benchmark pair, one
/// per CPU feature tier, from a single description.
///
/// Each pair runs one transpose implementation over every 1024-bit block. The
/// optional `guard = <expr>` is evaluated at runtime: when the required CPU
/// feature is absent the benchmark returns immediately. That keeps the suite
/// "just works" locally — `cargo bench` exercises whatever the host supports —
/// while in CI the walltime runner, which has every feature tier, measures them
/// all. No per-feature wiring is needed beyond this macro.
macro_rules! bench_feature {
    ($feature:ident, $transpose:expr, $untranspose:expr $(, guard = $guard:expr)?) => {
        ::paste::paste! {
            #[divan::bench]
            fn [<$feature _transpose>](bencher: Bencher) {
                $( if !$guard { return; } )?
                bench_blocks(bencher, $transpose);
            }

            #[divan::bench]
            fn [<$feature _untranspose>](bencher: Bencher) {
                $( if !$guard { return; } )?
                bench_blocks(bencher, $untranspose);
            }
        }
    };
}

// The portable scalar fallback and the dispatch entry point exist on every
// target, so they always run.
bench_feature!(
    scalar,
    fastlanes::scalar::transpose_bits,
    fastlanes::scalar::untranspose_bits
);
bench_feature!(
    dispatch,
    fastlanes::transpose_bits,
    fastlanes::untranspose_bits
);

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{bench_blocks, Bencher};
    use fastlanes::x86;

    bench_feature!(
        bmi2,
        // SAFETY: guarded by `has_bmi2`.
        |i, o| unsafe { x86::transpose_bits_bmi2(i, o) },
        |i, o| unsafe { x86::untranspose_bits_bmi2(i, o) },
        guard = x86::has_bmi2()
    );
    bench_feature!(
        vbmi,
        // SAFETY: guarded by `has_vbmi`.
        |i, o| unsafe { x86::transpose_bits_vbmi(i, o) },
        |i, o| unsafe { x86::untranspose_bits_vbmi(i, o) },
        guard = x86::has_vbmi()
    );
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{bench_blocks, Bencher};
    use fastlanes::aarch64;

    bench_feature!(
        neon,
        // SAFETY: NEON is always available on aarch64.
        |i, o| unsafe { aarch64::transpose_bits_neon(i, o) },
        |i, o| unsafe { aarch64::untranspose_bits_neon(i, o) }
    );
}
