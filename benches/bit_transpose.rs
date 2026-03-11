// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use fastlanes::bit_transpose::{transpose_bits, untranspose_bits_scalar};

fn main() {
    divan::main();
}

/// Generate deterministic test data.
fn generate_test_data(seed: u8) -> [u8; 128] {
    let mut data = [0u8; 128];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(17).wrapping_add(i as u8).wrapping_mul(31);
    }
    data
}

const BATCH_SIZE: usize = 1000;

// ============================================================================
// Transpose: single array
// ============================================================================

#[divan::bench]
fn transpose_scalar(bencher: Bencher) {
    let input = generate_test_data(42);

    bencher
        .with_inputs(|| (&input, [0u8; 128]))
        .bench_local_values(|(input, mut output)| {
            transpose_bits(&input, &mut output);
            output
        });
}

// ============================================================================
// Transpose: throughput (1000 arrays)
// ============================================================================

#[divan::bench]
fn transpose_scalar_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

    bencher
        .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
        .bench_local_values(|(inputs, mut outputs)| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                transpose_bits(input, output);
            }
            outputs
        });
}

// ============================================================================
// Untranspose: single array
// ============================================================================

#[divan::bench]
fn untranspose_scalar(bencher: Bencher) {
    let input = generate_test_data(42);

    bencher
        .with_inputs(|| (&input, [0u8; 128]))
        .bench_local_values(|(input, mut output)| {
            untranspose_bits_scalar(&input, &mut output);
            output
        });
}

// ============================================================================
// Untranspose: throughput (1000 arrays)
// ============================================================================

#[divan::bench]
fn untranspose_scalar_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

    bencher
        .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
        .bench_local_values(|(inputs, mut outputs)| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                untranspose_bits_scalar(input, output);
            }
            outputs
        });
}

// ============================================================================
// x86_64 benchmarks
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::{generate_test_data, BATCH_SIZE};
    use divan::Bencher;
    use fastlanes::bit_transpose::x86::{
        has_bmi2, has_vbmi, transpose_bits_bmi2, transpose_bits_vbmi, untranspose_bits_bmi2,
        untranspose_bits_vbmi,
    };

    // --- Transpose: single array ---

    #[divan::bench]
    fn transpose_bmi2(bencher: Bencher) {
        if !has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_local_values(|(input, mut output)| {
                unsafe { transpose_bits_bmi2(&input, &mut output) };
                output
            });
    }

    #[divan::bench]
    fn transpose_vbmi(bencher: Bencher) {
        if !has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_local_values(|(input, mut output)| {
                unsafe { transpose_bits_vbmi(&input, &mut output) };
                output
            });
    }

    // --- Untranspose: single array ---

    #[divan::bench]
    fn untranspose_bmi2(bencher: Bencher) {
        if !has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_local_values(|(input, mut output)| {
                unsafe { untranspose_bits_bmi2(&input, &mut output) };
                output
            });
    }

    #[divan::bench]
    fn untranspose_vbmi(bencher: Bencher) {
        if !has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_local_values(|(input, mut output)| {
                unsafe { untranspose_bits_vbmi(&input, &mut output) };
                output
            });
    }

    // --- Transpose: throughput (1000 arrays) ---

    #[divan::bench]
    fn transpose_bmi2_throughput(bencher: Bencher) {
        if !has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_local_values(|(inputs, mut outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { transpose_bits_bmi2(input, output) };
                }
                outputs
            });
    }

    #[divan::bench]
    fn transpose_vbmi_throughput(bencher: Bencher) {
        if !has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_local_values(|(inputs, mut outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { transpose_bits_vbmi(input, output) };
                }
                outputs
            });
    }

    // --- Untranspose: throughput (1000 arrays) ---

    #[divan::bench]
    fn untranspose_bmi2_throughput(bencher: Bencher) {
        if !has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_local_values(|(inputs, mut outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { untranspose_bits_bmi2(input, output) };
                }
                outputs
            });
    }

    #[divan::bench]
    fn untranspose_vbmi_throughput(bencher: Bencher) {
        if !has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_local_values(|(inputs, mut outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { untranspose_bits_vbmi(input, output) };
                }
                outputs
            });
    }
}

// ============================================================================
// aarch64 benchmarks
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{generate_test_data, Bencher, BATCH_SIZE};
    use fastlanes::bit_transpose::aarch64::{transpose_bits_neon, untranspose_bits_neon};

    // --- Transpose: single array ---

    #[divan::bench]
    fn transpose_neon(bencher: Bencher) {
        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_local_values(|(input, mut output)| {
                unsafe { transpose_bits_neon(&input, &mut output) };
                output
            });
    }

    // --- Untranspose: single array ---

    #[divan::bench]
    fn untranspose_neon(bencher: Bencher) {
        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_local_values(|(input, mut output)| {
                unsafe { untranspose_bits_neon(&input, &mut output) };
                output
            });
    }

    // --- Transpose: throughput (1000 arrays) ---

    #[divan::bench]
    fn transpose_neon_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_local_values(|(inputs, mut outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { transpose_bits_neon(input, output) };
                }
                outputs
            });
    }

    // --- Untranspose: throughput (1000 arrays) ---

    #[divan::bench]
    fn untranspose_neon_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_local_values(|(inputs, mut outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { untranspose_bits_neon(input, output) };
                }
                outputs
            });
    }
}
