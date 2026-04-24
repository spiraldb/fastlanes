use divan::Bencher;
use std::hint::black_box;

use fastlanes::Transpose;

fn main() {
    divan::main();
}

#[divan::bench]
fn transpose_u16(bencher: Bencher) {
    let mut values: [u16; 1024] = [0; 1024];
    for i in 0..1024 {
        values[i] = (i % u16::MAX as usize) as u16;
    }

    let mut transposed = [0; 1024];

    bencher.bench_local(|| {
        Transpose::transpose(black_box(&values), &mut transposed);
        black_box(&transposed);
    });
}
