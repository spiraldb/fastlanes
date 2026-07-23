use divan::Bencher;
use std::hint::black_box;

use fastlanes::Transpose;

mod shared;

use shared::Aligned;

fn main() {
    divan::main();
}

#[divan::bench]
fn transpose_u16(bencher: Bencher) {
    let mut values = Aligned([0u16; 1024]);
    for i in 0..1024 {
        values.0[i] = (i % u16::MAX as usize) as u16;
    }

    let mut transposed = Aligned([0; 1024]);

    bencher.bench_local(|| {
        Transpose::transpose(black_box(&values.0), &mut transposed.0);
        black_box(&transposed.0);
    });
}
