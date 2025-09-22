use divan::Bencher;
use std::hint::black_box;

use fastlanes::Transpose;

fn main() {
    divan::main();
}

#[divan::bench]
fn transpose_u16(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let mut values: [u16; 1024] = [0; 1024];
            for i in 0..1024 {
                values[i] = (i % u16::MAX as usize) as u16;
            }
            let transposed = [0; 1024];
            (values, transposed)
        })
        .bench_local_refs(|(values, transposed)| {
            Transpose::transpose(black_box(values), transposed);
            black_box(*transposed);
        });
}
