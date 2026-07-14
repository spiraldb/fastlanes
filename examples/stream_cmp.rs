//! Streaming working-set scaling test: is unpack+cmp memory bound or compute bound?
//!
//! For each working-set size, iterate over many distinct packed blocks (instead of re-running
//! one hot block) and report ns per 1024-value block. Flat times across sizes => compute bound;
//! climbing times past cache capacities => memory bound.
//!
//! Run: RUSTFLAGS="-C target-cpu=native" cargo run --release --example stream_cmp

use fastlanes::{BitPacking, BitPackingCompare};
use std::hint::black_box;
use std::time::Instant;

const W: usize = 3;
const B: usize = 1024 * W / u16::BITS as usize; // 192 u16 = 384 bytes per block
const COMPARE_VALUE: u16 = 3;
const TOTAL_BLOCKS: usize = 1 << 21; // ~2M block-iterations per measurement

fn time_ns_per_block(blocks: usize, mut f: impl FnMut()) -> f64 {
    let passes = (TOTAL_BLOCKS / blocks).max(1);
    f(); // warm-up
    let start = Instant::now();
    for _ in 0..passes {
        f();
    }
    start.elapsed().as_nanos() as f64 / (passes * blocks) as f64
}

fn main() {
    let values: [u16; 1024] = std::array::from_fn(|i| (i as u16).wrapping_mul(17) & 7);
    let mut packed_block = [0u16; B];
    <u16 as BitPacking>::pack::<W, B>(&values, &mut packed_block);

    println!(
        "{:>10} {:>12} {:>12} | {:>10} {:>12} {:>12} {:>12}",
        "blocks", "in KiB", "out KiB", "memcpy", "cmp_byte", "cmp_packed", "unpack"
    );

    for &blocks in &[8usize, 64, 512, 2048, 16384, 131072, 524288] {
        let input: Vec<u16> = packed_block
            .iter()
            .copied()
            .cycle()
            .take(blocks * B)
            .collect();
        let mut byte_out = vec![false; blocks * 1024];
        let mut packed_out = vec![0u64; blocks * 16];
        let mut unpack_out = vec![0u16; blocks * 1024];
        let mut copy_out = vec![0u16; blocks * B];

        // Baseline: pure streaming copy of the packed input (384 B/block read + write).
        let memcpy = time_ns_per_block(blocks, || {
            copy_out.copy_from_slice(black_box(&input));
            black_box(&copy_out);
        });

        let byte = time_ns_per_block(blocks, || {
            for (inp, out) in input.chunks_exact(B).zip(byte_out.chunks_exact_mut(1024)) {
                let inp: &[u16; B] = inp.try_into().unwrap();
                let out: &mut [bool; 1024] = out.try_into().unwrap();
                <u16 as BitPackingCompare>::unpack_cmp_byte::<W, B, _, _>(
                    black_box(inp),
                    out,
                    |a, b| a == b,
                    COMPARE_VALUE,
                );
            }
            black_box(&byte_out);
        });

        let packed = time_ns_per_block(blocks, || {
            for (inp, out) in input.chunks_exact(B).zip(packed_out.chunks_exact_mut(16)) {
                let inp: &[u16; B] = inp.try_into().unwrap();
                let out: &mut [u64; 16] = out.try_into().unwrap();
                <u16 as BitPackingCompare>::unpack_cmp::<W, B, _, _>(
                    black_box(inp),
                    out,
                    |a, b| a == b,
                    COMPARE_VALUE,
                );
            }
            black_box(&packed_out);
        });

        let unpack = time_ns_per_block(blocks, || {
            for (inp, out) in input.chunks_exact(B).zip(unpack_out.chunks_exact_mut(1024)) {
                let inp: &[u16; B] = inp.try_into().unwrap();
                let out: &mut [u16; 1024] = out.try_into().unwrap();
                <u16 as BitPacking>::unpack::<W, B>(black_box(inp), out);
            }
            black_box(&unpack_out);
        });

        println!(
            "{:>10} {:>12.0} {:>12.0} | {:>8.1}ns {:>10.1}ns {:>10.1}ns {:>10.1}ns",
            blocks,
            (blocks * B * 2) as f64 / 1024.0,
            (blocks * 1024) as f64 / 1024.0,
            memcpy,
            byte,
            packed,
            unpack,
        );
    }
}
