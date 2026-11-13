use arrayref::array_ref;
use divan::Bencher;
use fastlanes::RLE;
use std::hint::black_box;

mod shared;

fn main() {
    divan::main();
}

#[divan::bench]
fn rle_encode_u32(bencher: Bencher) {
    let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
    let mut rle_vals = [0u32; 1024];
    let mut rle_idxs = [0u16; 1024];

    bencher.bench_local(|| {
        // SAFETY: all arguments are 1024-element arrays.
        let unique_count =
            unsafe { u32::encode_unchecked(black_box(&input), &mut rle_vals, &mut rle_idxs) };
        black_box(unique_count);
    });
}

#[divan::bench]
fn rle_decode_u32(bencher: Bencher) {
    let input: [u32; 1024] = std::array::from_fn(|i| (i / 100 + 1) as u32);
    let mut rle_vals = [0u32; 1024];
    let mut rle_idxs = [0u16; 1024];
    // SAFETY: all arguments are 1024-element arrays.
    let unique_count = unsafe { u32::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

    let mut output = [0u32; 1024];

    bencher.bench_local(|| {
        // SAFETY: `encode_unchecked` only writes indices below the returned `unique_count`.
        unsafe {
            u32::decode_unchecked(
                black_box(&rle_vals[..unique_count]),
                black_box(&rle_idxs),
                &mut output,
            );
        }
        black_box(&output);
    });
}

#[divan::bench]
fn rle_throughput_encode_32(bencher: Bencher) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();

    with_counter!(bencher, input_data.len() * std::mem::size_of::<u32>()).bench_local(|| {
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];
        for batch in 0..NUM_BATCHES {
            let batch_start = batch * 1024;
            let input_batch = array_ref![input_data, batch_start, 1024];

            // SAFETY: all arguments are 1024-element arrays.
            let unique_count = unsafe {
                u32::encode_unchecked(black_box(input_batch), &mut rle_vals, &mut rle_idxs)
            };
            black_box(unique_count);
        }
    });
}

#[divan::bench]
fn rle_throughput_decode_32(bencher: Bencher) {
    const NUM_BATCHES: usize = 1024;
    const N: usize = 1024 * NUM_BATCHES;

    let input_data: Vec<u32> = (0..N).map(|i| (i / 100) as u32).collect();

    // Pre-encode all batches
    let mut encoded_batches = Vec::new();
    for batch in 0..NUM_BATCHES {
        let batch_start = batch * 1024;
        let input_batch: [u32; 1024] = std::array::from_fn(|i| input_data[batch_start + i]);

        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];
        // SAFETY: all arguments are 1024-element arrays.
        let unique_count =
            unsafe { u32::encode_unchecked(&input_batch, &mut rle_vals, &mut rle_idxs) };
        encoded_batches.push((rle_vals, rle_idxs, unique_count));
    }

    with_counter!(bencher, input_data.len() * std::mem::size_of::<u32>()).bench_local(|| {
        let mut output = [0u32; 1024];
        for (rle_vals, rle_idxs, unique_count) in &encoded_batches {
            // SAFETY: `encode_unchecked` only writes indices below the returned `unique_count`.
            unsafe {
                u32::decode_unchecked(
                    black_box(&rle_vals[..*unique_count]),
                    black_box(rle_idxs),
                    &mut output,
                );
            }
            black_box(&output);
        }
    });
}

/// 64 chunks of values in irregular runs of 1 to 15 (8 on average), RLE-encoded with the
/// indices narrowed to `u8` and, for Delta decoding, delta-encoded in transposed order with 1-bit
/// packed deltas.
struct IrregularRuns {
    chunks: Vec<IrregularChunk>,
}

struct IrregularChunk {
    vals: Vec<u32>,
    idxs: [u8; 1024],
    bases: [u8; 128],
    deltas: [u8; 128],
}

impl IrregularRuns {
    const CHUNKS: usize = 64;

    fn new() -> Self {
        use fastlanes::{BitPacking, Delta, Transpose};

        // A fixed-seed xorshift, so the runs are the same in every run of the benchmark.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut input = Vec::with_capacity(Self::CHUNKS * 1024);
        while input.len() < Self::CHUNKS * 1024 {
            let run = (next() % 15 + 1) as usize;
            // Below 100, so the values fit every benchmarked value type.
            let value = (next() % 100) as u32;
            input.extend(std::iter::repeat_n(value, run));
        }
        input.truncate(Self::CHUNKS * 1024);

        let chunks = input
            .as_chunks::<1024>()
            .0
            .iter()
            .map(|chunk| {
                let mut vals = [0u32; 1024];
                let mut idxs16 = [0u16; 1024];
                // SAFETY: all arguments are 1024-element arrays.
                let count = unsafe { u32::encode_unchecked(chunk, &mut vals, &mut idxs16) };
                let idxs = idxs16.map(|i| u8::try_from(i).expect("at most 256 runs per chunk"));

                let mut transposed = [0u8; 1024];
                u8::transpose(&idxs, &mut transposed);
                let bases: [u8; 128] = std::array::from_fn(|lane| transposed[lane]);
                let mut deltas = [0u8; 1024];
                u8::delta::<128>(&transposed, &bases, &mut deltas);
                let mut packed = [0u8; 128];
                // SAFETY: RLE index deltas are 0 or 1, and a 1-bit chunk packs into 128 bytes.
                unsafe { u8::unchecked_pack(1, &deltas, &mut packed) };

                IrregularChunk {
                    vals: vals[..count].to_vec(),
                    idxs,
                    bases,
                    deltas: packed,
                }
            })
            .collect();
        Self { chunks }
    }
}

#[divan::bench(types = [u8, u16, u32])]
fn rle_decode_irregular_runs<V>(bencher: Bencher)
where
    V: Copy + Default + PartialEq + TryFrom<u32>,
{
    let data = IrregularRuns::new();
    let vals: Vec<Vec<V>> = data
        .chunks
        .iter()
        .map(|chunk| {
            chunk
                .vals
                .iter()
                .map(|&v| V::try_from(v).ok().expect("values are below 100"))
                .collect()
        })
        .collect();
    with_counter!(
        bencher,
        IrregularRuns::CHUNKS * 1024 * std::mem::size_of::<V>()
    )
    .with_inputs(|| vec![[V::default(); 1024]; IrregularRuns::CHUNKS])
    .bench_local_refs(|output| {
        for ((chunk, vals), out) in data.chunks.iter().zip(&vals).zip(output.iter_mut()) {
            // SAFETY: every index is below the chunk's run count.
            unsafe { V::decode_unchecked(vals, &chunk.idxs, out) };
        }
    });
}

#[divan::bench]
fn rle_decode_delta_u32_irregular_runs(bencher: Bencher) {
    let data = IrregularRuns::new();
    with_counter!(
        bencher,
        IrregularRuns::CHUNKS * 1024 * std::mem::size_of::<u32>()
    )
    .with_inputs(|| vec![[0u32; 1024]; IrregularRuns::CHUNKS])
    .bench_local_refs(|output| {
        for (chunk, out) in data.chunks.iter().zip(output.iter_mut()) {
            // SAFETY: every lane's last index is an index of the chunk, below its run count.
            unsafe {
                u32::decode_delta_unchecked(&chunk.vals, &chunk.bases, Some(&chunk.deltas), out);
            }
        }
    });
}
