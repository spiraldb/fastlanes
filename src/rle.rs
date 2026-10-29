use crate::{FastLanes, transpose, unpack};

pub trait RLE: Sized {
    /// Encode an array using Run-Length Encoding
    ///
    /// Creates a dictionary of run values (`rle_vals`) and an index array
    /// (`rle_idxs`) that maps each input position to a dictionary entry.
    ///
    /// # Returns
    /// The number of run values in the dictionary
    ///
    /// # Safety
    ///
    /// - All three arguments must be valid for 1024 elements, as their types already
    ///   guarantee.
    ///
    /// Implementations perform unchecked buffer accesses that rely on these bounds; they are
    /// checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn encode_unchecked(
        input: &[Self; 1024],
        rle_vals: &mut [Self; 1024],
        rle_idxs: &mut [u16; 1024],
    ) -> usize;

    /// Decode RLE-encoded data back to original values
    ///
    /// Takes the dictionary of run values and indices, reconstructing the original array
    ///
    /// # Safety
    ///
    /// - Every element of `rle_idxs`, converted via `Into<usize>`, must be less than
    ///   `rle_vals.len()`.
    ///
    /// This is checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn decode_unchecked<I>(
        rle_vals: &[Self],
        rle_idxs: &[I; 1024],
        output: &mut [Self; 1024],
    ) where
        I: Copy + Into<usize>;

    /// Encode an array as run values plus a run-start bitmap.
    ///
    /// Bit `p` of the bitmap is set when position `p > 0` starts a new run. The bitmap is stored
    /// in the `FastLanes` `u16` layout, so it is bit-identical to
    /// `BitPack<1>(Delta(Transpose(rle_idxs), bases))` for the `u16` indices produced by
    /// [`RLE::encode_unchecked`], where each lane's base is the index preceding the lane.
    /// Those bases are prefix popcounts of the bitmap, so they need not be stored.
    ///
    /// # Returns
    /// The number of run values in the dictionary
    ///
    /// # Safety
    ///
    /// - All three arguments must be valid for their full length, as their types already
    ///   guarantee.
    unsafe fn encode_run_bitmap_unchecked(
        input: &[Self; 1024],
        rle_vals: &mut [Self; 1024],
        run_starts: &mut [u16; 64],
    ) -> usize;

    /// Decode RLE values whose `u16` indices are delta encoded and bit-packed to width `W`.
    ///
    /// This fuses `BitPacking::unpack`, `Delta::undelta` and [`RLE::decode_unchecked`] without
    /// materializing the indices: each lane is unpacked, prefix-summed from its base and used to
    /// gather directly into `output`, which is written in natural (untransposed) order.
    ///
    /// # Safety
    ///
    /// - Every reconstructed index, `bases[lane]` plus a prefix sum of the lane's deltas, must be
    ///   less than `rle_vals.len()`.
    ///
    /// This is checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn decode_delta_packed_unchecked<const W: usize, const B: usize>(
        rle_vals: &[Self],
        packed: &[u16; B],
        bases: &[u16; 64],
        output: &mut [Self; 1024],
    );

    /// Decode RLE values from a run-start bitmap produced by [`RLE::encode_run_bitmap_unchecked`].
    ///
    /// # Safety
    ///
    /// - The number of set bits in `run_starts` must be less than `rle_vals.len()`.
    ///
    /// This is checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn decode_run_bitmap_unchecked(
        rle_vals: &[Self],
        run_starts: &[u16; 64],
        output: &mut [Self; 1024],
    );
}

/// Number of `u16` lanes in a 1024-element `FastLanes` vector.
const U16_LANES: usize = <u16 as FastLanes>::LANES;

/// The `u16` lane holding the `chunk`-th run of 16 consecutive positions.
///
/// In the transposed `u16` layout, lane `l` covers positions `transpose(l)..transpose(l) + 16`.
const CHUNK_LANE: [u8; U16_LANES] = {
    let mut chunk_lane = [0u8; U16_LANES];
    let mut lane = 0;
    while lane < U16_LANES {
        chunk_lane[transpose(lane) / 16] = lane as u8;
        lane += 1;
    }
    chunk_lane
};

impl<T: PartialEq + Copy> RLE for T {
    #[inline(never)]
    unsafe fn encode_unchecked(
        input: &[Self; 1024],
        rle_vals: &mut [Self; 1024],
        rle_idxs: &mut [u16; 1024],
    ) -> usize {
        let mut pos_val = 0u16;
        let mut rle_val_idx = 0usize;

        let mut prev_val = unsafe { *input.get_unchecked(0) };
        unsafe { *rle_vals.get_unchecked_mut(rle_val_idx) = prev_val };
        rle_val_idx += 1;
        unsafe { *rle_idxs.get_unchecked_mut(0) = pos_val };

        for i in 1..1024 {
            let cur_val = unsafe { *input.get_unchecked(i) };
            if cur_val != prev_val {
                // SAFETY: `rle_val_idx` increments at most once per element, so it stays
                // below 1024.
                debug_assert!(rle_val_idx < rle_vals.len());
                unsafe { *rle_vals.get_unchecked_mut(rle_val_idx) = cur_val };
                rle_val_idx += 1;
                pos_val += 1;
                prev_val = cur_val;
            }
            unsafe { *rle_idxs.get_unchecked_mut(i) = pos_val };
        }

        rle_val_idx
    }

    #[inline(never)]
    unsafe fn decode_unchecked<I>(
        rle_vals: &[Self],
        rle_idxs: &[I; 1024],
        output: &mut [Self; 1024],
    ) where
        I: Copy + Into<usize>,
    {
        for i in 0..1024 {
            let rle_idx: usize = unsafe { *rle_idxs.get_unchecked(i) }.into();
            debug_assert!(rle_idx < rle_vals.len());

            // SAFETY: `i` is less than 1024, and the caller guarantees every index is
            // less than `rle_vals.len()`.
            unsafe { *output.get_unchecked_mut(i) = *rle_vals.get_unchecked(rle_idx) };
        }
    }

    #[inline(never)]
    unsafe fn encode_run_bitmap_unchecked(
        input: &[Self; 1024],
        rle_vals: &mut [Self; 1024],
        run_starts: &mut [u16; 64],
    ) -> usize {
        let mut prev_val = input[0];
        rle_vals[0] = prev_val;
        let mut rle_val_idx = 1;

        for (chunk, &lane) in CHUNK_LANE.iter().enumerate() {
            let mut bits = 0u16;
            for row in 0..16 {
                let cur_val = input[chunk * 16 + row];
                if cur_val != prev_val {
                    // `rle_val_idx` increments at most once per position after the first, so it
                    // stays below 1024.
                    rle_vals[rle_val_idx] = cur_val;
                    rle_val_idx += 1;
                    bits |= 1 << row;
                    prev_val = cur_val;
                }
            }
            run_starts[lane as usize] = bits;
        }

        rle_val_idx
    }

    #[inline(never)]
    unsafe fn decode_delta_packed_unchecked<const W: usize, const B: usize>(
        rle_vals: &[Self],
        packed: &[u16; B],
        bases: &[u16; 64],
        output: &mut [Self; 1024],
    ) {
        const {
            assert!(W <= 16);
            assert!(B == 1024 * W / 16);
        }

        for lane in 0..U16_LANES {
            let mut rle_idx = bases[lane];
            // Row `r` of lane `l` holds position `transpose(l) + r`.
            let mut pos = transpose(lane);
            unpack!(u16, W, packed, lane, |$idx, $delta| {
                debug_assert_eq!(transpose($idx), pos);
                rle_idx = rle_idx.wrapping_add($delta);
                debug_assert!((rle_idx as usize) < rle_vals.len());
                // SAFETY: `pos < 1024`, and the caller guarantees every reconstructed index is
                // less than `rle_vals.len()`.
                unsafe {
                    *output.get_unchecked_mut(pos) = *rle_vals.get_unchecked(rle_idx as usize);
                }
                pos += 1;
            });
        }
    }

    #[inline(never)]
    unsafe fn decode_run_bitmap_unchecked(
        rle_vals: &[Self],
        run_starts: &[u16; 64],
        output: &mut [Self; 1024],
    ) {
        debug_assert!(
            run_starts
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>()
                < rle_vals.len()
        );

        let (chunks, _) = output.as_chunks_mut::<16>();
        let mut base = 0u16;
        for (chunk_out, &lane) in chunks.iter_mut().zip(CHUNK_LANE.iter()) {
            let bits = run_starts[lane as usize];
            if bits == 0 {
                // A word without run starts continues the current run.
                // SAFETY: the caller guarantees the total popcount, which bounds `base`, is less
                // than `rle_vals.len()`.
                chunk_out.fill(unsafe { *rle_vals.get_unchecked(base as usize) });
                continue;
            }
            // Branch-free within the word so that short runs do not mispredict: the index of row
            // `r` is the base plus the run starts at or before it, a masked popcount.
            let rle_idxs: [u16; 16] = core::array::from_fn(|row| {
                base + (bits & (u16::MAX >> (15 - row))).count_ones() as u16
            });
            for (out, &rle_idx) in chunk_out.iter_mut().zip(rle_idxs.iter()) {
                // SAFETY: as above, the total popcount bounds every index.
                *out = unsafe { *rle_vals.get_unchecked(rle_idx as usize) };
            }
            base += bits.count_ones() as u16;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{BitPacking, Delta, Transpose};
    use alloc::{format, string::ToString, vec, vec::Vec};
    use hegel::TestCase;
    use hegel::generators as gs;

    fn reference_encode(input: &[u8; 1024]) -> (Vec<u8>, [u16; 1024]) {
        let mut values = vec![input[0]];
        let mut indices = [0u16; 1024];

        for index in 1..1024 {
            if input[index] != input[index - 1] {
                values.push(input[index]);
            }
            indices[index] = (values.len() - 1) as u16;
        }

        (values, indices)
    }

    #[test]
    fn test_rle_encode_unique_count() {
        let input: [u32; 1024] = core::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];

        // SAFETY: all arguments are 1024-element arrays.
        let unique_count = unsafe { u32::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

        assert_eq!(unique_count, 11);
    }

    #[test]
    fn test_rle_encode_values() {
        let input: [u32; 1024] = core::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];

        // SAFETY: all arguments are 1024-element arrays.
        let unique_count = unsafe { u32::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

        // Check that RLE values are 1, 2, 3, ..., 11
        for i in 0..unique_count {
            assert_eq!(rle_vals[i], i as u32 + 1);
        }
    }

    #[test]
    fn test_rle_encode_index_groups() {
        let input: [u32; 1024] = core::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];

        // SAFETY: all arguments are 1024-element arrays.
        unsafe { u32::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

        for i in 0..100 {
            assert_eq!(rle_idxs[i], 0);
        }

        for i in 100..200 {
            assert_eq!(rle_idxs[i], 1);
        }

        for i in 1000..1024 {
            assert_eq!(rle_idxs[i], 10);
        }
    }

    #[test]
    fn test_rle_encode_single_value() {
        let input = [42u16; 1024];
        let mut rle_vals = [0u16; 1024];
        let mut rle_idxs = [0u16; 1024];

        // SAFETY: all arguments are 1024-element arrays.
        let unique_count = unsafe { u16::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

        assert_eq!(unique_count, 1);
        assert_eq!(rle_vals[0], 42);

        for &idx in &rle_idxs {
            assert_eq!(idx, 0);
        }
    }

    #[test]
    fn test_rle_encode_all_different() {
        let input: [u8; 1024] = core::array::from_fn(|i| (i % 256) as u8);

        let mut rle_vals = [0u8; 1024];
        let mut rle_idxs = [0u16; 1024];

        // SAFETY: all arguments are 1024-element arrays.
        let unique_count = unsafe { u8::encode_unchecked(&input, &mut rle_vals, &mut rle_idxs) };

        // RLE creates a new dictionary entry every time the value changes,
        // not when we encounter a new unique value.
        assert_eq!(unique_count, 1024);
    }

    #[hegel::test]
    fn test_rle_encode_matches_reference(tc: TestCase) {
        let input: [u8; 1024] = tc.draw(gs::arrays(gs::integers::<u8>()));
        let (expected_values, expected_indices) = reference_encode(&input);
        let mut values = [u8::MAX; 1024];
        let mut indices = [u16::MAX; 1024];

        let count = unsafe { u8::encode_unchecked(&input, &mut values, &mut indices) };

        assert_eq!(count, expected_values.len());
        assert_eq!(&values[..count], expected_values);
        assert_eq!(indices, expected_indices);
    }

    #[hegel::test]
    fn test_rle_roundtrip_generated(tc: TestCase) {
        let input: [u8; 1024] = tc.draw(gs::arrays(gs::integers::<u8>()));
        let mut values = [u8::MAX; 1024];
        let mut indices = [u16::MAX; 1024];
        let count = unsafe { u8::encode_unchecked(&input, &mut values, &mut indices) };

        let mut actual = [u8::MAX; 1024];
        unsafe { u8::decode_unchecked(&values[..count], &indices, &mut actual) };

        assert_eq!(actual, input);
    }

    #[hegel::test]
    fn test_rle_decode_matches_index_model(tc: TestCase) {
        let dictionary_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(1024));
        let dictionary = tc.draw(
            gs::vecs(gs::integers::<u32>())
                .min_size(dictionary_len)
                .max_size(dictionary_len),
        );
        let indices: [u16; 1024] = tc.draw(gs::arrays(
            gs::integers::<u16>().max_value((dictionary_len - 1) as u16),
        ));
        let expected = indices.map(|index| dictionary[index as usize]);

        let mut actual = [u32::MAX; 1024];
        unsafe { u32::decode_unchecked(&dictionary, &indices, &mut actual) };

        assert_eq!(actual, expected);
    }

    /// The per-lane delta bases of transposed RLE indices: the index preceding each lane.
    fn lane_bases(indices: &[u16; 1024]) -> [u16; 64] {
        core::array::from_fn(|lane| {
            let start = transpose(lane);
            if start == 0 { 0 } else { indices[start - 1] }
        })
    }

    fn assert_run_bitmap_roundtrip(input: &[u32; 1024]) {
        let mut values = [u32::MAX; 1024];
        let mut indices = [u16::MAX; 1024];
        let count = unsafe { u32::encode_unchecked(input, &mut values, &mut indices) };

        let mut bitmap_values = [u32::MAX; 1024];
        let mut run_starts = [u16::MAX; 64];
        let bitmap_count =
            unsafe { u32::encode_run_bitmap_unchecked(input, &mut bitmap_values, &mut run_starts) };
        assert_eq!(bitmap_count, count);
        assert_eq!(bitmap_values[..count], values[..count]);

        // The bitmap is exactly the FastLanes Delta + 1-bit BitPacking of the RLE indices.
        let bases = lane_bases(&indices);
        let mut transposed = [0u16; 1024];
        Transpose::transpose(&indices, &mut transposed);
        let mut deltas = [0u16; 1024];
        Delta::delta(&transposed, &bases, &mut deltas);
        let mut packed = [0u16; 64];
        BitPacking::pack::<1, 64>(&deltas, &mut packed);
        assert_eq!(run_starts, packed);

        let mut actual = [u32::MAX; 1024];
        unsafe { u32::decode_run_bitmap_unchecked(&values[..count], &run_starts, &mut actual) };
        assert_eq!(&actual, input);

        let mut actual = [u32::MAX; 1024];
        unsafe {
            u32::decode_delta_packed_unchecked::<1, 64>(
                &values[..count],
                &run_starts,
                &bases,
                &mut actual,
            );
        }
        assert_eq!(&actual, input);
    }

    #[test]
    fn test_run_bitmap_fixed_runs() {
        for run_len in [1, 3, 16, 17, 100, 1024] {
            let input: [u32; 1024] = core::array::from_fn(|i| (i / run_len) as u32 * 7);
            assert_run_bitmap_roundtrip(&input);
        }
    }

    #[hegel::test]
    fn test_run_bitmap_roundtrip_generated(tc: TestCase) {
        // A small alphabet produces runs of varying length.
        let max_value = tc.draw(gs::integers::<u32>().max_value(4));
        let input: [u32; 1024] = tc.draw(gs::arrays(gs::integers::<u32>().max_value(max_value)));
        assert_run_bitmap_roundtrip(&input);
    }

    #[hegel::test]
    fn test_decode_delta_packed_full_width(tc: TestCase) {
        // With W = 16 the deltas are arbitrary, so the fused kernel must handle any index order.
        let dictionary_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(1024));
        let dictionary = tc.draw(
            gs::vecs(gs::integers::<u32>())
                .min_size(dictionary_len)
                .max_size(dictionary_len),
        );
        let indices: [u16; 1024] = tc.draw(gs::arrays(
            gs::integers::<u16>().max_value((dictionary_len - 1) as u16),
        ));
        let bases: [u16; 64] = tc.draw(gs::arrays(gs::integers::<u16>()));

        let mut transposed = [0u16; 1024];
        Transpose::transpose(&indices, &mut transposed);
        let mut deltas = [0u16; 1024];
        Delta::delta(&transposed, &bases, &mut deltas);
        let mut packed = [0u16; 1024];
        BitPacking::pack::<16, 1024>(&deltas, &mut packed);

        let mut actual = [u32::MAX; 1024];
        unsafe {
            u32::decode_delta_packed_unchecked::<16, 1024>(
                &dictionary,
                &packed,
                &bases,
                &mut actual,
            );
        }
        assert_eq!(actual, indices.map(|index| dictionary[index as usize]));
    }
}
