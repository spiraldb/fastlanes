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
}

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
        for (idx, output) in rle_idxs.iter().zip(output.iter_mut()) {
            debug_assert!((*idx).into() < rle_vals.len());
            // SAFETY: the caller guarantees every index is less than `rle_vals.len()`.
            *output = unsafe { *rle_vals.get_unchecked((*idx).into()) };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
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
}
