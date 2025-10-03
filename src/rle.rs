pub trait RLE: Sized {
    /// Encode an array using Run-Length Encoding
    ///
    /// Creates a dictionary of run values (`rle_vals`) and an index array
    /// (`rle_idxs`) that maps each input position to a dictionary entry.
    ///
    /// # Returns
    /// The number of run values in the dictionary
    fn encode(
        input: &[Self; 1024],
        rle_vals: &mut [Self; 1024],
        rle_idxs: &mut [u16; 1024],
    ) -> usize;

    /// Decode RLE-encoded data back to original values
    ///
    /// Takes the dictionary of run values and indices, reconstructing the original array
    fn decode<I>(rle_vals: &[Self], rle_idxs: &[I; 1024], output: &mut [Self; 1024])
    where
        I: Copy + Into<usize>;
}

impl<T: PartialEq + Copy> RLE for T {
    #[inline(never)]
    fn encode(
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
    fn decode<I>(rle_vals: &[Self], rle_idxs: &[I; 1024], output: &mut [Self; 1024])
    where
        I: Copy + Into<usize>,
    {
        for i in 0..1024 {
            let idx = unsafe { *rle_idxs.get_unchecked(i) }.into();
            unsafe { *output.get_unchecked_mut(i) = *rle_vals.get_unchecked(idx) };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rle_encode_unique_count() {
        let input: [u32; 1024] = core::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];

        let unique_count = u32::encode(&input, &mut rle_vals, &mut rle_idxs);

        assert_eq!(unique_count, 11);
    }

    #[test]
    fn test_rle_encode_values() {
        let input: [u32; 1024] = core::array::from_fn(|i| (i / 100 + 1) as u32);
        let mut rle_vals = [0u32; 1024];
        let mut rle_idxs = [0u16; 1024];

        let unique_count = u32::encode(&input, &mut rle_vals, &mut rle_idxs);

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

        u32::encode(&input, &mut rle_vals, &mut rle_idxs);

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

        let unique_count = u16::encode(&input, &mut rle_vals, &mut rle_idxs);

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

        let unique_count = u8::encode(&input, &mut rle_vals, &mut rle_idxs);

        // RLE creates a new dictionary entry every time the value changes,
        // not when we encounter a new unique value.
        assert_eq!(unique_count, 1024);
    }

    #[test]
    fn test_rle_round_trip() {
        let input: [u8; 1024] = core::array::from_fn(|i| (i % 256) as u8);

        let mut rle_vals = [0u8; 1024];
        let mut rle_idxs = [0u16; 1024];
        let unique_count = u8::encode(&input, &mut rle_vals, &mut rle_idxs);

        let mut decoded = [0u8; 1024];
        u8::decode(&rle_vals[..unique_count], &rle_idxs, &mut decoded);
        assert_eq!(input, decoded);
    }
}
