//! Run-length encoding of 1024-element chunks into run values and per-element indices.
//!
//! # Decoding
//!
//! Consecutive RLE indices differ by 0 or 1, so `W` consecutive outputs read from a window of at
//! most `W` values starting at the first index: one shuffle decodes them.
//! [`RLE::decode_unchecked`] does this for materialized indices. [`RLE::decode_delta_unchecked`]
//! does it for `u8` indices stored as [`Delta`](crate::Delta) over 1-bit deltas, fusing
//! bit-unpacking, undelta and the gather: each of the 128 lanes covers 8 consecutive outputs, and
//! the prefix sums of its delta byte are its indices relative to its base.
//!
//! | Target     | 1-byte values     | 2-byte values      | 4-byte values          | 8-byte values |
//! | ---------- | ----------------- | ------------------ | ---------------------- | ------------- |
//! | x86 AVX512 | `vpermb`, 64 wide | `vpermw`, 32 wide  | `vpermd`, 16 wide      | scalar        |
//! | x86 AVX2   | `pshufb`, 8 wide  | `pshufb`, 8 wide   | `vpermd`, 8 wide       | scalar        |
//! | aarch64    | `tbl`, 8 wide     | `tbl`, 8 wide      | `tbl` over 2 registers | scalar        |
//! | otherwise  | scalar            | scalar             | scalar                 | scalar        |
//!
//! Kernel selection follows [`Transpose`](crate::Transpose): runtime detection with the `runtime`
//! feature, `target_feature`s otherwise, and no AVX-512 for 4-byte values on Zen 4. The
//! `rle_decode_*` and `rle_decode_delta_*` kernels are also public.
//!
//! The scalar gather handles 8-byte values (no vector kernel measured faster), index types other
//! than `u8` and `u16`, and steps whose indices do not fit one window, which RLE indices never
//! produce. The AVX2 and NEON kernels also use it for groups whose window would pass the end of
//! the values; the AVX-512 kernels mask the load instead.

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "x86_64")]
mod x86;

use crate::FL_ORDER;
#[cfg(target_arch = "x86_64")]
use crate::transpose::{avx512_is_slower, has_avx2, has_avx512};

/// Number of 8-output groups in a chunk.
const GROUPS: usize = 1024 / 8;

/// Number of `u8` `FastLanes` lanes in a chunk.
const U8_LANES: usize = 128;

/// `DELTA_LUT[d][r]`: the number of set bits among bits `1..=r` of delta byte `d`.
///
/// A lane with base `b` reads index `b + (d & 1) + DELTA_LUT[d][r]` for row `r`. Entries are
/// below 8, so a lane fits one window.
static DELTA_LUT: [[u8; 8]; 256] = {
    let mut lut = [[0u8; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut sum = 0;
        let mut row = 0;
        while row < 8 {
            sum += (byte >> row) & 1;
            lut[byte][row] = (sum - (byte & 1)) as u8;
            row += 1;
        }
        byte += 1;
    }
    lut
};

/// The first of the 8 consecutive outputs of `u8` lane `lane`: transposed element
/// `row * 128 + lane` comes from position `(lane % 16) * 64 + FL_ORDER[lane / 16] * 8 + row`.
#[inline]
const fn u8_lane_start(lane: usize) -> usize {
    (lane % 16) * 64 + FL_ORDER[lane / 16] * 8
}

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
        I: Copy + Into<usize> + 'static;

    /// Decode RLE-encoded data whose `u8` indices are stored as [`Delta`](crate::Delta) over
    /// 1-bit packed deltas, without materializing the indices.
    ///
    /// `bases` is the first transposed row of the indices, one base per lane. `deltas` is the
    /// deltas packed at 1 bit, one byte per lane, or `None` when every delta is zero. RLE index
    /// deltas are 0 or 1, so `u8` RLE indices always pack this way.
    ///
    /// # Safety
    ///
    /// For every lane, `bases[lane]` plus the set bits of `deltas[lane]`, the lane's last index,
    /// must be less than `rle_vals.len()`.
    ///
    /// This is checked only with `debug_assert` (i.e., not checked on release builds).
    #[inline]
    unsafe fn decode_delta_unchecked(
        rle_vals: &[Self],
        bases: &[u8; 128],
        deltas: Option<&[u8; 128]>,
        output: &mut [Self; 1024],
    ) where
        Self: Copy,
    {
        // SAFETY: the caller bounds every lane's last index by `rle_vals.len()`.
        unsafe { decode_delta(rle_vals, bases, deltas, output) }
    }
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
        I: Copy + Into<usize> + 'static,
    {
        // SAFETY: the caller bounds every index by `rle_vals.len()`.
        unsafe { decode(rle_vals, rle_idxs, output) }
    }
}

/// Write `output[i] = rle_vals[rle_idxs[i]]` with the best kernel for this CPU.
///
/// # Safety
///
/// Every index must be less than `rle_vals.len()`.
#[inline]
unsafe fn decode<V: Copy, I: Copy + Into<usize> + 'static>(
    rle_vals: &[V],
    rle_idxs: &[I; 1024],
    output: &mut [V; 1024],
) {
    // SAFETY: each kernel runs only where its features are detected, and the caller bounds
    // every index by `rle_vals.len()`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx512() && !avx512_is_slower(size_of::<V>()) {
                return rle_decode_avx512(rle_vals, rle_idxs, output);
            }
            if has_avx2() {
                return rle_decode_avx2(rle_vals, rle_idxs, output);
            }
        }
        #[cfg(target_arch = "aarch64")]
        return rle_decode_neon(rle_vals, rle_idxs, output);
        #[allow(unreachable_code, reason = "aarch64 always returns above")]
        rle_decode_scalar(rle_vals, rle_idxs, output);
    }
}

/// See [`RLE::decode_delta_unchecked`].
///
/// # Safety
///
/// Every lane's last index must be less than `rle_vals.len()`.
#[inline]
unsafe fn decode_delta<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; U8_LANES],
    deltas: Option<&[u8; U8_LANES]>,
    output: &mut [V; 1024],
) {
    debug_assert!((0..U8_LANES).all(|lane| {
        usize::from(bases[lane]) + deltas.map_or(0, |d| d[lane].count_ones() as usize)
            < rle_vals.len()
    }));
    // SAFETY: each kernel runs only where its features are detected, and the caller bounds
    // every lane's last index by `rle_vals.len()`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx512() && !avx512_is_slower(size_of::<V>()) {
                return rle_decode_delta_avx512(rle_vals, bases, deltas, output);
            }
            if has_avx2() {
                return rle_decode_delta_avx2(rle_vals, bases, deltas, output);
            }
        }
        #[cfg(target_arch = "aarch64")]
        return rle_decode_delta_neon(rle_vals, bases, deltas, output);
        #[allow(unreachable_code, reason = "aarch64 always returns above")]
        rle_decode_delta_scalar(rle_vals, bases, deltas, output);
    }
}

/// [`RLE::decode_unchecked`] with the AVX-512 kernel.
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VBMI, and every index must be less than
/// `rle_vals.len()`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn rle_decode_avx512<V: Copy, I: Copy + Into<usize> + 'static>(
    rle_vals: &[V],
    rle_idxs: &[I; 1024],
    output: &mut [V; 1024],
) {
    // SAFETY: the caller guarantees the features and bounds every index.
    unsafe {
        if !avx512::decode(rle_vals, rle_idxs, output) {
            rle_decode_scalar(rle_vals, rle_idxs, output);
        }
    }
}

/// [`RLE::decode_unchecked`] with the AVX2 kernel.
///
/// # Safety
///
/// The CPU must support AVX2, and every index must be less than `rle_vals.len()`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn rle_decode_avx2<V: Copy, I: Copy + Into<usize> + 'static>(
    rle_vals: &[V],
    rle_idxs: &[I; 1024],
    output: &mut [V; 1024],
) {
    // SAFETY: the caller guarantees the features and bounds every index.
    unsafe {
        if !x86::decode(rle_vals, rle_idxs, output) {
            rle_decode_scalar(rle_vals, rle_idxs, output);
        }
    }
}

/// [`RLE::decode_unchecked`] with the NEON kernel.
///
/// # Safety
///
/// Every index must be less than `rle_vals.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn rle_decode_neon<V: Copy, I: Copy + Into<usize> + 'static>(
    rle_vals: &[V],
    rle_idxs: &[I; 1024],
    output: &mut [V; 1024],
) {
    // SAFETY: the caller bounds every index.
    unsafe {
        if !aarch64::decode(rle_vals, rle_idxs, output) {
            rle_decode_scalar(rle_vals, rle_idxs, output);
        }
    }
}

/// [`RLE::decode_delta_unchecked`] with the AVX-512 kernel.
///
/// # Safety
///
/// The CPU must support AVX-512 F, BW, VL and VBMI, and every lane's last index must be less
/// than `rle_vals.len()`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn rle_decode_delta_avx512<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; 128],
    deltas: Option<&[u8; 128]>,
    output: &mut [V; 1024],
) {
    // SAFETY: the caller guarantees the features and bounds every lane.
    unsafe {
        if !avx512::decode_delta(rle_vals, bases, deltas, output) {
            rle_decode_delta_scalar(rle_vals, bases, deltas, output);
        }
    }
}

/// [`RLE::decode_delta_unchecked`] with the AVX2 kernel.
///
/// # Safety
///
/// The CPU must support AVX2, and every lane's last index must be less than `rle_vals.len()`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn rle_decode_delta_avx2<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; 128],
    deltas: Option<&[u8; 128]>,
    output: &mut [V; 1024],
) {
    // SAFETY: the caller guarantees the features and bounds every lane.
    unsafe {
        if !x86::decode_delta(rle_vals, bases, deltas, output) {
            rle_decode_delta_scalar(rle_vals, bases, deltas, output);
        }
    }
}

/// [`RLE::decode_delta_unchecked`] with the NEON kernel.
///
/// # Safety
///
/// Every lane's last index must be less than `rle_vals.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn rle_decode_delta_neon<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; 128],
    deltas: Option<&[u8; 128]>,
    output: &mut [V; 1024],
) {
    // SAFETY: the caller bounds every lane.
    unsafe {
        if !aarch64::decode_delta(rle_vals, bases, deltas, output) {
            rle_decode_delta_scalar(rle_vals, bases, deltas, output);
        }
    }
}

/// Portable [`RLE::decode_unchecked`].
///
/// # Safety
///
/// Every index must be less than `rle_vals.len()`.
pub unsafe fn rle_decode_scalar<V: Copy, I: Copy + Into<usize>>(
    rle_vals: &[V],
    rle_idxs: &[I; 1024],
    output: &mut [V; 1024],
) {
    for i in 0..1024 {
        let rle_idx: usize = unsafe { *rle_idxs.get_unchecked(i) }.into();
        debug_assert!(rle_idx < rle_vals.len());

        // SAFETY: `i` is less than 1024, and the caller guarantees every index is
        // less than `rle_vals.len()`.
        unsafe { *output.get_unchecked_mut(i) = *rle_vals.get_unchecked(rle_idx) };
    }
}

/// Portable [`RLE::decode_delta_unchecked`].
///
/// # Safety
///
/// Every lane's last index must be less than `rle_vals.len()`.
pub unsafe fn rle_decode_delta_scalar<V: Copy>(
    rle_vals: &[V],
    bases: &[u8; U8_LANES],
    deltas: Option<&[u8; U8_LANES]>,
    output: &mut [V; 1024],
) {
    let out = output.as_mut_ptr();
    // In output order: lane `FL_ORDER[g] * 16 + l` covers the 8 outputs at `l * 64 + g * 8`.
    for l in 0..16 {
        for (g, &order) in FL_ORDER.iter().enumerate() {
            let lane = order * 16 + l;
            let bits = deltas.map_or(0, |d| d[lane]);
            let mut index = usize::from(bases[lane]);
            for row in 0..8 {
                index += usize::from((bits >> row) & 1);
                debug_assert!(index < rle_vals.len());
                // SAFETY: the caller bounds the lane's indices, and the output position is
                // below 1024.
                unsafe { *out.add(l * 64 + g * 8 + row) = *rle_vals.get_unchecked(index) };
            }
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

    /// RLE indices for one chunk, with runs of `1..=max_run`.
    fn rle_indices(tc: &TestCase, max_run: usize) -> [usize; 1024] {
        let mut indices = [0usize; 1024];
        let mut position = 0;
        let mut index = 0;
        while position < 1024 {
            let run = tc.draw(gs::integers::<usize>().min_value(1).max_value(max_run));
            for slot in indices.iter_mut().skip(position).take(run) {
                *slot = index;
            }
            position += run;
            index += 1;
        }
        indices
    }

    fn values<V: TryFrom<u64>>(len: usize) -> Vec<V> {
        (0..len as u64)
            .map(|i| V::try_from(i * 37 % 127 + 1).ok().expect("small value"))
            .collect()
    }

    /// Decode `indices` as `I` into values of `V`, with `extra` values past the last index.
    fn check_decode<V, I>(indices: &[usize; 1024], extra: usize)
    where
        V: Copy + PartialEq + core::fmt::Debug + TryFrom<u64>,
        I: Copy + Into<usize> + TryFrom<usize> + 'static,
    {
        let max = indices.iter().copied().max().expect("non-empty");
        let vals = values::<V>(max + 1 + extra);
        let typed: [I; 1024] = indices.map(|i| I::try_from(i).ok().expect("index fits"));
        let expected = indices.map(|i| vals[i]);

        let mut actual = [vals[0]; 1024];
        // SAFETY: every index is at most `max`, below `vals.len()`.
        unsafe { V::decode_unchecked(&vals, &typed, &mut actual) };
        assert_eq!(actual, expected);

        let mut actual = [vals[0]; 1024];
        // SAFETY: as above.
        unsafe { rle_decode_scalar(&vals, &typed, &mut actual) };
        assert_eq!(actual, expected);

        // Every vector kernel this CPU supports.
        #[cfg(target_arch = "x86_64")]
        {
            if crate::transpose::has_avx512() {
                let mut actual = [vals[0]; 1024];
                // SAFETY: AVX-512 is detected; indices as above.
                unsafe { rle_decode_avx512(&vals, &typed, &mut actual) };
                assert_eq!(actual, expected, "AVX-512");
            }
            if crate::transpose::has_avx2() {
                let mut actual = [vals[0]; 1024];
                // SAFETY: AVX2 is detected; indices as above.
                unsafe { rle_decode_avx2(&vals, &typed, &mut actual) };
                assert_eq!(actual, expected, "AVX2");
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let mut actual = [vals[0]; 1024];
            // SAFETY: as above.
            unsafe { rle_decode_neon(&vals, &typed, &mut actual) };
            assert_eq!(actual, expected, "NEON");
        }
    }

    fn check_decode_all_widths(indices: &[usize; 1024], extra: usize) {
        check_decode::<u8, u16>(indices, extra);
        check_decode::<u16, u16>(indices, extra);
        check_decode::<u32, u16>(indices, extra);
        check_decode::<u64, u16>(indices, extra);
        if indices.iter().all(|&i| u8::try_from(i).is_ok()) {
            check_decode::<u8, u8>(indices, extra);
            check_decode::<u16, u8>(indices, extra);
            check_decode::<u32, u8>(indices, extra);
            check_decode::<u64, u8>(indices, extra);
        }
    }

    /// Generated RLE indices decode correctly, with and without values past the last index.
    #[hegel::test]
    fn test_rle_decode_runs_matches_index_model(tc: TestCase) {
        let max_run = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
        let indices = rle_indices(&tc, max_run);
        let extra = tc.draw(gs::integers::<usize>().max_value(8));
        check_decode_all_widths(&indices, extra);
    }

    /// An index type whose [`Into`] conversion is not its bit pattern decodes through it.
    #[test]
    fn test_rle_decode_converts_indices() {
        #[derive(Clone, Copy)]
        struct Halved(u8);
        impl From<Halved> for usize {
            fn from(index: Halved) -> usize {
                usize::from(index.0 / 2)
            }
        }

        let vals: Vec<u32> = (0..300).map(|i| i * 3 + 1).collect();
        let raw: [u8; 1024] = core::array::from_fn(|i| (i / 4) as u8);
        let indices = raw.map(Halved);
        let expected = raw.map(|r| vals[usize::from(r / 2)]);

        let mut actual = [0u32; 1024];
        // SAFETY: every converted index is at most 127, below `vals.len()`.
        unsafe { u32::decode_unchecked(&vals, &indices, &mut actual) };
        assert_eq!(actual, expected);
    }

    /// Delta-encode `indices` as `Delta` does (transposed, first row as bases) and pack the
    /// deltas at 1 bit.
    fn delta_encode(indices: &[usize; 1024]) -> ([u8; 128], [u8; 128], bool) {
        use crate::{BitPacking, Delta, Transpose};

        let input = indices.map(|i| u8::try_from(i).expect("u8 index"));
        let mut transposed = [0u8; 1024];
        u8::transpose(&input, &mut transposed);
        let bases: [u8; 128] = core::array::from_fn(|lane| transposed[lane]);
        let mut deltas = [0u8; 1024];
        u8::delta::<128>(&transposed, &bases, &mut deltas);
        assert!(
            deltas.iter().all(|&d| d <= 1),
            "RLE index deltas are 0 or 1"
        );
        let mut packed = [0u8; 128];
        // SAFETY: a 1-bit chunk of 1024 `u8` values packs into 128 bytes.
        unsafe { u8::unchecked_pack(1, &deltas, &mut packed) };
        (bases, packed, deltas.iter().all(|&d| d == 0))
    }

    /// Check every Delta kernel this CPU supports against `expected`.
    fn check_delta_kernels<V>(
        vals: &[V],
        bases: &[u8; 128],
        deltas: Option<&[u8; 128]>,
        expected: &[V; 1024],
    ) where
        V: Copy + PartialEq + core::fmt::Debug,
    {
        type Kernel<V> = unsafe fn(&[V], &[u8; 128], Option<&[u8; 128]>, &mut [V; 1024]);
        let mut kernels: Vec<(&str, Kernel<V>)> = vec![
            ("dispatch", V::decode_delta_unchecked),
            ("scalar", rle_decode_delta_scalar),
        ];
        #[cfg(target_arch = "x86_64")]
        {
            if crate::transpose::has_avx512() {
                kernels.push(("AVX-512", rle_decode_delta_avx512));
            }
            if crate::transpose::has_avx2() {
                kernels.push(("AVX2", rle_decode_delta_avx2));
            }
        }
        #[cfg(target_arch = "aarch64")]
        kernels.push(("NEON", rle_decode_delta_neon));

        for (name, kernel) in kernels {
            let mut actual = [vals[0]; 1024];
            // SAFETY: `vals` covers every lane's last index, and each vector kernel runs only
            // where its features are detected.
            unsafe { kernel(vals, bases, deltas, &mut actual) };
            assert_eq!(&actual, expected, "{name}");
        }
    }

    fn check_decode_delta<V>(indices: &[usize; 1024], extra: usize)
    where
        V: Copy + PartialEq + core::fmt::Debug + TryFrom<u64>,
    {
        let (bases, packed, all_zero) = delta_encode(indices);
        let max = indices.iter().copied().max().expect("non-empty");
        let vals = values::<V>(max + 1 + extra);
        let expected = indices.map(|i| vals[i]);

        check_delta_kernels(&vals, &bases, Some(&packed), &expected);
        if all_zero {
            check_delta_kernels(&vals, &bases, None, &expected);
        }
    }

    /// Delta indices decode correctly, for lane-aligned runs (every delta zero) and generated
    /// runs.
    #[hegel::test]
    fn test_rle_decode_delta_matches_index_model(tc: TestCase) {
        let indices = if tc.draw(gs::booleans()) {
            let run = 8 * tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
            core::array::from_fn(|i| i / run)
        } else {
            // Fall back to runs of 8 when the generated indices do not fit `u8`.
            let mut indices = rle_indices(&tc, 8);
            let max = indices.iter().copied().max().expect("non-empty");
            if max > usize::from(u8::MAX) {
                indices = core::array::from_fn(|i| i / 8);
            }
            indices
        };
        let extra = tc.draw(gs::integers::<usize>().max_value(8));
        check_decode_delta::<u8>(&indices, extra);
        check_decode_delta::<u16>(&indices, extra);
        check_decode_delta::<u32>(&indices, extra);
        check_decode_delta::<u64>(&indices, extra);
    }

    /// Arbitrary bases and delta bits (lanes out of order, windows wider than a step, indices
    /// past 255) decode lane by lane, through the kernels' fallbacks.
    #[hegel::test]
    fn test_rle_decode_delta_any_lanes(tc: TestCase) {
        let bases: [u8; 128] = core::array::from_fn(|_| tc.draw(gs::integers::<u8>()));
        let deltas: [u8; 128] = core::array::from_fn(|_| tc.draw(gs::integers::<u8>()));
        let extra = tc.draw(gs::integers::<usize>().max_value(8));
        let mut indices = [0usize; 1024];
        for lane in 0..128 {
            let mut index = usize::from(bases[lane]);
            for row in 0..8 {
                index += usize::from((deltas[lane] >> row) & 1);
                indices[u8_lane_start(lane) + row] = index;
            }
        }
        let len = indices.iter().copied().max().expect("non-empty") + 1 + extra;
        check_any_lanes::<u8>(len, &bases, &deltas, &indices);
        check_any_lanes::<u16>(len, &bases, &deltas, &indices);
        check_any_lanes::<u32>(len, &bases, &deltas, &indices);
        check_any_lanes::<u64>(len, &bases, &deltas, &indices);
    }

    fn check_any_lanes<V>(
        len: usize,
        bases: &[u8; 128],
        deltas: &[u8; 128],
        indices: &[usize; 1024],
    ) where
        V: Copy + PartialEq + core::fmt::Debug + TryFrom<u64>,
    {
        let vals = values::<V>(len);
        let expected = indices.map(|i| vals[i]);
        check_delta_kernels(&vals, bases, Some(deltas), &expected);
    }
}
