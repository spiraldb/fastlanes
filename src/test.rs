use alloc::vec::Vec;

#[inline(never)]
#[must_use]
pub fn collect_bool_cmp<T: PartialEq + Copy>(unpacked: &[T; 1024], cmp: &T) -> Vec<u64> {
    collect_bool(unpacked.len(), |idx| unpacked[idx] == *cmp)
}

#[inline]
#[must_use]
pub fn ceil(value: usize, divisor: usize) -> usize {
    // Rewrite as `value.div_ceil(&divisor)` after
    // https://github.com/rust-lang/rust/issues/88581 is merged.
    value / divisor + usize::from(0 != value % divisor)
}

#[inline]
pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, mut f: F) -> Vec<u64> {
    let mut buffer = Vec::with_capacity(ceil(len, 64) * 8);

    let chunks = len / 64;
    let remainder = len % 64;
    for chunk in 0..chunks {
        let mut packed = 0;
        for bit_idx in 0..64 {
            let i = bit_idx + chunk * 64;
            packed |= u64::from(f(i)) << bit_idx;
        }

        // SAFETY: Already allocated sufficient capacity
        buffer.push(packed);
    }

    if remainder != 0 {
        let mut packed = 0;
        for bit_idx in 0..remainder {
            let i = bit_idx + chunks * 64;
            packed |= u64::from(f(i)) << bit_idx;
        }

        // SAFETY: Already allocated sufficient capacity
        buffer.push(packed);
    }

    buffer.truncate(ceil(len, 8));
    buffer
}
