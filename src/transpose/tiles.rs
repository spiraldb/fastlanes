//! Tile walks shared by the SIMD implementations of [`Transpose`](crate::Transpose).
//!
//! Both directions of the transpose move square tiles: [`untranspose_tiles`] reads tile rows 128
//! elements apart and writes them 64 apart, [`transpose_tiles`] the other way around. So each
//! element width needs one [`Tile`] kernel per instruction set.

// The walks must inline into the `target_feature` functions so their tiles compile with them.
#![allow(clippy::inline_always)]

use crate::FL_ORDER;

/// A kernel that transposes one `N`x`N` tile of `Elem`.
pub(super) trait Tile {
    type Elem;
    const N: usize;

    /// Set `dst[i][j] = src[j][i]` for the `N` rows of `N` values at each stride.
    ///
    /// # Safety
    /// `src` and `dst` must hold `N` rows of `N` values at their strides, and the CPU must support
    /// the instructions of the kernel.
    unsafe fn tile(
        src: *const Self::Elem,
        src_stride: usize,
        dst: *mut Self::Elem,
        dst_stride: usize,
    );
}

/// Untranspose 1024 elements tile by tile.
///
/// `output[a * 64 + b * 8 + c] = input[c * 128 + FL_ORDER[b] * 16 + a]`: for each `b`, the 8 rows
/// `c` of 16 lanes `a` become 16 rows `a` of 8 values `c`.
///
/// # Safety
/// `input` and `output` must each point to 1024 elements, and the CPU must support the
/// instructions of `K`.
#[inline(always)]
pub(super) unsafe fn untranspose_tiles<K: Tile>(input: *const K::Elem, output: *mut K::Elem) {
    for (b, &order) in FL_ORDER.iter().enumerate() {
        for a in (0..16).step_by(K::N) {
            for c in (0..8).step_by(K::N) {
                // SAFETY: the tiles stay within the 1024 elements, and the caller guarantees `K`.
                unsafe {
                    K::tile(
                        input.add(c * 128 + order * 16 + a),
                        128,
                        output.add(a * 64 + b * 8 + c),
                        64,
                    );
                }
            }
        }
    }
}

/// Transpose 1024 elements tile by tile, the inverse of [`untranspose_tiles`].
///
/// # Safety
/// As for [`untranspose_tiles`].
#[inline(always)]
pub(super) unsafe fn transpose_tiles<K: Tile>(input: *const K::Elem, output: *mut K::Elem) {
    for (b, &order) in FL_ORDER.iter().enumerate() {
        for a in (0..16).step_by(K::N) {
            for c in (0..8).step_by(K::N) {
                // SAFETY: the tiles stay within the 1024 elements, and the caller guarantees `K`.
                unsafe {
                    K::tile(
                        input.add(a * 64 + b * 8 + c),
                        64,
                        output.add(c * 128 + order * 16 + a),
                        128,
                    );
                }
            }
        }
    }
}
