//! Exports one wrapper per public `fastlanes` kernel and element type so that the linked library
//! contains every kernel. `scripts/codesize.py` attributes each symbol to a module by the
//! `fastlanes::<module>` path in its name or, for code inlined into a wrapper, by the wrapper's
//! `codesize__<module>__<function>__<type>` name.
#![allow(clippy::missing_safety_doc, improper_ctypes_definitions, unused_unsafe)]

use core::mem::MaybeUninit;
use fastlanes::*;
use pastey::paste;

/// Exports `codesize__<module>__<function>__<T>` with the given arguments and body.
macro_rules! export {
    ($T:ident, $module:ident, $name:ident, ($($arg:ident: $ty:ty),*) $(-> $ret:ty)?, $body:expr) => {
        paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__ $module __ $name __ $T>]($($arg: $ty),*) $(-> $ret)? {
                unsafe { $body }
            }
        }
    };
}

macro_rules! kernels {
    ($T:ident, $BITS:literal) => {
        export!($T, bitpacking, unchecked_pack, (w: usize, i: &[$T], o: &mut [$T]), $T::unchecked_pack(w, i, o));
        export!($T, bitpacking, unchecked_unpack, (w: usize, i: &[$T], o: &mut [$T]), $T::unchecked_unpack(w, i, o));
        export!($T, bitpacking, unchecked_unpack_single, (w: usize, i: &[$T], n: usize) -> $T, $T::unchecked_unpack_single(w, i, n));
        export!($T, bitpacking, unchecked_unpack_indices, (w: usize, i: &[$T], n: &[usize], o: &mut [MaybeUninit<$T>]), $T::unchecked_unpack_indices(w, i, n, o));
        // `for_pack` has no runtime-width variant, so instantiate every width here.
        export!($T, ffor, for_pack, (w: usize, i: &[$T; 1024], r: $T, o: &mut [$T]), seq_macro::seq!(W in 0..=$BITS {
            match w {
                #(W => $T::for_pack::<W, { 1024 * W / $BITS }>(i, r, fastlanes_codesize_cast(o)),)*
                _ => unreachable!(),
            }
        }));
        export!($T, ffor, unchecked_unfor_pack, (w: usize, i: &[$T], r: $T, o: &mut [$T]), $T::unchecked_unfor_pack(w, i, r, o));
        export!($T, delta, delta, (i: &[$T; 1024], b: &[$T; 1024 / $BITS], o: &mut [$T; 1024]), $T::delta(i, b, o));
        export!($T, delta, undelta, (i: &[$T; 1024], b: &[$T; 1024 / $BITS], o: &mut [$T; 1024]), $T::undelta(i, b, o));
        export!($T, delta, unchecked_undelta_pack, (w: usize, i: &[$T], b: &[$T; 1024 / $BITS], o: &mut [$T]), $T::unchecked_undelta_pack(w, i, b, o));
        // One representative comparison: each comparison closure is a separate instantiation.
        export!($T, bitpacking_cmp, unchecked_unpack_cmp, (w: usize, i: &[$T], o: &mut [u64; 16], v: $T), $T::unchecked_unpack_cmp(w, i, o, |a: $T, b: $T| a < b, v));
        export!($T, rle, encode_unchecked, (i: &[$T; 1024], v: &mut [$T; 1024], n: &mut [u16; 1024]) -> usize, $T::encode_unchecked(i, v, n));
        export!($T, rle, decode_unchecked, (v: &[$T], n: &[u16; 1024], o: &mut [$T; 1024]), $T::decode_unchecked(v, n, o));
        export!($T, transpose, transpose, (i: &[$T; 1024], o: &mut [$T; 1024]), $T::transpose(i, o));
        export!($T, transpose, untranspose, (i: &[$T; 1024], o: &mut [$T; 1024]), $T::untranspose(i, o));
        export!($T, bit_transpose, transpose_bits, (i: &[u64; 16], o: &mut [u64; 16]), transpose_bits::<$T>(i, o));
    };
}

kernels!(u8, 8);
kernels!(u16, 16);
kernels!(u32, 32);
kernels!(u64, 64);
export!(u64, bit_transpose, untranspose_bits, (i: &[u64; 16], o: &mut [u64; 16]), untranspose_bits(i, o));

/// Reinterprets `slice` as an array without a length check, like the `unchecked_*` kernels do.
unsafe fn fastlanes_codesize_cast<T, const N: usize>(slice: &mut [T]) -> &mut [T; N] {
    unsafe { &mut *slice.as_mut_ptr().cast() }
}
