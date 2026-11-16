//! Exports one wrapper per public `fastlanes` kernel and element type so that the linked
//! library contains every kernel. `scripts/codesize.py` attributes each symbol to a module,
//! either by the `fastlanes::<module>` path in its name or, for code inlined into a wrapper, by
//! the `codesize__<module>__` prefix of the wrapper.
#![allow(clippy::missing_safety_doc, improper_ctypes_definitions)]

use core::mem::MaybeUninit;
use fastlanes::{
    BitPacking, BitPackingCompare, Delta, FoR, RLE, Transpose, transpose_bits, untranspose_bits,
};
use pastey::paste;
use seq_macro::seq;

macro_rules! wrappers {
    ($T:ident, $BITS:literal) => {
        paste! {
            // bitpacking
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__bitpacking__unchecked_pack__ $T>](
                width: usize, input: &[$T], output: &mut [$T],
            ) {
                unsafe { <$T as BitPacking>::unchecked_pack(width, input, output) }
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__bitpacking__unchecked_unpack__ $T>](
                width: usize, input: &[$T], output: &mut [$T],
            ) {
                unsafe { <$T as BitPacking>::unchecked_unpack(width, input, output) }
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__bitpacking__unchecked_unpack_single__ $T>](
                width: usize, input: &[$T], index: usize,
            ) -> $T {
                unsafe { <$T as BitPacking>::unchecked_unpack_single(width, input, index) }
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__bitpacking__unchecked_unpack_indices__ $T>](
                width: usize, input: &[$T], indices: &[usize], output: &mut [MaybeUninit<$T>],
            ) {
                unsafe { <$T as BitPacking>::unchecked_unpack_indices(width, input, indices, output) }
            }

            // ffor
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__ffor__for_pack__ $T>](
                width: usize, input: &[$T; 1024], reference: $T, output: &mut [$T],
            ) {
                seq!(W in 0..=$BITS {
                    match width {
                        #(W => <$T as FoR>::for_pack::<W, { 1024 * W / $BITS }>(
                            input, reference, output.try_into().unwrap(),
                        ),)*
                        _ => unreachable!(),
                    }
                });
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__ffor__unchecked_unfor_pack__ $T>](
                width: usize, input: &[$T], reference: $T, output: &mut [$T],
            ) {
                unsafe { <$T as FoR>::unchecked_unfor_pack(width, input, reference, output) }
            }

            // delta
            #[unsafe(no_mangle)]
            pub extern "C" fn [<codesize__delta__delta__ $T>](
                input: &[$T; 1024], base: &[$T; 1024 / $BITS], output: &mut [$T; 1024],
            ) {
                <$T as Delta>::delta(input, base, output)
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn [<codesize__delta__undelta__ $T>](
                input: &[$T; 1024], base: &[$T; 1024 / $BITS], output: &mut [$T; 1024],
            ) {
                <$T as Delta>::undelta(input, base, output)
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__delta__unchecked_undelta_pack__ $T>](
                width: usize, input: &[$T], base: &[$T; 1024 / $BITS], output: &mut [$T],
            ) {
                unsafe { <$T as Delta>::unchecked_undelta_pack(width, input, base, output) }
            }

            // bitpacking_cmp (one representative comparison)
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__bitpacking_cmp__unchecked_unpack_cmp__ $T>](
                width: usize, input: &[$T], output: &mut [u64; 16], value: $T,
            ) {
                unsafe {
                    <$T as BitPackingCompare>::unchecked_unpack_cmp(
                        width, input, output, |a: $T, b: $T| a < b, value,
                    )
                }
            }

            // rle
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__rle__encode_unchecked__ $T>](
                input: &[$T; 1024], rle_vals: &mut [$T; 1024], rle_idxs: &mut [u16; 1024],
            ) -> usize {
                unsafe { <$T as RLE>::encode_unchecked(input, rle_vals, rle_idxs) }
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<codesize__rle__decode_unchecked__ $T>](
                rle_vals: &[$T], rle_idxs: &[u16; 1024], output: &mut [$T; 1024],
            ) {
                unsafe { <$T as RLE>::decode_unchecked(rle_vals, rle_idxs, output) }
            }

            // transpose
            #[unsafe(no_mangle)]
            pub extern "C" fn [<codesize__transpose__transpose__ $T>](
                input: &[$T; 1024], output: &mut [$T; 1024],
            ) {
                <$T as Transpose>::transpose(input, output)
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn [<codesize__transpose__untranspose__ $T>](
                input: &[$T; 1024], output: &mut [$T; 1024],
            ) {
                <$T as Transpose>::untranspose(input, output)
            }

            // bit_transpose
            #[unsafe(no_mangle)]
            pub extern "C" fn [<codesize__bit_transpose__transpose_bits__ $T>](
                input: &[u64; 16], output: &mut [u64; 16],
            ) {
                transpose_bits::<$T>(input, output)
            }
        }
    };
}

wrappers!(u8, 8);
wrappers!(u16, 16);
wrappers!(u32, 32);
wrappers!(u64, 64);

#[unsafe(no_mangle)]
pub extern "C" fn codesize__bit_transpose__untranspose_bits(
    input: &[u64; 16],
    output: &mut [u64; 16],
) {
    untranspose_bits(input, output)
}
