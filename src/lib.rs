#![no_std]

extern crate alloc;
extern crate core;

use core::mem::size_of;
use num_traits::{PrimInt, Unsigned};

mod bit_transpose;
mod bitpacking;
mod bitpacking_cmp;
mod delta;
mod ffor;
mod macros;
mod rle;
mod transpose;

pub use bit_transpose::*;
pub use bitpacking::*;
pub use bitpacking_cmp::*;
pub use delta::*;
pub use ffor::*;
pub use rle::*;
pub use transpose::*;

pub const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

/// The lane and row of position `index` (below 1024) in a vector of `T` laid out in the order the
/// pack, unpack and delta kernels use: the inverse of their position of row `row` of lane `lane`,
/// `FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane`.
#[inline]
pub(crate) const fn lane_row<T: FastLanes>(index: usize) -> (usize, usize) {
    let lane = index % T::LANES;
    let s = index / 128; // because `(FL_ORDER[o] * 16) + lane` is always < 128
    let fl_order = (index - s * 128 - lane) / 16; // value of FL_ORDER[o]
    let o = FL_ORDER[fl_order]; // because this transposition is invertible!
    (lane, o * 8 + s)
}

/// An unsigned integer type that the `FastLanes` kernels support.
///
/// The trait is sealed: only `u8`, `u16`, `u32` and `u64` implement it. The SIMD kernels read
/// the elements as these integers, so other types must not implement it.
pub trait FastLanes: Sized + Unsigned + PrimInt + sealed::Sealed {
    const T: usize = size_of::<Self>() * 8;
    const LANES: usize = 1024 / Self::T;
}

mod sealed {
    /// Prevents implementations of [`FastLanes`](super::FastLanes) outside this crate.
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

impl FastLanes for u8 {}
impl FastLanes for u16 {}
impl FastLanes for u32 {}
impl FastLanes for u64 {}

/// Reinterprets the first `N` elements of `slice` as a fixed-size array reference.
///
/// # Safety
/// `slice.len()` must be at least `N`. This is checked only with `debug_assert!`.
#[inline]
pub(crate) unsafe fn as_array_unchecked<T, const N: usize>(slice: &[T]) -> &[T; N] {
    debug_assert!(slice.len() >= N);
    // SAFETY: the caller guarantees `slice` has at least `N` elements, and `[T; N]`
    // has the same alignment as `T`.
    unsafe { &*slice.as_ptr().cast::<[T; N]>() }
}

/// Reinterprets the first `N` elements of `slice` as a mutable fixed-size array reference.
///
/// # Safety
/// `slice.len()` must be at least `N`. This is checked only with `debug_assert!`.
#[inline]
pub(crate) unsafe fn as_array_mut_unchecked<T, const N: usize>(slice: &mut [T]) -> &mut [T; N] {
    debug_assert!(slice.len() >= N);
    // SAFETY: the caller guarantees `slice` has at least `N` elements, and `[T; N]`
    // has the same alignment as `T`.
    unsafe { &mut *slice.as_mut_ptr().cast::<[T; N]>() }
}

// Macro for repeating a code block bit_size_of::<T> times.
#[macro_export]
macro_rules! seq_t {
    ($ident:ident in u8 $body:tt) => {seq_macro::seq!($ident in 0..8 $body)};
    ($ident:ident in u16 $body:tt) => {seq_macro::seq!($ident in 0..16 $body)};
    ($ident:ident in u32 $body:tt) => {seq_macro::seq!($ident in 0..32 $body)};
    ($ident:ident in u64 $body:tt) => {seq_macro::seq!($ident in 0..64 $body)};
}

/// Dispatches a runtime bit width to a kernel generic over the const width `W`.
///
/// `width_dispatch!(T, module, kernel<G..>(arg: Ty, ..) -> Ret where ..)` defines `module`, which
/// holds one `#[inline(never)]` `run` function per width of `T`, each in its own `w<W>` submodule
/// (`wfull` for `W == T`), and `module::dispatch(width, args..)`, which calls the one for `width`.
/// `kernel` must be an `#[inline(always)] unsafe fn kernel<const W: usize, const B: usize, G..>`
/// taking slices, where `B == 1024 * W / T` is the packed length.
///
/// rustc places a function in the codegen unit of the module that defines it, including every
/// instance of a generic function. A kernel instantiated once per width from a single module puts
/// every width into one codegen unit, which LLVM compiles on one thread. One module per width
/// spreads them across codegen units. When a kernel is generic only over `W` and `B`, each `run` is
/// an ordinary function, so it is compiled once, in this crate, rather than by its callers.
///
/// Single-element kernels such as [`BitPacking::unpack_single`] are left out: they are `#[inline]`
/// so that callers inline them, which a per-width call would prevent.
macro_rules! width_dispatch {
    (
        $T:ident, $module:ident,
        $kernel:ident $(<$($g:ident),+>)? ($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?
        $(where $($bound:tt)+)?
    ) => {
        mod $module {
            #[allow(unused_imports)]
            use super::*;

            $crate::seq_t_items!(W in $T {
                pub(super) mod w~W {
                    #[allow(unused_imports)]
                    use super::*;

                    /// # Safety
                    /// The kernel's safety contract at width `W`.
                    #[inline(never)]
                    pub(in super::super) unsafe fn run $(<$($g),+>)? ($($arg: $ty),*) $(-> $ret)?
                    $(where $($bound)+)?
                    {
                        const B: usize = 1024 * W / <$T>::T;
                        // SAFETY: forwarded to the caller.
                        unsafe { super::super::$kernel::<W, B $(, $($g),+)?>($($arg),*) }
                    }
                }
            });

            // `seq_t_items!` has an exclusive upper bound, so `W == T` gets its own module.
            pub(super) mod wfull {
                #[allow(unused_imports)]
                use super::*;

                /// # Safety
                /// The kernel's safety contract at width `T`.
                #[inline(never)]
                pub(in super::super) unsafe fn run $(<$($g),+>)? ($($arg: $ty),*) $(-> $ret)?
                $(where $($bound)+)?
                {
                    const W: usize = <$T>::T;
                    // SAFETY: forwarded to the caller.
                    unsafe { super::super::$kernel::<W, 1024 $(, $($g),+)?>($($arg),*) }
                }
            }

            /// Calls the kernel at width `width`.
            ///
            /// # Safety
            /// The kernel's safety contract at width `width`.
            ///
            /// # Panics
            /// Panics if `width` is greater than the bit width of `T`.
            #[inline(always)]
            pub(super) unsafe fn dispatch $(<$($g),+>)? (width: usize, $($arg: $ty),*) $(-> $ret)?
            $(where $($bound)+)?
            {
                // SAFETY: forwarded to the caller.
                unsafe {
                    $crate::seq_t!(W in $T {
                        match width {
                            #(W => w~W::run($($arg),*),)*
                            <$T>::T => wfull::run($($arg),*),
                            _ => unreachable!("Unsupported width: {}", width),
                        }
                    })
                }
            }
        }
    };
}
pub(crate) use width_dispatch;

/// [`seq_t!`] for item position.
macro_rules! seq_t_items {
    ($ident:ident in u8 $body:tt) => {seq_macro::seq!($ident in 0..8 $body);};
    ($ident:ident in u16 $body:tt) => {seq_macro::seq!($ident in 0..16 $body);};
    ($ident:ident in u32 $body:tt) => {seq_macro::seq!($ident in 0..32 $body);};
    ($ident:ident in u64 $body:tt) => {seq_macro::seq!($ident in 0..64 $body);};
}
pub(crate) use seq_t_items;

const fn supported_bit_width(width: usize, type_width: usize) -> bool {
    match type_width {
        8 => width <= 8,
        16 => width <= 16,
        32 => width <= 32,
        64 => width <= 64,
        _ => unreachable!(),
    }
}

pub trait FastLanesComparable: Copy {
    type Bitpacked: FastLanes;

    fn as_unpacked(inner: Self::Bitpacked) -> Self;
}

macro_rules! impl_fastlanes_comparable {
    ($value_type:ty, $bitpacked_type:ty) => {
        impl FastLanesComparable for $value_type {
            type Bitpacked = $bitpacked_type;

            #[inline]
            #[allow(unnecessary_transmutes, clippy::useless_transmute)]
            fn as_unpacked(inner: Self::Bitpacked) -> Self {
                unsafe { core::mem::transmute(inner) }
            }
        }
    };
}

impl_fastlanes_comparable!(u8, u8);
impl_fastlanes_comparable!(i8, u8);

impl_fastlanes_comparable!(u16, u16);
impl_fastlanes_comparable!(i16, u16);

impl_fastlanes_comparable!(u32, u32);
impl_fastlanes_comparable!(i32, u32);

impl_fastlanes_comparable!(u64, u64);
impl_fastlanes_comparable!(i64, u64);

// run the example code in the README as a test
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[cfg(test)]
mod tests {
    use crate::{BitPacking, FL_ORDER};

    #[test]
    fn test_ordering_is_own_inverse() {
        // Check that FL_ORDER "round-trips"; i.e., it is its own inverse permutation.
        for i in 0..8 {
            assert_eq!(FL_ORDER[FL_ORDER[i]], i);
        }
    }

    #[test]
    fn pack_u16_into_u3_no_unsafe() {
        const WIDTH: usize = 3;
        const B: usize = 128 * WIDTH / size_of::<u16>();

        // Generate some values.
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << WIDTH)) as u16;
        }

        // Pack the values.
        let mut packed = [0; B];
        BitPacking::pack::<WIDTH, B>(&values, &mut packed);

        // Unpack the values.
        let mut unpacked = [0u16; 1024];
        BitPacking::unpack::<WIDTH, B>(&packed, &mut unpacked);
        assert_eq!(values, unpacked);

        // Unpack a single value at index 14.
        // Note that for more than ~10 values, it can be faster to unpack all values and then
        // access the desired one.
        for i in 0..1024 {
            assert_eq!(BitPacking::unpack_single::<WIDTH, B>(&packed, i), values[i]);
        }
    }
}
