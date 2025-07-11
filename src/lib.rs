#![allow(incomplete_features)]
#![no_std]

extern crate alloc;
extern crate core;

use core::mem::size_of;
use num_traits::{PrimInt, Unsigned};

mod bitpacking;
mod bitpacking_cmp;
mod delta;
mod ffor;
mod macros;
mod transpose;

pub use bitpacking::*;
pub use bitpacking_cmp::*;
pub use delta::*;
pub use ffor::*;
pub use transpose::*;

pub const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

pub trait FastLanes: Sized + Unsigned + PrimInt {
    const T: usize = size_of::<Self>() * 8;
    const LANES: usize = 1024 / Self::T;
}

impl FastLanes for u8 {}
impl FastLanes for u16 {}
impl FastLanes for u32 {}
impl FastLanes for u64 {}

// Macro for repeating a code block bit_size_of::<T> times.
#[macro_export]
macro_rules! seq_t {
    ($ident:ident in u8 $body:tt) => {seq_macro::seq!($ident in 0..8 $body)};
    ($ident:ident in u16 $body:tt) => {seq_macro::seq!($ident in 0..16 $body)};
    ($ident:ident in u32 $body:tt) => {seq_macro::seq!($ident in 0..32 $body)};
    ($ident:ident in u64 $body:tt) => {seq_macro::seq!($ident in 0..64 $body)};
}

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
