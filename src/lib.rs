#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![no_std]

extern crate alloc;
extern crate core;

use core::mem::size_of;
use num_traits::{PrimInt, Unsigned};

mod bitpacking;
mod delta;
mod ffor;
mod macros;
mod transpose;

pub use bitpacking::*;
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

pub struct Pred<const B: bool>;

pub trait Satisfied {}

impl Satisfied for Pred<true> {}

// Macro for repeating a code block bit_size_of::<T> times.
#[macro_export]
macro_rules! seq_t {
    ($ident:ident in u8 $body:tt) => {seq_macro::seq!($ident in 0..8 $body)};
    ($ident:ident in u16 $body:tt) => {seq_macro::seq!($ident in 0..16 $body)};
    ($ident:ident in u32 $body:tt) => {seq_macro::seq!($ident in 0..32 $body)};
    ($ident:ident in u64 $body:tt) => {seq_macro::seq!($ident in 0..64 $body)};
}

#[cfg(test)]
mod test {
    use crate::FL_ORDER;

    #[test]
    fn test_ordering_is_own_inverse() {
        // Check that FL_ORDER "round-trips"; i.e., it is its own inverse permutation.
        for i in 0..8 {
            assert_eq!(FL_ORDER[FL_ORDER[i]], i);
        }
    }
}

// run the example code in the README as a test
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[cfg(test)]
mod tests {
    use crate::BitPacking;

    #[test]
    fn pack_u16_into_u3_no_unsafe() {
        const WIDTH: usize = 3;
    
        // Generate some values.
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << WIDTH)) as u16;
        }
    
        // Pack the values.
        let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
        BitPacking::pack::<WIDTH>(&values, &mut packed);
    
        // Unpack the values.
        let mut unpacked = [0u16; 1024];
        BitPacking::unpack::<WIDTH>(&packed, &mut unpacked);
        assert_eq!(values, unpacked);
    
        // Unpack a single value at index 14.
        // Note that for more than ~10 values, it can be faster to unpack all values and then 
        // access the desired one.
        for i in 0..1024 {
            assert_eq!(BitPacking::unpack_single::<WIDTH>(&packed, i), values[i]);
        }
    }
}