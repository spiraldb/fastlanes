#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::mem::size_of;

use num_traits::{PrimInt, Unsigned};

mod bitpacking;
mod delta;
mod transpose;

pub use bitpacking::*;
pub use delta::*;
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
macro_rules! seq_t {
    ($ident:ident in u8 $body:tt) => {seq!($ident in 0..8 $body);};
    ($ident:ident in u16 $body:tt) => {seq!($ident in 0..16 $body);};
    ($ident:ident in u32 $body:tt) => {seq!($ident in 0..32 $body);};
    ($ident:ident in u64 $body:tt) => {seq!($ident in 0..64 $body);};
}
pub(crate) use seq_t;

// Macro for repeating a code block size_of::<T> times.
macro_rules! seq_s {
    ($ident:ident in u8 $body:tt) => {seq!($ident in 0..1 $body);};
    ($ident:ident in u16 $body:tt) => {seq!($ident in 0..2 $body);};
    ($ident:ident in u32 $body:tt) => {seq!($ident in 0..4 $body);};
    ($ident:ident in u64 $body:tt) => {seq!($ident in 0..8 $body);};
}
pub(crate) use seq_s;

#[cfg(test)]
mod test {
    use crate::FastLanes;

    pub(crate) fn round_robin_values<T: FastLanes>() -> [T; 1024] {
        let mut values = [T::zero(); 1024];
        for i in 0..1024 {
            values[i] = T::from(i % T::max_value().to_usize().unwrap()).unwrap();
        }
        values
    }
}
