//! AVX-512 implementations of [`Transpose`](crate::Transpose).
//!
//! `FL_ORDER` reverses the 3 bits of the order, so the transpose moves the element at index `p`
//! to an index whose bits are the bits of `p` in a different sequence. Bits 0..3 of the input
//! index are the row, bits 3..6 the order and bits 6..10 the lane; bits 0..4 of the output index
//! are the lane, bits 4..7 the reversed order and bits 7..10 the row.
//!
//! The kernels hold the 1024 elements in 512-bit registers. Some index bits give the register
//! number, the other bits give the lane in the register. Three operations change which bits are
//! lane bits:
//!
//! - A load can fill the two 256-bit halves of a register from two addresses, which makes one
//!   register-number bit a lane bit, at the cost of one insert.
//! - A store can write the two halves to two addresses, which does the same in reverse, at the
//!   cost of one extract.
//! - One `vpermt2` on each register of a pair moves one register-number bit into the lanes and one
//!   lane bit into the register number.
//!
//! [`plan`] picks the bits that each step moves and computes the addresses and permute indices at
//! compile time. [`splits`] gives the loads and stores that each width uses.

// The kernels must inline into the `target_feature` functions to be compiled with them.
#![allow(clippy::inline_always)]

use crate::FastLanes;
use core::arch::x86_64::{
    __m512i, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_castsi256_si512,
    _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_loadu_si512,
    _mm512_permutex2var_epi8, _mm512_permutex2var_epi32, _mm512_permutex2var_epi64,
    _mm512_setzero_si512, _mm512_storeu_si512,
};
use core::mem::size_of;
use fearless_simd::Avx512;
use seq_macro::seq;

/// The most permute levels that a plan can have.
const MAX_LEVELS: usize = 4;

/// The most registers that a plan can have.
const MAX_REGS: usize = 128;

/// Where each register of a transpose loads and stores its halves, and how it permutes them.
struct Plan {
    /// The number of `vpermt2` levels. Level `t` pairs the registers whose numbers differ in bit
    /// `t`, so groups of `1 << levels` consecutive registers are independent.
    levels: usize,

    /// The number of registers.
    regs: usize,

    /// If each register loads two separate halves, not 64 contiguous bytes.
    split_load: bool,

    /// If each register stores two separate halves, not 64 contiguous bytes.
    split_store: bool,

    /// The element offsets of the two 256-bit halves that each register loads.
    load: [[usize; 2]; MAX_REGS],

    /// The element offsets of the two 256-bit halves that each register stores.
    store: [[usize; 2]; MAX_REGS],

    /// The `vpermt2` index vectors at each level, for the register of the pair with the clear bit
    /// and for the register with the set bit. `u8` and `u16` use byte indices, `u32` and `u64`
    /// element indices.
    index: [[[u8; 64]; 2]; MAX_LEVELS],
}

/// Give the output index bit of input index bit `bit` of the transpose, or of the untranspose
/// if `forward` is false.
const fn move_bit(forward: bool, bit: usize) -> usize {
    match (forward, bit) {
        (true, 0..3) => bit + 7,
        (true, 3..6) | (false, 4..7) => 9 - bit,
        (true, _) => bit - 6,
        (false, 0..4) => bit + 6,
        (false, _) => bit - 7,
    }
}

/// Give the input index bit that moves to output index bit `bit`.
const fn source_bit(forward: bool, bit: usize) -> usize {
    move_bit(!forward, bit)
}

/// Make a plan for elements of `size` bytes.
///
/// With `split_load` false, a register loads 64 contiguous bytes, and with `split_store` false it
/// stores 64 contiguous bytes. Otherwise the plan picks the halves that need the fewest permute
/// levels.
#[allow(clippy::too_many_lines)]
const fn plan(size: usize, forward: bool, split_load: bool, split_store: bool) -> Plan {
    let width = 64 / size;
    let lane_bits = width.trailing_zeros() as usize;
    let half = lane_bits - 1;
    let reg_bits = 10 - lane_bits;

    // Pick the input bit that selects the loaded half and the output bit that selects the
    // stored half. Bit `half` selects the upper half of 64 contiguous bytes.
    let mut best = (usize::MAX, half, half);
    let mut load_bit = half;
    while load_bit < 10 {
        let mut store_bit = half;
        while store_bit < 10 {
            let mut levels = 0;
            let mut bit = half;
            while bit < 10 {
                let out = move_bit(forward, bit);
                if bit != load_bit && (out < half || out == store_bit) {
                    levels += 1;
                }
                bit += 1;
            }
            if levels < best.0 {
                best = (levels, load_bit, store_bit);
            }
            store_bit = if split_store { store_bit + 1 } else { 10 };
        }
        load_bit = if split_load { load_bit + 1 } else { 10 };
    }
    let (levels, load_bit, store_bit) = best;
    assert!(levels <= MAX_LEVELS);

    // The input bits of the register number: first the bits that the permutes move into the
    // lanes, then the others. The permutes move in the bits that the stores need in the register
    // number in their place.
    let mut reg = [0; 10];
    let mut incoming = [0; MAX_LEVELS];
    let (mut outgoing_count, mut staying_count, mut incoming_count) = (0, 0, 0);
    let mut bit = half;
    while bit < 10 {
        let out = move_bit(forward, bit);
        if bit != load_bit && (out < half || out == store_bit) {
            reg[outgoing_count] = bit;
            outgoing_count += 1;
        }
        bit += 1;
    }
    let mut bit = 0;
    while bit < 10 {
        let out = move_bit(forward, bit);
        let in_reg = bit >= half && bit != load_bit;
        let stored_in_reg = out >= half && out != store_bit;
        if in_reg && stored_in_reg {
            reg[levels + staying_count] = bit;
            staying_count += 1;
        } else if !in_reg && stored_in_reg {
            incoming[incoming_count] = bit;
            incoming_count += 1;
        }
        bit += 1;
    }
    assert!(outgoing_count == levels && incoming_count == levels);
    assert!(levels + staying_count == reg_bits);

    let mut plan = Plan {
        levels,
        regs: 1 << reg_bits,
        split_load: load_bit != half,
        split_store: store_bit != half,
        load: [[0; 2]; MAX_REGS],
        store: [[0; 2]; MAX_REGS],
        index: [[[0; 64]; 2]; MAX_LEVELS],
    };

    // The input bit of each lane bit, after the load.
    let mut lanes = [0; 6];
    let mut m = 0;
    while m < half {
        lanes[m] = m;
        m += 1;
    }
    lanes[half] = load_bit;

    let mut t = 0;
    while t < levels {
        let (in_bit, out_bit) = (incoming[t], reg[t]);

        // After the last level, the lanes must be in the order of the output index.
        let mut next = lanes;
        if t + 1 == levels {
            let mut m = 0;
            while m < half {
                next[m] = source_bit(forward, m);
                m += 1;
            }
            next[half] = source_bit(forward, store_bit);
        } else {
            let mut m = 0;
            while m < lane_bits {
                if next[m] == in_bit {
                    next[m] = out_bit;
                }
                m += 1;
            }
        }

        let mut upper = 0;
        while upper < 2 {
            let mut lane = 0;
            while lane < width {
                // The value of each input bit at this lane of the result.
                let mut value = [0; 10];
                let mut m = 0;
                while m < lane_bits {
                    value[next[m]] = (lane >> m) & 1;
                    m += 1;
                }
                value[in_bit] = upper;

                let mut from = value[out_bit] * width;
                let mut m = 0;
                while m < lane_bits {
                    from |= value[lanes[m]] << m;
                    m += 1;
                }

                let mut byte = 0;
                while byte < size {
                    plan.index[t][upper][lane * size + byte] = match size {
                        1 | 2 => (from * size + byte) as u8,
                        _ if byte == 0 => from as u8,
                        _ => 0,
                    };
                    byte += 1;
                }
                lane += 1;
            }
            upper += 1;
        }

        lanes = next;
        t += 1;
    }

    let mut r = 0;
    while r < plan.regs {
        let (mut load, mut store) = (0, 0);
        let mut i = 0;
        while i < reg_bits {
            let bit = (r >> i) & 1;
            load |= bit << reg[i];
            let stored = if i < levels { incoming[i] } else { reg[i] };
            store |= bit << move_bit(forward, stored);
            i += 1;
        }
        plan.load[r] = [load, load | (1 << load_bit)];
        plan.store[r] = [store, store | (1 << store_bit)];
        r += 1;
    }

    plan
}

/// Give if elements of `size` bytes load and store split halves, in the direction `forward`.
///
/// Split loads and stores each save a permute level but cost an insert or an extract. These are
/// the fastest choices on Sapphire Rapids, Granite Rapids and Zen 5. Zen 4 runs 512-bit shuffles
/// at half rate, so there `u32` and `u64` are about 25% slower than with AVX2.
const fn splits(size: usize, forward: bool) -> (bool, bool) {
    match (size, forward) {
        (2, true) => (true, false),
        (2, false) | (4, _) => (true, true),
        _ => (false, true),
    }
}

/// An element width, with its `vpermt2` instruction.
trait Elem {
    /// Permute the elements of `a` and `b` by `index`.
    ///
    /// # Safety
    /// The CPU must support AVX-512 F and VBMI.
    unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i;
}

impl Elem for u8 {
    #[inline(always)]
    unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i {
        // SAFETY: the caller guarantees VBMI.
        unsafe { _mm512_permutex2var_epi8(a, index, b) }
    }
}

impl Elem for u16 {
    #[inline(always)]
    unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i {
        // A byte permute, because the word permute takes three micro-ops on Intel CPUs.
        // SAFETY: the caller guarantees VBMI.
        unsafe { _mm512_permutex2var_epi8(a, index, b) }
    }
}

impl Elem for u32 {
    #[inline(always)]
    unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i {
        // SAFETY: the caller guarantees AVX-512 F.
        unsafe { _mm512_permutex2var_epi32(a, index, b) }
    }
}

impl Elem for u64 {
    #[inline(always)]
    unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i {
        // SAFETY: the caller guarantees AVX-512 F.
        unsafe { _mm512_permutex2var_epi64(a, index, b) }
    }
}

/// A transpose of `E` in the direction `FORWARD`.
struct Kernel<E, const FORWARD: bool>(E);

impl<E: Elem, const FORWARD: bool> Kernel<E, FORWARD> {
    const PLAN: Plan = {
        let (split_load, split_store) = splits(size_of::<E>(), FORWARD);
        plan(size_of::<E>(), FORWARD, split_load, split_store)
    };

    /// Run the plan on 1024 elements.
    ///
    /// # Safety
    /// `input` and `output` must each point to 1024 elements of `E`, and the CPU must support
    /// AVX-512 F and VBMI.
    // `seq!` repeats the `if` statement with a separator that clippy reads as a semicolon.
    #[allow(clippy::unnecessary_semicolon)]
    #[inline(always)]
    unsafe fn run(input: *const E, output: *mut E) {
        let plan = &Self::PLAN;

        // SAFETY: the caller guarantees the CPU features.
        let index: [[__m512i; 2]; MAX_LEVELS] = core::array::from_fn(|t| {
            core::array::from_fn(|upper| unsafe {
                _mm512_loadu_si512(plan.index[t][upper].as_ptr().cast())
            })
        });

        // Unroll the groups, so that every address is a constant offset.
        seq!(G in 0..128 {
            if G << plan.levels < plan.regs {
                // SAFETY: as for this function.
                unsafe { Self::group(input, output, &index, G << plan.levels) }
            }
        });
    }

    /// Run the plan on the group of registers that starts at register `first`.
    ///
    /// # Safety
    /// As for [`Self::run`].
    #[inline(always)]
    unsafe fn group(
        input: *const E,
        output: *mut E,
        index: &[[__m512i; 2]; MAX_LEVELS],
        first: usize,
    ) {
        let plan = &Self::PLAN;
        let group = 1 << plan.levels;
        let half = 32 / size_of::<E>();

        // SAFETY: the plan keeps every address within the 1024 elements, and the caller
        // guarantees the CPU features.
        unsafe {
            let mut regs = [_mm512_setzero_si512(); 1 << MAX_LEVELS];

            for (r, reg) in regs.iter_mut().enumerate().take(group) {
                let [low, high] = plan.load[first + r];
                *reg = if plan.split_load {
                    _mm512_inserti64x4::<1>(
                        _mm512_castsi256_si512(_mm256_loadu_si256(input.add(low).cast())),
                        _mm256_loadu_si256(input.add(high).cast()),
                    )
                } else {
                    debug_assert_eq!(high, low + half);
                    _mm512_loadu_si512(input.add(low).cast())
                };
            }

            for (t, [index_low, index_high]) in index.iter().enumerate().take(plan.levels) {
                for r in 0..group {
                    if r & (1 << t) == 0 {
                        let (a, b) = (regs[r], regs[r | (1 << t)]);
                        regs[r] = E::permute(a, *index_low, b);
                        regs[r | (1 << t)] = E::permute(a, *index_high, b);
                    }
                }
            }

            for (r, reg) in regs.iter().enumerate().take(group) {
                let [low, high] = plan.store[first + r];
                if plan.split_store {
                    _mm256_storeu_si256(output.add(low).cast(), _mm512_castsi512_si256(*reg));
                    _mm256_storeu_si256(
                        output.add(high).cast(),
                        _mm512_extracti64x4_epi64::<1>(*reg),
                    );
                } else {
                    debug_assert_eq!(high, low + half);
                    _mm512_storeu_si512(output.add(low).cast(), *reg);
                }
            }
        }
    }
}

/// Run the kernel for the width of `T`.
///
/// # Safety
/// The CPU must support AVX-512 F and VBMI.
#[inline(always)]
unsafe fn dispatch<T: FastLanes, const FORWARD: bool>(input: &[T; 1024], output: &mut [T; 1024]) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());
    // SAFETY: both arrays hold 1024 elements of the width matched below, and the transpose only
    // moves elements, so reading them as unsigned integers of that width is exact. The caller
    // guarantees the CPU features.
    unsafe {
        match size_of::<T>() {
            1 => Kernel::<u8, FORWARD>::run(input.cast(), output.cast()),
            2 => Kernel::<u16, FORWARD>::run(input.cast(), output.cast()),
            4 => Kernel::<u32, FORWARD>::run(input.cast(), output.cast()),
            _ => Kernel::<u64, FORWARD>::run(input.cast(), output.cast()),
        }
    }
}

/// [`Transpose::transpose`](crate::Transpose::transpose) with AVX-512.
///
/// The `Avx512` token proves that the CPU supports AVX-512 F, BW and VBMI. Get it from
/// [`Level`](fearless_simd::Level).
#[inline]
pub fn transpose_avx512<T: FastLanes>(_avx512: Avx512, input: &[T; 1024], output: &mut [T; 1024]) {
    // SAFETY: the token proves the CPU features.
    unsafe { transpose_avx512_unchecked(input, output) }
}

/// [`transpose_avx512`] without the token.
///
/// # Safety
/// The CPU must support AVX-512 F, BW and VBMI.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
unsafe fn transpose_avx512_unchecked<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    // SAFETY: the caller guarantees the CPU features.
    unsafe { dispatch::<T, true>(input, output) }
}

/// [`Transpose::untranspose`](crate::Transpose::untranspose) with AVX-512.
///
/// The `Avx512` token proves that the CPU supports AVX-512 F, BW and VBMI. Get it from
/// [`Level`](fearless_simd::Level).
#[inline]
pub fn untranspose_avx512<T: FastLanes>(
    _avx512: Avx512,
    input: &[T; 1024],
    output: &mut [T; 1024],
) {
    // SAFETY: the token proves the CPU features.
    unsafe { untranspose_avx512_unchecked(input, output) }
}

/// [`untranspose_avx512`] without the token.
///
/// # Safety
/// The CPU must support AVX-512 F, BW and VBMI.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
unsafe fn untranspose_avx512_unchecked<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    // SAFETY: the caller guarantees the CPU features.
    unsafe { dispatch::<T, false>(input, output) }
}
