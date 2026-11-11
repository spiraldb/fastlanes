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
    _mm512_permutex2var_epi8, _mm512_permutex2var_epi16, _mm512_permutex2var_epi32,
    _mm512_permutex2var_epi64, _mm512_setzero_si512, _mm512_storeu_si512,
};
use core::mem::size_of;
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
    /// and for the register with the set bit. Each element of the index holds the element index
    /// in its low byte, and the other bytes are zero.
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

/// The roles of the 10 index bits in a plan.
#[derive(Clone, Copy)]
struct Layout {
    /// The direction: transpose if true, untranspose if false.
    forward: bool,

    /// The element size in bytes.
    size: usize,

    /// The number of lane layout, so a register holds `1 << lane_bits` elements.
    lane_bits: usize,

    /// The lane bit that selects the 256-bit half of a register.
    half: usize,

    /// The number of permute levels.
    levels: usize,

    /// The input bit that selects the loaded half.
    load_bit: usize,

    /// The output bit that selects the stored half.
    store_bit: usize,

    /// The input bits of the register number, from its low bit: first the `levels` bits that the
    /// permutes move into the lanes, then the bits that stay.
    reg: [usize; 10],

    /// The input bit that the permutes move into the register number at each level.
    incoming: [usize; MAX_LEVELS],
}

impl Layout {
    /// Give the number of register-number layout.
    const fn reg_bits(&self) -> usize {
        10 - self.lane_bits
    }

    /// Give if register-number bit `bit` must move into the lanes.
    const fn moves_out(&self, bit: usize) -> bool {
        moves_out(self.forward, self.half, self.load_bit, self.store_bit, bit)
    }
}

/// Give if input bit `bit`, which is a register-number bit after the load, must move into the
/// lanes before the store.
const fn moves_out(
    forward: bool,
    half: usize,
    load_bit: usize,
    store_bit: usize,
    bit: usize,
) -> bool {
    let out = move_bit(forward, bit);
    bit != load_bit && (out < half || out == store_bit)
}

/// Make a plan for elements of `size` bytes.
///
/// With `split_load` false, a register loads 64 contiguous bytes, and with `split_store` false it
/// stores 64 contiguous bytes. Otherwise the plan picks the halves that need the fewest permute
/// levels.
const fn plan(size: usize, forward: bool, split_load: bool, split_store: bool) -> Plan {
    let layout = order(halves(size, forward, split_load, split_store));
    let (load, store) = addresses(&layout);

    Plan {
        levels: layout.levels,
        regs: 1 << layout.reg_bits(),
        split_load: layout.load_bit != layout.half,
        split_store: layout.store_bit != layout.half,
        load,
        store,
        index: index(&layout),
    }
}

/// Pick the input bit that selects the loaded half and the output bit that selects the stored
/// half, with the fewest permute levels.
///
/// Bit `half` selects the upper half of 64 contiguous bytes. The result has no register order
/// yet: [`order`] sets `reg` and `incoming`.
const fn halves(size: usize, forward: bool, split_load: bool, split_store: bool) -> Layout {
    let lane_bits = (64 / size).trailing_zeros() as usize;
    let half = lane_bits - 1;

    let mut best = (usize::MAX, half, half);
    let mut load_bit = half;
    while load_bit < 10 {
        let mut store_bit = half;
        while store_bit < 10 {
            let mut levels = 0;
            let mut bit = half;
            while bit < 10 {
                if moves_out(forward, half, load_bit, store_bit, bit) {
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

    Layout {
        forward,
        size,
        lane_bits,
        half,
        levels,
        load_bit,
        store_bit,
        reg: [0; 10],
        incoming: [0; MAX_LEVELS],
    }
}

/// Set the order of the register-number layout, and the bits that the permutes move in.
///
/// The permutes move in the bits that the stores need in the register number, in place of the
/// bits that move out.
const fn order(mut layout: Layout) -> Layout {
    let (mut outgoing, mut staying, mut incoming) = (0, 0, 0);

    let mut bit = layout.half;
    while bit < 10 {
        if layout.moves_out(bit) {
            layout.reg[outgoing] = bit;
            outgoing += 1;
        }
        bit += 1;
    }

    let mut bit = 0;
    while bit < 10 {
        let out = move_bit(layout.forward, bit);
        let in_reg = bit >= layout.half && bit != layout.load_bit;
        let stored_in_reg = out >= layout.half && out != layout.store_bit;
        if in_reg && stored_in_reg {
            layout.reg[layout.levels + staying] = bit;
            staying += 1;
        } else if !in_reg && stored_in_reg {
            layout.incoming[incoming] = bit;
            incoming += 1;
        }
        bit += 1;
    }

    assert!(outgoing == layout.levels && incoming == layout.levels);
    assert!(layout.levels + staying == layout.reg_bits());
    layout
}

/// Give the `vpermt2` index vectors of each level.
const fn index(layout: &Layout) -> [[[u8; 64]; 2]; MAX_LEVELS] {
    let (lane_bits, half) = (layout.lane_bits, layout.half);
    let width = 1 << lane_bits;
    let mut index = [[[0; 64]; 2]; MAX_LEVELS];

    // The input bit of each lane bit, after the load.
    let mut lanes = [0; 6];
    let mut m = 0;
    while m < half {
        lanes[m] = m;
        m += 1;
    }
    lanes[half] = layout.load_bit;

    let mut t = 0;
    while t < layout.levels {
        let (in_bit, out_bit) = (layout.incoming[t], layout.reg[t]);

        // After the last level, the lanes must be in the order of the output index.
        let mut next = lanes;
        if t + 1 == layout.levels {
            let mut m = 0;
            while m < half {
                next[m] = source_bit(layout.forward, m);
                m += 1;
            }
            next[half] = source_bit(layout.forward, layout.store_bit);
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

                // The other bytes of the element stay zero.
                index[t][upper][lane * layout.size] = from as u8;
                lane += 1;
            }
            upper += 1;
        }

        lanes = next;
        t += 1;
    }

    index
}

/// Give the element offsets of the two halves that each register loads and stores.
const fn addresses(layout: &Layout) -> ([[usize; 2]; MAX_REGS], [[usize; 2]; MAX_REGS]) {
    let (mut load, mut store) = ([[0; 2]; MAX_REGS], [[0; 2]; MAX_REGS]);

    let mut r = 0;
    while r < 1 << layout.reg_bits() {
        let (mut low_load, mut low_store) = (0, 0);
        let mut i = 0;
        while i < layout.reg_bits() {
            let bit = (r >> i) & 1;
            low_load |= bit << layout.reg[i];
            let stored = if i < layout.levels {
                layout.incoming[i]
            } else {
                layout.reg[i]
            };
            low_store |= bit << move_bit(layout.forward, stored);
            i += 1;
        }

        load[r] = [low_load, low_load | (1 << layout.load_bit)];
        store[r] = [low_store, low_store | (1 << layout.store_bit)];
        r += 1;
    }

    (load, store)
}

/// Give if elements of `size` bytes load and store split halves, in the direction `forward`.
///
/// Split loads and stores each save a permute level but cost an insert or an extract. These are
/// the fastest choices on Sapphire Rapids, Granite Rapids and Zen 5. For `u8`, no split saves a
/// level.
const fn splits(size: usize, forward: bool) -> (bool, bool) {
    match (size, forward) {
        (1, _) => (false, false),
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
    /// The CPU must support AVX-512 F, BW and VBMI.
    unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i;
}

/// Implement [`Elem`] for each type with its `vpermt2` intrinsic.
macro_rules! impl_elem {
    ($($T:ty => $permute:ident;)*) => {
        $(
            impl Elem for $T {
                #[inline(always)]
                unsafe fn permute(a: __m512i, index: __m512i, b: __m512i) -> __m512i {
                    // SAFETY: the caller guarantees AVX-512 F, BW and VBMI.
                    unsafe { $permute(a, index, b) }
                }
            }
        )*
    };
}

impl_elem! {
    u8 => _mm512_permutex2var_epi8;

    u16 => _mm512_permutex2var_epi16;

    u32 => _mm512_permutex2var_epi32;

    u64 => _mm512_permutex2var_epi64;
}

/// A transpose of `E` in the direction `FORWARD`.
struct Kernel<E, const FORWARD: bool>(E);

impl<E: Elem, const FORWARD: bool> Kernel<E, FORWARD> {
    const PLAN: Plan = {
        let (split_load, split_store) = splits(size_of::<E>(), FORWARD);
        let plan = plan(size_of::<E>(), FORWARD, split_load, split_store);

        // `plan` splits only if a split saves a permute level, so `splits` must not ask for a
        // split that does not.
        assert!(plan.split_load == split_load && plan.split_store == split_store);

        plan
    };

    /// Run the plan on 1024 elements.
    ///
    /// # Safety
    /// `input` and `output` must each point to 1024 elements of `E`, and the CPU must support
    /// AVX-512 F, BW and VBMI.
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

/// [`Transpose::transpose`](crate::Transpose::transpose) with AVX-512.
///
/// # Safety
/// The CPU must support AVX-512 F, BW and VBMI.
#[inline]
pub unsafe fn transpose_avx512<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    // SAFETY: the caller guarantees the CPU features.
    unsafe { avx512::<T, true>(input, output) }
}

/// [`Transpose::untranspose`](crate::Transpose::untranspose) with AVX-512.
///
/// # Safety
/// The CPU must support AVX-512 F, BW and VBMI.
#[inline]
pub unsafe fn untranspose_avx512<T: FastLanes>(input: &[T; 1024], output: &mut [T; 1024]) {
    // SAFETY: the caller guarantees the CPU features.
    unsafe { avx512::<T, false>(input, output) }
}

/// Transpose with AVX-512, or untranspose if `FORWARD` is false.
///
/// # Safety
/// The CPU must support AVX-512 F, BW and VBMI.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
pub(super) unsafe fn avx512<T: FastLanes, const FORWARD: bool>(
    input: &[T; 1024],
    output: &mut [T; 1024],
) {
    let (input, output) = (input.as_ptr(), output.as_mut_ptr());

    // SAFETY: `FastLanes` is sealed, so `T` is the unsigned integer of the width matched below,
    // and both arrays hold 1024 of them. The caller guarantees the CPU features.
    unsafe {
        match size_of::<T>() {
            1 => Kernel::<u8, FORWARD>::run(input.cast(), output.cast()),
            2 => Kernel::<u16, FORWARD>::run(input.cast(), output.cast()),
            4 => Kernel::<u32, FORWARD>::run(input.cast(), output.cast()),
            8 => Kernel::<u64, FORWARD>::run(input.cast(), output.cast()),
            _ => unreachable!("FastLanes is sealed to u8, u16, u32 and u64"),
        }
    }
}
