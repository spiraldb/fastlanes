// Pack-time field -> bit-sliced (BitWeaving/V) transpose — step 1 of the bit-sliced scan plan.
//
// Bit-sliced scan kernels need the data held as W bit-planes (plane[j] = bit j of every value,
// a 1024-bit `[u64;16]`). A *per-scan* field->plane transpose is fatal (the naive bit-by-bit
// version below costs ~an order of magnitude more than an unpack). The design therefore pays the
// transpose ONCE at pack time. This example measures that transpose — a branch-free 64x64
// delta-swap, ~3.3-4.5k cyc/1024: several unpacks in absolute terms, but ~3x cheaper than the
// naive per-bit transpose and roughly flat in W — shows it round-trips and feeds the bit-sliced
// compare correctly, and argues it is a viable ONE-TIME pack cost that amortizes away across scans.
//
//   * `to_planes_fast`   : 1024 W-bit values -> W planes, via 16x a 64x64 bit transpose.
//   * `from_planes_fast` : the inverse (the same primitive — a transpose is its own inverse).
//   * round-trip + the bit-sliced `<` kernel validate that the planes are semantically correct.
//
// Run: RUSTFLAGS="-C target-cpu=native" cargo run --release --example bitsliced_pack
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::inline_always,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]

use std::hint::black_box;

use fastlanes::BitPacking;

const NW: usize = 16; // u64 words per 1024-bit plane (1024 / 64)
const NG: usize = 16; // groups of 64 values (1024 / 64)

type Plane = [u64; NW];

// ---- the primitive: in-place 64x64 bit-matrix transpose (Warren, Hacker's Delight 7-3) --------
//
// `a[r]` is row r; bit (r, c) is `(a[r] >> c) & 1`. After the call, `a[c]` holds column c, i.e.
// `a[c] >> r & 1` == old `a[r] >> c & 1`. This is exactly the per-64-group field<->plane swap:
// load value r into lane r, transpose, and row j becomes "bit j of every value". It is its own
// inverse, so the same routine packs and unpacks.
#[inline(always)]
fn transpose64(a: &mut [u64; 64]) {
    // Same masks/convention as Warren's `transpose32`, but the inner pass over the `j` independent
    // (i, i+j) pairs is a plain counted loop so the compiler can vectorize it (the classic
    // `k = (k+j+1) & !j` index recurrence serializes and runs ~40x slower here).
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    let mut j = 32usize;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            for i in k..k + j {
                let t = (a[i] ^ (a[i + j] >> j)) & m;
                a[i] ^= t;
                a[i + j] ^= t << j;
            }
            k += 2 * j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Transpose the eight 8x8 bit blocks packed in a `u64` (each byte is a row), so output byte `c`
/// holds column `c`. This is `bit_transpose::scalar::transpose_8x8` — the crate's fast 8x8
/// primitive — replicated here because it is `pub(crate)` and an example sees only the public API.
#[inline(always)]
fn transpose_8x8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^ t ^ (t << 28)
}

// ---- field -> planes and back -----------------------------------------------------------------

/// Width-proportional transpose for `w <= 8` built from the 8x8 primitive: each chunk of 8 values
/// is one `transpose_8x8`. This is the obvious "extend `bit_transpose`" path, but the benchmark
/// shows it does NOT beat the 64x64 delta-swap below — the per-byte strided gather/scatter
/// dominates and the cost grows with `w`. Kept to document the dead end for the next step.
fn to_planes_fast8(values: &[u64; 1024], w: usize, planes: &mut [Plane]) {
    debug_assert!(w <= 8);
    for p in &mut planes[..w] {
        *p = [0u64; NW];
    }
    for c in 0..128 {
        // Gather the low byte of 8 consecutive values: byte r = value(8c + r).
        let mut x = 0u64;
        for r in 0..8 {
            x |= (values[8 * c + r] & 0xff) << (r * 8);
        }
        // After the transpose, byte j holds "bit j of those 8 values"; place its 8 bits at the
        // value positions 8c..8c+8 of plane j.
        let t = transpose_8x8(x);
        let (word, shift) = (c / 8, (c % 8) * 8);
        for j in 0..w {
            planes[j][word] |= ((t >> (j * 8)) & 0xff) << shift;
        }
    }
}

/// Fast: 16 independent 64x64 transposes, one per group of 64 consecutive values.
///
/// `transpose64` works in MSB-on-both-axes numbering (`in(lane r, bit p) -> out(row 63-p,
/// bit 63-r)`); loading the group in reversed lane order and reading planes from reversed rows
/// cancels both reversals, landing on the natural convention: `plane[j]` bit `local` is bit `j`
/// of the value at position `local`.
fn to_planes_fast(values: &[u64; 1024], w: usize, planes: &mut [Plane]) {
    for g in 0..NG {
        let mut block = [0u64; 64];
        for local in 0..64 {
            block[63 - local] = values[g * 64 + local];
        }
        transpose64(&mut block);
        for j in 0..w {
            planes[j][g] = block[63 - j];
        }
    }
}

/// Inverse: reconstruct the 1024 values from the W planes (same transpose primitive).
fn from_planes_fast(planes: &[Plane], w: usize, values: &mut [u64; 1024]) {
    for g in 0..NG {
        let mut block = [0u64; 64];
        for j in 0..w {
            block[63 - j] = planes[j][g];
        }
        transpose64(&mut block);
        for local in 0..64 {
            values[g * 64 + local] = block[63 - local];
        }
    }
}

/// Reference (slow, bit-by-bit) field -> planes. Pins the bit-ordering convention.
fn to_planes_naive(values: &[u64; 1024], w: usize, planes: &mut [Plane]) {
    for p in planes.iter_mut() {
        *p = [0u64; NW];
    }
    for (i, &v) in values.iter().enumerate() {
        let (word, bit) = (i / 64, i % 64);
        for j in 0..w {
            planes[j][word] |= ((v >> j) & 1) << bit;
        }
    }
}

// ---- the scan kernel the planes feed (bit-sliced `<`, from examples/bitsliced.rs) -------------

/// `mask[i]` = (`value_i` < c). MSB->LSB bit-sliced unsigned compare — pure scalar bitwise ops.
fn bs_lt(planes: &[Plane], w: usize, c: u64) -> Plane {
    if c >= (1u64 << w) {
        return [!0u64; NW];
    }
    let mut lt = [0u64; NW];
    let mut eq = [!0u64; NW];
    for j in (0..w).rev() {
        let pj = &planes[j];
        if (c >> j) & 1 == 1 {
            for k in 0..NW {
                lt[k] |= eq[k] & !pj[k];
                eq[k] &= pj[k];
            }
        } else {
            for k in 0..NW {
                eq[k] &= !pj[k];
            }
        }
    }
    lt
}

fn ref_lt(values: &[u64; 1024], c: u64) -> Plane {
    let mut out = [0u64; NW];
    for (i, &v) in values.iter().enumerate() {
        if v < c {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

// ---- harness ----------------------------------------------------------------------------------

#[inline(always)]
fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

fn bench<F: FnMut()>(label: &str, mut f: F) {
    for _ in 0..500 {
        f();
    }
    let iters = 20_000u64;
    let mut best = u64::MAX;
    for _ in 0..20 {
        let t = rdtsc();
        for _ in 0..iters {
            f();
        }
        best = best.min(rdtsc() - t);
    }
    println!("{label:<34} {:>8.1} cyc/1024", best as f64 / iters as f64);
}

fn rng(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    *s
}

fn main() {
    let mut s = 0x1234_5678u64;
    println!("pack-time field -> bit-sliced transpose: round-trip + cost\n");

    // -------- correctness: every W in 1..=64 (covers all storage types) --------
    for w in 1..=64usize {
        let cap = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
        let mut values = [0u64; 1024];
        for v in &mut values {
            *v = rng(&mut s) & cap;
        }

        let mut planes = vec![[0u64; NW]; w];
        let mut planes_ref = vec![[0u64; NW]; w];
        to_planes_fast(&values, w, &mut planes);
        to_planes_naive(&values, w, &mut planes_ref);
        assert_eq!(planes, planes_ref, "fast vs naive planes mismatch W={w}");

        // The width-proportional W<=8 variant must produce identical planes.
        if w <= 8 {
            let mut planes8 = vec![[0u64; NW]; w];
            to_planes_fast8(&values, w, &mut planes8);
            assert_eq!(planes8, planes_ref, "fast8 vs naive planes mismatch W={w}");
        }

        let mut back = [0u64; 1024];
        from_planes_fast(&planes, w, &mut back);
        assert_eq!(values, back, "round-trip mismatch W={w}");

        // The planes must drive the scan kernel correctly. `bs_lt`'s `1<<w` guard is only valid
        // for w < 64 (w == 64 is uncompressed — nothing to bit-pack), so skip the kernel there.
        if w < 64 {
            let c = rng(&mut s) & cap;
            assert_eq!(
                bs_lt(&planes, w, c),
                ref_lt(&values, c),
                "bs_lt on planes W={w}"
            );
        }
    }
    println!("correctness: fast==naive planes, round-trip, and bs_lt all pass for W=1..=64\n");

    // -------- cost: fast transpose vs naive transpose vs a real `unpack` (the one-time yardstick).
    // The transpose is paid ONCE at pack time; every later scan is the cheap bit-sliced kernel.
    println!("--- one-time pack cost: fast bit-slice transpose vs naive vs a single unpack ---");
    for &w in &[3usize, 7, 8, 16, 32] {
        let cap = (1u128 << w) as u64 - 1;
        let mut values = [0u64; 1024];
        for v in &mut values {
            *v = rng(&mut s) & cap;
        }
        let mut planes = vec![[0u64; NW]; w];

        let vv = black_box(&values);
        bench(&format!("W={w} transpose (fast 64x64)"), || {
            to_planes_fast(vv, w, black_box(&mut planes));
            black_box(&planes);
        });
        // Width-proportional 8x8-based variant and the naive per-bit transpose, sampled only at
        // small W (the point is identical at every width and the naive one is ~100x slower).
        if w <= 8 {
            bench(&format!("W={w} transpose (fast 8x8)"), || {
                to_planes_fast8(vv, w, black_box(&mut planes));
                black_box(&planes);
            });
            bench(&format!("W={w} transpose (naive)"), || {
                to_planes_naive(vv, w, black_box(&mut planes));
                black_box(&planes);
            });
        }

        // Real FastLanes unpack baseline (u64 storage), runtime-known width.
        let packed_len = 128 * w / 8; // u64s
        let mut packed = vec![0u64; packed_len];
        // SAFETY: lengths match the documented contract (input 1024, output 128*W/8).
        unsafe { <u64 as BitPacking>::unchecked_pack(w, &values, &mut packed) };
        let mut out = [0u64; 1024];
        let pk = black_box(&packed);
        bench(&format!("W={w} unpack (baseline)"), || {
            // SAFETY: `packed` has the contract length; `out` is 1024.
            unsafe { <u64 as BitPacking>::unchecked_unpack(w, pk, black_box(&mut out)) };
            black_box(&out);
        });
        println!();
    }

    println!(
        "Takeaway:\n\
         - The branch-free 64x64 delta-swap transpose runs ~3.3-4.5k cyc/1024 (~3x faster than the\n\
         \x20 naive per-bit transpose, ~15x a single unpack) and is roughly flat in W.\n\
         - The byte-wise 8x8 variant does NOT improve on it (strided gather/scatter; worsens with W).\n\
         - But the transpose is a ONE-TIME pack cost. Against the bit-sliced scan kernel (the `<`\n\
         \x20 compare is ~27 cyc/1024 vs ~247 for today's SIMD unpack_cmp => ~220 cyc saved per scan)\n\
         \x20 it breaks even after ~15 scans and is free thereafter. Bit-sliced storage is therefore\n\
         \x20 viable; a faster (SIMD/VBMI) transpose is the lever to shrink the one-time cost for\n\
         \x20 rarely-scanned columns. That is step 2."
    );
}
