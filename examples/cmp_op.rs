use fastlanes::{BitPacking, BitPackingCompare, CmpOp, FastLanes};
use std::hint::black_box;
use std::time::Instant;

/// ns per call: best of `reps` samples, each ~`iters` calls.
fn time<F: FnMut()>(mut f: F, iters: usize, reps: usize) -> f64 {
    for _ in 0..iters / 4 { f(); }
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..iters { f(); }
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    best
}

// xorshift
struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }

fn ops<T: Copy>(a: T, b: T) -> [CmpOp<T>; 4] { [CmpOp::Eq(a), CmpOp::Lt(a), CmpOp::Gt(a), CmpOp::Between { lower: a, upper: b }] }

fn run<T>(name: &str)
where T: BitPacking + BitPackingCompare + FastLanes + Copy + num_traits::FromPrimitive,
{
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let packed: Vec<Vec<T>> = (0..=T::T).map(|w| {
        let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
        let vals: [T; 1024] = std::array::from_fn(|_| T::from_u64(rng.next() & mask).unwrap());
        let mut p = vec![T::zero(); 128 * w / size_of::<T>()];
        unsafe { T::unchecked_pack(w, &vals, &mut p) };
        p
    }).collect();
    let a = T::from_u64(3).unwrap(); let b = T::from_u64(9).unwrap();
    let mut out = [0u64; 16];
    // 1. hot: same width+op every call (column scan of one chunk type)
    for w in 1..=T::T {
        for (oi, op) in ops(a, b).into_iter().enumerate() {
            let p = &packed[w];
            let ns = time(|| { unsafe { T::unchecked_unpack_cmp_op(black_box(w), black_box(p), &mut out, black_box(op)) }; black_box(&out); }, 20_000, 7);
            println!("hot {name} {w} {oi} {ns:.3}");
        }
    }
    // 2. mixed: random (width, op) per call -> stresses the dispatch branches
    let seq: Vec<(usize, CmpOp<T>)> = (0..4096).map(|_| { let w = 1 + (rng.next() as usize % T::T); (w, ops(a, b)[rng.next() as usize % 4]) }).collect();
    let mut i = 0;
    let ns = time(|| { let (w, op) = seq[i & 4095]; i += 1; unsafe { T::unchecked_unpack_cmp_op(w, &packed[w], &mut out, op) }; black_box(&out); }, 100_000, 9);
    println!("mixed {name} 0 0 {ns:.3}");
    // 3. mixed op only, fixed mid width
    let w = T::T / 2 + 1; let mut i = 0;
    let os: Vec<CmpOp<T>> = (0..4096).map(|_| ops(a, b)[rng.next() as usize % 4]).collect();
    let ns = time(|| { let op = os[i & 4095]; i += 1; unsafe { T::unchecked_unpack_cmp_op(w, &packed[w], &mut out, op) }; black_box(&out); }, 100_000, 9);
    println!("mixedop {name} {w} 0 {ns:.3}");
}

fn main() { run::<u8>("u8"); run::<u16>("u16"); run::<u32>("u32"); run::<u64>("u64"); }
