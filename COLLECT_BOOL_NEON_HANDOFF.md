# AArch64 `collect_bool` SIMD handoff

## Outcome

Packing 1024 byte booleans into a final, logical-order `[u64; 16]` bitmap is substantially faster
with explicit AArch64 SIMD:

| Implementation | Median per 1024 booleans | Relative to scalar |
|---|---:|---:|
| Scalar shift/OR | 214.3 ns | 1.0x |
| Stable baseline NEON | 24.3 ns | 8.8x |
| Nightly NEON `udot` | **15.2 ns** | **14.1x** |

The `udot` path is approximately 1.6x faster than the stable NEON path. Every implementation
produces the same LSB-first 128-byte result, and all timed buffers are fixed stack arrays with no
allocation.

These measurements were taken on an Apple M1 Pro using 1024 input booleans and 256 kernel calls
per timed benchmark iteration. The nightly result used Rust 1.95.0-nightly (2026-02-13) with
`-C target-cpu=native`.

## Code to hand off

The implementations and exported assembly kernels are in
[`benches/bitpacking_cmp_stack.rs`](benches/bitpacking_cmp_stack.rs):

- `pack_bools_collect`: scalar Arrow/Vortex-style shift/OR baseline.
- `pack_bools_collect_neon`: stable baseline AArch64 NEON.
- `pack_bools_collect_neon_dotprod`: nightly AArch64 FEAT_DotProd path.
- `collect_bool_kernel`, `collect_bool_neon_kernel`, and
  `collect_bool_neon_dotprod_kernel`: non-inlined benchmark/assembly entry points.

The `nightly-dotprod` Cargo feature gates the unstable `vdotq_u32` intrinsic. Normal stable builds
do not compile or require the dot-product code.

## Stable NEON algorithm

NEON has no direct equivalent of an x86 byte movemask. The stable implementation therefore:

1. Loads 16 booleans represented as bytes containing zero or one.
2. Shifts them by repeating lane positions `0, 1, ..., 7` with `vshlq_u8`.
3. Uses three pairwise widening reductions to produce two packed mask bytes.
4. Repeats this for four vectors, narrows the eight results, and performs one 8-byte store.

It processes 64 input booleans per loop. LLVM lowers the final narrowing operations to one `tbl`,
so its hot loop contains paired vector loads, four `ushl` operations, twelve `uaddlp` operations,
one `tbl`, and one `str d`.

## Nightly `udot` algorithm

The dot-product version multiplies each group of four Boolean bytes by the repeating weights
`[1, 2, 4, 8, 16, 32, 64, 128]`. One `udot` produces four partial packed values, and one
`uaddlp` combines adjacent partials into the two output mask bytes for a 16-byte input vector.

It uses the same four-vector narrowing and single-store strategy as baseline NEON. The generated
64-input hot loop contains four `udot`, four `uaddlp`, one `tbl`, and one `str d`, plus loads and
zeroing of the destructive dot-product accumulators.

The intrinsic currently requires:

```rust
#![feature(stdarch_neon_dotprod)]
```

and the implementation is annotated with:

```rust
#[target_feature(enable = "dotprod")]
```

## Reproducing the results

Stable benchmark:

```bash
RUSTFLAGS="-C target-cpu=native" \
  cargo bench --bench bitpacking_cmp_stack -- --min-time 3 --max-time 5
```

Nightly benchmark including `udot`:

```bash
RUSTFLAGS="-C target-cpu=native" \
  cargo +nightly bench --bench bitpacking_cmp_stack \
  --features nightly-dotprod -- --min-time 3 --max-time 5
```

Inspect nightly assembly:

```bash
RUSTFLAGS="-C target-cpu=native" \
  cargo +nightly rustc --bench bitpacking_cmp_stack --release \
  --features nightly-dotprod -- --emit=asm

rg -n -A50 '^_?collect_bool_neon_dotprod_kernel:' target/release/deps/*.s
```

## Integration requirements

The benchmark deliberately isolates the packing operation and accepts already-materialized
`&[bool; 1024]` input. This proves the cost and benefit of the packing kernels, but it does **not**
include the cost of invoking or materializing results from the public Arrow/Vortex API:

```rust
pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, f: F) -> Self
```

An arbitrary scalar `FnMut` cannot be invoked by NEON lanes directly. A production integration
that preserves this public signature must collect callback results into stack blocks before SIMD
packing, or provide an internal fast path for callers that already have contiguous Boolean bytes;
that complete path must be benchmarked separately.

Before merging into Arrow or Vortex, the implementer should also:

1. Generalize the fixed 1024-element kernels to arbitrary `len`.
2. Handle full 64-boolean blocks with SIMD and finish the tail with a correct scalar path.
3. Preserve LSB-first ordering and clear unused high bits in the final byte.
4. Dispatch the `udot` path only when FEAT_DotProd is enabled or detected.
5. Keep baseline NEON for AArch64 CPUs without dot-product and scalar code for other targets.
6. Test zero length, every tail length from 1 through 63, and randomized inputs against scalar
   `collect_bool`.
7. Benchmark both packing-only and end-to-end callback workloads before choosing the production
   dispatch threshold.

## Validation completed

- Stable `cargo check --benches` passes.
- Stable and nightly test suites pass: 155 unit tests and one documentation test.
- Both SIMD variants were checked against the scalar logical bitmap before timing.
- The emitted nightly assembly contains the intended `udot` instructions.

Experimental branch: `agent/stack-bitpacking-compare`.
