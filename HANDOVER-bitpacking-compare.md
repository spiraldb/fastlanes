# Handover: bitpacked-compare benchmarks across x86 ISA tiers, and memory-bound analysis

Context for a blog post on FastLanes fused unpack+compare kernels. All measurements taken
2026-07-14 on this session's machine; raw numbers, methodology, and repro commands below.

## What was asked

1. Does the stack-benchmark table from a prior session ("raw byte-bool cmp 40.4 ns, direct
   FastLanes-order packed cmp 50.6 ns") reproduce on x86 AVX-512? (NEON was dropped from scope —
   no aarch64 hardware in this container.)
2. Were the hoped-for bigger AVX-512 wins missing because the work is memory bound?
3. Does the streaming win of bitpacked-vs-plain compare line up with the compression factor?

## Machine

- Intel Xeon @ 2.80 GHz (cloud VM), 4 cores, no SMT.
- AVX-512 F/BW/DQ/VL/CD + VNNI, AVX2, SSE4.2.
- Caches: 32 KB L1d per core, 1 MB L2 per core, 33 MB shared L3.
- Effective single-core copy bandwidth (measured, streaming memcpy incl. RFO): ~12–14.6 GB/s.
- Rust 1.97.0 stable. Branch: `claude/bitpacking-table-arch-compare-4gha5j`
  (= `agent/stack-bitpacking-compare` + the streaming harness).

## Benchmark shapes

All kernels operate on FastLanes blocks: 1024 × u16 packed at width W=3, so 384 B packed input
per block. Outputs per block: packed mask 128 B (`[u64; 16]`), byte bools 1024 B, full unpack
2048 B.

- `benches/bitpacking_cmp_stack.rs` (divan): one hot block, all stack arrays, BATCH=256 blocks
  per sample. **Per-block time = divan median / 256.** Working set ~1.5 KB → always L1-resident.
- `examples/stream_cmp.rs`: iterates over N distinct blocks so the working set scales from L1
  to DRAM; reports ns/block directly. Includes a streaming-memcpy bandwidth baseline and a
  `cmp_plain` kernel (compare over already-unpacked u16 → same packed mask) as the
  "not bitpacked" reference.

Repro:

```bash
RUSTFLAGS="-C target-cpu=native"    cargo bench --bench bitpacking_cmp_stack  # AVX-512
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench --bench bitpacking_cmp_stack  # AVX2
                                    cargo bench --bench bitpacking_cmp_stack  # SSE2 baseline
RUSTFLAGS="-C target-cpu=native"    cargo run --release --example stream_cmp
```

## Result 1: the stack table on x86, by ISA tier

Per-block (1024 values), divan medians / 256, ns:

| kernel                              | SSE2 (default) | AVX2 (v3) | AVX-512 (native) | prior-session table |
|-------------------------------------|---------------:|----------:|-----------------:|--------------------:|
| `cmp_byte` (raw byte-bool cmp)      | 75.3           | 52.7      | 30.6–33.3        | 40.4                |
| `cmp_packed_raw` (FastLanes-order packed cmp) | 68.9 | **35.2**  | 48.4–52.1        | 50.6                |
| `cmp_packed` (packed + untranspose) | 157.9          | 91.5      | 106.1            | —                   |
| `unpack_and_cmp` (unpack, then collect_bool mask) | 962.5 | 180.0 | 126.3–128.0    | —                   |
| `cmp_then_pack` (byte bools → collect_bool) | 329.9  | 299.4     | 284.5–287.7      | —                   |
| `cmp_then_set_true` (byte bools → set-true-bits) | 523.0 | 527.0  | 493.8–530.5      | —                   |

Findings:

- On AVX-512 the prior table's *shape* reproduces: byte-bool beats packed raw. Packed matches
  almost exactly (~50 vs 50.6 ns); byte-bool is faster here (~31–33 vs 40.4 ns), widening the
  gap from 1.25× to ~1.6×.
- **The ordering is not universal on x86.** On AVX2 it flips decisively: packed raw wins 1.5×
  (35.2 vs 52.7). It also wins slightly on SSE2.
- **`cmp_packed_raw` has an AVX-512 codegen regression**: 48–52 ns with `-C target-cpu=native`
  vs 35.2 ns with AVX2 codegen *on the same CPU*. LLVM's AVX-512 vectorization of the per-lane
  bit accumulator is worse than its AVX2 version. Worth filing/investigating before drawing
  hardware conclusions from the AVX-512 column.

## Result 2: the stack benchmark is compute bound; streaming is memory bound

`examples/stream_cmp.rs`, AVX-512 codegen, ns per block. "in KiB" is packed input
(plain input for `cmp_plain` is 5.33× larger; byte/unpack output sizes in second column):

| blocks | packed in KiB | bool out KiB | memcpy | cmp_byte | cmp_packed | unpack | cmp_plain |
|-------:|--------------:|-------------:|-------:|---------:|-----------:|-------:|----------:|
| 8      | 3             | 8            | 4.3    | 33.3     | 48.4       | 33.2   | 117.5     |
| 64     | 24            | 64           | 13.8   | 40.3     | 49.6       | 76.9   | 117.0     |
| 512    | 192           | 512          | 13.8   | 42.2     | 50.5       | 104.1  | 119.4     |
| 2048   | 768           | 2048         | 18.3   | 61.3     | 51.5       | 126.0  | 123.5     |
| 16384  | 6144          | 16384        | 33.5   | 106.9    | 52.7       | 358.4  | 252.3     |
| 131072 | 49152         | 131072       | 63.0   | 168.9    | 73.2       | 387.2  | 273.0     |
| 524288 | 196608        | 524288       | 78.7   | 149.5    | 68.8       | 351.3  | 229.4     |

Same harness with AVX2 codegen (before `cmp_plain` was added):

| blocks | memcpy | cmp_byte | cmp_packed | unpack |
|-------:|-------:|---------:|-----------:|-------:|
| 8      | 8.5    | 61.7     | 41.0       | 38.7   |
| 524288 | 56.6   | 195.7    | 74.4       | 543.7  |

Findings:

- **L1-resident (the divan bench): compute bound.** Moving the block's data costs ~4 ns vs
  30–57 ns kernel time; memory is <15% of the cost. The modest AVX-512 win is a codegen story,
  not a bandwidth one.
- **DRAM-resident: memory bound.** Three signatures:
  1. AVX2 and AVX-512 converge for `cmp_packed` at DRAM (74.4 vs 68.8–72 ns) despite differing
     ~20% in L1 — vector width stops mattering.
  2. Times sit on the bandwidth roofline. Traffic per block (read + write + RFO):
     `cmp_packed` 640 B → predicted ~44–55 ns at measured bandwidth, observed ~69–73 ns
     (imperfect compute/memory overlap); `cmp_byte` 2432 B → predicted ~137–206 ns,
     observed 150–196 ns.
  3. Degradation tracks *output size*: packed mask (128 B out) barely moves L1→DRAM (~1.4×);
     byte bools (1 KB out) degrade ~4×; full unpack (2 KB out) ~11×.
- **The stack-bench table flips under streaming**: packed mask wins 2–2.7× over byte bools
  (69–73 vs 150–196 ns) because its output writes 8× fewer bytes. The byte-bool "win" is an
  artifact of cache-resident microbenchmarking; for Vortex-scale scans the packed mask is the
  right output format.

## Result 3: streaming win vs compression factor — yes, once output traffic is counted

Per block at W=3: plain input 2048 B vs packed 384 B → raw compression factor 16/3 ≈ **5.33×**.
Both compare variants also write the 128 B mask (+128 B RFO):

- total traffic: plain 2304 B vs packed 640 B → ratio **3.6×**
- measured at DRAM: 229–273 ns vs 68.8–73.2 ns → **3.3–3.7×** ✔ lines up with 3.6×

So the streaming speedup of compare-on-bitpacked equals the *total-traffic* ratio, i.e. the
compression factor diluted by fixed mask-output traffic. 5.33× is the asymptotic ceiling as
compute and output become negligible. Note the ratio is W-dependent (narrower W → bigger ratio).

Cache-resident, the bitpacked kernel still wins (48 vs 117 ns) but for a different reason:
the fused kernel keeps the mask in vector registers, while the plain path is the bit-at-a-time
`collect_bool` pattern (as used by Arrow `MutableBuffer::collect_bool` and Vortex
`BitBufferMut::collect_bool`), which compiles to ~117 ns/block even from L1 with AVX-512.

## Result 4: hand-written AVX-512 k-mask kernels (`vpcmpeqw` → `kmov`)

`cmp_kmask_avx512` in `examples/stream_cmp.rs` compares 32 u16 lanes per `vpcmpeqw` into a
k-mask register and `kmov`s the mask bits straight into the output words — no `collect_bool`
loop, and the mask comes out in **logical order** (bit i = value i), not FastLanes-transposed
order. Two kernels use it:

- `plain+kmask`: raw u16 input → logical mask (the "not bitpacked" path with ideal codegen).
- `unpack+kmask`: FastLanes `unpack` into a 2 KB stack scratch (L1-resident), then k-mask
  compare → logical mask, from bitpacked input.

Full run (AVX-512 native, ns/block; two runs agreed within noise):

| blocks | memcpy | cmp_byte | cmp_packed | unpack | cmp_plain | plain+kmask | unpack+kmask |
|-------:|-------:|---------:|-----------:|-------:|----------:|------------:|-------------:|
| 8      | 4.3    | 34.4     | 50.8       | 33.5   | 105.4     | 35.5        | 56.8         |
| 64     | 13.9   | 40.5     | 50.6       | 79.4   | 107.7     | 38.8        | 57.0         |
| 512    | 14.5   | 42.3     | 50.9       | 110.5  | 111.5     | 57.1        | 57.0         |
| 2048   | 19.4   | 62.1     | 50.9       | 125.0  | 114.5     | 85.6        | 58.4         |
| 16384  | 32.7   | 105.3    | 52.8       | 401.5  | 270.4     | 180.3       | 61.2         |
| 131072 | 61.3   | 185.3    | 80.8       | 546.0  | 310.8     | 249.6       | 87.9         |
| 524288 | 51.2   | 207.9    | 78.2       | 549.4  | 312.9     | 258.6       | 87.0         |

Findings:

- **The collect_bool floor is codegen, confirmed.** Same input, same output, the intrinsic
  compare runs 35 ns/block in L1 vs 105–117 ns for the autovectorized bit-accumulate loop —
  3× faster compute for the plain path.
- **Better codegen cannot rescue the non-bitpacked path at DRAM.** `plain+kmask` still costs
  ~250–260 ns streaming vs ~78–90 ns for the bitpacked kernels: it must read 5.33× the input
  bytes, and bandwidth doesn't care how good the compare instruction is.
- **`unpack+kmask` is the cheapest logical-order mask on this (non-VBMI) AVX-512 machine.**
  57 ns L1 / ~87–90 ns DRAM, vs ~106 ns L1 for `unpack_cmp` + `untranspose_bits`, and only
  ~12% behind the transposed-order-only `unpack_cmp` at DRAM. The 2 KB scratch stays in L1 so
  it adds no DRAM traffic.
- **IMPORTANT caveat: this CPU has no AVX512-VBMI** (Cascade Lake era: F/BW/DQ/VL/CD/VNNI
  only), so `untranspose_bits` runtime-dispatched to its **BMI2 fallback** everywhere in this
  session — the ~55 ns/block untranspose cost (106 − 51) is the BMI2 path, not the SIMD
  (`vpermb`) kernel. On VBMI hardware (Ice Lake+ / Zen 4+) the SIMD untranspose should cost
  ~5–10 ns/block, putting `unpack_cmp` + VBMI-untranspose at roughly ~55–60 ns L1 — likely
  even with or ahead of `unpack+kmask`, in a register-only form. The "kmask wins for logical
  order" finding is conditional on missing VBMI; re-measure on VBMI hardware before publishing
  it as a general AVX-512 claim.

The harness now also times the full logical-order path directly (`packed+untr` column =
`unpack_cmp` + dispatched `untranspose_bits`) and prints which untranspose implementation the
runtime dispatch selected (`untranspose_bits dispatch: vbmi=... bmi2=...`). On this machine
(vbmi=false, bmi2=true): 106–115 ns L1 / 122–126 ns DRAM, vs 56–62 / 78–85 ns for
`unpack+kmask`. **Re-run `cargo run --release --example stream_cmp` (with
`-C target-cpu=native`) on an Ice Lake+/Zen 4+ box** — the dispatch line will read vbmi=true
and the `packed+untr` vs `unpack+kmask` columns settle the logical-order ranking directly.

Note: DRAM-row absolute numbers vary ~±20% between runs on this shared VM (e.g. `cmp_byte`
150–210 ns across the session); trust within-run ratios over cross-run absolutes.

## Caveats / open threads for the post

- **No AVX512-VBMI on this machine** — all `untranspose_bits` numbers are the BMI2 fallback,
  not the SIMD `vpermb` kernel. The logical-order-path ranking (Result 4) must be re-measured
  on Ice Lake+ / Zen 4+ before publication.
- Cloud-VM single-core DRAM bandwidth (~12–14.6 GB/s) is on the low side; desktop/server parts
  with more per-core bandwidth will shift the DRAM crossover but not the qualitative story.
- The `cmp_packed_raw` AVX-512-vs-AVX2 codegen regression (35 → 50 ns) is unexplained; worth a
  disassembly look before publishing AVX-512 hardware claims.
- The prior-session numbers (40.4 / 50.6 ns) came from a different machine; only the packed
  number reproduced exactly here.
- NEON was not measured (x86-only container). The AVX2-flip result suggests the byte-vs-packed
  ordering is ISA-sensitive, so NEON deserves a real measurement rather than extrapolation.
- Writes were regular (RFO-paying) stores throughout; non-temporal stores would change the
  byte-bool and unpack rooflines.

## Files

- `benches/bitpacking_cmp_stack.rs` — divan stack benchmark (from `agent/stack-bitpacking-compare`).
- `examples/stream_cmp.rs` — working-set scaling harness (this session).
- Branch `claude/bitpacking-table-arch-compare-4gha5j` on `spiraldb/fastlanes` contains both.
