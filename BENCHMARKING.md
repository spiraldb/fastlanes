# Benchmarking

FastLanes uses Divan microbenchmarks for individual kernels. They live in `benches/`, and CI runs
them on CodSpeed.

## Microbenchmarks

Run one benchmark file with the `runtime` feature, so the SIMD kernels are selected from the CPU at
runtime:

```bash
cargo bench --features runtime --bench <name>
```

Without `runtime`, kernel selection depends on the target features the benchmarks were compiled
with, which by default means the scalar kernels.

### Separate setup from measured work

Use `with_inputs` to construct fixtures outside the timed closure:

```rust
bencher
    .with_inputs(|| bench_fixture())
    .bench_refs(|(values, output)| T::transpose(values, output));
```

### Exclude destruction when it is not part of the operation

Divan drops the closure's return value after timing stops. Return the result instead of discarding
it inside the closure:

```rust
bencher
    .with_inputs(|| make_big_vec())
    .bench_values(|v| transform(v));
```

Use `bench_refs` when the operation mutates an input in place. Divan keeps the input alive until the
sample ends, so its destruction is outside the timed region:

```rust
bencher
    .with_inputs(|| make_big_vec())
    .bench_refs(|v| v.sort());
```

Other values dropped inside the closure contribute to the measurement. Keep those drops only when
they are part of the cost being measured.

### Prevent the compiler from removing work

The compiler can fold or remove work when it knows the inputs. Divan black-boxes inputs from
`with_inputs` and the closure's return value to prevent that optimization. Captured values do not
receive that protection:

```rust
let values = make_values();
bencher.bench(|| process(&values));
```

Pass the input through `with_inputs` instead:

```rust
let values = make_values();
bencher
    .with_inputs(|| &values)
    .bench_refs(|values| process(values));
```

An explicit `black_box` also protects a captured input:

```rust
bencher.bench(|| process(divan::black_box(&values)));
```

Inside a manual loop, black-box each result so the compiler cannot discard earlier iterations:

```rust
bencher.bench(|| {
    for value in divan::black_box(&values) {
        divan::black_box(process(value));
    }
});
```

### Choose reproducible inputs

Use a fixed-seed generator for random data, such as the xorshift in `benches/rle.rs`. Unseeded
random data can change the measured work between runs even when the code stays the same.

Choose inputs that exercise useful work. For tiny operations, repeat a fixed amount of work and
black-box each result. Size inputs by bytes when comparing element types of different widths.

Keep each iteration below 1 ms so the suite remains practical to run. This is the cost of one
closure invocation, not the total time of the benchmark binary. Check the per-iteration `HEAD`
value in the CodSpeed report before merging a new benchmark. Reduce the input size, split a larger
workload into parameterized cases, or keep it local if it cannot fit this budget.

For CPU kernels, keep the working set within the runner's L2 cache unless memory bandwidth is the
intended measurement.

### Parameterize workloads

Use Divan's `args`, `consts`, and `types` for distinct workloads. Keep shared parameter lists in named
constants:

```rust
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];
const VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE)]
fn my_bench<const N: usize>(bencher: Bencher, num_indices: usize) { /* ... */ }
```

## CodSpeed

The `bench-codspeed` job in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds the benchmarks with
`-C target-feature=+avx2` and the `runtime` feature, then runs them under
[CPU simulation](https://codspeed.io/docs/instruments/cpu). Simulation estimates cycles from one
execution's instruction trace, including cache and memory access costs. Divan's sampling settings
have no effect.

### Give short benchmarks enough measured work

Simulation has a fixed harness cost, so most of a very short benchmark's reported time can come from
that cost. Aim for at least a few microseconds of useful work per iteration, for example by
processing many 1024-element chunks per iteration, as the RLE decode benchmarks do.

### Keep incompatible benchmarks out of automatic runs

Simulation excludes system-call time and measures one cold iteration. Use automatic reports to track
maintained FastLanes entry points. Keep third-party kernels and frozen implementations available for
local comparisons, gated with `#[cfg(not(codspeed))]`. These baselines help compare implementations
locally, but unexplained changes in their automatic results do not identify a change in maintained
FastLanes code.

Use `#[cfg(not(codspeed))]` for a benchmark that cannot produce useful CodSpeed measurements. When
only one input case is unreliable, filter the benchmark's argument list instead. Record the reason
beside the exclusion and keep the supporting measurements in the PR. Keep CI commands independent of
individual benchmark names.

### Investigate reports on unchanged code

Check the exact base and head commits before attributing a change. An old comparison base can
include unrelated changes. Inspect the callgraph for changes in measured work or allocation.
Simulation avoids clock noise, but changes in code placement or allocation can still change the
instruction trace on unchanged code.

Compare independent builds to check sensitivity to code placement. Keep commands and environment
fixed. One repeatable executable does not establish that unrelated source changes cannot affect its
measurements.

Record the executable identity, base and head commits, hardware, commands, and environment with the
results. If a case remains noisy after fixing setup, exclude it from automatic runs with the evidence
in the PR. Before restoring it, demonstrate repeatability and check that a deliberate increase in
measured work remains visible. Stable results are not useful if the measurement no longer responds
to a real regression. Do not raise the regression threshold to hide noise.

## Attribution

Adapted from the Vortex
[benchmarking guide](https://github.com/vortex-data/vortex/blob/develop/docs/developer-guide/benchmarking.md),
Copyright the Vortex contributors, licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Changes: kept the microbenchmark and
CodSpeed sections, removed the Vortex-specific SQL benchmark, allocator, walltime and CPU-feature
guidance, and replaced Vortex examples with FastLanes ones.
