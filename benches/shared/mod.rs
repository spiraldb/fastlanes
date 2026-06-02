/// Prefix a benchmark's name with the active CPU/arch build variant.
///
/// Each `CodSpeed` matrix leg compiles with a different `target-feature` set
/// (e.g. `avx2`, `avx512`, `sve`) and exports a matching `BENCH_VARIANT`.
/// Prefixing the divan benchmark name with that variant turns the `CodSpeed`
/// URI into `benches/<file>.rs::<variant>::<name>`, so every variant appears
/// as its own entry under a per-variant group instead of overwriting the
/// others. Falls back to `local` (see `.cargo/config.toml`) for a plain
/// `cargo bench`.
#[macro_export]
macro_rules! variant {
    ($name:literal) => {
        concat!(env!("BENCH_VARIANT"), "::", $name)
    };
}

// Helper macro to conditionally add counter based on codspeed cfg
#[macro_export]
macro_rules! with_counter {
    ($bencher:expr, $bytes:expr) => {{
        #[cfg(not(codspeed))]
        let bencher = {
            use divan::counter::BytesCount;
            $bencher.counter(BytesCount::new($bytes))
        };
        #[cfg(codspeed)]
        let bencher = {
            let _ = $bytes; // Consume the bytes value to avoid unused variable warning
            $bencher
        };
        bencher
    }};
}
