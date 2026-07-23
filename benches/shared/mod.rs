/// A `T` pinned to 128-byte alignment, one full 1024-bit `FastLanes` vector.
///
/// Bench buffers declared as plain stack arrays inherit whatever alignment the
/// enclosing frame happens to give them, so unrelated code changes can move
/// them relative to cache-line and SIMD-lane boundaries and show up as phantom
/// performance changes. Wrapping the buffer pins its alignment across builds.
#[repr(C, align(128))]
pub struct Aligned<T>(pub T);

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
