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
