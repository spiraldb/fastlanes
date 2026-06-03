//! Dev-only attribute macro for the bit-transpose benchmarks.
//!
//! `#[bench(<tier>)]`, placed above `#[divan::bench]`, expands to the
//! `#[cfg(target_feature = …)]` gate for one Intel feature tier. The gates are
//! mutually exclusive, so every benchmark is compiled — and therefore run — by
//! exactly one CI matrix entry, i.e. on exactly one runner.

use proc_macro::TokenStream;

/// Gate a benchmark on its Intel feature tier: `baseline` (no SIMD feature),
/// `bmi2`, or `avx512` (AVX-512 VBMI).
#[proc_macro_attribute]
pub fn bench(attr: TokenStream, item: TokenStream) -> TokenStream {
    let gate = match attr.to_string().trim() {
        "baseline" => r#"#[cfg(not(any(target_feature = "bmi2", target_feature = "avx512vbmi")))]"#,
        "bmi2" => r#"#[cfg(all(target_feature = "bmi2", not(target_feature = "avx512vbmi")))]"#,
        "avx512" => r#"#[cfg(target_feature = "avx512vbmi")]"#,
        other => {
            return format!(
                "compile_error!(\"#[bench(..)] expects `baseline`, `bmi2`, or `avx512`, got `{other}`\");"
            )
            .parse()
            .expect("compile_error! is valid tokens");
        }
    };

    // Prepend the cfg gate; the rest of the attributes (e.g. `#[divan::bench]`)
    // and the function itself are left untouched.
    let mut out: TokenStream = gate.parse().expect("cfg attribute is valid tokens");
    out.extend(item);
    out
}
