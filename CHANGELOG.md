# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.2](https://github.com/spiraldb/fastlanes/compare/v0.7.1...v0.7.2) - 2026-09-02

### Other

- Assertions in functions are more optimisation friendly ([#195](https://github.com/spiraldb/fastlanes/pull/195))

## [0.7.1](https://github.com/spiraldb/fastlanes/compare/v0.7.0...v0.7.1) - 2026-09-01

### Added

- *(bitpacking)* Add batched index unpacking ([#190](https://github.com/spiraldb/fastlanes/pull/190))

### Other

- *(bitpacking)* more representative unpack_indices benchmarks ([#194](https://github.com/spiraldb/fastlanes/pull/194))

## [0.7.0](https://github.com/spiraldb/fastlanes/compare/v0.6.1...v0.7.0) - 2026-08-26

This release has the same contents as 0.6.2. Version 0.6.2 is yanked, because
it has breaking changes that a patch release must not have:

- The `std` feature is replaced by the `runtime` feature ([#175](https://github.com/spiraldb/fastlanes/pull/175))
- `Transpose::transpose`/`untranspose` use the same element order as
  `bit_transpose`/`bit_untranspose` ([#186](https://github.com/spiraldb/fastlanes/pull/186))

## [0.6.2](https://github.com/spiraldb/fastlanes/compare/v0.6.1...v0.6.2) - 2026-08-25

### Added

- Replace `std` feature with a `runtime` feature so its always `no_std` ([#175](https://github.com/spiraldb/fastlanes/pull/175))

### Other

- :transpose/untranspose and bit_transpose/untranspose have the same ordering ([#186](https://github.com/spiraldb/fastlanes/pull/186))
- Lock file maintenance ([#185](https://github.com/spiraldb/fastlanes/pull/185))
- Update Rust crate hegeltest to 0.31.0 ([#184](https://github.com/spiraldb/fastlanes/pull/184))
- Extend prop test coverage, and move them into the relevant files ([#174](https://github.com/spiraldb/fastlanes/pull/174))

## [0.6.1](https://github.com/spiraldb/fastlanes/compare/v0.6.0...v0.6.1) - 2026-07-27

### Fixed

- fused unpack_cmp panicks at full width ([#172](https://github.com/spiraldb/fastlanes/pull/172))

### Other

- Replace round-trip tests with hegel-based property testing ([#171](https://github.com/spiraldb/fastlanes/pull/171))
- Clean up dependencies and use const_for! in one more place ([#153](https://github.com/spiraldb/fastlanes/pull/153))
- Update taiki-e/install-action digest to 41049aa ([#169](https://github.com/spiraldb/fastlanes/pull/169))
- Update MarcoIeni/release-plz-action digest to 2eb1d8b ([#166](https://github.com/spiraldb/fastlanes/pull/166))
- Update Rust crate divan to v5 ([#167](https://github.com/spiraldb/fastlanes/pull/167))

## [0.6.0](https://github.com/spiraldb/fastlanes/compare/v0.5.2...v0.6.0) - 2026-07-23

### Other

- Update taiki-e/install-action digest to c44f6b0 ([#161](https://github.com/spiraldb/fastlanes/pull/161))
- Update dtolnay/rust-toolchain digest ([#160](https://github.com/spiraldb/fastlanes/pull/160))
- Update CodSpeedHQ/action digest to f99becd ([#159](https://github.com/spiraldb/fastlanes/pull/159))
- Update actions/checkout action to v7 ([#162](https://github.com/spiraldb/fastlanes/pull/162))
- Mark RLE encode/decode as unsafe due to index bounds requirements ([#163](https://github.com/spiraldb/fastlanes/pull/163))

## [0.5.2](https://github.com/spiraldb/fastlanes/compare/v0.5.1...v0.5.2) - 2026-06-12

### Other

- Remove array ref to reduce code size ([#151](https://github.com/spiraldb/fastlanes/pull/151))

## [0.5.1](https://github.com/spiraldb/fastlanes/compare/v0.5.0...v0.5.1) - 2026-06-04

### Added

- Add width-generic x86 BMI2/VBMI untranspose for u8/u16/u32 ([#145](https://github.com/spiraldb/fastlanes/pull/145))
- Fused bitpacking compare into a 1024-bit mask (+ SIMD bit-untranspose) ([#141](https://github.com/spiraldb/fastlanes/pull/141))
- Add FastLanes 1024-bit transpose with SIMD implementations ([#142](https://github.com/spiraldb/fastlanes/pull/142))

### Other

- *(deps)* update taiki-e/install-action digest to 25435dc ([#138](https://github.com/spiraldb/fastlanes/pull/138))
- clean up publish ([#139](https://github.com/spiraldb/fastlanes/pull/139))
- *(deps)* update marcoieni/release-plz-action digest to 064f4d1 ([#136](https://github.com/spiraldb/fastlanes/pull/136))
- *(deps)* update taiki-e/install-action digest to b550161 ([#135](https://github.com/spiraldb/fastlanes/pull/135))
- *(deps)* update codspeedhq/action digest to 3194d9a ([#134](https://github.com/spiraldb/fastlanes/pull/134))
- *(deps)* pin taiki-e/install-action action to 1329c29 ([#132](https://github.com/spiraldb/fastlanes/pull/132))
- *(deps)* update mozilla-actions/sccache-action action to v0.0.10 ([#133](https://github.com/spiraldb/fastlanes/pull/133))
- Reduce copies in benchmarks ([#131](https://github.com/spiraldb/fastlanes/pull/131))
- Bump toolchain to stable and MSRV to 1.91 while verifying MSRV respected ([#130](https://github.com/spiraldb/fastlanes/pull/130))
- Benchmark fused comparison for all widths ([#129](https://github.com/spiraldb/fastlanes/pull/129))
- Remove unused const generics benchmarks ([#128](https://github.com/spiraldb/fastlanes/pull/128))
- *(deps)* update codspeedhq/action digest to db35df7 ([#127](https://github.com/spiraldb/fastlanes/pull/127))
- *(deps)* pin dependencies ([#126](https://github.com/spiraldb/fastlanes/pull/126))
- Move memory allocation out of RLE benchmarks ([#123](https://github.com/spiraldb/fastlanes/pull/123))
- *(deps)* update codspeedhq/action digest to 1c8ae48 ([#122](https://github.com/spiraldb/fastlanes/pull/122))
- *(deps)* update swatinem/rust-cache digest to e18b497 ([#121](https://github.com/spiraldb/fastlanes/pull/121))
- *(deps)* update codspeedhq/action digest to 281164b ([#118](https://github.com/spiraldb/fastlanes/pull/118))
- *(deps)* update marcoieni/release-plz-action digest to 1528104 ([#119](https://github.com/spiraldb/fastlanes/pull/119))
- *(deps)* update marcoieni/release-plz-action digest to f708778 ([#115](https://github.com/spiraldb/fastlanes/pull/115))
- *(deps)* update actions/checkout digest to de0fac2 ([#114](https://github.com/spiraldb/fastlanes/pull/114))
- *(deps)* update codspeedhq/action digest to 4deb327 ([#113](https://github.com/spiraldb/fastlanes/pull/113))
- *(deps)* update codspeedhq/action digest to e736f0d ([#111](https://github.com/spiraldb/fastlanes/pull/111))
- *(deps)* update actions/checkout action to v6 ([#109](https://github.com/spiraldb/fastlanes/pull/109))
- *(deps)* update swatinem/rust-cache digest to 779680d ([#108](https://github.com/spiraldb/fastlanes/pull/108))
- *(deps)* update marcoieni/release-plz-action digest to e592230 ([#107](https://github.com/spiraldb/fastlanes/pull/107))
- *(deps)* update codspeedhq/action digest to 346a2d8 ([#105](https://github.com/spiraldb/fastlanes/pull/105))
- *(deps)* update actions/checkout digest to 93cb6ef ([#104](https://github.com/spiraldb/fastlanes/pull/104))
- *(deps)* update codspeedhq/action digest to daf3e64 ([#103](https://github.com/spiraldb/fastlanes/pull/103))
- clear up safety documentation ([#102](https://github.com/spiraldb/fastlanes/pull/102))
- *(deps)* update codspeedhq/action digest to bb005fe ([#101](https://github.com/spiraldb/fastlanes/pull/101))
- *(deps)* update codspeedhq/action digest to c6574d0 ([#99](https://github.com/spiraldb/fastlanes/pull/99))

## [0.5.0](https://github.com/spiraldb/fastlanes/compare/v0.4.0...v0.5.0) - 2025-10-16

### Added

- unchecked for bp ([#97](https://github.com/spiraldb/fastlanes/pull/97))

### Fixed

- fixup FoR doc str and bench ([#98](https://github.com/spiraldb/fastlanes/pull/98))

### Other

- *(deps)* update marcoieni/release-plz-action digest to d529f73 ([#96](https://github.com/spiraldb/fastlanes/pull/96))
- *(deps)* update codspeedhq/action digest to 7a5b8b0 ([#95](https://github.com/spiraldb/fastlanes/pull/95))
- *(deps)* update codspeedhq/action digest to 3959e9e ([#93](https://github.com/spiraldb/fastlanes/pull/93))
- *(deps)* update rust crate divan to v4 ([#90](https://github.com/spiraldb/fastlanes/pull/90))

## [0.4.0](https://github.com/spiraldb/fastlanes/compare/v0.3.0...v0.4.0) - 2025-10-03

### Added

- make rle index type generic for decode ([#91](https://github.com/spiraldb/fastlanes/pull/91))

## [0.3.0](https://github.com/spiraldb/fastlanes/compare/v0.2.2...v0.3.0) - 2025-09-30

### Fixed

- increase sample count to stabilize benchmarks ([#82](https://github.com/spiraldb/fastlanes/pull/82))
- black box input and output parameters ([#79](https://github.com/spiraldb/fastlanes/pull/79))

### Other

- Relax bounds on RLE compression ([#89](https://github.com/spiraldb/fastlanes/pull/89))
- Simplify unpack mask to match the pack mask ([#88](https://github.com/spiraldb/fastlanes/pull/88))
- Use c6.8xlarge for codspeed benchmarks ([#87](https://github.com/spiraldb/fastlanes/pull/87))
- Remove unsafe example from readme ([#86](https://github.com/spiraldb/fastlanes/pull/86))
- switch to metal instance for ci benchmarks ([#84](https://github.com/spiraldb/fastlanes/pull/84))
- move all benchmarks to divan ([#83](https://github.com/spiraldb/fastlanes/pull/83))
- Narrow down the scope of paste macro ([#77](https://github.com/spiraldb/fastlanes/pull/77))
- *(deps)* update marcoieni/release-plz-action digest to acb9246 ([#74](https://github.com/spiraldb/fastlanes/pull/74))

## [0.2.2](https://github.com/spiraldb/fastlanes/compare/v0.2.1...v0.2.2) - 2025-09-15

### Other

- use `get_unchecked` for index accesses in rle ([#76](https://github.com/spiraldb/fastlanes/pull/76))
- improve fls-rle comments ([#75](https://github.com/spiraldb/fastlanes/pull/75))
- *(deps)* update codspeedhq/action digest to 653fdc3 ([#73](https://github.com/spiraldb/fastlanes/pull/73))
- *(deps)* pin dependencies ([#67](https://github.com/spiraldb/fastlanes/pull/67))
- *(deps)* update codspeedhq/action action to v4 ([#69](https://github.com/spiraldb/fastlanes/pull/69))
- *(deps)* update actions/checkout action to v5 ([#68](https://github.com/spiraldb/fastlanes/pull/68))

## [0.2.1](https://github.com/spiraldb/fastlanes/compare/v0.2.0...v0.2.1) - 2025-09-05

### Added

- fastlanes rle ([#70](https://github.com/spiraldb/fastlanes/pull/70))

## [0.2.0](https://github.com/spiraldb/fastlanes/compare/v0.1.8...v0.2.0) - 2025-07-15

### Added

- Use stable Rust edition 2021 ([#65](https://github.com/spiraldb/fastlanes/pull/65))

### Other

- *(deps)* update mozilla-actions/sccache-action action to v0.0.9 ([#62](https://github.com/spiraldb/fastlanes/pull/62))
- Never inline kernel impls to improve compile times ([#61](https://github.com/spiraldb/fastlanes/pull/61))
- Compare into byte bool ([#60](https://github.com/spiraldb/fastlanes/pull/60))
- Bump toolchain & edition ([#59](https://github.com/spiraldb/fastlanes/pull/59))
- Setup Codspeed  ([#58](https://github.com/spiraldb/fastlanes/pull/58))
- bump rust-toolchain ([#54](https://github.com/spiraldb/fastlanes/pull/54))
- *(deps)* update mozilla-actions/sccache-action action to v0.0.7 ([#53](https://github.com/spiraldb/fastlanes/pull/53))
- add throughput benchmarks ([#51](https://github.com/spiraldb/fastlanes/pull/51))
- *(deps)* update mozilla-actions/sccache-action action to v0.0.6 ([#49](https://github.com/spiraldb/fastlanes/pull/49))

## [0.1.8](https://github.com/spiraldb/fastlanes/compare/v0.1.7...v0.1.8) - 2024-09-20

### Fixed

- fix the readme example and run it as a doctest ([#47](https://github.com/spiraldb/fastlanes/pull/47))

## [0.1.7](https://github.com/spiraldb/fastlanes/compare/v0.1.6...v0.1.7) - 2024-07-19

### Other
- Fast unpack_single ([#43](https://github.com/spiraldb/fastlanes/pull/43))

## [0.1.6](https://github.com/spiraldb/fastlanes/compare/v0.1.5...v0.1.6) - 2024-07-03

### Other
- reimplement Bitpacking::unpack_single ([#42](https://github.com/spiraldb/fastlanes/pull/42))
- clippy works now, and is pedantic ([#38](https://github.com/spiraldb/fastlanes/pull/38))
- aggressive clippy ([#37](https://github.com/spiraldb/fastlanes/pull/37))
- pin nightly-2024-06-19 ([#36](https://github.com/spiraldb/fastlanes/pull/36))
- Apache license ([#34](https://github.com/spiraldb/fastlanes/pull/34))

## [0.1.5](https://github.com/spiraldb/fastlanes/compare/v0.1.4...v0.1.5) - 2024-06-17

### Other
- Fix fused delta ([#31](https://github.com/spiraldb/fastlanes/pull/31))

## [0.1.4](https://github.com/spiraldb/fastlanes/compare/v0.1.3...v0.1.4) - 2024-06-17

### Other
- Unchecked unpack single ([#29](https://github.com/spiraldb/fastlanes/pull/29))

## [0.1.3](https://github.com/spiraldb/fastlanes/compare/v0.1.2...v0.1.3) - 2024-06-17

### Other
- *(deps)* update mozilla-actions/sccache-action action to v0.0.5 ([#28](https://github.com/spiraldb/fastlanes/pull/28))
- Bitunpack Single ([#27](https://github.com/spiraldb/fastlanes/pull/27))
- Unchecked bitpack ([#25](https://github.com/spiraldb/fastlanes/pull/25))

## [0.1.2](https://github.com/spiraldb/fastlanes/compare/v0.1.1...v0.1.2) - 2024-06-16

### Other
- Fused Delta + FoR ([#22](https://github.com/spiraldb/fastlanes/pull/22))

## [0.1.1](https://github.com/spiraldb/fastlanes-rs/compare/v0.1.0...v0.1.1) - 2024-06-14

### Other
- Delta Encoding ([#11](https://github.com/spiraldb/fastlanes-rs/pull/11))
- Transpose Masks ([#8](https://github.com/spiraldb/fastlanes-rs/pull/8))
- Remove old workflow ([#7](https://github.com/spiraldb/fastlanes-rs/pull/7))
