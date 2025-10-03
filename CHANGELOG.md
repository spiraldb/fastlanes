# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
