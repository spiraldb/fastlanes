# FastLanes Rust

A Rust implementation of the [FastLanes](https://github.com/cwida/FastLanes) compression library

> Azim Afroozeh and Peter Boncz. 2023. The FastLanes Compression Layout: Decoding > 100 Billion Integers per Second with
> Scalar Code.
> Proc. VLDB Endow. 16, 9 (May 2023), 2132–2144. https://doi.org/10.14778/3598581.3598587

FastLanes is a compression framework that can leverage LLVM's auto-vectorization to achieve high-performance
SIMD decoding without intrinsics or other explicit SIMD code.

## Usage

```rust
use fastlanes::BitPacking;

fn pack_u16_into_u3() {
    const WIDTH: usize = 3;

    // Generate some values.
    let mut values: [u16; 1024] = [0; 1024];
    for i in 0..1024 {
        values[i] = (i % (1 << WIDTH)) as u16;
    }

    // Pack the values.
    let values = [3u16; 1024];
    let mut packed = [0; 128 * WIDTH / size_of::<u16>()];
    BitPacking::bitpack::<WIDTH>(&values, &mut packed);

    // Unpack the values.
    let mut unpacked = [0u16; 1024];
    BitPacking::bitunpack::<WIDTH>(&packed, &mut unpacked);
    assert_eq!(values, unpacked);

    // Unpack a single value at index 14.
    // Note that for more than ~10 values, it can be faster to unpack all values and then 
    // access the desired one.
    assert_eq!(BitPacking::bitunpack_single::<WIDTH>(&packed, 14), 14);
}
```

## Differences to original FastLanes

> [!CAUTION]
> Rust FastLanes is not binary compatible with original FastLanes

The BitPacking implementation in this library is reordered vs the original to enable
fused kernels for transposed encodings (like Delta and RLE) in addition to the linear
kernels such as FoR.

## Verifying ASM

To validate the correctness of the generated assembly and ensure it is vectorized, you can use the following command:

```bash
RUSTFLAGS='-C target-cpu=native' cargo asm --profile release --bench bitpacking --rust BitPacking
```

Note, it requires `cargo install cargo-show-asm`.

## Benchmarking

```bash
RUSTFLAGS='-C target-cpu=native' cargo bench --profile release
```

## License

Licensed under the Apache 2.0 license.