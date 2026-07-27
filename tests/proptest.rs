// https://github.com/rust-lang/rust-clippy/issues/11024
#![allow(clippy::tests_outside_test_module)]

use fastlanes::BitPacking;
use hegel::TestCase;
use hegel::generators as gs;
use hegel::generators::Integer;
use std::fmt::Debug;

const BUFFER_SIZE: usize = 1024;

fn assert_bitpack_roundtrip<T>(tc: &TestCase)
where
    T: BitPacking + Debug + Integer + 'static,
{
    let width = tc.draw(gs::integers::<usize>().min_value(0).max_value(T::T));

    let max_value = if width == T::T {
        T::max_value()
    } else {
        (T::one() << width) - T::one()
    };
    let input = tc.draw(
        gs::vecs(
            gs::integers::<T>()
                .min_value(T::zero())
                .max_value(max_value),
        )
        .min_size(BUFFER_SIZE)
        .max_size(BUFFER_SIZE),
    );

    // We use dirty buffers here, to make sure its not a hidden invariant
    let mut packed_output = vec![T::one(); (BUFFER_SIZE * width) / T::T];
    let mut unpacked_output = vec![T::one(); BUFFER_SIZE];
    unsafe { BitPacking::unchecked_pack(width, &input, &mut packed_output) };
    unsafe {
        BitPacking::unchecked_unpack(width, &packed_output, &mut unpacked_output);
    }

    let mask = if width == 0 {
        T::zero()
    } else if width == T::T {
        T::max_value()
    } else {
        (T::one() << width) - T::one()
    };

    let expected = input.iter().copied().map(|v| v & mask).collect::<Vec<_>>();

    assert_eq!(expected, unpacked_output);

    for index in 0..BUFFER_SIZE {
        assert_eq!(
            unsafe { T::unchecked_unpack_single(width, &packed_output, index) },
            expected[index],
        );
    }
}

fn assert_bitpack_repack_roundtrip<T>(tc: &TestCase)
where
    T: BitPacking + Debug + Integer + 'static,
{
    let width = tc.draw(gs::integers::<usize>().min_value(0).max_value(T::T));
    let packed_length = (BUFFER_SIZE * width) / T::T;
    let packed_input = tc.draw(
        gs::vecs(gs::integers::<T>())
            .min_size(packed_length)
            .max_size(packed_length),
    );

    let mut unpacked_output = vec![T::one(); BUFFER_SIZE];
    let mut repacked_output = vec![T::one(); packed_length];
    unsafe {
        BitPacking::unchecked_unpack(width, &packed_input, &mut unpacked_output);
        BitPacking::unchecked_pack(width, &unpacked_output, &mut repacked_output);
    }

    assert_eq!(packed_input, repacked_output);
}

macro_rules! bitpack_property_test {
    ($name:ident, $type:ty, $property:ident) => {
        #[hegel::test(test_cases = 1000)]
        fn $name(tc: TestCase) {
            $property::<$type>(&tc);
        }
    };
}

bitpack_property_test!(test_bitpack_roundtrip_u8, u8, assert_bitpack_roundtrip);
bitpack_property_test!(test_bitpack_roundtrip_u16, u16, assert_bitpack_roundtrip);
bitpack_property_test!(test_bitpack_roundtrip_u32, u32, assert_bitpack_roundtrip);
bitpack_property_test!(test_bitpack_roundtrip_u64, u64, assert_bitpack_roundtrip);

bitpack_property_test!(
    test_bitpack_repack_roundtrip_u8,
    u8,
    assert_bitpack_repack_roundtrip
);
bitpack_property_test!(
    test_bitpack_repack_roundtrip_u16,
    u16,
    assert_bitpack_repack_roundtrip
);
bitpack_property_test!(
    test_bitpack_repack_roundtrip_u32,
    u32,
    assert_bitpack_repack_roundtrip
);
bitpack_property_test!(
    test_bitpack_repack_roundtrip_u64,
    u64,
    assert_bitpack_repack_roundtrip
);
