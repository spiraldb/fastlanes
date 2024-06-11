use seq_macro::seq;
use crate::{FastLanes, FL_ORDER};

pub trait Transpose: FastLanes {
    const MASK: [usize; 1024] = transpose_mask();
    const UNMASK: [usize; 1024] = untranspose_mask();

    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]);
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]);
}

impl<T> Transpose for T where T: FastLanes {
    #[inline(never)]
    fn transpose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        seq!(i in 0..1024 {
            output[i] = input[Self::MASK[i]];
        });
    }

    #[inline(never)]
    fn untranspose(input: &[Self; 1024], output: &mut [Self; 1024]) {
        seq!(i in 0..1024 {
            output[i] = input[Self::UNMASK[i]];
        });
    }
}

const fn transpose_mask() -> [usize; 1024] {
    let mut mask = [0; 1024];
    let mut mask_idx = 0;
    let mut row = 0;
    while row < 8 {
        let mut order = 0;
        while order < FL_ORDER.len() {
            let mut lane = 0;
            while lane < 16 {
                mask[mask_idx] = (lane * 64) + (FL_ORDER[order] as usize * 8) + row;
                mask_idx += 1;
                lane += 1;
            }
            order += 1;
        }
        row += 1;
    }
    mask
}

const fn untranspose_mask() -> [usize; 1024] {
    const MASK: [usize; 1024] = transpose_mask();
    let mut mask = [0; 1024];
    let mut mask_idx = 0;
    while mask_idx < 1024 {
        mask[mask_idx] = MASK[mask_idx];
        mask_idx += 1;
    }
    mask
}
