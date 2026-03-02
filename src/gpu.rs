use cubecl::prelude::*;
use std::time::Instant;

// CubeCL GPU kernel for RGB to Grayscale conversion
#[cube(launch_unchecked)]
pub fn gray_from_rgb_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x < width && y < height {
        let input_idx = (y * width + x) * 3;
        let output_idx = y * width + x;

        let r = input[input_idx as usize];
        let g = input[(input_idx + 1) as usize];
        let b = input[(input_idx + 2) as usize];

        let rw = F::cast_from(0.299f32);
        let gw = F::cast_from(0.587f32);
        let bw = F::cast_from(0.114f32);

        output[output_idx as usize] = r * rw + g * gw + b * bw;
    }
}

