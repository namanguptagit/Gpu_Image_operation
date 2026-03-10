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

// Runs the GPU benchmark using the CubeCL wgpu/cuda backend
pub fn run_cubecl_benchmark(
    width: usize,
    height: usize,
    image_data: &[f32],
    warmup: u32,
    iters: u32,
) -> Vec<f32> {
    println!("Benchmarking GPU (CubeCL) ...");
    
    #[cfg(feature = "cuda")]
    type Runtime = cubecl::cuda::CudaRuntime;
    #[cfg(not(feature = "cuda"))]
    type Runtime = cubecl::wgpu::WgpuRuntime;
    
    let device = Default::default();
    let client = Runtime::client(&device);

    let num_pixels = width * height;

    let input_handle = client.create_from_slice(bytemuck::cast_slice(image_data));
    let out_handle = client.empty(num_pixels * std::mem::size_of::<f32>());

    let block_size = 16;
    let blocks_x = (width as u32 + block_size - 1) / block_size;
    let blocks_y = (height as u32 + block_size - 1) / block_size;

    let run_gpu = || {
        unsafe {
            let _ = gray_from_rgb_kernel::launch_unchecked::<f32, Runtime>(
                &client,
                CubeCount::Static(blocks_x, blocks_y, 1),
                CubeDim::new_2d(block_size, block_size),
                ArrayArg::from_raw_parts::<f32>(&input_handle, num_pixels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&out_handle, num_pixels, 1),
                width as u32,
                height as u32,
            );
        };
    };

    // Warmup
    for _ in 0..warmup {
        run_gpu();
    }

    let _ = client.read_one(out_handle.clone());

    // Benchmark
    let start = Instant::now();
    for _ in 0..iters {
        run_gpu();
    }
    
    let actual_bytes = client.read_one(out_handle.clone());
    let gpu_duration = start.elapsed() / iters;
    println!("GPU (CubeCL) duration: {:?}", gpu_duration);

    let actual: &[f32] = bytemuck::cast_slice(&actual_bytes);
    println!("Validation: Output size {}", actual.len());
    
    actual.to_vec()
}
