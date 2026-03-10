mod cpu;
mod cubecl_gpu;
mod native_cuda;

use image::{ImageBuffer, Luma};

// Main benchmarking entry point.
pub fn main() {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| "input.png".to_string());
    println!("Loading image from: {}", img_path);
    
    let img = image::open(&img_path).expect("Failed to open image");
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let num_pixels = (width * height) as usize;

    let image_data: Vec<f32> = rgb_img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();

    let warmup = 10;
    let iters = 100;

    let cpu_result = cpu::run_cpu_benchmark(width as usize, height as usize, &image_data, warmup, iters);
    let cubecl_result = cubecl_gpu::run_cubecl_benchmark(width as usize, height as usize, &image_data, warmup, iters);
    let native_result = native_cuda::run_gpu_benchmark(width as usize, height as usize, &image_data, warmup, iters);

    // Save CPU result
    let cpu_raw: Vec<u8> = cpu_result.iter().map(|&p| (p.clamp(0.0, 1.0) * 255.0) as u8).collect();
    let cpu_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, cpu_raw).expect("Failed to create CPU image buffer");
    cpu_img.save("cpu_output.png").expect("Failed to save CPU image");
    println!("Saved CPU output to cpu_output.png");

    // Save CubeCL GPU result
    let cubecl_raw: Vec<u8> = cubecl_result.iter().map(|&p| (p.clamp(0.0, 1.0) * 255.0) as u8).collect();
    let cubecl_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, cubecl_raw).expect("Failed to create GPU image buffer");
    cubecl_img.save("cubecl_output.png").expect("Failed to save CubeCL image");
    println!("Saved CubeCL output to cubecl_output.png");

    // Save Native CUDA result
    let native_raw: Vec<u8> = native_result.iter().map(|&p| (p.clamp(0.0, 1.0) * 255.0) as u8).collect();
    let native_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, native_raw).expect("Failed to create GPU image buffer");
    native_img.save("native_cuda_output.png").expect("Failed to save Native CUDA image");
    println!("Saved Native CUDA output to native_cuda_output.png");

    // Accuracy Validation
    let validate = |name: &str, gpu_res: &[f32]| {
        let mut diff_sum = 0.0;
        let mut max_diff = 0.0f32;
        
        for (g_cpu, g_gpu) in cpu_result.iter().zip(gpu_res.iter()) {
            let diff = (g_cpu - g_gpu).abs();
            diff_sum += diff;
            if diff > max_diff {
                max_diff = diff;
            }
        }
        
        println!(
            "{name} Validation: average diff: {:.6}, max diff: {:.6}", 
            diff_sum / num_pixels as f32, 
            max_diff
        );
    };

    validate("CubeCL", &cubecl_result);
    validate("Native CUDA", &native_result);
}
