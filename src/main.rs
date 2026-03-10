mod cpu;
mod gpu;

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
    let gpu_result = gpu::run_gpu_benchmark(width as usize, height as usize, &image_data, warmup, iters);

    // Save CPU result
    let cpu_raw: Vec<u8> = cpu_result.iter().map(|&p| (p.clamp(0.0, 1.0) * 255.0) as u8).collect();
    let cpu_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, cpu_raw).expect("Failed to create CPU image buffer");
    cpu_img.save("cpu_output.png").expect("Failed to save CPU image");
    println!("Saved CPU output to cpu_output.png");

    // Save GPU result
    let gpu_raw: Vec<u8> = gpu_result.iter().map(|&p| (p.clamp(0.0, 1.0) * 255.0) as u8).collect();
    let gpu_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, gpu_raw).expect("Failed to create GPU image buffer");
    gpu_img.save("gpu_output.png").expect("Failed to save GPU image");
    println!("Saved GPU output to gpu_output.png");

    // Accuracy Validation
    let mut diff_sum = 0.0;
    let mut max_diff = 0.0f32;
    
    for (g_cpu, g_gpu) in cpu_result.iter().zip(gpu_result.iter()) {
        let diff = (g_cpu - g_gpu).abs();
        diff_sum += diff;
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    println!(
        "Validation: average diff: {:.6}, max diff: {:.6}", 
        diff_sum / num_pixels as f32, 
        max_diff
    );
}
