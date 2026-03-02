mod cpu;
mod gpu;

// Main benchmarking entry point.
pub fn main() {
    let width = 1920;
    let height = 1080;
    let num_pixels = width * height;
    
    // Generate simulated image data
    let mut image_data = vec![0.0f32; num_pixels * 3];
    for i in 0..image_data.len() {
        image_data[i] = (i % 255) as f32 / 255.0;
    }

    let warmup = 10;
    let iters = 100;

    let cpu_result = cpu::run_cpu_benchmark(width, height, &image_data, warmup, iters);
    let gpu_result = gpu::run_gpu_benchmark(width, height, &image_data, warmup, iters);

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
