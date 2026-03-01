mod cpu;

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
}
