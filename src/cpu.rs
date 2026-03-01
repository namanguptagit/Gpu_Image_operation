use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::color::gray_from_rgb;
use std::time::Instant;

// Runs the CPU benchmark using kornia-rs Rayon iteration.
pub fn run_cpu_benchmark(
    width: usize,
    height: usize,
    image_data: &[f32],
    warmup: u32,
    iters: u32,
) -> Vec<f32> {
    println!("Benchmarking CPU (kornia-rs) ...");

    let kornia_img = Image::<f32, 3, CpuAllocator>::new(
        ImageSize { width, height },
        image_data.to_vec(),
        CpuAllocator,
    ).unwrap();
    
    let mut kornia_gray = Image::<f32, 1, CpuAllocator>::from_size_val(
        ImageSize { width, height },
        0.0,
        CpuAllocator,
    ).unwrap();

    // Warmup
    for _ in 0..warmup {
        gray_from_rgb(&kornia_img, &mut kornia_gray).unwrap();
    }
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..iters {
        gray_from_rgb(&kornia_img, &mut kornia_gray).unwrap();
    }
    
    let cpu_duration = start.elapsed() / iters;
    println!("CPU (kornia-rs) duration: {:?}", cpu_duration);
    
    kornia_gray.as_slice().to_vec()
}
