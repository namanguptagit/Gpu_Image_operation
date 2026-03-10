use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use std::time::Instant;

pub fn run_gpu_benchmark(
    width: usize,
    height: usize,
    image_data: &[f32],
    warmup: u32,
    iters: u32,
) -> Vec<f32> {
    println!("Benchmarking GPU (Native CUDA via cudarc API 0.18) ...");

    // Initialize Driver and Device context
    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA device context");
    let stream = ctx.default_stream();
    
    // Compile PTX at runtime using NVRTC
    let ptx_src = include_str!("kernel.cu");
    let ptx_compiled = cudarc::nvrtc::compile_ptx(ptx_src).expect("Failed to compile PTX at runtime");
    
    // Load module and function
    let module = ctx.load_module(ptx_compiled).expect("Failed to load module");
    let kernel = module.load_function("gray_from_rgb_kernel").expect("Failed to load function");
    
    let num_pixels = width * height;
    
    // Host -> Device Data Transfers
    let input_dev = stream.clone_htod(image_data).expect("Failed to copy input data to GPU");
    let mut output_dev = stream.alloc_zeros::<f32>(num_pixels).expect("Failed to allocate output buffer");
    
    // Grid and Block dimensions
    let block_size = 16;
    let blocks_x = (width as u32 + block_size - 1) / block_size;
    let blocks_y = (height as u32 + block_size - 1) / block_size;
    
    let cfg = LaunchConfig {
        grid_dim: (blocks_x, blocks_y, 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };
    
    let w = width as u32;
    let h = height as u32;
    
    let mut run_gpu = || {
        let mut launch_args = stream.launch_builder(&kernel);
        launch_args.arg(&input_dev);
        launch_args.arg(&mut output_dev);
        launch_args.arg(&w);
        launch_args.arg(&h);
        unsafe { launch_args.launch(cfg) }.expect("Kernel launch failed");
    };
    
    // Warmup
    for _ in 0..warmup {
        run_gpu();
    }
    ctx.synchronize().unwrap();
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..iters {
        run_gpu();
    }
    ctx.synchronize().unwrap();
    
    let gpu_duration = start.elapsed() / iters;
    println!("GPU (Native CUDA) duration: {:?}", gpu_duration);
    
    // Device -> Host Transfer
    let actual = stream.clone_dtoh(&output_dev).expect("Failed to copy data back to Host");
    println!("Validation: Output size {}", actual.len());
    
    actual
}
