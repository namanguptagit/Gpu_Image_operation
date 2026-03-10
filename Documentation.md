# Project Architecture & In-Depth Documentation

The project execution environment for our GPU-accelerated RGB to Grayscale operation has been modularized into three core files: `src/main.rs`, `src/cpu.rs`, and `src/gpu.rs`.

---

## 1. The GPU Kernel (`src/gpu.rs`)

The `gpu.rs` file declares the `gray_from_rgb_kernel` function. This is pure Rust code transformed at compile-time by the `#[cube(launch_unchecked)]` macro into an AST that CubeCL can compile down to WGSL/SPIR-V for the GPU.

```rust
#[cube(launch_unchecked)]
pub fn gray_from_rgb_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] width: u32,
    #[comptime] height: u32,
)
```

### Argument Layout
- **`input`**: A flat 1D array on the GPU containing all color values. An $1920 \times 1080$ RGB image has $1920 \times 1080 \times 3 = 6,220,800$ floating point elements. It is formatted interleaved, i.e., `[R, G, B, R, G, B, ... ]`.
- **`output`**: A flat 1D array to hold the grayscale output. Size is $1920 \times 1080 = 2,073,600$ elements.
- **`#[comptime] width / height`**: These arguments are marked `#[comptime]`. This allows the CubeCL compiler engine to statically evaluate bounds checking branches within the compute shader resulting in a much faster kernel by avoiding branch prediction overhead on the hardware.

### Thread Geometry and Coordinate Mapping
```rust
let x = ABSOLUTE_POS_X;
let y = ABSOLUTE_POS_Y;

if x < width && y < height { ... }
```
When this kernel launches, the GPU fires up millions of threads grouped into *Compute Blocks* (e.g. $16 \times 16$ blocks). 
- `ABSOLUTE_POS_X/Y` represents the unique 2D coordinate of the *currently executing thread* within the entire image grid. 
- Because GPU Blocks must be uniform sizes (e.g. 16), a $1080p$ height ($1080/16 = 67.5$) actually spawns 68 blocks vertically. The `if x < width && y < height` check ensures threads spawned in the "padding" area outside the actual image border exit cleanly without causing an out-of-bounds memory crash.

### Data Fetching and Math
Because the `input` array is interleaved:
```rust
let input_idx = (y * width + x) * 3;
let output_idx = y * width + x;
```
It computes the 1D base offset by navigating down `y` rows and across `x` columns. It multiplies by `3` for the input to skip over exactly that many RGB clusters.

The math utilizes the standard Rec. 601 luma encoding constants:
```rust
let rw = F::cast_from(0.299f32);
let gw = F::cast_from(0.587f32);
let bw = F::cast_from(0.114f32);
output[output_idx as usize] = r * rw + g * gw + b * bw;
```
The kernel finishes when the dot-product is pushed into the `output` array at the target thread's pixel offset.

---

## 2. Main Entry Point (`src/main.rs`)

The main function acts as the host-level controller, mocking the fake image memory and dispatching the benchmarks. 

### Data Initialization
```rust
let img = image::open(&img_path).expect("Failed to open image");
let rgb_img = img.to_rgb8();
...
let image_data: Vec<f32> = rgb_img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
```
It loads a real RGB image (e.g., `input.png`) using the `image` crate, preventing hardcoded mock data. This allows us to process actual user inputs and save the resulting pixel arrays back to disk as actual `.png` images for visual verification. It converts the standard `u8` bytes into the `f32` vectors expected by the `kornia-rs` CPU functions and our `gpu.rs` CubeCL kernels.

### CPU Benchmark Execution (`src/cpu.rs`)
```rust
for _ in 0..warmup {
    gray_from_rgb(&kornia_img, &mut kornia_gray).unwrap();
}
```
In `run_cpu_benchmark`, it begins by "warming up" the CPU. Often, the first time you invoke a library on a system it has to wake up threads from OS sleep, move data into L2/L3 hardware caches, and allocate thread pools (Rayon does this). We throw away the first 10 cycles, then time the remaining 100 perfectly hot executions.

### GPU Benchmark Setup (`src/gpu.rs`)

```rust
type Runtime = cubecl::wgpu::WgpuRuntime;
let device = Default::default();
let client = Runtime::client(&device);
```
Initializes the **WebGPU** compute backend for the CubeCL compiler. It attaches to your computer's default graphics device. 

```rust
let input_handle = client.create_from_slice(bytemuck::cast_slice(&image_data));
let out_handle = client.empty(num_pixels * std::mem::size_of::<f32>());
```
Raw bytes from CPU RAM (`image_data`) are uploaded securely via `bytemuck` across the PCIe bus and into the GPU's isolated VRAM as `input_handle`. `out_handle` is initialized on the VRAM empty.

```rust
let block_size = 16;
let blocks_x = (width as u32 + block_size - 1) / block_size;
```
It defines the compute dispatch topology. `16x16` is 256 threads per block, a highly compatible execution size for AMD and NVIDIA Warp Schedulers. We apply ceiling division to compute the correct number of blocks `x` and `y`.

### GPU Kernel Launch
```rust
gray_from_rgb_kernel::launch_unchecked::<f32, Runtime>(
    &client,
    CubeCount::Static(blocks_x, blocks_y, 1),
    CubeDim::new_2d(block_size, block_size),
    ArrayArg::from_raw_parts::<f32>(&input_handle, num_pixels * 3, 1),
    ArrayArg::from_raw_parts::<f32>(&out_handle, num_pixels, 1),
    width as u32,
    height as u32,
)
```
Tells the driver to invoke our parsed CubeCL shader. The `ArrayArg::from_raw_parts` commands map our strongly typed `f32` shader inputs to the raw VRAM memory addresses we allocated previously. 

### Synchronization & Validation
We measure the `start.elapsed()` precisely, but because GPU drivers execute *asynchronously* (pushing instructions into an endless buffer without waiting), we have to explicitly force synchronization in Rust:
```rust
let _ = client.read_one(out_handle.clone());
```
Any `read_one` commands stall the CPU execution thread entirely until the GPU signals it has flushed all pending blocks. We use this to capture the exact cycle timestamps for benchmarking execution within `run_gpu_benchmark`. 

Finally, `main.rs` loops across the CPU result and the downloaded GPU array `actual_bytes` and mathematically subtracts them tracking the variance to `0.0000000` to validate algorithm purity.
