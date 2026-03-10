# CubeCL vs Native CUDA Image Operation Benchmark

This project demonstrates a single GPU-accelerated image operation (RGB to Grayscale conversion) in Rust using both [CubeCL](https://github.com/tracel-ai/cubecl) and raw Native CUDA C++ (via `cudarc`), benchmarked against the CPU implementation from [kornia-rs](https://github.com/kornia/kornia-rs).

It fulfills the objective of creating a simple, done-properly GPU operation in Rust and measuring high-level compute frameworks against bare-metal GPU dispatch.

## Implementation Details

- **CPU Baseline:** Uses `imgproc::color::gray_from_rgb` from `kornia-imgproc`.
- **GPU (CubeCL):** Uses `#[cube(launch_unchecked)]` to compile a WGPU or CUDA shader directly from Rust.
- **GPU (Native CUDA):** Uses a dynamically runtime compiled (`nvrtc`) `.cu` C++ CUDA kernel dispatched via Driver API wrapper `cudarc`.
- **Validation:** Ensures that CPU, CubeCL, and Native CUDA backends generate the exact identical output tensors up to precision limits.

## How to Run

1. Ensure you have Rust and Cargo installed.
2. Clone the repository (the workspace assumes the `kornia-rs` clone is available locally for reference):
   ```bash
   git clone https://github.com/kornia/kornia-rs.git reference_kornia_rs
   ```
3. Run the benchmark in release mode:
   ```bash
   cargo run --release
   ```

## Benchmark Results

### Apple Silicon (Mac)
Testing on a real RGB image (`input.png`):

| Implementation       | Avg Duration per Run (100 iters) | Note |
|----------------------|----------------------------------|------|
| **CPU (kornia-rs)**  | ~2.21 ms                         | Rayon multi-threaded CPU iteration |
| **GPU (CubeCL)**     | ~2.41 ms                         | WGPU backend via CubeCL (includes dispatch/sync overhead) |

### NVIDIA GPU (Windows PC)
Testing on a real RGB image (`input.png`):

| Implementation | Avg Duration per Run (100 iters) | Note |
|---|---|---|
| **CPU (kornia-rs)** | ~7.01 ms | Rayon multi-threaded CPU iteration |
| **GPU (CubeCL)** | ~1.23 ms | WGPU/CUDA backend via CubeCL (includes dispatch/sync overhead) |
| **GPU (Native CUDA)** | ~1.20 ms | Bare-metal driver dispatch via `cudarc` and `nvrtc` |

*The zero difference output during execution validates mathematical accuracy.*

### Output Validation

The pipeline outputs the mathematically verified results back to `cpu_output.png` and `gpu_output.png`.

**Original Image (`input.png`):**

![Input Image](input.png)

**Grayscale Output (`gpu_output.png`):**

![GPU Output](gpu_output.png)

*Note: Due to overheads of the graphics-API compute pipeline, simplistic pixel-wise loops could run very fast on CPU, while hitting fixed GPU-host dispatch time costs.*
