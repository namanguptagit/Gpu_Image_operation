extern "C" {
    __global__ void gray_from_rgb_kernel(const float* input, float* output, unsigned int width, unsigned int height) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            unsigned int input_idx = (y * width + x) * 3;
            unsigned int output_idx = y * width + x;

            float r = input[input_idx];
            float g = input[input_idx + 1];
            float b = input[input_idx + 2];

            float rw = 0.299f;
            float gw = 0.587f;
            float bw = 0.114f;

            output[output_idx] = r * rw + g * gw + b * bw;
        }
    }
}
