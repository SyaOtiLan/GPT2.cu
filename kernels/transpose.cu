#include <cuda_runtime.h>

__global__ void transpose_kernel(
    const float* in,
    float* out,
    int rows,
    int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int r = idx / cols;
    int c = idx % cols;

    out[c * rows + r] = in[r * cols + c];
}

void launch_transpose(const float* in, float* out, int rows, int cols)
{
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_kernel<<<blocks, threads>>>(in, out, rows, cols);
}