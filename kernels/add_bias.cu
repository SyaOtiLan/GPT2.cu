#include <cuda_runtime.h>

#include "../include/kernels.h"

__global__ void add_bias_kernel(float* x, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    x[idx] += bias[col];
}

void launch_add_bias(float* x, const float* bias, int rows, int cols) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(x, bias, rows, cols);
}
