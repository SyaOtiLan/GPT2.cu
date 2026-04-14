#include <cuda_runtime.h>
#include "../include/kernels.h"

#define NEG_INF -1e20f

__global__ void causal_mask_kernel(float* scores, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;

    if (col > row) {
        scores[idx] = NEG_INF;
    }
}

void launch_causal_mask(float* scores, int rows, int cols)
{
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_mask_kernel<<<blocks, threads>>>(scores, rows, cols);
}