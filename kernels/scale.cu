#include <cuda_runtime.h>
#include <stdio.h>

__global__ void scale_kernel(float* x, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scale;
    }
}

void launch_scale(float* x, float scale, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(x, scale, n);
}
