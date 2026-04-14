#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "../include/kernels.h"

#define MAX_THREADS 1024

__global__ void softmax_kernel(
    const float* x,
    float* y,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    __shared__ float sdata[MAX_THREADS];
    __shared__ float row_max;
    __shared__ float row_sum;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = x[row * cols + col];
        local_max = fmaxf(local_max, v);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = expf(x[row * cols + col] - row_max);
        y[row * cols + col] = v;
        local_sum += v;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) row_sum = sdata[0];
    __syncthreads();

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row * cols + col] /= row_sum;
    }
}

void launch_softmax(const float* x, float* y, int rows, int cols)
{
    int threads = 1;
    while (threads < cols) threads <<= 1;
    if (threads > 1024) threads = 1024;

    int blocks = rows;
    softmax_kernel<<<blocks, threads>>>(x, y, rows, cols);
}