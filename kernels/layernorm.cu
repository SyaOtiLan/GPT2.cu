#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define EPS 1e-5f
#define MAX_THREADS 1024

__global__ void layernorm_kernel(
    const float* x,
    float* y,
    const float* gamma,
    const float* beta,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    int cols4 = cols / 4;   // 每个 float4 覆盖 4 个 float

    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float t_sum[MAX_THREADS];

    // -------------------------
    // 1) 每线程局部和
    // -------------------------
    float local_sum = 0.0f;

    if (tid < cols4) {
        int col = tid * 4;
        int idx = row * cols + col;
        float4 x_vec = *reinterpret_cast<const float4*>(&x[idx]);
        local_sum = x_vec.x + x_vec.y + x_vec.z + x_vec.w;
    } else {
        local_sum = 0.0f;
    }

    t_sum[tid] = local_sum;
    __syncthreads();

    // block reduction 求总和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t_sum[tid] += t_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_mean = t_sum[0] / cols;
    }
    __syncthreads();

    float mean = s_mean;

    // -------------------------
    // 2) 每线程局部平方差和
    // -------------------------
    float local_var = 0.0f;

    if (tid < cols4) {
        int col = tid * 4;
        int idx = row * cols + col;
        float4 x_vec = *reinterpret_cast<const float4*>(&x[idx]);

        float dx = x_vec.x - mean;
        float dy = x_vec.y - mean;
        float dz = x_vec.z - mean;
        float dw = x_vec.w - mean;

        local_var = dx * dx + dy * dy + dz * dz + dw * dw;
    } else {
        local_var = 0.0f;
    }

    t_sum[tid] = local_var;
    __syncthreads();

    // block reduction 求平方差和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t_sum[tid] += t_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_var = t_sum[0] / cols;
    }
    __syncthreads();

    float inv_std = rsqrtf(s_var + EPS);

    // -------------------------
    // 3) normalize + affine
    // -------------------------
    if (tid < cols4) {
        int col = tid * 4;
        int idx = row * cols + col;

        float4 x_vec = *reinterpret_cast<const float4*>(&x[idx]);

        float4 y_vec;
        y_vec.x = ((x_vec.x - mean) * inv_std) * gamma[col + 0] + beta[col + 0];
        y_vec.y = ((x_vec.y - mean) * inv_std) * gamma[col + 1] + beta[col + 1];
        y_vec.z = ((x_vec.z - mean) * inv_std) * gamma[col + 2] + beta[col + 2];
        y_vec.w = ((x_vec.w - mean) * inv_std) * gamma[col + 3] + beta[col + 3];

        *reinterpret_cast<float4*>(&y[idx]) = y_vec;
    }
}

void launch_layernorm(
    const float* x,
    float* y,
    const float* gamma,
    const float* beta,
    int rows,
    int cols)
{
    int cols4 = cols / 4;
    int threads = 1;
    while (threads < cols4) threads <<= 1;
    if (threads > 1024) threads = 1024;

    int blocks = rows;
    layernorm_kernel<<<blocks, threads>>>(x, y, gamma, beta, rows, cols);
}