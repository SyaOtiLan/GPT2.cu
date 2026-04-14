#include <cuda_runtime.h>
#include <stdio.h>

__global__ void residual_add_kernel(
    const float* a,
    const float* b,
    float* y,
    int n)
{
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = idx4 * 4;

    if (idx + 3 < n) {
        float4 a_vec = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[idx]);

        float4 y_vec;
        y_vec.x = a_vec.x + b_vec.x;
        y_vec.y = a_vec.y + b_vec.y;
        y_vec.z = a_vec.z + b_vec.z;
        y_vec.w = a_vec.w + b_vec.w;

        *reinterpret_cast<float4*>(&y[idx]) = y_vec;
    } else {
        for (int i = idx; i < n; ++i) {
            y[i] = a[i] + b[i];
        }
    }
}

void launch_residual_add(const float* a, const float* b, float* y, int n)
{
    int threads = 256;
    int blocks = ((n + 3) / 4 + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads>>>(a, b, y, n);
}