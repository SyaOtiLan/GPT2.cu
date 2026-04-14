#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(
    const float* x,
    float* y,
    int n)
{
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x; // 第几个 float4
    int idx = idx4 * 4;                               // 转成 float 下标

    if (idx + 3 < n) {
        float4 x_vec = *reinterpret_cast<const float4*>(&x[idx]);

        float4 y_vec;

        float x0 = x_vec.x;
        float x1 = x_vec.y;
        float x2 = x_vec.z;
        float x3 = x_vec.w;

        y_vec.x = 0.5f * x0 * (1.0f + tanhf(0.7978845608f * (x0 + 0.044715f * x0 * x0 * x0)));
        y_vec.y = 0.5f * x1 * (1.0f + tanhf(0.7978845608f * (x1 + 0.044715f * x1 * x1 * x1)));
        y_vec.z = 0.5f * x2 * (1.0f + tanhf(0.7978845608f * (x2 + 0.044715f * x2 * x2 * x2)));
        y_vec.w = 0.5f * x3 * (1.0f + tanhf(0.7978845608f * (x3 + 0.044715f * x3 * x3 * x3)));

        *reinterpret_cast<float4*>(&y[idx]) = y_vec;
    } else {
        for (int i = idx; i < n; ++i) {
            float v = x[i];
            y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        }
    }
    
}
void launch_gelu(const float* x, float* y, int n)
{
    int threads = 256;
    int blocks = ((n + 3) / 4 + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(x, y, n);
}