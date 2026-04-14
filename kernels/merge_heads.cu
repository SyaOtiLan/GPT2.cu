#include <cuda_runtime.h>
#include <stdio.h>

__global__ void merge_heads_kernel(
    const float* x,   // [heads, tokens, head_dim]
    float* y,         // [tokens, hidden]
    int tokens,
    int heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden = heads * head_dim;
    int total = tokens * hidden;

    if (idx >= total) return;

    // 输出位置: [tokens, hidden]
    int token = idx / hidden;
    int hidden_idx = idx % hidden;

    int head = hidden_idx / head_dim;
    int dim  = hidden_idx % head_dim;

    // 输入位置: [heads, tokens, head_dim]
    int in_idx = (head * tokens + token) * head_dim + dim;

    y[idx] = x[in_idx];
}

void launch_merge_heads(
    const float* x,
    float* y,
    int tokens,
    int heads,
    int head_dim)
{
    int hidden = heads * head_dim;
    int total = tokens * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    merge_heads_kernel<<<blocks, threads>>>(
        x, y, tokens, heads, head_dim
    );
}