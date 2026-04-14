#include <cuda_runtime.h>
#include <stdio.h>

__global__ void qkv_split_reshape_kernel(
    const float* qkv,   // [tokens, 3 * hidden]
    float* q,           // [heads, tokens, head_dim]
    float* k,           // [heads, tokens, head_dim]
    float* v,           // [heads, tokens, head_dim]
    int tokens,
    int hidden,
    int heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * hidden;

    if (idx >= total) return;

    // idx 对应的是 (token, hidden_idx)
    int token = idx / hidden;
    int hidden_idx = idx % hidden;

    int head = hidden_idx / head_dim;
    int dim  = hidden_idx % head_dim;

    // 目标位置: [heads, tokens, head_dim]
    int out_idx = (head * tokens + token) * head_dim + dim;

    // 源位置: [tokens, 3 * hidden]
    int base = token * (3 * hidden);

    q[out_idx] = qkv[base + 0 * hidden + hidden_idx];
    k[out_idx] = qkv[base + 1 * hidden + hidden_idx];
    v[out_idx] = qkv[base + 2 * hidden + hidden_idx];
}

void launch_qkv_split_reshape(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    int tokens,
    int hidden,
    int heads)
{
    int total = tokens * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    int head_dim = hidden / heads;

    qkv_split_reshape_kernel<<<blocks, threads>>>(
        qkv, q, k, v, tokens, hidden, heads, head_dim
    );
}