#include <cuda_runtime.h>
#include "../include/embedding.h"

__global__ void embedding_kernel(
    const int* input_ids,
    const float* wte,
    const float* wpe,
    float* x,
    int tokens,
    int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * hidden;
    if (idx >= total) return;

    int token_pos = idx / hidden;
    int dim = idx % hidden;

    int token_id = input_ids[token_pos];

    x[idx] = wte[token_id * hidden + dim] + wpe[token_pos * hidden + dim];
}

void launch_embedding(
    const int* input_ids,
    const float* wte,
    const float* wpe,
    float* x,
    int tokens,
    int hidden)
{
    int total = tokens * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(input_ids, wte, wpe, x, tokens, hidden);
}