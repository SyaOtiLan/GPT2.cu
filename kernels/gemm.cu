#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../include/kernels.h"

#define TILE_SIZE 32
#define PADDING 1

__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + PADDING];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + PADDING];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;   // 0~7
    int ty = threadIdx.y;   // 0~31

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx * 4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // load A tile
        int a_base = row * K + k_tile + tx * 4;
        // float4 loads require each row stride to stay 16-byte aligned.
        // That is true for hidden/head_dim GEMMs, but not for attention PV
        // when K=tokens and tokens is 5, 6, 7, ...
        if (row < M && (K % 4 == 0) && (k_tile + tx * 4 + 3) < K) {
            float4 a_vec = *reinterpret_cast<const float4*>(&A[a_base]);
            As[ty][tx * 4 + 0] = a_vec.x;
            As[ty][tx * 4 + 1] = a_vec.y;
            As[ty][tx * 4 + 2] = a_vec.z;
            As[ty][tx * 4 + 3] = a_vec.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int kk = k_tile + tx * 4 + i;
                if (row < M && kk < K)
                    As[ty][tx * 4 + i] = A[row * K + kk];
                else
                    As[ty][tx * 4 + i] = 0.0f;
            }
        }

        // load B tile
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int out_col = col + i;
            int kk = k_tile + ty;
            if (out_col < N && kk < K)
                Bs[ty][tx * 4 + i] = B[kk * N + out_col];
            else
                Bs[ty][tx * 4 + i] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val = As[ty][k];
            sum0 += a_val * Bs[k][tx * 4 + 0];
            sum1 += a_val * Bs[k][tx * 4 + 1];
            sum2 += a_val * Bs[k][tx * 4 + 2];
            sum3 += a_val * Bs[k][tx * 4 + 3];
        }

        __syncthreads();
    }

    if (row < M) {
        if (col + 0 < N) C[row * N + col + 0] = sum0;
        if (col + 1 < N) C[row * N + col + 1] = sum1;
        if (col + 2 < N) C[row * N + col + 2] = sum2;
        if (col + 3 < N) C[row * N + col + 3] = sum3;
    }
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 blockDim(8, 32);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void gemm_pv(const float* P, const float* V, float* C, int tokens, int head_dim)
{
    // [tokens, tokens] x [tokens, head_dim] -> [tokens, head_dim]
    gemm(P, V, C, tokens, head_dim, tokens);
}

void gemm_qk(const float* Q, const float* K, float* S, int tokens, int head_dim)
{
    float* K_T = nullptr;
    cudaMalloc(&K_T, sizeof(float) * tokens * head_dim);

    // K: [tokens, head_dim] -> K_T: [head_dim, tokens]
    launch_transpose(K, K_T, tokens, head_dim);

    // Q[tokens, head_dim] x K_T[head_dim, tokens] -> S[tokens, tokens]
    gemm(Q, K_T, S, tokens, tokens, head_dim);

    cudaFree(K_T);
}
