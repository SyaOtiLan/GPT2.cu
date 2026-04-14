#pragma once

// Standard row-major GEMM: A[M, K] * B[K, N] -> C[M, N]
void gemm(const float* A, const float* B, float* C, int M, int N, int K);

// QK^T: Q[tokens, head_dim] * K^T[head_dim, tokens] -> scores[tokens, tokens]
void gemm_qk(const float* Q, const float* K, float* S, int tokens, int head_dim);

// P * V: P[tokens, tokens] * V[tokens, head_dim] -> context[tokens, head_dim]
void gemm_pv(const float* P, const float* V, float* C, int tokens, int head_dim);
