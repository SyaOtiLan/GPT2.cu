#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/attention.h"
#include "reference/cpu_reference.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void print_matrix(const char* name, const float* x, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.5f ", x[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_head_matrices(const char* name, const float* x, int heads, int rows, int cols) {
    printf("%s:\n", name);
    for (int h = 0; h < heads; ++h) {
        printf("head %d:\n", h);
        const float* p = x + h * rows * cols;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%8.5f ", p[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    const int tokens = 4;
    const int hidden = 4;
    const int heads = 2;
    const int head_dim = hidden / heads;

    const int x_size       = tokens * hidden;                 // 16
    const int w_qkv_size   = hidden * (3 * hidden);          // 4 * 12 = 48
    const int b_qkv_size   = 3 * hidden;
    const int w_o_size     = hidden * hidden;                // 16
    const int b_o_size     = hidden;

    const int qkv_size     = tokens * 3 * hidden;            // 48
    const int qkv_each     = heads * tokens * head_dim;      // 16
    const int scores_size  = heads * tokens * tokens;        // 32
    const int merged_size  = tokens * hidden;                // 16
    const int proj_out_sz  = tokens * hidden;                // 16
    const int out_size     = tokens * hidden;                // 16

    // host data
    float h_x[x_size] = {
        1.0f,  0.5f, -1.0f,  2.0f,
        0.3f, -0.7f,  1.2f, -1.5f,
        2.0f,  1.0f,  0.0f, -0.5f,
       -1.0f,  2.5f,  0.8f,  1.1f
    };

    float h_w_qkv[w_qkv_size];
    float h_b_qkv[b_qkv_size];
    float h_w_o[w_o_size];
    float h_b_o[b_o_size];

    // 小权重，避免数值太大
    for (int i = 0; i < w_qkv_size; ++i) {
        h_w_qkv[i] = 0.02f * ((i % 7) - 3);
    }
    for (int i = 0; i < w_o_size; ++i) {
        h_w_o[i] = 0.02f * ((i % 5) - 2);
    }
    for (int i = 0; i < b_qkv_size; ++i) {
        h_b_qkv[i] = 0.01f * ((i % 5) - 2);
    }
    for (int i = 0; i < b_o_size; ++i) {
        h_b_o[i] = 0.01f * ((i % 3) - 1);
    }

    // device buffers
    float *d_x = nullptr, *d_out = nullptr;
    float *d_w_qkv = nullptr, *d_b_qkv = nullptr, *d_w_o = nullptr, *d_b_o = nullptr;
    float *d_qkv = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_scores = nullptr, *d_probs = nullptr, *d_context = nullptr;
    float *d_merged = nullptr, *d_proj_out = nullptr;

    check_cuda(cudaMalloc(&d_x,        x_size      * sizeof(float)), "malloc d_x");
    check_cuda(cudaMalloc(&d_out,      out_size    * sizeof(float)), "malloc d_out");
    check_cuda(cudaMalloc(&d_w_qkv,    w_qkv_size  * sizeof(float)), "malloc d_w_qkv");
    check_cuda(cudaMalloc(&d_b_qkv,    b_qkv_size  * sizeof(float)), "malloc d_b_qkv");
    check_cuda(cudaMalloc(&d_w_o,      w_o_size    * sizeof(float)), "malloc d_w_o");
    check_cuda(cudaMalloc(&d_b_o,      b_o_size    * sizeof(float)), "malloc d_b_o");

    check_cuda(cudaMalloc(&d_qkv,      qkv_size    * sizeof(float)), "malloc d_qkv");
    check_cuda(cudaMalloc(&d_q,        qkv_each    * sizeof(float)), "malloc d_q");
    check_cuda(cudaMalloc(&d_k,        qkv_each    * sizeof(float)), "malloc d_k");
    check_cuda(cudaMalloc(&d_v,        qkv_each    * sizeof(float)), "malloc d_v");

    check_cuda(cudaMalloc(&d_scores,   scores_size * sizeof(float)), "malloc d_scores");
    check_cuda(cudaMalloc(&d_probs,    scores_size * sizeof(float)), "malloc d_probs");
    check_cuda(cudaMalloc(&d_context,  qkv_each    * sizeof(float)), "malloc d_context");
    check_cuda(cudaMalloc(&d_merged,   merged_size * sizeof(float)), "malloc d_merged");
    check_cuda(cudaMalloc(&d_proj_out, proj_out_sz * sizeof(float)), "malloc d_proj_out");

    check_cuda(cudaMemcpy(d_x,     h_x,     x_size     * sizeof(float), cudaMemcpyHostToDevice), "memcpy x");
    check_cuda(cudaMemcpy(d_w_qkv, h_w_qkv, w_qkv_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy w_qkv");
    check_cuda(cudaMemcpy(d_b_qkv, h_b_qkv, b_qkv_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy b_qkv");
    check_cuda(cudaMemcpy(d_w_o,   h_w_o,   w_o_size   * sizeof(float), cudaMemcpyHostToDevice), "memcpy w_o");
    check_cuda(cudaMemcpy(d_b_o,   h_b_o,   b_o_size   * sizeof(float), cudaMemcpyHostToDevice), "memcpy b_o");

    // run attention
    attention_forward(
        d_x,
        d_x,
        d_out,
        d_w_qkv,
        d_b_qkv,
        d_w_o,
        d_b_o,
        tokens,
        hidden,
        heads,
        d_qkv,
        d_q,
        d_k,
        d_v,
        d_scores,
        d_probs,
        d_context,
        d_merged,
        d_proj_out
    );

    check_cuda(cudaDeviceSynchronize(), "sync after attention_forward");

    // copy back
    float h_out[out_size];
    float h_probs[scores_size];
    float h_scores[scores_size];
    float h_q[qkv_each], h_k[qkv_each], h_v[qkv_each];
    float h_merged[merged_size];
    float h_out_ref[out_size];

    check_cuda(cudaMemcpy(h_out,    d_out,    out_size    * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
    check_cuda(cudaMemcpy(h_probs,  d_probs,  scores_size * sizeof(float), cudaMemcpyDeviceToHost), "copy probs");
    check_cuda(cudaMemcpy(h_scores, d_scores, scores_size * sizeof(float), cudaMemcpyDeviceToHost), "copy scores");
    check_cuda(cudaMemcpy(h_q,      d_q,      qkv_each    * sizeof(float), cudaMemcpyDeviceToHost), "copy q");
    check_cuda(cudaMemcpy(h_k,      d_k,      qkv_each    * sizeof(float), cudaMemcpyDeviceToHost), "copy k");
    check_cuda(cudaMemcpy(h_v,      d_v,      qkv_each    * sizeof(float), cudaMemcpyDeviceToHost), "copy v");
    check_cuda(cudaMemcpy(h_merged, d_merged, merged_size * sizeof(float), cudaMemcpyDeviceToHost), "copy merged");

    attention_cpu_ref(h_x, h_x, h_out_ref, h_w_qkv, h_b_qkv, h_w_o, h_b_o, tokens, hidden, heads);

    // print
    print_matrix("input x", h_x, tokens, hidden);
    print_head_matrices("Q", h_q, heads, tokens, head_dim);
    print_head_matrices("K", h_k, heads, tokens, head_dim);
    print_head_matrices("V", h_v, heads, tokens, head_dim);
    print_head_matrices("scores(after mask+scale)", h_scores, heads, tokens, tokens);
    print_head_matrices("probs(after softmax)", h_probs, heads, tokens, tokens);
    print_matrix("merged", h_merged, tokens, hidden);
    print_matrix("attention out", h_out, tokens, hidden);

    // 检查 softmax 每行和
    printf("softmax row sums:\n");
    for (int h = 0; h < heads; ++h) {
        printf("head %d:\n", h);
        const float* p = h_probs + h * tokens * tokens;
        for (int i = 0; i < tokens; ++i) {
            float s = 0.0f;
            for (int j = 0; j < tokens; ++j) {
                s += p[i * tokens + j];
            }
            printf("row %d sum = %.6f\n", i, s);
        }
        printf("\n");
    }

    const float tol = 1e-4f;
    float max_err = max_abs_diff_cpu_ref(h_out, h_out_ref, out_size);
    if (!all_finite_cpu_ref(h_out, out_size)) {
        fprintf(stderr, "Attention output contains NaN/Inf\n");
        return 1;
    }
    if (max_err > tol) {
        fprintf(stderr, "Attention output mismatch: max_err=%g tol=%g\n", max_err, tol);
        return 1;
    }
    for (int h = 0; h < heads; ++h) {
        const float* p = h_probs + h * tokens * tokens;
        for (int i = 0; i < tokens; ++i) {
            float row_sum = 0.0f;
            for (int j = 0; j < tokens; ++j) {
                row_sum += p[i * tokens + j];
                if (j > i && fabsf(p[i * tokens + j]) > tol) {
                    fprintf(stderr, "Causal mask violated at head=%d row=%d col=%d\n", h, i, j);
                    return 1;
                }
            }
            if (fabsf(row_sum - 1.0f) > 1e-5f) {
                fprintf(stderr, "Softmax row sum mismatch at head=%d row=%d: %g\n", h, i, row_sum);
                return 1;
            }
        }
    }

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_w_qkv);
    cudaFree(d_b_qkv);
    cudaFree(d_w_o);
    cudaFree(d_b_o);
    cudaFree(d_qkv);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_scores);
    cudaFree(d_probs);
    cudaFree(d_context);
    cudaFree(d_merged);
    cudaFree(d_proj_out);

    return 0;
}
