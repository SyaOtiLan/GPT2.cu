#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/transformer_block.h"
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

static bool has_nan_or_inf(const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        if (isnan(x[i]) || isinf(x[i])) return true;
    }
    return false;
}

int main() {
    const int tokens = 4;
    const int hidden = 4;
    const int heads = 2;
    const int head_dim = hidden / heads;

    // sizes
    const int x_size        = tokens * hidden;
    const int ln_size       = hidden;

    const int w_qkv_size    = hidden * (3 * hidden);
    const int w_o_size      = hidden * hidden;
    const int w_fc1_size    = hidden * (4 * hidden);
    const int w_fc2_size    = (4 * hidden) * hidden;
    const int b_qkv_size    = 3 * hidden;
    const int b_o_size      = hidden;
    const int b_fc1_size    = 4 * hidden;
    const int b_fc2_size    = hidden;

    const int qkv_size      = tokens * 3 * hidden;
    const int qkv_each      = heads * tokens * head_dim;
    const int scores_size   = heads * tokens * tokens;
    const int hidden_size   = tokens * hidden;
    const int fc1_size      = tokens * 4 * hidden;

    // -------------------------
    // host inputs / weights
    // -------------------------
    float h_x[x_size] = {
         1.0f,  0.5f, -1.0f,  2.0f,
         0.3f, -0.7f,  1.2f, -1.5f,
         2.0f,  1.0f,  0.0f, -0.5f,
        -1.0f,  2.5f,  0.8f,  1.1f
    };

    float h_ln1_gamma[ln_size] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_ln1_beta [ln_size] = {0.0f, 0.0f, 0.0f, 0.0f};

    float h_ln2_gamma[ln_size] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_ln2_beta [ln_size] = {0.0f, 0.0f, 0.0f, 0.0f};

    float h_w_qkv[w_qkv_size];
    float h_w_o[w_o_size];
    float h_w_fc1[w_fc1_size];
    float h_w_fc2[w_fc2_size];
    float h_b_qkv[b_qkv_size];
    float h_b_o[b_o_size];
    float h_b_fc1[b_fc1_size];
    float h_b_fc2[b_fc2_size];

    for (int i = 0; i < w_qkv_size; ++i) {
        h_w_qkv[i] = 0.02f * ((i % 7) - 3);
    }
    for (int i = 0; i < w_o_size; ++i) {
        h_w_o[i] = 0.02f * ((i % 5) - 2);
    }
    for (int i = 0; i < w_fc1_size; ++i) {
        h_w_fc1[i] = 0.01f * ((i % 9) - 4);
    }
    for (int i = 0; i < w_fc2_size; ++i) {
        h_w_fc2[i] = 0.01f * ((i % 5) - 2);
    }
    for (int i = 0; i < b_qkv_size; ++i) {
        h_b_qkv[i] = 0.01f * ((i % 5) - 2);
    }
    for (int i = 0; i < b_o_size; ++i) {
        h_b_o[i] = 0.01f * ((i % 3) - 1);
        h_b_fc2[i] = 0.01f * ((i % 3) - 1);
    }
    for (int i = 0; i < b_fc1_size; ++i) {
        h_b_fc1[i] = 0.01f * ((i % 5) - 2);
    }

    // -------------------------
    // device weights / input
    // -------------------------
    float *d_x = nullptr, *d_out = nullptr;

    float *d_ln1_gamma = nullptr, *d_ln1_beta = nullptr;
    float *d_ln2_gamma = nullptr, *d_ln2_beta = nullptr;

    float *d_w_qkv = nullptr, *d_b_qkv = nullptr, *d_w_o = nullptr, *d_b_o = nullptr;
    float *d_w_fc1 = nullptr, *d_b_fc1 = nullptr, *d_w_fc2 = nullptr, *d_b_fc2 = nullptr;

    // workspace
    float *d_ln1_out = nullptr;
    float *d_qkv = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_scores = nullptr, *d_probs = nullptr, *d_context = nullptr;
    float *d_merged = nullptr, *d_attn_proj = nullptr, *d_attn_out = nullptr;
    float *d_ln2_out = nullptr;
    float *d_fc1_out = nullptr, *d_gelu_out = nullptr, *d_fc2_out = nullptr;

    check_cuda(cudaMalloc(&d_x, x_size * sizeof(float)), "malloc d_x");
    check_cuda(cudaMalloc(&d_out, x_size * sizeof(float)), "malloc d_out");

    check_cuda(cudaMalloc(&d_ln1_gamma, ln_size * sizeof(float)), "malloc d_ln1_gamma");
    check_cuda(cudaMalloc(&d_ln1_beta,  ln_size * sizeof(float)), "malloc d_ln1_beta");
    check_cuda(cudaMalloc(&d_ln2_gamma, ln_size * sizeof(float)), "malloc d_ln2_gamma");
    check_cuda(cudaMalloc(&d_ln2_beta,  ln_size * sizeof(float)), "malloc d_ln2_beta");

    check_cuda(cudaMalloc(&d_w_qkv, w_qkv_size * sizeof(float)), "malloc d_w_qkv");
    check_cuda(cudaMalloc(&d_b_qkv, b_qkv_size * sizeof(float)), "malloc d_b_qkv");
    check_cuda(cudaMalloc(&d_w_o,   w_o_size   * sizeof(float)), "malloc d_w_o");
    check_cuda(cudaMalloc(&d_b_o,   b_o_size   * sizeof(float)), "malloc d_b_o");
    check_cuda(cudaMalloc(&d_w_fc1, w_fc1_size * sizeof(float)), "malloc d_w_fc1");
    check_cuda(cudaMalloc(&d_b_fc1, b_fc1_size * sizeof(float)), "malloc d_b_fc1");
    check_cuda(cudaMalloc(&d_w_fc2, w_fc2_size * sizeof(float)), "malloc d_w_fc2");
    check_cuda(cudaMalloc(&d_b_fc2, b_fc2_size * sizeof(float)), "malloc d_b_fc2");

    check_cuda(cudaMalloc(&d_ln1_out, hidden_size * sizeof(float)), "malloc d_ln1_out");

    check_cuda(cudaMalloc(&d_qkv,     qkv_size    * sizeof(float)), "malloc d_qkv");
    check_cuda(cudaMalloc(&d_q,       qkv_each    * sizeof(float)), "malloc d_q");
    check_cuda(cudaMalloc(&d_k,       qkv_each    * sizeof(float)), "malloc d_k");
    check_cuda(cudaMalloc(&d_v,       qkv_each    * sizeof(float)), "malloc d_v");
    check_cuda(cudaMalloc(&d_scores,  scores_size * sizeof(float)), "malloc d_scores");
    check_cuda(cudaMalloc(&d_probs,   scores_size * sizeof(float)), "malloc d_probs");
    check_cuda(cudaMalloc(&d_context, qkv_each    * sizeof(float)), "malloc d_context");
    check_cuda(cudaMalloc(&d_merged,  hidden_size * sizeof(float)), "malloc d_merged");
    check_cuda(cudaMalloc(&d_attn_proj, hidden_size * sizeof(float)), "malloc d_attn_proj");
    check_cuda(cudaMalloc(&d_attn_out,  hidden_size * sizeof(float)), "malloc d_attn_out");

    check_cuda(cudaMalloc(&d_ln2_out, hidden_size * sizeof(float)), "malloc d_ln2_out");
    check_cuda(cudaMalloc(&d_fc1_out, fc1_size * sizeof(float)), "malloc d_fc1_out");
    check_cuda(cudaMalloc(&d_gelu_out, fc1_size * sizeof(float)), "malloc d_gelu_out");
    check_cuda(cudaMalloc(&d_fc2_out, hidden_size * sizeof(float)), "malloc d_fc2_out");

    // copy inputs / weights
    check_cuda(cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice), "copy x");

    check_cuda(cudaMemcpy(d_ln1_gamma, h_ln1_gamma, ln_size * sizeof(float), cudaMemcpyHostToDevice), "copy ln1 gamma");
    check_cuda(cudaMemcpy(d_ln1_beta,  h_ln1_beta,  ln_size * sizeof(float), cudaMemcpyHostToDevice), "copy ln1 beta");
    check_cuda(cudaMemcpy(d_ln2_gamma, h_ln2_gamma, ln_size * sizeof(float), cudaMemcpyHostToDevice), "copy ln2 gamma");
    check_cuda(cudaMemcpy(d_ln2_beta,  h_ln2_beta,  ln_size * sizeof(float), cudaMemcpyHostToDevice), "copy ln2 beta");

    check_cuda(cudaMemcpy(d_w_qkv, h_w_qkv, w_qkv_size * sizeof(float), cudaMemcpyHostToDevice), "copy w_qkv");
    check_cuda(cudaMemcpy(d_b_qkv, h_b_qkv, b_qkv_size * sizeof(float), cudaMemcpyHostToDevice), "copy b_qkv");
    check_cuda(cudaMemcpy(d_w_o,   h_w_o,   w_o_size   * sizeof(float), cudaMemcpyHostToDevice), "copy w_o");
    check_cuda(cudaMemcpy(d_b_o,   h_b_o,   b_o_size   * sizeof(float), cudaMemcpyHostToDevice), "copy b_o");
    check_cuda(cudaMemcpy(d_w_fc1, h_w_fc1, w_fc1_size * sizeof(float), cudaMemcpyHostToDevice), "copy w_fc1");
    check_cuda(cudaMemcpy(d_b_fc1, h_b_fc1, b_fc1_size * sizeof(float), cudaMemcpyHostToDevice), "copy b_fc1");
    check_cuda(cudaMemcpy(d_w_fc2, h_w_fc2, w_fc2_size * sizeof(float), cudaMemcpyHostToDevice), "copy w_fc2");
    check_cuda(cudaMemcpy(d_b_fc2, h_b_fc2, b_fc2_size * sizeof(float), cudaMemcpyHostToDevice), "copy b_fc2");

    // weights struct
    BlockWeights w;
    w.ln1_gamma = d_ln1_gamma;
    w.ln1_beta  = d_ln1_beta;
    w.w_qkv     = d_w_qkv;
    w.b_qkv     = d_b_qkv;
    w.w_o       = d_w_o;
    w.b_o       = d_b_o;
    w.ln2_gamma = d_ln2_gamma;
    w.ln2_beta  = d_ln2_beta;
    w.w_fc1     = d_w_fc1;
    w.b_fc1     = d_b_fc1;
    w.w_fc2     = d_w_fc2;
    w.b_fc2     = d_b_fc2;

    // workspace struct
    BlockWorkspace ws;
    ws.ln1_out   = d_ln1_out;
    ws.qkv       = d_qkv;
    ws.q         = d_q;
    ws.k         = d_k;
    ws.v         = d_v;
    ws.scores    = d_scores;
    ws.probs     = d_probs;
    ws.context   = d_context;
    ws.merged    = d_merged;
    ws.attn_proj = d_attn_proj;
    ws.attn_out  = d_attn_out;
    ws.ln2_out   = d_ln2_out;
    ws.fc1_out   = d_fc1_out;
    ws.gelu_out  = d_gelu_out;
    ws.fc2_out   = d_fc2_out;

    // run one transformer block
    transformer_block_forward(
        d_x,
        d_out,
        w,
        tokens,
        hidden,
        heads,
        ws
    );

    check_cuda(cudaDeviceSynchronize(), "sync after transformer_block_forward");

    // copy outputs
    float h_out[x_size];
    float h_attn_out[x_size];
    float h_ln1_out[x_size];
    float h_ln2_out[x_size];
    float h_out_ref[x_size];

    check_cuda(cudaMemcpy(h_out,      d_out,      x_size * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
    check_cuda(cudaMemcpy(h_attn_out, d_attn_out, x_size * sizeof(float), cudaMemcpyDeviceToHost), "copy attn_out");
    check_cuda(cudaMemcpy(h_ln1_out,  d_ln1_out,  x_size * sizeof(float), cudaMemcpyDeviceToHost), "copy ln1_out");
    check_cuda(cudaMemcpy(h_ln2_out,  d_ln2_out,  x_size * sizeof(float), cudaMemcpyDeviceToHost), "copy ln2_out");

    CPUBlockWeights w_ref;
    w_ref.ln1_gamma = h_ln1_gamma;
    w_ref.ln1_beta  = h_ln1_beta;
    w_ref.w_qkv     = h_w_qkv;
    w_ref.b_qkv     = h_b_qkv;
    w_ref.w_o       = h_w_o;
    w_ref.b_o       = h_b_o;
    w_ref.ln2_gamma = h_ln2_gamma;
    w_ref.ln2_beta  = h_ln2_beta;
    w_ref.w_fc1     = h_w_fc1;
    w_ref.b_fc1     = h_b_fc1;
    w_ref.w_fc2     = h_w_fc2;
    w_ref.b_fc2     = h_b_fc2;
    transformer_block_cpu_ref(h_x, h_out_ref, w_ref, tokens, hidden, heads);

    print_matrix("input x", h_x, tokens, hidden);
    print_matrix("ln1_out", h_ln1_out, tokens, hidden);
    print_matrix("attn_out", h_attn_out, tokens, hidden);
    print_matrix("ln2_out", h_ln2_out, tokens, hidden);
    print_matrix("block out", h_out, tokens, hidden);

    const float tol = 1e-3f;
    float max_err = max_abs_diff_cpu_ref(h_out, h_out_ref, x_size);
    if (has_nan_or_inf(h_out, x_size)) {
        fprintf(stderr, "Found NaN or Inf in block output\n");
        return 1;
    }
    if (max_err > tol) {
        fprintf(stderr, "Block output mismatch: max_err=%g tol=%g\n", max_err, tol);
        return 1;
    }

    // free
    cudaFree(d_x);
    cudaFree(d_out);

    cudaFree(d_ln1_gamma);
    cudaFree(d_ln1_beta);
    cudaFree(d_ln2_gamma);
    cudaFree(d_ln2_beta);

    cudaFree(d_w_qkv);
    cudaFree(d_b_qkv);
    cudaFree(d_w_o);
    cudaFree(d_b_o);
    cudaFree(d_w_fc1);
    cudaFree(d_b_fc1);
    cudaFree(d_w_fc2);
    cudaFree(d_b_fc2);

    cudaFree(d_ln1_out);
    cudaFree(d_qkv);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_scores);
    cudaFree(d_probs);
    cudaFree(d_context);
    cudaFree(d_merged);
    cudaFree(d_attn_proj);
    cudaFree(d_attn_out);
    cudaFree(d_ln2_out);
    cudaFree(d_fc1_out);
    cudaFree(d_gelu_out);
    cudaFree(d_fc2_out);

    return 0;
}
