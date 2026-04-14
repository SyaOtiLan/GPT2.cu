#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>

#include "../include/gpt2.h"
#include "reference/cpu_reference.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static bool has_nan_or_inf(const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        if (isnan(x[i]) || isinf(x[i])) return true;
    }
    return false;
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

static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

static bool ensure_dir(const char* path) {
    if (mkdir(path, 0777) == 0) {
        return true;
    }
    return errno == EEXIST;
}

static bool write_tensor(const char* path, const float* data, size_t count) {
    FILE* fp = fopen(path, "wb");
    if (fp == nullptr) {
        return false;
    }
    bool ok = fwrite(data, sizeof(float), count, fp) == count;
    fclose(fp);
    return ok;
}

static bool write_named_tensor(const char* dir, const char* name, const float* data, size_t count) {
    char path[1024];
    int written = snprintf(path, sizeof(path), "%s/%s", dir, name);
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return false;
    }
    return write_tensor(path, data, count);
}

static bool write_block_tensor(
    const char* dir,
    int layer,
    const char* tensor_name,
    const float* data,
    size_t count)
{
    char path[1024];
    int written = snprintf(path, sizeof(path), "%s/block_%d_%s.bin", dir, layer, tensor_name);
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return false;
    }
    return write_tensor(path, data, count);
}

int main() {
    const int tokens = 4;
    const int hidden = 4;
    const int heads = 2;
    const int head_dim = hidden / heads;
    const int n_layer = 1;
    const int vocab_size = 8;
    const int max_position = 8;
    const char* weight_dir = "/tmp/gpt2_test_weights";

    GPT2Config cfg;
    cfg.hidden = hidden;
    cfg.heads = heads;
    cfg.n_layer = n_layer;
    cfg.vocab_size = vocab_size;
    cfg.max_position = max_position;

    // -------------------------
    // host input ids
    // -------------------------
    int h_input_ids[tokens] = {1, 3, 2, 5};

    // embeddings
    float h_wte[vocab_size * hidden];
    float h_wpe[max_position * hidden];

    for (int i = 0; i < vocab_size * hidden; ++i) {
        h_wte[i] = 0.02f * ((i % 7) - 3);
    }
    for (int i = 0; i < max_position * hidden; ++i) {
        h_wpe[i] = 0.01f * ((i % 5) - 2);
    }

    // one block weights
    float h_ln1_gamma[hidden] = {1.f, 1.f, 1.f, 1.f};
    float h_ln1_beta [hidden] = {0.f, 0.f, 0.f, 0.f};

    float h_ln2_gamma[hidden] = {1.f, 1.f, 1.f, 1.f};
    float h_ln2_beta [hidden] = {0.f, 0.f, 0.f, 0.f};

    float h_ln_f_gamma[hidden] = {1.f, 1.f, 1.f, 1.f};
    float h_ln_f_beta [hidden] = {0.f, 0.f, 0.f, 0.f};

    float h_w_qkv[hidden * (3 * hidden)];
    float h_b_qkv[3 * hidden];
    float h_w_o[hidden * hidden];
    float h_b_o[hidden];
    float h_w_fc1[hidden * (4 * hidden)];
    float h_b_fc1[4 * hidden];
    float h_w_fc2[(4 * hidden) * hidden];
    float h_b_fc2[hidden];
    float h_lm_head[hidden * vocab_size];

    for (int i = 0; i < hidden * (3 * hidden); ++i) {
        h_w_qkv[i] = 0.02f * ((i % 7) - 3);
    }
    for (int i = 0; i < hidden * hidden; ++i) {
        h_w_o[i] = 0.02f * ((i % 5) - 2);
    }
    for (int i = 0; i < 3 * hidden; ++i) {
        h_b_qkv[i] = 0.01f * ((i % 5) - 2);
    }
    for (int i = 0; i < hidden; ++i) {
        h_b_o[i] = 0.01f * ((i % 3) - 1);
        h_b_fc2[i] = 0.01f * ((i % 3) - 1);
    }
    for (int i = 0; i < hidden * (4 * hidden); ++i) {
        h_w_fc1[i] = 0.01f * ((i % 9) - 4);
    }
    for (int i = 0; i < 4 * hidden; ++i) {
        h_b_fc1[i] = 0.01f * ((i % 5) - 2);
    }
    for (int i = 0; i < (4 * hidden) * hidden; ++i) {
        h_w_fc2[i] = 0.01f * ((i % 5) - 2);
    }
    for (int i = 0; i < hidden * vocab_size; ++i) {
        h_lm_head[i] = 0.02f * ((i % 11) - 5);
    }

    // -------------------------
    // write weights and load them through loader
    // -------------------------
    if (!ensure_dir(weight_dir)) {
        fprintf(stderr, "failed to create weight dir: %s\n", weight_dir);
        return 1;
    }
    if (!write_named_tensor(weight_dir, "wte.bin", h_wte, vocab_size * hidden)) return 1;
    if (!write_named_tensor(weight_dir, "wpe.bin", h_wpe, max_position * hidden)) return 1;
    if (!write_named_tensor(weight_dir, "ln_f_gamma.bin", h_ln_f_gamma, hidden)) return 1;
    if (!write_named_tensor(weight_dir, "ln_f_beta.bin", h_ln_f_beta, hidden)) return 1;
    if (!write_named_tensor(weight_dir, "lm_head.bin", h_lm_head, hidden * vocab_size)) return 1;

    if (!write_block_tensor(weight_dir, 0, "ln1_gamma", h_ln1_gamma, hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "ln1_beta", h_ln1_beta, hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "w_qkv", h_w_qkv, hidden * (3 * hidden))) return 1;
    if (!write_block_tensor(weight_dir, 0, "b_qkv", h_b_qkv, 3 * hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "w_o", h_w_o, hidden * hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "b_o", h_b_o, hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "ln2_gamma", h_ln2_gamma, hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "ln2_beta", h_ln2_beta, hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "w_fc1", h_w_fc1, hidden * (4 * hidden))) return 1;
    if (!write_block_tensor(weight_dir, 0, "b_fc1", h_b_fc1, 4 * hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "w_fc2", h_w_fc2, (4 * hidden) * hidden)) return 1;
    if (!write_block_tensor(weight_dir, 0, "b_fc2", h_b_fc2, hidden)) return 1;

    int* d_input_ids = nullptr;
    check_cuda(cudaMalloc(&d_input_ids, tokens * sizeof(int)), "malloc d_input_ids");

    check_cuda(cudaMemcpy(d_input_ids, h_input_ids, tokens * sizeof(int), cudaMemcpyHostToDevice), "copy input_ids");

    // -------------------------
    // load weights / workspace
    // -------------------------
    GPT2Weights w = {};
    if (!load_gpt2_weights_from_dir(&w, cfg, weight_dir)) {
        fprintf(stderr, "failed to load GPT2 weights from %s\n", weight_dir);
        return 1;
    }

    GPT2Workspace ws;
    if (!create_gpt2_workspace(&ws, cfg, tokens)) {
        fprintf(stderr, "failed to create GPT2 workspace\n");
        destroy_gpt2_weights(&w, cfg);
        cudaFree(d_input_ids);
        return 1;
    }

    // -------------------------
    // run gpt2 forward
    // -------------------------
    gpt2_forward(
        d_input_ids,
        ws.logits,
        w,
        ws,
        cfg,
        tokens
    );

    check_cuda(cudaDeviceSynchronize(), "sync after gpt2_forward");

    // -------------------------
    // copy back
    // -------------------------
    float h_logits[tokens * vocab_size];
    float h_x[tokens * hidden];
    float h_logits_ref[tokens * vocab_size];

    check_cuda(cudaMemcpy(h_logits, ws.logits, tokens * vocab_size * sizeof(float), cudaMemcpyDeviceToHost), "copy logits");
    check_cuda(cudaMemcpy(h_x, ws.x, tokens * hidden * sizeof(float), cudaMemcpyDeviceToHost), "copy embedding out");

    CPUBlockWeights cpu_block_w;
    cpu_block_w.ln1_gamma = h_ln1_gamma;
    cpu_block_w.ln1_beta  = h_ln1_beta;
    cpu_block_w.w_qkv     = h_w_qkv;
    cpu_block_w.b_qkv     = h_b_qkv;
    cpu_block_w.w_o       = h_w_o;
    cpu_block_w.b_o       = h_b_o;
    cpu_block_w.ln2_gamma = h_ln2_gamma;
    cpu_block_w.ln2_beta  = h_ln2_beta;
    cpu_block_w.w_fc1     = h_w_fc1;
    cpu_block_w.b_fc1     = h_b_fc1;
    cpu_block_w.w_fc2     = h_w_fc2;
    cpu_block_w.b_fc2     = h_b_fc2;

    CPUBlockWeights cpu_blocks[n_layer];
    cpu_blocks[0] = cpu_block_w;

    CPUGPT2Weights w_ref;
    w_ref.wte = h_wte;
    w_ref.wpe = h_wpe;
    w_ref.blocks = cpu_blocks;
    w_ref.ln_f_gamma = h_ln_f_gamma;
    w_ref.ln_f_beta  = h_ln_f_beta;
    w_ref.lm_head    = h_lm_head;

    gpt2_forward_cpu_ref(
        h_input_ids,
        h_logits_ref,
        w_ref,
        tokens,
        hidden,
        heads,
        n_layer,
        vocab_size
    );

    print_matrix("embedding out", h_x, tokens, hidden);
    print_matrix("logits", h_logits, tokens, vocab_size);

    const float tol = 1e-3f;
    float max_err = max_abs_diff_cpu_ref(h_logits, h_logits_ref, tokens * vocab_size);
    if (has_nan_or_inf(h_logits, tokens * vocab_size)) {
        fprintf(stderr, "Found NaN/Inf in logits\n");
        return 1;
    }
    if (max_err > tol) {
        fprintf(stderr, "GPT2 logits mismatch: max_err=%g tol=%g\n", max_err, tol);
        return 1;
    }

    // last token argmax
    const float* last_logits = h_logits + (tokens - 1) * vocab_size;
    int next_token = argmax(last_logits, vocab_size);
    const float* last_logits_ref = h_logits_ref + (tokens - 1) * vocab_size;
    int next_token_ref = argmax(last_logits_ref, vocab_size);
    printf("argmax(next token) = %d\n", next_token);
    if (next_token != next_token_ref) {
        fprintf(stderr, "GPT2 argmax mismatch: got %d expected %d\n", next_token, next_token_ref);
        return 1;
    }

    // -------------------------
    // free
    // -------------------------
    cudaFree(d_input_ids);
    destroy_gpt2_weights(&w, cfg);
    destroy_gpt2_workspace(&ws);

    return 0;
}
