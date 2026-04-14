#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "../include/gpt2.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
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

static float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main() {
    const int prompt_len = 4;
    const int total_tokens = 6;

    const int hidden = 4;
    const int heads = 2;
    const int n_layer = 1;
    const int vocab_size = 8;
    const int max_position = 8;
    const char* weight_dir = "/tmp/kv_cache_test_weights";

    GPT2Config cfg = {};
    cfg.hidden = hidden;
    cfg.heads = heads;
    cfg.n_layer = n_layer;
    cfg.vocab_size = vocab_size;
    cfg.max_position = max_position;

    int h_input_ids[total_tokens] = {1, 3, 2, 5, 7, 4};

    float h_wte[vocab_size * hidden];
    float h_wpe[max_position * hidden];

    float h_ln1_gamma[hidden] = {1.f, 1.f, 1.f, 1.f};
    float h_ln1_beta[hidden] = {0.f, 0.f, 0.f, 0.f};
    float h_ln2_gamma[hidden] = {1.f, 1.f, 1.f, 1.f};
    float h_ln2_beta[hidden] = {0.f, 0.f, 0.f, 0.f};
    float h_ln_f_gamma[hidden] = {1.f, 1.f, 1.f, 1.f};
    float h_ln_f_beta[hidden] = {0.f, 0.f, 0.f, 0.f};

    float h_w_qkv[hidden * (3 * hidden)];
    float h_b_qkv[3 * hidden];
    float h_w_o[hidden * hidden];
    float h_b_o[hidden];
    float h_w_fc1[hidden * (4 * hidden)];
    float h_b_fc1[4 * hidden];
    float h_w_fc2[(4 * hidden) * hidden];
    float h_b_fc2[hidden];
    float h_lm_head[hidden * vocab_size];

    for (int i = 0; i < vocab_size * hidden; ++i) {
        h_wte[i] = 0.02f * ((i % 7) - 3);
    }
    for (int i = 0; i < max_position * hidden; ++i) {
        h_wpe[i] = 0.01f * ((i % 5) - 2);
    }
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

    GPT2Weights w = {};
    if (!load_gpt2_weights_from_dir(&w, cfg, weight_dir)) {
        fprintf(stderr, "failed to load GPT2 weights from %s\n", weight_dir);
        return 1;
    }

    GPT2Workspace ws_cache;
    if (!create_gpt2_workspace(&ws_cache, cfg, max_position)) {
        fprintf(stderr, "failed to create cache workspace\n");
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    GPT2Workspace ws_full;
    if (!create_gpt2_workspace(&ws_full, cfg, max_position)) {
        fprintf(stderr, "failed to create full workspace\n");
        destroy_gpt2_workspace(&ws_cache);
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    int* d_input_ids = nullptr;
    check_cuda(cudaMalloc(&d_input_ids, total_tokens * sizeof(int)), "malloc d_input_ids");
    check_cuda(
        cudaMemcpy(d_input_ids, h_input_ids, total_tokens * sizeof(int), cudaMemcpyHostToDevice),
        "copy input ids"
    );

    float h_decode_logits[vocab_size];
    float h_full_logits[vocab_size];

    gpt2_prefill(d_input_ids, ws_cache.logits, w, ws_cache, cfg, prompt_len);
    check_cuda(cudaDeviceSynchronize(), "sync after gpt2_prefill");

    gpt2_decode_step(d_input_ids + prompt_len, ws_cache.logits, w, ws_cache, cfg, prompt_len);
    check_cuda(cudaDeviceSynchronize(), "sync after first gpt2_decode_step");
    check_cuda(
        cudaMemcpy(h_decode_logits, ws_cache.logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost),
        "copy first decode logits"
    );

    gpt2_forward(d_input_ids, ws_full.logits, w, ws_full, cfg, prompt_len + 1);
    check_cuda(cudaDeviceSynchronize(), "sync after first full forward");
    check_cuda(
        cudaMemcpy(
            h_full_logits,
            ws_full.logits + prompt_len * vocab_size,
            vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy first full logits"
    );

    float diff1 = max_abs_diff(h_decode_logits, h_full_logits, vocab_size);
    if (diff1 > 1e-4f) {
        fprintf(stderr, "first decode logits mismatch: max diff = %.8f\n", diff1);
        return 1;
    }

    gpt2_decode_step(d_input_ids + prompt_len + 1, ws_cache.logits, w, ws_cache, cfg, prompt_len + 1);
    check_cuda(cudaDeviceSynchronize(), "sync after second gpt2_decode_step");
    check_cuda(
        cudaMemcpy(h_decode_logits, ws_cache.logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost),
        "copy second decode logits"
    );

    gpt2_forward(d_input_ids, ws_full.logits, w, ws_full, cfg, prompt_len + 2);
    check_cuda(cudaDeviceSynchronize(), "sync after second full forward");
    check_cuda(
        cudaMemcpy(
            h_full_logits,
            ws_full.logits + (prompt_len + 1) * vocab_size,
            vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy second full logits"
    );

    float diff2 = max_abs_diff(h_decode_logits, h_full_logits, vocab_size);
    if (diff2 > 1e-4f) {
        fprintf(stderr, "second decode logits mismatch: max diff = %.8f\n", diff2);
        return 1;
    }

    printf("kv_cache_test passed: diff1=%.8f diff2=%.8f\n", diff1, diff2);

    cudaFree(d_input_ids);
    destroy_gpt2_workspace(&ws_cache);
    destroy_gpt2_workspace(&ws_full);
    destroy_gpt2_weights(&w, cfg);
    return 0;
}
