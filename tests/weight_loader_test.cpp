#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <vector>

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

static bool copy_and_check(const char* name, const float* device, const float* expected, size_t count) {
    std::vector<float> got(count);
    check_cuda(cudaMemcpy(got.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost), name);

    const float tol = 1e-6f;
    for (size_t i = 0; i < count; ++i) {
        if (fabsf(got[i] - expected[i]) > tol) {
            fprintf(stderr, "%s mismatch at %zu: got %f expected %f\n", name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const char* dir = "/tmp/gpt2_weight_loader_test";
    if (!ensure_dir(dir)) {
        fprintf(stderr, "failed to create test dir: %s\n", dir);
        return 1;
    }

    GPT2Config cfg;
    cfg.hidden = 4;
    cfg.heads = 2;
    cfg.n_layer = 1;
    cfg.vocab_size = 8;
    cfg.max_position = 8;

    std::vector<float> wte(cfg.vocab_size * cfg.hidden);
    std::vector<float> wpe(cfg.max_position * cfg.hidden);
    std::vector<float> ln_f_gamma(cfg.hidden);
    std::vector<float> ln_f_beta(cfg.hidden);
    std::vector<float> lm_head(cfg.hidden * cfg.vocab_size);

    std::vector<float> ln1_gamma(cfg.hidden);
    std::vector<float> ln1_beta(cfg.hidden);
    std::vector<float> w_qkv(cfg.hidden * 3 * cfg.hidden);
    std::vector<float> b_qkv(3 * cfg.hidden);
    std::vector<float> w_o(cfg.hidden * cfg.hidden);
    std::vector<float> b_o(cfg.hidden);
    std::vector<float> ln2_gamma(cfg.hidden);
    std::vector<float> ln2_beta(cfg.hidden);
    std::vector<float> w_fc1(cfg.hidden * 4 * cfg.hidden);
    std::vector<float> b_fc1(4 * cfg.hidden);
    std::vector<float> w_fc2(4 * cfg.hidden * cfg.hidden);
    std::vector<float> b_fc2(cfg.hidden);

    for (size_t i = 0; i < wte.size(); ++i) wte[i] = 0.01f * (float)((int)(i % 7) - 3);
    for (size_t i = 0; i < wpe.size(); ++i) wpe[i] = 0.01f * (float)((int)(i % 5) - 2);
    for (size_t i = 0; i < lm_head.size(); ++i) lm_head[i] = 0.02f * (float)((int)(i % 11) - 5);
    for (size_t i = 0; i < w_qkv.size(); ++i) w_qkv[i] = 0.02f * (float)((int)(i % 7) - 3);
    for (size_t i = 0; i < b_qkv.size(); ++i) b_qkv[i] = 0.01f * (float)((int)(i % 5) - 2);
    for (size_t i = 0; i < w_o.size(); ++i) w_o[i] = 0.02f * (float)((int)(i % 5) - 2);
    for (size_t i = 0; i < w_fc1.size(); ++i) w_fc1[i] = 0.01f * (float)((int)(i % 9) - 4);
    for (size_t i = 0; i < b_fc1.size(); ++i) b_fc1[i] = 0.01f * (float)((int)(i % 5) - 2);
    for (size_t i = 0; i < w_fc2.size(); ++i) w_fc2[i] = 0.01f * (float)((int)(i % 5) - 2);

    for (int i = 0; i < cfg.hidden; ++i) {
        ln_f_gamma[i] = 1.0f;
        ln_f_beta[i] = 0.0f;
        ln1_gamma[i] = 1.0f;
        ln1_beta[i] = 0.0f;
        b_o[i] = 0.01f * (float)((i % 3) - 1);
        ln2_gamma[i] = 1.0f;
        ln2_beta[i] = 0.0f;
        b_fc2[i] = 0.01f * (float)((i % 3) - 1);
    }

    if (!write_named_tensor(dir, "wte.bin", wte.data(), wte.size())) return 1;
    if (!write_named_tensor(dir, "wpe.bin", wpe.data(), wpe.size())) return 1;
    if (!write_named_tensor(dir, "ln_f_gamma.bin", ln_f_gamma.data(), ln_f_gamma.size())) return 1;
    if (!write_named_tensor(dir, "ln_f_beta.bin", ln_f_beta.data(), ln_f_beta.size())) return 1;
    if (!write_named_tensor(dir, "lm_head.bin", lm_head.data(), lm_head.size())) return 1;

    if (!write_block_tensor(dir, 0, "ln1_gamma", ln1_gamma.data(), ln1_gamma.size())) return 1;
    if (!write_block_tensor(dir, 0, "ln1_beta", ln1_beta.data(), ln1_beta.size())) return 1;
    if (!write_block_tensor(dir, 0, "w_qkv", w_qkv.data(), w_qkv.size())) return 1;
    if (!write_block_tensor(dir, 0, "b_qkv", b_qkv.data(), b_qkv.size())) return 1;
    if (!write_block_tensor(dir, 0, "w_o", w_o.data(), w_o.size())) return 1;
    if (!write_block_tensor(dir, 0, "b_o", b_o.data(), b_o.size())) return 1;
    if (!write_block_tensor(dir, 0, "ln2_gamma", ln2_gamma.data(), ln2_gamma.size())) return 1;
    if (!write_block_tensor(dir, 0, "ln2_beta", ln2_beta.data(), ln2_beta.size())) return 1;
    if (!write_block_tensor(dir, 0, "w_fc1", w_fc1.data(), w_fc1.size())) return 1;
    if (!write_block_tensor(dir, 0, "b_fc1", b_fc1.data(), b_fc1.size())) return 1;
    if (!write_block_tensor(dir, 0, "w_fc2", w_fc2.data(), w_fc2.size())) return 1;
    if (!write_block_tensor(dir, 0, "b_fc2", b_fc2.data(), b_fc2.size())) return 1;

    GPT2Weights weights = {};
    if (!load_gpt2_weights_from_dir(&weights, cfg, dir)) {
        fprintf(stderr, "load_gpt2_weights_from_dir failed\n");
        return 1;
    }

    bool ok = true;
    ok = ok && copy_and_check("wte", weights.wte, wte.data(), wte.size());
    ok = ok && copy_and_check("wpe", weights.wpe, wpe.data(), wpe.size());
    ok = ok && copy_and_check("ln_f_gamma", weights.ln_f_gamma, ln_f_gamma.data(), ln_f_gamma.size());
    ok = ok && copy_and_check("ln_f_beta", weights.ln_f_beta, ln_f_beta.data(), ln_f_beta.size());
    ok = ok && copy_and_check("lm_head", weights.lm_head, lm_head.data(), lm_head.size());
    ok = ok && copy_and_check("ln1_gamma", weights.blocks[0].ln1_gamma, ln1_gamma.data(), ln1_gamma.size());
    ok = ok && copy_and_check("ln1_beta", weights.blocks[0].ln1_beta, ln1_beta.data(), ln1_beta.size());
    ok = ok && copy_and_check("w_qkv", weights.blocks[0].w_qkv, w_qkv.data(), w_qkv.size());
    ok = ok && copy_and_check("b_qkv", weights.blocks[0].b_qkv, b_qkv.data(), b_qkv.size());
    ok = ok && copy_and_check("w_o", weights.blocks[0].w_o, w_o.data(), w_o.size());
    ok = ok && copy_and_check("b_o", weights.blocks[0].b_o, b_o.data(), b_o.size());
    ok = ok && copy_and_check("ln2_gamma", weights.blocks[0].ln2_gamma, ln2_gamma.data(), ln2_gamma.size());
    ok = ok && copy_and_check("ln2_beta", weights.blocks[0].ln2_beta, ln2_beta.data(), ln2_beta.size());
    ok = ok && copy_and_check("w_fc1", weights.blocks[0].w_fc1, w_fc1.data(), w_fc1.size());
    ok = ok && copy_and_check("b_fc1", weights.blocks[0].b_fc1, b_fc1.data(), b_fc1.size());
    ok = ok && copy_and_check("w_fc2", weights.blocks[0].w_fc2, w_fc2.data(), w_fc2.size());
    ok = ok && copy_and_check("b_fc2", weights.blocks[0].b_fc2, b_fc2.data(), b_fc2.size());

    destroy_gpt2_weights(&weights, cfg);

    if (!ok) {
        return 1;
    }

    printf("weight_loader_test passed\n");
    return 0;
}
