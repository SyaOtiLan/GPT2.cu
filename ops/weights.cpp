#include <cuda_runtime.h>
#include <new>
#include <stdio.h>
#include <vector>

#include "../include/gpt2.h"

static void reset_block_weights(BlockWeights* w) {
    w->ln1_gamma = nullptr;
    w->ln1_beta = nullptr;
    w->w_qkv = nullptr;
    w->b_qkv = nullptr;
    w->w_o = nullptr;
    w->b_o = nullptr;
    w->ln2_gamma = nullptr;
    w->ln2_beta = nullptr;
    w->w_fc1 = nullptr;
    w->b_fc1 = nullptr;
    w->w_fc2 = nullptr;
    w->b_fc2 = nullptr;
}

static void reset_gpt2_weights(GPT2Weights* w) {
    w->wte = nullptr;
    w->wpe = nullptr;
    w->blocks = nullptr;
    w->ln_f_gamma = nullptr;
    w->ln_f_beta = nullptr;
    w->lm_head = nullptr;
}

static void safe_cuda_free(const float*& ptr) {
    if (ptr != nullptr) {
        cudaFree((void*)ptr);
        ptr = nullptr;
    }
}

static bool make_path(char* out, size_t out_size, const char* dir, const char* file_name) {
    int written = snprintf(out, out_size, "%s/%s", dir, file_name);
    return written > 0 && (size_t)written < out_size;
}

static bool make_block_file_name(char* out, size_t out_size, int layer, const char* tensor_name) {
    int written = snprintf(out, out_size, "block_%d_%s.bin", layer, tensor_name);
    return written > 0 && (size_t)written < out_size;
}

static bool load_tensor_from_file(const char* path, size_t count, const float** dst) {
    FILE* fp = fopen(path, "rb");
    if (fp == nullptr) {
        fprintf(stderr, "load_tensor_from_file: failed to open %s\n", path);
        return false;
    }

    std::vector<float> host(count);
    size_t items_read = fread(host.data(), sizeof(float), count, fp);
    bool exact_size = (items_read == count) && (fgetc(fp) == EOF);
    fclose(fp);
    if (!exact_size) {
        fprintf(stderr, "load_tensor_from_file: size mismatch for %s\n", path);
        return false;
    }

    float* device = nullptr;
    cudaError_t err = cudaMalloc((void**)&device, count * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "load_tensor_from_file: cudaMalloc failed for %s: %s\n", path, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(device, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "load_tensor_from_file: cudaMemcpy failed for %s: %s\n", path, cudaGetErrorString(err));
        cudaFree(device);
        return false;
    }

    *dst = device;
    return true;
}

static bool load_named_tensor(
    const char* dir,
    const char* file_name,
    size_t count,
    const float** dst)
{
    char path[1024];
    if (!make_path(path, sizeof(path), dir, file_name)) {
        return false;
    }
    return load_tensor_from_file(path, count, dst);
}

static bool load_block_tensor(
    const char* dir,
    int layer,
    const char* tensor_name,
    size_t count,
    const float** dst)
{
    char file_name[256];
    char path[1024];
    if (!make_block_file_name(file_name, sizeof(file_name), layer, tensor_name)) {
        return false;
    }
    if (!make_path(path, sizeof(path), dir, file_name)) {
        return false;
    }
    return load_tensor_from_file(path, count, dst);
}

bool load_gpt2_weights_from_dir(GPT2Weights* w, const GPT2Config& cfg, const char* dir) {
    if (w == nullptr || dir == nullptr) {
        fprintf(stderr, "load_gpt2_weights_from_dir: null argument\n");
        return false;
    }
    if (!validate_config(cfg)) {
        fprintf(stderr, "load_gpt2_weights_from_dir: invalid config\n");
        return false;
    }

    reset_gpt2_weights(w);

    w->blocks = new (std::nothrow) BlockWeights[cfg.n_layer];
    if (w->blocks == nullptr) {
        fprintf(stderr, "load_gpt2_weights_from_dir: failed to allocate block table\n");
        return false;
    }
    for (int layer = 0; layer < cfg.n_layer; ++layer) {
        reset_block_weights(&w->blocks[layer]);
    }

    if (!load_named_tensor(dir, "wte.bin", (size_t)cfg.vocab_size * cfg.hidden, &w->wte)) goto fail;
    if (!load_named_tensor(dir, "wpe.bin", (size_t)cfg.max_position * cfg.hidden, &w->wpe)) goto fail;
    if (!load_named_tensor(dir, "ln_f_gamma.bin", cfg.hidden, &w->ln_f_gamma)) goto fail;
    if (!load_named_tensor(dir, "ln_f_beta.bin", cfg.hidden, &w->ln_f_beta)) goto fail;
    if (!load_named_tensor(dir, "lm_head.bin", (size_t)cfg.hidden * cfg.vocab_size, &w->lm_head)) goto fail;

    for (int layer = 0; layer < cfg.n_layer; ++layer) {
        BlockWeights& bw = w->blocks[layer];
        if (!load_block_tensor(dir, layer, "ln1_gamma", cfg.hidden, &bw.ln1_gamma)) goto fail;
        if (!load_block_tensor(dir, layer, "ln1_beta", cfg.hidden, &bw.ln1_beta)) goto fail;
        if (!load_block_tensor(dir, layer, "w_qkv", (size_t)cfg.hidden * 3 * cfg.hidden, &bw.w_qkv)) goto fail;
        if (!load_block_tensor(dir, layer, "b_qkv", 3 * cfg.hidden, &bw.b_qkv)) goto fail;
        if (!load_block_tensor(dir, layer, "w_o", (size_t)cfg.hidden * cfg.hidden, &bw.w_o)) goto fail;
        if (!load_block_tensor(dir, layer, "b_o", cfg.hidden, &bw.b_o)) goto fail;
        if (!load_block_tensor(dir, layer, "ln2_gamma", cfg.hidden, &bw.ln2_gamma)) goto fail;
        if (!load_block_tensor(dir, layer, "ln2_beta", cfg.hidden, &bw.ln2_beta)) goto fail;
        if (!load_block_tensor(dir, layer, "w_fc1", (size_t)cfg.hidden * 4 * cfg.hidden, &bw.w_fc1)) goto fail;
        if (!load_block_tensor(dir, layer, "b_fc1", 4 * cfg.hidden, &bw.b_fc1)) goto fail;
        if (!load_block_tensor(dir, layer, "w_fc2", (size_t)4 * cfg.hidden * cfg.hidden, &bw.w_fc2)) goto fail;
        if (!load_block_tensor(dir, layer, "b_fc2", cfg.hidden, &bw.b_fc2)) goto fail;
    }

    return true;

fail:
    destroy_gpt2_weights(w, cfg);
    return false;
}

void destroy_gpt2_weights(GPT2Weights* w, const GPT2Config& cfg) {
    if (w == nullptr) {
        return;
    }

    safe_cuda_free(w->wte);
    safe_cuda_free(w->wpe);
    safe_cuda_free(w->ln_f_gamma);
    safe_cuda_free(w->ln_f_beta);
    safe_cuda_free(w->lm_head);

    if (w->blocks != nullptr) {
        for (int layer = 0; layer < cfg.n_layer; ++layer) {
            BlockWeights& bw = w->blocks[layer];
            safe_cuda_free(bw.ln1_gamma);
            safe_cuda_free(bw.ln1_beta);
            safe_cuda_free(bw.w_qkv);
            safe_cuda_free(bw.b_qkv);
            safe_cuda_free(bw.w_o);
            safe_cuda_free(bw.b_o);
            safe_cuda_free(bw.ln2_gamma);
            safe_cuda_free(bw.ln2_beta);
            safe_cuda_free(bw.w_fc1);
            safe_cuda_free(bw.b_fc1);
            safe_cuda_free(bw.w_fc2);
            safe_cuda_free(bw.b_fc2);
        }
        delete[] w->blocks;
        w->blocks = nullptr;
    }
}
