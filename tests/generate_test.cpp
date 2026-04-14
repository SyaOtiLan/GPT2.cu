#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "../include/generate.h"
#include "reference/cpu_reference.h"

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

int main() {
    const int input_len = 4;
    const int max_new_tokens = 4;

    const int hidden = 4;
    const int heads = 2;
    const int n_layer = 1;
    const int vocab_size = 8;
    const int max_position = 16;
    const char* weight_dir = "/tmp/generate_test_weights";

    GPT2Config cfg;
    cfg.hidden = hidden;
    cfg.heads = heads;
    cfg.n_layer = n_layer;
    cfg.vocab_size = vocab_size;
    cfg.max_position = max_position;

    int h_input_ids[input_len] = {1, 3, 2, 5};
    int h_output_ids[input_len + max_new_tokens];
    int h_output_ids_ref[input_len + max_new_tokens];

    // -------------------------
    // host weights
    // -------------------------
    float h_wte[vocab_size * hidden];
    float h_wpe[max_position * hidden];

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

    // -------------------------
    // block weights
    // -------------------------
    CPUBlockWeights cpu_blocks[n_layer];
    cpu_blocks[0].ln1_gamma = h_ln1_gamma;
    cpu_blocks[0].ln1_beta  = h_ln1_beta;
    cpu_blocks[0].w_qkv     = h_w_qkv;
    cpu_blocks[0].b_qkv     = h_b_qkv;
    cpu_blocks[0].w_o       = h_w_o;
    cpu_blocks[0].b_o       = h_b_o;
    cpu_blocks[0].ln2_gamma = h_ln2_gamma;
    cpu_blocks[0].ln2_beta  = h_ln2_beta;
    cpu_blocks[0].w_fc1     = h_w_fc1;
    cpu_blocks[0].b_fc1     = h_b_fc1;
    cpu_blocks[0].w_fc2     = h_w_fc2;
    cpu_blocks[0].b_fc2     = h_b_fc2;

    GPT2Weights w = {};
    if (!load_gpt2_weights_from_dir(&w, cfg, weight_dir)) {
        fprintf(stderr, "failed to load GPT2 weights from %s\n", weight_dir);
        return 1;
    }

    CPUGPT2Weights w_ref;
    w_ref.wte = h_wte;
    w_ref.wpe = h_wpe;
    w_ref.blocks = cpu_blocks;
    w_ref.ln_f_gamma = h_ln_f_gamma;
    w_ref.ln_f_beta  = h_ln_f_beta;
    w_ref.lm_head    = h_lm_head;

    // -------------------------
    // workspace allocate
    // -------------------------
    GPT2Workspace ws;
    if (!create_gpt2_workspace(&ws, cfg, max_position)) {
        fprintf(stderr, "failed to create GPT2 workspace\n");
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    // -------------------------
    // run greedy generate
    // -------------------------
    generate_greedy(
        h_input_ids,
        input_len,
        max_new_tokens,
        h_output_ids,
        w,
        ws,
        cfg
    );

    generate_greedy_cpu_ref(
        h_input_ids,
        input_len,
        max_new_tokens,
        h_output_ids_ref,
        w_ref,
        hidden,
        heads,
        n_layer,
        vocab_size
    );

    printf("input ids: ");
    for (int i = 0; i < input_len; ++i) {
        printf("%d ", h_input_ids[i]);
    }
    printf("\n");

    printf("generated ids: ");
    for (int i = 0; i < input_len + max_new_tokens; ++i) {
        printf("%d ", h_output_ids[i]);
    }
    printf("\n");

    if (!ids_in_range_cpu_ref(h_output_ids, input_len + max_new_tokens, vocab_size)) {
        fprintf(stderr, "Generated ids out of range\n");
        return 1;
    }
    for (int i = 0; i < input_len; ++i) {
        if (h_output_ids[i] != h_input_ids[i]) {
            fprintf(stderr, "Prefix token changed at position %d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < input_len + max_new_tokens; ++i) {
        if (h_output_ids[i] != h_output_ids_ref[i]) {
            fprintf(stderr, "Generated ids mismatch at position %d: got %d expected %d\n",
                i, h_output_ids[i], h_output_ids_ref[i]);
            return 1;
        }
    }

    // -------------------------
    // free
    // -------------------------
    destroy_gpt2_weights(&w, cfg);
    destroy_gpt2_workspace(&ws);

    return 0;
}
