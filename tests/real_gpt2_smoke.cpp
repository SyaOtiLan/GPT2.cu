#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "../include/gpt2.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

int main() {
    const char* model_dir = "models/gpt2-bin";
    const char* config_path = "models/gpt2-bin/gpt2_config.json";
    const int tokens = 4;
    int h_input_ids[tokens] = {15496, 11, 314, 716}; // "Hello, I am" in GPT-2 ids

    GPT2Config cfg = {};
    if (!load_gpt2_config_from_json(&cfg, config_path)) {
        fprintf(stderr, "failed to load config from %s\n", config_path);
        return 1;
    }

    if (tokens > cfg.max_position) {
        fprintf(stderr, "tokens exceeds max_position\n");
        return 1;
    }

    GPT2Weights w = {};
    if (!load_gpt2_weights_from_dir(&w, cfg, model_dir)) {
        fprintf(stderr, "failed to load weights from %s\n", model_dir);
        return 1;
    }

    GPT2Workspace ws;
    if (!create_gpt2_workspace(&ws, cfg, tokens)) {
        fprintf(stderr, "failed to create workspace\n");
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    int* d_input_ids = nullptr;
    check_cuda(cudaMalloc(&d_input_ids, tokens * sizeof(int)), "malloc d_input_ids");
    check_cuda(cudaMemcpy(d_input_ids, h_input_ids, tokens * sizeof(int), cudaMemcpyHostToDevice), "copy input ids");

    gpt2_forward(d_input_ids, ws.logits, w, ws, cfg, tokens);
    check_cuda(cudaDeviceSynchronize(), "sync after gpt2_forward");

    std::vector<float> last_logits(cfg.vocab_size);
    check_cuda(
        cudaMemcpy(
            last_logits.data(),
            ws.logits + (tokens - 1) * cfg.vocab_size,
            cfg.vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy last logits"
    );

    int next_token = argmax(last_logits.data(), cfg.vocab_size);
    printf("real_gpt2_smoke next token id: %d\n", next_token);
    printf("input ids: %d %d %d %d\n", h_input_ids[0], h_input_ids[1], h_input_ids[2], h_input_ids[3]);

    cudaFree(d_input_ids);
    destroy_gpt2_workspace(&ws);
    destroy_gpt2_weights(&w, cfg);
    return 0;
}
