#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/generate.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static int argmax_host(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

void generate_greedy(
    const int* input_ids_host,
    int input_len,
    int max_new_tokens,
    int* output_ids_host,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg
)
{
    if (!validate_config(cfg)) {
        fprintf(stderr, "generate_greedy: invalid GPT2Config\n");
        exit(1);
    }
    if (input_len <= 0) {
        fprintf(stderr, "generate_greedy: input_len must be > 0\n");
        exit(1);
    }
    if (max_new_tokens < 0) {
        fprintf(stderr, "generate_greedy: max_new_tokens must be >= 0\n");
        exit(1);
    }

    int total_len = input_len + max_new_tokens;
    if (total_len > cfg.max_position) {
        fprintf(stderr, "generate_greedy: total length exceeds max_position\n");
        exit(1);
    }
    if (total_len > ws.max_tokens) {
        fprintf(stderr, "generate_greedy: total length exceeds ws.max_tokens\n");
        exit(1);
    }
    if (ws.logits == nullptr) {
        fprintf(stderr, "generate_greedy: ws.logits is null\n");
        exit(1);
    }
    if (ws.k_cache == nullptr || ws.v_cache == nullptr) {
        fprintf(stderr, "generate_greedy: kv cache is null\n");
        exit(1);
    }

    // 先把初始输入拷到输出 buffer
    for (int i = 0; i < input_len; ++i) {
        output_ids_host[i] = input_ids_host[i];
    }

    if (max_new_tokens == 0) {
        return;
    }

    int* d_input_ids = nullptr;
    float* d_logits = ws.logits;
    float* last_logits_host = (float*)malloc(cfg.vocab_size * sizeof(float));
    if (last_logits_host == nullptr) {
        fprintf(stderr, "generate_greedy: failed to allocate host logits buffer\n");
        exit(1);
    }

    check_cuda(cudaMalloc(&d_input_ids, total_len * sizeof(int)), "cudaMalloc d_input_ids");
    check_cuda(
        cudaMemcpy(d_input_ids, input_ids_host, input_len * sizeof(int), cudaMemcpyHostToDevice),
        "memcpy prompt -> device"
    );

    // Prefill once on the whole prompt.
    gpt2_prefill(
        d_input_ids,
        d_logits,
        w,
        ws,
        cfg,
        input_len
    );
    check_cuda(cudaDeviceSynchronize(), "sync after gpt2_prefill");

    check_cuda(
        cudaMemcpy(
            last_logits_host,
            d_logits + (input_len - 1) * cfg.vocab_size,
            cfg.vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "memcpy prefill last logits -> host"
    );

    int next_token = argmax_host(last_logits_host, cfg.vocab_size);
    output_ids_host[input_len] = next_token;

    for (int step = 1; step < max_new_tokens; ++step) {
        int cur_pos = input_len + step - 1;

        check_cuda(
            cudaMemcpy(d_input_ids + cur_pos, output_ids_host + cur_pos, sizeof(int), cudaMemcpyHostToDevice),
            "memcpy decode token -> device"
        );

        gpt2_decode_step(
            d_input_ids + cur_pos,
            d_logits,
            w,
            ws,
            cfg,
            cur_pos
        );

        check_cuda(cudaDeviceSynchronize(), "sync after gpt2_decode_step");

        check_cuda(
            cudaMemcpy(
                last_logits_host,
                d_logits,
                cfg.vocab_size * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "memcpy decode logits -> host"
        );

        next_token = argmax_host(last_logits_host, cfg.vocab_size);
        output_ids_host[cur_pos + 1] = next_token;
    }

    free(last_logits_host);
    cudaFree(d_input_ids);
}
