#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "../include/gpt2.h"

struct BenchArgs {
    std::string model_dir = "models/gpt2-bin";
    int prompt_len = 32;
    int gen_len = 32;
    int warmup = 1;
    int iters = 5;
};

struct KVBenchResult {
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
};

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static bool parse_int(const char* text, int* value) {
    if (text == nullptr || value == nullptr) {
        return false;
    }

    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *value = (int)parsed;
    return true;
}

static void print_usage(const char* argv0) {
    printf(
        "Usage: %s [--model-dir DIR] [--prompt-len N] [--gen-len N] [--warmup N] [--iters N]\n",
        argv0
    );
}

static BenchArgs parse_args(int argc, char** argv) {
    BenchArgs args;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--model-dir requires a path\n");
                exit(1);
            }
            args.model_dir = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--prompt-len") == 0) {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &args.prompt_len) || args.prompt_len <= 0) {
                fprintf(stderr, "--prompt-len requires a positive integer\n");
                exit(1);
            }
            ++i;
            continue;
        }
        if (strcmp(argv[i], "--gen-len") == 0) {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &args.gen_len) || args.gen_len < 0) {
                fprintf(stderr, "--gen-len requires a non-negative integer\n");
                exit(1);
            }
            ++i;
            continue;
        }
        if (strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &args.warmup) || args.warmup < 0) {
                fprintf(stderr, "--warmup requires a non-negative integer\n");
                exit(1);
            }
            ++i;
            continue;
        }
        if (strcmp(argv[i], "--iters") == 0) {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &args.iters) || args.iters <= 0) {
                fprintf(stderr, "--iters requires a positive integer\n");
                exit(1);
            }
            ++i;
            continue;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }

        fprintf(stderr, "unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        exit(1);
    }

    return args;
}

static std::vector<int> make_input_ids(int total_len, int vocab_size) {
    const int seed_ids[] = {15496, 11, 314, 716, 257, 1310, 1643, 286, 262, 290, 287, 340};
    const int seed_len = (int)(sizeof(seed_ids) / sizeof(seed_ids[0]));

    std::vector<int> ids(total_len);
    for (int i = 0; i < total_len; ++i) {
        ids[i] = seed_ids[i % seed_len] % vocab_size;
    }
    return ids;
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    return ms;
}

static void run_full_recompute_once(
    const int* d_input_ids,
    int prompt_len,
    int gen_len,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg)
{
    for (int step = 0; step < gen_len; ++step) {
        int cur_len = prompt_len + step;
        gpt2_forward(d_input_ids, ws.logits, w, ws, cfg, cur_len);
    }
}

static float benchmark_full_recompute(
    const int* d_input_ids,
    int prompt_len,
    int gen_len,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int warmup,
    int iters)
{
    if (gen_len == 0) {
        return 0.0f;
    }

    for (int i = 0; i < warmup; ++i) {
        run_full_recompute_once(d_input_ids, prompt_len, gen_len, w, ws, cfg);
    }
    check_cuda(cudaDeviceSynchronize(), "sync after full-recompute warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create full start event");
    check_cuda(cudaEventCreate(&stop), "create full stop event");

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaEventRecord(start), "record full start");
        run_full_recompute_once(d_input_ids, prompt_len, gen_len, w, ws, cfg);
        check_cuda(cudaEventRecord(stop), "record full stop");
        check_cuda(cudaEventSynchronize(stop), "sync full stop event");
        total_ms += elapsed_ms(start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / iters;
}

static void run_kv_once(
    const int* d_input_ids,
    int prompt_len,
    int gen_len,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg)
{
    gpt2_prefill(d_input_ids, ws.logits, w, ws, cfg, prompt_len);
    for (int step = 0; step < gen_len; ++step) {
        int position = prompt_len + step;
        gpt2_decode_step(d_input_ids + position, ws.logits, w, ws, cfg, position);
    }
}

static KVBenchResult benchmark_kv_cache(
    const int* d_input_ids,
    int prompt_len,
    int gen_len,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int warmup,
    int iters)
{
    for (int i = 0; i < warmup; ++i) {
        run_kv_once(d_input_ids, prompt_len, gen_len, w, ws, cfg);
    }
    check_cuda(cudaDeviceSynchronize(), "sync after kv warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create kv start event");
    check_cuda(cudaEventCreate(&stop), "create kv stop event");

    float total_prefill_ms = 0.0f;
    float total_decode_ms = 0.0f;

    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaEventRecord(start), "record prefill start");
        gpt2_prefill(d_input_ids, ws.logits, w, ws, cfg, prompt_len);
        check_cuda(cudaEventRecord(stop), "record prefill stop");
        check_cuda(cudaEventSynchronize(stop), "sync prefill stop event");
        total_prefill_ms += elapsed_ms(start, stop);

        check_cuda(cudaEventRecord(start), "record decode start");
        for (int step = 0; step < gen_len; ++step) {
            int position = prompt_len + step;
            gpt2_decode_step(d_input_ids + position, ws.logits, w, ws, cfg, position);
        }
        check_cuda(cudaEventRecord(stop), "record decode stop");
        check_cuda(cudaEventSynchronize(stop), "sync decode stop event");
        total_decode_ms += elapsed_ms(start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    KVBenchResult result;
    result.prefill_ms = total_prefill_ms / iters;
    result.decode_ms = total_decode_ms / iters;
    return result;
}

int main(int argc, char** argv) {
    BenchArgs args = parse_args(argc, argv);

    std::string config_path = args.model_dir + "/gpt2_config.json";
    GPT2Config cfg = {};
    if (!load_gpt2_config_from_json(&cfg, config_path.c_str())) {
        fprintf(stderr, "failed to load config from %s\n", config_path.c_str());
        return 1;
    }

    const int total_len = args.prompt_len + args.gen_len;
    if (total_len > cfg.max_position) {
        fprintf(stderr, "total length %d exceeds max_position %d\n", total_len, cfg.max_position);
        return 1;
    }

    std::vector<int> input_ids = make_input_ids(total_len, cfg.vocab_size);

    GPT2Weights w = {};
    if (!load_gpt2_weights_from_dir(&w, cfg, args.model_dir.c_str())) {
        fprintf(stderr, "failed to load weights from %s\n", args.model_dir.c_str());
        return 1;
    }

    GPT2Workspace ws_full;
    if (!create_gpt2_workspace(&ws_full, cfg, total_len)) {
        fprintf(stderr, "failed to create full benchmark workspace\n");
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    GPT2Workspace ws_kv;
    if (!create_gpt2_workspace(&ws_kv, cfg, total_len)) {
        fprintf(stderr, "failed to create kv benchmark workspace\n");
        destroy_gpt2_workspace(&ws_full);
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    int* d_input_ids = nullptr;
    check_cuda(cudaMalloc(&d_input_ids, total_len * sizeof(int)), "malloc d_input_ids");
    check_cuda(
        cudaMemcpy(d_input_ids, input_ids.data(), total_len * sizeof(int), cudaMemcpyHostToDevice),
        "copy benchmark input ids"
    );

    float full_ms = benchmark_full_recompute(
        d_input_ids,
        args.prompt_len,
        args.gen_len,
        w,
        ws_full,
        cfg,
        args.warmup,
        args.iters
    );
    KVBenchResult kv = benchmark_kv_cache(
        d_input_ids,
        args.prompt_len,
        args.gen_len,
        w,
        ws_kv,
        cfg,
        args.warmup,
        args.iters
    );

    float kv_total_ms = kv.prefill_ms + kv.decode_ms;
    float decode_ms_per_token = (args.gen_len > 0) ? (kv.decode_ms / args.gen_len) : 0.0f;
    float full_ms_per_token = (args.gen_len > 0) ? (full_ms / args.gen_len) : 0.0f;
    float kv_speedup = (kv_total_ms > 0.0f) ? (full_ms / kv_total_ms) : 0.0f;
    float decode_tokens_per_s = (kv.decode_ms > 0.0f) ? (1000.0f * args.gen_len / kv.decode_ms) : 0.0f;

    printf("model_dir: %s\n", args.model_dir.c_str());
    printf(
        "config: hidden=%d heads=%d layers=%d vocab=%d max_position=%d\n",
        cfg.hidden,
        cfg.heads,
        cfg.n_layer,
        cfg.vocab_size,
        cfg.max_position
    );
    printf(
        "benchmark: prompt_len=%d gen_len=%d warmup=%d iters=%d\n",
        args.prompt_len,
        args.gen_len,
        args.warmup,
        args.iters
    );
    printf("full_recompute_total_ms: %.3f\n", full_ms);
    printf("full_recompute_ms_per_token: %.3f\n", full_ms_per_token);
    printf("kv_prefill_ms: %.3f\n", kv.prefill_ms);
    printf("kv_decode_total_ms: %.3f\n", kv.decode_ms);
    printf("kv_decode_ms_per_token: %.3f\n", decode_ms_per_token);
    printf("kv_total_ms: %.3f\n", kv_total_ms);
    printf("kv_speedup_vs_full: %.3fx\n", kv_speedup);
    printf("kv_decode_tokens_per_s: %.2f\n", decode_tokens_per_s);

    cudaFree(d_input_ids);
    destroy_gpt2_workspace(&ws_kv);
    destroy_gpt2_workspace(&ws_full);
    destroy_gpt2_weights(&w, cfg);
    return 0;
}
