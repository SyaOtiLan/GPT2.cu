#pragma once
#include "transformer_block.h"

struct GPT2Config {
    int hidden;
    int heads;
    int n_layer;
    int vocab_size;
    int max_position;
};

bool validate_config(const GPT2Config& cfg);
bool load_gpt2_config_from_json(GPT2Config* cfg, const char* path);

struct GPT2Weights {
    const float* wte;        // [vocab_size, hidden]
    const float* wpe;        // [max_position, hidden]

    BlockWeights* blocks;    // [n_layer]

    const float* ln_f_gamma; // [hidden]
    const float* ln_f_beta;  // [hidden]

    const float* lm_head;    // [hidden, vocab_size]
};

// Load weights from a directory of raw float32 .bin files.
// Caller should zero-initialize GPT2Weights before first load, and call
// destroy_gpt2_weights before reusing the same object for another load.
bool load_gpt2_weights_from_dir(GPT2Weights* w, const GPT2Config& cfg, const char* dir);
void destroy_gpt2_weights(GPT2Weights* w, const GPT2Config& cfg);

struct GPT2Workspace {
    int max_tokens;
    float* x;            // [tokens, hidden]
    float* block_in;     // [tokens, hidden]
    float* block_out;    // [tokens, hidden]
    float* ln_f_out;     // [tokens, hidden]
    float* logits;       // [tokens, vocab_size]
    float* k_cache;      // [n_layer, heads, max_tokens, head_dim]
    float* v_cache;      // [n_layer, heads, max_tokens, head_dim]

    BlockWorkspace block_ws;
};

bool create_gpt2_workspace(GPT2Workspace* ws, const GPT2Config& cfg, int max_tokens);
void destroy_gpt2_workspace(GPT2Workspace* ws);

void gpt2_forward(
    const int* input_ids,   // [tokens]
    float* logits,          // [tokens, vocab_size]
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int tokens
);

void gpt2_prefill(
    const int* input_ids,   // [tokens]
    float* logits,          // [tokens, vocab_size]
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int tokens
);

void gpt2_decode_step(
    const int* input_id,    // [1]
    float* logits,          // [1, vocab_size]
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int position
);
