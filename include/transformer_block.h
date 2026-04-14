#pragma once

struct BlockWeights {
    const float* ln1_gamma;
    const float* ln1_beta;

    const float* w_qkv;
    const float* b_qkv;
    const float* w_o;
    const float* b_o;

    const float* ln2_gamma;
    const float* ln2_beta;

    const float* w_fc1;
    const float* b_fc1;
    const float* w_fc2;
    const float* b_fc2;
};

struct BlockWorkspace {
    // LN1
    float* ln1_out;      // [tokens, hidden]

    // Attention
    float* qkv;          // [tokens, 3*hidden]
    float* q;            // [heads, tokens, head_dim]
    float* k;            // [heads, tokens, head_dim]
    float* v;            // [heads, tokens, head_dim]
    float* k_t;          // [head_dim, max_tokens]
    float* scores;       // [heads, tokens, tokens]
    float* probs;        // [heads, tokens, tokens]
    float* context;      // [heads, tokens, head_dim]
    float* merged;       // [tokens, hidden]
    float* attn_proj;    // [tokens, hidden]
    float* attn_out;     // [tokens, hidden]

    // LN2
    float* ln2_out;      // [tokens, hidden]

    // MLP
    float* fc1_out;      // [tokens, 4*hidden]
    float* gelu_out;     // [tokens, 4*hidden]
    float* fc2_out;      // [tokens, hidden]
};

void transformer_block_forward(
    const float* x,      // [tokens, hidden]
    float* out,          // [tokens, hidden]
    const BlockWeights& w,
    int tokens,
    int hidden,
    int heads,
    BlockWorkspace& ws
);

void transformer_block_prefill(
    const float* x,      // [tokens, hidden]
    float* out,          // [tokens, hidden]
    const BlockWeights& w,
    int tokens,
    int hidden,
    int heads,
    int max_tokens,
    float* k_cache,      // [heads, max_tokens, head_dim]
    float* v_cache,      // [heads, max_tokens, head_dim]
    BlockWorkspace& ws
);

void transformer_block_decode(
    const float* x,      // [1, hidden]
    float* out,          // [1, hidden]
    const BlockWeights& w,
    int position,
    int hidden,
    int heads,
    int max_tokens,
    float* k_cache,      // [heads, max_tokens, head_dim]
    float* v_cache,      // [heads, max_tokens, head_dim]
    BlockWorkspace& ws
);
