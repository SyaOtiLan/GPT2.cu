#pragma once

void attention_forward(
    const float* x,          // normalized input [tokens, hidden]
    const float* residual,   // residual input [tokens, hidden]
    float* out,              // [tokens, hidden]

    const float* w_qkv,      // [hidden, 3*hidden]
    const float* b_qkv,      // [3*hidden]
    const float* w_o,        // [hidden, hidden]
    const float* b_o,        // [hidden]

    int tokens,
    int hidden,
    int heads,

    // workspace
    float* qkv,              // [tokens, 3*hidden]
    float* q,                // [heads, tokens, head_dim]
    float* k,                // [heads, tokens, head_dim]
    float* v,                // [heads, tokens, head_dim]
    float* scores,           // [heads, tokens, tokens]
    float* probs,            // [heads, tokens, tokens]
    float* context,          // [heads, tokens, head_dim]
    float* merged,           // [tokens, hidden]
    float* proj_out          // [tokens, hidden]
);

void attention_write_kv_cache(
    const float* k,          // [heads, tokens, head_dim]
    const float* v,          // [heads, tokens, head_dim]
    float* k_cache,          // [heads, max_tokens, head_dim]
    float* v_cache,          // [heads, max_tokens, head_dim]
    int heads,
    int tokens,
    int head_dim,
    int cache_start,
    int cache_capacity
);

void attention_forward_decode(
    const float* x,          // normalized input [1, hidden]
    const float* residual,   // residual input [1, hidden]
    float* out,              // [1, hidden]

    const float* w_qkv,      // [hidden, 3*hidden]
    const float* b_qkv,      // [3*hidden]
    const float* w_o,        // [hidden, hidden]
    const float* b_o,        // [hidden]

    int position,
    int hidden,
    int heads,
    int max_tokens,

    // workspace
    float* qkv,              // [1, 3*hidden]
    float* q,                // [heads, 1, head_dim]
    float* k,                // [heads, 1, head_dim]
    float* v,                // [heads, 1, head_dim]
    float* scores,           // [heads, max_tokens, max_tokens]
    float* probs,            // [heads, max_tokens, max_tokens]
    float* context,          // first heads*head_dim floats used as [heads, 1, head_dim]
    float* merged,           // [1, hidden]
    float* proj_out,         // [1, hidden]
    float* k_t,              // [head_dim, max_tokens]
    float* k_cache,          // [heads, max_tokens, head_dim]
    float* v_cache           // [heads, max_tokens, head_dim]
);
