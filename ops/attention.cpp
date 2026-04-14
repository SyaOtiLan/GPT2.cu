#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/attention.h"
#include "../include/gemm.h"
#include "../include/kernels.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void attention_forward(
    const float* x,
    const float* residual,
    float* out,
    const float* w_qkv,
    const float* b_qkv,
    const float* w_o,
    const float* b_o,
    int tokens,
    int hidden,
    int heads,
    float* qkv,
    float* q,
    float* k,
    float* v,
    float* scores,
    float* probs,
    float* context,
    float* merged,
    float* proj_out)
{
    int head_dim = hidden / heads;

    // 1) qkv projection
    // [tokens, hidden] x [hidden, 3*hidden] -> [tokens, 3*hidden]
    gemm(x, w_qkv, qkv, tokens, 3 * hidden, hidden);
    launch_add_bias(qkv, b_qkv, tokens, 3 * hidden);

    // 2) split / reshape
    launch_qkv_split_reshape(qkv, q, k, v, tokens, hidden, heads);

    // 3) per-head attention
    for (int h = 0; h < heads; ++h) {
        const float* q_h = q + h * tokens * head_dim;         // [tokens, head_dim]
        const float* k_h = k + h * tokens * head_dim;         // [tokens, head_dim]
        const float* v_h = v + h * tokens * head_dim;         // [tokens, head_dim]

        float* scores_h  = scores  + h * tokens * tokens;     // [tokens, tokens]
        float* probs_h   = probs   + h * tokens * tokens;     // [tokens, tokens]
        float* context_h = context + h * tokens * head_dim;   // [tokens, head_dim]

        // scores = Q * K^T
        gemm_qk(q_h, k_h, scores_h, tokens, head_dim);

        // scale by 1/sqrt(head_dim)
        launch_scale(scores_h, 1.0f / sqrtf((float)head_dim), tokens * tokens);

        // causal mask
        launch_causal_mask(scores_h, tokens, tokens);

        // softmax
        launch_softmax(scores_h, probs_h, tokens, tokens);

        // context = P * V
        gemm_pv(probs_h, v_h, context_h, tokens, head_dim);
    }

    // 4) merge heads
    launch_merge_heads(context, merged, tokens, heads, head_dim);

    // 5) output projection
    // [tokens, hidden] x [hidden, hidden] -> [tokens, hidden]
    gemm(merged, w_o, proj_out, tokens, hidden, hidden);
    launch_add_bias(proj_out, b_o, tokens, hidden);

    // 6) residual add
    launch_residual_add(residual, proj_out, out, tokens * hidden);
}

void attention_write_kv_cache(
    const float* k,
    const float* v,
    float* k_cache,
    float* v_cache,
    int heads,
    int tokens,
    int head_dim,
    int cache_start,
    int cache_capacity)
{
    if (k == nullptr || v == nullptr || k_cache == nullptr || v_cache == nullptr) {
        fprintf(stderr, "attention_write_kv_cache: null pointer\n");
        exit(1);
    }
    if (tokens <= 0 || head_dim <= 0 || heads <= 0) {
        fprintf(stderr, "attention_write_kv_cache: invalid shape\n");
        exit(1);
    }
    if (cache_start < 0 || cache_start + tokens > cache_capacity) {
        fprintf(stderr, "attention_write_kv_cache: cache range out of bounds\n");
        exit(1);
    }

    size_t elems_per_copy = (size_t)tokens * head_dim;
    for (int h = 0; h < heads; ++h) {
        const float* k_src = k + (size_t)h * tokens * head_dim;
        const float* v_src = v + (size_t)h * tokens * head_dim;
        float* k_dst = k_cache + ((size_t)h * cache_capacity + cache_start) * head_dim;
        float* v_dst = v_cache + ((size_t)h * cache_capacity + cache_start) * head_dim;

        check_cuda(
            cudaMemcpy(k_dst, k_src, elems_per_copy * sizeof(float), cudaMemcpyDeviceToDevice),
            "copy k -> kv cache"
        );
        check_cuda(
            cudaMemcpy(v_dst, v_src, elems_per_copy * sizeof(float), cudaMemcpyDeviceToDevice),
            "copy v -> kv cache"
        );
    }
}

void attention_forward_decode(
    const float* x,
    const float* residual,
    float* out,
    const float* w_qkv,
    const float* b_qkv,
    const float* w_o,
    const float* b_o,
    int position,
    int hidden,
    int heads,
    int max_tokens,
    float* qkv,
    float* q,
    float* k,
    float* v,
    float* scores,
    float* probs,
    float* context,
    float* merged,
    float* proj_out,
    float* k_t,
    float* k_cache,
    float* v_cache)
{
    if (position < 0 || position >= max_tokens) {
        fprintf(stderr, "attention_forward_decode: position out of range\n");
        exit(1);
    }

    int total_tokens = position + 1;
    int head_dim = hidden / heads;

    // Current token projection.
    gemm(x, w_qkv, qkv, 1, 3 * hidden, hidden);
    launch_add_bias(qkv, b_qkv, 1, 3 * hidden);
    launch_qkv_split_reshape(qkv, q, k, v, 1, hidden, heads);

    // Append this token's K/V into the persistent cache.
    attention_write_kv_cache(k, v, k_cache, v_cache, heads, 1, head_dim, position, max_tokens);

    for (int h = 0; h < heads; ++h) {
        const float* q_h = q + (size_t)h * head_dim;                            // [1, head_dim]
        const float* k_cache_h = k_cache + (size_t)h * max_tokens * head_dim;   // [max_tokens, head_dim]
        const float* v_cache_h = v_cache + (size_t)h * max_tokens * head_dim;   // [max_tokens, head_dim]

        float* scores_h = scores + (size_t)h * max_tokens * max_tokens;         // use first row
        float* probs_h = probs + (size_t)h * max_tokens * max_tokens;           // use first row
        float* context_h = context + (size_t)h * head_dim;                      // [1, head_dim]

        // scores = q * K_cache^T, only over valid cache positions [0, total_tokens).
        launch_transpose(k_cache_h, k_t, total_tokens, head_dim);
        gemm(q_h, k_t, scores_h, 1, total_tokens, head_dim);

        launch_scale(scores_h, 1.0f / sqrtf((float)head_dim), total_tokens);
        launch_softmax(scores_h, probs_h, 1, total_tokens);

        // context = probs * V_cache
        gemm(probs_h, v_cache_h, context_h, 1, head_dim, total_tokens);
    }

    launch_merge_heads(context, merged, 1, heads, head_dim);
    gemm(merged, w_o, proj_out, 1, hidden, hidden);
    launch_add_bias(proj_out, b_o, 1, hidden);
    launch_residual_add(residual, proj_out, out, hidden);
}
