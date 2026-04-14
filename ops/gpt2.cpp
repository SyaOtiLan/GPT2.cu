#include <cstddef>

#include "../include/gpt2.h"
#include "../include/embedding.h"
#include "../include/gemm.h"
#include "../include/kernels.h"

static float* layer_cache_ptr(float* cache, int layer, const GPT2Workspace& ws, const GPT2Config& cfg) {
    return cache + (size_t)layer * ws.max_tokens * cfg.hidden;
}

void gpt2_forward(
    const int* input_ids,
    float* logits,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int tokens)
{
    if (!validate_config(cfg)) {
        return;
    }
    if (tokens <= 0 || tokens > cfg.max_position || tokens > ws.max_tokens) {
        return;
    }

    // 1) embedding
    launch_embedding(input_ids, w.wte, w.wpe, ws.x, tokens, cfg.hidden);

    // 2) block stack
    const float* cur_in = ws.x;
    float* cur_out = ws.block_out;

    for (int l = 0; l < cfg.n_layer; ++l) {
        transformer_block_forward(
            cur_in,
            cur_out,
            w.blocks[l],
            tokens,
            cfg.hidden,
            cfg.heads,
            ws.block_ws
        );

        // swap
        const float* tmp_in = cur_out;
        cur_out = (cur_out == ws.block_out) ? ws.block_in : ws.block_out;
        cur_in = tmp_in;
    }

    // 3) final LN
    launch_layernorm(cur_in, ws.ln_f_out, w.ln_f_gamma, w.ln_f_beta, tokens, cfg.hidden);

    // 4) lm_head
    // [tokens, hidden] x [hidden, vocab_size] -> [tokens, vocab_size]
    gemm(ws.ln_f_out, w.lm_head, logits, tokens, cfg.vocab_size, cfg.hidden);
}

void gpt2_prefill(
    const int* input_ids,
    float* logits,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int tokens)
{
    if (!validate_config(cfg)) {
        return;
    }
    if (tokens <= 0 || tokens > cfg.max_position || tokens > ws.max_tokens) {
        return;
    }
    if (ws.k_cache == nullptr || ws.v_cache == nullptr) {
        return;
    }

    // 1) embedding for the whole prompt
    launch_embedding(input_ids, w.wte, w.wpe, ws.x, tokens, cfg.hidden);

    // 2) block stack, while seeding KV cache for each layer
    const float* cur_in = ws.x;
    float* cur_out = ws.block_out;

    for (int l = 0; l < cfg.n_layer; ++l) {
        transformer_block_prefill(
            cur_in,
            cur_out,
            w.blocks[l],
            tokens,
            cfg.hidden,
            cfg.heads,
            ws.max_tokens,
            layer_cache_ptr(ws.k_cache, l, ws, cfg),
            layer_cache_ptr(ws.v_cache, l, ws, cfg),
            ws.block_ws
        );

        const float* tmp_in = cur_out;
        cur_out = (cur_out == ws.block_out) ? ws.block_in : ws.block_out;
        cur_in = tmp_in;
    }

    launch_layernorm(cur_in, ws.ln_f_out, w.ln_f_gamma, w.ln_f_beta, tokens, cfg.hidden);
    gemm(ws.ln_f_out, w.lm_head, logits, tokens, cfg.vocab_size, cfg.hidden);
}

void gpt2_decode_step(
    const int* input_id,
    float* logits,
    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg,
    int position)
{
    if (!validate_config(cfg)) {
        return;
    }
    if (position < 0 || position >= cfg.max_position || position >= ws.max_tokens) {
        return;
    }
    if (ws.k_cache == nullptr || ws.v_cache == nullptr) {
        return;
    }

    // Embed just the newest token, but use its absolute position.
    launch_embedding(input_id, w.wte, w.wpe + (size_t)position * cfg.hidden, ws.x, 1, cfg.hidden);

    const float* cur_in = ws.x;
    float* cur_out = ws.block_out;

    for (int l = 0; l < cfg.n_layer; ++l) {
        transformer_block_decode(
            cur_in,
            cur_out,
            w.blocks[l],
            position,
            cfg.hidden,
            cfg.heads,
            ws.max_tokens,
            layer_cache_ptr(ws.k_cache, l, ws, cfg),
            layer_cache_ptr(ws.v_cache, l, ws, cfg),
            ws.block_ws
        );

        const float* tmp_in = cur_out;
        cur_out = (cur_out == ws.block_out) ? ws.block_in : ws.block_out;
        cur_in = tmp_in;
    }

    launch_layernorm(cur_in, ws.ln_f_out, w.ln_f_gamma, w.ln_f_beta, 1, cfg.hidden);
    gemm(ws.ln_f_out, w.lm_head, logits, 1, cfg.vocab_size, cfg.hidden);
}
