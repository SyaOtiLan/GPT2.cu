#include <cuda_runtime.h>
#include <stddef.h>

#include "../include/gpt2.h"

static void safe_cuda_free(float*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

static void reset_block_workspace(BlockWorkspace* ws) {
    ws->ln1_out = nullptr;
    ws->qkv = nullptr;
    ws->q = nullptr;
    ws->k = nullptr;
    ws->v = nullptr;
    ws->k_t = nullptr;
    ws->scores = nullptr;
    ws->probs = nullptr;
    ws->context = nullptr;
    ws->merged = nullptr;
    ws->attn_proj = nullptr;
    ws->attn_out = nullptr;
    ws->ln2_out = nullptr;
    ws->fc1_out = nullptr;
    ws->gelu_out = nullptr;
    ws->fc2_out = nullptr;
}

static void reset_gpt2_workspace(GPT2Workspace* ws) {
    ws->max_tokens = 0;
    ws->x = nullptr;
    ws->block_in = nullptr;
    ws->block_out = nullptr;
    ws->ln_f_out = nullptr;
    ws->logits = nullptr;
    ws->k_cache = nullptr;
    ws->v_cache = nullptr;
    reset_block_workspace(&ws->block_ws);
}

static bool alloc_float_buffer(float** ptr, size_t count) {
    return cudaMalloc((void**)ptr, count * sizeof(float)) == cudaSuccess;
}

bool create_gpt2_workspace(GPT2Workspace* ws, const GPT2Config& cfg, int max_tokens) {
    if (ws == nullptr) {
        return false;
    }
    reset_gpt2_workspace(ws);

    if (!validate_config(cfg)) {
        return false;
    }
    if (max_tokens <= 0 || max_tokens > cfg.max_position) {
        return false;
    }

    int head_dim = cfg.hidden / cfg.heads;
    BlockWorkspace& bws = ws->block_ws;

    ws->max_tokens = max_tokens;

    if (!alloc_float_buffer(&ws->x, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&ws->block_in, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&ws->block_out, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&ws->ln_f_out, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&ws->logits, (size_t)max_tokens * cfg.vocab_size)) goto fail;
    if (!alloc_float_buffer(&ws->k_cache, (size_t)cfg.n_layer * max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&ws->v_cache, (size_t)cfg.n_layer * max_tokens * cfg.hidden)) goto fail;

    if (!alloc_float_buffer(&bws.ln1_out, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.qkv, (size_t)max_tokens * 3 * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.q, (size_t)cfg.heads * max_tokens * head_dim)) goto fail;
    if (!alloc_float_buffer(&bws.k, (size_t)cfg.heads * max_tokens * head_dim)) goto fail;
    if (!alloc_float_buffer(&bws.v, (size_t)cfg.heads * max_tokens * head_dim)) goto fail;
    if (!alloc_float_buffer(&bws.k_t, (size_t)max_tokens * head_dim)) goto fail;
    if (!alloc_float_buffer(&bws.scores, (size_t)cfg.heads * max_tokens * max_tokens)) goto fail;
    if (!alloc_float_buffer(&bws.probs, (size_t)cfg.heads * max_tokens * max_tokens)) goto fail;
    if (!alloc_float_buffer(&bws.context, (size_t)cfg.heads * max_tokens * head_dim)) goto fail;
    if (!alloc_float_buffer(&bws.merged, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.attn_proj, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.attn_out, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.ln2_out, (size_t)max_tokens * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.fc1_out, (size_t)max_tokens * 4 * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.gelu_out, (size_t)max_tokens * 4 * cfg.hidden)) goto fail;
    if (!alloc_float_buffer(&bws.fc2_out, (size_t)max_tokens * cfg.hidden)) goto fail;

    return true;

fail:
    destroy_gpt2_workspace(ws);
    return false;
}

void destroy_gpt2_workspace(GPT2Workspace* ws) {
    if (ws == nullptr) {
        return;
    }

    safe_cuda_free(ws->x);
    safe_cuda_free(ws->block_in);
    safe_cuda_free(ws->block_out);
    safe_cuda_free(ws->ln_f_out);
    safe_cuda_free(ws->logits);
    safe_cuda_free(ws->k_cache);
    safe_cuda_free(ws->v_cache);

    BlockWorkspace& bws = ws->block_ws;
    safe_cuda_free(bws.ln1_out);
    safe_cuda_free(bws.qkv);
    safe_cuda_free(bws.q);
    safe_cuda_free(bws.k);
    safe_cuda_free(bws.v);
    safe_cuda_free(bws.k_t);
    safe_cuda_free(bws.scores);
    safe_cuda_free(bws.probs);
    safe_cuda_free(bws.context);
    safe_cuda_free(bws.merged);
    safe_cuda_free(bws.attn_proj);
    safe_cuda_free(bws.attn_out);
    safe_cuda_free(bws.ln2_out);
    safe_cuda_free(bws.fc1_out);
    safe_cuda_free(bws.gelu_out);
    safe_cuda_free(bws.fc2_out);

    ws->max_tokens = 0;
}
