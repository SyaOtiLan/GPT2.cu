#include "../include/transformer_block.h"
#include "../include/attention.h"
#include "../include/mlp.h"
#include "../include/kernels.h"

void transformer_block_forward(
    const float* x,
    float* out,
    const BlockWeights& w,
    int tokens,
    int hidden,
    int heads,
    BlockWorkspace& ws)
{
    // 1) LN1
    launch_layernorm(
        x,
        ws.ln1_out,
        w.ln1_gamma,
        w.ln1_beta,
        tokens,
        hidden
    );

    // 2) Attention
    attention_forward(
        ws.ln1_out,
        x,
        ws.attn_out,
        w.w_qkv,
        w.b_qkv,
        w.w_o,
        w.b_o,
        tokens,
        hidden,
        heads,
        ws.qkv,
        ws.q,
        ws.k,
        ws.v,
        ws.scores,
        ws.probs,
        ws.context,
        ws.merged,
        ws.attn_proj
    );

    // 3) LN2
    launch_layernorm(
        ws.attn_out,
        ws.ln2_out,
        w.ln2_gamma,
        w.ln2_beta,
        tokens,
        hidden
    );

    // 4) MLP
    mlp_forward(
        ws.ln2_out,
        ws.attn_out,
        out,
        w.w_fc1,
        w.b_fc1,
        w.w_fc2,
        w.b_fc2,
        tokens,
        hidden,
        ws.fc1_out,
        ws.gelu_out,
        ws.fc2_out
    );
}

void transformer_block_prefill(
    const float* x,
    float* out,
    const BlockWeights& w,
    int tokens,
    int hidden,
    int heads,
    int max_tokens,
    float* k_cache,
    float* v_cache,
    BlockWorkspace& ws)
{
    // 1) LN1
    launch_layernorm(
        x,
        ws.ln1_out,
        w.ln1_gamma,
        w.ln1_beta,
        tokens,
        hidden
    );

    // 2) Attention
    attention_forward(
        ws.ln1_out,
        x,
        ws.attn_out,
        w.w_qkv,
        w.b_qkv,
        w.w_o,
        w.b_o,
        tokens,
        hidden,
        heads,
        ws.qkv,
        ws.q,
        ws.k,
        ws.v,
        ws.scores,
        ws.probs,
        ws.context,
        ws.merged,
        ws.attn_proj
    );
    attention_write_kv_cache(ws.k, ws.v, k_cache, v_cache, heads, tokens, hidden / heads, 0, max_tokens);

    // 3) LN2
    launch_layernorm(
        ws.attn_out,
        ws.ln2_out,
        w.ln2_gamma,
        w.ln2_beta,
        tokens,
        hidden
    );

    // 4) MLP
    mlp_forward(
        ws.ln2_out,
        ws.attn_out,
        out,
        w.w_fc1,
        w.b_fc1,
        w.w_fc2,
        w.b_fc2,
        tokens,
        hidden,
        ws.fc1_out,
        ws.gelu_out,
        ws.fc2_out
    );
}

void transformer_block_decode(
    const float* x,
    float* out,
    const BlockWeights& w,
    int position,
    int hidden,
    int heads,
    int max_tokens,
    float* k_cache,
    float* v_cache,
    BlockWorkspace& ws)
{
    // 1) LN1
    launch_layernorm(
        x,
        ws.ln1_out,
        w.ln1_gamma,
        w.ln1_beta,
        1,
        hidden
    );

    // 2) Attention using persistent KV cache
    attention_forward_decode(
        ws.ln1_out,
        x,
        ws.attn_out,
        w.w_qkv,
        w.b_qkv,
        w.w_o,
        w.b_o,
        position,
        hidden,
        heads,
        max_tokens,
        ws.qkv,
        ws.q,
        ws.k,
        ws.v,
        ws.scores,
        ws.probs,
        ws.context,
        ws.merged,
        ws.attn_proj,
        ws.k_t,
        k_cache,
        v_cache
    );

    // 3) LN2
    launch_layernorm(
        ws.attn_out,
        ws.ln2_out,
        w.ln2_gamma,
        w.ln2_beta,
        1,
        hidden
    );

    // 4) MLP
    mlp_forward(
        ws.ln2_out,
        ws.attn_out,
        out,
        w.w_fc1,
        w.b_fc1,
        w.w_fc2,
        w.b_fc2,
        1,
        hidden,
        ws.fc1_out,
        ws.gelu_out,
        ws.fc2_out
    );
}
