#pragma once

#include <float.h>
#include <math.h>
#include <vector>

struct CPUBlockWeights {
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

struct CPUGPT2Weights {
    const float* wte;
    const float* wpe;
    const CPUBlockWeights* blocks;
    const float* ln_f_gamma;
    const float* ln_f_beta;
    const float* lm_head;
};

inline float gelu_cpu_ref(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline void matmul_cpu_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

inline void layernorm_cpu_ref(
    const float* x,
    float* y,
    const float* gamma,
    const float* beta,
    int rows,
    int cols)
{
    const float eps = 1e-5f;

    for (int row = 0; row < rows; ++row) {
        const float* in_row = x + row * cols;
        float* out_row = y + row * cols;

        float mean = 0.0f;
        for (int col = 0; col < cols; ++col) {
            mean += in_row[col];
        }
        mean /= cols;

        float var = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float diff = in_row[col] - mean;
            var += diff * diff;
        }
        var /= cols;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int col = 0; col < cols; ++col) {
            out_row[col] = ((in_row[col] - mean) * inv_std) * gamma[col] + beta[col];
        }
    }
}

inline void softmax_rows_cpu_ref(const float* x, float* y, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const float* in_row = x + row * cols;
        float* out_row = y + row * cols;

        float row_max = -FLT_MAX;
        for (int col = 0; col < cols; ++col) {
            if (in_row[col] > row_max) row_max = in_row[col];
        }

        float row_sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float v = expf(in_row[col] - row_max);
            out_row[col] = v;
            row_sum += v;
        }

        for (int col = 0; col < cols; ++col) {
            out_row[col] /= row_sum;
        }
    }
}

inline void embedding_cpu_ref(
    const int* input_ids,
    const float* wte,
    const float* wpe,
    float* x,
    int tokens,
    int hidden)
{
    for (int token = 0; token < tokens; ++token) {
        int token_id = input_ids[token];
        for (int dim = 0; dim < hidden; ++dim) {
            x[token * hidden + dim] = wte[token_id * hidden + dim] + wpe[token * hidden + dim];
        }
    }
}

inline void attention_cpu_ref(
    const float* x,
    const float* residual,
    float* out,
    const float* w_qkv,
    const float* b_qkv,
    const float* w_o,
    const float* b_o,
    int tokens,
    int hidden,
    int heads)
{
    const float neg_inf = -1e20f;
    int head_dim = hidden / heads;

    std::vector<float> qkv(tokens * 3 * hidden);
    std::vector<float> q(heads * tokens * head_dim);
    std::vector<float> k(heads * tokens * head_dim);
    std::vector<float> v(heads * tokens * head_dim);
    std::vector<float> scores(heads * tokens * tokens);
    std::vector<float> probs(heads * tokens * tokens);
    std::vector<float> context(heads * tokens * head_dim);
    std::vector<float> merged(tokens * hidden);
    std::vector<float> proj(tokens * hidden);

    matmul_cpu_ref(x, w_qkv, qkv.data(), tokens, 3 * hidden, hidden);
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < 3 * hidden; ++j) {
            qkv[i * 3 * hidden + j] += b_qkv[j];
        }
    }

    for (int token = 0; token < tokens; ++token) {
        int base = token * 3 * hidden;
        for (int hidden_idx = 0; hidden_idx < hidden; ++hidden_idx) {
            int head = hidden_idx / head_dim;
            int dim = hidden_idx % head_dim;
            int out_idx = (head * tokens + token) * head_dim + dim;
            q[out_idx] = qkv[base + hidden_idx];
            k[out_idx] = qkv[base + hidden + hidden_idx];
            v[out_idx] = qkv[base + 2 * hidden + hidden_idx];
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    for (int head = 0; head < heads; ++head) {
        const float* q_h = q.data() + head * tokens * head_dim;
        const float* k_h = k.data() + head * tokens * head_dim;
        const float* v_h = v.data() + head * tokens * head_dim;
        float* scores_h = scores.data() + head * tokens * tokens;
        float* probs_h = probs.data() + head * tokens * tokens;
        float* context_h = context.data() + head * tokens * head_dim;

        for (int i = 0; i < tokens; ++i) {
            for (int j = 0; j < tokens; ++j) {
                float sum = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim) {
                    sum += q_h[i * head_dim + dim] * k_h[j * head_dim + dim];
                }
                scores_h[i * tokens + j] = (j > i) ? neg_inf : sum * scale;
            }
        }

        softmax_rows_cpu_ref(scores_h, probs_h, tokens, tokens);
        matmul_cpu_ref(probs_h, v_h, context_h, tokens, head_dim, tokens);
    }

    for (int token = 0; token < tokens; ++token) {
        for (int hidden_idx = 0; hidden_idx < hidden; ++hidden_idx) {
            int head = hidden_idx / head_dim;
            int dim = hidden_idx % head_dim;
            merged[token * hidden + hidden_idx] = context[(head * tokens + token) * head_dim + dim];
        }
    }

    matmul_cpu_ref(merged.data(), w_o, proj.data(), tokens, hidden, hidden);
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < hidden; ++j) {
            proj[i * hidden + j] += b_o[j];
        }
    }

    for (int i = 0; i < tokens * hidden; ++i) {
        out[i] = residual[i] + proj[i];
    }
}

inline void mlp_cpu_ref(
    const float* x,
    const float* residual,
    float* out,
    const float* w_fc1,
    const float* b_fc1,
    const float* w_fc2,
    const float* b_fc2,
    int tokens,
    int hidden)
{
    std::vector<float> fc1(tokens * 4 * hidden);
    std::vector<float> gelu(tokens * 4 * hidden);
    std::vector<float> fc2(tokens * hidden);

    matmul_cpu_ref(x, w_fc1, fc1.data(), tokens, 4 * hidden, hidden);
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < 4 * hidden; ++j) {
            fc1[i * 4 * hidden + j] += b_fc1[j];
        }
    }
    for (int i = 0; i < tokens * 4 * hidden; ++i) {
        gelu[i] = gelu_cpu_ref(fc1[i]);
    }
    matmul_cpu_ref(gelu.data(), w_fc2, fc2.data(), tokens, hidden, 4 * hidden);
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < hidden; ++j) {
            fc2[i * hidden + j] += b_fc2[j];
        }
    }

    for (int i = 0; i < tokens * hidden; ++i) {
        out[i] = residual[i] + fc2[i];
    }
}

inline void transformer_block_cpu_ref(
    const float* x,
    float* out,
    const CPUBlockWeights& w,
    int tokens,
    int hidden,
    int heads)
{
    std::vector<float> ln1(tokens * hidden);
    std::vector<float> attn(tokens * hidden);
    std::vector<float> ln2(tokens * hidden);

    layernorm_cpu_ref(x, ln1.data(), w.ln1_gamma, w.ln1_beta, tokens, hidden);
    attention_cpu_ref(ln1.data(), x, attn.data(), w.w_qkv, w.b_qkv, w.w_o, w.b_o, tokens, hidden, heads);
    layernorm_cpu_ref(attn.data(), ln2.data(), w.ln2_gamma, w.ln2_beta, tokens, hidden);
    mlp_cpu_ref(ln2.data(), attn.data(), out, w.w_fc1, w.b_fc1, w.w_fc2, w.b_fc2, tokens, hidden);
}

inline void gpt2_forward_cpu_ref(
    const int* input_ids,
    float* logits,
    const CPUGPT2Weights& w,
    int tokens,
    int hidden,
    int heads,
    int n_layer,
    int vocab_size)
{
    std::vector<float> x(tokens * hidden);
    std::vector<float> block_a(tokens * hidden);
    std::vector<float> block_b(tokens * hidden);
    std::vector<float> ln_f(tokens * hidden);

    embedding_cpu_ref(input_ids, w.wte, w.wpe, x.data(), tokens, hidden);

    const float* cur_in = x.data();
    float* cur_out = block_a.data();

    for (int layer = 0; layer < n_layer; ++layer) {
        transformer_block_cpu_ref(cur_in, cur_out, w.blocks[layer], tokens, hidden, heads);
        const float* tmp = cur_out;
        cur_out = (cur_out == block_a.data()) ? block_b.data() : block_a.data();
        cur_in = tmp;
    }

    layernorm_cpu_ref(cur_in, ln_f.data(), w.ln_f_gamma, w.ln_f_beta, tokens, hidden);
    matmul_cpu_ref(ln_f.data(), w.lm_head, logits, tokens, vocab_size, hidden);
}

inline void generate_greedy_cpu_ref(
    const int* input_ids,
    int input_len,
    int max_new_tokens,
    int* output_ids,
    const CPUGPT2Weights& w,
    int hidden,
    int heads,
    int n_layer,
    int vocab_size)
{
    for (int i = 0; i < input_len; ++i) {
        output_ids[i] = input_ids[i];
    }

    std::vector<float> logits((input_len + max_new_tokens) * vocab_size);
    for (int step = 0; step < max_new_tokens; ++step) {
        int cur_len = input_len + step;
        gpt2_forward_cpu_ref(output_ids, logits.data(), w, cur_len, hidden, heads, n_layer, vocab_size);

        const float* last_logits = logits.data() + (cur_len - 1) * vocab_size;
        int best = 0;
        for (int i = 1; i < vocab_size; ++i) {
            if (last_logits[i] > last_logits[best]) best = i;
        }
        output_ids[cur_len] = best;
    }
}

inline float max_abs_diff_cpu_ref(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

inline bool all_finite_cpu_ref(const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) return false;
    }
    return true;
}

inline bool ids_in_range_cpu_ref(const int* ids, int n, int vocab_size) {
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= vocab_size) return false;
    }
    return true;
}
