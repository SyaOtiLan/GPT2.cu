#pragma once

void launch_gelu(const float* x, float* y, int n);
void launch_add_bias(float* x, const float* bias, int rows, int cols);
void launch_residual_add(const float* a, const float* b, float* y, int n);
void launch_layernorm(const float* x, float* y, const float* gamma, const float* beta, int rows, int cols);
void launch_softmax(const float* x, float* y, int rows, int cols);
void launch_scale(float* x, float scale, int n);
void launch_causal_mask(float* scores, int rows, int cols);
void launch_qkv_split_reshape(const float* qkv, float* q, float* k, float* v, int tokens, int hidden, int heads);
void launch_merge_heads(const float* x, float* y, int tokens, int heads, int head_dim);
void launch_transpose(const float* in, float* out, int rows, int cols);
