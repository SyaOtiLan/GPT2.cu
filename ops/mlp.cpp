#include "../include/mlp.h"
#include "../include/gemm.h"
#include "../include/kernels.h"

void mlp_forward(
    const float* x,
    const float* residual,
    float* out,
    const float* w_fc1,
    const float* b_fc1,
    const float* w_fc2,
    const float* b_fc2,
    int tokens,
    int hidden,
    float* fc1_out,
    float* gelu_out,
    float* fc2_out)
{
    gemm(x, w_fc1, fc1_out, tokens, 4 * hidden, hidden);
    launch_add_bias(fc1_out, b_fc1, tokens, 4 * hidden);
    launch_gelu(fc1_out, gelu_out, tokens * 4 * hidden);
    gemm(gelu_out, w_fc2, fc2_out, tokens, hidden, 4 * hidden);
    launch_add_bias(fc2_out, b_fc2, tokens, hidden);
    launch_residual_add(residual, fc2_out, out, tokens * hidden);
}
