#pragma once

void mlp_forward(
    const float* x,        // normalized input [tokens, hidden]
    const float* residual, // residual input [tokens, hidden]
    float* out,            // [tokens, hidden]

    const float* w_fc1,    // [hidden, 4*hidden]
    const float* b_fc1,    // [4*hidden]
    const float* w_fc2,    // [4*hidden, hidden]
    const float* b_fc2,    // [hidden]

    int tokens,
    int hidden,

    // workspace
    float* fc1_out,        // [tokens, 4*hidden]
    float* gelu_out,       // [tokens, 4*hidden]
    float* fc2_out         // [tokens, hidden]
);
