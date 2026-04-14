#pragma once

void launch_embedding(
    const int* input_ids,         // [tokens]
    const float* wte,             // [vocab_size, hidden]
    const float* wpe,             // [max_position, hidden]
    float* x,                     // [tokens, hidden]
    int tokens,
    int hidden
);