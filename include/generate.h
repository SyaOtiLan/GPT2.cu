#pragma once
#include "gpt2.h"

void generate_greedy(
    const int* input_ids_host,   // 初始输入 token ids（host）
    int input_len,
    int max_new_tokens,
    int* output_ids_host,        // 输出长度至少 input_len + max_new_tokens（host）

    const GPT2Weights& w,
    GPT2Workspace& ws,
    const GPT2Config& cfg
);
