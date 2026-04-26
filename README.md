# GPT2.cu

[English](#gpt2cu) | [中文说明](#中文说明)

A learning-oriented GPT-2 inference engine written in C++ and CUDA.

Chinese write-up:
[从零手写 CUDA GPT-2 推理：从 full recompute 到 KV cache](https://zhuanlan.zhihu.com/p/2027774235276325692)

This project runs real Hugging Face GPT-2 weights without calling PyTorch or a
deep learning runtime during inference. The goal is to make the GPT-2 inference
path visible end to end: token ids go through embeddings, transformer blocks,
final layernorm, the LM head, greedy generation, and finally a `prefill +
decode` path with KV cache.

## What This Project Does

- Loads real GPT-2 weights exported from Hugging Face into raw `.bin` files.
- Implements the main GPT-2 inference path in C++/CUDA:
  - token and position embedding
  - pre-LN transformer blocks
  - multi-head causal self-attention
  - MLP with GELU
  - residual add
  - final layernorm
  - LM head
- Provides CUDA kernels for GEMM, layernorm, softmax, GELU, embedding,
  residual add, transpose, QKV split/reshape, head merge, scaling, and masking.
- Supports text input/output through a tokenizer bridge.
- Supports greedy generation.
- Implements both naive full recompute generation and `prefill + decode` with
  KV cache.
- Includes unit tests, a real GPT-2 smoke test, and a benchmark comparing full
  recompute against KV-cache decoding.

Example output from a real GPT-2 run:

```text
input text: Hello, I am
generated text: Hello, I am a little bit of a fan of the
```

## Inference Flow

At the model level, the data flow is:

```text
input text
  -> tokenizer
  -> token ids
  -> token + position embedding
  -> transformer block x N
  -> final layernorm
  -> lm head
  -> logits
  -> next token
```

Each transformer block follows the GPT-2 pre-LN structure:

```text
x
  -> layernorm
  -> attention
  -> residual add
  -> layernorm
  -> MLP
  -> residual add
```

Attention uses GPT-2's packed QKV projection:

```text
x -> c_attn -> [Q, K, V] -> causal attention -> c_proj
```

The MLP expands hidden size to `4 * hidden`, applies GELU, then projects back to
`hidden`.

## Why KV Cache Matters

The simplest generation loop is full recompute:

```text
prompt length 4 -> run forward on 4 tokens -> sample token 5
prompt length 5 -> run forward on 5 tokens -> sample token 6
prompt length 6 -> run forward on 6 tokens -> sample token 7
```

This is easy to implement, but it repeatedly recomputes old tokens. During
causal self-attention, the historical K/V tensors are exactly the values future
tokens need. KV cache stores those per-layer historical K/V tensors so decode can
process only the new token:

```text
prefill:
  run the full prompt once
  write each layer's prompt K/V into cache

decode:
  compute Q/K/V for one new token
  append the new K/V to cache
  attend with current Q over cached K/V
```

The cache layout is:

```text
[n_layer, heads, max_tokens, head_dim]
```

So instead of rebuilding an increasingly large `t x t` attention matrix at every
generation step, decode mostly does the current token's `1 x t` attention work.

## Benchmark

The benchmark compares:

- `full recompute`: every generated token reruns `gpt2_forward` over the full
  current prefix.
- `KV cache`: one `gpt2_prefill` over the prompt, followed by one
  `gpt2_decode_step` per generated token.

Sample results from the author's local GPU:

| prompt_len | gen_len | full recompute | kv prefill | kv decode | kv total | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 478.863 ms | 13.796 ms | 306.482 ms | 320.278 ms | 1.495x |
| 32 | 128 | 2395.783 ms | 13.902 ms | 1259.238 ms | 1273.140 ms | 1.882x |
| 128 | 128 | 3441.195 ms | 21.385 ms | 1335.563 ms | 1356.948 ms | 2.536x |

The current decode path is functional first, not heavily optimized, but the
algorithmic win is already visible and grows with longer prompts or generations.

## Repo Layout

```text
.
|-- bench/      benchmark programs
|-- include/    headers
|-- kernels/    CUDA kernels
|-- ops/        GPT-2 ops and model glue
|-- tests/      correctness and smoke tests
|-- tools/      Hugging Face exporter and tokenizer bridge
|-- main.cpp    CLI entry point
`-- Makefile
```

The main code path is organized as:

```text
main.cpp
  -> ops/generate.cpp
  -> ops/gpt2.cpp
  -> ops/transformer_block.cpp
  -> ops/attention.cpp
  -> ops/mlp.cpp
  -> kernels/*.cu
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit with `nvcc`
- C++17 compiler support through `nvcc`
- Python 3
- `torch` and `transformers` for exporting weights and tokenization

Install Python dependencies:

```bash
pip install torch transformers
```

## Export GPT-2 Weights

Export a Hugging Face GPT-2 checkpoint into the raw format used by this project:

```bash
python3 tools/export_hf_gpt2.py \
  --model openai-community/gpt2 \
  --output-dir models/gpt2-bin
```

The exporter writes:

- `gpt2_config.json`
- raw float32 `.bin` weights
- tokenizer files such as `vocab.json`, `merges.txt`, and `tokenizer.json`

The loader then reads those files directly in C++ and moves them to GPU memory
with `cudaMalloc` and `cudaMemcpy`.

## Build

```bash
make main
```

Useful targets:

```bash
make main
make bench
make real_smoke
make gpt2_test
make kv_cache_test
make generate_test
make test
```

## Run

Next-token prediction:

```bash
./build/gpt2_main --text "Hello, I am"
```

Greedy generation:

```bash
./build/gpt2_main --generate 8 --text "Hello, I am"
```

You can also pass token ids directly:

```bash
./build/gpt2_main 15496 11 314 716
```

Use a different exported model directory:

```bash
./build/gpt2_main --model-dir models/gpt2-bin --generate 8 --text "Hello, I am"
```

## Run Benchmark

```bash
make bench
./build/gpt2_bench --prompt-len 32 --gen-len 32 --warmup 1 --iters 5
```

Reported fields include:

- `full_recompute_total_ms`
- `full_recompute_ms_per_token`
- `kv_prefill_ms`
- `kv_decode_total_ms`
- `kv_decode_ms_per_token`
- `kv_total_ms`
- `kv_speedup_vs_full`
- `kv_decode_tokens_per_s`

## Run Tests

```bash
make test
./build/gpt2_test
./build/generate_test
./build/kv_cache_test
```

Real GPT-2 smoke test:

```bash
make real_smoke
./build/real_gpt2_smoke
```

## Current Limitations

- FP32 only.
- Batch size 1.
- Greedy decoding only.
- Tokenization currently goes through a Python bridge.
- Decode path is implemented for clarity first and still has optimization room.
- GEMM is handwritten CUDA rather than cuBLAS.

## Roadmap

- Top-k, top-p, and temperature sampling.
- Batch inference.
- FP16/BF16 support.
- Faster GEMM and decode kernels.
- Fewer transposes and memory copies in the attention path.
- Native C++ tokenizer.

## Project Goal

This is not meant to be a production LLM runtime. It is a from-scratch CUDA
implementation for understanding how GPT-2 inference is actually assembled from
weights, tensor shapes, kernels, residual paths, attention, generation loops, and
KV cache.

---

# 中文说明

这是一个用 C++/CUDA 手写的 GPT-2 推理项目，目标不是做一个生产级 LLM
推理框架，而是把 GPT-2 推理链路从输入到输出完整拆开：权重怎么加载，token
怎么进入 embedding，transformer block 怎么组织，attention / MLP /
layernorm / residual add 这些算子怎么串起来，以及生成时为什么需要
`prefill + decode + KV cache`。

项目目前已经可以加载 Hugging Face GPT-2 权重导出的真实参数，并在 C++ 侧完成
GPT-2 推理和 greedy generation。

## 项目已经实现了什么

- 用 C++/CUDA 实现 GPT-2 推理主链路，包括 embedding、transformer block、
  final layernorm 和 lm head。
- 实现 attention、MLP、layernorm、residual add、GELU、softmax、GEMM 等基础
  CUDA 算子。
- 支持从 Hugging Face GPT-2 导出权重，并转换成项目自己的 raw `.bin` 格式。
- 支持在 C++ 侧加载真实 GPT-2 配置和权重，而不是只跑手造测试数据。
- 通过 tokenizer bridge 支持文本输入和输出。
- 支持 greedy generate。
- 将生成流程从 naive full recompute 改成 `prefill + decode`，并加入 KV cache。
- 增加 benchmark，对比 full recompute 和 KV cache 两条生成路径的耗时。

示例输出：

```text
input text: Hello, I am
generated text: Hello, I am a little bit of a fan of the
```

## GPT-2 推理链路

整体数据流可以概括成：

```text
input text
  -> tokenizer
  -> token ids
  -> token + position embedding
  -> transformer blocks
  -> final layernorm
  -> lm head
  -> logits
  -> next token
```

每一层 transformer block 的结构是：

```text
x
  -> layernorm
  -> attention
  -> residual add
  -> layernorm
  -> MLP
  -> residual add
```

attention 内部先用 GPT-2 的 packed QKV projection 得到 Q/K/V，再按 head 拆开做
causal self-attention，最后合并多头结果并经过输出投影。MLP 则先把 hidden
维度扩展到 `4 * hidden`，经过 GELU 后再投影回 hidden。

代码组织上，主线大致对应：

```text
main.cpp
  -> ops/generate.cpp
  -> ops/gpt2.cpp
  -> ops/transformer_block.cpp
  -> ops/attention.cpp
  -> ops/mlp.cpp
  -> kernels/*.cu
```

## 从 Hugging Face 权重到真实模型

导出脚本 `tools/export_hf_gpt2.py` 会把 Hugging Face GPT-2 checkpoint 里的参数
翻译成这个项目自己的文件组织方式。例如：

- `wte.bin` / `wpe.bin`：token embedding 和 position embedding。
- `ln_f_gamma.bin` / `ln_f_beta.bin`：final layernorm 参数。
- `lm_head.bin`：输出词表投影。
- `block_0_w_qkv.bin` / `block_0_b_qkv.bin`：第 0 层 attention 的 QKV 投影。
- `block_0_w_o.bin` / `block_0_b_o.bin`：attention 输出投影。
- `block_0_w_fc1.bin` / `block_0_w_fc2.bin`：MLP 两层线性层。

C++ 侧的 loader 只需要按约定文件名读取 `.bin`，检查大小，然后通过
`cudaMalloc` 和 `cudaMemcpy` 放到 GPU 上。

## 为什么要做 KV cache

最朴素的 generate 做法是 full recompute：每生成一个 token，都把当前完整前缀
重新跑一遍 forward。

```text
长度 4 -> 跑 4 个 token -> 生成第 5 个 token
长度 5 -> 跑 5 个 token -> 生成第 6 个 token
长度 6 -> 跑 6 个 token -> 生成第 7 个 token
```

这个逻辑容易实现，但会反复重算历史 token。对于自回归生成来说，未来 token
真正需要复用的是每一层历史 token 的 K/V。历史 token 的 Q 不会再被未来步骤
使用，所以缓存 K/V 就能避免每一步重新计算整段前缀。

KV cache 之后生成流程变成：

```text
prefill:
  对 prompt 做一次完整 forward
  把每一层 prompt 的 K/V 写入 cache

decode:
  每次只处理一个新 token
  计算当前 token 的 Q/K/V
  把当前 K/V 追加进 cache
  用当前 Q attend 到历史 K/V
```

cache layout 是：

```text
[n_layer, heads, max_tokens, head_dim]
```

所以 full recompute 里不断增长的 `t x t` attention，会在 decode 阶段变成更接近
`1 x t` 的当前 token attention。

## Benchmark 结果

本地真实 GPT-2 权重测试结果：

| prompt_len | gen_len | full recompute | kv prefill | kv decode | kv total | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 478.863 ms | 13.796 ms | 306.482 ms | 320.278 ms | 1.495x |
| 32 | 128 | 2395.783 ms | 13.902 ms | 1259.238 ms | 1273.140 ms | 1.882x |
| 128 | 128 | 3441.195 ms | 21.385 ms | 1335.563 ms | 1356.948 ms | 2.536x |

可以看到，KV cache 的收益是明确存在的，并且随着 prompt length 和 generation
length 增大，优势会进一步拉开。

## 快速运行

导出 Hugging Face GPT-2 权重：

```bash
python3 tools/export_hf_gpt2.py \
  --model openai-community/gpt2 \
  --output-dir models/gpt2-bin
```

编译：

```bash
make main
```

运行 next-token prediction：

```bash
./build/gpt2_main --text "Hello, I am"
```

运行 greedy generation：

```bash
./build/gpt2_main --generate 8 --text "Hello, I am"
```

运行 benchmark：

```bash
make bench
./build/gpt2_bench --prompt-len 32 --gen-len 32 --warmup 1 --iters 5
```

运行测试：

```bash
make test
./build/gpt2_test
./build/generate_test
./build/kv_cache_test
```

## 当前限制和后续方向

目前还没有做：

- top-k、top-p、temperature 等更完整采样策略。
- batch 推理。
- FP16 / BF16。
- 更高性能的 GEMM 和 decode kernel。
- 原生 C++ tokenizer。

后续还可以继续优化 decode 路径，减少 transpose 和 memcpy，引入更高效的 GEMM，
或者补上更完整的 sampling 策略。但到目前为止，这个项目已经跑通了真实 GPT-2
权重的推理路径，并把 naive generate、prefill/decode、KV cache 这些关键概念
落实到了可运行的 C++/CUDA 代码里。
