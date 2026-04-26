# GPT2.cu

[English](README.md) | [简体中文](README.zh-CN.md)

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
