# GPT2.cu

Handwritten CUDA GPT-2 inference project for learning transformer inference end to end.

Current status:

- GPT-2 style pre-LN transformer blocks
- CUDA kernels for embedding, layernorm, GEMM, attention pieces, GELU, residual add
- Hugging Face GPT-2 export script
- Raw `.bin` weight loader
- Text input via tokenizer bridge
- Greedy generation
- `prefill + decode`
- KV cache
- Unit tests plus real GPT-2 smoke test
- Benchmark for full recompute vs KV cache

## Repo Layout

- `include/`: headers
- `kernels/`: CUDA kernels
- `ops/`: higher-level GPT-2 ops and model glue
- `tools/`: Hugging Face export script and tokenizer helper
- `tests/`: correctness and smoke tests
- `bench/`: benchmark programs
- `models/gpt2-bin/`: exported GPT-2 weights/config/tokenizer files

## Export a Hugging Face GPT-2 Model

Install Python deps:

```bash
pip install torch transformers
```

Export:

```bash
python3 tools/export_hf_gpt2.py \
  --model openai-community/gpt2 \
  --output-dir models/gpt2-bin
```

This writes:

- `gpt2_config.json`
- raw float32 `.bin` weights
- tokenizer files (`vocab.json`, `merges.txt`, `tokenizer.json`, ...)

## Build and Run Main

Build:

```bash
make main
```

Useful targets:

```bash
make main
make bench
make real_smoke
make test
```

Run next-token prediction:

```bash
cd /home/lxy/GPT2.cu
./build/gpt2_main --text "Hello, I am"
```

Run greedy generation:

```bash
cd /home/lxy/GPT2.cu
./build/gpt2_main --generate 8 --text "Hello, I am"
```

## Benchmark

Build:

```bash
make bench
```

Run:

```bash
cd /home/lxy/GPT2.cu
./build/gpt2_bench --prompt-len 32 --gen-len 32 --warmup 1 --iters 5
```

The benchmark reports:

- `full_recompute_total_ms`
- `kv_prefill_ms`
- `kv_decode_total_ms`
- `kv_decode_ms_per_token`
- `kv_total_ms`
- `kv_speedup_vs_full`

Sample results on the author's local GPU:

| prompt_len | gen_len | full recompute | kv prefill | kv decode | kv total | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 478.863 ms | 13.796 ms | 306.482 ms | 320.278 ms | 1.495x |
| 32 | 128 | 2395.783 ms | 13.902 ms | 1259.238 ms | 1273.140 ms | 1.882x |
| 128 | 128 | 3441.195 ms | 21.385 ms | 1335.563 ms | 1356.948 ms | 2.536x |

Takeaway:

- KV cache already gives a clear win in this handwritten implementation.
- The speedup grows when generation gets longer.
- The current decode path is still not heavily optimized, so there is still room to improve beyond these numbers.

## Tests

Examples:

```bash
make gpt2_test
```

```bash
make kv_cache_test
```

```bash
./build/gpt2_test
./build/kv_cache_test
```

## Notes

- Current tokenizer path is a Python bridge, not a native C++ tokenizer.
- Current implementation is FP32 and batch size 1.
- KV cache is implemented functionally first; it is not yet heavily optimized.
- The benchmark is meant to compare algorithmic paths, especially full recompute vs KV cache.
