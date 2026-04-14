#!/usr/bin/env python3
"""
Export a Hugging Face GPT-2 checkpoint to raw float32 .bin files that match
the loader format used by ops/weights.cpp.

Example:
    python3 tools/export_hf_gpt2.py \
        --model openai-community/gpt2 \
        --output-dir models/gpt2-bin

You can also point --model at a local Hugging Face checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hugging Face GPT-2 weights to GPT2.cu .bin format.")
    parser.add_argument(
        "--model",
        default="openai-community/gpt2",
        help="Hugging Face repo id or local checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the exported .bin files will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use already-downloaded local/cached files.",
    )
    return parser.parse_args()


def require_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install them with:\n"
            "  pip install torch transformers"
        ) from exc
    return torch, GPT2LMHeadModel, GPT2TokenizerFast


def save_tensor(path: Path, tensor: Any) -> None:
    array = tensor.detach().cpu().float().contiguous().numpy()
    array.tofile(path)


def export_config(path: Path, model_name: str, cfg: Any) -> None:
    data = {
        "source_model": model_name,
        "hidden": int(cfg.n_embd),
        "heads": int(cfg.n_head),
        "n_layer": int(cfg.n_layer),
        "vocab_size": int(cfg.vocab_size),
        "max_position": int(cfg.n_positions),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def export_top_level(out_dir: Path, model: Any) -> None:
    transformer = model.transformer
    save_tensor(out_dir / "wte.bin", transformer.wte.weight)
    save_tensor(out_dir / "wpe.bin", transformer.wpe.weight)
    save_tensor(out_dir / "ln_f_gamma.bin", transformer.ln_f.weight)
    save_tensor(out_dir / "ln_f_beta.bin", transformer.ln_f.bias)
    save_tensor(out_dir / "lm_head.bin", model.lm_head.weight.t())


def export_block(out_dir: Path, layer_idx: int, block: Any) -> None:
    prefix = f"block_{layer_idx}_"
    save_tensor(out_dir / f"{prefix}ln1_gamma.bin", block.ln_1.weight)
    save_tensor(out_dir / f"{prefix}ln1_beta.bin", block.ln_1.bias)
    save_tensor(out_dir / f"{prefix}w_qkv.bin", block.attn.c_attn.weight)
    save_tensor(out_dir / f"{prefix}b_qkv.bin", block.attn.c_attn.bias)
    save_tensor(out_dir / f"{prefix}w_o.bin", block.attn.c_proj.weight)
    save_tensor(out_dir / f"{prefix}b_o.bin", block.attn.c_proj.bias)
    save_tensor(out_dir / f"{prefix}ln2_gamma.bin", block.ln_2.weight)
    save_tensor(out_dir / f"{prefix}ln2_beta.bin", block.ln_2.bias)
    save_tensor(out_dir / f"{prefix}w_fc1.bin", block.mlp.c_fc.weight)
    save_tensor(out_dir / f"{prefix}b_fc1.bin", block.mlp.c_fc.bias)
    save_tensor(out_dir / f"{prefix}w_fc2.bin", block.mlp.c_proj.weight)
    save_tensor(out_dir / f"{prefix}b_fc2.bin", block.mlp.c_proj.bias)


def main() -> None:
    args = parse_args()
    _, GPT2LMHeadModel, GPT2TokenizerFast = require_deps()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    tokenizer.save_pretrained(out_dir)

    export_config(out_dir / "gpt2_config.json", args.model, model.config)
    export_top_level(out_dir, model)

    for layer_idx, block in enumerate(model.transformer.h):
        export_block(out_dir, layer_idx, block)

    print(f"Exported {len(model.transformer.h)} GPT-2 blocks to {out_dir}")
    print(f"Config written to {out_dir / 'gpt2_config.json'}")


if __name__ == "__main__":
    main()
