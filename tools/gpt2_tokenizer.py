#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GPT-2 tokenizer helper for GPT2.cu.")
    parser.add_argument("--model-dir", required=True, help="Directory containing tokenizer files.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--encode", help="Encode a text string into token ids.")
    mode.add_argument("--encode-file", help="Read text from a file and encode it.")
    mode.add_argument("--decode", nargs="+", type=int, help="Decode token ids back into text.")
    return parser.parse_args()


def main() -> None:
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install transformers") from exc

    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_dir, local_files_only=True)

    if args.encode is not None:
        ids = tokenizer.encode(args.encode, add_special_tokens=False)
        print(" ".join(str(x) for x in ids))
        return

    if args.encode_file is not None:
        text = Path(args.encode_file).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(" ".join(str(x) for x in ids))
        return

    print(tokenizer.decode(args.decode))


if __name__ == "__main__":
    main()
