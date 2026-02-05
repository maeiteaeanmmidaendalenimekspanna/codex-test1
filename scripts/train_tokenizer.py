#!/usr/bin/env python3
"""Train a BPE tokenizer on Python-only JSONL shards."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<file_sep>", "<tool_call>", "<tool_result>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Python-only tokenizer.")
    parser.add_argument("--input", required=True, help="JSONL file or directory")
    parser.add_argument("--output", required=True, help="Output tokenizer.json path")
    parser.add_argument("--vocab-size", type=int, default=64000)
    return parser.parse_args()


def iter_texts(paths: Iterable[pathlib.Path]) -> Iterable[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    yield text


def resolve_paths(input_path: pathlib.Path) -> list[pathlib.Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))
    return [input_path]


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_texts(resolve_paths(input_path)), trainer=trainer)
    tokenizer.save(str(output_path))
    print(f"Saved tokenizer to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
