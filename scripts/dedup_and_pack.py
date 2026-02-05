#!/usr/bin/env python3
"""Deduplicate JSONL code samples and pack to fixed-length shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dedup and pack JSONL dataset.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def iter_records(path: pathlib.Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_sequences(texts: List[str], max_len: int) -> List[str]:
    packed = []
    buffer = ""
    for text in texts:
        if not buffer:
            buffer = text
            continue
        if len(buffer) + len(text) + 1 <= max_len:
            buffer = buffer + "\n" + text
        else:
            packed.append(buffer)
            buffer = text
    if buffer:
        packed.append(buffer)
    return packed


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    texts = []
    for record in iter_records(input_path):
        text = record.get("text", "")
        digest = hash_text(text)
        if digest in seen:
            continue
        seen.add(digest)
        texts.append(text)

    random.shuffle(texts)
    packed = pack_sequences(texts, args.max_seq_len)

    shard = []
    shard_index = 0
    for item in packed:
        shard.append({"id": f"shard-{shard_index}-{len(shard)}", "text": item})
        if len(shard) >= args.shard_size:
            shard_path = output_dir / f"shard-{shard_index:05d}.jsonl"
            with shard_path.open("w", encoding="utf-8") as handle:
                for record in shard:
                    handle.write(json.dumps(record) + "\n")
            shard_index += 1
            shard = []

    if shard:
        shard_path = output_dir / f"shard-{shard_index:05d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for record in shard:
                handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(packed)} packed samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
