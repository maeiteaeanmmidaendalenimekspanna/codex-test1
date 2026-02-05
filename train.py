#!/usr/bin/env python3
"""Minimal training loop for a 7B Python-only GPT model.

This is a reference implementation meant to be swapped with distributed
training frameworks (e.g., FSDP, DeepSpeed, Megatron-LM) for full-scale runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from tokenizer_utils import PythonTokenizer


@dataclass
class TrainingConfig:
    seed: int
    run_name: str
    model: dict
    optimizer: dict
    training: dict
    data: dict
    output_dir: str


class GPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        hidden_size = config["hidden_size"]
        ffn_size = config["ffn_size"]
        vocab_size = config["vocab_size"]
        max_seq_len = config["max_seq_len"]

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=n_heads,
                    dim_feedforward=ffn_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)


class JsonlDataset:
    def __init__(self, paths: List[str], tokenizer: PythonTokenizer, max_seq_len: int) -> None:
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterable[torch.Tensor]:
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    tokens = self.tokenizer.encode(record.get("text", ""))
                    if len(tokens) <= 1:
                        continue
                    for start in range(0, len(tokens) - 1, self.max_seq_len):
                        chunk = tokens[start : start + self.max_seq_len + 1]
                        if len(chunk) < 2:
                            continue
                        yield torch.tensor(chunk, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Python-only GPT model.")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    return parser.parse_args()


def load_config(path: str) -> TrainingConfig:
    data = yaml.safe_load(pathlib.Path(path).read_text())
    return TrainingConfig(
        seed=data["seed"],
        run_name=data["run_name"],
        model=data["model"],
        optimizer=data["optimizer"],
        training=data["training"],
        data=data["data"],
        output_dir=data["output_dir"],
    )


def iter_shards(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.training.get("device", "cpu"))
    dtype = torch.bfloat16 if cfg.training.get("dtype") == "bf16" else torch.float32

    tokenizer = PythonTokenizer.from_file(args.tokenizer)

    train_paths = iter_shards(cfg.data["train_shards"])
    if not train_paths:
        raise FileNotFoundError("No training shards found.")

    dataset = JsonlDataset(train_paths, tokenizer, cfg.data["pack_seq_len"])

    model = GPT(cfg.model).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer["lr"],
        betas=tuple(cfg.optimizer["betas"]),
        weight_decay=cfg.optimizer["weight_decay"],
    )

    grad_accum_steps = cfg.training["grad_accum_steps"]
    micro_batch_size = cfg.training["micro_batch_size"]
    total_steps = cfg.training["total_steps"]
    log_every = cfg.training["log_every"]
    save_every = cfg.training["save_every"]

    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for sample in dataset:
        if step >= total_steps:
            break
        sample = sample[: cfg.data["pack_seq_len"] + 1]
        if sample.numel() < 2:
            continue

        x = sample[:-1].unsqueeze(0).repeat(micro_batch_size, 1).to(device)
        y = sample[1:].unsqueeze(0).repeat(micro_batch_size, 1).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % log_every == 0:
            print(f"step {step} | loss {loss.item() * grad_accum_steps:.4f}")

        if step % save_every == 0 and step > 0:
            ckpt_path = output_dir / f"checkpoint-{step}.pt"
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
            print(f"saved {ckpt_path}")

        step += 1

    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)
    print(f"training complete -> {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
