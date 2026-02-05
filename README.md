# Python-Only 7B Code LLM Plan

This repository contains a focused training blueprint and scripts for a **7B-parameter, Python-only coding model** that is trained from scratch. The goal is to maximize coding capability, reasoning, and tool-use while minimizing wasted capacity on non-Python text.

## Goals
- **7B parameters** decoder-only Transformer.
- **Python-only** data pipeline (strict extraction and filtering).
- **Reasoning-first** training with a tool-augmented curriculum.
- **Internet + tool access** via a controlled tool API during supervised fine-tuning.
- **Fully from scratch**, no model warm-start.
- **Highly optimized** training for throughput and stability.

## Repository Layout
- `docs/architecture.md`: model architecture and optimization choices.
- `docs/data_pipeline.md`: data sources, Python extraction, and dedup strategy.
- `docs/training_plan.md`: staged training schedule and reasoning curriculum.
- `scripts/`: data prep utilities (download, extract, dedup, pack).
- `train.py`: entry point to launch pretraining.
- `configs/train.yaml`: reference hyperparameters.

## Quick Start (high level)
1. Download and curate data with `scripts/download_data.py`.
2. Extract Python-only content with `scripts/extract_python.py`.
3. Deduplicate and pack into shards with `scripts/dedup_and_pack.py`.
4. Launch training with `train.py --config configs/train.yaml`.

> This is a blueprint and starter implementation. You can swap in your preferred tokenizer/training stack while keeping the same data discipline and curriculum.
