# Architecture: 7B Python-Only Code Model

## Target model
- **Type:** Decoder-only Transformer (GPT-style).
- **Parameters:** ~7B.
- **Context length:** 8k (train) with extension path to 32k via RoPE scaling.
- **Vocabulary:** 50k–64k, trained on Python-only corpus with a tokenizer biased to Python tokens.

## Suggested configuration (reference)
| Component | Value |
| --- | --- |
| Layers | 32 |
| Hidden size | 4096 |
| Attention heads | 32 |
| FFN size | 11008 (SwiGLU) |
| RoPE | NTK-aware scaling | 
| Norm | RMSNorm |
| Activation | SiLU (SwiGLU) |
| Optimizer | AdamW + decoupled weight decay |
| Precision | bfloat16 |
| Dropout | 0.0–0.1 (prefer 0.0 for code) |

## Optimization choices
- **SwiGLU** for strong coding performance and efficient compute.
- **RMSNorm** to improve stability in large-batch pretraining.
- **RoPE scaling** to allow short-to-long context generalization without full retraining.
- **Tokenizer** trained on Python-only data to avoid vocab waste.
- **FlashAttention** for throughput (if available).

## Tool-use interface
The model is trained with a tool call format:
```
<tool_call name="browser.search">
{"query": "python list comprehension examples"}
</tool_call>
```
Tool outputs are inserted as assistant-visible messages, then continued with code generation.

## Safety constraint
The model is aligned to output **only Python code** for user prompts, except for minimal tool call wrappers when needed.
