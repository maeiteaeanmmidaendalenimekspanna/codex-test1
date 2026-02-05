# Training Plan (From Scratch)

## Stage 0: Tokenizer
- Train a BPE/Unigram tokenizer on Python-only corpus.
- Include special tokens: `<file_sep>`, `<tool_call>`, `<tool_result>`.

## Stage 1: Base pretraining
- Mix: 90% real Python code, 10% synthetic tasks.
- Objective: next-token prediction only.
- Target tokens: 1.2T.

## Stage 2: Reasoning curriculum
- Add curated tasks that require multi-step reasoning (e.g., algorithmic problems).
- Use chain-of-thought **hidden** (stored in data but masked from loss) and expose only final code.

## Stage 3: Tool-use training
- Include tool call traces with deterministic tool results.
- Enforce tool schema and format.

## Stage 4: Alignment to “code-only” output
- Supervised finetuning with instruction prompts that require **Python-only** responses.
- Reject any natural language outputs during evaluation.

## Evaluation
- Functional tests with unit-test harnesses.
- Code-only benchmarks (HumanEval-Python, MBPP-Python).
- Static analysis to ensure no non-Python text.
