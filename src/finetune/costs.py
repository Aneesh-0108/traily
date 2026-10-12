"""Cost model for the Tinker LoRA experiment (pure functions, tested).

Prices copied from the official Models & Pricing page on 2026-10-11
(per million tokens, USD). Re-verify before any paid run:
https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/

- Qwen/Qwen3.5-4B (64K, 4B dense, Hybrid): prefill $0.33 ($0.066 cached),
  sample $1.005, TRAIN $0.737.
- Checkpoint storage $0.10/GB/month (a rank-8..32 adapter is MBs ≈ $0.00).

Token counts below were MEASURED with the model's own GGUF vocabulary
(`llama-tokenize -m models/llm/Qwen3.5-4B-Q4_K_M.gguf`): train content
averages ~294 tokens/example (system 42 / user 176 / assistant 77).
Estimates add 10% for chat-template special tokens.

Uncertainty (stated, not hidden): billing granularity/minimums are not
documented on the pages fetched; forward-only passes are billed at the
train price; exact step counts depend on batching. Estimates below are
therefore upper-ish bounds; the dry run is unambiguously cents.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen3.5-4B"
TRAIN_PRICE_PER_M = 0.737
SEQ_LEN_MEASURED = 294
TEMPLATE_OVERHEAD = 1.10

# Minimal conversion smoke test: 2 examples, batch 2, 2 steps, rank 4.
# Rationale: no documented minimum batch/steps exists; 1 forward_backward
# + 1 optim_step + save_state is the smallest chain that still performs an
# optimizer update and yields an exportable checkpoint. Two steps (not one)
# guard against step-counting edge cases; rank 4 keeps the adapter tiny
# (real runs use 32+). Batch 2 covers the whole subset per step.
DRYRUN = {
    "n_examples": 2,
    "lora_rank": 4,
    "learning_rate": 5e-4,
    "batch_size": 2,
    "steps": 2,
    "seed": 7,
}

# Intended full run on the current 36-example set (revisit if data grows).
FULLRUN = {
    "n_examples": 36,
    "lora_rank": 32,
    "learning_rate": 5e-4,
    "batch_size": 8,
    "epochs": 3,
    "seed": 7,
}


def train_tokens(n_examples: int, steps: int, batch_size: int,
                 seq_len: int = SEQ_LEN_MEASURED) -> int:
    """Approximate billed train tokens for a run configuration."""
    return int(steps * batch_size * seq_len * TEMPLATE_OVERHEAD)


def full_run_steps(n_examples: int = FULLRUN["n_examples"],
                   epochs: int = FULLRUN["epochs"],
                   batch_size: int = FULLRUN["batch_size"]) -> int:
    import math

    return math.ceil(n_examples * epochs / batch_size)


def cost_usd(tokens: int, price_per_m: float = TRAIN_PRICE_PER_M) -> float:
    return tokens / 1_000_000 * price_per_m


def dryrun_estimate() -> dict:
    tokens = train_tokens(DRYRUN["n_examples"], DRYRUN["steps"], DRYRUN["batch_size"])
    return {"tokens": tokens, "usd": cost_usd(tokens)}


def fullrun_estimate() -> dict:
    steps = full_run_steps()
    tokens = train_tokens(FULLRUN["n_examples"], steps, FULLRUN["batch_size"])
    return {"tokens": tokens, "usd": cost_usd(tokens), "steps": steps}
