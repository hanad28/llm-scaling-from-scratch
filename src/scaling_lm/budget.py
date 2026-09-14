"""Token budget per model size: how many passes over the fixed corpus each size gets.

One pass over the training split delivers the same tokens to every size, so tokens per
non-embedding parameter falls as the model grows: about 150 for the smallest size and
1.2 for the largest. MIN_TOKENS_PER_PARAMETER is a soft floor for catching severe
under-training. A size that lands clearly below it in one pass, by more than
REPETITION_TOLERANCE of the floor, repeats the split, reshuffled, until it reaches the
floor or MAX_EPOCHS passes; a size only a little under it stays at one pass. Muennighoff
et al. (2023) find that up to four passes over the same data cost little against the same
number of fresh tokens. The rule is applied by the sweep and the ablation; `python -m
scaling_lm.train --epochs N` overrides it for one run.

    python -m scaling_lm.budget    # log the plan for every size from data/corpus_stats.json
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from scaling_lm.config import (
    CONTEXT_LENGTH,
    CORPUS_STATS_PATH,
    DEFAULT_POSITIONAL_SCHEME,
    DEFAULT_TRAINING_CONFIG,
    MAX_EPOCHS,
    MIN_TOKENS_PER_PARAMETER,
    MODEL_SIZES,
    MODEL_SIZES_BY_NAME,
    REPETITION_TOLERANCE,
    ModelSize,
    TrainingConfig,
)
from scaling_lm.model import GPT, GPTConfig
from scaling_lm.validation import require_non_negative, require_positive

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenBudget:
    """What one size sees in training under the repetition rule, and how far short it falls."""

    model_size: str
    non_embedding_params: int
    tokens_per_epoch: int
    epochs: int
    target_tokens_per_parameter: float
    repetition_tolerance: float

    @property
    def tokens_seen(self) -> int:
        return self.tokens_per_epoch * self.epochs

    @property
    def one_epoch_tokens_per_parameter(self) -> float:
        return self.tokens_per_epoch / self.non_embedding_params

    @property
    def tokens_per_parameter(self) -> float:
        return self.tokens_seen / self.non_embedding_params

    @property
    def meets_target(self) -> bool:
        return self.tokens_per_parameter >= self.target_tokens_per_parameter

    @property
    def within_tolerance(self) -> bool:
        """Not clearly short of the target after training: at or above the repeat threshold."""
        threshold = repetition_threshold(
            self.target_tokens_per_parameter, self.repetition_tolerance
        )
        return self.tokens_per_parameter >= threshold

    @property
    def target_status(self) -> str:
        if self.meets_target:
            return "yes"
        if self.within_tolerance:
            return f"no, within {self.repetition_tolerance:.0%}"
        return "no"


def steps_per_epoch(train_window_count: int, training: TrainingConfig) -> int:
    """Optimiser steps one pass over the split gives: only full batches and full steps count."""
    require_non_negative("train_window_count", train_window_count)
    windows_per_step = training.batch_size_sequences * training.gradient_accumulation_steps
    return train_window_count // windows_per_step


def tokens_per_epoch(train_window_count: int, training: TrainingConfig) -> int:
    """Training tokens one pass delivers, after the partial batch at the end is dropped."""
    return steps_per_epoch(train_window_count, training) * training.tokens_per_step


def non_embedding_parameter_count(size: ModelSize) -> int:
    """Exact count from the model itself, built on the meta device so no memory is allocated.

    Token and position embeddings are excluded, so the count is the same for every
    positional scheme; the default scheme is used to build it.
    """
    config = GPTConfig.from_model_size(size, DEFAULT_POSITIONAL_SCHEME)
    with torch.device("meta"):
        model = GPT(config)
    return model.count_parameters()["non_embedding"]


def repetition_threshold(
    target_tokens_per_parameter: float = MIN_TOKENS_PER_PARAMETER,
    tolerance: float = REPETITION_TOLERANCE,
) -> float:
    """Tokens per parameter below which one pass counts as clearly short of the target."""
    require_positive("target_tokens_per_parameter", target_tokens_per_parameter)
    if not 0 <= tolerance < 1:
        raise ValueError(f"tolerance must lie in [0, 1), got {tolerance}")
    return target_tokens_per_parameter * (1 - tolerance)


def planned_epochs(
    epoch_tokens: int,
    non_embedding_params: int,
    target_tokens_per_parameter: float = MIN_TOKENS_PER_PARAMETER,
    max_epochs: int = MAX_EPOCHS,
    tolerance: float = REPETITION_TOLERANCE,
) -> int:
    """Passes for one size: one, unless a single pass falls clearly short of the target.

    The target is a soft floor. A one-pass ratio within `tolerance` (a fraction of the
    target) below it is a borderline case, not severe under-training, and stays at one
    pass. A size clearly below it takes the fewest passes that reach the target itself,
    or `max_epochs` if none does.
    """
    require_positive("epoch_tokens", epoch_tokens)
    require_positive("non_embedding_params", non_embedding_params)
    require_positive("max_epochs", max_epochs)
    one_pass = epoch_tokens / non_embedding_params
    if one_pass >= repetition_threshold(target_tokens_per_parameter, tolerance):
        return 1
    for epochs in range(2, max_epochs + 1):
        if epochs * one_pass >= target_tokens_per_parameter:
            return epochs
    return max_epochs


def token_budget(size_name: str, train_window_count: int, training: TrainingConfig) -> TokenBudget:
    """Apply the repetition rule to one size given the number of windows in the training split."""
    if size_name not in MODEL_SIZES_BY_NAME:
        raise ValueError(
            f"unknown model size {size_name!r}, expected one of {sorted(MODEL_SIZES_BY_NAME)}"
        )
    epoch_tokens = tokens_per_epoch(train_window_count, training)
    parameter_count = non_embedding_parameter_count(MODEL_SIZES_BY_NAME[size_name])
    return TokenBudget(
        model_size=size_name,
        non_embedding_params=parameter_count,
        tokens_per_epoch=epoch_tokens,
        epochs=planned_epochs(epoch_tokens, parameter_count),
        target_tokens_per_parameter=MIN_TOKENS_PER_PARAMETER,
        repetition_tolerance=REPETITION_TOLERANCE,
    )


def train_window_count_from_stats(stats_path: Path = CORPUS_STATS_PATH) -> int:
    """Windows in the training split, from corpus_stats.json, without reading the token file."""
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found; run `python -m scaling_lm.tokenizer`")
    stats = json.loads(stats_path.read_text())
    train_tokens = stats["tokens"]["train"]
    # Mirrors TokenWindows: the last window's target needs one extra token.
    return (train_tokens - 1) // CONTEXT_LENGTH


def budget_table(budgets: Sequence[TokenBudget]) -> list[str]:
    """Markdown table of the plan, one row per size, shared by the CLI and the report."""
    lines = [
        "| Size | Non-embedding params | Tokens/param, one pass | Epochs | Tokens seen | "
        "Tokens/param | Reaches target |",
        "|---|---|---|---|---|---|---|",
    ]
    for budget in budgets:
        lines.append(
            f"| {budget.model_size} | {budget.non_embedding_params:,} | "
            f"{budget.one_epoch_tokens_per_parameter:.1f} | {budget.epochs} | "
            f"{budget.tokens_seen:,} | {budget.tokens_per_parameter:.1f} | "
            f"{budget.target_status} |"
        )
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_window_count = train_window_count_from_stats()
    budgets = [
        token_budget(size.name, train_window_count, DEFAULT_TRAINING_CONFIG) for size in MODEL_SIZES
    ]
    logger.info(
        "Target %.0f tokens per non-embedding parameter; sizes below %.2f after one pass "
        "repeat, at most %d epochs; one pass is %s training tokens.",
        MIN_TOKENS_PER_PARAMETER,
        repetition_threshold(),
        MAX_EPOCHS,
        f"{budgets[0].tokens_per_epoch:,}",
    )
    logger.info("\n".join(budget_table(budgets)))


if __name__ == "__main__":
    main()
