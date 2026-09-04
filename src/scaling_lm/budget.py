"""Token budget per model size: how many passes over the fixed corpus each size gets."""

from __future__ import annotations

from scaling_lm.config import TrainingConfig
from scaling_lm.validation import require_non_negative


def steps_per_epoch(train_window_count: int, training: TrainingConfig) -> int:
    """Optimiser steps one pass over the split gives: only full batches and full steps count."""
    require_non_negative("train_window_count", train_window_count)
    windows_per_step = training.batch_size_sequences * training.gradient_accumulation_steps
    return train_window_count // windows_per_step
