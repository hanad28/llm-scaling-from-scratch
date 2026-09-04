"""Batching over the flat token files produced by scaling_lm.tokenizer.

The training split is cut into non-overlapping windows of CONTEXT_LENGTH tokens.
One epoch visits every window exactly once in a seeded random order; a run of several
epochs repeats the split with a fresh shuffle each time, so every model in the sweep
sees the same windows and two runs with the same seed see them in the same order.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor

from scaling_lm.config import CONTEXT_LENGTH
from scaling_lm.tokenizer import load_tokens
from scaling_lm.validation import require_non_negative, require_positive


class TokenWindows:
    """Random access to (input, target) windows over a memory-mapped token stream."""

    def __init__(self, split_name: str, context_length: int = CONTEXT_LENGTH) -> None:
        require_positive("context_length", context_length)
        self.tokens = load_tokens(split_name)
        self.context_length = context_length
        # The target of the final window needs one extra token, hence the -1.
        self.num_windows = (len(self.tokens) - 1) // context_length

    def __len__(self) -> int:
        return self.num_windows

    def batch(self, window_indices: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return inputs and next-token targets, each (batch, context_length), for these windows."""
        starts = window_indices.astype(np.int64) * self.context_length
        offsets = np.arange(self.context_length + 1)
        # Gather (batch, context + 1) then slice to shift targets by one position.
        chunks = torch.from_numpy(self.tokens[starts[:, None] + offsets[None, :]].astype(np.int64))
        inputs = chunks[:, :-1].to(device, non_blocking=True)
        targets = chunks[:, 1:].to(device, non_blocking=True)
        return inputs, targets


def epoch_batches(
    windows: TokenWindows, batch_size: int, seed: int, epochs: int = 1
) -> Iterator[np.ndarray]:
    """Yield window-index arrays for `epochs` shuffled passes, dropping each pass's partial batch.

    One generator is seeded once and draws one permutation per pass, so the first pass of
    a multi-epoch run is identical to a single-epoch run with the same seed.
    """
    require_positive("batch_size", batch_size)
    require_non_negative("seed", seed)
    require_positive("epochs", epochs)
    generator = np.random.default_rng(seed)
    full_batches = len(windows) // batch_size
    for _ in range(epochs):
        permutation = generator.permutation(len(windows))
        for batch_index in range(full_batches):
            yield permutation[batch_index * batch_size : (batch_index + 1) * batch_size]


def sequential_batches(windows: TokenWindows, batch_size: int) -> Iterator[np.ndarray]:
    """Yield window indices in order, including a final partial batch (used for evaluation)."""
    require_positive("batch_size", batch_size)
    for start in range(0, len(windows), batch_size):
        yield np.arange(start, min(start + batch_size, len(windows)))
