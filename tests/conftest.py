"""Shared fixtures: a synthetic corpus small enough to train real runs inside a test."""

from __future__ import annotations

import numpy as np
import pytest

from scaling_lm import dataset as dataset_module
from scaling_lm import runs as runs_module
from scaling_lm.config import CONTEXT_LENGTH, VOCAB_SIZE, TrainingConfig

STUB_CORPUS_FINGERPRINT = "d" * 64
SYNTHETIC_WINDOWS = 8
# One optimiser step on two windows; enough for real result files. Only CLI options are
# set, because `load_run` rebuilds a run's request from the CLI options it records.
SMOKE_TRAINING = TrainingConfig(
    batch_size_sequences=2, max_steps=1, eval_interval_steps=1, use_mixed_precision=False
)


def synthetic_tokens(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    length = SYNTHETIC_WINDOWS * CONTEXT_LENGTH + 1
    return generator.integers(0, VOCAB_SIZE, size=length, dtype=np.uint16)


@pytest.fixture
def synthetic_corpus(monkeypatch):
    """Serve every split from an in-memory token stream the test can swap out.

    The corpus fingerprint is pinned, so swapping the tokens changes what a retrained
    run measures without changing the run's identity: a regenerated run of the same name.
    """
    stream = {"tokens": synthetic_tokens(seed=0)}
    monkeypatch.setattr(dataset_module, "load_tokens", lambda _split_name: stream["tokens"])
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: STUB_CORPUS_FINGERPRINT)

    def regenerate(seed: int) -> None:
        stream["tokens"] = synthetic_tokens(seed)

    return regenerate
