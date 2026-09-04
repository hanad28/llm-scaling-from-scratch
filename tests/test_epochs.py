"""A run of several epochs repeats the training split, reshuffled, and is planned accordingly."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import SMOKE_TRAINING, SYNTHETIC_WINDOWS

from scaling_lm import train
from scaling_lm.budget import steps_per_epoch
from scaling_lm.config import ResultsPaths, RunConfig, TrainingConfig
from scaling_lm.dataset import TokenWindows, epoch_batches

BATCH_SIZE = 3
FULL_BATCHES_PER_EPOCH = SYNTHETIC_WINDOWS // BATCH_SIZE


def passes(batches: list[np.ndarray]) -> list[np.ndarray]:
    """Split a multi-epoch batch stream back into the window indices of each pass."""
    return [
        np.concatenate(batches[start : start + FULL_BATCHES_PER_EPOCH])
        for start in range(0, len(batches), FULL_BATCHES_PER_EPOCH)
    ]


def test_each_epoch_covers_the_same_full_batch_windows_once(synthetic_corpus):
    windows = TokenWindows("train")
    batches = list(epoch_batches(windows, BATCH_SIZE, seed=0, epochs=3))
    assert len(batches) == 3 * FULL_BATCHES_PER_EPOCH
    assert all(len(batch) == BATCH_SIZE for batch in batches)
    for pass_indices in passes(batches):
        assert len(np.unique(pass_indices)) == len(pass_indices)
        assert len(pass_indices) == FULL_BATCHES_PER_EPOCH * BATCH_SIZE


def test_epochs_are_reshuffled_and_the_first_matches_a_single_pass(synthetic_corpus):
    windows = TokenWindows("train")
    single = np.concatenate(list(epoch_batches(windows, BATCH_SIZE, seed=0)))
    first, second, third = passes(list(epoch_batches(windows, BATCH_SIZE, seed=0, epochs=3)))
    assert np.array_equal(first, single)
    assert not np.array_equal(first, second)
    assert not np.array_equal(second, third)


def test_multi_epoch_order_is_reproducible(synthetic_corpus):
    windows = TokenWindows("train")
    first = list(epoch_batches(windows, BATCH_SIZE, seed=4, epochs=2))
    again = list(epoch_batches(windows, BATCH_SIZE, seed=4, epochs=2))
    assert all(np.array_equal(left, right) for left, right in zip(first, again, strict=True))


def test_epoch_batches_rejects_a_non_positive_epoch_count(synthetic_corpus):
    windows = TokenWindows("train")
    with pytest.raises(ValueError, match="epochs"):
        next(epoch_batches(windows, BATCH_SIZE, seed=0, epochs=0))


def test_planned_steps_multiplies_by_epochs_then_caps():
    training = TrainingConfig(batch_size_sequences=4)
    assert steps_per_epoch(9, training) == 2
    assert train.planned_steps(9, training) == 2
    assert train.planned_steps(9, training, epochs=3) == 6
    assert train.planned_steps(9, replace(training, max_steps=5), epochs=3) == 5
    accumulating = TrainingConfig(batch_size_sequences=4, gradient_accumulation_steps=2)
    assert train.planned_steps(17, accumulating, epochs=2) == 4
    with pytest.raises(ValueError, match="epochs"):
        train.planned_steps(9, training, epochs=0)


def test_run_config_rejects_a_non_positive_epoch_count():
    with pytest.raises(ValueError, match="epochs"):
        RunConfig("tiny", epochs=0)


def test_two_epochs_double_the_steps_and_are_recorded(synthetic_corpus, tmp_path):
    training = replace(SMOKE_TRAINING, max_steps=None)
    one_epoch = RunConfig("tiny", training=training, epochs=1)
    two_epochs = RunConfig("tiny", training=training, epochs=2)
    single = train.train_run(one_epoch, ResultsPaths(tmp_path / "one"))
    repeated = train.train_run(two_epochs, ResultsPaths(tmp_path / "two"))
    epoch_steps = steps_per_epoch(SYNTHETIC_WINDOWS, training)
    assert (single.epochs, single.total_steps) == (1, epoch_steps)
    assert (repeated.epochs, repeated.total_steps) == (2, 2 * epoch_steps)
    assert repeated.tokens_seen == 2 * single.tokens_seen
    assert repeated.history[-1].step == repeated.total_steps - 1
    assert repeated.identity != single.identity
