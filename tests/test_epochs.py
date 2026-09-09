"""A run of several epochs repeats the training split, reshuffled, and is planned accordingly."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest
from conftest import SMOKE_TRAINING, SYNTHETIC_WINDOWS

from scaling_lm import train
from scaling_lm.budget import steps_per_epoch
from scaling_lm.config import ResultsPaths, RunConfig, TrainingConfig
from scaling_lm.dataset import TokenWindows, epoch_batches
from scaling_lm.runs import planned_epochs_of, stopped_early

BATCH_SIZE = 3
FULL_BATCHES_PER_EPOCH = SYNTHETIC_WINDOWS // BATCH_SIZE
# Two-window batches accumulated three per step: four full batches per pass, of which
# three make a step and the fourth (two windows) must be dropped, not carried over.
ACCUMULATING = TrainingConfig(
    batch_size_sequences=2,
    gradient_accumulation_steps=3,
    max_steps=None,
    eval_interval_steps=1,
    use_mixed_precision=False,
)
WINDOWS_PER_ACCUMULATED_STEP = (
    ACCUMULATING.batch_size_sequences * ACCUMULATING.gradient_accumulation_steps
)
DROPPED_WINDOWS_PER_PASS = SYNTHETIC_WINDOWS % WINDOWS_PER_ACCUMULATED_STEP


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


def accumulated_batches(epochs: int) -> list[np.ndarray]:
    return list(
        epoch_batches(
            TokenWindows("train"),
            ACCUMULATING.batch_size_sequences,
            seed=0,
            epochs=epochs,
            micro_batches_per_step=ACCUMULATING.gradient_accumulation_steps,
        )
    )


def test_no_optimiser_step_mixes_windows_from_two_epochs(synthetic_corpus):
    assert DROPPED_WINDOWS_PER_PASS > 0, "the fixture must leave a remainder to prove anything"
    epochs = 2
    batches = accumulated_batches(epochs)
    steps = steps_per_epoch(SYNTHETIC_WINDOWS, ACCUMULATING)
    micro_batches_per_pass = steps * ACCUMULATING.gradient_accumulation_steps
    assert len(batches) == epochs * micro_batches_per_pass
    # Rebuild each pass's permutation the way the loader must: one generator, one draw per
    # pass. Every step's micro-batches then lie inside a single pass's prefix.
    generator = np.random.default_rng(0)
    for epoch_index in range(epochs):
        permutation = generator.permutation(SYNTHETIC_WINDOWS)
        start = epoch_index * micro_batches_per_pass
        seen = np.concatenate(batches[start : start + micro_batches_per_pass])
        expected = permutation[: SYNTHETIC_WINDOWS - DROPPED_WINDOWS_PER_PASS]
        assert np.array_equal(seen, expected)
        assert len(np.unique(seen)) == len(seen)


def test_dropped_windows_are_logged_per_pass(synthetic_corpus, caplog):
    with caplog.at_level(logging.INFO, logger="scaling_lm.dataset"):
        accumulated_batches(epochs=2)
    expected = f"{DROPPED_WINDOWS_PER_PASS} of {SYNTHETIC_WINDOWS} windows"
    assert expected in caplog.text


def test_accumulated_multi_epoch_run_drops_the_remainder_and_says_so(
    synthetic_corpus, tmp_path, caplog
):
    run_config = RunConfig("tiny", training=ACCUMULATING, epochs=2)
    with caplog.at_level(logging.INFO):
        result = train.train_run(run_config, ResultsPaths(tmp_path))
    steps = steps_per_epoch(SYNTHETIC_WINDOWS, ACCUMULATING)
    assert result.total_steps == 2 * steps
    assert result.tokens_seen == 2 * steps * ACCUMULATING.tokens_per_step
    assert f"{DROPPED_WINDOWS_PER_PASS} of {SYNTHETIC_WINDOWS} windows" in caplog.text


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
    assert (single.epochs_completed, single.total_steps) == (1.0, epoch_steps)
    assert (repeated.epochs_completed, repeated.total_steps) == (2.0, 2 * epoch_steps)
    assert repeated.tokens_seen == 2 * single.tokens_seen
    assert not stopped_early(single) and not stopped_early(repeated)
    assert repeated.history[-1].step == repeated.total_steps - 1
    assert repeated.identity != single.identity


def test_a_run_capped_by_max_steps_records_the_fraction_it_completed(
    synthetic_corpus, tmp_path, caplog
):
    epoch_steps = steps_per_epoch(SYNTHETIC_WINDOWS, SMOKE_TRAINING)
    capped_steps = epoch_steps + 1
    capped = replace(SMOKE_TRAINING, max_steps=capped_steps)
    run_config = RunConfig("tiny", training=capped, epochs=3)
    with caplog.at_level(logging.INFO, logger="scaling_lm.train"):
        result = train.train_run(run_config, ResultsPaths(tmp_path))
    assert result.total_steps == capped_steps
    assert result.tokens_seen == capped_steps * capped.tokens_per_step
    assert result.epochs_completed == pytest.approx(capped_steps / epoch_steps)
    assert result.epochs_completed < run_config.epochs
    assert planned_epochs_of(result) == run_config.epochs
    assert stopped_early(result)
    assert "max_steps" in caplog.text and "of 3 passes" in caplog.text
