"""The epoch-repetition rule: which sizes repeat the corpus, how often, and what they get."""

from __future__ import annotations

import json

import pytest
from conftest import SMOKE_TRAINING, SYNTHETIC_WINDOWS

from scaling_lm.budget import (
    budget_table,
    non_embedding_parameter_count,
    planned_epochs,
    steps_per_epoch,
    token_budget,
    tokens_per_epoch,
    train_window_count_from_stats,
)
from scaling_lm.config import (
    CONTEXT_LENGTH,
    CORPUS_STATS_PATH,
    DEFAULT_TRAINING_CONFIG,
    MAX_EPOCHS,
    MIN_TOKENS_PER_PARAMETER,
    MODEL_SIZES,
    MODEL_SIZES_BY_NAME,
)
from scaling_lm.model import GPT, GPTConfig
from scaling_lm.sweep import sweep_run_config

# The committed corpus: 118,703,695 training tokens, 463,686 windows of 256, 3,622 full
# steps of 128 sequences. Each size's epoch count is the plan the README documents.
EXPECTED_EPOCHS = {
    "tiny": 1,
    "small": 1,
    "medium": 1,
    "large": 2,
    "xlarge": 3,
    "xxlarge": 4,
}


def test_sweep_has_six_sizes_in_ascending_order_with_a_50m_point():
    assert [size.name for size in MODEL_SIZES] == list(EXPECTED_EPOCHS)
    counts = [non_embedding_parameter_count(size) for size in MODEL_SIZES]
    assert counts == sorted(counts)
    xlarge = MODEL_SIZES_BY_NAME["xlarge"]
    assert (xlarge.n_layer, xlarge.d_model, xlarge.n_head) == (10, 640, 10)
    assert non_embedding_parameter_count(xlarge) == pytest.approx(50e6, rel=0.02)


@pytest.mark.parametrize("size", MODEL_SIZES, ids=lambda size: size.name)
def test_meta_device_count_matches_a_real_model(size):
    real = GPT(GPTConfig.from_model_size(size, "rope")).count_parameters()["non_embedding"]
    assert non_embedding_parameter_count(size) == real


def test_planned_epochs_is_the_fewest_passes_that_reach_the_target():
    assert planned_epochs(epoch_tokens=100, non_embedding_params=10) == 1
    assert planned_epochs(epoch_tokens=50, non_embedding_params=10) == 1
    assert planned_epochs(epoch_tokens=49, non_embedding_params=10) == 2
    assert planned_epochs(epoch_tokens=25, non_embedding_params=10) == 2
    assert planned_epochs(epoch_tokens=17, non_embedding_params=10) == 3
    assert planned_epochs(epoch_tokens=13, non_embedding_params=10) == 4


def test_planned_epochs_is_capped_even_when_the_target_is_out_of_reach():
    assert planned_epochs(epoch_tokens=1, non_embedding_params=10**9) == MAX_EPOCHS
    assert planned_epochs(epoch_tokens=1, non_embedding_params=10**9, max_epochs=2) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epoch_tokens": 0, "non_embedding_params": 10},
        {"epoch_tokens": 10, "non_embedding_params": 0},
        {"epoch_tokens": 10, "non_embedding_params": 10, "target_tokens_per_parameter": 0},
        {"epoch_tokens": 10, "non_embedding_params": 10, "max_epochs": 0},
    ],
)
def test_planned_epochs_rejects_non_positive_inputs(kwargs):
    with pytest.raises(ValueError):
        planned_epochs(**kwargs)


def test_tokens_per_epoch_counts_full_batches_only():
    assert steps_per_epoch(SYNTHETIC_WINDOWS, SMOKE_TRAINING) == 4
    assert steps_per_epoch(SYNTHETIC_WINDOWS + 1, SMOKE_TRAINING) == 4
    assert tokens_per_epoch(SYNTHETIC_WINDOWS, SMOKE_TRAINING) == 4 * 2 * CONTEXT_LENGTH
    with pytest.raises(ValueError, match="train_window_count"):
        steps_per_epoch(-1, SMOKE_TRAINING)


def test_token_budget_rejects_an_unknown_size():
    with pytest.raises(ValueError, match="unknown model size"):
        token_budget("huge", SYNTHETIC_WINDOWS, SMOKE_TRAINING)


def test_committed_corpus_gives_the_documented_plan():
    train_window_count = train_window_count_from_stats(CORPUS_STATS_PATH)
    stats = json.loads(CORPUS_STATS_PATH.read_text())
    assert train_window_count == (stats["tokens"]["train"] - 1) // CONTEXT_LENGTH
    budgets = {
        size.name: token_budget(size.name, train_window_count, DEFAULT_TRAINING_CONFIG)
        for size in MODEL_SIZES
    }
    assert {name: budget.epochs for name, budget in budgets.items()} == EXPECTED_EPOCHS
    one_pass_tokens = steps_per_epoch(train_window_count, DEFAULT_TRAINING_CONFIG) * (
        DEFAULT_TRAINING_CONFIG.tokens_per_step
    )
    for budget in budgets.values():
        assert budget.tokens_per_epoch == one_pass_tokens
        assert budget.tokens_seen == one_pass_tokens * budget.epochs
        if budget.epochs > 1:
            assert budget.one_epoch_tokens_per_parameter < MIN_TOKENS_PER_PARAMETER
            previous_pass = (budget.epochs - 1) * budget.one_epoch_tokens_per_parameter
            assert previous_pass < MIN_TOKENS_PER_PARAMETER
    # The rule does not rescue the largest size: about 4.8 tokens per parameter at the cap.
    largest = budgets["xxlarge"]
    assert largest.epochs == MAX_EPOCHS
    assert not largest.meets_target
    assert largest.tokens_per_parameter == pytest.approx(4.78, abs=0.01)
    assert all(budget.meets_target for name, budget in budgets.items() if name != "xxlarge")


def test_missing_corpus_stats_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="scaling_lm.tokenizer"):
        train_window_count_from_stats(tmp_path / "corpus_stats.json")


def test_budget_table_has_one_row_per_size():
    budgets = [token_budget(size.name, SYNTHETIC_WINDOWS, SMOKE_TRAINING) for size in MODEL_SIZES]
    lines = budget_table(budgets)
    assert len(lines) == 2 + len(MODEL_SIZES)
    for line, budget in zip(lines[2:], budgets, strict=True):
        assert line.startswith(f"| {budget.model_size} |")


def test_sweep_run_config_carries_the_planned_epochs():
    # The synthetic corpus is tiny, so every size hits the cap.
    run_config = sweep_run_config("tiny", SMOKE_TRAINING, SYNTHETIC_WINDOWS)
    assert run_config.epochs == MAX_EPOCHS
    assert run_config.epochs == token_budget("tiny", SYNTHETIC_WINDOWS, SMOKE_TRAINING).epochs
