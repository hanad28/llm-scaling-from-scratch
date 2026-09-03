import argparse

import numpy as np
import pytest

from scaling_lm import ablation, data, train
from scaling_lm.config import RunConfig, TrainingConfig
from scaling_lm.dataset import epoch_batches, sequential_batches
from scaling_lm.model import GPTConfig
from scaling_lm.validation import require_unique


class FakeWindows:
    def __len__(self) -> int:
        return 8


@pytest.fixture
def training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(exit_on_error=False)
    train.add_training_arguments(parser)
    return parser


@pytest.mark.parametrize(
    "option",
    ["--batch-size", "--grad-accumulation", "--eval-interval", "--max-steps"],
)
@pytest.mark.parametrize("value", ["0", "-3"])
def test_training_cli_rejects_non_positive_counts(training_parser, option, value):
    with pytest.raises(argparse.ArgumentError, match="positive integer"):
        training_parser.parse_args([option, value])


def test_training_cli_accepts_valid_counts(training_parser):
    args = training_parser.parse_args(["--batch-size", "4", "--eval-interval", "2"])
    config = train.training_config_from_args(args)
    assert (config.batch_size_sequences, config.eval_interval_steps) == (4, 2)


@pytest.mark.parametrize("argv", [["--seed", "-1"], ["--batch-size", "0", "--model-size", "tiny"]])
def test_train_main_parser_rejects_bad_values(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["train"] + argv)
    with pytest.raises(SystemExit):
        train.parse_args()


def test_ablation_parser_rejects_negative_seed(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ablation", "--seeds", "0", "-1"])
    with pytest.raises(SystemExit):
        ablation.parse_args()


@pytest.mark.parametrize("argv", [["--shards", "0"], ["--max-documents", "0"]])
def test_data_parser_rejects_non_positive_counts(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["data"] + argv)
    with pytest.raises(SystemExit):
        data.parse_args()


def test_download_shards_rejects_out_of_range_count():
    with pytest.raises(ValueError, match="shard_count"):
        data.download_shards(0)
    with pytest.raises(ValueError, match="shard_count"):
        data.download_shards(data.HF_PARQUET_SHARD_COUNT + 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size_sequences": 0},
        {"gradient_accumulation_steps": -1},
        {"eval_interval_steps": 0},
        {"max_steps": 0},
        {"grad_clip_norm": 0.0},
        {"adam_beta2": 1.0},
        {"weight_decay": -0.1},
    ],
)
def test_training_config_rejects_degenerate_values(overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        TrainingConfig(**overrides)


@pytest.mark.parametrize(
    ("model_size", "scheme", "seed", "message"),
    [
        ("huge", "learned", 0, "unknown model size"),
        ("tiny", "alibi", 0, "unknown positional scheme"),
        ("tiny", "learned", -1, "seed"),
    ],
)
def test_run_config_rejects_unknown_choices(model_size, scheme, seed, message):
    with pytest.raises(ValueError, match=message):
        RunConfig(model_size, scheme, seed, TrainingConfig())


@pytest.mark.parametrize(
    "overrides",
    [{"n_layer": 0}, {"d_model": 0, "n_head": 1}, {"context_length": -1}, {"vocab_size": 0}],
)
def test_gpt_config_rejects_non_positive_dimensions(overrides):
    fields = {"n_layer": 2, "d_model": 64, "n_head": 2, **overrides}
    with pytest.raises(ValueError, match="must be positive"):
        GPTConfig(**fields)


def test_planned_steps_rejects_a_split_too_small_for_one_step():
    with pytest.raises(ValueError, match="not enough for one step"):
        train.planned_steps(3, TrainingConfig(batch_size_sequences=4))
    assert train.planned_steps(9, TrainingConfig(batch_size_sequences=4)) == 2


@pytest.mark.parametrize("batch_size", [0, -2])
def test_batch_iterators_reject_non_positive_batch_size(batch_size):
    windows = FakeWindows()
    with pytest.raises(ValueError, match="batch_size"):
        next(epoch_batches(windows, batch_size, seed=0))
    with pytest.raises(ValueError, match="batch_size"):
        next(sequential_batches(windows, batch_size))


def test_batch_iterators_still_work_for_valid_sizes():
    windows = FakeWindows()
    assert [len(batch) for batch in sequential_batches(windows, 3)] == [3, 3, 2]
    assert sorted(np.concatenate(list(epoch_batches(windows, 4, seed=0)))) == list(range(8))


def test_require_unique_reports_the_duplicates():
    with pytest.raises(ValueError, match="repeated entries \\[1\\]"):
        require_unique("seeds", [0, 1, 1])
    with pytest.raises(ValueError, match="must not be empty"):
        require_unique("sizes", [])
    require_unique("sizes", ["tiny", "small"])
