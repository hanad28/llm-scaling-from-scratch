import json
from dataclasses import asdict

import pytest
from scipy import stats

from scaling_lm import train as train_module
from scaling_lm.ablation import SchemeSummary, compare_schemes, summarise_scheme
from scaling_lm.config import ResultsPaths, RunConfig, TrainingConfig
from scaling_lm.train import RESULT_FILENAME, RunResult, train_or_load

FINGERPRINT = "a" * 64


def make_result(run_config: RunConfig, fingerprint: str, validation_loss: float = 3.0) -> RunResult:
    return RunResult(
        run_name=run_config.run_name,
        model_size=run_config.model_size,
        positional_scheme=run_config.positional_scheme,
        seed=run_config.seed,
        parameters={"total": 1, "embedding": 1, "non_embedding": 1},
        architecture={"n_layer": 1, "d_model": 1, "n_head": 1, "context_length": 1},
        training=asdict(run_config.training),
        corpus_fingerprint=fingerprint,
        total_steps=1,
        tokens_seen=1,
        peak_learning_rate=1e-3,
        final_validation_loss=validation_loss,
        final_test_loss=validation_loss,
        wall_time_seconds=1.0,
        device="cpu",
    )


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """A finished run on disk plus a stubbed corpus fingerprint; training itself is disabled."""
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4))
    run_dir = paths.run_directory(run_config.run_name)
    run_dir.mkdir(parents=True)
    result = make_result(run_config, FINGERPRINT)
    (run_dir / RESULT_FILENAME).write_text(json.dumps(asdict(result)))
    monkeypatch.setattr(train_module, "corpus_fingerprint", lambda: FINGERPRINT)

    def refuse_to_train(*_args: object) -> RunResult:
        raise AssertionError("train_run should not be called when a matching result exists")

    monkeypatch.setattr(train_module, "train_run", refuse_to_train)
    return run_config, paths


def test_train_or_load_reuses_matching_run(saved_run):
    run_config, paths = saved_run
    assert train_or_load(run_config, paths).final_validation_loss == 3.0


def test_train_or_load_rejects_different_training_config(saved_run):
    run_config, paths = saved_run
    changed = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=8))
    with pytest.raises(RuntimeError, match="trained with"):
        train_or_load(changed, paths)


def test_train_or_load_rejects_rebuilt_corpus(saved_run, monkeypatch):
    run_config, paths = saved_run
    monkeypatch.setattr(train_module, "corpus_fingerprint", lambda: "b" * 64)
    with pytest.raises(RuntimeError, match="token files have changed"):
        train_or_load(run_config, paths)


def test_compare_schemes_uses_paired_test_on_matched_seeds():
    seeds = [0, 1, 2]
    learned_losses = [3.10, 3.20, 3.00]
    rope_losses = [3.05, 3.12, 2.97]
    training = TrainingConfig()
    learned = summarise_scheme(
        "learned",
        [
            make_result(RunConfig("medium", "learned", seed, training), FINGERPRINT, loss)
            for seed, loss in zip(seeds, learned_losses, strict=True)
        ],
    )
    rope = summarise_scheme(
        "rope",
        [
            make_result(RunConfig("medium", "rope", seed, training), FINGERPRINT, loss)
            for seed, loss in zip(seeds, rope_losses, strict=True)
        ],
    )
    comparison = compare_schemes(learned, rope)
    expected = stats.ttest_rel(rope_losses, learned_losses)
    assert comparison["mean_difference"] == pytest.approx(-0.16 / 3)
    assert comparison["paired_t_statistic"] == pytest.approx(float(expected.statistic))
    assert comparison["paired_p_value"] == pytest.approx(float(expected.pvalue))
    independent = stats.ttest_ind(rope_losses, learned_losses, equal_var=False)
    assert comparison["paired_p_value"] != pytest.approx(float(independent.pvalue))


def test_compare_schemes_requires_matched_seeds():
    learned = SchemeSummary("learned", [0, 1], [3.0, 3.1], 3.05, 0.07, [3.0, 3.1], 3.05)
    rope = SchemeSummary("rope", [0, 2], [3.0, 3.1], 3.05, 0.07, [3.0, 3.1], 3.05)
    with pytest.raises(ValueError, match="same seeds"):
        compare_schemes(learned, rope)
