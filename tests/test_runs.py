import json
from dataclasses import asdict

import pytest
from scipy import stats

from scaling_lm import runs as runs_module
from scaling_lm import train as train_module
from scaling_lm.ablation import SchemeSummary, compare_schemes, summarise_scheme
from scaling_lm.config import ResultsPaths, RunConfig, TrainingConfig
from scaling_lm.report import build_report
from scaling_lm.runs import (
    RESULT_FILENAME,
    RunResult,
    StaleRunError,
    fingerprint_specification,
    load_run,
    run_fingerprint,
    run_specification,
)
from scaling_lm.sweep import summarise_run
from scaling_lm.train import train_or_load

CORPUS_FINGERPRINT = "a" * 64
REBUILT_CORPUS_FINGERPRINT = "b" * 64


@pytest.fixture(autouse=True)
def stub_corpus(monkeypatch):
    """The token files are not available in the test environment; stand in a fixed digest."""
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: CORPUS_FINGERPRINT)


def make_result(run_config: RunConfig, validation_loss: float = 3.0) -> RunResult:
    specification = run_specification(run_config)
    return RunResult(
        run_name=run_config.run_name,
        model_size=run_config.model_size,
        positional_scheme=run_config.positional_scheme,
        seed=run_config.seed,
        parameters={"total": 1, "embedding": 1, "non_embedding": 1},
        specification=specification,
        run_fingerprint=fingerprint_specification(specification),
        total_steps=1,
        tokens_seen=1,
        peak_learning_rate=1e-3,
        final_validation_loss=validation_loss,
        final_test_loss=validation_loss,
        wall_time_seconds=1.0,
        device="cpu",
    )


def write_result(result: RunResult, paths: ResultsPaths) -> None:
    run_dir = paths.run_directory(result.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RESULT_FILENAME).write_text(json.dumps(asdict(result)))


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """A finished run on disk; training itself is disabled so reuse is the only way through."""
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4))
    write_result(make_result(run_config), paths)

    def refuse_to_train(*_args: object) -> RunResult:
        raise AssertionError("train_run should not be called when a matching result exists")

    monkeypatch.setattr(train_module, "train_run", refuse_to_train)
    return run_config, paths


def test_fingerprint_covers_corpus_architecture_and_schedule(monkeypatch):
    training = TrainingConfig(max_steps=4)
    base = RunConfig("tiny", "learned", 0, training)
    baseline = run_fingerprint(base)
    assert run_fingerprint(base) == baseline
    variants = [
        RunConfig("tiny", "rope", 0, training),
        RunConfig("tiny", "learned", 1, training),
        RunConfig("small", "learned", 0, training),
        RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4, weight_decay=0)),
    ]
    assert all(run_fingerprint(variant) != baseline for variant in variants)
    monkeypatch.setattr(runs_module, "WARMUP_FRACTION", 0.5)
    assert run_fingerprint(base) != baseline, "a schedule constant must change the fingerprint"


def test_specification_hash_is_over_the_whole_dictionary():
    specification = run_specification(RunConfig("tiny", "learned", 0, TrainingConfig()))
    baseline = fingerprint_specification(specification)
    specification["architecture"]["init_std"] = 0.05
    assert fingerprint_specification(specification) != baseline
    specification["architecture"]["init_std"] = runs_module.INIT_STD
    assert fingerprint_specification(specification) == baseline


def test_train_or_load_reuses_matching_run(saved_run):
    run_config, paths = saved_run
    assert train_or_load(run_config, paths).final_validation_loss == 3.0


def test_train_or_load_rejects_different_training_config(saved_run):
    run_config, paths = saved_run
    changed = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=8))
    with pytest.raises(StaleRunError, match="changed: \\['schedule'\\]"):
        train_or_load(changed, paths)


def test_train_or_load_rejects_rebuilt_corpus(saved_run, monkeypatch):
    run_config, paths = saved_run
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: REBUILT_CORPUS_FINGERPRINT)
    with pytest.raises(StaleRunError, match="changed: \\['corpus'\\]"):
        train_or_load(run_config, paths)


def test_train_or_load_rejects_changed_architecture_constant(saved_run, monkeypatch):
    run_config, paths = saved_run
    monkeypatch.setattr(runs_module, "LAYER_NORM_EPS", 1e-6)
    with pytest.raises(StaleRunError, match="changed: \\['architecture'\\]"):
        train_or_load(run_config, paths)


def test_load_run_by_name_applies_the_same_check(saved_run, monkeypatch):
    run_config, paths = saved_run
    assert load_run(run_config.run_name, paths).run_fingerprint == run_fingerprint(run_config)
    monkeypatch.setattr(runs_module, "KAPLAN_LR_SLOPE", 0.0)
    with pytest.raises(StaleRunError, match="changed: \\['schedule'\\]"):
        load_run(run_config.run_name, paths)


def test_load_run_rejects_edited_result_file(saved_run):
    run_config, paths = saved_run
    result_file = paths.run_directory(run_config.run_name) / RESULT_FILENAME
    payload = json.loads(result_file.read_text())
    payload["specification"]["schedule"]["max_steps"] = 8
    result_file.write_text(json.dumps(payload))
    with pytest.raises(StaleRunError, match="edited after training"):
        load_run(run_config.run_name, paths)


def test_report_loads_runs_through_the_shared_check(tmp_path, monkeypatch):
    paths = ResultsPaths(tmp_path)
    sizes = ["tiny", "small", "medium"]
    losses = [4.0, 3.5, 3.2]
    summaries = []
    for size, loss in zip(sizes, losses, strict=True):
        result = make_result(RunConfig(size, "learned", 0, TrainingConfig(max_steps=4)), loss)
        result.parameters["non_embedding"] = {"tiny": 1_000, "small": 10_000, "medium": 100_000}[
            size
        ]
        write_result(result, paths)
        summaries.append(summarise_run(result))
    paths.sweep_summary.write_text(json.dumps(summaries))
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: REBUILT_CORPUS_FINGERPRINT)
    with pytest.raises(StaleRunError, match="changed: \\['corpus'\\]"):
        build_report(paths, title="test")


def test_compare_schemes_uses_paired_test_on_matched_seeds():
    seeds = [0, 1, 2]
    learned_losses = [3.10, 3.20, 3.00]
    rope_losses = [3.05, 3.12, 2.97]
    training = TrainingConfig()
    learned = summarise_scheme(
        "learned",
        [
            make_result(RunConfig("medium", "learned", seed, training), loss)
            for seed, loss in zip(seeds, learned_losses, strict=True)
        ],
    )
    rope = summarise_scheme(
        "rope",
        [
            make_result(RunConfig("medium", "rope", seed, training), loss)
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
