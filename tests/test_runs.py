import json
from dataclasses import asdict, dataclass

import pytest
import torch
from scipy import stats

from scaling_lm import runs as runs_module
from scaling_lm import train as train_module
from scaling_lm.ablation import SchemeSummary, compare_schemes, summarise_scheme
from scaling_lm.config import INIT_STD, ResultsPaths, RunConfig, TrainingConfig
from scaling_lm.model import GPTConfig
from scaling_lm.report import build_report
from scaling_lm.runs import (
    RESULT_FILENAME,
    RunResult,
    StaleRunError,
    hash_specification,
    load_run,
    run_identity,
)
from scaling_lm.sweep import write_sweep_manifest
from scaling_lm.train import train_or_load

CORPUS_FINGERPRINT = "a" * 64
REBUILT_CORPUS_FINGERPRINT = "b" * 64


@pytest.fixture(autouse=True)
def stub_corpus(monkeypatch):
    """The token files are not available in the test environment; stand in a fixed digest."""
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: CORPUS_FINGERPRINT)


def make_result(run_config: RunConfig, validation_loss: float = 3.0) -> RunResult:
    return RunResult(
        run_name=run_config.run_name,
        model_size=run_config.model_size,
        positional_scheme=run_config.positional_scheme,
        seed=run_config.seed,
        parameters={"total": 1, "embedding": 1, "non_embedding": 1},
        identity=run_identity(run_config),
        epochs=run_config.epochs,
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


def test_fingerprint_covers_corpus_architecture_and_schedule():
    training = TrainingConfig(max_steps=4)
    base = RunConfig("tiny", "learned", 0, training)
    baseline = run_identity(base).fingerprint
    assert run_identity(base).fingerprint == baseline
    variants = [
        RunConfig("tiny", "rope", 0, training),
        RunConfig("tiny", "learned", 1, training),
        RunConfig("small", "learned", 0, training),
        RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4, weight_decay=0)),
        RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4, warmup_fraction=0.5)),
    ]
    assert all(run_identity(variant).fingerprint != baseline for variant in variants)


def test_fingerprint_distinguishes_epoch_counts_at_the_same_size():
    training = TrainingConfig(max_steps=4)
    one_epoch = run_identity(RunConfig("tiny", "learned", 0, training, epochs=1))
    assert one_epoch.specification["schedule"]["epochs"] == 1
    repeated = {
        epochs: run_identity(RunConfig("tiny", "learned", 0, training, epochs=epochs))
        for epochs in (2, 3, 4)
    }
    for epochs, identity in repeated.items():
        assert identity.specification["schedule"]["epochs"] == epochs
        assert identity.fingerprint != one_epoch.fingerprint
    assert len({identity.fingerprint for identity in repeated.values()}) == len(repeated)


def pretend_to_be_an_a40(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device=None: "NVIDIA A40")


def test_fingerprint_covers_torch_version_and_device(monkeypatch):
    run_config = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4))
    baseline = run_identity(run_config)
    assert baseline.specification["environment"] == {
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    with pytest.MonkeyPatch.context() as other_version:
        other_version.setattr(torch, "__version__", "0.0.0+changed")
        assert run_identity(run_config).fingerprint != baseline.fingerprint
    pretend_to_be_an_a40(monkeypatch)
    on_gpu = run_identity(run_config)
    assert on_gpu.specification["environment"]["device"] == "cuda:NVIDIA A40"
    assert on_gpu.fingerprint != baseline.fingerprint


def test_cpu_smoke_result_is_not_reused_on_a_gpu(saved_run, monkeypatch):
    run_config, paths = saved_run
    pretend_to_be_an_a40(monkeypatch)
    with pytest.raises(StaleRunError, match="changed: \\['environment'\\]"):
        train_or_load(run_config, paths)
    with pytest.raises(StaleRunError, match="changed: \\['environment'\\]"):
        load_run(run_config.run_name, paths)


def test_result_from_another_torch_version_is_not_reused(saved_run, monkeypatch):
    run_config, paths = saved_run
    monkeypatch.setattr(torch, "__version__", "0.0.0+changed")
    with pytest.raises(StaleRunError, match="changed: \\['environment'\\]"):
        train_or_load(run_config, paths)


def test_specification_hash_is_over_the_whole_dictionary():
    identity = run_identity(RunConfig("tiny", "learned", 0, TrainingConfig()))
    specification = identity.specification
    assert hash_specification(specification) == identity.fingerprint
    specification["architecture"]["init_std"] = 0.05
    assert hash_specification(specification) != identity.fingerprint
    specification["architecture"]["init_std"] = INIT_STD
    assert hash_specification(specification) == identity.fingerprint


def test_train_or_load_reuses_matching_run(saved_run):
    run_config, paths = saved_run
    assert train_or_load(run_config, paths).final_validation_loss == 3.0


def test_train_or_load_rejects_different_training_config(saved_run):
    run_config, paths = saved_run
    changed = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=8))
    with pytest.raises(StaleRunError, match="changed: \\['schedule'\\]"):
        train_or_load(changed, paths)


def test_train_or_load_rejects_different_epoch_count(saved_run):
    run_config, paths = saved_run
    assert run_config.epochs == 1
    repeated = RunConfig("tiny", "learned", 0, run_config.training, epochs=2)
    assert repeated.run_name == run_config.run_name
    with pytest.raises(StaleRunError, match="changed: \\['schedule'\\]"):
        train_or_load(repeated, paths)


def test_load_run_restores_the_recorded_epoch_count(tmp_path):
    paths = ResultsPaths(tmp_path)
    repeated = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4), epochs=3)
    write_result(make_result(repeated), paths)
    loaded = load_run(repeated.run_name, paths)
    assert loaded.epochs == 3
    assert loaded.identity == run_identity(repeated)
    assert loaded.identity != run_identity(RunConfig("tiny", "learned", 0, repeated.training))


def test_load_run_rejects_a_record_without_an_epoch_count(tmp_path):
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4))
    write_result(make_result(run_config), paths)
    result_file = paths.run_directory(run_config.run_name) / RESULT_FILENAME
    payload = json.loads(result_file.read_text())
    del payload["identity"]["specification"]["schedule"]["epochs"]
    # Rehash so the edited-file guard does not fire first; the missing key is what is tested.
    payload["identity"]["fingerprint"] = hash_specification(payload["identity"]["specification"])
    result_file.write_text(json.dumps(payload))
    with pytest.raises(StaleRunError, match="lacks \\['epochs'\\]"):
        load_run(run_config.run_name, paths)


def write_pre_epoch_result(run_config: RunConfig, paths: ResultsPaths) -> None:
    """A result.json as PR1 wrote it: no epoch count anywhere, fingerprint consistent."""
    payload = asdict(make_result(run_config))
    del payload["epochs"]
    del payload["identity"]["specification"]["schedule"]["epochs"]
    payload["identity"]["fingerprint"] = hash_specification(payload["identity"]["specification"])
    run_dir = paths.run_directory(run_config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RESULT_FILENAME).write_text(json.dumps(payload))


def test_pre_epoch_result_is_stale_for_the_loader_not_a_type_error(tmp_path):
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4))
    write_pre_epoch_result(run_config, paths)
    with pytest.raises(StaleRunError, match="epochs.*Delete the run directory"):
        load_run(run_config.run_name, paths)


def test_pre_epoch_result_is_stale_for_a_new_request_not_a_type_error(tmp_path, monkeypatch):
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", "learned", 0, TrainingConfig(max_steps=4))
    write_pre_epoch_result(run_config, paths)
    monkeypatch.setattr(train_module, "train_run", lambda *_args: pytest.fail("trained"))
    with pytest.raises(StaleRunError, match="epochs.*Delete the run directory"):
        train_or_load(run_config, paths)


def test_train_or_load_rejects_rebuilt_corpus(saved_run, monkeypatch):
    run_config, paths = saved_run
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: REBUILT_CORPUS_FINGERPRINT)
    with pytest.raises(StaleRunError, match="changed: \\['corpus'\\]"):
        train_or_load(run_config, paths)


def test_train_or_load_rejects_changed_architecture_default(saved_run, monkeypatch):
    @dataclass(frozen=True)
    class ChangedGPTConfig(GPTConfig):
        layer_norm_eps: float = 1e-6

    run_config, paths = saved_run
    monkeypatch.setattr(runs_module, "GPTConfig", ChangedGPTConfig)
    with pytest.raises(StaleRunError, match="changed: \\['architecture'\\]"):
        train_or_load(run_config, paths)


def test_load_run_by_name_applies_the_same_check(saved_run, monkeypatch):
    @dataclass(frozen=True)
    class ChangedTrainingConfig(TrainingConfig):
        kaplan_lr_slope: float = 0.0

    run_config, paths = saved_run
    assert load_run(run_config.run_name, paths).identity == run_identity(run_config)
    monkeypatch.setattr(runs_module, "TrainingConfig", ChangedTrainingConfig)
    with pytest.raises(StaleRunError, match="changed: \\['schedule'\\]"):
        load_run(run_config.run_name, paths)


def test_load_run_rejects_edited_result_file(saved_run):
    run_config, paths = saved_run
    result_file = paths.run_directory(run_config.run_name) / RESULT_FILENAME
    payload = json.loads(result_file.read_text())
    payload["identity"]["specification"]["schedule"]["max_steps"] = 8
    result_file.write_text(json.dumps(payload))
    with pytest.raises(StaleRunError, match="edited after training"):
        load_run(run_config.run_name, paths)


def test_report_loads_runs_through_the_shared_check(tmp_path, monkeypatch):
    paths = ResultsPaths(tmp_path)
    sizes = ["tiny", "small", "medium"]
    losses = [4.0, 3.5, 3.2]
    results = []
    for size, loss in zip(sizes, losses, strict=True):
        result = make_result(RunConfig(size, "learned", 0, TrainingConfig(max_steps=4)), loss)
        result.parameters["non_embedding"] = {"tiny": 1_000, "small": 10_000, "medium": 100_000}[
            size
        ]
        write_result(result, paths)
        results.append(result)
    write_sweep_manifest(results, paths)
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
    welch = stats.ttest_ind(rope_losses, learned_losses, equal_var=False)
    assert comparison["welch_t_statistic"] == pytest.approx(float(welch.statistic))
    assert comparison["welch_p_value"] == pytest.approx(float(welch.pvalue))
    assert comparison["paired_p_value"] != pytest.approx(float(welch.pvalue))


def test_compare_schemes_requires_matched_seeds():
    learned = SchemeSummary("learned", [0, 1], [3.0, 3.1], 3.05, 0.07, [3.0, 3.1], 3.05)
    rope = SchemeSummary("rope", [0, 2], [3.0, 3.1], 3.05, 0.07, [3.0, 3.1], 3.05)
    with pytest.raises(ValueError, match="same seeds"):
        compare_schemes(learned, rope)
