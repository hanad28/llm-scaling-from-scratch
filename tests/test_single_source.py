"""There is one way to say what a run is, and one place a run's loss can be read from.

The pipeline tests here drive only the public entry points (train_run, run_sweep,
run_ablation, build_report) over a tiny synthetic corpus, so they hold regardless of
how results happen to be laid out on disk.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest
from conftest import SMOKE_TRAINING, SYNTHETIC_WINDOWS

import scaling_lm
from scaling_lm import report as report_module
from scaling_lm.ablation import run_ablation
from scaling_lm.budget import token_budget
from scaling_lm.config import ABLATION_POSITIONAL_SCHEMES, ResultsPaths, RunConfig
from scaling_lm.report import build_report
from scaling_lm.runs import resolve_run
from scaling_lm.scaling_fit import fit_power_law
from scaling_lm.sweep import run_sweep, sweep_run_config
from scaling_lm.train import train_run

PACKAGE_DIR = Path(scaling_lm.__file__).parent
SWEEP_SIZES = ("tiny", "small", "medium")
ABLATION_SEEDS = (0, 1)
FIT_BOOTSTRAP_RESAMPLES = 50

# Names that turn configs into a run specification or hash one into a fingerprint.
IDENTITY_NAMES = {
    "run_specification",
    "fingerprint_specification",
    "run_fingerprint",
    "run_identity",
    "hash_specification",
    "sha256",
}
# The one function allowed to assemble a run's identity, and the module it lives in.
IDENTITY_ASSEMBLER = ("runs.py", "run_identity")
# Hashes, but of something other than a run: the token files (an input to run_identity)
# and an already-assembled specification (to detect a hand-edited result file).
ALLOWED_HASHERS = {
    ("tokenizer.py", "corpus_fingerprint"),
    ("runs.py", "hash_specification"),
    ("runs.py", "verify_result"),
}


@pytest.fixture
def quick_bootstrap(monkeypatch):
    monkeypatch.setattr(
        report_module,
        "fit_power_law",
        lambda counts, losses: fit_power_law(counts, losses, FIT_BOOTSTRAP_RESAMPLES),
    )


def retrain(run_config: RunConfig, paths: ResultsPaths, regenerate, seed: int) -> float:
    """Delete a finished run and train it again on different tokens; return the new loss."""
    shutil.rmtree(paths.run_directory(run_config.run_name))
    regenerate(seed)
    return train_run(run_config, paths).final_validation_loss


def recorded_loss(run_config: RunConfig, paths: ResultsPaths) -> float:
    result = resolve_run(run_config, paths)
    assert result is not None
    return result.final_validation_loss


def calls_in(function: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def identity_calls() -> dict[tuple[str, str], set[str]]:
    """{(module, function): identity-producing names it calls}, over the whole package."""
    found = {}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                names = calls_in(node) & IDENTITY_NAMES
                if names:
                    found[(path.name, node.name)] = names
    return found


def identity_names_imported_outside_runs() -> dict[str, set[str]]:
    imported = {}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name == IDENTITY_ASSEMBLER[0]:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names} & IDENTITY_NAMES
                if names:
                    imported[path.name] = names
    return imported


def test_run_identity_is_assembled_by_one_function_and_recorded_verbatim():
    """`run_identity` is the only function that builds a specification or hashes one.

    Any other function may call `run_identity` to obtain the finished identity (training
    does, to record it) but may not call anything that assembles or hashes one itself.
    """
    calls = identity_calls()
    assert IDENTITY_ASSEMBLER in calls
    assembling_elsewhere = {
        function: names - {IDENTITY_ASSEMBLER[1]}
        for function, names in calls.items()
        if function not in ALLOWED_HASHERS | {IDENTITY_ASSEMBLER}
        and names - {IDENTITY_ASSEMBLER[1]}
    }
    assert assembling_elsewhere == {}, f"identity assembled in {assembling_elsewhere}"
    imported = identity_names_imported_outside_runs()
    assert imported == {"train.py": {IDENTITY_ASSEMBLER[1]}}, imported


def test_report_fits_the_loss_in_result_json_not_a_copy_in_the_sweep_file(
    tmp_path, synthetic_corpus, quick_bootstrap
):
    paths = ResultsPaths(tmp_path)
    run_sweep(SWEEP_SIZES, SMOKE_TRAINING, paths)
    sweep_configs = [
        sweep_run_config(size, SMOKE_TRAINING, SYNTHETIC_WINDOWS) for size in SWEEP_SIZES
    ]
    stale_loss = recorded_loss(sweep_configs[0], paths)

    fresh_loss = retrain(sweep_configs[0], paths, synthetic_corpus, seed=1)
    assert fresh_loss != stale_loss

    fit = build_report(paths, title="test")
    current = [resolve_run(config, paths) for config in sweep_configs]
    expected = fit_power_law(
        [result.parameters["non_embedding"] for result in current],
        [result.final_validation_loss for result in current],
        FIT_BOOTSTRAP_RESAMPLES,
    )
    assert fit.alpha == pytest.approx(expected.alpha)
    assert f"{fresh_loss:.4f}" in paths.summary_markdown.read_text()


def test_report_reads_ablation_losses_through_load_run(
    tmp_path, synthetic_corpus, quick_bootstrap, monkeypatch
):
    paths = ResultsPaths(tmp_path)
    run_sweep(SWEEP_SIZES, SMOKE_TRAINING, paths)
    run_ablation(ABLATION_SEEDS, SMOKE_TRAINING, paths, model_size="tiny")
    baseline_scheme, alternative_scheme = ABLATION_POSITIONAL_SCHEMES
    epochs = token_budget("tiny", SYNTHETIC_WINDOWS, SMOKE_TRAINING).epochs
    regenerated = RunConfig(
        "tiny", alternative_scheme, ABLATION_SEEDS[-1], SMOKE_TRAINING, epochs=epochs
    )
    stale_loss = recorded_loss(regenerated, paths)
    fresh_loss = retrain(regenerated, paths, synthetic_corpus, seed=1)
    assert fresh_loss != stale_loss

    loaded_names = []
    verified_load_run = report_module.load_run

    def observed_load_run(run_name: str, paths: ResultsPaths):
        loaded_names.append(run_name)
        return verified_load_run(run_name, paths)

    monkeypatch.setattr(report_module, "load_run", observed_load_run)
    build_report(paths, title="test")

    ablation_names = {
        RunConfig("tiny", scheme, seed, SMOKE_TRAINING, epochs=epochs).run_name
        for scheme in (baseline_scheme, alternative_scheme)
        for seed in ABLATION_SEEDS
    }
    assert ablation_names <= set(loaded_names), sorted(ablation_names - set(loaded_names))
    assert f"{fresh_loss:.4f}" in paths.summary_markdown.read_text()
