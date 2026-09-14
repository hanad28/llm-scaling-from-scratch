"""The summary separates fit uncertainty from seed-to-seed variance and discloses the budget."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import STUB_CORPUS_FINGERPRINT
from test_runs import make_result, write_result

from scaling_lm import runs as runs_module
from scaling_lm.ablation import AblationManifest, analyse_ablation
from scaling_lm.config import (
    ABLATION_POSITIONAL_SCHEMES,
    DEFAULT_POSITIONAL_SCHEME,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.plots import training_curve_label, training_curves_title
from scaling_lm.report import (
    data_constraint_note,
    fit_section,
    seed_spread_of_sweep_scheme,
    seed_variance_section,
    sweep_section,
    verified_generations,
)
from scaling_lm.runs import RESULT_FILENAME, RunResult
from scaling_lm.scaling_fit import alpha_std_from_loss_noise, fit_power_law

PARAMETER_COUNTS = [793_344, 4_739_072, 12_422_016, 25_220_096, 49_236_480, 99_231_744]
EPOCHS = [1, 1, 1, 1, 3, 4]
TOKENS_PER_EPOCH = 118_685_696


@pytest.fixture(autouse=True)
def stub_corpus(monkeypatch):
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: STUB_CORPUS_FINGERPRINT)


def sweep_results() -> list[RunResult]:
    sizes = ["tiny", "small", "medium", "large", "xlarge", "xxlarge"]
    results = []
    for size, count, epochs in zip(sizes, PARAMETER_COUNTS, EPOCHS, strict=True):
        config = RunConfig(size, DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(), epochs=epochs)
        result = make_result(config, validation_loss=10.0 * count**-0.076)
        results.append(
            replace(
                result,
                parameters={"total": count + 1, "embedding": 1, "non_embedding": count},
                tokens_seen=TOKENS_PER_EPOCH * epochs,
            )
        )
    return results


CAPPED_FRACTION = 0.5


def capped_sweep_results() -> list[RunResult]:
    """The sweep with xxlarge stopped by max_steps halfway through its first of four passes."""
    results = sweep_results()
    results[-1] = replace(
        results[-1],
        epochs_completed=CAPPED_FRACTION,
        tokens_seen=int(TOKENS_PER_EPOCH * CAPPED_FRACTION),
    )
    return results


def ablation_report(seed_losses: list[float]):
    """Both ablation schemes over the same seeds, with these losses for the default scheme."""
    per_scheme = {
        scheme: [
            make_result(RunConfig("medium", scheme, seed, TrainingConfig()), validation_loss=loss)
            for seed, loss in enumerate(seed_losses)
        ]
        for scheme in ABLATION_POSITIONAL_SCHEMES
    }
    manifest = AblationManifest(
        model_size="medium",
        seeds=list(range(len(seed_losses))),
        run_names={
            scheme: [result.run_name for result in results]
            for scheme, results in per_scheme.items()
        },
    )
    return manifest, analyse_ablation(per_scheme)


def test_sweep_table_reports_epochs_and_tokens_per_parameter():
    lines = sweep_section(sweep_results())
    assert "| Epochs |" in lines[0] and "| Tokens/param |" in lines[0]
    xxlarge_row = next(line for line in lines if line.startswith("| xxlarge |"))
    assert "| 4 |" in xxlarge_row
    assert "| 4.8 |" in xxlarge_row
    tiny_row = next(line for line in lines if line.startswith("| tiny |"))
    assert "| 1 |" in tiny_row
    assert "| 149.6 |" in tiny_row


def test_sweep_table_shows_a_capped_run_as_a_fraction_of_its_plan():
    lines = sweep_section(capped_sweep_results())
    xxlarge_row = next(line for line in lines if line.startswith("| xxlarge |"))
    assert "| 0.5 of 4 |" in xxlarge_row
    assert "| 0.6 |" in xxlarge_row
    assert "| 4 |" not in xxlarge_row
    note = " ".join(lines)
    assert "Stopped early by max_steps" in note
    assert "xxlarge (0.5 of 4 passes)" in note


def test_data_constraint_note_is_silent_about_max_steps_when_every_run_finished():
    assert "max_steps" not in " ".join(data_constraint_note(sweep_results()))


def test_data_constraint_note_names_the_repeated_borderline_and_still_short_sizes():
    note = " ".join(data_constraint_note(sweep_results()))
    assert "clearly below 5 tokens per non-embedding parameter" in note
    assert "Repeated here: xlarge, xxlarge." in note
    assert "Within 10% of the target and left at one pass: large." in note
    assert "Still below the target after training: xxlarge." in note
    assert "Muennighoff et al., 2023" in note


def test_fit_section_labels_the_intervals_as_fit_uncertainty():
    results = sweep_results()
    fit = fit_power_law(
        [result.parameters["non_embedding"] for result in results],
        [result.final_validation_loss for result in results],
        bootstrap_resamples=50,
    )
    text = "\n".join(fit_section(fit))
    assert "over 6 sizes" in text
    assert "Fit uncertainty across the 6 sweep points" in text
    assert "regression standard error" in text


def test_seed_variance_section_uses_the_ablation_spread_for_the_sweep_scheme():
    results = sweep_results()
    counts = [result.parameters["non_embedding"] for result in results]
    losses = [result.final_validation_loss for result in results]
    fit = fit_power_law(counts, losses, bootstrap_resamples=50)
    ablation = ablation_report([3.00, 3.02, 3.01])
    spread = seed_spread_of_sweep_scheme(ablation)
    assert spread is not None
    assert (spread.model_size, spread.seed_count) == ("medium", 3)
    assert spread.loss_std == pytest.approx(float(np.std([3.00, 3.02, 3.01], ddof=1)))

    text = "\n".join(seed_variance_section(results, fit, ablation))
    expected_alpha_std = alpha_std_from_loss_noise(counts, losses, spread.loss_std)
    assert "one seed per size" in text
    assert "Rough scale only" in text
    assert f"about {expected_alpha_std:.4f} from seed noise alone" in text
    assert f"regression standard error of {fit.alpha_standard_error:.4f}" in text
    assert "not added to the fit interval" in text


@pytest.mark.parametrize("seed_losses", [None, [3.0]], ids=["no ablation", "one seed"])
def test_seed_variance_section_says_when_the_spread_is_unmeasured(seed_losses):
    ablation = None if seed_losses is None else ablation_report(seed_losses)
    results = sweep_results()
    fit = fit_power_law(
        [result.parameters["non_embedding"] for result in results],
        [result.final_validation_loss for result in results],
        bootstrap_resamples=50,
    )
    assert seed_spread_of_sweep_scheme(ablation) is None
    text = "\n".join(seed_variance_section(results, fit, ablation))
    assert "no measurement of this spread" in text


def test_training_curve_labels_carry_each_models_epoch_count():
    results = sweep_results()
    labels = [training_curve_label(result) for result in results]
    assert labels[0] == "tiny (0.8M, 1 pass)"
    assert labels[4] == "xlarge (49.2M, 3 passes)"
    assert labels[5] == "xxlarge (99.2M, 4 passes)"
    assert training_curves_title(results) == "Validation loss during training (1 to 4 passes)"
    assert training_curves_title(results[:3]) == "Validation loss during the single training pass"
    assert "single" not in training_curves_title(results[5:])


def test_training_curve_labels_and_title_show_when_max_steps_cut_a_run_short():
    results = capped_sweep_results()
    assert training_curve_label(results[-1]) == "xxlarge (99.2M, 0.5 of 4 passes)"
    title = training_curves_title(results)
    assert "max_steps" in title
    assert "1 to 4 passes" not in title
    assert "single" not in training_curves_title([results[-1]])


def test_verified_generations_keeps_samples_matching_the_current_run(tmp_path):
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(max_steps=4))
    result = make_result(run_config)
    write_result(result, paths)
    generations = {
        run_config.run_name: {
            "fingerprint": result.identity.fingerprint,
            "samples": {"prompt": "a continuation"},
        }
    }
    assert verified_generations(generations, paths) == {
        run_config.run_name: {"prompt": "a continuation"}
    }


def test_verified_generations_drops_samples_after_a_config_change_same_size_name(tmp_path):
    """Fail-then-pass proof: samples generated under one config must not survive display

    once the same run name is retrained under a changed config (`verified_generations`
    did not exist before this fix; generation_section trusted generations.json outright).
    """
    paths = ResultsPaths(tmp_path)
    original = RunConfig("tiny", DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(max_steps=4))
    original_fingerprint = make_result(original).identity.fingerprint
    stale_samples = {"fingerprint": original_fingerprint, "samples": {"p": "old"}}
    generations = {original.run_name: stale_samples}

    # Retrain the same run name under a changed config: same size, different max_steps.
    changed = RunConfig("tiny", DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(max_steps=8))
    assert changed.run_name == original.run_name
    write_result(make_result(changed), paths)

    assert verified_generations(generations, paths) == {}


def test_verified_generations_drops_samples_for_a_run_that_no_longer_verifies(tmp_path):
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(max_steps=4))
    generations = {run_config.run_name: {"fingerprint": "stale", "samples": {"p": "old"}}}
    # No result.json on disk at all: load_run fails, not just a fingerprint mismatch.
    assert verified_generations(generations, paths) == {}


def test_verified_generations_drops_samples_for_a_run_with_an_unparseable_result(tmp_path):
    """Fail-then-pass proof: one corrupted result.json referenced from generations.json

    must not take the whole report down. Before this fix, `verified_generations` only
    caught `StaleRunError` and `FileNotFoundError`; a genuinely malformed record (never
    valid JSON, or missing/mistyped pieces `read_result` indexes before validating)
    raised `json.JSONDecodeError`, `KeyError` or `TypeError` straight out of this
    function instead of being dropped like any other unverifiable entry.
    """
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(max_steps=4))
    run_dir = paths.run_directory(run_config.run_name)
    run_dir.mkdir(parents=True)
    (run_dir / RESULT_FILENAME).write_text("{not valid json")
    generations = {run_config.run_name: {"fingerprint": "whatever", "samples": {"p": "old"}}}
    assert verified_generations(generations, paths) == {}


def test_verified_generations_tolerates_a_matching_entry_with_no_samples_key(tmp_path):
    """Fail-then-pass proof: a hand-edited entry with the right fingerprint but no

    "samples" key (not something `generate.py` itself would ever write) must not crash
    the report either. Before this fix, `entry["samples"]` was indexed unconditionally
    once the fingerprint matched, raising `KeyError` for this entry.
    """
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", DEFAULT_POSITIONAL_SCHEME, 0, TrainingConfig(max_steps=4))
    result = make_result(run_config)
    write_result(result, paths)
    generations = {run_config.run_name: {"fingerprint": result.identity.fingerprint}}
    assert verified_generations(generations, paths) == {run_config.run_name: {}}


def test_verified_generations_handles_no_generations_file():
    assert verified_generations(None, ResultsPaths()) == {}
