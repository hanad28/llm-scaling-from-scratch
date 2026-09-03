"""Run identity and the single place that decides whether a saved result may be reused.

A run is identified by a fingerprint over everything that determines its result: the
corpus (token files), the resolved architecture (GPTConfig plus the initialisation
constants) and the resolved schedule (TrainingConfig, seed and the learning-rate
constants). The fingerprint is a hash of that whole specification, so adding a new
constant to `run_specification` is the only step needed to make it part of the check.

Every reader of a result.json goes through `verify_result`, via either
`resolve_run` (a requested RunConfig, used before training) or `load_run` (a run
name, used by report and generate). There is no other loading path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields

from scaling_lm.config import (
    FINAL_LR_FRACTION,
    KAPLAN_LR_INTERCEPT,
    KAPLAN_LR_SLOPE,
    MODEL_SIZES_BY_NAME,
    WARMUP_FRACTION,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.model import INIT_STD, LAYER_NORM_EPS, GPTConfig
from scaling_lm.tokenizer import corpus_fingerprint

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "model.pt"
RESULT_FILENAME = "result.json"
FINGERPRINT_PREVIEW_CHARS = 12


@dataclass
class EvalPoint:
    step: int
    tokens_seen: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass
class RunResult:
    """Everything recorded about a finished run, written to results/runs/<run_name>/result.json.

    `specification` is the output of `run_specification` at training time and
    `run_fingerprint` its hash; together they are the only record of what was trained.
    """

    run_name: str
    model_size: str
    positional_scheme: str
    seed: int
    parameters: dict[str, int]
    specification: dict[str, object]
    run_fingerprint: str
    total_steps: int
    tokens_seen: int
    peak_learning_rate: float
    final_validation_loss: float
    final_test_loss: float
    wall_time_seconds: float
    device: str
    history: list[EvalPoint] = field(default_factory=list)


class StaleRunError(RuntimeError):
    """A result.json exists for the run name but was produced by a different run."""


def run_specification(run_config: RunConfig) -> dict[str, object]:
    """Everything that determines a run's result, resolved to plain JSON-friendly values."""
    size = MODEL_SIZES_BY_NAME[run_config.model_size]
    model = GPTConfig.from_model_size(size, run_config.positional_scheme)
    return {
        "corpus": corpus_fingerprint(),
        "architecture": {
            **asdict(model),
            "init_std": INIT_STD,
            "layer_norm_eps": LAYER_NORM_EPS,
        },
        "schedule": {
            **asdict(run_config.training),
            "seed": run_config.seed,
            "warmup_fraction": WARMUP_FRACTION,
            "final_lr_fraction": FINAL_LR_FRACTION,
            "kaplan_lr_intercept": KAPLAN_LR_INTERCEPT,
            "kaplan_lr_slope": KAPLAN_LR_SLOPE,
        },
    }


def fingerprint_specification(specification: dict[str, object]) -> str:
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_fingerprint(run_config: RunConfig) -> str:
    return fingerprint_specification(run_specification(run_config))


def run_config_of(result: RunResult) -> RunConfig:
    """Rebuild the RunConfig a saved result claims to have been trained with."""
    schedule = result.specification["schedule"]
    if not isinstance(schedule, dict):
        raise StaleRunError(f"{result.run_name}: result.json has no schedule section")
    training_fields = {entry.name for entry in fields(TrainingConfig)}
    training = {name: value for name, value in schedule.items() if name in training_fields}
    return RunConfig(
        model_size=result.model_size,
        positional_scheme=result.positional_scheme,
        seed=result.seed,
        training=TrainingConfig(**training),
    )


def result_path(run_name: str, paths: ResultsPaths) -> str:
    return str(paths.run_directory(run_name) / RESULT_FILENAME)


def read_result(run_name: str, paths: ResultsPaths) -> RunResult:
    """Parse a result.json without checking it. Only `verify_result` callers should use this."""
    payload = json.loads((paths.run_directory(run_name) / RESULT_FILENAME).read_text())
    payload["history"] = [EvalPoint(**point) for point in payload["history"]]
    return RunResult(**payload)


def changed_sections(recorded: dict[str, object], current: dict[str, object]) -> list[str]:
    return sorted(
        key for key in recorded.keys() | current.keys() if recorded.get(key) != current.get(key)
    )


def verify_result(result: RunResult, run_config: RunConfig, paths: ResultsPaths) -> RunResult:
    """The reuse decision: accept `result` only if it is exactly the run `run_config` describes.

    Raises StaleRunError when the run name, or the fingerprint over corpus, architecture
    and schedule, differs from what the current code and data would produce.
    """
    if result.run_name != run_config.run_name:
        raise StaleRunError(
            f"{result_path(run_config.run_name, paths)} records run {result.run_name!r}, "
            f"not {run_config.run_name!r}"
        )
    if fingerprint_specification(result.specification) != result.run_fingerprint:
        raise StaleRunError(
            f"{result_path(run_config.run_name, paths)} records a specification that does not "
            "hash to its own fingerprint; the file was edited after training. "
            "Delete the run directory or use another --results-dir"
        )
    current_specification = run_specification(run_config)
    current_fingerprint = fingerprint_specification(current_specification)
    if result.run_fingerprint != current_fingerprint:
        sections = changed_sections(result.specification, current_specification)
        raise StaleRunError(
            f"{result_path(run_config.run_name, paths)} was produced by a different run: "
            f"fingerprint {result.run_fingerprint[:FINGERPRINT_PREVIEW_CHARS]} recorded, "
            f"{current_fingerprint[:FINGERPRINT_PREVIEW_CHARS]} now; changed: {sections}. "
            "Delete the run directory or use another --results-dir"
        )
    return result


def resolve_run(run_config: RunConfig, paths: ResultsPaths) -> RunResult | None:
    """Return the verified saved result for this run, or None if it has not been run yet."""
    if not (paths.run_directory(run_config.run_name) / RESULT_FILENAME).exists():
        return None
    result = verify_result(read_result(run_config.run_name, paths), run_config, paths)
    logger.info("%s already finished, loading result", run_config.run_name)
    return result


def load_run(run_name: str, paths: ResultsPaths) -> RunResult:
    """Load a finished run by name, checking it against the current code and corpus."""
    result = read_result(run_name, paths)
    return verify_result(result, run_config_of(result), paths)
