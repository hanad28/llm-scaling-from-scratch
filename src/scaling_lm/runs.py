"""Run identity and the single place that decides whether a saved result may be reused.

`run_identity` is the only function that says what a run is: a specification covering
everything that determines its result (the corpus token files, the resolved GPTConfig,
the resolved TrainingConfig plus seed and epoch count, the torch version and the device)
and the SHA-256 of that whole specification. The environment section is deliberately
narrow: it separates a CPU smoke run from an A40 run of the same name and a torch upgrade
from the run before it, not every package or OS difference (see README, Limitations).
Training records its output verbatim in result.json; `verify_result` recomputes it and
compares. There is no list of constants to maintain: the model and training code read
every tunable from the two config objects (tests/test_constant_staleness.py enforces
this), so a constant is part of the identity by construction, and nothing outside this
module assembles or hashes a specification (tests/test_single_source.py enforces that).

Every reader of a result.json goes through `verify_result`, via either `resolve_run`
(a requested RunConfig, used before training) or `load_run` (a run name, used by report
and generate). Losses are read from the RunResult those return and from nowhere else.

`resolve_run` answers "is this exactly the run asked for?"; it says nothing about
whether that run finished. `reject_partial_reuse` is the second, independent check
`train_or_load` layers on top: a full (uncapped) request never gets back a partial
result silently, only with an explicit `--allow-partial`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

import torch

from scaling_lm.config import MODEL_SIZES_BY_NAME, ResultsPaths, RunConfig, TrainingConfig
from scaling_lm.model import GPTConfig
from scaling_lm.tokenizer import corpus_fingerprint

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "model.pt"
RESULT_FILENAME = "result.json"
PARTIAL_WRITE_SUFFIX = ".partial"
FINGERPRINT_PREVIEW_CHARS = 12
EPOCHS_KEY = "epochs"


@dataclass
class EvalPoint:
    step: int
    tokens_seen: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class RunIdentity:
    """What a run is: its resolved specification and the fingerprint (hash) of it."""

    specification: dict[str, object]
    fingerprint: str


@dataclass
class RunResult:
    """Everything recorded about a finished run, written to results/runs/<run_name>/result.json.

    `identity` is `run_identity(run_config)` as computed when training started; it is the
    only record of what was trained. `final_validation_loss` and `final_test_loss` here
    are the only place a loss is read from.

    `epochs_completed` is the number of passes over the training split the run actually
    made, `total_steps / steps_per_epoch`. It equals the requested count (kept in the
    identity's schedule, see `planned_epochs_of`) unless `max_steps` stopped the run
    early, in which case it is the fraction reached, so a capped run never reads as a
    finished one.
    """

    run_name: str
    model_size: str
    positional_scheme: str
    seed: int
    parameters: dict[str, int]
    identity: RunIdentity
    epochs_completed: float
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


class PartialRunReuseError(RuntimeError):
    """A saved result is a partial run (max_steps cut it short) but a full run was asked for."""


def hash_specification(specification: dict[str, object]) -> str:
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    """`cpu`, or `cuda:<hardware name>` such as `cuda:NVIDIA A40`."""
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return device.type


def environment_specification() -> dict[str, str]:
    return {"torch_version": torch.__version__, "device": describe_device(select_device())}


def run_identity(run_config: RunConfig) -> RunIdentity:
    """The one assembly of everything that determines a run's result, and its fingerprint.

    The corpus enters as `corpus_fingerprint()`, a hash of the token files kept in
    tokenizer.py because it describes the data, not a run; it is an input here, not a
    second identity. Architecture and schedule are `asdict()` of the resolved configs; the
    schedule also carries the two per-run settings RunConfig holds outside TrainingConfig,
    the seed and the epoch count.
    """
    size = MODEL_SIZES_BY_NAME[run_config.model_size]
    model = GPTConfig.from_model_size(size, run_config.positional_scheme)
    specification: dict[str, object] = {
        "corpus": corpus_fingerprint(),
        "architecture": asdict(model),
        "schedule": {
            **asdict(run_config.training),
            "seed": run_config.seed,
            "epochs": run_config.epochs,
        },
        "environment": environment_specification(),
    }
    return RunIdentity(specification, hash_specification(specification))


def run_config_of(result: RunResult) -> RunConfig:
    """Rebuild the request a saved result answers: its name plus the options its CLI was given.

    Only the CLI options and the per-run epoch count are taken from the saved record. Every
    other TrainingConfig field comes from the current code, so checking the rebuilt request
    against the record catches a changed constant just as `resolve_run` does for a fresh
    request.
    """
    schedule = result.identity.specification["schedule"]
    if not isinstance(schedule, dict):
        raise StaleRunError(f"{result.run_name}: result.json has no schedule section")
    option_names = TrainingConfig.cli_option_names()
    missing = sorted((option_names | {EPOCHS_KEY}) - schedule.keys())
    if missing:
        raise StaleRunError(f"{result.run_name}: result.json schedule lacks {missing}")
    training = {name: schedule[name] for name in option_names}
    return RunConfig(
        model_size=result.model_size,
        positional_scheme=result.positional_scheme,
        seed=result.seed,
        training=TrainingConfig(**training),
        epochs=schedule[EPOCHS_KEY],
    )


def planned_epochs_of(result: RunResult) -> int:
    """The epoch count the run was asked for, read from the identity it was trained under."""
    return run_config_of(result).epochs


def stopped_early(result: RunResult) -> bool:
    return result.epochs_completed < planned_epochs_of(result)


def format_epochs(epochs: float) -> str:
    """`3` for a whole number of passes, otherwise three significant figures (`0.5`, `3.45e-05`)."""
    return f"{epochs:.3g}"


def epochs_summary(result: RunResult) -> str:
    """`3`, or `0.5 of 3` when max_steps stopped the run before its planned passes."""
    planned = planned_epochs_of(result)
    if stopped_early(result):
        return f"{format_epochs(result.epochs_completed)} of {planned}"
    return str(planned)


def describe_passes(result: RunResult) -> str:
    """`epochs_summary` with its noun: `1 pass`, `3 passes`, `0.5 of 3 passes`."""
    noun = "pass" if planned_epochs_of(result) == 1 else "passes"
    return f"{epochs_summary(result)} {noun}"


def result_path(run_name: str, paths: ResultsPaths) -> str:
    return str(paths.run_directory(run_name) / RESULT_FILENAME)


def write_atomically(final_path: Path, write: Callable[[Path], None]) -> None:
    """Run `write` against a temporary sibling of `final_path`, then rename it into place.

    The rename is atomic on POSIX, so `final_path` is either absent (or its previous
    content) or complete; a process killed mid-write leaves only the `.partial` file,
    which the next save overwrites. A run directory is only ever written by one process.
    """
    partial_path = final_path.with_name(final_path.name + PARTIAL_WRITE_SUFFIX)
    try:
        write(partial_path)
        os.replace(partial_path, final_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def required_result_fields() -> set[str]:
    return {
        result_field.name
        for result_field in fields(RunResult)
        if result_field.default is MISSING and result_field.default_factory is MISSING
    }


def read_result(run_name: str, paths: ResultsPaths) -> RunResult:
    """Parse a result.json without checking it. Only `verify_result` callers should use this.

    A record written by an older version of the code, which did not know about a field the
    current RunResult requires, is stale rather than malformed: it is reported like any
    other stale run instead of failing inside the dataclass constructor.
    """
    payload = json.loads((paths.run_directory(run_name) / RESULT_FILENAME).read_text())
    missing = sorted(required_result_fields() - payload.keys())
    if missing:
        raise StaleRunError(
            f"{result_path(run_name, paths)} was written by an older version of the code and "
            f"lacks {missing}. Delete the run directory or use another --results-dir"
        )
    payload["history"] = [EvalPoint(**point) for point in payload["history"]]
    payload["identity"] = RunIdentity(**payload["identity"])
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
    recorded = result.identity
    if hash_specification(recorded.specification) != recorded.fingerprint:
        raise StaleRunError(
            f"{result_path(run_config.run_name, paths)} records a specification that does not "
            "hash to its own fingerprint; the file was edited after training. "
            "Delete the run directory or use another --results-dir"
        )
    current = run_identity(run_config)
    if recorded != current:
        sections = changed_sections(recorded.specification, current.specification)
        raise StaleRunError(
            f"{result_path(run_config.run_name, paths)} was produced by a different run: "
            f"fingerprint {recorded.fingerprint[:FINGERPRINT_PREVIEW_CHARS]} recorded, "
            f"{current.fingerprint[:FINGERPRINT_PREVIEW_CHARS]} now; changed: {sections}. "
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


def reject_partial_reuse(
    result: RunResult, run_config: RunConfig, paths: ResultsPaths, allow_partial: bool
) -> None:
    """Refuse a partial `result` when `run_config` asks for a full (uncapped) run.

    `resolve_run` already checked that `result` is an exact fingerprint match for
    `run_config`; that identity check does not by itself say whether the saved run
    actually finished its planned passes, only that nothing about the request has
    changed. This is a second, independent check, on `epochs_completed` against the
    plan rather than on identity, so a partial result cannot be handed back as if it
    were the finished run a full request asked for, however it came to be on disk (a
    genuinely interrupted run, or a result.json placed there by another process). A
    request that itself caps `max_steps` is not asking for a full run, so it is exempt:
    reusing a matching capped result under the same cap is the intended resumption path.
    """
    if run_config.training.max_steps is not None or allow_partial:
        return
    if stopped_early(result):
        raise PartialRunReuseError(
            f"{result_path(run_config.run_name, paths)} is a partial result "
            f"({epochs_summary(result)}), but a full run was requested. Pass --allow-partial "
            "to reuse it anyway, or delete the run directory to retrain."
        )
