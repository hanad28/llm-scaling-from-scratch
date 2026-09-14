"""Train every model size on the same training split and collect the final losses.

    python -m scaling_lm.sweep [--sizes tiny small ...] [--results-dir PATH]

Each size gets the number of passes over the split that `budget.token_budget` assigns it,
so the smaller sizes see the corpus once and the larger ones repeat it. Finished runs
(those with a result.json) are skipped, so the sweep can be resumed; see
`runs.resolve_run` for the fingerprint check that guards the reuse.

scaling_sweep.json is a manifest: the run names in sweep order and nothing else. Losses
and parameter counts live only in each run's result.json and are read through
`runs.load_run`, so a regenerated run can never be reported from a stale copy.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from scaling_lm.budget import token_budget
from scaling_lm.config import (
    DEFAULT_POSITIONAL_SCHEME,
    MODEL_SIZES,
    MODEL_SIZES_BY_NAME,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.dataset import TokenWindows
from scaling_lm.runs import RunResult
from scaling_lm.train import add_training_arguments, train_or_load, training_config_from_args
from scaling_lm.validation import require_unique

logger = logging.getLogger(__name__)

SWEEP_SEED = 0
MANIFEST_KEY = "run_names"


def write_sweep_manifest(results: Sequence[RunResult], paths: ResultsPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    manifest = {MANIFEST_KEY: [result.run_name for result in results]}
    paths.sweep_summary.write_text(json.dumps(manifest, indent=2))


def read_sweep_manifest(paths: ResultsPaths) -> list[str]:
    """Run names in sweep order. Raises if the sweep has not been run or lists no runs."""
    if not paths.sweep_summary.exists():
        raise FileNotFoundError(f"{paths.sweep_summary} not found; run the sweep first")
    manifest = json.loads(paths.sweep_summary.read_text())
    run_names = manifest.get(MANIFEST_KEY) if isinstance(manifest, dict) else None
    if not isinstance(run_names, list) or not run_names:
        raise ValueError(f"{paths.sweep_summary} lists no runs")
    require_unique("run_names", run_names)
    return [str(run_name) for run_name in run_names]


def sweep_run_config(
    size_name: str, training: TrainingConfig, train_window_count: int
) -> RunConfig:
    """The run the sweep trains for one size: default scheme, the sweep seed, planned epochs."""
    budget = token_budget(size_name, train_window_count, training)
    return RunConfig(
        model_size=size_name,
        positional_scheme=DEFAULT_POSITIONAL_SCHEME,
        seed=SWEEP_SEED,
        training=training,
        epochs=budget.epochs,
    )


def run_sweep(
    size_names: Sequence[str],
    training: TrainingConfig,
    paths: ResultsPaths,
    allow_partial: bool = False,
) -> list[RunResult]:
    """Train the requested sizes in order and write results/scaling_sweep.json."""
    require_unique("sizes", list(size_names))
    train_window_count = len(TokenWindows("train"))
    results = []
    for size_name in size_names:
        run_config = sweep_run_config(size_name, training, train_window_count)
        logger.info("%s: %d epoch(s) planned", run_config.run_name, run_config.epochs)
        results.append(train_or_load(run_config, paths, allow_partial))
    write_sweep_manifest(results, paths)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[size.name for size in MODEL_SIZES],
        choices=sorted(MODEL_SIZES_BY_NAME),
    )
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_sweep(
        args.sizes,
        training_config_from_args(args),
        ResultsPaths(args.results_dir),
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
