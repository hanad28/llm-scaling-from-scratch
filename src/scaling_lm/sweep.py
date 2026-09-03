"""Train every model size on the same training tokens and collect the final losses.

    python -m scaling_lm.sweep [--sizes tiny small ...] [--results-dir PATH]

Finished runs (those with a result.json) are skipped, so the sweep can be resumed; see
`runs.resolve_run` for the fingerprint check that guards the reuse.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from scaling_lm.config import (
    DEFAULT_POSITIONAL_SCHEME,
    MODEL_SIZES,
    MODEL_SIZES_BY_NAME,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.runs import RunResult
from scaling_lm.train import add_training_arguments, train_or_load, training_config_from_args
from scaling_lm.validation import require_unique

logger = logging.getLogger(__name__)

SWEEP_SEED = 0


def summarise_run(result: RunResult) -> dict[str, object]:
    architecture = result.specification["architecture"]
    if not isinstance(architecture, dict):
        raise ValueError(f"{result.run_name}: result has no architecture section")
    return {
        "run_name": result.run_name,
        "model_size": result.model_size,
        "positional_scheme": result.positional_scheme,
        "seed": result.seed,
        "n_layer": architecture["n_layer"],
        "d_model": architecture["d_model"],
        "n_head": architecture["n_head"],
        "run_fingerprint": result.run_fingerprint,
        "non_embedding_params": result.parameters["non_embedding"],
        "total_params": result.parameters["total"],
        "total_steps": result.total_steps,
        "tokens_seen": result.tokens_seen,
        "peak_learning_rate": result.peak_learning_rate,
        "final_validation_loss": result.final_validation_loss,
        "final_test_loss": result.final_test_loss,
        "wall_time_seconds": result.wall_time_seconds,
        "device": result.device,
    }


def run_sweep(
    size_names: Sequence[str], training: TrainingConfig, paths: ResultsPaths
) -> list[RunResult]:
    """Train the requested sizes in order and write results/scaling_sweep.json."""
    require_unique("sizes", list(size_names))
    results = []
    for size_name in size_names:
        run_config = RunConfig(
            model_size=size_name,
            positional_scheme=DEFAULT_POSITIONAL_SCHEME,
            seed=SWEEP_SEED,
            training=training,
        )
        results.append(train_or_load(run_config, paths))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.sweep_summary.write_text(
        json.dumps([summarise_run(result) for result in results], indent=2)
    )
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
    run_sweep(args.sizes, training_config_from_args(args), ResultsPaths(args.results_dir))


if __name__ == "__main__":
    main()
