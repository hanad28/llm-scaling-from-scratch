"""Train every model size on the same training tokens and collect the final losses.

    python -m scaling_lm.sweep [--sizes tiny small ...] [--results-dir PATH]

Finished runs (those with a result.json) are skipped, so the sweep can be resumed.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict

from scaling_lm.config import (
    DEFAULT_POSITIONAL_SCHEME,
    MODEL_SIZES,
    MODEL_SIZES_BY_NAME,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.train import (
    RunResult,
    add_training_arguments,
    load_result,
    result_exists,
    train_run,
    training_config_from_args,
)

logger = logging.getLogger(__name__)

SWEEP_SEED = 0


def train_or_load(run_config: RunConfig, paths: ResultsPaths) -> RunResult:
    """Reuse a finished run with the same name, refusing one trained under a different config."""
    if not result_exists(run_config.run_name, paths):
        return train_run(run_config, paths)
    result = load_result(run_config.run_name, paths)
    requested = asdict(run_config.training)
    if result.training != requested:
        raise RuntimeError(
            f"{run_config.run_name} exists but was trained with {result.training}, "
            f"not the requested {requested}; delete it or use another --results-dir"
        )
    logger.info("%s already finished, loading result", run_config.run_name)
    return result


def summarise_run(result: RunResult) -> dict[str, object]:
    return {
        "run_name": result.run_name,
        "model_size": result.model_size,
        "positional_scheme": result.positional_scheme,
        "seed": result.seed,
        "n_layer": result.architecture["n_layer"],
        "d_model": result.architecture["d_model"],
        "n_head": result.architecture["n_head"],
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
