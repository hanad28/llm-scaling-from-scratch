"""Positional encoding ablation: learned embeddings against RoPE at one model size, several seeds.

    python -m scaling_lm.ablation [--seeds 0 1 2] [--results-dir PATH]

The learned-embedding seed-0 run is shared with the sweep, so it is reused when present.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from scaling_lm.config import (
    ABLATION_MODEL_SIZE,
    ABLATION_POSITIONAL_SCHEMES,
    ABLATION_SEEDS,
    MODEL_SIZES_BY_NAME,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.sweep import summarise_run
from scaling_lm.train import (
    RunResult,
    add_training_arguments,
    train_or_load,
    training_config_from_args,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemeSummary:
    positional_scheme: str
    seeds: list[int]
    validation_losses: list[float]
    mean_validation_loss: float
    std_validation_loss: float
    test_losses: list[float]
    mean_test_loss: float


def summarise_scheme(scheme: str, results: Sequence[RunResult]) -> SchemeSummary:
    validation = np.array([result.final_validation_loss for result in results])
    test = np.array([result.final_test_loss for result in results])
    # Sample standard deviation (ddof=1); with one seed there is no spread to report.
    spread = float(validation.std(ddof=1)) if len(validation) > 1 else float("nan")
    return SchemeSummary(
        positional_scheme=scheme,
        seeds=[result.seed for result in results],
        validation_losses=validation.tolist(),
        mean_validation_loss=float(validation.mean()),
        std_validation_loss=spread,
        test_losses=test.tolist(),
        mean_test_loss=float(test.mean()),
    )


def compare_schemes(baseline: SchemeSummary, alternative: SchemeSummary) -> dict[str, float | str]:
    """Mean loss difference (alternative minus baseline) with a paired t-test over matched seeds.

    The two schemes are trained with the same seeds, so run k of each visits the
    training windows in the same order. That makes the design paired: the test is on
    the per-seed differences, not on two independent samples.
    """
    if baseline.seeds != alternative.seeds:
        raise ValueError(
            f"schemes must use the same seeds in the same order for a paired comparison, "
            f"got {baseline.seeds} and {alternative.seeds}"
        )
    paired_differences = np.array(alternative.validation_losses) - np.array(
        baseline.validation_losses
    )
    comparison: dict[str, float | str] = {
        "baseline": baseline.positional_scheme,
        "alternative": alternative.positional_scheme,
        "mean_difference": float(paired_differences.mean()),
    }
    if len(paired_differences) > 1:
        comparison["std_difference"] = float(paired_differences.std(ddof=1))
        test = stats.ttest_rel(alternative.validation_losses, baseline.validation_losses)
        comparison["paired_t_statistic"] = float(test.statistic)
        comparison["paired_p_value"] = float(test.pvalue)
    return comparison


def run_ablation(
    seeds: Sequence[int],
    training: TrainingConfig,
    paths: ResultsPaths,
    model_size: str = ABLATION_MODEL_SIZE,
) -> dict[str, object]:
    """Train every (scheme, seed) pair and write results/positional_ablation.json."""
    per_scheme: dict[str, list[RunResult]] = {}
    for scheme in ABLATION_POSITIONAL_SCHEMES:
        for seed in seeds:
            run_config = RunConfig(
                model_size=model_size,
                positional_scheme=scheme,
                seed=seed,
                training=training,
            )
            per_scheme.setdefault(scheme, []).append(train_or_load(run_config, paths))

    summaries = {
        scheme: summarise_scheme(scheme, results) for scheme, results in per_scheme.items()
    }
    baseline_scheme, alternative_scheme = ABLATION_POSITIONAL_SCHEMES
    payload: dict[str, object] = {
        "model_size": model_size,
        "schemes": {scheme: asdict(summary) for scheme, summary in summaries.items()},
        "comparison": compare_schemes(summaries[baseline_scheme], summaries[alternative_scheme]),
        "runs": [summarise_run(result) for results in per_scheme.values() for result in results],
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ablation_summary.write_text(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ABLATION_SEEDS))
    parser.add_argument(
        "--model-size",
        default=ABLATION_MODEL_SIZE,
        choices=sorted(MODEL_SIZES_BY_NAME),
        help="override the ablation size (smoke tests only; reported results use the default)",
    )
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_ablation(
        args.seeds,
        training_config_from_args(args),
        ResultsPaths(args.results_dir),
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
