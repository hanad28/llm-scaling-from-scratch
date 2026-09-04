"""Positional encoding ablation: learned embeddings against RoPE at one model size, several seeds.

    python -m scaling_lm.ablation [--seeds 0 1 2] [--results-dir PATH]

The learned-embedding seed-0 run is shared with the sweep, so it is reused when present.

positional_ablation.json is a manifest (model size, seeds, run names per scheme). The
statistics are computed from the verified result.json of each run by `analyse_ablation`,
both here for the log and again by the report, so no loss is ever copied into a file.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
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
from scaling_lm.runs import RunResult
from scaling_lm.train import add_training_arguments, train_or_load, training_config_from_args
from scaling_lm.validation import non_negative_int, require_unique

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationManifest:
    """Which runs make up the ablation: one run per (scheme, seed), in seed order."""

    model_size: str
    seeds: list[int]
    run_names: dict[str, list[str]]

    def all_run_names(self) -> list[str]:
        return [name for names in self.run_names.values() for name in names]


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
    """Mean loss difference (alternative minus baseline) with a paired t-test over seeds.

    Why paired: for a given seed the two runs are identical in everything except the
    positional scheme. `GPT` initialises the learned position table last, so every
    parameter the schemes share (token embeddings, attention, MLPs, norms) is bitwise
    identical across schemes for the same seed (tests/test_model.py checks this), and the
    training windows are visited in the same order because the shuffle is seeded
    separately. Seed k is therefore a block that holds both nuisance factors fixed, and
    the per-seed difference isolates the scheme; the paired t-test is the test on those
    differences. It does not remove noise the seed does not control (non-deterministic
    GPU kernels, bf16 accumulation), which stays inside the differences and inflates
    their standard deviation honestly rather than being assumed away.

    Welch's unpaired t-test is reported alongside as the value that makes no pairing
    assumption at all; if pairing bought nothing the two p-values will be similar.
    """
    if baseline.seeds != alternative.seeds:
        raise ValueError(
            f"schemes must use the same seeds in the same order for a paired comparison, "
            f"got {baseline.seeds} and {alternative.seeds}"
        )
    alternative_losses = np.array(alternative.validation_losses)
    baseline_losses = np.array(baseline.validation_losses)
    paired_differences = alternative_losses - baseline_losses
    comparison: dict[str, float | str] = {
        "baseline": baseline.positional_scheme,
        "alternative": alternative.positional_scheme,
        "mean_difference": float(paired_differences.mean()),
    }
    if len(paired_differences) > 1:
        comparison["std_difference"] = float(paired_differences.std(ddof=1))
        paired = stats.ttest_rel(alternative_losses, baseline_losses)
        comparison["paired_t_statistic"] = float(paired.statistic)
        comparison["paired_p_value"] = float(paired.pvalue)
        welch = stats.ttest_ind(alternative_losses, baseline_losses, equal_var=False)
        comparison["welch_t_statistic"] = float(welch.statistic)
        comparison["welch_p_value"] = float(welch.pvalue)
    return comparison


@dataclass(frozen=True)
class AblationAnalysis:
    schemes: dict[str, SchemeSummary]
    comparison: dict[str, float | str]


def analyse_ablation(per_scheme: Mapping[str, Sequence[RunResult]]) -> AblationAnalysis:
    """Per-scheme summaries and the baseline-versus-alternative comparison, from results."""
    summaries = {
        scheme: summarise_scheme(scheme, results) for scheme, results in per_scheme.items()
    }
    baseline_scheme, alternative_scheme = ABLATION_POSITIONAL_SCHEMES
    return AblationAnalysis(
        schemes=summaries,
        comparison=compare_schemes(summaries[baseline_scheme], summaries[alternative_scheme]),
    )


def write_ablation_manifest(manifest: AblationManifest, paths: ResultsPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ablation_summary.write_text(json.dumps(asdict(manifest), indent=2))


def read_ablation_manifest(paths: ResultsPaths) -> AblationManifest | None:
    """The ablation manifest, or None when the ablation has not been run."""
    if not paths.ablation_summary.exists():
        return None
    payload = json.loads(paths.ablation_summary.read_text())
    manifest = AblationManifest(**payload)
    require_unique("ablation run_names", manifest.all_run_names())
    return manifest


def run_ablation(
    seeds: Sequence[int],
    training: TrainingConfig,
    paths: ResultsPaths,
    model_size: str = ABLATION_MODEL_SIZE,
) -> AblationAnalysis:
    """Train every (scheme, seed) pair, write the manifest and return the analysis."""
    require_unique("seeds", list(seeds))
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

    manifest = AblationManifest(
        model_size=model_size,
        seeds=list(seeds),
        run_names={
            scheme: [result.run_name for result in results]
            for scheme, results in per_scheme.items()
        },
    )
    write_ablation_manifest(manifest, paths)
    analysis = analyse_ablation(per_scheme)
    logger.info("ablation comparison: %s", analysis.comparison)
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=non_negative_int, default=list(ABLATION_SEEDS))
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
