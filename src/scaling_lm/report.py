"""Turn the raw run outputs into the fitted scaling law, figures and a markdown summary.

    python -m scaling_lm.report [--results-dir PATH] [--title TEXT]

Reads the sweep and ablation manifests (which runs to report), generations.json
(optional) and data/corpus_stats.json, then writes scaling_fit.json, the figures and
summary.md into the results directory. Every number about a run comes from its verified
result.json via `runs.load_run`; the manifests contribute run names only.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scaling_lm.ablation import (
    AblationAnalysis,
    AblationManifest,
    analyse_ablation,
    read_ablation_manifest,
)
from scaling_lm.config import (
    CORPUS_STATS_PATH,
    DEFAULT_POSITIONAL_SCHEME,
    KAPLAN_ALPHA_N,
    MAX_EPOCHS,
    MIN_TOKENS_PER_PARAMETER,
    ResultsPaths,
)
from scaling_lm.plots import plot_ablation, plot_scaling_law, plot_training_curves
from scaling_lm.runs import RunResult, load_run
from scaling_lm.scaling_fit import PowerLawFit, alpha_std_from_loss_noise, fit_power_law
from scaling_lm.sweep import read_sweep_manifest

logger = logging.getLogger(__name__)

SCALING_FIGURE = "scaling_law.png"
CURVES_FIGURE = "training_curves.png"
ABLATION_FIGURE = "positional_ablation.png"
GENERATION_PREVIEW_CHARS = 400


def load_json(path: Path) -> object | None:
    return json.loads(path.read_text()) if path.exists() else None


def fit_from_results(results: Sequence[RunResult]) -> PowerLawFit:
    counts = [result.parameters["non_embedding"] for result in results]
    losses = [result.final_validation_loss for result in results]
    return fit_power_law(counts, losses)


def architecture_of(result: RunResult) -> Mapping[str, object]:
    architecture = result.identity.specification["architecture"]
    if not isinstance(architecture, dict):
        raise ValueError(f"{result.run_name}: result has no architecture section")
    return architecture


def format_int(value: object) -> str:
    return f"{int(value):,}"


def format_millions(value: object) -> str:
    return f"{int(value) / 1e6:.2f}M"


def corpus_section(stats: Mapping[str, object] | None) -> list[str]:
    if stats is None:
        return ["Corpus statistics not found (run `python -m scaling_lm.tokenizer`)."]
    documents = stats["documents"]
    tokens = stats["tokens"]
    lines = [
        f"Source: `{stats['dataset_repo']}` at revision `{stats['dataset_revision']}`, "
        f"vocabulary size {format_int(stats['vocab_size'])}.",
        "",
        "| Split | Documents | Tokens |",
        "|---|---|---|",
    ]
    for split_name in documents:
        lines.append(
            f"| {split_name} | {format_int(documents[split_name])} | "
            f"{format_int(tokens[split_name])} |"
        )
    return lines


def tokens_per_parameter(result: RunResult) -> float:
    return result.tokens_seen / result.parameters["non_embedding"]


def sweep_section(results: Sequence[RunResult]) -> list[str]:
    lines = [
        "| Size | Layers | d_model | Heads | Non-embedding params | Total params | Epochs | "
        "Steps | Tokens seen | Tokens/param | Peak LR | Validation loss | Test loss | "
        "Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        architecture = architecture_of(result)
        lines.append(
            f"| {result.model_size} | {architecture['n_layer']} | {architecture['d_model']} | "
            f"{architecture['n_head']} | {format_millions(result.parameters['non_embedding'])} | "
            f"{format_millions(result.parameters['total'])} | {result.epochs} | "
            f"{format_int(result.total_steps)} | {format_millions(result.tokens_seen)} | "
            f"{tokens_per_parameter(result):.1f} | {result.peak_learning_rate:.2e} | "
            f"{result.final_validation_loss:.4f} | {result.final_test_loss:.4f} | "
            f"{result.wall_time_seconds / 60:.1f} min |"
        )
    lines.extend(["", *data_constraint_note(results)])
    return lines


def data_constraint_note(results: Sequence[RunResult]) -> list[str]:
    """Say which sizes repeated the corpus and which still fall short of the target."""
    repeated = [result.model_size for result in results if result.epochs > 1]
    short = [
        result.model_size
        for result in results
        if tokens_per_parameter(result) < MIN_TOKENS_PER_PARAMETER
    ]
    lines = [
        f"Sizes below {MIN_TOKENS_PER_PARAMETER:.0f} tokens per non-embedding parameter after "
        f"one pass repeat the corpus, up to {MAX_EPOCHS} epochs (Muennighoff et al., 2023). "
        + (f"Repeated here: {', '.join(repeated)}." if repeated else "No size needed to.")
    ]
    if short:
        lines.append(
            f"Still below the target at the cap: {', '.join(short)}. Those points are trained "
            "in a more data-constrained regime than the rest, and the fit treats them the same."
        )
    return lines


def fit_section(fit: PowerLawFit) -> list[str]:
    verdict = "inside" if fit.kaplan_within_ci else "outside"
    return [
        f"Fitted `loss = {fit.coefficient:.3f} * N^-{fit.alpha:.4f}` over {fit.n_points} sizes "
        f"(R^2 = {fit.r_squared:.4f} in log-log space).",
        "",
        f"Fit uncertainty across the {fit.n_points} sweep points, taking each point's loss as "
        "measured:",
        "",
        f"- alpha = {fit.alpha:.4f}, regression standard error {fit.alpha_standard_error:.4f}",
        f"- {fit.confidence_level:.0%} t-interval: {fit.alpha_ci_low:.4f} to "
        f"{fit.alpha_ci_high:.4f}",
        f"- {fit.confidence_level:.0%} bootstrap interval: {fit.alpha_bootstrap_ci_low:.4f} to "
        f"{fit.alpha_bootstrap_ci_high:.4f}",
        f"- Kaplan et al. (2020) alpha_N = {fit.kaplan_alpha} lies {verdict} the t-interval",
    ]


AblationReport = tuple[AblationManifest, AblationAnalysis]


@dataclass(frozen=True)
class SeedSpread:
    """Seed-to-seed spread of validation loss at one size, for the sweep's positional scheme."""

    model_size: str
    seed_count: int
    loss_std: float


def seed_spread_of_sweep_scheme(ablation: AblationReport | None) -> SeedSpread | None:
    """The ablation's spread for the sweep's scheme, or None if it was not measured."""
    if ablation is None:
        return None
    manifest, analysis = ablation
    summary = analysis.schemes.get(DEFAULT_POSITIONAL_SCHEME)
    if summary is None or len(summary.seeds) < 2:
        return None
    return SeedSpread(manifest.model_size, len(summary.seeds), summary.std_validation_loss)


def seed_variance_section(
    results: Sequence[RunResult], fit: PowerLawFit, ablation: AblationReport | None
) -> list[str]:
    """The second source of uncertainty: what the sweep's one seed per size cannot show."""
    lines = [
        "Run-to-run training variance at a single size is a separate source of uncertainty. "
        "The sweep trains one seed per size, so each point is a single draw and the intervals "
        "above do not include how far that draw might sit from the size's average.",
        "",
    ]
    spread = seed_spread_of_sweep_scheme(ablation)
    if spread is None:
        lines.append(
            "The ablation has not been run with at least two seeds for the sweep's positional "
            "scheme, so there is no measurement of this spread."
        )
        return lines
    counts = [result.parameters["non_embedding"] for result in results]
    losses = [result.final_validation_loss for result in results]
    alpha_std = alpha_std_from_loss_noise(counts, losses, spread.loss_std)
    lines.extend(
        [
            f"Rough scale only: the ablation's {spread.seed_count} `{DEFAULT_POSITIONAL_SCHEME}` "
            f"runs at the {spread.model_size} size have a validation-loss standard deviation of "
            f"{spread.loss_std:.4f} nats. If every sweep point had that spread, independently, "
            f"the exponent would carry a standard deviation of about {alpha_std:.4f} from seed "
            f"noise alone, against a regression standard error of {fit.alpha_standard_error:.4f}.",
            "",
            f"This assumes the spread at {spread.model_size} applies at every size, and a "
            f"standard deviation from {spread.seed_count} seeds is itself imprecise. It is not "
            "added to the fit interval, because the two are not measured on the same footing.",
        ]
    )
    return lines


def ablation_section(ablation: AblationReport | None) -> list[str]:
    if ablation is None:
        return ["Ablation not run (run `python -m scaling_lm.ablation`)."]
    manifest, analysis = ablation
    lines = [
        f"Model size: {manifest.model_size}.",
        "",
        "| Scheme | Seeds | Validation losses | Mean | Std (ddof=1) | Mean test loss |",
        "|---|---|---|---|---|---|",
    ]
    for scheme, summary in analysis.schemes.items():
        losses = ", ".join(f"{loss:.4f}" for loss in summary.validation_losses)
        lines.append(
            f"| {scheme} | {len(summary.seeds)} | {losses} | "
            f"{summary.mean_validation_loss:.4f} | {summary.std_validation_loss:.4f} | "
            f"{summary.mean_test_loss:.4f} |"
        )
    comparison = analysis.comparison
    lines.extend(
        [
            "",
            f"Mean difference ({comparison['alternative']} minus {comparison['baseline']}): "
            f"{comparison['mean_difference']:+.4f} nats per token.",
        ]
    )
    if "paired_p_value" in comparison:
        lines.extend(
            [
                "",
                "Paired t-test on the per-seed differences: "
                f"t = {comparison['paired_t_statistic']:.2f}, "
                f"p = {comparison['paired_p_value']:.3f}, "
                f"std of the differences = {comparison['std_difference']:.4f}. "
                "The pairing is by seed: for one seed the two schemes start from bitwise "
                "identical shared weights (the learned position table is initialised after "
                "everything else) and visit the training windows in the same order, so each "
                "difference isolates the scheme. Kernel-level non-determinism on the GPU is "
                "not controlled by the seed and remains inside the differences.",
                "Welch's unpaired t-test, which assumes no pairing: "
                f"t = {comparison['welch_t_statistic']:.2f}, "
                f"p = {comparison['welch_p_value']:.3f}.",
            ]
        )
    return lines


def generation_section(generations: Mapping[str, Mapping[str, str]] | None) -> list[str]:
    if generations is None:
        return ["No samples (run `python -m scaling_lm.generate`)."]
    lines = []
    for run_name, samples in generations.items():
        lines.append(f"### {run_name}")
        lines.append("")
        for prompt, continuation in samples.items():
            preview = continuation[:GENERATION_PREVIEW_CHARS].replace("\n", " ")
            lines.append(f"- **{prompt}** ... {preview}")
        lines.append("")
    return lines


def write_figures(
    results: Sequence[RunResult],
    fit: PowerLawFit,
    ablation: AblationReport | None,
    paths: ResultsPaths,
) -> None:
    plot_scaling_law(
        parameter_counts=[result.parameters["non_embedding"] for result in results],
        losses=[result.final_validation_loss for result in results],
        labels=[result.model_size for result in results],
        fit=fit,
        output_path=paths.figures / SCALING_FIGURE,
        kaplan_alpha=KAPLAN_ALPHA_N,
    )
    plot_training_curves(results, paths.figures / CURVES_FIGURE)
    if ablation is not None:
        manifest, analysis = ablation
        scheme_losses = {
            scheme: summary.validation_losses for scheme, summary in analysis.schemes.items()
        }
        plot_ablation(scheme_losses, manifest.model_size, paths.figures / ABLATION_FIGURE)


def load_ablation(paths: ResultsPaths) -> AblationReport | None:
    """The ablation manifest with its statistics recomputed from the verified results."""
    manifest = read_ablation_manifest(paths)
    if manifest is None:
        return None
    per_scheme = {
        scheme: [load_run(run_name, paths) for run_name in run_names]
        for scheme, run_names in manifest.run_names.items()
    }
    return manifest, analyse_ablation(per_scheme)


def build_report(paths: ResultsPaths, title: str) -> PowerLawFit:
    """Fit the power law, draw the figures and write summary.md. Returns the fit."""
    results = [load_run(run_name, paths) for run_name in read_sweep_manifest(paths)]
    ablation = load_ablation(paths)
    generations = load_json(paths.generations)
    corpus_stats = load_json(CORPUS_STATS_PATH)

    fit = fit_from_results(results)
    paths.scaling_fit.write_text(json.dumps(fit.to_dict(), indent=2))
    write_figures(results, fit, ablation, paths)

    sections = [
        f"# {title}",
        "",
        "## Corpus",
        "",
        *corpus_section(corpus_stats),
        "",
        "## Scaling sweep",
        "",
        *sweep_section(results),
        "",
        f"![Scaling law]({paths.figures.name}/{SCALING_FIGURE})",
        "",
        f"![Training curves]({paths.figures.name}/{CURVES_FIGURE})",
        "",
        "## Power-law fit",
        "",
        *fit_section(fit),
        "",
        "### Run-to-run variance",
        "",
        *seed_variance_section(results, fit, ablation),
        "",
        "## Positional encoding ablation",
        "",
        *ablation_section(ablation),
        "",
    ]
    if ablation is not None:
        sections.extend([f"![Ablation]({paths.figures.name}/{ABLATION_FIGURE})", ""])
    sections.extend(
        ["## Sample generations (qualitative only)", "", *generation_section(generations)]
    )
    paths.summary_markdown.write_text("\n".join(sections) + "\n")
    logger.info("wrote %s", paths.summary_markdown)
    return fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ResultsPaths().root)
    parser.add_argument("--title", default="Scaling sweep results")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    build_report(ResultsPaths(args.results_dir), args.title)


if __name__ == "__main__":
    main()
