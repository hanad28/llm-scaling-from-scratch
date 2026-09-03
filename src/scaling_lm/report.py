"""Turn the raw run outputs into the fitted scaling law, figures and a markdown summary.

    python -m scaling_lm.report [--results-dir PATH] [--title TEXT]

Reads scaling_sweep.json, positional_ablation.json (optional), generations.json
(optional) and data/corpus_stats.json, then writes scaling_fit.json, the figures
and summary.md into the results directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from scaling_lm.config import CORPUS_STATS_PATH, KAPLAN_ALPHA_N, ResultsPaths
from scaling_lm.plots import plot_ablation, plot_scaling_law, plot_training_curves
from scaling_lm.runs import RunResult, load_run
from scaling_lm.scaling_fit import PowerLawFit, fit_power_law

logger = logging.getLogger(__name__)

SCALING_FIGURE = "scaling_law.png"
CURVES_FIGURE = "training_curves.png"
ABLATION_FIGURE = "positional_ablation.png"
GENERATION_PREVIEW_CHARS = 400


def load_json(path: Path) -> object | None:
    return json.loads(path.read_text()) if path.exists() else None


def fit_from_sweep(sweep: Sequence[Mapping[str, object]]) -> PowerLawFit:
    counts = np.array([entry["non_embedding_params"] for entry in sweep], dtype=float)
    losses = np.array([entry["final_validation_loss"] for entry in sweep], dtype=float)
    return fit_power_law(counts, losses)


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


def sweep_section(sweep: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Size | Layers | d_model | Heads | Non-embedding params | Total params | Steps | "
        "Tokens seen | Peak LR | Validation loss | Test loss | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in sweep:
        lines.append(
            f"| {entry['model_size']} | {entry['n_layer']} | {entry['d_model']} | "
            f"{entry['n_head']} | {format_millions(entry['non_embedding_params'])} | "
            f"{format_millions(entry['total_params'])} | {format_int(entry['total_steps'])} | "
            f"{format_millions(entry['tokens_seen'])} | {entry['peak_learning_rate']:.2e} | "
            f"{entry['final_validation_loss']:.4f} | {entry['final_test_loss']:.4f} | "
            f"{entry['wall_time_seconds'] / 60:.1f} min |"
        )
    return lines


def fit_section(fit: PowerLawFit) -> list[str]:
    verdict = "inside" if fit.kaplan_within_ci else "outside"
    return [
        f"Fitted `loss = {fit.coefficient:.3f} * N^-{fit.alpha:.4f}` over {fit.n_points} sizes "
        f"(R^2 = {fit.r_squared:.4f} in log-log space).",
        "",
        f"- alpha = {fit.alpha:.4f}, standard error {fit.alpha_standard_error:.4f}",
        f"- {fit.confidence_level:.0%} t-interval: {fit.alpha_ci_low:.4f} to "
        f"{fit.alpha_ci_high:.4f}",
        f"- {fit.confidence_level:.0%} bootstrap interval: {fit.alpha_bootstrap_ci_low:.4f} to "
        f"{fit.alpha_bootstrap_ci_high:.4f}",
        f"- Kaplan et al. (2020) alpha_N = {fit.kaplan_alpha} lies {verdict} the t-interval",
    ]


def ablation_section(ablation: Mapping[str, object] | None) -> list[str]:
    if ablation is None:
        return ["Ablation not run (run `python -m scaling_lm.ablation`)."]
    schemes = ablation["schemes"]
    lines = [
        f"Model size: {ablation['model_size']}.",
        "",
        "| Scheme | Seeds | Validation losses | Mean | Std (ddof=1) | Mean test loss |",
        "|---|---|---|---|---|---|",
    ]
    for scheme, summary in schemes.items():
        losses = ", ".join(f"{loss:.4f}" for loss in summary["validation_losses"])
        lines.append(
            f"| {scheme} | {len(summary['seeds'])} | {losses} | "
            f"{summary['mean_validation_loss']:.4f} | {summary['std_validation_loss']:.4f} | "
            f"{summary['mean_test_loss']:.4f} |"
        )
    comparison = ablation["comparison"]
    lines.extend(
        [
            "",
            f"Mean difference ({comparison['alternative']} minus {comparison['baseline']}): "
            f"{comparison['mean_difference']:+.4f} nats per token.",
        ]
    )
    if "paired_p_value" in comparison:
        lines.append(
            f"Paired t-test on the per-seed differences (runs are matched by seed, so each "
            f"pair saw the training windows in the same order): "
            f"t = {comparison['paired_t_statistic']:.2f}, "
            f"p = {comparison['paired_p_value']:.3f}, "
            f"std of the differences = {comparison['std_difference']:.4f}."
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
    sweep: Sequence[Mapping[str, object]],
    results: Sequence[RunResult],
    fit: PowerLawFit,
    ablation: Mapping[str, object] | None,
    paths: ResultsPaths,
) -> None:
    plot_scaling_law(
        parameter_counts=[int(entry["non_embedding_params"]) for entry in sweep],
        losses=[float(entry["final_validation_loss"]) for entry in sweep],
        labels=[str(entry["model_size"]) for entry in sweep],
        fit=fit,
        output_path=paths.figures / SCALING_FIGURE,
        kaplan_alpha=KAPLAN_ALPHA_N,
    )
    plot_training_curves(results, paths.figures / CURVES_FIGURE)
    if ablation is not None:
        scheme_losses = {
            scheme: summary["validation_losses"] for scheme, summary in ablation["schemes"].items()
        }
        plot_ablation(scheme_losses, str(ablation["model_size"]), paths.figures / ABLATION_FIGURE)


def build_report(paths: ResultsPaths, title: str) -> PowerLawFit:
    """Fit the power law, draw the figures and write summary.md. Returns the fit."""
    sweep = load_json(paths.sweep_summary)
    if sweep is None:
        raise FileNotFoundError(f"{paths.sweep_summary} not found; run the sweep first")
    if not sweep:
        raise ValueError(f"{paths.sweep_summary} lists no runs")
    results = [load_run(str(entry["run_name"]), paths) for entry in sweep]
    ablation = load_json(paths.ablation_summary)
    generations = load_json(paths.generations)
    corpus_stats = load_json(CORPUS_STATS_PATH)

    fit = fit_from_sweep(sweep)
    paths.scaling_fit.write_text(json.dumps(fit.to_dict(), indent=2))
    write_figures(sweep, results, fit, ablation, paths)

    sections = [
        f"# {title}",
        "",
        "## Corpus",
        "",
        *corpus_section(corpus_stats),
        "",
        "## Scaling sweep",
        "",
        *sweep_section(sweep),
        "",
        f"![Scaling law]({paths.figures.name}/{SCALING_FIGURE})",
        "",
        f"![Training curves]({paths.figures.name}/{CURVES_FIGURE})",
        "",
        "## Power-law fit",
        "",
        *fit_section(fit),
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
