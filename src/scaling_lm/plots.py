"""Matplotlib figures for the scaling sweep and the positional encoding ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scaling_lm.runs import RunResult  # noqa: E402
from scaling_lm.scaling_fit import PowerLawFit  # noqa: E402

FIGURE_DPI = 150
FIGURE_SIZE = (7.0, 4.5)
FIT_LINE_POINTS = 100


def plot_scaling_law(
    parameter_counts: Sequence[int],
    losses: Sequence[float],
    labels: Sequence[str],
    fit: PowerLawFit,
    output_path: Path,
    kaplan_alpha: float,
) -> None:
    """Log-log validation loss against non-embedding parameters with the fitted power law.

    A dashed reference line with Kaplan et al.'s exponent is drawn through the
    geometric centre of the data so the two slopes can be compared by eye.
    """
    counts = np.asarray(parameter_counts, dtype=float)
    losses_array = np.asarray(losses, dtype=float)
    grid = np.logspace(np.log10(counts.min()), np.log10(counts.max()), FIT_LINE_POINTS)

    centre_log_n = np.log(counts).mean()
    centre_log_loss = np.log(losses_array).mean()
    kaplan_reference = np.exp(centre_log_loss - kaplan_alpha * (np.log(grid) - centre_log_n))

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(
        grid,
        fit.predict(grid),
        color="tab:blue",
        label=(
            f"fit: alpha = {fit.alpha:.3f} "
            f"({fit.confidence_level:.0%} CI {fit.alpha_ci_low:.3f} to {fit.alpha_ci_high:.3f})"
        ),
    )
    axis.plot(
        grid,
        kaplan_reference,
        color="tab:grey",
        linestyle="--",
        label=f"Kaplan et al. (2020) slope, alpha = {kaplan_alpha}",
    )
    axis.scatter(counts, losses_array, color="tab:red", zorder=3, label="validation loss")
    for count, loss, label in zip(counts, losses_array, labels, strict=True):
        axis.annotate(label, (count, loss), textcoords="offset points", xytext=(6, 6), fontsize=8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("non-embedding parameters")
    axis.set_ylabel("validation loss (nats per token)")
    axis.set_title("Validation loss against model size, fixed training tokens")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    save(figure, output_path)


def plot_training_curves(results: Sequence[RunResult], output_path: Path) -> None:
    """Periodic validation loss against tokens seen, one line per model size."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for result in results:
        tokens = [point.tokens_seen for point in result.history]
        losses = [point.validation_loss for point in result.history]
        label = f"{result.model_size} ({result.parameters['non_embedding'] / 1e6:.1f}M)"
        axis.plot(tokens, losses, label=label)
    axis.set_xlabel("training tokens seen")
    axis.set_ylabel("validation loss (nats per token)")
    axis.set_title("Validation loss during the single training pass")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    save(figure, output_path)


def plot_ablation(
    scheme_losses: Mapping[str, Sequence[float]], model_size: str, output_path: Path
) -> None:
    """Per-seed validation losses for each positional scheme, with the mean marked."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for position, losses in enumerate(scheme_losses.values()):
        losses_array = np.asarray(losses, dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(losses_array))
        axis.scatter(position + jitter, losses_array, color="tab:red", zorder=3, label=None)
        axis.hlines(losses_array.mean(), position - 0.2, position + 0.2, color="tab:blue")
    axis.set_xticks(range(len(scheme_losses)), list(scheme_losses))
    axis.set_ylabel("final validation loss (nats per token)")
    axis.set_title(f"Positional encoding ablation ({model_size} model, one point per seed)")
    axis.grid(True, axis="y", alpha=0.3)
    save(figure, output_path)


def save(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
