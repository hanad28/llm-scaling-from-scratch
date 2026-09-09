"""Matplotlib figures for the scaling sweep and the positional encoding ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scaling_lm.runs import (  # noqa: E402
    RunResult,
    describe_passes,
    format_epochs,
    planned_epochs_of,
    stopped_early,
)
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
            f"fit: alpha = {fit.alpha:.3f} ({fit.confidence_level:.0%} fit interval "
            f"{fit.alpha_ci_low:.3f} to {fit.alpha_ci_high:.3f}; one seed per size)"
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
    axis.set_title("Validation loss against model size, fixed corpus")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    save(figure, output_path)


def training_curve_label(result: RunResult) -> str:
    """Legend entry: size, parameter count and the passes made (as a fraction of the plan if
    max_steps cut the run short)."""
    non_embedding_millions = result.parameters["non_embedding"] / 1e6
    return f"{result.model_size} ({non_embedding_millions:.1f}M, {describe_passes(result)})"


def epoch_range(epoch_counts: Sequence[float]) -> str:
    """`3`, or `1 to 4` when the counts differ."""
    low, high = min(epoch_counts), max(epoch_counts)
    return format_epochs(low) if low == high else f"{format_epochs(low)} to {format_epochs(high)}"


def training_curves_title(results: Sequence[RunResult]) -> str:
    """State the range of passes actually made, and say so if max_steps stopped any run early."""
    planned = [planned_epochs_of(result) for result in results]
    if any(stopped_early(result) for result in results):
        completed = [result.epochs_completed for result in results]
        return (
            f"Validation loss during training, cut short by max_steps\n"
            f"({epoch_range(completed)} of {epoch_range(planned)} planned passes)"
        )
    if set(planned) == {1}:
        return "Validation loss during the single training pass"
    return f"Validation loss during training ({epoch_range(planned)} passes)"


def plot_training_curves(results: Sequence[RunResult], output_path: Path) -> None:
    """Periodic validation loss against tokens seen, one line per model size.

    Sizes train for different numbers of passes over the corpus, so the legend carries each
    model's pass count and the title the range across the figure.
    """
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for result in results:
        tokens = [point.tokens_seen for point in result.history]
        losses = [point.validation_loss for point in result.history]
        axis.plot(tokens, losses, label=training_curve_label(result))
    axis.set_xlabel("training tokens seen")
    axis.set_ylabel("validation loss (nats per token)")
    axis.set_title(training_curves_title(results))
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
