"""Fit loss = a * N^-alpha to (parameter count, validation loss) pairs and quantify the uncertainty.

Taking logs turns the power law into a straight line, ln L = ln a - alpha ln N,
so the exponent is the negative slope of an ordinary least-squares fit in log-log
space. Two uncertainty estimates are reported: the regression standard error
turned into a t-interval, and a non-parametric bootstrap over the data points.

Both describe how well the sweep points, taken as measured, pin down the slope. The
sweep trains one seed per size, so neither measures how much a single point would move
if its run were repeated with another seed. `alpha_std_from_loss_noise` propagates an
externally measured per-run loss spread (the ablation's seed spread) through the same
regression to give a rough, separately labelled scale for that second source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from scaling_lm.config import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL, KAPLAN_ALPHA_N
from scaling_lm.validation import require_non_negative, require_positive, require_unit_interval

MIN_POINTS_FOR_FIT = 3


@dataclass(frozen=True)
class PowerLawFit:
    """Result of fitting L(N) = coefficient * N ** -alpha."""

    alpha: float
    coefficient: float
    alpha_standard_error: float
    alpha_ci_low: float
    alpha_ci_high: float
    alpha_bootstrap_ci_low: float
    alpha_bootstrap_ci_high: float
    r_squared: float
    n_points: int
    confidence_level: float
    kaplan_alpha: float

    @property
    def kaplan_within_ci(self) -> bool:
        return bool(self.alpha_ci_low <= self.kaplan_alpha <= self.alpha_ci_high)

    def predict(self, parameter_counts: np.ndarray) -> np.ndarray:
        return self.coefficient * np.asarray(parameter_counts, dtype=float) ** (-self.alpha)

    def to_dict(self) -> dict[str, float | int | bool]:
        payload: dict[str, float | int | bool] = dict(asdict(self))
        payload["kaplan_within_ci"] = self.kaplan_within_ci
        return payload


def validate_inputs(
    parameter_counts: np.ndarray,
    losses: np.ndarray,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> None:
    """Reject anything that would make the log-log regression or the bootstrap meaningless."""
    if parameter_counts.ndim != 1 or parameter_counts.shape != losses.shape:
        raise ValueError("parameter_counts and losses must be 1-D arrays of the same length")
    if len(parameter_counts) < MIN_POINTS_FOR_FIT:
        raise ValueError(f"need at least {MIN_POINTS_FOR_FIT} points to fit a power law")
    if not np.all(np.isfinite(losses)):
        raise ValueError(
            f"losses contain NaN or infinite values (a diverged run?): {losses.tolist()}"
        )
    if not np.all(np.isfinite(parameter_counts)):
        raise ValueError(f"parameter counts must be finite: {parameter_counts.tolist()}")
    if np.any(parameter_counts <= 0) or np.any(losses <= 0):
        raise ValueError("parameter counts and losses must be positive for a log-log fit")
    if len(np.unique(parameter_counts)) != len(parameter_counts):
        raise ValueError(
            "parameter counts must be distinct, one per model size; got "
            f"{parameter_counts.tolist()}"
        )
    require_positive("bootstrap_resamples", bootstrap_resamples)
    require_unit_interval("confidence_level", confidence_level)
    require_non_negative("seed", seed)


def alpha_std_from_loss_noise(
    parameter_counts: np.ndarray, losses: np.ndarray, loss_noise_std: float
) -> float:
    """Standard deviation of alpha if every point's loss had independent noise of this std.

    The regression is on ln(loss), where a loss perturbation of `loss_noise_std` becomes
    about `loss_noise_std / loss` at each point. The OLS slope is a fixed linear
    combination of the ln(loss) values, so its variance follows directly. This treats the
    supplied spread as the same at every size, which is an assumption, not a measurement.
    """
    parameter_counts = np.asarray(parameter_counts, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if parameter_counts.ndim != 1 or parameter_counts.shape != losses.shape:
        raise ValueError("parameter_counts and losses must be 1-D arrays of the same length")
    if len(parameter_counts) < MIN_POINTS_FOR_FIT:
        raise ValueError(f"need at least {MIN_POINTS_FOR_FIT} points to propagate loss noise")
    if np.any(parameter_counts <= 0) or np.any(losses <= 0):
        raise ValueError("parameter counts and losses must be positive for a log-log fit")
    require_non_negative("loss_noise_std", loss_noise_std)
    log_counts = np.log(parameter_counts)
    centred = log_counts - log_counts.mean()
    sum_of_squares = float(np.sum(centred**2))
    if sum_of_squares == 0.0:
        raise ValueError("parameter counts must not all be equal")
    slope_weights = centred / sum_of_squares
    log_loss_noise = loss_noise_std / losses
    return float(np.sqrt(np.sum((slope_weights * log_loss_noise) ** 2)))


def log_log_slope(parameter_counts: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of ln(loss) against ln(parameters)."""
    slope, intercept = np.polyfit(np.log(parameter_counts), np.log(losses), deg=1)
    return float(slope), float(intercept)


def bootstrap_alpha(
    parameter_counts: np.ndarray,
    losses: np.ndarray,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Exponents from refitting on resampled points; draws with under 2 distinct N are skipped."""
    generator = np.random.default_rng(seed)
    n_points = len(parameter_counts)
    alphas = []
    for _ in range(resamples):
        chosen = generator.integers(0, n_points, size=n_points)
        if len(np.unique(parameter_counts[chosen])) < 2:
            continue
        slope, _ = log_log_slope(parameter_counts[chosen], losses[chosen])
        alphas.append(-slope)
    if not alphas:
        raise ValueError(
            f"none of the {resamples} bootstrap resamples had two distinct parameter counts; "
            "increase bootstrap_resamples"
        )
    return np.asarray(alphas)


def fit_power_law(
    parameter_counts: np.ndarray,
    losses: np.ndarray,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = 0,
) -> PowerLawFit:
    """Least-squares power-law fit in log-log space with t-interval and bootstrap uncertainty."""
    parameter_counts = np.asarray(parameter_counts, dtype=float)
    losses = np.asarray(losses, dtype=float)
    validate_inputs(parameter_counts, losses, bootstrap_resamples, confidence_level, seed)

    regression = stats.linregress(np.log(parameter_counts), np.log(losses))
    alpha = -float(regression.slope)
    degrees_of_freedom = len(parameter_counts) - 2
    t_critical = float(stats.t.ppf(0.5 + confidence_level / 2, degrees_of_freedom))
    half_width = t_critical * float(regression.stderr)

    bootstrap = bootstrap_alpha(parameter_counts, losses, bootstrap_resamples, seed)
    tail = (1.0 - confidence_level) / 2
    bootstrap_low, bootstrap_high = np.quantile(bootstrap, [tail, 1.0 - tail])

    return PowerLawFit(
        alpha=alpha,
        coefficient=float(np.exp(regression.intercept)),
        alpha_standard_error=float(regression.stderr),
        alpha_ci_low=alpha - half_width,
        alpha_ci_high=alpha + half_width,
        alpha_bootstrap_ci_low=float(bootstrap_low),
        alpha_bootstrap_ci_high=float(bootstrap_high),
        r_squared=float(regression.rvalue**2),
        n_points=len(parameter_counts),
        confidence_level=confidence_level,
        kaplan_alpha=KAPLAN_ALPHA_N,
    )
