import numpy as np
import pytest

from scaling_lm.scaling_fit import alpha_std_from_loss_noise, fit_power_law

PARAMETER_COUNTS = np.array([1e6, 5e6, 1.2e7, 2.5e7, 1e8])


def test_recovers_exact_power_law():
    true_alpha, true_coefficient = 0.076, 12.0
    losses = true_coefficient * PARAMETER_COUNTS**-true_alpha
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=200)
    assert fit.alpha == pytest.approx(true_alpha, abs=1e-9)
    assert fit.coefficient == pytest.approx(true_coefficient, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.alpha_standard_error == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fit.predict(PARAMETER_COUNTS), losses)


def test_noise_widens_interval_and_interval_contains_truth():
    generator = np.random.default_rng(0)
    true_alpha = 0.076
    losses = 10.0 * PARAMETER_COUNTS**-true_alpha * np.exp(generator.normal(0, 0.01, 5))
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=2000)
    assert fit.alpha_standard_error > 0
    assert fit.alpha_ci_low < fit.alpha < fit.alpha_ci_high
    assert fit.alpha_ci_low < true_alpha < fit.alpha_ci_high
    assert fit.kaplan_within_ci
    assert fit.alpha_bootstrap_ci_low <= fit.alpha <= fit.alpha_bootstrap_ci_high


def test_kaplan_flag_false_when_exponent_clearly_different():
    losses = 10.0 * PARAMETER_COUNTS**-0.3
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=200)
    assert not fit.kaplan_within_ci


def test_rejects_too_few_points_and_non_positive_values():
    with pytest.raises(ValueError):
        fit_power_law(np.array([1e6, 1e7]), np.array([3.0, 2.5]))
    with pytest.raises(ValueError):
        fit_power_law(np.array([1e6, 1e7, 1e8]), np.array([3.0, -2.5, 2.0]))


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), -float("inf")])
def test_rejects_non_finite_losses(bad_loss):
    losses = np.array([3.0, bad_loss, 2.0, 1.9, 1.8])
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_power_law(PARAMETER_COUNTS, losses)


@pytest.mark.parametrize("resamples", [0, -5])
def test_rejects_degenerate_bootstrap_resamples(resamples):
    losses = 10.0 * PARAMETER_COUNTS**-0.08
    with pytest.raises(ValueError, match="bootstrap_resamples must be positive"):
        fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=resamples)


def test_rejects_repeated_parameter_counts():
    counts = np.array([1e6, 1e7, 1e7, 1e8])
    losses = np.array([4.0, 3.5, 3.4, 3.0])
    with pytest.raises(ValueError, match="distinct"):
        fit_power_law(counts, losses)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, 1.5, float("nan")])
def test_rejects_confidence_level_outside_unit_interval(confidence_level):
    losses = 10.0 * PARAMETER_COUNTS**-0.08
    with pytest.raises(ValueError, match="confidence_level"):
        fit_power_law(PARAMETER_COUNTS, losses, confidence_level=confidence_level)


def test_alpha_std_from_loss_noise_matches_monte_carlo():
    """The closed form agrees with refitting under independent Gaussian loss noise."""
    losses = 10.0 * PARAMETER_COUNTS**-0.076
    loss_std = 0.01
    predicted = alpha_std_from_loss_noise(PARAMETER_COUNTS, losses, loss_std)
    generator = np.random.default_rng(0)
    alphas = []
    for _ in range(4000):
        noisy = losses + generator.normal(0, loss_std, len(losses))
        slope, _ = np.polyfit(np.log(PARAMETER_COUNTS), np.log(noisy), deg=1)
        alphas.append(-slope)
    assert predicted == pytest.approx(np.std(alphas, ddof=1), rel=0.1)


def test_alpha_std_from_loss_noise_scales_with_the_noise_and_is_zero_without_it():
    losses = 10.0 * PARAMETER_COUNTS**-0.076
    assert alpha_std_from_loss_noise(PARAMETER_COUNTS, losses, 0.0) == 0.0
    single = alpha_std_from_loss_noise(PARAMETER_COUNTS, losses, 0.01)
    assert alpha_std_from_loss_noise(PARAMETER_COUNTS, losses, 0.02) == pytest.approx(2 * single)


def test_alpha_std_from_loss_noise_is_separate_from_the_regression_error():
    """An exact power law has zero regression error but non-zero propagated seed noise."""
    losses = 10.0 * PARAMETER_COUNTS**-0.076
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=100)
    assert fit.alpha_standard_error == pytest.approx(0.0, abs=1e-9)
    assert alpha_std_from_loss_noise(PARAMETER_COUNTS, losses, 0.01) > 0


def test_alpha_std_from_loss_noise_validates_inputs():
    losses = 10.0 * PARAMETER_COUNTS**-0.076
    with pytest.raises(ValueError, match="loss_noise_std"):
        alpha_std_from_loss_noise(PARAMETER_COUNTS, losses, -0.01)
    with pytest.raises(ValueError, match="at least 3"):
        alpha_std_from_loss_noise(PARAMETER_COUNTS[:2], losses[:2], 0.01)
    with pytest.raises(ValueError, match="same length"):
        alpha_std_from_loss_noise(PARAMETER_COUNTS, losses[:-1], 0.01)
    with pytest.raises(ValueError, match="positive"):
        alpha_std_from_loss_noise(PARAMETER_COUNTS, -losses, 0.01)


def test_to_dict_is_json_friendly():
    losses = 10.0 * PARAMETER_COUNTS**-0.08
    payload = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=100).to_dict()
    assert payload["n_points"] == 5
    assert isinstance(payload["kaplan_within_ci"], bool)
