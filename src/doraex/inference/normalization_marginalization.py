"""Marginalize a bounded uniform divisor in an independent Gaussian likelihood."""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from scipy.stats import truncnorm


# Restrict quadrature to the resolved likelihood peak. The omitted Gaussian or
# exponential tail is below float64 precision relative to the retained mass.
_LOG_LIKELIHOOD_DROP = 48.0


@lru_cache(maxsize=8)
def _legendre_rule(order):
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order < 2:
        raise ValueError("quadrature_order must be an integer of at least 2.")
    return np.polynomial.legendre.leggauss(order)


def _normalization_quadrature(
    observed, prediction, variance, lower, upper, quadrature_order
):
    """Return the centered likelihood, inverse-A center, displacements, and weights."""

    if not jax.config.x64_enabled:
        raise ValueError("Normalization marginalization requires JAX X64 enabled.")
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("Normalization bounds must satisfy 0 < lower < upper.")
    nodes, weights = _legendre_rule(quadrature_order)
    observed, prediction, variance = jnp.broadcast_arrays(
        jnp.asarray(observed, dtype=jnp.float64),
        jnp.asarray(prediction, dtype=jnp.float64),
        jnp.asarray(variance, dtype=jnp.float64),
    )
    if observed.size == 0:
        raise ValueError("The likelihood requires at least one observation.")

    # In s = 1 / A, the likelihood is Gaussian and the original Uniform(A)
    # density contributes 1 / ((upper - lower) * s**2), not a uniform s prior.
    s_lower, s_upper = 1.0 / upper, 1.0 / lower
    full_width = s_upper - s_lower
    precision = jnp.sum(prediction**2 / variance)
    rhs = jnp.sum(prediction * observed / variance)
    safe_precision = jnp.where(precision > 0.0, precision, 1.0)
    center = jnp.clip(rhs / safe_precision, s_lower, s_upper)
    residual = observed - center * prediction
    slope = -jnp.sum(prediction * residual / variance)
    center_log_likelihood = -0.5 * jnp.sum(
        residual**2 / variance + jnp.log(2.0 * jnp.pi * variance)
    )

    # Solve 0.5 * precision * d**2 +/- slope * d = likelihood_drop.
    # Rationalized roots resolve both a narrow interior Gaussian and an
    # arbitrarily far outside peak, whose in-bounds mass is an exponential tail.
    # Capping the radius at full_width also handles prediction == 0 smoothly.
    drop = _LOG_LIKELIHOOD_DROP
    root = jnp.sqrt(
        jnp.maximum(slope**2 + 2.0 * precision * drop, jnp.finfo(jnp.float64).tiny)
    )
    minimum_denominator = 2.0 * drop / full_width
    left_radius = 2.0 * drop / jnp.maximum(root - slope, minimum_denominator)
    right_radius = 2.0 * drop / jnp.maximum(root + slope, minimum_denominator)
    left = jnp.maximum(s_lower - center, -left_radius)
    right = jnp.minimum(s_upper - center, right_radius)
    half_width = 0.5 * (right - left)
    displacement = 0.5 * (left + right) + half_width * jnp.asarray(nodes)
    inverse_a = center + displacement
    log_weights = (
        jnp.log(jnp.asarray(weights) * half_width)
        - 2.0 * jnp.log(inverse_a)
        - slope * displacement
        - 0.5 * precision * displacement**2
        - jnp.log(upper - lower)
    )
    return center_log_likelihood, center, displacement, log_weights


def marginal_normalization_log_likelihood(
    observed, prediction, variance, lower=1.0, upper=1.2, quadrature_order=64
):
    """Integrate ``Normal(observed; prediction / A, variance)`` over Uniform(A).

    ``prediction`` must exclude A. ``variance`` can be scalar or broadcastable
    to the observations and must be finite and positive. All entries share one
    A; call once per chip. Bounds and quadrature order are static arguments.

    The scalar result includes Gaussian and uniform-prior normalizing constants.
    Evaluation and differentiation require JAX X64 enabled; arrays are promoted
    to float64 without modifying global configuration. Narrow peaks and peaks
    outside the prior bounds are integrated on their local likelihood scale.
    This is numerical quadrature of the original target, with tails below a
    relative likelihood of exp(-48) omitted. Check order convergence when using
    bounds substantially wider than the default interval.
    """

    center_log_likelihood, _, _, log_weights = _normalization_quadrature(
        observed, prediction, variance, lower, upper, quadrature_order
    )
    return center_log_likelihood + logsumexp(log_weights)


def conditional_normalization_moments(
    observed, prediction, variance, lower=1.0, upper=1.2, quadrature_order=64
):
    """Return the conditional mean and variance of A under the same target.

    Inputs and precision requirements match
    :func:`marginal_normalization_log_likelihood`. These are conditional moments,
    not posterior draws. The posterior total variance includes both the average
    conditional variance and the variance of the conditional means.
    """

    _, center, displacement, log_weights = _normalization_quadrature(
        observed, prediction, variance, lower, upper, quadrature_order
    )
    a_values = 1.0 / (center + displacement)
    probabilities = jax.nn.softmax(log_weights)
    mean = jnp.sum(probabilities * a_values)
    variance = jnp.sum(probabilities * (a_values - mean) ** 2)
    return mean, variance


def conditional_inverse_normalization_moments(
    observed, prediction, variance, lower=1.0, upper=1.2, quadrature_order=64
):
    """Return conditional mean and variance of 1/A with the original A prior.

    These inverse-divisor moments describe the amplitude uncertainty remaining
    after applying the finite A bounds. They can be used in a local metric;
    they do not replace the marginalized likelihood or its exact gradient.
    Precision and input requirements match the other quadrature helpers.
    """

    _, center, displacement, log_weights = _normalization_quadrature(
        observed, prediction, variance, lower, upper, quadrature_order
    )
    probabilities = jax.nn.softmax(log_weights)
    mean_displacement = jnp.sum(probabilities * displacement)
    # Work on the narrow local scale, avoiding reciprocal round trips and
    # subtraction of nearly equal inverse-A values when computing the variance.
    variance = jnp.sum(probabilities * (displacement - mean_displacement) ** 2)
    return center + mean_displacement, variance


def draw_conditional_normalization(
    observed,
    prediction,
    variance,
    rng,
    lower=1.0,
    upper=1.2,
    *,
    max_trials=10000,
):
    """Draw one continuous conditional A on CPU, without quadrature nodes.

    The inverse divisor s = 1 / A has a truncated Gaussian likelihood times
    the original uniform-A prior's s**-2 Jacobian. A truncated Gaussian proposal
    is therefore accepted with probability (s_lower / s)**2. The acceptance
    probability is at least (lower / upper)**2, or 0.694 for the default bounds.
    No likelihood tails are discarded. SciPy evaluates Gaussian tail quantiles
    in log space; numerical failures and exhausted rejection trials raise errors.

    ``rng`` must be a ``numpy.random.Generator``. Finite observations/predictions
    and strictly positive finite scalar or broadcastable variances are required.
    All entries share one A. This helper uses NumPy float64, requires no JAX X64
    configuration, and is intended for restoring A after marginalized inference.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("Normalization bounds must satisfy 0 < lower < upper.")
    if (
        not isinstance(max_trials, (int, np.integer))
        or isinstance(max_trials, bool)
        or max_trials < 1
    ):
        raise ValueError("max_trials must be a positive integer.")
    observed, prediction, variance = np.broadcast_arrays(
        np.asarray(observed, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
        np.asarray(variance, dtype=np.float64),
    )
    if observed.size == 0 or not all(
        np.all(np.isfinite(values)) for values in (observed, prediction, variance)
    ):
        raise ValueError("Observations, predictions, and variances must be nonempty and finite.")
    if np.any(variance <= 0.0):
        raise ValueError("Variances must be strictly positive.")
    if np.all(prediction == 0.0):
        return float(rng.uniform(lower, upper))

    precision = np.sum(prediction**2 / variance)
    rhs = np.sum(prediction * observed / variance)
    if not np.isfinite(precision) or precision <= 0.0 or not np.isfinite(rhs):
        raise ValueError("Conditional Gaussian sufficient statistics are not representable.")
    mean = rhs / precision
    scale = 1.0 / np.sqrt(precision)
    s_lower, s_upper = 1.0 / upper, 1.0 / lower
    left, right = (s_lower - mean) / scale, (s_upper - mean) / scale
    if not (np.isfinite(left) and np.isfinite(right) and left < right):
        raise ValueError("Conditional Gaussian truncation bounds are not representable.")

    for _ in range(max_trials):
        inverse_a = float(truncnorm.rvs(left, right, loc=mean, scale=scale, random_state=rng))
        if not (np.isfinite(inverse_a) and s_lower <= inverse_a <= s_upper):
            raise FloatingPointError("Truncated Gaussian proposal left the normalization bounds.")
        if rng.random() < (s_lower / inverse_a) ** 2:
            return 1.0 / inverse_a
    raise RuntimeError("Conditional normalization rejection sampler exhausted max_trials.")
