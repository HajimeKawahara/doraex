"""Conditional map posterior reconstruction."""

import jax.numpy as jnp


def conditional_map_posterior(
    data,
    design_matrix,
    prior_mean,
    prior_covariance,
    noise_variance,
):
    """Compute the conditional Gaussian posterior of the map.

    This implements the explicit Gaussian conditioning formula derived in
    Ureshino et al. Appendix C for fixed nonlinear parameters ``eta``. It is
    used after sampling ``eta`` from the analytically marginalized likelihood
    to reconstruct map moments.

    Args:
        data: Flattened observed data vector ``d``.
        design_matrix: Linear operator ``W`` mapping map coefficients to data.
        prior_mean: Prior mean vector ``mu_a`` for the map coefficients.
        prior_covariance: Prior covariance matrix ``Sigma_a`` for the map
            coefficients.
        noise_variance: One-dimensional diagonal of the independent data-noise
            covariance ``Sigma_d``.

    Returns:
        A tuple ``(posterior_mean, posterior_covariance)`` for
        ``p(a | d, eta)``.
    """
    precision_noise_design = design_matrix.T / noise_variance
    prior_precision = jnp.linalg.inv(prior_covariance)
    posterior_precision = prior_precision + precision_noise_design @ design_matrix
    posterior_covariance = jnp.linalg.inv(posterior_precision)
    residual = data.reshape(-1) - design_matrix @ prior_mean
    posterior_mean = prior_mean + posterior_covariance @ (precision_noise_design @ residual)
    return posterior_mean, posterior_covariance


def posterior_moments_from_conditionals(means, covariance_diagonals):
    """Combine conditional map moments over nonlinear posterior samples.

    Args:
        means: Sequence of conditional posterior mean vectors for map
            coefficients, one per nonlinear posterior sample.
        covariance_diagonals: Sequence of conditional posterior covariance
            diagonals, one per nonlinear posterior sample.

    Returns:
        A tuple ``(mean, variance)``. The variance is the total posterior
        marginal variance from the law of total variance: the average
        conditional variance plus the variance of conditional means.
    """
    mean_stack = jnp.stack(means, axis=0)
    mean = jnp.mean(mean_stack, axis=0)
    within = jnp.mean(jnp.stack(covariance_diagonals, axis=0), axis=0)
    between = jnp.mean((mean_stack - mean[None, :]) ** 2, axis=0)
    return mean, within + between
