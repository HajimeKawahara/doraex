"""Conditional map posterior reconstruction."""

import jax.numpy as jnp


def conditional_map_posterior(
    data,
    design_matrix,
    prior_mean,
    prior_covariance,
    noise_variance,
):
    """Return p(a | d, eta) for a linear Gaussian map model."""
    precision_noise_design = design_matrix.T / noise_variance
    prior_precision = jnp.linalg.inv(prior_covariance)
    posterior_precision = prior_precision + precision_noise_design @ design_matrix
    posterior_covariance = jnp.linalg.inv(posterior_precision)
    residual = data.reshape(-1) - design_matrix @ prior_mean
    posterior_mean = prior_mean + posterior_covariance @ (precision_noise_design @ residual)
    return posterior_mean, posterior_covariance


def posterior_moments_from_conditionals(means, covariance_diagonals):
    """Combine conditional Gaussian moments over nonlinear posterior samples."""
    mean_stack = jnp.stack(means, axis=0)
    mean = jnp.mean(mean_stack, axis=0)
    within = jnp.mean(jnp.stack(covariance_diagonals, axis=0), axis=0)
    between = jnp.mean((mean_stack - mean[None, :]) ** 2, axis=0)
    return mean, within + between
