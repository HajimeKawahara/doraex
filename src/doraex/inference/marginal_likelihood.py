"""Analytically marginalized map likelihoods."""

import jax.numpy as jnp
from numpyro import distributions as dist


def diagonal_noise_variance(size, sigma, jitter=1.0e-6):
    """Return a diagonal noise variance vector."""
    return sigma**2 * jnp.ones(size) + jitter


def marginalized_map_distribution(mean, design_matrix, map_covariance, noise_variance):
    """Return p(d | eta) after analytically marginalizing a Gaussian map."""
    factor = design_matrix @ jnp.linalg.cholesky(map_covariance)
    return dist.LowRankMultivariateNormal(
        loc=mean,
        cov_factor=factor,
        cov_diag=noise_variance,
    )


def dense_marginal_covariance(design_matrix, map_covariance, noise_variance):
    """Return the dense marginalized data covariance."""
    return design_matrix @ map_covariance @ design_matrix.T + jnp.diag(noise_variance)
