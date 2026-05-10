"""Analytically marginalized map likelihoods."""

import jax.numpy as jnp
from numpyro import distributions as dist


def diagonal_noise_variance(size, sigma, jitter=1.0e-6):
    """Build the diagonal entries of an independent Gaussian noise covariance.

    Args:
        size: Number of flattened data points.
        sigma: Per-datum Gaussian noise standard deviation.
        jitter: Additional variance floor added to each diagonal entry.

    Returns:
        A one-dimensional variance vector representing ``Sigma_d`` for the
        independent-noise model in Ureshino et al. Eq. (15).
    """
    return sigma**2 * jnp.ones(size) + jitter


def marginalized_map_distribution(mean, design_matrix, map_covariance, noise_variance):
    """Return the data distribution after analytically marginalizing the map.

    For the linear Gaussian model ``d = mean + W a + eps`` with
    ``a ~ N(0, Sigma_a)`` and independent noise ``eps``, this constructs
    ``p(d | eta) = N(mean, W Sigma_a W.T + Sigma_d)`` without materializing the
    dense covariance.

    Args:
        mean: Flattened deterministic model component.
        design_matrix: Linear operator ``W`` mapping map coefficients to data.
        map_covariance: Prior covariance ``Sigma_a`` of the map coefficients.
        noise_variance: One-dimensional diagonal of the data-noise covariance
            ``Sigma_d``.

    Returns:
        A ``numpyro.distributions.LowRankMultivariateNormal`` representing the
        marginalized likelihood used to sample nonlinear parameters only.
    """
    factor = design_matrix @ jnp.linalg.cholesky(map_covariance)
    return dist.LowRankMultivariateNormal(
        loc=mean,
        cov_factor=factor,
        cov_diag=noise_variance,
    )


def dense_marginal_covariance(design_matrix, map_covariance, noise_variance):
    """Materialize the dense marginalized data covariance.

    Args:
        design_matrix: Linear operator ``W`` mapping map coefficients to data.
        map_covariance: Prior covariance ``Sigma_a`` of the map coefficients.
        noise_variance: One-dimensional diagonal of the data-noise covariance
            ``Sigma_d``.

    Returns:
        Dense covariance ``W @ Sigma_a @ W.T + diag(noise_variance)``. This is
        mainly intended for tests and diagnostics; inference should prefer the
        low-rank representation when possible.
    """
    return design_matrix @ map_covariance @ design_matrix.T + jnp.diag(noise_variance)
