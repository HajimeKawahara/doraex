"""Gaussian-process priors on spherical maps."""

import jax.numpy as jnp


def squared_exponential_covariance(distance_matrix, amplitude, length_scale):
    """Evaluate a squared-exponential covariance on the sphere.

    Args:
        distance_matrix: Pairwise great-circle angular distances between map
            pixels, in radians.
        amplitude: Standard deviation of the surface-map variations.
        length_scale: Angular correlation length scale in radians.

    Returns:
        The covariance matrix ``amplitude**2 * exp(-d**2 / (2 * l**2))`` used
        for the Gaussian-process map prior in Ureshino et al. Eq. (17).
    """
    return amplitude**2 * jnp.exp(-(distance_matrix**2) / (2.0 * length_scale**2))


def add_diagonal_jitter(matrix, jitter=1.0e-6):
    """Add diagonal jitter to a square covariance-like matrix.

    Args:
        matrix: Square matrix to regularize.
        jitter: Positive value added to each diagonal element.

    Returns:
        ``matrix + jitter * I`` with the identity using the input matrix dtype.
        This is useful before Cholesky factorization or explicit inversion of
        nearly singular GP covariance matrices.
    """
    return matrix + jitter * jnp.eye(matrix.shape[0], dtype=matrix.dtype)


def project_zero_mean_covariance(covariance, weights=None):
    """Project a covariance matrix onto the weighted zero-mean subspace.

    Args:
        covariance: Square covariance matrix for a map vector.
        weights: Optional pixel weights defining the constrained mean. When
            omitted, all pixels receive equal weight.

    Returns:
        ``P @ covariance @ P.T`` where ``P`` removes the weighted monopole
        component. The resulting covariance is positive semidefinite and should
        generally receive diagonal jitter before Cholesky factorization.
    """

    covariance = jnp.asarray(covariance)
    n_pixel = covariance.shape[0]
    if weights is None:
        weights = jnp.ones(n_pixel, dtype=covariance.dtype) / n_pixel
    else:
        weights = jnp.asarray(weights, dtype=covariance.dtype)
        weights = weights / jnp.sum(weights)
    ones = jnp.ones(n_pixel, dtype=covariance.dtype)
    projector = jnp.eye(n_pixel, dtype=covariance.dtype) - jnp.outer(ones, weights)
    return projector @ covariance @ projector.T
