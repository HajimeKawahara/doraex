"""Gaussian-process priors on spherical maps."""

from jax import lax
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


def zero_mean_basis(n_pixel, weights=None):
    """Return an orthonormal basis for the weighted zero-mean subspace.

    Args:
        n_pixel: Number of map pixels.
        weights: Optional weights defining the constrained mean. When omitted,
            all pixels receive equal weight.

    Returns:
        A matrix with shape ``(n_pixel, n_pixel - 1)`` whose columns span the
        subspace satisfying ``weights @ x = 0``.
    """

    if weights is None:
        index = jnp.arange(n_pixel - 1)
        row = jnp.arange(n_pixel)[:, None]
        scale = jnp.sqrt((index + 1.0) * (index + 2.0))
        positive = row <= index
        negative = row == (index + 1)
        return (
            jnp.where(positive, 1.0 / scale, 0.0)
            + jnp.where(negative, -(index + 1.0) / scale, 0.0)
        )
    else:
        weights = jnp.asarray(weights)
        weights = weights / jnp.sum(weights)
    weights = weights.astype(jnp.result_type(weights, 1.0))
    q, _ = jnp.linalg.qr(weights[:, None], mode="complete")
    return q[:, 1:]


def zero_mean_covariance_factor(covariance, jitter=1.0e-6, weights=None):
    """Build a Cholesky factor on the weighted zero-mean subspace.

    This avoids Cholesky factorization of the singular projected covariance
    ``P @ covariance @ P.T``. Instead, it factors the reduced covariance
    ``B.T @ covariance @ B`` where columns of ``B`` span the zero-mean subspace.

    Args:
        covariance: Square covariance matrix for an unconstrained map vector.
        jitter: Positive value added to the reduced covariance diagonal.
        weights: Optional weights defining the constrained mean. When omitted,
            all pixels receive equal weight.

    Returns:
        A matrix with shape ``(n_pixel, n_pixel - 1)``. Multiplying standard
        normal reduced coordinates by its transpose gives a zero-mean map draw.
    """

    covariance = jnp.asarray(covariance)
    basis = zero_mean_basis(covariance.shape[0], weights=weights).astype(
        covariance.dtype
    )
    projected_basis = jnp.matmul(
        covariance,
        basis,
        precision=lax.Precision.HIGHEST,
    )
    reduced_covariance = jnp.matmul(
        basis.T,
        projected_basis,
        precision=lax.Precision.HIGHEST,
    )
    reduced_covariance = 0.5 * (reduced_covariance + reduced_covariance.T)
    reduced_covariance = add_diagonal_jitter(reduced_covariance, jitter=jitter)
    reduced_factor = jnp.linalg.cholesky(reduced_covariance)
    return jnp.matmul(
        basis,
        reduced_factor,
        precision=lax.Precision.HIGHEST,
    )


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
