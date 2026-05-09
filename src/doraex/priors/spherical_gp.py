"""Gaussian-process priors on spherical maps."""

import jax.numpy as jnp


def squared_exponential_covariance(distance_matrix, amplitude, length_scale):
    """Return a squared-exponential covariance on angular distances."""
    return amplitude**2 * jnp.exp(-(distance_matrix**2) / (2.0 * length_scale**2))


def add_diagonal_jitter(matrix, jitter=1.0e-6):
    """Add diagonal jitter to a square matrix."""
    return matrix + jitter * jnp.eye(matrix.shape[0], dtype=matrix.dtype)
