"""Limb-darkening parameterizations."""

import jax.numpy as jnp


def linear_limb_darkening(u, mu):
    """Return the linear limb-darkening law."""
    return 1.0 - u * (1.0 - mu)


def quadratic_limb_darkening(u1, u2, mu):
    """Return the quadratic limb-darkening law."""
    one_minus_mu = 1.0 - mu
    return 1.0 - u1 * one_minus_mu - u2 * one_minus_mu**2


def kipping_q_to_u(q1, q2):
    """Convert Kipping's q1, q2 sampling variables to quadratic coefficients."""
    sqrt_q1 = jnp.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2
