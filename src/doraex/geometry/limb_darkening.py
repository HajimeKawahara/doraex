"""Limb-darkening parameterizations."""

import jax.numpy as jnp


def linear_limb_darkening(u, mu):
    """Evaluate the linear limb-darkening law.

    Args:
        u: Linear limb-darkening coefficient.
        mu: Projected direction cosine between the local surface normal and
            the observer line of sight.

    Returns:
        The multiplicative limb-darkening factor ``1 - u * (1 - mu)`` from
        Ureshino et al. Eq. (8).
    """
    return 1.0 - u * (1.0 - mu)


def quadratic_limb_darkening(u1, u2, mu):
    """Evaluate the quadratic limb-darkening law.

    The first-stage implementation uses this law in the design matrix so that
    the linear-law formulation in Ureshino et al. can be generalized without
    changing the Doppler-imaging operator.

    Args:
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        mu: Projected direction cosine between the local surface normal and
            the observer line of sight.

    Returns:
        The multiplicative factor ``1 - u1 * (1 - mu) - u2 * (1 - mu)**2``.
    """
    one_minus_mu = 1.0 - mu
    return 1.0 - u1 * one_minus_mu - u2 * one_minus_mu**2


def kipping_q_to_u(q1, q2):
    """Convert Kipping sampling variables to quadratic coefficients.

    Args:
        q1: Kipping's first bounded sampling variable. Values should be in
            ``[0, 1]`` for physically valid quadratic coefficients.
        q2: Kipping's second bounded sampling variable. Values should be in
            ``[0, 1]`` for physically valid quadratic coefficients.

    Returns:
        A tuple ``(u1, u2)`` of quadratic limb-darkening coefficients.
    """
    sqrt_q1 = jnp.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2
