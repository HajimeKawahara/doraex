"""Pixel-wise Doppler shift operators."""

import jax.numpy as jnp

from doraex.constants import SPEED_OF_LIGHT


def doppler_factor(vlos):
    """Compute the relativistic wavelength Doppler factor.

    Args:
        vlos: Line-of-sight velocity in km/s. Positive values correspond to a
            redshift.

    Returns:
        The factor ``D = (1 + beta) / sqrt(1 - beta**2)`` from Ureshino et al.
        Eq. (5), where ``beta = vlos / c``. The returned value maps an
        observed wavelength to the rest-frame wavelength through ``lambda / D``.
    """
    beta = vlos / (SPEED_OF_LIGHT * 1.0e-3)
    return (1.0 + beta) / jnp.sqrt(1.0 - beta**2)


def shifted_profile(wavelengths, rest_profile, doppler_factors):
    """Interpolate a rest-frame profile onto Doppler-shifted pixel grids.

    This is the compact JAX implementation of the interpolation operator
    ``C^(jk)`` in Ureshino et al. Eq. (7) and Appendix B.

    Args:
        wavelengths: One-dimensional wavelength grid on which both the
            observed profile and rest profile are represented.
        rest_profile: Rest-frame local spectrum or line profile sampled on
            ``wavelengths``.
        doppler_factors: Doppler factors for one or more surface pixels.

    Returns:
        A two-dimensional array with shape ``(n_wavelength, n_pixel)`` when
        ``doppler_factors`` is one-dimensional. Column ``j`` is the local
        profile evaluated at ``wavelengths / doppler_factors[j]``.
    """
    return jnp.interp(wavelengths[:, None] / doppler_factors, wavelengths, rest_profile)
