"""Pixel-wise Doppler shift operators."""

import jax.numpy as jnp

from doraex.constants import SPEED_OF_LIGHT


def doppler_factor(vlos):
    """Return the relativistic Doppler wavelength factor for km/s velocities."""
    beta = vlos / (SPEED_OF_LIGHT * 1.0e-3)
    return (1.0 + beta) / jnp.sqrt(1.0 - beta**2)


def shifted_profile(wavelengths, rest_profile, doppler_factors):
    """Interpolate a rest-frame profile onto per-pixel Doppler-shifted grids."""
    return jnp.interp(wavelengths[:, None] / doppler_factors, wavelengths, rest_profile)
