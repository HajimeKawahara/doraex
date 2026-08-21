"""Pixel-wise Doppler shift operators."""

import jax.numpy as jnp
import numpy as np

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


def doppler_padded_wavelengths(
    wavelengths,
    max_abs_velocity,
    *,
    guard_points=1,
):
    """Extend an observed grid to cover a bounded Doppler shift.

    Args:
        wavelengths: Positive, strictly increasing observed wavelengths.
        max_abs_velocity: Maximum absolute line-of-sight velocity in km/s.
        guard_points: Number of extra log-wavelength cells beyond each
            theoretical Doppler boundary.

    Returns:
        A NumPy array containing the original grid with logarithmically spaced
        padding on both sides.
    """

    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("wavelengths must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths must contain only finite positive values.")
    log_wavelengths = np.log(wavelengths)
    if np.any(np.diff(log_wavelengths) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing.")
    max_abs_velocity = float(max_abs_velocity)
    speed_of_light_kms = SPEED_OF_LIGHT * 1.0e-3
    if (
        not np.isfinite(max_abs_velocity)
        or max_abs_velocity < 0.0
        or max_abs_velocity >= speed_of_light_kms
    ):
        raise ValueError(
            "max_abs_velocity must be finite, non-negative, and below the "
            "speed of light."
        )
    if not isinstance(guard_points, (int, np.integer)) or guard_points < 0:
        raise ValueError("guard_points must be a non-negative integer.")
    if max_abs_velocity == 0.0:
        return wavelengths.copy()

    beta = max_abs_velocity / speed_of_light_kms
    log_doppler_factor = np.arctanh(beta)
    log_step = np.min(np.diff(log_wavelengths))
    extension_count = int(np.ceil(log_doppler_factor / log_step)) + guard_points
    offsets = log_step * np.arange(extension_count, 0, -1)
    left = np.exp(log_wavelengths[0] - offsets)
    offsets = log_step * np.arange(1, extension_count + 1)
    right = np.exp(log_wavelengths[-1] + offsets)
    return np.concatenate((left, wavelengths, right))


def shifted_profile(
    wavelengths,
    rest_profile,
    doppler_factors,
    *,
    rest_wavelengths=None,
):
    """Interpolate a rest-frame profile onto Doppler-shifted pixel grids.

    This is the compact JAX implementation of the interpolation operator
    ``C^(jk)`` in Ureshino et al. Eq. (7) and Appendix B.

    Args:
        wavelengths: One-dimensional observed wavelength grid.
        rest_profile: Rest-frame local spectrum or line profile.
        doppler_factors: Doppler factors for one or more surface pixels.
        rest_wavelengths: Wavelength grid on which ``rest_profile`` is sampled.
            It must cover every ``wavelengths / doppler_factors`` query;
            out-of-range queries return ``NaN``. If omitted, ``wavelengths``
            and the legacy endpoint-clamping behavior are used.

    Returns:
        A two-dimensional array with shape ``(n_wavelength, n_pixel)`` when
        ``doppler_factors`` is one-dimensional. Column ``j`` is the local
        profile evaluated at ``wavelengths / doppler_factors[j]``.
    """
    queries = wavelengths[:, None] / doppler_factors
    if rest_wavelengths is None:
        return jnp.interp(queries, wavelengths, rest_profile)
    return jnp.interp(
        queries,
        rest_wavelengths,
        rest_profile,
        left=jnp.nan,
        right=jnp.nan,
    )
