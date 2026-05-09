"""Design-matrix builders for Doppler retrieval."""

import jax
import jax.numpy as jnp

from doraex.geometry.limb_darkening import quadratic_limb_darkening
from doraex.geometry.rotation import (
    incline,
    line_of_sight_velocity,
    projected_mu,
    rotate_longitude,
    visible_mask,
)
from doraex.operators.doppler import doppler_factor, shifted_profile


def exposure_matrix_from_angles(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    phase,
    wavelengths,
    line_profile,
    pixel_area=1.0,
):
    """Build the design matrix for one exposure from explicit pixel angles."""
    phi_rot = rotate_longitude(phi, phase)
    theta_obs, phi_obs = incline(theta, phi_rot, jnp.pi / 2.0 - inclination)

    vlos = line_of_sight_velocity(vrot, inclination, theta, phi_rot)
    factors = doppler_factor(vlos)
    local_profiles = shifted_profile(wavelengths, line_profile, factors)

    mu = projected_mu(theta_obs, phi_obs)
    limb = quadratic_limb_darkening(u1, u2, mu)
    weight = visible_mask(phi_obs) * mu * limb * pixel_area
    return weight[None, :] * local_profiles


def full_design_matrix_from_angles(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    phases,
    wavelengths,
    line_profile,
    weights=None,
    pixel_area=1.0,
):
    """Build the stacked design matrix for all phases."""
    phases = jnp.asarray(phases)
    if weights is None:
        weights = jnp.ones_like(phases)

    def build_one(phase, weight):
        matrix = exposure_matrix_from_angles(
            theta,
            phi,
            vrot,
            inclination,
            u1,
            u2,
            phase,
            wavelengths,
            line_profile,
            pixel_area=pixel_area,
        )
        return weight * matrix

    stack = jax.vmap(build_one)(phases, weights)
    return stack.reshape((len(phases) * len(wavelengths), -1))


def full_design_matrix_from_times(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    obs_times,
    period,
    wavelengths,
    line_profile,
    weights=None,
    t0=0.0,
    pixel_area=1.0,
):
    """Build the stacked design matrix from observation times and period."""
    phases = (jnp.asarray(obs_times) - t0) / period
    return full_design_matrix_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        line_profile,
        weights=weights,
        pixel_area=pixel_area,
    )


def two_column_operator_from_angles(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    phases,
    wavelengths,
    clear_profile,
    cloudy_profile,
    mean_cloud_fraction,
    weights=None,
    pixel_area=1.0,
):
    """Return the uniform component and contrast-map matrix for two columns."""
    base_profile = (
        1.0 - mean_cloud_fraction
    ) * clear_profile + mean_cloud_fraction * cloudy_profile
    delta_profile = cloudy_profile - clear_profile

    base_matrix = full_design_matrix_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        base_profile,
        weights=weights,
        pixel_area=pixel_area,
    )
    contrast_matrix = full_design_matrix_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        delta_profile,
        weights=weights,
        pixel_area=pixel_area,
    )
    uniform_map = jnp.ones(base_matrix.shape[1])
    return base_matrix @ uniform_map, contrast_matrix
