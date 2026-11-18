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
from doraex.operators.doppler import (
    doppler_factor,
    shifted_pixel_profiles,
    shifted_profile,
)


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
    *,
    rest_wavelengths=None,
):
    """Build the one-exposure Doppler-imaging design matrix.

    This implements the discretized forward model in Ureshino et al.
    Eq. (10)-(11) for fixed nonlinear geometry and a fixed rest-frame line
    profile. The returned matrix maps a pixel-brightness vector ``a`` to one
    phase-resolved spectrum ``d_k``.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        phase: Rotational phase in cycles.
        wavelengths: One-dimensional wavelength grid.
        line_profile: Rest-frame local line profile.
        pixel_area: Optional equal-area pixel solid-angle factor. The default
            preserves the original BayesianDI convention used by the tests.
        rest_wavelengths: Optional wavelength grid on which ``line_profile``
            is sampled. If omitted, ``wavelengths`` is used.

    Returns:
        A matrix with shape ``(n_wavelength, n_pixel)``. Each column is the
        Doppler-shifted local profile for one pixel multiplied by visibility,
        projected area, limb darkening, and ``pixel_area``.
    """
    phi_rot = rotate_longitude(phi, phase)
    theta_obs, phi_obs = incline(theta, phi_rot, jnp.pi / 2.0 - inclination)

    vlos = line_of_sight_velocity(vrot, inclination, theta, phi_rot)
    factors = doppler_factor(vlos)
    local_profiles = shifted_profile(
        wavelengths,
        line_profile,
        factors,
        rest_wavelengths=rest_wavelengths,
    )

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
    *,
    rest_wavelengths=None,
):
    """Build the phase-stacked Doppler-imaging design matrix.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        phases: Rotational phases in cycles.
        wavelengths: One-dimensional wavelength grid.
        line_profile: Rest-frame local line profile.
        weights: Optional per-phase multiplicative weights for exposure time,
            normalization, or signal-to-noise weighting. If omitted, all phases
            use unit weight.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Optional wavelength grid on which ``line_profile``
            is sampled. If omitted, ``wavelengths`` is used.

    Returns:
        The block-stacked matrix ``W`` from Ureshino et al. Eq. (13)-(14), with
        shape ``(n_phase * n_wavelength, n_pixel)``.
    """
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
            rest_wavelengths=rest_wavelengths,
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
    *,
    rest_wavelengths=None,
):
    """Build the phase-stacked design matrix from observation times.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        obs_times: Observation times in the same units as ``period`` and
            ``t0``.
        period: Rotation period in the same units as ``obs_times``.
        wavelengths: One-dimensional wavelength grid.
        line_profile: Rest-frame local line profile.
        weights: Optional per-exposure multiplicative weights.
        t0: Reference epoch used to convert times to phases.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Optional wavelength grid on which ``line_profile``
            is sampled. If omitted, ``wavelengths`` is used.

    Returns:
        The block-stacked matrix ``W`` with shape
        ``(n_time * n_wavelength, n_pixel)``.
    """
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
        rest_wavelengths=rest_wavelengths,
    )


def integrate_pixel_profiles_from_angles(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    phases,
    wavelengths,
    pixel_profiles,
    weights=None,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Integrate one rest-frame spectrum per pixel over rotation phases.

    Unlike the linear design-matrix builders, this operator accepts a distinct
    local spectrum for every surface pixel. It is therefore suitable for an
    exact nonlinear atmospheric map whose local radiative-transfer result
    cannot be represented by one shared profile and a linear contrast.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        phases: Rotational phases in cycles.
        wavelengths: One-dimensional observed wavelength grid.
        pixel_profiles: Local rest-frame spectra with shape
            ``(n_pixel, n_rest_wavelength)``.
        weights: Optional per-phase multiplicative weights.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Common wavelength grid for ``pixel_profiles``. If
            omitted, ``wavelengths`` is used.

    Returns:
        Disk-integrated spectra with shape
        ``(n_phase, n_wavelength)``.
    """

    phases = jnp.asarray(phases)
    if weights is None:
        weights = jnp.ones_like(phases)
    else:
        weights = jnp.asarray(weights)

    def integrate_one(phase, phase_weight):
        phi_rot = rotate_longitude(phi, phase)
        theta_obs, phi_obs = incline(
            theta,
            phi_rot,
            jnp.pi / 2.0 - inclination,
        )
        vlos = line_of_sight_velocity(vrot, inclination, theta, phi_rot)
        local_profiles = shifted_pixel_profiles(
            wavelengths,
            pixel_profiles,
            doppler_factor(vlos),
            rest_wavelengths=rest_wavelengths,
        )
        mu = projected_mu(theta_obs, phi_obs)
        limb = quadratic_limb_darkening(u1, u2, mu)
        pixel_weight = visible_mask(phi_obs) * mu * limb * pixel_area
        return phase_weight * jnp.sum(
            local_profiles * pixel_weight[None, :],
            axis=1,
        )

    return jax.vmap(integrate_one)(phases, weights)


def integrate_pixel_profiles_from_times(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    obs_times,
    period,
    wavelengths,
    pixel_profiles,
    weights=None,
    t0=0.0,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Integrate per-pixel spectra from observation times."""

    phases = (jnp.asarray(obs_times) - t0) / period
    return integrate_pixel_profiles_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        pixel_profiles,
        weights=weights,
        pixel_area=pixel_area,
        rest_wavelengths=rest_wavelengths,
    )


def local_pixel_pressure_operator_from_angles(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    phases,
    wavelengths,
    pixel_profiles,
    pixel_pressure_derivatives,
    weights=None,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Linearize exact per-pixel spectra with respect to pixel pressures.

    ``pixel_profiles`` and ``pixel_pressure_derivatives`` are evaluated at one
    pressure per pixel. The returned Jacobian maps an additive perturbation in
    each pixel's log pressure to the corresponding first-order perturbation of
    the disk-integrated spectrum.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        phases: Rotational phases in cycles.
        wavelengths: One-dimensional observed wavelength grid.
        pixel_profiles: Local spectra with shape
            ``(n_pixel, n_rest_wavelength)``.
        pixel_pressure_derivatives: Derivatives of ``pixel_profiles`` with
            respect to each pixel's scalar log pressure, with the same shape.
        weights: Optional per-phase multiplicative weights.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Common rest wavelength grid. If omitted,
            ``wavelengths`` is used.

    Returns:
        A tuple ``(baseline, jacobian)``. ``baseline`` has shape
        ``(n_phase * n_wavelength,)`` and ``jacobian`` has shape
        ``(n_phase * n_wavelength, n_pixel)``.
    """

    phases = jnp.asarray(phases)
    if weights is None:
        weights = jnp.ones_like(phases)
    else:
        weights = jnp.asarray(weights)

    def linearize_one(phase, phase_weight):
        phi_rot = rotate_longitude(phi, phase)
        theta_obs, phi_obs = incline(
            theta,
            phi_rot,
            jnp.pi / 2.0 - inclination,
        )
        vlos = line_of_sight_velocity(vrot, inclination, theta, phi_rot)
        factors = doppler_factor(vlos)
        local_profiles = shifted_pixel_profiles(
            wavelengths,
            pixel_profiles,
            factors,
            rest_wavelengths=rest_wavelengths,
        )
        local_derivatives = shifted_pixel_profiles(
            wavelengths,
            pixel_pressure_derivatives,
            factors,
            rest_wavelengths=rest_wavelengths,
        )
        mu = projected_mu(theta_obs, phi_obs)
        limb = quadratic_limb_darkening(u1, u2, mu)
        pixel_weight = (
            phase_weight
            * visible_mask(phi_obs)
            * mu
            * limb
            * pixel_area
        )
        jacobian = local_derivatives * pixel_weight[None, :]
        baseline = jnp.sum(local_profiles * pixel_weight[None, :], axis=1)
        return baseline, jacobian

    baseline, jacobian = jax.vmap(linearize_one)(phases, weights)
    return baseline.reshape(-1), jacobian.reshape(
        (-1, pixel_profiles.shape[0])
    )


def local_pixel_pressure_operator_from_times(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    obs_times,
    period,
    wavelengths,
    pixel_profiles,
    pixel_pressure_derivatives,
    weights=None,
    t0=0.0,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Linearize exact per-pixel spectra from observation times."""

    phases = (jnp.asarray(obs_times) - t0) / period
    return local_pixel_pressure_operator_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        pixel_profiles,
        pixel_pressure_derivatives,
        weights=weights,
        pixel_area=pixel_area,
        rest_wavelengths=rest_wavelengths,
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
    *,
    rest_wavelengths=None,
):
    """Build the two-column Doppler-retrieval linear operator.

    The local spectrum is parameterized as
    ``s_j = (1 - f0 - b_j) * clear_profile + (f0 + b_j) * cloudy_profile``.
    For fixed atmospheric columns this can be written as
    ``d = m0 + W_delta b``, preserving the Ureshino-style linear inverse
    problem for the surface contrast map ``b``.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        phases: Rotational phases in cycles.
        wavelengths: One-dimensional wavelength grid.
        clear_profile: Rest-frame local spectrum for the clear atmospheric
            column.
        cloudy_profile: Rest-frame local spectrum for the cloudy atmospheric
            column.
        mean_cloud_fraction: Uniform baseline cloudy-column fraction ``f0``.
        weights: Optional per-phase multiplicative weights.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Optional wavelength grid on which both local spectra
            are sampled. If omitted, ``wavelengths`` is used.

    Returns:
        A tuple ``(m0, W_delta)``. ``m0`` is the flattened spectrum from the
        uniform baseline column mixture, and ``W_delta`` maps the pixel-level
        cloud contrast map to the flattened residual spectrum.
    """
    base_profile = (
        1.0 - mean_cloud_fraction
    ) * clear_profile + mean_cloud_fraction * cloudy_profile
    delta_profile = cloudy_profile - clear_profile
    return linear_profile_operator_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        base_profile,
        delta_profile,
        weights=weights,
        pixel_area=pixel_area,
        rest_wavelengths=rest_wavelengths,
    )


def linear_profile_operator_from_angles(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    phases,
    wavelengths,
    base_profile,
    contrast_profile,
    weights=None,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Build a Doppler-retrieval operator for a linear local spectrum.

    The local spectrum is parameterized as
    ``s_j = base_profile + b_j * contrast_profile``. This preserves the
    analytically marginalized Gaussian map problem for any contrast coordinate
    whose first-order spectral response is represented by ``contrast_profile``.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        phases: Rotational phases in cycles.
        wavelengths: One-dimensional wavelength grid.
        base_profile: Rest-frame local spectrum for the uniform baseline
            atmosphere.
        contrast_profile: Rest-frame local spectral response to a unit map
            perturbation.
        weights: Optional per-phase multiplicative weights.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Optional wavelength grid on which both local spectra
            are sampled. If omitted, ``wavelengths`` is used.

    Returns:
        A tuple ``(m0, W_delta)`` for the flattened baseline spectrum and the
        contrast-map design matrix.
    """

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
        rest_wavelengths=rest_wavelengths,
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
        contrast_profile,
        weights=weights,
        pixel_area=pixel_area,
        rest_wavelengths=rest_wavelengths,
    )
    uniform_map = jnp.ones(base_matrix.shape[1])
    return base_matrix @ uniform_map, contrast_matrix


def two_column_operator_from_times(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    obs_times,
    period,
    wavelengths,
    clear_profile,
    cloudy_profile,
    mean_cloud_fraction,
    weights=None,
    t0=0.0,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Build the two-column Doppler-retrieval operator from times.

    Args:
        theta: Pixel colatitudes in the co-rotating frame, in radians.
        phi: Pixel longitudes in the co-rotating frame, in radians.
        vrot: Equatorial rotation velocity, in km/s.
        inclination: Spin inclination angle in radians.
        u1: First quadratic limb-darkening coefficient.
        u2: Second quadratic limb-darkening coefficient.
        obs_times: Observation times in the same units as ``period`` and
            ``t0``.
        period: Rotation period in the same units as ``obs_times``.
        wavelengths: One-dimensional wavelength grid.
        clear_profile: Rest-frame local spectrum for the clear atmospheric
            column.
        cloudy_profile: Rest-frame local spectrum for the cloudy atmospheric
            column.
        mean_cloud_fraction: Uniform baseline cloudy-column fraction.
        weights: Optional per-exposure multiplicative weights.
        t0: Reference epoch used to convert times to phases.
        pixel_area: Optional equal-area pixel solid-angle factor.
        rest_wavelengths: Optional wavelength grid on which both local spectra
            are sampled. If omitted, ``wavelengths`` is used.

    Returns:
        A tuple ``(m0, W_delta)`` for the flattened baseline spectrum and the
        cloud-contrast design matrix.
    """
    phases = (jnp.asarray(obs_times) - t0) / period
    return two_column_operator_from_angles(
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
        weights=weights,
        pixel_area=pixel_area,
        rest_wavelengths=rest_wavelengths,
    )


def linear_profile_operator_from_times(
    theta,
    phi,
    vrot,
    inclination,
    u1,
    u2,
    obs_times,
    period,
    wavelengths,
    base_profile,
    contrast_profile,
    weights=None,
    t0=0.0,
    pixel_area=1.0,
    *,
    rest_wavelengths=None,
):
    """Build a linear-profile Doppler operator from observation times."""

    phases = (jnp.asarray(obs_times) - t0) / period
    return linear_profile_operator_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        phases,
        wavelengths,
        base_profile,
        contrast_profile,
        weights=weights,
        pixel_area=pixel_area,
        rest_wavelengths=rest_wavelengths,
    )
