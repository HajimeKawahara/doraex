"""Milestone 2 workflow helpers for fixed two-column Doppler retrieval."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.geometry.healpix import angular_distance_matrix, healpix_pixel_angles
from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.map_posterior import conditional_map_posterior
from doraex.operators.design_matrix import (
    linear_profile_operator_from_times,
    two_column_operator_from_times,
)
from doraex.priors.spherical_gp import (
    add_diagonal_jitter,
    project_zero_mean_covariance,
    squared_exponential_covariance,
)
from doraex.spectra.exojax_forward import (
    load_cloud_profile_grid,
    load_t0_alpha_vmr_cloud_profile_grid,
    load_t0_cloud_profile_grid,
    load_t0_vmr_cloud_profile_grid,
    load_two_column_profiles,
    save_cloud_profile_grid,
    save_t0_cloud_profile_grid,
    save_two_column_profiles,
    synthetic_cloud_profile_grid,
    synthetic_t0_cloud_profile_grid,
    synthetic_two_column_profiles,
)


@dataclass(frozen=True)
class Luhman16BGeometry:
    """HEALPix geometry needed by the Luhman 16B workflows."""

    theta: jnp.ndarray
    phi: jnp.ndarray
    distance_matrix: jnp.ndarray
    nside: int


def build_luhman16b_geometry(nside=8, order="ring"):
    """Build pixel angles and angular-distance matrix for a HEALPix grid."""

    theta, phi = healpix_pixel_angles(nside, order=order)
    return Luhman16BGeometry(
        theta=theta,
        phi=phi,
        distance_matrix=angular_distance_matrix(theta, phi),
        nside=nside,
    )


def _interpolate_profile_grid(parameter_grid, profile_grid, parameter):
    """Linearly interpolate a precomputed one-parameter profile grid."""

    parameter_grid = jnp.asarray(parameter_grid)
    profile_grid = jnp.asarray(profile_grid)
    parameter = jnp.asarray(parameter)
    index = jnp.searchsorted(parameter_grid, parameter, side="right") - 1
    index = jnp.clip(index, 0, parameter_grid.shape[0] - 2)
    left = parameter_grid[index]
    right = parameter_grid[index + 1]
    fraction = (parameter - left) / (right - left)
    return (1.0 - fraction) * profile_grid[index] + fraction * profile_grid[index + 1]


def _interpolate_profile_grid_2d(x_grid, y_grid, profile_grid, x, y):
    """Bilinearly interpolate a precomputed two-parameter profile grid."""

    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    profile_grid = jnp.asarray(profile_grid)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x_index = jnp.searchsorted(x_grid, x, side="right") - 1
    y_index = jnp.searchsorted(y_grid, y, side="right") - 1
    x_index = jnp.clip(x_index, 0, x_grid.shape[0] - 2)
    y_index = jnp.clip(y_index, 0, y_grid.shape[0] - 2)
    x_fraction = (x - x_grid[x_index]) / (x_grid[x_index + 1] - x_grid[x_index])
    y_fraction = (y - y_grid[y_index]) / (y_grid[y_index + 1] - y_grid[y_index])
    p00 = profile_grid[x_index, y_index]
    p10 = profile_grid[x_index + 1, y_index]
    p01 = profile_grid[x_index, y_index + 1]
    p11 = profile_grid[x_index + 1, y_index + 1]
    return (
        (1.0 - x_fraction) * (1.0 - y_fraction) * p00
        + x_fraction * (1.0 - y_fraction) * p10
        + (1.0 - x_fraction) * y_fraction * p01
        + x_fraction * y_fraction * p11
    )


def _interpolate_profile_grid_3d(x_grid, y_grid, z_grid, profile_grid, x, y, z):
    """Trilinearly interpolate a precomputed three-parameter profile grid."""

    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    z_grid = jnp.asarray(z_grid)
    profile_grid = jnp.asarray(profile_grid)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)
    x_index = jnp.searchsorted(x_grid, x, side="right") - 1
    y_index = jnp.searchsorted(y_grid, y, side="right") - 1
    z_index = jnp.searchsorted(z_grid, z, side="right") - 1
    x_index = jnp.clip(x_index, 0, x_grid.shape[0] - 2)
    y_index = jnp.clip(y_index, 0, y_grid.shape[0] - 2)
    z_index = jnp.clip(z_index, 0, z_grid.shape[0] - 2)
    x_fraction = (x - x_grid[x_index]) / (x_grid[x_index + 1] - x_grid[x_index])
    y_fraction = (y - y_grid[y_index]) / (y_grid[y_index + 1] - y_grid[y_index])
    z_fraction = (z - z_grid[z_index]) / (z_grid[z_index + 1] - z_grid[z_index])
    p000 = profile_grid[x_index, y_index, z_index]
    p100 = profile_grid[x_index + 1, y_index, z_index]
    p010 = profile_grid[x_index, y_index + 1, z_index]
    p110 = profile_grid[x_index + 1, y_index + 1, z_index]
    p001 = profile_grid[x_index, y_index, z_index + 1]
    p101 = profile_grid[x_index + 1, y_index, z_index + 1]
    p011 = profile_grid[x_index, y_index + 1, z_index + 1]
    p111 = profile_grid[x_index + 1, y_index + 1, z_index + 1]
    return (
        (1.0 - x_fraction) * (1.0 - y_fraction) * (1.0 - z_fraction) * p000
        + x_fraction * (1.0 - y_fraction) * (1.0 - z_fraction) * p100
        + (1.0 - x_fraction) * y_fraction * (1.0 - z_fraction) * p010
        + x_fraction * y_fraction * (1.0 - z_fraction) * p110
        + (1.0 - x_fraction) * (1.0 - y_fraction) * z_fraction * p001
        + x_fraction * (1.0 - y_fraction) * z_fraction * p101
        + (1.0 - x_fraction) * y_fraction * z_fraction * p011
        + x_fraction * y_fraction * z_fraction * p111
    )


def _interpolate_profile_grid_4d(
    w_grid,
    x_grid,
    y_grid,
    z_grid,
    profile_grid,
    w,
    x,
    y,
    z,
):
    """Linearly interpolate a precomputed four-parameter profile grid."""

    w_grid = jnp.asarray(w_grid)
    profile_grid = jnp.asarray(profile_grid)
    w = jnp.asarray(w)
    w_index = jnp.searchsorted(w_grid, w, side="right") - 1
    w_index = jnp.clip(w_index, 0, w_grid.shape[0] - 2)
    w_fraction = (w - w_grid[w_index]) / (w_grid[w_index + 1] - w_grid[w_index])
    left = _interpolate_profile_grid_3d(
        x_grid,
        y_grid,
        z_grid,
        profile_grid[w_index],
        x,
        y,
        z,
    )
    right = _interpolate_profile_grid_3d(
        x_grid,
        y_grid,
        z_grid,
        profile_grid[w_index + 1],
        x,
        y,
        z,
    )
    return (1.0 - w_fraction) * left + w_fraction * right


def load_milestone2_fixed_inputs(
    data_dir,
    profiles_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
):
    """Load Luhman 16B data, geometry, and fixed two-column spectra."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profiles_path is None:
        if not smoke_test:
            raise ValueError("profiles_path is required unless smoke_test=True")
        clear_profile, cloudy_profile = synthetic_two_column_profiles(chip_data.wavelengths)
    else:
        clear_profile, cloudy_profile = load_two_column_profiles(
            profiles_path, expected_wavelengths=chip_data.wavelengths
        )
    return chip_data, geometry, clear_profile, cloudy_profile


def load_milestone2_free_cloud_inputs(
    data_dir,
    profile_grid_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_log_p_cloud_grid=None,
):
    """Load Luhman 16B data, geometry, and a cloudy-profile grid."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profile_grid_path is None:
        if not smoke_test:
            raise ValueError("profile_grid_path is required unless smoke_test=True")
        if smoke_log_p_cloud_grid is None:
            smoke_log_p_cloud_grid = np.linspace(0.0, 2.0, 5)
        clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
            chip_data.wavelengths,
            smoke_log_p_cloud_grid,
        )
        log_p_cloud_grid = np.asarray(smoke_log_p_cloud_grid)
    else:
        clear_profile, log_p_cloud_grid, cloudy_profile_grid = load_cloud_profile_grid(
            profile_grid_path,
            expected_wavelengths=chip_data.wavelengths,
        )
    return chip_data, geometry, clear_profile, log_p_cloud_grid, cloudy_profile_grid


def load_milestone2_free_t0_cloud_inputs(
    data_dir,
    profile_grid_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_t0_grid=None,
    smoke_log_p_cloud_grid=None,
):
    """Load Luhman 16B data, geometry, and T0/cloud profile grids."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profile_grid_path is None:
        if not smoke_test:
            raise ValueError("profile_grid_path is required unless smoke_test=True")
        if smoke_t0_grid is None:
            smoke_t0_grid = np.linspace(1000.0, 1700.0, 5)
        if smoke_log_p_cloud_grid is None:
            smoke_log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
        clear_profile_grid, cloudy_profile_grid = synthetic_t0_cloud_profile_grid(
            chip_data.wavelengths,
            smoke_t0_grid,
            smoke_log_p_cloud_grid,
        )
        t0_grid = np.asarray(smoke_t0_grid)
        log_p_cloud_grid = np.asarray(smoke_log_p_cloud_grid)
    else:
        (
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_t0_cloud_profile_grid(
            profile_grid_path,
            expected_wavelengths=chip_data.wavelengths,
        )
    return (
        chip_data,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
    )


def load_milestone2_joint_free_t0_cloud_inputs(
    data_dir,
    chip_indices=(0, 1, 2, 3),
    profile_grid_template=None,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_t0_grid=None,
    smoke_log_p_cloud_grid=None,
):
    """Load multi-chip Luhman 16B data and T0/cloud profile grids."""

    chip_data_list = []
    t0_grids = []
    log_p_cloud_grids = []
    clear_profile_grids = []
    cloudy_profile_grids = []
    geometry = build_luhman16b_geometry(nside=nside)
    for chip_index in chip_indices:
        profile_grid_path = None
        if profile_grid_template is not None:
            profile_grid_path = str(profile_grid_template).format(chip=chip_index)
        (
            chip_data,
            _,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_milestone2_free_t0_cloud_inputs(
            data_dir,
            profile_grid_path=profile_grid_path,
            chip_index=chip_index,
            nside=nside,
            smoke_test=smoke_test,
            smoke_wavelength_step=smoke_wavelength_step,
            smoke_phase_count=smoke_phase_count,
            smoke_t0_grid=smoke_t0_grid,
            smoke_log_p_cloud_grid=smoke_log_p_cloud_grid,
        )
        chip_data_list.append(chip_data)
        t0_grids.append(t0_grid)
        log_p_cloud_grids.append(log_p_cloud_grid)
        clear_profile_grids.append(clear_profile_grid)
        cloudy_profile_grids.append(cloudy_profile_grid)

    return (
        chip_data_list,
        geometry,
        np.asarray(t0_grids),
        np.asarray(log_p_cloud_grids),
        np.asarray(clear_profile_grids),
        np.asarray(cloudy_profile_grids),
    )


def load_milestone2_free_t0_vmr_cloud_inputs(
    data_dir,
    profile_grid_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_t0_grid=None,
    smoke_log_p_cloud_grid=None,
    smoke_zeta_vmr_grid=None,
):
    """Load Luhman 16B data, geometry, and T0/VMR/cloud profile grids."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profile_grid_path is None:
        if not smoke_test:
            raise ValueError("profile_grid_path is required unless smoke_test=True")
        if smoke_t0_grid is None:
            smoke_t0_grid = np.linspace(1000.0, 1700.0, 5)
        if smoke_log_p_cloud_grid is None:
            smoke_log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
        if smoke_zeta_vmr_grid is None:
            smoke_zeta_vmr_grid = np.linspace(-0.5, 0.5, 5)
        clear_grid_1d, cloudy_grid_2d = synthetic_t0_cloud_profile_grid(
            chip_data.wavelengths,
            smoke_t0_grid,
            smoke_log_p_cloud_grid,
        )
        clear_profile_grid = np.repeat(
            clear_grid_1d[:, None, :],
            len(smoke_zeta_vmr_grid),
            axis=1,
        )
        cloudy_profile_grid = np.repeat(
            cloudy_grid_2d[:, :, None, :],
            len(smoke_zeta_vmr_grid),
            axis=2,
        )
        t0_grid = np.asarray(smoke_t0_grid)
        log_p_cloud_grid = np.asarray(smoke_log_p_cloud_grid)
        zeta_vmr_grid = np.asarray(smoke_zeta_vmr_grid)
    else:
        (
            t0_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_t0_vmr_cloud_profile_grid(
            profile_grid_path,
            expected_wavelengths=chip_data.wavelengths,
        )
    return (
        chip_data,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        clear_profile_grid,
        cloudy_profile_grid,
    )


def load_milestone2_joint_free_t0_vmr_cloud_inputs(
    data_dir,
    chip_indices=(0, 1, 2, 3),
    profile_grid_template=None,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_t0_grid=None,
    smoke_log_p_cloud_grid=None,
    smoke_zeta_vmr_grid=None,
):
    """Load multi-chip Luhman 16B data and T0/VMR/cloud profile grids."""

    chip_data_list = []
    t0_grids = []
    log_p_cloud_grids = []
    zeta_vmr_grids = []
    clear_profile_grids = []
    cloudy_profile_grids = []
    geometry = build_luhman16b_geometry(nside=nside)
    for chip_index in chip_indices:
        profile_grid_path = None
        if profile_grid_template is not None:
            profile_grid_path = str(profile_grid_template).format(chip=chip_index)
        (
            chip_data,
            _,
            t0_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_milestone2_free_t0_vmr_cloud_inputs(
            data_dir,
            profile_grid_path=profile_grid_path,
            chip_index=chip_index,
            nside=nside,
            smoke_test=smoke_test,
            smoke_wavelength_step=smoke_wavelength_step,
            smoke_phase_count=smoke_phase_count,
            smoke_t0_grid=smoke_t0_grid,
            smoke_log_p_cloud_grid=smoke_log_p_cloud_grid,
            smoke_zeta_vmr_grid=smoke_zeta_vmr_grid,
        )
        chip_data_list.append(chip_data)
        t0_grids.append(t0_grid)
        log_p_cloud_grids.append(log_p_cloud_grid)
        zeta_vmr_grids.append(zeta_vmr_grid)
        clear_profile_grids.append(clear_profile_grid)
        cloudy_profile_grids.append(cloudy_profile_grid)

    return (
        chip_data_list,
        geometry,
        np.asarray(t0_grids),
        np.asarray(log_p_cloud_grids),
        np.asarray(zeta_vmr_grids),
        np.asarray(clear_profile_grids),
        np.asarray(cloudy_profile_grids),
    )


def load_milestone2_free_t0_alpha_vmr_cloud_inputs(
    data_dir,
    profile_grid_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_t0_grid=None,
    smoke_alpha_grid=None,
    smoke_log_p_cloud_grid=None,
    smoke_zeta_vmr_grid=None,
):
    """Load Luhman 16B data, geometry, and T0/alpha/VMR/cloud profile grids."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profile_grid_path is None:
        if not smoke_test:
            raise ValueError("profile_grid_path is required unless smoke_test=True")
        if smoke_t0_grid is None:
            smoke_t0_grid = np.linspace(1000.0, 1700.0, 5)
        if smoke_alpha_grid is None:
            smoke_alpha_grid = np.linspace(0.05, 0.20, 3)
        if smoke_log_p_cloud_grid is None:
            smoke_log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
        if smoke_zeta_vmr_grid is None:
            smoke_zeta_vmr_grid = np.linspace(-0.5, 0.5, 5)
        clear_grid_1d, cloudy_grid_2d = synthetic_t0_cloud_profile_grid(
            chip_data.wavelengths,
            smoke_t0_grid,
            smoke_log_p_cloud_grid,
        )
        alpha_scale = 1.0 + 0.05 * (
            np.asarray(smoke_alpha_grid) - np.mean(smoke_alpha_grid)
        ) / max(float(np.ptp(smoke_alpha_grid)), 1.0e-6)
        clear_profile_grid = (
            clear_grid_1d[:, None, None, :]
            * alpha_scale[None, :, None, None]
            * np.ones((1, 1, len(smoke_zeta_vmr_grid), 1))
        )
        cloudy_profile_grid = (
            cloudy_grid_2d[:, None, :, None, :]
            * alpha_scale[None, :, None, None, None]
            * np.ones((1, 1, 1, len(smoke_zeta_vmr_grid), 1))
        )
        t0_grid = np.asarray(smoke_t0_grid)
        alpha_grid = np.asarray(smoke_alpha_grid)
        log_p_cloud_grid = np.asarray(smoke_log_p_cloud_grid)
        zeta_vmr_grid = np.asarray(smoke_zeta_vmr_grid)
    else:
        (
            t0_grid,
            alpha_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_t0_alpha_vmr_cloud_profile_grid(
            profile_grid_path,
            expected_wavelengths=chip_data.wavelengths,
        )
    return (
        chip_data,
        geometry,
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        clear_profile_grid,
        cloudy_profile_grid,
    )


def load_milestone2_joint_free_t0_alpha_vmr_cloud_inputs(
    data_dir,
    chip_indices=(0, 1, 2, 3),
    profile_grid_template=None,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_t0_grid=None,
    smoke_alpha_grid=None,
    smoke_log_p_cloud_grid=None,
    smoke_zeta_vmr_grid=None,
):
    """Load multi-chip Luhman 16B data and T0/alpha/VMR/cloud grids."""

    chip_data_list = []
    t0_grids = []
    alpha_grids = []
    log_p_cloud_grids = []
    zeta_vmr_grids = []
    clear_profile_grids = []
    cloudy_profile_grids = []
    geometry = build_luhman16b_geometry(nside=nside)
    for chip_index in chip_indices:
        profile_grid_path = None
        if profile_grid_template is not None:
            profile_grid_path = str(profile_grid_template).format(chip=chip_index)
        (
            chip_data,
            _,
            t0_grid,
            alpha_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_milestone2_free_t0_alpha_vmr_cloud_inputs(
            data_dir,
            profile_grid_path=profile_grid_path,
            chip_index=chip_index,
            nside=nside,
            smoke_test=smoke_test,
            smoke_wavelength_step=smoke_wavelength_step,
            smoke_phase_count=smoke_phase_count,
            smoke_t0_grid=smoke_t0_grid,
            smoke_alpha_grid=smoke_alpha_grid,
            smoke_log_p_cloud_grid=smoke_log_p_cloud_grid,
            smoke_zeta_vmr_grid=smoke_zeta_vmr_grid,
        )
        chip_data_list.append(chip_data)
        t0_grids.append(t0_grid)
        alpha_grids.append(alpha_grid)
        log_p_cloud_grids.append(log_p_cloud_grid)
        zeta_vmr_grids.append(zeta_vmr_grid)
        clear_profile_grids.append(clear_profile_grid)
        cloudy_profile_grids.append(cloudy_profile_grid)

    return (
        chip_data_list,
        geometry,
        np.asarray(t0_grids),
        np.asarray(alpha_grids),
        np.asarray(log_p_cloud_grids),
        np.asarray(zeta_vmr_grids),
        np.asarray(clear_profile_grids),
        np.asarray(cloudy_profile_grids),
    )


def make_fixed_two_column_nuts_kernel(
    model,
    n_phase,
    period_mode="sampled",
    fixed_period=5.0,
    fixed_ell_b=None,
    fix_geometry=False,
    target_accept_prob=0.9,
    dense_mass=True,
    max_tree_depth=10,
    init_log_p_cloud=None,
    init_t0=None,
):
    """Create the NUTS kernel for Milestone 2."""

    from numpyro.infer import NUTS, init_to_value

    init_values = {
        "log_w": jnp.zeros(n_phase),
        "f_cloud": 0.5,
        "surface_scale": 0.0077,
        "sigma_d": 0.039,
        "sigma_b": 0.05,
    }
    if not fix_geometry:
        init_values.update(
            {
                "cosi": 0.485,
                "v": 31.2,
                "q1": 0.81,
                "q2": 0.59,
            }
        )
    if fixed_ell_b is None:
        init_values["ell_b"] = 0.4
    if period_mode == "sampled":
        init_values["P"] = 4.83
    elif period_mode == "fixed":
        init_values["P"] = fixed_period
    if init_log_p_cloud is not None:
        init_values["log_p_cloud"] = init_log_p_cloud
    if init_t0 is not None:
        init_values["T0"] = init_t0
    return NUTS(
        model,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=init_values),
    )


def run_fixed_two_column_mcmc(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    seed=0,
    period_mode="sampled",
    fixed_period=5.0,
    target_accept_prob=0.9,
    dense_mass=True,
    max_tree_depth=10,
    sigma_b_scale=0.1,
    fixed_ell_b=None,
    fix_geometry=False,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    progress_bar=True,
):
    """Run fixed-atmosphere two-column Doppler retrieval."""

    from numpyro.infer import MCMC
    from doraex.inference.numpyro_models import fixed_two_column_doppler_model

    data = jnp.asarray(chip_data.flux)
    obs_times = jnp.asarray(chip_data.obs_times)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    clear_profile = jnp.asarray(clear_profile)
    cloudy_profile = jnp.asarray(cloudy_profile)

    def model(data_observed):
        return fixed_two_column_doppler_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            clear_profile,
            cloudy_profile,
            period_mode=period_mode,
            fixed_period=fixed_period,
            sigma_b_scale=sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=fix_geometry,
            fixed_cosi=fixed_cosi,
            fixed_v=fixed_v,
            fixed_q1=fixed_q1,
            fixed_q2=fixed_q2,
        )

    kernel = make_fixed_two_column_nuts_kernel(
        model,
        n_phase=chip_data.flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return mcmc


def run_free_cloud_two_column_mcmc(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    seed=0,
    period_mode="fixed",
    fixed_period=4.83,
    log_p_cloud_bounds=(0.0, 2.0),
    init_log_p_cloud=1.35,
    target_accept_prob=0.98,
    dense_mass=True,
    max_tree_depth=10,
    sigma_b_scale=0.1,
    fixed_ell_b=0.4,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    progress_bar=True,
):
    """Run Milestone 2-2 with free cloud-top pressure."""

    from numpyro.infer import MCMC
    from doraex.inference.numpyro_models import free_cloud_two_column_doppler_model

    data = jnp.asarray(chip_data.flux)
    obs_times = jnp.asarray(chip_data.obs_times)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    clear_profile = jnp.asarray(clear_profile)
    log_p_cloud_grid_np = np.asarray(log_p_cloud_grid)
    grid_min = float(np.min(log_p_cloud_grid_np))
    grid_max = float(np.max(log_p_cloud_grid_np))
    if log_p_cloud_bounds[0] < grid_min or log_p_cloud_bounds[1] > grid_max:
        raise ValueError(
            "log_p_cloud_bounds must be covered by log_p_cloud_grid: "
            f"bounds={log_p_cloud_bounds}, grid=({grid_min}, {grid_max})"
        )
    log_p_cloud_grid = jnp.asarray(log_p_cloud_grid)
    cloudy_profile_grid = jnp.asarray(cloudy_profile_grid)

    def model(data_observed):
        return free_cloud_two_column_doppler_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            clear_profile,
            log_p_cloud_grid,
            cloudy_profile_grid,
            period_mode=period_mode,
            fixed_period=fixed_period,
            log_p_cloud_bounds=log_p_cloud_bounds,
            sigma_b_scale=sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=fix_geometry,
            fixed_cosi=fixed_cosi,
            fixed_v=fixed_v,
            fixed_q1=fixed_q1,
            fixed_q2=fixed_q2,
        )

    kernel = make_fixed_two_column_nuts_kernel(
        model,
        n_phase=chip_data.flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_log_p_cloud=init_log_p_cloud,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return mcmc


def run_free_t0_cloud_two_column_mcmc(
    chip_data,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    zeta_vmr_grid=None,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    seed=0,
    period_mode="fixed",
    fixed_period=4.83,
    t0_bounds=(1000.0, 1700.0),
    log_p_cloud_bounds=(-2.0, 2.0),
    zeta_vmr_bounds=(-0.5, 0.5),
    init_t0=1215.0,
    init_log_p_cloud=1.28,
    init_zeta_vmr=0.0,
    target_accept_prob=0.98,
    dense_mass=True,
    max_tree_depth=10,
    sigma_b_scale=0.1,
    fixed_ell_b=0.4,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    progress_bar=True,
):
    """Run grid-based Milestone 2-3a with free T0 and cloud-top pressure."""

    from numpyro.infer import MCMC
    from doraex.inference.numpyro_models import free_t0_cloud_two_column_doppler_model

    data = jnp.asarray(chip_data.flux)
    obs_times = jnp.asarray(chip_data.obs_times)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    t0_grid_np = np.asarray(t0_grid)
    log_p_cloud_grid_np = np.asarray(log_p_cloud_grid)
    t0_grid_min = float(np.min(t0_grid_np))
    t0_grid_max = float(np.max(t0_grid_np))
    log_p_grid_min = float(np.min(log_p_cloud_grid_np))
    log_p_grid_max = float(np.max(log_p_cloud_grid_np))
    if t0_bounds[0] < t0_grid_min or t0_bounds[1] > t0_grid_max:
        raise ValueError(
            "t0_bounds must be covered by t0_grid: "
            f"bounds={t0_bounds}, grid=({t0_grid_min}, {t0_grid_max})"
        )
    if log_p_cloud_bounds[0] < log_p_grid_min or log_p_cloud_bounds[1] > log_p_grid_max:
        raise ValueError(
            "log_p_cloud_bounds must be covered by log_p_cloud_grid: "
            f"bounds={log_p_cloud_bounds}, grid=({log_p_grid_min}, {log_p_grid_max})"
        )
    t0_grid = jnp.asarray(t0_grid)
    log_p_cloud_grid = jnp.asarray(log_p_cloud_grid)
    clear_profile_grid = jnp.asarray(clear_profile_grid)
    cloudy_profile_grid = jnp.asarray(cloudy_profile_grid)

    def model(data_observed):
        return free_t0_cloud_two_column_doppler_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            period_mode=period_mode,
            fixed_period=fixed_period,
            t0_bounds=t0_bounds,
            log_p_cloud_bounds=log_p_cloud_bounds,
            sigma_b_scale=sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=fix_geometry,
            fixed_cosi=fixed_cosi,
            fixed_v=fixed_v,
            fixed_q1=fixed_q1,
            fixed_q2=fixed_q2,
        )

    kernel = make_fixed_two_column_nuts_kernel(
        model,
        n_phase=chip_data.flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_log_p_cloud=init_log_p_cloud,
        init_t0=init_t0,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return mcmc


def make_joint_free_t0_cloud_nuts_kernel(
    model,
    n_chip,
    n_phase,
    period_mode="fixed",
    fixed_period=4.83,
    fixed_ell_b=0.3,
    fix_geometry=True,
    target_accept_prob=0.98,
    dense_mass=True,
    max_tree_depth=10,
    init_log_p_cloud=1.28,
    init_t0=1215.0,
    init_alpha=0.128,
    init_zeta_vmr=0.0,
    shared_atmosphere=False,
    normalization_mode="surface_scale",
    column_mode="clear_cloud",
    sample_zeta_vmr=False,
    sample_alpha=False,
):
    """Create a NUTS kernel for joint multi-chip Milestone 2-4 runs."""

    from numpyro.infer import NUTS, init_to_value

    atmosphere_shape = () if shared_atmosphere else (n_chip,)
    pressure_name = "log_p_mid" if column_mode == "double_cloud" else "log_p_cloud"
    fraction_name = "h_high" if column_mode == "double_cloud" else "f_cloud"
    uses_fraction_map = column_mode != "pressure_perturbation"
    init_values = {
        "T0": jnp.full(atmosphere_shape, init_t0),
        pressure_name: jnp.full(atmosphere_shape, init_log_p_cloud),
        "log_w": jnp.zeros((n_chip, n_phase)),
        "sigma_d": jnp.full((n_chip,), 0.039),
    }
    if uses_fraction_map:
        init_values[fraction_name] = jnp.full(atmosphere_shape, 0.5)
        init_values["sigma_b"] = 0.05
    else:
        init_values["sigma_log_p"] = 0.03
    if normalization_mode == "surface_scale":
        init_values["surface_scale"] = jnp.full((n_chip,), 0.0077)
    elif normalization_mode == "yama":
        init_values["A"] = jnp.full((n_chip,), 1.1)
    else:
        raise ValueError("normalization_mode must be 'surface_scale' or 'yama'")
    if not fix_geometry:
        init_values.update({"cosi": 0.485, "v": 31.2, "q1": 0.81, "q2": 0.59})
    if sample_zeta_vmr:
        init_values["zeta_vmr"] = jnp.full(atmosphere_shape, init_zeta_vmr)
    if sample_alpha:
        init_values["alpha"] = jnp.full(atmosphere_shape, init_alpha)
    if fixed_ell_b is None:
        init_values["ell_b"] = 0.3
    if period_mode == "sampled":
        init_values["P"] = 4.83
    elif period_mode == "fixed":
        init_values["P"] = fixed_period
    return NUTS(
        model,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=init_values),
    )


def run_joint_free_t0_cloud_two_column_mcmc(
    chip_data_list,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    alpha_grid=None,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    seed=0,
    period_mode="fixed",
    fixed_period=4.83,
    t0_bounds=(1000.0, 1700.0),
    alpha_bounds=(0.05, 0.20),
    log_p_cloud_bounds=(-2.0, 2.0),
    zeta_vmr_grid=None,
    zeta_vmr_bounds=(-0.5, 0.5),
    init_t0=1215.0,
    init_alpha=0.128,
    init_log_p_cloud=1.28,
    init_zeta_vmr=0.0,
    target_accept_prob=0.98,
    dense_mass=True,
    max_tree_depth=10,
    sigma_b_scale=0.1,
    fixed_ell_b=0.3,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    shared_atmosphere=False,
    normalization_mode="surface_scale",
    column_mode="clear_cloud",
    fixed_cloud_delta=1.0,
    pressure_derivative_step=0.025,
    zero_mean_pressure_map=False,
    progress_bar=True,
):
    """Run joint multi-chip grid-based Milestone 2-4 retrieval."""

    from numpyro.infer import MCMC
    from doraex.inference.numpyro_models import (
        joint_free_t0_cloud_two_column_doppler_model,
    )

    data = jnp.asarray(np.stack([chip.flux for chip in chip_data_list], axis=0))
    obs_times = jnp.asarray(chip_data_list[0].obs_times)
    wavelengths = jnp.asarray(np.stack([chip.wavelengths for chip in chip_data_list], axis=0))
    t0_grid = jnp.asarray(t0_grid)
    alpha_grid = None if alpha_grid is None else jnp.asarray(alpha_grid)
    log_p_cloud_grid = jnp.asarray(log_p_cloud_grid)
    zeta_vmr_grid = None if zeta_vmr_grid is None else jnp.asarray(zeta_vmr_grid)
    clear_profile_grid = jnp.asarray(clear_profile_grid)
    cloudy_profile_grid = jnp.asarray(cloudy_profile_grid)
    if column_mode not in ("clear_cloud", "double_cloud", "pressure_perturbation"):
        raise ValueError(
            "column_mode must be 'clear_cloud', 'double_cloud', or "
            "'pressure_perturbation'"
        )
    if column_mode == "double_cloud":
        log_p_grid_np = np.asarray(log_p_cloud_grid)
        grid_min = float(np.min(log_p_grid_np))
        grid_max = float(np.max(log_p_grid_np))
        deep_bound = log_p_cloud_bounds[1] + 0.5 * fixed_cloud_delta
        high_bound = log_p_cloud_bounds[0] - 0.5 * fixed_cloud_delta
        if high_bound < grid_min or deep_bound > grid_max:
            raise ValueError(
                "double-cloud endpoints must be covered by log_p_cloud_grid: "
                f"mid_bounds={log_p_cloud_bounds}, fixed_cloud_delta={fixed_cloud_delta}, "
                f"grid=({grid_min}, {grid_max})"
            )
    if column_mode == "pressure_perturbation":
        log_p_grid_np = np.asarray(log_p_cloud_grid)
        grid_min = float(np.min(log_p_grid_np))
        grid_max = float(np.max(log_p_grid_np))
        lower_bound = log_p_cloud_bounds[0] - pressure_derivative_step
        upper_bound = log_p_cloud_bounds[1] + pressure_derivative_step
        if lower_bound < grid_min or upper_bound > grid_max:
            raise ValueError(
                "pressure-perturbation finite-difference points must be covered "
                "by log_p_cloud_grid: "
                f"log_p_bounds={log_p_cloud_bounds}, "
                f"pressure_derivative_step={pressure_derivative_step}, "
                f"grid=({grid_min}, {grid_max})"
            )

    def model(data_observed):
        return joint_free_t0_cloud_two_column_doppler_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            alpha_grid=alpha_grid,
            zeta_vmr_grid=zeta_vmr_grid,
            period_mode=period_mode,
            fixed_period=fixed_period,
            t0_bounds=t0_bounds,
            alpha_bounds=alpha_bounds,
            log_p_cloud_bounds=log_p_cloud_bounds,
            zeta_vmr_bounds=zeta_vmr_bounds,
            sigma_b_scale=sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=fix_geometry,
            fixed_cosi=fixed_cosi,
            fixed_v=fixed_v,
            fixed_q1=fixed_q1,
            fixed_q2=fixed_q2,
            shared_atmosphere=shared_atmosphere,
            normalization_mode=normalization_mode,
            column_mode=column_mode,
            fixed_cloud_delta=fixed_cloud_delta,
            pressure_derivative_step=pressure_derivative_step,
            zero_mean_pressure_map=zero_mean_pressure_map,
        )

    kernel = make_joint_free_t0_cloud_nuts_kernel(
        model,
        n_chip=len(chip_data_list),
        n_phase=chip_data_list[0].flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_log_p_cloud=init_log_p_cloud,
        init_t0=init_t0,
        init_alpha=init_alpha,
        init_zeta_vmr=init_zeta_vmr,
        shared_atmosphere=shared_atmosphere,
        normalization_mode=normalization_mode,
        column_mode=column_mode,
        sample_zeta_vmr=zeta_vmr_grid is not None,
        sample_alpha=alpha_grid is not None,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return mcmc


def save_fixed_two_column_samples(
    output_path,
    samples,
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    period_mode,
    sigma_b_scale=None,
    fixed_ell_b=None,
    fix_geometry=False,
):
    """Save Milestone 2-1 samples and fixed profile metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray(chip_data.wavelengths),
            "obs_times": np.asarray(chip_data.obs_times),
            "clear_profile": np.asarray(clear_profile),
            "cloudy_profile": np.asarray(cloudy_profile),
            "chip_index": np.asarray(chip_data.chip_index),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
            "sigma_b_scale": np.asarray(
                np.nan if sigma_b_scale is None else sigma_b_scale
            ),
            "fixed_ell_b": np.asarray(np.nan if fixed_ell_b is None else fixed_ell_b),
            "fix_geometry": np.asarray(fix_geometry),
        }
    )
    np.savez(output_path, **save_data)


def save_free_cloud_two_column_samples(
    output_path,
    samples,
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    period_mode,
    log_p_cloud_bounds,
    sigma_b_scale=None,
    fixed_ell_b=None,
    fix_geometry=True,
):
    """Save Milestone 2-2a samples and cloudy-profile grid metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray(chip_data.wavelengths),
            "obs_times": np.asarray(chip_data.obs_times),
            "clear_profile": np.asarray(clear_profile),
            "log_p_cloud_grid": np.asarray(log_p_cloud_grid),
            "cloudy_profile_grid": np.asarray(cloudy_profile_grid),
            "chip_index": np.asarray(chip_data.chip_index),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
            "log_p_cloud_bounds": np.asarray(log_p_cloud_bounds),
            "sigma_b_scale": np.asarray(
                np.nan if sigma_b_scale is None else sigma_b_scale
            ),
            "fixed_ell_b": np.asarray(np.nan if fixed_ell_b is None else fixed_ell_b),
            "fix_geometry": np.asarray(fix_geometry),
        }
    )
    np.savez(output_path, **save_data)


def save_free_t0_cloud_two_column_samples(
    output_path,
    samples,
    chip_data,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    period_mode,
    t0_bounds,
    log_p_cloud_bounds,
    zeta_vmr_grid=None,
    zeta_vmr_bounds=(-0.5, 0.5),
    sigma_b_scale=None,
    fixed_ell_b=None,
    fix_geometry=True,
):
    """Save grid-based T0/cloud samples and profile metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray(chip_data.wavelengths),
            "obs_times": np.asarray(chip_data.obs_times),
            "t0_grid": np.asarray(t0_grid),
            "log_p_cloud_grid": np.asarray(log_p_cloud_grid),
            "clear_profile_grid": np.asarray(clear_profile_grid),
            "cloudy_profile_grid": np.asarray(cloudy_profile_grid),
            "chip_index": np.asarray(chip_data.chip_index),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
            "t0_bounds": np.asarray(t0_bounds),
            "log_p_cloud_bounds": np.asarray(log_p_cloud_bounds),
            "zeta_vmr_bounds": np.asarray(zeta_vmr_bounds),
            "sigma_b_scale": np.asarray(
                np.nan if sigma_b_scale is None else sigma_b_scale
            ),
            "fixed_ell_b": np.asarray(np.nan if fixed_ell_b is None else fixed_ell_b),
            "fix_geometry": np.asarray(fix_geometry),
        }
    )
    if zeta_vmr_grid is not None:
        save_data["zeta_vmr_grid"] = np.asarray(zeta_vmr_grid)
    np.savez(output_path, **save_data)


def save_joint_free_t0_cloud_two_column_samples(
    output_path,
    samples,
    chip_data_list,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    period_mode,
    t0_bounds,
    log_p_cloud_bounds,
    alpha_grid=None,
    alpha_bounds=(0.05, 0.20),
    zeta_vmr_grid=None,
    zeta_vmr_bounds=(-0.5, 0.5),
    sigma_b_scale=None,
    fixed_ell_b=None,
    fix_geometry=True,
    shared_atmosphere=False,
    normalization_mode="surface_scale",
    column_mode="clear_cloud",
    fixed_cloud_delta=1.0,
    pressure_derivative_step=0.025,
    zero_mean_pressure_map=False,
):
    """Save joint multi-chip T0/cloud samples and profile metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray([chip.wavelengths for chip in chip_data_list]),
            "obs_times": np.asarray(chip_data_list[0].obs_times),
            "t0_grid": np.asarray(t0_grid),
            "log_p_cloud_grid": np.asarray(log_p_cloud_grid),
            "clear_profile_grid": np.asarray(clear_profile_grid),
            "cloudy_profile_grid": np.asarray(cloudy_profile_grid),
            "chip_indices": np.asarray([chip.chip_index for chip in chip_data_list]),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
            "t0_bounds": np.asarray(t0_bounds),
            "alpha_bounds": np.asarray(alpha_bounds),
            "log_p_cloud_bounds": np.asarray(log_p_cloud_bounds),
            "zeta_vmr_bounds": np.asarray(zeta_vmr_bounds),
            "sigma_b_scale": np.asarray(
                np.nan if sigma_b_scale is None else sigma_b_scale
            ),
            "fixed_ell_b": np.asarray(np.nan if fixed_ell_b is None else fixed_ell_b),
            "fix_geometry": np.asarray(fix_geometry),
            "shared_atmosphere": np.asarray(shared_atmosphere),
            "normalization_mode": np.asarray(normalization_mode),
            "column_mode": np.asarray(column_mode),
            "fixed_cloud_delta": np.asarray(fixed_cloud_delta),
            "pressure_derivative_step": np.asarray(pressure_derivative_step),
            "zero_mean_pressure_map": np.asarray(zero_mean_pressure_map),
        }
    )
    if column_mode == "double_cloud":
        save_data["log_p_mid_bounds"] = np.asarray(log_p_cloud_bounds)
    if alpha_grid is not None:
        save_data["alpha_grid"] = np.asarray(alpha_grid)
    if zeta_vmr_grid is not None:
        save_data["zeta_vmr_grid"] = np.asarray(zeta_vmr_grid)
    np.savez(output_path, **save_data)


def save_synthetic_smoke_profiles(path, wavelengths):
    """Save synthetic two-column profiles for reproducible smoke runs."""

    clear_profile, cloudy_profile = synthetic_two_column_profiles(wavelengths)
    save_two_column_profiles(
        path,
        wavelengths,
        clear_profile,
        cloudy_profile,
        metadata={"profile_source": "synthetic_smoke"},
    )
    return clear_profile, cloudy_profile


def save_synthetic_smoke_cloud_profile_grid(path, wavelengths, log_p_cloud_grid=None):
    """Save synthetic clear/cloudy grid profiles for smoke runs."""

    if log_p_cloud_grid is None:
        log_p_cloud_grid = np.linspace(0.0, 2.0, 5)
    clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
        wavelengths,
        log_p_cloud_grid,
    )
    save_cloud_profile_grid(
        path,
        wavelengths,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        metadata={"profile_source": "synthetic_smoke_cloud_grid"},
    )
    return clear_profile, np.asarray(log_p_cloud_grid), cloudy_profile_grid


def _fixed_sample_at(samples, index):
    sample_names = {
        "cosi",
        "v",
        "q1",
        "q2",
        "u1",
        "u2",
        "log_w",
        "f_cloud",
        "surface_scale",
        "A",
        "sigma_d",
        "sigma_b",
        "sigma_log_p",
        "ell_b",
        "P",
        "log_p_cloud",
        "log_p_mid",
        "T0",
        "alpha",
        "zeta_vmr",
        "h_high",
    }
    result = {
        name: jnp.asarray(samples[name])[index]
        for name in sample_names
        if name in samples
    }
    if "fixed_cloud_delta" in samples:
        result["fixed_cloud_delta"] = jnp.asarray(samples["fixed_cloud_delta"])
    if "pressure_derivative_step" in samples:
        result["pressure_derivative_step"] = jnp.asarray(
            samples["pressure_derivative_step"]
        )
    if "zero_mean_pressure_map" in samples:
        result["zero_mean_pressure_map"] = jnp.asarray(
            samples["zero_mean_pressure_map"]
        )
    return result


def two_column_operator_from_sample(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    sample,
):
    """Build ``(baseline, W_delta)`` for one Milestone 2-1 sample."""

    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    u1 = jnp.asarray(sample.get("u1", kipping_q_to_u(sample["q1"], sample["q2"])[0]))
    u2 = jnp.asarray(sample.get("u2", kipping_q_to_u(sample["q1"], sample["q2"])[1]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    baseline, contrast_matrix = two_column_operator_from_times(
        geometry.theta,
        geometry.phi,
        jnp.asarray(sample["v"]),
        inclination,
        u1,
        u2,
        jnp.asarray(chip_data.obs_times),
        jnp.asarray(sample["P"]),
        jnp.asarray(chip_data.wavelengths),
        jnp.asarray(clear_profile),
        jnp.asarray(cloudy_profile),
        jnp.asarray(sample["h_high"] if "h_high" in sample else sample["f_cloud"]),
        weights=weights,
    )
    if "A" in sample:
        norm = jnp.asarray(sample["A"]) * jnp.mean(baseline)
        return baseline / norm, contrast_matrix / norm
    surface_scale = jnp.asarray(sample.get("surface_scale", 1.0))
    return surface_scale * baseline, surface_scale * contrast_matrix


def linear_profile_operator_from_sample(
    chip_data,
    geometry,
    base_profile,
    contrast_profile,
    sample,
):
    """Build ``(baseline, W_delta)`` for a linear local spectral response."""

    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    u1 = jnp.asarray(sample.get("u1", kipping_q_to_u(sample["q1"], sample["q2"])[0]))
    u2 = jnp.asarray(sample.get("u2", kipping_q_to_u(sample["q1"], sample["q2"])[1]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    baseline, contrast_matrix = linear_profile_operator_from_times(
        geometry.theta,
        geometry.phi,
        jnp.asarray(sample["v"]),
        inclination,
        u1,
        u2,
        jnp.asarray(chip_data.obs_times),
        jnp.asarray(sample["P"]),
        jnp.asarray(chip_data.wavelengths),
        jnp.asarray(base_profile),
        jnp.asarray(contrast_profile),
        weights=weights,
    )
    if "A" in sample:
        norm = jnp.asarray(sample["A"]) * jnp.mean(baseline)
        return baseline / norm, contrast_matrix / norm
    surface_scale = jnp.asarray(sample.get("surface_scale", 1.0))
    return surface_scale * baseline, surface_scale * contrast_matrix


def two_column_operator_from_free_cloud_sample(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    sample,
):
    """Build ``(baseline, W_delta)`` for one Milestone 2-2a sample."""

    cloudy_profile = _interpolate_profile_grid(
        log_p_cloud_grid,
        cloudy_profile_grid,
        sample["log_p_cloud"],
    )
    return two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )


def two_column_operator_from_free_t0_cloud_sample(
    chip_data,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    sample,
    alpha_grid=None,
    zeta_vmr_grid=None,
):
    """Build ``(baseline, W_delta)`` for one grid-based T0/cloud sample."""

    pressure_perturbation = "sigma_log_p" in sample
    double_cloud = "h_high" in sample
    log_p_name = "log_p_mid" if double_cloud else "log_p_cloud"
    if alpha_grid is not None:
        clear_profile = _interpolate_profile_grid_3d(
            t0_grid,
            alpha_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            sample["T0"],
            sample["alpha"],
            sample["zeta_vmr"],
        )
        if pressure_perturbation:
            pressure_derivative_step = float(
                np.asarray(sample.get("pressure_derivative_step", 0.025))
            )
            base_profile = _interpolate_profile_grid_4d(
                t0_grid,
                alpha_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample["alpha"],
                sample[log_p_name],
                sample["zeta_vmr"],
            )
            deeper_profile = _interpolate_profile_grid_4d(
                t0_grid,
                alpha_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample["alpha"],
                sample[log_p_name] + pressure_derivative_step,
                sample["zeta_vmr"],
            )
            higher_profile = _interpolate_profile_grid_4d(
                t0_grid,
                alpha_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample["alpha"],
                sample[log_p_name] - pressure_derivative_step,
                sample["zeta_vmr"],
            )
            contrast_profile = (
                deeper_profile - higher_profile
            ) / (2.0 * pressure_derivative_step)
        elif double_cloud:
            fixed_cloud_delta = float(np.asarray(sample.get("fixed_cloud_delta", 1.0)))
            clear_profile = _interpolate_profile_grid_4d(
                t0_grid,
                alpha_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample["alpha"],
                sample[log_p_name] + 0.5 * fixed_cloud_delta,
                sample["zeta_vmr"],
            )
            cloudy_profile = _interpolate_profile_grid_4d(
                t0_grid,
                alpha_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample["alpha"],
                sample[log_p_name] - 0.5 * fixed_cloud_delta,
                sample["zeta_vmr"],
            )
            base_profile = None
            contrast_profile = None
        else:
            cloudy_profile = _interpolate_profile_grid_4d(
                t0_grid,
                alpha_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample["alpha"],
                sample[log_p_name],
                sample["zeta_vmr"],
            )
            base_profile = None
            contrast_profile = None
    elif zeta_vmr_grid is None:
        clear_profile = _interpolate_profile_grid(
            t0_grid,
            clear_profile_grid,
            sample["T0"],
        )
        if pressure_perturbation:
            pressure_derivative_step = float(
                np.asarray(sample.get("pressure_derivative_step", 0.025))
            )
            base_profile = _interpolate_profile_grid_2d(
                t0_grid,
                log_p_cloud_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name],
            )
            deeper_profile = _interpolate_profile_grid_2d(
                t0_grid,
                log_p_cloud_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] + pressure_derivative_step,
            )
            higher_profile = _interpolate_profile_grid_2d(
                t0_grid,
                log_p_cloud_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] - pressure_derivative_step,
            )
            contrast_profile = (
                deeper_profile - higher_profile
            ) / (2.0 * pressure_derivative_step)
        elif double_cloud:
            fixed_cloud_delta = float(np.asarray(sample.get("fixed_cloud_delta", 1.0)))
            clear_profile = _interpolate_profile_grid_2d(
                t0_grid,
                log_p_cloud_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] + 0.5 * fixed_cloud_delta,
            )
            cloudy_profile = _interpolate_profile_grid_2d(
                t0_grid,
                log_p_cloud_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] - 0.5 * fixed_cloud_delta,
            )
            base_profile = None
            contrast_profile = None
        else:
            cloudy_profile = _interpolate_profile_grid_2d(
                t0_grid,
                log_p_cloud_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name],
            )
            base_profile = None
            contrast_profile = None
    else:
        clear_profile = _interpolate_profile_grid_2d(
            t0_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            sample["T0"],
            sample["zeta_vmr"],
        )
        if pressure_perturbation:
            pressure_derivative_step = float(
                np.asarray(sample.get("pressure_derivative_step", 0.025))
            )
            base_profile = _interpolate_profile_grid_3d(
                t0_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name],
                sample["zeta_vmr"],
            )
            deeper_profile = _interpolate_profile_grid_3d(
                t0_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] + pressure_derivative_step,
                sample["zeta_vmr"],
            )
            higher_profile = _interpolate_profile_grid_3d(
                t0_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] - pressure_derivative_step,
                sample["zeta_vmr"],
            )
            contrast_profile = (
                deeper_profile - higher_profile
            ) / (2.0 * pressure_derivative_step)
        elif double_cloud:
            fixed_cloud_delta = float(np.asarray(sample.get("fixed_cloud_delta", 1.0)))
            clear_profile = _interpolate_profile_grid_3d(
                t0_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] + 0.5 * fixed_cloud_delta,
                sample["zeta_vmr"],
            )
            cloudy_profile = _interpolate_profile_grid_3d(
                t0_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name] - 0.5 * fixed_cloud_delta,
                sample["zeta_vmr"],
            )
            base_profile = None
            contrast_profile = None
        else:
            cloudy_profile = _interpolate_profile_grid_3d(
                t0_grid,
                log_p_cloud_grid,
                zeta_vmr_grid,
                cloudy_profile_grid,
                sample["T0"],
                sample[log_p_name],
                sample["zeta_vmr"],
            )
            base_profile = None
            contrast_profile = None
    if pressure_perturbation:
        return linear_profile_operator_from_sample(
            chip_data,
            geometry,
            base_profile,
            contrast_profile,
            sample,
        )
    return two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )


def _chip_sample(sample, chip_position):
    """Return a view of chip-specific sample entries for one chip."""

    result = dict(sample)
    for name in (
        "T0",
        "alpha",
        "log_p_cloud",
        "log_p_mid",
        "f_cloud",
        "h_high",
        "surface_scale",
        "zeta_vmr",
        "A",
        "sigma_d",
        "log_w",
    ):
        if name in result:
            value = jnp.asarray(result[name])
            result[name] = value if value.ndim == 0 else value[chip_position]
    return result


def joint_operators_from_free_t0_cloud_sample(
    chip_data_list,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    sample,
    alpha_grid=None,
    zeta_vmr_grid=None,
):
    """Build concatenated joint baseline and contrast matrix for one sample."""

    baselines = []
    contrast_matrices = []
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_sample = _chip_sample(sample, chip_position)
        baseline, contrast_matrix = two_column_operator_from_free_t0_cloud_sample(
            chip_data,
            geometry,
            t0_grid[chip_position],
            log_p_cloud_grid[chip_position],
            clear_profile_grid[chip_position],
            cloudy_profile_grid[chip_position],
            chip_sample,
            None if alpha_grid is None else alpha_grid[chip_position],
            None if zeta_vmr_grid is None else zeta_vmr_grid[chip_position],
        )
        baselines.append(baseline)
        contrast_matrices.append(contrast_matrix)
    return jnp.concatenate(baselines, axis=0), jnp.concatenate(
        contrast_matrices,
        axis=0,
    )


def conditional_contrast_map_for_sample(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    sample,
    gp_jitter=1.0e-6,
):
    """Compute conditional cloud-contrast posterior for one sample."""

    baseline, contrast_matrix = two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )
    prior_covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(sample["sigma_b"]),
        jnp.asarray(sample["ell_b"]),
    )
    zero_mean_pressure_map = bool(np.asarray(sample.get("zero_mean_pressure_map", False)))
    if zero_mean_pressure_map:
        prior_covariance = project_zero_mean_covariance(prior_covariance)
    prior_covariance = add_diagonal_jitter(prior_covariance, jitter=gp_jitter)
    prior_mean = jnp.zeros(contrast_matrix.shape[1])
    noise_variance = (
        jnp.asarray(sample["sigma_d"]) ** 2 * jnp.ones(contrast_matrix.shape[0])
        + 1.0e-6
    )
    posterior_mean, posterior_covariance = conditional_map_posterior(
        jnp.asarray(chip_data.flux).reshape(-1) - baseline,
        contrast_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )
    if zero_mean_pressure_map:
        posterior_mean = posterior_mean - jnp.mean(posterior_mean)
        posterior_covariance = project_zero_mean_covariance(posterior_covariance)
    return posterior_mean, posterior_covariance


def conditional_contrast_map_for_free_cloud_sample(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    sample,
    gp_jitter=1.0e-6,
):
    """Compute conditional cloud-contrast posterior for one M2-2a sample."""

    baseline, contrast_matrix = two_column_operator_from_free_cloud_sample(
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        sample,
    )
    prior_covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(sample["sigma_b"]),
        jnp.asarray(sample["ell_b"]),
    )
    zero_mean_pressure_map = bool(np.asarray(sample.get("zero_mean_pressure_map", False)))
    if zero_mean_pressure_map:
        prior_covariance = project_zero_mean_covariance(prior_covariance)
    prior_covariance = add_diagonal_jitter(prior_covariance, jitter=gp_jitter)
    prior_mean = jnp.zeros(contrast_matrix.shape[1])
    noise_variance = (
        jnp.asarray(sample["sigma_d"]) ** 2 * jnp.ones(contrast_matrix.shape[0])
        + 1.0e-6
    )
    posterior_mean, posterior_covariance = conditional_map_posterior(
        jnp.asarray(chip_data.flux).reshape(-1) - baseline,
        contrast_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )
    if zero_mean_pressure_map:
        posterior_mean = posterior_mean - jnp.mean(posterior_mean)
        posterior_covariance = project_zero_mean_covariance(posterior_covariance)
    return posterior_mean, posterior_covariance


def conditional_contrast_map_for_free_t0_cloud_sample(
    chip_data,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    sample,
    alpha_grid=None,
    zeta_vmr_grid=None,
    gp_jitter=1.0e-6,
):
    """Compute conditional cloud-contrast posterior for one T0/cloud sample."""

    baseline, contrast_matrix = two_column_operator_from_free_t0_cloud_sample(
        chip_data,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        sample,
        alpha_grid,
        zeta_vmr_grid,
    )
    prior_covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(sample["sigma_b"]),
        jnp.asarray(sample["ell_b"]),
    )
    zero_mean_pressure_map = bool(np.asarray(sample.get("zero_mean_pressure_map", False)))
    if zero_mean_pressure_map:
        prior_covariance = project_zero_mean_covariance(prior_covariance)
    prior_covariance = add_diagonal_jitter(prior_covariance, jitter=gp_jitter)
    prior_mean = jnp.zeros(contrast_matrix.shape[1])
    noise_variance = (
        jnp.asarray(sample["sigma_d"]) ** 2 * jnp.ones(contrast_matrix.shape[0])
        + 1.0e-6
    )
    posterior_mean, posterior_covariance = conditional_map_posterior(
        jnp.asarray(chip_data.flux).reshape(-1) - baseline,
        contrast_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )
    if zero_mean_pressure_map:
        posterior_mean = posterior_mean - jnp.mean(posterior_mean)
        posterior_covariance = project_zero_mean_covariance(posterior_covariance)
    return posterior_mean, posterior_covariance


def conditional_contrast_map_for_joint_free_t0_cloud_sample(
    chip_data_list,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    sample,
    alpha_grid=None,
    zeta_vmr_grid=None,
    gp_jitter=1.0e-6,
):
    """Compute conditional shared contrast-map posterior for one joint sample."""

    baseline, contrast_matrix = joint_operators_from_free_t0_cloud_sample(
        chip_data_list,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        sample,
        alpha_grid,
        zeta_vmr_grid,
    )
    prior_covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(sample["sigma_b"]),
        jnp.asarray(sample["ell_b"]),
    )
    zero_mean_pressure_map = bool(np.asarray(sample.get("zero_mean_pressure_map", False)))
    if zero_mean_pressure_map:
        prior_covariance = project_zero_mean_covariance(prior_covariance)
    prior_covariance = add_diagonal_jitter(prior_covariance, jitter=gp_jitter)
    prior_mean = jnp.zeros(contrast_matrix.shape[1])
    residual = jnp.concatenate(
        [jnp.asarray(chip.flux).reshape(-1) for chip in chip_data_list],
        axis=0,
    ) - baseline
    noise_variance = jnp.concatenate(
        [
            jnp.asarray(sample["sigma_d"])[chip_position] ** 2
            * jnp.ones(chip.flux.size)
            + 1.0e-6
            for chip_position, chip in enumerate(chip_data_list)
        ],
        axis=0,
    )
    posterior_mean, posterior_covariance = conditional_map_posterior(
        residual,
        contrast_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )
    if zero_mean_pressure_map:
        posterior_mean = posterior_mean - jnp.mean(posterior_mean)
        posterior_covariance = project_zero_mean_covariance(posterior_covariance)
    return posterior_mean, posterior_covariance


def compute_contrast_map_moments(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    samples,
    sample_indices=None,
):
    """Compute posterior moments for the cloud-fraction contrast map."""

    sample_count = len(np.asarray(samples["f_cloud"]))
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    f_cloud_values = []
    for index in sample_indices:
        sample = _fixed_sample_at(samples, int(index))
        mean, covariance = conditional_contrast_map_for_sample(
            chip_data,
            geometry,
            clear_profile,
            cloudy_profile,
            sample,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        f_cloud_values.append(jnp.asarray(sample["f_cloud"]))

    mean_stack = jnp.stack(conditional_means, axis=0)
    contrast_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - contrast_mean[None, :]) ** 2, axis=0)
    contrast_variance = within + between

    f_cloud_stack = jnp.stack(f_cloud_values, axis=0)
    cloud_fraction_samples = f_cloud_stack[:, None] + mean_stack
    cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
    cloud_fraction_variance = (
        jnp.mean((cloud_fraction_samples - cloud_fraction_mean[None, :]) ** 2, axis=0)
        + within
    )
    return contrast_mean, contrast_variance, cloud_fraction_mean, cloud_fraction_variance


def compute_free_cloud_contrast_map_moments(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    samples,
    sample_indices=None,
):
    """Compute posterior cloud-map moments for Milestone 2-2a."""

    sample_count = len(np.asarray(samples["f_cloud"]))
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    f_cloud_values = []
    for index in sample_indices:
        sample = _fixed_sample_at(samples, int(index))
        mean, covariance = conditional_contrast_map_for_free_cloud_sample(
            chip_data,
            geometry,
            clear_profile,
            log_p_cloud_grid,
            cloudy_profile_grid,
            sample,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        f_cloud_values.append(jnp.asarray(sample["f_cloud"]))

    mean_stack = jnp.stack(conditional_means, axis=0)
    contrast_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - contrast_mean[None, :]) ** 2, axis=0)
    contrast_variance = within + between

    f_cloud_stack = jnp.stack(f_cloud_values, axis=0)
    cloud_fraction_samples = f_cloud_stack[:, None] + mean_stack
    cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
    cloud_fraction_variance = (
        jnp.mean((cloud_fraction_samples - cloud_fraction_mean[None, :]) ** 2, axis=0)
        + within
    )
    return contrast_mean, contrast_variance, cloud_fraction_mean, cloud_fraction_variance


def compute_free_t0_cloud_contrast_map_moments(
    chip_data,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    samples,
    sample_indices=None,
    alpha_grid=None,
    zeta_vmr_grid=None,
):
    """Compute posterior cloud-map moments for grid-based free T0/cloud runs."""

    sample_count = len(np.asarray(samples["f_cloud"]))
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    f_cloud_values = []
    for index in sample_indices:
        sample = _fixed_sample_at(samples, int(index))
        mean, covariance = conditional_contrast_map_for_free_t0_cloud_sample(
            chip_data,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            sample,
            alpha_grid=alpha_grid,
            zeta_vmr_grid=zeta_vmr_grid,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        f_cloud_values.append(jnp.asarray(sample["f_cloud"]))

    mean_stack = jnp.stack(conditional_means, axis=0)
    contrast_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - contrast_mean[None, :]) ** 2, axis=0)
    contrast_variance = within + between

    f_cloud_stack = jnp.stack(f_cloud_values, axis=0)
    cloud_fraction_samples = f_cloud_stack[:, None] + mean_stack
    cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
    cloud_fraction_variance = (
        jnp.mean((cloud_fraction_samples - cloud_fraction_mean[None, :]) ** 2, axis=0)
        + within
    )
    return contrast_mean, contrast_variance, cloud_fraction_mean, cloud_fraction_variance


def compute_joint_free_t0_cloud_contrast_map_moments(
    chip_data_list,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    samples,
    sample_indices=None,
    alpha_grid=None,
    zeta_vmr_grid=None,
):
    """Compute shared contrast-map moments for joint multi-chip runs."""

    sample_count = len(np.asarray(samples["sigma_b"]))
    uses_fraction_map = "f_cloud" in samples or "h_high" in samples
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    mean_fraction_values = []
    for index in sample_indices:
        sample = _fixed_sample_at(samples, int(index))
        mean, covariance = conditional_contrast_map_for_joint_free_t0_cloud_sample(
            chip_data_list,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            sample,
            alpha_grid=alpha_grid,
            zeta_vmr_grid=zeta_vmr_grid,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        if uses_fraction_map:
            mean_fraction_values.append(
                jnp.asarray(
                    sample["h_high"] if "h_high" in sample else sample["f_cloud"]
                )
            )
        else:
            mean_fraction_values.append(jnp.asarray(0.0))

    mean_stack = jnp.stack(conditional_means, axis=0)
    contrast_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - contrast_mean[None, :]) ** 2, axis=0)
    contrast_variance = within + between

    mean_fraction_stack = jnp.stack(mean_fraction_values, axis=0)
    if mean_fraction_stack.ndim == 1:
        cloud_fraction_samples = mean_fraction_stack[:, None] + mean_stack
        cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
        cloud_fraction_variance = (
            jnp.mean(
                (cloud_fraction_samples - cloud_fraction_mean[None, :]) ** 2,
                axis=0,
            )
            + within
        )
    else:
        cloud_fraction_samples = mean_fraction_stack[:, :, None] + mean_stack[:, None, :]
        cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
        cloud_fraction_variance = (
            jnp.mean(
                (cloud_fraction_samples - cloud_fraction_mean[None, :, :]) ** 2,
                axis=0,
            )
            + within[None, :]
        )
    return contrast_mean, contrast_variance, cloud_fraction_mean, cloud_fraction_variance


def fixed_two_column_median_sample(samples):
    """Return marginal posterior medians for Milestone 2-1 samples."""

    result = {}
    skip_names = {
        "wavelengths",
        "obs_times",
        "clear_profile",
        "cloudy_profile",
        "chip_index",
        "nside",
        "period_mode",
        "sigma_b_scale",
        "fixed_ell_b",
        "fix_geometry",
        "log_p_cloud_grid",
        "cloudy_profile_grid",
        "cloud_pressure_derivative_grid",
        "log_p_cloud_bounds",
        "log_p_mid_bounds",
        "t0_grid",
        "alpha_grid",
        "clear_profile_grid",
        "t0_bounds",
        "alpha_bounds",
        "zeta_vmr_grid",
        "zeta_vmr_bounds",
        "column_mode",
        "normalization_mode",
        "pressure_derivative_step",
        "pressure_derivative_method",
        "zero_mean_pressure_map",
        "standardized_parameter_names",
        "standardized_parameter_centers",
        "standardized_parameter_scales",
        "atmosphere_rotation_slope_names",
        "atmosphere_rotation_slopes",
    }
    for name, values in samples.items():
        array = np.asarray(values)
        if array.ndim == 0 or name in skip_names:
            continue
        if array.dtype.kind not in "biufc":
            continue
        result[name] = jnp.asarray(np.median(array, axis=0))
    if "fixed_cloud_delta" in samples:
        result["fixed_cloud_delta"] = jnp.asarray(samples["fixed_cloud_delta"])
    if "pressure_derivative_step" in samples:
        result["pressure_derivative_step"] = jnp.asarray(
            samples["pressure_derivative_step"]
        )
    if "zero_mean_pressure_map" in samples:
        result["zero_mean_pressure_map"] = jnp.asarray(
            samples["zero_mean_pressure_map"]
        )
    if "u1" not in result or "u2" not in result:
        result["u1"], result["u2"] = kipping_q_to_u(result["q1"], result["q2"])
    return result


def reconstruct_fixed_two_column_timeseries(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    samples,
    contrast_map,
):
    """Reconstruct spectra from median nonlinear parameters and contrast map."""

    sample = fixed_two_column_median_sample(samples)
    baseline, contrast_matrix = two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )
    model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
    return model.reshape(chip_data.flux.shape), sample


def reconstruct_free_cloud_two_column_timeseries(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    samples,
    contrast_map,
):
    """Reconstruct M2-2a spectra from median nonlinear parameters."""

    sample = fixed_two_column_median_sample(samples)
    baseline, contrast_matrix = two_column_operator_from_free_cloud_sample(
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        sample,
    )
    model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
    return model.reshape(chip_data.flux.shape), sample


def reconstruct_free_t0_cloud_two_column_timeseries(
    chip_data,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    samples,
    contrast_map,
    alpha_grid=None,
    zeta_vmr_grid=None,
):
    """Reconstruct spectra from median T0/cloud nonlinear parameters."""

    sample = fixed_two_column_median_sample(samples)
    baseline, contrast_matrix = two_column_operator_from_free_t0_cloud_sample(
        chip_data,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        sample,
        alpha_grid,
        zeta_vmr_grid,
    )
    model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
    return model.reshape(chip_data.flux.shape), sample


def reconstruct_joint_free_t0_cloud_two_column_timeseries(
    chip_data_list,
    geometry,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    samples,
    contrast_map,
    alpha_grid=None,
    zeta_vmr_grid=None,
):
    """Reconstruct per-chip spectra from median joint parameters."""

    sample = fixed_two_column_median_sample(samples)
    models = []
    chip_samples = []
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_sample = _chip_sample(sample, chip_position)
        baseline, contrast_matrix = two_column_operator_from_free_t0_cloud_sample(
            chip_data,
            geometry,
            t0_grid[chip_position],
            log_p_cloud_grid[chip_position],
            clear_profile_grid[chip_position],
            cloudy_profile_grid[chip_position],
            chip_sample,
            None if alpha_grid is None else alpha_grid[chip_position],
            None if zeta_vmr_grid is None else zeta_vmr_grid[chip_position],
        )
        model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
        models.append(model.reshape(chip_data.flux.shape))
        chip_samples.append(chip_sample)
    return models, sample, chip_samples
