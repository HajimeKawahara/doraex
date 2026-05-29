"""Evaluate the Milestone 4 cloud-pressure linearization error."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402
from doraex.geometry.limb_darkening import quadratic_limb_darkening  # noqa: E402
from doraex.geometry.rotation import (  # noqa: E402
    incline,
    line_of_sight_velocity,
    projected_mu,
    rotate_longitude,
    visible_mask,
)
from doraex.operators.doppler import doppler_factor  # noqa: E402
from doraex.workflows.luhman16b_milestone1 import build_luhman16b_geometry  # noqa: E402


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare exact profile-grid cloud-pressure spectra against the "
            "Milestone 4 first-order pressure perturbation model."
        )
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument(
        "--samples",
        default=str(
            ROOT
            / "results"
            / "milestone4_2"
            / "mcmc_joint_chips_pressure_variation_shared_atmosphere.npz"
        ),
    )
    parser.add_argument(
        "--product-dir",
        default=str(ROOT / "results" / "milestone4_2"),
        help="Directory containing cloud-pressure perturbation map products.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to --product-dir.",
    )
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument(
        "--clip-out-of-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clip per-pixel log pressure values to the profile-grid bounds. "
            "By default, out-of-grid pixels raise an error."
        ),
    )
    parser.add_argument(
        "--save-model-arrays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save exact, linearized, and difference data-space arrays.",
    )
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_chip_data_list(args):
    chip_data_list = []
    for chip_index in args.chip_indices:
        chip_data = load_luhman16b_chip(args.data_dir, chip_index=chip_index)
        if args.smoke_test:
            chip_data = subset_chip_data(
                chip_data,
                wavelength_step=args.smoke_wavelength_step,
                phase_count=args.smoke_phase_count,
            )
        chip_data_list.append(chip_data)
    return chip_data_list


def _posterior_median_sample(samples):
    """Return posterior medians for nonlinear parameters."""

    names = (
        "A",
        "P",
        "T0",
        "alpha",
        "cosi",
        "ell_b",
        "log_p_cloud",
        "log_w",
        "q1",
        "q2",
        "sigma_d",
        "sigma_log_p",
        "u1",
        "u2",
        "v",
        "zeta_vmr",
    )
    sample = {}
    for name in names:
        if name in samples:
            sample[name] = np.median(np.asarray(samples[name]), axis=0)
    if "pressure_derivative_step" in samples:
        sample["pressure_derivative_step"] = float(
            np.asarray(samples["pressure_derivative_step"])
        )
    return sample


def _chip_value(sample, name, chip_position):
    value = np.asarray(sample[name])
    if value.ndim == 0:
        return value.item()
    return value[chip_position]


def _load_pressure_perturbation_map(product_dir, chip_index, chip_position):
    chip_path = product_dir / f"cloud_pressure_perturbation_mean_chip{chip_index}.npy"
    if chip_path.exists():
        return np.asarray(np.load(chip_path), dtype=float)
    joint_path = product_dir / "cloud_pressure_perturbation_mean_joint_by_chip.npy"
    if joint_path.exists():
        return np.asarray(np.load(joint_path)[chip_position], dtype=float)
    raise FileNotFoundError(
        "Pressure perturbation map product was not found. Run "
        "make_milestone2_joint_chip_products.py with the M4 preset first."
    )


def _interpolate_profile_nd(grids, profile_grid, values):
    """Multilinearly interpolate a profile grid at one coordinate."""

    grids = [np.asarray(grid, dtype=float) for grid in grids]
    profile_grid = np.asarray(profile_grid, dtype=float)
    values = [float(value) for value in values]
    lower_indices = []
    fractions = []
    for grid, value in zip(grids, values):
        index = np.searchsorted(grid, value, side="right") - 1
        index = int(np.clip(index, 0, len(grid) - 2))
        lower_indices.append(index)
        denom = grid[index + 1] - grid[index]
        fractions.append((value - grid[index]) / denom)

    result = np.zeros(profile_grid.shape[-1], dtype=float)
    for offsets in product((0, 1), repeat=len(grids)):
        weight = 1.0
        indices = []
        for index, fraction, offset in zip(lower_indices, fractions, offsets):
            weight *= fraction if offset else 1.0 - fraction
            indices.append(index + offset)
        result = result + weight * profile_grid[tuple(indices)]
    return result


def _interpolate_cloud_profiles_for_map(
    t0_grid,
    alpha_grid,
    log_p_cloud_grid,
    zeta_vmr_grid,
    cloudy_profile_grid,
    t0,
    alpha,
    log_p_cloud_values,
    zeta_vmr,
    clip_out_of_grid=False,
):
    """Interpolate cloudy profiles for all per-pixel pressure values."""

    log_p_cloud_values = np.asarray(log_p_cloud_values, dtype=float)
    grid_min = float(np.min(log_p_cloud_grid))
    grid_max = float(np.max(log_p_cloud_grid))
    outside = (log_p_cloud_values < grid_min) | (log_p_cloud_values > grid_max)
    if np.any(outside) and not clip_out_of_grid:
        raise ValueError(
            "Per-pixel log_p_cloud values exceed the profile-grid bounds: "
            f"min={float(np.min(log_p_cloud_values)):.6g}, "
            f"max={float(np.max(log_p_cloud_values)):.6g}, "
            f"grid=({grid_min:.6g}, {grid_max:.6g}). "
            "Use --clip-out-of-grid to measure the clipped-grid error."
        )
    log_p_values = np.clip(log_p_cloud_values, grid_min, grid_max)
    grids = (t0_grid, alpha_grid, log_p_cloud_grid, zeta_vmr_grid)
    profiles = [
        _interpolate_profile_nd(
            grids,
            cloudy_profile_grid,
            (t0, alpha, log_p_value, zeta_vmr),
        )
        for log_p_value in log_p_values
    ]
    return np.asarray(profiles), outside


def _single_cloud_profile(
    t0_grid,
    alpha_grid,
    log_p_cloud_grid,
    zeta_vmr_grid,
    cloudy_profile_grid,
    t0,
    alpha,
    log_p_cloud,
    zeta_vmr,
):
    return _interpolate_profile_nd(
        (t0_grid, alpha_grid, log_p_cloud_grid, zeta_vmr_grid),
        cloudy_profile_grid,
        (t0, alpha, log_p_cloud, zeta_vmr),
    )


def _variable_profile_timeseries(chip_data, geometry, sample, profiles_by_pixel):
    """Doppler-integrate one local spectrum per surface pixel."""

    profiles_by_pixel = jnp.asarray(profiles_by_pixel)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    phases = jnp.asarray(chip_data.obs_times) / jnp.asarray(sample["P"])
    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    u1 = jnp.asarray(sample["u1"])
    u2 = jnp.asarray(sample["u2"])

    def one_phase(phase, phase_weight):
        phi_rot = rotate_longitude(geometry.phi, phase)
        theta_obs, phi_obs = incline(
            geometry.theta,
            phi_rot,
            jnp.pi / 2.0 - inclination,
        )
        vlos = line_of_sight_velocity(
            jnp.asarray(sample["v"]),
            inclination,
            geometry.theta,
            phi_rot,
        )
        factors = doppler_factor(vlos)

        def shift_one(profile, factor):
            return jnp.interp(wavelengths / factor, wavelengths, profile)

        local_profiles = jax.vmap(shift_one)(profiles_by_pixel, factors).T
        mu = projected_mu(theta_obs, phi_obs)
        limb = quadratic_limb_darkening(u1, u2, mu)
        pixel_weight = visible_mask(phi_obs) * mu * limb
        return phase_weight * jnp.sum(local_profiles * pixel_weight[None, :], axis=1)

    return jax.vmap(one_phase)(phases, weights)


def _chip_linearization_metrics(
    chip_data,
    geometry,
    samples,
    median_sample,
    q_map,
    chip_position,
    clip_out_of_grid=False,
):
    """Compute local and data-space linearization errors for one chip."""

    t0_grid = np.asarray(samples["t0_grid"])[chip_position]
    alpha_grid = np.asarray(samples["alpha_grid"])[chip_position]
    log_p_cloud_grid = np.asarray(samples["log_p_cloud_grid"])[chip_position]
    zeta_vmr_grid = np.asarray(samples["zeta_vmr_grid"])[chip_position]
    cloudy_profile_grid = np.asarray(samples["cloudy_profile_grid"])[chip_position]
    chip_sample = {
        "A": float(_chip_value(median_sample, "A", chip_position)),
        "P": float(np.asarray(median_sample["P"])),
        "T0": float(np.asarray(median_sample["T0"])),
        "alpha": float(np.asarray(median_sample["alpha"])),
        "cosi": float(np.asarray(median_sample["cosi"])),
        "log_p_cloud": float(np.asarray(median_sample["log_p_cloud"])),
        "log_w": np.asarray(_chip_value(median_sample, "log_w", chip_position)),
        "sigma_d": float(_chip_value(median_sample, "sigma_d", chip_position)),
        "u1": float(np.asarray(median_sample["u1"])),
        "u2": float(np.asarray(median_sample["u2"])),
        "v": float(np.asarray(median_sample["v"])),
        "zeta_vmr": float(np.asarray(median_sample["zeta_vmr"])),
    }
    derivative_step = float(median_sample["pressure_derivative_step"])
    q_map = np.asarray(q_map, dtype=float)
    log_p_map = chip_sample["log_p_cloud"] + q_map

    base_profile = _single_cloud_profile(
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        cloudy_profile_grid,
        chip_sample["T0"],
        chip_sample["alpha"],
        chip_sample["log_p_cloud"],
        chip_sample["zeta_vmr"],
    )
    deeper_profile = _single_cloud_profile(
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        cloudy_profile_grid,
        chip_sample["T0"],
        chip_sample["alpha"],
        chip_sample["log_p_cloud"] + derivative_step,
        chip_sample["zeta_vmr"],
    )
    higher_profile = _single_cloud_profile(
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        cloudy_profile_grid,
        chip_sample["T0"],
        chip_sample["alpha"],
        chip_sample["log_p_cloud"] - derivative_step,
        chip_sample["zeta_vmr"],
    )
    derivative_profile = (deeper_profile - higher_profile) / (2.0 * derivative_step)
    exact_profiles, outside_grid = _interpolate_cloud_profiles_for_map(
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        cloudy_profile_grid,
        chip_sample["T0"],
        chip_sample["alpha"],
        log_p_map,
        chip_sample["zeta_vmr"],
        clip_out_of_grid=clip_out_of_grid,
    )
    linear_profiles = base_profile[None, :] + q_map[:, None] * derivative_profile[None, :]
    local_delta = exact_profiles - linear_profiles

    base_profiles = np.tile(base_profile[None, :], (q_map.size, 1))
    base_model = _variable_profile_timeseries(chip_data, geometry, chip_sample, base_profiles)
    norm = chip_sample["A"] * jnp.mean(base_model)
    exact_model = _variable_profile_timeseries(
        chip_data,
        geometry,
        chip_sample,
        exact_profiles,
    ) / norm
    linear_model = _variable_profile_timeseries(
        chip_data,
        geometry,
        chip_sample,
        linear_profiles,
    ) / norm
    data_delta = np.asarray(exact_model - linear_model)
    sigma_d = chip_sample["sigma_d"]
    abs_over_sigma = np.abs(data_delta) / sigma_d

    metrics = {
        "chip_index": int(chip_data.chip_index),
        "log_p_cloud_center": chip_sample["log_p_cloud"],
        "sigma_d": sigma_d,
        "q_min": float(np.min(q_map)),
        "q_q01": float(np.quantile(q_map, 0.01)),
        "q_q05": float(np.quantile(q_map, 0.05)),
        "q_median": float(np.median(q_map)),
        "q_q95": float(np.quantile(q_map, 0.95)),
        "q_q99": float(np.quantile(q_map, 0.99)),
        "q_max": float(np.max(q_map)),
        "log_p_cloud_map_min": float(np.min(log_p_map)),
        "log_p_cloud_map_max": float(np.max(log_p_map)),
        "out_of_grid_pixel_fraction": float(np.mean(outside_grid)),
        "local_profile_error_rms": float(np.sqrt(np.mean(local_delta**2))),
        "local_profile_error_abs_max": float(np.max(np.abs(local_delta))),
        "local_profile_error_relative_rms": float(
            np.sqrt(np.mean(local_delta**2)) / np.sqrt(np.mean(exact_profiles**2))
        ),
        "data_error_rms": float(np.sqrt(np.mean(data_delta**2))),
        "data_error_abs_median": float(np.median(np.abs(data_delta))),
        "data_error_abs_p95": float(np.quantile(np.abs(data_delta), 0.95)),
        "data_error_abs_max": float(np.max(np.abs(data_delta))),
        "data_error_rms_over_sigma_d": float(np.sqrt(np.mean(abs_over_sigma**2))),
        "data_error_abs_median_over_sigma_d": float(np.median(abs_over_sigma)),
        "data_error_abs_p95_over_sigma_d": float(np.quantile(abs_over_sigma, 0.95)),
        "data_error_abs_max_over_sigma_d": float(np.max(abs_over_sigma)),
    }
    arrays = {
        "exact_model": np.asarray(exact_model),
        "linear_model": np.asarray(linear_model),
        "data_delta": data_delta,
        "local_delta_rms_by_pixel": np.sqrt(np.mean(local_delta**2, axis=1)),
    }
    return metrics, arrays


def main():
    """Run the linearization-error evaluator."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    product_dir = Path(args.product_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else product_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    chip_data_list = _load_chip_data_list(args)
    geometry = build_luhman16b_geometry(nside=args.nside)
    samples = dict(np.load(args.samples, allow_pickle=False))
    if "sigma_log_p" not in samples:
        raise ValueError("The sample file is not a pressure-perturbation M4 run.")
    if "alpha_grid" not in samples or "zeta_vmr_grid" not in samples:
        raise ValueError("This evaluator requires the alpha/zeta/cloud profile grid.")

    median_sample = _posterior_median_sample(samples)
    chip_metrics = []
    weighted_squared_error = 0.0
    data_count = 0
    for chip_position, chip_data in enumerate(chip_data_list):
        q_map = _load_pressure_perturbation_map(
            product_dir,
            chip_data.chip_index,
            chip_position,
        )
        metrics, arrays = _chip_linearization_metrics(
            chip_data,
            geometry,
            samples,
            median_sample,
            q_map,
            chip_position,
            clip_out_of_grid=args.clip_out_of_grid,
        )
        chip_metrics.append(metrics)
        weighted_squared_error += metrics["data_error_rms_over_sigma_d"] ** 2 * chip_data.flux.size
        data_count += chip_data.flux.size
        if args.save_model_arrays:
            chip_index = chip_data.chip_index
            np.save(
                out_dir / f"pressure_linearization_exact_model_chip{chip_index}.npy",
                arrays["exact_model"],
            )
            np.save(
                out_dir / f"pressure_linearization_linear_model_chip{chip_index}.npy",
                arrays["linear_model"],
            )
            np.save(
                out_dir / f"pressure_linearization_data_delta_chip{chip_index}.npy",
                arrays["data_delta"],
            )
            np.save(
                out_dir / f"pressure_linearization_local_delta_rms_chip{chip_index}.npy",
                arrays["local_delta_rms_by_pixel"],
            )

    diagnostics = {
        "samples": str(Path(args.samples)),
        "product_dir": str(product_dir),
        "chip_indices": [int(chip.chip_index) for chip in chip_data_list],
        "pressure_derivative_step": float(median_sample["pressure_derivative_step"]),
        "log_p_cloud_median": float(np.asarray(median_sample["log_p_cloud"])),
        "sigma_log_p_median": float(np.asarray(median_sample["sigma_log_p"])),
        "aggregate_data_error_rms_over_sigma_d": float(
            np.sqrt(weighted_squared_error / data_count)
        ),
        "chips": chip_metrics,
    }
    output_path = out_dir / "pressure_linearization_diagnostics.json"
    output_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"Pressure linearization diagnostics saved to {output_path}")


if __name__ == "__main__":
    main()
