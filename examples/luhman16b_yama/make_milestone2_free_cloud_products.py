"""Create Milestone 2-2a free-cloud diagnostic products."""

import argparse
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    compute_free_cloud_contrast_map_moments,
    load_milestone2_free_cloud_inputs,
    reconstruct_free_cloud_two_column_timeseries,
)
from make_milestone2_fixed_products import (  # noqa: E402
    _plot_cloud_fraction,
    _plot_delta_s,
    _plot_figure9,
    _write_cloud_fraction_diagnostics,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build Milestone 2-2a maps and spectral residual diagnostics."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--samples",
        default=str(
            ROOT / "results" / "milestone2_2a" / "mcmc_chip1_fixed_free_cloud.npz"
        ),
    )
    parser.add_argument(
        "--profile-grid",
        default=str(ROOT / "data" / "milestone2_cloud_grid_profiles_chip1.npz"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone2_2a"))
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument(
        "--max-map-samples",
        type=int,
        default=None,
        help="Use at most this many posterior samples for map moments.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _select_sample_indices(sample_count, max_map_samples):
    if max_map_samples is None or max_map_samples >= sample_count:
        return None
    return np.linspace(0, sample_count - 1, max_map_samples, dtype=int)


def _interpolate_profile_grid(parameter_grid, profile_grid, parameter):
    parameter_grid = np.asarray(parameter_grid)
    profile_grid = np.asarray(profile_grid)
    index = np.searchsorted(parameter_grid, parameter, side="right") - 1
    index = np.clip(index, 0, len(parameter_grid) - 2)
    left = parameter_grid[index]
    right = parameter_grid[index + 1]
    fraction = (parameter - left) / (right - left)
    return (1.0 - fraction) * profile_grid[index] + fraction * profile_grid[index + 1]


def main():
    """Compute and save Milestone 2-2a diagnostic products."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_grid_path = None if args.smoke_test else args.profile_grid
    chip_data, geometry, clear_profile, log_p_cloud_grid, cloudy_profile_grid = (
        load_milestone2_free_cloud_inputs(
            args.data_dir,
            profile_grid_path=profile_grid_path,
            chip_index=args.chip_index,
            nside=args.nside,
            smoke_test=args.smoke_test,
            smoke_wavelength_step=args.smoke_wavelength_step,
            smoke_phase_count=args.smoke_phase_count,
        )
    )
    samples = dict(np.load(args.samples, allow_pickle=False))
    sample_indices = _select_sample_indices(
        len(samples["log_p_cloud"]),
        args.max_map_samples,
    )

    contrast_mean, contrast_var, cloud_mean, cloud_var = (
        compute_free_cloud_contrast_map_moments(
            chip_data,
            geometry,
            clear_profile,
            log_p_cloud_grid,
            cloudy_profile_grid,
            samples,
            sample_indices=sample_indices,
        )
    )
    contrast_mean = np.asarray(contrast_mean)
    contrast_var = np.asarray(contrast_var)
    cloud_mean = np.asarray(cloud_mean)
    cloud_var = np.asarray(cloud_var)
    cloud_std = np.sqrt(cloud_var)
    clipped_cloud_mean = np.clip(cloud_mean, 0.0, 1.0)

    model, median_sample = reconstruct_free_cloud_two_column_timeseries(
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        samples,
        contrast_mean,
    )
    model = np.asarray(model)
    residual = chip_data.flux - model
    sigma_d = float(np.asarray(median_sample["sigma_d"]))

    cloudy_median = _interpolate_profile_grid(
        log_p_cloud_grid,
        cloudy_profile_grid,
        float(np.asarray(median_sample["log_p_cloud"])),
    )
    delta_profile = cloudy_median - np.asarray(clear_profile)
    delta_scale = float(np.sqrt(np.mean(delta_profile**2)))
    delta_s_mean = contrast_mean * delta_scale
    delta_s_var = contrast_var * delta_scale**2

    np.save(out_dir / f"contrast_mean_chip{args.chip_index}.npy", contrast_mean)
    np.save(out_dir / f"contrast_var_chip{args.chip_index}.npy", contrast_var)
    np.save(out_dir / f"cloud_fraction_mean_chip{args.chip_index}.npy", cloud_mean)
    np.save(out_dir / f"cloud_fraction_var_chip{args.chip_index}.npy", cloud_var)
    np.save(
        out_dir / f"cloud_fraction_clipped_mean_chip{args.chip_index}.npy",
        clipped_cloud_mean,
    )
    np.save(out_dir / f"delta_s_mean_chip{args.chip_index}.npy", delta_s_mean)
    np.save(out_dir / f"delta_s_var_chip{args.chip_index}.npy", delta_s_var)
    np.save(out_dir / f"model_spectrum_chip{args.chip_index}.npy", model)
    np.save(out_dir / f"residual_chip{args.chip_index}.npy", residual)
    np.savez(
        out_dir / f"posterior_median_parameters_chip{args.chip_index}.npz",
        **{key: np.asarray(value) for key, value in median_sample.items()},
    )

    _plot_cloud_fraction(
        cloud_mean,
        cloud_std,
        out_dir / f"figure8_cloud_fraction_chip{args.chip_index}.png",
    )
    _plot_cloud_fraction(
        clipped_cloud_mean,
        cloud_std,
        out_dir / f"figure8_cloud_fraction_clipped_chip{args.chip_index}.png",
    )
    _plot_delta_s(
        delta_s_mean,
        np.sqrt(delta_s_var),
        out_dir / f"figure8_delta_s_chip{args.chip_index}.png",
    )
    _plot_figure9(
        chip_data.wavelengths,
        chip_data.flux,
        model,
        sigma_d,
        out_dir / f"figure9_free_cloud_chip{args.chip_index}.png",
    )
    _write_cloud_fraction_diagnostics(
        out_dir / f"cloud_fraction_diagnostics_chip{args.chip_index}.json",
        cloud_mean,
        cloud_std,
        contrast_mean,
    )
    print(f"Products saved to {out_dir}")


if __name__ == "__main__":
    main()
