"""Create Milestone 2-2 free-cloud diagnostic products."""

import argparse
import json
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
from chip_paths import cloud_grid_path, free_cloud_sample_path  # noqa: E402

DEFAULT_M22A_OUT = ROOT / "results" / "milestone2_2a"
DEFAULT_M22B_OUT = ROOT / "results" / "milestone2_2b"


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build Milestone 2-2 maps and spectral residual diagnostics."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--samples",
        default=None,
    )
    parser.add_argument(
        "--profile-grid",
        default=None,
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_M22A_OUT))
    parser.add_argument(
        "--m2-2b",
        action="store_true",
        help="Use Milestone 2-2b wide-cloud defaults.",
    )
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
    args = parser.parse_args()
    if args.m2_2b:
        if args.out_dir == str(DEFAULT_M22A_OUT):
            args.out_dir = str(DEFAULT_M22B_OUT)
    if args.samples is None:
        args.samples = str(
            free_cloud_sample_path(
                args.out_dir,
                args.chip_index,
                "fixed",
                wide=args.m2_2b,
            )
        )
    if args.profile_grid is None:
        args.profile_grid = str(cloud_grid_path(args.chip_index, wide=args.m2_2b))
    return args


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


def _safe_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _write_free_cloud_diagnostics(
    path,
    samples,
    cloud_mean,
    cloud_std,
    clipped_cloud_mean,
    log_p_cloud_grid,
):
    log_p_cloud = np.asarray(samples["log_p_cloud"], dtype=float)
    bounds = np.asarray(
        samples.get(
            "log_p_cloud_bounds",
            np.asarray([np.min(log_p_cloud_grid), np.max(log_p_cloud_grid)]),
        ),
        dtype=float,
    )
    lower, upper = float(bounds[0]), float(bounds[1])
    width = upper - lower
    edge_width = 0.05 * width if width > 0.0 else 0.0
    diagnostics = {
        "log_p_cloud_min": float(np.min(log_p_cloud)),
        "log_p_cloud_max": float(np.max(log_p_cloud)),
        "log_p_cloud_median": float(np.median(log_p_cloud)),
        "log_p_cloud_q16": float(np.quantile(log_p_cloud, 0.16)),
        "log_p_cloud_q84": float(np.quantile(log_p_cloud, 0.84)),
        "log_p_cloud_prior_lower": lower,
        "log_p_cloud_prior_upper": upper,
        "fraction_log_p_cloud_near_lower_edge": float(
            np.mean(log_p_cloud <= lower + edge_width)
        ),
        "fraction_log_p_cloud_near_upper_edge": float(
            np.mean(log_p_cloud >= upper - edge_width)
        ),
        "cloud_fraction_mean_min": float(np.min(cloud_mean)),
        "cloud_fraction_mean_max": float(np.max(cloud_mean)),
        "cloud_fraction_std_min": float(np.min(cloud_std)),
        "cloud_fraction_std_max": float(np.max(cloud_std)),
        "fraction_pixels_below_zero": float(np.mean(cloud_mean < 0.0)),
        "fraction_pixels_above_one": float(np.mean(cloud_mean > 1.0)),
        "mean_abs_clipping_shift": float(
            np.mean(np.abs(clipped_cloud_mean - cloud_mean))
        ),
        "max_abs_clipping_shift": float(
            np.max(np.abs(clipped_cloud_mean - cloud_mean))
        ),
        "corr_log_p_cloud_f_cloud": _safe_correlation(
            log_p_cloud,
            samples["f_cloud"],
        ),
        "corr_log_p_cloud_sigma_b": _safe_correlation(
            log_p_cloud,
            samples["sigma_b"],
        ),
        "corr_log_p_cloud_surface_scale": _safe_correlation(
            log_p_cloud,
            samples["surface_scale"],
        ),
        "corr_f_cloud_sigma_b": _safe_correlation(
            samples["f_cloud"],
            samples["sigma_b"],
        ),
        "log_p_cloud_grid_min": float(np.min(log_p_cloud_grid)),
        "log_p_cloud_grid_max": float(np.max(log_p_cloud_grid)),
        "log_p_cloud_grid_count": int(len(log_p_cloud_grid)),
    }
    path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


def main():
    """Compute and save Milestone 2-2 diagnostic products."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_grid_path = None if args.smoke_test else args.profile_grid
    smoke_log_p_cloud_grid = None
    if args.smoke_test and args.m2_2b:
        smoke_log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
    chip_data, geometry, clear_profile, log_p_cloud_grid, cloudy_profile_grid = (
        load_milestone2_free_cloud_inputs(
            args.data_dir,
            profile_grid_path=profile_grid_path,
            chip_index=args.chip_index,
            nside=args.nside,
            smoke_test=args.smoke_test,
            smoke_wavelength_step=args.smoke_wavelength_step,
            smoke_phase_count=args.smoke_phase_count,
            smoke_log_p_cloud_grid=smoke_log_p_cloud_grid,
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
    _write_free_cloud_diagnostics(
        out_dir / f"free_cloud_diagnostics_chip{args.chip_index}.json",
        samples,
        cloud_mean,
        cloud_std,
        clipped_cloud_mean,
        log_p_cloud_grid,
    )
    print(f"Products saved to {out_dir}")


if __name__ == "__main__":
    main()
