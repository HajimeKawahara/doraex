"""Create Milestone 2-4 joint multi-chip diagnostic products."""

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

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402
from doraex.workflows.luhman16b_milestone1 import build_luhman16b_geometry  # noqa: E402
from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    compute_joint_free_t0_cloud_contrast_map_moments,
    reconstruct_joint_free_t0_cloud_two_column_timeseries,
)
from make_milestone2_fixed_products import (  # noqa: E402
    _plot_cloud_fraction,
    _plot_delta_s,
    _plot_figure9,
    _write_cloud_fraction_diagnostics,
)
from make_milestone2_free_t0_cloud_products import (  # noqa: E402
    _interpolate_1d,
    _interpolate_2d,
    _select_sample_indices,
)


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build Milestone 2-4 joint multi-chip products."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument(
        "--samples",
        default=str(ROOT / "results" / "milestone2_4a" / "mcmc_joint_chips_free_t0_cloud.npz"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone2_4a"))
    parser.add_argument(
        "--m2-4b",
        action="store_true",
        help="Use Milestone 2-4b shared-atmosphere defaults.",
    )
    parser.add_argument(
        "--m2-4c",
        action="store_true",
        help="Use Milestone 2-4c ExoMol-consistent shared-atmosphere defaults.",
    )
    parser.add_argument(
        "--m2-4d",
        action="store_true",
        help="Use Milestone 2-4d Yama-normalized shared-atmosphere defaults.",
    )
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--max-map-samples", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument(
        "--cloud-fraction-cmap",
        default="afmhot",
        help="Matplotlib colormap for the cloud-fraction map upper panel.",
    )
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.m2_4b or args.m2_4c or args.m2_4d:
        default_samples = str(
            ROOT / "results" / "milestone2_4a" / "mcmc_joint_chips_free_t0_cloud.npz"
        )
        default_out = str(ROOT / "results" / "milestone2_4a")
        if args.m2_4d:
            milestone = "milestone2_4d"
        elif args.m2_4c:
            milestone = "milestone2_4c"
        else:
            milestone = "milestone2_4b"
        if args.samples == default_samples:
            args.samples = str(
                ROOT
                / "results"
                / milestone
                / "mcmc_joint_chips_free_t0_cloud_shared_atmosphere.npz"
            )
        if args.out_dir == default_out:
            args.out_dir = str(ROOT / "results" / milestone)
    return args


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


def _write_joint_diagnostics(path, samples, residuals, contrast_mean, cloud_fraction_mean):
    cloud_fraction_mean = np.asarray(cloud_fraction_mean)
    if cloud_fraction_mean.ndim == 1:
        cloud_rows = [cloud_fraction_mean]
    else:
        cloud_rows = [row for row in cloud_fraction_mean]
    diagnostics = {
        "chip_indices": [int(value) for value in np.asarray(samples["chip_indices"])],
        "residual_rms_by_chip": [
            float(np.sqrt(np.mean(np.asarray(residual) ** 2))) for residual in residuals
        ],
        "residual_abs_median_by_chip": [
            float(np.median(np.abs(np.asarray(residual)))) for residual in residuals
        ],
        "contrast_mean_min": float(np.min(contrast_mean)),
        "contrast_mean_max": float(np.max(contrast_mean)),
        "contrast_mean_std": float(np.std(contrast_mean)),
        "cloud_fraction_mean_min_by_chip": [
            float(np.min(row)) for row in cloud_rows
        ],
        "cloud_fraction_mean_max_by_chip": [
            float(np.max(row)) for row in cloud_rows
        ],
    }
    for name in ("T0", "log_p_cloud", "f_cloud", "sigma_d", "surface_scale", "A"):
        if name in samples:
            values = np.asarray(samples[name], dtype=float)
            if values.ndim == 1:
                diagnostics[f"{name}_median"] = float(np.median(values))
                diagnostics[f"{name}_q05"] = float(np.quantile(values, 0.05))
                diagnostics[f"{name}_q95"] = float(np.quantile(values, 0.95))
            else:
                diagnostics[f"{name}_median_by_chip"] = [
                    float(value) for value in np.median(values, axis=0)
                ]
                diagnostics[f"{name}_q05_by_chip"] = [
                    float(value) for value in np.quantile(values, 0.05, axis=0)
                ]
                diagnostics[f"{name}_q95_by_chip"] = [
                    float(value) for value in np.quantile(values, 0.95, axis=0)
                ]
    for name in ("sigma_b", "ell_b"):
        if name in samples:
            values = np.asarray(samples[name], dtype=float)
            diagnostics[f"{name}_median"] = float(np.median(values))
            diagnostics[f"{name}_q05"] = float(np.quantile(values, 0.05))
            diagnostics[f"{name}_q95"] = float(np.quantile(values, 0.95))
    path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


def main():
    """Compute and save joint multi-chip diagnostic products."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_data_list = _load_chip_data_list(args)
    geometry = build_luhman16b_geometry(nside=args.nside)
    samples = dict(np.load(args.samples, allow_pickle=False))
    t0_grid = np.asarray(samples["t0_grid"])
    log_p_cloud_grid = np.asarray(samples["log_p_cloud_grid"])
    clear_profile_grid = np.asarray(samples["clear_profile_grid"])
    cloudy_profile_grid = np.asarray(samples["cloudy_profile_grid"])
    sample_indices = _select_sample_indices(len(samples["sigma_b"]), args.max_map_samples)

    contrast_mean, contrast_var, cloud_mean, cloud_var = (
        compute_joint_free_t0_cloud_contrast_map_moments(
            chip_data_list,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            samples,
            sample_indices=sample_indices,
        )
    )
    contrast_mean = np.asarray(contrast_mean)
    contrast_var = np.asarray(contrast_var)
    cloud_mean = np.asarray(cloud_mean)
    cloud_var = np.asarray(cloud_var)
    if cloud_mean.ndim == 1:
        cloud_mean_by_chip = np.tile(cloud_mean[None, :], (len(chip_data_list), 1))
        cloud_var_by_chip = np.tile(cloud_var[None, :], (len(chip_data_list), 1))
    else:
        cloud_mean_by_chip = cloud_mean
        cloud_var_by_chip = cloud_var

    models, median_sample, chip_samples = reconstruct_joint_free_t0_cloud_two_column_timeseries(
        chip_data_list,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        samples,
        contrast_mean,
    )
    residuals = [np.asarray(chip.flux) - np.asarray(model) for chip, model in zip(chip_data_list, models)]

    np.save(out_dir / "contrast_mean_joint.npy", contrast_mean)
    np.save(out_dir / "contrast_var_joint.npy", contrast_var)
    np.save(out_dir / "cloud_fraction_mean_joint_by_chip.npy", cloud_mean_by_chip)
    np.save(out_dir / "cloud_fraction_var_joint_by_chip.npy", cloud_var_by_chip)
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_index = chip_data.chip_index
        np.save(
            out_dir / f"cloud_fraction_mean_chip{chip_index}.npy",
            cloud_mean_by_chip[chip_position],
        )
        np.save(
            out_dir / f"cloud_fraction_var_chip{chip_index}.npy",
            cloud_var_by_chip[chip_position],
        )
        np.save(out_dir / f"model_spectrum_chip{chip_index}.npy", np.asarray(models[chip_position]))
        np.save(out_dir / f"residual_chip{chip_index}.npy", residuals[chip_position])

        t0_median = float(np.asarray(chip_samples[chip_position]["T0"]))
        log_p_median = float(np.asarray(chip_samples[chip_position]["log_p_cloud"]))
        clear_median = _interpolate_1d(
            t0_grid[chip_position],
            clear_profile_grid[chip_position],
            t0_median,
        )
        cloudy_median = _interpolate_2d(
            t0_grid[chip_position],
            log_p_cloud_grid[chip_position],
            cloudy_profile_grid[chip_position],
            t0_median,
            log_p_median,
        )
        delta_scale = float(np.sqrt(np.mean((cloudy_median - clear_median) ** 2)))
        delta_s_mean = contrast_mean * delta_scale
        delta_s_var = contrast_var * delta_scale**2
        np.save(out_dir / f"delta_s_mean_chip{chip_index}.npy", delta_s_mean)
        np.save(out_dir / f"delta_s_var_chip{chip_index}.npy", delta_s_var)
        _plot_cloud_fraction(
            cloud_mean_by_chip[chip_position],
            np.sqrt(cloud_var_by_chip[chip_position]),
            out_dir / f"figure8_cloud_fraction_chip{chip_index}.png",
            cmap=args.cloud_fraction_cmap,
        )
        _plot_delta_s(
            delta_s_mean,
            np.sqrt(delta_s_var),
            out_dir / f"figure8_delta_s_chip{chip_index}.png",
        )
        _plot_figure9(
            chip_data.wavelengths,
            chip_data.flux,
            np.asarray(models[chip_position]),
            float(np.asarray(chip_samples[chip_position]["sigma_d"])),
            out_dir / f"figure9_joint_chip{chip_index}.png",
        )
        _write_cloud_fraction_diagnostics(
            out_dir / f"cloud_fraction_diagnostics_chip{chip_index}.json",
            cloud_mean_by_chip[chip_position],
            np.sqrt(cloud_var_by_chip[chip_position]),
            contrast_mean,
        )

    _plot_delta_s(
        contrast_mean,
        np.sqrt(contrast_var),
        out_dir / "figure8_shared_contrast_joint.png",
    )
    _write_joint_diagnostics(
        out_dir / "joint_chip_diagnostics.json",
        samples,
        residuals,
        contrast_mean,
        cloud_mean_by_chip,
    )
    print(f"Joint products saved to {out_dir}")


if __name__ == "__main__":
    main()
