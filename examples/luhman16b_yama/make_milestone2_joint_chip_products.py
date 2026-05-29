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
    _plot_two_panel_map,
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
    parser.add_argument(
        "--m2-5a",
        action="store_true",
        help="Use Milestone 2-5a shared zeta_vmr atmosphere defaults.",
    )
    parser.add_argument(
        "--m2-5b",
        action="store_true",
        help="Use Milestone 2-5b shared alpha/zeta_vmr atmosphere defaults.",
    )
    parser.add_argument(
        "--m3-1",
        action="store_true",
        help="Use Milestone 3-1 double-cloud shared-atmosphere defaults.",
    )
    parser.add_argument(
        "--m4-1",
        action="store_true",
        help="Use Milestone 4-1 pressure-variation shared-atmosphere defaults.",
    )
    parser.add_argument(
        "--m4-2",
        action="store_true",
        help="Use Milestone 4-2 zero-mean pressure-map shared-atmosphere defaults.",
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
    if (
        args.m2_4b
        or args.m2_4c
        or args.m2_4d
        or args.m2_5a
        or args.m2_5b
        or args.m3_1
        or args.m4_1
        or args.m4_2
    ):
        default_samples = str(
            ROOT / "results" / "milestone2_4a" / "mcmc_joint_chips_free_t0_cloud.npz"
        )
        default_out = str(ROOT / "results" / "milestone2_4a")
        if args.m4_2:
            milestone = "milestone4_2"
        elif args.m4_1:
            milestone = "milestone4_1"
        elif args.m3_1:
            milestone = "milestone3_1"
        elif args.m2_5b:
            milestone = "milestone2_5b"
        elif args.m2_5a:
            milestone = "milestone2_5a"
        elif args.m2_4d:
            milestone = "milestone2_4d"
        elif args.m2_4c:
            milestone = "milestone2_4c"
        else:
            milestone = "milestone2_4b"
        if args.samples == default_samples:
            if args.m4_1 or args.m4_2:
                sample_name = "mcmc_joint_chips_pressure_variation_shared_atmosphere.npz"
            else:
                sample_name = "mcmc_joint_chips_free_t0_cloud_shared_atmosphere.npz"
            args.samples = str(ROOT / "results" / milestone / sample_name)
        if args.out_dir == default_out:
            args.out_dir = str(ROOT / "results" / milestone)
    return args


def _interpolate_3d(x_grid, y_grid, z_grid, profile_grid, x, y, z):
    """Trilinear interpolation helper for T0/log10 Pc/zeta_vmr products."""

    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    z_grid = np.asarray(z_grid)
    profile_grid = np.asarray(profile_grid)
    x_index = np.searchsorted(x_grid, x, side="right") - 1
    y_index = np.searchsorted(y_grid, y, side="right") - 1
    z_index = np.searchsorted(z_grid, z, side="right") - 1
    x_index = np.clip(x_index, 0, len(x_grid) - 2)
    y_index = np.clip(y_index, 0, len(y_grid) - 2)
    z_index = np.clip(z_index, 0, len(z_grid) - 2)
    x_fraction = (x - x_grid[x_index]) / (x_grid[x_index + 1] - x_grid[x_index])
    y_fraction = (y - y_grid[y_index]) / (y_grid[y_index + 1] - y_grid[y_index])
    z_fraction = (z - z_grid[z_index]) / (z_grid[z_index + 1] - z_grid[z_index])
    result = 0.0
    for dx in (0, 1):
        wx = x_fraction if dx else 1.0 - x_fraction
        for dy in (0, 1):
            wy = y_fraction if dy else 1.0 - y_fraction
            for dz in (0, 1):
                wz = z_fraction if dz else 1.0 - z_fraction
                result = result + wx * wy * wz * profile_grid[
                    x_index + dx,
                    y_index + dy,
                    z_index + dz,
                ]
    return result


def _interpolate_4d(w_grid, x_grid, y_grid, z_grid, profile_grid, w, x, y, z):
    """Four-dimensional linear interpolation helper."""

    w_grid = np.asarray(w_grid)
    profile_grid = np.asarray(profile_grid)
    w_index = np.searchsorted(w_grid, w, side="right") - 1
    w_index = np.clip(w_index, 0, len(w_grid) - 2)
    w_fraction = (w - w_grid[w_index]) / (w_grid[w_index + 1] - w_grid[w_index])
    left = _interpolate_3d(
        x_grid,
        y_grid,
        z_grid,
        profile_grid[w_index],
        x,
        y,
        z,
    )
    right = _interpolate_3d(
        x_grid,
        y_grid,
        z_grid,
        profile_grid[w_index + 1],
        x,
        y,
        z,
    )
    return (1.0 - w_fraction) * left + w_fraction * right


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


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return None
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _center_values_by_chip(samples, name, chip_count):
    """Return posterior samples of a shared or chip-specific scalar by chip."""

    values = np.asarray(samples[name], dtype=float)
    if values.ndim == 1:
        return np.tile(values[:, None], (1, chip_count))
    if values.ndim == 2:
        return values
    raise ValueError(f"{name} must be one- or two-dimensional.")


def _plot_log_p_cloud_map(mean_map, std_map, out_path, cmap="viridis"):
    _plot_two_panel_map(
        mean_map,
        std_map,
        "Posterior mean cloud pressure",
        "Posterior std. dev. of cloud pressure",
        "log10 P_cloud [bar]",
        "dex",
        out_path,
        top_cmap=cmap,
    )


def _plot_p_cloud_map(mean_map, std_map, out_path, cmap="viridis"):
    _plot_two_panel_map(
        mean_map,
        std_map,
        "Posterior mean cloud pressure",
        "Posterior std. dev. of cloud pressure",
        "P_cloud [bar]",
        "bar",
        out_path,
        top_cmap=cmap,
    )


def _write_pressure_map_diagnostics(
    path,
    log_p_cloud_mean,
    log_p_cloud_std,
    p_cloud_mean,
    p_cloud_std,
    perturbation_mean,
    perturbation_std,
):
    diagnostics = {
        "log_p_cloud_mean_min": float(np.min(log_p_cloud_mean)),
        "log_p_cloud_mean_max": float(np.max(log_p_cloud_mean)),
        "log_p_cloud_std_min": float(np.min(log_p_cloud_std)),
        "log_p_cloud_std_max": float(np.max(log_p_cloud_std)),
        "p_cloud_mean_min": float(np.min(p_cloud_mean)),
        "p_cloud_mean_max": float(np.max(p_cloud_mean)),
        "p_cloud_std_min": float(np.min(p_cloud_std)),
        "p_cloud_std_max": float(np.max(p_cloud_std)),
        "pressure_perturbation_mean_min": float(np.min(perturbation_mean)),
        "pressure_perturbation_mean_max": float(np.max(perturbation_mean)),
        "pressure_perturbation_std_min": float(np.min(perturbation_std)),
        "pressure_perturbation_std_max": float(np.max(perturbation_std)),
    }
    path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


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
    for name in (
        "T0",
        "alpha",
        "log_p_cloud",
        "log_p_mid",
        "zeta_vmr",
        "f_cloud",
        "h_high",
        "sigma_log_p",
        "sigma_d",
        "surface_scale",
        "A",
    ):
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
            bounds_name = f"{name.lower()}_bounds"
            if bounds_name in samples and values.ndim == 1:
                bounds = np.asarray(samples[bounds_name], dtype=float)
                edge = 0.05 * float(bounds[1] - bounds[0])
                diagnostics[f"{name}_prior_lower"] = float(bounds[0])
                diagnostics[f"{name}_prior_upper"] = float(bounds[1])
                diagnostics[f"{name}_fraction_near_lower_edge"] = float(
                    np.mean(values <= bounds[0] + edge)
                )
                diagnostics[f"{name}_fraction_near_upper_edge"] = float(
                    np.mean(values >= bounds[1] - edge)
                )
    for left in (
        "T0",
        "alpha",
        "log_p_cloud",
        "log_p_mid",
        "zeta_vmr",
        "f_cloud",
        "h_high",
        "sigma_b",
        "sigma_log_p",
    ):
        if left not in samples:
            continue
        left_values = np.asarray(samples[left], dtype=float)
        if left_values.ndim != 1:
            continue
        for right in (
            "T0",
            "alpha",
            "log_p_cloud",
            "log_p_mid",
            "zeta_vmr",
            "f_cloud",
            "h_high",
            "sigma_b",
            "sigma_log_p",
        ):
            if right <= left or right not in samples:
                continue
            right_values = np.asarray(samples[right], dtype=float)
            if right_values.ndim != 1:
                continue
            diagnostics[f"corr_{left}_{right}"] = _safe_correlation(
                left_values,
                right_values,
            )
        for right in ("A", "sigma_d", "surface_scale"):
            if right not in samples:
                continue
            right_values = np.asarray(samples[right], dtype=float)
            if right_values.ndim == 2:
                diagnostics[f"corr_{left}_{right}_by_chip"] = [
                    _safe_correlation(left_values, right_values[:, chip])
                    for chip in range(right_values.shape[1])
                ]
                diagnostics[f"{right}_q05_by_chip"] = [
                    float(value) for value in np.quantile(values, 0.05, axis=0)
                ]
                diagnostics[f"{right}_q95_by_chip"] = [
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
    if "sigma_log_p" in samples and "h_high" not in samples and "f_cloud" not in samples:
        fraction_stem = "cloud_pressure_perturbation"
    else:
        fraction_stem = "high_cloud_fraction" if "h_high" in samples else "cloud_fraction"
    t0_grid = np.asarray(samples["t0_grid"])
    alpha_grid = (
        np.asarray(samples["alpha_grid"]) if "alpha_grid" in samples else None
    )
    log_p_cloud_grid = np.asarray(samples["log_p_cloud_grid"])
    clear_profile_grid = np.asarray(samples["clear_profile_grid"])
    cloudy_profile_grid = np.asarray(samples["cloudy_profile_grid"])
    zeta_vmr_grid = (
        np.asarray(samples["zeta_vmr_grid"]) if "zeta_vmr_grid" in samples else None
    )
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
            alpha_grid=alpha_grid,
            zeta_vmr_grid=zeta_vmr_grid,
        )
    )
    contrast_mean = np.asarray(contrast_mean)
    contrast_var = np.asarray(contrast_var)
    cloud_mean = np.asarray(cloud_mean)
    cloud_var = np.asarray(cloud_var)
    pressure_perturbation_mode = "sigma_log_p" in samples
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
        alpha_grid=alpha_grid,
        zeta_vmr_grid=zeta_vmr_grid,
    )
    residuals = [np.asarray(chip.flux) - np.asarray(model) for chip, model in zip(chip_data_list, models)]

    np.save(out_dir / "contrast_mean_joint.npy", contrast_mean)
    np.save(out_dir / "contrast_var_joint.npy", contrast_var)
    np.save(out_dir / f"{fraction_stem}_mean_joint_by_chip.npy", cloud_mean_by_chip)
    np.save(out_dir / f"{fraction_stem}_var_joint_by_chip.npy", cloud_var_by_chip)
    if pressure_perturbation_mode:
        center_by_chip = _center_values_by_chip(
            samples,
            "log_p_cloud",
            len(chip_data_list),
        )
        center_mean_by_chip = np.mean(center_by_chip, axis=0)
        center_var_by_chip = np.var(center_by_chip, axis=0)
        log_p_cloud_mean_by_chip = (
            center_mean_by_chip[:, None] + cloud_mean_by_chip
        )
        log_p_cloud_var_by_chip = (
            center_var_by_chip[:, None] + cloud_var_by_chip
        )
        p_cloud_mean_by_chip = 10.0**log_p_cloud_mean_by_chip
        p_cloud_std_by_chip = (
            np.log(10.0)
            * p_cloud_mean_by_chip
            * np.sqrt(log_p_cloud_var_by_chip)
        )
        np.save(out_dir / "log_p_cloud_mean_joint_by_chip.npy", log_p_cloud_mean_by_chip)
        np.save(out_dir / "log_p_cloud_var_joint_by_chip.npy", log_p_cloud_var_by_chip)
        np.save(out_dir / "p_cloud_mean_joint_by_chip.npy", p_cloud_mean_by_chip)
        np.save(out_dir / "p_cloud_std_joint_by_chip.npy", p_cloud_std_by_chip)
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_index = chip_data.chip_index
        np.save(
            out_dir / f"{fraction_stem}_mean_chip{chip_index}.npy",
            cloud_mean_by_chip[chip_position],
        )
        np.save(
            out_dir / f"{fraction_stem}_var_chip{chip_index}.npy",
            cloud_var_by_chip[chip_position],
        )
        if pressure_perturbation_mode:
            np.save(
                out_dir / f"log_p_cloud_mean_chip{chip_index}.npy",
                log_p_cloud_mean_by_chip[chip_position],
            )
            np.save(
                out_dir / f"log_p_cloud_var_chip{chip_index}.npy",
                log_p_cloud_var_by_chip[chip_position],
            )
            np.save(
                out_dir / f"p_cloud_mean_chip{chip_index}.npy",
                p_cloud_mean_by_chip[chip_position],
            )
            np.save(
                out_dir / f"p_cloud_std_chip{chip_index}.npy",
                p_cloud_std_by_chip[chip_position],
            )
        np.save(out_dir / f"model_spectrum_chip{chip_index}.npy", np.asarray(models[chip_position]))
        np.save(out_dir / f"residual_chip{chip_index}.npy", residuals[chip_position])

        t0_median = float(np.asarray(chip_samples[chip_position]["T0"]))
        pressure_perturbation = "sigma_log_p" in chip_samples[chip_position]
        double_cloud = "h_high" in chip_samples[chip_position]
        log_p_name = "log_p_mid" if double_cloud else "log_p_cloud"
        log_p_median = float(np.asarray(chip_samples[chip_position][log_p_name]))
        cloud_delta = float(np.asarray(samples.get("fixed_cloud_delta", 1.0)))
        derivative_step = float(np.asarray(samples.get("pressure_derivative_step", 0.025)))
        if alpha_grid is not None:
            alpha_median = float(np.asarray(chip_samples[chip_position]["alpha"]))
            zeta_median = float(np.asarray(chip_samples[chip_position]["zeta_vmr"]))
            clear_median = _interpolate_3d(
                t0_grid[chip_position],
                alpha_grid[chip_position],
                zeta_vmr_grid[chip_position],
                clear_profile_grid[chip_position],
                t0_median,
                alpha_median,
                zeta_median,
            )
            if pressure_perturbation:
                deeper_median = _interpolate_4d(
                    t0_grid[chip_position],
                    alpha_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    alpha_median,
                    log_p_median + derivative_step,
                    zeta_median,
                )
                higher_median = _interpolate_4d(
                    t0_grid[chip_position],
                    alpha_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    alpha_median,
                    log_p_median - derivative_step,
                    zeta_median,
                )
                cloudy_median = clear_median + (
                    deeper_median - higher_median
                ) / (2.0 * derivative_step)
            elif double_cloud:
                clear_median = _interpolate_4d(
                    t0_grid[chip_position],
                    alpha_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    alpha_median,
                    log_p_median + 0.5 * cloud_delta,
                    zeta_median,
                )
                cloudy_median = _interpolate_4d(
                    t0_grid[chip_position],
                    alpha_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    alpha_median,
                    log_p_median - 0.5 * cloud_delta,
                    zeta_median,
                )
            else:
                cloudy_median = _interpolate_4d(
                    t0_grid[chip_position],
                    alpha_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    alpha_median,
                    log_p_median,
                    zeta_median,
                )
        elif zeta_vmr_grid is None:
            clear_median = _interpolate_1d(
                t0_grid[chip_position],
                clear_profile_grid[chip_position],
                t0_median,
            )
            if pressure_perturbation:
                deeper_median = _interpolate_2d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median + derivative_step,
                )
                higher_median = _interpolate_2d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median - derivative_step,
                )
                cloudy_median = clear_median + (
                    deeper_median - higher_median
                ) / (2.0 * derivative_step)
            elif double_cloud:
                clear_median = _interpolate_2d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median + 0.5 * cloud_delta,
                )
                cloudy_median = _interpolate_2d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median - 0.5 * cloud_delta,
                )
            else:
                cloudy_median = _interpolate_2d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median,
                )
        else:
            zeta_median = float(np.asarray(chip_samples[chip_position]["zeta_vmr"]))
            clear_median = _interpolate_2d(
                t0_grid[chip_position],
                zeta_vmr_grid[chip_position],
                clear_profile_grid[chip_position],
                t0_median,
                zeta_median,
            )
            if pressure_perturbation:
                deeper_median = _interpolate_3d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median + derivative_step,
                    zeta_median,
                )
                higher_median = _interpolate_3d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median - derivative_step,
                    zeta_median,
                )
                cloudy_median = clear_median + (
                    deeper_median - higher_median
                ) / (2.0 * derivative_step)
            elif double_cloud:
                clear_median = _interpolate_3d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median + 0.5 * cloud_delta,
                    zeta_median,
                )
                cloudy_median = _interpolate_3d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median - 0.5 * cloud_delta,
                    zeta_median,
                )
            else:
                cloudy_median = _interpolate_3d(
                    t0_grid[chip_position],
                    log_p_cloud_grid[chip_position],
                    zeta_vmr_grid[chip_position],
                    cloudy_profile_grid[chip_position],
                    t0_median,
                    log_p_median,
                    zeta_median,
                )
        delta_scale = float(np.sqrt(np.mean((cloudy_median - clear_median) ** 2)))
        delta_s_mean = contrast_mean * delta_scale
        delta_s_var = contrast_var * delta_scale**2
        np.save(out_dir / f"delta_s_mean_chip{chip_index}.npy", delta_s_mean)
        np.save(out_dir / f"delta_s_var_chip{chip_index}.npy", delta_s_var)
        _plot_cloud_fraction(
            cloud_mean_by_chip[chip_position],
            np.sqrt(cloud_var_by_chip[chip_position]),
            out_dir / f"figure8_{fraction_stem}_chip{chip_index}.png",
            cmap=args.cloud_fraction_cmap,
        )
        if pressure_perturbation_mode:
            _plot_log_p_cloud_map(
                log_p_cloud_mean_by_chip[chip_position],
                np.sqrt(log_p_cloud_var_by_chip[chip_position]),
                out_dir / f"figure8_log_p_cloud_chip{chip_index}.png",
            )
            _plot_p_cloud_map(
                p_cloud_mean_by_chip[chip_position],
                p_cloud_std_by_chip[chip_position],
                out_dir / f"figure8_p_cloud_chip{chip_index}.png",
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
            out_dir / f"{fraction_stem}_diagnostics_chip{chip_index}.json",
            cloud_mean_by_chip[chip_position],
            np.sqrt(cloud_var_by_chip[chip_position]),
            contrast_mean,
        )
        if pressure_perturbation_mode:
            _write_pressure_map_diagnostics(
                out_dir / f"cloud_pressure_map_diagnostics_chip{chip_index}.json",
                log_p_cloud_mean_by_chip[chip_position],
                np.sqrt(log_p_cloud_var_by_chip[chip_position]),
                p_cloud_mean_by_chip[chip_position],
                p_cloud_std_by_chip[chip_position],
                cloud_mean_by_chip[chip_position],
                np.sqrt(cloud_var_by_chip[chip_position]),
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
