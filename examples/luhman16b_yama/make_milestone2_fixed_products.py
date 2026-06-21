"""Create Milestone 2-1 fixed two-column diagnostic products."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    compute_contrast_map_moments,
    load_milestone2_fixed_inputs,
    reconstruct_fixed_two_column_timeseries,
)
from chip_paths import fixed_profile_path, fixed_sample_path  # noqa: E402


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build Milestone 2-1 maps and spectral residual diagnostics."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--samples",
        default=None,
    )
    parser.add_argument(
        "--profiles",
        default=None,
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone2_1"))
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
    if args.samples is None:
        args.samples = str(fixed_sample_path(args.out_dir, args.chip_index, "sampled"))
    if args.profiles is None:
        args.profiles = str(fixed_profile_path(args.chip_index))
    return args


def _select_sample_indices(sample_count, max_map_samples):
    if max_map_samples is None or max_map_samples >= sample_count:
        return None
    return np.linspace(0, sample_count - 1, max_map_samples, dtype=int)


def _plot_pixel_fallback(
    top_map,
    bottom_map,
    top_title,
    bottom_title,
    top_unit,
    bottom_unit,
    out_path,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    top_map = np.asarray(top_map)
    bottom_map = np.asarray(bottom_map)
    pixel_index = np.arange(top_map.size)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(pixel_index, top_map, color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel(top_unit)
    axes[0].set_title(f"{top_title} (pixel-order fallback)")
    axes[1].plot(pixel_index, bottom_map, color="tab:orange", linewidth=0.8)
    axes[1].set_xlabel("HEALPix pixel index")
    axes[1].set_ylabel(bottom_unit)
    axes[1].set_title(bottom_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _check_interpolation_as_method(method, interpolation, fname):
    """Compatibility helper removed from newer NumPy private APIs."""

    if interpolation is None:
        return method
    if method != "linear":
        raise TypeError(
            f"{fname} received both method={method!r} and interpolation="
            f"{interpolation!r}."
        )
    return interpolation


def _ensure_numpy_astropy_compat():
    """Provide local compatibility aliases for Astropy with newer NumPy."""

    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    function_base = importlib.import_module("numpy.lib._function_base_impl")
    if not hasattr(function_base, "_check_interpolation_as_method"):
        function_base._check_interpolation_as_method = _check_interpolation_as_method


def _plot_two_panel_map(
    top_map,
    bottom_map,
    top_title,
    bottom_title,
    top_unit,
    bottom_unit,
    out_path,
    top_cmap="afmhot",
    bottom_cmap="viridis",
    colorbar_orientation="horizontal",
    invert_top_colorbar=False,
    invert_bottom_colorbar=False,
    top_colorbar_unit=None,
    bottom_colorbar_unit=None,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    def add_vertical_colorbar(fig, ax, map_values, cmap, unit, invert=False):
        values = np.asarray(map_values)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return
        colorbar = fig.colorbar(
            ScalarMappable(
                norm=Normalize(
                    vmin=float(np.min(finite_values)),
                    vmax=float(np.max(finite_values)),
                ),
                cmap=cmap,
            ),
            ax=ax,
            orientation="vertical",
            fraction=0.035,
            pad=0.02,
        )
        if unit:
            colorbar.set_label(unit)
        if invert:
            colorbar.ax.invert_yaxis()

    try:
        _ensure_numpy_astropy_compat()
        import healpy as hp
    except Exception as exc:
        print(f"healpy map plotting unavailable; using pixel fallback: {exc}")
        _plot_pixel_fallback(
            top_map,
            bottom_map,
            top_title,
            bottom_title,
            top_unit,
            bottom_unit,
            out_path,
        )
        return

    use_vertical_colorbar = colorbar_orientation == "vertical"
    if top_colorbar_unit is None:
        top_colorbar_unit = top_unit
    if bottom_colorbar_unit is None:
        bottom_colorbar_unit = bottom_unit
    fig = plt.figure(
        figsize=(
            9.4 if use_vertical_colorbar else 9,
            8.0 if use_vertical_colorbar else 7.0,
        )
    )
    margins = (0.01, 0.04, 0.0, 0.04) if use_vertical_colorbar else None
    hp.mollview(
        top_map,
        fig=fig.number,
        sub=(2, 1, 1),
        cmap=top_cmap,
        title=top_title,
        unit="" if use_vertical_colorbar else top_unit,
        flip="geo",
        cbar=not use_vertical_colorbar,
        margins=margins,
    )
    hp.graticule()
    if use_vertical_colorbar:
        add_vertical_colorbar(
            fig,
            plt.gca(),
            top_map,
            top_cmap,
            top_colorbar_unit,
            invert=invert_top_colorbar,
        )
    hp.mollview(
        bottom_map,
        fig=fig.number,
        sub=(2, 1, 2),
        cmap=bottom_cmap,
        title=bottom_title,
        unit="" if use_vertical_colorbar else bottom_unit,
        flip="geo",
        cbar=not use_vertical_colorbar,
        margins=margins,
    )
    hp.graticule()
    if use_vertical_colorbar:
        add_vertical_colorbar(
            fig,
            plt.gca(),
            bottom_map,
            bottom_cmap,
            bottom_colorbar_unit,
            invert=invert_bottom_colorbar,
        )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cloud_fraction(
    mean_map,
    std_map,
    out_path,
    cmap="afmhot",
    mean_title="Posterior mean cloud fraction",
    std_title="Posterior std. dev. of cloud fraction",
    mean_unit="f_cloud + b",
    std_unit="std",
    colorbar_orientation="horizontal",
    invert_mean_colorbar=False,
    invert_std_colorbar=False,
    mean_colorbar_unit=None,
    std_colorbar_unit=None,
):
    _plot_two_panel_map(
        mean_map,
        std_map,
        mean_title,
        std_title,
        mean_unit,
        std_unit,
        out_path,
        top_cmap=cmap,
        colorbar_orientation=colorbar_orientation,
        invert_top_colorbar=invert_mean_colorbar,
        invert_bottom_colorbar=invert_std_colorbar,
        top_colorbar_unit=mean_colorbar_unit,
        bottom_colorbar_unit=std_colorbar_unit,
    )


def _write_cloud_fraction_diagnostics(
    path,
    cloud_mean,
    cloud_std,
    contrast_mean,
):
    diagnostics = {
        "cloud_fraction_mean_min": float(np.min(cloud_mean)),
        "cloud_fraction_mean_max": float(np.max(cloud_mean)),
        "cloud_fraction_std_min": float(np.min(cloud_std)),
        "cloud_fraction_std_max": float(np.max(cloud_std)),
        "fraction_pixels_below_zero": float(np.mean(cloud_mean < 0.0)),
        "fraction_pixels_above_one": float(np.mean(cloud_mean > 1.0)),
        "contrast_mean_min": float(np.min(contrast_mean)),
        "contrast_mean_max": float(np.max(contrast_mean)),
    }
    path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


def _plot_delta_s(
    delta_s_mean,
    delta_s_std,
    out_path,
    mean_title="Mean delta_s contribution",
    std_title="Std. dev.",
):
    _plot_two_panel_map(
        delta_s_mean,
        delta_s_std,
        mean_title,
        std_title,
        "",
        "std",
        out_path,
    )


def _plot_figure9(wavelengths, observed, model, sigma_d, out_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    residual = observed - model
    offsets = np.arange(observed.shape[0])[:, None] * 0.12
    fig = plt.figure(figsize=(10, 7))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.0, 1.0],
        width_ratios=[1.0, 0.035],
        hspace=0.08,
        wspace=0.04,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
    ]
    colorbar_axis = fig.add_subplot(grid[1, 1])
    axes[1].sharex(axes[0])
    for phase_index in range(observed.shape[0]):
        axes[0].plot(
            wavelengths,
            observed[phase_index] + offsets[phase_index, 0],
            color="black",
            linewidth=0.7,
        )
        axes[0].plot(
            wavelengths,
            model[phase_index] + offsets[phase_index, 0],
            color="tab:red",
            linewidth=0.7,
        )
    axes[0].set_ylabel("Flux + offset")
    axes[0].set_title("Observed spectra and fixed two-column reconstruction")

    image = axes[1].imshow(
        residual / sigma_d,
        aspect="auto",
        origin="lower",
        extent=[wavelengths[0], wavelengths[-1], 0, observed.shape[0] - 1],
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0,
    )
    axes[1].set_xlabel("Wavelength [Angstrom]")
    axes[1].set_ylabel("Phase index")
    fig.colorbar(image, cax=colorbar_axis, label="Residual / sigma_d")
    axes[0].tick_params(labelbottom=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Compute and save Milestone 2-1 diagnostic products."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_path = None if args.smoke_test else args.profiles
    chip_data, geometry, clear_profile, cloudy_profile = load_milestone2_fixed_inputs(
        args.data_dir,
        profiles_path=profiles_path,
        chip_index=args.chip_index,
        nside=args.nside,
        smoke_test=args.smoke_test,
        smoke_wavelength_step=args.smoke_wavelength_step,
        smoke_phase_count=args.smoke_phase_count,
    )
    samples = dict(np.load(args.samples, allow_pickle=False))
    sample_indices = _select_sample_indices(len(samples["v"]), args.max_map_samples)

    contrast_mean, contrast_var, cloud_mean, cloud_var = compute_contrast_map_moments(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        samples,
        sample_indices=sample_indices,
    )
    contrast_mean = np.asarray(contrast_mean)
    contrast_var = np.asarray(contrast_var)
    cloud_mean = np.asarray(cloud_mean)
    cloud_var = np.asarray(cloud_var)
    cloud_std = np.sqrt(cloud_var)
    clipped_cloud_mean = np.clip(cloud_mean, 0.0, 1.0)

    model, median_sample = reconstruct_fixed_two_column_timeseries(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        samples,
        contrast_mean,
    )
    model = np.asarray(model)
    residual = chip_data.flux - model
    sigma_d = float(np.asarray(median_sample["sigma_d"]))

    delta_profile = np.asarray(cloudy_profile) - np.asarray(clear_profile)
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
        out_dir / f"figure9_fixed_two_column_chip{args.chip_index}.png",
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
