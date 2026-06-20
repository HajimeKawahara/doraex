"""Create mean-subtracted prediction figures around selected spectral lines."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from doraex.spectra.exojax_forward import FixedPowerLawAtmosphere
from doraex.diagnostics.mean_subtracted_plotting import (
    plot_mean_subtracted_spectra,
    plot_mean_subtracted_spectra_axes,
)

YAMA_L16B_EXOMOL_ATMOSPHERE = FixedPowerLawAtmosphere(
    t0=1219.0,
    alpha=0.129,
    logg=4.97,
    log_vmr_co=-2.96,
    log_vmr_h2o=-3.25,
    log_vmr_ch4=-4.65,
    log_vmr_hf=-7.08,
    rv=25.66,
    log_p_cloud=1.45,
)


SPEED_OF_LIGHT_KMS = 299792.458
DEFAULT_LINE_STRENGTH_MOLECULES = ("CO", "H2O")
EXOMOL_ELOWER_MAX = {
    "CO": 58242.689,
    "H2O": 23726.625476,
    "CH4": 9900.0,
    "HF": 20000.0,
}


def _molecule_paths(database_dir):
    """Return ExoMol database paths for the Luhman 16B diagnostic molecules."""

    database = Path(os.path.expanduser(database_dir))
    return {
        "CO": database / "CO" / "12C-16O" / "Li2015",
        "H2O": database / "H2O" / "1H2-16O" / "POKAZATEL",
        "CH4": database / "CH4" / "12C-1H4" / "MM",
        "HF": database / "HF" / "1H-19F" / "Coxon-Hajig",
    }


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_highlight_lines(text):
    """Parse highlighted line markers as chip:center:linestyle[:color] entries."""

    if text is None or not text.strip():
        return []
    lines = []
    for item in text.split(","):
        fields = [field.strip() for field in item.split(":")]
        if len(fields) not in (2, 3, 4):
            raise argparse.ArgumentTypeError(
                "Highlighted lines must use chip:center[:linestyle[:color]] entries."
            )
        lines.append(
            {
                "chip_index": int(fields[0]),
                "center_wavelength": float(fields[1]),
                "linestyle": fields[2] if len(fields) >= 3 else "solid",
                "color": fields[3] if len(fields) == 4 else "0.05",
            }
        )
    return lines


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create mean-subtracted prediction figures in wavelength windows "
            "centered on selected model minima or temperature-dependent "
            "molecular line strengths."
        )
    )
    default_database = Path.home() / "data_mol" / ".database"
    parser.add_argument(
        "--samples",
        required=True,
        help="Posterior NPZ containing embedded flux and wavelength arrays.",
    )
    parser.add_argument(
        "--product-dir",
        required=True,
        help="Directory containing model_spectrum_chip*.npy products.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Defaults to product-dir/minimum_window_figures."
        ),
    )
    parser.add_argument("--chip-indices", type=parse_chips, default=None)
    parser.add_argument(
        "--center-source",
        choices=("model-minima", "line-strength"),
        default="model-minima",
        help=(
            "Source used to select window centers. model-minima uses local "
            "minima of the phase-mean model spectrum. line-strength uses the "
            "strongest ExoMol lines at the selected temperature."
        ),
    )
    parser.add_argument(
        "--database-dir",
        default=str(default_database),
        help="ExoMol database directory used for line-strength center selection.",
    )
    parser.add_argument(
        "--line-strength-temperature",
        type=float,
        default=None,
        help=(
            "Temperature in K for ExoMol line-strength center selection. "
            "Defaults to the posterior median T0 when available."
        ),
    )
    parser.add_argument(
        "--line-strength-molecules",
        default=",".join(DEFAULT_LINE_STRENGTH_MOLECULES),
        help="Comma-separated molecule names used for line-strength centers.",
    )
    parser.add_argument(
        "--radial-velocity",
        type=float,
        default=float(YAMA_L16B_EXOMOL_ATMOSPHERE.rv),
        help=(
            "Radial velocity in km/s used to shift rest-frame line-strength "
            "centers onto the observed wavelength grid."
        ),
    )
    parser.add_argument(
        "--window-half-width",
        type=float,
        default=5.0,
        help="Half-width in Angstrom around each selected minimum.",
    )
    parser.add_argument(
        "--edge-exclusion",
        type=float,
        default=None,
        help=(
            "Exclude model-spectrum minima within this many Angstrom of a chip "
            "edge. Defaults to window-half-width."
        ),
    )
    parser.add_argument(
        "--minima-percentile",
        type=float,
        default=35.0,
        help="Keep local minima below this model-mean flux percentile.",
    )
    parser.add_argument(
        "--minimum-separation",
        type=float,
        default=8.0,
        help="Minimum separation in Angstrom between selected minima.",
    )
    parser.add_argument(
        "--max-minima-per-chip",
        type=int,
        default=None,
        help="Maximum number of minimum-centered windows per chip.",
    )
    parser.add_argument(
        "--individual-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create one figure for each selected line-centered window.",
    )
    parser.add_argument(
        "--combined-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create one aligned stack figure per chip by averaging all selected "
            "minimum-centered windows."
        ),
    )
    parser.add_argument(
        "--joint-combined-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create one aligned stack figure averaged over all selected chips "
            "and minimum-centered windows."
        ),
    )
    parser.add_argument(
        "--composite-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create a five-column summary figure: all-chip minimum stack plus "
            "full-chip mean-subtracted figures for chip0-3."
        ),
    )
    parser.add_argument(
        "--single-line-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create one relative-wavelength profile figure for a single "
            "selected line. With line-strength centers, the strongest selected "
            "line is used by default."
        ),
    )
    parser.add_argument(
        "--single-line-rank",
        type=int,
        default=0,
        help=(
            "Zero-based rank of the selected line used for the single-line "
            "profile figure. Rank 0 is the strongest line-strength center or "
            "the deepest model minimum."
        ),
    )
    parser.add_argument(
        "--single-line-chip",
        type=int,
        default=None,
        help="Optional chip index restriction for the single-line profile figure.",
    )
    parser.add_argument(
        "--composite-figure-width",
        type=float,
        default=36.0,
        help="Figure width in inches for the five-column summary figure.",
    )
    parser.add_argument(
        "--composite-output-name",
        default="figure9_mean_subtracted_composite_summary.png",
        help="Output filename for the five-column summary figure.",
    )
    parser.add_argument(
        "--highlight-lines",
        type=parse_highlight_lines,
        default=[],
        help=(
            "Comma-separated chip:center[:linestyle] markers to emphasize in "
            "the full-chip panels."
        ),
    )
    parser.add_argument("--highlight-linewidth", type=float, default=1.4)
    parser.add_argument("--highlight-line-alpha", type=float, default=0.85)
    parser.add_argument("--publication-style", action="store_true")
    parser.add_argument("--font-size", type=float, default=12.0)
    parser.add_argument("--label-size", type=float, default=12.0)
    parser.add_argument("--title-size", type=float, default=13.0)
    parser.add_argument("--tick-size", type=float, default=10.5)
    parser.add_argument("--composite-wspace", type=float, default=0.22)
    parser.add_argument("--composite-hspace", type=float, default=0.08)
    parser.add_argument("--composite-mean-height-ratio", type=float, default=1.0)
    parser.add_argument("--composite-residual-height-ratio", type=float, default=0.4)
    parser.add_argument("--composite-max-xticks", type=int, default=4)
    parser.add_argument("--composite-max-yticks", type=int, default=4)
    parser.add_argument(
        "--composite-chip-delta-scale",
        type=float,
        default=None,
        help=(
            "Display delta-flux scale for the full-chip panels in the composite "
            "summary. Defaults to --delta-scale."
        ),
    )
    parser.add_argument(
        "--composite-chip-offset-scale",
        type=float,
        default=None,
        help=(
            "Vertical phase-offset scale for the full-chip panels in the "
            "composite summary. Defaults to --offset-scale."
        ),
    )
    parser.add_argument(
        "--combined-grid-count",
        type=int,
        default=201,
        help="Relative-wavelength grid size for combined minimum stacks.",
    )
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--offset-scale", type=float, default=0.5)
    parser.add_argument("--top-height-ratio", type=float, default=6.0)
    parser.add_argument("--figure-height", type=float, default=12.0)
    parser.add_argument("--observed-alpha", type=float, default=0.48)
    parser.add_argument("--observed-linewidth", type=float, default=0.6)
    parser.add_argument("--model-alpha", type=float, default=0.98)
    parser.add_argument("--model-linewidth", type=float, default=1.8)
    parser.add_argument("--observed-color-by-phase", action="store_true")
    parser.add_argument("--observed-cmap", default="viridis")
    parser.add_argument(
        "--mark-window-minima",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw faint vertical lines at model minima inside each window.",
    )
    parser.add_argument("--model-minima-line-alpha", type=float, default=0.18)
    parser.add_argument("--model-minima-linewidth", type=float, default=0.6)
    return parser.parse_args()


def _local_minimum_indices(
    values,
    percentile,
    wavelengths,
    minimum_separation,
    max_count,
    edge_exclusion,
):
    """Select separated local minima of a one-dimensional spectrum."""

    values = np.asarray(values, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    local_mask = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    candidates = np.where(local_mask)[0] + 1
    if candidates.size == 0:
        return candidates
    threshold = np.nanpercentile(values, percentile)
    candidates = candidates[values[candidates] <= threshold]
    if edge_exclusion > 0.0:
        lower = np.min(wavelengths) + edge_exclusion
        upper = np.max(wavelengths) - edge_exclusion
        candidates = candidates[
            (wavelengths[candidates] >= lower)
            & (wavelengths[candidates] <= upper)
        ]
    candidates = candidates[np.argsort(values[candidates])]

    selected = []
    for index in candidates:
        wavelength = wavelengths[index]
        if all(
            abs(wavelength - wavelengths[existing]) >= minimum_separation
            for existing in selected
        ):
            selected.append(int(index))
        if max_count is not None and len(selected) >= max_count:
            break
    return np.asarray(sorted(selected, key=lambda item: wavelengths[item]), dtype=int)


def _parse_molecules(text):
    """Parse comma-separated molecule names."""

    molecules = tuple(item.strip() for item in text.split(",") if item.strip())
    if not molecules:
        raise ValueError("At least one molecule is required.")
    return molecules


def _posterior_temperature(samples, explicit_temperature):
    """Return the temperature used for line-strength ranking."""

    if explicit_temperature is not None:
        return float(explicit_temperature)
    if "T0" in samples:
        return float(np.nanmedian(np.asarray(samples["T0"], dtype=float)))
    raise KeyError(
        "line-strength center selection requires --line-strength-temperature "
        "when T0 is absent from the samples file."
    )


def _wavelength_to_wavenumber(wavelength_angstrom):
    """Convert vacuum wavelength in Angstrom to wavenumber in cm^-1."""

    return 1.0e8 / np.asarray(wavelength_angstrom, dtype=float)


def _wavenumber_to_wavelength(wavenumber):
    """Convert wavenumber in cm^-1 to vacuum wavelength in Angstrom."""

    return 1.0e8 / np.asarray(wavenumber, dtype=float)


def _observed_frame_center(rest_wavelength, radial_velocity):
    """Return the line center shifted by the model radial velocity."""

    return float(rest_wavelength) * (1.0 + float(radial_velocity) / SPEED_OF_LIGHT_KMS)


def _nearest_indices_for_centers(wavelengths, centers):
    """Map selected wavelength centers onto the nearest model wavelength pixels."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    indices = []
    for center in centers:
        indices.append(int(np.argmin(np.abs(wavelengths - center))))
    return np.asarray(indices, dtype=int)


def _line_strength_center_indices(
    wavelengths,
    temperature,
    molecule_paths,
    molecules,
    minimum_separation,
    max_count,
    edge_exclusion,
    radial_velocity,
):
    """Select separated wavelength centers from strongest ExoMol lines."""

    from exojax.database import MdbExomol

    wavelengths = np.asarray(wavelengths, dtype=float)
    lower_wavelength = float(np.min(wavelengths))
    upper_wavelength = float(np.max(wavelengths))
    if edge_exclusion > 0.0:
        lower_wavelength += edge_exclusion
        upper_wavelength -= edge_exclusion
    if lower_wavelength >= upper_wavelength:
        return np.asarray([], dtype=int), []

    lower_nu = float(_wavelength_to_wavenumber(upper_wavelength))
    upper_nu = float(_wavelength_to_wavenumber(lower_wavelength))
    candidates = []
    for molecule in molecules:
        if molecule not in molecule_paths:
            raise KeyError(f"Unknown molecule for line-strength centers: {molecule}")
        try:
            mdb = MdbExomol(
                molecule_paths[molecule],
                nurange=[lower_nu, upper_nu],
                Ttyp=temperature,
                broadf=False,
                broadf_download=False,
                gpu_transfer=False,
                elower_max=EXOMOL_ELOWER_MAX.get(molecule),
            )
        except ValueError as exc:
            if "No line found" in str(exc):
                print(f"No {molecule} lines found in this chip range; skipping.")
                continue
            raise
        strengths = np.asarray(mdb.line_strength(temperature), dtype=float)
        line_wavelengths = np.asarray(
            _wavenumber_to_wavelength(mdb.nu_lines),
            dtype=float,
        )
        del mdb
        finite = np.isfinite(strengths) & np.isfinite(line_wavelengths)
        finite &= strengths > 0.0
        finite &= (line_wavelengths >= lower_wavelength) & (
            line_wavelengths <= upper_wavelength
        )
        for center, strength in zip(line_wavelengths[finite], strengths[finite]):
            observed_center = _observed_frame_center(center, radial_velocity)
            if not (lower_wavelength <= observed_center <= upper_wavelength):
                continue
            candidates.append(
                {
                    "center_wavelength": float(observed_center),
                    "rest_frame_center_wavelength": float(center),
                    "line_strength": float(strength),
                    "molecule": molecule,
                    "radial_velocity_kms": float(radial_velocity),
                }
            )

    candidates.sort(key=lambda item: item["line_strength"], reverse=True)
    selected = []
    for candidate in candidates:
        center = candidate["center_wavelength"]
        if all(
            abs(center - existing["center_wavelength"]) >= minimum_separation
            for existing in selected
        ):
            selected.append(candidate)
        if max_count is not None and len(selected) >= max_count:
            break
    selected.sort(key=lambda item: item["center_wavelength"])
    indices = _nearest_indices_for_centers(
        wavelengths,
        [item["center_wavelength"] for item in selected],
    )
    return indices, selected


def _window_slice(wavelengths, center, half_width):
    """Return a boolean mask for a wavelength window."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    return np.abs(wavelengths - center) <= half_width


def _interpolate_centered_window(values, wavelengths, center, relative_grid):
    """Interpolate phase spectra onto a relative wavelength grid."""

    order = np.argsort(wavelengths)
    sorted_relative = np.asarray(wavelengths, dtype=float)[order] - center
    sorted_values = np.asarray(values, dtype=float)[:, order]
    if (
        relative_grid[0] < sorted_relative[0]
        or relative_grid[-1] > sorted_relative[-1]
    ):
        return None
    return np.vstack(
        [
            np.interp(relative_grid, sorted_relative, phase_values)
            for phase_values in sorted_values
        ]
    )


def _combined_minimum_stack(observed, model, wavelengths, centers, half_width, grid_count):
    """Return spectra averaged over line-centered wavelength windows."""

    relative_grid = np.linspace(-half_width, half_width, grid_count)
    observed_windows = []
    model_windows = []
    used_centers = []
    for center in centers:
        center = float(center)
        observed_window = _interpolate_centered_window(
            observed,
            wavelengths,
            center,
            relative_grid,
        )
        model_window = _interpolate_centered_window(
            model,
            wavelengths,
            center,
            relative_grid,
        )
        if observed_window is None or model_window is None:
            continue
        observed_windows.append(observed_window)
        model_windows.append(model_window)
        used_centers.append(center)
    if not observed_windows:
        return None, None, relative_grid, []
    return (
        np.mean(np.stack(observed_windows, axis=0), axis=0),
        np.mean(np.stack(model_windows, axis=0), axis=0),
        relative_grid,
        used_centers,
    )


def _centered_windows(observed, model, wavelengths, centers, half_width, grid_count):
    """Return all valid centered windows for later joint stacking."""

    relative_grid = np.linspace(-half_width, half_width, grid_count)
    windows = []
    used_centers = []
    for center in centers:
        center = float(center)
        observed_window = _interpolate_centered_window(
            observed,
            wavelengths,
            center,
            relative_grid,
        )
        model_window = _interpolate_centered_window(
            model,
            wavelengths,
            center,
            relative_grid,
        )
        if observed_window is None or model_window is None:
            continue
        windows.append((observed_window, model_window))
        used_centers.append(center)
    return windows, used_centers, relative_grid


def _centered_window_at_center(observed, model, wavelengths, center, half_width, grid_count):
    """Return one centered observed/model window around an exact line center."""

    relative_grid = np.linspace(-half_width, half_width, grid_count)
    observed_window = _interpolate_centered_window(
        observed,
        wavelengths,
        center,
        relative_grid,
    )
    model_window = _interpolate_centered_window(
        model,
        wavelengths,
        center,
        relative_grid,
    )
    return observed_window, model_window, relative_grid


def _sigma_d_for_chip(samples, chip_position):
    """Return a representative noise scale for one chip position."""

    if "sigma_d" not in samples:
        return 1.0
    values = np.asarray(samples["sigma_d"], dtype=float)
    if values.ndim == 1:
        return float(values[chip_position])
    if values.ndim == 2:
        return float(np.median(values[:, chip_position]))
    raise ValueError("sigma_d must be one- or two-dimensional.")


def _plot_kwargs(args):
    """Return display parameters for the shared plotting helper."""

    return {
        "delta_display_scale": args.delta_scale,
        "offset_scale": args.offset_scale,
        "top_height_ratio": args.top_height_ratio,
        "figure_height": args.figure_height,
        "observed_alpha": args.observed_alpha,
        "observed_linewidth": args.observed_linewidth,
        "model_alpha": args.model_alpha,
        "model_linewidth": args.model_linewidth,
        "observed_color_by_phase": args.observed_color_by_phase,
        "observed_cmap": args.observed_cmap,
        "mark_model_minima": args.mark_window_minima,
        "model_minima_percentile": 100.0,
        "model_minima_line_alpha": args.model_minima_line_alpha,
        "model_minima_linewidth": args.model_minima_linewidth,
    }


def _axes_plot_kwargs(args):
    """Return display parameters accepted by the axes-level plotting helper."""

    kwargs = _plot_kwargs(args)
    kwargs.pop("top_height_ratio", None)
    kwargs.pop("figure_height", None)
    return kwargs


def _format_wavelength(value):
    """Format a wavelength for stable output filenames."""

    return f"{value:.2f}".replace(".", "p")


def _plot_composite_summary(
    out_dir,
    samples,
    product_dir,
    chip_indices,
    relative_grid,
    joint_observed,
    joint_model,
    args,
):
    """Create a five-column summary figure from data arrays."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if args.publication_style:
        plt.rcParams.update(
            {
                "font.size": args.font_size,
                "axes.labelsize": args.label_size,
                "axes.titlesize": args.title_size,
                "xtick.labelsize": args.tick_size,
                "ytick.labelsize": args.tick_size,
                "legend.fontsize": args.tick_size,
            }
        )
    axes_kwargs = _axes_plot_kwargs(args)
    chip_axes_kwargs = dict(axes_kwargs)
    if args.composite_chip_delta_scale is not None:
        chip_axes_kwargs["delta_display_scale"] = args.composite_chip_delta_scale
    if args.composite_chip_offset_scale is not None:
        chip_axes_kwargs["offset_scale"] = args.composite_chip_offset_scale
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(args.composite_figure_width, args.figure_height),
        sharex="col",
        gridspec_kw={
            "height_ratios": [
                args.top_height_ratio,
                args.composite_mean_height_ratio,
                args.composite_residual_height_ratio,
            ],
            "width_ratios": [1.0 / 3.0, 1.0, 1.0, 1.0, 1.0],
            "wspace": args.composite_wspace,
            "hspace": args.composite_hspace,
        },
    )
    plot_mean_subtracted_spectra_axes(
        axes[:, 0],
        relative_grid,
        joint_observed,
        joint_model,
        **axes_kwargs,
        show_title=True,
        title="All chips line stack",
        show_legend=False,
    )
    axes[2, 0].set_xlabel("Relative wavelength [A]")
    axes[1, 0].set_ylabel("Mean", labelpad=10)
    axes[2, 0].set_ylabel("Residual", labelpad=10)

    for column, chip_index in enumerate(chip_indices, start=1):
        wavelengths = np.asarray(samples[f"wavelengths_chip{chip_index}"], dtype=float)
        observed = np.asarray(samples[f"flux_chip{chip_index}"], dtype=float)
        model = np.asarray(
            np.load(product_dir / f"model_spectrum_chip{chip_index}.npy"),
            dtype=float,
        )
        plot_mean_subtracted_spectra_axes(
            axes[:, column],
            wavelengths,
            observed,
            model,
            **chip_axes_kwargs,
            show_title=True,
            title=f"Chip {chip_index}",
            show_legend=False,
        )
        for line in args.highlight_lines:
            if line["chip_index"] != int(chip_index):
                continue
            for row in range(3):
                axes[row, column].axvline(
                    line["center_wavelength"],
                    color=line["color"],
                    linestyle=line["linestyle"],
                    linewidth=args.highlight_linewidth,
                    alpha=args.highlight_line_alpha,
                    zorder=5,
                )

    for column in range(1, 5):
        for row in range(3):
            axes[row, column].set_ylabel("")
        axes[2, column].set_xlabel("")
    for row in range(2):
        for column in range(5):
            axes[row, column].tick_params(labelbottom=False)
    for column in range(5):
        axes[2, column].xaxis.set_major_locator(
            MaxNLocator(nbins=args.composite_max_xticks)
        )
    for row in range(3):
        for column in range(5):
            axes[row, column].yaxis.set_major_locator(
                MaxNLocator(nbins=args.composite_max_yticks)
            )
            axes[row, column].tick_params(axis="both", pad=3)
    fig.canvas.draw()
    right_panel_boxes = [axes[2, column].get_position() for column in range(1, 5)]
    shared_label_x = 0.5 * (right_panel_boxes[0].x0 + right_panel_boxes[-1].x1)
    shared_label_y = max(0.01, min(box.y0 for box in right_panel_boxes) - 0.045)
    fig.text(
        shared_label_x,
        shared_label_y,
        "Wavelength [A]",
        ha="center",
        va="top",
        fontsize=args.label_size if args.publication_style else None,
    )
    out_path = out_dir / args.composite_output_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _single_line_score(metadata, center_source):
    """Return the ranking score for one selected line candidate."""

    if center_source == "line-strength":
        return float(metadata["line_strength"])
    return -float(metadata["model_mean_flux"])


def _plot_single_line_profile(
    out_dir,
    samples,
    product_dir,
    candidates,
    args,
):
    """Create a single-line profile figure from the strongest selected line."""

    ranked_candidates = sorted(
        candidates,
        key=lambda item: item["rank_score"],
        reverse=True,
    )
    if args.single_line_chip is not None:
        ranked_candidates = [
            item
            for item in ranked_candidates
            if item["chip_index"] == args.single_line_chip
        ]
    if args.single_line_rank >= len(ranked_candidates):
        return None

    for candidate in ranked_candidates[args.single_line_rank :]:
        chip_index = candidate["chip_index"]
        chip_position = candidate["chip_position"]
        center = candidate["center_wavelength"]
        wavelengths = np.asarray(samples[f"wavelengths_chip{chip_index}"], dtype=float)
        observed = np.asarray(samples[f"flux_chip{chip_index}"], dtype=float)
        model = np.asarray(
            np.load(product_dir / f"model_spectrum_chip{chip_index}.npy"),
            dtype=float,
        )
        observed_window, model_window, relative_grid = _centered_window_at_center(
            observed,
            model,
            wavelengths,
            center,
            args.window_half_width,
            args.combined_grid_count,
        )
        if observed_window is None or model_window is None:
            continue
        center_label = _format_wavelength(center)
        out_path = out_dir / (
            f"figure9_single_line_profile_chip{chip_index}_lambda{center_label}.png"
        )
        sigma_d = _sigma_d_for_chip(samples, chip_position)
        plot_mean_subtracted_spectra(
            relative_grid,
            observed_window,
            model_window,
            out_path,
            **_plot_kwargs(args),
        )
        result = dict(candidate)
        result["output_path"] = str(out_path)
        result["relative_wavelength_grid_size"] = int(len(relative_grid))
        return result
    return None


def main():
    """Create minimum-centered prediction-window figures."""

    args = parse_args()
    if args.single_line_rank < 0:
        raise ValueError("--single-line-rank must be non-negative.")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    product_dir = Path(args.product_dir)
    out_dir = (
        product_dir / "minimum_window_figures"
        if args.out_dir is None
        else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = dict(np.load(args.samples, allow_pickle=False))
    chip_indices = (
        [int(value) for value in np.asarray(samples["chip_indices"])]
        if args.chip_indices is None
        else args.chip_indices
    )
    line_strength_temperature = None
    line_strength_molecules = _parse_molecules(args.line_strength_molecules)
    molecule_paths = None
    if args.center_source == "line-strength":
        line_strength_temperature = _posterior_temperature(
            samples,
            args.line_strength_temperature,
        )
        molecule_paths = _molecule_paths(args.database_dir)
    plot_kwargs = _plot_kwargs(args)
    summary = {
        "sample_path": str(args.samples),
        "product_dir": str(product_dir),
        "center_source": args.center_source,
        "window_half_width": args.window_half_width,
        "edge_exclusion": (
            args.window_half_width
            if args.edge_exclusion is None
            else args.edge_exclusion
        ),
        "minima_percentile": args.minima_percentile,
        "minimum_separation": args.minimum_separation,
        "chips": [],
    }
    if args.center_source == "line-strength":
        summary["line_strength_selection"] = {
            "temperature": float(line_strength_temperature),
            "molecules": list(line_strength_molecules),
            "database_dir": str(Path(args.database_dir).expanduser()),
            "radial_velocity_kms": float(args.radial_velocity),
        }
    joint_observed_windows = []
    joint_model_windows = []
    joint_sigma_d = []
    joint_centers = []
    single_line_candidates = []
    relative_grid = np.linspace(
        -args.window_half_width,
        args.window_half_width,
        args.combined_grid_count,
    )

    for chip_position, chip_index in enumerate(chip_indices):
        wavelength_key = f"wavelengths_chip{chip_index}"
        flux_key = f"flux_chip{chip_index}"
        model_path = product_dir / f"model_spectrum_chip{chip_index}.npy"
        if wavelength_key not in samples or flux_key not in samples:
            raise KeyError(f"Missing {wavelength_key} or {flux_key} in samples.")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model spectrum: {model_path}")

        wavelengths = np.asarray(samples[wavelength_key], dtype=float)
        observed = np.asarray(samples[flux_key], dtype=float)
        model = np.asarray(np.load(model_path), dtype=float)
        model_mean = np.mean(model, axis=0)
        if args.center_source == "model-minima":
            minima = _local_minimum_indices(
                model_mean,
                args.minima_percentile,
                wavelengths,
                args.minimum_separation,
                args.max_minima_per_chip,
                summary["edge_exclusion"],
                args.radial_velocity,
            )
            selected_center_metadata = [
                {
                    "center_wavelength": float(wavelengths[index]),
                    "model_mean_flux": float(model_mean[index]),
                }
                for index in minima
            ]
        else:
            minima, selected_center_metadata = _line_strength_center_indices(
                wavelengths,
                line_strength_temperature,
                molecule_paths,
                line_strength_molecules,
                args.minimum_separation,
                args.max_minima_per_chip,
                summary["edge_exclusion"],
                args.radial_velocity,
            )
        chip_summary = {
            "chip_index": int(chip_index),
            "minimum_count": int(len(minima)),
            "selected_centers": selected_center_metadata,
            "windows": [],
        }
        window_centers = [
            float(metadata["center_wavelength"])
            for metadata in selected_center_metadata
        ]
        for metadata in selected_center_metadata:
            candidate = dict(metadata)
            candidate["chip_index"] = int(chip_index)
            candidate["chip_position"] = int(chip_position)
            candidate["rank_score"] = _single_line_score(metadata, args.center_source)
            single_line_candidates.append(candidate)
        sigma_d = _sigma_d_for_chip(samples, chip_position)
        centered_windows, used_joint_centers, relative_grid = _centered_windows(
            observed,
            model,
            wavelengths,
            window_centers,
            args.window_half_width,
            args.combined_grid_count,
        )
        for observed_window, model_window in centered_windows:
            joint_observed_windows.append(observed_window)
            joint_model_windows.append(model_window)
            joint_sigma_d.append(sigma_d)
        joint_centers.extend(
            [
                {
                    "chip_index": int(chip_index),
                    "center_wavelength": float(center),
                }
                for center in used_joint_centers
            ]
        )
        if args.combined_stack:
            stacked_observed, stacked_model, relative_grid, used_centers = (
                _combined_minimum_stack(
                    observed,
                    model,
                    wavelengths,
                    window_centers,
                    args.window_half_width,
                    args.combined_grid_count,
                )
            )
            if stacked_observed is not None and stacked_model is not None:
                combined_path = (
                    out_dir / f"figure9_minimum_combined_stack_chip{chip_index}.png"
                )
                plot_mean_subtracted_spectra(
                    relative_grid,
                    stacked_observed,
                    stacked_model,
                    combined_path,
                    **plot_kwargs,
                )
                chip_summary["combined_stack"] = {
                    "output_path": str(combined_path),
                    "stacked_minimum_count": int(len(used_centers)),
                    "center_wavelengths": [float(value) for value in used_centers],
                }
        if args.individual_windows:
            for window_number, metadata in enumerate(selected_center_metadata):
                center = float(metadata["center_wavelength"])
                mask = _window_slice(wavelengths, center, args.window_half_width)
                if int(np.sum(mask)) < 3:
                    continue
                center_label = _format_wavelength(center)
                out_path = (
                    out_dir
                    / f"figure9_minimum_window_chip{chip_index}_{window_number:02d}"
                    f"_lambda{center_label}.png"
                )
                plot_mean_subtracted_spectra(
                    wavelengths[mask],
                    observed[:, mask],
                    model[:, mask],
                    out_path,
                    **plot_kwargs,
                )
                chip_summary["windows"].append(
                    {
                        "center_wavelength": center,
                        "output_path": str(out_path),
                        "pixel_count": int(np.sum(mask)),
                    }
                )
        summary["chips"].append(chip_summary)

    if args.single_line_profile and single_line_candidates:
        single_line_profile = _plot_single_line_profile(
            out_dir,
            samples,
            product_dir,
            single_line_candidates,
            args,
        )
        if single_line_profile is not None:
            summary["single_line_profile"] = single_line_profile
        else:
            print("Single-line profile skipped: no valid line window found.")

    if args.joint_combined_stack and joint_observed_windows:
        joint_observed = np.mean(np.stack(joint_observed_windows, axis=0), axis=0)
        joint_model = np.mean(np.stack(joint_model_windows, axis=0), axis=0)
        joint_sigma = float(np.mean(joint_sigma_d)) if joint_sigma_d else 1.0
        joint_path = out_dir / "figure9_minimum_combined_stack_all_chips.png"
        plot_mean_subtracted_spectra(
            relative_grid,
            joint_observed,
            joint_model,
            joint_path,
            **plot_kwargs,
        )
        summary["joint_combined_stack"] = {
            "output_path": str(joint_path),
            "stacked_window_count": int(len(joint_observed_windows)),
            "centers": joint_centers,
        }
        if args.composite_summary:
            if len(chip_indices) != 4:
                print("Composite summary skipped: expected exactly four chip indices.")
            else:
                composite_path = _plot_composite_summary(
                    out_dir,
                    samples,
                    product_dir,
                    chip_indices,
                    relative_grid,
                    joint_observed,
                    joint_model,
                    args,
                )
                summary["composite_summary"] = {
                    "output_path": str(composite_path),
                }

    summary_path = out_dir / "minimum_window_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Minimum-window figures saved to {out_dir}")


if __name__ == "__main__":
    main()
