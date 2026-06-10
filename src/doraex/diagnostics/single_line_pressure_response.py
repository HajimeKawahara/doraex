"""Compare pure single-line RT predictions with matched normalization."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Overlay pure single-line RT predictions to compare whether "
            "spectral variability scales with line depth."
        )
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated single_line_rt_prediction*.npz files.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated labels. Defaults to line metadata from JSON files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--normalization",
        choices=("line-depth", "raw"),
        default="line-depth",
        help=(
            "line-depth divides mean-subtracted variability and mean profiles "
            "by each line depth. raw keeps the original normalized flux scale."
        ),
    )
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--offset-scale", type=float, default=1.25)
    parser.add_argument(
        "--offset-delta-scale",
        type=float,
        default=1.0,
        help=(
            "Delta scale used only to determine vertical offsets. Keeping this "
            "at 1 lets --delta-scale change the visible contrast without also "
            "expanding the phase spacing."
        ),
    )
    parser.add_argument("--top-height-ratio", type=float, default=8.0)
    parser.add_argument("--response-height-ratio", type=float, default=1.3)
    parser.add_argument("--figure-width", type=float, default=8.0)
    parser.add_argument("--figure-height", type=float, default=12.0)
    parser.add_argument("--hspace", type=float, default=0.08)
    parser.add_argument("--left-margin", type=float, default=0.14)
    parser.add_argument("--right-margin", type=float, default=0.78)
    parser.add_argument("--top-margin", type=float, default=0.98)
    parser.add_argument("--bottom-margin", type=float, default=0.08)
    parser.add_argument(
        "--legend-placement",
        choices=("outside", "inside"),
        default="outside",
        help="Place the line-style legend outside the axes or inside the top panel.",
    )
    parser.add_argument(
        "--top-headroom-scale",
        type=float,
        default=0.0,
        help="Extra upper y-limit headroom in units of the phase offset step.",
    )
    parser.add_argument("--linewidth", type=float, default=1.15)
    parser.add_argument("--alpha", type=float, default=0.92)
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--phase-cmap", default="turbo")
    parser.add_argument("--linestyles", default="solid,dashed,dashdot,dotted")
    parser.add_argument("--colors", default=None)
    parser.add_argument("--zero-line-alpha", type=float, default=0.22)
    parser.add_argument("--zero-linewidth", type=float, default=0.55)
    parser.add_argument("--publication-style", action="store_true")
    parser.add_argument("--font-size", type=float, default=12.0)
    parser.add_argument("--label-size", type=float, default=12.0)
    parser.add_argument("--tick-size", type=float, default=10.5)
    parser.add_argument("--legend-size", type=float, default=10.5)
    parser.add_argument(
        "--pressure-response-panel",
        action="store_true",
        help="Add a middle panel showing dF/dlog10(P_cloud) for each line.",
    )
    return parser.parse_args()


def _split_csv(text):
    """Split a comma-separated string."""

    return [item.strip() for item in text.split(",") if item.strip()]


def _json_path_for_npz(npz_path):
    """Return the sibling metadata JSON path for a prediction NPZ."""

    return npz_path.with_suffix(".json")


def _line_label(npz_path, fallback):
    """Return a compact label from single-line RT metadata."""

    json_path = _json_path_for_npz(npz_path)
    if not json_path.exists():
        return fallback
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    line = metadata.get("selected_line_metadata", {})
    molecule = line.get("molecule", "line")
    wavelength = line.get(
        "selected_center_wavelength",
        line.get("requested_center_wavelength", None),
    )
    strength = line.get("line_strength", None)
    if wavelength is None or strength is None:
        return fallback
    return f"{molecule} {float(wavelength):.2f} A line"


def _load_prediction(path, label):
    """Load one single-line prediction product."""

    data = np.load(path)
    relative = np.asarray(data["relative_wavelengths"], dtype=float)
    prediction = np.asarray(data["prediction"], dtype=float)
    mean_prediction = np.asarray(data["mean_prediction"], dtype=float)
    pressure_response = np.asarray(data["pressure_response_profile"], dtype=float)
    delta = prediction - mean_prediction[None, :]
    continuum = float(np.nanmax(mean_prediction))
    line_depth = continuum - float(np.nanmin(mean_prediction))
    if not np.isfinite(line_depth) or line_depth <= 0.0:
        raise ValueError(f"Invalid line depth for {path}: {line_depth}")
    return {
        "path": str(path),
        "label": label,
        "relative_wavelengths": relative,
        "prediction": prediction,
        "mean_prediction": mean_prediction,
        "pressure_response": pressure_response,
        "delta": delta,
        "continuum": continuum,
        "line_depth": line_depth,
    }


def _common_relative_grid(products):
    """Return a common relative wavelength grid for interpolation."""

    lower = max(float(np.min(item["relative_wavelengths"])) for item in products)
    upper = min(float(np.max(item["relative_wavelengths"])) for item in products)
    if lower >= upper:
        raise ValueError("Input products do not overlap in relative wavelength.")
    count = min(len(item["relative_wavelengths"]) for item in products)
    return np.linspace(lower, upper, count)


def _interp_rows(values, x, x_new):
    """Interpolate a two-dimensional phase-by-wavelength array."""

    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    values_sorted = np.asarray(values)[:, order]
    return np.vstack([np.interp(x_new, x_sorted, row) for row in values_sorted])


def _prepare_products(products, normalization, common_grid):
    """Interpolate products and apply the requested normalization."""

    prepared = []
    for item in products:
        mean = np.interp(
            common_grid,
            item["relative_wavelengths"],
            item["mean_prediction"],
        )
        delta = _interp_rows(
            item["delta"],
            item["relative_wavelengths"],
            common_grid,
        )
        pressure_response = np.interp(
            common_grid,
            item["relative_wavelengths"],
            item["pressure_response"],
        )
        if normalization == "line-depth":
            mean_display = (mean - item["continuum"]) / item["line_depth"]
            delta_display = delta / item["line_depth"]
            response_display = pressure_response / item["line_depth"]
            mean_ylabel = "Mean"
            delta_ylabel = "Delta/depth"
            response_ylabel = "dF/dlogP"
        else:
            mean_display = mean
            delta_display = delta
            response_display = pressure_response
            mean_ylabel = "Mean flux"
            delta_ylabel = "Delta flux"
            response_ylabel = "Response"
        prepared.append(
            {
                **item,
                "relative_grid": common_grid,
                "mean_display": mean_display,
                "delta_display": delta_display,
                "response_display": response_display,
                "mean_ylabel": mean_ylabel,
                "delta_ylabel": delta_ylabel,
                "response_ylabel": response_ylabel,
            }
        )
    return prepared


def _plot_comparison(products, args):
    """Plot overlaid single-line RT predictions."""

    import matplotlib.pyplot as plt

    if args.publication_style:
        plt.rcParams.update(
            {
                "font.size": args.font_size,
                "axes.labelsize": args.label_size,
                "xtick.labelsize": args.tick_size,
                "ytick.labelsize": args.tick_size,
                "legend.fontsize": args.legend_size,
            }
        )
    product_colors = _split_csv(args.colors) if args.colors else None
    linestyles = _split_csv(args.linestyles)
    phase_cmap = plt.get_cmap(args.phase_cmap)
    phase_count = min(item["delta_display"].shape[0] for item in products)
    if args.phase_count is not None:
        phase_count = min(phase_count, args.phase_count)
    offset_delta = [
        args.offset_delta_scale * item["delta_display"][:phase_count]
        for item in products
    ]
    scaled_delta = [
        args.delta_scale * item["delta_display"][:phase_count]
        for item in products
    ]
    variation_scale = max(
        float(np.nanpercentile(np.abs(delta), 95.0))
        for delta in offset_delta
    )
    if not np.isfinite(variation_scale) or variation_scale <= 0.0:
        variation_scale = 0.01
    offset_step = args.offset_scale * max(2.0 * variation_scale, 0.01)
    offsets = np.arange(phase_count)[:, None] * offset_step

    panel_count = 3 if args.pressure_response_panel else 2
    height_ratios = (
        [args.top_height_ratio, args.response_height_ratio, 1.0]
        if args.pressure_response_panel
        else [args.top_height_ratio, 1.0]
    )
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": args.hspace,
        },
    )
    fig.subplots_adjust(
        left=args.left_margin,
        right=args.right_margin,
        top=args.top_margin,
        bottom=args.bottom_margin,
    )
    mean_axis_index = 2 if args.pressure_response_panel else 1
    for phase_index in range(phase_count):
        axes[0].axhline(
            offsets[phase_index, 0],
            color="0.55",
            linestyle=":",
            linewidth=args.zero_linewidth,
            alpha=args.zero_line_alpha,
            zorder=0,
        )
    for product_index, item in enumerate(products):
        linestyle = linestyles[product_index % len(linestyles)]
        mean_color = (
            product_colors[product_index % len(product_colors)]
            if product_colors
            else phase_cmap((product_index + 0.5) / max(len(products), 1))
        )
        for phase_index in range(phase_count):
            color = (
                product_colors[product_index % len(product_colors)]
                if product_colors
                else phase_cmap(phase_index / max(phase_count - 1, 1))
            )
            axes[0].plot(
                item["relative_grid"],
                scaled_delta[product_index][phase_index] + offsets[phase_index, 0],
                color=color,
                linestyle=linestyle,
                alpha=args.alpha,
                linewidth=args.linewidth,
                label=item["label"] if phase_index == 0 else None,
            )
        if args.pressure_response_panel:
            axes[1].plot(
                item["relative_grid"],
                item["response_display"],
                color=mean_color,
                linestyle=linestyle,
                alpha=args.alpha,
                linewidth=max(args.linewidth, 1.4),
                label=item["label"],
            )
        axes[mean_axis_index].plot(
            item["relative_grid"],
            item["mean_display"],
            color=mean_color,
            linestyle=linestyle,
            alpha=args.alpha,
            linewidth=max(args.linewidth, 1.4),
            label=(
                f"{item['label']} (depth={item['line_depth']:.3e})"
            ),
        )
    for ax in axes:
        ax.axvline(0.0, color="0.6", alpha=0.45, linewidth=0.7)
    axes[0].set_ylabel(
        f"{products[0]['delta_ylabel']}\nx {args.delta_scale:g} + offset",
        labelpad=8,
    )
    axes[0].set_ylim(
        float(offsets[0, 0] - offset_step),
        float(offsets[-1, 0] + (1.0 + args.top_headroom_scale) * offset_step),
    )
    if args.legend_placement == "inside":
        axes[0].legend(loc="upper right", frameon=False)
    else:
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    if args.pressure_response_panel:
        axes[1].axhline(0.0, color="0.65", alpha=0.5, linewidth=0.7)
        axes[1].set_ylabel(products[0]["response_ylabel"], labelpad=8)
    axes[mean_axis_index].set_xlabel("Relative wavelength [Angstrom]")
    axes[mean_axis_index].set_ylabel(products[0]["mean_ylabel"], labelpad=8)
    return fig


def main():
    """Create an overlaid comparison figure."""

    args = parse_args()
    input_paths = [Path(value) for value in _split_csv(args.inputs)]
    if len(input_paths) < 2:
        raise ValueError("At least two input NPZ files are required.")
    if args.labels is None:
        labels = [
            _line_label(path, f"line {index + 1}")
            for index, path in enumerate(input_paths)
        ]
    else:
        labels = _split_csv(args.labels)
        if len(labels) != len(input_paths):
            raise ValueError("--labels must match the number of --inputs.")
    products = [
        _load_prediction(path, label)
        for path, label in zip(input_paths, labels)
    ]
    common_grid = _common_relative_grid(products)
    products = _prepare_products(products, args.normalization, common_grid)
    fig = _plot_comparison(products, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    summary_path = out_path.with_suffix(".json")
    summary = {
        "output": str(out_path),
        "normalization": args.normalization,
        "input_products": [
            {
                "path": item["path"],
                "label": item["label"],
                "continuum": item["continuum"],
                "line_depth": item["line_depth"],
            }
            for item in products
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Single-line comparison figure saved to {out_path}")


if __name__ == "__main__":
    main()
