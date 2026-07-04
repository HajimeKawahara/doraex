"""Build a four-column line-strength comparison panel from diagnostic data."""

import argparse
import json
from pathlib import Path

import numpy as np

from doraex.diagnostics.mean_subtracted_line_stack import _interpolate_centered_window
from doraex.diagnostics.mean_subtracted_plotting import _phase_colors
from doraex.diagnostics.single_line_pressure_response import (
    _common_relative_grid,
    _load_prediction,
    _prepare_products,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a four-column comparison panel for all-line, weak-line, "
            "strong-line, and pure-line pressure-response diagnostics."
        )
    )
    parser.add_argument("--samples", required=True)
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--single-line-inputs", required=True)
    parser.add_argument("--single-line-labels", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure-width", type=float, default=7.25)
    parser.add_argument("--figure-height", type=float, default=10.25)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title-size", type=float, default=7.5)
    parser.add_argument("--label-size", type=float, default=8.0)
    parser.add_argument("--tick-size", type=float, default=6.8)
    parser.add_argument("--legend-size", type=float, default=5.6)
    parser.add_argument("--wspace", type=float, default=0.10)
    parser.add_argument("--hspace", type=float, default=0.08)
    parser.add_argument("--window-half-width", type=float, default=5.0)
    parser.add_argument("--grid-count", type=int, default=201)
    parser.add_argument("--stack-delta-scale", type=float, default=7.0)
    parser.add_argument("--offset-stack-delta-scale", type=float, default=3.0)
    parser.add_argument("--pure-delta-scale", type=float, default=2.0)
    parser.add_argument("--offset-pure-delta-scale", type=float, default=2.0)
    parser.add_argument("--observed-alpha", type=float, default=0.68)
    parser.add_argument("--observed-linewidth", type=float, default=0.55)
    parser.add_argument("--model-alpha", type=float, default=0.98)
    parser.add_argument("--model-linewidth", type=float, default=1.25)
    parser.add_argument("--stack-observed-bin-size", type=int, default=3)
    parser.add_argument("--strong-observed-bin-size", type=int, default=3)
    parser.add_argument("--weak-reference-rest-center", type=float, default=None)
    parser.add_argument("--weak-max-line-strength", type=float, default=None)
    parser.add_argument("--strong-rest-centers", default=None)
    parser.add_argument("--line-match-tolerance", type=float, default=0.25)
    parser.add_argument("--phase-cmap", default="turbo_dark")
    return parser.parse_args()


def _split_csv(text):
    """Split a comma-separated option."""

    return [item.strip() for item in text.split(",") if item.strip()]


def _reference_wavelength(record):
    """Return the preferred wavelength used for line identity matching."""

    return float(record.get("rest_frame_center_wavelength", record["center_wavelength"]))


def _nearest_line_record(records, center, tolerance):
    """Return the nearest line record to a rest-frame wavelength center."""

    finite_records = [
        record
        for record in records
        if np.isfinite(_reference_wavelength(record))
    ]
    if not finite_records:
        raise ValueError("No line records with finite wavelength metadata are available.")
    nearest = min(
        finite_records,
        key=lambda record: abs(_reference_wavelength(record) - center),
    )
    separation = abs(_reference_wavelength(nearest) - center)
    if separation > tolerance:
        raise ValueError(
            f"No line record found within {tolerance:g} A of {center:g} A "
            f"(nearest separation {separation:g} A)."
        )
    return nearest


def _resolve_stack_records(summary, args):
    """Resolve all/weak/strong records, applying optional selection overrides."""

    all_records = summary["joint_combined_stack"]["centers"]
    weak_records = summary["strength_split_stack"]["weak_lines"]
    strong_records = summary["strength_split_stack"]["strong_lines"]

    weak_max_strength = args.weak_max_line_strength
    if args.weak_reference_rest_center is not None:
        reference_record = _nearest_line_record(
            all_records,
            args.weak_reference_rest_center,
            args.line_match_tolerance,
        )
        weak_max_strength = float(reference_record["line_strength"])
    if weak_max_strength is not None:
        weak_records = [
            record
            for record in all_records
            if float(record.get("line_strength", np.inf)) <= weak_max_strength
        ]
        weak_records.sort(key=lambda record: float(record["line_strength"]), reverse=True)

    if args.strong_rest_centers:
        strong_records = []
        seen = set()
        for value in _split_csv(args.strong_rest_centers):
            record = _nearest_line_record(
                all_records,
                float(value),
                args.line_match_tolerance,
            )
            key = (int(record["chip_index"]), float(record["center_wavelength"]))
            if key not in seen:
                strong_records.append(record)
                seen.add(key)

    return all_records, weak_records, strong_records


def _load_stack(samples, product_dir, records, half_width, grid_count):
    """Load and stack centered observed/model windows listed in summary records."""

    relative_grid = np.linspace(-half_width, half_width, grid_count)
    observed_windows = []
    model_windows = []
    for record in records:
        chip_index = int(record["chip_index"])
        center = float(record["center_wavelength"])
        wavelengths = np.asarray(samples[f"wavelengths_chip{chip_index}"], dtype=float)
        observed = np.asarray(samples[f"flux_chip{chip_index}"], dtype=float)
        model = np.asarray(
            np.load(product_dir / f"model_spectrum_chip{chip_index}.npy"),
            dtype=float,
        )
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
    if not observed_windows:
        raise ValueError("No valid windows found for stack panel.")
    return {
        "x": relative_grid,
        "observed": np.mean(np.stack(observed_windows, axis=0), axis=0),
        "model": np.mean(np.stack(model_windows, axis=0), axis=0),
        "count": len(observed_windows),
    }


def _load_pure_products(paths, labels):
    """Load and normalize pure-line products on a common relative grid."""

    products = [
        _load_prediction(path, label)
        for path, label in zip(paths, labels)
    ]
    common_grid = _common_relative_grid(products)
    return _prepare_products(products, "line-depth", common_grid)


def _stack_panel_arrays(panel, delta_scale):
    """Return top-panel arrays for a stack panel."""

    observed_delta = panel["observed"] - np.mean(panel["observed"], axis=0, keepdims=True)
    model_delta = panel["model"] - np.mean(panel["model"], axis=0, keepdims=True)
    return delta_scale * observed_delta, delta_scale * model_delta


def _pure_panel_arrays(products, delta_scale):
    """Return top-panel arrays for pure-line products."""

    return [
        delta_scale * np.asarray(item["delta_display"], dtype=float)
        for item in products
    ]


def _bin_last_axis(array, bin_size):
    """Average an array along its last axis with non-overlapping bins."""

    if bin_size <= 1:
        return array
    array = np.asarray(array, dtype=float)
    usable = (array.shape[-1] // bin_size) * bin_size
    if usable == 0:
        return array
    trimmed = array[..., :usable]
    return trimmed.reshape(*array.shape[:-1], usable // bin_size, bin_size).mean(axis=-1)


def _shared_offset_step(stack_panels, pure_products, args):
    """Compute one phase-offset step for all top panels."""

    scales = []
    for panel in stack_panels:
        observed_delta, model_delta = _stack_panel_arrays(
            panel,
            args.offset_stack_delta_scale,
        )
        scales.append(float(np.nanpercentile(np.abs(observed_delta), 95.0)))
        scales.append(float(np.nanpercentile(np.abs(model_delta), 95.0)))
    for delta in _pure_panel_arrays(pure_products, args.offset_pure_delta_scale):
        scales.append(float(np.nanpercentile(np.abs(delta), 95.0)))
    scale = max(value for value in scales if np.isfinite(value))
    if scale <= 0.0:
        scale = 0.01
    return max(2.8 * scale, 0.025)


def _plot_stack_column(
    axes,
    panel,
    title,
    offsets,
    offset_step,
    colors,
    args,
    show_ylabel,
    observed_bin_size=1,
):
    """Plot one observed/model stack column."""

    observed_delta, model_delta = _stack_panel_arrays(panel, args.stack_delta_scale)
    observed_x = _bin_last_axis(panel["x"], observed_bin_size)
    observed_delta_plot = _bin_last_axis(observed_delta, observed_bin_size)
    for phase_index in range(panel["observed"].shape[0]):
        axes[0].plot(
            observed_x,
            observed_delta_plot[phase_index] + offsets[phase_index],
            color=colors[phase_index],
            alpha=args.observed_alpha,
            linewidth=args.observed_linewidth,
            zorder=1,
        )
        axes[0].plot(
            panel["x"],
            model_delta[phase_index] + offsets[phase_index],
            color="firebrick",
            alpha=args.model_alpha,
            linewidth=args.model_linewidth,
            zorder=2,
        )
    observed_mean = np.mean(panel["observed"], axis=0)
    model_mean = np.mean(panel["model"], axis=0)
    observed_mean_plot = _bin_last_axis(observed_mean, observed_bin_size)
    model_mean_for_residual = _bin_last_axis(model_mean, observed_bin_size)
    axes[1].plot(
        observed_x,
        observed_mean_plot,
        color="0.45",
        linewidth=0.9,
        label="Observed",
    )
    axes[1].plot(panel["x"], model_mean, color="firebrick", linewidth=1.2, label="Model")
    axes[2].plot(
        observed_x,
        observed_mean_plot - model_mean_for_residual,
        color="0.35",
        linewidth=0.85,
    )
    axes[0].set_title(title, fontsize=args.title_size)
    axes[0].set_xlim(-args.window_half_width, args.window_half_width)
    axes[2].set_xlabel("Relative wavelength [A]")
    axes[2].set_ylim(-0.1, 0.1)
    if show_ylabel:
        axes[0].set_ylabel(
            "Delta flux\n+ offset",
            fontsize=args.label_size,
        )
        axes[1].set_ylabel("Mean flux", fontsize=args.label_size)
        axes[2].set_ylabel("Residual", fontsize=args.label_size)
        axes[1].legend(frameon=False, fontsize=args.legend_size, loc="best")
    else:
        for axis in axes:
            axis.set_ylabel("")
            axis.tick_params(labelleft=False)


def _plot_pure_column(axes, products, offsets, colors, args):
    """Plot the pure-line pressure-response column."""

    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    deltas = _pure_panel_arrays(products, args.pure_delta_scale)
    for product_index, item in enumerate(products):
        linestyle = linestyles[product_index % len(linestyles)]
        for phase_index in range(item["delta_display"].shape[0]):
            axes[0].plot(
                item["relative_grid"],
                deltas[product_index][phase_index] + offsets[phase_index],
                color=colors[phase_index],
                linestyle=linestyle,
                alpha=0.95,
                linewidth=0.95,
                label=item["label"] if phase_index == 0 else None,
            )
        color = colors[min(product_index * 7, len(colors) - 1)]
        axes[1].plot(
            item["relative_grid"],
            item["response_display"],
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
        )
        axes[2].plot(
            item["relative_grid"],
            item["mean_display"],
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
        )
    axes[0].set_title("(4) Pure-line prediction", fontsize=args.title_size)
    axes[0].legend(
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="none",
        fontsize=args.legend_size,
        loc="upper right",
    )
    axes[0].set_xlim(-args.window_half_width, args.window_half_width)
    axes[0].tick_params(labelleft=False)
    axes[1].axhline(0.0, color="0.75", linewidth=0.6)
    axes[1].set_ylabel("dF/dlogP", fontsize=args.label_size)
    axes[2].set_ylabel("Mean", fontsize=args.label_size)
    for axis in axes[1:]:
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()
    axes[2].set_xlabel("Relative wavelength [A]")


def main():
    """Create the comparison panel."""

    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = dict(np.load(args.samples, allow_pickle=False))
    product_dir = Path(args.product_dir)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    single_line_paths = [Path(value) for value in _split_csv(args.single_line_inputs)]
    if args.single_line_labels:
        single_line_labels = _split_csv(args.single_line_labels)
    else:
        single_line_labels = [f"line {index + 1}" for index in range(len(single_line_paths))]

    all_records, weak_records, strong_records = _resolve_stack_records(summary, args)
    stack_panels = [
        _load_stack(samples, product_dir, all_records, args.window_half_width, args.grid_count),
        _load_stack(samples, product_dir, weak_records, args.window_half_width, args.grid_count),
        _load_stack(samples, product_dir, strong_records, args.window_half_width, args.grid_count),
    ]
    pure_products = _load_pure_products(single_line_paths, single_line_labels)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": args.tick_size,
            "axes.labelsize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
        }
    )
    phase_count = min(panel["observed"].shape[0] for panel in stack_panels)
    phase_count = min(phase_count, min(item["delta_display"].shape[0] for item in pure_products))
    colors = _phase_colors(plt, args.phase_cmap, phase_count)
    offset_step = _shared_offset_step(stack_panels, pure_products, args)
    offsets = np.arange(phase_count, dtype=float) * offset_step
    top_ylim = (float(-offset_step), float(offsets[-1] + offset_step))

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(args.figure_width, args.figure_height),
        sharex="col",
        gridspec_kw={
            "height_ratios": [6.0, 1.15, 0.75],
            "width_ratios": [1.0, 1.0, 1.0, 1.0],
            "wspace": args.wspace,
            "hspace": args.hspace,
        },
    )
    titles = [
        f"(1) All\nN={stack_panels[0]['count']}",
        f"(2) Weak\nN={stack_panels[1]['count']}",
        f"(3) Strong\nN={stack_panels[2]['count']}",
    ]
    for column, (panel, title) in enumerate(zip(stack_panels, titles)):
        _plot_stack_column(
            axes[:, column],
            panel,
            title,
            offsets,
            offset_step,
            colors,
            args,
            show_ylabel=(column == 0),
            observed_bin_size=(
                args.strong_observed_bin_size
                if column == 2
                else args.stack_observed_bin_size
            ),
        )
    _plot_pure_column(axes[:, 3], pure_products, offsets, colors, args)
    for column in range(4):
        axes[0, column].set_ylim(top_ylim)
        axes[0, column].tick_params(labelbottom=False)
        axes[1, column].tick_params(labelbottom=False)
        axes[0, column].axvline(0.0, color="0.8", linewidth=0.6, zorder=0)
        axes[1, column].axvline(0.0, color="0.88", linewidth=0.6, zorder=0)
        axes[2, column].axvline(0.0, color="0.88", linewidth=0.6, zorder=0)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary_out = {
        "output": str(out_path),
        "input_products": {
            "samples": str(args.samples),
            "product_dir": str(product_dir),
            "summary": str(args.summary),
            "single_line_inputs": [str(path) for path in single_line_paths],
        },
        "panels": titles + ["(4) Pure-line prediction"],
        "shared_top_ylim": list(top_ylim),
        "shared_phase_offset_step": float(offset_step),
        "stack_selection": {
            "weak_reference_rest_center": args.weak_reference_rest_center,
            "weak_max_line_strength": args.weak_max_line_strength,
            "strong_rest_centers": _split_csv(args.strong_rest_centers)
            if args.strong_rest_centers
            else None,
            "line_match_tolerance": float(args.line_match_tolerance),
            "all_count": int(stack_panels[0]["count"]),
            "weak_count": int(stack_panels[1]["count"]),
            "strong_count": int(stack_panels[2]["count"]),
        },
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(summary_out, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Line-strength comparison panel saved to {out_path}")


if __name__ == "__main__":
    main()
