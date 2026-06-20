"""Plotting helpers for mean-subtracted spectral prediction diagnostics."""

import os

import numpy as np


def plot_mean_subtracted_spectra(
    wavelengths,
    observed,
    model,
    out_path,
    delta_display_scale=3.0,
    offset_scale=0.5,
    top_height_ratio=6.0,
    figure_height=12.0,
    observed_alpha=0.48,
    observed_linewidth=0.6,
    model_alpha=0.98,
    model_linewidth=1.8,
    observed_color_by_phase=False,
    observed_cmap="viridis",
    mark_model_minima=False,
    model_minima_percentile=35.0,
    model_minima_line_alpha=0.18,
    model_minima_linewidth=0.6,
):
    """Save a three-panel mean-subtracted prediction diagnostic."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [top_height_ratio, 1.0, 0.4]},
    )
    plot_mean_subtracted_spectra_axes(
        axes,
        wavelengths,
        observed,
        model,
        delta_display_scale=delta_display_scale,
        offset_scale=offset_scale,
        observed_alpha=observed_alpha,
        observed_linewidth=observed_linewidth,
        model_alpha=model_alpha,
        model_linewidth=model_linewidth,
        observed_color_by_phase=observed_color_by_phase,
        observed_cmap=observed_cmap,
        mark_model_minima=mark_model_minima,
        model_minima_percentile=model_minima_percentile,
        model_minima_line_alpha=model_minima_line_alpha,
        model_minima_linewidth=model_minima_linewidth,
        show_title=True,
        show_legend=True,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_subtracted_spectra_axes(
    axes,
    wavelengths,
    observed,
    model,
    delta_display_scale=3.0,
    offset_scale=0.5,
    observed_alpha=0.48,
    observed_linewidth=0.6,
    model_alpha=0.98,
    model_linewidth=1.8,
    observed_color_by_phase=False,
    observed_cmap="viridis",
    mark_model_minima=False,
    model_minima_percentile=35.0,
    model_minima_line_alpha=0.18,
    model_minima_linewidth=0.6,
    show_title=True,
    title="Mean-subtracted spectra and reconstruction",
    show_legend=True,
):
    """Draw a mean-subtracted prediction diagnostic on existing axes."""

    import matplotlib.pyplot as plt

    observed_delta = observed - np.mean(observed, axis=0, keepdims=True)
    model_delta = model - np.mean(model, axis=0, keepdims=True)
    observed_mean = np.mean(observed, axis=0)
    model_mean = np.mean(model, axis=0)
    mean_residual = observed_mean - model_mean
    local_minimum_mask = (
        (model_mean[1:-1] < model_mean[:-2])
        & (model_mean[1:-1] < model_mean[2:])
    )
    local_minimum_indices = np.where(local_minimum_mask)[0] + 1
    if local_minimum_indices.size > 0:
        minimum_threshold = np.nanpercentile(model_mean, model_minima_percentile)
        local_minimum_indices = local_minimum_indices[
            model_mean[local_minimum_indices] <= minimum_threshold
        ]
    observed_delta_display = delta_display_scale * observed_delta
    model_delta_display = delta_display_scale * model_delta
    observed_scale = float(np.nanpercentile(np.abs(observed_delta_display), 95.0))
    model_scale = float(np.nanpercentile(np.abs(model_delta), 95.0))
    if not np.isfinite(model_scale) or model_scale <= 0.0:
        model_scale = float(np.nanpercentile(np.abs(observed_delta), 68.0))
    if not np.isfinite(model_scale) or model_scale <= 0.0:
        model_scale = 0.01
    offset_step = offset_scale * max(
        2.0 * observed_scale,
        12.0 * delta_display_scale * model_scale,
    )
    offsets = np.arange(observed.shape[0])[:, None] * offset_step
    if observed_color_by_phase:
        cmap = plt.get_cmap(observed_cmap)
        observed_colors = [
            cmap(value)
            for value in np.linspace(0.0, 1.0, observed.shape[0])
        ]
    else:
        observed_colors = ["black"] * observed.shape[0]
    for phase_index in range(observed.shape[0]):
        axes[0].plot(
            wavelengths,
            observed_delta_display[phase_index] + offsets[phase_index, 0],
            color=observed_colors[phase_index],
            alpha=observed_alpha,
            linewidth=observed_linewidth,
            zorder=1,
        )
        axes[0].plot(
            wavelengths,
            model_delta_display[phase_index] + offsets[phase_index, 0],
            color="firebrick",
            linewidth=model_linewidth,
            alpha=model_alpha,
            zorder=2,
        )
    axes[0].set_ylabel(f"Delta flux x {delta_display_scale:g} + offset")
    if show_title:
        axes[0].set_title(title)
    axes[0].set_ylim(
        float(offsets[0, 0] - offset_step),
        float(offsets[-1, 0] + offset_step),
    )

    axes[1].plot(
        wavelengths,
        observed_mean,
        color="black",
        alpha=observed_alpha,
        linewidth=max(observed_linewidth, 0.8),
        label="Observed mean",
    )
    axes[1].plot(
        wavelengths,
        model_mean,
        color="firebrick",
        alpha=model_alpha,
        linewidth=max(model_linewidth, 1.2),
        label="Model mean",
    )
    axes[1].set_ylabel("Mean flux")
    if show_legend:
        axes[1].legend(loc="best", frameon=False)
    axes[2].plot(
        wavelengths,
        mean_residual,
        color="0.2",
        linewidth=0.8,
    )
    axes[2].axhline(0.0, color="0.5", alpha=0.5, linewidth=0.7)
    axes[2].set_xlabel("Wavelength [Angstrom]")
    axes[2].set_ylabel("Mean residual")
    axes[2].set_ylim(-0.1, 0.1)
    if mark_model_minima:
        for minimum_index in local_minimum_indices:
            wavelength = wavelengths[minimum_index]
            for ax in axes:
                ax.axvline(
                    wavelength,
                    color="0.2",
                    alpha=model_minima_line_alpha,
                    linewidth=model_minima_linewidth,
                    zorder=0,
                )
