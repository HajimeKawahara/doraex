"""Create Milestone 1 Figure 8 and Figure 9 reproduction products."""

import argparse
import os
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone1 import (  # noqa: E402
    build_luhman16b_geometry,
    compute_posterior_map_moments,
    reconstruct_spectral_timeseries,
)
from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build posterior maps and spectral residual plots for Milestone 1."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
    )
    parser.add_argument(
        "--samples",
        default=str(ROOT / "results" / "milestone1" / "mcmc_chip1_sampled.npz"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone1"))
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


def _plot_figure8(mean_map, std_map, out_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import healpy as hp
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 6))
    hp.mollview(
        mean_map,
        fig=fig.number,
        sub=(2, 1, 1),
        cmap="inferno",
        title="Posterior mean map",
        unit="Intensity",
        flip="geo",
    )
    hp.graticule()
    hp.mollview(
        std_map,
        fig=fig.number,
        sub=(2, 1, 2),
        cmap="viridis",
        title="Posterior standard deviation",
        unit="Intensity",
        flip="geo",
    )
    hp.graticule()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_figure9(wavelengths, observed, model, sigma_d, out_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    residual = observed - model
    offsets = np.arange(observed.shape[0])[:, None] * 0.12
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
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
    axes[0].set_title("Observed spectra and posterior reconstruction")

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
    fig.colorbar(image, ax=axes[1], label="Residual / sigma_d")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Compute and save Milestone 1 products."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_data = load_luhman16b_chip(args.data_dir, chip_index=args.chip_index)
    if args.smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=args.smoke_wavelength_step,
            phase_count=args.smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=args.nside)
    samples = dict(np.load(args.samples, allow_pickle=False))
    sample_indices = _select_sample_indices(len(samples["v"]), args.max_map_samples)

    map_mean, map_var = compute_posterior_map_moments(
        chip_data,
        geometry,
        samples,
        sample_indices=sample_indices,
    )
    map_mean = np.asarray(map_mean)
    map_var = np.asarray(map_var)
    map_std = np.sqrt(map_var)

    model, median_sample = reconstruct_spectral_timeseries(
        chip_data, geometry, samples, map_mean
    )
    model = np.asarray(model)
    residual = chip_data.flux - model
    sigma_d = float(np.asarray(median_sample["sigma_d"]))

    np.save(out_dir / f"posterior_mean_chip{args.chip_index}.npy", map_mean)
    np.save(out_dir / f"posterior_var_chip{args.chip_index}.npy", map_var)
    np.save(out_dir / f"model_spectrum_chip{args.chip_index}.npy", model)
    np.save(out_dir / f"residual_chip{args.chip_index}.npy", residual)
    np.savez(
        out_dir / f"posterior_median_parameters_chip{args.chip_index}.npz",
        **{key: np.asarray(value) for key, value in median_sample.items()},
    )

    _plot_figure8(
        map_mean,
        map_std,
        out_dir / f"figure8_chip{args.chip_index}_mean_uncertainty.png",
    )
    _plot_figure9(
        chip_data.wavelengths,
        chip_data.flux,
        model,
        sigma_d,
        out_dir / f"figure9_chip{args.chip_index}_spectral_fit_residual.png",
    )
    print(f"Products saved to {out_dir}")


if __name__ == "__main__":
    main()
