"""Reconstruct E2/v1 eigenmaps and their physical-parameter projections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from doraex.data.luhman16b import Luhman16BChipData  # noqa: E402
from doraex.inference.map_posterior import conditional_map_posterior  # noqa: E402
from doraex.operators.design_matrix import linear_profile_operator_from_times  # noqa: E402
from doraex.priors.spherical_gp import (  # noqa: E402
    add_diagonal_jitter,
    project_zero_mean_covariance,
    squared_exponential_covariance,
)
from doraex.workflows.luhman16b_milestone2 import build_luhman16b_geometry  # noqa: E402
from m6_v1_run import build_response_functions, load_eigen_basis  # noqa: E402


DEFAULT_SAMPLES = ROOT / "results" / "e2" / "v1_run" / "samples.npz"
DEFAULT_BASIS = ROOT / "results" / "e2" / "v1_basis" / "eigen_response_basis.npz"
DEFAULT_OUT_DIR = ROOT / "results" / "e2" / "v1_primary_products"

PARAMETER_LABELS = {
    "log_p_cloud": r"$\delta\log_{10} P_{\rm cloud}$ [dex]",
    "T0": r"$\delta T_0$ [K]",
    "alpha": r"$\delta\alpha$",
    "log_vmr_co": r"$\delta\log_{10}{\rm VMR}_{\rm CO}$ [dex]",
    "log_vmr_h2o": r"$\delta\log_{10}{\rm VMR}_{\rm H_2O}$ [dex]",
}


def parse_args() -> argparse.Namespace:
    """Parse product-generation options."""

    parser = argparse.ArgumentParser(
        description="Build E2/v1 conditional eigenmap and physical-map products."
    )
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--basis", default=str(DEFAULT_BASIS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-map-samples", type=int, default=None)
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--gp-jitter", type=float, default=5.0e-6)
    parser.add_argument("--noise-jitter", type=float, default=1.0e-6)
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument(
        "--database-dir", default=str(Path.home() / "data_mol" / ".database")
    )
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _select_indices(sample_count: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples >= sample_count:
        return np.arange(sample_count)
    return np.linspace(0, sample_count - 1, max_samples, dtype=int)


def _load_chip_data(samples: dict[str, np.ndarray], chip_indices: tuple[int, ...]):
    obs_times = np.asarray(samples["obs_times"])
    chips = []
    for chip_index in chip_indices:
        chips.append(
            Luhman16BChipData(
                wavelengths=np.asarray(samples[f"wavelengths_chip{chip_index}"]),
                flux=np.asarray(samples[f"flux_chip{chip_index}"]),
                line_profile=np.ones_like(samples[f"wavelengths_chip{chip_index}"]),
                obs_times=obs_times,
                chip_index=chip_index,
            )
        )
    return chips


def _sample_at(samples: dict[str, np.ndarray], index: int) -> dict[str, jnp.ndarray]:
    """Extract one nonlinear posterior sample and fixed metadata."""

    names = (
        "T0", "alpha", "logg", "log_p_cloud", "log_vmr_co", "log_vmr_h2o",
        "log_vmr_ch4", "log_vmr_hf", "P", "cosi", "v", "q1", "q2", "u1",
        "u2", "A", "log_w", "sigma_d", "sigma_eigen", "ell_eigen",
    )
    sample = {}
    for name in names:
        value = np.asarray(samples[name])
        sample[name] = jnp.asarray(value if value.ndim == 0 else value[index])
    return sample


def _joint_operator(chips, geometry, response_functions, sample, mode_count: int):
    """Return E2 baseline and stacked eigenmode design matrix for one sample."""

    inclination = jnp.arccos(sample["cosi"])
    baselines = []
    by_mode = [[] for _ in range(mode_count)]
    noise = []
    for chip_position, chip in enumerate(chips):
        base_profile, eigen_profiles = response_functions[chip_position](
            sample["T0"],
            sample["alpha"],
            sample["log_vmr_co"],
            sample["log_vmr_h2o"],
            sample["log_vmr_ch4"],
            sample["log_vmr_hf"],
            sample["logg"],
            sample["log_p_cloud"],
        )
        baseline = None
        norm = None
        for mode in range(mode_count):
            mode_baseline, mode_matrix = linear_profile_operator_from_times(
                geometry.theta,
                geometry.phi,
                sample["v"],
                inclination,
                sample["u1"],
                sample["u2"],
                jnp.asarray(chip.obs_times),
                sample["P"],
                jnp.asarray(chip.wavelengths),
                base_profile,
                eigen_profiles[mode],
                weights=jnp.exp(sample["log_w"][chip_position]),
            )
            if baseline is None:
                norm = sample["A"][chip_position] * jnp.mean(mode_baseline)
                baseline = mode_baseline / norm
            by_mode[mode].append(mode_matrix / norm)
        baselines.append(baseline)
        noise.append(
            sample["sigma_d"][chip_position] ** 2
            * jnp.ones(chip.flux.size, dtype=baseline.dtype)
        )
    design = jnp.concatenate(
        [jnp.concatenate(mode_matrices, axis=0) for mode_matrices in by_mode], axis=1
    )
    return jnp.concatenate(baselines), design, jnp.concatenate(noise)


def _prior_covariance(geometry, sample, mode_count: int, gp_jitter: float):
    """Return the E2 block-diagonal zero-mean GP covariance."""

    factors = []
    for mode in range(mode_count):
        covariance = squared_exponential_covariance(
            geometry.distance_matrix, sample["sigma_eigen"][mode], sample["ell_eigen"][mode]
        )
        covariance = project_zero_mean_covariance(covariance)
        covariance = add_diagonal_jitter(covariance, jitter=jnp.asarray(gp_jitter))
        factors.append(covariance)
    return jsp_linalg.block_diag(*factors)


def _conditional_for_sample(
    chips, geometry, response_functions, sample, mode_count, gp_jitter, noise_jitter
):
    baseline, design, noise_variance = _joint_operator(
        chips, geometry, response_functions, sample, mode_count
    )
    data = jnp.concatenate([jnp.asarray(chip.flux).reshape(-1) for chip in chips])
    mean, covariance = conditional_map_posterior(
        data - baseline,
        design,
        jnp.zeros(design.shape[1], dtype=design.dtype),
        _prior_covariance(geometry, sample, mode_count, gp_jitter),
        noise_variance + noise_jitter,
    )
    pixel_count = geometry.theta.shape[0]
    mean = mean.reshape(mode_count, pixel_count)
    covariance = covariance.reshape(mode_count, pixel_count, mode_count, pixel_count)
    # The GP is zero-mean in each mode; remove numerical monopoles after conditioning.
    mean = mean - jnp.mean(mean, axis=1, keepdims=True)
    return mean, covariance


def _figure10_grid(values, longitude_shift_deg=180.0, nlon=721, nlat=361):
    """Sample a HEALPix map on the rotated M7 Figure-10 Aitoff grid."""

    import healpy as hp

    longitude = np.linspace(-np.pi, np.pi, nlon)
    latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nlat)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    theta = 0.5 * np.pi - latitude_grid
    phi = np.mod(longitude_grid + np.deg2rad(longitude_shift_deg), 2.0 * np.pi)
    pixel_index = hp.ang2pix(hp.npix2nside(values.size), theta, phi)
    return longitude_grid, latitude_grid, np.asarray(values)[pixel_index]


def _plot_figure10_style_map(
    values, title, unit, out_path, *, cmap, symmetric=False, invert_colorbar=False
):
    """Write one M7-Figure-10-style rotated Aitoff primary map."""

    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if symmetric:
        limit = float(np.max(np.abs(finite)))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    longitude_grid, latitude_grid, projected = _figure10_grid(values)
    fig = plt.figure(figsize=(8.3, 4.75), dpi=240)
    axis = fig.add_axes((0.02, 0.22, 0.96, 0.76), projection="aitoff")
    mesh = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        projected,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    axis.set_xticks(np.linspace(-np.pi, np.pi, 11)[1:-1])
    axis.set_yticks(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 11)[1:-1])
    axis.grid(color="0.55", alpha=0.6, linewidth=0.8)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_title(title, fontsize=18, pad=22)
    colorbar_axis = fig.add_axes((0.18, 0.065, 0.64, 0.045))
    colorbar = fig.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(unit)
    colorbar.ax.tick_params(labelsize=16)
    colorbar.ax.xaxis.label.set_size(16)
    if invert_colorbar:
        colorbar.ax.invert_xaxis()
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    """Compute E2 eigenmap and physical-map posterior products."""

    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = dict(np.load(args.samples, allow_pickle=False))
    basis = load_eigen_basis(args.basis)
    mode_count = basis["mode_count"]
    parameter_names = basis["parameter_names"]
    parameter_scales = np.asarray(basis["parameter_scales"], dtype=float)
    projection = parameter_scales[:, None] * np.asarray(basis["selected_v"], dtype=float)
    chip_indices = tuple(int(v) for v in np.asarray(samples["chip_indices"]))
    nside = int(np.asarray(samples["nside"])) if args.nside is None else args.nside
    chips = _load_chip_data(samples, chip_indices)
    geometry = build_luhman16b_geometry(nside=nside)

    # build_response_functions honors the frozen eigenspectra stored in the basis.
    args.frozen_eigen_spectra = True
    response_functions = build_response_functions(args, chips, basis)
    indices = _select_indices(len(np.asarray(samples["T0"])), args.max_map_samples)
    pixel_count = geometry.theta.shape[0]
    mode_means = []
    physical_means = []
    physical_absolute_means = []
    mode_diag_sum = np.zeros((mode_count, pixel_count))
    physical_diag_sum = np.zeros((len(parameter_names), pixel_count))

    for count, index in enumerate(indices, start=1):
        start = time.time()
        sample = _sample_at(samples, int(index))
        mean, covariance = _conditional_for_sample(
            chips,
            geometry,
            response_functions,
            sample,
            mode_count,
            args.gp_jitter,
            args.noise_jitter,
        )
        mean_np = np.asarray(mean)
        covariance_np = np.asarray(covariance)
        mode_means.append(mean_np)
        mode_diag_sum += np.stack(
            [np.diag(covariance_np[mode, :, mode, :]) for mode in range(mode_count)]
        )
        physical_mean = projection @ mean_np
        physical_means.append(physical_mean)
        global_values = np.asarray([sample[name] for name in parameter_names], dtype=float)
        physical_absolute_means.append(physical_mean + global_values[:, None])
        for parameter_index, coefficients in enumerate(projection):
            diagonal = np.zeros(pixel_count)
            for left in range(mode_count):
                for right in range(mode_count):
                    diagonal += coefficients[left] * coefficients[right] * np.diag(
                        covariance_np[left, :, right, :]
                    )
            physical_diag_sum[parameter_index] += diagonal
        if count == 1 or count % 10 == 0 or count == len(indices):
            print(
                f"Processed map sample {count}/{len(indices)} "
                f"(posterior index {int(index)}, {time.time() - start:.2f} s)",
                flush=True,
            )

    mode_stack = np.asarray(mode_means)
    physical_stack = np.asarray(physical_means)
    physical_absolute_stack = np.asarray(physical_absolute_means)
    eigen_mean = mode_stack.mean(axis=0)
    eigen_var = mode_diag_sum / len(indices) + np.mean(
        (mode_stack - eigen_mean[None, :, :]) ** 2, axis=0
    )
    physical_mean = physical_stack.mean(axis=0)
    physical_var = physical_diag_sum / len(indices) + np.mean(
        (physical_stack - physical_mean[None, :, :]) ** 2, axis=0
    )
    physical_absolute_mean = physical_absolute_stack.mean(axis=0)
    physical_absolute_var = physical_diag_sum / len(indices) + np.mean(
        (physical_absolute_stack - physical_absolute_mean[None, :, :]) ** 2, axis=0
    )

    np.savez(
        out_dir / "e2_v1_map_posterior.npz",
        sample_indices=indices,
        theta=np.asarray(geometry.theta),
        phi=np.asarray(geometry.phi),
        eigen_mean=eigen_mean,
        eigen_var=eigen_var,
        physical_parameter_names=np.asarray(parameter_names),
        physical_projection=projection,
        physical_perturbation_mean=physical_mean,
        physical_perturbation_var=physical_var,
        physical_absolute_mean=physical_absolute_mean,
        physical_absolute_var=physical_absolute_var,
    )
    for mode in range(mode_count):
        np.save(out_dir / f"eigenmap_{mode + 1}_mean.npy", eigen_mean[mode])
        np.save(out_dir / f"eigenmap_{mode + 1}_var.npy", eigen_var[mode])
        _plot_figure10_style_map(
            eigen_mean[mode],
            f"Eigenmap {mode + 1}",
            "dimensionless coefficient",
            out_dir / f"eigenmap_{mode + 1}.png",
            cmap="RdBu_r",
            symmetric=True,
        )
    for parameter_index, name in enumerate(parameter_names):
        label = PARAMETER_LABELS.get(name, name)
        np.save(out_dir / f"{name}_perturbation_mean.npy", physical_mean[parameter_index])
        np.save(out_dir / f"{name}_perturbation_var.npy", physical_var[parameter_index])
        np.save(out_dir / f"{name}_absolute_mean.npy", physical_absolute_mean[parameter_index])
        np.save(out_dir / f"{name}_absolute_var.npy", physical_absolute_var[parameter_index])
        unit = "dex" if name.startswith("log_") else ("K" if name == "T0" else "")
        _plot_figure10_style_map(
            physical_mean[parameter_index],
            label,
            unit,
            out_dir / f"{name}_perturbation.png",
            cmap="RdBu_r",
            symmetric=True,
        )
        if name == "log_p_cloud":
            _plot_figure10_style_map(
                physical_absolute_mean[parameter_index],
                r"$\log_{10} P_{\rm cloud}$",
                "dex",
                out_dir / "log_p_cloud_absolute.png",
                cmap="inferno",
                invert_colorbar=True,
            )

    summary = {
        "samples": str(args.samples),
        "basis": str(args.basis),
        "sample_count": int(len(indices)),
        "sample_indices_min": int(indices.min()),
        "sample_indices_max": int(indices.max()),
        "nside": nside,
        "mode_count": mode_count,
        "parameter_names": list(parameter_names),
        "physical_projection": projection.tolist(),
        "gp_jitter": args.gp_jitter,
        "noise_jitter": args.noise_jitter,
        "frozen_eigen_spectra": True,
        "map_projection": "M7 Figure-10 style Aitoff; longitude +180 deg",
    }
    (out_dir / "product_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"E2/v1 primary map products saved to {out_dir}")


if __name__ == "__main__":
    main()
