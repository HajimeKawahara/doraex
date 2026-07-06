"""Synthetic M7/v1 fixed-eta pressure-spot latitude-PSF experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402
from doraex.geometry.healpix import angular_distance  # noqa: E402
from doraex.geometry.limb_darkening import kipping_q_to_u  # noqa: E402
from doraex.inference.map_posterior import conditional_map_posterior  # noqa: E402
from doraex.diagnostics.mean_subtracted_plotting import (  # noqa: E402
    plot_mean_subtracted_spectra_axes,
)
from doraex.operators.design_matrix import (  # noqa: E402
    full_design_matrix_from_times,
    linear_profile_operator_from_times,
)
from doraex.priors.spherical_gp import (  # noqa: E402
    add_diagonal_jitter,
    project_zero_mean_covariance,
    squared_exponential_covariance,
)
from doraex.workflows.luhman16b_milestone2 import build_luhman16b_geometry  # noqa: E402

from m6_v1_run import (  # noqa: E402
    LOG_VMR_NAMES,
    _cia_paths,
    _molecule_paths,
    _response_function,
)
from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel  # noqa: E402
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
)
from make_milestone2_fixed_products import _plot_figure9  # noqa: E402


DEFAULT_SAMPLES = ROOT / "results" / "m7" / "v1_zero_mean_log_w_run" / "samples.npz"
DEFAULT_OUT_DIR = ROOT / "results" / "syn" / "m7v1_pressure_spot_fixed_eta"


def parse_chips(text: str) -> list[int]:
    """Parse a comma-separated chip list."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic pressure-spot data at fixed M7/v1 nonlinear "
            "parameters and compare fixed-eta Doppler-retrieval and Bayesian "
            "DI conditional map reconstructions."
        )
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument("--opacity-cache-dir", type=Path, default=ROOT / "data" / "opacities" / "luhman16b_powerlaw")
    parser.add_argument("--database-dir", type=Path, default=default_database)
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=128)
    parser.add_argument("--smoke-phase-count", type=int, default=4)

    parser.add_argument("--baseline-p-cloud-bar", type=float, default=20.0)
    parser.add_argument("--spot-p-cloud-bar", type=float, default=50.0)
    parser.add_argument("--spot-lat-deg", type=float, default=30.0)
    parser.add_argument("--spot-lon-deg", type=float, default=0.0)
    parser.add_argument("--spot-sigma-deg", type=float, default=15.0)
    parser.add_argument("--rescale-spot-peak", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fixed-cosi", type=float, default=0.485)
    parser.add_argument("--fixed-v", type=float, default=31.2)
    parser.add_argument("--fixed-period", type=float, default=4.83)
    parser.add_argument("--fixed-q1", type=float, default=0.81)
    parser.add_argument("--fixed-q2", type=float, default=0.59)
    parser.add_argument("--fixed-ell-b", type=float, default=0.4)
    parser.add_argument("--gp-jitter", type=float, default=1.0e-6)
    parser.add_argument("--noise-jitter", type=float, default=1.0e-6)
    parser.add_argument("--noise-seed", type=int, default=7)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.25,
        help="Multiplicative scale applied to the M7/v1 median sigma_d noise.",
    )
    parser.add_argument(
        "--noise-mode",
        choices=("posterior", "none"),
        default="posterior",
        help="Use M7/v1 median sigma_d noise or noiseless synthetic data.",
    )
    parser.add_argument("--intensity-spot-amplitude", type=float, default=0.2)
    parser.add_argument("--intensity-prior-sigma", type=float, default=0.2)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def posterior_median(samples: np.lib.npyio.NpzFile, name: str):
    """Return the posterior median for a sampled quantity."""

    return np.median(np.asarray(samples[name]), axis=0)


def load_chip_data(args: argparse.Namespace):
    """Load the selected Luhman 16B chips."""

    chips = []
    for chip_index in args.chip_indices:
        chip = load_luhman16b_chip(args.data_dir, chip_index=chip_index)
        if args.smoke_test:
            chip = subset_chip_data(
                chip,
                wavelength_step=args.smoke_wavelength_step,
                phase_count=args.smoke_phase_count,
            )
        chips.append(chip)
    return chips


def build_response_functions(args: argparse.Namespace, chip_data_list):
    """Build pressure-response functions matching the M7 on-the-fly model."""

    responses = []
    for chip_data in chip_data_list:
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx,
        )
        responses.append(jax.jit(_response_function(model.cloudy_at_log_vmrs)))
    return responses


def zero_mean_gaussian_spot(
    theta,
    phi,
    *,
    center_lat_deg: float,
    center_lon_deg: float,
    sigma_deg: float,
    peak_delta: float,
    rescale_peak: bool,
) -> jnp.ndarray:
    """Return a zero-mean spherical Gaussian spot in log-pressure units."""

    center_theta = jnp.asarray(0.5 * jnp.pi - np.deg2rad(center_lat_deg))
    center_phi = jnp.asarray(np.deg2rad(center_lon_deg))
    distance = angular_distance(theta, phi, center_theta, center_phi)
    raw = jnp.exp(-0.5 * (distance / np.deg2rad(sigma_deg)) ** 2)
    spot = raw - jnp.mean(raw)
    if rescale_peak:
        spot = spot * (peak_delta / jnp.max(spot))
    else:
        spot = peak_delta * spot
    return spot


def _select_chip_values(
    samples: np.lib.npyio.NpzFile,
    values: np.ndarray,
    chip_indices: list[int],
) -> np.ndarray:
    """Select chip-position values from an M7 sample archive by chip index."""

    if "chip_indices" not in samples.files:
        return np.asarray(values, dtype=float)[: len(chip_indices)]
    saved_chips = [int(value) for value in np.asarray(samples["chip_indices"]).reshape(-1)]
    positions = [saved_chips.index(chip_index) for chip_index in chip_indices]
    return np.asarray(values, dtype=float)[positions]


def fixed_eta_from_m7(samples: np.lib.npyio.NpzFile, args: argparse.Namespace) -> dict[str, object]:
    """Collect M7/v1 posterior-median fixed parameters for the synthetic run."""

    return {
        "T0": float(posterior_median(samples, "T0")),
        "alpha": float(posterior_median(samples, "alpha")),
        "logg": float(posterior_median(samples, "logg")),
        "log_vmr_co": float(posterior_median(samples, "log_vmr_co")),
        "log_vmr_h2o": float(posterior_median(samples, "log_vmr_h2o")),
        "log_vmr_ch4": float(posterior_median(samples, "log_vmr_ch4")),
        "log_vmr_hf": float(posterior_median(samples, "log_vmr_hf")),
        "sigma_log_p": float(posterior_median(samples, "sigma_log_p")),
        "A": _select_chip_values(samples, posterior_median(samples, "A"), args.chip_indices),
        "log_w": _select_chip_values(
            samples,
            posterior_median(samples, "log_w"),
            args.chip_indices,
        ),
        "sigma_d": _select_chip_values(
            samples,
            posterior_median(samples, "sigma_d"),
            args.chip_indices,
        ),
        "cosi": float(args.fixed_cosi),
        "inclination_deg": float(np.degrees(np.arccos(args.fixed_cosi))),
        "vrot": float(args.fixed_v),
        "period": float(args.fixed_period),
        "q1": float(args.fixed_q1),
        "q2": float(args.fixed_q2),
        "ell_b": float(args.fixed_ell_b),
    }


def build_operators(
    args: argparse.Namespace,
    chip_data_list,
    geometry,
    responses,
    eta: dict[str, object],
    log_p0: float,
):
    """Build joint pressure and intensity linear operators."""

    inclination = jnp.arccos(jnp.asarray(eta["cosi"]))
    u1, u2 = kipping_q_to_u(jnp.asarray(eta["q1"]), jnp.asarray(eta["q2"]))
    baselines = []
    pressure_matrices = []
    intensity_matrices = []
    base_profiles = []
    for position, chip_data in enumerate(chip_data_list):
        base_profile, pressure_profile = responses[position](
            jnp.asarray(eta["T0"]),
            jnp.asarray(eta["alpha"]),
            jnp.asarray(eta["log_vmr_co"]),
            jnp.asarray(eta["log_vmr_h2o"]),
            jnp.asarray(eta["log_vmr_ch4"]),
            jnp.asarray(eta["log_vmr_hf"]),
            jnp.asarray(eta["logg"]),
            jnp.asarray(log_p0),
        )
        base_profile.block_until_ready()
        pressure_profile.block_until_ready()
        weights = jnp.exp(jnp.asarray(eta["log_w"])[position])
        baseline, pressure_matrix = linear_profile_operator_from_times(
            geometry.theta,
            geometry.phi,
            jnp.asarray(eta["vrot"]),
            inclination,
            u1,
            u2,
            jnp.asarray(chip_data.obs_times),
            jnp.asarray(eta["period"]),
            jnp.asarray(chip_data.wavelengths),
            base_profile,
            pressure_profile,
            weights=weights,
        )
        intensity_matrix = full_design_matrix_from_times(
            geometry.theta,
            geometry.phi,
            jnp.asarray(eta["vrot"]),
            inclination,
            u1,
            u2,
            jnp.asarray(chip_data.obs_times),
            jnp.asarray(eta["period"]),
            jnp.asarray(chip_data.wavelengths),
            base_profile,
            weights=weights,
        )
        norm = jnp.asarray(eta["A"])[position] * jnp.mean(baseline)
        baselines.append(baseline / norm)
        pressure_matrices.append(pressure_matrix / norm)
        intensity_matrices.append(intensity_matrix / norm)
        base_profiles.append(np.asarray(base_profile))
    return (
        jnp.concatenate(baselines, axis=0),
        jnp.concatenate(pressure_matrices, axis=0),
        jnp.concatenate(intensity_matrices, axis=0),
        base_profiles,
    )


def noise_variance_for_chips(args: argparse.Namespace, chip_data_list, eta: dict[str, object]) -> jnp.ndarray:
    """Return the diagonal noise variance used for synthetic data and recovery."""

    variances = []
    for position, chip_data in enumerate(chip_data_list):
        if args.noise_mode == "none":
            sigma = 0.0
        else:
            sigma = args.noise_scale * float(np.asarray(eta["sigma_d"])[position])
        variances.append((sigma**2 + args.noise_jitter) * jnp.ones(chip_data.flux.size))
    return jnp.concatenate(variances, axis=0)


def split_joint_vector(vector: np.ndarray, chip_data_list) -> list[np.ndarray]:
    """Split a joint flattened spectral vector into per-chip phase/wavelength arrays."""

    vector = np.asarray(vector)
    arrays = []
    offset = 0
    for chip_data in chip_data_list:
        size = int(np.prod(chip_data.flux.shape))
        arrays.append(vector[offset : offset + size].reshape(chip_data.flux.shape))
        offset += size
    return arrays


def add_noise(data: jnp.ndarray, noise_variance: jnp.ndarray, seed: int, mode: str) -> jnp.ndarray:
    """Add diagonal Gaussian noise to synthetic data."""

    if mode == "none":
        return data
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(np.asarray(noise_variance)), size=data.shape)
    return data + jnp.asarray(noise, dtype=data.dtype)


def zero_mean_prior_covariance(geometry, amplitude: float, ell: float, jitter: float) -> jnp.ndarray:
    """Build a jittered zero-mean spherical GP covariance."""

    covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(amplitude),
        jnp.asarray(ell),
    )
    covariance = project_zero_mean_covariance(covariance)
    return add_diagonal_jitter(covariance, jitter=jitter)


def recover_map(
    residual: jnp.ndarray,
    design_matrix: jnp.ndarray,
    prior_covariance: jnp.ndarray,
    noise_variance: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recover a zero-mean map with a conditional Gaussian posterior."""

    prior_mean = jnp.zeros(design_matrix.shape[1], dtype=design_matrix.dtype)
    mean, covariance = conditional_map_posterior(
        residual,
        design_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )
    mean = mean - jnp.mean(mean)
    covariance = project_zero_mean_covariance(covariance)
    return mean, covariance


def spot_shape_metrics(values, theta, phi, reference) -> dict[str, float]:
    """Measure longitude/latitude RMS and elongation of the positive spot."""

    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if np.dot(values, reference) < 0.0:
        values = -values
    weights = np.clip(values, 0.0, None)
    if not np.any(weights > 0.0):
        weights = np.abs(values)
    weights = weights / np.sum(weights)
    lon = np.asarray(phi, dtype=float)
    lon = (lon + np.pi) % (2.0 * np.pi) - np.pi
    lat = 0.5 * np.pi - np.asarray(theta, dtype=float)
    lon_mean = np.sum(weights * lon)
    lat_mean = np.sum(weights * lat)
    dlon = lon - lon_mean
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
    dlat = lat - lat_mean
    cov_ll = np.sum(weights * dlon * dlon)
    cov_bb = np.sum(weights * dlat * dlat)
    cov_lb = np.sum(weights * dlon * dlat)
    covariance = np.array([[cov_ll, cov_lb], [cov_lb, cov_bb]])
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sigma_lon = math.degrees(math.sqrt(cov_ll))
    sigma_lat = math.degrees(math.sqrt(cov_bb))
    return {
        "peak": float(np.max(values)),
        "trough": float(np.min(values)),
        "weighted_lon_deg": float(math.degrees(lon_mean)),
        "weighted_lat_deg": float(math.degrees(lat_mean)),
        "sigma_lon_deg": float(sigma_lon),
        "sigma_lat_deg": float(sigma_lat),
        "sigma_lat_over_lon": float(sigma_lat / sigma_lon) if sigma_lon > 0.0 else float("nan"),
        "major_sigma_deg": float(math.degrees(math.sqrt(eigenvalues[-1]))),
        "minor_sigma_deg": float(math.degrees(math.sqrt(eigenvalues[0]))),
        "major_over_minor": float(math.sqrt(eigenvalues[-1] / eigenvalues[0])) if eigenvalues[0] > 0.0 else float("nan"),
    }


def write_map_products(args, geometry, maps: dict[str, np.ndarray]) -> None:
    """Write quick-look Mollweide maps."""

    if not args.plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import healpy as hp

    plot_specs = [
        ("true_pressure_q", "Injected pressure perturbation [dex]", "RdBu_r"),
        ("dr_pressure_recovered", "DR recovered pressure perturbation [dex]", "RdBu_r"),
        ("true_pressure_bar", "Injected cloud pressure [bar]", "inferno"),
        ("dr_pressure_recovered_bar", "DR recovered cloud pressure [bar]", "inferno"),
        ("true_intensity_contrast", "Injected DI intensity contrast", "RdBu_r"),
        ("di_self_recovered", "DI self recovered intensity contrast", "RdBu_r"),
        ("di_on_pressure_recovered", "DI fit to pressure mock", "RdBu_r"),
    ]
    for name, title, cmap in plot_specs:
        values = maps[name]
        if name.endswith("_bar"):
            vmin = np.nanpercentile(values, 1.0)
            vmax = np.nanpercentile(values, 99.0)
        else:
            vmax = np.nanmax(np.abs(values))
            vmin = -vmax
        hp.mollview(
            values,
            title=title,
            cmap=cmap,
            min=vmin,
            max=vmax,
            unit="",
        )
        plt.savefig(args.out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), subplot_kw={"projection": "mollweide"})
    lon = np.asarray(geometry.phi)
    lon = (lon + np.pi) % (2.0 * np.pi) - np.pi
    lat = 0.5 * np.pi - np.asarray(geometry.theta)
    for axis, (name, title, cmap) in zip(axes.flat, plot_specs):
        values = maps[name]
        if name.endswith("_bar"):
            vmin = np.nanpercentile(values, 1.0)
            vmax = np.nanpercentile(values, 99.0)
        else:
            vmax = np.nanmax(np.abs(values))
            vmin = -vmax
        sc = axis.scatter(lon, lat, c=values, s=8, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=9)
        axis.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=axis, orientation="horizontal", fraction=0.05, pad=0.08)
    for axis in axes.flat[len(plot_specs):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "syn001_reconstruction_comparison.png", dpi=180)
    plt.close(fig)
    write_figure10_style_map_comparison(
        args,
        maps,
        longitude_shift_deg=0.0,
        suffix="centered",
    )
    write_figure10_style_map_comparison(
        args,
        maps,
        longitude_shift_deg=180.0,
        suffix="shift180",
    )


def _aitoff_grid_values(
    values: np.ndarray,
    *,
    longitude_shift_deg: float = 180.0,
    nlon: int = 721,
    nlat: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a HEALPix map on a Figure-10-style shifted Aitoff grid."""

    import healpy as hp

    values = np.asarray(values, dtype=float)
    lon = np.linspace(-np.pi, np.pi, nlon)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nlat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    theta = 0.5 * np.pi - lat_grid
    phi = np.mod(lon_grid + np.deg2rad(longitude_shift_deg), 2.0 * np.pi)
    nside = hp.npix2nside(values.size)
    pixel_index = hp.ang2pix(nside, theta, phi)
    return lon_grid, lat_grid, values[pixel_index]


def _plot_figure10_style_panel(
    values: np.ndarray,
    *,
    title: str,
    unit: str,
    out_path: Path,
    cmap: str = "inferno",
    invert_colorbar: bool = False,
    longitude_shift_deg: float = 0.0,
    dpi: int = 220,
) -> None:
    """Save one map panel matching the Ureshino Figure-10 visual style."""

    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    lon_grid, lat_grid, projected = _aitoff_grid_values(
        values,
        longitude_shift_deg=longitude_shift_deg,
    )
    fig = plt.figure(figsize=(8.3, 4.75), dpi=dpi)
    ax = fig.add_axes((0.02, 0.22, 0.96, 0.76), projection="aitoff")
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        projected,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_xticks(np.linspace(-np.pi, np.pi, 11)[1:-1])
    ax.set_yticks(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 11)[1:-1])
    ax.grid(color="0.55", alpha=0.6, linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = fig.add_axes((0.18, 0.065, 0.64, 0.045))
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    colorbar.set_label(unit)
    colorbar.ax.tick_params(labelsize=16)
    colorbar.ax.xaxis.label.set_size(16)
    if invert_colorbar:
        colorbar.ax.invert_xaxis()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _trim_white_border(image, padding: int = 12):
    """Trim nearly white borders from a rendered panel image."""

    from PIL import ImageOps

    grayscale = ImageOps.grayscale(image)
    mask = grayscale.point(lambda value: 0 if value > 246 else 255)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, upper, right, lower = bbox
    return image.crop(
        (
            max(left - padding, 0),
            max(upper - padding, 0),
            min(right + padding, image.width),
            min(lower + padding, image.height),
        )
    )


def _fit_panel_width(image, width: int):
    """Resize a panel to a common width while preserving aspect ratio."""

    from PIL import Image

    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _add_panel_label(image, label: str, index: str):
    """Add a panel label above an image."""

    from PIL import Image, ImageDraw, ImageFont

    top_margin = 108
    canvas = Image.new("RGB", (image.width, image.height + top_margin), "white")
    canvas.paste(image, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        size=max(22, image.width // 36),
    )
    draw.text((8, 18), f"({index}) {label}", fill=(20, 20, 20), font=font)
    return canvas


def write_figure10_style_map_comparison(
    args,
    maps: dict[str, np.ndarray],
    *,
    longitude_shift_deg: float = 0.0,
    suffix: str = "",
) -> None:
    """Write input/DR/DI maps with Ureshino-Figure-10-like styling."""

    from PIL import Image

    panel_specs = [
        (
            "true_pressure_bar",
            "Injected cloud-pressure map",
            "bar",
            True,
        ),
        (
            "di_self_recovered",
            "Bayesian DI recovered intensity map",
            "DI contrast",
            False,
        ),
        (
            "dr_pressure_recovered_bar",
            "Doraex DR recovered cloud-pressure map",
            "bar",
            True,
        ),
    ]
    panel_images = []
    for key, title, unit, invert in panel_specs:
        path = args.out_dir / f"figure10_style_{key}.png"
        _plot_figure10_style_panel(
            maps[key],
            title=title,
            unit=unit,
            out_path=path,
            cmap="inferno",
            invert_colorbar=invert,
            longitude_shift_deg=longitude_shift_deg,
        )
        panel_images.append(_trim_white_border(Image.open(path).convert("RGB")))
    width = max(image.width for image in panel_images)
    labeled = [
        _add_panel_label(_fit_panel_width(image, width), title, chr(ord("a") + index))
        for index, (image, (_, title, _, _)) in enumerate(zip(panel_images, panel_specs))
    ]
    gap = 28
    height = sum(image.height for image in labeled) + gap * (len(labeled) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in labeled:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    suffix_text = f"_{suffix}" if suffix else ""
    out_png = args.out_dir / f"syn_figure10_style_input_di_dr_comparison{suffix_text}.png"
    out_pdf = args.out_dir / f"syn_figure10_style_input_di_dr_comparison{suffix_text}.pdf"
    canvas.save(out_png)
    canvas.save(out_pdf)


def write_prediction_products(
    args,
    chip_data_list,
    *,
    pressure_data,
    intensity_data,
    dr_pressure_prediction,
    di_pressure_prediction,
    di_intensity_prediction,
    eta: dict[str, object],
) -> dict[str, object]:
    """Write Figure-9-like and mean-subtracted synthetic prediction diagnostics."""

    if not args.plot:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    products_dir = args.out_dir / "spectral_products"
    products_dir.mkdir(parents=True, exist_ok=True)
    pressure_observed = split_joint_vector(pressure_data, chip_data_list)
    intensity_observed = split_joint_vector(intensity_data, chip_data_list)
    dr_models = split_joint_vector(dr_pressure_prediction, chip_data_list)
    di_pressure_models = split_joint_vector(di_pressure_prediction, chip_data_list)
    di_intensity_models = split_joint_vector(di_intensity_prediction, chip_data_list)

    product_paths: dict[str, str] = {}
    for position, chip_data in enumerate(chip_data_list):
        chip_index = chip_data.chip_index
        sigma_d = (
            args.noise_scale * float(np.asarray(eta["sigma_d"])[position])
            if args.noise_mode != "none"
            else float(np.sqrt(args.noise_jitter))
        )
        np.save(products_dir / f"pressure_mock_observed_chip{chip_index}.npy", pressure_observed[position])
        np.save(products_dir / f"intensity_mock_observed_chip{chip_index}.npy", intensity_observed[position])
        np.save(products_dir / f"pressure_mock_dr_model_chip{chip_index}.npy", dr_models[position])
        np.save(products_dir / f"pressure_mock_di_model_chip{chip_index}.npy", di_pressure_models[position])
        np.save(products_dir / f"intensity_mock_di_model_chip{chip_index}.npy", di_intensity_models[position])

        figure9_dr = products_dir / f"figure9_pressure_mock_dr_chip{chip_index}.png"
        figure9_di = products_dir / f"figure9_pressure_mock_di_chip{chip_index}.png"
        figure9_di_self = products_dir / f"figure9_intensity_mock_di_self_chip{chip_index}.png"
        _plot_figure9(
            chip_data.wavelengths,
            pressure_observed[position],
            dr_models[position],
            sigma_d,
            figure9_dr,
        )
        _plot_figure9(
            chip_data.wavelengths,
            pressure_observed[position],
            di_pressure_models[position],
            sigma_d,
            figure9_di,
        )
        _plot_figure9(
            chip_data.wavelengths,
            intensity_observed[position],
            di_intensity_models[position],
            sigma_d,
            figure9_di_self,
        )
        product_paths[f"figure9_pressure_mock_dr_chip{chip_index}"] = str(figure9_dr)
        product_paths[f"figure9_pressure_mock_di_chip{chip_index}"] = str(figure9_di)
        product_paths[f"figure9_intensity_mock_di_self_chip{chip_index}"] = str(figure9_di_self)

    summary_specs = [
        (
            "pressure_mock_dr",
            pressure_observed,
            dr_models,
            "Pressure mock: DR prediction",
        ),
        (
            "pressure_mock_di",
            pressure_observed,
            di_pressure_models,
            "Pressure mock: DI prediction",
        ),
        (
            "intensity_mock_di_self",
            intensity_observed,
            di_intensity_models,
            "Intensity mock: DI self prediction",
        ),
    ]
    for stem, observed_by_chip, model_by_chip, title in summary_specs:
        fig, axes = plt.subplots(
            3,
            len(chip_data_list),
            figsize=(4.6 * len(chip_data_list), 10.5),
            sharex=False,
            squeeze=False,
            gridspec_kw={"height_ratios": [5.5, 1.0, 0.55]},
        )
        for position, chip_data in enumerate(chip_data_list):
            plot_mean_subtracted_spectra_axes(
                axes[:, position],
                chip_data.wavelengths,
                observed_by_chip[position],
                model_by_chip[position],
                delta_display_scale=3.0,
                offset_scale=0.5,
                observed_alpha=0.45,
                observed_linewidth=0.55,
                model_alpha=0.98,
                model_linewidth=1.15,
                observed_color_by_phase=True,
                observed_cmap="turbo_dark",
                mark_model_minima=True,
                model_minima_percentile=35.0,
                mean_ylim_edge_fraction=0.05,
                mean_ylim_percentile=1.0,
                show_title=True,
                title=f"{title}, chip {chip_data.chip_index}",
                show_legend=(position == 0),
            )
        fig.tight_layout()
        out_path = products_dir / f"{stem}_mean_subtracted_line_stack_summary.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        product_paths[f"{stem}_mean_subtracted_line_stack_summary"] = str(out_path)
    return product_paths


def main() -> None:
    """Run the synthetic fixed-eta experiment."""

    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = np.load(args.samples, allow_pickle=True)
    eta = fixed_eta_from_m7(samples, args)
    chip_data_list = load_chip_data(args)
    geometry = build_luhman16b_geometry(nside=args.nside)
    responses = build_response_functions(args, chip_data_list)

    log_p0 = math.log10(args.baseline_p_cloud_bar)
    peak_delta = math.log10(args.spot_p_cloud_bar) - log_p0
    q_true = zero_mean_gaussian_spot(
        geometry.theta,
        geometry.phi,
        center_lat_deg=args.spot_lat_deg,
        center_lon_deg=args.spot_lon_deg,
        sigma_deg=args.spot_sigma_deg,
        peak_delta=peak_delta,
        rescale_peak=args.rescale_spot_peak,
    )

    baseline, pressure_matrix, intensity_matrix, base_profiles = build_operators(
        args,
        chip_data_list,
        geometry,
        responses,
        eta,
        log_p0,
    )
    noise_variance = noise_variance_for_chips(args, chip_data_list, eta)
    pressure_data_clean = baseline + pressure_matrix @ q_true
    pressure_data = add_noise(pressure_data_clean, noise_variance, args.noise_seed, args.noise_mode)

    intensity_shape = q_true / jnp.max(jnp.abs(q_true))
    intensity_true = args.intensity_spot_amplitude * intensity_shape
    intensity_data_clean = baseline + intensity_matrix @ intensity_true
    intensity_data = add_noise(
        intensity_data_clean,
        noise_variance,
        args.noise_seed + 1,
        args.noise_mode,
    )

    pressure_prior = zero_mean_prior_covariance(
        geometry,
        amplitude=float(eta["sigma_log_p"]),
        ell=float(eta["ell_b"]),
        jitter=args.gp_jitter,
    )
    intensity_prior = zero_mean_prior_covariance(
        geometry,
        amplitude=args.intensity_prior_sigma,
        ell=float(eta["ell_b"]),
        jitter=args.gp_jitter,
    )

    dr_mean, dr_covariance = recover_map(
        pressure_data - baseline,
        pressure_matrix,
        pressure_prior,
        noise_variance,
    )
    di_self_mean, di_self_covariance = recover_map(
        intensity_data - baseline,
        intensity_matrix,
        intensity_prior,
        noise_variance,
    )
    di_on_pressure_mean, di_on_pressure_covariance = recover_map(
        pressure_data - baseline,
        intensity_matrix,
        intensity_prior,
        noise_variance,
    )
    dr_pressure_prediction = baseline + pressure_matrix @ dr_mean
    di_pressure_prediction = baseline + intensity_matrix @ di_on_pressure_mean
    di_intensity_prediction = baseline + intensity_matrix @ di_self_mean

    maps = {
        "true_pressure_q": np.asarray(q_true),
        "dr_pressure_recovered": np.asarray(dr_mean),
        "true_pressure_bar": np.asarray(10.0 ** (log_p0 + q_true)),
        "dr_pressure_recovered_bar": np.asarray(10.0 ** (log_p0 + dr_mean)),
        "true_intensity_contrast": np.asarray(intensity_true),
        "di_self_recovered": np.asarray(di_self_mean),
        "di_on_pressure_recovered": np.asarray(di_on_pressure_mean),
    }
    covariance_diagonals = {
        "dr_pressure_var": np.asarray(jnp.diag(dr_covariance)),
        "di_self_var": np.asarray(jnp.diag(di_self_covariance)),
        "di_on_pressure_var": np.asarray(jnp.diag(di_on_pressure_covariance)),
    }

    metrics = {
        name: spot_shape_metrics(values, geometry.theta, geometry.phi, maps["true_pressure_q"])
        for name, values in maps.items()
        if not name.endswith("_bar")
    }
    spectral_products = write_prediction_products(
        args,
        chip_data_list,
        pressure_data=np.asarray(pressure_data),
        intensity_data=np.asarray(intensity_data),
        dr_pressure_prediction=np.asarray(dr_pressure_prediction),
        di_pressure_prediction=np.asarray(di_pressure_prediction),
        di_intensity_prediction=np.asarray(di_intensity_prediction),
        eta=eta,
    )
    pressure_residual_dr = np.asarray(pressure_data - dr_pressure_prediction)
    pressure_residual_di = np.asarray(pressure_data - di_pressure_prediction)
    intensity_residual_di = np.asarray(intensity_data - di_intensity_prediction)
    summary = {
        "experiment": "syn001_m7v1_pressure_spot_fixed_eta",
        "samples": str(args.samples),
        "chip_indices": args.chip_indices,
        "nside": args.nside,
        "smoke_test": args.smoke_test,
        "x64": args.x64,
        "noise_mode": args.noise_mode,
        "noise_scale": args.noise_scale,
        "fixed_eta": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in eta.items()
        },
        "baseline_p_cloud_bar": args.baseline_p_cloud_bar,
        "spot_p_cloud_bar": args.spot_p_cloud_bar,
        "log_p0": log_p0,
        "spot_peak_delta_logp": peak_delta,
        "spot_lat_deg": args.spot_lat_deg,
        "spot_lon_deg": args.spot_lon_deg,
        "spot_sigma_deg": args.spot_sigma_deg,
        "rescale_spot_peak": args.rescale_spot_peak,
        "pressure_data_rms": float(np.sqrt(np.mean(np.asarray(pressure_data - baseline) ** 2))),
        "pressure_residual_rms_dr": float(np.sqrt(np.mean(pressure_residual_dr**2))),
        "pressure_residual_rms_di": float(np.sqrt(np.mean(pressure_residual_di**2))),
        "intensity_residual_rms_di_self": float(np.sqrt(np.mean(intensity_residual_di**2))),
        "noise_sigma_by_chip": (
            [0.0 for _ in args.chip_indices]
            if args.noise_mode == "none"
            else (args.noise_scale * np.asarray(eta["sigma_d"])).tolist()
        ),
        "metrics": metrics,
        "spectral_products": spectral_products,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
        file_obj.write("\n")

    np.savez(
        args.out_dir / "maps_and_reconstructions.npz",
        theta=np.asarray(geometry.theta),
        phi=np.asarray(geometry.phi),
        pressure_data=np.asarray(pressure_data),
        pressure_data_clean=np.asarray(pressure_data_clean),
        intensity_data=np.asarray(intensity_data),
        intensity_data_clean=np.asarray(intensity_data_clean),
        baseline=np.asarray(baseline),
        dr_pressure_prediction=np.asarray(dr_pressure_prediction),
        di_pressure_prediction=np.asarray(di_pressure_prediction),
        di_intensity_prediction=np.asarray(di_intensity_prediction),
        noise_variance=np.asarray(noise_variance),
        **maps,
        **covariance_diagonals,
        **{f"base_profile_chip{chip.chip_index}": base_profiles[i] for i, chip in enumerate(chip_data_list)},
    )
    write_map_products(args, geometry, maps)
    print(f"Wrote synthetic fixed-eta experiment to {args.out_dir}")
    print(json.dumps({"metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
