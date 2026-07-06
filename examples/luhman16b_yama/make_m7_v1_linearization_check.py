"""Check the M7 v1 cloud-pressure linear approximation on chip 0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.data.luhman16b import Luhman16BChipData  # noqa: E402
from doraex.geometry.limb_darkening import quadratic_limb_darkening  # noqa: E402
from doraex.geometry.rotation import (  # noqa: E402
    incline,
    line_of_sight_velocity,
    projected_mu,
    rotate_longitude,
    visible_mask,
)
from doraex.operators.doppler import doppler_factor  # noqa: E402
from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel  # noqa: E402
from doraex.workflows.luhman16b_milestone2 import build_luhman16b_geometry  # noqa: E402
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
    _cia_paths,
    _molecule_paths,
)


DEFAULT_SAMPLES = ROOT / "results" / "m7" / "v1_zero_mean_log_w_run" / "samples.npz"
DEFAULT_PRODUCT_DIR = (
    ROOT / "results" / "m7" / "v1_zero_mean_log_w_f64_prod" / "baseline_f64"
)
DEFAULT_OUT_DIR = ROOT / "results" / "m7" / "v1" / "check_linapprox"


def parse_indices(text: str) -> tuple[int, ...]:
    """Parse comma-separated integer indices."""

    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one index is required.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact ExoJAX spectra s(logP0 + q_j) with the M7 v1 "
            "linearized pressure-response spectra on one chip."
        )
    )
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--product-dir", default=str(DEFAULT_PRODUCT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--chip-index", type=int, default=0)
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase-indices", type=parse_indices, default=parse_indices("0,4,8,12"))
    parser.add_argument(
        "--percent-denominator-floor",
        type=float,
        default=0.002,
        help=(
            "Minimum absolute exact perturbation amplitude used when plotting "
            "percentage residuals. Smaller denominators are masked."
        ),
    )
    parser.add_argument("--percent-ylim", type=float, default=80.0)
    parser.add_argument("--ratio-ylim", type=float, default=10.0)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _posterior_median(samples: dict[str, np.ndarray], name: str):
    """Return a posterior median value from an NPZ payload."""

    return np.median(np.asarray(samples[name]), axis=0)


def _median_sample_for_chip(samples: dict[str, np.ndarray], chip_index: int) -> dict[str, np.ndarray]:
    """Return posterior median atmospheric and chip-local nuisance values."""

    chip_indices = [int(value) for value in np.asarray(samples["chip_indices"])]
    chip_position = chip_indices.index(int(chip_index))
    sample = {
        "T0": float(_posterior_median(samples, "T0")),
        "alpha": float(_posterior_median(samples, "alpha")),
        "logg": float(_posterior_median(samples, "logg")),
        "log_vmr_co": float(_posterior_median(samples, "log_vmr_co")),
        "log_vmr_h2o": float(_posterior_median(samples, "log_vmr_h2o")),
        "log_vmr_ch4": float(_posterior_median(samples, "log_vmr_ch4")),
        "log_vmr_hf": float(_posterior_median(samples, "log_vmr_hf")),
        "log_p_cloud": float(_posterior_median(samples, "log_p_cloud")),
        "P": float(_posterior_median(samples, "P")),
        "cosi": float(_posterior_median(samples, "cosi")),
        "v": float(_posterior_median(samples, "v")),
        "q1": float(_posterior_median(samples, "q1")),
        "q2": float(_posterior_median(samples, "q2")),
        "u1": float(_posterior_median(samples, "u1")),
        "u2": float(_posterior_median(samples, "u2")),
        "A": float(_posterior_median(samples, "A")[chip_position]),
        "log_w": np.asarray(_posterior_median(samples, "log_w")[chip_position], dtype=float),
        "sigma_d": float(_posterior_median(samples, "sigma_d")[chip_position]),
        "chip_position": chip_position,
    }
    return sample


def _chip_data_from_samples(samples: dict[str, np.ndarray], chip_index: int) -> Luhman16BChipData:
    """Build a chip data object from arrays embedded in samples.npz."""

    wavelengths = np.asarray(samples[f"wavelengths_chip{chip_index}"], dtype=float)
    flux = np.asarray(samples[f"flux_chip{chip_index}"], dtype=float)
    return Luhman16BChipData(
        wavelengths=wavelengths,
        flux=flux,
        line_profile=np.ones_like(wavelengths),
        obs_times=np.asarray(samples["obs_times"], dtype=float),
        chip_index=int(chip_index),
    )


def _load_q_map(product_dir: Path, chip_index: int, chip_position: int) -> np.ndarray:
    """Load the posterior-mean cloud-pressure perturbation map for one chip."""

    chip_path = product_dir / f"cloud_pressure_perturbation_mean_chip{chip_index}.npy"
    if chip_path.exists():
        return np.asarray(np.load(chip_path), dtype=float)
    joint_path = product_dir / "cloud_pressure_perturbation_mean_joint_by_chip.npy"
    if joint_path.exists():
        return np.asarray(np.load(joint_path)[chip_position], dtype=float)
    raise FileNotFoundError(f"No pressure perturbation map found under {product_dir}")


def _build_column_model(args: argparse.Namespace, chip_data: Luhman16BChipData):
    """Build the ExoJAX local-column model for the selected chip."""

    return Luhman16BPowerLawColumnModel(
        chip_data.wavelengths,
        molecule_paths=_molecule_paths(args.database_dir),
        cia_paths=_cia_paths(args.database_dir),
        opacity_cache_dir=args.opacity_cache_dir,
        parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
        nx=args.nx,
    )


def _local_spectrum_functions(model, sample: dict[str, np.ndarray]):
    """Return JIT-compiled local spectrum and pressure-response functions."""

    def spectrum_at_pressure(log_p_cloud):
        return model.cloudy_at_log_vmrs(
            jnp.asarray(sample["T0"]),
            jnp.asarray(sample["alpha"]),
            jnp.asarray(sample["log_vmr_co"]),
            jnp.asarray(sample["log_vmr_h2o"]),
            jnp.asarray(sample["log_vmr_ch4"]),
            jnp.asarray(sample["log_vmr_hf"]),
            log_p_cloud,
            logg=jnp.asarray(sample["logg"]),
        )

    @jax.jit
    def response(log_p_cloud):
        return jax.jvp(
            spectrum_at_pressure,
            (log_p_cloud,),
            (jnp.ones_like(log_p_cloud),),
        )

    return jax.jit(spectrum_at_pressure), response


def _evaluate_exact_profiles(spectrum_at_pressure, log_p_values: np.ndarray, batch_size: int) -> np.ndarray:
    """Evaluate exact local spectra for all map pressures in small batches."""

    batch_function = jax.jit(jax.vmap(spectrum_at_pressure))
    profiles = []
    for start in range(0, len(log_p_values), batch_size):
        print(
            f"Evaluating exact local spectra {start + 1}-{min(start + batch_size, len(log_p_values))}/"
            f"{len(log_p_values)}",
            flush=True,
        )
        batch = jnp.asarray(log_p_values[start : start + batch_size])
        profiles.append(np.asarray(batch_function(batch).block_until_ready()))
    return np.concatenate(profiles, axis=0)


def _integrate_profiles(
    chip_data: Luhman16BChipData,
    geometry,
    sample: dict[str, np.ndarray],
    profiles_by_pixel,
):
    """Doppler-integrate one local spectrum per surface pixel."""

    profiles_by_pixel = jnp.asarray(profiles_by_pixel)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    phases = jnp.asarray(chip_data.obs_times) / jnp.asarray(sample["P"])
    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    theta = geometry.theta
    phi = geometry.phi

    def one_phase(phase, phase_weight):
        phi_rot = rotate_longitude(phi, phase)
        theta_obs, phi_obs = incline(theta, phi_rot, jnp.pi / 2.0 - inclination)
        vlos = line_of_sight_velocity(jnp.asarray(sample["v"]), inclination, theta, phi_rot)
        factors = doppler_factor(vlos)

        def shift_one(profile, factor):
            return jnp.interp(wavelengths / factor, wavelengths, profile)

        local_profiles = jax.vmap(shift_one)(profiles_by_pixel, factors).T
        mu = projected_mu(theta_obs, phi_obs)
        limb = quadratic_limb_darkening(
            jnp.asarray(sample["u1"]),
            jnp.asarray(sample["u2"]),
            mu,
        )
        pixel_weight = visible_mask(phi_obs) * mu * limb
        return phase_weight * jnp.sum(local_profiles * pixel_weight[None, :], axis=1)

    return jax.jit(jax.vmap(one_phase))(phases, weights)


def _rms(values: np.ndarray) -> float:
    """Return root-mean-square as a Python float."""

    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def _plot_linearization_check(
    out_path: Path,
    wavelengths: np.ndarray,
    local_payload: dict[str, np.ndarray],
    integrated_payload: dict[str, np.ndarray],
    phase_indices: tuple[int, ...],
    summary: dict[str, object],
) -> None:
    """Create the 2x2 local/data-space linearization figure."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    ax_lt, ax_rt = axes[0]
    ax_lb, ax_rb = axes[1]

    local_colors = {"q16": "tab:blue", "q84": "tab:red"}
    for label, color in local_colors.items():
        delta = float(local_payload[f"{label}_delta"])
        exact = local_payload[f"{label}_exact"]
        linear = local_payload[f"{label}_linear"]
        residual = exact - linear
        ax_lt.plot(wavelengths, exact, color=color, lw=1.1, label=f"exact {label} ({delta:+.3f} dex)")
        ax_lt.plot(wavelengths, linear, color=color, lw=1.1, ls="--", label=f"linear {label}")
        ax_lb.plot(wavelengths, residual, color=color, lw=1.0, label=f"{label}")

    exact_delta = integrated_payload["exact_delta"]
    linear_delta = integrated_payload["linear_delta"]
    integrated_residual = exact_delta - linear_delta
    offsets = _phase_offsets(exact_delta, phase_indices)
    cmap = plt.get_cmap("viridis")
    for order, phase_index in enumerate(phase_indices):
        color = cmap(order / max(len(phase_indices) - 1, 1))
        offset = offsets[order]
        ax_rt.plot(
            wavelengths,
            exact_delta[phase_index] + offset,
            color=color,
            lw=1.0,
            label=f"phase {phase_index} exact",
        )
        ax_rt.plot(
            wavelengths,
            linear_delta[phase_index] + offset,
            color=color,
            lw=1.0,
            ls="--",
            label=f"phase {phase_index} linear",
        )
        ax_rb.plot(
            wavelengths,
            integrated_residual[phase_index] + offset * 0.15,
            color=color,
            lw=1.0,
            label=f"phase {phase_index}",
        )

    ax_lt.set_title("Local pressure response")
    ax_rt.set_title("Doppler-integrated map response")
    ax_lb.set_xlabel("Wavelength [A]")
    ax_rb.set_xlabel("Wavelength [A]")
    ax_lt.set_ylabel(r"$s(\log P_0+\Delta)-s(\log P_0)$")
    ax_lb.set_ylabel("exact - linear")
    ax_rt.set_ylabel(r"$F(q)-F_0$ + offset")
    ax_rb.set_ylabel("exact - linear + offset")
    for ax in axes.ravel():
        ax.axhline(0.0, color="0.65", lw=0.7, zorder=0)
        ax.tick_params(direction="in", top=True, right=True)
    ax_lt.legend(fontsize=8, ncol=2, frameon=False)
    ax_rt.legend(fontsize=7, ncol=2, frameon=False)

    metric_text = (
        "chip {chip_index}\n"
        r"$q_{{16,84}}$ = {q16:+.3f}, {q84:+.3f} dex" "\n"
        r"RMS int. residual / RMS exact = {frac:.3e}" "\n"
        r"RMS int. residual / $\sigma_d$ = {noise:.3e}"
    ).format(
        chip_index=summary["chip_index"],
        q16=summary["q_quantiles"]["q16"],
        q84=summary["q_quantiles"]["q84"],
        frac=summary["integrated_residual_rms_over_exact_delta_rms"],
        noise=summary["integrated_residual_rms_over_sigma_d"],
    )
    ax_rt.text(
        0.02,
        0.98,
        metric_text,
        transform=ax_rt.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_linearization_check_percent_residuals(
    out_path: Path,
    wavelengths: np.ndarray,
    local_payload: dict[str, np.ndarray],
    integrated_payload: dict[str, np.ndarray],
    phase_indices: tuple[int, ...],
    summary: dict[str, object],
    denominator_floor: float,
    percent_ylim: float,
) -> None:
    """Create a 2x2 figure with un-offset percentage residual panels."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    ax_lt, ax_rt = axes[0]
    ax_lb, ax_rb = axes[1]

    local_colors = {"q16": "tab:blue", "q84": "tab:red"}
    for label, color in local_colors.items():
        delta = float(local_payload[f"{label}_delta"])
        exact = local_payload[f"{label}_exact"]
        linear = local_payload[f"{label}_linear"]
        ax_lt.plot(wavelengths, exact, color=color, lw=1.1, label=f"exact {label} ({delta:+.3f} dex)")
        ax_lt.plot(wavelengths, linear, color=color, lw=1.1, ls="--", label=f"linear {label}")
        percent = _masked_percent_residual(exact - linear, exact, denominator_floor)
        ax_lb.plot(wavelengths, percent, color=color, lw=1.0, label=f"{label}")

    exact_delta = integrated_payload["exact_delta"]
    linear_delta = integrated_payload["linear_delta"]
    integrated_residual = exact_delta - linear_delta
    cmap = plt.get_cmap("viridis")
    for order, phase_index in enumerate(phase_indices):
        color = cmap(order / max(len(phase_indices) - 1, 1))
        ax_rt.plot(
            wavelengths,
            exact_delta[phase_index],
            color=color,
            lw=1.0,
            label=f"phase {phase_index} exact",
        )
        ax_rt.plot(
            wavelengths,
            linear_delta[phase_index],
            color=color,
            lw=1.0,
            ls="--",
            label=f"phase {phase_index} linear",
        )
        percent = _masked_percent_residual(
            integrated_residual[phase_index],
            exact_delta[phase_index],
            denominator_floor,
        )
        ax_rb.plot(wavelengths, percent, color=color, lw=1.0, label=f"phase {phase_index}")

    ax_lt.set_title("Local pressure response")
    ax_rt.set_title("Doppler-integrated map response")
    ax_lb.set_xlabel("Wavelength [A]")
    ax_rb.set_xlabel("Wavelength [A]")
    ax_lt.set_ylabel(r"$s(\log P_0+\Delta)-s(\log P_0)$")
    ax_rt.set_ylabel(r"$F(q)-F_0$")
    ax_lb.set_ylabel("100 x (exact - linear) / exact [%]")
    ax_rb.set_ylabel("100 x (exact - linear) / exact [%]")
    ax_lb.set_ylim(-percent_ylim, percent_ylim)
    ax_rb.set_ylim(-percent_ylim, percent_ylim)
    for ax in axes.ravel():
        ax.axhline(0.0, color="0.65", lw=0.7, zorder=0)
        ax.tick_params(direction="in", top=True, right=True)
    ax_lt.legend(fontsize=8, ncol=2, frameon=False)
    ax_rt.legend(fontsize=7, ncol=2, frameon=False)

    metric_text = (
        "chip {chip_index}\n"
        r"$q_{{16,84}}$ = {q16:+.3f}, {q84:+.3f} dex" "\n"
        r"RMS int. residual / RMS exact = {frac:.3e}" "\n"
        r"percent denom. floor = {floor:.3g}"
    ).format(
        chip_index=summary["chip_index"],
        q16=summary["q_quantiles"]["q16"],
        q84=summary["q_quantiles"]["q84"],
        frac=summary["integrated_residual_rms_over_exact_delta_rms"],
        floor=denominator_floor,
    )
    ax_rt.text(
        0.02,
        0.98,
        metric_text,
        transform=ax_rt.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_linearization_check_ratio_residuals(
    out_path: Path,
    wavelengths: np.ndarray,
    base_profile: np.ndarray,
    local_payload: dict[str, np.ndarray],
    integrated_payload: dict[str, np.ndarray],
    phase_indices: tuple[int, ...],
    summary: dict[str, object],
    ratio_ylim: float,
) -> None:
    """Create a 2x2 figure with residuals as linear/exact - 1."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    ax_lt, ax_rt = axes[0]
    ax_lb, ax_rb = axes[1]

    local_colors = {"q16": "tab:blue", "q84": "tab:red"}
    local_offsets = {"q16": -0.35, "q84": 0.35}
    for label, color in local_colors.items():
        delta = float(local_payload[f"{label}_delta"])
        exact = base_profile + local_payload[f"{label}_exact"]
        linear = base_profile + local_payload[f"{label}_linear"]
        offset = local_offsets[label]
        ax_lt.plot(
            wavelengths,
            exact + offset,
            color=color,
            lw=1.1,
            alpha=0.6,
            label=f"exact {label} ({delta:+.3f} dex)",
        )
        ax_lt.plot(
            wavelengths,
            linear + offset,
            color=color,
            lw=1.1,
            ls="--",
            label=f"linear {label}",
        )
        ax_lb.plot(
            wavelengths,
            100.0 * (linear / exact - 1.0),
            color=color,
            lw=1.0,
            label=f"{label}",
        )

    exact_model = integrated_payload["exact_model"]
    linear_model = integrated_payload["linear_model"]
    cmap = plt.get_cmap("viridis")
    offsets = _positive_phase_offsets(exact_model, phase_indices, scale=0.32)
    for order, phase_index in enumerate(phase_indices):
        color = cmap(order / max(len(phase_indices) - 1, 1))
        offset = offsets[order]
        ax_rt.plot(
            wavelengths,
            exact_model[phase_index] + offset,
            color=color,
            lw=1.0,
            alpha=0.6,
            label=f"phase {phase_index} exact",
        )
        ax_rt.plot(
            wavelengths,
            linear_model[phase_index] + offset,
            color=color,
            lw=1.0,
            ls="--",
            label=f"phase {phase_index} linear",
        )
        ax_rb.plot(
            wavelengths,
            100.0 * (linear_model[phase_index] / exact_model[phase_index] - 1.0),
            color=color,
            lw=1.0,
            label=f"phase {phase_index}",
        )

    ax_lt.set_title("Local pressure-shifted spectrum")
    ax_rt.set_title("Doppler-integrated map spectrum")
    ax_lb.set_xlabel("Wavelength [A]")
    ax_rb.set_xlabel("Wavelength [A]")
    ax_lt.set_ylabel(r"$s(\log P)$ + offset")
    ax_rt.set_ylabel(r"$F$ + offset")
    ax_lb.set_ylabel(r"$100(F_\mathrm{lin}/F_\mathrm{exact}-1)$ [%]")
    ax_rb.set_ylabel(r"$100(F_\mathrm{lin}/F_\mathrm{exact}-1)$ [%]")
    ax_rt.set_ylim(0.4, 1.5)
    ax_lb.set_ylim(-ratio_ylim, ratio_ylim)
    ax_rb.set_ylim(-ratio_ylim, ratio_ylim)
    for ax in axes.ravel():
        ax.axhline(0.0, color="0.65", lw=0.7, zorder=0)
        ax.tick_params(direction="in", top=True, right=True)
    ax_lt.legend(fontsize=8, ncol=2, frameon=False)
    ax_rt.legend(fontsize=7, ncol=2, frameon=False)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _masked_percent_residual(residual: np.ndarray, reference: np.ndarray, floor: float) -> np.ndarray:
    """Return percentage residuals, masking near-zero reference values."""

    residual = np.asarray(residual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    percent = np.full_like(residual, np.nan)
    mask = np.abs(reference) >= floor
    percent[mask] = 100.0 * residual[mask] / reference[mask]
    return percent


def _phase_offsets(values: np.ndarray, phase_indices: tuple[int, ...]) -> np.ndarray:
    """Return compact vertical offsets for selected phase traces."""

    selected = np.asarray(values)[list(phase_indices)]
    scale = float(np.nanpercentile(np.abs(selected), 98))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    center = 0.5 * (len(phase_indices) - 1)
    return (np.arange(len(phase_indices)) - center) * 2.4 * scale


def _positive_phase_offsets(
    values: np.ndarray,
    phase_indices: tuple[int, ...],
    scale: float = 0.32,
) -> np.ndarray:
    """Return offsets with phase0 at zero and later plotted phases shifted upward."""

    selected = np.asarray(values)[list(phase_indices)]
    amplitude = float(np.nanpercentile(selected, 95) - np.nanpercentile(selected, 5))
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        amplitude = 1.0
    return np.arange(len(phase_indices)) * scale * amplitude


def main() -> None:
    """Run the linearization check and write artifacts."""

    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = dict(np.load(args.samples, allow_pickle=False))
    sample = _median_sample_for_chip(samples, args.chip_index)
    chip_data = _chip_data_from_samples(samples, args.chip_index)
    geometry = build_luhman16b_geometry(nside=int(np.asarray(samples["nside"])))
    q_map = _load_q_map(Path(args.product_dir), args.chip_index, sample["chip_position"])
    print(
        f"Loaded chip {args.chip_index}: {chip_data.flux.shape[0]} phases, "
        f"{chip_data.flux.shape[1]} wavelengths, {len(q_map)} map pixels",
        flush=True,
    )

    setup_start = time.time()
    print("Building ExoJAX column model...", flush=True)
    model = _build_column_model(args, chip_data)
    spectrum_at_pressure, response = _local_spectrum_functions(model, sample)
    setup_seconds = time.time() - setup_start
    print(f"Built ExoJAX column model in {setup_seconds:.2f} s", flush=True)

    response_start = time.time()
    print("Evaluating local baseline spectrum and pressure derivative...", flush=True)
    base_profile, pressure_derivative = response(jnp.asarray(sample["log_p_cloud"]))
    base_profile.block_until_ready()
    pressure_derivative.block_until_ready()
    response_seconds = time.time() - response_start
    print(f"Evaluated local pressure response in {response_seconds:.2f} s", flush=True)
    base_profile = np.asarray(base_profile)
    pressure_derivative = np.asarray(pressure_derivative)

    q_quantiles = np.quantile(q_map, [0.05, 0.16, 0.50, 0.84, 0.95])
    local_payload = {}
    for label, delta in (("q16", q_quantiles[1]), ("q84", q_quantiles[3])):
        exact_profile = np.asarray(
            spectrum_at_pressure(jnp.asarray(sample["log_p_cloud"] + delta)).block_until_ready()
        )
        exact_delta = exact_profile - base_profile
        linear_delta = pressure_derivative * delta
        local_payload[f"{label}_delta"] = np.asarray(delta)
        local_payload[f"{label}_exact"] = exact_delta
        local_payload[f"{label}_linear"] = linear_delta

    exact_start = time.time()
    print("Evaluating exact local spectra for the posterior-mean pressure map...", flush=True)
    exact_profiles = _evaluate_exact_profiles(
        spectrum_at_pressure,
        sample["log_p_cloud"] + q_map,
        batch_size=args.batch_size,
    )
    exact_profile_seconds = time.time() - exact_start
    linear_profiles = base_profile[None, :] + q_map[:, None] * pressure_derivative[None, :]
    base_profiles = np.broadcast_to(base_profile[None, :], exact_profiles.shape)

    integrate_start = time.time()
    print("Doppler-integrating base, exact, and linearized profile sets...", flush=True)
    raw_base = np.asarray(_integrate_profiles(chip_data, geometry, sample, base_profiles))
    raw_exact = np.asarray(_integrate_profiles(chip_data, geometry, sample, exact_profiles))
    raw_linear = np.asarray(_integrate_profiles(chip_data, geometry, sample, linear_profiles))
    integrate_seconds = time.time() - integrate_start
    print(f"Finished Doppler integration in {integrate_seconds:.2f} s", flush=True)
    norm = sample["A"] * float(np.mean(raw_base))
    base_model = raw_base / norm
    exact_model = raw_exact / norm
    linear_model = raw_linear / norm
    exact_delta = exact_model - base_model
    linear_delta = linear_model - base_model
    integrated_residual = exact_delta - linear_delta

    q_names = ("q05", "q16", "q50", "q84", "q95")
    q_summary = {name: float(value) for name, value in zip(q_names, q_quantiles)}
    summary = {
        "chip_index": int(args.chip_index),
        "samples": str(args.samples),
        "product_dir": str(args.product_dir),
        "out_dir": str(out_dir),
        "x64": bool(args.x64),
        "nx": int(args.nx),
        "batch_size": int(args.batch_size),
        "phase_indices_plotted": [int(value) for value in args.phase_indices],
        "setup_seconds": setup_seconds,
        "response_seconds": response_seconds,
        "exact_profile_seconds": exact_profile_seconds,
        "integrate_seconds": integrate_seconds,
        "log_p_cloud": float(sample["log_p_cloud"]),
        "sigma_d": float(sample["sigma_d"]),
        "A": float(sample["A"]),
        "q_min": float(np.min(q_map)),
        "q_max": float(np.max(q_map)),
        "q_quantiles": q_summary,
        "local_q16_residual_rms_over_exact_delta_rms": _rms(
            local_payload["q16_exact"] - local_payload["q16_linear"]
        )
        / max(_rms(local_payload["q16_exact"]), 1.0e-30),
        "local_q84_residual_rms_over_exact_delta_rms": _rms(
            local_payload["q84_exact"] - local_payload["q84_linear"]
        )
        / max(_rms(local_payload["q84_exact"]), 1.0e-30),
        "integrated_exact_delta_rms": _rms(exact_delta),
        "integrated_linear_delta_rms": _rms(linear_delta),
        "integrated_residual_rms": _rms(integrated_residual),
        "integrated_residual_rms_over_exact_delta_rms": _rms(integrated_residual)
        / max(_rms(exact_delta), 1.0e-30),
        "integrated_residual_rms_over_sigma_d": _rms(integrated_residual)
        / max(float(sample["sigma_d"]), 1.0e-30),
        "integrated_residual_max_abs": float(np.max(np.abs(integrated_residual))),
        "integrated_ratio_residual_rms": _rms(linear_model / exact_model - 1.0),
        "integrated_ratio_residual_q05": float(
            np.quantile(linear_model / exact_model - 1.0, 0.05)
        ),
        "integrated_ratio_residual_q50": float(
            np.quantile(linear_model / exact_model - 1.0, 0.50)
        ),
        "integrated_ratio_residual_q95": float(
            np.quantile(linear_model / exact_model - 1.0, 0.95)
        ),
        "percent_denominator_floor": float(args.percent_denominator_floor),
        "percent_ylim": float(args.percent_ylim),
        "ratio_ylim": float(args.ratio_ylim),
    }

    arrays_path = out_dir / "m7_v1_chip0_linearization_check_arrays.npz"
    np.savez(
        arrays_path,
        wavelengths=np.asarray(chip_data.wavelengths),
        obs_times=np.asarray(chip_data.obs_times),
        q_map=q_map,
        base_profile=base_profile,
        pressure_derivative=pressure_derivative,
        exact_profiles=exact_profiles,
        linear_profiles=linear_profiles,
        base_model=base_model,
        exact_model=exact_model,
        linear_model=linear_model,
        exact_delta=exact_delta,
        linear_delta=linear_delta,
        integrated_residual=integrated_residual,
        **local_payload,
    )
    summary["arrays_path"] = str(arrays_path)

    figure_path = out_dir / "m7_v1_chip0_linearization_check.png"
    _plot_linearization_check(
        figure_path,
        np.asarray(chip_data.wavelengths),
        local_payload,
        {"exact_delta": exact_delta, "linear_delta": linear_delta},
        args.phase_indices,
        summary,
    )
    summary["figure_path"] = str(figure_path)

    percent_figure_path = out_dir / "m7_v1_chip0_linearization_check_percent_residuals.png"
    _plot_linearization_check_percent_residuals(
        percent_figure_path,
        np.asarray(chip_data.wavelengths),
        local_payload,
        {"exact_delta": exact_delta, "linear_delta": linear_delta},
        args.phase_indices,
        summary,
        args.percent_denominator_floor,
        args.percent_ylim,
    )
    summary["percent_figure_path"] = str(percent_figure_path)

    ratio_figure_path = out_dir / "m7_v1_chip0_linearization_check_ratio_residuals.png"
    _plot_linearization_check_ratio_residuals(
        ratio_figure_path,
        np.asarray(chip_data.wavelengths),
        base_profile,
        local_payload,
        {"exact_model": exact_model, "linear_model": linear_model},
        args.phase_indices,
        summary,
        args.ratio_ylim,
    )
    summary["ratio_figure_path"] = str(ratio_figure_path)

    summary_path = out_dir / "m7_v1_chip0_linearization_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
