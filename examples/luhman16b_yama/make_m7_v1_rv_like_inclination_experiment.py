"""Measure rotation-period RV-like curves from the M7/v1 cloud map.

This deliberately small experiment reuses the exact local spectra saved by
the M7/v1 cloud-pressure linearization check.  It views the same posterior-
mean cloud map at several inclinations, Doppler-integrates the visible disk,
and measures the apparent velocity of every phase against the phase-mean
empirical template.

The resulting velocity is an ``RV-like`` line-shape projection.  No companion
or center-of-mass motion is injected.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_ARRAYS = (
    ROOT
    / "results"
    / "m7"
    / "v1_zero_mean_log_w_f64_prod"
    / "baseline_f64"
    / "products"
    / "linearization_check"
    / "m7_v1_chip0_linearization_check_arrays.npz"
)
DEFAULT_OUT_DIR = ROOT / "results" / "m7" / "v1_rv_like_inclination"
SPEED_OF_LIGHT_KMS = 299792.458


def parse_inclinations(text: str) -> list[float]:
    """Parse a comma-separated list of inclinations in degrees."""

    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one inclination is required.")
    if any(value < 0.0 or value > 90.0 for value in values):
        raise argparse.ArgumentTypeError("Inclinations must lie between 0 and 90 degrees.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--inclinations-deg",
        type=parse_inclinations,
        default=parse_inclinations("5,10,15,20"),
    )
    parser.add_argument("--phase-count", type=int, default=64)
    parser.add_argument("--period-hours", type=float, default=4.83)
    parser.add_argument("--vrot-kms", type=float, default=31.2)
    parser.add_argument("--q1", type=float, default=0.81)
    parser.add_argument("--q2", type=float, default=0.59)
    parser.add_argument(
        "--edge-trim",
        type=int,
        default=32,
        help="Ignore this many wavelength pixels at both interpolation edges in the RV fit.",
    )
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def infer_nside(pixel_count: int) -> int:
    """Infer and validate the HEALPix nside from a pixel count."""

    nside = int(round(np.sqrt(pixel_count / 12.0)))
    if 12 * nside**2 != pixel_count:
        raise ValueError(f"Pixel count {pixel_count} is not a valid HEALPix size.")
    return nside


def measure_template_rvs(
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    *,
    edge_trim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit each spectrum with a continuum, template, and template RV derivative.

    The phase-mean spectrum is the empirical template.  For a small velocity
    shift, ``T(lambda / D)`` is represented by ``T + v dT/dv``.  A quadratic
    continuum and a free line-depth scale are fitted simultaneously.  This is
    a compact approximation to a conventional template-matching RV analysis.
    """

    wavelengths = np.asarray(wavelengths, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    spectra = spectra / np.mean(spectra, axis=1, keepdims=True)
    template = np.mean(spectra, axis=0)
    derivative = (
        -wavelengths
        / SPEED_OF_LIGHT_KMS
        * np.gradient(template, wavelengths)
    )
    coordinate = np.linspace(-1.0, 1.0, wavelengths.size)
    template_lines = template - 1.0

    if edge_trim < 0 or 2 * edge_trim >= wavelengths.size:
        raise ValueError("edge_trim must leave at least one wavelength pixel.")
    fit_slice = slice(edge_trim, -edge_trim if edge_trim else None)
    design = np.column_stack(
        [
            np.ones(wavelengths.size),
            coordinate,
            coordinate**2,
            template_lines,
            derivative,
        ]
    )[fit_slice]

    velocities = []
    orthogonal_rms = []
    for spectrum in spectra:
        coefficients, *_ = np.linalg.lstsq(design, spectrum[fit_slice], rcond=None)
        line_scale = coefficients[-2]
        if abs(line_scale) < 1.0e-12:
            raise RuntimeError("The empirical template has no measurable line contrast.")
        velocities.append(coefficients[-1] / line_scale)
        residual = spectrum[fit_slice] - design @ coefficients
        orthogonal_rms.append(float(np.sqrt(np.mean(residual**2))))

    velocities = np.asarray(velocities)
    velocities -= np.mean(velocities)
    return velocities, np.asarray(orthogonal_rms), template


def integrate_exact_spectra(
    wavelengths: np.ndarray,
    profiles_by_pixel: np.ndarray,
    *,
    inclinations_deg: list[float],
    phases: np.ndarray,
    vrot_kms: float,
    q1: float,
    q2: float,
    x64: bool,
) -> dict[float, np.ndarray]:
    """Doppler-integrate one exact local spectrum per surface pixel."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", x64)

    from doraex.geometry.healpix import healpix_pixel_angles
    from doraex.geometry.limb_darkening import (
        kipping_q_to_u,
        quadratic_limb_darkening,
    )
    from doraex.geometry.rotation import (
        incline,
        line_of_sight_velocity,
        projected_mu,
        rotate_longitude,
        visible_mask,
    )
    from doraex.operators.doppler import doppler_factor

    nside = infer_nside(profiles_by_pixel.shape[0])
    theta, phi = healpix_pixel_angles(nside)
    wavelengths_jax = jnp.asarray(wavelengths)
    profiles_jax = jnp.asarray(profiles_by_pixel)
    u1, u2 = kipping_q_to_u(jnp.asarray(q1), jnp.asarray(q2))

    @jax.jit
    def integrate_one(phase, inclination):
        phi_rot = rotate_longitude(phi, phase)
        theta_observer, phi_observer = incline(
            theta,
            phi_rot,
            jnp.pi / 2.0 - inclination,
        )
        velocity = line_of_sight_velocity(vrot_kms, inclination, theta, phi_rot)
        factors = doppler_factor(velocity)

        def shift_one(profile, factor):
            return jnp.interp(wavelengths_jax / factor, wavelengths_jax, profile)

        local_profiles = jax.vmap(shift_one)(profiles_jax, factors).T
        mu = projected_mu(theta_observer, phi_observer)
        limb = quadratic_limb_darkening(u1, u2, mu)
        weights = visible_mask(phi_observer) * mu * limb
        return jnp.sum(local_profiles * weights[None, :], axis=1)

    output = {}
    for inclination_deg in inclinations_deg:
        inclination = jnp.asarray(np.deg2rad(inclination_deg))
        spectra = [
            np.asarray(integrate_one(jnp.asarray(phase), inclination).block_until_ready())
            for phase in phases
        ]
        output[inclination_deg] = np.asarray(spectra)
    return output


def write_outputs(
    args: argparse.Namespace,
    phases: np.ndarray,
    results: dict[float, dict[str, object]],
    *,
    input_metadata: dict[str, object],
) -> None:
    """Write CSV, JSON, and publication-friendly quick-look figures."""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "rv_like_curves.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["inclination_deg", "phase", "time_hours", "rv_like_ms"],
        )
        writer.writeheader()
        for inclination_deg, result in results.items():
            for phase, velocity in zip(phases, result["rv_kms"]):
                writer.writerow(
                    {
                        "inclination_deg": inclination_deg,
                        "phase": float(phase),
                        "time_hours": float(phase * args.period_hours),
                        "rv_like_ms": float(1000.0 * velocity),
                    }
                )

    summary = {
        "description": "M7/v1 exact-spectrum rotation-induced RV-like experiment",
        "companion_injected": False,
        "input_arrays": str(args.arrays),
        "chip_index": input_metadata["chip_index"],
        "nside": input_metadata["nside"],
        "wavelength_min": input_metadata["wavelength_min"],
        "wavelength_max": input_metadata["wavelength_max"],
        "phase_count": args.phase_count,
        "period_hours": args.period_hours,
        "vrot_kms": args.vrot_kms,
        "q1": args.q1,
        "q2": args.q2,
        "edge_trim": args.edge_trim,
        "rv_method": "phase-mean empirical template plus first-order RV derivative",
        "results": [
            {
                "inclination_deg": inclination_deg,
                "vsini_kms": result["vsini_kms"],
                "rv_semi_amplitude_ms": result["semi_amplitude_ms"],
                "rv_rms_ms": result["rms_ms"],
                "maximum_rv_ms": result["maximum_rv_ms"],
                "maximum_phase": result["maximum_phase"],
                "maximum_time_hours": result["maximum_time_hours"],
                "minimum_rv_ms": result["minimum_rv_ms"],
                "minimum_phase": result["minimum_phase"],
                "minimum_time_hours": result["minimum_time_hours"],
                "mean_non_rv_shape_rms": result["mean_orthogonal_rms"],
            }
            for inclination_deg, result in results.items()
        ],
        "curve_csv": str(csv_path),
    }
    summary_path = args.out_dir / "rv_like_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, max(len(results), 2))
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for color, (inclination_deg, result) in zip(colors, results.items()):
        closed_times = np.append(phases, 1.0) * args.period_hours
        closed_velocities = 1000.0 * np.append(
            result["rv_kms"],
            result["rv_kms"][0],
        )
        axis.plot(
            closed_times,
            closed_velocities,
            color=color,
            lw=2.5,
            label=rf"$i={inclination_deg:g}^\circ$",
        )
    axis.axhline(0.0, color="0.65", lw=1.0)
    axis.set_ylabel(r"RV-like shift [m s$^{-1}$]", fontsize=17)
    axis.set_xlabel("Time over one rotation [hr]", fontsize=17)
    axis.set_xlim(0.0, args.period_hours)
    axis.tick_params(axis="both", labelsize=15)
    axis.legend(frameon=False, fontsize=15)

    figure_path = args.out_dir / "cloud_pressure_rv_like_inclination.png"
    fig.savefig(figure_path, dpi=220)
    fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    """Run the inclination experiment."""

    args = parse_args()
    if args.phase_count < 4:
        raise ValueError("phase_count must be at least four.")
    if not args.arrays.exists():
        raise FileNotFoundError(args.arrays)

    with np.load(args.arrays, allow_pickle=False) as arrays:
        wavelengths = np.asarray(arrays["wavelengths"], dtype=float)
        profiles_by_pixel = np.asarray(arrays["exact_profiles"], dtype=float)

    phases = np.arange(args.phase_count, dtype=float) / args.phase_count
    spectra_by_inclination = integrate_exact_spectra(
        wavelengths,
        profiles_by_pixel,
        inclinations_deg=args.inclinations_deg,
        phases=phases,
        vrot_kms=args.vrot_kms,
        q1=args.q1,
        q2=args.q2,
        x64=args.x64,
    )

    results: dict[float, dict[str, object]] = {}
    for inclination_deg in args.inclinations_deg:
        velocities, orthogonal_rms, _ = measure_template_rvs(
            wavelengths,
            spectra_by_inclination[inclination_deg],
            edge_trim=args.edge_trim,
        )
        results[inclination_deg] = {
            "rv_kms": velocities,
            "vsini_kms": float(args.vrot_kms * np.sin(np.deg2rad(inclination_deg))),
            "semi_amplitude_ms": float(500.0 * (np.max(velocities) - np.min(velocities))),
            "rms_ms": float(1000.0 * np.std(velocities)),
            "maximum_rv_ms": float(1000.0 * np.max(velocities)),
            "maximum_phase": float(phases[np.argmax(velocities)]),
            "maximum_time_hours": float(
                args.period_hours * phases[np.argmax(velocities)]
            ),
            "minimum_rv_ms": float(1000.0 * np.min(velocities)),
            "minimum_phase": float(phases[np.argmin(velocities)]),
            "minimum_time_hours": float(
                args.period_hours * phases[np.argmin(velocities)]
            ),
            "mean_orthogonal_rms": float(np.mean(orthogonal_rms)),
        }

    metadata = {
        "chip_index": 0,
        "nside": infer_nside(profiles_by_pixel.shape[0]),
        "wavelength_min": float(np.min(wavelengths)),
        "wavelength_max": float(np.max(wavelengths)),
    }
    write_outputs(args, phases, results, input_metadata=metadata)

    for inclination_deg, result in results.items():
        print(
            f"i={inclination_deg:5.1f} deg  "
            f"vsini={result['vsini_kms']:6.3f} km/s  "
            f"K_RV-like={result['semi_amplitude_ms']:7.2f} m/s  "
            f"RMS={result['rms_ms']:7.2f} m/s"
        )
    print(f"Wrote products to {args.out_dir}")


if __name__ == "__main__":
    main()
