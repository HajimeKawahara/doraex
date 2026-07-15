"""Generate an observation-aware atmospheric-response eigenbasis for E2 v1."""

import argparse
import json
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

from doraex.operators.design_matrix import linear_profile_operator_from_times  # noqa: E402
from doraex.priors.spherical_gp import (  # noqa: E402
    squared_exponential_covariance,
    zero_mean_covariance_factor,
)
from doraex.workflows.luhman16b_milestone2 import build_luhman16b_geometry  # noqa: E402
from generate_e1_v1_basis import (  # noqa: E402
    ATMOSPHERE_NAMES,
    DEFAULT_PARAMETER_NAMES,
    DEFAULT_PARAMETER_SCALES,
    _basis_center,
    _build_chip_jacobian,
    _load_chip_data_from_samples,
    parse_chips,
    parse_names,
    parse_scales,
)


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description="Generate E2 v1 observation-aware atmospheric eigenbasis."
    )
    parser.add_argument(
        "--samples",
        default=str(ROOT / "results" / "m7" / "v1_zero_mean_log_w_run" / "samples.npz"),
        help="Posterior sample archive used for eta, nuisance, noise, and data grids.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "e2" / "v1_basis" / "eigen_response_basis.npz"),
    )
    parser.add_argument("--chip-indices", type=parse_chips, default=None)
    parser.add_argument(
        "--parameter-names",
        type=parse_names,
        default=DEFAULT_PARAMETER_NAMES,
        help="Comma-separated atmospheric parameters included in the basis.",
    )
    parser.add_argument(
        "--parameter-scales",
        type=parse_scales,
        default=None,
        help="Comma-separated perturbation scales matching --parameter-names.",
    )
    parser.add_argument("--mode-count", type=int, default=2)
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--ell", type=float, default=0.4)
    parser.add_argument("--gp-jitter", type=float, default=5.0e-6)
    parser.add_argument("--noise-jitter", type=float, default=1.0e-6)
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _median_array(samples, name):
    """Return posterior median preserving non-sample dimensions."""

    value = np.asarray(samples[name])
    if value.ndim == 0:
        return value
    return np.median(value, axis=0)


def _median_scalar(samples, name, default):
    """Return scalar posterior median or a default."""

    if name not in samples.files:
        return float(default)
    return float(np.asarray(_median_array(samples, name)))


def _kipping_q_to_u(q1, q2):
    """Convert Kipping q coefficients to quadratic limb darkening."""

    sqrt_q1 = np.sqrt(q1)
    return 2.0 * sqrt_q1 * q2, sqrt_q1 * (1.0 - 2.0 * q2)


def _stabilize_eigenvector_signs(vectors):
    """Make eigenvector signs deterministic."""

    vectors = np.array(vectors, copy=True)
    for col in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, col])))
        if vectors[pivot, col] < 0.0:
            vectors[:, col] *= -1.0
    return vectors


def _build_observation_designs(
    samples,
    chip_data_list,
    spectra,
    jacobians,
    parameter_scales,
    geometry,
    ell,
    gp_jitter,
    noise_jitter,
):
    """Build whitened prior-propagated designs for each physical direction."""

    n_parameter = len(parameter_scales)
    sample_chip_indices = [int(v) for v in np.asarray(samples["chip_indices"])]
    sample_chip_positions = {
        chip_index: pos for pos, chip_index in enumerate(sample_chip_indices)
    }
    obs_times = jnp.asarray(chip_data_list[0].obs_times)
    period = _median_scalar(samples, "P", 4.83)
    cosi = _median_scalar(samples, "cosi", 0.485)
    vrot = _median_scalar(samples, "v", 31.2)
    q1 = _median_scalar(samples, "q1", 0.81)
    q2 = _median_scalar(samples, "q2", 0.59)
    inclination = np.arccos(cosi)
    u1, u2 = _kipping_q_to_u(q1, q2)
    a_median = np.asarray(_median_array(samples, "A"), dtype=float)
    log_w_median = np.asarray(_median_array(samples, "log_w"), dtype=float)
    sigma_d_median = np.asarray(_median_array(samples, "sigma_d"), dtype=float)

    unit_covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(1.0),
        jnp.asarray(ell),
    )
    unit_factor = zero_mean_covariance_factor(
        unit_covariance,
        jitter=jnp.asarray(gp_jitter),
    )

    propagated = []
    for parameter_index in range(n_parameter):
        rows = []
        for chip_position, chip_data in enumerate(chip_data_list):
            chip_index = chip_data.chip_index
            sample_chip_position = sample_chip_positions[chip_index]
            baseline, contrast_matrix = linear_profile_operator_from_times(
                geometry.theta,
                geometry.phi,
                jnp.asarray(vrot),
                jnp.asarray(inclination),
                jnp.asarray(u1),
                jnp.asarray(u2),
                obs_times,
                jnp.asarray(period),
                jnp.asarray(chip_data.wavelengths),
                jnp.asarray(spectra[chip_position]),
                jnp.asarray(jacobians[chip_position][:, parameter_index])
                * jnp.asarray(parameter_scales[parameter_index]),
                weights=jnp.exp(jnp.asarray(log_w_median[sample_chip_position])),
            )
            norm = jnp.asarray(a_median[sample_chip_position]) * jnp.mean(baseline)
            whiten = jnp.sqrt(
                jnp.asarray(sigma_d_median[sample_chip_position]) ** 2
                + jnp.asarray(noise_jitter)
            )
            rows.append((contrast_matrix / norm / whiten).astype(jnp.float32))
        design = jnp.concatenate(rows, axis=0)
        projected = design @ unit_factor.astype(jnp.float32)
        propagated.append(np.asarray(projected.block_until_ready(), dtype=np.float32))
        print(
            json.dumps(
                {
                    "parameter_index": parameter_index,
                    "propagated_shape": list(propagated[-1].shape),
                    "propagated_rms": float(np.sqrt(np.mean(propagated[-1] ** 2))),
                }
            ),
            flush=True,
        )
    return propagated


def main():
    """Generate and save the E2 v1 observation-aware eigenbasis."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    samples = np.load(args.samples, allow_pickle=False)
    chip_indices = (
        tuple(int(v) for v in np.asarray(samples["chip_indices"]))
        if args.chip_indices is None
        else tuple(args.chip_indices)
    )
    nside = int(np.asarray(samples["nside"])) if args.nside is None else args.nside
    parameter_names = tuple(args.parameter_names)
    if args.parameter_scales is None:
        parameter_scales = tuple(DEFAULT_PARAMETER_SCALES[name] for name in parameter_names)
    else:
        parameter_scales = tuple(args.parameter_scales)
    if len(parameter_scales) != len(parameter_names):
        raise ValueError("--parameter-scales must match --parameter-names length.")
    if not 1 <= args.mode_count <= len(parameter_names):
        raise ValueError("--mode-count must be between 1 and the number of parameters.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    center = _basis_center(samples)
    chip_data_list = _load_chip_data_from_samples(samples, chip_indices)
    geometry = build_luhman16b_geometry(nside=nside)

    start = time.time()
    spectra = []
    jacobians = []
    for chip_data in chip_data_list:
        jacobian, spectrum = _build_chip_jacobian(
            args,
            chip_data,
            center,
            parameter_names,
        )
        jacobians.append(np.asarray(jacobian))
        spectra.append(np.asarray(spectrum))

    propagated = _build_observation_designs(
        samples,
        chip_data_list,
        spectra,
        jacobians,
        np.asarray(parameter_scales, dtype=float),
        geometry,
        args.ell,
        args.gp_jitter,
        args.noise_jitter,
    )
    n_parameter = len(parameter_names)
    observation_information = np.empty((n_parameter, n_parameter), dtype=np.float64)
    for i in range(n_parameter):
        for j in range(i, n_parameter):
            value = float(np.vdot(propagated[i], propagated[j]))
            observation_information[i, j] = value
            observation_information[j, i] = value
    observation_information = 0.5 * (
        observation_information + observation_information.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(observation_information)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _stabilize_eigenvector_signs(eigenvectors[:, order])
    selected_v = eigenvectors[:, : args.mode_count]
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))

    local_scaled_response = np.concatenate(
        [
            jacobian * np.asarray(parameter_scales, dtype=float)[None, :]
            for jacobian in jacobians
        ],
        axis=0,
    )
    payload = {
        "parameter_names": np.asarray(parameter_names),
        "parameter_scales": np.asarray(parameter_scales, dtype=float),
        "atmosphere_names": np.asarray(ATMOSPHERE_NAMES),
        "basis_center": np.asarray([center[name] for name in ATMOSPHERE_NAMES], dtype=float),
        "chip_indices": np.asarray(chip_indices, dtype=int),
        "mode_count": np.asarray(args.mode_count, dtype=int),
        "singular_values": singular_values,
        "v_matrix": eigenvectors,
        "selected_v": selected_v,
        "observation_information": observation_information,
        "local_scaled_response": local_scaled_response,
        "ell": np.asarray(args.ell, dtype=float),
        "nside": np.asarray(nside, dtype=int),
    }
    for chip_position, chip_index in enumerate(chip_indices):
        payload[f"spectrum_chip{chip_index}"] = spectra[chip_position]
        payload[f"jacobian_chip{chip_index}"] = jacobians[chip_position]
        payload[f"eigenspectra_chip{chip_index}"] = (
            jacobians[chip_position] * np.asarray(parameter_scales)[None, :]
        ) @ selected_v
    np.savez(out_path, **payload)
    positive_eigenvalues = np.maximum(eigenvalues, 0.0)
    variance_fractions = positive_eigenvalues / np.sum(positive_eigenvalues)
    metadata = {
        "samples": str(args.samples),
        "out": str(out_path),
        "chip_indices": list(chip_indices),
        "nside": nside,
        "ell": args.ell,
        "parameter_names": list(parameter_names),
        "parameter_scales": list(parameter_scales),
        "mode_count": args.mode_count,
        "singular_values": singular_values.tolist(),
        "variance_fractions": variance_fractions.tolist(),
        "selected_v": selected_v.tolist(),
        "basis_center": center,
        "elapsed_seconds": time.time() - start,
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
