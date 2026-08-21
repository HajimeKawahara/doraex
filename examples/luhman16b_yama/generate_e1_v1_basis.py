"""Generate the fixed atmospheric-response eigenbasis for E1 v1."""

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

from doraex.data.luhman16b import Luhman16BChipData  # noqa: E402
from doraex.operators.doppler import doppler_padded_wavelengths  # noqa: E402
from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel  # noqa: E402
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
    _cia_paths,
    _molecule_paths,
)


DEFAULT_PARAMETER_NAMES = (
    "log_p_cloud",
    "T0",
    "alpha",
    "log_vmr_co",
    "log_vmr_h2o",
)
DEFAULT_PARAMETER_SCALES = {
    "log_p_cloud": 0.20,
    "T0": 50.0,
    "alpha": 0.01,
    "log_vmr_co": 0.10,
    "log_vmr_h2o": 0.10,
    "log_vmr_ch4": 0.10,
    "log_vmr_hf": 0.10,
    "logg": 0.09,
}
ATMOSPHERE_NAMES = (
    "T0",
    "alpha",
    "logg",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
    "log_p_cloud",
)


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_names(text):
    """Parse comma-separated atmospheric parameter names."""

    names = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = sorted(set(names) - set(DEFAULT_PARAMETER_SCALES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown parameter names: {unknown}")
    return names


def parse_scales(text):
    """Parse comma-separated parameter scales."""

    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one scale is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description="Generate the E1 v1 fixed atmospheric-response eigenbasis."
    )
    parser.add_argument(
        "--samples",
        default=str(ROOT / "results" / "m7" / "v1_zero_mean_log_w_run" / "samples.npz"),
        help="Posterior sample archive used to set the basis center.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "e1" / "v1_basis" / "eigen_response_basis.npz"),
    )
    parser.add_argument("--chip-indices", type=parse_chips, default=None)
    parser.add_argument(
        "--parameter-names",
        type=parse_names,
        default=DEFAULT_PARAMETER_NAMES,
        help="Comma-separated atmospheric parameters included in the SVD.",
    )
    parser.add_argument(
        "--parameter-scales",
        type=parse_scales,
        default=None,
        help=(
            "Comma-separated perturbation scales matching --parameter-names. "
            "Defaults to conservative E1 v1 scales."
        ),
    )
    parser.add_argument("--mode-count", type=int, default=2)
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_chip_data_from_samples(samples, chip_indices):
    """Load chip wavelength grids and spectra embedded in a sample archive."""

    obs_times = np.asarray(samples["obs_times"])
    chip_data_list = []
    for chip_index in chip_indices:
        wavelengths = np.asarray(samples[f"wavelengths_chip{chip_index}"])
        flux = np.asarray(samples[f"flux_chip{chip_index}"])
        chip_data_list.append(
            Luhman16BChipData(
                wavelengths=wavelengths,
                flux=flux,
                line_profile=np.ones_like(wavelengths),
                obs_times=obs_times,
                chip_index=int(chip_index),
            )
        )
    return chip_data_list


def _median_or_default(samples, name, default):
    """Return posterior median for a scalar parameter when available."""

    if name not in samples.files:
        return float(default)
    value = np.asarray(samples[name])
    if value.ndim == 0:
        return float(value)
    return float(np.median(value))


def _basis_center(samples):
    """Build the physical atmospheric center used for the SVD basis."""

    defaults = YAMA_L16B_EXOMOL_ATMOSPHERE
    return {
        "T0": _median_or_default(samples, "T0", defaults.t0),
        "alpha": _median_or_default(samples, "alpha", defaults.alpha),
        "logg": _median_or_default(samples, "logg", defaults.logg),
        "log_vmr_co": _median_or_default(samples, "log_vmr_co", defaults.log_vmr_co),
        "log_vmr_h2o": _median_or_default(samples, "log_vmr_h2o", defaults.log_vmr_h2o),
        "log_vmr_ch4": _median_or_default(samples, "log_vmr_ch4", defaults.log_vmr_ch4),
        "log_vmr_hf": _median_or_default(samples, "log_vmr_hf", defaults.log_vmr_hf),
        "log_p_cloud": _median_or_default(samples, "log_p_cloud", defaults.log_p_cloud),
    }


def _profile_wavelengths_from_samples(samples, chip_data_list):
    """Load saved profile grids or reconstruct them from the sampled velocity."""

    max_abs_velocity = abs(_median_or_default(samples, "v", 31.2))
    profile_wavelengths = []
    for chip_data in chip_data_list:
        key = f"profile_wavelengths_chip{chip_data.chip_index}"
        if key in samples.files:
            profile_grid = np.asarray(samples[key])
        else:
            profile_grid = doppler_padded_wavelengths(
                chip_data.wavelengths,
                max_abs_velocity,
            )
        profile_wavelengths.append(profile_grid)
    return profile_wavelengths


def _xi_vector(center, parameter_names):
    """Return a vector of selected atmospheric parameters."""

    return jnp.asarray([center[name] for name in parameter_names])


def _center_with_xi(center, parameter_names, xi):
    """Return an atmospheric parameter dict with selected values replaced."""

    values = dict(center)
    for index, name in enumerate(parameter_names):
        values[name] = xi[index]
    return values


def _spectrum_function(model, center, parameter_names):
    """Return a vector-input local spectrum function."""

    def spectrum(xi):
        values = _center_with_xi(center, parameter_names, xi)
        return model.cloudy_at_log_vmrs(
            values["T0"],
            values["alpha"],
            values["log_vmr_co"],
            values["log_vmr_h2o"],
            values["log_vmr_ch4"],
            values["log_vmr_hf"],
            values["log_p_cloud"],
            logg=values["logg"],
        )

    return spectrum


def _build_chip_jacobian(
    args,
    chip_data,
    center,
    parameter_names,
    profile_wavelengths=None,
):
    """Evaluate the local spectrum and Jacobian for one chip."""

    model = Luhman16BPowerLawColumnModel(
        chip_data.wavelengths,
        molecule_paths=_molecule_paths(args.database_dir),
        cia_paths=_cia_paths(args.database_dir),
        opacity_cache_dir=args.opacity_cache_dir,
        parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
        nx=args.nx,
        sampling_wavelengths=profile_wavelengths,
    )
    spectrum = _spectrum_function(model, center, parameter_names)
    xi0 = _xi_vector(center, parameter_names)
    return jax.jit(jax.jacfwd(spectrum))(xi0), jax.jit(spectrum)(xi0)


def main():
    """Generate and save the E1 v1 eigen-response basis."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    samples = np.load(args.samples, allow_pickle=False)
    chip_indices = (
        tuple(int(v) for v in np.asarray(samples["chip_indices"]))
        if args.chip_indices is None
        else tuple(args.chip_indices)
    )
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
    profile_wavelengths = _profile_wavelengths_from_samples(
        samples,
        chip_data_list,
    )

    start = time.time()
    spectra = []
    jacobians = []
    whitened_blocks = []
    for chip_position, chip_data in enumerate(chip_data_list):
        jacobian, spectrum = _build_chip_jacobian(
            args,
            chip_data,
            center,
            parameter_names,
            profile_wavelengths[chip_position],
        )
        jacobian = np.asarray(jacobian)
        spectrum = np.asarray(spectrum)
        sigma_d = np.asarray(samples["sigma_d"])
        sigma_chip = float(np.median(sigma_d[:, chip_position]))
        scaled = jacobian * np.asarray(parameter_scales)[None, :]
        whitened_blocks.append(scaled / sigma_chip)
        jacobians.append(jacobian)
        spectra.append(spectrum)

    whitened_response = np.concatenate(whitened_blocks, axis=0)
    _, singular_values, vh = np.linalg.svd(whitened_response, full_matrices=False)
    v_matrix = vh.T
    selected_v = v_matrix[:, : args.mode_count]

    payload = {
        "parameter_names": np.asarray(parameter_names),
        "parameter_scales": np.asarray(parameter_scales, dtype=float),
        "atmosphere_names": np.asarray(ATMOSPHERE_NAMES),
        "basis_center": np.asarray([center[name] for name in ATMOSPHERE_NAMES], dtype=float),
        "chip_indices": np.asarray(chip_indices, dtype=int),
        "mode_count": np.asarray(args.mode_count, dtype=int),
        "singular_values": singular_values,
        "v_matrix": v_matrix,
        "selected_v": selected_v,
        "whitened_response": whitened_response,
    }
    for chip_position, chip_index in enumerate(chip_indices):
        payload[f"spectrum_chip{chip_index}"] = spectra[chip_position]
        payload[f"jacobian_chip{chip_index}"] = jacobians[chip_position]
        payload[f"profile_wavelengths_chip{chip_index}"] = profile_wavelengths[
            chip_position
        ]
        payload[f"eigenspectra_chip{chip_index}"] = (
            jacobians[chip_position] * np.asarray(parameter_scales)[None, :]
        ) @ selected_v
    np.savez(out_path, **payload)
    metadata = {
        "samples": str(args.samples),
        "out": str(out_path),
        "chip_indices": list(chip_indices),
        "parameter_names": list(parameter_names),
        "parameter_scales": list(parameter_scales),
        "mode_count": args.mode_count,
        "singular_values": singular_values.tolist(),
        "basis_center": center,
        "elapsed_seconds": time.time() - start,
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
