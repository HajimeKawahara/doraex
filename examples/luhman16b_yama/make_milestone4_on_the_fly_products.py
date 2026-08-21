"""Create products for on-the-fly autodiff pressure retrievals."""

import argparse
import hashlib
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

from doraex.data.luhman16b import Luhman16BChipData  # noqa: E402
from doraex.inference.map_posterior import conditional_map_posterior  # noqa: E402
from doraex.operators.design_matrix import linear_profile_operator_from_times  # noqa: E402
from doraex.priors.spherical_gp import (  # noqa: E402
    add_diagonal_jitter,
    project_zero_mean_covariance,
    squared_exponential_covariance,
)
from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel  # noqa: E402
from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    _chip_sample,
    build_luhman16b_geometry,
    fixed_two_column_median_sample,
)
from doraex.workflows.on_the_fly_pressure_retrieval import (  # noqa: E402
    build_fixed_pressure_gp_eigendecomposition,
)
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
    _cia_paths,
    _molecule_paths,
)
from make_milestone2_fixed_products import (  # noqa: E402
    _plot_delta_s,
    _plot_figure9,
    _write_cloud_fraction_diagnostics,
)
from make_milestone2_free_t0_cloud_products import _select_sample_indices  # noqa: E402
from make_milestone2_joint_chip_products import (  # noqa: E402
    _center_values_by_chip,
    _map_plot_cmap,
    _plot_surface_map_figures,
    _write_joint_diagnostics,
    _write_pressure_map_diagnostics,
)


LOG_VMR_NAMES = (
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
)


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description="Build products for on-the-fly autodiff pressure retrievals."
    )
    parser.add_argument(
        "--samples",
        default=str(
            ROOT
            / "results"
            / "milestone4_on_the_fly_autodiff_full_joint"
            / "mcmc_on_the_fly_autodiff_pressure.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "milestone4_on_the_fly_autodiff_full_joint"),
    )
    parser.add_argument("--chip-indices", type=parse_chips, default=None)
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--max-map-samples", type=int, default=None)
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--gp-jitter", type=float, default=0.5e-6)
    parser.add_argument("--noise-jitter", type=float, default=1.0e-6)
    parser.add_argument(
        "--reuse-map-products",
        action="store_true",
        help="Reuse saved pressure-map arrays and rebuild downstream products.",
    )
    parser.add_argument(
        "--cloud-fraction-cmap",
        default=None,
        help=(
            "Matplotlib colormap for the pressure-perturbation map. Defaults "
            "to the joint-product pressure-map colormap."
        ),
    )
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_chip_data_from_samples(samples, chip_indices):
    """Load chip data arrays embedded in the on-the-fly sample NPZ."""

    obs_times = np.asarray(samples["obs_times"])
    chip_data_list = []
    for chip_index in chip_indices:
        wavelength_key = f"wavelengths_chip{chip_index}"
        flux_key = f"flux_chip{chip_index}"
        if wavelength_key not in samples or flux_key not in samples:
            raise KeyError(f"Missing {wavelength_key} or {flux_key} in samples.")
        wavelengths = np.asarray(samples[wavelength_key])
        flux = np.asarray(samples[flux_key])
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


def _load_profile_wavelengths(samples, chip_data_list):
    """Load the padded local-profile grids used by the retrieval."""

    profile_wavelengths = []
    for chip_data in chip_data_list:
        key = f"profile_wavelengths_chip{chip_data.chip_index}"
        if key not in samples:
            raise ValueError(
                f"Missing {key}; rerun the retrieval with Doppler-padded "
                "local profiles before building products."
            )
        profile_wavelengths.append(np.asarray(samples[key]))
    return profile_wavelengths


def _load_observation_masks(samples, chip_data_list):
    """Load and validate the retrieval's static likelihood-row masks."""

    chip_indices = [int(chip.chip_index) for chip in chip_data_list]
    flux_by_chip = [np.asarray(chip.flux) for chip in chip_data_list]
    for chip_index, flux in zip(chip_indices, flux_by_chip):
        if not np.all(np.isfinite(flux)):
            raise ValueError(
                f"Non-finite flux values found for chip {chip_index}."
            )

    contract_names = {
        "mask_zero_flux",
        "observation_mask_rule",
        "observation_valid_count",
        "observation_excluded_count",
        "observation_indices",
        *(f"observation_mask_chip{chip_index}" for chip_index in chip_indices),
    }
    contract_present = any(name in samples for name in contract_names)
    if not contract_present:
        masks = [np.ones(flux.shape, dtype=bool) for flux in flux_by_chip]
        total = int(sum(mask.size for mask in masks))
        return masks, {
            "mask_zero_flux": False,
            "observation_mask_rule": "all_observations",
            "observation_mask_source": "legacy_archive_default",
            "observation_valid_count": total,
            "observation_excluded_count": 0,
            "observation_valid_count_by_chip": {
                str(chip_index): int(mask.size)
                for chip_index, mask in zip(chip_indices, masks)
            },
            "observation_excluded_count_by_chip": {
                str(chip_index): 0 for chip_index in chip_indices
            },
        }

    missing = sorted(name for name in contract_names if name not in samples)
    if missing:
        raise ValueError(
            "Incomplete observation-mask contract; missing "
            + ", ".join(missing)
            + "."
        )
    if "chip_indices" not in samples:
        raise ValueError("Observation-mask contract requires chip_indices.")
    archived_chip_indices = [int(value) for value in np.asarray(samples["chip_indices"])]
    if archived_chip_indices != chip_indices:
        raise ValueError(
            "Masked product reconstruction requires the archived chip order; "
            f"got {chip_indices}, expected {archived_chip_indices}."
        )

    mask_zero_flux_value = np.asarray(samples["mask_zero_flux"])
    if mask_zero_flux_value.shape != () or mask_zero_flux_value.dtype.kind != "b":
        raise ValueError("mask_zero_flux must be a scalar boolean.")
    mask_zero_flux = bool(mask_zero_flux_value.item())
    rule_value = np.asarray(samples["observation_mask_rule"])
    if rule_value.shape != ():
        raise ValueError("observation_mask_rule must be a scalar string.")
    rule = str(rule_value.item())
    expected_rule = (
        "finite_and_nonzero_flux" if mask_zero_flux else "all_observations"
    )
    if rule != expected_rule:
        raise ValueError(
            f"Observation-mask rule mismatch: got {rule!r}, "
            f"expected {expected_rule!r}."
        )

    masks = []
    for chip_index, flux in zip(chip_indices, flux_by_chip):
        key = f"observation_mask_chip{chip_index}"
        mask = np.asarray(samples[key])
        if mask.dtype.kind != "b" or mask.shape != flux.shape:
            raise ValueError(
                f"{key} must be boolean with shape {flux.shape}, got "
                f"dtype={mask.dtype}, shape={mask.shape}."
            )
        expected = flux != 0.0 if mask_zero_flux else np.ones(flux.shape, dtype=bool)
        if not np.array_equal(mask, expected):
            raise ValueError(
                f"{key} does not match the declared exact-zero flux rule."
            )
        if not np.any(mask):
            raise ValueError(f"No valid observations remain for chip {chip_index}.")
        masks.append(mask)

    flat_mask = np.concatenate([mask.reshape(-1) for mask in masks])
    expected_indices = np.flatnonzero(flat_mask).astype(np.int64, copy=False)
    stored_indices = np.asarray(samples["observation_indices"])
    if not np.issubdtype(stored_indices.dtype, np.integer):
        raise ValueError("observation_indices must use an integer dtype.")
    if stored_indices.ndim != 1 or not np.array_equal(
        stored_indices.astype(np.int64, copy=False),
        expected_indices,
    ):
        raise ValueError(
            "observation_indices do not match the flattened per-chip masks."
        )

    valid_count = int(np.count_nonzero(flat_mask))
    excluded_count = int(flat_mask.size - valid_count)
    for name, expected in (
        ("observation_valid_count", valid_count),
        ("observation_excluded_count", excluded_count),
    ):
        value = np.asarray(samples[name])
        if value.shape != () or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must be a scalar integer.")
        if int(value.item()) != expected:
            raise ValueError(
                f"{name}={int(value.item())} does not match the masks ({expected})."
            )

    valid_by_chip = {
        str(chip_index): int(np.count_nonzero(mask))
        for chip_index, mask in zip(chip_indices, masks)
    }
    excluded_by_chip = {
        str(chip_index): int(mask.size - np.count_nonzero(mask))
        for chip_index, mask in zip(chip_indices, masks)
    }
    return masks, {
        "mask_zero_flux": mask_zero_flux,
        "observation_mask_rule": rule,
        "observation_mask_source": "retrieval_archive",
        "observation_valid_count": valid_count,
        "observation_excluded_count": excluded_count,
        "observation_valid_count_by_chip": valid_by_chip,
        "observation_excluded_count_by_chip": excluded_by_chip,
    }


def _validate_reused_observation_mask_summary(existing_summary, current_summary):
    """Reject conditional maps built under a different row-selection contract."""

    keys = (
        "mask_zero_flux",
        "observation_mask_rule",
        "observation_mask_source",
        "observation_valid_count",
        "observation_excluded_count",
        "observation_valid_count_by_chip",
        "observation_excluded_count_by_chip",
    )
    missing = [key for key in keys if key not in existing_summary]
    if missing:
        raise ValueError(
            "Reused map products lack observation-mask metadata: "
            + ", ".join(missing)
            + ". Recompute the map products."
        )
    mismatches = [
        key
        for key in keys
        if existing_summary[key] != current_summary[key]
    ]
    if mismatches:
        raise ValueError(
            "Reused map products do not match the archived observation-mask "
            "contract: "
            + ", ".join(mismatches)
            + ". Recompute the map products."
        )


def _response_function(
    spectrum_function,
    *,
    independent_log_vmrs=False,
    explicit_logg=False,
):
    """Return a pressure JVP for legacy or independent-VMR samples."""

    def response(t0, alpha, *parameters):
        if independent_log_vmrs:
            log_vmrs = parameters[: len(LOG_VMR_NAMES)]
            tail = parameters[len(LOG_VMR_NAMES) :]
        else:
            log_vmrs = parameters[:1]
            tail = parameters[1:]
        if explicit_logg:
            logg, log_p_cloud = tail
        else:
            (log_p_cloud,) = tail
            logg = None

        def spectrum_at_pressure(pressure):
            return spectrum_function(
                t0,
                alpha,
                *log_vmrs,
                pressure,
                logg=logg,
            )

        return jax.jvp(
            spectrum_at_pressure,
            (log_p_cloud,),
            (jnp.ones_like(log_p_cloud),),
        )

    return response


def _build_response_functions(
    args,
    samples,
    chip_data_list,
    profile_wavelengths=None,
):
    """Build ExoJAX on-the-fly response functions for each chip."""

    if profile_wavelengths is None:
        profile_wavelengths = _load_profile_wavelengths(samples, chip_data_list)
    independent_log_vmrs = all(name in samples for name in LOG_VMR_NAMES)
    explicit_logg = "logg" in samples
    response_functions = []
    for chip_position, chip_data in enumerate(chip_data_list):
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx,
            sampling_wavelengths=profile_wavelengths[chip_position],
        )
        spectrum_function = (
            model.cloudy_at_log_vmrs
            if independent_log_vmrs
            else model.cloudy_at_parameters
        )
        response = jax.jit(
            _response_function(
                spectrum_function,
                independent_log_vmrs=independent_log_vmrs,
                explicit_logg=explicit_logg,
            )
        )
        response.profile_wavelengths = profile_wavelengths[chip_position]
        response.independent_log_vmrs = independent_log_vmrs
        response.explicit_logg = explicit_logg
        response_functions.append(response)
    return response_functions


def _sample_at(samples, index):
    """Return one posterior sample while preserving fixed scalar metadata."""

    sample_names = {
        "cosi",
        "v",
        "q1",
        "q2",
        "u1",
        "u2",
        "log_w",
        "A",
        "sigma_d",
        "sigma_b",
        "sigma_log_p",
        "ell_b",
        "P",
        "T0",
        "t0",
        "log_p_cloud",
        "alpha",
        "logg",
        "log_vmr_co",
        "log_vmr_h2o",
        "log_vmr_ch4",
        "log_vmr_hf",
        "zeta_vmr",
    }
    result = {}
    for name in sample_names:
        if name not in samples:
            continue
        value = jnp.asarray(samples[name])
        result[name] = value if value.ndim == 0 else value[index]
    if "zero_mean_pressure_map" in samples:
        result["zero_mean_pressure_map"] = jnp.asarray(
            samples["zero_mean_pressure_map"]
        )
    return result


def _evaluate_response(response, sample):
    """Evaluate one response using the parameterization stored in samples."""

    t0 = sample.get(
        "T0",
        sample.get("t0", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.t0)),
    )
    alpha = sample.get(
        "alpha",
        jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.alpha),
    )
    parameters = [jnp.asarray(t0), jnp.asarray(alpha)]
    if all(name in sample for name in LOG_VMR_NAMES):
        parameters.extend(jnp.asarray(sample[name]) for name in LOG_VMR_NAMES)
    else:
        parameters.append(jnp.asarray(sample.get("zeta_vmr", 0.0)))
    if "logg" in sample:
        parameters.append(jnp.asarray(sample["logg"]))
    parameters.append(jnp.asarray(sample["log_p_cloud"]))
    return response(*parameters)


def _build_retrieval_precision_geometry(samples, nside):
    """Rebuild geometry constants in the retrieval's configured precision."""

    retrieval_x64 = bool(np.asarray(samples.get("x64", False)).item())
    with jax.enable_x64(retrieval_x64):
        return build_luhman16b_geometry(nside=nside)


def _build_product_geometry(samples, nside, product_x64):
    """Promote retrieval geometry constants without recomputing their values."""

    retrieval_geometry = _build_retrieval_precision_geometry(samples, nside)
    numpy_dtype = np.float64 if product_x64 else np.float32
    with jax.enable_x64(product_x64):
        return retrieval_geometry.__class__(
            theta=jnp.asarray(np.asarray(retrieval_geometry.theta, dtype=numpy_dtype)),
            phi=jnp.asarray(np.asarray(retrieval_geometry.phi, dtype=numpy_dtype)),
            distance_matrix=jnp.asarray(
                np.asarray(retrieval_geometry.distance_matrix, dtype=numpy_dtype)
            ),
            nside=retrieval_geometry.nside,
        )


def _linear_profile_operator_from_sample(
    chip_data,
    geometry,
    base_profile,
    contrast_profile,
    sample,
    profile_wavelengths=None,
):
    """Build the on-the-fly linear pressure-response operator for one chip."""

    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    baseline, contrast_matrix = linear_profile_operator_from_times(
        geometry.theta,
        geometry.phi,
        jnp.asarray(sample["v"]),
        inclination,
        jnp.asarray(sample["u1"]),
        jnp.asarray(sample["u2"]),
        jnp.asarray(chip_data.obs_times),
        jnp.asarray(sample["P"]),
        jnp.asarray(chip_data.wavelengths),
        base_profile,
        contrast_profile,
        weights=weights,
        rest_wavelengths=profile_wavelengths,
    )
    norm = jnp.asarray(sample["A"]) * jnp.mean(baseline)
    return baseline / norm, contrast_matrix / norm


def _joint_operator_from_sample(
    chip_data_list,
    geometry,
    response_functions,
    sample,
    profile_wavelengths=None,
):
    """Build the concatenated baseline and pressure-response matrix."""

    baselines = []
    contrast_matrices = []
    if profile_wavelengths is None:
        profile_wavelengths = [
            getattr(response, "profile_wavelengths", None)
            for response in response_functions
        ]
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_sample = _chip_sample(sample, chip_position)
        base_profile, contrast_profile = _evaluate_response(
            response_functions[chip_position],
            sample,
        )
        baseline, contrast_matrix = _linear_profile_operator_from_sample(
            chip_data,
            geometry,
            base_profile,
            contrast_profile,
            chip_sample,
            profile_wavelengths[chip_position],
        )
        baselines.append(baseline)
        contrast_matrices.append(contrast_matrix)
    return jnp.concatenate(baselines, axis=0), jnp.concatenate(
        contrast_matrices,
        axis=0,
    )


def _prepare_fixed_pressure_gp_eigendecomposition(samples, geometry):
    """Rebuild the fixed zero-mean GP basis and retain saved eigenvalues."""

    factorization = str(
        np.asarray(samples.get("pressure_gp_factorization", "cholesky")).item()
    )
    if factorization != "fixed_eigen":
        return None
    if not bool(np.asarray(samples.get("zero_mean_pressure_map", False)).item()):
        raise ValueError("fixed_eigen products require a zero-mean pressure map.")
    if "pressure_gp_eigenvalues" not in samples:
        raise ValueError("Missing saved pressure_gp_eigenvalues.")

    retrieval_geometry = _build_retrieval_precision_geometry(
        samples,
        geometry.nside,
    )
    with jax.enable_x64(bool(np.asarray(samples.get("x64", False)).item())):
        rebuilt = build_fixed_pressure_gp_eigendecomposition(
            retrieval_geometry.distance_matrix,
            float(np.asarray(samples["fixed_ell_b"]).item()),
            theta=retrieval_geometry.theta,
            phi=retrieval_geometry.phi,
        )
    saved_eigenvalues = np.asarray(
        samples["pressure_gp_eigenvalues"],
        dtype=np.float64,
    )
    rebuilt_eigenvalues = np.asarray(rebuilt["eigenvalues"], dtype=np.float64)
    if saved_eigenvalues.shape != rebuilt_eigenvalues.shape or not np.allclose(
        saved_eigenvalues,
        rebuilt_eigenvalues,
        rtol=5.0e-7,
        atol=1.0e-6,
    ):
        maximum_difference = (
            float(np.max(np.abs(saved_eigenvalues - rebuilt_eigenvalues)))
            if saved_eigenvalues.shape == rebuilt_eigenvalues.shape
            else float("inf")
        )
        raise ValueError(
            "Saved and rebuilt pressure-GP eigenvalues disagree: "
            f"max_abs_difference={maximum_difference:.6e}."
        )

    pixel_eigenvectors = np.asarray(
        rebuilt["pixel_eigenvectors"],
        dtype=np.float64,
    )
    pixel_count = int(np.asarray(retrieval_geometry.theta).size)
    expected_shape = (pixel_count, pixel_count - 1)
    if pixel_eigenvectors.shape != expected_shape:
        raise ValueError(
            "Rebuilt pressure-GP eigenvectors have shape "
            f"{pixel_eigenvectors.shape}, expected {expected_shape}."
        )
    return {
        "eigenvalues": saved_eigenvalues,
        "pixel_eigenvectors": pixel_eigenvectors,
        "eigenvalue_source": "saved_samples",
        "eigenvector_source": "rebuilt_from_retrieval_precision_angles",
        "rebuilt_eigenvalue_max_abs_difference": float(
            np.max(np.abs(saved_eigenvalues - rebuilt_eigenvalues))
        ),
        "eigenvalue_validation_rtol": 5.0e-7,
        "eigenvalue_validation_atol": 1.0e-6,
    }


def _conditional_pressure_map_fixed_eigen(
    residual,
    contrast_matrix,
    noise_variance,
    sigma_log_p,
    gp_jitter,
    pressure_gp_eigendecomposition,
    return_variance_diagonal=False,
):
    """Condition the map in the same reduced GP coordinates as retrieval."""

    contrast_matrix = jnp.asarray(contrast_matrix)
    dtype = contrast_matrix.dtype
    eigenvalues = jnp.asarray(
        pressure_gp_eigendecomposition["eigenvalues"],
        dtype=dtype,
    )
    pixel_eigenvectors = jnp.asarray(
        pressure_gp_eigendecomposition["pixel_eigenvectors"],
        dtype=dtype,
    )
    prior_variance = jnp.asarray(
        sigma_log_p, dtype=dtype
    ) ** 2 * eigenvalues + jnp.asarray(gp_jitter, dtype=dtype)
    reduced_design = contrast_matrix @ pixel_eigenvectors
    noise_variance = jnp.asarray(noise_variance, dtype=dtype)
    weighted_design = reduced_design.T / noise_variance
    posterior_precision = (
        jnp.diag(1.0 / prior_variance) + weighted_design @ reduced_design
    )
    precision_factor = jnp.linalg.cholesky(posterior_precision)
    posterior_mean_reduced = jsp_linalg.cho_solve(
        (precision_factor, True),
        weighted_design @ jnp.asarray(residual, dtype=dtype).reshape(-1),
    )
    posterior_mean = pixel_eigenvectors @ posterior_mean_reduced
    if return_variance_diagonal:
        whitened_pixel_basis = jsp_linalg.solve_triangular(
            precision_factor,
            pixel_eigenvectors.T,
            lower=True,
        )
        posterior_variance_diagonal = jnp.sum(
            whitened_pixel_basis**2,
            axis=0,
        )
        return posterior_mean, posterior_variance_diagonal

    inverse_factor = jsp_linalg.solve_triangular(
        precision_factor,
        jnp.eye(precision_factor.shape[0], dtype=dtype),
        lower=True,
    )
    posterior_covariance_reduced = inverse_factor.T @ inverse_factor
    posterior_covariance = (
        pixel_eigenvectors @ posterior_covariance_reduced @ pixel_eigenvectors.T
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    return posterior_mean, posterior_covariance


def _select_observation_rows(
    residual,
    contrast_matrix,
    noise_variance,
    observation_masks,
):
    """Apply the archived likelihood-row mask to conditional-map inputs."""

    flat_mask = np.concatenate(
        [np.asarray(mask, dtype=bool).reshape(-1) for mask in observation_masks]
    )
    residual = jnp.asarray(residual).reshape(-1)
    contrast_matrix = jnp.asarray(contrast_matrix)
    noise_variance = jnp.asarray(noise_variance).reshape(-1)
    expected_size = int(flat_mask.size)
    if (
        residual.shape[0] != expected_size
        or contrast_matrix.shape[0] != expected_size
        or noise_variance.shape[0] != expected_size
    ):
        raise ValueError(
            "Observation masks and conditional-map rows disagree: "
            f"mask={expected_size}, residual={residual.shape[0]}, "
            f"contrast={contrast_matrix.shape[0]}, noise={noise_variance.shape[0]}."
        )
    indices = jnp.asarray(np.flatnonzero(flat_mask), dtype=jnp.int32)
    return (
        jnp.take(residual, indices, axis=0),
        jnp.take(contrast_matrix, indices, axis=0),
        jnp.take(noise_variance, indices, axis=0),
    )


def _conditional_pressure_map_for_sample(
    chip_data_list,
    geometry,
    response_functions,
    profile_wavelengths,
    sample,
    gp_jitter,
    noise_jitter,
    pressure_gp_eigendecomposition=None,
    observation_masks=None,
    return_variance_diagonal=False,
):
    """Compute the conditional pressure-perturbation map posterior."""

    baseline, contrast_matrix = _joint_operator_from_sample(
        chip_data_list,
        geometry,
        response_functions,
        sample,
        profile_wavelengths=profile_wavelengths,
    )
    residual = (
        jnp.concatenate(
            [jnp.asarray(chip.flux).reshape(-1) for chip in chip_data_list],
            axis=0,
        )
        - baseline
    )
    noise_variance = jnp.concatenate(
        [
            jnp.asarray(sample["sigma_d"])[chip_position] ** 2
            * jnp.ones(chip.flux.size, dtype=baseline.dtype)
            + noise_jitter
            for chip_position, chip in enumerate(chip_data_list)
        ],
        axis=0,
    )
    if observation_masks is not None:
        residual, contrast_matrix, noise_variance = _select_observation_rows(
            residual,
            contrast_matrix,
            noise_variance,
            observation_masks,
        )
    if pressure_gp_eigendecomposition is not None:
        mean, covariance = _conditional_pressure_map_fixed_eigen(
            residual,
            contrast_matrix,
            noise_variance,
            sample["sigma_log_p"],
            gp_jitter,
            pressure_gp_eigendecomposition,
            return_variance_diagonal=return_variance_diagonal,
        )
    else:
        prior_covariance = squared_exponential_covariance(
            geometry.distance_matrix,
            jnp.asarray(sample["sigma_log_p"]),
            jnp.asarray(sample["ell_b"]),
        )
        if bool(np.asarray(sample.get("zero_mean_pressure_map", False))):
            prior_covariance = project_zero_mean_covariance(prior_covariance)
        prior_covariance = add_diagonal_jitter(
            prior_covariance,
            jitter=gp_jitter,
        )
        prior_mean = jnp.zeros(contrast_matrix.shape[1])
        mean, covariance = conditional_map_posterior(
            residual,
            contrast_matrix,
            prior_mean,
            prior_covariance,
            noise_variance,
        )
        if bool(np.asarray(sample.get("zero_mean_pressure_map", False))):
            mean = mean - jnp.mean(mean)
            covariance = project_zero_mean_covariance(covariance)
        if return_variance_diagonal:
            covariance = jnp.diag(covariance)
    return mean, covariance


def _compute_pressure_map_moments(
    chip_data_list,
    geometry,
    response_functions,
    profile_wavelengths,
    samples,
    sample_indices,
    gp_jitter,
    noise_jitter,
    pressure_gp_eigendecomposition=None,
    observation_masks=None,
):
    """Compute posterior moments for the shared pressure perturbation map."""

    conditional_means = []
    conditional_variance_diagonals = []
    for count, index in enumerate(sample_indices, start=1):
        start = time.time()
        sample = _sample_at(samples, int(index))
        mean, covariance = _conditional_pressure_map_for_sample(
            chip_data_list,
            geometry,
            response_functions,
            profile_wavelengths,
            sample,
            gp_jitter,
            noise_jitter,
            pressure_gp_eigendecomposition,
            observation_masks=observation_masks,
            return_variance_diagonal=True,
        )
        conditional_means.append(mean)
        conditional_variance_diagonals.append(covariance)
        if count == 1 or count % 25 == 0 or count == len(sample_indices):
            elapsed = time.time() - start
            print(
                f"Processed map sample {count}/{len(sample_indices)} "
                f"(posterior index {int(index)}, {elapsed:.2f} s)"
            )

    moments = _combine_pressure_map_conditionals(
        jnp.stack(conditional_means, axis=0),
        jnp.stack(conditional_variance_diagonals, axis=0),
        jnp.asarray(samples["log_p_cloud"])[jnp.asarray(sample_indices)],
    )
    return (
        moments["perturbation_mean"],
        moments["perturbation_var"],
        moments,
    )


def _combine_pressure_map_conditionals(
    conditional_means,
    conditional_variance_diagonals,
    log_p_cloud_centers,
):
    """Combine conditional Gaussian moments, including absolute pressure."""

    conditional_means = jnp.asarray(conditional_means)
    conditional_variance_diagonals = jnp.asarray(conditional_variance_diagonals)
    log_p_cloud_centers = jnp.asarray(log_p_cloud_centers)
    if conditional_means.ndim != 2:
        raise ValueError("conditional_means must have shape (sample, pixel).")
    if conditional_variance_diagonals.shape != conditional_means.shape:
        raise ValueError("conditional_variance_diagonals must match conditional_means.")
    if log_p_cloud_centers.shape != (conditional_means.shape[0],):
        raise ValueError("log_p_cloud_centers must contain one scalar per sample.")

    perturbation_mean = jnp.mean(conditional_means, axis=0)
    perturbation_second = jnp.mean(
        conditional_variance_diagonals + conditional_means**2,
        axis=0,
    )
    perturbation_var = jnp.maximum(
        perturbation_second - perturbation_mean**2,
        0.0,
    )

    absolute_conditional_means = log_p_cloud_centers[:, None] + conditional_means
    log_p_cloud_mean = jnp.mean(absolute_conditional_means, axis=0)
    log_p_cloud_second = jnp.mean(
        conditional_variance_diagonals + absolute_conditional_means**2,
        axis=0,
    )
    log_p_cloud_var = jnp.maximum(
        log_p_cloud_second - log_p_cloud_mean**2,
        0.0,
    )

    log_ten = jnp.log(jnp.asarray(10.0, dtype=conditional_means.dtype))
    p_cloud_conditional_mean = jnp.exp(
        log_ten * absolute_conditional_means
        + 0.5 * log_ten**2 * conditional_variance_diagonals
    )
    p_cloud_conditional_second = jnp.exp(
        2.0 * log_ten * absolute_conditional_means
        + 2.0 * log_ten**2 * conditional_variance_diagonals
    )
    p_cloud_mean = jnp.mean(p_cloud_conditional_mean, axis=0)
    p_cloud_var = jnp.maximum(
        jnp.mean(p_cloud_conditional_second, axis=0) - p_cloud_mean**2,
        0.0,
    )
    return {
        "perturbation_mean": perturbation_mean,
        "perturbation_var": perturbation_var,
        "log_p_cloud_mean": log_p_cloud_mean,
        "log_p_cloud_var": log_p_cloud_var,
        "p_cloud_mean": p_cloud_mean,
        "p_cloud_std": jnp.sqrt(p_cloud_var),
    }


def _reconstruct_median_models(
    chip_data_list,
    geometry,
    response_functions,
    profile_wavelengths,
    samples,
    contrast_map,
):
    """Reconstruct spectra from median nonlinear parameters and the map."""

    sample = fixed_two_column_median_sample(samples)
    models = []
    chip_samples = []
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_sample = _chip_sample(sample, chip_position)
        base_profile, contrast_profile = _evaluate_response(
            response_functions[chip_position],
            sample,
        )
        baseline, contrast_matrix = _linear_profile_operator_from_sample(
            chip_data,
            geometry,
            base_profile,
            contrast_profile,
            chip_sample,
            profile_wavelengths[chip_position],
        )
        model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
        models.append(model.reshape(chip_data.flux.shape))
        chip_samples.append(chip_sample)
    return models, sample, chip_samples


def _select_fit_residuals(residuals, observation_masks):
    """Return likelihood-used residuals without altering full saved arrays."""

    if len(residuals) != len(observation_masks):
        raise ValueError("Residual and observation-mask chip counts disagree.")
    selected = []
    for chip_position, (residual, mask) in enumerate(
        zip(residuals, observation_masks)
    ):
        residual = np.asarray(residual)
        mask = np.asarray(mask)
        if mask.dtype.kind != "b" or mask.shape != residual.shape:
            raise ValueError(
                f"Observation mask {chip_position} must be boolean with shape "
                f"{residual.shape}."
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError(
                f"Full residual array for chip position {chip_position} is non-finite."
            )
        fit_residual = residual[mask]
        if fit_residual.size == 0:
            raise ValueError(
                f"No fit residuals remain for chip position {chip_position}."
            )
        selected.append(fit_residual)
    return selected


def _save_pressure_maps(
    out_dir,
    samples,
    chip_data_list,
    perturbation_mean,
    perturbation_var,
    absolute_moments=None,
):
    """Save pressure-perturbation, log-pressure, and pressure maps."""

    chip_count = len(chip_data_list)
    perturbation_mean = np.asarray(perturbation_mean)
    perturbation_var = np.asarray(perturbation_var)
    perturbation_mean_by_chip = np.tile(perturbation_mean[None, :], (chip_count, 1))
    perturbation_var_by_chip = np.tile(perturbation_var[None, :], (chip_count, 1))
    if absolute_moments is None:
        center_by_chip = _center_values_by_chip(
            samples,
            "log_p_cloud",
            chip_count,
        )
        center_mean_by_chip = np.mean(center_by_chip, axis=0)
        center_var_by_chip = np.var(center_by_chip, axis=0)
        log_p_cloud_mean_by_chip = (
            center_mean_by_chip[:, None] + perturbation_mean_by_chip
        )
        log_p_cloud_var_by_chip = center_var_by_chip[:, None] + perturbation_var_by_chip
        p_cloud_mean_by_chip = 10.0**log_p_cloud_mean_by_chip
        p_cloud_std_by_chip = (
            np.log(10.0) * p_cloud_mean_by_chip * np.sqrt(log_p_cloud_var_by_chip)
        )
    else:
        log_p_cloud_mean_by_chip = np.tile(
            np.asarray(absolute_moments["log_p_cloud_mean"])[None, :],
            (chip_count, 1),
        )
        log_p_cloud_var_by_chip = np.tile(
            np.asarray(absolute_moments["log_p_cloud_var"])[None, :],
            (chip_count, 1),
        )
        p_cloud_mean_by_chip = np.tile(
            np.asarray(absolute_moments["p_cloud_mean"])[None, :],
            (chip_count, 1),
        )
        p_cloud_std_by_chip = np.tile(
            np.asarray(absolute_moments["p_cloud_std"])[None, :],
            (chip_count, 1),
        )

    np.save(out_dir / "contrast_mean_joint.npy", perturbation_mean)
    np.save(out_dir / "contrast_var_joint.npy", perturbation_var)
    np.save(
        out_dir / "cloud_pressure_perturbation_mean_joint_by_chip.npy",
        perturbation_mean_by_chip,
    )
    np.save(
        out_dir / "cloud_pressure_perturbation_var_joint_by_chip.npy",
        perturbation_var_by_chip,
    )
    np.save(out_dir / "log_p_cloud_mean_joint_by_chip.npy", log_p_cloud_mean_by_chip)
    np.save(out_dir / "log_p_cloud_var_joint_by_chip.npy", log_p_cloud_var_by_chip)
    np.save(out_dir / "p_cloud_mean_joint_by_chip.npy", p_cloud_mean_by_chip)
    np.save(out_dir / "p_cloud_std_joint_by_chip.npy", p_cloud_std_by_chip)
    return {
        "perturbation_mean_by_chip": perturbation_mean_by_chip,
        "perturbation_var_by_chip": perturbation_var_by_chip,
        "log_p_cloud_mean_by_chip": log_p_cloud_mean_by_chip,
        "log_p_cloud_var_by_chip": log_p_cloud_var_by_chip,
        "p_cloud_mean_by_chip": p_cloud_mean_by_chip,
        "p_cloud_std_by_chip": p_cloud_std_by_chip,
    }


def _load_pressure_maps(out_dir):
    """Load saved pressure-map arrays from a previous product run."""

    return {
        "perturbation_mean_by_chip": np.load(
            out_dir / "cloud_pressure_perturbation_mean_joint_by_chip.npy"
        ),
        "perturbation_var_by_chip": np.load(
            out_dir / "cloud_pressure_perturbation_var_joint_by_chip.npy"
        ),
        "log_p_cloud_mean_by_chip": np.load(
            out_dir / "log_p_cloud_mean_joint_by_chip.npy"
        ),
        "log_p_cloud_var_by_chip": np.load(
            out_dir / "log_p_cloud_var_joint_by_chip.npy"
        ),
        "p_cloud_mean_by_chip": np.load(out_dir / "p_cloud_mean_joint_by_chip.npy"),
        "p_cloud_std_by_chip": np.load(out_dir / "p_cloud_std_joint_by_chip.npy"),
    }


def _file_sha256(path):
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_pressure_gp_eigendecomposition(out_dir, decomposition):
    """Save the reconstructed pixel basis and frozen retrieval eigenvalues."""

    if decomposition is None:
        return {"factorization": "cholesky"}
    eigenvalue_path = out_dir / "pressure_gp_eigenvalues.npy"
    eigenvector_path = out_dir / "pressure_gp_pixel_eigenvectors.npy"
    np.save(eigenvalue_path, decomposition["eigenvalues"])
    np.save(eigenvector_path, decomposition["pixel_eigenvectors"])

    eigenvalues = np.asarray(decomposition["eigenvalues"])
    eigenvectors = np.asarray(decomposition["pixel_eigenvectors"])
    return {
        "factorization": "fixed_eigen",
        "eigenvalue_path": str(eigenvalue_path),
        "eigenvalue_sha256": _file_sha256(eigenvalue_path),
        "eigenvalue_shape": list(eigenvalues.shape),
        "eigenvalue_dtype": str(eigenvalues.dtype),
        "eigenvalue_source": decomposition["eigenvalue_source"],
        "eigenvector_path": str(eigenvector_path),
        "eigenvector_sha256": _file_sha256(eigenvector_path),
        "eigenvector_shape": list(eigenvectors.shape),
        "eigenvector_dtype": str(eigenvectors.dtype),
        "eigenvector_source": decomposition["eigenvector_source"],
        "rebuilt_eigenvalue_max_abs_difference": decomposition[
            "rebuilt_eigenvalue_max_abs_difference"
        ],
        "eigenvalue_validation_rtol": decomposition["eigenvalue_validation_rtol"],
        "eigenvalue_validation_atol": decomposition["eigenvalue_validation_atol"],
        "eigenvector_max_abs_column_sum": float(
            np.max(np.abs(np.sum(eigenvectors, axis=0)))
        ),
        "eigenvector_max_abs_orthogonality_error": float(
            np.max(
                np.abs(eigenvectors.T @ eigenvectors - np.eye(eigenvectors.shape[1]))
            )
        ),
    }


def _load_pressure_gp_eigendecomposition(out_dir, samples, summary):
    """Load a frozen product basis without overwriting it during reuse."""

    factorization = str(
        np.asarray(samples.get("pressure_gp_factorization", "cholesky")).item()
    )
    if factorization != "fixed_eigen":
        return None, {"factorization": "cholesky"}
    if summary.get("factorization") != "fixed_eigen":
        raise ValueError("Reused fixed-eigen products lack frozen basis metadata.")

    eigenvalue_path = out_dir / "pressure_gp_eigenvalues.npy"
    eigenvector_path = out_dir / "pressure_gp_pixel_eigenvectors.npy"
    for path, hash_key in (
        (eigenvalue_path, "eigenvalue_sha256"),
        (eigenvector_path, "eigenvector_sha256"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen pressure-GP basis: {path}")
        if _file_sha256(path) != summary.get(hash_key):
            raise ValueError(f"Frozen pressure-GP basis hash mismatch: {path}")

    eigenvalues = np.load(eigenvalue_path, allow_pickle=False)
    eigenvectors = np.load(eigenvector_path, allow_pickle=False)
    saved_eigenvalues = np.asarray(samples["pressure_gp_eigenvalues"])
    pixel_count = 12 * int(np.asarray(samples["nside"]).item()) ** 2
    if not np.array_equal(eigenvalues, saved_eigenvalues):
        raise ValueError(
            "Frozen product eigenvalues differ from the retrieval samples."
        )
    if eigenvectors.shape != (pixel_count, pixel_count - 1):
        raise ValueError(
            "Frozen product eigenvectors have shape "
            f"{eigenvectors.shape}, expected {(pixel_count, pixel_count - 1)}."
        )
    if not np.all(np.isfinite(eigenvectors)):
        raise ValueError("Frozen product eigenvectors must be finite.")
    return {
        "eigenvalues": eigenvalues,
        "pixel_eigenvectors": eigenvectors,
    }, dict(summary)


def main():
    """Compute and save on-the-fly pressure-retrieval products."""

    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = dict(np.load(args.samples, allow_pickle=False))
    sample_sha256 = _file_sha256(args.samples)
    existing_summary_path = out_dir / "on_the_fly_product_summary.json"
    existing_summary = (
        json.loads(existing_summary_path.read_text(encoding="utf-8"))
        if args.reuse_map_products and existing_summary_path.exists()
        else {}
    )
    fixed_eigen = (
        str(np.asarray(samples.get("pressure_gp_factorization", "cholesky")).item())
        == "fixed_eigen"
    )
    if args.reuse_map_products and fixed_eigen:
        if existing_summary.get("sample_sha256") != sample_sha256:
            raise ValueError(
                "Reused fixed-eigen maps do not match the input samples SHA256."
            )
    chip_indices = (
        [int(value) for value in np.asarray(samples["chip_indices"])]
        if args.chip_indices is None
        else args.chip_indices
    )
    nside = int(np.asarray(samples["nside"])) if args.nside is None else args.nside
    chip_data_list = _load_chip_data_from_samples(samples, chip_indices)
    observation_masks, observation_mask_summary = _load_observation_masks(
        samples,
        chip_data_list,
    )
    if args.reuse_map_products:
        _validate_reused_observation_mask_summary(
            existing_summary,
            observation_mask_summary,
        )
    profile_wavelengths = _load_profile_wavelengths(samples, chip_data_list)
    geometry = _build_product_geometry(samples, nside, args.x64)
    if args.reuse_map_products:
        (
            pressure_gp_eigendecomposition,
            pressure_gp_summary,
        ) = _load_pressure_gp_eigendecomposition(
            out_dir,
            samples,
            existing_summary.get("pressure_gp", {}),
        )
    else:
        pressure_gp_eigendecomposition = _prepare_fixed_pressure_gp_eigendecomposition(
            samples, geometry
        )
        pressure_gp_summary = _save_pressure_gp_eigendecomposition(
            out_dir,
            pressure_gp_eigendecomposition,
        )

    setup_start = time.time()
    response_functions = _build_response_functions(
        args,
        samples,
        chip_data_list,
        profile_wavelengths,
    )
    setup_seconds = time.time() - setup_start
    print(f"Built on-the-fly ExoJAX response functions in {setup_seconds:.2f} s")

    sample_indices = _select_sample_indices(
        len(np.asarray(samples["sigma_log_p"])),
        args.max_map_samples,
    )
    if sample_indices is None:
        sample_indices = np.arange(len(np.asarray(samples["sigma_log_p"])))
    if args.reuse_map_products:
        maps = _load_pressure_maps(out_dir)
        perturbation_mean = np.load(out_dir / "contrast_mean_joint.npy")
        perturbation_var = np.load(out_dir / "contrast_var_joint.npy")
        absolute_pressure_moment_method = existing_summary.get(
            "absolute_pressure_moment_method",
            "unknown_reused_products",
        )
        print(f"Reused saved pressure-map products from {out_dir}")
    else:
        (
            perturbation_mean,
            perturbation_var,
            absolute_moments,
        ) = _compute_pressure_map_moments(
            chip_data_list,
            geometry,
            response_functions,
            profile_wavelengths,
            samples,
            sample_indices,
            args.gp_jitter,
            args.noise_jitter,
            pressure_gp_eigendecomposition,
            observation_masks=observation_masks,
        )
        maps = _save_pressure_maps(
            out_dir,
            samples,
            chip_data_list,
            perturbation_mean,
            perturbation_var,
            absolute_moments,
        )
        absolute_pressure_moment_method = (
            "conditional_gaussian_exact_lognormal_transform"
        )
    pressure_maps = {
        "log_p_cloud_mean_by_chip": maps["log_p_cloud_mean_by_chip"],
        "log_p_cloud_var_by_chip": maps["log_p_cloud_var_by_chip"],
        "p_cloud_mean_by_chip": maps["p_cloud_mean_by_chip"],
        "p_cloud_std_by_chip": maps["p_cloud_std_by_chip"],
    }
    _plot_surface_map_figures(
        out_dir,
        "cloud_pressure_perturbation",
        maps["perturbation_mean_by_chip"],
        maps["perturbation_var_by_chip"],
        _map_plot_cmap("cloud_pressure_perturbation", args.cloud_fraction_cmap),
        chip_data_list,
        pressure_maps=pressure_maps,
    )

    models, median_sample, chip_samples = _reconstruct_median_models(
        chip_data_list,
        geometry,
        response_functions,
        profile_wavelengths,
        samples,
        perturbation_mean,
    )
    residuals = [
        np.asarray(chip.flux) - np.asarray(model)
        for chip, model in zip(chip_data_list, models)
    ]
    fit_residuals = _select_fit_residuals(residuals, observation_masks)
    delta_scale_by_chip = []
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_index = chip_data.chip_index
        base_profile, contrast_profile = _evaluate_response(
            response_functions[chip_position],
            median_sample,
        )
        delta_scale = float(np.sqrt(np.mean(np.asarray(contrast_profile) ** 2)))
        delta_scale_by_chip.append(delta_scale)
        delta_s_mean = np.asarray(perturbation_mean) * delta_scale
        delta_s_var = np.asarray(perturbation_var) * delta_scale**2
        np.save(out_dir / f"delta_s_mean_chip{chip_index}.npy", delta_s_mean)
        np.save(out_dir / f"delta_s_var_chip{chip_index}.npy", delta_s_var)
        np.save(
            out_dir / f"model_spectrum_chip{chip_index}.npy",
            np.asarray(models[chip_position]),
        )
        np.save(out_dir / f"residual_chip{chip_index}.npy", residuals[chip_position])
        np.save(
            out_dir / f"cloud_pressure_perturbation_mean_chip{chip_index}.npy",
            maps["perturbation_mean_by_chip"][chip_position],
        )
        np.save(
            out_dir / f"cloud_pressure_perturbation_var_chip{chip_index}.npy",
            maps["perturbation_var_by_chip"][chip_position],
        )
        np.save(
            out_dir / f"log_p_cloud_mean_chip{chip_index}.npy",
            maps["log_p_cloud_mean_by_chip"][chip_position],
        )
        np.save(
            out_dir / f"log_p_cloud_var_chip{chip_index}.npy",
            maps["log_p_cloud_var_by_chip"][chip_position],
        )
        np.save(
            out_dir / f"p_cloud_mean_chip{chip_index}.npy",
            maps["p_cloud_mean_by_chip"][chip_position],
        )
        np.save(
            out_dir / f"p_cloud_std_chip{chip_index}.npy",
            maps["p_cloud_std_by_chip"][chip_position],
        )
        _plot_delta_s(
            delta_s_mean,
            np.sqrt(delta_s_var),
            out_dir / f"figure8_delta_s_chip{chip_index}.png",
        )
        _plot_figure9(
            chip_data.wavelengths,
            np.where(
                observation_masks[chip_position],
                chip_data.flux,
                np.nan,
            ),
            np.where(
                observation_masks[chip_position],
                np.asarray(models[chip_position]),
                np.nan,
            ),
            float(np.asarray(chip_samples[chip_position]["sigma_d"])),
            out_dir / f"figure9_joint_chip{chip_index}.png",
        )
        _write_cloud_fraction_diagnostics(
            out_dir / f"cloud_pressure_perturbation_diagnostics_chip{chip_index}.json",
            maps["perturbation_mean_by_chip"][chip_position],
            np.sqrt(maps["perturbation_var_by_chip"][chip_position]),
            perturbation_mean,
        )
        _write_pressure_map_diagnostics(
            out_dir / f"cloud_pressure_map_diagnostics_chip{chip_index}.json",
            maps["log_p_cloud_mean_by_chip"][chip_position],
            np.sqrt(maps["log_p_cloud_var_by_chip"][chip_position]),
            maps["p_cloud_mean_by_chip"][chip_position],
            maps["p_cloud_std_by_chip"][chip_position],
            maps["perturbation_mean_by_chip"][chip_position],
            np.sqrt(maps["perturbation_var_by_chip"][chip_position]),
        )

    _plot_delta_s(
        np.asarray(perturbation_mean),
        np.sqrt(np.asarray(perturbation_var)),
        out_dir / "figure8_shared_contrast_joint.png",
        mean_title="Mean shared contrast",
    )
    joint_diagnostics_path = out_dir / "joint_chip_diagnostics.json"
    _write_joint_diagnostics(
        joint_diagnostics_path,
        {name: value for name, value in samples.items() if np.asarray(value).ndim > 0},
        fit_residuals,
        perturbation_mean,
        maps["perturbation_mean_by_chip"],
    )
    joint_diagnostics = json.loads(
        joint_diagnostics_path.read_text(encoding="utf-8")
    )
    joint_diagnostics.update(
        {
            "residual_statistic_scope": "likelihood_observations",
            **observation_mask_summary,
        }
    )
    joint_diagnostics_path.write_text(
        json.dumps(joint_diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    product_summary = {
        "sample_path": str(args.samples),
        "sample_sha256": sample_sha256,
        "chip_indices": chip_indices,
        "nside": nside,
        "map_sample_count": int(len(sample_indices)),
        "map_sample_indices_min": int(np.min(sample_indices)),
        "map_sample_indices_max": int(np.max(sample_indices)),
        "setup_seconds": setup_seconds,
        "delta_scale_by_chip": delta_scale_by_chip,
        "pressure_derivative_method": "on_the_fly_autodiff",
        "retrieval_x64": bool(np.asarray(samples.get("x64", False)).item()),
        "product_x64": bool(args.x64),
        "geometry_method": "retrieval_precision_constants_promoted",
        "gp_jitter": float(args.gp_jitter),
        "noise_jitter": float(args.noise_jitter),
        **observation_mask_summary,
        "pressure_gp": pressure_gp_summary,
        "absolute_pressure_moment_method": absolute_pressure_moment_method,
    }
    (out_dir / "on_the_fly_product_summary.json").write_text(
        json.dumps(product_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"On-the-fly products saved to {out_dir}")


if __name__ == "__main__":
    main()
