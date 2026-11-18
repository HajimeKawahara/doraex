"""Build primary map and spectrum products for the M9 v1 exact-RT chain."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.operators.design_matrix import (  # noqa: E402
    integrate_pixel_profiles_from_times,
)
from doraex.workflows.on_the_fly_pressure_retrieval import (  # noqa: E402
    EXACT_RT_DISK_NORMALIZATION,
    build_response_functions,
)
from make_milestone2_fixed_products import (  # noqa: E402
    _plot_figure9,
)
from make_milestone2_free_t0_cloud_products import (  # noqa: E402
    _select_sample_indices,
)
from make_milestone2_joint_chip_products import (  # noqa: E402
    _map_plot_cmap,
    _plot_surface_map_figures,
    _write_joint_diagnostics,
    _write_pressure_map_diagnostics,
)
from make_milestone4_on_the_fly_products import (  # noqa: E402
    _build_product_geometry,
    _file_sha256,
    _load_chip_data_from_samples,
    _load_observation_masks,
    _load_profile_wavelengths,
    _save_pressure_maps,
    _select_fit_residuals,
)


DEFAULT_SAMPLES = (
    ROOT
    / "results"
    / "m9"
    / "v1"
    / "exact_rt_diffgrid_monotone_preconditioned_m8map_seed0"
    / "samples.npz"
)
DEFAULT_OUT_DIR = ROOT / "results" / "m9" / "v1" / "exact_rt_diffgrid_products"
DEFAULT_OPACITY_CACHE = ROOT / "data" / "opacities" / "luhman16b_diffgrid"
REPRESENTATIVE_SAMPLE_NAMES = (
    "T0",
    "alpha",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
    "logg",
    "log_p_cloud",
    "A",
    "sigma_d",
    "log_w",
    "v",
    "cosi",
    "u1",
    "u2",
    "P",
)


def parse_args(*, default_samples=DEFAULT_SAMPLES, default_out_dir=DEFAULT_OUT_DIR):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build direct posterior products for M9 exact pixel RT."
    )
    parser.add_argument("--samples", default=str(default_samples))
    parser.add_argument("--out-dir", default=str(default_out_dir))
    parser.add_argument(
        "--database-dir",
        default=str(Path.home() / "data_mol" / ".database"),
    )
    parser.add_argument("--opacity-cache-dir", default=str(DEFAULT_OPACITY_CACHE))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--max-map-samples", type=int, default=None)
    parser.add_argument(
        "--x64",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 64-bit JAX evaluation instead of the retrieval's 32-bit target.",
    )
    return parser.parse_args()


def _scalar(samples, name):
    """Return one required scalar archive value."""

    if name not in samples:
        raise KeyError(f"Missing required archive value: {name}")
    value = np.asarray(samples[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {value.shape}.")
    return value.item()


def _exact_rt_local_normalization(samples):
    """Return the archived local normalization with a legacy fallback."""

    if "exact_rt_local_normalization" not in samples:
        return "per_pixel_mean"
    normalization = str(_scalar(samples, "exact_rt_local_normalization"))
    if normalization not in ("per_pixel_mean", "none"):
        raise ValueError(
            "exact_rt_local_normalization must be 'per_pixel_mean' or 'none'."
        )
    return normalization


def _exact_rt_disk_normalization(samples):
    """Return and validate the exact disk-normalization contract."""

    if "exact_rt_disk_normalization" not in samples:
        return EXACT_RT_DISK_NORMALIZATION
    normalization = str(_scalar(samples, "exact_rt_disk_normalization"))
    if normalization != EXACT_RT_DISK_NORMALIZATION:
        raise ValueError(
            "Unexpected exact_rt_disk_normalization: " + normalization
        )
    return normalization


def _validate_exact_archive(samples):
    """Validate the saved deterministic map contract for exact pixel RT."""

    if not bool(_scalar(samples, "exact_pixel_rt")):
        raise ValueError("M9 exact products require exact_pixel_rt=True.")
    if str(_scalar(samples, "opacity_mode")) != "diffgrid":
        raise ValueError("M9 exact products require opacity_mode='diffgrid'.")
    if not bool(_scalar(samples, "zero_mean_pressure_map")):
        raise ValueError("M9 exact products require a zero-mean pressure map.")
    _exact_rt_local_normalization(samples)
    _exact_rt_disk_normalization(samples)

    latent = np.asarray(samples["pressure_map_latent"])
    delta_log_p = np.asarray(samples["delta_log_p"], dtype=np.float64)
    log_p_cloud = np.asarray(samples["log_p_cloud"], dtype=np.float64)
    log_p_cloud_map = np.asarray(samples["log_p_cloud_map"], dtype=np.float64)
    sample_count = log_p_cloud.shape[0]
    pixel_count = 12 * int(_scalar(samples, "nside")) ** 2
    expected_rank = pixel_count - 1
    if latent.shape != (sample_count, expected_rank):
        raise ValueError(
            "pressure_map_latent has shape "
            f"{latent.shape}, expected {(sample_count, expected_rank)}."
        )
    expected_map_shape = (sample_count, pixel_count)
    if delta_log_p.shape != expected_map_shape:
        raise ValueError(
            f"delta_log_p has shape {delta_log_p.shape}, "
            f"expected {expected_map_shape}."
        )
    if log_p_cloud_map.shape != expected_map_shape:
        raise ValueError(
            f"log_p_cloud_map has shape {log_p_cloud_map.shape}, "
            f"expected {expected_map_shape}."
        )
    if not all(
        np.all(np.isfinite(values))
        for values in (latent, delta_log_p, log_p_cloud, log_p_cloud_map)
    ):
        raise ValueError("Exact pressure-map samples must all be finite.")
    if not np.allclose(
        log_p_cloud_map,
        log_p_cloud[:, None] + delta_log_p,
        rtol=2.0e-6,
        atol=2.0e-6,
    ):
        raise ValueError("Saved log-pressure maps violate center + perturbation.")
    maximum_mean = float(np.max(np.abs(np.mean(delta_log_p, axis=1))))
    if maximum_mean >= 1.0e-5:
        raise ValueError(
            "Saved pressure perturbations are not zero mean: "
            f"max_abs_mean={maximum_mean:.6e}."
        )
    return sample_count, pixel_count


def _direct_pressure_map_moments(samples, sample_indices):
    """Compute map moments directly from exact posterior samples."""

    indices = np.asarray(sample_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("sample_indices must be a non-empty one-dimensional array.")
    delta_log_p = np.asarray(samples["delta_log_p"], dtype=np.float64)[indices]
    log_p_cloud_map = np.asarray(
        samples["log_p_cloud_map"], dtype=np.float64
    )[indices]
    pressure_map = np.power(10.0, log_p_cloud_map)
    return {
        "perturbation_mean": np.mean(delta_log_p, axis=0),
        "perturbation_var": np.var(delta_log_p, axis=0),
        "log_p_cloud_mean": np.mean(log_p_cloud_map, axis=0),
        "log_p_cloud_var": np.var(log_p_cloud_map, axis=0),
        "p_cloud_mean": np.mean(pressure_map, axis=0),
        "p_cloud_std": np.std(pressure_map, axis=0),
    }


def _posterior_median_sample(samples):
    """Return marginal medians needed by the representative exact model."""

    missing = [name for name in REPRESENTATIVE_SAMPLE_NAMES if name not in samples]
    if missing:
        raise KeyError(
            "Missing representative-model samples: " + ", ".join(missing)
        )
    return {
        name: np.median(np.asarray(samples[name]), axis=0)
        for name in REPRESENTATIVE_SAMPLE_NAMES
    }


def _write_pressure_perturbation_diagnostics(path, mean, variance):
    """Write map diagnostics with pressure-specific field names."""

    mean = np.asarray(mean)
    std = np.sqrt(np.asarray(variance))
    payload = {
        "pressure_perturbation_mean_min": float(np.min(mean)),
        "pressure_perturbation_mean_max": float(np.max(mean)),
        "pressure_perturbation_mean_rms": float(np.sqrt(np.mean(mean**2))),
        "pressure_perturbation_std_min": float(np.min(std)),
        "pressure_perturbation_std_median": float(np.median(std)),
        "pressure_perturbation_std_max": float(np.max(std)),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _exact_model_for_chip(
    response,
    chip_data,
    geometry,
    profile_wavelengths,
    sample,
    perturbation_mean,
    chip_position,
):
    """Evaluate the retrieval's exact disk integration for one chip."""

    center = jnp.asarray(sample["log_p_cloud"])
    profile_pressures = jnp.concatenate(
        (jnp.atleast_1d(center), center + jnp.asarray(perturbation_mean))
    )
    profiles = response(
        jnp.asarray(sample["T0"]),
        jnp.asarray(sample["alpha"]),
        jnp.asarray(sample["log_vmr_co"]),
        jnp.asarray(sample["log_vmr_h2o"]),
        jnp.asarray(sample["log_vmr_ch4"]),
        jnp.asarray(sample["log_vmr_hf"]),
        jnp.asarray(sample["logg"]),
        profile_pressures,
    )
    pixel_profiles = profiles[1:]
    if pixel_profiles.shape[0] != geometry.theta.shape[0]:
        raise ValueError(
            "Exact response returned "
            f"{pixel_profiles.shape[0]} pixel profiles for "
            f"{geometry.theta.shape[0]} pixels."
        )
    integration_kwargs = {"rest_wavelengths": profile_wavelengths}
    integration_args = (
        geometry.theta,
        geometry.phi,
        jnp.asarray(sample["v"]),
        jnp.arccos(jnp.asarray(sample["cosi"])),
        jnp.asarray(sample["u1"]),
        jnp.asarray(sample["u2"]),
        jnp.asarray(chip_data.obs_times),
        jnp.asarray(sample["P"]),
        jnp.asarray(chip_data.wavelengths),
    )
    weights = jnp.exp(jnp.asarray(sample["log_w"])[chip_position])
    raw_exact = integrate_pixel_profiles_from_times(
        *integration_args,
        pixel_profiles,
        weights=weights,
        **integration_kwargs,
    )
    uniform_profiles = jnp.broadcast_to(profiles[0], pixel_profiles.shape)
    raw_uniform = integrate_pixel_profiles_from_times(
        *integration_args,
        uniform_profiles,
        weights=weights,
        **integration_kwargs,
    )
    normalization = (
        jnp.asarray(sample["A"])[chip_position] * jnp.mean(raw_uniform)
    )
    return (raw_exact / normalization).reshape(chip_data.flux.shape)


def _build_exact_responses(args, chip_data_list, profile_wavelengths, samples):
    """Build the same DiffGrid exact-response functions as the retrieval."""

    response_args = SimpleNamespace(
        database_dir=args.database_dir,
        opacity_cache_dir=args.opacity_cache_dir,
        nx=args.nx,
        opacity_mode=str(_scalar(samples, "opacity_mode")),
        diffgrid_temperature_nodes=int(
            _scalar(samples, "diffgrid_temperature_nodes")
        ),
        exact_pixel_rt=True,
        exact_rt_local_normalization=_exact_rt_local_normalization(samples),
        rt_pixel_batch_size=int(_scalar(samples, "rt_pixel_batch_size")),
    )
    return [
        jax.jit(response)
        for response in build_response_functions(
            response_args,
            chip_data_list,
            profile_wavelengths=profile_wavelengths,
        )
    ]


def _save_chip_products(
    out_dir,
    samples,
    chip_data_list,
    observation_masks,
    maps,
    models,
    representative_sample,
    perturbation_mean,
):
    """Save exact spectra, map arrays, figures, and diagnostics."""

    residuals = [
        np.asarray(chip.flux) - np.asarray(model)
        for chip, model in zip(chip_data_list, models)
    ]
    fit_residuals = _select_fit_residuals(residuals, observation_masks)
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_index = int(chip_data.chip_index)
        arrays = {
            "cloud_pressure_perturbation_mean": maps[
                "perturbation_mean_by_chip"
            ][chip_position],
            "cloud_pressure_perturbation_var": maps[
                "perturbation_var_by_chip"
            ][chip_position],
            "log_p_cloud_mean": maps["log_p_cloud_mean_by_chip"][chip_position],
            "log_p_cloud_var": maps["log_p_cloud_var_by_chip"][chip_position],
            "p_cloud_mean": maps["p_cloud_mean_by_chip"][chip_position],
            "p_cloud_std": maps["p_cloud_std_by_chip"][chip_position],
        }
        for stem, values in arrays.items():
            np.save(out_dir / f"{stem}_chip{chip_index}.npy", values)
        np.save(out_dir / f"model_spectrum_chip{chip_index}.npy", models[chip_position])
        np.save(out_dir / f"residual_chip{chip_index}.npy", residuals[chip_position])
        _plot_figure9(
            chip_data.wavelengths,
            np.where(observation_masks[chip_position], chip_data.flux, np.nan),
            np.where(observation_masks[chip_position], models[chip_position], np.nan),
            float(np.asarray(representative_sample["sigma_d"])[chip_position]),
            out_dir / f"figure9_joint_chip{chip_index}.png",
            title="Observed spectra and exact pixel-RT reconstruction",
        )
        _write_pressure_perturbation_diagnostics(
            out_dir
            / f"cloud_pressure_perturbation_diagnostics_chip{chip_index}.json",
            arrays["cloud_pressure_perturbation_mean"],
            arrays["cloud_pressure_perturbation_var"],
        )
        _write_pressure_map_diagnostics(
            out_dir / f"cloud_pressure_map_diagnostics_chip{chip_index}.json",
            arrays["log_p_cloud_mean"],
            np.sqrt(arrays["log_p_cloud_var"]),
            arrays["p_cloud_mean"],
            arrays["p_cloud_std"],
            arrays["cloud_pressure_perturbation_mean"],
            np.sqrt(arrays["cloud_pressure_perturbation_var"]),
        )

    diagnostics_path = out_dir / "joint_chip_diagnostics.json"
    _write_joint_diagnostics(
        diagnostics_path,
        {name: value for name, value in samples.items() if np.asarray(value).ndim > 0},
        fit_residuals,
        perturbation_mean,
        maps["perturbation_mean_by_chip"],
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["pressure_perturbation_mean_min_by_chip"] = diagnostics.pop(
        "cloud_fraction_mean_min_by_chip"
    )
    diagnostics["pressure_perturbation_mean_max_by_chip"] = diagnostics.pop(
        "cloud_fraction_mean_max_by_chip"
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    return residuals, diagnostics_path


def main(*, default_samples=DEFAULT_SAMPLES, default_out_dir=DEFAULT_OUT_DIR):
    """Build direct exact-map moments and one representative exact spectrum."""

    args = parse_args(
        default_samples=default_samples,
        default_out_dir=default_out_dir,
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    samples_path = Path(args.samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = dict(np.load(samples_path, allow_pickle=False))
    sample_count, _ = _validate_exact_archive(samples)
    chip_indices = [int(value) for value in np.asarray(samples["chip_indices"])]
    chip_data_list = _load_chip_data_from_samples(samples, chip_indices)
    observation_masks, observation_mask_summary = _load_observation_masks(
        samples, chip_data_list
    )
    profile_wavelengths = _load_profile_wavelengths(samples, chip_data_list)
    geometry = _build_product_geometry(
        samples,
        int(_scalar(samples, "nside")),
        args.x64,
    )

    sample_indices = _select_sample_indices(sample_count, args.max_map_samples)
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    moments = _direct_pressure_map_moments(samples, sample_indices)
    maps = _save_pressure_maps(
        out_dir,
        samples,
        chip_data_list,
        moments["perturbation_mean"],
        moments["perturbation_var"],
        moments,
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
        _map_plot_cmap("cloud_pressure_perturbation", None),
        chip_data_list,
        pressure_maps=pressure_maps,
    )

    setup_start = time.time()
    responses = _build_exact_responses(
        args, chip_data_list, profile_wavelengths, samples
    )
    setup_seconds = time.time() - setup_start
    representative_sample = _posterior_median_sample(samples)
    models = []
    for chip_position, (response, chip_data, profile_grid) in enumerate(
        zip(responses, chip_data_list, profile_wavelengths)
    ):
        start = time.time()
        model = _exact_model_for_chip(
            response,
            chip_data,
            geometry,
            profile_grid,
            representative_sample,
            moments["perturbation_mean"],
            chip_position,
        )
        model = np.asarray(jax.block_until_ready(model))
        if not np.all(np.isfinite(model)):
            raise ValueError(f"Exact representative model for chip {chip_data.chip_index} is non-finite.")
        models.append(model)
        print(
            f"Built exact representative model for chip {chip_data.chip_index} "
            f"in {time.time() - start:.2f} s"
        )

    _, diagnostics_path = _save_chip_products(
        out_dir,
        samples,
        chip_data_list,
        observation_masks,
        maps,
        models,
        representative_sample,
        moments["perturbation_mean"],
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.update(
        {
            "residual_statistic_scope": "likelihood_observations",
            "model_spectrum_method": (
                "exact_rt_diffgrid_at_marginal_medians_and_mean_map"
            ),
            "exact_rt_local_normalization": _exact_rt_local_normalization(
                samples
            ),
            "exact_rt_disk_normalization": _exact_rt_disk_normalization(
                samples
            ),
            **observation_mask_summary,
        }
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    np.save(out_dir / "pressure_gp_eigenvalues.npy", samples["pressure_gp_eigenvalues"])
    summary = {
        "sample_path": str(samples_path),
        "sample_sha256": _file_sha256(samples_path),
        "chip_indices": chip_indices,
        "nside": int(_scalar(samples, "nside")),
        "map_sample_count": int(len(sample_indices)),
        "map_sample_indices_min": int(np.min(sample_indices)),
        "map_sample_indices_max": int(np.max(sample_indices)),
        "setup_seconds": setup_seconds,
        "pressure_derivative_method": str(
            _scalar(samples, "pressure_derivative_method")
        ),
        "pressure_map_moment_method": "direct_exact_rt_posterior_samples",
        "absolute_pressure_moment_method": "direct_exact_rt_posterior_samples",
        "model_spectrum_method": (
            "exact_rt_diffgrid_at_marginal_medians_and_mean_map"
        ),
        "representative_pressure_map_statistic": "posterior_mean_delta_log_p",
        "representative_shared_parameter_statistic": "marginal_posterior_median",
        "retrieval_x64": bool(_scalar(samples, "x64")),
        "product_x64": bool(args.x64),
        "opacity_mode": str(_scalar(samples, "opacity_mode")),
        "exact_rt_local_normalization": _exact_rt_local_normalization(samples),
        "exact_rt_disk_normalization": _exact_rt_disk_normalization(samples),
        "diffgrid_interpolation_limiter": str(
            _scalar(samples, "diffgrid_interpolation_limiter")
        ),
        "diffgrid_temperature_nodes": int(
            _scalar(samples, "diffgrid_temperature_nodes")
        ),
        "rt_pixel_batch_size": int(_scalar(samples, "rt_pixel_batch_size")),
        "geometry_method": "retrieval_precision_constants_promoted",
        **observation_mask_summary,
        "pressure_gp": {
            "factorization": str(_scalar(samples, "pressure_gp_factorization")),
            "eigenvalue_source": "saved_samples",
            "pixel_eigenvectors_saved": False,
        },
    }
    (out_dir / "on_the_fly_product_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"M9 exact products saved to {out_dir}")


if __name__ == "__main__":
    main()
