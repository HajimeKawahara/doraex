"""Create products for on-the-fly autodiff pressure retrievals."""

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


def _response_function(spectrum_function):
    """Return a function that evaluates the spectrum and pressure JVP."""

    def response(t0, alpha, zeta_vmr, log_p_cloud):
        def spectrum_at_pressure(pressure):
            return spectrum_function(t0, alpha, zeta_vmr, pressure)

        return jax.jvp(
            spectrum_at_pressure,
            (log_p_cloud,),
            (jnp.ones_like(log_p_cloud),),
        )

    return response


def _build_response_functions(args, samples, chip_data_list):
    """Build ExoJAX on-the-fly response functions for each chip."""

    response_functions = []
    for chip_data in chip_data_list:
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx,
        )
        response_functions.append(jax.jit(_response_function(model.cloudy_at_parameters)))
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


def _linear_profile_operator_from_sample(chip_data, geometry, base_profile, contrast_profile, sample):
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
    )
    norm = jnp.asarray(sample["A"]) * jnp.mean(baseline)
    return baseline / norm, contrast_matrix / norm


def _joint_operator_from_sample(chip_data_list, geometry, response_functions, sample):
    """Build the concatenated baseline and pressure-response matrix."""

    baselines = []
    contrast_matrices = []
    t0 = sample.get("T0", sample.get("t0", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.t0)))
    alpha = sample.get("alpha", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.alpha))
    zeta_vmr = sample.get("zeta_vmr", jnp.asarray(0.0))
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_sample = _chip_sample(sample, chip_position)
        base_profile, contrast_profile = response_functions[chip_position](
            jnp.asarray(t0),
            jnp.asarray(alpha),
            jnp.asarray(zeta_vmr),
            jnp.asarray(sample["log_p_cloud"])
        )
        baseline, contrast_matrix = _linear_profile_operator_from_sample(
            chip_data,
            geometry,
            base_profile,
            contrast_profile,
            chip_sample,
        )
        baselines.append(baseline)
        contrast_matrices.append(contrast_matrix)
    return jnp.concatenate(baselines, axis=0), jnp.concatenate(
        contrast_matrices,
        axis=0,
    )


def _conditional_pressure_map_for_sample(
    chip_data_list,
    geometry,
    response_functions,
    sample,
    gp_jitter,
    noise_jitter,
):
    """Compute the conditional pressure-perturbation map posterior."""

    baseline, contrast_matrix = _joint_operator_from_sample(
        chip_data_list,
        geometry,
        response_functions,
        sample,
    )
    prior_covariance = squared_exponential_covariance(
        geometry.distance_matrix,
        jnp.asarray(sample["sigma_log_p"]),
        jnp.asarray(sample["ell_b"]),
    )
    if bool(np.asarray(sample.get("zero_mean_pressure_map", False))):
        prior_covariance = project_zero_mean_covariance(prior_covariance)
    prior_covariance = add_diagonal_jitter(prior_covariance, jitter=gp_jitter)
    residual = jnp.concatenate(
        [jnp.asarray(chip.flux).reshape(-1) for chip in chip_data_list],
        axis=0,
    ) - baseline
    noise_variance = jnp.concatenate(
        [
            jnp.asarray(sample["sigma_d"])[chip_position] ** 2
            * jnp.ones(chip.flux.size)
            + noise_jitter
            for chip_position, chip in enumerate(chip_data_list)
        ],
        axis=0,
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
    return mean, covariance


def _compute_pressure_map_moments(
    chip_data_list,
    geometry,
    response_functions,
    samples,
    sample_indices,
    gp_jitter,
    noise_jitter,
):
    """Compute posterior moments for the shared pressure perturbation map."""

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    for count, index in enumerate(sample_indices, start=1):
        start = time.time()
        sample = _sample_at(samples, int(index))
        mean, covariance = _conditional_pressure_map_for_sample(
            chip_data_list,
            geometry,
            response_functions,
            sample,
            gp_jitter,
            noise_jitter,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        if count == 1 or count % 25 == 0 or count == len(sample_indices):
            elapsed = time.time() - start
            print(
                f"Processed map sample {count}/{len(sample_indices)} "
                f"(posterior index {int(index)}, {elapsed:.2f} s)"
            )

    mean_stack = jnp.stack(conditional_means, axis=0)
    perturbation_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - perturbation_mean[None, :]) ** 2, axis=0)
    perturbation_var = within + between
    return perturbation_mean, perturbation_var


def _reconstruct_median_models(chip_data_list, geometry, response_functions, samples, contrast_map):
    """Reconstruct spectra from median nonlinear parameters and the map."""

    sample = fixed_two_column_median_sample(samples)
    models = []
    chip_samples = []
    t0 = sample.get("T0", sample.get("t0", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.t0)))
    alpha = sample.get("alpha", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.alpha))
    zeta_vmr = sample.get("zeta_vmr", jnp.asarray(0.0))
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_sample = _chip_sample(sample, chip_position)
        base_profile, contrast_profile = response_functions[chip_position](
            jnp.asarray(t0),
            jnp.asarray(alpha),
            jnp.asarray(zeta_vmr),
            jnp.asarray(sample["log_p_cloud"])
        )
        baseline, contrast_matrix = _linear_profile_operator_from_sample(
            chip_data,
            geometry,
            base_profile,
            contrast_profile,
            chip_sample,
        )
        model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
        models.append(model.reshape(chip_data.flux.shape))
        chip_samples.append(chip_sample)
    return models, sample, chip_samples


def _save_pressure_maps(out_dir, samples, chip_data_list, perturbation_mean, perturbation_var):
    """Save pressure-perturbation, log-pressure, and pressure maps."""

    chip_count = len(chip_data_list)
    perturbation_mean = np.asarray(perturbation_mean)
    perturbation_var = np.asarray(perturbation_var)
    perturbation_mean_by_chip = np.tile(perturbation_mean[None, :], (chip_count, 1))
    perturbation_var_by_chip = np.tile(perturbation_var[None, :], (chip_count, 1))
    center_by_chip = _center_values_by_chip(samples, "log_p_cloud", chip_count)
    center_mean_by_chip = np.mean(center_by_chip, axis=0)
    center_var_by_chip = np.var(center_by_chip, axis=0)
    log_p_cloud_mean_by_chip = center_mean_by_chip[:, None] + perturbation_mean_by_chip
    log_p_cloud_var_by_chip = center_var_by_chip[:, None] + perturbation_var_by_chip
    p_cloud_mean_by_chip = 10.0**log_p_cloud_mean_by_chip
    p_cloud_std_by_chip = (
        np.log(10.0) * p_cloud_mean_by_chip * np.sqrt(log_p_cloud_var_by_chip)
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
        "log_p_cloud_mean_by_chip": np.load(out_dir / "log_p_cloud_mean_joint_by_chip.npy"),
        "log_p_cloud_var_by_chip": np.load(out_dir / "log_p_cloud_var_joint_by_chip.npy"),
        "p_cloud_mean_by_chip": np.load(out_dir / "p_cloud_mean_joint_by_chip.npy"),
        "p_cloud_std_by_chip": np.load(out_dir / "p_cloud_std_joint_by_chip.npy"),
    }


def main():
    """Compute and save on-the-fly pressure-retrieval products."""

    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = dict(np.load(args.samples, allow_pickle=False))
    chip_indices = (
        [int(value) for value in np.asarray(samples["chip_indices"])]
        if args.chip_indices is None
        else args.chip_indices
    )
    nside = int(np.asarray(samples["nside"])) if args.nside is None else args.nside
    chip_data_list = _load_chip_data_from_samples(samples, chip_indices)
    geometry = build_luhman16b_geometry(nside=nside)

    setup_start = time.time()
    response_functions = _build_response_functions(args, samples, chip_data_list)
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
        print(f"Reused saved pressure-map products from {out_dir}")
    else:
        perturbation_mean, perturbation_var = _compute_pressure_map_moments(
            chip_data_list,
            geometry,
            response_functions,
            samples,
            sample_indices,
            args.gp_jitter,
            args.noise_jitter,
        )
        maps = _save_pressure_maps(
            out_dir,
            samples,
            chip_data_list,
            perturbation_mean,
            perturbation_var,
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
        samples,
        perturbation_mean,
    )
    residuals = [
        np.asarray(chip.flux) - np.asarray(model)
        for chip, model in zip(chip_data_list, models)
    ]
    delta_scale_by_chip = []
    for chip_position, chip_data in enumerate(chip_data_list):
        chip_index = chip_data.chip_index
        base_profile, contrast_profile = response_functions[chip_position](
            jnp.asarray(median_sample.get("T0", median_sample.get("t0", YAMA_L16B_EXOMOL_ATMOSPHERE.t0))),
            jnp.asarray(median_sample.get("alpha", YAMA_L16B_EXOMOL_ATMOSPHERE.alpha)),
            jnp.asarray(median_sample.get("zeta_vmr", 0.0)),
            jnp.asarray(median_sample["log_p_cloud"])
        )
        delta_scale = float(np.sqrt(np.mean(np.asarray(contrast_profile) ** 2)))
        delta_scale_by_chip.append(delta_scale)
        delta_s_mean = np.asarray(perturbation_mean) * delta_scale
        delta_s_var = np.asarray(perturbation_var) * delta_scale**2
        np.save(out_dir / f"delta_s_mean_chip{chip_index}.npy", delta_s_mean)
        np.save(out_dir / f"delta_s_var_chip{chip_index}.npy", delta_s_var)
        np.save(out_dir / f"model_spectrum_chip{chip_index}.npy", np.asarray(models[chip_position]))
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
            chip_data.flux,
            np.asarray(models[chip_position]),
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
    _write_joint_diagnostics(
        out_dir / "joint_chip_diagnostics.json",
        {name: value for name, value in samples.items() if np.asarray(value).ndim > 0},
        residuals,
        perturbation_mean,
        maps["perturbation_mean_by_chip"],
    )
    product_summary = {
        "sample_path": str(args.samples),
        "chip_indices": chip_indices,
        "nside": nside,
        "map_sample_count": int(len(sample_indices)),
        "map_sample_indices_min": int(np.min(sample_indices)),
        "map_sample_indices_max": int(np.max(sample_indices)),
        "setup_seconds": setup_seconds,
        "delta_scale_by_chip": delta_scale_by_chip,
        "pressure_derivative_method": "on_the_fly_autodiff",
    }
    (out_dir / "on_the_fly_product_summary.json").write_text(
        json.dumps(product_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"On-the-fly products saved to {out_dir}")


if __name__ == "__main__":
    main()
