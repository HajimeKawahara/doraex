"""Build M9 v3 primary products on CPU from a completed archived chain."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

import analyze_m9_v3_pilots as review
import m9_v3_run as runner
import make_m9_v1_exact_products as exact
from make_milestone2_joint_chip_products import _plot_p_cloud_map
from doraex.products import primary
from doraex.workflows.exact_pressure_investigation import ExactPressureProblem

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = ("linearization_check", "ureshino_fig10_three_map_comparison",
            "log_w_combined_diagnostic", "single_line_pressure_response",
            "line_strength_comparison_panel")


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def write_json(path, value):
    path.write_text(json.dumps(review.finite_json(value), indent=2, allow_nan=False) + "\n")


def product_archive(samples, contract, metric, arguments):
    """Join saved draws and the observation contract for existing plotters."""
    result = dict(samples)
    count, pixels = samples["delta_log_p"].shape
    rank = pixels - 1
    latent = samples["position"][:, -rank:]
    reconstructed = (latent * np.sqrt(
        samples["sigma_log_p"][:, None] ** 2 * metric["eigenvalues"] + 5e-7
    )) @ metric["eigenvectors"].T
    np.testing.assert_allclose(reconstructed, samples["delta_log_p"], rtol=0, atol=1e-9)
    result.update({
        "pressure_map_latent": latent,
        "pressure_gp_eigenvalues": metric["eigenvalues"],
        "log_p_cloud_map": samples["log_p_cloud"][:, None] + samples["delta_log_p"],
        "obs_times": contract[f"obs_times_chip{samples['chip_indices'][0]}"],
    })
    for name in ("P", "cosi", "v", "u1", "u2", "logg"):
        result[name] = np.full(count, contract[f"fixed_{name}"].item())
    masks, weights = [], []
    for chip in samples["chip_indices"]:
        suffix = f"chip{chip}"
        shape = tuple(contract[f"shape_{suffix}"])
        indices = contract[f"valid_indices_{suffix}"]
        flux = np.zeros(np.prod(shape))
        flux[indices] = contract[f"observed_{suffix}"]
        mask = np.zeros(flux.size, dtype=bool)
        mask[indices] = True
        result[f"flux_{suffix}"] = flux.reshape(shape)
        result[f"observation_mask_{suffix}"] = mask.reshape(shape)
        for name in ("wavelengths", "profile_wavelengths"):
            result[f"{name}_{suffix}"] = contract[f"{name}_{suffix}"]
        np.testing.assert_array_equal(result["obs_times"], contract[f"obs_times_{suffix}"])
        masks.append(mask)
        weights.append(np.log(contract[f"weights_{suffix}"]))
    result["log_w"] = np.broadcast_to(np.stack(weights), (count, len(weights), len(weights[0])))
    result["observation_indices"] = np.flatnonzero(np.concatenate(masks))
    metadata = {
        "nside": int(np.sqrt(pixels / 12)), "exact_pixel_rt": True,
        "zero_mean_pressure_map": True, "opacity_mode": "diffgrid",
        "exact_rt_local_normalization": "none",
        "exact_rt_disk_normalization": exact.EXACT_RT_DISK_NORMALIZATION,
        "diffgrid_temperature_nodes": arguments.nodes,
        "diffgrid_interpolation_limiter": "fritsch_carlson_monotone_log_slopes",
        "rt_pixel_batch_size": arguments.rt_pixel_batch_size,
        "pressure_gp_factorization": "fixed_eigen", "gp_jitter": 5e-7,
        "fixed_ell_b": .4, "pressure_derivative_method": "not_used_exact_pixel_rt",
        "direct_sigma_log_p": False, "sigma_log_p_parameterization": "log_halfnormal",
        "fix_logg": True, "fix_a": False, "fix_sigma_d": False, "fix_log_w": True,
        "zero_mean_log_w": True, "fix_nuisance": False, "x64": True,
        "mask_zero_flux": True, "observation_mask_rule": "finite_and_nonzero_flux",
        "observation_valid_count": sum(mask.sum() for mask in masks),
        "observation_excluded_count": sum((~mask).sum() for mask in masks),
    }
    result.update({key: np.asarray(value) for key, value in metadata.items()})
    exact._validate_exact_archive(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--pressure-max", type=float, default=50.)
    options = parser.parse_args()
    run, out, bundle = (path.resolve() for path in
                        (options.run_dir, options.out_dir, options.primary_dir))
    if out.exists() or bundle.exists():
        raise FileExistsError("Use new product and bundle directories.")
    started = time.time()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    verified = review.load_run(run)
    if not verified["contract_verified"]:
        raise ValueError("A verified observation contract is required.")
    provenance = json.loads((run / "provenance.json").read_text())
    for path, expected in provenance["source_sha256"].items():
        if runner.sha256(path) != expected:
            raise ValueError(f"Inference source changed since the run: {path}")
    arguments = SimpleNamespace(**provenance["arguments"])
    for name in ("samples", "state_file", "opacity_cache_dir", "database_dir", "forward_validation"):
        value = Path(getattr(arguments, name))
        setattr(arguments, name, value if value.is_absolute() else ROOT / value)
    _, commit = runner.active_exojax_checkout()
    if commit != provenance["exojax_commit"]:
        raise ValueError("ExoJAX differs from the archived run.")
    runner.validate_forward_gate(arguments, commit)
    samples, contract, metric = (load(run / name) for name in
                                  ("samples.npz", "observation_contract.npz", "metric.npz"))
    archive = product_archive(samples, contract, metric, arguments)
    out.mkdir(parents=True)
    (out / "RUNNING").touch()
    archive_path = out / "product_samples.npz"
    np.savez_compressed(archive_path, **archive)
    sampling = json.loads((run / "sampling.json").read_text())
    run_info = {**sampling, "mode": "m9_v3_exact_raw_intensity", "chip_indices": samples["chip_indices"],
                "num_chains": 1, "nside": 8, "full_data": True, "x64": True,
                "fix_logg": True, "fix_a": False, "fix_log_w": True, "fix_sigma_d": False,
                "marginalized_A": bool(samples["marginalized_A"]),
                "run_seconds": sampling["seconds"], "divergence_count": sampling["divergences"],
                "mean_accept_prob": sampling["acceptance"], "max_num_steps": int(samples["extra_num_steps"].max())}
    write_json(out / "diagnostics.json", run_info)
    chips = exact._load_chip_data_from_samples(archive, samples["chip_indices"])
    masks, mask_info = exact._load_observation_masks(archive, chips)
    moments = exact._direct_pressure_map_moments(archive, np.arange(len(samples["position"])))
    maps = exact._save_pressure_maps(out, archive, chips, moments["perturbation_mean"],
                                     moments["perturbation_var"], moments)
    exact._plot_surface_map_figures(out, "cloud_pressure_perturbation", maps["perturbation_mean_by_chip"],
                                   maps["perturbation_var_by_chip"], "inferno", chips, pressure_maps=maps)
    shutil.copy2(out / "figure8_p_cloud_joint.png", out / "figure8_p_cloud_joint_full_range.png")
    _plot_p_cloud_map(moments["p_cloud_mean"], np.sqrt(moments["log_p_cloud_var"]),
                     out / "figure8_p_cloud_joint.png", vmax=options.pressure_max)
    commands = []
    for filename, limit in (("figure8_p_cloud_observer.png", options.pressure_max),
                            ("figure8_p_cloud_observer_full_range.png", None)):
        command = [sys.executable, str(ROOT / "examples/luhman16b_yama/make_observer_view_cloud_pressure_map.py"),
                   "--product-dir", str(out), "--output", str(out / filename),
                   "--cosi", str(float(contract["fixed_cosi"]))]
        if limit is not None:
            command += ["--pressure-max", str(limit)]
        commands.append(command)
        subprocess.run(command, check=True, cwd=ROOT)
    print(f"Pressure maps ready: {out}", flush=True)

    # CPU and GPU float32 geometry construction can round differently. Reuse
    # the archived angles and GP basis, including their original roundoff.
    grids = exact._load_profile_wavelengths(archive, chips)
    response_args = SimpleNamespace(nx=4500, database_dir=arguments.database_dir,
                                    opacity_cache_dir=arguments.opacity_cache_dir)
    responses = exact._build_exact_responses(response_args, chips, grids, archive)
    geometry = SimpleNamespace(theta=jnp.asarray(contract["theta"]),
                               phi=jnp.asarray(contract["phi"]), nside=8)
    problem = ExactPressureProblem(
        chips, responses, grids, geometry, jnp.asarray(metric["eigenvalues"]),
        jnp.asarray(metric["eigenvectors"]), load(arguments.state_file),
        marginalized=arguments.marginalize_a, quadrature_order=arguments.quadrature_order)
    rebuilt = runner._observation_contract(problem)
    for key, value in rebuilt.items():
        np.testing.assert_array_equal(value, contract[key], err_msg=key)
    for column, chip in enumerate(chips):
        np.testing.assert_array_equal(problem.observed[column], contract[f"observed_chip{chip.chip_index}"])
    representative = exact._posterior_median_sample(archive)
    models, replay_errors = [], []
    draw = len(samples["position"]) // 2
    coherent = {name: archive[name][draw] for name in exact.REPRESENTATIVE_SAMPLE_NAMES}
    for column, (response, chip, grid) in enumerate(zip(problem.responses, chips, problem.profile_wavelengths)):
        begin = time.time()
        model_function = jax.jit(lambda state, delta, response=response, chip=chip, grid=grid, column=column:
            exact._exact_model_for_chip(response, chip, problem.geometry, grid, state, delta, column))
        model = np.asarray(model_function(representative, moments["perturbation_mean"]))
        replay = np.asarray(model_function(coherent, samples["delta_log_p"][draw]))
        reference = np.asarray(problem.raw_functions[column](samples["position"][draw])) / samples["A"][draw, column]
        error = float(np.max(np.abs(replay.reshape(-1)[problem.indices[column]] - reference)))
        np.testing.assert_allclose(replay.reshape(-1)[problem.indices[column]], reference, rtol=1e-9, atol=1e-10)
        if not np.all(np.isfinite(model)):
            raise ValueError("Non-finite representative spectrum.")
        models.append(model)
        replay_errors.append(error)
        np.save(out / f"coherent_draw{draw}_model_chip{chip.chip_index}.npy", replay)
        print(f"Chip {chip.chip_index}: representative and saved-draw check in {time.time() - begin:.1f} s; max error {error:.3g}", flush=True)
    potential_error = float(problem.potential(samples["position"][draw])) - float(samples["extra_potential_energy"][draw])
    if abs(potential_error) > 1e-5:
        raise ValueError(f"Saved-draw potential replay differs: {potential_error}")
    print(f"Saved-draw potential replay error: {potential_error:.3g}", flush=True)
    _, diagnostics_path = exact._save_chip_products(out, archive, chips, masks, maps, models,
                                                   representative, moments["perturbation_mean"])
    diagnostics = json.loads(diagnostics_path.read_text())
    method = "exact_rt_diffgrid_at_marginal_medians_and_mean_map"
    diagnostics.update({**mask_info, "residual_statistic_scope": "likelihood_observations",
                        "model_spectrum_method": method})
    write_json(diagnostics_path, diagnostics)
    np.save(out / "pressure_gp_eigenvalues.npy", metric["eigenvalues"])
    summary = {
        "source_run": str(run), "sample_path": str(run / "samples.npz"),
        "sample_sha256": runner.sha256(run / "samples.npz"),
        "product_archive_path": str(archive_path), "product_archive_sha256": runner.sha256(archive_path),
        "nside": 8, "chip_indices": samples["chip_indices"], "map_sample_count": len(samples["position"]),
        "map_sample_indices_min": 0, "map_sample_indices_max": len(samples["position"]) - 1,
        "pressure_map_moment_method": "direct_exact_rt_posterior_samples",
        "absolute_pressure_moment_method": "direct_exact_rt_posterior_samples",
        "model_spectrum_method": method, "representative_pressure_map_statistic": "posterior_mean_delta_log_p",
        "representative_shared_parameter_statistic": "marginal_posterior_median",
        "predictive_scope": "One representative reconstruction, not posterior predictive intervals; a coherent saved draw is checked separately.",
        "coherent_draw_index": draw, "coherent_replay_maximum_errors": replay_errors,
        "coherent_potential_replay_error": potential_error,
        "pressure_display_max_bar": options.pressure_max, "full_range_maps_also_saved": True,
        "retrieval_x64": True, "product_x64": True, "backend": jax.default_backend(),
        "diffgrid_temperature_nodes": arguments.nodes, "exojax_commit": commit,
        "exact_rt_local_normalization": "none", "exact_rt_disk_normalization": exact.EXACT_RT_DISK_NORMALIZATION,
        "geometry_method": "verified_archived_observation_contract", **mask_info,
        "interpretation": "Single-chain q50/q95 products for inspection; independent-chain agreement remains to be assessed.",
    }
    write_json(out / "on_the_fly_product_summary.json", summary)
    result = primary.generate_primary_products(primary.PrimaryProductConfig(
        samples_path=archive_path, product_dir=out, primary_dir=bundle, prefix="m9_v3",
        project_root=ROOT, chip_indices=tuple(int(chip) for chip in samples["chip_indices"]),
        run_map_products=False, run_single_line_products=False, run_log_w_products=False,
        run_linearization_check=False, excluded_product_names=EXCLUDED, x64=True))
    if result.missing:
        raise ValueError(f"Missing primary products: {result.missing}")
    run_summary = json.loads((bundle / "info/run_summary.json").read_text())
    if run_summary["sampled_parameter_count"] != samples["position"].shape[1]:
        raise ValueError("Primary bundle reports the wrong sampled dimension.")
    for name in ("figure8_p_cloud_joint.png", "figure8_log_p_cloud_joint.png",
                 "figure8_p_cloud_observer_full_range.png", "figure8_p_cloud_joint_full_range.png"):
        shutil.copy2(out / name, bundle / name)
    paths = [run / name for name in ("samples.npz", "mcmc_raw.npz", "metric.npz", "provenance.json", "observation_contract.npz")]
    paths += [Path(__file__), Path(exact.__file__), Path(primary.__file__)]
    write_json(out / "generation_commands.json", {"argv": sys.argv, "observer_commands": commands,
               "input_and_source_sha256": {str(path): runner.sha256(path) for path in paths},
               "backend": "cpu", "seconds": time.time() - started})
    shutil.copy2(out / "on_the_fly_product_summary.json", bundle / "info/product_reconstruction.json")
    (out / "RUNNING").rename(out / "COMPLETE")
    (bundle / "COMPLETE").touch()
    print(f"Primary products complete: {bundle}", flush=True)


if __name__ == "__main__":
    main()
