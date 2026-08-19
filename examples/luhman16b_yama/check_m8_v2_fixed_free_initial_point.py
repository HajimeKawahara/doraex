"""Replay the M8 v2 fixed/free sigma control at one physical initial point."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model, log_density


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from examples.luhman16b_yama import m8_v2_run  # noqa: E402


FORWARD_RTOL = 2.0e-5
FORWARD_ATOL = 2.0e-6
GRADIENT_RTOL = 5.0e-4
GRADIENT_ATOL = 2.0e-3
# At the established potential-energy magnitude near 8.8e4, one float32 ULP
# is 0.0078125. This tolerance covers subtraction of two such log joints.
LOG_JOINT_ATOL = 2.0e-2
EXPECTED_WORKFLOW_SHA256 = (
    "582f79edad35a1aa819cf6e619caeaabbac614a1ea173dca25fbbf4f8ce41230"
)
EXPECTED_FORWARD_SHA256 = (
    "63390bbf64f436a5ed3cfd022454b22b6e6377aebe909acb21a32a5be0859a21"
)
EXPECTED_INIT_SHA256 = (
    "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89"
)
EXPECTED_FREE_WRAPPER_SHA256 = (
    "1d280912b78940da0450c97a8a0f12de5954264c0a307be07013499596812c61"
)
EXPECTED_FREE_SAMPLES_SHA256 = (
    "d0dac818908d3180ff0c22cb9ff6b1f4b389351f9c458f08d24963fe636cabc9"
)
EXPECTED_FREE_DIAGNOSTICS_SHA256 = (
    "7f4f449fa961a064337b0138edf832d0557d5dfda9882833078d5587148f725d"
)


def parse_args():
    """Parse replay-guard arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-from",
        default=str(m8_v2_run.DEFAULT_INIT_FROM),
    )
    parser.add_argument(
        "--output",
        default=str(
            m8_v2_run.DEFAULT_OUT_DIR / "initial_point_replay.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(path, expected):
    """Return a verified file digest or raise on provenance drift."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing replay input: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}."
        )
    return actual


def _fixed_wrapper_args(init_from, seed):
    """Replay the dedicated wrapper's exact option routing without sampling."""

    captured = {}
    original_main = retrieval.main
    original_argv = sys.argv

    def capture_args():
        captured["args"] = retrieval.parse_args()

    try:
        retrieval.main = capture_args
        sys.argv = [
            str(Path(m8_v2_run.__file__)),
            "--init-from",
            str(init_from),
            "--seed",
            str(seed),
        ]
        m8_v2_run.main()
    finally:
        retrieval.main = original_main
        sys.argv = original_argv
    return captured["args"]


def _array_comparison(first, second, *, rtol, atol):
    """Summarize one fixed/free array comparison."""

    first = np.asarray(jax.device_get(first))
    second = np.asarray(jax.device_get(second))
    shapes_match = first.shape == second.shape
    finite_values = bool(
        np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    )
    maximum_absolute_difference = (
        float(np.max(np.abs(first - second)))
        if shapes_match and first.size
        else None
    )
    return {
        "shape": list(first.shape),
        "shapes_match": shapes_match,
        "finite": finite_values,
        "maximum_absolute_difference": maximum_absolute_difference,
        "matches": bool(
            shapes_match
            and finite_values
            and np.allclose(first, second, rtol=rtol, atol=atol)
        ),
    }


def replay_fixed_free_initial_point(
    fixed_model,
    free_model,
    fixed_initial_values,
    free_initial_values,
    *,
    physical_sigma_log_p,
    sigma_log_p_scale,
    rng_key,
):
    """Compare the two target dimensions at a common physical point."""

    fixed_initial_values = dict(fixed_initial_values)
    free_initial_values = dict(free_initial_values)
    physical_sigma_log_p = float(physical_sigma_log_p)

    fixed_trace = handlers.trace(
        handlers.substitute(
            handlers.seed(fixed_model, rng_key),
            data=fixed_initial_values,
        )
    ).get_trace()
    free_trace = handlers.trace(
        handlers.substitute(
            handlers.seed(free_model, rng_key),
            data=free_initial_values,
        )
    ).get_trace()
    fixed_sigma = np.asarray(
        jax.device_get(fixed_trace["sigma_log_p"]["value"])
    )
    free_sigma = np.asarray(
        jax.device_get(free_trace["sigma_log_p"]["value"])
    )
    expected_sigma = np.asarray(physical_sigma_log_p, dtype=free_sigma.dtype)
    sigma_exact_match = bool(
        np.array_equal(fixed_sigma, free_sigma)
        and np.array_equal(free_sigma, expected_sigma)
    )
    fixed_observation = fixed_trace["obs"]["fn"]
    free_observation = free_trace["obs"]["fn"]
    forward = {
        name: _array_comparison(
            getattr(fixed_observation, name),
            getattr(free_observation, name),
            rtol=FORWARD_RTOL,
            atol=FORWARD_ATOL,
        )
        for name in ("loc", "cov_factor", "cov_diag")
    }
    del fixed_trace, free_trace, fixed_observation, free_observation

    fixed_constrained_log_joint, _ = log_density(
        fixed_model,
        (),
        {},
        fixed_initial_values,
    )
    free_constrained_log_joint, _ = log_density(
        free_model,
        (),
        {},
        free_initial_values,
    )
    expected_prior_contribution = dist.HalfNormal(
        sigma_log_p_scale
    ).log_prob(jnp.asarray(physical_sigma_log_p))
    constrained_delta = (
        free_constrained_log_joint - fixed_constrained_log_joint
    )
    constrained_delta_error = constrained_delta - expected_prior_contribution

    fixed_model_info = initialize_model(
        rng_key,
        fixed_model,
        init_strategy=init_to_value(values=fixed_initial_values),
    )
    fixed_unconstrained = {
        name: np.asarray(jax.device_get(value))
        for name, value in fixed_model_info.param_info.z.items()
    }
    fixed_gradients = {
        name: np.asarray(jax.device_get(value))
        for name, value in fixed_model_info.param_info.z_grad.items()
    }
    fixed_unconstrained_log_joint = float(
        jax.device_get(-fixed_model_info.param_info.potential_energy)
    )
    del fixed_model_info
    gc.collect()
    jax.clear_caches()

    free_model_info = initialize_model(
        rng_key,
        free_model,
        init_strategy=init_to_value(values=free_initial_values),
    )
    free_unconstrained = {
        name: np.asarray(jax.device_get(value))
        for name, value in free_model_info.param_info.z.items()
    }
    free_gradients = {
        name: np.asarray(jax.device_get(value))
        for name, value in free_model_info.param_info.z_grad.items()
    }
    free_unconstrained_log_joint = float(
        jax.device_get(-free_model_info.param_info.potential_energy)
    )
    del free_model_info

    sigma_unconstrained = jnp.asarray(free_unconstrained["sigma_log_p"])
    positive_transform = biject_to(dist.constraints.positive)
    sigma_jacobian = positive_transform.log_abs_det_jacobian(
        sigma_unconstrained,
        jnp.asarray(physical_sigma_log_p),
    ).sum()
    expected_unconstrained_contribution = float(
        expected_prior_contribution + sigma_jacobian
    )
    unconstrained_delta = (
        free_unconstrained_log_joint - fixed_unconstrained_log_joint
    )
    unconstrained_delta_error = (
        unconstrained_delta - expected_unconstrained_contribution
    )

    fixed_sites = set(fixed_unconstrained)
    free_sites = set(free_unconstrained)
    shared_sites = sorted(fixed_sites & free_sites)
    site_structure_matches = free_sites == fixed_sites | {"sigma_log_p"}
    shared_gradients = {
        name: _array_comparison(
            fixed_gradients[name],
            free_gradients[name],
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        )
        for name in shared_sites
    }
    free_sigma_gradient_finite = bool(
        np.all(np.isfinite(free_gradients["sigma_log_p"]))
    )
    log_joint_values = np.asarray(
        [
            fixed_constrained_log_joint,
            free_constrained_log_joint,
            fixed_unconstrained_log_joint,
            free_unconstrained_log_joint,
            expected_prior_contribution,
            sigma_jacobian,
        ],
        dtype=float,
    )
    log_joint_values_finite = bool(np.all(np.isfinite(log_joint_values)))
    constrained_delta_matches = bool(
        abs(float(constrained_delta_error)) <= LOG_JOINT_ATOL
    )
    unconstrained_delta_matches = bool(
        abs(float(unconstrained_delta_error)) <= LOG_JOINT_ATOL
    )
    passed = bool(
        sigma_exact_match
        and all(item["matches"] for item in forward.values())
        and site_structure_matches
        and all(item["matches"] for item in shared_gradients.values())
        and free_sigma_gradient_finite
        and log_joint_values_finite
        and constrained_delta_matches
        and unconstrained_delta_matches
    )
    return {
        "passed": passed,
        "conditional_intervention": (
            "The fixed arm has one fewer target dimension than the free arm."
        ),
        "tolerances": {
            "forward_rtol": FORWARD_RTOL,
            "forward_atol": FORWARD_ATOL,
            "gradient_rtol": GRADIENT_RTOL,
            "gradient_atol": GRADIENT_ATOL,
            "log_joint_atol": LOG_JOINT_ATOL,
        },
        "physical_sigma_log_p": {
            "expected": physical_sigma_log_p,
            "fixed": float(fixed_sigma),
            "free": float(free_sigma),
            "exact_match": sigma_exact_match,
        },
        "sample_sites": {
            "fixed": sorted(fixed_sites),
            "free": sorted(free_sites),
            "shared": shared_sites,
            "free_adds_only_sigma_log_p": site_structure_matches,
        },
        "forward": forward,
        "shared_unconstrained_gradients": shared_gradients,
        "free_sigma_unconstrained_gradient_finite": (
            free_sigma_gradient_finite
        ),
        "log_joint": {
            "all_values_finite": log_joint_values_finite,
            "fixed_constrained": float(fixed_constrained_log_joint),
            "free_constrained": float(free_constrained_log_joint),
            "constrained_free_minus_fixed": float(constrained_delta),
            "expected_free_prior_contribution": float(
                expected_prior_contribution
            ),
            "constrained_delta_error": float(constrained_delta_error),
            "constrained_delta_matches": constrained_delta_matches,
            "fixed_unconstrained": fixed_unconstrained_log_joint,
            "free_unconstrained": free_unconstrained_log_joint,
            "unconstrained_free_minus_fixed": unconstrained_delta,
            "free_transform_log_abs_det_jacobian": float(sigma_jacobian),
            "expected_prior_plus_jacobian": (
                expected_unconstrained_contribution
            ),
            "unconstrained_delta_error": unconstrained_delta_error,
            "unconstrained_delta_matches": unconstrained_delta_matches,
        },
    }


def _build_models(args):
    """Build the source-matched fixed/free M8 linear targets."""

    if args.direct_sigma_log_p or args.fixed_sigma_log_p is None:
        raise ValueError("The replay wrapper must select the fixed arm.")
    if args.fixed_sigma_log_p != m8_v2_run.V17_INITIAL_SIGMA_LOG_P:
        raise ValueError("The wrapper fixed sigma_log_p has drifted.")
    if args.eigen_basis is not None or args.map_init:
        raise ValueError("The clean replay does not support alternate targets.")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    chip_data_list = retrieval.build_chip_data(args)
    geometry = retrieval.build_luhman16b_geometry(nside=args.nside)
    pressure_gp = retrieval.build_fixed_pressure_gp_eigendecomposition(
        geometry.distance_matrix,
        args.fixed_ell_b,
        theta=geometry.theta,
        phi=geometry.phi,
    )
    data = jnp.asarray(
        np.stack([chip.flux for chip in chip_data_list], axis=0)
    )
    wavelengths = [jnp.asarray(chip.wavelengths) for chip in chip_data_list]
    obs_times = jnp.asarray(chip_data_list[0].obs_times)
    response_functions = retrieval.build_response_functions(
        args,
        chip_data_list,
        eigen_basis=None,
    )
    physical_initial_values = retrieval.load_initial_values(
        args,
        len(chip_data_list),
        len(obs_times),
    )
    loaded_sigma = float(physical_initial_values["sigma_log_p"])
    if loaded_sigma != args.fixed_sigma_log_p:
        raise ValueError(
            "The fixed value does not exactly match the free v17 archive "
            f"initialization ({args.fixed_sigma_log_p!r} != {loaded_sigma!r})."
        )
    parameter_centers, parameter_scales = (
        retrieval.build_parameter_reparameterization(
            args,
            physical_initial_values,
        )
    )
    fixed_nuisance_values = retrieval.build_fixed_nuisance_values(
        args,
        physical_initial_values,
    )
    atmosphere_rotation = retrieval.validate_atmosphere_rotation(
        args.atmosphere_rotation_matrix,
        dimension=7,
    )
    fixed_initial_values = retrieval.build_sampling_initial_values(
        args,
        physical_initial_values,
        fixed_nuisance_values,
        atmosphere_rotation,
    )
    free_initial_values = dict(fixed_initial_values)
    free_initial_values["sigma_log_p"] = jnp.asarray(loaded_sigma)
    if set(fixed_initial_values) != {"atmosphere_rotated", "A"}:
        raise ValueError(
            "Unexpected fixed-arm sample sites: "
            f"{sorted(fixed_initial_values)}."
        )

    def model_with_sigma_control(fixed_sigma_log_p, direct_sigma_log_p):
        return retrieval.on_the_fly_pressure_model(
            data=data,
            wavelengths=wavelengths,
            obs_times=obs_times,
            theta=geometry.theta,
            phi=geometry.phi,
            distance_matrix=geometry.distance_matrix,
            response_functions=response_functions,
            fixed_period=args.fixed_period,
            fixed_cosi=args.fixed_cosi,
            fixed_v=args.fixed_v,
            fixed_q1=args.fixed_q1,
            fixed_q2=args.fixed_q2,
            fix_logg=args.fix_logg,
            fixed_logg=args.init_logg,
            logg_prior_mean=args.logg_prior_mean,
            logg_prior_sigma=args.logg_prior_sigma,
            logg_bounds=(args.logg_min, args.logg_max),
            t0_bounds=(args.t0_min, args.t0_max),
            alpha_bounds=(args.alpha_min, args.alpha_max),
            log_vmr_bounds={
                "log_vmr_co": (args.log_vmr_co_min, args.log_vmr_co_max),
                "log_vmr_h2o": (
                    args.log_vmr_h2o_min,
                    args.log_vmr_h2o_max,
                ),
                "log_vmr_ch4": (
                    args.log_vmr_ch4_min,
                    args.log_vmr_ch4_max,
                ),
                "log_vmr_hf": (args.log_vmr_hf_min, args.log_vmr_hf_max),
            },
            log_p_cloud_bounds=(
                args.log_p_cloud_min,
                args.log_p_cloud_max,
            ),
            sigma_log_p_scale=args.sigma_log_p_scale,
            standardized_parameters=args.standardized_parameters,
            gaussianized_atmosphere=args.gaussianized_atmosphere,
            atmosphere_rotation=atmosphere_rotation,
            parameter_centers=parameter_centers,
            parameter_scales=parameter_scales,
            fixed_ell_b=args.fixed_ell_b,
            zero_mean_pressure_map=args.zero_mean_pressure_map,
            log_w_scale=args.log_w_scale,
            zero_mean_log_w=args.zero_mean_log_w,
            zero_sum_log_w_basis=args.zero_sum_log_w_basis,
            eigen_mode_count=None,
            eigen_sigma_scale=args.eigen_sigma_scale,
            eigen_sigma_center=jnp.asarray(args.init_sigma_eigen),
            eigen_sigma_log_raw_scale=jnp.asarray(
                args.sigma_eigen_log_raw_scale
            ),
            fixed_sigma_eigen=None,
            eigen_fixed_ell=args.fixed_ell_b,
            fixed_nuisance_values=fixed_nuisance_values,
            gp_jitter=args.gp_jitter,
            noise_jitter=args.noise_jitter,
            fixed_sigma_log_p=fixed_sigma_log_p,
            direct_sigma_log_p=direct_sigma_log_p,
            pressure_gp_eigenvalues=pressure_gp["eigenvalues"],
            pressure_gp_pixel_eigenvectors=pressure_gp[
                "pixel_eigenvectors"
            ],
            joint_atmosphere_a_sigma_d=False,
            joint_atmosphere_a_sigma_d_rotation=None,
        )

    def fixed_model():
        return model_with_sigma_control(args.fixed_sigma_log_p, False)

    def free_model():
        return model_with_sigma_control(None, True)

    physical_summary = {
        name: (
            float(value)
            if np.shape(value) == ()
            else np.asarray(value).tolist()
        )
        for name, value in physical_initial_values.items()
    }
    return (
        fixed_model,
        free_model,
        fixed_initial_values,
        free_initial_values,
        physical_summary,
    )


def _matched_configuration(args):
    """Return the fixed-arm settings that must match the reused free arm."""

    return {
        "chip_indices": args.chip_indices,
        "full_data": args.full_data,
        "nside": args.nside,
        "fixed_ell_b": args.fixed_ell_b,
        "pressure_gp_factorization": args.pressure_gp_factorization,
        "fix_logg": args.fix_logg,
        "fixed_logg": args.init_logg,
        "zero_mean_pressure_map": args.zero_mean_pressure_map,
        "zero_mean_log_w": args.zero_mean_log_w,
        "zero_sum_log_w_basis": args.zero_sum_log_w_basis,
        "fix_a": args.fix_a,
        "fix_log_w": args.fix_log_w,
        "fix_sigma_d": args.fix_sigma_d,
        "gaussianized_atmosphere": args.gaussianized_atmosphere,
        "atmosphere_rotation_label": args.atmosphere_rotation_label,
        "atmosphere_rotation_matrix": np.asarray(
            args.atmosphere_rotation_matrix
        ).reshape(7, 7).tolist(),
        "sigma_log_p_scale": args.sigma_log_p_scale,
        "dense_mass": args.dense_mass,
        "adapt_mass_matrix": args.adapt_mass_matrix,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "seed": args.seed,
        "target_accept_prob": args.target_accept_prob,
        "warmup_max_tree_depth": args.warmup_max_tree_depth,
        "max_tree_depth": args.max_tree_depth,
        "x64": args.x64,
    }


def main():
    """Run the guarded source-matched replay and write its report."""

    cli_args = parse_args()
    output = Path(cli_args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {output}")
    init_path = Path(cli_args.init_from)
    provenance = {
        "retrieval_workflow": _checked_sha256(
            retrieval.__file__,
            EXPECTED_WORKFLOW_SHA256,
        ),
        "forward_source": _checked_sha256(
            ROOT / "src" / "doraex" / "spectra" / "exojax_forward.py",
            EXPECTED_FORWARD_SHA256,
        ),
        "initialization_archive": _checked_sha256(
            init_path,
            EXPECTED_INIT_SHA256,
        ),
        "reused_free_wrapper": _checked_sha256(
            ROOT / "examples" / "luhman16b_yama" / "m7_p2_v17_run.py",
            EXPECTED_FREE_WRAPPER_SHA256,
        ),
        "reused_free_samples": _checked_sha256(
            ROOT / "results" / "m7" / "p2" / "v17" / "samples.npz",
            EXPECTED_FREE_SAMPLES_SHA256,
        ),
        "reused_free_diagnostics": _checked_sha256(
            ROOT / "results" / "m7" / "p2" / "v17" / "diagnostics.json",
            EXPECTED_FREE_DIAGNOSTICS_SHA256,
        ),
    }
    args = _fixed_wrapper_args(init_path, cli_args.seed)
    (
        fixed_model,
        free_model,
        fixed_initial_values,
        free_initial_values,
        physical_initial_values,
    ) = _build_models(args)
    replay = replay_fixed_free_initial_point(
        fixed_model,
        free_model,
        fixed_initial_values,
        free_initial_values,
        physical_sigma_log_p=m8_v2_run.V17_INITIAL_SIGMA_LOG_P,
        sigma_log_p_scale=args.sigma_log_p_scale,
        rng_key=jax.random.PRNGKey(args.seed),
    )
    reused_free_diagnostics = json.loads(
        (
            ROOT
            / "results"
            / "m7"
            / "p2"
            / "v17"
            / "diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    report = {
        "mode": "m8_v2_fixed_free_initial_point_replay",
        "passed": replay["passed"],
        "provenance_sha256": provenance,
        "fixed_wrapper": str(Path(m8_v2_run.__file__)),
        "fixed_run_label": args.run_label,
        "init_from": str(init_path),
        "physical_initial_values": physical_initial_values,
        "matched_configuration": _matched_configuration(args),
        "reused_free_arm": {
            "run_label": reused_free_diagnostics["mode"],
            "num_warmup": reused_free_diagnostics["num_warmup"],
            "num_samples": reused_free_diagnostics["num_samples"],
            "divergence_count": reused_free_diagnostics[
                "divergence_count"
            ],
            "tree_depth_cap_count": reused_free_diagnostics[
                "tree_depth_cap_count"
            ],
            "tree_depth_cap_fraction": reused_free_diagnostics[
                "tree_depth_cap_fraction"
            ],
            "median_num_steps": float(
                np.median(
                    np.load(
                        ROOT
                        / "results"
                        / "m7"
                        / "p2"
                        / "v17"
                        / "samples.npz",
                        allow_pickle=False,
                    )["extra_num_steps"]
                )
            ),
        },
        "replay": replay,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"Initial-point replay failed; see {output}.")


if __name__ == "__main__":
    main()
