"""Check M9's actual monotone opacity and raw disk flux against its teacher.

This command builds a separate, explicitly real-space DiffGrid cache. It does
not run inference. Pass a coherent physical state and its source archive;
``--prepare-cache-only`` needs only the archive's wavelength grids.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_SAMPLES = ROOT / "results/m8/v12/masked_free_sigma_d_highest_precision_long_seed0/samples.npz"
DEFAULT_CACHE = ROOT / "data/opacities/luhman16b_diffgrid_m9_v3_real_space"
DEFAULT_TEACHER_CACHE = ROOT / "data/opacities/luhman16b_premodit_m9_v3"
ATMOSPHERE_NAMES = (
    "T0", "alpha", "log_vmr_co", "log_vmr_h2o", "log_vmr_ch4",
    "log_vmr_hf", "logg", "log_p_cloud",
)


def comma_list(value, converter=int):
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison_metrics(prediction, reference, observed, mask, variance):
    """Compare equal-noise Gaussian means on exactly the likelihood entries."""

    prediction, reference, observed = (
        np.asarray(value, dtype=np.float64) for value in (prediction, reference, observed)
    )
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError("Predictions and the likelihood mask must have identical shapes.")
    if not mask.any() or not np.isfinite(variance) or variance <= 0:
        raise ValueError("The mask must contain data and variance must be positive.")
    if not all(np.all(np.isfinite(value[mask])) for value in (prediction, reference, observed)):
        raise FloatingPointError("Nonfinite flux on a likelihood entry.")
    delta = (prediction - reference)[mask]
    residual = (observed - reference)[mask]
    squared_noise_distance = np.sum(delta**2) / variance
    return {
        "observation_count": int(mask.sum()),
        "flux_difference_rms": float(np.sqrt(np.mean(delta**2))),
        "flux_difference_rms_over_noise": float(np.sqrt(np.mean(delta**2) / variance)),
        "maximum_absolute_flux_difference_over_noise": float(np.max(np.abs(delta)) / np.sqrt(variance)),
        "gaussian_mean_kl": float(0.5 * squared_noise_distance),
        "delta_log_likelihood": float(np.sum(residual * delta) / variance - 0.5 * squared_noise_distance),
        "reference_log_likelihood": float(-0.5 * (np.sum(residual**2) / variance + mask.sum() * np.log(2 * np.pi * variance))),
    }


def load_state(path, samples):
    """Read one joint physical state, with explicit limb-darkening conversion."""

    with np.load(path, allow_pickle=False) as archive:
        state = {name: np.asarray(archive[name]) for name in archive.files}
    if "P" not in state and "period" in state:
        state["P"] = state["period"]
    if "u1" not in state or "u2" not in state:
        q1, q2 = float(state["q1"]), float(state["q2"])
        state["u1"] = np.asarray(2 * np.sqrt(q1) * q2)
        state["u2"] = np.asarray(np.sqrt(q1) * (1 - 2 * q2))
    required = (*ATMOSPHERE_NAMES, "P", "cosi", "v", "u1", "u2", "A", "sigma_d", "log_w", "delta_log_p")
    for name in required:
        if name not in state or not np.all(np.isfinite(state[name])):
            raise ValueError(f"Missing or nonfinite physical state: {name}.")
    chip_count = len(samples["chip_indices"])
    pixel_count = 12 * int(samples["nside"]) ** 2
    if state["delta_log_p"].shape != (pixel_count,):
        raise ValueError("The physical map does not match the archived nside.")
    for name in ("A", "sigma_d"):
        if state[name].shape != (chip_count,) or np.any(state[name] <= 0):
            raise ValueError(f"{name} must contain one positive value per archived chip.")
    if state["log_w"].shape != (chip_count, len(samples["obs_times"])):
        raise ValueError("log_w must have shape (archived chips, observation times).")
    return state


def summarize_checks(report, args):
    """Aggregate chips without confusing skipped checks with successful checks."""

    def derivative_passed(row):
        return any(
            check["derivative_relative_rms_error"] <= args.gradient_relative_tolerance
            or check["derivative_difference_rms_over_noise"] <= args.gradient_absolute_noise_tolerance
            for check in row["finite_difference_checks"]
        )

    report["forward_checks"] = {}
    report["passed_by_nodes"] = {}
    for nodes in args.nodes:
        label = f"nodes{nodes}"
        variants = [chip["variants"].get(label, {}) for chip in report["chips"].values()]
        complete = len(variants) == len(args.chip_indices) and all("disk" in item for item in variants)
        kl = sum(item.get("disk", {}).get("gaussian_mean_kl", 0.0) for item in variants)
        delta = sum(item.get("disk", {}).get("delta_log_likelihood", 0.0) for item in variants)
        accuracy_passed = complete and kl <= args.max_joint_kl and abs(delta) <= args.max_absolute_delta_log_likelihood
        gradient_rows = [row for item in variants for row in item.get("gradients", {}).values()]
        teacher_rows = [row for chip in report["chips"].values() for row in chip.get("teacher_gradients", {}).values()]
        expected_count = len(args.chip_indices) * len(args.gradient_directions)
        gradients_complete = not args.skip_gradients and expected_count > 0 and len(gradient_rows) == len(teacher_rows) == expected_count
        gradients_passed = gradients_complete and all(derivative_passed(row) for row in (*gradient_rows, *teacher_rows)) and all(
            row["teacher_derivative_relative_rms_error"] <= args.gradient_relative_tolerance
            or row["teacher_derivative_difference_rms_over_noise"] <= args.gradient_absolute_noise_tolerance
            for row in gradient_rows
        )
        report["forward_checks"][label] = {
            "complete": complete, "joint_gaussian_mean_kl": kl,
            "joint_delta_log_likelihood": delta, "accuracy_passed": accuracy_passed,
            "gradients_complete": gradients_complete, "gradients_passed": gradients_passed,
        }
        report["passed_by_nodes"][str(nodes)] = bool(accuracy_passed and gradients_passed)
    report["passed"] = all(report["passed_by_nodes"].values())


def cache_namespace(model, root, nodes, nx):
    from doraex.spectra.exojax_forward import _opacity_cache_namespace

    return _opacity_cache_namespace(
        root, model.internal_wavelength_bounds, nx=nx, pressure_top=1e-4,
        pressure_btm=1e2, nlayer=101, t_low=210.0, t_high=3500.0,
        opacity_mode="diffgrid", diffgrid_temperature_nodes=nodes,
    )


def build_diffgrid_model(teacher_model, root, nodes, nx, *, legacy=False):
    """Build or load tables while retaining the exact production slope limiter."""

    import jax
    from exojax.opacity import OpaDiffgrid, saveopa
    from doraex.spectra.exojax_forward import _prepare_monotone_diffgrid

    model = copy.copy(teacher_model)
    model.opacity_mode = "diffgrid"
    model.diffgrid_temperature_nodes = nodes
    model.opacity_cache_dir = cache_namespace(model, root, nodes, nx)
    model.opacities = {}
    records = {}
    for molecule, teacher in teacher_model.opacities.items():
        path = model.opacity_cache_dir / f"opa{molecule}.zarr"
        if path.exists():
            opa = OpaDiffgrid.from_saved_opa(str(path))
            origin = "loaded"
        elif legacy:
            raise FileNotFoundError(f"Legacy comparisons require an existing archive: {path}")
        else:
            opa = OpaDiffgrid(
                teacher, temperature_grid=1 / np.linspace(1 / 3500.0, 1 / 210.0, nodes),
                pressure_grid=np.asarray(model.art.pressure), profile_kernel="real_space",
            )
            jax.block_until_ready(opa.log_cross_section_derivative_grid)
            path.parent.mkdir(parents=True, exist_ok=True)
            saveopa(opa, str(path), format="zarr", aux={"molmass": model.mol_masses[molecule]})
            origin = "constructed"
        kernel = getattr(opa, "teacher_profile_kernel", None)
        if not legacy and kernel != "real_space":
            raise ValueError(f"Fresh cache has an unverified teacher kernel: {path}: {kernel!r}")
        opa.check_pressure_grid(np.asarray(model.art.pressure))
        if not np.array_equal(np.asarray(opa.nu_grid), np.asarray(model.nu_grid)):
            raise ValueError(f"The cached opacity wavelength grid differs from the teacher: {path}")
        if opa.diffgrid_info.inverse_temperature_grid.size != nodes:
            raise ValueError(f"The cached temperature node count is incorrect: {path}")
        _prepare_monotone_diffgrid(opa)
        model.opacities[molecule] = opa
        metadata_path = path / "zarr.json"
        records[molecule] = {
            "path": str(path), "origin": origin, "teacher_profile_kernel": kernel,
            "metadata_sha256": sha256(metadata_path),
            "table_dtype": str(opa.log_cross_section_grid.dtype),
        }
    return model, records


def opacity_comparison(model, teacher_model, state, midpoint_count):
    """Check the actual monotone evaluator, including temperature midpoints."""

    import jax
    import jax.numpy as jnp
    from doraex.spectra.exojax_forward import _monotone_diffgrid_xsmatrix

    profiles = [("state", model.art.powerlaw_temperature(state["T0"], state["alpha"]))]
    grid = np.asarray(next(iter(model.opacities.values())).diffgrid_info.inverse_temperature_grid)
    indices = np.unique(np.linspace(0, grid.size - 2, min(midpoint_count, grid.size - 1), dtype=int))
    for index in indices:
        temperature = 2 / (grid[index] + grid[index + 1])
        profiles.append((f"inverse_temperature_midpoint_{index}", jnp.full_like(model.art.pressure, temperature)))
    rows = []
    for label, temperature in profiles:
        for molecule, opa in model.opacities.items():
            actual = np.asarray(jax.block_until_ready(_monotone_diffgrid_xsmatrix(opa, temperature)), dtype=float)
            expected = np.asarray(jax.block_until_ready(teacher_model.opacities[molecule].xsmatrix(temperature, model.art.pressure)), dtype=float)
            if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
                raise FloatingPointError(f"Nonfinite opacity: {label}, {molecule}.")
            floor = float(np.exp(np.asarray(opa.diffgrid_info.log_cross_section_floor, dtype=float)))
            error = np.abs(np.log(np.maximum(actual, floor)) - np.log(np.maximum(expected, floor)))
            rows.append({
                "profile": label, "molecule": molecule,
                "absolute_log_opacity_error_rms": float(np.sqrt(np.mean(error**2))),
                "absolute_log_opacity_error_p99": float(np.quantile(error, 0.99)),
                "absolute_log_opacity_error_max": float(np.max(error)),
                "teacher_nonpositive_fraction": float(np.mean(expected <= 0)),
                "teacher_at_or_below_floor_fraction": float(np.mean(expected <= floor)),
            })
    return rows


def make_prediction(model, chip, geometry, grid, state, chip_position, batch_size):
    import jax
    import jax.numpy as jnp
    from doraex.workflows.on_the_fly_pressure_retrieval import _exact_response_function
    from make_m9_v1_exact_products import _exact_model_for_chip

    response = _exact_response_function(model.cloudy_raw_map_at_log_vmrs, batch_size)

    def predict(atmosphere, delta, amplitude):
        current = {**state, **dict(zip(ATMOSPHERE_NAMES, atmosphere, strict=True))}
        current["A"] = jnp.asarray(state["A"]).at[chip_position].set(amplitude)
        return _exact_model_for_chip(response, chip, geometry, grid, current, delta, chip_position)

    return jax.jit(predict)


def direction_vector(name, state):
    """Use dimensionless test steps with explicitly reported physical scales."""

    atmosphere = np.zeros(len(ATMOSPHERE_NAMES))
    delta = np.zeros_like(state["delta_log_p"], dtype=float)
    amplitude = 0.0
    scales = {"T0": 10.0, "alpha": 0.005, "log_p_cloud": 0.05}
    if name in scales:
        atmosphere[ATMOSPHERE_NAMES.index(name)] = scales[name]
    elif name == "A":
        amplitude = 0.01
    elif name == "map_scale":
        delta = np.asarray(state["delta_log_p"], dtype=float)
        norm = np.sqrt(np.mean(delta**2))
        if norm == 0:
            raise ValueError("A nonzero map is required for the map_scale direction.")
        delta = 0.05 * delta / norm
    else:
        raise ValueError(f"Unknown gradient direction: {name}")
    return atmosphere, delta, np.asarray(amplitude)


def gradient_check(predict, point, direction, observed, mask, variance, steps):
    import jax
    import jax.numpy as jnp

    tangent = tuple(jnp.asarray(value, dtype=primal.dtype) for value, primal in zip(direction, point, strict=True))
    value, derivative = jax.jvp(predict, point, tangent)
    value, derivative = (np.asarray(jax.block_until_ready(item), dtype=float) for item in (value, derivative))
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Nonfinite forward directional derivative.")
    rows = []
    for step in steps:
        plus = tuple(primal + step * vector for primal, vector in zip(point, tangent, strict=True))
        minus = tuple(primal - step * vector for primal, vector in zip(point, tangent, strict=True))
        finite = (np.asarray(predict(*plus), dtype=float) - np.asarray(predict(*minus), dtype=float)) / (2 * step)
        error = (finite - derivative)[mask]
        derivative_rms = np.sqrt(np.mean(derivative[mask] ** 2))
        rows.append({
            "dimensionless_step": step,
            "derivative_difference_rms_over_noise": float(np.sqrt(np.mean(error**2) / variance)),
            "derivative_relative_rms_error": float(np.sqrt(np.mean(error**2)) / max(derivative_rms, np.finfo(float).tiny)),
        })
    return {
        "ad_log_likelihood_directional_derivative": float(np.sum((observed - value)[mask] * derivative[mask]) / variance),
        "ad_flux_derivative_rms_over_noise": float(np.sqrt(np.mean(derivative[mask] ** 2) / variance)),
        "finite_difference_checks": rows,
    }, derivative


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chip-indices", default="0")
    parser.add_argument("--nodes", default="21")
    parser.add_argument("--opacity-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--teacher-cache-dir", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--legacy-cache-dir", type=Path)
    parser.add_argument("--database-dir", type=Path, default=Path.home() / "data_mol/.database")
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--rt-pixel-batch-size", type=int, default=16)
    parser.add_argument("--noise-jitter", type=float, default=1e-6)
    parser.add_argument("--max-joint-kl", type=float, default=1.0)
    parser.add_argument("--max-absolute-delta-log-likelihood", type=float, default=3.0)
    parser.add_argument("--gradient-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--gradient-absolute-noise-tolerance", type=float, default=0.001)
    parser.add_argument("--midpoint-count", type=int, default=3)
    parser.add_argument("--gradient-directions", default="T0,alpha,log_p_cloud,map_scale")
    parser.add_argument("--fd-steps", default="0.03,0.01,0.003")
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--x64", action="store_true")
    args = parser.parse_args(argv)
    args.chip_indices = comma_list(args.chip_indices)
    args.nodes = comma_list(args.nodes)
    args.gradient_directions = comma_list(args.gradient_directions, str)
    args.fd_steps = comma_list(args.fd_steps, float)
    if not args.chip_indices or not args.nodes or min(args.nodes) < 2:
        parser.error("Specify at least one chip and temperature node counts >= 2.")
    if args.midpoint_count < 0 or args.rt_pixel_batch_size < 1 or args.noise_jitter < 0:
        parser.error("Invalid midpoint count, pixel batch size, or noise jitter.")
    if not args.fd_steps or any(step <= 0 for step in args.fd_steps):
        parser.error("Finite-difference steps must be positive.")
    if not args.prepare_cache_only and args.state_file is None:
        parser.error("--state-file is required unless --prepare-cache-only is specified.")
    protected = ROOT / "data/opacities/luhman16b_diffgrid"
    if args.opacity_cache_dir.resolve() == protected.resolve():
        parser.error("Use a separate M9/v3 cache; the archived M9 cache is protected.")
    return args


def _run(args):
    summary_path = args.out_dir / "diffgrid_validation.json"
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Use a new, empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import jax
    jax.config.update("jax_enable_x64", args.x64)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    from m9_v3_run import active_exojax_checkout
    from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel
    from doraex.workflows.on_the_fly_pressure_retrieval import _molecule_paths, _cia_paths, YAMA_L16B_EXOMOL_ATMOSPHERE
    from make_m9_v1_exact_products import _build_product_geometry, _load_chip_data_from_samples, _load_observation_masks, _load_profile_wavelengths

    exojax_root, exojax_commit = active_exojax_checkout()
    with np.load(args.samples, allow_pickle=False) as archive:
        samples = {name: archive[name] for name in archive.files}
    archived_chips = tuple(int(value) for value in samples["chip_indices"])
    if not set(args.chip_indices).issubset(archived_chips):
        raise ValueError("Requested chips must be present in the sample archive.")
    all_chips = _load_chip_data_from_samples(samples, list(archived_chips))
    all_masks, _ = _load_observation_masks(samples, all_chips)
    chips = [all_chips[archived_chips.index(chip)] for chip in args.chip_indices]
    masks = [all_masks[archived_chips.index(chip)] for chip in args.chip_indices]
    grids = _load_profile_wavelengths(samples, chips)
    state = None if args.prepare_cache_only else load_state(args.state_file, samples)
    source_hash = sha256(args.samples)
    if state is not None and "samples_sha256" in state and str(state["samples_sha256"].item()) != source_hash:
        raise ValueError("The physical state was built from a different sample archive.")
    geometry = _build_product_geometry(samples, int(samples["nside"]), args.x64)
    report = {
        "mode": "m9_v3_monotone_real_space_diffgrid_validation", "status": "running",
        "samples": str(args.samples.resolve()), "samples_sha256": source_hash,
        "state_file": None if state is None else str(args.state_file.resolve()),
        "state_sha256": None if state is None else sha256(args.state_file),
        "sample_index": None if state is None else int(state.get("sample_index", -1)),
        "exojax_root": str(exojax_root),
        "exojax_commit": exojax_commit,
        "script_sha256": sha256(__file__), "jax_version": jax.__version__,
        "jax_x64": bool(jax.config.jax_enable_x64), "x64": bool(jax.config.jax_enable_x64), "matmul_precision": "highest",
        "chip_indices": args.chip_indices, "cache_root": str(args.opacity_cache_dir.resolve()),
        "teacher_cache_root": str(args.teacher_cache_dir.resolve()),
        "teacher_profile_kernel": "real_space",
        "backend": jax.default_backend(), "nx": args.nx, "nodes": args.nodes,
        "noise_jitter": args.noise_jitter, "disk_normalization": "A_chip_times_global_mean_uniform_disk_spectrum",
        "local_normalization": "none", "interpolation": "fritsch_carlson_monotone_log_slopes_in_inverse_temperature",
        "spectral_interpolation": "existing_linear_interpolation",
        "gradient_direction_scales": {"T0": "10 K", "alpha": "0.005", "log_p_cloud": "0.05 dex", "map_scale": "0.05 dex RMS along the state map", "A": "0.01"},
        "acceptance": "Numerical checks at the listed states only; these do not establish posterior convergence or scientific adequacy.",
        "thresholds": {name: getattr(args, name) for name in ("max_joint_kl", "max_absolute_delta_log_likelihood", "gradient_relative_tolerance", "gradient_absolute_noise_tolerance")},
        "chips": {},
    }

    def save_report():
        summarize_checks(report, args)
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    for chip, mask, grid in zip(chips, masks, grids, strict=True):
        started = time.monotonic()
        position = archived_chips.index(chip.chip_index)
        print(f"Chip {chip.chip_index}: constructing the real-space teacher", flush=True)
        teacher = Luhman16BPowerLawColumnModel(
            chip.wavelengths, _molecule_paths(args.database_dir), _cia_paths(args.database_dir),
            args.teacher_cache_dir, parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx, sampling_wavelengths=grid, save_opacity_cache=True,
        )
        teacher.opacities = {name: opa.with_profile_kernel("real_space") for name, opa in teacher.opacities.items()}
        chip_report = {"teacher_cache_dir": str(teacher.opacity_cache_dir), "variants": {}}
        report["chips"][str(chip.chip_index)] = chip_report
        arrays = {"observation_mask": mask, "observed": chip.flux}
        if state is not None:
            dtype = jnp.float64 if args.x64 else jnp.float32
            point = (jnp.asarray([state[name] for name in ATMOSPHERE_NAMES], dtype=dtype), jnp.asarray(state["delta_log_p"], dtype=dtype), jnp.asarray(state["A"][position], dtype=dtype))
            variance = float(state["sigma_d"][position]) ** 2 + args.noise_jitter
            teacher_predict = make_prediction(teacher, chip, geometry, grid, state, position, args.rt_pixel_batch_size)
            reference = np.asarray(jax.block_until_ready(teacher_predict(*point)))
            arrays["teacher_prediction"] = reference
            chip_report["teacher_self_comparison"] = comparison_metrics(reference, reference, chip.flux, mask, variance)
            teacher_gradients = {}
            if not args.skip_gradients:
                chip_report["teacher_gradients"] = {}
                for name in args.gradient_directions:
                    print(f"Chip {chip.chip_index}: teacher derivative {name}", flush=True)
                    row, derivative = gradient_check(teacher_predict, point, direction_vector(name, state), chip.flux, mask, variance, args.fd_steps)
                    chip_report["teacher_gradients"][name] = row
                    teacher_gradients[name] = derivative
                    arrays[f"teacher_derivative_{name}"] = derivative
        variants = [(f"nodes{nodes}", args.opacity_cache_dir, nodes, False) for nodes in args.nodes]
        if args.legacy_cache_dir is not None:
            variants.append(("legacy_nodes21", args.legacy_cache_dir, 21, True))
        for label, root, nodes, legacy in variants:
            print(f"Chip {chip.chip_index}: {label}", flush=True)
            model, cache_records = build_diffgrid_model(teacher, root, nodes, args.nx, legacy=legacy)
            variant_report = {"cache": cache_records}
            chip_report["variants"][label] = variant_report
            save_report()
            if state is not None:
                variant_report["opacity"] = opacity_comparison(model, teacher, state, args.midpoint_count)
                predict = make_prediction(model, chip, geometry, grid, state, position, args.rt_pixel_batch_size)
                prediction = np.asarray(jax.block_until_ready(predict(*point)))
                arrays[f"{label}_prediction"] = prediction
                variant_report["disk"] = comparison_metrics(prediction, reference, chip.flux, mask, variance)
                variant_report["gradients"] = {}
                if not args.skip_gradients:
                    for name in args.gradient_directions:
                        row, derivative = gradient_check(predict, point, direction_vector(name, state), chip.flux, mask, variance, args.fd_steps)
                        difference = (derivative - teacher_gradients[name])[mask]
                        row["teacher_derivative_difference_rms_over_noise"] = float(np.sqrt(np.mean(difference**2) / variance))
                        teacher_rms = np.sqrt(np.mean(teacher_gradients[name][mask] ** 2))
                        row["teacher_derivative_relative_rms_error"] = float(np.sqrt(np.mean(difference**2)) / max(teacher_rms, np.finfo(float).tiny))
                        variant_report["gradients"][name] = row
                        arrays[f"{label}_derivative_{name}"] = derivative
                del predict
            del model
            save_report()
        if state is not None:
            np.savez_compressed(args.out_dir / f"chip{chip.chip_index}_predictions.npz", **arrays)
            del teacher_predict
        chip_report["elapsed_seconds"] = time.monotonic() - started
        del teacher
        jax.clear_caches()
        save_report()
    report["status"] = "complete"
    save_report()
    print(f"Validation saved to {summary_path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Use a new, empty output directory: {args.out_dir}")
    failure = None
    try:
        _run(args)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if failure is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = args.out_dir / "diffgrid_validation.json"
            report = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            report.update(status="failed", failure_reason=failure, passed=False)
            report["passed_by_nodes"] = {str(nodes): False for nodes in args.nodes}
            summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
