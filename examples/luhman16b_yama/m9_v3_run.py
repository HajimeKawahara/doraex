"""Evaluate, initialize, or sample the M9 v3 exact raw-intensity target."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CACHE = ROOT / "data/opacities/luhman16b_diffgrid_m9_v3_real_space"
DEFAULT_SAMPLES = ROOT / "results/m8/v12/masked_free_sigma_d_highest_precision_long_seed0/samples.npz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("evaluate", "initialize", "sample"), required=True)
    parser.add_argument("--state-file", type=Path, default=ROOT / "results/m9/v3/seeds/m8_q50_mean.npz")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                        help="Immutable observation, padded-grid, and geometry archive used by forward validation.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chip-indices", default="0,1,2,3")
    parser.add_argument("--opacity-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--database-dir", type=Path, default=Path.home() / "data_mol/.database")
    parser.add_argument("--nodes", type=int, default=81)
    parser.add_argument("--rt-pixel-batch-size", type=int, default=16)
    parser.add_argument("--marginalize-a", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-step-norm", type=float, default=5.)
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-tree-depth", type=int, default=7)
    parser.add_argument("--warmup-max-tree-depth", type=int, default=None,
                        help="Warmup tree depth; defaults to --max-tree-depth, which sets the sampling limit.")
    parser.add_argument("--target-accept-prob", type=float, default=.9)
    parser.add_argument("--adapt-mass-matrix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-scale", type=float, default=1.)
    parser.add_argument("--initial-map-source", choices=("archived", "raw-linear"), default="archived",
                        help="Use the archived map, or compare it with scaled corrected raw-linear conditional maps.")
    observations = parser.add_mutually_exclusive_group()
    observations.add_argument("--synthetic-seed", type=int, default=None,
                              help="Generate exact observations from the input state before initialization.")
    observations.add_argument("--observations-from", type=Path, default=None,
                              help="Reuse archived synthetic observations after checking the fixed geometry and mask.")
    parser.add_argument("--forward-validation", type=Path, default=None,
                        help="An aggregate passing M9 v3 forward report, required for initialization and sampling.")
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/doraex_m7_gpu.lock"))
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args()


@contextmanager
def gpu_lock(path):
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another retrieval holds {path}; no job started.") from exc
        yield


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def active_exojax_checkout():
    """Identify the imported ExoJAX checkout for reproducible forward validation."""
    import exojax.opacity.diffgrid.api as diffgrid_api

    root = Path(diffgrid_api.__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("M9 v3 requires an ExoJAX Git checkout on PYTHONPATH.") from exc
    return root, commit


def build_mcmc(args, potential, covariance):
    """Build the sampler with independently recorded warmup and sampling limits."""
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from doraex.workflows.on_the_fly_pressure_retrieval import build_nuts_max_tree_depth

    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    kernel = NUTS(potential_fn=potential, dense_mass=True,
                  inverse_mass_matrix=jnp.asarray(covariance),
                  adapt_mass_matrix=args.adapt_mass_matrix,
                  target_accept_prob=args.target_accept_prob, max_tree_depth=specification)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, progress_bar=True)
    return mcmc, {
        "num_warmup": args.num_warmup, "num_samples": args.num_samples,
        "warmup_max_tree_depth": warmup_depth, "max_tree_depth": sampling_depth,
        "warmup_tree_depth_cap_num_steps": 2**warmup_depth - 1,
        "tree_depth_cap_num_steps": 2**sampling_depth - 1,
        "target_accept_prob": args.target_accept_prob, "adapt_mass_matrix": args.adapt_mass_matrix,
    }


def select_raw_linear_initial_position(problem, archived_position, normalizations):
    """Select among archived and scaled raw-linear maps using the exact target."""
    linear_position, diagnostics = problem.raw_linear_initial_position(
        archived_position, normalizations=normalizations,
    )
    candidates = [("archived", None, archived_position)]
    candidates += [
        ("raw-linear", scale, linear_position.at[problem.map_slice].multiply(scale))
        for scale in (0., .25, .5, .75, 1.)
    ]

    def evaluate_candidate(source, scale, position):
        objective = float(problem.potential(position))
        finite = bool(np.isfinite(objective))
        return {
            "source": source, "scale": scale, "finite": finite,
            "exact_objective": objective if finite else None,
        }

    trials = [evaluate_candidate(*candidate) for candidate in candidates]
    backtracking = {"additional_trials": 0, "max_additional_trials": 8,
                    "stop_reason": "no_finite_raw_candidate"}
    raw_indices = [index for index in range(1, len(trials)) if trials[index]["finite"]]
    if raw_indices:
        raw_best = min(raw_indices, key=lambda index: trials[index]["exact_objective"])
        backtracking["stop_reason"] = "coarse_minimum_above_quarter"
        if trials[raw_best]["scale"] in (0., .25):
            zero_objective = trials[1]["exact_objective"] if trials[1]["finite"] else np.inf
            previous = trials[2]["exact_objective"] if trials[2]["finite"] else np.inf
            improved = previous < zero_objective
            scale = .25
            for _ in range(backtracking["max_additional_trials"]):
                scale *= .5
                candidate = ("raw-linear", scale, linear_position.at[problem.map_slice].multiply(scale))
                candidates.append(candidate)
                trial = evaluate_candidate(*candidate)
                trials.append(trial)
                backtracking["additional_trials"] += 1
                objective = trial["exact_objective"] if trial["finite"] else np.inf
                if improved and objective >= previous:
                    backtracking["stop_reason"] = "improvement_stopped"
                    break
                improved = improved or objective < zero_objective
                previous = objective
            else:
                backtracking["stop_reason"] = "maximum_additional_trials"
    finite_indices = [index for index, trial in enumerate(trials) if trial["finite"]]
    if not finite_indices:
        raise FloatingPointError("Every archived/raw-linear initial candidate has a non-finite exact objective.")
    selected = min(finite_indices, key=lambda index: trials[index]["exact_objective"])
    return candidates[selected][2], {
        "conditional_solve": diagnostics, "trials": trials, "backtracking": backtracking,
        "selected_index": selected,
        "selected_source": trials[selected]["source"], "selected_scale": trials[selected]["scale"],
        "selected_exact_objective": trials[selected]["exact_objective"],
        "marginalized_A": bool(problem.marginalized),
        "interpretation": "Initialization selection only; all shared parameters stay fixed and the target remains exact.",
    }


def validate_forward_gate(args, current_commit):
    """Require unchanged numerical-validation evidence before running inference."""
    if args.forward_validation is None:
        raise ValueError("Initialization and sampling require --forward-validation.")
    gate = json.loads(args.forward_validation.read_text())
    if not gate.get("passed", False):
        raise ValueError("The forward validation did not pass.")
    if Path(gate["cache_root"]).resolve() != args.opacity_cache_dir.resolve() or int(gate["selected_nodes"]) != args.nodes:
        raise ValueError("Forward validation does not match this cache and node count.")
    if sorted(map(int, args.chip_indices.split(","))) != sorted(gate["chip_indices"]):
        raise ValueError("Forward validation does not cover the requested chips.")
    if gate["exojax_commit"] != current_commit or not gate["x64"]:
        raise ValueError("Forward validation does not match the active ExoJAX or precision.")
    if gate.get("nx") != 4500 or gate.get("noise_jitter") != 1e-6:
        raise ValueError("Forward validation does not match the spectral grid or noise jitter.")
    expected_samples = gate.get("source_samples_sha256", {})
    if expected_samples.get(str(args.samples.resolve())) != sha256(args.samples):
        raise ValueError("The observation archive does not match forward validation.")
    for field in ("cache_metadata_sha256", "report_sha256", "inputstates"):
        if not gate.get(field):
            raise ValueError(f"Forward validation is missing {field}.")
        for filename, expected in gate[field].items():
            if sha256(filename) != expected:
                raise ValueError(f"Forward evidence changed: {filename}")

    requested_chips = [int(chip) for chip in args.chip_indices.split(",")]

    def fixed_operator(path):
        with np.load(path, allow_pickle=False) as archive:
            scalar_names = ("P", "cosi", "v", "u1", "u2", "logg")
            operator = {name: np.asarray(archive[name]) for name in scalar_names}
            if any(value.shape != () or not np.isfinite(value) for value in operator.values()):
                raise ValueError(f"The fixed operator must contain finite scalars: {path}")
            chips = np.asarray(archive["chip_indices"])
            if chips.ndim != 1 or np.unique(chips).size != chips.size:
                raise ValueError(f"The fixed operator requires unique chip indices: {path}")
            if not set(requested_chips).issubset(chips.tolist()):
                raise ValueError(f"The fixed operator does not cover the requested chips: {path}")
            log_w = np.asarray(archive["log_w"], dtype=float)
            if log_w.ndim != 2 or log_w.shape[0] != chips.size or not np.all(np.isfinite(log_w)):
                raise ValueError(f"The fixed operator requires finite per-chip phase weights: {path}")
            positions = [chips.tolist().index(chip) for chip in requested_chips]
            # A phase-independent chip factor cancels in the disk normalization.
            weights = log_w[positions]
            operator["log_w"] = weights - weights.mean(axis=1, keepdims=True)
            return operator

    current_operator = fixed_operator(args.state_file)
    for path in gate["inputstates"]:
        validated_operator = fixed_operator(path)
        for name, current_value in current_operator.items():
            if not np.array_equal(current_value, validated_operator[name]):
                raise ValueError(f"The fixed operator differs from forward validation: {name} in {path}")
    return gate


def _observation_contract(problem):
    """Record the fixed observation operator independently of an initial state."""
    contract = {
        "format_version": np.asarray(1),
        "target": np.asarray("m9_v3_exact_raw_intensity"),
        "chip_indices": np.asarray([chip.chip_index for chip in problem.chip_data]),
        "theta": np.asarray(problem.geometry.theta),
        "phi": np.asarray(problem.geometry.phi),
        "noise_jitter": np.asarray(problem.noise_jitter),
    }
    contract.update({f"fixed_{name}": np.asarray(problem.fixed_state[name])
                     for name in ("P", "cosi", "v", "u1", "u2", "logg")})
    for position, chip in enumerate(problem.chip_data):
        suffix = f"chip{chip.chip_index}"
        shape = np.asarray(np.shape(chip.flux), dtype="<i8")
        indices = np.asarray(problem.indices[position], dtype="<i8")
        mask = np.zeros(int(np.prod(shape)), dtype=bool)
        mask[indices] = True
        mask_hash = hashlib.sha256(shape.tobytes() + np.packbits(mask).tobytes()).hexdigest()
        profile_grid = problem.profile_wavelengths[position]
        contract.update({
            f"shape_{suffix}": shape, f"valid_indices_{suffix}": indices,
            f"mask_sha256_{suffix}": np.asarray(mask_hash),
            f"wavelengths_{suffix}": np.asarray(chip.wavelengths),
            f"profile_wavelengths_{suffix}": np.asarray(profile_grid) if profile_grid is not None else np.empty(0),
            f"obs_times_{suffix}": np.asarray(chip.obs_times),
            f"weights_{suffix}": np.asarray(problem.weights[position]),
        })
    return contract


def _replace_observations(problem, observations):
    """Retain the mask and replace density closures, including already traced ones."""
    import jax
    import jax.numpy as jnp
    problem.observed = [jnp.asarray(value) for value in observations]
    # Fresh function identities prevent a previous JIT trace from retaining data.
    problem.potential = jax.jit(lambda position: problem.negative_log_density(position))
    problem.value_and_grad = jax.jit(jax.value_and_grad(
        lambda position: problem.negative_log_density(position)))


def generate_synthetic_observations(problem, position, state, seed, path, state_file):
    """Generate one reproducible dataset and archive the actual truth and mask."""
    truth = problem.physical(position)
    rng = np.random.default_rng(seed)
    archive = _observation_contract(problem)
    archive.update({f"truth_{name}": np.asarray(value) for name, value in truth.items()})
    archive.update(
        synthetic_seed=np.asarray(seed), truth_state_file=np.asarray(str(Path(state_file).resolve())),
        truth_state_sha256=np.asarray(sha256(state_file)),
        truth_A=np.asarray(state["A"])[problem.source_positions],
    )
    observations = []
    for chip in range(problem.nchip):
        a = float(archive["truth_A"][chip])
        mean = np.asarray(problem.raw_functions[chip](position)) / a
        noise = float(np.sqrt(truth["sigma_d"][chip]**2 + problem.noise_jitter))
        if not np.all(np.isfinite(mean)) or not np.isfinite(noise) or noise <= 0:
            raise ValueError("Synthetic means and effective noise must be finite with positive noise.")
        observed = mean + rng.normal(size=mean.shape) * noise
        observations.append(observed)
        archive[f"chip{problem.chip_data[chip].chip_index}"] = observed
    np.savez(path, **archive)
    _replace_observations(problem, observations)
    return {"source": str(Path(path).resolve()), "sha256": sha256(path),
            "synthetic_seed": int(seed), "truth_state_sha256": str(archive["truth_state_sha256"])}


def load_observations(problem, path):
    """Reuse a dataset for requested chips without changing its generating truth."""
    contract = _observation_contract(problem)
    observations = []
    with np.load(path, allow_pickle=False) as archive:
        archived_chips = np.asarray(archive["chip_indices"])
        if archived_chips.ndim != 1 or np.unique(archived_chips).size != archived_chips.size:
            raise ValueError("The observation archive must contain unique chip indices.")
        if not set(contract["chip_indices"].tolist()).issubset(archived_chips.tolist()):
            raise ValueError("The observation archive does not cover the requested chips.")
        for key, expected in contract.items():
            if key != "chip_indices" and (key not in archive or not np.array_equal(archive[key], expected)):
                raise ValueError(f"The observation archive does not match the target: {key}.")
        for chip_position, chip in enumerate(problem.chip_data):
            key = f"chip{chip.chip_index}"
            observed = np.asarray(archive[key])
            if observed.shape != np.shape(problem.observed[chip_position]) or not np.all(np.isfinite(observed)):
                raise ValueError(f"The archived observations must match the finite masked rows: {key}.")
            observations.append(observed)
        metadata = {"source": str(Path(path).resolve()), "sha256": sha256(path),
                    "synthetic_seed": int(archive["synthetic_seed"]),
                    "truth_state_sha256": str(archive["truth_state_sha256"])}
    _replace_observations(problem, observations)
    return metadata


def build_problem(args, state):
    from types import SimpleNamespace
    from doraex.workflows import on_the_fly_pressure_retrieval as retrieval
    from doraex.workflows.exact_pressure_investigation import ExactPressureProblem
    from make_m9_v1_exact_products import (
        _build_product_geometry, _load_chip_data_from_samples, _load_profile_wavelengths,
    )
    chips = [int(value) for value in args.chip_indices.split(",")]
    with np.load(args.samples, allow_pickle=False) as archive:
        archived = {name: archive[name] for name in archive.files}
    if int(archived["nside"]) != 8:
        raise ValueError("M9 v3 requires the archived nside=8 geometry.")
    chip_data = _load_chip_data_from_samples(archived, chips)
    profile_wavelengths = _load_profile_wavelengths(archived, chip_data)
    geometry = _build_product_geometry(archived, 8, True)
    gp = retrieval.build_fixed_pressure_gp_eigendecomposition(
        geometry.distance_matrix, .4, theta=geometry.theta, phi=geometry.phi)
    response_args = SimpleNamespace(
        database_dir=args.database_dir, opacity_cache_dir=args.opacity_cache_dir,
        nx=4500, opacity_mode="diffgrid", diffgrid_temperature_nodes=args.nodes,
        exact_pixel_rt=True, exact_rt_local_normalization="none",
        rt_pixel_batch_size=args.rt_pixel_batch_size,
    )
    # Never accept an old analytic-teacher table under the new namespace.
    from doraex.spectra.exojax_forward import _internal_spectrum_wavelength_bounds, _opacity_cache_namespace
    for chip, profile_grid in zip(chip_data, profile_wavelengths):
        bounds = _internal_spectrum_wavelength_bounds(
            chip.wavelengths, profile_grid, systemic_rv=retrieval.YAMA_L16B_EXOMOL_ATMOSPHERE.rv)
        directory = _opacity_cache_namespace(args.opacity_cache_dir, bounds,
                                             4500, 1e-4, 1e2, 101, 210., 3500.,
                                             "diffgrid", args.nodes)
        archives = list(directory.glob("opa*.zarr"))
        if len(archives) != 4:
            raise ValueError(f"Expected four molecular archives in {directory}.")
        for archive in archives:
            if (archive / "zarr.json").exists():
                metadata = json.loads((archive / "zarr.json").read_text())["attributes"]
            else:
                metadata = json.loads((archive / ".zattrs").read_text())
            if metadata.get("opa_state", {}).get("teacher_profile_kernel") != "real_space":
                raise ValueError(f"The cache does not record the real_space teacher: {archive}")
    responses = retrieval.build_response_functions(response_args, chip_data,
                                                   profile_wavelengths=profile_wavelengths)
    return ExactPressureProblem(
        chip_data, responses, profile_wavelengths, geometry,
        gp["eigenvalues"], gp["pixel_eigenvectors"], state,
        marginalized=args.marginalize_a, quadrature_order=args.quadrature_order,
    )


def metric_from_curvature(curvature):
    """Invert a symmetric local precision with a recorded numerical floor."""
    eigenvalues, vectors = np.linalg.eigh(.5 * (curvature + curvature.T))
    floor = max(1e-8, float(eigenvalues[-1]) * 1e-10)
    retained = np.maximum(eigenvalues, floor)
    covariance = (vectors / retained[None, :]) @ vectors.T
    return .5 * (covariance + covariance.T), {
        "minimum_precision_eigenvalue": float(eigenvalues[0]),
        "maximum_precision_eigenvalue": float(eigenvalues[-1]),
        "precision_floor": floor, "floor_clipped_count": int(np.sum(eigenvalues < floor)),
        "condition_number": float(retained[-1] / retained[0]),
    }


def save_state(path, problem, position, original):
    state = {name: np.asarray(value) for name, value in problem.physical(position).items()}
    fixed = ("logg", "P", "period", "cosi", "v", "q1", "q2", "u1", "u2", "chip_indices", "log_w",
             "source_samples", "samples_sha256")
    state.update({name: original[name] for name in fixed if name in original})
    # A conditional mean is only an initialization value, never a saved A draw.
    if problem.marginalized:
        moments = problem.normalization_moments(position)
        state["A"] = np.asarray([float(pair[0]) for pair in moments])
        state["A_initialization_kind"] = np.asarray("conditional_mean")
    # The output state carries only the chips used by this problem.
    state["chip_indices"] = np.asarray([chip.chip_index for chip in problem.chip_data])
    state["log_w"] = np.asarray(original["log_w"])[problem.source_positions]
    np.savez(path, **state)


def run(args):
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    _, commit = active_exojax_checkout()
    if args.stage != "evaluate":
        validate_forward_gate(args, commit)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "RUNNING").touch()
    started = time.time()
    try:
        source_paths = [Path(__file__), ROOT / "src/doraex/workflows/exact_pressure_investigation.py",
                        ROOT / "src/doraex/workflows/on_the_fly_pressure_retrieval.py",
                        ROOT / "src/doraex/spectra/exojax_forward.py",
                        ROOT / "src/doraex/operators/design_matrix.py", ROOT / "src/doraex/operators/doppler.py",
                        ROOT / "src/doraex/inference/normalization_marginalization.py",
                        ROOT / "src/doraex/inference/damped_gauss_newton.py"]
        provenance = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                      "source_sha256": {str(path): sha256(path) for path in source_paths},
                      "state_sha256": sha256(args.state_file), "exojax_commit": commit,
                      "samples_sha256": sha256(args.samples),
                      "devices": [str(device) for device in jax.devices()], "x64": True,
                      "jax_version": jax.__version__,
                      "environment": {name: os.environ.get(name) for name in (
                          "JAX_PLATFORMS", "XLA_FLAGS", "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS",
                          "JAX_COMPILATION_CACHE_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE")},
                      "interpretation": "M9 v3 investigation; computational completion is not scientific acceptance."}
        if args.forward_validation is not None:
            provenance["forward_validation_sha256"] = sha256(args.forward_validation)
        if getattr(args, "observations_from", None) is not None:
            provenance["observations_sha256"] = sha256(args.observations_from)
        snapshot_dir = args.out_dir / "source"
        snapshot_dir.mkdir()
        for path in source_paths:
            (snapshot_dir / path.name).write_bytes(path.read_bytes())
        write_json(args.out_dir / "provenance.json", provenance)
        state = dict(np.load(args.state_file, allow_pickle=False))
        problem = build_problem(args, state)
        position = problem.encode(state)
        if args.synthetic_seed is not None:
            metadata = generate_synthetic_observations(
                problem, position, state, args.synthetic_seed,
                args.out_dir / "synthetic_observations.npz", args.state_file,
            )
            write_json(args.out_dir / "observations.json", metadata)
        elif args.observations_from is not None:
            write_json(args.out_dir / "observations.json", load_observations(problem, args.observations_from))
        contract_path = args.out_dir / "observation_contract.npz"
        contract = _observation_contract(problem)
        contract.update({f"observed_chip{chip.chip_index}": np.asarray(problem.observed[index])
                         for index, chip in enumerate(problem.chip_data)})
        np.savez(contract_path, **contract)
        provenance["observation_contract_sha256"] = sha256(contract_path)
        write_json(args.out_dir / "provenance.json", provenance)
        position = position.at[problem.map_slice].multiply(args.map_scale)
        if args.initial_map_source == "raw-linear":
            position, scan = select_raw_linear_initial_position(
                problem, position, np.asarray(state["A"])[problem.source_positions],
            )
            write_json(args.out_dir / "raw_linear_initialization.json", scan)
            save_state(args.out_dir / "selected_seed.npz", problem, position, state)
        print(f"Prepared {problem.dimension} coordinates and {sum(y.size for y in problem.observed)} observations.", flush=True)
        t0 = time.time()
        initial_value, gradient = problem.value_and_grad(position)
        jax.block_until_ready(gradient)
        if not np.isfinite(initial_value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("The initial exact potential or gradient is non-finite.")
        first_gradient_seconds = time.time() - t0
        benchmark_start = time.time()
        jax.block_until_ready(problem.value_and_grad(position))
        evaluation = {"initial_potential": float(initial_value), "initial_gradient_norm": float(jnp.linalg.norm(gradient)),
                      "initial_prior_potential": float(problem.prior_potential(position)),
                      "compile_and_gradient_seconds": first_gradient_seconds,
                      "gradient_seconds": time.time() - benchmark_start,
                      "dimension": problem.dimension, "marginalized_A": problem.marginalized}
        write_json(args.out_dir / "evaluation.json", evaluation)
        np.savez(args.out_dir / "evaluation_arrays.npz", position=np.asarray(position),
                 gradient=np.asarray(gradient), prior_gradient=np.asarray(jax.grad(problem.prior_potential)(position)))
        print(json.dumps(evaluation), flush=True)
        if args.stage == "initialize":
            from doraex.inference.damped_gauss_newton import damped_gauss_newton
            history = damped_gauss_newton(
                np.asarray(position), objective_function=lambda x: float(problem.potential(jnp.asarray(x))),
                normal_equations_function=problem.normal_equations,
                max_iterations=args.iterations, max_step_norm=args.max_step_norm,
            )
            write_json(args.out_dir / "initialization.json", history)
            position = jnp.asarray(history["position"])
            save_state(args.out_dir / "initial_state.npz", problem, position, state)
        elif args.stage == "sample":
            _, curvature = problem.normal_equations(position)
            covariance, metric_diagnostics = metric_from_curvature(curvature)
            write_json(args.out_dir / "metric.json", metric_diagnostics)
            np.savez(args.out_dir / "metric.npz", covariance=covariance,
                     eigenvalues=np.asarray(problem.eigenvalues), eigenvectors=np.asarray(problem.eigenvectors))
            mcmc, sampler_settings = build_mcmc(args, problem.potential, covariance)
            write_json(args.out_dir / "sampler_settings.json", sampler_settings)
            print(json.dumps({"sampler_settings": sampler_settings}), flush=True)
            sample_start = time.time()
            mcmc.run(jax.random.PRNGKey(args.seed), init_params=position,
                     extra_fields=("potential_energy", "energy", "num_steps", "accept_prob", "diverging"))
            positions = mcmc.get_samples()
            extras = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}
            sample_metadata = {
                **{name: np.asarray(sampler_settings[name]) for name in (
                    "warmup_max_tree_depth", "max_tree_depth",
                    "warmup_tree_depth_cap_num_steps", "tree_depth_cap_num_steps")},
                "chip_indices": np.asarray([chip.chip_index for chip in problem.chip_data]),
                "marginalized_A": np.asarray(problem.marginalized), "seed": np.asarray(args.seed),
            }
            # Preserve completed HMC draws before expensive physical/A recovery.
            raw_path = args.out_dir / "mcmc_raw.npz"
            temporary_path = raw_path.with_suffix(".npz.tmp")
            with temporary_path.open("wb") as stream:
                np.savez(stream, position=np.asarray(positions),
                         **{f"extra_{name}": value for name, value in extras.items()}, **sample_metadata)
            temporary_path.replace(raw_path)
            print(f"Saved HMC draws before physical/A recovery: {raw_path}", flush=True)
            physical = {name: np.asarray(value) for name, value in jax.vmap(problem.physical)(positions).items()}
            if problem.marginalized:
                from doraex.inference.normalization_marginalization import draw_conditional_normalization
                moments = [problem.normalization_moments(point) for point in positions]
                physical["A_conditional_mean"] = np.asarray([[float(pair[0]) for pair in row] for row in moments])
                physical["A_conditional_variance"] = np.asarray([[float(pair[1]) for pair in row] for row in moments])
                recovery_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 903]))
                physical["A"] = np.asarray([
                    [draw_conditional_normalization(
                        np.asarray(problem.observed[chip]), np.asarray(problem.raw_functions[chip](point)),
                        physical["sigma_d"][draw, chip]**2 + problem.noise_jitter, recovery_rng,
                    ) for chip in range(problem.nchip)] for draw, point in enumerate(positions)
                ])
            np.savez(args.out_dir / "samples.npz", position=np.asarray(positions), **physical,
                     **{f"extra_{name}": value for name, value in extras.items()},
                     **sample_metadata)
            sample_diagnostics = {**sampler_settings, "seconds": time.time() - sample_start,
                                  "divergences": int(extras["diverging"].sum()),
                                  "acceptance": float(extras["accept_prob"].mean()),
                                  "median_steps": float(np.median(extras["num_steps"])),
                                  "depth_cap_fraction": float(np.mean(extras["num_steps"] == sampler_settings["tree_depth_cap_num_steps"])),
                                  "step_size": float(mcmc.last_state.adapt_state.step_size),
                                  "interpretation": "Single-chain samples; independent-chain diagnostics required."}
            write_json(args.out_dir / "sampling.json", sample_diagnostics)
        write_json(args.out_dir / "completion.json", {"seconds": time.time() - started, "stage": args.stage})
        (args.out_dir / "COMPLETE").touch()
    except (Exception, KeyboardInterrupt) as exc:
        write_json(args.out_dir / "failure.json", {"exception": type(exc).__name__, "message": str(exc), "seconds": time.time() - started})
        (args.out_dir / "FAILED").touch()
        raise
    finally:
        (args.out_dir / "RUNNING").unlink(missing_ok=True)


def main():
    args = parse_args()
    with gpu_lock(args.gpu_lock):
        run(args)


if __name__ == "__main__":
    main()
