"""Lightweight checks of reusable synthetic data in the M9 v3 runner."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "m9_v3_runner_test_module", ROOT / "examples/luhman16b_yama/m9_v3_run.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make_problem(chips=(3, 1)):
    """Build only the fixed operator and an inexpensive differentiable density."""
    problem = SimpleNamespace()
    problem.chip_data = []
    problem.indices = []
    problem.observed = []
    problem.weights = []
    problem.profile_wavelengths = []
    problem.raw_functions = []
    problem.nchip = len(chips)
    problem.source_positions = np.asarray(chips)
    problem.geometry = SimpleNamespace(theta=np.asarray([.4, 1.1, 1.8]), phi=np.asarray([0., 2., 4.]))
    problem.fixed_state = {"P": 4.83, "cosi": .485, "v": 31.2, "u1": 1.062, "u2": -.162, "logg": 4.86}
    problem.noise_jitter = 1e-6
    for chip in chips:
        flux = np.ones((2, 3))
        flux[0, chip % 3] = 0.
        problem.chip_data.append(SimpleNamespace(
            chip_index=chip, flux=flux, wavelengths=np.asarray([1000., 1001., 1002.]),
            obs_times=np.asarray([.1, .9]),
        ))
        problem.indices.append(np.flatnonzero(flux.reshape(-1)))
        problem.observed.append(jnp.ones(5))
        problem.weights.append(jnp.exp(jnp.asarray([.01 * chip, -.01 * chip])))
        problem.profile_wavelengths.append(np.linspace(999., 1003., 7))
        problem.raw_functions.append(lambda position, c=chip: jnp.ones(5) * (1. + position[0] + .01 * c))
    problem.physical = lambda position: {
        "sigma_d": jnp.full((len(chips),), .03),
        "delta_log_p": jnp.asarray([position[0], -position[0], 0.]),
        "log_p_cloud": jnp.asarray(.4),
    }
    problem.negative_log_density = lambda position: (
        .5 * position @ position
        + .5 * sum(jnp.sum((values - (1. + position[0]))**2) for values in problem.observed)
    )
    problem.potential = jax.jit(problem.negative_log_density)
    problem.value_and_grad = jax.jit(jax.value_and_grad(problem.negative_log_density))
    return problem


@pytest.fixture
def synthetic_archive(tmp_path):
    with jax.enable_x64():
        state = {"A": np.asarray([1.03, 1.07, 1.09, 1.12])}
        state_file = tmp_path / "truth.npz"
        np.savez(state_file, **state)
        path = tmp_path / "observations.npz"
        problem = make_problem()
        position = jnp.asarray([.2])
        metadata = runner.generate_synthetic_observations(problem, position, state, 17, path, state_file)
        yield problem, position, state, state_file, path, metadata


def test_generation_preserves_masks_truth_and_random_seed(synthetic_archive, tmp_path):
    problem, position, state, state_file, path, metadata = synthetic_archive
    clone = make_problem()
    copy_path = tmp_path / "copy.npz"
    runner.generate_synthetic_observations(clone, position, state, 17, copy_path, state_file)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["chip_indices"], [3, 1])
        np.testing.assert_array_equal(archive["truth_A"], state["A"][[3, 1]])
        np.testing.assert_allclose(archive["truth_delta_log_p"], [.2, -.2, 0.])
        assert metadata["truth_state_sha256"] == runner.sha256(state_file)
        for index, chip in enumerate(problem.chip_data):
            np.testing.assert_array_equal(problem.observed[index], clone.observed[index])
            np.testing.assert_array_equal(problem.observed[index], archive[f"chip{chip.chip_index}"])
            np.testing.assert_array_equal(problem.indices[index], np.flatnonzero(chip.flux.reshape(-1)))
            assert np.count_nonzero(chip.flux == 0.) == 1
            assert len(str(archive[f"mask_sha256_chip{chip.chip_index}"])) == 64
    other = make_problem()
    runner.generate_synthetic_observations(other, position, state, 18, tmp_path / "other.npz", state_file)
    assert not np.array_equal(problem.observed[0], other.observed[0])


def test_reuse_keeps_original_data_for_subset_and_new_initial_position(synthetic_archive):
    source, _, _, _, path, original_metadata = synthetic_archive
    restored = make_problem(chips=(1,))
    position = jnp.asarray([-.4])
    # A previous trace must not hold on to the original observational data.
    old_value = float(restored.potential(position))
    old_gradient = np.asarray(restored.value_and_grad(position)[1])
    metadata = runner.load_observations(restored, path)
    np.testing.assert_array_equal(restored.observed[0], source.observed[1])
    assert metadata["truth_state_sha256"] == original_metadata["truth_state_sha256"]
    expected = .5 * float(position @ position) + .5 * np.sum((np.asarray(source.observed[1]) - (1. + float(position[0])))**2)
    np.testing.assert_allclose(restored.potential(position), expected, rtol=1e-12)
    value, gradient = restored.value_and_grad(position)
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    assert not np.isclose(float(value), old_value)
    assert not np.allclose(gradient, old_gradient)
    reordered = make_problem(chips=(1, 3))
    runner.load_observations(reordered, path)
    np.testing.assert_array_equal(reordered.observed[0], source.observed[1])
    np.testing.assert_array_equal(reordered.observed[1], source.observed[0])


@pytest.mark.parametrize("change", ["mask", "geometry", "wavelengths", "nonfinite"])
def test_reuse_rejects_mismatched_operator_before_replacing_data(synthetic_archive, tmp_path, change):
    _, _, _, _, path, _ = synthetic_archive
    restored = make_problem()
    with np.load(path, allow_pickle=False) as original:
        contents = {key: original[key] for key in original.files}
    key = {"mask": "valid_indices_chip3", "geometry": "theta",
           "wavelengths": "wavelengths_chip3", "nonfinite": "chip3"}[change]
    contents[key] = contents[key].copy()
    contents[key].flat[0] = np.nan if change == "nonfinite" else contents[key].flat[0] + 1
    altered = tmp_path / f"{change}.npz"
    np.savez(altered, **contents)
    previous = [np.asarray(value).copy() for value in restored.observed]
    with pytest.raises(ValueError, match="match"):
        runner.load_observations(restored, altered)
    for old, current in zip(previous, restored.observed, strict=True):
        np.testing.assert_array_equal(old, current)


def test_synthetic_generation_and_reuse_are_mutually_exclusive(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["m9_v3_run.py", "--stage", "evaluate", "--out-dir", str(tmp_path),
                                     "--synthetic-seed", "2", "--observations-from", "data.npz"])
    with pytest.raises(SystemExit):
        runner.parse_args()


@pytest.fixture
def forward_gate(tmp_path):
    cache_metadata = tmp_path / "cache_metadata.json"
    report = tmp_path / "report.json"
    cache_metadata.write_text('{"teacher_profile_kernel": "real_space"}')
    report.write_text('{"passed": true}')
    samples = tmp_path / "source_samples.npz"
    np.savez(samples, immutable_input=np.asarray([1., 2.]))
    state_file = tmp_path / "validated_state.npz"
    np.savez(state_file, P=np.asarray(4.83), cosi=np.asarray(.485),
             v=np.asarray(31.2), u1=np.asarray(1.062), u2=np.asarray(-.162),
             logg=np.asarray(4.86), chip_indices=np.arange(4), log_w=np.zeros((4, 14)))
    path = tmp_path / "gate.json"
    args = SimpleNamespace(
        forward_validation=path, opacity_cache_dir=tmp_path / "cache",
        nodes=21, chip_indices="0,1,2,3", samples=samples, state_file=state_file,
    )
    gate = {
        "passed": True, "cache_root": str(args.opacity_cache_dir),
        "selected_nodes": 21, "chip_indices": [0, 1, 2, 3],
        "exojax_commit": "verified-commit", "x64": True,
        "nx": 4500, "noise_jitter": 1e-6,
        "inputstates": {str(state_file): runner.sha256(state_file)},
        "source_samples_sha256": {str(samples.resolve()): runner.sha256(samples)},
        "cache_metadata_sha256": {str(cache_metadata): runner.sha256(cache_metadata)},
        "report_sha256": {str(report): runner.sha256(report)},
    }
    runner.write_json(path, gate)
    return args, gate


def test_forward_gate_accepts_unchanged_evidence_and_chip_reordering(forward_gate):
    args, gate = forward_gate
    args.chip_indices = "3,1,0,2"
    assert runner.validate_forward_gate(args, "verified-commit") == gate


@pytest.mark.parametrize("field", ["cache_metadata_sha256", "report_sha256", "source_samples_sha256"])
def test_forward_gate_rejects_changed_evidence(forward_gate, field):
    args, gate = forward_gate
    path = Path(next(iter(gate[field])))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="evidence changed|observation archive"):
        runner.validate_forward_gate(args, "verified-commit")


@pytest.mark.parametrize("change, value", [
    ("passed", False), ("cache_root", "/different/cache"), ("selected_nodes", 41),
    ("chip_indices", [0, 1, 2]), ("exojax_commit", "different-commit"),
    ("x64", False), ("cache_metadata_sha256", {}), ("report_sha256", {}),
    ("source_samples_sha256", {}),
])
def test_forward_gate_rejects_target_or_evidence_mismatch(forward_gate, change, value):
    args, gate = forward_gate
    gate[change] = value
    runner.write_json(args.forward_validation, gate)
    with pytest.raises(ValueError):
        runner.validate_forward_gate(args, "verified-commit")


def test_forward_gate_requires_a_report(forward_gate):
    args, _ = forward_gate
    args.forward_validation = None
    with pytest.raises(ValueError, match="require --forward-validation"):
        runner.validate_forward_gate(args, "verified-commit")


@pytest.mark.parametrize("name,value", [("nx", 9000), ("noise_jitter", 1e-5)])
def test_forward_gate_rejects_a_different_grid_or_jitter(forward_gate, name, value):
    args, gate = forward_gate
    gate[name] = value
    runner.write_json(args.forward_validation, gate)
    with pytest.raises(ValueError, match="spectral grid or noise jitter"):
        runner.validate_forward_gate(args, "verified-commit")


def test_forward_gate_rechecks_validated_state_hash(forward_gate):
    args, _ = forward_gate
    args.state_file.write_bytes(args.state_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        runner.validate_forward_gate(args, "verified-commit")


@pytest.mark.parametrize("name", ["P", "cosi", "v", "u1", "u2", "logg", "log_w"])
def test_forward_gate_rejects_changed_fixed_operator(forward_gate, tmp_path, name):
    args, _ = forward_gate
    with np.load(args.state_file) as archive:
        state = {key: archive[key] for key in archive.files}
    state[name] = state[name].copy()
    state[name].flat[0] += .01
    args.state_file = tmp_path / "changed_initial_state.npz"
    np.savez(args.state_file, **state)
    with pytest.raises(ValueError, match="fixed operator differs"):
        runner.validate_forward_gate(args, "verified-commit")


def test_forward_gate_allows_new_dynamic_state_and_reorders_phase_weights(forward_gate, tmp_path):
    args, gate = forward_gate
    validated_path = args.state_file
    with np.load(validated_path) as archive:
        state = {key: archive[key] for key in archive.files}
    state["log_w"] = np.arange(56, dtype=float).reshape(4, 14) ** 2 / 10000
    np.savez(validated_path, **state)
    gate["inputstates"][str(validated_path)] = runner.sha256(validated_path)
    runner.write_json(args.forward_validation, gate)
    order = np.asarray([3, 0, 2, 1])
    state["log_w"] = state["log_w"][order]
    state["chip_indices"] = state["chip_indices"][order]
    state["T0"] = np.asarray(1350.)
    state["delta_log_p"] = np.linspace(-.1, .1, 768)
    args.state_file = tmp_path / "refined_reordered_state.npz"
    np.savez(args.state_file, **state)
    assert runner.validate_forward_gate(args, "verified-commit") == gate


@pytest.fixture
def mocked_cache_problem(monkeypatch, tmp_path):
    from doraex.spectra.exojax_forward import (
        _internal_spectrum_wavelength_bounds,
        _opacity_cache_namespace,
    )
    from doraex.workflows import exact_pressure_investigation, on_the_fly_pressure_retrieval

    args = SimpleNamespace(
        chip_indices="1", database_dir=tmp_path / "database", opacity_cache_dir=tmp_path / "cache",
        nodes=21, rt_pixel_batch_size=16, marginalize_a=False, quadrature_order=64,
        samples=tmp_path / "samples.npz",
    )
    np.savez(args.samples, nside=np.asarray(8))
    chip = SimpleNamespace(chip_index=1, wavelengths=np.asarray([22000., 22010.]))
    padded = np.asarray([21996., 22014.])
    state = {"v": 31.2}
    geometry = SimpleNamespace(theta=np.asarray([.4, 1., 2.]), phi=np.asarray([0., 2., 4.]),
                               distance_matrix=np.eye(3))
    calls = []

    def responses(response_args, chip_data, **kwargs):
        calls.append(response_args)
        assert chip_data == [chip]
        np.testing.assert_array_equal(kwargs["profile_wavelengths"][0], padded)
        return ["response"]

    product_helpers = SimpleNamespace(
        _load_chip_data_from_samples=lambda archive, chips: [chip],
        _load_profile_wavelengths=lambda archive, chips: [padded],
        _build_product_geometry=lambda archive, nside, x64: geometry,
    )
    monkeypatch.setitem(sys.modules, "make_m9_v1_exact_products", product_helpers)
    monkeypatch.setattr(on_the_fly_pressure_retrieval, "build_fixed_pressure_gp_eigendecomposition",
                        lambda *args, **kwargs: {"eigenvalues": np.ones(2), "pixel_eigenvectors": np.ones((3, 2))})
    monkeypatch.setattr(on_the_fly_pressure_retrieval, "build_response_functions", responses)
    sentinel = object()
    monkeypatch.setattr(exact_pressure_investigation, "ExactPressureProblem", lambda *args, **kwargs: sentinel)
    bounds = _internal_spectrum_wavelength_bounds(
        chip.wavelengths, padded, systemic_rv=on_the_fly_pressure_retrieval.YAMA_L16B_EXOMOL_ATMOSPHERE.rv,
    )

    def namespace(grid):
        return _opacity_cache_namespace(args.opacity_cache_dir, grid, 4500, 1e-4, 1e2, 101, 210., 3500., "diffgrid", 21)

    return args, state, namespace(bounds), namespace(chip.wavelengths), calls, sentinel


def _write_mock_cache(directory, version=3, kernel="real_space", count=4):
    for molecule in ("CO", "H2O", "CH4", "HF")[:count]:
        archive = directory / f"opa{molecule}.zarr"
        archive.mkdir(parents=True)
        state = {"opa_state": {"teacher_profile_kernel": kernel}}
        path = archive / ("zarr.json" if version == 3 else ".zattrs")
        path.write_text(json.dumps({"attributes": state} if version == 3 else state))


@pytest.mark.parametrize("version", [2, 3])
def test_cache_prevalidation_accepts_real_space_internal_grid_before_building_responses(mocked_cache_problem, version):
    args, state, internal_directory, _, calls, sentinel = mocked_cache_problem
    _write_mock_cache(internal_directory, version=version)
    assert runner.build_problem(args, state) is sentinel
    assert len(calls) == 1
    assert calls[0].exact_rt_local_normalization == "none"
    assert calls[0].opacity_mode == "diffgrid"


@pytest.mark.parametrize("failure", ["missing", "partial", "wrong_grid", "analytic_teacher"])
def test_cache_prevalidation_rejects_before_constructor_can_create_tables(mocked_cache_problem, failure):
    args, state, internal_directory, observed_directory, calls, _ = mocked_cache_problem
    if failure == "partial":
        _write_mock_cache(internal_directory, count=3)
    elif failure == "wrong_grid":
        _write_mock_cache(observed_directory)
    elif failure == "analytic_teacher":
        _write_mock_cache(internal_directory, kernel="analytic")
    with pytest.raises(ValueError, match="four molecular archives|real_space teacher"):
        runner.build_problem(args, state)
    assert calls == []


@pytest.mark.parametrize("archived_latent, target, expected_source, expected_scale", [
    (1.5, 1., "raw-linear", .25), (.8, .8, "archived", None),
    (1.5, 2., "raw-linear", .5),
])
def test_raw_linear_selection_uses_exact_objective_and_keeps_archived_candidate(
    archived_latent, target, expected_source, expected_scale,
):
    with jax.enable_x64():
        seen_normalizations = []
        problem = SimpleNamespace(map_slice=slice(1, 2), marginalized=True)

        def propose(position, normalizations):
            seen_normalizations.append(normalizations)
            return position.at[problem.map_slice].set(4.), {"latent": [4.]}

        problem.raw_linear_initial_position = propose
        problem.potential = lambda position: (position[1] - target)**2 if position[1] < 3. else np.inf
        initial = jnp.asarray([.13, archived_latent])
        candidate, scan = runner.select_raw_linear_initial_position(problem, initial, np.asarray([1.12]))
        assert scan["selected_source"] == expected_source
        assert scan["selected_scale"] == expected_scale
        assert len(scan["trials"]) == (6 if expected_scale == .5 else 7)
        assert [trial["scale"] for trial in scan["trials"][1:6]] == [0., .25, .5, .75, 1.]
        assert scan["trials"][5]["exact_objective"] is None
        assert candidate[0] == initial[0]
        assert float(problem.potential(candidate)) <= float(problem.potential(initial))
        np.testing.assert_array_equal(seen_normalizations[0], [1.12])
        json.dumps(scan, allow_nan=False)


def test_raw_linear_backtracking_finds_improvement_below_coarse_scales():
    problem = SimpleNamespace(map_slice=slice(1, 2), marginalized=False)
    problem.raw_linear_initial_position = lambda position, normalizations: (
        position.at[problem.map_slice].set(1.), {},
    )
    problem.potential = lambda position: (position[1] - .02)**2
    initial = jnp.asarray([.13, .8])

    candidate, scan = runner.select_raw_linear_initial_position(problem, initial, np.asarray([1.1]))

    zero_objective = scan["trials"][1]["exact_objective"]
    assert all(trial["exact_objective"] > zero_objective for trial in scan["trials"][2:6])
    assert scan["selected_scale"] == .015625
    assert scan["selected_exact_objective"] < zero_objective
    assert scan["backtracking"]["additional_trials"] == 5
    assert scan["backtracking"]["stop_reason"] == "improvement_stopped"
    assert scan["trials"][-1]["exact_objective"] > scan["selected_exact_objective"]
    assert candidate[0] == initial[0]
    json.dumps(scan, allow_nan=False)


def test_raw_linear_backtracking_stops_for_nondescending_marginal_direction():
    problem = SimpleNamespace(map_slice=slice(1, 2), marginalized=True)
    problem.raw_linear_initial_position = lambda position, normalizations: (
        position.at[problem.map_slice].set(1.), {},
    )
    evaluated = []

    def objective(position):
        evaluated.append(float(position[1]))
        return position[1]**2

    problem.potential = objective
    initial = jnp.asarray([.13, .8])
    candidate, scan = runner.select_raw_linear_initial_position(problem, initial, np.asarray([1.1]))

    assert len(evaluated) == 14
    assert scan["backtracking"]["additional_trials"] == 8
    assert scan["backtracking"]["stop_reason"] == "maximum_additional_trials"
    assert scan["selected_scale"] == 0.
    assert scan["selected_exact_objective"] == 0.
    assert candidate[0] == initial[0]
    assert candidate[1] == 0.
    json.dumps(scan, allow_nan=False)


def test_raw_linear_selection_rejects_all_nonfinite_candidates():
    problem = SimpleNamespace(map_slice=slice(0, 1), marginalized=False)
    problem.raw_linear_initial_position = lambda position, normalizations: (position, {})
    problem.potential = lambda position: np.nan
    with pytest.raises(FloatingPointError, match="Every archived/raw-linear"):
        runner.select_raw_linear_initial_position(problem, jnp.asarray([1.]), np.asarray([1.1]))


def test_initial_map_source_defaults_to_archived_and_accepts_raw_linear(monkeypatch, tmp_path):
    argv = ["m9_v3_run.py", "--stage", "evaluate", "--out-dir", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)
    assert runner.parse_args().initial_map_source == "archived"
    assert runner.parse_args().nodes == 81
    monkeypatch.setattr("sys.argv", [*argv, "--initial-map-source", "raw-linear"])
    assert runner.parse_args().initial_map_source == "raw-linear"


@pytest.mark.parametrize("depth_options, counts, expected_depths, expected_steps", [
    ([], (3, 4), (7, 7), (127, 127)),
    (["--warmup-max-tree-depth", "9", "--max-tree-depth", "11"],
     (2000, 1500), (9, 11), (511, 2047)),
])
def test_sampler_factory_applies_cli_depths_at_actual_numpyro_phase_boundary(
    monkeypatch, tmp_path, depth_options, counts, expected_depths, expected_steps,
):
    monkeypatch.setattr("sys.argv", [
        "m9_v3_run.py", "--stage", "sample", "--out-dir", str(tmp_path),
        "--num-warmup", str(counts[0]), "--num-samples", str(counts[1]),
        *depth_options,
    ])
    args = runner.parse_args()
    if not depth_options:
        assert args.warmup_max_tree_depth is None
    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64():
        mcmc, settings = runner.build_mcmc(
            args, lambda position: .5 * position @ position, jnp.eye(2),
        )
        assert (mcmc.num_warmup, mcmc.num_samples) == counts
        assert (settings["warmup_max_tree_depth"], settings["max_tree_depth"]) == expected_depths
        assert (settings["warmup_tree_depth_cap_num_steps"],
                settings["tree_depth_cap_num_steps"]) == expected_steps
        kernel = mcmc.sampler
        # Force cap-reaching trajectories without changing the factory's depths.
        # Jump the iteration counter to test production boundaries cheaply.
        kernel._step_size = 1e-5
        kernel._adapt_step_size = False
        kernel._adapt_mass_matrix = False
        state = kernel.init(
            jax.random.PRNGKey(17), num_warmup=mcmc.num_warmup, init_params=jnp.zeros(2),
        )
        sample = jax.jit(kernel.sample)
        last_warmup = sample(state._replace(i=jnp.asarray(mcmc.num_warmup - 1)), (), {})
        first_saved = sample(last_warmup, (), {})
        last_saved = sample(first_saved._replace(
            i=jnp.asarray(mcmc.num_warmup + mcmc.num_samples - 1),
        ), (), {})
        assert int(last_warmup.i) == mcmc.num_warmup
        assert int(first_saved.i) == mcmc.num_warmup + 1
        assert int(last_saved.i) == mcmc.num_warmup + mcmc.num_samples
        assert int(last_warmup.num_steps) == expected_steps[0]
        assert int(first_saved.num_steps) == expected_steps[1]
        assert int(last_saved.num_steps) == expected_steps[1]


@pytest.mark.parametrize("option", ["--warmup-max-tree-depth", "--max-tree-depth"])
@pytest.mark.parametrize("depth", [0, -1])
def test_sampler_factory_rejects_nonpositive_depths(monkeypatch, tmp_path, option, depth):
    monkeypatch.setattr("sys.argv", [
        "m9_v3_run.py", "--stage", "sample", "--out-dir", str(tmp_path), option, str(depth),
    ])
    with pytest.raises(ValueError, match=f"{option} must be at least 1"):
        runner.build_mcmc(runner.parse_args(), lambda position: .5 * position @ position, np.eye(2))


@pytest.mark.parametrize("fail_recovery", [False, True])
def test_completed_hmc_draws_survive_conditional_a_recovery_failure(monkeypatch, tmp_path, fail_recovery):
    from doraex.inference import normalization_marginalization

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    state_file, samples_file = tmp_path / "state.npz", tmp_path / "input.npz"
    np.savez(state_file, A=np.full(4, 1.1))
    np.savez(samples_file, unused=np.asarray(1.))
    out_dir = tmp_path / "run"
    monkeypatch.setattr("sys.argv", [
        "m9_v3_run.py", "--stage", "sample", "--platform", "cpu",
        "--state-file", str(state_file), "--samples", str(samples_file), "--out-dir", str(out_dir),
        "--marginalize-a", "--num-warmup", "3", "--num-samples", "4", "--max-tree-depth", "2",
    ])
    args = runner.parse_args()
    monkeypatch.setattr(runner, "active_exojax_checkout", lambda: (tmp_path / "exojax", "test-commit"))
    monkeypatch.setattr(runner, "validate_forward_gate", lambda *args: None)

    problem = make_problem()
    problem.dimension, problem.marginalized = 1, True
    problem.map_slice = slice(0, 1)
    problem.encode = lambda state: jnp.asarray([.2])
    problem.prior_potential = lambda position: .5 * position @ position
    problem.normal_equations = lambda position: (jax.grad(problem.prior_potential)(position), jnp.eye(1))
    problem.eigenvalues, problem.eigenvectors = np.ones(1), np.ones((3, 1))
    problem.normalization_moments = lambda position: [(jnp.asarray(1.1), jnp.asarray(1e-6))] * problem.nchip
    monkeypatch.setattr(runner, "build_problem", lambda *args: problem)
    built = []
    original_build_mcmc = runner.build_mcmc

    def capture_mcmc(*args):
        result = original_build_mcmc(*args)
        built.append(result[0])
        return result

    monkeypatch.setattr(runner, "build_mcmc", capture_mcmc)
    if fail_recovery:
        def fail_conditional_draw(*args, **kwargs):
            assert (out_dir / "mcmc_raw.npz").is_file()
            assert not (out_dir / "mcmc_raw.npz.tmp").exists()
            raise FloatingPointError("Injected conditional A recovery failure")

        monkeypatch.setattr(normalization_marginalization, "draw_conditional_normalization", fail_conditional_draw)
    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(), jax.default_matmul_precision("highest"):
        if fail_recovery:
            with pytest.raises(FloatingPointError, match="Injected conditional A"):
                runner.run(args)
        else:
            runner.run(args)

    with np.load(out_dir / "mcmc_raw.npz", allow_pickle=False) as archive:
        raw = dict(archive)
    np.testing.assert_array_equal(raw["position"], built[0].get_samples())
    assert raw["position"].shape == (4, 1)
    for name, value in built[0].get_extra_fields().items():
        np.testing.assert_array_equal(raw[f"extra_{name}"], value)
    np.testing.assert_array_equal(raw["chip_indices"], [3, 1])
    assert bool(raw["marginalized_A"]) and int(raw["seed"]) == args.seed
    assert int(raw["max_tree_depth"]) == 2
    assert not (out_dir / "RUNNING").exists()
    assert not (out_dir / "mcmc_raw.npz.tmp").exists()
    if fail_recovery:
        assert (out_dir / "FAILED").exists()
        assert not (out_dir / "COMPLETE").exists()
        assert not (out_dir / "samples.npz").exists()
        assert json.loads((out_dir / "failure.json").read_text())["exception"] == "FloatingPointError"
    else:
        assert (out_dir / "COMPLETE").exists()
        assert not (out_dir / "FAILED").exists()
        with np.load(out_dir / "samples.npz", allow_pickle=False) as final:
            for name, value in raw.items():
                np.testing.assert_array_equal(final[name], value)
            assert final["A"].shape == (4, problem.nchip)
            assert final["delta_log_p"].shape == (4, 3)
