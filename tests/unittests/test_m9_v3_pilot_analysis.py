"""Offline pilot analysis must not confuse incompatible or duplicate runs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "examples/luhman16b_yama/analyze_m9_v3_pilots.py"
SPEC = importlib.util.spec_from_file_location("m9_v3_pilot_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write_run(tmp_path, name, seed, marginalized=False, count=128, shift=0.0, data_shift=0.0, source="original"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "source").mkdir()
    source_path = directory / "source" / "model.py"
    source_path.write_text(source)
    theta = analysis.fallback_ring_angles()
    phi = np.concatenate([np.arange(np.sum(theta == angle)) * 2 * np.pi / np.sum(theta == angle)
                          for angle in np.unique(theta)])
    chips = np.asarray([0, 1])
    contract = {
        "format_version": np.asarray(1), "target": np.asarray("m9_v3_exact_raw_intensity"),
        "chip_indices": chips, "theta": theta, "phi": phi,
        "fixed_P": np.asarray(5.0), "fixed_cosi": np.asarray(.4), "noise_jitter": np.asarray(1e-6),
        "observed_chip0": np.linspace(.8, 1.0, 20) + data_shift,
        "observed_chip1": np.linspace(.9, 1.1, 20),
    }
    np.savez(directory / "observation_contract.npz", **contract)
    rng = np.random.default_rng(seed)
    axisymmetric_amplitude = .2 + .02 * rng.normal(size=count)
    nonaxisymmetric_amplitude = .04 + .01 * rng.normal(size=count)
    delta = axisymmetric_amplitude[:, None] * np.cos(theta)[None, :] + nonaxisymmetric_amplitude[:, None] * np.sin(phi)[None, :]
    a = 1.1 + .01 * rng.normal(size=(count, 2))
    payload = {
        "seed": np.asarray(seed), "marginalized_A": np.asarray(marginalized), "chip_indices": chips,
        "A": a, "log_p_cloud": 1.0 + shift + .03 * rng.normal(size=count),
        "sigma_log_p": .3 + .01 * rng.normal(size=count), "sigma_d": .03 + .001 * rng.normal(size=(count, 2)),
        "delta_log_p": delta, "extra_energy": rng.normal(size=count),
        "extra_potential_energy": np.arange(count, dtype=float) * 1e6,
        "extra_diverging": np.zeros(count, dtype=bool), "extra_num_steps": np.full(count, 7),
        "extra_accept_prob": np.full(count, .92),
    }
    if marginalized:
        payload.update(A_conditional_mean=a + .001, A_conditional_variance=np.full_like(a, 1e-6))
    np.savez(directory / "samples.npz", **payload)
    provenance = {
        "arguments": {"num_samples": count, "seed": seed, "marginalize_a": marginalized,
                      "chip_indices": "0,1", "nodes": 21, "quadrature_order": 64,
                      "rt_pixel_batch_size": 16, "max_tree_depth": 5},
        "source_sha256": {"/test/model.py": analysis.file_sha256(source_path)},
        "state_sha256": f"initial_state_{seed}", "exojax_commit": "fixed_commit", "x64": True,
        "observation_contract_sha256": analysis.file_sha256(directory / "observation_contract.npz"),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance))
    (directory / "sampling.json").write_text(json.dumps({"seconds": 100.0, "num_warmup": 50, "num_samples": count}))
    (directory / "completion.json").write_text(json.dumps({"seconds": 120.0}))
    (directory / "COMPLETE").touch()
    return directory


def test_independent_chains_report_rhat_physical_maps_energy_and_measured_cost(tmp_path):
    first = write_run(tmp_path, "seed0", 0)
    second = write_run(tmp_path, "seed1", 1, shift=.5)
    result = analysis.analyze_runs([first, second])
    json.dumps(result, allow_nan=False)
    group = result["groups"][0]
    assert group["independent_chain_rhat_available"]
    assert group["parameters"]["log_p_cloud"]["independent_chain_rank_rhat"] > 1.1
    rms = group["parameters"]["map_rms_dex"]
    assert rms["rank_bulk_ess"] > 0
    assert rms["rank_tail_ess"] > 0
    assert rms["bulk_ess_per_sampling_second"] == rms["rank_bulk_ess"] / 200.0
    assert rms["bulk_ess_per_total_second"] == rms["rank_bulk_ess"] / 240.0
    assert len(group["latitude_rings"]) == 31
    assert sum(ring["pixel_count"] for ring in group["latitude_rings"]) == 768
    assert all(run["energy_ebfmi"] > 1.0 for run in group["runs"])
    assert all(run["divergences"] == 0 and run["depth_cap_fraction"] == 0 for run in group["runs"])
    loaded = analysis.load_run(first)["series"]
    np.testing.assert_allclose(
        loaded["map_rms_dex"]**2,
        loaded["axisymmetric_rms_dex"]**2 + loaded["nonaxisymmetric_rms_dex"]**2,
        rtol=1e-13,
    )
    assert not group["scientifically_accepted"]


def test_single_chain_and_ten_draw_pilots_are_not_accepted(tmp_path):
    directory = write_run(tmp_path, "short", 0, count=10)
    group = analysis.analyze_runs([directory])["groups"][0]
    assert not group["independent_chain_rhat_available"]
    assert group["parameters"]["A_common_mean"]["independent_chain_rank_rhat"] is None
    assert any("Fewer than 100" in note for note in group["limitations"])
    assert not group["scientifically_accepted"]


@pytest.mark.parametrize("warmup_depth, identical", [(None, True), (5, True), (3, False)])
def test_sampler_comparison_resolves_legacy_and_separate_warmup_depths(tmp_path, warmup_depth, identical):
    legacy = write_run(tmp_path, "legacy_depth", 0)
    separate = write_run(tmp_path, "separate_depth", 1)
    provenance_path = separate / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["arguments"]["warmup_max_tree_depth"] = warmup_depth
    provenance_path.write_text(json.dumps(provenance))

    result = analysis.analyze_runs([legacy, separate])
    assert result["comparison"]["identical_sampler_settings"] is identical
    runs = result["groups"][0]["runs"]
    assert runs[0]["sampler_arguments"]["warmup_max_tree_depth"] == 5
    assert runs[1]["sampler_arguments"]["warmup_max_tree_depth"] == (5 if warmup_depth is None else warmup_depth)
    # Saved draws have seven steps: only the sampling depth determines its cap.
    assert runs[1]["depth_cap_fraction"] == 0


def test_float32_angular_roundoff_does_not_split_healpix_latitude_rings(tmp_path):
    directory = write_run(tmp_path, "roundoff", 0, count=8)
    with np.load(directory / "samples.npz") as archive:
        samples = dict(archive)
    theta = analysis.fallback_ring_angles().astype(np.float32)
    theta[::2] = np.nextafter(theta[::2], np.float32(np.inf))
    assert np.unique(theta).size > 31
    samples["delta_log_p"][:] = 0
    samples["delta_log_p"][:, 0] = 1
    samples["delta_log_p"][:, 1] = -1
    series, rings = analysis.physical_series(samples, theta.astype(float))
    assert len(rings) == 31
    np.testing.assert_array_equal(
        [ring["pixel_count"] for ring in rings],
        4 * np.minimum(np.minimum(np.arange(1, 32), 32 - np.arange(1, 32)), 8),
    )
    # Opposite perturbations in the same polar ring have zero longitude mean.
    np.testing.assert_allclose(series["axisymmetric_rms_dex"], 0., atol=1e-15)
    np.testing.assert_allclose(series["nonaxisymmetric_rms_dex"], np.sqrt(2 / 768))


def test_repeated_seeds_cannot_inflate_effective_chain_count(tmp_path):
    directories = [write_run(tmp_path, name, 0) for name in ("copy0", "copy1")]
    group = analysis.analyze_runs(directories)["groups"][0]
    assert group["unique_seed_count"] == 1
    assert not group["independent_chain_rhat_available"]
    assert group["parameters"]["log_p_cloud"]["rank_bulk_ess"] is None
    with pytest.raises(ValueError, match="supplied twice"):
        analysis.analyze_runs([directories[0], directories[0]])


def test_variants_compare_only_with_identical_data_operator_and_sources(tmp_path):
    explicit = write_run(tmp_path, "explicit", 0)
    marginal = write_run(tmp_path, "marginal", 1, marginalized=True)
    result = analysis.analyze_runs([explicit, marginal])
    assert result["comparison"]["controlled_comparison_allowed"]
    assert result["comparison"]["identical_data_content_hashes"]
    assert result["comparison"]["identical_operator_content_hashes"]
    assert "A_common_mean" in result["comparison"]["marginalized_to_explicit_bulk_ess_per_second"]
    other_data = write_run(tmp_path, "other_data", 1, marginalized=True, data_shift=.01)
    result = analysis.analyze_runs([explicit, other_data])
    assert not result["comparison"]["controlled_comparison_allowed"]
    assert not result["comparison"]["identical_data_content_hashes"]
    other_source = write_run(tmp_path, "other_source", 2, marginalized=True, source="changed target")
    result = analysis.analyze_runs([explicit, other_source])
    assert not result["comparison"]["controlled_comparison_allowed"]
    assert not result["comparison"]["identical_source_snapshots"]


def test_tampered_evidence_and_missing_legacy_contract_are_distinguished(tmp_path):
    tampered = write_run(tmp_path, "tampered", 0)
    (tampered / "source/model.py").write_text("changed after completion")
    with pytest.raises(ValueError, match="snapshot"):
        analysis.analyze_runs([tampered])
    other = write_run(tmp_path, "tampered_data", 1)
    (other / "observation_contract.npz").write_bytes(b"invalid archive")
    with pytest.raises(ValueError, match="changed after"):
        analysis.analyze_runs([other])
    legacy = write_run(tmp_path, "legacy", 2)
    (legacy / "observation_contract.npz").unlink()
    result = analysis.analyze_runs([legacy])
    assert not result["comparison"]["same_target_verified"]
    assert any("unverified" in note for note in result["groups"][0]["limitations"])


def test_content_hash_ignores_archive_layout_and_preserves_array_shapes():
    first = {"a": np.arange(8).reshape(2, 4), "b": np.asarray(1)}
    reordered = {"b": np.asarray(1), "a": np.asfortranarray(first["a"])}
    assert analysis.content_sha256(first) == analysis.content_sha256(reordered)
    assert analysis.content_sha256({"b": np.asarray(1)}) != analysis.content_sha256({"b": np.asarray([1])})


@pytest.mark.parametrize("change", ["geometry", "nodes", "exojax_commit", "forward_validation_sha256"])
def test_target_contract_mismatches_prevent_controlled_comparison(tmp_path, change):
    explicit = write_run(tmp_path, "explicit", 0)
    marginal = write_run(tmp_path, "marginal", 1, marginalized=True)
    provenance_path = marginal / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if change == "geometry":
        with np.load(marginal / "observation_contract.npz") as archive:
            arrays = dict(archive)
        arrays["fixed_P"] = np.asarray(6.0)
        np.savez(marginal / "observation_contract.npz", **arrays)
        provenance["observation_contract_sha256"] = analysis.file_sha256(marginal / "observation_contract.npz")
    elif change == "nodes":
        provenance["arguments"]["nodes"] = 25
    else:
        provenance[change] = "changed"
    provenance_path.write_text(json.dumps(provenance))
    comparison = analysis.analyze_runs([explicit, marginal])["comparison"]
    assert not comparison["controlled_comparison_allowed"]
    assert comparison["identical_data_content_hashes"]
