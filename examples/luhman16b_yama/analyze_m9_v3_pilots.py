"""Compare archived M9 v3 pilots on CPU without evaluating the forward model."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/doraex_m9_v3_mpl")

import arviz as az
import numpy as np


TARGET_ARGUMENTS = ("nodes", "quadrature_order", "rt_pixel_batch_size")
REQUIRED_SERIES = ("A", "log_p_cloud", "sigma_log_p", "sigma_d", "delta_log_p")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(arrays):
    """Hash array values, names, shapes, and dtypes independently of ZIP metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.asarray(value)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def fallback_ring_angles():
    """Return the fixed nside=8 RING colatitudes for legacy diagnostic displays."""
    nside = 8
    ring = np.arange(1, 4 * nside)
    counts = 4 * np.minimum(np.minimum(ring, 4 * nside - ring), nside)
    z = np.where(
        ring < nside, 1 - ring**2 / (3 * nside**2),
        np.where(ring <= 3 * nside, (2 * nside - ring) * 2 / (3 * nside),
                 -1 + (4 * nside - ring)**2 / (3 * nside**2)),
    )
    return np.repeat(np.arccos(z), counts)


def latitude_ring_masks(theta, tolerance=1e-6):
    """Group latitude rings despite float32 angular roundoff in saved geometry."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or not np.all(np.isfinite(theta)):
        raise ValueError("Ring angles must be a finite one-dimensional array.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Ring tolerance must be finite and positive.")
    order = np.argsort(theta)
    boundaries = np.r_[0, np.flatnonzero(np.diff(theta[order]) > tolerance) + 1, theta.size]
    masks = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop > start:
            mask = np.zeros(theta.size, dtype=bool)
            mask[order[start:stop]] = True
            masks.append(mask)
    return masks


def physical_series(samples, theta):
    """Separate every draw into longitude means and residual map structure."""
    delta = np.asarray(samples["delta_log_p"], dtype=float)
    if delta.ndim != 2 or delta.shape[1] != theta.size:
        raise ValueError("delta_log_p does not match the archived pixel geometry.")
    series = {}
    for name in REQUIRED_SERIES[:-1]:
        array = np.asarray(samples[name], dtype=float)
        if array.ndim == 1:
            series[name] = array
        elif array.ndim == 2:
            for column, chip in enumerate(samples["chip_indices"]):
                series[f"{name}[chip{chip}]"] = array[:, column]
        else:
            raise ValueError(f"Unexpected sample shape for {name}.")
    series["A_common_mean"] = np.mean(samples["A"], axis=1)
    zonal = np.empty_like(delta)
    rings = []
    for index, mask in enumerate(latitude_ring_masks(theta)):
        angle = float(np.mean(theta[mask]))
        values = delta[:, mask].mean(axis=1)
        zonal[:, mask] = values[:, None]
        latitude = 90 - np.rad2deg(angle)
        if abs(latitude) < 1e-5:
            latitude = 0.0
        name = f"axisymmetric_ring[{index}]"
        series[name] = values
        rings.append({"parameter": name, "latitude_deg": float(latitude), "pixel_count": int(mask.sum())})
    series["map_rms_dex"] = np.sqrt(np.mean(delta**2, axis=1))
    series["axisymmetric_rms_dex"] = np.sqrt(np.mean(zonal**2, axis=1))
    series["nonaxisymmetric_rms_dex"] = np.sqrt(np.mean((delta - zonal)**2, axis=1))
    return series, rings


def load_run(directory):
    directory = Path(directory).resolve()
    if not (directory / "COMPLETE").exists() or any((directory / mark).exists() for mark in ("FAILED", "RUNNING")):
        raise ValueError(f"Run is not complete: {directory}")
    provenance = json.loads((directory / "provenance.json").read_text())
    sampling = json.loads((directory / "sampling.json").read_text())
    arguments = provenance["arguments"]
    with np.load(directory / "samples.npz", allow_pickle=False) as archive:
        samples = dict(archive)
    count = len(samples["log_p_cloud"])
    seed = int(samples["seed"])
    marginalized = bool(samples["marginalized_A"])
    if count < 1 or count != sampling["num_samples"] or count != arguments["num_samples"]:
        raise ValueError(f"Archived draw counts disagree: {directory}")
    if seed != arguments["seed"] or marginalized != arguments["marginalize_a"]:
        raise ValueError(f"Archived variant or seed disagrees with provenance: {directory}")
    for key in REQUIRED_SERIES:
        values = np.asarray(samples[key])
        if values.shape[0] != count or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid physical samples for {key}: {directory}")
    chips = np.asarray(samples["chip_indices"], dtype=int)
    if chips.tolist() != [int(value) for value in arguments["chip_indices"].split(",")]:
        raise ValueError(f"Archived chip order disagrees with provenance: {directory}")
    for key in ("A", "sigma_d"):
        if np.shape(samples[key]) != (count, len(chips)):
            raise ValueError(f"Invalid per-chip sample shape for {key}: {directory}")
    if marginalized and not all(key in samples for key in ("A_conditional_mean", "A_conditional_variance")):
        raise ValueError(f"Marginalized run lacks the continuous-A recovery record: {directory}")

    sources = {}
    for filename, expected in provenance["source_sha256"].items():
        name = Path(filename).name
        if name in sources or file_sha256(directory / "source" / name) != expected:
            raise ValueError(f"Source snapshot does not match provenance: {filename}")
        sources[name] = expected
    if not sources:
        raise ValueError(f"Missing source provenance: {directory}")
    contract_path = directory / "observation_contract.npz"
    verified = contract_path.exists() and bool(provenance.get("observation_contract_sha256"))
    contract = None
    if verified:
        if file_sha256(contract_path) != provenance["observation_contract_sha256"]:
            raise ValueError(f"Observation contract changed after the run: {directory}")
        with np.load(contract_path, allow_pickle=False) as archive:
            arrays = dict(archive)
        if str(arrays.get("target", "")) != "m9_v3_exact_raw_intensity":
            raise ValueError(f"Observation contract is not the M9 v3 target: {directory}")
        if not np.array_equal(arrays["chip_indices"], chips):
            raise ValueError(f"Observation contract has different chips: {directory}")
        data = {f"observed_chip{chip}": arrays[f"observed_chip{chip}"] for chip in chips}
        if not all(np.all(np.isfinite(value)) for value in data.values()):
            raise ValueError(f"Observation contract contains non-finite data: {directory}")
        operator = {key: value for key, value in arrays.items() if not key.startswith("observed_chip")}
        theta = np.asarray(arrays["theta"], dtype=float)
        contract = {
            "data_sha256": content_sha256(data), "operator_sha256": content_sha256(operator),
            "source_sha256": sources, "exojax_commit": provenance["exojax_commit"],
            "x64": provenance["x64"],
            "forward_validation_sha256": provenance.get("forward_validation_sha256"),
            "opacity_cache_dir": arguments.get("opacity_cache_dir"),
            "target_arguments": {key: arguments[key] for key in TARGET_ARGUMENTS},
        }
    else:
        theta = fallback_ring_angles()
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() if verified else str(directory)
    series, rings = physical_series(samples, theta)
    for key in ("extra_energy", "extra_diverging", "extra_num_steps", "extra_accept_prob"):
        if key not in samples or np.shape(samples[key]) != (count,) or not np.all(np.isfinite(samples[key])):
            raise ValueError(f"Missing or invalid sampling diagnostic {key}: {directory}")
    energy = np.asarray(samples["extra_energy"], dtype=float)
    ebfmi = float(az.bfmi(energy[None, :])[0]) if count > 1 and np.var(energy) > 0 else None
    seconds = float(sampling["seconds"])
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid sampling elapsed time: {directory}")
    completion = directory / "completion.json"
    total_seconds = float(json.loads(completion.read_text())["seconds"]) if completion.exists() else None
    steps, counts = np.unique(samples["extra_num_steps"], return_counts=True)
    warmup_max_tree_depth = arguments.get("warmup_max_tree_depth")
    if warmup_max_tree_depth is None:
        warmup_max_tree_depth = arguments["max_tree_depth"]
    return {
        "directory": str(directory), "seed": seed,
        "variant": "marginalized_A" if marginalized else "explicit_A", "draws": count,
        "warmup": int(sampling["num_warmup"]), "initial_state_sha256": provenance["state_sha256"],
        "sampler_arguments": {
            **{key: arguments.get(key) for key in
               ("max_tree_depth", "target_accept_prob", "adapt_mass_matrix")},
            "warmup_max_tree_depth": warmup_max_tree_depth,
        },
        "sampling_seconds_including_warmup_and_recovery": seconds, "total_seconds": total_seconds,
        "contract_verified": verified, "contract": contract, "contract_fingerprint": fingerprint,
        "A_kind": "continuous_conditional_draw" if marginalized else "joint_sample",
        "energy_ebfmi": ebfmi, "divergences": int(np.sum(samples["extra_diverging"])),
        "acceptance_mean": float(np.mean(samples["extra_accept_prob"])),
        "depth_cap_fraction": float(np.mean(samples["extra_num_steps"] == 2**arguments["max_tree_depth"] - 1)),
        "num_steps_counts": dict(zip(map(str, steps.tolist()), counts.tolist())),
        "rings": rings, "series": series,
    }


def correlations(series):
    names = [name for name, values in series.items() if np.std(values) > 0]
    if len(names) < 2:
        return {"strongest_pairs": [], "A_chip_pairs": [], "A_common_links": []}
    matrix = np.corrcoef([series[name] for name in names])
    pairs = [{"first": first, "second": names[j], "pearson_r": float(matrix[i, j])}
             for i, first in enumerate(names) for j in range(i + 1, len(names))]
    pairs.sort(key=lambda pair: abs(pair["pearson_r"]), reverse=True)
    return {
        "strongest_pairs": pairs[:20],
        "A_chip_pairs": [pair for pair in pairs if pair["first"].startswith("A[chip") and pair["second"].startswith("A[chip")],
        "A_common_links": [pair for pair in pairs if "A_common_mean" in (pair["first"], pair["second"])][:15],
    }


def summarize_group(runs):
    used = min(run["draws"] for run in runs)
    unique_seeds = len({run["seed"] for run in runs})
    independent = unique_seeds == len(runs)
    rhat_available = independent and unique_seeds >= 2 and used >= 4
    elapsed = sum(run["sampling_seconds_including_warmup_and_recovery"] for run in runs)
    total_elapsed = sum(run["total_seconds"] for run in runs) if all(run["total_seconds"] is not None for run in runs) else None
    parameters, pooled = {}, {}
    for name in runs[0]["series"]:
        values = np.stack([run["series"][name][:used] for run in runs])
        pooled[name] = values.ravel()
        constant = np.ptp(values) == 0
        bulk = float(az.ess(values, method="bulk")) if independent and used >= 4 and not constant else None
        tail = float(az.ess(values, method="tail")) if independent and used >= 4 and not constant else None
        rhat = float(az.rhat(values, method="rank")) if rhat_available and not constant else None
        parameters[name] = {
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "quantiles_025_50_975": np.quantile(values, [.025, .5, .975]).tolist(),
            "rank_bulk_ess": bulk, "rank_tail_ess": tail, "independent_chain_rank_rhat": rhat,
            "bulk_ess_per_sampling_second": bulk / elapsed if bulk is not None else None,
            "bulk_ess_per_total_second": bulk / total_elapsed if bulk is not None and total_elapsed else None,
            "constant_series": constant,
            "constant_within_chain_count": int(np.sum(np.ptp(values, axis=1) == 0)),
        }
    limitations = ["Pilot diagnostics do not establish scientific acceptance or synthetic recovery."]
    if not rhat_available:
        limitations.append("Independent-chain R-hat is unavailable: at least two distinct seeds and four draws per chain are required.")
    if not independent:
        limitations.append("Repeated seeds are not independent replicates; pooled ESS and R-hat are suppressed.")
    if used < 100:
        limitations.append("Fewer than 100 saved draws per chain: this is a smoke or short pilot, not an acceptable converged result.")
    if not all(run["contract_verified"] for run in runs):
        limitations.append("Archived observation/operator provenance is missing; controlled target compatibility is unverified.")
    return {
        "variant": runs[0]["variant"], "contract_fingerprint": runs[0]["contract_fingerprint"],
        "contract": runs[0]["contract"], "run_count": len(runs), "unique_seed_count": unique_seeds,
        "unique_initial_state_count": len({run["initial_state_sha256"] for run in runs}),
        "draws_used_per_chain": used,
        "discarded_draws_by_run": {run["directory"]: run["draws"] - used for run in runs},
        "independent_chain_rhat_available": rhat_available, "scientifically_accepted": False,
        "summed_sampling_seconds": elapsed, "summed_total_seconds": total_elapsed,
        "parameters": parameters, "latitude_rings": runs[0]["rings"],
        "pooled_correlations": correlations(pooled), "limitations": limitations,
        "runs": [{**{key: value for key, value in run.items() if key not in ("series", "rings", "contract")},
                  "correlations": correlations(run["series"])} for run in runs],
    }


def analyze_runs(directories):
    runs = [load_run(directory) for directory in directories]
    if len({run["directory"] for run in runs}) != len(runs):
        raise ValueError("A run directory cannot be supplied twice.")
    grouped = {}
    for run in runs:
        grouped.setdefault((run["variant"], run["contract_fingerprint"]), []).append(run)
    groups = [summarize_group(group) for group in grouped.values()]
    contracts = [run["contract"] for run in runs if run["contract_verified"]]
    both_variants = {run["variant"] for run in runs} == {"explicit_A", "marginalized_A"}
    same_target = all(run["contract_verified"] for run in runs) and len({run["contract_fingerprint"] for run in runs}) == 1
    comparison = {"both_variants_present": both_variants, "same_target_verified": same_target,
                  "controlled_comparison_allowed": both_variants and same_target,
                  "identical_sampler_settings": len({json.dumps(run["sampler_arguments"], sort_keys=True) for run in runs}) == 1,
                  "both_variants_have_independent_chain_rhat": both_variants and all(group["independent_chain_rhat_available"] for group in groups)}
    if contracts:
        comparison["identical_data_content_hashes"] = len({contract["data_sha256"] for contract in contracts}) == 1
        comparison["identical_operator_content_hashes"] = len({contract["operator_sha256"] for contract in contracts}) == 1
        comparison["identical_source_snapshots"] = len({json.dumps(contract["source_sha256"], sort_keys=True) for contract in contracts}) == 1
    if both_variants and same_target:
        explicit = next(group for group in groups if group["variant"] == "explicit_A")
        marginal = next(group for group in groups if group["variant"] == "marginalized_A")
        comparison["marginalized_to_explicit_bulk_ess_per_second"] = {
            name: (marginal["parameters"][name]["bulk_ess_per_sampling_second"] / values["bulk_ess_per_sampling_second"])
            if values["bulk_ess_per_sampling_second"] and marginal["parameters"][name]["bulk_ess_per_sampling_second"] is not None else None
            for name, values in explicit["parameters"].items()
        }
    return finite_json({
        "format_version": 1, "analysis": "M9 v3 offline pilot diagnostics", "arviz_version": az.__version__,
        "scientifically_accepted": False,
        "conventions": {
            "ess": "ArviZ rank-normalized bulk and tail ESS, computed from equal-length prefixes.",
            "rhat": "ArviZ rank-normalized folded split R-hat, only with distinct independently seeded chains.",
            "energy": "E-BFMI uses archived total Hamiltonian energy, not potential energy.",
            "cost": "ESS/sec divides by summed measured chain costs, including warmup and conditional-A recovery; total cost also includes setup and metric construction. No cost is removed for discarded draws.",
            "maps": "Equal-area full-sphere longitude means within latitude rings grouped at 1e-6 rad tolerance for float32 angular roundoff; nonaxisymmetric RMS uses the remaining map.",
            "correlations": "Descriptive posterior correlations, not a convergence or identifiability test; pooled values can include between-chain shifts.",
            "acceptance": "All outputs remain pilot evidence. Short chains, successful execution, or high ESS alone do not establish scientific acceptance.",
        },
        "comparison": comparison, "groups": groups,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_runs(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "comparison": result["comparison"]}, allow_nan=False))


if __name__ == "__main__":
    main()
