"""Aggregate completed M9/v3 forward checks into an inference gate.

Each listed state must cover chips 0--3. This checks only the supplied states;
passing this gate does not establish posterior coverage or convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHIPS = (0, 1, 2, 3)
MOLECULES = ("CO", "H2O", "CH4", "HF")
GRADIENT_DIRECTIONS = ("T0", "alpha", "log_p_cloud", "map_scale")
MAX_JOINT_KL = 1.0
MAX_ABSOLUTE_DELTA_LOG_LIKELIHOOD = 3.0
GRADIENT_RELATIVE_TOLERANCE = 0.01
GRADIENT_ABSOLUTE_NOISE_TOLERANCE = 0.001


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number {value} in {path}.")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}.")
    return actual


def finite_number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return float(value)


def gradient_pair_passes(row, relative_key, absolute_key):
    relative = finite_number(row[relative_key], relative_key, nonnegative=True)
    absolute = finite_number(row[absolute_key], absolute_key, nonnegative=True)
    return relative <= GRADIENT_RELATIVE_TOLERANCE or absolute <= GRADIENT_ABSOLUTE_NOISE_TOLERANCE


def check_gradients(chip_report, variant):
    """Recheck recorded derivative errors instead of trusting pass booleans."""

    teacher = chip_report.get("teacher_gradients", {})
    candidate = variant.get("gradients", {})
    if not set(GRADIENT_DIRECTIONS).issubset(teacher) or set(teacher) != set(candidate):
        return False
    passed = True
    for name, row in candidate.items():
        for derivative in (teacher[name], row):
            checks = derivative.get("finite_difference_checks", [])
            if not checks:
                passed = False
            for check in checks:
                if finite_number(check["dimensionless_step"], "dimensionless_step") <= 0:
                    raise ValueError("Finite-difference steps must be positive.")
            passed &= any(gradient_pair_passes(check, "derivative_relative_rms_error", "derivative_difference_rms_over_noise") for check in checks)
        passed &= gradient_pair_passes(row, "teacher_derivative_relative_rms_error", "teacher_derivative_difference_rms_over_noise")
    return bool(passed)


def report_contract(report):
    required = {
        "disk_normalization": "A_chip_times_global_mean_uniform_disk_spectrum",
        "local_normalization": "none",
        "interpolation": "fritsch_carlson_monotone_log_slopes_in_inverse_temperature",
        "spectral_interpolation": "existing_linear_interpolation",
        "teacher_profile_kernel": "real_space",
    }
    for name, expected in required.items():
        if report.get(name) != expected:
            raise ValueError(f"The report has an unsupported {name} contract.")
    if type(report.get("x64")) is not bool or report.get("jax_x64") != report["x64"]:
        raise ValueError("The report must consistently identify its x64 setting.")
    commit = report["exojax_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("The report must record a full ExoJAX commit hash.")
    if type(report.get("nx")) is not int or report["nx"] < 2:
        raise ValueError("The report must record a valid spectral grid size.")
    noise_jitter = finite_number(report["noise_jitter"], "noise_jitter", nonnegative=True)
    return {
        **required, "cache_root": str(source_path(report["cache_root"])),
        "exojax_commit": commit, "x64": report["x64"], "nx": report["nx"],
        "noise_jitter": noise_jitter,
    }


def verify_cache(variant, contract, selected_nodes):
    records = variant["cache"]
    if set(records) != set(MOLECULES):
        raise ValueError("Each checked chip must identify all four opacity archives.")
    root = Path(contract["cache_root"])
    metadata_hashes = {}
    for molecule, record in records.items():
        path = source_path(record["path"])
        if not path.is_relative_to(root) or path.name != f"opa{molecule}.zarr":
            raise ValueError(f"Opacity path does not match the declared cache: {path}.")
        if record.get("teacher_profile_kernel") != "real_space":
            raise ValueError(f"Unverified teacher kernel for {path}.")
        metadata_path = path / "zarr.json"
        digest = verify_hash(metadata_path, record["metadata_sha256"])
        metadata = read_json(metadata_path)["attributes"]
        if metadata.get("opa_type") != "OpaDiffgrid" or metadata.get("opa_state", {}).get("teacher_profile_kernel") != "real_space":
            raise ValueError(f"The saved opacity metadata has an unsupported kernel: {metadata_path}.")
        shapes = metadata["array_shapes"]
        if shapes.get("inverse_temperature_grid") != [selected_nodes] or shapes.get("nu_grid") != [contract["nx"]]:
            raise ValueError(f"The saved opacity grids disagree with the report: {metadata_path}.")
        metadata_hashes[str(metadata_path)] = digest
    return metadata_hashes


def aggregate(reports, selected_nodes):
    if not isinstance(selected_nodes, int) or selected_nodes < 2:
        raise ValueError("Select at least two temperature nodes.")
    if not reports:
        raise ValueError("At least one completed report is required.")
    label = f"nodes{selected_nodes}"
    contract = None
    states = {}
    report_hashes, inputstates, source_samples, cache_hashes = {}, {}, {}, {}
    for supplied_path in reports:
        path = Path(supplied_path).resolve()
        if str(path) in report_hashes:
            raise ValueError(f"Duplicate input report: {path}.")
        report = read_json(path)
        if report.get("mode") != "m9_v3_monotone_real_space_diffgrid_validation" or report.get("status") != "complete":
            raise ValueError(f"Only completed M9/v3 validation reports are accepted: {path}.")
        current_contract = report_contract(report)
        if contract is None:
            contract = current_contract
        elif contract != current_contract:
            raise ValueError(f"Forward contracts differ across reports: {path}.")
        if selected_nodes not in report.get("nodes", []):
            raise ValueError(f"The selected node count was not evaluated in {path}.")
        if not report.get("state_file"):
            raise ValueError(f"A physical state is required; cache-only reports are insufficient: {path}.")
        state_path = source_path(report["state_file"])
        state_hash = verify_hash(state_path, report["state_sha256"])
        samples_path = source_path(report["samples"])
        samples_hash = verify_hash(samples_path, report["samples_sha256"])
        inputstates[str(state_path)] = state_hash
        source_samples[str(samples_path)] = samples_hash
        report_hashes[str(path)] = sha256(path)
        chip_indices = report.get("chip_indices", [])
        if not chip_indices or any(type(chip) is not int or chip not in CHIPS for chip in chip_indices) or len(set(chip_indices)) != len(chip_indices):
            raise ValueError(f"Invalid chip indices in {path}.")
        if set(report["chips"]) != {str(chip) for chip in chip_indices}:
            raise ValueError(f"Report chip contents do not match its declared chip indices: {path}.")
        state = states.setdefault(state_hash, {
            "state_file": str(state_path), "state_sha256": state_hash,
            "sample_index": report.get("sample_index"), "chips": {},
        })
        if state["sample_index"] != report.get("sample_index"):
            raise ValueError(f"The same state is assigned conflicting sample indices: {state_path}.")
        for chip_index in chip_indices:
            key = str(chip_index)
            if key in state["chips"]:
                raise ValueError(f"Duplicate chip {chip_index} for state {state_path}; refusing double counting.")
            chip = report["chips"][key]
            variant = chip["variants"][label]
            if "disk" not in variant:
                raise ValueError(f"Missing final disk comparison for chip {chip_index}: {path}.")
            metadata = verify_cache(variant, contract, selected_nodes)
            cache_hashes.update(metadata)
            disk = variant["disk"]
            state["chips"][key] = {
                "report": str(path), "report_sha256": report_hashes[str(path)],
                "gaussian_mean_kl": finite_number(disk["gaussian_mean_kl"], "gaussian_mean_kl", nonnegative=True),
                "delta_log_likelihood": finite_number(disk["delta_log_likelihood"], "delta_log_likelihood"),
                "gradients_passed": check_gradients(chip, variant),
                "cache_metadata_sha256": metadata,
            }
    for state in states.values():
        chips = state["chips"]
        state["chip_indices"] = sorted(int(chip) for chip in chips)
        state["missing_chip_indices"] = sorted(set(CHIPS) - set(state["chip_indices"]))
        state["joint_gaussian_mean_kl"] = math.fsum(chip["gaussian_mean_kl"] for chip in chips.values())
        state["joint_delta_log_likelihood"] = math.fsum(chip["delta_log_likelihood"] for chip in chips.values())
        state["gradients_passed"] = all(chip["gradients_passed"] for chip in chips.values())
        reasons = []
        if state["missing_chip_indices"]:
            reasons.append("Missing requested chips.")
        if state["joint_gaussian_mean_kl"] > MAX_JOINT_KL:
            reasons.append("Joint Gaussian mean KL exceeds 1.")
        if abs(state["joint_delta_log_likelihood"]) > MAX_ABSOLUTE_DELTA_LOG_LIKELIHOOD:
            reasons.append("Absolute joint delta log likelihood exceeds 3.")
        if not state["gradients_passed"]:
            reasons.append("Required teacher and DiffGrid gradient checks are missing or failed.")
        state["reasons"] = reasons
        state["passed"] = not reasons
    return {
        "mode": "m9_v3_aggregate_forward_validation", "status": "complete",
        "passed": all(state["passed"] for state in states.values()),
        **contract, "selected_nodes": selected_nodes, "chip_indices": list(CHIPS),
        "report_sha256": report_hashes, "inputstates": inputstates,
        "source_samples_sha256": source_samples, "cache_metadata_sha256": cache_hashes,
        "states": list(states.values()),
        "required_gradient_directions": list(GRADIENT_DIRECTIONS),
        "thresholds": {"max_joint_kl": MAX_JOINT_KL, "max_absolute_delta_log_likelihood": MAX_ABSOLUTE_DELTA_LOG_LIKELIHOOD, "gradient_relative_tolerance": GRADIENT_RELATIVE_TOLERANCE, "gradient_absolute_noise_tolerance": GRADIENT_ABSOLUTE_NOISE_TOLERANCE},
        "scope": "Numerical validation at the listed physical states only. Additional support checks and posterior diagnostics remain separate requirements.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--selected-nodes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Use a new output path: {args.out}.")
    result = aggregate(args.reports, args.selected_nodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Forward gate passed={result['passed']}: {args.out}", flush=True)


if __name__ == "__main__":
    main()
