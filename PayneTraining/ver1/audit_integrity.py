"""Read-only CPU audit of the version 1 training arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
NAMES = (
    "T0", "alpha", "log_vmr_co", "log_vmr_h2o", "log_vmr_ch4",
    "log_vmr_hf", "log_p_cloud", "mu",
)
LOWER = np.array([1000., .05, -6., -6., -6., -10., -3., 0.])
UPPER = np.array([1700., .20, -1., -1., -1., -5., 5., 1.])
EDGES = np.array([
    [22866.027997218360, 23008.110368176436],
    [23028.459924926683, 23165.628572390491],
    [23183.280818327999, 23315.944332905266],
    [23331.697042423235, 23459.163543595641],
])


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(directory):
    started = time.monotonic()
    paths = [directory / f"{name}.npy" for name in ("parameters", "spectra", "wavelength")]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    parameters, spectra, wavelength = (
        np.load(path, allow_pickle=False, mmap_mode="r") for path in paths
    )
    assert parameters.shape == (100000, 8), parameters.shape
    assert spectra.shape == (100000, 18000), spectra.shape
    assert wavelength.shape == (18000,), wavelength.shape
    assert all(a.dtype == np.float64 for a in (parameters, spectra, wavelength))
    files = {
        path.name: {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "file_bytes": path.stat().st_size, "array_bytes": array.nbytes,
        }
        for path, array in zip(paths, (parameters, spectra, wavelength))
    }
    unit = np.random.default_rng(0).random(parameters.shape)
    unit[:, 7] = 1. - unit[:, 7]
    expected = LOWER + unit * (UPPER - LOWER)
    seed_exact = np.array_equal(parameters.view(np.uint64), expected.view(np.uint64))
    normalized = (parameters - LOWER) / (UPPER - LOWER)
    parameter_finite = bool(np.all(np.isfinite(parameters)))
    in_bounds = bool(np.all(parameters >= LOWER) and np.all(parameters <= UPPER)
                     and np.all(parameters[:, 7] > 0))
    unique_parameters = len(np.unique(parameters, axis=0))
    # Treat the eight KS tests and 28 pairwise correlations as one family.
    test_count = 36
    parameter_checks = []
    for column, name in enumerate(NAMES):
        ks = stats.kstest(normalized[:, column], "uniform")
        parameter_checks.append({
            "name": name, "minimum": float(parameters[:, column].min()),
            "maximum": float(parameters[:, column].max()),
            "normalized_mean": float(normalized[:, column].mean()),
            "ks_statistic": float(ks.statistic), "ks_p_value": float(ks.pvalue),
            "ks_p_bonferroni_36": min(1., test_count * float(ks.pvalue)),
        })
    correlations = []
    for i in range(8):
        for j in range(i + 1, 8):
            result = stats.pearsonr(normalized[:, i], normalized[:, j])
            correlations.append({
                "parameters": [NAMES[i], NAMES[j]], "r": float(result.statistic),
                "p_value": float(result.pvalue),
                "p_bonferroni_36": min(1., test_count * float(result.pvalue)),
            })

    nonfinite = nonpositive = constant = duplicates = hash_collisions = 0
    duplicate_examples = []
    seen = {}
    chip_min = np.full(4, np.inf)
    chip_max = np.full(4, -np.inf)
    constant_chip = np.zeros(4, dtype=np.int64)
    for start in range(0, len(spectra), 256):
        block = spectra[start:start + 256]
        nonfinite += int(np.count_nonzero(~np.isfinite(block)))
        nonpositive += int(np.count_nonzero(block <= 0))
        constant += int(np.count_nonzero(np.ptp(block, axis=1) == 0))
        for chip in range(4):
            section = block[:, chip * 4500:(chip + 1) * 4500]
            chip_min[chip] = min(chip_min[chip], float(section.min()))
            chip_max[chip] = max(chip_max[chip], float(section.max()))
            constant_chip[chip] += np.count_nonzero(np.ptp(section, axis=1) == 0)
        for offset, row in enumerate(block):
            digest = hashlib.sha256(memoryview(row)).digest()
            index = start + offset
            if digest in seen:
                previous = seen[digest]
                if np.array_equal(row.view(np.uint64), spectra[previous].view(np.uint64)):
                    duplicates += 1
                    if len(duplicate_examples) < 10:
                        duplicate_examples.append([previous, index])
                else:
                    hash_collisions += 1
            else:
                seen[digest] = index

    wavelength_finite = bool(np.all(np.isfinite(wavelength)))
    wavelength_increasing = bool(np.all(wavelength > 0) and np.all(np.diff(wavelength) > 0))
    wave_checks = []
    for chip in range(4):
        wave = wavelength[chip * 4500:(chip + 1) * 4500]
        steps = np.diff(np.log(wave))
        relative_spacing_error = float(np.max(np.abs(steps / steps.mean() - 1.)))
        edge_error = float(np.max(np.abs(wave[[0, -1]] - EDGES[chip])))
        wave_checks.append({
            "chip": chip, "minimum_angstrom": float(wave[0]),
            "maximum_angstrom": float(wave[-1]),
            "endpoint_max_error_angstrom": edge_error,
            "log_spacing_max_relative_error": relative_spacing_error,
        })
    for path in paths:
        files[path.name]["sha256"] = file_hash(path)
    unchanged = before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    checks = {
        "arrays_unchanged_during_audit": unchanged,
        "parameters_finite": parameter_finite, "parameters_in_bounds": in_bounds,
        "parameters_bitwise_equal_seed_0": seed_exact,
        "no_duplicate_parameter_rows": unique_parameters == len(parameters),
        "all_spectra_finite_and_positive": nonfinite == 0 and nonpositive == 0,
        "no_constant_spectrum_rows": constant == 0,
        "no_duplicate_spectrum_rows": duplicates == 0 and hash_collisions == 0,
        "wavelength_finite_positive_increasing": wavelength_finite and wavelength_increasing,
        "wavelength_matches_expected_grid": all(
            c["endpoint_max_error_angstrom"] < 1e-10
            and c["log_spacing_max_relative_error"] < 5e-9 for c in wave_checks
        ),
    }
    return {
        "integrity_pass": all(checks.values()), "checks": checks, "files": files,
        "total_file_bytes": sum(f["file_bytes"] for f in files.values()),
        "parameters": {
            "duplicate_rows": len(parameters) - unique_parameters, "columns": parameter_checks,
            "pairwise_correlations": correlations,
            "statistical_test_note": (
                "KS uniformity and Pearson correlation are descriptive checks of a fixed seeded draw, "
                "not proofs of uniformity or independence. Bonferroni-adjusted p-values account for "
                "all 36 tests; compare adjusted values with 0.05. Exact RNG reproduction is checked separately."
            ),
        },
        "spectra": {
            "values_scanned": spectra.size, "nonfinite_values": nonfinite,
            "nonpositive_values": nonpositive, "constant_rows": constant,
            "duplicate_rows": duplicates, "duplicate_examples": duplicate_examples,
            "row_hash_collisions": hash_collisions,
            "chips": [{"chip": chip, "minimum": float(chip_min[chip]),
                       "maximum": float(chip_max[chip]), "constant_rows": int(constant_chip[chip])}
                      for chip in range(4)],
        },
        "wavelength": {
            "chips": wave_checks,
            "gaps_angstrom": [float(wavelength[i] - wavelength[i - 1]) for i in (4500, 9000, 13500)],
        },
        "seconds": time.monotonic() - started,
        "limitations": "This audit checks stored arrays; physical accuracy and label-to-spectrum correspondence require RT replay.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE / "audit_integrity.json")
    args = parser.parse_args()
    report = audit(args.data)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"integrity_pass": report["integrity_pass"], "seconds": report["seconds"],
                      "report": str(args.out)}, indent=2))
