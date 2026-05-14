"""Summarize Milestone 2-3d chip-to-chip retrieval products."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_chips(text):
    """Parse comma-separated chip indices."""

    chips = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not chips:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return chips


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Summarize Milestone 2-3d chip-to-chip products."
    )
    parser.add_argument("--chips", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument(
        "--results-template",
        default=str(ROOT / "results" / "milestone2_3d_chip{chip}"),
        help="Directory template containing '{chip}'.",
    )
    parser.add_argument(
        "--samples-template",
        default=None,
        help=(
            "Optional sample path template containing '{chip}'. Defaults to "
            "{results_dir}/mcmc_chip{chip}_fixed_free_t0_cloud.npz."
        ),
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "results" / "milestone2_3d_chip_comparison.json"),
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "results" / "milestone2_3d_chip_comparison.csv"),
    )
    parser.add_argument(
        "--low-lmax-values",
        default="2,3,5",
        help="Comma-separated spherical-harmonic lmax values for low-l map comparison.",
    )
    return parser.parse_args()


def _format_template(template, chip, results_dir=None):
    values = {"chip": chip}
    if results_dir is not None:
        values["results_dir"] = str(results_dir)
    return Path(template.format(**values))


def _sample_path(args, chip, results_dir):
    if args.samples_template is not None:
        return _format_template(args.samples_template, chip, results_dir=results_dir)
    return results_dir / f"mcmc_chip{chip}_fixed_free_t0_cloud.npz"


def _posterior_stats(samples, name):
    if name not in samples:
        return {}
    values = np.asarray(samples[name], dtype=float)
    if values.ndim == 0:
        return {}
    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_median": float(np.median(values)),
        f"{name}_std": float(np.std(values)),
        f"{name}_q05": float(np.quantile(values, 0.05)),
        f"{name}_q95": float(np.quantile(values, 0.95)),
    }


def _safe_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_array(path):
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=float)


def _safe_corr(left, right):
    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.shape != right.shape:
        return None
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _standardize_map(values):
    if values is None:
        return None
    values = np.asarray(values, dtype=float).ravel()
    std = float(np.std(values))
    if values.size < 2 or std == 0.0:
        return None
    return (values - float(np.mean(values))) / std


def _standardized_rmse(left, right):
    left = _standardize_map(left)
    right = _standardize_map(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _sign_agreement(left, right):
    left = _standardize_map(left)
    right = _standardize_map(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.mean(np.sign(left) == np.sign(right)))


def _tail_overlap(left, right, fraction=0.2, low=False):
    left = _standardize_map(left)
    right = _standardize_map(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    count = max(1, int(fraction * left.size))
    if low:
        left_indices = set(np.argsort(left)[:count])
        right_indices = set(np.argsort(right)[:count])
    else:
        left_indices = set(np.argsort(left)[-count:])
        right_indices = set(np.argsort(right)[-count:])
    return float(len(left_indices & right_indices) / count)


def _ensure_numpy_astropy_compat():
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    try:
        import importlib

        function_base = importlib.import_module("numpy.lib._function_base_impl")
    except Exception:
        return
    if hasattr(function_base, "_check_interpolation_as_method"):
        return

    def _check_interpolation_as_method(method, interpolation, fname):
        if interpolation is None:
            return method
        if method != "linear":
            raise TypeError(
                f"{fname} received both method={method!r} and interpolation="
                f"{interpolation!r}."
            )
        return interpolation

    function_base._check_interpolation_as_method = _check_interpolation_as_method


def _low_l_maps(maps, lmax):
    _ensure_numpy_astropy_compat()
    import healpy as hp

    filtered = {}
    for chip, values in maps.items():
        standardized = _standardize_map(values)
        if standardized is None:
            continue
        alm = hp.map2alm(standardized, lmax=lmax, iter=0)
        low_l_map = hp.alm2map(alm, nside=hp.npix2nside(standardized.size), verbose=False)
        filtered[chip] = _standardize_map(low_l_map)
    return filtered


def _entry_for_chip(args, chip):
    results_dir = _format_template(args.results_template, chip)
    sample_path = _sample_path(args, chip, results_dir)
    diagnostics_path = results_dir / f"free_t0_cloud_diagnostics_chip{chip}.json"
    cloud_diag_path = results_dir / f"cloud_fraction_diagnostics_chip{chip}.json"
    residual_path = results_dir / f"residual_chip{chip}.npy"
    cloud_path = results_dir / f"cloud_fraction_mean_chip{chip}.npy"
    contrast_path = results_dir / f"contrast_mean_chip{chip}.npy"
    delta_s_path = results_dir / f"delta_s_mean_chip{chip}.npy"

    entry = {
        "chip": int(chip),
        "results_dir": str(results_dir),
        "sample_path": str(sample_path),
        "sample_available": sample_path.exists(),
        "products_available": residual_path.exists() and cloud_path.exists(),
    }
    if sample_path.exists():
        samples = np.load(sample_path, allow_pickle=False)
        for name in (
            "T0",
            "log_p_cloud",
            "f_cloud",
            "sigma_b",
            "sigma_d",
            "surface_scale",
            "ell_b",
        ):
            entry.update(_posterior_stats(samples, name))
        for name in ("fixed_ell_b", "sigma_b_scale", "nside"):
            if name in samples:
                entry[name] = float(np.asarray(samples[name]))
        if "wavelengths" in samples:
            wavelengths = np.asarray(samples["wavelengths"], dtype=float)
            entry["wavelength_min"] = float(np.min(wavelengths))
            entry["wavelength_max"] = float(np.max(wavelengths))
            entry["n_wavelength"] = int(wavelengths.size)
        if "obs_times" in samples:
            entry["n_phase"] = int(np.asarray(samples["obs_times"]).size)

    diagnostics = _safe_json(diagnostics_path)
    cloud_diagnostics = _safe_json(cloud_diag_path)
    entry.update({f"diagnostics_{key}": value for key, value in diagnostics.items()})
    entry.update(
        {f"cloud_diagnostics_{key}": value for key, value in cloud_diagnostics.items()}
    )

    residual = _safe_array(residual_path)
    if residual is not None:
        entry["residual_rms"] = float(np.sqrt(np.mean(residual**2)))
        entry["residual_abs_median"] = float(np.median(np.abs(residual)))
        entry["residual_abs_p95"] = float(np.quantile(np.abs(residual), 0.95))

    cloud_mean = _safe_array(cloud_path)
    if cloud_mean is not None:
        entry["cloud_fraction_mean_min"] = float(np.min(cloud_mean))
        entry["cloud_fraction_mean_max"] = float(np.max(cloud_mean))
        entry["cloud_fraction_mean_range"] = float(np.max(cloud_mean) - np.min(cloud_mean))
        entry["cloud_fraction_mean_std"] = float(np.std(cloud_mean))

    delta_s_mean = _safe_array(delta_s_path)
    if delta_s_mean is not None:
        entry["delta_s_mean_min"] = float(np.min(delta_s_mean))
        entry["delta_s_mean_max"] = float(np.max(delta_s_mean))
        entry["delta_s_mean_std"] = float(np.std(delta_s_mean))

    return entry, cloud_mean, _safe_array(contrast_path), delta_s_mean


def _pairwise_map_metrics(entries, cloud_maps, contrast_maps, delta_s_maps):
    pairs = []
    for left_index, left in enumerate(entries):
        for right_index in range(left_index + 1, len(entries)):
            right = entries[right_index]
            pairs.append(
                {
                    "chip_left": left["chip"],
                    "chip_right": right["chip"],
                    "cloud_fraction_corr": _safe_corr(
                        cloud_maps.get(left["chip"]),
                        cloud_maps.get(right["chip"]),
                    ),
                    "delta_s_corr": _safe_corr(
                        delta_s_maps.get(left["chip"]),
                        delta_s_maps.get(right["chip"]),
                    ),
                    "contrast_corr": _safe_corr(
                        contrast_maps.get(left["chip"]),
                        contrast_maps.get(right["chip"]),
                    ),
                    "contrast_sign_agreement": _sign_agreement(
                        contrast_maps.get(left["chip"]),
                        contrast_maps.get(right["chip"]),
                    ),
                    "contrast_hot20_overlap": _tail_overlap(
                        contrast_maps.get(left["chip"]),
                        contrast_maps.get(right["chip"]),
                    ),
                    "contrast_cold20_overlap": _tail_overlap(
                        contrast_maps.get(left["chip"]),
                        contrast_maps.get(right["chip"]),
                        low=True,
                    ),
                    "contrast_standardized_rmse": _standardized_rmse(
                        contrast_maps.get(left["chip"]),
                        contrast_maps.get(right["chip"]),
                    ),
                }
            )
    return pairs


def _low_l_pairwise_map_metrics(entries, contrast_maps, lmax_values):
    metrics = []
    for lmax in lmax_values:
        try:
            low_l_maps = _low_l_maps(contrast_maps, lmax=lmax)
        except Exception as exc:
            metrics.append({"lmax": int(lmax), "available": False, "error": str(exc)})
            continue
        for left_index, left in enumerate(entries):
            for right_index in range(left_index + 1, len(entries)):
                right = entries[right_index]
                metrics.append(
                    {
                        "lmax": int(lmax),
                        "available": True,
                        "chip_left": left["chip"],
                        "chip_right": right["chip"],
                        "contrast_corr": _safe_corr(
                            low_l_maps.get(left["chip"]),
                            low_l_maps.get(right["chip"]),
                        ),
                    }
                )
    return metrics


def _write_csv(path, entries):
    keys = sorted({key for entry in entries for key in entry})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def main():
    """Summarize chip-to-chip posterior and product diagnostics."""

    args = parse_args()
    lmax_values = parse_chips(args.low_lmax_values)
    entries = []
    cloud_maps = {}
    contrast_maps = {}
    delta_s_maps = {}
    for chip in args.chips:
        entry, cloud_mean, contrast_mean, delta_s_mean = _entry_for_chip(args, chip)
        entries.append(entry)
        if cloud_mean is not None:
            cloud_maps[chip] = cloud_mean
        if contrast_mean is not None:
            contrast_maps[chip] = contrast_mean
        if delta_s_mean is not None:
            delta_s_maps[chip] = delta_s_mean

    pairs = _pairwise_map_metrics(entries, cloud_maps, contrast_maps, delta_s_maps)
    low_l_pairs = _low_l_pairwise_map_metrics(entries, contrast_maps, lmax_values)
    summary = {
        "entries": entries,
        "pairwise_map_metrics": pairs,
        "low_l_pairwise_map_metrics": low_l_pairs,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(Path(args.out_csv), entries)
    print(f"Chip comparison summary saved to {out_json}")


if __name__ == "__main__":
    main()
