"""Summarize Milestone 2-3 fixed-ell sensitivity runs."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_milestone2_fixed_ell_sensitivity import ell_tag, parse_ell_values  # noqa: E402


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Summarize fixed-ell Milestone 2-3 sensitivity runs."
    )
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "milestone2_3c"))
    parser.add_argument(
        "--ell-values",
        type=parse_ell_values,
        default=parse_ell_values("0.25,0.30,0.35,0.40,0.50"),
    )
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument("--period-mode", choices=("sampled", "fixed"), default="fixed")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path. Defaults to results-dir/ell_sensitivity_summary.json.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path. Defaults to results-dir/ell_sensitivity_summary.csv.",
    )
    return parser.parse_args()


def _sample_path(results_dir, chip_index, period_mode, ell_b, smoke_test):
    suffix = f"_{ell_tag(ell_b)}_smoke" if smoke_test else f"_{ell_tag(ell_b)}"
    return (
        results_dir
        / f"mcmc_chip{chip_index}_{period_mode}_free_t0_cloud{suffix}.npz"
    )


def _products_dir(results_dir, ell_b):
    return results_dir / ell_tag(ell_b)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q16": float(np.quantile(values, 0.16)),
        "q84": float(np.quantile(values, 0.84)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _load_residual_metrics(products_dir, chip_index):
    residual_path = products_dir / f"residual_chip{chip_index}.npy"
    model_path = products_dir / f"model_spectrum_chip{chip_index}.npy"
    diagnostics_path = products_dir / f"free_t0_cloud_diagnostics_chip{chip_index}.json"
    metrics = {
        "products_available": residual_path.exists(),
        "residual_rms": None,
        "residual_abs_median": None,
        "model_mean": None,
    }
    if residual_path.exists():
        residual = np.load(residual_path)
        metrics["residual_rms"] = float(np.sqrt(np.mean(residual**2)))
        metrics["residual_abs_median"] = float(np.median(np.abs(residual)))
    if model_path.exists():
        model = np.load(model_path)
        metrics["model_mean"] = float(np.mean(model))
    if diagnostics_path.exists():
        with diagnostics_path.open("r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
        for key in (
            "fraction_pixels_below_zero",
            "fraction_pixels_above_one",
            "mean_abs_clipping_shift",
            "max_abs_clipping_shift",
            "cloud_fraction_mean_min",
            "cloud_fraction_mean_max",
        ):
            metrics[key] = diagnostics.get(key)
    return metrics


def _row_from_entry(entry):
    row = {
        "ell_b": entry["ell_b"],
        "sample_path": entry["sample_path"],
        "sample_available": entry["sample_available"],
        "products_available": entry["products"]["products_available"],
        "residual_rms": entry["products"]["residual_rms"],
        "residual_abs_median": entry["products"]["residual_abs_median"],
    }
    for parameter in ("T0", "log_p_cloud", "f_cloud", "sigma_b", "sigma_d", "surface_scale"):
        summary = entry["posterior"].get(parameter)
        if summary is None:
            continue
        for key in ("mean", "std", "median", "q05", "q95"):
            row[f"{parameter}_{key}"] = summary[key]
    for key in (
        "fraction_pixels_below_zero",
        "fraction_pixels_above_one",
        "mean_abs_clipping_shift",
        "max_abs_clipping_shift",
        "cloud_fraction_mean_min",
        "cloud_fraction_mean_max",
    ):
        row[key] = entry["products"].get(key)
    return row


def main():
    """Summarize fixed-ell posterior samples and optional products."""

    args = parse_args()
    results_dir = Path(args.results_dir)
    out_json = Path(args.out_json) if args.out_json else results_dir / "ell_sensitivity_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else results_dir / "ell_sensitivity_summary.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for ell_b in args.ell_values:
        sample_path = _sample_path(
            results_dir,
            args.chip_index,
            args.period_mode,
            ell_b,
            args.smoke_test,
        )
        entry = {
            "ell_b": float(ell_b),
            "ell_b_deg": float(np.degrees(ell_b)),
            "sample_path": str(sample_path),
            "sample_available": sample_path.exists(),
            "posterior": {},
            "products": _load_residual_metrics(_products_dir(results_dir, ell_b), args.chip_index),
        }
        if sample_path.exists():
            samples = np.load(sample_path, allow_pickle=False)
            for parameter in (
                "T0",
                "log_p_cloud",
                "f_cloud",
                "sigma_b",
                "sigma_d",
                "surface_scale",
            ):
                if parameter in samples:
                    entry["posterior"][parameter] = _summary(samples[parameter])
        entries.append(entry)

    payload = {"entries": entries}
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = [_row_from_entry(entry) for entry in entries]
    fieldnames = sorted({key for row in rows for key in row})
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary JSON saved to {out_json}")
    print(f"Summary CSV saved to {out_csv}")


if __name__ == "__main__":
    main()
