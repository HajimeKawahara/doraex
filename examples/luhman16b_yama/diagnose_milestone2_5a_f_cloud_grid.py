"""Diagnose M2-5a mean-spectrum placement on the clear/cloudy segment."""

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

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402
from doraex.spectra.exojax_forward import load_t0_vmr_cloud_profile_grid  # noqa: E402


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fast M2-5a f_cloud diagnostic over T0/log10 Pc/zeta_vmr grids."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument(
        "--profile-grid-template",
        default=str(
            ROOT
            / "data"
            / "milestone2_t0_vmr_cloud_grid_profiles_exomol_chip{chip}.npz"
        ),
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "results" / "milestone2_5a" / "f_cloud_grid_diagnostic.json"),
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "results" / "milestone2_5a" / "f_cloud_grid_diagnostic.csv"),
    )
    parser.add_argument("--f-min", type=float, default=-0.25)
    parser.add_argument("--f-max", type=float, default=1.25)
    parser.add_argument("--f-count", type=int, default=151)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    return parser.parse_args()


def _best_f_for_profiles(observed_mean, clear_profile, cloudy_profile, f_grid):
    delta_profile = cloudy_profile - clear_profile
    best = None
    for f_cloud in f_grid:
        mixed = clear_profile + f_cloud * delta_profile
        normalized = mixed / np.mean(mixed)
        inv_a = float(np.dot(normalized, observed_mean) / np.dot(normalized, normalized))
        model = inv_a * normalized
        loss = float(np.mean((observed_mean - model) ** 2))
        row = {
            "f_cloud": float(f_cloud),
            "A": float(1.0 / inv_a),
            "loss": loss,
        }
        if best is None or loss < best["loss"]:
            best = row
    return best


def _chip_rows(chip_index, args, f_grid):
    chip_data = load_luhman16b_chip(args.data_dir, chip_index=chip_index)
    if args.smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=args.smoke_wavelength_step,
            phase_count=args.smoke_phase_count,
        )
    profile_grid_path = args.profile_grid_template.format(chip=chip_index)
    (
        t0_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        clear_profile_grid,
        cloudy_profile_grid,
    ) = load_t0_vmr_cloud_profile_grid(
        profile_grid_path,
        expected_wavelengths=chip_data.wavelengths,
    )
    observed_mean = np.mean(np.asarray(chip_data.flux), axis=0)
    rows = []
    for t0_index, t0 in enumerate(t0_grid):
        for log_p_index, log_p_cloud in enumerate(log_p_cloud_grid):
            for zeta_index, zeta_vmr in enumerate(zeta_vmr_grid):
                best = _best_f_for_profiles(
                    observed_mean,
                    clear_profile_grid[t0_index, zeta_index],
                    cloudy_profile_grid[t0_index, log_p_index, zeta_index],
                    f_grid,
                )
                best.update(
                    {
                        "chip_index": int(chip_index),
                        "T0": float(t0),
                        "log_p_cloud": float(log_p_cloud),
                        "zeta_vmr": float(zeta_vmr),
                    }
                )
                rows.append(best)
    rows.sort(key=lambda row: row["loss"])
    return rows


def main():
    """Run the diagnostic and write compact JSON/CSV summaries."""

    args = parse_args()
    f_grid = np.linspace(args.f_min, args.f_max, args.f_count)
    per_chip = {}
    all_rows = []
    for chip_index in args.chip_indices:
        rows = _chip_rows(chip_index, args, f_grid)
        per_chip[str(chip_index)] = rows[: args.top_k]
        all_rows.extend(rows[: args.top_k])

    summary = {
        "chip_indices": args.chip_indices,
        "f_grid_min": float(f_grid[0]),
        "f_grid_max": float(f_grid[-1]),
        "f_grid_count": int(len(f_grid)),
        "top_k": int(args.top_k),
        "best_by_chip": {chip: rows[0] for chip, rows in per_chip.items()},
        "top_by_chip": per_chip,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["chip_index", "loss", "f_cloud", "A", "T0", "log_p_cloud", "zeta_vmr"]
    with out_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_rows, key=lambda item: (item["chip_index"], item["loss"])):
            writer.writerow({name: row[name] for name in fieldnames})
    print(f"Diagnostic saved to {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
