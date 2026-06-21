"""Primary product generation for Doraex post-processing runs.

The primary product set is the small, stable collection of figures that should
exist whenever a run is considered to have completed product generation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PrimaryProductDefinition:
    """Definition of one primary product path."""

    name: str
    relative_path: str
    description: str


@dataclass(frozen=True)
class SingleLineSpec:
    """Single-line pressure response input definition."""

    chip_index: int
    molecule: str
    line_center: float
    label: str

    @property
    def stem(self) -> str:
        wavelength = f"{self.line_center:.2f}".replace(".", "p")
        return f"single_line_rt_prediction_chip{self.chip_index}_lambda{wavelength}"


@dataclass(frozen=True)
class PrimaryProductConfig:
    """Configuration for generating the default primary products."""

    samples_path: Path
    product_dir: Path
    prefix: str = "m6"
    project_root: Path = Path(".")
    python_executable: str = sys.executable
    chip_indices: tuple[int, ...] = (0, 1, 2, 3)
    run_map_products: bool = True
    run_line_stack: bool = True
    run_single_line_products: bool = True
    run_corner_products: bool = True
    run_log_w_products: bool = True
    x64: bool = True
    primary_dir: Path | None = None


@dataclass(frozen=True)
class PrimaryProductResult:
    """Generated primary products and any missing required paths."""

    paths: Mapping[str, Path]
    missing: tuple[Path, ...]
    manifest_path: Path
    bundle_dir: Path | None = None


DEFAULT_SINGLE_LINE_SPECS = (
    SingleLineSpec(2, "CO", 23220.85, "CO 23220.85 A line"),
    SingleLineSpec(3, "CO", 23427.59, "CO 23427.59 A line"),
)

DEFAULT_ATMOSPHERE_CORNER_KEYS = (
    "T0",
    "alpha",
    "log_p_cloud",
    "sigma_log_p",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
)

DEFAULT_ATMOSPHERE_SUMMARY_KEYS = (
    "T0",
    "alpha",
    "logg",
    "log_p_cloud",
    "sigma_log_p",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
)

DEFAULT_ATMOSPHERE_CORNER_LABELS = {
    "T0": r"$T_0$",
    "alpha": r"$\alpha$",
    "log_p_cloud": r"$\log_{10} p_\mathrm{cloud}$",
    "sigma_log_p": r"$\sigma_{\log p}$",
    "log_vmr_co": r"$\log_{10}\mathrm{VMR}_{CO}$",
    "log_vmr_h2o": r"$\log_{10}\mathrm{VMR}_{H_2O}$",
    "log_vmr_ch4": r"$\log_{10}\mathrm{VMR}_{CH_4}$",
    "log_vmr_hf": r"$\log_{10}\mathrm{VMR}_{HF}$",
}

DEFAULT_PRIMARY_PRODUCT_DEFINITIONS = (
    PrimaryProductDefinition(
        "cloud_pressure_map",
        "figure8_p_cloud_joint.png",
        "64-bit cloud-pressure joint map product.",
    ),
    PrimaryProductDefinition(
        "corner_atm",
        "corner_atm.png",
        "Atmospheric-parameter posterior corner plot.",
    ),
    PrimaryProductDefinition(
        "corner_nuisance",
        "corner_nuisance.png",
        "Nuisance-parameter posterior corner plot.",
    ),
    PrimaryProductDefinition(
        "log_w_summary",
        "log_w_summary.png",
        "Chip/exposure log-weight posterior summary.",
    ),
    PrimaryProductDefinition(
        "log_w_corr_ellipse",
        "log_w_corr_ellipse.png",
        "Chip/exposure log-weight posterior correlation ellipse plot.",
    ),
    PrimaryProductDefinition(
        "mean_subtracted_line_stack",
        "products/{prefix}_mean_subtracted_line_stack_summary.png",
        "M5-style mean-subtracted line-stack summary.",
    ),
    PrimaryProductDefinition(
        "single_line_pressure_response",
        "products/{prefix}_single_line_pressure_response_comparison.png",
        "Single-line pressure-response comparison.",
    ),
)


def primary_product_paths(
    product_dir: Path | str,
    *,
    prefix: str = "m6",
    chip_indices: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, Path]:
    """Return the expected primary product paths for a product directory."""

    root = Path(product_dir)
    paths = {
        definition.name: root / definition.relative_path.format(prefix=prefix)
        for definition in primary_product_definitions(chip_indices=chip_indices)
    }
    return paths


def primary_product_definitions(
    *, chip_indices: Sequence[int] = (0, 1, 2, 3)
) -> tuple[PrimaryProductDefinition, ...]:
    """Return the full primary product definition set."""

    joint_chip_definitions = tuple(
        PrimaryProductDefinition(
            f"joint_chip_{chip_index}",
            f"figure9_joint_chip{chip_index}.png",
            f"64-bit joint chip model/residual product for chip {chip_index}.",
        )
        for chip_index in chip_indices
    )
    return DEFAULT_PRIMARY_PRODUCT_DEFINITIONS + joint_chip_definitions


def generate_primary_products(config: PrimaryProductConfig) -> PrimaryProductResult:
    """Generate all enabled primary products for one run."""

    samples_path = Path(config.samples_path)
    product_dir = Path(config.product_dir)
    project_root = Path(config.project_root)
    products_dir = product_dir / "products"
    products_dir.mkdir(parents=True, exist_ok=True)

    env = _postprocess_environment(project_root)
    if config.run_map_products:
        _run_map_products(config, env=env)
    if config.run_line_stack:
        _run_line_stack(config, env=env)
    if config.run_single_line_products:
        _run_single_line_products(config, env=env)
    if config.run_corner_products:
        generate_corner_products(samples_path, product_dir)
    if config.run_log_w_products:
        generate_log_w_products(samples_path, product_dir)

    paths = primary_product_paths(
        product_dir, prefix=config.prefix, chip_indices=config.chip_indices
    )
    missing = tuple(path for path in paths.values() if not path.exists())
    manifest_path = write_primary_product_manifest(
        product_dir,
        paths=paths,
        missing=missing,
        prefix=config.prefix,
        samples_path=samples_path,
    )
    bundle_dir = None
    if config.primary_dir is not None:
        bundle_dir = collect_primary_products(
            samples_path,
            product_dir,
            config.primary_dir,
            prefix=config.prefix,
            chip_indices=config.chip_indices,
        )
    return PrimaryProductResult(
        paths=paths,
        missing=missing,
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
    )


def collect_primary_products(
    samples_path: Path | str,
    product_dir: Path | str,
    primary_dir: Path | str,
    *,
    prefix: str = "m6",
    chip_indices: Sequence[int] = (0, 1, 2, 3),
) -> Path:
    """Collect primary figures and paper-facing summary values in one directory."""

    samples_path = Path(samples_path)
    product_dir = Path(product_dir)
    primary_dir = Path(primary_dir)
    info_dir = primary_dir / "info"
    primary_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    source_paths = primary_product_paths(
        product_dir,
        prefix=prefix,
        chip_indices=chip_indices,
    )
    missing = [path for path in source_paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing primary products:\n{missing_text}")

    figure_entries = []
    for name, source in sorted(source_paths.items()):
        destination = primary_dir / source.name
        shutil.copy2(source, destination)
        figure_entries.append(
            {
                "name": name,
                "source": str(source),
                "path": str(destination.relative_to(primary_dir)),
            }
        )

    summary_payload = write_primary_info_products(
        samples_path,
        product_dir,
        info_dir,
        prefix=prefix,
        chip_indices=chip_indices,
    )
    manifest = {
        "product_set": "primary_bundle",
        "prefix": prefix,
        "samples": str(samples_path),
        "source_product_dir": str(product_dir),
        "primary_dir": str(primary_dir),
        "figures": figure_entries,
        "info": summary_payload["info_files"],
    }
    manifest_path = primary_dir / "primary_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_primary_bundle_readme(primary_dir, manifest, summary_payload)
    return primary_dir


def write_primary_info_products(
    samples_path: Path,
    product_dir: Path,
    info_dir: Path,
    *,
    prefix: str,
    chip_indices: Sequence[int],
) -> dict[str, object]:
    """Write summary tables and JSON values needed alongside primary figures."""

    samples = np.load(samples_path, allow_pickle=True)
    diagnostics = _read_json_if_exists(samples_path.parent / "diagnostics.json")
    joint_diagnostics = _read_json_if_exists(product_dir / "joint_chip_diagnostics.json")
    product_summary = _read_json_if_exists(product_dir / "on_the_fly_product_summary.json")
    line_summary = _read_json_if_exists(product_dir / "products" / "minimum_window_summary.json")
    single_line_summary = _read_json_if_exists(
        product_dir / "products" / f"{prefix}_single_line_pressure_response_comparison.json"
    )

    info_files = []

    run_summary = _build_run_summary(samples, diagnostics, product_summary)
    info_files.append(_write_json(info_dir / "run_summary.json", run_summary))

    atmosphere_rows, atmosphere_json = _posterior_summary_rows(
        samples,
        DEFAULT_ATMOSPHERE_SUMMARY_KEYS,
        include_derived_cloud_pressure=True,
    )
    info_files.append(_write_json(info_dir / "posterior_atmosphere.json", atmosphere_json))
    info_files.append(
        _write_csv(
            info_dir / "posterior_atmosphere.csv",
            atmosphere_rows,
            fieldnames=[
                "name",
                "n",
                "median",
                "q16",
                "q84",
                "minus",
                "plus",
                "q05",
                "q95",
                "mean",
                "std",
            ],
        )
    )

    nuisance_payload = _build_nuisance_summary(samples, chip_indices)
    info_files.append(_write_json(info_dir / "posterior_nuisance.json", nuisance_payload))
    info_files.append(
        _write_csv(
            info_dir / "posterior_nuisance_by_chip.csv",
            nuisance_payload["by_chip"],
            fieldnames=[
                "name",
                "chip_index",
                "n",
                "median",
                "q16",
                "q84",
                "minus",
                "plus",
                "q05",
                "q95",
                "mean",
                "std",
            ],
        )
    )
    if nuisance_payload["log_w_by_phase"]:
        info_files.append(
            _write_csv(
                info_dir / "posterior_log_w_by_phase.csv",
                nuisance_payload["log_w_by_phase"],
                fieldnames=[
                    "chip_index",
                    "phase_index",
                    "n",
                    "median",
                    "q16",
                    "q84",
                    "minus",
                    "plus",
                    "q05",
                    "q95",
                    "mean",
                    "std",
                ],
            )
        )

    cloud_payload = _build_cloud_map_summary(product_dir, chip_indices)
    info_files.append(_write_json(info_dir / "cloud_map_summary.json", cloud_payload))
    info_files.append(
        _write_csv(
            info_dir / "cloud_map_summary.csv",
            cloud_payload["rows"],
            fieldnames=[
                "name",
                "chip_index",
                "n",
                "min",
                "q05",
                "q16",
                "median",
                "q84",
                "q95",
                "max",
                "mean",
                "std",
            ],
        )
    )

    model_fit_summary = _build_model_fit_summary(joint_diagnostics)
    info_files.append(_write_json(info_dir / "model_fit_summary.json", model_fit_summary))

    line_payload = _build_line_stack_info(line_summary)
    info_files.append(_write_json(info_dir / "line_stack_summary.json", line_payload))
    if line_payload["selected_lines"]:
        info_files.append(
            _write_csv(
                info_dir / "line_stack_selected_lines.csv",
                line_payload["selected_lines"],
                fieldnames=[
                    "chip_index",
                    "molecule",
                    "center_wavelength",
                    "rest_frame_center_wavelength",
                    "line_strength",
                    "radial_velocity_kms",
                ],
            )
        )

    single_line_payload = _build_single_line_info(product_dir, single_line_summary)
    info_files.append(
        _write_json(info_dir / "single_line_pressure_response_summary.json", single_line_payload)
    )
    if single_line_payload["lines"]:
        info_files.append(
            _write_csv(
                info_dir / "single_line_pressure_response_summary.csv",
                single_line_payload["lines"],
                fieldnames=[
                    "label",
                    "path",
                    "continuum",
                    "line_depth",
                    "molecule",
                    "requested_center_wavelength",
                    "selected_center_wavelength",
                    "observed_frame_center_wavelength",
                    "line_strength",
                    "line_strength_temperature",
                    "radial_velocity_kms",
                ],
            )
        )

    source_payload = {
        "samples": str(samples_path),
        "product_dir": str(product_dir),
        "diagnostics": str(samples_path.parent / "diagnostics.json"),
        "joint_chip_diagnostics": str(product_dir / "joint_chip_diagnostics.json"),
        "on_the_fly_product_summary": str(product_dir / "on_the_fly_product_summary.json"),
        "minimum_window_summary": str(product_dir / "products" / "minimum_window_summary.json"),
        "single_line_comparison": str(
            product_dir / "products" / f"{prefix}_single_line_pressure_response_comparison.json"
        ),
    }
    info_files.append(_write_json(info_dir / "source_paths.json", source_payload))

    return {
        "info_files": [
            str(path.relative_to(info_dir.parent)) for path in info_files
        ],
        "run_summary": run_summary,
        "atmosphere": atmosphere_json,
        "cloud_map": cloud_payload,
        "line_stack": line_payload,
        "single_line": single_line_payload,
    }


def _build_run_summary(
    samples: np.lib.npyio.NpzFile,
    diagnostics: Mapping[str, object],
    product_summary: Mapping[str, object],
) -> dict[str, object]:
    sampled_parameter_count = _count_sampled_parameters(samples)
    keys = [
        "mode",
        "chip_indices",
        "full_data",
        "n_chip",
        "n_phase",
        "n_wavelength",
        "nside",
        "num_warmup",
        "num_samples",
        "num_chains",
        "target_accept_prob",
        "max_tree_depth",
        "dense_mass",
        "fix_nuisance",
        "fix_logg",
        "fixed_logg",
        "fixed_ell_b",
        "x64",
        "divergence_count",
        "mean_accept_prob",
        "max_num_steps",
        "run_seconds",
        "setup_seconds",
    ]
    payload = {key: diagnostics.get(key) for key in keys if key in diagnostics}
    payload["sampled_parameter_count"] = sampled_parameter_count
    payload["samples_recorded"] = int(_sample_count(samples))
    if product_summary:
        payload["product_reconstruction"] = {
            key: product_summary.get(key)
            for key in [
                "nside",
                "map_sample_count",
                "pressure_derivative_method",
                "override_ell_b",
                "setup_seconds",
            ]
            if key in product_summary
        }
    return payload


def _posterior_summary_rows(
    samples: np.lib.npyio.NpzFile,
    parameter_names: Sequence[str],
    *,
    include_derived_cloud_pressure: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for name in parameter_names:
        if name not in samples.files:
            continue
        values = np.asarray(samples[name], dtype=float).reshape(-1)
        rows.append({"name": name, **_posterior_stats(values)})
    if include_derived_cloud_pressure and "log_p_cloud" in samples.files:
        values = 10.0 ** np.asarray(samples["log_p_cloud"], dtype=float).reshape(-1)
        rows.append({"name": "p_cloud_bar", **_posterior_stats(values)})
    return rows, {"parameters": rows}


def _build_nuisance_summary(
    samples: np.lib.npyio.NpzFile,
    chip_indices: Sequence[int],
) -> dict[str, object]:
    by_chip = []
    if "A" in samples.files:
        values = np.asarray(samples["A"], dtype=float)
        for position, chip_index in enumerate(chip_indices):
            by_chip.append(
                {
                    "name": "A",
                    "chip_index": int(chip_index),
                    **_posterior_stats(values[:, position]),
                }
            )
    if "sigma_d" in samples.files:
        values = np.asarray(samples["sigma_d"], dtype=float)
        for position, chip_index in enumerate(chip_indices):
            by_chip.append(
                {
                    "name": "sigma_d",
                    "chip_index": int(chip_index),
                    **_posterior_stats(values[:, position]),
                }
            )
    log_w_by_phase = []
    log_w_by_chip = []
    if "log_w" in samples.files:
        values = np.asarray(samples["log_w"], dtype=float)
        for position, chip_index in enumerate(chip_indices):
            chip_values = values[:, position, :]
            log_w_by_chip.append(
                {
                    "name": "log_w",
                    "chip_index": int(chip_index),
                    **_posterior_stats(chip_values.reshape(-1)),
                }
            )
            for phase_index in range(chip_values.shape[1]):
                log_w_by_phase.append(
                    {
                        "chip_index": int(chip_index),
                        "phase_index": int(phase_index),
                        **_posterior_stats(chip_values[:, phase_index]),
                    }
                )
    return {
        "by_chip": by_chip + log_w_by_chip,
        "log_w_by_phase": log_w_by_phase,
    }


def _build_cloud_map_summary(
    product_dir: Path,
    chip_indices: Sequence[int],
) -> dict[str, object]:
    rows = []
    array_specs = [
        ("p_cloud_mean_bar", "p_cloud_mean_joint_by_chip.npy"),
        ("p_cloud_std_bar", "p_cloud_std_joint_by_chip.npy"),
        ("log_p_cloud_mean", "log_p_cloud_mean_joint_by_chip.npy"),
        ("log_p_cloud_var", "log_p_cloud_var_joint_by_chip.npy"),
        ("contrast_mean", "contrast_mean_joint.npy"),
        ("contrast_var", "contrast_var_joint.npy"),
    ]
    for name, filename in array_specs:
        path = product_dir / filename
        if not path.exists():
            continue
        array = np.asarray(np.load(path), dtype=float)
        if array.ndim == 2 and array.shape[0] == len(chip_indices):
            rows.append({"name": name, "chip_index": "all", **_array_stats(array)})
            for position, chip_index in enumerate(chip_indices):
                rows.append(
                    {
                        "name": name,
                        "chip_index": int(chip_index),
                        **_array_stats(array[position]),
                    }
                )
        else:
            rows.append({"name": name, "chip_index": "joint", **_array_stats(array)})
    return {"rows": rows}


def _build_model_fit_summary(joint_diagnostics: Mapping[str, object]) -> dict[str, object]:
    if not joint_diagnostics:
        return {}
    keys = [
        "chip_indices",
        "residual_rms_by_chip",
        "residual_abs_median_by_chip",
        "contrast_mean_min",
        "contrast_mean_max",
        "contrast_mean_std",
        "corr_T0_alpha",
        "corr_T0_log_p_cloud",
    ]
    return {key: joint_diagnostics.get(key) for key in keys if key in joint_diagnostics}


def _build_line_stack_info(line_summary: Mapping[str, object]) -> dict[str, object]:
    if not line_summary:
        return {"selected_lines": [], "total_selected_line_count": 0}
    selected_lines = []
    molecule_counts: dict[str, int] = {}
    for chip in line_summary.get("chips", []):
        chip_index = int(chip.get("chip_index"))
        for line in chip.get("selected_centers", []):
            molecule = str(line.get("molecule", "unknown"))
            molecule_counts[molecule] = molecule_counts.get(molecule, 0) + 1
            selected_lines.append(
                {
                    "chip_index": chip_index,
                    "molecule": molecule,
                    "center_wavelength": line.get("center_wavelength"),
                    "rest_frame_center_wavelength": line.get(
                        "rest_frame_center_wavelength"
                    ),
                    "line_strength": line.get("line_strength"),
                    "radial_velocity_kms": line.get("radial_velocity_kms"),
                }
            )
    return {
        "center_source": line_summary.get("center_source"),
        "window_half_width": line_summary.get("window_half_width"),
        "edge_exclusion": line_summary.get("edge_exclusion"),
        "minimum_separation": line_summary.get("minimum_separation"),
        "total_selected_line_count": len(selected_lines),
        "molecule_counts": molecule_counts,
        "selected_lines": selected_lines,
    }


def _build_single_line_info(
    product_dir: Path,
    single_line_summary: Mapping[str, object],
) -> dict[str, object]:
    if not single_line_summary:
        return {"lines": []}
    lines = []
    for item in single_line_summary.get("input_products", []):
        path = Path(item["path"])
        metadata = _read_json_if_exists(path.with_suffix(".json"))
        selected = metadata.get("selected_line_metadata", {})
        lines.append(
            {
                "label": item.get("label"),
                "path": item.get("path"),
                "continuum": item.get("continuum"),
                "line_depth": item.get("line_depth"),
                "molecule": selected.get("molecule"),
                "requested_center_wavelength": selected.get(
                    "requested_center_wavelength"
                ),
                "selected_center_wavelength": selected.get("selected_center_wavelength"),
                "observed_frame_center_wavelength": selected.get(
                    "observed_frame_center_wavelength"
                ),
                "line_strength": selected.get("line_strength"),
                "line_strength_temperature": selected.get(
                    "line_strength_temperature"
                ),
                "radial_velocity_kms": selected.get("radial_velocity_kms"),
            }
        )
    return {
        "normalization": single_line_summary.get("normalization"),
        "lines": lines,
    }


def _posterior_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "median": None,
            "q16": None,
            "q84": None,
            "minus": None,
            "plus": None,
            "q05": None,
            "q95": None,
            "mean": None,
            "std": None,
        }
    q05, q16, median, q84, q95 = np.nanpercentile(
        values,
        [5.0, 16.0, 50.0, 84.0, 95.0],
    )
    return {
        "n": int(values.size),
        "median": float(median),
        "q16": float(q16),
        "q84": float(q84),
        "minus": float(median - q16),
        "plus": float(q84 - median),
        "q05": float(q05),
        "q95": float(q95),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
    }


def _array_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "min": None,
            "q05": None,
            "q16": None,
            "median": None,
            "q84": None,
            "q95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    q05, q16, median, q84, q95 = np.nanpercentile(
        values,
        [5.0, 16.0, 50.0, 84.0, 95.0],
    )
    return {
        "n": int(values.size),
        "min": float(np.nanmin(values)),
        "q05": float(q05),
        "q16": float(q16),
        "median": float(median),
        "q84": float(q84),
        "q95": float(q95),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
    }


def _count_sampled_parameters(samples: np.lib.npyio.NpzFile) -> int:
    sample_site_names = [
        name
        for name in samples.files
        if name.endswith("_raw") or name in {"A", "log_w", "sigma_d"}
    ]
    if not sample_site_names:
        sample_site_names = [
            name
            for name in (
                "T0",
                "alpha",
                "logg",
                "log_vmr_co",
                "log_vmr_h2o",
                "log_vmr_ch4",
                "log_vmr_hf",
                "log_p_cloud",
                "sigma_log_p",
                "A",
                "log_w",
                "sigma_d",
            )
            if name in samples.files
        ]
    count = 0
    for name in sample_site_names:
        values = np.asarray(samples[name])
        if values.ndim == 0:
            continue
        if not np.issubdtype(values.dtype, np.number):
            continue
        count += int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
    return count


def _sample_count(samples: np.lib.npyio.NpzFile) -> int:
    for name in samples.files:
        values = np.asarray(samples[name])
        if values.ndim >= 1 and values.shape[0] > 10:
            return int(values.shape[0])
    raise ValueError("Could not infer sample count.")


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(dict(row)))
    return path


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_primary_bundle_readme(
    primary_dir: Path,
    manifest: Mapping[str, object],
    summary_payload: Mapping[str, object],
) -> None:
    readme = [
        "# Primary Product Bundle",
        "",
        "This directory collects the publication-facing primary products for one Doraex run.",
        "",
        "## Figures",
        "",
    ]
    for item in manifest["figures"]:
        readme.append(f"- `{item['path']}`: {item['name']}")
    readme.extend(
        [
            "",
            "## Info Tables",
            "",
            "- `info/run_summary.json`: run configuration and sampler diagnostics.",
            "- `info/posterior_atmosphere.csv`: atmospheric posterior medians and intervals.",
            "- `info/posterior_nuisance_by_chip.csv`: chip-level nuisance summaries.",
            "- `info/posterior_log_w_by_phase.csv`: phase-level `log_w` summaries.",
            "- `info/cloud_map_summary.csv`: cloud-map pixel statistics.",
            "- `info/line_stack_selected_lines.csv`: line-stack wavelength selections.",
            "- `info/single_line_pressure_response_summary.csv`: pure-line diagnostic numbers.",
            "",
            "## Sources",
            "",
            f"- Samples: `{manifest['samples']}`",
            f"- Product directory: `{manifest['source_product_dir']}`",
        ]
    )
    primary_dir.joinpath("README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def generate_corner_products(samples_path: Path | str, product_dir: Path | str) -> tuple[Path, Path]:
    """Generate atmospheric and nuisance corner plots from a samples archive."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "corner products require matplotlib and corner to be installed"
        ) from exc

    samples = np.load(samples_path, allow_pickle=True)
    product_root = Path(product_dir)
    product_root.mkdir(parents=True, exist_ok=True)

    atm_keys = [key for key in DEFAULT_ATMOSPHERE_CORNER_KEYS if key in samples.files]
    if atm_keys:
        atm_data = np.column_stack([np.asarray(samples[key]) for key in atm_keys])
        atm_figure = corner.corner(
            atm_data,
            labels=[DEFAULT_ATMOSPHERE_CORNER_LABELS.get(key, key) for key in atm_keys],
            bins=36,
            quantiles=(0.16, 0.5, 0.84),
            show_titles=True,
            title_fmt=".3g",
            title_kwargs={"fontsize": 9},
            label_kwargs={"fontsize": 10},
            plot_datapoints=False,
            fill_contours=True,
            smooth=1.0,
            color="tab:blue",
        )
        atm_path = product_root / "corner_atm.png"
        atm_figure.savefig(atm_path, dpi=180, bbox_inches="tight")
        plt.close(atm_figure)
    else:
        raise ValueError(f"No atmospheric corner keys found in {samples_path}")

    nuisance_data, nuisance_labels = _nuisance_corner_data(samples)
    nuisance_figure = corner.corner(
        nuisance_data,
        labels=nuisance_labels,
        bins=36,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        color="tab:green",
    )
    nuisance_path = product_root / "corner_nuisance.png"
    nuisance_figure.savefig(nuisance_path, dpi=180, bbox_inches="tight")
    plt.close(nuisance_figure)
    return atm_path, nuisance_path


def generate_log_w_products(samples_path: Path | str, product_dir: Path | str) -> tuple[Path, Path]:
    """Generate compact diagnostics for the chip/exposure log-weight posterior."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Ellipse, Rectangle
    from matplotlib.colors import Normalize

    samples = np.load(samples_path, allow_pickle=True)
    if "log_w" not in samples.files:
        raise ValueError(f"No log_w samples found in {samples_path}")
    log_w = np.asarray(samples["log_w"], dtype=float)
    if log_w.ndim != 3:
        raise ValueError(f"Expected log_w with shape (sample, chip, exposure), got {log_w.shape}")

    product_root = Path(product_dir)
    product_root.mkdir(parents=True, exist_ok=True)
    summary_path = product_root / "log_w_summary.png"
    corr_path = product_root / "log_w_corr_ellipse.png"

    _plot_log_w_summary(log_w, summary_path, plt)
    _plot_log_w_corr_ellipse(log_w, corr_path, plt, PatchCollection, Ellipse, Rectangle, Normalize)
    return summary_path, corr_path


def _plot_log_w_summary(log_w, output_path, plt):
    n_sample, n_chip, n_phase = log_w.shape
    median_log_w = np.median(log_w, axis=0)
    median_rho_percent = 100.0 * np.expm1(median_log_w)
    half_width = 0.5 * (
        np.percentile(100.0 * np.expm1(log_w), 84, axis=0)
        - np.percentile(100.0 * np.expm1(log_w), 16, axis=0)
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 7.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]},
    )

    limit = max(0.5, float(np.nanmax(np.abs(median_rho_percent))) * 1.15)
    image = axes[0].imshow(
        median_rho_percent,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set_title(r"Posterior median of $100[\exp(\log w_{c,m})-1]$")
    _style_log_w_heatmap_axis(axes[0], n_chip, n_phase)
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.012)
    colorbar.set_label("relative normalization (%)")

    uncertainty_limit = max(3.0, float(np.nanmax(half_width)) * 1.05)
    image = axes[1].imshow(
        half_width,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=uncertainty_limit,
    )
    axes[1].set_title(r"Posterior 68% half-width of $100[\exp(\log w_{c,m})-1]$")
    _style_log_w_heatmap_axis(axes[1], n_chip, n_phase)
    colorbar = fig.colorbar(image, ax=axes[1], pad=0.012)
    colorbar.set_label("relative normalization (%)")

    phase = np.arange(n_phase)
    chip_colors = plt.get_cmap("tab10")(np.arange(n_chip))
    q16 = np.percentile(100.0 * np.expm1(log_w), 16, axis=0)
    q84 = np.percentile(100.0 * np.expm1(log_w), 84, axis=0)
    for chip_index in range(n_chip):
        axes[2].fill_between(
            phase,
            q16[chip_index],
            q84[chip_index],
            color=chip_colors[chip_index],
            alpha=0.16,
            linewidth=0.0,
        )
        axes[2].plot(
            phase,
            median_rho_percent[chip_index],
            color=chip_colors[chip_index],
            lw=1.8,
            marker="o",
            ms=3.8,
            label=f"chip {chip_index}",
        )
    axes[2].axhline(0.0, color="0.25", lw=0.9, ls="--")
    axes[2].set_xlim(-0.4, n_phase - 0.6)
    axes[2].set_xlabel("exposure index")
    axes[2].set_ylabel("relative normalization (%)")
    axes[2].set_title(r"Median and 68% interval of $\log w_{c,m}$ weights")
    axes[2].legend(ncol=n_chip, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    axes[2].grid(True, color="0.88", lw=0.6)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _style_log_w_heatmap_axis(axis, n_chip, n_phase):
    axis.set_ylabel("chip index")
    axis.set_xlabel("exposure index")
    axis.set_yticks(np.arange(n_chip))
    axis.set_xticks(np.arange(n_phase))
    axis.tick_params(axis="both", labelsize=9)


def _plot_log_w_corr_ellipse(
    log_w,
    output_path,
    plt,
    PatchCollection,
    Ellipse,
    Rectangle,
    Normalize,
):
    n_sample, n_chip, n_phase = log_w.shape
    labels = [f"c{chip}m{phase}" for chip in range(n_chip) for phase in range(n_phase)]
    flat = log_w.reshape(n_sample, n_chip * n_phase)
    corr = np.corrcoef(flat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    patches = []
    colors = []
    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = float(np.clip(corr[row, col], -1.0, 1.0))
            width = 0.86
            height = 0.86 * np.sqrt(max(0.0, 1.0 - abs(value)))
            angle = 45.0 if value >= 0.0 else -45.0
            patches.append(Ellipse((col, row), width=width, height=height, angle=angle))
            colors.append(value)

    fig, ax = plt.subplots(figsize=(13.0, 12.0), constrained_layout=True)
    collection = PatchCollection(
        patches,
        array=np.asarray(colors),
        cmap="RdBu_r",
        norm=Normalize(vmin=-1.0, vmax=1.0),
        edgecolor="0.72",
        linewidth=0.15,
    )
    ax.add_collection(collection)

    for boundary in range(n_phase, n_chip * n_phase, n_phase):
        ax.axhline(boundary - 0.5, color="0.1", lw=1.0)
        ax.axvline(boundary - 0.5, color="0.1", lw=1.0)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            n_chip * n_phase,
            n_chip * n_phase,
            fill=False,
            edgecolor="0.1",
            linewidth=1.0,
        )
    )

    ax.set_xlim(-0.5, n_chip * n_phase - 0.5)
    ax.set_ylim(n_chip * n_phase - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(r"Posterior correlation of $\log w_{c,m}$")
    ax.set_xlabel(r"$\log w_{c,m}$ index")
    ax.set_ylabel(r"$\log w_{c,m}$ index")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.4)
    ax.set_yticklabels(labels, fontsize=5.4)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(collection, ax=ax, fraction=0.035, pad=0.018)
    colorbar.set_label("posterior correlation")

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_primary_product_manifest(
    product_dir: Path | str,
    *,
    paths: Mapping[str, Path],
    missing: Iterable[Path],
    prefix: str,
    samples_path: Path | str,
) -> Path:
    """Write a machine-readable primary product manifest."""

    root = Path(product_dir)
    missing_set = {Path(path).resolve() for path in missing}
    manifest = {
        "product_set": "primary",
        "prefix": prefix,
        "samples": str(samples_path),
        "products": [
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "missing": path.resolve() in missing_set,
            }
            for name, path in sorted(paths.items())
        ],
    }
    manifest_path = root / "primary_products.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _run_map_products(config: PrimaryProductConfig, *, env: Mapping[str, str]) -> None:
    script = _example_script(config.project_root, "make_milestone4_on_the_fly_products.py")
    command = [
        config.python_executable,
        str(script),
        "--samples",
        str(config.samples_path),
        "--out-dir",
        str(config.product_dir),
        "--reuse-map-products",
    ]
    command.append("--x64" if config.x64 else "--no-x64")
    _run(command, cwd=config.project_root, env=env)


def _run_line_stack(config: PrimaryProductConfig, *, env: Mapping[str, str]) -> None:
    script = _example_script(config.project_root, "make_milestone5_minimum_window_figures.py")
    command = [
        config.python_executable,
        str(script),
        "--samples",
        str(config.samples_path),
        "--product-dir",
        str(config.product_dir),
        "--out-dir",
        str(Path(config.product_dir) / "products"),
        "--center-source",
        "line-strength",
        "--edge-exclusion",
        "20.0",
        "--minima-percentile",
        "35.0",
        "--minimum-separation",
        "8.0",
        "--max-minima-per-chip",
        "8",
        "--no-individual-windows",
        "--no-combined-stack",
        "--joint-combined-stack",
        "--composite-summary",
        "--composite-output-name",
        f"{config.prefix}_mean_subtracted_line_stack_summary.png",
        "--composite-figure-width",
        "24.0",
        "--figure-height",
        "19.5",
        "--font-size",
        "18.0",
        "--label-size",
        "22.0",
        "--title-size",
        "20.0",
        "--tick-size",
        "16.0",
        "--delta-scale",
        "3.0",
        "--offset-scale",
        "0.62",
        "--composite-chip-offset-scale",
        "0.62",
        "--composite-wspace",
        "0.20",
        "--composite-hspace",
        "0.10",
        "--mean-ylim-edge-fraction",
        "0.025",
        "--mean-ylim-percentile",
        "0.8",
        "--mean-ylim-pad-fraction",
        "0.08",
        "--observed-alpha",
        "0.48",
        "--observed-linewidth",
        "0.6",
        "--model-alpha",
        "0.98",
        "--model-linewidth",
        "1.8",
        "--observed-color-by-phase",
        "--observed-cmap",
        "turbo",
        "--highlight-lines",
        "2:23222.836820905963:-:deepskyblue,3:23429.59216554832:--:orange",
        "--highlight-linewidth",
        "1.4",
        "--highlight-line-alpha",
        "0.85",
        "--publication-style",
    ]
    _run(command, cwd=config.project_root, env=env)


def _run_single_line_products(config: PrimaryProductConfig, *, env: Mapping[str, str]) -> None:
    product_dir = Path(config.product_dir)
    products_dir = product_dir / "products"
    single_line_script = _example_script(
        config.project_root, "make_milestone5_single_line_rt_prediction.py"
    )
    comparison_script = _example_script(
        config.project_root, "compare_milestone5_single_line_rt_predictions.py"
    )

    input_paths = []
    labels = []
    for spec in DEFAULT_SINGLE_LINE_SPECS:
        command = [
            config.python_executable,
            str(single_line_script),
            "--samples",
            str(config.samples_path),
            "--product-dir",
            str(product_dir),
            "--out-dir",
            str(products_dir),
            "--chip-index",
            str(spec.chip_index),
            "--molecule",
            spec.molecule,
            "--line-center",
            str(spec.line_center),
        ]
        command.append("--x64" if config.x64 else "--no-x64")
        _run(command, cwd=config.project_root, env=env)
        input_paths.append(products_dir / f"{spec.stem}.npz")
        labels.append(spec.label)

    comparison_output = products_dir / f"{config.prefix}_single_line_pressure_response_comparison.png"
    command = [
        config.python_executable,
        str(comparison_script),
        "--inputs",
        ",".join(str(path) for path in input_paths),
        "--labels",
        ",".join(labels),
        "--out",
        str(comparison_output),
        "--normalization",
        "line-depth",
        "--delta-scale",
        "2.0",
        "--figure-width",
        "5.8",
        "--figure-height",
        "14.8",
        "--left-margin",
        "0.20",
        "--right-margin",
        "0.96",
        "--top-margin",
        "0.985",
        "--bottom-margin",
        "0.09",
        "--legend-placement",
        "inside",
        "--top-headroom-scale",
        "1.4",
        "--font-size",
        "17.0",
        "--label-size",
        "19.0",
        "--tick-size",
        "16.0",
        "--legend-size",
        "13.5",
        "--linewidth",
        "1.45",
        "--pressure-response-panel",
        "--publication-style",
    ]
    _run(command, cwd=config.project_root, env=env)


def _nuisance_corner_data(samples: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    arrays = []
    labels = []
    if "A" in samples.files:
        amplitudes = np.asarray(samples["A"])
        for chip_index in range(amplitudes.shape[1]):
            arrays.append(amplitudes[:, chip_index])
            labels.append(rf"$A_{chip_index}$")
    if "sigma_d" in samples.files:
        sigmas = np.asarray(samples["sigma_d"])
        for chip_index in range(sigmas.shape[1]):
            arrays.append(sigmas[:, chip_index])
            labels.append(rf"$\sigma_{{d,{chip_index}}}$")
    if not arrays:
        raise ValueError("No nuisance keys found; expected A and/or sigma_d in samples")
    return np.column_stack(arrays), labels


def _postprocess_environment(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str((Path(project_root) / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = src_path
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache" / "jax"))
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    xla_flags = env.get("XLA_FLAGS", "")
    if "--xla_gpu_autotune_level" not in xla_flags:
        env["XLA_FLAGS"] = f"{xla_flags} --xla_gpu_autotune_level=0".strip()
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _example_script(project_root: Path, name: str) -> Path:
    script = Path(project_root) / "examples" / "luhman16b_yama" / name
    if not script.exists():
        raise FileNotFoundError(f"Required postprocess script does not exist: {script}")
    return script


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    print("+ " + shlex.join([str(part) for part in command]), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, env=dict(env), check=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the default Doraex primary products for one run."
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--product-dir", required=True, type=Path)
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory that receives a self-contained primary-product "
            "bundle with figures at the root and paper-facing numbers under info/."
        ),
    )
    parser.add_argument("--prefix", default="m6")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--chip-indices", default="0,1,2,3")
    parser.add_argument("--run-map-products", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-line-stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-single-line-products", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--run-corner-products", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-log-w-products", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip product regeneration and only build --primary-dir from existing products.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = PrimaryProductConfig(
        samples_path=args.samples,
        product_dir=args.product_dir,
        prefix=args.prefix,
        project_root=args.project_root,
        python_executable=args.python_executable,
        chip_indices=tuple(int(item) for item in args.chip_indices.split(",") if item),
        run_map_products=args.run_map_products and not args.collect_only,
        run_line_stack=args.run_line_stack and not args.collect_only,
        run_single_line_products=args.run_single_line_products and not args.collect_only,
        run_corner_products=args.run_corner_products and not args.collect_only,
        run_log_w_products=args.run_log_w_products and not args.collect_only,
        x64=args.x64,
        primary_dir=args.primary_dir,
    )
    result = generate_primary_products(config)
    print(f"Wrote primary product manifest: {result.manifest_path}")
    if result.bundle_dir is not None:
        print(f"Wrote primary product bundle: {result.bundle_dir}")
    if result.missing:
        print("Missing primary products:")
        for path in result.missing:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
