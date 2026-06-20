"""Primary product generation for Doraex post-processing runs.

The primary product set is the small, stable collection of figures that should
exist whenever a run is considered to have completed product generation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
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
    x64: bool = True


@dataclass(frozen=True)
class PrimaryProductResult:
    """Generated primary products and any missing required paths."""

    paths: Mapping[str, Path]
    missing: tuple[Path, ...]
    manifest_path: Path


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
    return PrimaryProductResult(paths=paths, missing=missing, manifest_path=manifest_path)


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

    centers = _standardized_parameter_centers(samples)
    atm_keys = [key for key in DEFAULT_ATMOSPHERE_CORNER_KEYS if key in samples.files]
    if atm_keys:
        atm_data = np.column_stack([np.asarray(samples[key]) for key in atm_keys])
        atm_truths = [centers.get(key) for key in atm_keys]
        atm_figure = corner.corner(
            atm_data,
            labels=[DEFAULT_ATMOSPHERE_CORNER_LABELS.get(key, key) for key in atm_keys],
            truths=atm_truths if any(truth is not None for truth in atm_truths) else None,
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
        "--delta-scale",
        "3.0",
        "--offset-scale",
        "0.5",
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
        "--pressure-response-panel",
        "--publication-style",
    ]
    _run(command, cwd=config.project_root, env=env)


def _standardized_parameter_centers(samples: np.lib.npyio.NpzFile) -> dict[str, float]:
    if (
        "standardized_parameter_names" not in samples.files
        or "standardized_parameter_centers" not in samples.files
    ):
        return {}
    names = [str(name) for name in samples["standardized_parameter_names"]]
    centers = np.asarray(samples["standardized_parameter_centers"], dtype=float)
    return {name: float(center) for name, center in zip(names, centers, strict=False)}


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
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
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
        run_map_products=args.run_map_products,
        run_line_stack=args.run_line_stack,
        run_single_line_products=args.run_single_line_products,
        run_corner_products=args.run_corner_products,
        x64=args.x64,
    )
    result = generate_primary_products(config)
    print(f"Wrote primary product manifest: {result.manifest_path}")
    if result.missing:
        print("Missing primary products:")
        for path in result.missing:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
