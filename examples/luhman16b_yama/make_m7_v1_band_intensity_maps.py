"""Build M7/v1 CRIRES-band intensity maps and paper-facing comparisons.

The retrieval spectra are wavelength-normalized before entering the
likelihood.  This script instead evaluates the unnormalized ExoJAX emergent
flux on a cloud-pressure grid, integrates it over all four CRIRES chips, and
uses the resulting lookup table to transform the retrieved pressure map into
a model-derived relative band-intensity map.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# This product is intentionally CPU-oriented so it can run beside GPU sampling.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel  # noqa: E402
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
)
from m6_v1_run import _cia_paths, _molecule_paths  # noqa: E402
from make_m7_v1_linearization_check import (  # noqa: E402
    _chip_data_from_samples,
    _median_sample_for_chip,
)


DEFAULT_SAMPLES = ROOT / "results" / "m7" / "v1_zero_mean_log_w_run" / "samples.npz"
DEFAULT_PRODUCT_DIR = (
    ROOT / "results" / "m7" / "v1_zero_mean_log_w_f64_prod" / "baseline_f64"
)
DEFAULT_OUT_DIR = ROOT / "results" / "m7" / "v1_band_intensity_maps"
DEFAULT_REFERENCE_PDF = ROOT / "references" / "ureshino.pdf"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--product-dir", type=Path, default=DEFAULT_PRODUCT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-pdf", type=Path, default=DEFAULT_REFERENCE_PDF)
    parser.add_argument("--database-dir", type=Path, default=Path.home() / "data_mol" / ".database")
    parser.add_argument(
        "--opacity-cache-dir",
        type=Path,
        default=ROOT / "data" / "opacities" / "luhman16b_powerlaw",
    )
    parser.add_argument("--chip-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--nx", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-p-min", type=float, default=0.75)
    parser.add_argument("--log-p-max", type=float, default=2.10)
    parser.add_argument("--pressure-grid-count", type=int, default=129)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--reference-page", type=int, default=15)
    parser.add_argument(
        "--global-pressure-output",
        type=Path,
        default=None,
        help="Optional direct output path for the rotated global pressure panel.",
    )
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-lookup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def evaluate_band_intensity_lookup(
    args: argparse.Namespace,
    samples: np.lib.npyio.NpzFile,
    log_p_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate raw band-integrated intensity on a common pressure grid."""

    per_chip = []
    for chip_index in args.chip_indices:
        chip_data = _chip_data_from_samples(samples, chip_index)
        sample = _median_sample_for_chip(samples, chip_index)
        print(f"Building raw-flux model for chip {chip_index} on CPU...", flush=True)
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx,
            save_opacity_cache=False,
        )

        def raw_spectrum(log_p_cloud, *, column_model=model):
            return column_model.cloudy_raw_at_log_vmrs(
                jnp.asarray(sample["T0"]),
                jnp.asarray(sample["alpha"]),
                jnp.asarray(sample["log_vmr_co"]),
                jnp.asarray(sample["log_vmr_h2o"]),
                jnp.asarray(sample["log_vmr_ch4"]),
                jnp.asarray(sample["log_vmr_hf"]),
                log_p_cloud,
                logg=jnp.asarray(sample["logg"]),
            )

        evaluate_batch = jax.jit(jax.vmap(raw_spectrum))
        chip_values = []
        for start in range(0, log_p_grid.size, args.batch_size):
            stop = min(start + args.batch_size, log_p_grid.size)
            print(
                f"  chip {chip_index}: pressures {start + 1}-{stop}/{log_p_grid.size}",
                flush=True,
            )
            profiles = np.asarray(
                evaluate_batch(jnp.asarray(log_p_grid[start:stop])).block_until_ready()
            )
            chip_values.append(
                np.trapezoid(profiles, np.asarray(chip_data.wavelengths), axis=1)
            )
        per_chip.append(np.concatenate(chip_values))
        del model, evaluate_batch
        jax.clear_caches()
    return np.asarray(per_chip)


def relative_intensity(log_p_values: np.ndarray, lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Interpolate the total band intensity and normalize it by its map mean."""

    values = np.interp(
        np.asarray(log_p_values, dtype=float),
        lookup["log_p_cloud_grid"],
        lookup["band_intensity_total"],
    )
    return values / np.mean(values)


def longitude_grid_values(
    values: np.ndarray,
    *,
    longitude_shift_deg: float,
    nlon: int = 721,
    nlat: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a HEALPix map on a shifted Aitoff grid."""

    lon = np.linspace(-np.pi, np.pi, nlon)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nlat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    theta = 0.5 * np.pi - lat_grid
    phi = np.mod(lon_grid + np.deg2rad(longitude_shift_deg), 2.0 * np.pi)
    pixels = hp.ang2pix(hp.npix2nside(values.size), theta, phi)
    return lon_grid, lat_grid, np.asarray(values)[pixels]


def save_aitoff_panel(
    values: np.ndarray,
    out_path: Path,
    *,
    unit: str,
    longitude_shift_deg: float,
    invert_colorbar: bool = False,
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Image.Image:
    """Save one Aitoff map panel and return a tightly cropped RGB image."""

    lon, lat, projected = longitude_grid_values(
        values, longitude_shift_deg=longitude_shift_deg
    )
    fig = plt.figure(figsize=(8.3, 4.75), dpi=dpi)
    axis = fig.add_axes((0.02, 0.22, 0.96, 0.76), projection="aitoff")
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if not vmin < vmax:
        raise ValueError("The Aitoff color minimum must be smaller than the maximum.")
    mesh = axis.pcolormesh(
        lon,
        lat,
        projected,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    axis.set_xticks(np.linspace(-np.pi, np.pi, 11)[1:-1])
    axis.set_yticks(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 11)[1:-1])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.grid(color="0.55", alpha=0.6, linewidth=0.8)
    color_axis = fig.add_axes((0.18, 0.065, 0.64, 0.045))
    colorbar = fig.colorbar(mesh, cax=color_axis, orientation="horizontal")
    colorbar.set_label(unit, fontsize=16)
    colorbar.ax.tick_params(labelsize=16)
    if invert_colorbar:
        colorbar.ax.invert_xaxis()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return trim_white_border(Image.open(out_path).convert("RGB"))


def render_pdf_page(pdf_path: Path, page: int, dpi: int) -> Image.Image:
    """Render a single reference-PDF page."""

    with tempfile.TemporaryDirectory(prefix="doraex_intensity_fig_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(dpi),
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
        )
        rendered = sorted(Path(tmp).glob("page-*.png"))
        if not rendered:
            raise FileNotFoundError(f"Could not render page {page} of {pdf_path}")
        return Image.open(rendered[0]).convert("RGB")


def trim_white_border(image: Image.Image, padding: int = 12) -> Image.Image:
    """Trim nearly white borders from an image."""

    grayscale = ImageOps.grayscale(image)
    mask = grayscale.point(lambda value: 0 if value > 246 else 255)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, upper, right, lower = bbox
    return image.crop(
        (
            max(left - padding, 0),
            max(upper - padding, 0),
            min(right + padding, image.width),
            min(lower + padding, image.height),
        )
    )


def crop_reference_panels(page_image: Image.Image, dpi: int) -> tuple[Image.Image, Image.Image]:
    """Crop the Bayesian-DI and Crossfield panels from Ureshino Figure 10."""

    scale = dpi / 180.0
    boxes = ((135, 165, 790, 480), (135, 480, 790, 800))
    panels = []
    for box in boxes:
        scaled = tuple(int(round(value * scale)) for value in box)
        panels.append(trim_white_border(page_image.crop(scaled)))
    return panels[0], panels[1]


def fit_width(image: Image.Image, width: int) -> Image.Image:
    """Resize an image to a common width while preserving aspect ratio."""

    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def add_panel_label(image: Image.Image, label: str, letter: str) -> Image.Image:
    """Add a paper-facing panel label above a raster panel."""

    top_margin = 108
    canvas = Image.new("RGB", (image.width, image.height + top_margin), "white")
    canvas.paste(image, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        size=max(22, image.width // 36),
    )
    draw.text((8, 18), f"({letter}) {label}", fill=(20, 20, 20), font=font)
    return canvas


def compose_vertical(panels: list[Image.Image], output: Path) -> None:
    """Stack labeled panels into a single comparison image."""

    width = max(panel.width for panel in panels)
    gap = 28
    canvas = Image.new(
        "RGB",
        (width, sum(panel.height for panel in panels) + gap * (len(panels) - 1)),
        "white",
    )
    y = 0
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, y))
        y += panel.height + gap
    canvas.save(output)


def main() -> None:
    """Generate the lookup table and real-data map figures."""

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", args.x64)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    if jax.default_backend() != "cpu":
        raise RuntimeError("This figure generator must run on the CPU backend.")

    samples = np.load(args.samples, allow_pickle=True)
    lookup_path = args.out_dir / "m7_v1_band_intensity_lookup.npz"
    if args.reuse_lookup and lookup_path.exists():
        lookup_payload = np.load(lookup_path)
        lookup = {name: np.asarray(lookup_payload[name]) for name in lookup_payload.files}
        print(f"Reusing {lookup_path}", flush=True)
    else:
        log_p_grid = np.linspace(args.log_p_min, args.log_p_max, args.pressure_grid_count)
        per_chip = evaluate_band_intensity_lookup(args, samples, log_p_grid)
        lookup = {
            "log_p_cloud_grid": log_p_grid,
            "p_cloud_grid_bar": 10.0**log_p_grid,
            "band_intensity_by_chip": per_chip,
            "band_intensity_total": np.sum(per_chip, axis=0),
            "chip_indices": np.asarray(args.chip_indices, dtype=int),
        }
        np.savez(lookup_path, **lookup)
        print(f"Wrote {lookup_path}", flush=True)

    pressure_map = np.asarray(
        np.load(args.product_dir / "p_cloud_mean_joint_by_chip.npy")[0], dtype=float
    )
    intensity_map = relative_intensity(np.log10(pressure_map), lookup)
    map_path = args.out_dir / "m7_v1_relative_band_intensity.npy"
    np.save(map_path, intensity_map)

    global_pressure_path = (
        args.out_dir / "m7_v1_cloud_pressure_global.png"
        if args.global_pressure_output is None
        else args.global_pressure_output
    )
    global_pressure_path.parent.mkdir(parents=True, exist_ok=True)
    save_aitoff_panel(
        pressure_map,
        global_pressure_path,
        unit="Cloud pressure [bar]",
        longitude_shift_deg=180.0,
        invert_colorbar=True,
        dpi=args.dpi,
    )

    intensity_panel_path = args.out_dir / "m7_v1_band_intensity_shift180.png"
    doraex = save_aitoff_panel(
        intensity_map,
        intensity_panel_path,
        unit=r"Relative intensity $I/\langle I\rangle$",
        longitude_shift_deg=180.0,
        dpi=args.dpi,
    )
    page = render_pdf_page(args.reference_pdf, args.reference_page, args.dpi)
    bayesian_di, crossfield = crop_reference_panels(page, args.dpi)
    width = max(doraex.width, bayesian_di.width, crossfield.width)
    panels = [
        add_panel_label(
            fit_width(doraex, width),
            "Doraex DR-derived CRIRES-band intensity",
            "a",
        ),
        add_panel_label(
            fit_width(bayesian_di, width),
            "Ureshino et al. Bayesian DI intensity",
            "b",
        ),
        add_panel_label(
            fit_width(crossfield, width),
            "Crossfield et al. (2014) surface intensity",
            "c",
        ),
    ]
    comparison_path = args.out_dir / "m7_v1_intensity_map_comparison.png"
    compose_vertical(panels, comparison_path)
    Image.open(comparison_path).save(comparison_path.with_suffix(".pdf"))

    summary = {
        "jax_backend": jax.default_backend(),
        "samples": str(args.samples),
        "product_dir": str(args.product_dir),
        "chip_indices": args.chip_indices,
        "nx": args.nx,
        "pressure_grid_count": int(lookup["log_p_cloud_grid"].size),
        "log_p_cloud_grid_min": float(lookup["log_p_cloud_grid"][0]),
        "log_p_cloud_grid_max": float(lookup["log_p_cloud_grid"][-1]),
        "lookup": str(lookup_path),
        "relative_intensity_map": str(map_path),
        "relative_intensity_min": float(np.min(intensity_map)),
        "relative_intensity_max": float(np.max(intensity_map)),
        "relative_intensity_peak_to_peak": float(np.ptp(intensity_map)),
        "global_pressure_figure": str(global_pressure_path),
        "intensity_comparison_figure": str(comparison_path),
    }
    summary_path = args.out_dir / "m7_v1_band_intensity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
