"""Plot a cloud-pressure map as seen by the observer at selected phases."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCT_DIR = (
    ROOT / "results" / "m7" / "v1_zero_mean_log_w_f64_prod" / "baseline_f64"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a cloud-pressure map in observer-view orthographic projections."
    )
    parser.add_argument("--product-dir", type=Path, default=DEFAULT_PRODUCT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cosi", type=float, default=0.485)
    parser.add_argument("--grid-size", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def observer_view(values: np.ndarray, phase: float, inclination: float, grid_size: int):
    """Sample the HEALPix map on the observer-facing orthographic disk."""

    coordinate = np.linspace(-1.0, 1.0, grid_size)
    y_observer, z_observer = np.meshgrid(coordinate, coordinate)
    radius_squared = y_observer**2 + z_observer**2
    visible = radius_squared <= 1.0
    x_observer = np.sqrt(np.clip(1.0 - radius_squared, 0.0, None))

    alpha = 0.5 * np.pi - inclination
    x_rot = np.cos(alpha) * x_observer - np.sin(alpha) * z_observer
    y_rot = y_observer
    z_rot = np.sin(alpha) * x_observer + np.cos(alpha) * z_observer
    theta = np.arccos(np.clip(z_rot, -1.0, 1.0))
    phi_body = np.mod(np.arctan2(y_rot, x_rot) - 2.0 * np.pi * phase, 2.0 * np.pi)

    nside = hp.npix2nside(values.size)
    sampled = values[hp.ang2pix(nside, theta, phi_body)]
    return np.ma.array(sampled, mask=~visible)


def projected_equator(inclination: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the visible body equator in observer-plane coordinates."""

    alpha = 0.5 * np.pi - inclination
    longitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 400)
    return np.sin(longitude), -np.sin(alpha) * np.cos(longitude)


def main() -> None:
    args = parse_args()
    if not -1.0 <= args.cosi <= 1.0:
        raise ValueError("--cosi must lie between -1 and 1.")

    values = np.asarray(
        np.load(args.product_dir / "p_cloud_mean_joint_by_chip.npy")[0], dtype=float
    )
    inclination = np.arccos(args.cosi)
    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    equator_x, equator_y = projected_equator(inclination)

    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.75), constrained_layout=True)
    image = None
    for axis, phase in zip(axes.flat, (0.00, 0.25, 0.50, 0.75)):
        image = axis.imshow(
            observer_view(values, phase, inclination, args.grid_size),
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap="inferno",
            norm=norm,
            interpolation="none",
        )
        axis.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="black", linewidth=0.8))
        axis.plot(equator_x, equator_y, color="0.82", linewidth=0.65, alpha=0.85)
        axis.set_aspect("equal")
        axis.set_xlim(-1.035, 1.035)
        axis.set_ylim(-1.035, 1.035)
        axis.set_axis_off()
        axis.set_title(rf"$\varphi={phase:.2f}$", fontsize=12, pad=2)

    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.075, pad=0.035)
    colorbar.set_label("Cloud pressure [bar]", fontsize=12, labelpad=2)
    colorbar.ax.tick_params(labelsize=11)
    colorbar.ax.invert_xaxis()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
