"""Extract coherent M8 atmosphere/map states for M9 v3 initialization checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = ROOT / "results/m8/v12/masked_free_sigma_d_highest_precision_long_seed0/samples.npz"
DEFAULT_MAPS = ROOT / "results/m8/v12/check_exact_rt/m8_v12_exact_rt_linearization_arrays.npz"
STATE_NAMES = (
    "T0", "alpha", "log_vmr_co", "log_vmr_h2o", "log_vmr_ch4", "log_vmr_hf",
    "logg", "log_p_cloud", "sigma_log_p", "P", "cosi", "v", "q1", "q2",
    "u1", "u2", "A", "sigma_d", "log_w",
)


def sha256(path):
    """Hash a source archive without importing the retrieval runtime."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_seeds(samples_path, maps_path, out_dir):
    """Keep each physical state paired with its archived conditional map."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    hashes = {"samples_sha256": sha256(samples_path), "maps_sha256": sha256(maps_path)}
    outputs = []
    with np.load(samples_path, allow_pickle=False) as samples, np.load(maps_path, allow_pickle=False) as maps:
        for point, label in ((0, "q05"), (2, "q50"), (4, "q95")):
            sample_index = int(maps["posterior_indices"][point])
            for map_index, kind in ((0, "mean"), (1, "draw0")):
                state = {name: np.asarray(samples[name][sample_index], dtype=np.float64) for name in STATE_NAMES}
                raw_map = np.asarray(maps["q_maps"][point, map_index], dtype=np.float64)
                if raw_map.shape != (768,) or not np.all(np.isfinite(raw_map)) or abs(raw_map.mean()) > 1e-6:
                    raise ValueError("The archived seed must be a finite zero-mean nside=8 map.")
                state.update(
                    delta_log_p=raw_map - raw_map.mean(),
                    period=state["P"],
                    sample_index=np.asarray(sample_index),
                    map_index=np.asarray(map_index),
                    chip_indices=np.asarray(samples["chip_indices"]),
                    source_samples=np.asarray(str(Path(samples_path).resolve())),
                    source_maps=np.asarray(str(Path(maps_path).resolve())),
                    source_kind=np.asarray("M8 per-pixel-normalized linear conditional map; initialization only"),
                    **{key: np.asarray(value) for key, value in hashes.items()},
                )
                name = f"m8_{label}_{kind}.npz"
                np.savez(out_dir / name, **state)
                outputs.append({"path": name, "sample_index": sample_index, "map_index": map_index,
                                "removed_map_mean": float(raw_map.mean()), "sha256": sha256(out_dir / name)})
    manifest = {"source_samples": str(samples_path), "source_maps": str(maps_path), **hashes,
                "interpretation": "Coherent initialization candidates, not samples from the raw exact target.",
                "outputs": outputs}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/m9/v3/seeds")
    args = parser.parse_args()
    result = prepare_seeds(args.samples, args.maps, args.out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
