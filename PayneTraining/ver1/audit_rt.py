"""Read-only spectral replay and independent real-space PreMODIT comparison."""

import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

import generate
from io_training import load


HERE = Path(__file__).resolve().parent
ROOT = generate.ROOT
PREMODIT_CACHE = Path(os.environ.get(
    "PAYNE_PREMODIT_CACHE", ROOT / "data/opacities/luhman16b_premodit_m9_v3",
)).expanduser().resolve()


def relative_metrics(actual, expected):
    difference = np.asarray(actual) - np.asarray(expected)
    pointwise = np.abs(difference) / np.maximum(np.abs(expected), np.finfo(float).tiny)
    return {
        "max_pointwise_relative": float(np.max(pointwise)),
        "p99_pointwise_relative": float(np.quantile(pointwise, .99)),
        "rms_over_mean_reference": float(np.sqrt(np.mean(difference**2)) / np.mean(expected)),
    }


def run():
    jax = generate.prepare()
    import jax.numpy as jnp
    from doraex.spectra.exojax_forward import _opacity_cache_namespace
    from exojax.opacity import OpaPremodit

    start = time.monotonic()
    parameters, spectra, wavelength = load(HERE / "data")
    rng = np.random.default_rng(20261115)
    random_rows = rng.choice(len(parameters), 256, replace=False)
    extremes = np.concatenate([np.argmin(parameters, axis=0), np.argmax(parameters, axis=0)])
    indices = np.unique(np.r_[random_rows, extremes, 0, 63, 64, 99967, 99968, 99999])
    teacher_indices = np.unique(np.r_[random_rows[:16], extremes])
    with np.load(generate.BASELINE / "observation_contract.npz", allow_pickle=False) as archive:
        contract = {name: archive[name] for name in archive.files}
    report = {
        "seed": 20261115, "replay_indices": indices.tolist(),
        "teacher_indices": teacher_indices.tolist(), "chips": [],
        "scope": "Stored values vs M9 DiffGrid; selected broad-domain rays vs PreMODIT. No global accuracy guarantee.",
    }
    for chip in range(4):
        print(f"Chip {chip}: loading DiffGrid", flush=True)
        model = generate.build_model(chip, contract)
        atmosphere, spectrum = generate.functions(model)
        replay = jax.jit(jax.vmap(spectrum))
        actual = []
        for offset in range(0, len(indices), 32):
            p = parameters[indices[offset:offset + 32]]
            padded = np.pad(p, ((0, 32 - len(p)), (0, 0)), mode="edge")
            actual.append(np.asarray(replay(padded))[:len(p)])
        actual = np.concatenate(actual)
        stored = np.asarray(spectra[indices, chip * 4500:(chip + 1) * 4500])
        replay_metrics = relative_metrics(actual, stored)
        if not np.allclose(actual, stored, rtol=1e-11, atol=0):
            raise RuntimeError(f"Stored spectra do not match RT: chip {chip}, {replay_metrics}")
        np.testing.assert_array_equal(
            wavelength[chip * 4500:(chip + 1) * 4500], 1e8 / np.asarray(model.nu_grid)[::-1],
        )
        # Independent NumPy Planck formula and bottom-up transfer for eight rays.
        evaluate = jax.jit(atmosphere)
        independent_error = 0.
        bottom_fraction = 0.
        for index in indices[:8]:
            p = parameters[index]
            dtau, temperature = (np.asarray(x) for x in evaluate(p))
            nu = np.asarray(model.nu_grid)
            source = (3.741771790075259e-5 / np.pi) * nu[None, :]**3 / np.expm1(
                1.4387773538277202 * nu[None, :] / temperature[:, None],
            )
            intensity = np.zeros(4500)
            for layer in range(len(dtau) - 1, -1, -1):
                x = dtau[layer] / p[7]
                intensity = intensity * np.exp(-x) + source[layer] * -np.expm1(-x)
            reference = spectra[index, chip * 4500:(chip + 1) * 4500]
            independent_error = max(independent_error, relative_metrics(intensity[::-1], reference)["max_pointwise_relative"])
            bottom_fraction = max(bottom_fraction, float(np.max(
                source[-1] * np.exp(-dtau.sum(axis=0) / p[7]) / reference[::-1],
            )))
        if independent_error > 1e-11:
            raise RuntimeError(f"Independent Planck/transfer check failed: {independent_error}")
        print(f"Chip {chip}: replay {replay_metrics}; independent {independent_error:.3g}", flush=True)
        teacher = copy.copy(model)
        teacher.opacity_mode = "premodit"
        directory = _opacity_cache_namespace(
            PREMODIT_CACHE, model.internal_wavelength_bounds,
            4500, 1e-4, 1e2, 101, 210., 3500., "premodit", 81,
        )
        teacher.opacities = {
            name: OpaPremodit.from_saved_opa(str(directory / f"opa{name}.zarr")).with_profile_kernel("real_space")
            for name in generate.MOLECULES
        }
        for name, opa in teacher.opacities.items():
            np.testing.assert_array_equal(np.asarray(opa.nu_grid), np.asarray(model.nu_grid))
        _, teacher_spectrum = generate.functions(teacher)
        predict_teacher = jax.jit(teacher_spectrum)
        comparisons = []
        for index in teacher_indices:
            reference = np.asarray(predict_teacher(jnp.asarray(parameters[index])))
            stored = np.asarray(spectra[index, chip * 4500:(chip + 1) * 4500])
            if not np.all(np.isfinite(reference)) or np.any(reference <= 0):
                raise RuntimeError(f"Nonfinite or nonpositive PreMODIT intensity: chip {chip}, row {index}")
            comparisons.append({"row": int(index), **relative_metrics(stored, reference)})
        row = {"chip": chip, "replay": replay_metrics,
               "independent_planck_recursion_max_relative": independent_error,
               "bottom_blackbody_fraction_max_for_eight_rays": bottom_fraction,
               "premodit_comparisons": comparisons}
        report["chips"].append(row)
        report["seconds"] = time.monotonic() - start
        (HERE / "audit_rt.json").write_text(json.dumps(report, indent=2) + "\n")
        worst = max(comparisons, key=lambda x: x["rms_over_mean_reference"])
        print(f"Chip {chip}: worst PreMODIT comparison {worst}", flush=True)
        del model, teacher, atmosphere, spectrum, replay, evaluate, teacher_spectrum, predict_teacher
        jax.clear_caches()
        gc.collect()
    report["completed"] = True
    (HERE / "audit_rt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Complete in {report['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    with open("/tmp/doraex_m7_gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
