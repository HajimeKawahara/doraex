"""Generate single-pixel intensities using the M9 v3 opacity model on a GPU."""

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from io_training import create


ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(os.environ.get(
    "PAYNE_BASELINE",
    ROOT / "results/m9/v3/production_draw0_w2000_s1500_d9_d11_20260912T233435Z/q95_seed1",
)).expanduser().resolve()
CACHE = Path(os.environ.get(
    "PAYNE_OPACITY_CACHE", ROOT / "data/opacities/luhman16b_diffgrid_m9_v3_real_space",
)).expanduser().resolve()
DATABASE = Path(os.environ.get(
    "DORAEX_DATABASE_DIR", Path.home() / "data_mol/.database",
)).expanduser().resolve()
EXOJAX = Path(os.environ.get("EXOJAX_DIR", ROOT.parent / "exojax")).expanduser().resolve()
LOWER = np.array([1000., .05, -6., -6., -6., -10., -3., 0.])
UPPER = np.array([1700., .20, -1., -1., -1., -5., 5., 1.])
MOLECULES = {
    "CO": "CO/12C-16O/Li2015", "H2O": "H2O/1H2-16O/POKAZATEL",
    "CH4": "CH4/12C-1H4/MM", "HF": "HF/1H-19F/Coxon-Hajig",
}


def prepare():
    """Require the archived forward source and configure float64 CUDA."""
    source = ROOT / "src/doraex/spectra/exojax_forward.py"
    if source.read_bytes() != (BASELINE / "source/exojax_forward.py").read_bytes():
        raise RuntimeError("The forward source differs from the M9 v3 baseline.")
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path[:0] = [str(EXOJAX / "src"), str(ROOT / "src")]
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("A CUDA GPU is required; CPU fallback is disabled.")
    print(f"Device: {device.device_kind}; float64=True", flush=True)
    return jax


def build_model(chip, contract):
    from doraex.spectra.exojax_forward import (
        FixedPowerLawAtmosphere, Luhman16BPowerLawColumnModel,
        _internal_spectrum_wavelength_bounds, _opacity_cache_namespace,
    )

    observed = contract[f"wavelengths_chip{chip}"]
    sampling = contract[f"profile_wavelengths_chip{chip}"]
    bounds = _internal_spectrum_wavelength_bounds(observed, sampling, systemic_rv=25.66)
    directory = _opacity_cache_namespace(
        CACHE, bounds, 4500, 1e-4, 1e2, 101, 210., 3500., "diffgrid", 81,
    )
    # Require all caches before model construction; never rebuild a teacher.
    for molecule in MOLECULES:
        archive = directory / f"opa{molecule}.zarr"
        metadata_path = archive / "zarr.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())["attributes"]
        else:
            metadata = json.loads((archive / ".zattrs").read_text())
        if metadata["opa_state"]["teacher_profile_kernel"] != "real_space":
            raise RuntimeError(f"Wrong teacher kernel: {archive}")
        if (metadata["array_shapes"]["log_cross_section_grid"] != [101, 81, 4500]
                or metadata["array_dtypes"]["log_cross_section_grid"] != "float64"):
            raise RuntimeError(f"Wrong opacity grid or precision: {archive}")
    return Luhman16BPowerLawColumnModel(
        observed, {key: DATABASE / value for key, value in MOLECULES.items()},
        {"H2H2": DATABASE / "H2-H2_2011.cia", "H2He": DATABASE / "H2-He_2011.cia"},
        CACHE, parameters=FixedPowerLawAtmosphere(
            logg=float(contract["fixed_logg"]), rv=25.66,
        ),
        sampling_wavelengths=sampling, nx=4500, opacity_mode="diffgrid",
        diffgrid_temperature_nodes=81, save_opacity_cache=False,
    )


def functions(model):
    import jax.numpy as jnp
    from exojax.rt.planck import piBarr
    from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity

    def atmosphere(p):
        temperature = model.art.powerlaw_temperature(p[0], p[1])
        vmrs = {name: p[2 + i] for i, name in enumerate(MOLECULES)}
        dtau = model._dtau_molecular_and_cia(
            temperature, 10.**model.parameters.logg, log_vmr_values=vmrs,
        ) + model._cloud_dtau(p[6])[:, None]
        return dtau, temperature

    def spectrum(p):
        dtau, temperature = atmosphere(p)
        source = piBarr(temperature, model.nu_grid) / jnp.pi
        intensity = rtrun_emis_pureabs_ibased_intensity(dtau, source, p[7:8])[0]
        return intensity[::-1]  # Ascending wavelength, still per wavenumber.

    return atmosphere, spectrum


def validate(model, atmosphere, spectrum, jax):
    """Check units, angle integration, wavelength order and a layer recursion."""
    import jax.numpy as jnp
    from exojax.rt.planck import piBarr
    from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity

    evaluate = jax.jit(atmosphere)
    predict = jax.jit(spectrum)
    cases = np.array([
        [1244.75, .1558, -3.0355, -3.3214, -4.7682, -7.1746, .89, .5],
        [1000., .20, -6., -6., -6., -10., -3., 1e-8],
        [1700., .05, -1., -1., -1., -5., 5., 1.],
        [1700., .20, -6., -6., -6., -10., 2., .1],
    ])
    angle_error = recursion_error = 0.
    for p in cases:
        dtau, temperature = evaluate(jnp.asarray(p))
        source = piBarr(temperature, model.nu_grid) / jnp.pi
        intensities = rtrun_emis_pureabs_ibased_intensity(dtau, source, model.art.mus)
        integrated = 2 * jnp.pi * jnp.sum(
            (model.art.mus * model.art.weights)[:, None] * intensities, axis=0,
        )
        reference = np.asarray(model.art.run(dtau, temperature))
        error = np.max(np.abs(np.asarray(integrated) - reference)) / np.max(reference)
        angle_error = max(angle_error, float(error))
        optical_depth, planck = np.asarray(dtau), np.asarray(source)
        recursive = np.zeros(4500)
        for layer in range(len(optical_depth) - 1, -1, -1):
            x = optical_depth[layer] / p[7]
            recursive = recursive * np.exp(-x) + planck[layer] * -np.expm1(-x)
        actual = np.asarray(predict(jnp.asarray(p)))
        if not np.all(np.isfinite(actual)) or np.any(actual <= 0):
            raise RuntimeError("Nonpositive or nonfinite validation intensity.")
        error = np.max(np.abs(actual - recursive[::-1])) / np.max(recursive)
        recursion_error = max(recursion_error, float(error))
    if max(angle_error, recursion_error) > 1e-11:
        raise RuntimeError(f"RT validation failed: {angle_error=}, {recursion_error=}")
    return {"angle_relative_error": angle_error, "recursion_relative_error": recursion_error}


def run(args):
    jax = prepare()
    start = time.monotonic()
    output = args.out.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output or partial dataset already exists: {output}")
    if output.with_suffix(".json").exists():
        raise FileExistsError(f"Run report already exists: {output.with_suffix('.json')}")
    rng = np.random.default_rng(args.seed)
    unit = rng.random((args.samples, 8))
    unit[:, 7] = 1. - unit[:, 7]  # Uniform (0, 1], never exactly zero.
    parameters = LOWER + unit * (UPPER - LOWER)
    with np.load(BASELINE / "observation_contract.npz", allow_pickle=False) as archive:
        contract = {name: archive[name] for name in archive.files}
    spectra = None
    validations = []
    for chip in range(4):
        print(f"Chip {chip}: loading M9 opacity cache", flush=True)
        model = build_model(chip, contract)
        atmosphere, spectrum = functions(model)
        checks = validate(model, atmosphere, spectrum, jax)
        validations.append(checks)
        print(f"Chip {chip}: validation {checks}", flush=True)
        if spectra is None:
            from doraex.spectra.exojax_forward import _internal_spectrum_wavelength_bounds
            from exojax.utils.grids import wavenumber_grid

            wavelengths = []
            for c in range(4):
                bounds = _internal_spectrum_wavelength_bounds(
                    contract[f"wavelengths_chip{c}"], contract[f"profile_wavelengths_chip{c}"],
                    systemic_rv=25.66,
                )
                nu, _, _ = wavenumber_grid(*bounds, 4500, unit="AA", xsmode="diffgrid")
                wavelengths.append(1e8 / np.asarray(nu)[::-1])
            wavelength = np.concatenate(wavelengths)
            spectra = create(partial, parameters, wavelength)
        np.testing.assert_array_equal(
            wavelength[chip * 4500:(chip + 1) * 4500], 1e8 / np.asarray(model.nu_grid)[::-1],
        )
        batch = jax.jit(jax.vmap(spectrum))
        last_report = time.monotonic()
        for begin in range(0, args.samples, args.batch_size):
            stop = min(begin + args.batch_size, args.samples)
            p = parameters[begin:stop]
            if len(p) < args.batch_size:
                p = np.pad(p, ((0, args.batch_size - len(p)), (0, 0)), mode="edge")
            values = np.asarray(batch(p))[:stop - begin]
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise RuntimeError(f"Invalid intensity at chip {chip}, rows {begin}:{stop}.")
            spectra[begin:stop, chip * 4500:(chip + 1) * 4500] = values
            now = time.monotonic()
            if now - last_report >= 30 or stop == args.samples:
                spectra.flush()
                print(f"Chip {chip}: {stop}/{args.samples}; elapsed {now - start:.1f} s", flush=True)
                last_report = now
        del batch, atmosphere, spectrum, model
        jax.clear_caches()
        gc.collect()
    spectra.flush()
    del spectra
    partial.rename(output)
    report = {
        "samples": args.samples, "seed": args.seed, "batch_size": args.batch_size,
        "seconds": time.monotonic() - start, "gpu": jax.devices()[0].device_kind,
        "baseline": str(BASELINE),
        "fixed_logg": float(contract["fixed_logg"]), "validation": validations,
        "exojax_commit": subprocess.check_output(
            ["git", "-C", str(EXOJAX), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Complete: {output} ({report['seconds']:.1f} s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("samples and batch-size must be positive")
    with open("/tmp/doraex_m7_gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
