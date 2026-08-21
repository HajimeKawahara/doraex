"""Recompute a pure single-line plus CIA RT prediction for Milestone 5."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


SPEED_OF_LIGHT_KMS = 299792.458
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.data.luhman16b import Luhman16BChipData  # noqa: E402
from doraex.operators.doppler import doppler_padded_wavelengths  # noqa: E402
from doraex.spectra.exojax_forward import Luhman16BPowerLawColumnModel  # noqa: E402
from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    _chip_sample,
    fixed_two_column_median_sample,
)
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
    _cia_paths,
    _molecule_paths,
)
from make_milestone4_on_the_fly_products import (  # noqa: E402
    _build_product_geometry,
    _linear_profile_operator_from_sample,
)


EXOMOL_ELOWER_MAX = {
    "CO": 58242.689,
    "H2O": 23726.625476,
    "CH4": 9900.0,
    "HF": 20000.0,
}
VMR_PARAMETER_NAMES = {
    "CO": "log_vmr_co",
    "H2O": "log_vmr_h2o",
    "CH4": "log_vmr_ch4",
    "HF": "log_vmr_hf",
}
LOG_VMR_NAMES = tuple(VMR_PARAMETER_NAMES.values())
SUMMARY_PARAMETER_NAMES = frozenset(
    {
        "A",
        "P",
        "T0",
        "alpha",
        "cloud_delta",
        "cosi",
        "ell_b",
        "fixed_cloud_delta",
        "fixed_sigma_eigen",
        "log_p_cloud",
        "log_p_mid",
        "log_vmr_ch4",
        "log_vmr_co",
        "log_vmr_h2o",
        "log_vmr_hf",
        "log_w",
        "logg",
        "pressure_derivative_step",
        "q1",
        "q2",
        "sigma_b",
        "sigma_d",
        "sigma_eigen",
        "sigma_log_p",
        "t0",
        "u1",
        "u2",
        "v",
        "zero_mean_pressure_map",
        "zeta_vmr",
    }
)


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    default_product_dir = (
        ROOT
        / "results"
        / "milestone5"
        / "milestone5_on_the_fly_atmosphere_stage1_rotated_diag2_cholfix_ta095_prod_f32"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the Milestone 5 prediction using only one selected "
            "molecular line plus H2-H2/H2-He CIA. No observed spectrum is "
            "plotted because this is a pure-line diagnostic prediction."
        )
    )
    parser.add_argument(
        "--samples",
        default=str(default_product_dir / "mcmc_on_the_fly_atmosphere_pressure.npz"),
    )
    parser.add_argument("--product-dir", default=str(default_product_dir))
    parser.add_argument(
        "--minimum-summary",
        default=None,
        help=(
            "minimum_window_summary.json containing a single_line_profile entry. "
            "Defaults to product-dir/minimum_window_figures/minimum_window_summary.json."
        ),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_single_line"),
    )
    parser.add_argument("--chip-index", type=int, default=None)
    parser.add_argument("--molecule", default=None)
    parser.add_argument("--line-center", type=float, default=None)
    parser.add_argument("--window-half-width", type=float, default=5.0)
    parser.add_argument("--line-strength-temperature", type=float, default=None)
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--offset-scale", type=float, default=0.5)
    parser.add_argument("--top-height-ratio", type=float, default=10.0)
    parser.add_argument("--figure-height", type=float, default=12.0)
    parser.add_argument("--model-alpha", type=float, default=0.98)
    parser.add_argument("--model-linewidth", type=float, default=1.6)
    parser.add_argument("--color-by-phase", action="store_true")
    parser.add_argument("--phase-cmap", default="turbo")
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _format_wavelength(value):
    """Format a wavelength for stable output filenames."""

    return f"{value:.2f}".replace(".", "p")


def _wavenumber_to_wavelength(wavenumber):
    """Convert wavenumber in cm^-1 to wavelength in Angstrom."""

    return 1.0e8 / np.asarray(wavenumber, dtype=float)


def _observed_frame_center(rest_wavelength, radial_velocity):
    """Return the line center shifted by the model radial velocity."""

    return float(rest_wavelength) * (1.0 + float(radial_velocity) / SPEED_OF_LIGHT_KMS)


def _load_single_line_metadata(args):
    """Return chip index, molecule, and center wavelength for the selected line."""

    if (
        args.chip_index is not None
        and args.molecule is not None
        and args.line_center is not None
    ):
        return {
            "chip_index": int(args.chip_index),
            "molecule": args.molecule,
            "center_wavelength": float(args.line_center),
            "source": "command_line",
        }

    summary_path = (
        Path(args.product_dir) / "minimum_window_figures" / "minimum_window_summary.json"
        if args.minimum_summary is None
        else Path(args.minimum_summary)
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "single_line_profile" not in summary:
        raise KeyError(
            "The minimum summary does not contain single_line_profile. "
            "Provide --chip-index, --molecule, and --line-center explicitly."
        )
    line = dict(summary["single_line_profile"])
    if args.chip_index is not None:
        line["chip_index"] = int(args.chip_index)
    if args.molecule is not None:
        line["molecule"] = args.molecule
    if args.line_center is not None:
        line["center_wavelength"] = float(args.line_center)
    line["source"] = str(summary_path)
    return line


class SingleLinePowerLawColumnModel(Luhman16BPowerLawColumnModel):
    """Power-law RT model with one molecular transition plus CIA only."""

    def __init__(
        self,
        *args,
        single_line_molecule,
        single_line_center,
        line_strength_temperature,
        **kwargs,
    ):
        self.single_line_molecule = single_line_molecule
        self.single_line_center = float(single_line_center)
        self.line_strength_temperature = float(line_strength_temperature)
        self.selected_line_metadata = {}
        super().__init__(*args, **kwargs)

    def _load_molecular_opacities(self, t_low, t_high):
        from exojax.database import MdbExomol
        from exojax.opacity import OpaDirect

        molecule = self.single_line_molecule
        if molecule not in self.molecule_paths:
            raise KeyError(f"Unknown molecule: {molecule}")
        mdb = MdbExomol(
            self.molecule_paths[molecule],
            nurange=self.nu_grid,
            Ttyp=self.line_strength_temperature,
            broadf=False,
            broadf_download=False,
            gpu_transfer=False,
            elower_max=EXOMOL_ELOWER_MAX.get(molecule),
        )
        line_wavelengths = np.asarray(_wavenumber_to_wavelength(mdb.nu_lines))
        strengths = np.asarray(mdb.line_strength(self.line_strength_temperature))
        line_index = int(np.argmin(np.abs(line_wavelengths - self.single_line_center)))
        mask = np.zeros(len(line_wavelengths), dtype=bool)
        mask[line_index] = True
        self.selected_line_metadata = {
            "molecule": molecule,
            "requested_center_wavelength": float(self.single_line_center),
            "selected_center_wavelength": float(line_wavelengths[line_index]),
            "selected_wavenumber": float(mdb.nu_lines[line_index]),
            "line_strength": float(strengths[line_index]),
            "line_strength_temperature": float(self.line_strength_temperature),
        }
        molmass = float(mdb.molmass)
        _apply_mdb_mask_in_place(mdb, mask)
        opa = OpaDirect(mdb=mdb, nu_grid=self.nu_grid)
        return {molecule: opa}, {molecule: molmass}

    def _abundances(self, zeta_vmr=0.0, log_vmr_values=None):
        jnp = self.jnp
        params = self.parameters
        molecule = self.single_line_molecule
        log_vmr_name = VMR_PARAMETER_NAMES[molecule]
        if log_vmr_values is None:
            log_vmr = getattr(params, log_vmr_name) + zeta_vmr
        else:
            log_vmr = log_vmr_values[molecule]
        vmr = 10.0 ** log_vmr
        vmr_h2 = (1.0 - vmr) * params.h2_fraction_ratio
        vmr_he = (1.0 - vmr) * params.he_fraction_ratio
        molmass_h2 = self.molinfo.molmass_isotope("H2")
        molmass_he = self.molinfo.molmass_isotope("He", db_HIT=False)
        mmw = vmr_h2 * molmass_h2 + vmr_he * molmass_he + vmr * self.mol_masses[molecule]
        mmr = {
            molecule: jnp.asarray(vmr * self.mol_masses[molecule] / mmw),
        }
        return mmr, jnp.asarray(vmr_h2), jnp.asarray(vmr_he), jnp.asarray(mmw)

    def _dtau_molecular_and_cia(
        self,
        temperature,
        gravity,
        zeta_vmr=0.0,
        log_vmr_values=None,
    ):
        mmr, vmr_h2, vmr_he, mmw = self._abundances(
            zeta_vmr=zeta_vmr,
            log_vmr_values=log_vmr_values,
        )
        dtau = 0.0
        for molecule, opa in self.opacities.items():
            xsmatrix = opa.xsmatrix(temperature, self.art.pressure)
            profile = self.art.constant_mmr_profile(mmr[molecule])
            dtau = dtau + self.art.opacity_profile_xs(
                xsmatrix, profile, self.mol_masses[molecule], gravity
            )
        dtau = dtau + self.art.opacity_profile_cia(
            self.opa_cia_h2h2.logacia_matrix(temperature),
            temperature,
            vmr_h2,
            vmr_h2,
            mmw,
            gravity,
        )
        dtau = dtau + self.art.opacity_profile_cia(
            self.opa_cia_h2he.logacia_matrix(temperature),
            temperature,
            vmr_h2,
            vmr_he,
            mmw,
            gravity,
        )
        return dtau

    def _evaluate_at(
        self,
        t0,
        alpha,
        log_p_cloud,
        *,
        zeta_vmr=0.0,
        log_vmr_values=None,
        logg=None,
    ):
        from exojax.postproc.response import ipgauss_sampling

        jnp = self.jnp
        params = self.parameters
        temperature = self.art.powerlaw_temperature(t0, alpha)
        gravity = 10.0 ** (params.logg if logg is None else logg)
        dtau = self._dtau_molecular_and_cia(
            temperature,
            gravity,
            zeta_vmr=zeta_vmr,
            log_vmr_values=log_vmr_values,
        )
        dtau = dtau + self._cloud_dtau(log_p_cloud)[:, None]
        flux = self.art.run(dtau, temperature)
        flux = flux / jnp.average(flux)
        return ipgauss_sampling(
            self.sampling_nu,
            self.nu_grid,
            flux,
            self.beta_inst,
            params.rv,
            self.velocity_kernel,
        )

    def cloudy_at_parameters(self, t0, alpha, zeta_vmr, log_p_cloud, logg=None):
        """Return the cloudy local spectrum at explicit shared-VMR parameters."""

        return self._evaluate_at(
            t0,
            alpha,
            log_p_cloud,
            zeta_vmr=zeta_vmr,
            logg=logg,
        )

    def cloudy_at_log_vmrs(
        self,
        t0,
        alpha,
        log_vmr_co,
        log_vmr_h2o,
        log_vmr_ch4,
        log_vmr_hf,
        log_p_cloud,
        logg=None,
    ):
        """Return the cloudy local spectrum at explicit independent VMRs."""

        return self._evaluate_at(
            t0,
            alpha,
            log_p_cloud,
            log_vmr_values={
                "CO": log_vmr_co,
                "H2O": log_vmr_h2o,
                "CH4": log_vmr_ch4,
                "HF": log_vmr_hf,
            },
            logg=logg,
        )


def _response_function(spectrum_function, *, independent_log_vmrs=False, free_logg=False):
    """Return a pressure-response JVP for shared or independent VMR parameters."""

    if independent_log_vmrs:

        def response(
            t0,
            alpha,
            log_vmr_co,
            log_vmr_h2o,
            log_vmr_ch4,
            log_vmr_hf,
            *tail,
        ):
            if free_logg:
                logg, log_p_cloud = tail
            else:
                (log_p_cloud,) = tail
                logg = None

            def spectrum_at_pressure(pressure):
                return spectrum_function(
                    t0,
                    alpha,
                    log_vmr_co,
                    log_vmr_h2o,
                    log_vmr_ch4,
                    log_vmr_hf,
                    pressure,
                    logg=logg,
                )

            return jax.jvp(
                spectrum_at_pressure,
                (log_p_cloud,),
                (jnp.ones_like(log_p_cloud),),
            )

        return response

    def response(t0, alpha, zeta_vmr, *tail):
        if free_logg:
            logg, log_p_cloud = tail
        else:
            (log_p_cloud,) = tail
            logg = None

        def spectrum_at_pressure(pressure):
            return spectrum_function(t0, alpha, zeta_vmr, pressure, logg=logg)

        return jax.jvp(
            spectrum_at_pressure,
            (log_p_cloud,),
            (jnp.ones_like(log_p_cloud),),
        )

    return response


def _summary_median_sample(sample):
    """Return scalar model parameters for summary provenance."""

    return {
        key: float(np.asarray(value))
        for key, value in sample.items()
        if key in SUMMARY_PARAMETER_NAMES and np.asarray(value).ndim == 0
    }


def _model_parameter_provenance():
    """Describe defaults separately from posterior evaluation parameters."""

    return {
        "model_defaults": asdict(YAMA_L16B_EXOMOL_ATMOSPHERE),
        "evaluation_parameters_source": "posterior_median_sample",
    }


def _chip_data_for_window(samples, chip_index, center, half_width):
    """Build chip data on the selected wavelength window."""

    wavelengths = np.asarray(samples[f"wavelengths_chip{chip_index}"], dtype=float)
    flux = np.asarray(samples[f"flux_chip{chip_index}"], dtype=float)
    mask = np.abs(wavelengths - center) <= half_width
    if int(np.sum(mask)) < 3:
        raise ValueError("Selected wavelength window contains fewer than three pixels.")
    return Luhman16BChipData(
        wavelengths=wavelengths[mask],
        flux=np.zeros((flux.shape[0], int(np.sum(mask))), dtype=float),
        line_profile=np.ones(int(np.sum(mask)), dtype=float),
        obs_times=np.asarray(samples["obs_times"], dtype=float),
        chip_index=int(chip_index),
    )


def _padded_profile_wavelengths(wavelengths, saved_velocity):
    """Return a profile grid covering every saved rotational velocity."""

    velocity = np.asarray(saved_velocity, dtype=float)
    if velocity.size == 0 or not np.all(np.isfinite(velocity)):
        raise ValueError("Saved rotational velocities must be finite and non-empty.")
    max_abs_velocity = float(np.max(np.abs(velocity)))
    return (
        doppler_padded_wavelengths(wavelengths, max_abs_velocity),
        max_abs_velocity,
    )


def _apply_mdb_mask_in_place(mdb, mask):
    """Apply a line mask to MDB arrays needed by OpaDirect."""

    for name in (
        "A",
        "logsij0",
        "nu_lines",
        "gamma_natural",
        "alpha_ref",
        "n_Texp",
        "elower",
        "jlower",
        "jupper",
        "line_strength_ref_original",
        "gpp",
    ):
        if hasattr(mdb, name):
            setattr(mdb, name, getattr(mdb, name)[mask])
    if hasattr(mdb, "dev_nu_lines"):
        mdb.dev_nu_lines = mdb.dev_nu_lines[mask]


def _reconstruct_pure_line_prediction(
    args,
    samples,
    product_dir,
    line_metadata,
):
    """Recompute the single-line RT profiles and projected time-series prediction."""

    chip_index = int(line_metadata["chip_index"])
    center = float(line_metadata["center_wavelength"])
    molecule = line_metadata["molecule"]
    temperature = (
        float(args.line_strength_temperature)
        if args.line_strength_temperature is not None
        else float(np.nanmedian(np.asarray(samples["T0"], dtype=float)))
    )
    chip_indices = [int(value) for value in np.asarray(samples["chip_indices"])]
    chip_position = chip_indices.index(chip_index)
    nside = int(np.asarray(samples["nside"])) if args.nside is None else args.nside
    geometry = _build_product_geometry(samples, nside, args.x64)
    radial_velocity = float(YAMA_L16B_EXOMOL_ATMOSPHERE.rv)
    observed_center = _observed_frame_center(center, radial_velocity)
    chip_data = _chip_data_for_window(
        samples,
        chip_index,
        observed_center,
        args.window_half_width,
    )
    profile_wavelengths, max_abs_velocity = _padded_profile_wavelengths(
        chip_data.wavelengths,
        samples["v"],
    )
    model = SingleLinePowerLawColumnModel(
        chip_data.wavelengths,
        molecule_paths=_molecule_paths(args.database_dir),
        cia_paths=_cia_paths(args.database_dir),
        opacity_cache_dir=args.opacity_cache_dir,
        parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
        nx=args.nx,
        single_line_molecule=molecule,
        single_line_center=center,
        line_strength_temperature=temperature,
        sampling_wavelengths=profile_wavelengths,
    )
    sample = fixed_two_column_median_sample(samples)
    has_independent_log_vmrs = all(name in sample for name in LOG_VMR_NAMES)
    has_free_logg = "logg" in sample
    spectrum_function = (
        model.cloudy_at_log_vmrs
        if has_independent_log_vmrs
        else model.cloudy_at_parameters
    )
    response = _response_function(
        spectrum_function,
        independent_log_vmrs=has_independent_log_vmrs,
        free_logg=has_free_logg,
    )
    chip_sample = _chip_sample(sample, chip_position)
    t0 = sample.get("T0", sample.get("t0", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.t0)))
    alpha = sample.get("alpha", jnp.asarray(YAMA_L16B_EXOMOL_ATMOSPHERE.alpha))
    if has_independent_log_vmrs and has_free_logg:
        base_profile, contrast_profile = response(
            jnp.asarray(t0),
            jnp.asarray(alpha),
            jnp.asarray(sample["log_vmr_co"]),
            jnp.asarray(sample["log_vmr_h2o"]),
            jnp.asarray(sample["log_vmr_ch4"]),
            jnp.asarray(sample["log_vmr_hf"]),
            jnp.asarray(sample["logg"]),
            jnp.asarray(sample["log_p_cloud"]),
        )
    elif has_independent_log_vmrs:
        base_profile, contrast_profile = response(
            jnp.asarray(t0),
            jnp.asarray(alpha),
            jnp.asarray(sample["log_vmr_co"]),
            jnp.asarray(sample["log_vmr_h2o"]),
            jnp.asarray(sample["log_vmr_ch4"]),
            jnp.asarray(sample["log_vmr_hf"]),
            jnp.asarray(sample["log_p_cloud"]),
        )
    elif has_free_logg:
        zeta_vmr = sample.get("zeta_vmr", jnp.asarray(0.0))
        base_profile, contrast_profile = response(
            jnp.asarray(t0),
            jnp.asarray(alpha),
            jnp.asarray(zeta_vmr),
            jnp.asarray(sample["logg"]),
            jnp.asarray(sample["log_p_cloud"]),
        )
    else:
        zeta_vmr = sample.get("zeta_vmr", jnp.asarray(0.0))
        base_profile, contrast_profile = response(
            jnp.asarray(t0),
            jnp.asarray(alpha),
            jnp.asarray(zeta_vmr),
            jnp.asarray(sample["log_p_cloud"]),
        )
    baseline, contrast_matrix = _linear_profile_operator_from_sample(
        chip_data,
        geometry,
        base_profile,
        contrast_profile,
        chip_sample,
        profile_wavelengths=profile_wavelengths,
    )
    contrast_map = np.load(product_dir / "contrast_mean_joint.npy")
    prediction = baseline + contrast_matrix @ jnp.asarray(contrast_map)
    prediction = np.asarray(prediction).reshape(chip_data.flux.shape)
    base_profile_padded = np.asarray(base_profile)
    contrast_profile_padded = np.asarray(contrast_profile)
    return {
        "chip_data": chip_data,
        "prediction": prediction,
        "profile_wavelengths": profile_wavelengths,
        "max_abs_velocity_kms": max_abs_velocity,
        "base_profile": np.interp(
            chip_data.wavelengths,
            profile_wavelengths,
            base_profile_padded,
        ),
        "contrast_profile": np.interp(
            chip_data.wavelengths,
            profile_wavelengths,
            contrast_profile_padded,
        ),
        "base_profile_padded": base_profile_padded,
        "contrast_profile_padded": contrast_profile_padded,
        "line_metadata": model.selected_line_metadata,
        "observed_frame_center_wavelength": float(observed_center),
        "radial_velocity_kms": float(radial_velocity),
        "median_sample": _summary_median_sample(sample),
    }


def _plot_pure_line_prediction(path, wavelengths, prediction, center, args):
    """Plot a pure-line prediction with no observed-data overlay."""

    import matplotlib.pyplot as plt

    relative = np.asarray(wavelengths, dtype=float) - float(center)
    mean_prediction = np.mean(prediction, axis=0)
    delta = prediction - mean_prediction[None, :]
    delta_display = args.delta_scale * delta
    scale = float(np.nanpercentile(np.abs(delta_display), 95.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 0.01
    offset_step = args.offset_scale * max(2.0 * scale, 0.01)
    offsets = np.arange(prediction.shape[0])[:, None] * offset_step

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, args.figure_height),
        sharex=True,
        gridspec_kw={
            "height_ratios": [args.top_height_ratio, 1.0],
            "hspace": 0.08,
        },
    )
    if args.color_by_phase:
        cmap = plt.get_cmap(args.phase_cmap)
        colors = [cmap(value) for value in np.linspace(0.0, 1.0, prediction.shape[0])]
    else:
        colors = ["firebrick"] * prediction.shape[0]
    for phase_index in range(prediction.shape[0]):
        axes[0].plot(
            relative,
            delta_display[phase_index] + offsets[phase_index, 0],
            color=colors[phase_index],
            alpha=args.model_alpha,
            linewidth=args.model_linewidth,
        )
    axes[0].axvline(0.0, color="0.6", alpha=0.45, linewidth=0.7)
    axes[0].set_ylabel(f"Delta flux x {args.delta_scale:g} + offset")
    axes[0].set_title("Pure single-line plus CIA prediction")
    axes[0].set_ylim(float(offsets[0, 0] - offset_step), float(offsets[-1, 0] + offset_step))

    axes[1].plot(
        relative,
        mean_prediction,
        color="firebrick",
        alpha=args.model_alpha,
        linewidth=max(args.model_linewidth, 1.2),
    )
    axes[1].axvline(0.0, color="0.6", alpha=0.45, linewidth=0.7)
    axes[1].set_xlabel("Relative wavelength [Angstrom]")
    axes[1].set_ylabel("Mean flux")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary_json(path, summary):
    """Write a standards-compliant JSON summary."""

    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def main():
    """Run the pure single-line plus CIA RT diagnostic."""

    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)
    samples = dict(np.load(args.samples, allow_pickle=False))
    product_dir = Path(args.product_dir)
    out_dir = (
        product_dir / "minimum_window_figures"
        if args.out_dir is None
        else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    line_metadata = _load_single_line_metadata(args)
    result = _reconstruct_pure_line_prediction(
        args,
        samples,
        product_dir,
        line_metadata,
    )
    chip_data = result["chip_data"]
    rest_center = result["line_metadata"]["selected_center_wavelength"]
    observed_center = result["observed_frame_center_wavelength"]
    radial_velocity = result["radial_velocity_kms"]
    result["line_metadata"]["observed_frame_center_wavelength"] = float(observed_center)
    result["line_metadata"]["radial_velocity_kms"] = float(radial_velocity)
    center_label = _format_wavelength(rest_center)
    prefix = f"single_line_rt_prediction_chip{chip_data.chip_index}_lambda{center_label}"
    npz_path = out_dir / f"{prefix}.npz"
    figure_path = out_dir / f"figure9_{prefix}.png"
    relative = np.asarray(chip_data.wavelengths, dtype=float) - observed_center
    np.savez(
        npz_path,
        wavelengths=np.asarray(chip_data.wavelengths),
        relative_wavelengths=relative,
        rest_center_wavelength=np.asarray(rest_center),
        observed_frame_center_wavelength=np.asarray(observed_center),
        prediction=result["prediction"],
        mean_prediction=np.mean(result["prediction"], axis=0),
        mean_subtracted_prediction=(
            result["prediction"] - np.mean(result["prediction"], axis=0, keepdims=True)
        ),
        profile_wavelengths=result["profile_wavelengths"],
        max_abs_velocity_kms=np.asarray(result["max_abs_velocity_kms"]),
        base_profile=result["base_profile"],
        pressure_response_profile=result["contrast_profile"],
        base_profile_padded=result["base_profile_padded"],
        pressure_response_profile_padded=result["contrast_profile_padded"],
    )
    _plot_pure_line_prediction(
        figure_path,
        chip_data.wavelengths,
        result["prediction"],
        observed_center,
        args,
    )
    summary = {
        "samples": str(args.samples),
        "product_dir": str(product_dir),
        "output_npz": str(npz_path),
        "output_figure": str(figure_path),
        "input_line_metadata": line_metadata,
        "selected_line_metadata": result["line_metadata"],
        "window_half_width": args.window_half_width,
        "max_abs_velocity_kms": result["max_abs_velocity_kms"],
        "profile_wavelength_count": int(len(result["profile_wavelengths"])),
        "retrieval_x64": bool(np.asarray(samples.get("x64", False)).item()),
        "product_x64": bool(args.x64),
        "geometry_method": "retrieval_precision_constants_promoted",
        **_model_parameter_provenance(),
        "median_sample": result["median_sample"],
    }
    summary_path = out_dir / f"{prefix}.json"
    _write_summary_json(summary_path, summary)
    print(f"Pure single-line RT prediction saved to {figure_path}")


if __name__ == "__main__":
    main()
