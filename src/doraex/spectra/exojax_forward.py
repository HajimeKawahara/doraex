"""Generic ExoJAX forward-model adapters."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FixedPowerLawAtmosphere:
    """Fixed power-law atmospheric parameters for a two-column run."""

    t0: float = 1215.0
    alpha: float = 0.128
    logg: float = 4.96
    log_vmr_co: float = -2.86
    log_vmr_h2o: float = -3.16
    log_vmr_ch4: float = -4.61
    log_vmr_hf: float = -7.00
    rv: float = 25.54
    log_p_cloud: float = 1.35
    cloud_width: float = 0.3
    cloud_column_optical_depth: float = 500.0
    h2_fraction_ratio: float = 6.0 / 7.0
    he_fraction_ratio: float = 1.0 / 7.0


def synthetic_two_column_profiles(wavelengths, line_center=None):
    """Build lightweight fixed clear/cloudy spectra for smoke tests.

    Args:
        wavelengths: One-dimensional wavelength grid.
        line_center: Optional center of the synthetic absorption feature. If
            omitted, the midpoint of ``wavelengths`` is used.

    Returns:
        A tuple ``(clear_profile, cloudy_profile)`` sampled on ``wavelengths``.
    """

    wavelengths = np.asarray(wavelengths)
    center = float(np.mean(wavelengths) if line_center is None else line_center)
    width = 0.12 * (float(np.max(wavelengths)) - float(np.min(wavelengths)))
    if width <= 0.0:
        width = 1.0
    gaussian = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    clear = 1.0 - 0.10 * gaussian
    cloudy = 0.94 - 0.16 * np.exp(-0.5 * ((wavelengths - center) / (1.25 * width)) ** 2)
    return clear, cloudy


def synthetic_cloud_profile_grid(wavelengths, log_p_cloud_grid, line_center=None):
    """Build lightweight cloudy-profile grid for Milestone 2-2 smoke tests.

    Args:
        wavelengths: One-dimensional wavelength grid.
        log_p_cloud_grid: Grid of ``log10 Pc`` values.
        line_center: Optional center of the synthetic absorption feature. If
            omitted, the midpoint of ``wavelengths`` is used.

    Returns:
        A tuple ``(clear_profile, cloudy_profile_grid)``. The cloudy grid has
        shape ``(n_log_p_cloud, n_wavelength)``.
    """

    wavelengths = np.asarray(wavelengths)
    log_p_cloud_grid = np.asarray(log_p_cloud_grid)
    clear_profile, _ = synthetic_two_column_profiles(
        wavelengths, line_center=line_center
    )
    center = float(np.mean(wavelengths) if line_center is None else line_center)
    width = 0.12 * (float(np.max(wavelengths)) - float(np.min(wavelengths)))
    if width <= 0.0:
        width = 1.0
    profiles = []
    for log_p_cloud in log_p_cloud_grid:
        depth_scale = 0.75 + 0.25 * np.tanh(float(log_p_cloud) - 1.0)
        width_scale = 1.15 + 0.15 * np.tanh(1.0 - float(log_p_cloud))
        continuum = 0.96 - 0.015 * np.tanh(float(log_p_cloud) - 1.0)
        gaussian = np.exp(
            -0.5 * ((wavelengths - center) / (width_scale * width)) ** 2
        )
        profiles.append(continuum - 0.16 * depth_scale * gaussian)
    return clear_profile, np.asarray(profiles)


def save_two_column_profiles(path, wavelengths, clear_profile, cloudy_profile, metadata=None):
    """Save fixed clear/cloudy profiles to an NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wavelengths": np.asarray(wavelengths),
        "clear_profile": np.asarray(clear_profile),
        "cloudy_profile": np.asarray(cloudy_profile),
    }
    if metadata:
        for key, value in metadata.items():
            payload[key] = np.asarray(value)
    np.savez(path, **payload)


def save_cloud_profile_grid(
    path,
    wavelengths,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    metadata=None,
):
    """Save clear spectrum and cloudy profile grid to an NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wavelengths": np.asarray(wavelengths),
        "clear_profile": np.asarray(clear_profile),
        "log_p_cloud_grid": np.asarray(log_p_cloud_grid),
        "cloudy_profile_grid": np.asarray(cloudy_profile_grid),
    }
    if metadata:
        for key, value in metadata.items():
            payload[key] = np.asarray(value)
    np.savez(path, **payload)


def load_two_column_profiles(path, expected_wavelengths=None):
    """Load fixed clear/cloudy profiles from an NPZ file.

    Args:
        path: NPZ file containing ``wavelengths``, ``clear_profile``, and
            ``cloudy_profile``.
        expected_wavelengths: Optional grid used to validate the loaded
            profiles.

    Returns:
        A tuple ``(clear_profile, cloudy_profile)``.
    """

    profiles = np.load(path)
    wavelengths = profiles["wavelengths"]
    if expected_wavelengths is not None and not np.allclose(
        wavelengths, np.asarray(expected_wavelengths)
    ):
        raise ValueError("Profile wavelength grid does not match the data wavelength grid.")
    return profiles["clear_profile"], profiles["cloudy_profile"]


def load_cloud_profile_grid(path, expected_wavelengths=None):
    """Load clear spectrum and cloudy profile grid from an NPZ file.

    Args:
        path: NPZ file containing ``wavelengths``, ``clear_profile``,
            ``log_p_cloud_grid``, and ``cloudy_profile_grid``.
        expected_wavelengths: Optional grid used to validate loaded profiles.

    Returns:
        A tuple ``(clear_profile, log_p_cloud_grid, cloudy_profile_grid)``.
    """

    profiles = np.load(path)
    wavelengths = profiles["wavelengths"]
    if expected_wavelengths is not None and not np.allclose(
        wavelengths, np.asarray(expected_wavelengths)
    ):
        raise ValueError("Profile wavelength grid does not match the data wavelength grid.")
    return (
        profiles["clear_profile"],
        profiles["log_p_cloud_grid"],
        profiles["cloudy_profile_grid"],
    )


class Luhman16BPowerLawColumnModel:
    """ExoJAX fixed-power-law clear/cloudy local spectrum generator.

    This class is intentionally imported lazily by production scripts because
    ExoJAX opacity setup can require large molecular databases and substantial
    memory. It is not used by unit tests.
    """

    def __init__(
        self,
        observed_wavelengths,
        molecule_paths,
        cia_paths,
        opacity_cache_dir,
        parameters=FixedPowerLawAtmosphere(),
        nx=4500,
        pressure_top=1.0e-4,
        pressure_btm=1.0e2,
        nlayer=101,
        t_low=210.0,
        t_high=3500.0,
        resolving_power=100000.0,
    ):
        """Initialize the ExoJAX objects for fixed-profile generation."""

        self.observed_wavelengths = np.asarray(observed_wavelengths)
        self.parameters = parameters
        self.molecule_paths = {key: str(value) for key, value in molecule_paths.items()}
        self.cia_paths = {key: str(value) for key, value in cia_paths.items()}
        self.opacity_cache_dir = Path(opacity_cache_dir)
        self.opacity_cache_dir.mkdir(parents=True, exist_ok=True)

        import jax.numpy as jnp
        from exojax.database import molinfo
        from exojax.database.cia.api import CdbCIA
        from exojax.opacity.opacont import OpaCIA
        from exojax.rt.emis import ArtEmisPure
        from exojax.utils.grids import wavenumber_grid
        from exojax.utils.grids import velocity_grid
        from exojax.utils.instfunc import resolution_to_gaussian_std

        self.jnp = jnp
        self.molinfo = molinfo
        self.nu_grid, self.wav_grid, resolution = wavenumber_grid(
            np.min(self.observed_wavelengths) - 5.0,
            np.max(self.observed_wavelengths) + 5.0,
            nx,
            unit="AA",
            xsmode="premodit",
        )
        self.observed_nu = jnp.asarray(1.0e8 / self.observed_wavelengths)
        self.art = ArtEmisPure(
            nu_grid=self.nu_grid,
            pressure_top=pressure_top,
            pressure_btm=pressure_btm,
            nlayer=nlayer,
        )
        self.art.change_temperature_range(t_low, t_high)
        self.beta_inst = resolution_to_gaussian_std(resolving_power)
        self.velocity_kernel = velocity_grid(resolution, 100.0)
        self.opacities, self.mol_masses = self._load_molecular_opacities(t_low, t_high)
        self.opa_cia_h2h2 = OpaCIA(
            cdb=CdbCIA(self.cia_paths["H2H2"], self.nu_grid), nu_grid=self.nu_grid
        )
        self.opa_cia_h2he = OpaCIA(
            cdb=CdbCIA(self.cia_paths["H2He"], self.nu_grid), nu_grid=self.nu_grid
        )

    def _load_molecular_opacities(self, t_low, t_high):
        from exojax.database import MdbExomol
        from exojax.opacity import OpaPremodit, saveopa

        elower_max = {
            "CO": 58242.689,
            "H2O": 23726.625476,
            "CH4": 9900.0,
            "HF": 20000.0,
        }
        opacities = {}
        mol_masses = {}
        for molecule, max_elower in elower_max.items():
            cache_path = self.opacity_cache_dir / f"opa{molecule}.zarr"
            if cache_path.exists():
                opa = OpaPremodit.from_saved_opa(str(cache_path))
                molmass = opa.aux["molmass"]
            else:
                mdb = MdbExomol(
                    self.molecule_paths[molecule],
                    nurange=self.nu_grid,
                    gpu_transfer=False,
                    elower_max=max_elower,
                )
                molmass = mdb.molmass
                snapshot = mdb.to_snapshot()
                del mdb
                opa = OpaPremodit.from_snapshot(
                    snapshot,
                    nu_grid=self.nu_grid,
                    diffmode=0,
                    auto_trange=[t_low, t_high],
                    dit_grid_resolution=1.0,
                )
                saveopa(opa, str(cache_path), format="zarr", aux={"molmass": molmass})
            opacities[molecule] = opa
            mol_masses[molecule] = molmass
        return opacities, mol_masses

    def _abundances(self):
        jnp = self.jnp
        params = self.parameters
        vmr = {
            "CO": 10.0**params.log_vmr_co,
            "H2O": 10.0**params.log_vmr_h2o,
            "CH4": 10.0**params.log_vmr_ch4,
            "HF": 10.0**params.log_vmr_hf,
        }
        heavy_sum = sum(vmr.values())
        vmr_h2 = (1.0 - heavy_sum) * params.h2_fraction_ratio
        vmr_he = (1.0 - heavy_sum) * params.he_fraction_ratio
        molmass_h2 = self.molinfo.molmass_isotope("H2")
        molmass_he = self.molinfo.molmass_isotope("He", db_HIT=False)
        mmw = (
            vmr_h2 * molmass_h2
            + vmr_he * molmass_he
            + sum(vmr[name] * self.mol_masses[name] for name in vmr)
        )
        mmr = {
            name: jnp.asarray(vmr[name] * self.mol_masses[name] / mmw)
            for name in vmr
        }
        return mmr, jnp.asarray(vmr_h2), jnp.asarray(vmr_he), jnp.asarray(mmw)

    def _dtau_molecular_and_cia(self, temperature, gravity):
        jnp = self.jnp
        mmr, vmr_h2, vmr_he, mmw = self._abundances()
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

    def _cloud_dtau(self):
        jnp = self.jnp
        params = self.parameters
        log_pressure = jnp.log10(self.art.pressure)
        norm = params.cloud_column_optical_depth / (
            jnp.sqrt(2.0 * jnp.pi) * params.cloud_width
        )
        return norm * jnp.exp(
            -((log_pressure - params.log_p_cloud) ** 2)
            / (2.0 * params.cloud_width**2)
        )

    def evaluate(self, cloudy):
        """Evaluate one fixed local spectrum."""

        from exojax.postproc.response import ipgauss_sampling

        jnp = self.jnp
        params = self.parameters
        temperature = self.art.powerlaw_temperature(params.t0, params.alpha)
        gravity = 10.0**params.logg
        dtau = self._dtau_molecular_and_cia(temperature, gravity)
        if cloudy:
            dtau = dtau + self._cloud_dtau()[:, None]
        flux = self.art.run(dtau, temperature)
        flux = flux / jnp.average(flux)
        return ipgauss_sampling(
            self.observed_nu,
            self.nu_grid,
            flux,
            self.beta_inst,
            params.rv,
            self.velocity_kernel,
        )

    def clear(self):
        """Return the fixed clear-sky local spectrum."""

        return self.evaluate(cloudy=False)

    def cloudy(self):
        """Return the fixed cloudy local spectrum."""

        return self.evaluate(cloudy=True)
