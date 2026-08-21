"""Unit tests for the ExoJAX forward-model adapter."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from doraex.constants import SPEED_OF_LIGHT
from doraex.operators.doppler import doppler_factor
from doraex.spectra.exojax_forward import (
    FixedPowerLawAtmosphere,
    Luhman16BPowerLawColumnModel,
    _internal_spectrum_wavelength_bounds,
)
from doraex.workflows import on_the_fly_pressure_retrieval
from doraex.workflows.on_the_fly_pressure_retrieval import (
    build_chip_data,
    build_doppler_profile_wavelengths,
    build_observation_masks,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXAMPLE_DIR = ROOT / "examples" / "luhman16b_yama"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import m6_v1_run  # noqa: E402
import make_milestone5_single_line_rt_prediction as single_line_product  # noqa: E402


class _Archive(dict):
    """Minimal in-memory stand-in for a NumPy ``NpzFile``."""

    @property
    def files(self):
        return list(self)


def test_single_line_summary_json_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        single_line_product._write_summary_json(path, {"value": np.nan})

    assert not path.exists()


def test_single_line_median_sample_ignores_empty_arrays():
    samples = {
        "A": np.arange(6.0).reshape(3, 2),
        "T0": np.asarray([1100.0, 1200.0, 1300.0]),
        "q1": np.asarray([0.7, 0.8, 0.9]),
        "q2": np.asarray([0.4, 0.5, 0.6]),
        "fixed_sigma_eigen": np.empty(0),
    }

    sample = single_line_product.fixed_two_column_median_sample(samples)

    assert "fixed_sigma_eigen" not in sample
    np.testing.assert_allclose(sample["T0"], 1200.0)
    np.testing.assert_allclose(sample["A"], [2.0, 3.0])


def test_single_line_summary_keeps_only_scalar_model_parameters():
    summary = single_line_product._summary_median_sample(
        {
            "T0": np.asarray(1200.0),
            "A": np.asarray([1.0, 1.1]),
            "extra_accept_prob": np.asarray(0.95),
            "pressure_gp_eigenvalues": np.asarray(1.0),
            "wavelengths_chip0": np.asarray(23000.0),
        }
    )

    assert summary == {"T0": 1200.0}


def test_single_line_summary_separates_defaults_from_evaluation_parameters():
    provenance = single_line_product._model_parameter_provenance()

    assert "model_parameters" not in provenance
    assert provenance["model_defaults"]["logg"] == 4.97
    assert provenance["evaluation_parameters_source"] == "posterior_median_sample"


class _StubColumnModel:
    """Avoid ExoJAX setup while exercising response archive validation."""

    def __init__(self, observed_wavelengths, *, sampling_wavelengths=None, **kwargs):
        self.wavelengths = np.asarray(
            observed_wavelengths
            if sampling_wavelengths is None
            else sampling_wavelengths
        )

    def cloudy_at_log_vmrs(self, *args, **kwargs):
        return jnp.ones(len(self.wavelengths))


@pytest.fixture(params=(on_the_fly_pressure_retrieval, m6_v1_run))
def response_workflow(request, monkeypatch):
    monkeypatch.setattr(
        request.param,
        "Luhman16BPowerLawColumnModel",
        _StubColumnModel,
    )
    return request.param


def _frozen_basis(payload, mode_count=2):
    return {
        "path": "basis.npz",
        "npz": _Archive(payload),
        "mode_count": mode_count,
    }


def _response_args():
    return SimpleNamespace(
        frozen_eigen_spectra=True,
        database_dir=".",
        opacity_cache_dir=".",
        nx=8,
    )


def _chip(wavelengths):
    return SimpleNamespace(chip_index=0, wavelengths=np.asarray(wavelengths))


def test_cloud_column_optical_depth_is_layer_resolution_independent():
    """The integrated cloud optical depth must not scale with layer count."""

    parameters = FixedPowerLawAtmosphere(
        log_p_cloud=-1.0,
        cloud_width=0.3,
        cloud_column_optical_depth=500.0,
    )
    totals = []
    for nlayer in (51, 101, 201):
        log_pressure = np.linspace(-4.0, 2.0, nlayer)
        layer_width = log_pressure[1] - log_pressure[0]
        log_pressure_boundary = np.linspace(
            log_pressure[0] - 0.5 * layer_width,
            log_pressure[-1] + 0.5 * layer_width,
            nlayer + 1,
        )
        model = SimpleNamespace(
            jnp=jnp,
            parameters=parameters,
            art=SimpleNamespace(
                pressure=10.0**log_pressure,
                pressure_boundary=10.0**log_pressure_boundary,
            ),
        )
        dtau = Luhman16BPowerLawColumnModel._cloud_dtau(model)
        totals.append(float(jnp.sum(dtau)))

    np.testing.assert_allclose(
        totals,
        parameters.cloud_column_optical_depth,
        rtol=2.0e-5,
    )


def test_cold_opacity_cache_allows_32bit_generation(monkeypatch, tmp_path):
    """Cold-cache construction must support the production 32-bit mode."""

    calls = []

    class _StubMdbExomol:
        def __init__(self, *args, **kwargs):
            self.molmass = 1.0

        def to_snapshot(self):
            return object()

    class _StubOpaPremodit:
        @classmethod
        def from_snapshot(cls, snapshot, **kwargs):
            calls.append(kwargs)
            return object()

    exojax_module = ModuleType("exojax")
    database_module = ModuleType("exojax.database")
    database_module.MdbExomol = _StubMdbExomol
    opacity_module = ModuleType("exojax.opacity")
    opacity_module.OpaPremodit = _StubOpaPremodit
    opacity_module.saveopa = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "exojax", exojax_module)
    monkeypatch.setitem(sys.modules, "exojax.database", database_module)
    monkeypatch.setitem(sys.modules, "exojax.opacity", opacity_module)

    model = object.__new__(Luhman16BPowerLawColumnModel)
    model.opacity_cache_dir = tmp_path
    model.molecule_paths = {
        molecule: f"{molecule}.database"
        for molecule in ("CO", "H2O", "CH4", "HF")
    }
    model.nu_grid = np.linspace(1.0, 2.0, 4)
    model.save_opacity_cache = False

    opacities, mol_masses = model._load_molecular_opacities(210.0, 3500.0)

    assert set(opacities) == {"CO", "H2O", "CH4", "HF"}
    assert set(mol_masses) == set(opacities)
    assert len(calls) == len(opacities)
    assert all(call["allow_32bit"] is True for call in calls)


def test_smoke_data_use_native_spacing_for_doppler_profile_padding():
    args = SimpleNamespace(
        chip_indices=(1,),
        data_dir=DATA_DIR,
        full_data=False,
        smoke_wavelength_step=128,
        smoke_phase_count=4,
        fixed_v=31.2,
    )
    chip_data_list, native_wavelengths = build_chip_data(
        args,
        return_native_wavelengths=True,
    )
    assert len(chip_data_list[0].wavelengths) == 8
    assert len(native_wavelengths[0]) == 1024
    for max_abs_velocity in (args.fixed_v, 120.0):
        profile_wavelengths = build_doppler_profile_wavelengths(
            native_wavelengths,
            max_abs_velocity,
        )
        assert len(profile_wavelengths[0]) > len(native_wavelengths[0])

        bounds = _internal_spectrum_wavelength_bounds(
            chip_data_list[0].wavelengths,
            profile_wavelengths[0],
            systemic_rv=FixedPowerLawAtmosphere().rv,
        )
        rv_factor = 1.0 + FixedPowerLawAtmosphere().rv / (
            SPEED_OF_LIGHT * 1.0e-3
        )
        source_wavelengths = np.asarray(profile_wavelengths[0]) / rv_factor
        assert bounds[0] <= np.min(source_wavelengths) - 5.0
        assert bounds[1] >= np.max(source_wavelengths) + 5.0


@pytest.mark.skipif(
    not (DATA_DIR / "fainterspectral-fits_6.pickle").exists(),
    reason="Luhman 16B spectra are not available.",
)
def test_full_data_zero_flux_sentinel_mask_matches_detector_edges():
    args = SimpleNamespace(
        chip_indices=(0, 1, 2, 3),
        data_dir=DATA_DIR,
        full_data=True,
        smoke_wavelength_step=128,
        smoke_phase_count=4,
    )
    chip_data_list = build_chip_data(args)
    masks = build_observation_masks(chip_data_list, mask_zero_flux=True)

    assert [int(mask.size - np.count_nonzero(mask)) for mask in masks] == [
        0,
        0,
        56,
        56,
    ]
    for mask in masks[2:]:
        excluded = np.argwhere(~mask)
        np.testing.assert_array_equal(np.unique(excluded[:, 0]), np.arange(14))
        np.testing.assert_array_equal(
            np.unique(excluded[:, 1]),
            [0, 1, 1022, 1023],
        )


def test_single_line_model_samples_rt_on_profile_grid(monkeypatch):
    """The single-line subclass must query RT on its padded sampling grid."""

    import exojax.postproc.response as exojax_response

    captured = {}

    def _stub_ipgauss_sampling(query_nu, *args):
        captured["query_nu"] = np.asarray(query_nu)
        return query_nu

    monkeypatch.setattr(exojax_response, "ipgauss_sampling", _stub_ipgauss_sampling)
    monkeypatch.setattr(
        single_line_product.SingleLinePowerLawColumnModel,
        "_dtau_molecular_and_cia",
        lambda self, temperature, gravity, **kwargs: jnp.zeros((2, 6)),
    )
    monkeypatch.setattr(
        single_line_product.SingleLinePowerLawColumnModel,
        "_cloud_dtau",
        lambda self, log_p_cloud=None: jnp.zeros(2),
    )

    model = object.__new__(single_line_product.SingleLinePowerLawColumnModel)
    model.jnp = jnp
    model.parameters = SimpleNamespace(logg=4.86, rv=25.66)
    model.art = SimpleNamespace(
        powerlaw_temperature=lambda t0, alpha: jnp.asarray([t0, t0 * alpha]),
        run=lambda dtau, temperature: jnp.linspace(1.0, 2.0, 6),
    )
    model.observed_nu = jnp.asarray([10.0, 11.0, 12.0])
    model.sampling_nu = jnp.asarray([9.0, 10.0, 11.0, 12.0, 13.0])
    model.nu_grid = jnp.linspace(8.0, 14.0, 6)
    model.beta_inst = jnp.asarray(1.0)
    model.velocity_kernel = jnp.ones(3)

    output = model._evaluate_at(1200.0, 0.1, 1.0)

    np.testing.assert_array_equal(captured["query_nu"], model.sampling_nu)
    np.testing.assert_array_equal(output, model.sampling_nu)
    assert output.shape != model.observed_nu.shape


def test_single_line_product_propagates_saved_velocity_padding(monkeypatch, tmp_path):
    """Single-line RT and Doppler integration must share one padded grid."""

    captured = {}
    center = 23220.85
    observed_center = single_line_product._observed_frame_center(
        center,
        single_line_product.YAMA_L16B_EXOMOL_ATMOSPHERE.rv,
    )
    wavelengths = np.linspace(observed_center - 6.0, observed_center + 6.0, 25)
    samples = {
        "T0": np.asarray([1200.0, 1210.0]),
        "chip_indices": np.asarray([2]),
        "nside": np.asarray(1),
        "obs_times": np.asarray([0.0, 1.0]),
        "v": np.asarray([31.2, 45.0]),
        "wavelengths_chip2": wavelengths,
        "flux_chip2": np.zeros((2, len(wavelengths))),
    }
    median_sample = {
        "T0": jnp.asarray(1205.0),
        "alpha": jnp.asarray(0.13),
        "log_vmr_co": jnp.asarray(-3.1),
        "log_vmr_h2o": jnp.asarray(-3.4),
        "log_vmr_ch4": jnp.asarray(-4.7),
        "log_vmr_hf": jnp.asarray(-7.3),
        "logg": jnp.asarray(4.86),
        "log_p_cloud": jnp.asarray(1.1),
        "cosi": jnp.asarray(0.485),
        "v": jnp.asarray(31.2),
        "u1": jnp.asarray(1.062),
        "u2": jnp.asarray(-0.162),
        "P": jnp.asarray(4.83),
        "A": jnp.asarray([1.1]),
        "log_w": jnp.zeros((1, 2)),
    }

    class _StubSingleLineModel:
        def __init__(self, observed_wavelengths, *, sampling_wavelengths, **kwargs):
            captured["model_observed_wavelengths"] = np.asarray(observed_wavelengths)
            captured["model_profile_wavelengths"] = np.asarray(sampling_wavelengths)
            self.profile_wavelengths = jnp.asarray(sampling_wavelengths)
            self.selected_line_metadata = {
                "selected_center_wavelength": center,
                "molecule": "CO",
            }

        def cloudy_at_log_vmrs(
            self,
            t0,
            alpha,
            log_vmr_co,
            log_vmr_h2o,
            log_vmr_ch4,
            log_vmr_hf,
            log_p_cloud,
            *,
            logg,
        ):
            return 1.0 + 1.0e-4 * self.profile_wavelengths + log_p_cloud

    def _stub_operator(
        chip_data,
        geometry,
        base_profile,
        contrast_profile,
        sample,
        profile_wavelengths=None,
    ):
        captured["operator_profile_wavelengths"] = np.asarray(profile_wavelengths)
        captured["operator_base_profile"] = np.asarray(base_profile)
        captured["operator_contrast_profile"] = np.asarray(contrast_profile)
        data_size = chip_data.flux.size
        return jnp.ones(data_size), jnp.zeros((data_size, 2))

    monkeypatch.setattr(
        single_line_product,
        "SingleLinePowerLawColumnModel",
        _StubSingleLineModel,
    )
    monkeypatch.setattr(
        single_line_product,
        "fixed_two_column_median_sample",
        lambda values: median_sample,
    )
    monkeypatch.setattr(
        single_line_product,
        "_build_product_geometry",
        lambda values, nside, product_x64: object(),
    )
    monkeypatch.setattr(
        single_line_product,
        "_linear_profile_operator_from_sample",
        _stub_operator,
    )
    np.save(tmp_path / "contrast_mean_joint.npy", np.zeros(2))
    args = SimpleNamespace(
        line_strength_temperature=None,
        nside=None,
        window_half_width=5.0,
        database_dir=".",
        opacity_cache_dir=".",
        nx=8,
        x64=True,
    )

    result = single_line_product._reconstruct_pure_line_prediction(
        args,
        samples,
        tmp_path,
        {"chip_index": 2, "molecule": "CO", "center_wavelength": center},
    )

    native_wavelengths = np.asarray(result["chip_data"].wavelengths)
    profile_wavelengths = np.asarray(result["profile_wavelengths"])
    np.testing.assert_array_equal(
        captured["model_observed_wavelengths"],
        native_wavelengths,
    )
    np.testing.assert_array_equal(
        captured["model_profile_wavelengths"],
        profile_wavelengths,
    )
    np.testing.assert_array_equal(
        captured["operator_profile_wavelengths"],
        profile_wavelengths,
    )
    assert len(profile_wavelengths) > len(native_wavelengths)
    assert captured["operator_base_profile"].shape == profile_wavelengths.shape
    assert captured["operator_contrast_profile"].shape == profile_wavelengths.shape
    assert result["base_profile"].shape == native_wavelengths.shape
    assert result["contrast_profile"].shape == native_wavelengths.shape
    assert result["max_abs_velocity_kms"] == 45.0

    factors = np.asarray(doppler_factor(jnp.asarray([-45.0, 45.0])))
    queries = native_wavelengths[:, None] / factors[None, :]
    assert profile_wavelengths[0] < float(np.min(queries))
    assert profile_wavelengths[-1] > float(np.max(queries))


def test_legacy_frozen_basis_does_not_require_profile_grid_key(response_workflow):
    wavelengths = np.linspace(100.0, 103.0, 4)
    basis = _frozen_basis(
        {"eigenspectra_chip0": np.arange(8.0).reshape(4, 2)}
    )

    response = response_workflow.build_response_functions(
        _response_args(),
        [_chip(wavelengths)],
        eigen_basis=basis,
    )[0]
    _, eigen_profiles = response(*([0.0] * 8))

    assert eigen_profiles.shape == (2, 4)


def test_padded_frozen_basis_requires_profile_grid_key(response_workflow):
    wavelengths = np.linspace(100.0, 103.0, 4)
    profile_wavelengths = np.linspace(99.0, 104.0, 6)
    basis = _frozen_basis(
        {"eigenspectra_chip0": np.arange(12.0).reshape(6, 2)}
    )

    with pytest.raises(ValueError, match="Missing profile_wavelengths_chip0"):
        response_workflow.build_response_functions(
            _response_args(),
            [_chip(wavelengths)],
            eigen_basis=basis,
            profile_wavelengths=[profile_wavelengths],
        )


def test_padded_frozen_basis_requires_exact_profile_grid_match(response_workflow):
    wavelengths = np.linspace(100.0, 103.0, 4)
    profile_wavelengths = np.linspace(99.0, 104.0, 6)
    basis = _frozen_basis(
        {
            "eigenspectra_chip0": np.arange(12.0).reshape(6, 2),
            "profile_wavelengths_chip0": profile_wavelengths + 1.0e-12,
        }
    )

    with pytest.raises(ValueError, match="does not match"):
        response_workflow.build_response_functions(
            _response_args(),
            [_chip(wavelengths)],
            eigen_basis=basis,
            profile_wavelengths=[profile_wavelengths],
        )


def test_frozen_basis_requires_all_selected_eigenmodes(response_workflow):
    wavelengths = np.linspace(100.0, 103.0, 4)
    basis = _frozen_basis(
        {"eigenspectra_chip0": np.arange(4.0).reshape(4, 1)}
    )

    with pytest.raises(ValueError, match=r"must have shape \(2, 4\)"):
        response_workflow.build_response_functions(
            _response_args(),
            [_chip(wavelengths)],
            eigen_basis=basis,
        )
