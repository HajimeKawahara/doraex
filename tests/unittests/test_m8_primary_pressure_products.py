"""Focused tests for M8 pressure-map primary-product reconstruction."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "luhman16b_yama"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import make_milestone4_on_the_fly_products as products  # noqa: E402


def _chip_data(chip_index, flux):
    flux = np.asarray(flux, dtype=float)
    wavelengths = np.linspace(2.0, 2.1, flux.shape[1])
    return products.Luhman16BChipData(
        wavelengths=wavelengths,
        flux=flux,
        line_profile=np.ones_like(wavelengths),
        obs_times=np.arange(flux.shape[0], dtype=float),
        chip_index=chip_index,
    )


def _masked_archive(chip_data_list):
    masks = [np.asarray(chip.flux) != 0.0 for chip in chip_data_list]
    flat_mask = np.concatenate([mask.reshape(-1) for mask in masks])
    archive = {
        "chip_indices": np.asarray(
            [chip.chip_index for chip in chip_data_list],
            dtype=int,
        ),
        "mask_zero_flux": np.asarray(True),
        "observation_mask_rule": np.asarray("finite_and_nonzero_flux"),
        "observation_valid_count": np.asarray(np.count_nonzero(flat_mask)),
        "observation_excluded_count": np.asarray(
            flat_mask.size - np.count_nonzero(flat_mask)
        ),
        "observation_indices": np.flatnonzero(flat_mask),
    }
    for chip, mask in zip(chip_data_list, masks):
        archive[f"observation_mask_chip{chip.chip_index}"] = mask
    return archive


def test_observation_mask_archive_uses_exact_zero_and_preserves_legacy_default():
    chip_data_list = [
        _chip_data(0, [[1.0, 1.0e-300, -0.2], [0.8, 0.9, 1.1]]),
        _chip_data(2, [[0.0, 0.7, -0.0], [0.6, 0.5, 0.4]]),
    ]
    archive = _masked_archive(chip_data_list)

    masks, summary = products._load_observation_masks(archive, chip_data_list)

    np.testing.assert_array_equal(masks[0], np.ones((2, 3), dtype=bool))
    np.testing.assert_array_equal(
        masks[1],
        np.asarray([[False, True, False], [True, True, True]]),
    )
    assert summary["observation_valid_count"] == 10
    assert summary["observation_excluded_count"] == 2
    assert summary["observation_excluded_count_by_chip"] == {"0": 0, "2": 2}
    assert summary["observation_mask_source"] == "retrieval_archive"

    legacy_masks, legacy_summary = products._load_observation_masks(
        {},
        chip_data_list,
    )
    assert all(np.all(mask) for mask in legacy_masks)
    assert legacy_summary["observation_excluded_count"] == 0
    assert legacy_summary["observation_mask_source"] == "legacy_archive_default"


def test_observation_mask_archive_rejects_mask_and_index_drift():
    chip_data_list = [_chip_data(2, [[0.0, 0.7], [0.6, 0.5]])]
    archive = _masked_archive(chip_data_list)
    archive["observation_mask_chip2"] = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="exact-zero flux rule"):
        products._load_observation_masks(archive, chip_data_list)

    archive = _masked_archive(chip_data_list)
    archive["observation_indices"] = np.asarray([0, 1, 2], dtype=int)
    with pytest.raises(ValueError, match="flattened per-chip masks"):
        products._load_observation_masks(archive, chip_data_list)


def test_reused_maps_require_the_same_observation_mask_contract():
    chip_data_list = [_chip_data(2, [[0.0, 0.7], [0.6, 0.5]])]
    _, summary = products._load_observation_masks(
        _masked_archive(chip_data_list),
        chip_data_list,
    )

    products._validate_reused_observation_mask_summary(dict(summary), summary)

    stale_summary = dict(summary)
    stale_summary["observation_excluded_count"] = 0
    with pytest.raises(ValueError, match="observation-mask contract"):
        products._validate_reused_observation_mask_summary(stale_summary, summary)

    missing_summary = dict(summary)
    del missing_summary["observation_mask_rule"]
    with pytest.raises(ValueError, match="lack observation-mask metadata"):
        products._validate_reused_observation_mask_summary(missing_summary, summary)


def test_response_function_routes_independent_vmrs_and_logg():
    def spectrum(t0, alpha, co, h2o, ch4, hf, log_p_cloud, *, logg=None):
        offset = t0 + alpha + co + h2o + ch4 + hf + logg
        return jnp.asarray([offset + 3.0 * log_p_cloud])

    response = products._response_function(
        spectrum,
        independent_log_vmrs=True,
        explicit_logg=True,
    )
    base, derivative = response(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

    np.testing.assert_allclose(base, [52.0])
    np.testing.assert_allclose(derivative, [3.0])


def test_response_function_preserves_legacy_zeta_route():
    def spectrum(t0, alpha, zeta_vmr, log_p_cloud, *, logg=None):
        assert logg is None
        return jnp.asarray([t0 + alpha + zeta_vmr + 2.0 * log_p_cloud])

    response = products._response_function(spectrum)
    base, derivative = response(1.0, 2.0, 3.0, 4.0)

    np.testing.assert_allclose(base, [14.0])
    np.testing.assert_allclose(derivative, [2.0])


def test_fixed_eigen_conditioning_matches_direct_reduced_solution():
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    inverse_sqrt_six = 1.0 / np.sqrt(6.0)
    pixel_eigenvectors = np.asarray(
        [
            [inverse_sqrt_two, inverse_sqrt_six],
            [-inverse_sqrt_two, inverse_sqrt_six],
            [0.0, -2.0 * inverse_sqrt_six],
        ]
    )
    contrast = np.asarray(
        [
            [1.0, 0.2, -0.4],
            [0.3, -0.7, 0.6],
            [-0.1, 0.8, 0.5],
            [0.9, -0.2, 0.1],
        ]
    )
    residual = np.asarray([0.2, -0.1, 0.3, 0.05])
    noise_variance = np.asarray([0.4, 0.3, 0.5, 0.2])
    eigenvalues = np.asarray([0.7, 1.4])
    sigma = 0.35
    jitter = 5.0e-7

    reduced_design = contrast @ pixel_eigenvectors
    prior_variance = sigma**2 * eigenvalues + jitter
    precision = (
        np.diag(1.0 / prior_variance)
        + (reduced_design.T / noise_variance) @ reduced_design
    )
    covariance_reduced = np.linalg.inv(precision)
    mean_reduced = covariance_reduced @ ((reduced_design.T / noise_variance) @ residual)
    expected_mean = pixel_eigenvectors @ mean_reduced
    expected_covariance = pixel_eigenvectors @ covariance_reduced @ pixel_eigenvectors.T

    with jax.enable_x64():
        mean, covariance = products._conditional_pressure_map_fixed_eigen(
            residual,
            contrast,
            noise_variance,
            sigma,
            jitter,
            {
                "eigenvalues": eigenvalues,
                "pixel_eigenvectors": pixel_eigenvectors,
            },
        )
        diagonal_mean, variance_diagonal = (
            products._conditional_pressure_map_fixed_eigen(
                residual,
                contrast,
                noise_variance,
                sigma,
                jitter,
                {
                    "eigenvalues": eigenvalues,
                    "pixel_eigenvectors": pixel_eigenvectors,
                },
                return_variance_diagonal=True,
            )
        )
        mean = np.asarray(mean)
        covariance = np.asarray(covariance)
        diagonal_mean = np.asarray(diagonal_mean)
        variance_diagonal = np.asarray(variance_diagonal)

    np.testing.assert_allclose(mean, expected_mean, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        covariance,
        expected_covariance,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.sum(mean), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.sum(covariance, axis=0), 0.0, atol=1.0e-7)
    np.testing.assert_array_equal(diagonal_mean, mean)
    np.testing.assert_allclose(
        variance_diagonal,
        np.diag(expected_covariance),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_conditional_map_reuses_archived_observation_rows(monkeypatch):
    chip_data_list = [_chip_data(2, [[0.2, 0.0, 0.3]])]
    baseline = np.asarray([0.1, 10.0, 0.4])
    contrast = np.asarray(
        [
            [1.0, -0.2],
            [8.0, 7.0],
            [0.3, 0.5],
        ]
    )
    monkeypatch.setattr(
        products,
        "_joint_operator_from_sample",
        lambda *args, **kwargs: (jnp.asarray(baseline), jnp.asarray(contrast)),
    )
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    decomposition = {
        "eigenvalues": np.asarray([1.2]),
        "pixel_eigenvectors": np.asarray(
            [[inverse_sqrt_two], [-inverse_sqrt_two]]
        ),
    }
    sample = {
        "sigma_d": np.asarray([0.2]),
        "sigma_log_p": np.asarray(0.35),
    }
    masks = [np.asarray([[True, False, True]])]
    noise_variance = np.full(2, 0.2**2 + 1.0e-6)
    expected_mean, expected_covariance = (
        products._conditional_pressure_map_fixed_eigen(
            np.asarray([0.1, -0.1]),
            contrast[[0, 2]],
            noise_variance,
            0.35,
            5.0e-7,
            decomposition,
        )
    )

    actual_mean, actual_covariance = products._conditional_pressure_map_for_sample(
        chip_data_list,
        geometry=None,
        response_functions=[],
        profile_wavelengths=[],
        sample=sample,
        gp_jitter=5.0e-7,
        noise_jitter=1.0e-6,
        pressure_gp_eigendecomposition=decomposition,
        observation_masks=masks,
    )

    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_covariance, expected_covariance)


def test_fit_residual_selection_keeps_full_arrays_finite_and_unmodified():
    residual = np.asarray([[1.0, -10.0], [3.0, 4.0]])
    original = residual.copy()
    selected = products._select_fit_residuals(
        [residual],
        [np.asarray([[True, False], [True, True]])],
    )

    np.testing.assert_array_equal(residual, original)
    assert np.all(np.isfinite(residual))
    np.testing.assert_array_equal(selected[0], np.asarray([1.0, 3.0, 4.0]))


def test_absolute_pressure_moments_include_center_map_covariance_and_lognormality():
    conditional_means = np.asarray([[0.2, -0.1], [-0.2, 0.3]])
    conditional_variances = np.asarray([[0.01, 0.02], [0.03, 0.04]])
    centers = np.asarray([0.0, 1.0])
    absolute_means = centers[:, None] + conditional_means

    expected_log_mean = np.mean(absolute_means, axis=0)
    expected_log_var = (
        np.mean(
            conditional_variances + absolute_means**2,
            axis=0,
        )
        - expected_log_mean**2
    )
    log_ten = np.log(10.0)
    expected_p_mean = np.mean(
        np.exp(log_ten * absolute_means + 0.5 * log_ten**2 * conditional_variances),
        axis=0,
    )
    expected_p_second = np.mean(
        np.exp(
            2.0 * log_ten * absolute_means + 2.0 * log_ten**2 * conditional_variances
        ),
        axis=0,
    )
    expected_p_std = np.sqrt(expected_p_second - expected_p_mean**2)

    with jax.enable_x64():
        actual = products._combine_pressure_map_conditionals(
            conditional_means,
            conditional_variances,
            centers,
        )

    np.testing.assert_allclose(actual["log_p_cloud_mean"], expected_log_mean)
    np.testing.assert_allclose(actual["log_p_cloud_var"], expected_log_var)
    np.testing.assert_allclose(actual["p_cloud_mean"], expected_p_mean)
    np.testing.assert_allclose(actual["p_cloud_std"], expected_p_std)
    naive_first_pixel_variance = (
        np.var(centers)
        + np.var(conditional_means[:, 0])
        + np.mean(conditional_variances[:, 0])
    )
    assert not np.isclose(expected_log_var[0], naive_first_pixel_variance)


def test_saved_pressure_maps_use_exact_absolute_moments(tmp_path):
    exact = {
        "log_p_cloud_mean": np.asarray([0.1, 0.2, 0.3]),
        "log_p_cloud_var": np.asarray([0.01, 0.02, 0.03]),
        "p_cloud_mean": np.asarray([1.4, 1.7, 2.1]),
        "p_cloud_std": np.asarray([0.2, 0.3, 0.4]),
    }
    maps = products._save_pressure_maps(
        tmp_path,
        {"log_p_cloud": np.asarray([9.0, 10.0])},
        [object(), object()],
        np.asarray([-0.1, 0.0, 0.1]),
        np.asarray([0.04, 0.05, 0.06]),
        exact,
    )

    for name in (
        "log_p_cloud_mean",
        "log_p_cloud_var",
        "p_cloud_mean",
        "p_cloud_std",
    ):
        expected = np.tile(exact[name][None, :], (2, 1))
        np.testing.assert_array_equal(maps[f"{name}_by_chip"], expected)


def test_fixed_eigen_preparation_replays_retrieval_precision_and_freezes_basis(
    tmp_path,
):
    product_geometry = products._build_product_geometry(
        {"x64": np.asarray(False)},
        nside=1,
        product_x64=True,
    )
    with jax.enable_x64(False):
        retrieval_geometry = products.build_luhman16b_geometry(nside=1)
        rebuilt = products.build_fixed_pressure_gp_eigendecomposition(
            retrieval_geometry.distance_matrix,
            0.4,
            theta=retrieval_geometry.theta,
            phi=retrieval_geometry.phi,
        )
    assert np.asarray(product_geometry.theta).dtype == np.float64
    np.testing.assert_array_equal(
        np.asarray(product_geometry.theta),
        np.asarray(retrieval_geometry.theta, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(product_geometry.distance_matrix),
        np.asarray(retrieval_geometry.distance_matrix, dtype=np.float64),
    )
    saved = np.array(rebuilt["eigenvalues"], dtype=np.float64, copy=True)
    saved[-1] += 2.0e-7 * max(saved[-1], 1.0)
    samples = {
        "pressure_gp_factorization": np.asarray("fixed_eigen"),
        "zero_mean_pressure_map": np.asarray(True),
        "pressure_gp_eigenvalues": saved,
        "fixed_ell_b": np.asarray(0.4),
        "nside": np.asarray(1),
        "x64": np.asarray(False),
    }
    with jax.enable_x64():
        decomposition = products._prepare_fixed_pressure_gp_eigendecomposition(
            samples,
            product_geometry,
        )

    np.testing.assert_array_equal(decomposition["eigenvalues"], saved)
    np.testing.assert_array_equal(
        decomposition["pixel_eigenvectors"],
        np.asarray(rebuilt["pixel_eigenvectors"], dtype=np.float64),
    )
    assert decomposition["pixel_eigenvectors"].shape == (12, 11)
    assert decomposition["rebuilt_eigenvalue_max_abs_difference"] > 0.0

    summary = products._save_pressure_gp_eigendecomposition(
        tmp_path,
        decomposition,
    )
    loaded, loaded_summary = products._load_pressure_gp_eigendecomposition(
        tmp_path,
        samples,
        summary,
    )
    np.testing.assert_array_equal(loaded["eigenvalues"], saved)
    np.testing.assert_array_equal(
        loaded["pixel_eigenvectors"],
        decomposition["pixel_eigenvectors"],
    )
    assert loaded_summary == summary

    bad_summary = dict(summary)
    bad_summary["eigenvector_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="basis hash mismatch"):
        products._load_pressure_gp_eigendecomposition(
            tmp_path,
            samples,
            bad_summary,
        )
