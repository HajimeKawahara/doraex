import json

import numpy as np
import pytest

from doraex.products import primary


M8_EXCLUDED_PRODUCTS = (
    "linearization_check",
    "ureshino_fig10_three_map_comparison",
    "log_w_combined_diagnostic",
)


def test_primary_product_paths_exclude_named_products(tmp_path):
    paths = primary.primary_product_paths(
        tmp_path,
        prefix="m8_v11",
        excluded_product_names=M8_EXCLUDED_PRODUCTS,
    )

    assert not set(M8_EXCLUDED_PRODUCTS) & paths.keys()
    assert "cloud_pressure_map" in paths
    assert "joint_chip_3" in paths


def test_primary_product_paths_reject_unknown_exclusion(tmp_path):
    with pytest.raises(ValueError, match="unknown_product"):
        primary.primary_product_paths(
            tmp_path,
            excluded_product_names=("unknown_product",),
        )


def test_collect_primary_products_ignores_excluded_products(tmp_path, monkeypatch):
    product_dir = tmp_path / "products"
    bundle_dir = tmp_path / "bundle"
    product_dir.mkdir()
    cloud_path = product_dir / "figure8_p_cloud_observer.png"
    cloud_path.write_bytes(b"figure")
    excluded = tuple(
        definition.name
        for definition in primary.DEFAULT_PRIMARY_PRODUCT_DEFINITIONS
        if definition.name != "cloud_pressure_map"
    )
    monkeypatch.setattr(
        primary,
        "write_primary_info_products",
        lambda *args, **kwargs: {"info_files": []},
    )

    primary.collect_primary_products(
        tmp_path / "samples.npz",
        product_dir,
        bundle_dir,
        chip_indices=(),
        excluded_product_names=excluded,
    )

    manifest = json.loads(
        (bundle_dir / "primary_bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert (bundle_dir / cloud_path.name).read_bytes() == b"figure"
    assert [item["name"] for item in manifest["figures"]] == ["cloud_pressure_map"]
    assert manifest["excluded_products"] == sorted(excluded)
    readme = (bundle_dir / "README.md").read_text(encoding="utf-8")
    assert "## Excluded Products" in readme
    assert "`linearization_check`" in readme
    assert "`fixed` and `statistic_scope`" in readme


def test_nuisance_corner_data_omits_fixed_columns(tmp_path):
    path = tmp_path / "samples.npz"
    sample_count = 16
    amplitudes = np.column_stack(
        [np.linspace(1.0 + index, 1.1 + index, sample_count) for index in range(4)]
    )
    np.savez(
        path,
        A=amplitudes,
        sigma_d=np.broadcast_to(np.arange(4, dtype=float), (sample_count, 4)),
    )

    with np.load(path, allow_pickle=False) as samples:
        data, labels = primary._nuisance_corner_data(samples)

    assert data.shape == (sample_count, 4)
    assert labels == ["$A_0$", "$A_1$", "$A_2$", "$A_3$"]


def test_nuisance_summary_marks_fixed_sites_and_separates_phase_spread(tmp_path):
    path = tmp_path / "samples.npz"
    sample_count = 16
    fixed_log_w = np.broadcast_to(
        np.arange(6, dtype=float).reshape(1, 2, 3),
        (sample_count, 2, 3),
    )
    np.savez(
        path,
        A=np.column_stack(
            [
                np.linspace(1.0, 1.1, sample_count),
                np.linspace(1.1, 1.2, sample_count),
            ]
        ),
        sigma_d=np.broadcast_to(np.asarray([0.03, 0.04]), (sample_count, 2)),
        log_w=fixed_log_w,
        fix_a=np.asarray(False),
        fix_log_w=np.asarray(True),
        fix_sigma_d=np.asarray(True),
        fixed_nuisance_sites=np.asarray(["log_w", "sigma_d"]),
    )

    with np.load(path, allow_pickle=False) as samples:
        summary = primary._build_nuisance_summary(samples, (0, 1))

    assert summary["fixed_parameters"] == {
        "A": False,
        "sigma_d": True,
        "log_w": True,
    }
    by_name = {
        (row["name"], row["chip_index"]): row for row in summary["by_chip"]
    }
    assert by_name[("A", 0)]["statistic_scope"] == "posterior_samples"
    assert by_name[("A", 0)]["n"] == sample_count
    assert by_name[("sigma_d", 0)]["fixed"] is True
    assert by_name[("sigma_d", 0)]["statistic_scope"] == "fixed_value"
    assert by_name[("sigma_d", 0)]["n"] == 1
    assert by_name[("log_w", 0)]["fixed"] is True
    assert by_name[("log_w", 0)]["statistic_scope"] == "fixed_values_across_phase"
    assert by_name[("log_w", 0)]["n"] == 3
    assert by_name[("log_w", 0)]["std"] == pytest.approx(np.std([0.0, 1.0, 2.0]))
    phase_row = summary["log_w_by_phase"][0]
    assert phase_row["fixed"] is True
    assert phase_row["statistic_scope"] == "fixed_value"
    assert phase_row["n"] == 1
    assert phase_row["std"] == 0.0


def test_run_summary_includes_fixed_nuisance_metadata(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(
        path,
        atmosphere_rotated=np.zeros((16, 7)),
        sigma_log_p=np.ones(16),
        A=np.ones((16, 4)),
        mask_zero_flux=np.asarray(True),
        observation_mask_rule=np.asarray("finite_and_nonzero_flux"),
        observation_valid_count=np.asarray(57232),
        observation_excluded_count=np.asarray(112),
    )
    diagnostics = {
        "fix_nuisance": False,
        "fix_a": False,
        "fix_log_w": True,
        "fix_sigma_d": True,
        "fixed_nuisance_sites": ["log_w", "sigma_d"],
    }

    with np.load(path, allow_pickle=False) as samples:
        summary = primary._build_run_summary(samples, diagnostics, {})

    assert summary["fix_a"] is False
    assert summary["fix_log_w"] is True
    assert summary["fix_sigma_d"] is True
    assert summary["fixed_nuisance_sites"] == ["log_w", "sigma_d"]
    assert summary["mask_zero_flux"] is True
    assert summary["observation_mask_rule"] == "finite_and_nonzero_flux"
    assert summary["observation_valid_count"] == 57232
    assert summary["observation_excluded_count"] == 112


def test_run_summary_omits_mask_metadata_for_legacy_archive(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, A=np.ones((16, 4)))

    with np.load(path, allow_pickle=False) as samples:
        summary = primary._build_run_summary(samples, {}, {})

    assert "mask_zero_flux" not in summary
    assert "observation_mask_rule" not in summary
    assert "observation_valid_count" not in summary
    assert "observation_excluded_count" not in summary


def test_count_sampled_parameters_uses_active_m8_sites(tmp_path):
    path = tmp_path / "samples.npz"
    sample_count = 16
    np.savez(
        path,
        atmosphere_rotated=np.zeros((sample_count, 7)),
        sigma_log_p=np.ones(sample_count),
        A=np.ones((sample_count, 4)),
        log_w=np.zeros((sample_count, 4, 14)),
        sigma_d=np.ones((sample_count, 4)),
        direct_sigma_log_p=np.asarray(True),
        fixed_nuisance_sites=np.asarray(["log_w", "sigma_d"]),
    )
    diagnostics = {
        "fix_a": False,
        "fix_log_w": True,
        "fix_sigma_d": True,
        "direct_sigma_log_p": True,
        "fixed_nuisance_sites": ["log_w", "sigma_d"],
    }

    with np.load(path, allow_pickle=False) as samples:
        count = primary._count_sampled_parameters(samples, diagnostics)

    assert count == 12


def test_count_sampled_parameters_keeps_free_log_w_outside_joint_coordinates(
    tmp_path,
):
    path = tmp_path / "samples.npz"
    sample_count = 16
    np.savez(
        path,
        atmosphere_a_sigma_d_rotated=np.zeros((sample_count, 15)),
        log_w=np.zeros((sample_count, 4, 14)),
        fix_log_w=np.asarray(False),
    )

    with np.load(path, allow_pickle=False) as samples:
        count = primary._count_sampled_parameters(samples)

    assert count == 15 + 4 * 14


def test_count_sampled_parameters_counts_raw_nuisance_with_encoded_atmosphere(
    tmp_path,
):
    path = tmp_path / "samples.npz"
    sample_count = 16
    np.savez(
        path,
        atmosphere_rotated=np.zeros((sample_count, 7)),
        log_w=np.zeros((sample_count, 4, 14)),
        log_w_raw=np.zeros((sample_count, 4, 14)),
    )

    with np.load(path, allow_pickle=False) as samples:
        count = primary._count_sampled_parameters(samples)

    assert count == 7 + 4 * 14
