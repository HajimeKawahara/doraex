"""Reusable workflow entry points.

The workflow package keeps imports lazy so post-processing scripts can run in
environments where NumPyro is not importable. NumPyro is only required when a
caller accesses MCMC entry points.
"""

__all__ = [
    "Luhman16BGeometry",
    "build_luhman16b_geometry",
    "compute_posterior_map_moments",
    "load_milestone1_inputs",
    "median_parameter_sample",
    "reconstruct_spectral_timeseries",
    "run_luhman16b_mcmc",
    "save_mcmc_samples",
    "load_milestone2_fixed_inputs",
    "load_milestone2_free_cloud_inputs",
    "load_milestone2_free_t0_cloud_inputs",
    "compute_contrast_map_moments",
    "compute_free_cloud_contrast_map_moments",
    "compute_free_t0_cloud_contrast_map_moments",
    "fixed_two_column_median_sample",
    "reconstruct_fixed_two_column_timeseries",
    "reconstruct_free_cloud_two_column_timeseries",
    "reconstruct_free_t0_cloud_two_column_timeseries",
    "run_fixed_two_column_mcmc",
    "run_free_cloud_two_column_mcmc",
    "run_free_t0_cloud_two_column_mcmc",
    "save_fixed_two_column_samples",
    "save_free_cloud_two_column_samples",
    "save_free_t0_cloud_two_column_samples",
    "save_synthetic_smoke_cloud_profile_grid",
    "save_synthetic_smoke_profiles",
]

_MILESTONE1_EXPORTS = {
    "Luhman16BGeometry",
    "build_luhman16b_geometry",
    "compute_posterior_map_moments",
    "load_milestone1_inputs",
    "median_parameter_sample",
    "reconstruct_spectral_timeseries",
    "run_luhman16b_mcmc",
    "save_mcmc_samples",
}

_MILESTONE2_EXPORTS = {
    "load_milestone2_fixed_inputs",
    "load_milestone2_free_cloud_inputs",
    "load_milestone2_free_t0_cloud_inputs",
    "compute_contrast_map_moments",
    "compute_free_cloud_contrast_map_moments",
    "compute_free_t0_cloud_contrast_map_moments",
    "fixed_two_column_median_sample",
    "reconstruct_fixed_two_column_timeseries",
    "reconstruct_free_cloud_two_column_timeseries",
    "reconstruct_free_t0_cloud_two_column_timeseries",
    "run_fixed_two_column_mcmc",
    "run_free_cloud_two_column_mcmc",
    "run_free_t0_cloud_two_column_mcmc",
    "save_fixed_two_column_samples",
    "save_free_cloud_two_column_samples",
    "save_free_t0_cloud_two_column_samples",
    "save_synthetic_smoke_cloud_profile_grid",
    "save_synthetic_smoke_profiles",
}


def __getattr__(name):
    """Load workflow symbols on first access."""

    if name in _MILESTONE1_EXPORTS:
        from doraex.workflows import luhman16b_milestone1

        return getattr(luhman16b_milestone1, name)
    if name in _MILESTONE2_EXPORTS:
        from doraex.workflows import luhman16b_milestone2

        return getattr(luhman16b_milestone2, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
