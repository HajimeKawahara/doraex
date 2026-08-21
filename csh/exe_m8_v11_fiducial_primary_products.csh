#!/usr/bin/env -S tcsh -f

# Build the M8/v11 fiducial pressure-map products from the completed chain.

set script_path = `realpath "$0"`
set repo_root = "$script_path:h:h"
cd "$repo_root"

set run_tag = "m8_v11_fiducial_primary"
set samples_dir = "results/m8/v11/free_sigma_highest_precision_long_seed0"
set samples = "$samples_dir/samples.npz"
set product_dir = "results/m8/v11/fiducial_f64_products"
set primary_dir = "results/m8/v11/primary_bundle"
set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_held = 0
set validate_only = 0

if ( $#argv > 1 ) then
    echo "[$run_tag] expected no argument or --validate-only"
    exit 2
endif
if ( $#argv == 1 ) then
    if ( "$argv[1]" == "--m8-v11-fiducial-primary-lock-held" ) then
        set lock_held = 1
    else if ( "$argv[1]" == "--validate-only" ) then
        set validate_only = 1
        set lock_held = 1
    else
        echo "[$run_tag] unsupported argument: $argv[1]"
        exit 2
    endif
endif

if ( ! -f "$samples_dir/COMPLETE" || ! -f "$samples" ) then
    echo "[$run_tag] completed M8/v11 samples are missing"
    exit 1
endif
sha256sum -c "$samples_dir/outputs.sha256"
if ( $status != 0 ) then
    echo "[$run_tag] M8/v11 sample integrity check failed"
    exit 1
endif
foreach database ( CO H2O CH4 HF )
    if ( ! -d "/home/kawahara/data_mol/.database/$database" ) then
        echo "[$run_tag] missing molecular database: $database"
        exit 1
    endif
end
if ( -e "$product_dir" || -e "$primary_dir" ) then
    echo "[$run_tag] refusing output collision"
    echo "[$run_tag] product_dir=$product_dir"
    echo "[$run_tag] primary_dir=$primary_dir"
    exit 1
endif

if ( $validate_only == 1 ) then
    echo "[$run_tag] validation passed"
    exit 0
endif

if ( $lock_held == 0 ) then
    flock --exclusive --nonblock --conflict-exit-code 75 \
        "$gpu_lock" "$script_path" --m8-v11-fiducial-primary-lock-held
    set lock_status = $status
    if ( $lock_status == 75 ) then
        echo "[$run_tag] another Luhman 16B GPU job holds $gpu_lock"
        exit 2
    endif
    exit $lock_status
endif

mkdir -p "$product_dir"
if ( $status != 0 ) exit 1
touch "$product_dir/RUNNING"
set log = "$product_dir/run.log"

sha256sum \
    csh/exe_m8_v11_fiducial_primary_products.csh \
    examples/luhman16b_yama/generate_milestone2_t0_alpha_cloud_zeta_grid_profiles.py \
    examples/luhman16b_yama/make_line_strength_comparison_panel.py \
    examples/luhman16b_yama/make_milestone2_fixed_products.py \
    examples/luhman16b_yama/make_milestone2_free_t0_cloud_products.py \
    examples/luhman16b_yama/make_milestone2_joint_chip_products.py \
    examples/luhman16b_yama/make_milestone4_on_the_fly_products.py \
    examples/luhman16b_yama/make_milestone5_minimum_window_figures.py \
    examples/luhman16b_yama/make_milestone5_single_line_rt_prediction.py \
    examples/luhman16b_yama/make_observer_view_cloud_pressure_map.py \
    examples/luhman16b_yama/compare_milestone5_single_line_rt_predictions.py \
    src/doraex/data/luhman16b.py \
    src/doraex/diagnostics/mean_subtracted_line_stack.py \
    src/doraex/diagnostics/single_line_pressure_response.py \
    src/doraex/constants.py \
    src/doraex/geometry/healpix.py \
    src/doraex/geometry/limb_darkening.py \
    src/doraex/geometry/rotation.py \
    src/doraex/inference/map_posterior.py \
    src/doraex/operators/design_matrix.py \
    src/doraex/operators/doppler.py \
    src/doraex/priors/spherical_gp.py \
    src/doraex/products/primary.py \
    src/doraex/spectra/exojax_forward.py \
    src/doraex/workflows/luhman16b_milestone2.py \
    src/doraex/workflows/on_the_fly_pressure_retrieval.py \
    tests/unittests/test_exojax_forward.py \
    tests/unittests/test_m8_primary_pressure_products.py \
    tests/unittests/test_primary_products.py \
    >! "$product_dir/fiducial_provenance.sha256"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] provenance capture failed"
    exit 1
endif

setenv PYTHONPATH "$repo_root/src"
setenv NUMBA_CACHE_DIR "/tmp/numba-cache"
setenv MPLCONFIGDIR "/tmp/matplotlib-codex"
setenv JAX_COMPILATION_CACHE_DIR "/home/kawahara/.cache/jax"
setenv JAX_ENABLE_COMPILATION_CACHE "1"
setenv JAX_DEFAULT_MATMUL_PRECISION "highest"
setenv XLA_FLAGS "--xla_gpu_autotune_level=0"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"

python -c 'import jax; assert jax.default_backend() == "gpu", jax.default_backend(); assert any(device.platform == "gpu" for device in jax.devices()); assert jax.config.jax_default_matmul_precision == "highest"'
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] GPU or JAX precision preflight failed"
    exit 1
endif

echo "[$run_tag] map reconstruction start: `date`" | tee "$log"
python examples/luhman16b_yama/make_milestone4_on_the_fly_products.py \
    --samples "$samples" \
    --out-dir "$product_dir" \
    --max-map-samples 1500 \
    --gp-jitter 5e-7 \
    --noise-jitter 1e-6 \
    --x64 \
    >>& "$log"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] map reconstruction failed: `date`" | tee -a "$log"
    exit 1
endif

python examples/luhman16b_yama/make_observer_view_cloud_pressure_map.py \
    --product-dir "$product_dir" \
    --output "$product_dir/figure8_p_cloud_observer.png" \
    --cosi 0.485 \
    >>& "$log"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] observer-view map failed: `date`" | tee -a "$log"
    exit 1
endif

echo "[$run_tag] primary bundle start: `date`" | tee -a "$log"
python -m doraex.products.primary \
    --samples "$samples" \
    --product-dir "$product_dir" \
    --primary-dir "$primary_dir" \
    --prefix m8_v11 \
    --project-root "$repo_root" \
    --no-run-map-products \
    --no-run-log-w-products \
    --no-run-linearization-check \
    --exclude-products linearization_check,ureshino_fig10_three_map_comparison,log_w_combined_diagnostic \
    --x64 \
    >>& "$log"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] primary bundle failed: `date`" | tee -a "$log"
    exit 1
endif

python -c 'import json, pathlib; p = pathlib.Path("results/m8/v11/fiducial_f64_products/primary_products.json"); d = json.loads(p.read_text()); assert all(x["exists"] and not x["missing"] for x in d["products"]); assert set(d["excluded_products"]) == {"linearization_check", "ureshino_fig10_three_map_comparison", "log_w_combined_diagnostic"}'
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] product manifest validation failed: `date`" | tee -a "$log"
    exit 1
endif

python -c 'import hashlib, json, pathlib, numpy as np; r = pathlib.Path("results/m8/v11/fiducial_f64_products"); s = pathlib.Path("results/m8/v11/free_sigma_highest_precision_long_seed0/samples.npz"); a = {n: np.load(r / n) for n in ("pressure_gp_eigenvalues.npy", "pressure_gp_pixel_eigenvectors.npy", "contrast_mean_joint.npy", "contrast_var_joint.npy", "log_p_cloud_mean_joint_by_chip.npy", "log_p_cloud_var_joint_by_chip.npy", "p_cloud_mean_joint_by_chip.npy", "p_cloud_std_joint_by_chip.npy")}; assert a["pressure_gp_eigenvalues.npy"].shape == (767,); assert a["pressure_gp_pixel_eigenvectors.npy"].shape == (768, 767); assert a["contrast_mean_joint.npy"].shape == (768,) and a["contrast_var_joint.npy"].shape == (768,); assert all(a[n].shape == (4, 768) for n in ("log_p_cloud_mean_joint_by_chip.npy", "log_p_cloud_var_joint_by_chip.npy", "p_cloud_mean_joint_by_chip.npy", "p_cloud_std_joint_by_chip.npy")); assert all(np.all(np.isfinite(v)) for v in a.values()); assert np.all(a["contrast_var_joint.npy"] >= 0.0) and np.all(a["log_p_cloud_var_joint_by_chip.npy"] >= 0.0) and np.all(a["p_cloud_std_joint_by_chip.npy"] >= 0.0); u = a["pressure_gp_pixel_eigenvectors.npy"]; assert np.max(np.abs(np.sum(u, axis=0))) < 1e-5; assert np.max(np.abs(u.T @ u - np.eye(767))) < 1e-5; q = json.loads((r / "on_the_fly_product_summary.json").read_text()); assert q["sample_sha256"] == hashlib.sha256(s.read_bytes()).hexdigest(); assert q["map_sample_count"] == 1500; assert q["absolute_pressure_moment_method"] == "conditional_gaussian_exact_lognormal_transform"; assert q["pressure_gp"]["factorization"] == "fixed_eigen"'
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] numerical artifact validation failed: `date`" | tee -a "$log"
    exit 1
endif

sha256sum -c "$product_dir/fiducial_provenance.sha256" >>& "$log"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] source provenance changed during the run: `date`" | tee -a "$log"
    exit 1
endif

find "$product_dir" "$primary_dir" -type f \
    ! -name RUNNING \
    ! -name run.log \
    ! -name fiducial_outputs.sha256 \
    ! -name fiducial_provenance.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >! "$product_dir/fiducial_outputs.sha256"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] output hashing failed: `date`" | tee -a "$log"
    exit 1
endif

mv "$product_dir/RUNNING" "$product_dir/COMPLETE"
touch "$primary_dir/COMPLETE"
echo "[$run_tag] done: `date`" | tee -a "$log"
