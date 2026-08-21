#!/usr/bin/env -S tcsh -f

# Build the M8/v12 primary pressure-map products from the masked completed chain.

set script_path = `realpath "$0"`
set repo_root = "$script_path:h:h"
cd "$repo_root"
setenv PYTHONPATH "$repo_root/src"

set run_tag = "m8_v12_masked_primary"
set samples_dir = "results/m8/v12/masked_free_sigma_d_highest_precision_long_seed0"
set samples = "$samples_dir/samples.npz"
set product_dir = "results/m8/v12/masked_f64_products"
set primary_dir = "results/m8/v12/primary_bundle"
set product_script = "examples/luhman16b_yama/make_milestone4_on_the_fly_products.py"
set primary_source = "src/doraex/products/primary.py"
set mask_test = "tests/unittests/test_m8_primary_pressure_products.py"
set primary_test = "tests/unittests/test_primary_products.py"
set expected_samples_sha256 = "d87544c90746e8fb8a4da4d7ad47a87092e0e44f4650a45f5daa9c329fd20c74"
set expected_product_script_sha256 = "f96dd2bd84d3ed6d0358dd35558af9c156a140c52e17a367914828ccd7dadb9d"
set expected_primary_source_sha256 = "eaeae57f301e63167e025497dd72a5fef1df0819b6c04139bb840c20f9704627"
set expected_mask_test_sha256 = "8f3939b683d50ea82a203ef27ff6e9b12bdc6fc225fa2f98e7e9495efb07b02f"
set expected_primary_test_sha256 = "db1337f6215d62b62d21c5ea793aedeb37c895c832b3c779b3d33b38e92e3f4f"
set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_held = 0
set validate_only = 0

if ( $#argv > 1 ) then
    echo "[$run_tag] expected no argument or --validate-only"
    exit 2
endif
if ( $#argv == 1 ) then
    if ( "$argv[1]" == "--m8-v12-masked-primary-lock-held" ) then
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
    echo "[$run_tag] completed M8/v12 samples are missing"
    exit 1
endif
if ( -e "$samples_dir/FAILED" || -e "$samples_dir/RUNNING" ) then
    echo "[$run_tag] M8/v12 sample state is not cleanly complete"
    exit 1
endif
sha256sum -c "$samples_dir/outputs.sha256"
if ( $status != 0 ) then
    echo "[$run_tag] M8/v12 sample integrity check failed"
    exit 1
endif
sha256sum -c "$samples_dir/provenance.sha256"
if ( $status != 0 ) then
    echo "[$run_tag] M8/v12 retrieval provenance check failed"
    exit 1
endif

set actual_samples_sha256 = `sha256sum "$samples" | awk '{print $1}'`
set actual_product_script_sha256 = `sha256sum "$product_script" | awk '{print $1}'`
set actual_primary_source_sha256 = `sha256sum "$primary_source" | awk '{print $1}'`
set actual_mask_test_sha256 = `sha256sum "$mask_test" | awk '{print $1}'`
set actual_primary_test_sha256 = `sha256sum "$primary_test" | awk '{print $1}'`
if ( "$actual_samples_sha256" != "$expected_samples_sha256" ) then
    echo "[$run_tag] unexpected M8/v12 samples SHA256: $actual_samples_sha256"
    exit 1
endif
if ( "$actual_product_script_sha256" != "$expected_product_script_sha256" ) then
    echo "[$run_tag] masked product implementation SHA256 mismatch"
    exit 1
endif
if ( "$actual_primary_source_sha256" != "$expected_primary_source_sha256" ) then
    echo "[$run_tag] primary implementation SHA256 mismatch"
    exit 1
endif
if ( "$actual_mask_test_sha256" != "$expected_mask_test_sha256" ) then
    echo "[$run_tag] masked product regression test SHA256 mismatch"
    exit 1
endif
if ( "$actual_primary_test_sha256" != "$expected_primary_test_sha256" ) then
    echo "[$run_tag] primary regression test SHA256 mismatch"
    exit 1
endif

python -c 'import sys,numpy as np; a=np.load(sys.argv[1],allow_pickle=False); chips=np.asarray(a["chip_indices"],dtype=int).tolist(); masks=[np.asarray(a[f"observation_mask_chip{c}"],dtype=bool) for c in chips]; flux=[np.asarray(a[f"flux_chip{c}"]) for c in chips]; expected=[np.isfinite(f)&(f!=0.0) for f in flux]; indices=np.flatnonzero(np.concatenate([m.reshape(-1) for m in expected])); stored=np.asarray(a["observation_indices"],dtype=int); sigma=np.asarray(a["sigma_d"],dtype=float); valid=[int(m.sum()) for m in masks]; ok=chips==[0,1,2,3] and all(np.array_equal(m,e) for m,e in zip(masks,expected)) and np.array_equal(stored,indices) and bool(np.asarray(a["mask_zero_flux"]).item()) and str(np.asarray(a["observation_mask_rule"]).item())=="finite_and_nonzero_flux" and int(np.asarray(a["observation_valid_count"]).item())==57232 and int(np.asarray(a["observation_excluded_count"]).item())==112 and valid==[14336,14336,14280,14280] and not bool(np.asarray(a["fix_sigma_d"]).item()) and sigma.shape==(1500,4) and np.all(np.isfinite(sigma)) and np.all(sigma>0.0) and np.all(np.ptp(sigma,axis=0)>0.0); print({"valid_by_chip":valid,"excluded_total":112,"free_sigma_d":not bool(np.asarray(a["fix_sigma_d"]).item()),"contract_ok":ok}); sys.exit(0 if ok else 3)' "$samples"
if ( $status != 0 ) then
    echo "[$run_tag] M8/v12 mask or free-sigma_d archive contract failed"
    exit 1
endif

python -c 'import sys,numpy as np; from doraex.operators.doppler import doppler_padded_wavelengths; a=np.load(sys.argv[1],allow_pickle=False); chips=[int(x) for x in np.asarray(a["chip_indices"])]; log_w=np.asarray(a["log_w"],dtype=float); vmr=[np.asarray(a[k],dtype=float) for k in ("log_vmr_co","log_vmr_h2o","log_vmr_ch4","log_vmr_hf")]; logg=np.asarray(a["logg"],dtype=float); eig=np.asarray(a["pressure_gp_eigenvalues"],dtype=float); vmax=float(np.max(np.abs(np.asarray(a["v"],dtype=float)))); profiles=[np.asarray(a[f"profile_wavelengths_chip{c}"]) for c in chips]; observed=[np.asarray(a[f"wavelengths_chip{c}"]) for c in chips]; profile_ok=[np.array_equal(p,doppler_padded_wavelengths(w,vmax)) for p,w in zip(profiles,observed)]; ok=bool(np.asarray(a["fix_log_w"]).item()) and set(np.asarray(a["fixed_nuisance_sites"],dtype=str).tolist())=={"log_w"} and log_w.shape==(1500,4,14) and np.all(np.isfinite(log_w)) and np.max(np.ptp(log_w,axis=0))==0.0 and all(x.shape==(1500,) and np.all(np.isfinite(x)) for x in vmr) and bool(np.asarray(a["fix_logg"]).item()) and float(np.asarray(a["fixed_logg"]).item())==4.86 and logg.shape==(1500,) and np.all(np.isfinite(logg)) and np.ptp(logg)==0.0 and np.all(logg==np.float32(4.86)) and str(np.asarray(a["pressure_gp_factorization"]).item())=="fixed_eigen" and bool(np.asarray(a["zero_mean_pressure_map"]).item()) and float(np.asarray(a["fixed_ell_b"]).item())==0.4 and eig.shape==(767,) and np.all(np.isfinite(eig)) and [len(p) for p in profiles]==[1066,1068,1070,1072] and all(profile_ok); print({"fixed_log_w":bool(np.asarray(a["fix_log_w"]).item()),"vmr_shapes":[x.shape for x in vmr],"fixed_logg":float(np.asarray(a["fixed_logg"]).item()),"pressure_gp":str(np.asarray(a["pressure_gp_factorization"]).item()),"profile_lengths":[len(p) for p in profiles],"profile_match":profile_ok,"target_contract_ok":ok}); sys.exit(0 if ok else 4)' "$samples"
if ( $status != 0 ) then
    echo "[$run_tag] M8/v12 retrieval target contract failed"
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

python -m pytest -q "$mask_test" "$primary_test"
if ( $status != 0 ) then
    echo "[$run_tag] primary product regression tests failed"
    exit 1
endif

if ( $validate_only == 1 ) then
    echo "[$run_tag] validation passed"
    exit 0
endif

if ( $lock_held == 0 ) then
    flock --exclusive --nonblock --conflict-exit-code 75 \
        "$gpu_lock" "$script_path" --m8-v12-masked-primary-lock-held
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
    csh/exe_m8_v12_masked_primary_products.csh \
    "$samples" \
    "$samples_dir/diagnostics.json" \
    "$samples_dir/provenance.sha256" \
    "$samples_dir/outputs.sha256" \
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
    "$primary_test" \
    >! "$product_dir/primary_provenance.sha256"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] provenance capture failed"
    exit 1
endif

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
python "$product_script" \
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
    --prefix m8_v12 \
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

python -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); p=Path(sys.argv[2]); d=json.loads((r/"primary_products.json").read_text()); assert d["prefix"]=="m8_v12"; assert all(x["exists"] and not x["missing"] for x in d["products"]); assert set(d["excluded_products"])=={"linearization_check","ureshino_fig10_three_map_comparison","log_w_combined_diagnostic"}; q=json.loads((p/"info/run_summary.json").read_text()); assert q["mask_zero_flux"] is True and q["observation_mask_rule"]=="finite_and_nonzero_flux" and q["observation_valid_count"]==57232 and q["observation_excluded_count"]==112 and q["fix_sigma_d"] is False and q["fix_log_w"] is True and q["fixed_nuisance_sites"]==["log_w"]' "$product_dir" "$primary_dir"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] product manifest validation failed: `date`" | tee -a "$log"
    exit 1
endif

python -c 'import hashlib,json,sys; from pathlib import Path; import numpy as np; r=Path(sys.argv[1]); p=Path(sys.argv[2]); s=Path(sys.argv[3]); a={n:np.load(r/n) for n in ("pressure_gp_eigenvalues.npy","pressure_gp_pixel_eigenvectors.npy","contrast_mean_joint.npy","contrast_var_joint.npy","log_p_cloud_mean_joint_by_chip.npy","log_p_cloud_var_joint_by_chip.npy","p_cloud_mean_joint_by_chip.npy","p_cloud_std_joint_by_chip.npy")}; assert a["pressure_gp_eigenvalues.npy"].shape==(767,); assert a["pressure_gp_pixel_eigenvectors.npy"].shape==(768,767); assert a["contrast_mean_joint.npy"].shape==(768,) and a["contrast_var_joint.npy"].shape==(768,); assert all(a[n].shape==(4,768) for n in ("log_p_cloud_mean_joint_by_chip.npy","log_p_cloud_var_joint_by_chip.npy","p_cloud_mean_joint_by_chip.npy","p_cloud_std_joint_by_chip.npy")); assert all(np.all(np.isfinite(v)) for v in a.values()); assert np.all(a["contrast_var_joint.npy"]>=0.0) and np.all(a["log_p_cloud_var_joint_by_chip.npy"]>=0.0) and np.all(a["p_cloud_std_joint_by_chip.npy"]>=0.0); u=a["pressure_gp_pixel_eigenvectors.npy"]; assert np.max(np.abs(np.sum(u,axis=0)))<1e-5; assert np.max(np.abs(u.T@u-np.eye(767)))<1e-5; q=json.loads((r/"on_the_fly_product_summary.json").read_text()); j=json.loads((r/"joint_chip_diagnostics.json").read_text()); expected={"mask_zero_flux":True,"observation_mask_rule":"finite_and_nonzero_flux","observation_mask_source":"retrieval_archive","observation_valid_count":57232,"observation_excluded_count":112,"observation_valid_count_by_chip":{"0":14336,"1":14336,"2":14280,"3":14280},"observation_excluded_count_by_chip":{"0":0,"1":0,"2":56,"3":56}}; assert all(all(d.get(k)==v for k,v in expected.items()) for d in (q,j)); assert j["residual_statistic_scope"]=="likelihood_observations"; assert q["sample_sha256"]==hashlib.sha256(s.read_bytes()).hexdigest(); assert q["map_sample_count"]==1500 and q["absolute_pressure_moment_method"]=="conditional_gaussian_exact_lognormal_transform" and q["pressure_gp"]["factorization"]=="fixed_eigen"; n=json.loads((p/"info/posterior_nuisance.json").read_text()); assert n["fixed_parameters"]=={"A":False,"sigma_d":False,"log_w":True}; rows=[x for x in n["by_chip"] if x["name"]=="sigma_d"]; assert len(rows)==4 and all(not x["fixed"] and x["statistic_scope"]=="posterior_samples" and x["n"]==1500 and np.isfinite(x["median"]) for x in rows)' "$product_dir" "$primary_dir" "$samples"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] masked numerical artifact validation failed: `date`" | tee -a "$log"
    exit 1
endif

sha256sum -c "$product_dir/primary_provenance.sha256" >>& "$log"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] source provenance changed during the run: `date`" | tee -a "$log"
    exit 1
endif

find "$product_dir" "$primary_dir" -type f \
    ! -name RUNNING \
    ! -name run.log \
    ! -name primary_outputs.sha256 \
    ! -name primary_provenance.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >! "$product_dir/primary_outputs.sha256"
if ( $status != 0 ) then
    mv "$product_dir/RUNNING" "$product_dir/FAILED"
    echo "[$run_tag] output hashing failed: `date`" | tee -a "$log"
    exit 1
endif

mv "$product_dir/RUNNING" "$product_dir/COMPLETE"
touch "$primary_dir/COMPLETE"
echo "[$run_tag] done: `date`" | tee -a "$log"
