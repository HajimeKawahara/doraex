#!/bin/tcsh -f

set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0
set validate_only = 0

if ( $#argv > 1 ) then
    echo "[m8_v5_sigma_geometry] expected no argument or --validate-only"
    exit 2
endif
if ( $#argv == 1 ) then
    if ( "$argv[1]" == "--m8-v5-sigma-geometry-gpu-lock-held" ) then
        set gpu_lock_held = 1
    else if ( "$argv[1]" == "--validate-only" ) then
        set gpu_lock_held = 1
        set validate_only = 1
    else
        echo "[m8_v5_sigma_geometry] unsupported argument: $argv[1]"
        exit 2
    endif
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        /bin/tcsh -f "$script_path" --m8-v5-sigma-geometry-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[m8_v5_sigma_geometry] not started: another Luhman 16B run holds $gpu_lock"
        echo "[m8_v5_sigma_geometry] wait for it to finish, then retry."
        exit 2
    endif
    exit $lock_status
endif

set diagnostic_script = "examples/luhman16b_yama/check_m8_v5_free_sigma_geometry.py"
set init_archive = "results/m7/p2/v6/samples.npz"
set free_wrapper = "examples/luhman16b_yama/m8_v1_run.py"
set free_launcher = "csh/exe_m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod.csh"
set free_samples = "results/m8/v1/samples.npz"
set free_diagnostics = "results/m8/v1/diagnostics.json"
set fixed_wrapper = "examples/luhman16b_yama/m8_v2_run.py"
set replay_helper = "examples/luhman16b_yama/check_m8_v2_fixed_free_initial_point.py"
set fixed_launcher = "csh/exe_m8_v3_fixed_sigma_log_p_v17_initial_full_dense_long_control.csh"
set fixed_samples = "results/m8/v3/fixed_seed0/samples.npz"
set fixed_diagnostics = "results/m8/v3/fixed_seed0/diagnostics.json"
set initial_replay = "results/m8/v3/fixed_seed0/initial_point_replay.json"
set data_spectra = "data/fainterspectral-fits_6.pickle"
set data_template = "data/posterior_predictive_vsini=0.npz"
set workflow = "src/doraex/workflows/on_the_fly_pressure_retrieval.py"
set forward_source = "src/doraex/spectra/exojax_forward.py"
set chip_data_source = "src/doraex/data/luhman16b.py"
set healpix_source = "src/doraex/geometry/healpix.py"
set gp_source = "src/doraex/priors/spherical_gp.py"
set design_source = "src/doraex/operators/design_matrix.py"
set doppler_source = "src/doraex/operators/doppler.py"
set rotation_source = "src/doraex/geometry/rotation.py"
set limb_source = "src/doraex/geometry/limb_darkening.py"
set marginal_likelihood_source = "src/doraex/inference/marginal_likelihood.py"
set constants_source = "src/doraex/constants.py"
set database_dir = "/home/kawahara/data_mol/.database"
set opacity_cache_dir = "data/opacities/luhman16b_powerlaw"

set output_dir = "results/m8/v5/free_sigma_geometry"
set summary = "$output_dir/m8_v5_free_sigma_geometry_summary.json"
set arrays = "$output_dir/m8_v5_free_sigma_geometry_arrays.npz"
set log = "$output_dir/run.log"
set provenance = "$output_dir/provenance.sha256"
set provenance_tmp = "$provenance.tmp"
set output_hashes = "$output_dir/outputs.sha256"
set output_hashes_tmp = "$output_hashes.tmp"

set expected_exojax_version = "2.2.dev271+ga4c8f69b6.d20260429"
set expected_exojax_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/exojax/__init__.py"
set expected_numpyro_version = "0.21.0"
set expected_numpyro_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/__init__.py"
set numpyro_continuous_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/distributions/continuous.py"
set gpu_python_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/(m[78](_p[0-9]+)?_v[0-9]+_run|run_milestone5_on_the_fly_atmosphere|make_m7_v1_linearization_check|check_m8_.*(linearization|convergence|validity|geometry))[.]py'

set required_files = ( \
    "$diagnostic_script" \
    "$init_archive" \
    "$free_wrapper" \
    "$free_launcher" \
    "$free_samples" \
    "$free_diagnostics" \
    "$fixed_wrapper" \
    "$replay_helper" \
    "$fixed_launcher" \
    "$fixed_samples" \
    "$fixed_diagnostics" \
    "$initial_replay" \
    "$data_spectra" \
    "$data_template" \
    "$workflow" \
    "$forward_source" \
    "$chip_data_source" \
    "$healpix_source" \
    "$gp_source" \
    "$design_source" \
    "$doppler_source" \
    "$rotation_source" \
    "$limb_source" \
    "$marginal_likelihood_source" \
    "$constants_source" \
    "$expected_numpyro_path" \
    "$numpyro_continuous_source" \
)
set expected_hashes = ( \
    "b1e1cd60e10410d32e5921812f1f9036ede5d7d1b73141818748a42e2b32dfc5" \
    "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89" \
    "fb17a96b1718d65a680ae6162b010cd7f0fac0a577de275d4c8df8a562312423" \
    "c7d6166173dd82b5f82ab63f22312931ac5f9aa3dafb5d302dbac2a6da720203" \
    "aa1bd9bb6bc9e5f146e3377aa62e3adb68c8c9a961812628144c73cd9c3a5083" \
    "f45d32470ee996c62b0dd8683190d4fe5eaedddd37d4f95d0ea21003278cf8c4" \
    "abb1b8ea4b2d0c6ca52af86035a501519503572255f484bdc472039e5b7faf3d" \
    "ed4268237691c120f759626db3f2400f3fd45bd17f2546ef3644ed423cec53ed" \
    "8f8a8fb259819e8de761d13d7205958d93b9e3eb6d74b2f22a76a5daae8fef7c" \
    "3d9e052befbef2541954be71aed5efdb9ad7cb2bcf9d910f0520cba0f5fbc6d2" \
    "8de84b82f08222b3070551ad85222e643573ce1d055db9b8108a6f2a11de1518" \
    "4e535403123ccb92e9ed4d0cc514fab8620035fc913fafca302f22266859fde1" \
    "4ce83dcfa2e5b8f4adbac34b075e02eb8175c476bc999fe6ac25bffa36aae362" \
    "cb98cc5c5401de09a0f574472ea4c95fab742cefc091c371365c014c5f62284a" \
    "582f79edad35a1aa819cf6e619caeaabbac614a1ea173dca25fbbf4f8ce41230" \
    "63390bbf64f436a5ed3cfd022454b22b6e6377aebe909acb21a32a5be0859a21" \
    "c3fe182a9418c16d5c8a010a326869b36201c0fc048714379ab6590f3d020c44" \
    "59c4bee2b82a5b4f72fb47b82645290a8d90f438e0c505f67d93d76455a41719" \
    "8272b2722a7028bdd2a1fb56fde22b36b88ae77bafa060915baff61e47107e62" \
    "0f65ee811f4218fb66fd881cd0906ab5675e7a8f28585a8387406e60512fb45d" \
    "7f1d900c2cc14e14b7e3251d1d5f66b9f746c7c3b8cb7130079f5e2ec92b6637" \
    "fe87269df5ef51846f3fc2a51081e079efbb6fc321c575cf2facae1b49d8c066" \
    "5e42a01f1806fb2f21920de315ce1b69cf80bafa568e61fbbd0326ccef1311f3" \
    "d12f573aefeba9f0275d0bce5f872e42d717b016a7fad14216e8663f307096f3" \
    "c89ce9edaa238fece7c687e429a538c90126307afd31aca7a4e7854e38ec8fa9" \
    "a76bb4e041234be9bd16a2cd968e7ed9fae472f7ab2e2486e117052a3c78400d" \
    "e781bc57cefd43bf91990c4b01b8e9c53da7001641a5c8f6cdc0d38f62f85fc8" \
)

if ( -e "$output_dir" ) then
    echo "[m8_v5_sigma_geometry] refusing output collision: $output_dir"
    exit 1
endif
if ( $#required_files != $#expected_hashes ) then
    echo "[m8_v5_sigma_geometry] internal provenance list mismatch"
    exit 1
endif
foreach required ( $required_files )
    if ( ! -f "$required" ) then
        echo "[m8_v5_sigma_geometry] missing required input: $required"
        exit 1
    endif
end
if ( ! -d "$database_dir" ) then
    echo "[m8_v5_sigma_geometry] missing molecular database: $database_dir"
    exit 1
endif
if ( ! -d "$opacity_cache_dir" ) then
    echo "[m8_v5_sigma_geometry] missing opacity cache: $opacity_cache_dir"
    exit 1
endif

@ hash_index = 1
while ( $hash_index <= $#required_files )
    set actual_hash = `sha256sum "$required_files[$hash_index]" | awk '{print $1}'`
    if ( "$actual_hash" != "$expected_hashes[$hash_index]" ) then
        echo "[m8_v5_sigma_geometry] SHA256 mismatch: $required_files[$hash_index]"
        echo "[m8_v5_sigma_geometry] expected $expected_hashes[$hash_index]"
        echo "[m8_v5_sigma_geometry] actual   $actual_hash"
        exit 1
    endif
    @ hash_index++
end

# Override the broad interactive PYTHONPATH so the frozen run uses the same
# site-packages ExoJAX build as the M8 artifacts, not a local develop checkout.
setenv PYTHONPATH "$repo_root/src"
python -c 'import inspect, os, sys, exojax, numpyro; import numpyro.distributions.continuous as continuous; exojax_path=os.path.realpath(inspect.getfile(exojax)); numpyro_path=os.path.realpath(inspect.getfile(numpyro)); continuous_path=os.path.realpath(inspect.getfile(continuous)); expected_paths=[os.path.realpath(value) for value in (sys.argv[2],sys.argv[4],sys.argv[5])]; print(f"exojax={exojax.__version__} path={exojax_path} numpyro={numpyro.__version__} path={numpyro_path} continuous={continuous_path}"); ok=exojax.__version__ == sys.argv[1] and numpyro.__version__ == sys.argv[3] and [exojax_path,numpyro_path,continuous_path] == expected_paths; sys.exit(0 if ok else 4)' "$expected_exojax_version" "$expected_exojax_path" "$expected_numpyro_version" "$expected_numpyro_path" "$numpyro_continuous_source"
set package_status = $status
if ( $package_status != 0 ) then
    echo "[m8_v5_sigma_geometry] frozen ExoJAX/NumPyro validation failed"
    exit $package_status
endif

set diagnostic_args = ( \
    --init-from "$init_archive" \
    --free-samples "$free_samples" \
    --free-diagnostics "$free_diagnostics" \
    --fixed-samples "$fixed_samples" \
    --fixed-diagnostics "$fixed_diagnostics" \
    --initial-replay "$initial_replay" \
    --out-dir "$output_dir" \
    --seed 0 \
    --profile-sigma-min 0.05 \
    --profile-sigma-max 0.70 \
    --profile-num 33 \
    --sigma-markers 0.27526917,0.32316479086875916,0.42335291,0.53313493 \
    --curvature-sigmas 0.27526917,0.32316479086875916,0.42335291,0.53313493 \
    --cross-u-steps 0.005,0.01,0.02 \
    --potential-atol 0.05 \
    --no-x64 \
)

python "$diagnostic_script" $diagnostic_args --validate-only
set validation_status = $status
if ( $validation_status != 0 ) then
    echo "[m8_v5_sigma_geometry] Python validation failed with status $validation_status"
    exit $validation_status
endif
if ( $validate_only == 1 ) then
    echo "[m8_v5_sigma_geometry] launcher validation passed; no GPU work was started"
    echo "[m8_v5_sigma_geometry] planned log: $log"
    exit 0
endif

set active_gpu_pids = (`pgrep -f "$gpu_python_pattern"`)
if ( $#active_gpu_pids > 0 ) then
    echo "[m8_v5_sigma_geometry] refusing concurrent GPU work; active PID(s): $active_gpu_pids"
    pgrep -af "$gpu_python_pattern"
    exit 2
endif

set output_parent = "$output_dir:h"
mkdir -p "$output_parent"
if ( $status != 0 ) then
    echo "[m8_v5_sigma_geometry] failed to create output parent: $output_parent"
    exit 1
endif
mkdir "$output_dir"
if ( $status != 0 ) then
    echo "[m8_v5_sigma_geometry] failed to claim fresh output directory: $output_dir"
    exit 1
endif
touch "$output_dir/RUNNING"
if ( $status != 0 ) then
    echo "[m8_v5_sigma_geometry] failed to create RUNNING sentinel"
    exit 1
endif

mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /tmp/doraex-jax-cache-m8-v5-sigma-geometry
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache-m8-v5-sigma-geometry
setenv XLA_FLAGS "--xla_gpu_autotune_level=0"
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

echo "[m8_v5_sigma_geometry] start: `date`" | tee "$log"
echo "[m8_v5_sigma_geometry] log: $log" | tee -a "$log"
echo "[m8_v5_sigma_geometry] repository: $repo_root" | tee -a "$log"
echo "[m8_v5_sigma_geometry] git HEAD: `git rev-parse HEAD`" | tee -a "$log"
echo "[m8_v5_sigma_geometry] host: `hostname`" | tee -a "$log"
echo "[m8_v5_sigma_geometry] molecular database: `realpath "$database_dir"`" | tee -a "$log"
echo "[m8_v5_sigma_geometry] opacity cache root: `realpath "$opacity_cache_dir"`" | tee -a "$log"
echo "[m8_v5_sigma_geometry] target: frozen M8 linear marginalized free-sigma target" | tee -a "$log"
echo "[m8_v5_sigma_geometry] diagnostic only: no posterior sampling and no exact RT" | tee -a "$log"
python --version |& tee -a "$log"
git status --short --untracked-files=all >>& "$log"

python -c 'import inspect, os, sys, jax, jaxlib, numpyro, exojax; import numpyro.distributions.continuous as continuous; backend=jax.default_backend(); exojax_path=os.path.realpath(inspect.getfile(exojax)); numpyro_path=os.path.realpath(inspect.getfile(numpyro)); continuous_path=os.path.realpath(inspect.getfile(continuous)); expected_paths=[os.path.realpath(value) for value in (sys.argv[2],sys.argv[4],sys.argv[5])]; print(f"jax={jax.__version__} jaxlib={jaxlib.__version__} numpyro={numpyro.__version__} numpyro_path={numpyro_path} continuous={continuous_path} exojax={exojax.__version__} exojax_path={exojax_path} backend={backend} devices={jax.devices()}"); ok=backend == "gpu" and exojax.__version__ == sys.argv[1] and numpyro.__version__ == sys.argv[3] and [exojax_path,numpyro_path,continuous_path] == expected_paths; sys.exit(0 if ok else 3)' "$expected_exojax_version" "$expected_exojax_path" "$expected_numpyro_version" "$expected_numpyro_path" "$numpyro_continuous_source" >>& "$log"
set backend_status = $status
tail -n 1 "$log"
if ( $backend_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] GPU/ExoJAX preflight failed; refusing fallback." | tee -a "$log"
    exit $backend_status
endif

sha256sum "$script_path" $required_files > "$provenance_tmp"
set provenance_status = $status
if ( $provenance_status != 0 || ! -s "$provenance_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] failed to write provenance hashes" | tee -a "$log"
    if ( $provenance_status == 0 ) set provenance_status = 1
    exit $provenance_status
endif
mv "$provenance_tmp" "$provenance"
set provenance_move_status = $status
if ( $provenance_move_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] failed to finalize provenance hashes" | tee -a "$log"
    exit $provenance_move_status
endif
cat "$provenance" >> "$log"

echo "[m8_v5_sigma_geometry] diagnostic start: `date`" | tee -a "$log"
python "$diagnostic_script" $diagnostic_args >>& "$log"
set run_status = $status
if ( $run_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] diagnostic failed with status ${run_status}: `date`" | tee -a "$log"
    exit $run_status
endif
if ( ! -f "$summary" || ! -f "$arrays" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] diagnostic returned without complete artifacts" | tee -a "$log"
    exit 1
endif

python -c 'import json, sys, numpy as np; summary=json.load(open(sys.argv[1], encoding="utf-8")); arrays=np.load(sys.argv[2], allow_pickle=False); required={"profile_total_potential","cross_hessian_column_raw","curvature_uu_total","probe_cross_hessian_column_raw"}; missing=required-set(arrays.files); finite=(not missing) and all(np.all(np.isfinite(arrays[name])) for name in required); arrays.close(); ok=summary.get("execution_completed") is True and summary.get("numerical_integrity_passed") is True and not missing and finite; print("artifact_gate", summary.get("execution_completed"), summary.get("numerical_integrity_passed"), sorted(missing), finite); sys.exit(0 if ok else 5)' "$summary" "$arrays" >>& "$log"
set artifact_status = $status
if ( $artifact_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] artifact schema validation failed" | tee -a "$log"
    exit $artifact_status
endif

sha256sum "$summary" "$arrays" > "$output_hashes_tmp"
set output_hash_status = $status
if ( $output_hash_status != 0 || ! -s "$output_hashes_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] failed to write output hashes" | tee -a "$log"
    if ( $output_hash_status == 0 ) set output_hash_status = 1
    exit $output_hash_status
endif
mv "$output_hashes_tmp" "$output_hashes"
set output_hash_move_status = $status
if ( $output_hash_move_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v5_sigma_geometry] failed to finalize output hashes" | tee -a "$log"
    exit $output_hash_move_status
endif
cat "$output_hashes" >> "$log"

mv "$output_dir/RUNNING" "$output_dir/COMPLETE"
set complete_status = $status
if ( $complete_status != 0 ) then
    echo "[m8_v5_sigma_geometry] failed to mark completion" | tee -a "$log"
    exit $complete_status
endif
touch "$output_dir/COMPLETE"
if ( $status != 0 ) then
    echo "[m8_v5_sigma_geometry] failed to timestamp completion" | tee -a "$log"
    exit 1
endif
echo "[m8_v5_sigma_geometry] summary: $summary" | tee -a "$log"
echo "[m8_v5_sigma_geometry] arrays: $arrays" | tee -a "$log"
echo "[m8_v5_sigma_geometry] done: `date`" | tee -a "$log"
exit 0
