#!/usr/bin/env -S tcsh -f

# Repeat the M8/v10 long chain with the cloud-column and Doppler-edge fixes.
# Sampling settings are intentionally identical to v10; only output metadata
# and the frozen source provenance differ.

set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

set run_tag = "m8_v11_highest_long"
set run_label = "m8_v11_free_sigma_highest_precision_long"
set output_dir = "results/m8/v11/free_sigma_highest_precision_long_seed0"
set nohup_parent = "results/m8/v11/nohup"
set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0
set validate_only = 0
set submit_nohup = 0

if ( $#argv > 1 ) then
    echo "[$run_tag] expected no argument, --validate-only, or --nohup"
    exit 2
endif
if ( $#argv == 1 ) then
    if ( "$argv[1]" == "--m8-v11-highest-long-gpu-lock-held" ) then
        set gpu_lock_held = 1
    else if ( "$argv[1]" == "--validate-only" ) then
        set gpu_lock_held = 1
        set validate_only = 1
    else if ( "$argv[1]" == "--nohup" ) then
        set submit_nohup = 1
    else
        echo "[$run_tag] unsupported argument: $argv[1]"
        exit 2
    endif
endif

if ( $submit_nohup == 1 ) then
    if ( -e "$output_dir" ) then
        echo "[$run_tag] refusing output collision: $output_dir"
        exit 1
    endif
    mkdir -p "$nohup_parent"
    if ( $status != 0 ) exit 1
    set outer_log = `mktemp "$nohup_parent/m8_v11_highest_long_nohup.XXXXXX.log"`
    if ( $status != 0 || "$outer_log" == "" ) then
        echo "[$run_tag] failed to claim a unique nohup log"
        exit 1
    endif
    set outer_log = `realpath "$outer_log"`
    setenv DORAEX_M8_V11_OUTER_LOG "$outer_log"
    /usr/bin/nohup "$script_path" < /dev/null >&! "$outer_log" &
    set submit_status = $status
    set submitted_pid = $!
    unsetenv DORAEX_M8_V11_OUTER_LOG
    if ( $submit_status != 0 ) then
        echo "[$run_tag] nohup submission failed with status $submit_status"
        exit $submit_status
    endif
    echo "[$run_tag] submitted PID: $submitted_pid"
    echo "[$run_tag] outer log: $outer_log"
    echo "[$run_tag] run log: $repo_root/$output_dir/run.log"
    exit 0
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        "$script_path" --m8-v11-highest-long-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[$run_tag] not started: another Luhman 16B run holds $gpu_lock"
        exit 2
    endif
    exit $lock_status
endif

set sampler = "examples/luhman16b_yama/m8_v1_run.py"
set init_archive = "results/m7/p2/v6/samples.npz"
set v10_samples = "results/m8/v10/free_sigma_highest_precision_long_seed0/samples.npz"
set v10_diagnostics = "results/m8/v10/free_sigma_highest_precision_long_seed0/diagnostics.json"
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
set doppler_test = "tests/unittests/test_ureshino_reproduction.py"
set exojax_test = "tests/unittests/test_exojax_forward.py"
set m6_source = "examples/luhman16b_yama/m6_v1_run.py"
set pytest_conftest = "tests/unittests/conftest.py"
set database_dir = "/home/kawahara/data_mol/.database"
set opacity_cache_dir = "data/opacities/luhman16b_powerlaw"

set expected_jax_version = "0.10.0"
set expected_jax_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jax/__init__.py"
set expected_jaxlib_version = "0.10.0"
set expected_jaxlib_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jaxlib/__init__.py"
set expected_exojax_version = "2.2.dev271+ga4c8f69b6.d20260429"
set expected_exojax_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/exojax/__init__.py"
set expected_numpyro_version = "0.21.0"
set expected_numpyro_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/__init__.py"
set expected_numpy_version = "2.1.0"
set expected_numpy_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpy/__init__.py"
set expected_scipy_version = "1.17.1"
set expected_scipy_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/scipy/__init__.py"
set numpyro_continuous_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/distributions/continuous.py"
set scipy_linalg_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/scipy/linalg/__init__.py"
set numpyro_hmc_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/infer/hmc_util.py"
set numpyro_initialization_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/infer/initialization.py"
set numpyro_util_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/infer/util.py"
set numpyro_transforms_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/distributions/transforms.py"
set jax_config_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jax/_src/config.py"
set jax_lax_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jax/_src/lax/lax.py"

set required_files = ( \
    "$sampler" \
    "$init_archive" \
    "$v10_samples" \
    "$v10_diagnostics" \
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
    "$doppler_test" \
    "$exojax_test" \
    "$m6_source" \
    "$pytest_conftest" \
    "$expected_exojax_path" \
    "$expected_jax_path" \
    "$expected_jaxlib_path" \
    "$expected_numpyro_path" \
    "$numpyro_continuous_source" \
    "$expected_numpy_path" \
    "$expected_scipy_path" \
    "$scipy_linalg_source" \
    "$numpyro_hmc_source" \
    "$numpyro_initialization_source" \
    "$numpyro_util_source" \
    "$numpyro_transforms_source" \
    "$jax_config_source" \
    "$jax_lax_source" \
)
set expected_hashes = ( \
    "fb17a96b1718d65a680ae6162b010cd7f0fac0a577de275d4c8df8a562312423" \
    "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89" \
    "c79d8f2c24e25a35bd5827628af8b2e0e5b14202ca7e8229f4545b338e46209f" \
    "140b52e99977c82f931da150f7d9c265cd909ba5f9f28be801290aa9d07a26d3" \
    "4ce83dcfa2e5b8f4adbac34b075e02eb8175c476bc999fe6ac25bffa36aae362" \
    "cb98cc5c5401de09a0f574472ea4c95fab742cefc091c371365c014c5f62284a" \
    "a3ece523a003e123bbf3b333088d0549f89bae77d917d1fa2aa26555e24632c3" \
    "6e4626b6a1d01c1f5876d1d03ae59a6b01ed6c3882b91d410e3a0d9095d4cf72" \
    "c3fe182a9418c16d5c8a010a326869b36201c0fc048714379ab6590f3d020c44" \
    "59c4bee2b82a5b4f72fb47b82645290a8d90f438e0c505f67d93d76455a41719" \
    "8272b2722a7028bdd2a1fb56fde22b36b88ae77bafa060915baff61e47107e62" \
    "2b341eed491a3c37b40bdb78fd1618c51b9ef09576fe90276c808cb636d92f94" \
    "09f2672ebf56ebe35feb81a4e32c94ab2717e952134d75ad66f13551a6ac6757" \
    "fe87269df5ef51846f3fc2a51081e079efbb6fc321c575cf2facae1b49d8c066" \
    "5e42a01f1806fb2f21920de315ce1b69cf80bafa568e61fbbd0326ccef1311f3" \
    "d12f573aefeba9f0275d0bce5f872e42d717b016a7fad14216e8663f307096f3" \
    "c89ce9edaa238fece7c687e429a538c90126307afd31aca7a4e7854e38ec8fa9" \
    "edf1496a300329f4f9177dac7096965bcfb55a48ab4979ca7106dca8269ef04a" \
    "99ec0eaa0967d05cc29de0bc3367fbb1e304798dbd409655aa3f9631366fd912" \
    "bcd857b9ce6937d392fb159ec667a3513440fd0fc2ecfc19a63afa3e0870b9dc" \
    "4eff04d66a5a3077dee4868559fa66d287251067c53038867da31838caa88384" \
    "f8bef220648b129cd61146b0df2a0262e1ae82b0a3be3d454b4e26a41dadb303" \
    "8317d2884580454b67ec3342bea57c3f3e83c9027702ee682e410ddb09a9cad2" \
    "2a5b37b2bc9802769f45ca43de7cdc1a8b532f0afd2bf9542131ab68f649c4a3" \
    "a76bb4e041234be9bd16a2cd968e7ed9fae472f7ab2e2486e117052a3c78400d" \
    "e781bc57cefd43bf91990c4b01b8e9c53da7001641a5c8f6cdc0d38f62f85fc8" \
    "39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1" \
    "335bc6e0a9909dc7534f9569a3685a92dc8001cb8c63a6da4c239849ff02d4d0" \
    "8840662d267e8d06503125608c6333e5f50053750f1e2c273b63e5a6c0b79c1c" \
    "37eefae77a9a7df97c460c4cfbfdba34eb7b88a3a3a403e8936ff75a2bafb031" \
    "0c5bd0e8ab55ada98e029ba2f49edbc611d5fd0b60ae607e49a3cf7733b8a9b3" \
    "9e73fafb9ce8f072c431edd8297f5dadc07d7edc5a2a769b756a0f3cef0a9149" \
    "7932fa3f2bb4f68f6db9f8ad3550a45dd2a3f3cbb468bfb4e18748d897472a43" \
    "a9a2bccf86b35734171699cb34e2c9142add5c9bfbd5422941e1a6b192d25aad" \
    "d4da422029482f0c107a376b483b7e99f5cdde3c18a96f6201dc82b1a5fd0cc6" \
)

set samples = "$output_dir/samples.npz"
set diagnostics = "$output_dir/diagnostics.json"
set log = "$output_dir/run.log"
set provenance = "$output_dir/provenance.sha256"
set provenance_tmp = "$provenance.tmp"
set output_hashes = "$output_dir/outputs.sha256"
set output_hashes_tmp = "$output_hashes.tmp"

if ( -e "$output_dir" && $validate_only == 0 ) then
    echo "[$run_tag] refusing output collision: $output_dir"
    exit 1
endif
if ( $#required_files != $#expected_hashes ) then
    echo "[$run_tag] source provenance list is malformed"
    exit 1
endif
foreach required ( $required_files )
    if ( ! -f "$required" ) then
        echo "[$run_tag] missing required input: $required"
        exit 1
    endif
end
if ( ! -d "$database_dir" || ! -d "$opacity_cache_dir" ) then
    echo "[$run_tag] molecular database or opacity cache root is missing"
    exit 1
endif
if ( $?XLA_FLAGS ) then
    echo "[$run_tag] refusing ambient XLA_FLAGS: $XLA_FLAGS"
    exit 1
endif
if ( $?JAX_DEFAULT_MATMUL_PRECISION ) then
    echo "[$run_tag] refusing ambient JAX_DEFAULT_MATMUL_PRECISION: $JAX_DEFAULT_MATMUL_PRECISION"
    exit 1
endif

@ hash_index = 1
while ( $hash_index <= $#required_files )
    set actual_hash = `sha256sum "$required_files[$hash_index]" | awk '{print $1}'`
    if ( "$actual_hash" != "$expected_hashes[$hash_index]" ) then
        echo "[$run_tag] SHA256 mismatch: $required_files[$hash_index]"
        echo "[$run_tag] expected $expected_hashes[$hash_index]"
        echo "[$run_tag] actual   $actual_hash"
        exit 1
    endif
    @ hash_index++
end

setenv PYTHONPATH "$repo_root/src"
setenv PYTHONDONTWRITEBYTECODE 1
setenv JAX_DEFAULT_MATMUL_PRECISION highest

python -c 'import inspect,os,sys,jax,jaxlib,exojax,numpyro,numpy,scipy; versions=[jax.__version__,jaxlib.__version__,exojax.__version__,numpyro.__version__,numpy.__version__,scipy.__version__]; wanted=list(sys.argv[1:7]); paths=[os.path.realpath(inspect.getfile(x)) for x in (jax,jaxlib,exojax,numpyro,numpy,scipy)]; expected=[os.path.realpath(x) for x in sys.argv[7:13]]; precision=jax.config.jax_default_matmul_precision; print(f"versions={versions} paths={paths} precision={precision}"); sys.exit(0 if versions==wanted and paths==expected and precision=="highest" else 4)' \
    "$expected_jax_version" "$expected_jaxlib_version" "$expected_exojax_version" \
    "$expected_numpyro_version" "$expected_numpy_version" "$expected_scipy_version" \
    "$expected_jax_path" "$expected_jaxlib_path" "$expected_exojax_path" \
    "$expected_numpyro_path" "$expected_numpy_path" "$expected_scipy_path"
if ( $status != 0 ) then
    echo "[$run_tag] frozen package or precision validation failed"
    exit 1
endif

python -m pytest -q "$doppler_test" "$exojax_test"
if ( $status != 0 ) then
    echo "[$run_tag] cloud/Doppler regression tests failed"
    exit 1
endif

if ( $validate_only == 1 ) then
    echo "[$run_tag] launcher validation passed; no GPU work was started"
    echo "[$run_tag] one-line submission: $script_path --nohup"
    echo "[$run_tag] planned output: $output_dir"
    exit 0
endif

set gpu_python_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/(m[78](_p[0-9]+)?_v[0-9]+_run|m8_v1_run|check_m8_.*)[.]py'
set active_gpu_pids = (`pgrep -f "$gpu_python_pattern"`)
if ( $#active_gpu_pids > 0 ) then
    echo "[$run_tag] refusing concurrent GPU work; active PID(s): $active_gpu_pids"
    pgrep -af "$gpu_python_pattern"
    exit 2
endif

mkdir -p "$output_dir:h"
if ( $status != 0 ) exit 1
mkdir "$output_dir"
if ( $status != 0 ) then
    echo "[$run_tag] failed to claim fresh output directory"
    exit 1
endif
touch "$output_dir/RUNNING"
if ( $status != 0 ) exit 1

mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /tmp/doraex-jax-cache-m8-v11-highest-long
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache-m8-v11-highest-long
setenv PYTHONUNBUFFERED 1
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

echo "[$run_tag] start: `date`" | tee "$log"
echo "[$run_tag] log: $log" | tee -a "$log"
if ( $?DORAEX_M8_V11_OUTER_LOG ) then
    echo "[$run_tag] outer nohup log: $DORAEX_M8_V11_OUTER_LOG" | tee -a "$log"
endif
echo "[$run_tag] repository: $repo_root" | tee -a "$log"
echo "[$run_tag] git HEAD: `git rev-parse HEAD`" | tee -a "$log"
echo "[$run_tag] host: `hostname`" | tee -a "$log"
echo "[$run_tag] intervention: cloud layer-width and Doppler-profile padding fixes" | tee -a "$log"
echo "[$run_tag] schedule: seed=0 chains=1 warmup=2000 samples=1500 target_accept=0.95 depths=9,11" | tee -a "$log"
python --version |& tee -a "$log"
git status --short --untracked-files=normal >>& "$log"

python -c 'import inspect,os,sys,jax,jaxlib,exojax,numpyro,numpy,scipy; backend=jax.default_backend(); devices=jax.devices(); versions=[jax.__version__,jaxlib.__version__,exojax.__version__,numpyro.__version__,numpy.__version__,scipy.__version__]; wanted=list(sys.argv[1:7]); paths=[os.path.realpath(inspect.getfile(x)) for x in (jax,jaxlib,exojax,numpyro,numpy,scipy)]; expected=[os.path.realpath(x) for x in sys.argv[7:13]]; precision=jax.config.jax_default_matmul_precision; print(f"versions={versions} paths={paths} precision={precision} backend={backend} devices={devices}"); gpu=backend=="gpu" and devices and all(x.platform=="gpu" for x in devices); sys.exit(0 if gpu and versions==wanted and paths==expected and precision=="highest" else 3)' \
    "$expected_jax_version" "$expected_jaxlib_version" "$expected_exojax_version" \
    "$expected_numpyro_version" "$expected_numpy_version" "$expected_scipy_version" \
    "$expected_jax_path" "$expected_jaxlib_path" "$expected_exojax_path" \
    "$expected_numpyro_path" "$expected_numpy_path" "$expected_scipy_path" >>& "$log"
set backend_status = $status
tail -n 1 "$log"
if ( $backend_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] GPU/package/precision preflight failed" | tee -a "$log"
    exit $backend_status
endif

sha256sum "$script_path" $required_files > "$provenance_tmp"
if ( $status != 0 || ! -s "$provenance_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif
mv "$provenance_tmp" "$provenance"
sha256sum -c "$provenance" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif

echo "[$run_tag] sampler start: `date`" | tee -a "$log"
python -u "$sampler" \
    --init-from "$repo_root/$init_archive" \
    --out-dir "$repo_root/$output_dir" \
    --run-label "$run_label" \
    --num-chains 1 \
    --seed 0 \
    --num-warmup 2000 \
    --num-samples 1500 \
    --target-accept-prob 0.95 \
    --warmup-max-tree-depth 9 \
    --max-tree-depth 11 \
    --dense-mass \
    --adapt-mass-matrix \
    --no-x64 \
    --print-summary \
    >>& "$log"
set run_status = $status
if ( $run_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] sampler failed with status ${run_status}: `date`" | tee -a "$log"
    exit $run_status
endif
if ( ! -f "$samples" || ! -f "$diagnostics" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] sampler returned without required artifacts" | tee -a "$log"
    exit 1
endif

sha256sum -c "$provenance" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] frozen inputs changed during sampling" | tee -a "$log"
    exit 1
endif

python -c 'import json,sys,numpy as np; from doraex.operators.doppler import doppler_padded_wavelengths; path,label,diag_path=sys.argv[1:4]; a=np.load(path,allow_pickle=False); chips=[int(x) for x in np.asarray(a["chip_indices"])]; vmax=float(np.max(np.abs(np.asarray(a["v"],dtype=float)))); beta=vmax/299792.458; factor=np.sqrt((1.0+beta)/(1.0-beta)); grids=all((f"profile_wavelengths_chip{c}" in a.files) for c in chips); coverage=grids and all((lambda o,p: p.ndim==1 and np.all(np.isfinite(p)) and np.all(np.diff(p)>0.0) and p[0]<o[0]/factor and p[-1]>o[-1]*factor and np.array_equal(p,doppler_padded_wavelengths(o,vmax)))(np.asarray(a[f"wavelengths_chip{c}"]),np.asarray(a[f"profile_wavelengths_chip{c}"])) for c in chips); sample_keys=("A","sigma_log_p","extra_num_steps","extra_diverging","extra_accept_prob","extra_potential_energy"); finite=all(k in a.files and np.asarray(a[k]).shape[0]==1500 and np.all(np.isfinite(np.asarray(a[k],dtype=float))) for k in sample_keys); metadata=str(np.asarray(a["run_label"]).item())==label; d=json.load(open(diag_path,encoding="utf-8")); diagnostics=d.get("mode")==label and d.get("num_warmup")==2000 and d.get("num_samples")==1500; print({"profile_grids":coverage,"finite_samples":finite,"metadata":metadata,"diagnostics":diagnostics}); sys.exit(0 if coverage and finite and metadata and diagnostics else 6)' \
    "$samples" "$run_label" "$diagnostics" >>& "$log"
set artifact_status = $status
tail -n 1 "$log"
if ( $artifact_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] padded-profile artifact validation failed" | tee -a "$log"
    exit $artifact_status
endif

sha256sum "$samples" "$diagnostics" "$provenance" > "$output_hashes_tmp"
if ( $status != 0 || ! -s "$output_hashes_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif
mv "$output_hashes_tmp" "$output_hashes"
sha256sum -c "$output_hashes" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif

mv "$output_dir/RUNNING" "$output_dir/COMPLETE"
if ( $status != 0 ) exit 1
touch "$output_dir/COMPLETE"
echo "[$run_tag] outputs: $output_hashes" | tee -a "$log"
echo "[$run_tag] done: `date`" | tee -a "$log"
