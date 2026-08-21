#!/usr/bin/env -S tcsh -f

# Repeat the M8/v11 physical target after excluding exact-zero flux sentinels.
# sigma_d is sampled because the sentinel-contaminated fixed values are not
# part of the M8/v12 target. All other physical priors and sampling settings
# match M8/v11.

set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

set run_tag = "m8_v12_masked_free_sigma_d_highest_long"
set run_label = "m8_v12_masked_free_sigma_d_highest_precision_long"
set output_dir = "results/m8/v12/masked_free_sigma_d_highest_precision_long_seed0"
set nohup_parent = "results/m8/v12/nohup"
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
    if ( "$argv[1]" == "--m8-v12-masked-free-sigma-d-gpu-lock-held" ) then
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
    set outer_log = `mktemp "$nohup_parent/m8_v12_masked_free_sigma_d_nohup.XXXXXX.log"`
    if ( $status != 0 || "$outer_log" == "" ) then
        echo "[$run_tag] failed to claim a unique nohup log"
        exit 1
    endif
    set outer_log = `realpath "$outer_log"`
    setenv DORAEX_M8_V12_OUTER_LOG "$outer_log"
    /usr/bin/nohup "$script_path" < /dev/null >&! "$outer_log" &
    set submit_status = $status
    set submitted_pid = $!
    unsetenv DORAEX_M8_V12_OUTER_LOG
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
        "$script_path" --m8-v12-masked-free-sigma-d-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[$run_tag] not started: another Luhman 16B run holds $gpu_lock"
        exit 2
    endif
    exit $lock_status
endif

set workflow = "src/doraex/workflows/on_the_fly_pressure_retrieval.py"
set v11_dir = "results/m8/v11/free_sigma_highest_precision_long_seed0"
set init_archive = "$v11_dir/samples.npz"
set v11_complete = "$v11_dir/COMPLETE"
set v11_outputs = "$v11_dir/outputs.sha256"
set data_spectra = "data/fainterspectral-fits_6.pickle"
set data_template = "data/posterior_predictive_vsini=0.npz"
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
set mask_test = "tests/unittests/test_pressure_retrieval_reparameterization.py"
set exojax_test = "tests/unittests/test_exojax_forward.py"
set doppler_test = "tests/unittests/test_ureshino_reproduction.py"
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
    "$init_archive" \
    "$v11_complete" \
    "$v11_outputs" \
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
    "$mask_test" \
    "$exojax_test" \
    "$doppler_test" \
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
    "e1708e6e9c681691c40dc2ae8d7c5dac62fa0c3663aef41b12d115565c8d83f9" \
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" \
    "4b1d4b3130aaabcfd466d87b0b5c9e5510db279a339fc1ffd68ab467c035f31f" \
    "4ce83dcfa2e5b8f4adbac34b075e02eb8175c476bc999fe6ac25bffa36aae362" \
    "cb98cc5c5401de09a0f574472ea4c95fab742cefc091c371365c014c5f62284a" \
    "17db94df684c76069d82bd8a9708bdde8cb9c6d2c6425bbd5ba62a40375f487a" \
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
    "9a8f9cba7793cbc2a704925021682525d14f7ccd2fffc1e884be5a830453f319" \
    "f79406140b0896bdcadd7506cc15eb754df2a9afb9dac2e7c0c0a33793088a6b" \
    "edf1496a300329f4f9177dac7096965bcfb55a48ab4979ca7106dca8269ef04a" \
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
foreach expected_hash ( $expected_hashes )
    if ( "$expected_hash" =~ "REPLACE_WITH_FROZEN_*" ) then
        echo "[$run_tag] source provenance contains an unfrozen SHA256 placeholder"
        exit 1
    endif
end
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

sha256sum -c "$v11_outputs"
if ( $status != 0 ) then
    echo "[$run_tag] M8/v11 initialization lineage validation failed"
    exit 1
endif
if ( -e "$v11_dir/FAILED" || -e "$v11_dir/RUNNING" ) then
    echo "[$run_tag] M8/v11 initialization lineage is not cleanly complete"
    exit 1
endif

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

python -c 'import sys,numpy as np; from doraex.data.luhman16b import load_luhman16b_chip; expected=[[],[],[0,1,1022,1023],[0,1,1022,1023]]; chips=[load_luhman16b_chip(sys.argv[1],chip_index=c) for c in range(4)]; actual=[np.flatnonzero(np.any(c.flux==0.0,axis=0)).tolist() for c in chips]; static=all(np.array_equal(c.flux==0.0,np.broadcast_to(np.all(c.flux==0.0,axis=0),c.flux.shape)) for c in chips); finite=all(np.all(np.isfinite(c.flux)) for c in chips); counts=[int(np.count_nonzero(c.flux==0.0)) for c in chips]; print({"zero_wavelength_indices":actual,"zero_counts":counts,"static":static,"finite":finite}); sys.exit(0 if actual==expected and counts==[0,0,56,56] and static and finite else 5)' "$repo_root/data"
if ( $status != 0 ) then
    echo "[$run_tag] zero-flux sentinel contract validation failed"
    exit 1
endif

python -m pytest -q "$mask_test" "$exojax_test" "$doppler_test"
if ( $status != 0 ) then
    echo "[$run_tag] mask/cloud/Doppler regression tests failed"
    exit 1
endif

if ( $validate_only == 1 ) then
    echo "[$run_tag] launcher validation passed; no GPU work was started"
    echo "[$run_tag] one-line submission: cd $repo_root && tcsh $script_path --nohup"
    echo "[$run_tag] planned output: $output_dir"
    exit 0
endif

set gpu_python_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/(m[78](_p[0-9]+)?_v[0-9]+_run|m8_v1_run|check_m8_.*)[.]py|python[0-9.]*[[:space:]].*src/doraex/workflows/on_the_fly_pressure_retrieval[.]py'
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
mkdir -p /tmp/doraex-jax-cache-m8-v12-masked-free-sigma-d
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache-m8-v12-masked-free-sigma-d
setenv PYTHONUNBUFFERED 1
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

echo "[$run_tag] start: `date`" | tee "$log"
echo "[$run_tag] log: $log" | tee -a "$log"
if ( $?DORAEX_M8_V12_OUTER_LOG ) then
    echo "[$run_tag] outer nohup log: $DORAEX_M8_V12_OUTER_LOG" | tee -a "$log"
endif
echo "[$run_tag] repository: $repo_root" | tee -a "$log"
echo "[$run_tag] git HEAD: `git rev-parse HEAD`" | tee -a "$log"
echo "[$run_tag] host: `hostname`" | tee -a "$log"
echo "[$run_tag] intervention: mask exact-zero flux sentinels and sample sigma_d" | tee -a "$log"
echo "[$run_tag] initialization: $init_archive" | tee -a "$log"
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
@ provenance_hash_index = 1
@ provenance_line = 2
while ( $provenance_hash_index <= $#required_files )
    set recorded_hash = `sed -n "${provenance_line}p" "$provenance_tmp" | awk '{print $1}'`
    if ( "$recorded_hash" != "$expected_hashes[$provenance_hash_index]" ) then
        mv "$output_dir/RUNNING" "$output_dir/FAILED"
        touch "$output_dir/FAILED"
        echo "[$run_tag] frozen input changed during launch: $required_files[$provenance_hash_index]" | tee -a "$log"
        exit 1
    endif
    @ provenance_hash_index++
    @ provenance_line++
end
mv "$provenance_tmp" "$provenance"
sha256sum -c "$provenance" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif

set atmosphere_rotation = "0.06397838453308154,-0.08270374528574881,0.661096938499562,-0.36984147980573834,0.10453274377854778,-0.45478058927129034,0.4443900264970414,-0.035438566534672096,0.03834018998567858,-0.3978996143501355,0.20653622618972178,-0.08275782597878363,0.08670949384597251,0.8842654736680561,-0.6478268435624228,-0.2843719841960052,-0.398029456745186,-0.3751918752750208,0.34845340110808926,-0.2770970455352638,-0.0453212783022881,0.7566871785043404,-0.28994429525482274,-0.4075684513482396,-0.25494300664708303,0.2660214234647052,-0.200389668030026,-0.03640636713949352,0.04598378094654632,0.9092414409507297,-0.17874600852050315,-0.2443038173464034,0.20136930817744947,-0.19610754465349314,-0.022874349603378547,-0.0011736004283130113,0.0057974028290333855,0.15751970890971337,0.6073715195466296,0.774514568391326,-0.079364919875776,0.008988001657735915,0.01672971946829533,0.001180160777794506,0.1527240962593539,-0.43152134275069376,0.3868002873643504,0.7899115950690732,0.12887441386769152"

echo "[$run_tag] sampler start: `date`" | tee -a "$log"
python -u "$workflow" \
    --data-dir "$repo_root/data" \
    --opacity-cache-dir "$repo_root/$opacity_cache_dir" \
    --database-dir "$database_dir" \
    --chip-indices 0,1,2,3 \
    --init-from "$repo_root/$init_archive" \
    --out-dir "$repo_root/$output_dir" \
    --run-label "$run_label" \
    --full-data \
    --mask-zero-flux \
    --nx 4500 \
    --nside 8 \
    --fixed-period 4.83 \
    --fixed-cosi 0.485 \
    --fixed-v 31.2 \
    --fixed-q1 0.81 \
    --fixed-q2 0.59 \
    --fixed-ell-b 0.4 \
    --gp-jitter 0.5e-6 \
    --noise-jitter 1.0e-6 \
    --pressure-gp-factorization fixed_eigen \
    --zero-mean-pressure-map \
    --fix-logg \
    --init-logg 4.86 \
    --zero-mean-log-w \
    --zero-sum-log-w-basis \
    --log-w-scale 0.1 \
    --fix-log-w \
    --gaussianized-atmosphere \
    --direct-sigma-log-p \
    --atmosphere-rotation-matrix "$atmosphere_rotation" \
    --atmosphere-rotation-label m7_v1_yama_cdf_gaussianized_bounded7_polar_n1500 \
    --t0-min 1000.0 \
    --t0-max 1700.0 \
    --alpha-min 0.05 \
    --alpha-max 0.20 \
    --log-vmr-co-min -6.0 \
    --log-vmr-co-max -1.0 \
    --log-vmr-h2o-min -6.0 \
    --log-vmr-h2o-max -1.0 \
    --log-vmr-ch4-min -6.0 \
    --log-vmr-ch4-max -1.0 \
    --log-vmr-hf-min -10.0 \
    --log-vmr-hf-max -5.0 \
    --log-p-cloud-min -2.0 \
    --log-p-cloud-max 2.0 \
    --sigma-log-p-scale 0.3 \
    --init-t0 1219.0 \
    --init-alpha 0.129 \
    --init-log-vmr-co -2.96 \
    --init-log-vmr-h2o -3.25 \
    --init-log-vmr-ch4 -4.65 \
    --init-log-vmr-hf -7.08 \
    --init-log-p-cloud 1.45 \
    --init-sigma-log-p 0.2 \
    --num-chains 1 \
    --seed 0 \
    --num-warmup 2000 \
    --num-samples 1500 \
    --target-accept-prob 0.95 \
    --warmup-max-tree-depth 9 \
    --max-tree-depth 11 \
    --dense-mass \
    --adapt-mass-matrix \
    --no-preflight-autodiff \
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

python -c 'import sys,numpy as np; a=np.load(sys.argv[1],allow_pickle=False); chips=np.asarray(a["chip_indices"],dtype=int).tolist(); shapes=[np.asarray(a[f"flux_chip{c}"]).shape for c in chips]; print({"chip_indices":chips,"flux_shapes":shapes}); sys.exit(0 if chips==[0,1,2,3] and shapes==[(14,1024)]*4 else 7)' "$samples" >>& "$log"
set data_contract_status = $status
tail -n 1 "$log"
if ( $data_contract_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] chip data contract validation failed" | tee -a "$log"
    exit $data_contract_status
endif

python -c 'import json,sys,numpy as np; from pathlib import Path; from doraex.operators.doppler import doppler_padded_wavelengths; path,label,diag_path,init_path=sys.argv[1:5]; a=np.load(path,allow_pickle=False); chips=[int(x) for x in np.asarray(a["chip_indices"])]; vmax=float(np.max(np.abs(np.asarray(a["v"],dtype=float)))); beta=vmax/299792.458; factor=np.sqrt((1.0+beta)/(1.0-beta)); grids=all((f"profile_wavelengths_chip{c}" in a.files) for c in chips); coverage=grids and all((lambda o,p: p.ndim==1 and np.all(np.isfinite(p)) and np.all(np.diff(p)>0.0) and p[0]<o[0]/factor and p[-1]>o[-1]*factor and np.array_equal(p,doppler_padded_wavelengths(o,vmax)))(np.asarray(a[f"wavelengths_chip{c}"]),np.asarray(a[f"profile_wavelengths_chip{c}"])) for c in chips); sample_keys=("A","sigma_log_p","sigma_d","extra_num_steps","extra_diverging","extra_accept_prob","extra_potential_energy"); finite=all(k in a.files and np.asarray(a[k]).shape[0]==1500 and np.all(np.isfinite(np.asarray(a[k],dtype=float))) for k in sample_keys); sigma=np.asarray(a["sigma_d"],dtype=float); free_sigma=sigma.shape==(1500,4) and np.all(sigma>0.0) and np.all(np.ptp(sigma,axis=0)>0.0) and not bool(np.asarray(a["fix_sigma_d"]).item()); masks=[np.asarray(a[f"observation_mask_chip{c}"],dtype=bool) for c in chips]; fluxes=[np.asarray(a[f"flux_chip{c}"]) for c in chips]; expected_masks=[np.isfinite(f)&(f!=0.0) for f in fluxes]; mask_match=all(m.shape==f.shape and np.array_equal(m,e) for m,f,e in zip(masks,fluxes,expected_masks)); expected_indices=np.flatnonzero(np.stack(expected_masks,axis=0).reshape(-1)); indices=np.asarray(a["observation_indices"],dtype=int); counts=[int(m.sum()) for m in masks]; mask_contract=bool(np.asarray(a["mask_zero_flux"]).item()) and str(np.asarray(a["observation_mask_rule"]).item())=="finite_and_nonzero_flux" and int(np.asarray(a["observation_valid_count"]).item())==57232 and int(np.asarray(a["observation_excluded_count"]).item())==112 and counts==[14336,14336,14280,14280] and np.array_equal(indices,expected_indices); metadata=str(np.asarray(a["run_label"]).item())==label and str(Path(str(np.asarray(a["init_from"]).item())).resolve())==str(Path(init_path).resolve()) and bool(np.asarray(a["fix_log_w"]).item()) and not bool(np.asarray(a["fix_a"]).item()) and set(np.asarray(a["fixed_nuisance_sites"],dtype=str).tolist())=={"log_w"} and not bool(np.asarray(a["x64"]).item()); d=json.load(open(diag_path,encoding="utf-8")); diagnostics=d.get("mode")==label and d.get("num_warmup")==2000 and d.get("num_samples")==1500 and d.get("mask_zero_flux") is True and d.get("observation_mask_rule")=="finite_and_nonzero_flux" and d.get("observation_valid_count")==57232 and d.get("observation_excluded_count")==112 and d.get("observation_excluded_count_by_chip")=={"0":0,"1":0,"2":56,"3":56} and d.get("fix_sigma_d") is False; print({"profile_grids":coverage,"finite_samples":finite,"free_sigma_d":free_sigma,"mask_contract":mask_contract,"metadata":metadata,"diagnostics":diagnostics}); sys.exit(0 if coverage and finite and free_sigma and mask_match and mask_contract and metadata and diagnostics else 6)' \
    "$samples" "$run_label" "$diagnostics" "$repo_root/$init_archive" >>& "$log"
set artifact_status = $status
tail -n 1 "$log"
if ( $artifact_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] masked free-sigma artifact validation failed" | tee -a "$log"
    exit $artifact_status
endif

python -c 'import json,sys,numpy as np; a=np.load(sys.argv[1],allow_pickle=False); d=json.load(open(sys.argv[2],encoding="utf-8")); diverging=np.asarray(a["extra_diverging"],dtype=bool); steps=np.asarray(a["extra_num_steps"],dtype=int); accept=np.asarray(a["extra_accept_prob"],dtype=float); final_step=float(d.get("final_step_size",np.nan)); mean_accept=float(d.get("mean_accept_prob",np.nan)); tree_cap=int(d.get("tree_depth_cap_num_steps",0)); healthy=not np.any(diverging) and int(d.get("divergence_count",-1))==0 and int(d.get("tree_depth_cap_count",-1))==0 and tree_cap>0 and int(np.max(steps))<tree_cap and int(d.get("max_num_steps",-1))==int(np.max(steps)) and np.isfinite(final_step) and final_step>0.0 and np.isfinite(mean_accept) and 0.90<=mean_accept<=1.0 and np.isclose(mean_accept,float(np.mean(accept)),rtol=1.0e-6,atol=1.0e-8); print({"divergence_count":int(d.get("divergence_count",-1)),"tree_depth_cap_count":int(d.get("tree_depth_cap_count",-1)),"max_num_steps":int(np.max(steps)),"final_step_size":final_step,"mean_accept_prob":mean_accept,"healthy":healthy}); sys.exit(0 if healthy else 8)' "$samples" "$diagnostics" >>& "$log"
set sampler_health_status = $status
tail -n 1 "$log"
if ( $sampler_health_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[$run_tag] sampler health validation failed" | tee -a "$log"
    exit $sampler_health_status
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
