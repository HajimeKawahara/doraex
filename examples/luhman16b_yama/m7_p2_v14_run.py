"""Run the M7 P2 v14 free-A, fixed-sigma_d isolation pilot."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "results" / "m7" / "p2" / "v14"
DEFAULT_INIT_FROM = ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"

# Orthogonal PCA rotation computed from all 1500 divergence-free M7 v1
# posterior draws after mapping the atmosphere through the P2/Yama prior CDFs.
# It is embedded so this source imports no other m?-run implementation.
# Source NPZ SHA256:
# beb3b10568018bdf356b298fd5cd2ee40bf4aac2d9913233b52577de0884184b
ATMOSPHERE_ROTATION = (
    (0.06397925829127843, -0.08270337684024437, 0.6611067983744282, -0.3698251352599803, 0.10451635727216951, -0.45478619470525417, 0.4443741595196928, -0.003380970275951089),
    (-0.03543646850019254, 0.038341074682813336, -0.3978759391989165, 0.20657547208256347, -0.08279717262584731, 0.08669603429414899, 0.8842273744940578, -0.008118255417685905),
    (-0.6478246782125295, -0.2843710711133721, -0.39800502197699095, -0.37515137017812283, 0.3484127920242713, -0.27711093693666494, -0.045360599887880854, -0.008378729556761874),
    (0.7566887513777871, -0.28994363200708106, -0.407550702346945, -0.2549135844315112, 0.26599192571479824, -0.20039975850845726, -0.03643492967252088, -0.00608616707806528),
    (0.04598488428272582, 0.9092419062044544, -0.17873355798589596, -0.24428317830834648, 0.20134861615371508, -0.19611462290250697, -0.022894385593010755, -0.00426931254170235),
    (-0.0011734616468206326, 0.005797461350278052, 0.15752127498160137, 0.6073741155971859, 0.7745119656760449, -0.07936581020274938, 0.008985481460724299, -0.0005370090974782552),
    (0.016732523454770137, 0.001181343160078176, 0.15275573768586678, -0.4314688912996855, 0.386747701257082, 0.7898936066129074, 0.12882349498640894, -0.010849906715884448),
    (-0.0005168268517514174, -0.00021793504310921655, -0.0058321033272078215, -0.009667777836081234, 0.00969259748889216, 0.0033156069942970596, 0.009385296735157978, 0.9998395784055493),
)


def _has_option(name):
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def _ensure_option(name, value=None):
    if _has_option(name):
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(str(value))


def _rotation_text():
    return ",".join(str(value) for row in ATMOSPHERE_ROTATION for value in row)


def main():
    """Fix sigma_d and log_w while sampling A and the atmosphere."""

    _ensure_option("--chip-indices", "0,1,2,3")
    _ensure_option("--out-dir", DEFAULT_OUT_DIR)
    _ensure_option("--run-label", "m7_p2_v14_free_a_fixed_sigma_d")
    _ensure_option("--init-from", DEFAULT_INIT_FROM)
    _ensure_option("--full-data")
    _ensure_option("--nside", 8)
    _ensure_option("--fixed-ell-b", 0.4)
    _ensure_option("--pressure-gp-factorization", "fixed_eigen")
    _ensure_option("--fix-logg")
    _ensure_option("--init-logg", 4.86)
    _ensure_option("--zero-mean-log-w")
    _ensure_option("--zero-sum-log-w-basis")

    # Relative to v8, release only A. sigma_d and log_w remain fixed at the
    # same component-wise v6 posterior medians used by that control run.
    _ensure_option("--fix-sigma-d")
    _ensure_option("--fix-log-w")
    _ensure_option("--dense-atmosphere-mass")

    # Match the successful v8 adaptation schedule. Its second mass update at
    # iteration 150 was essential; sampling retains max depth 11.
    _ensure_option("--no-preflight-autodiff")
    _ensure_option("--num-warmup", 200)
    _ensure_option("--num-samples", 20)
    _ensure_option("--target-accept-prob", 0.95)
    _ensure_option("--warmup-max-tree-depth", 9)
    _ensure_option("--max-tree-depth", 11)

    _ensure_option("--gaussianized-atmosphere")
    _ensure_option("--atmosphere-rotation-matrix", _rotation_text())
    _ensure_option(
        "--atmosphere-rotation-label",
        "m7_v1_yama_cdf_gaussianized_pca_n1500",
    )
    _ensure_option("--t0-min", 1000.0)
    _ensure_option("--t0-max", 1700.0)
    _ensure_option("--alpha-min", 0.05)
    _ensure_option("--alpha-max", 0.20)
    _ensure_option("--log-vmr-co-min", -6.0)
    _ensure_option("--log-vmr-co-max", -1.0)
    _ensure_option("--log-vmr-h2o-min", -6.0)
    _ensure_option("--log-vmr-h2o-max", -1.0)
    _ensure_option("--log-vmr-ch4-min", -6.0)
    _ensure_option("--log-vmr-ch4-max", -1.0)
    _ensure_option("--log-vmr-hf-min", -10.0)
    _ensure_option("--log-vmr-hf-max", -5.0)
    _ensure_option("--log-p-cloud-min", -2.0)
    _ensure_option("--log-p-cloud-max", 2.0)
    _ensure_option("--sigma-log-p-scale", 0.3)

    _ensure_option("--init-t0", 1219.0)
    _ensure_option("--init-alpha", 0.129)
    _ensure_option("--init-log-vmr-co", -2.96)
    _ensure_option("--init-log-vmr-h2o", -3.25)
    _ensure_option("--init-log-vmr-ch4", -4.65)
    _ensure_option("--init-log-vmr-hf", -7.08)
    _ensure_option("--init-log-p-cloud", 1.45)
    _ensure_option("--init-sigma-log-p", 0.2)

    from doraex.workflows.on_the_fly_pressure_retrieval import main as run_main

    run_main()


if __name__ == "__main__":
    main()
