"""Run the M7 P2 v5 fixed-sigma pressure-map diagnostic pilot.

The default condition fixes ``sigma_log_p=0.245``, near the divergence-free
P1 v2 chain.  A high-amplitude control can be run by overriding
``--fixed-sigma-log-p 0.371``.  All other physical atmosphere priors are the
published Yama et al. priors used by P2 v4.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "results" / "m7" / "p2" / "v5" / "low_sigma"
DEFAULT_INIT_FROM = ROOT / "results" / "m7" / "p1" / "v2" / "samples.npz"

# Seven-dimensional orthogonal rotation for the still-sampled atmosphere
# parameters.  It is the polar-orthogonalized 7x7 bounded-parameter block of
# the P2 v4 rotation.  Embedding it here keeps this pilot independent of every
# other m?-run source file at runtime.
ATMOSPHERE_ROTATION = (
    (
        0.06397838453308154,
        -0.08270374528574881,
        0.661096938499562,
        -0.36984147980573834,
        0.10453274377854778,
        -0.45478058927129034,
        0.4443900264970414,
    ),
    (
        -0.035438566534672096,
        0.03834018998567858,
        -0.3978996143501355,
        0.20653622618972178,
        -0.08275782597878363,
        0.08670949384597251,
        0.8842654736680561,
    ),
    (
        -0.6478268435624228,
        -0.2843719841960052,
        -0.398029456745186,
        -0.3751918752750208,
        0.34845340110808926,
        -0.2770970455352638,
        -0.0453212783022881,
    ),
    (
        0.7566871785043404,
        -0.28994429525482274,
        -0.4075684513482396,
        -0.25494300664708303,
        0.2660214234647052,
        -0.200389668030026,
        -0.03640636713949352,
    ),
    (
        0.04598378094654632,
        0.9092414409507297,
        -0.17874600852050315,
        -0.2443038173464034,
        0.20136930817744947,
        -0.19610754465349314,
        -0.022874349603378547,
    ),
    (
        -0.0011736004283130113,
        0.0057974028290333855,
        0.15751970890971337,
        0.6073715195466296,
        0.774514568391326,
        -0.079364919875776,
        0.008988001657735915,
    ),
    (
        0.01672971946829533,
        0.001180160777794506,
        0.1527240962593539,
        -0.43152134275069376,
        0.3868002873643504,
        0.7899115950690732,
        0.12887441386769152,
    ),
)


def _has_option(name):
    """Return whether a command-line option was explicitly supplied."""

    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def _ensure_option(name, value=None):
    """Append a default command-line option when the user did not set it."""

    if _has_option(name):
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(str(value))


def _rotation_text():
    """Serialize the embedded row-major rotation for the shared workflow."""

    return ",".join(str(value) for row in ATMOSPHERE_ROTATION for value in row)


def main():
    """Run one fixed-sigma diagnostic condition."""

    # M7 data, geometry, and short diagnostic configuration.
    _ensure_option("--chip-indices", "0,1,2,3")
    _ensure_option("--out-dir", DEFAULT_OUT_DIR)
    _ensure_option("--run-label", "m7_p2_v5_low_sigma")
    _ensure_option("--init-from", DEFAULT_INIT_FROM)
    _ensure_option("--full-data")
    _ensure_option("--nside", 8)
    _ensure_option("--fixed-ell-b", 0.4)
    _ensure_option("--fix-logg")
    _ensure_option("--init-logg", 4.86)
    _ensure_option("--fixed-sigma-log-p", 0.245)
    _ensure_option("--zero-mean-log-w")
    _ensure_option("--zero-sum-log-w-basis")
    _ensure_option("--no-preflight-autodiff")
    _ensure_option("--num-warmup", 100)
    _ensure_option("--num-samples", 50)
    _ensure_option("--target-accept-prob", 0.95)
    _ensure_option("--max-tree-depth", 11)

    # Preserve the seven still-active Yama et al. priors exactly.  The pressure
    # amplitude is deliberately fixed, so it is absent from the NUTS state.
    _ensure_option("--gaussianized-atmosphere")
    _ensure_option("--atmosphere-rotation-matrix", _rotation_text())
    _ensure_option(
        "--atmosphere-rotation-label",
        "m7_p2_v4_bounded_block_polar_orthogonalized",
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

    # Yama et al. posterior medians initialize the atmosphere only.  The P1 v2
    # NPZ supplies nuisance initial values and is not part of any prior.
    _ensure_option("--manual-atmosphere-init")
    _ensure_option("--init-t0", 1219.0)
    _ensure_option("--init-alpha", 0.129)
    _ensure_option("--init-log-vmr-co", -2.96)
    _ensure_option("--init-log-vmr-h2o", -3.25)
    _ensure_option("--init-log-vmr-ch4", -4.65)
    _ensure_option("--init-log-vmr-hf", -7.08)
    _ensure_option("--init-log-p-cloud", 1.45)

    from doraex.workflows.on_the_fly_pressure_retrieval import main as run_main

    run_main()


if __name__ == "__main__":
    main()
