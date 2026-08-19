"""Summarize the reused free v17 arm and completed M8 v2 fixed arm."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size


ROOT = Path(__file__).resolve().parents[2]
MAJOR_SITES = (
    "T0",
    "alpha",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
    "log_p_cloud",
    "A",
    "sigma_log_p",
)


def parse_args():
    """Parse summary paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--free-samples",
        default=str(ROOT / "results" / "m7" / "p2" / "v17" / "samples.npz"),
    )
    parser.add_argument(
        "--free-diagnostics",
        default=str(
            ROOT / "results" / "m7" / "p2" / "v17" / "diagnostics.json"
        ),
    )
    parser.add_argument(
        "--fixed-dir",
        default=str(ROOT / "results" / "m8" / "v2" / "fixed_seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _json_value(value):
    """Convert one NumPy value to JSON-compatible scalars and lists."""

    value = np.asarray(value)
    return float(value) if value.shape == () else value.tolist()


def _site_ess(values):
    """Compute single-chain ESS for one retained sample site."""

    values = np.asarray(values)
    ess = np.asarray(
        effective_sample_size(jnp.asarray(values[None, ...]))
    )
    return ess


def summarize_arm(samples_path, diagnostics_path, *, case, seed):
    """Summarize one short NUTS arm with an explicit work proxy."""

    samples_path = Path(samples_path)
    diagnostics_path = Path(diagnostics_path)
    if not samples_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"Missing {case} arm outputs: {samples_path}, {diagnostics_path}"
        )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    with np.load(samples_path, allow_pickle=False) as samples:
        steps = np.asarray(samples["extra_num_steps"], dtype=int)
        accept_prob = np.asarray(samples["extra_accept_prob"], dtype=float)
        potential = np.asarray(samples["extra_potential_energy"], dtype=float)
        divergence = np.asarray(samples["extra_diverging"], dtype=bool)
        retained_step_proxy = int(np.sum(steps))
        site_ess = {}
        ess_per_step_proxy = {}
        finite_sites = {}
        for name in MAJOR_SITES:
            if name not in samples:
                continue
            values = np.asarray(samples[name])
            ess = _site_ess(values)
            site_ess[name] = _json_value(ess)
            ess_per_step_proxy[name] = _json_value(
                ess / retained_step_proxy
            )
            finite_sites[name] = bool(np.all(np.isfinite(values)))

    sampling_depth = int(diagnostics["max_tree_depth"])
    cap_steps = 2**sampling_depth - 1
    cap_count = int(np.sum(steps == cap_steps))
    cap_fraction = float(cap_count / steps.size)
    median_steps = float(np.median(steps))
    divergence_count = int(np.sum(divergence))
    passes_diagnostic_targets = bool(
        divergence_count == 0
        and cap_fraction < 0.05
        and median_steps < 512
    )
    return {
        "case": case,
        "seed": seed,
        "run_label": diagnostics["mode"],
        "target_dimension_note": (
            "fixed sigma_log_p is absent from the target"
            if case == "fixed"
            else "sigma_log_p is a direct HalfNormal target coordinate"
        ),
        "num_warmup": diagnostics["num_warmup"],
        "num_samples": diagnostics["num_samples"],
        "warmup_depth_cap_count": None,
        "warmup_depth_cap_fraction": None,
        "warmup_depth_cap_limitation": (
            "Unavailable because the source-matched workflow did not retain "
            "warmup extra fields; no value is inferred."
        ),
        "sampling_depth_cap_num_steps": cap_steps,
        "sampling_depth_cap_count": cap_count,
        "sampling_depth_cap_fraction": cap_fraction,
        "step_count": {
            "minimum": int(np.min(steps)),
            "q25": float(np.quantile(steps, 0.25)),
            "median": median_steps,
            "q75": float(np.quantile(steps, 0.75)),
            "maximum": int(np.max(steps)),
            "mean": float(np.mean(steps)),
        },
        "divergence_count": divergence_count,
        "final_step_size": diagnostics.get("final_step_size"),
        "accept_prob": {
            "minimum": float(np.min(accept_prob)),
            "median": float(np.median(accept_prob)),
            "mean": float(np.mean(accept_prob)),
            "maximum": float(np.max(accept_prob)),
        },
        "retained_sampling_leapfrog_step_proxy": retained_step_proxy,
        "work_proxy_limitation": (
            "The sum of retained num_steps is a documented leapfrog/gradient "
            "work proxy. It excludes warmup and is not an exact total gradient count."
        ),
        "ess": site_ess,
        "ess_per_retained_sampling_step_proxy": ess_per_step_proxy,
        "finite": {
            "major_sites": finite_sites,
            "accept_prob": bool(np.all(np.isfinite(accept_prob))),
            "potential_energy": bool(np.all(np.isfinite(potential))),
            "num_steps": bool(np.all(np.isfinite(steps))),
        },
        "passes_short_pilot_diagnostic_targets": passes_diagnostic_targets,
    }


def interpret(fixed, free):
    """Return the prespecified causal interpretation for the short pilot."""

    fixed_passes = fixed["passes_short_pilot_diagnostic_targets"]
    free_passes = free["passes_short_pilot_diagnostic_targets"]
    if fixed_passes and not free_passes:
        conclusion = (
            "The matched fixed arm passes while the reused free arm fails, "
            "strengthening the claim that free-scale coupling drives the "
            "linear-target geometry problem."
        )
    elif not fixed_passes and not free_passes:
        conclusion = (
            "Both arms fail; atmosphere, A, or metric geometry remains a "
            "candidate beyond the free scale."
        )
    elif fixed_passes and free_passes:
        conclusion = (
            "Both arms pass; the earlier fixed/free evidence was confounded "
            "by settings, adaptation, or initialization."
        )
    else:
        conclusion = (
            "The free arm passes while the fixed arm fails; this does not "
            "support the proposed free-scale mechanism."
        )
    return {
        "short_pilot_conclusion": conclusion,
        "scope": (
            "This is a conditional intervention on a rejected linearized "
            "target, not production inference or forward-model validation."
        ),
        "production_valid": False,
    }


def main():
    """Write the matched fixed/free short-pilot report."""

    args = parse_args()
    fixed_dir = Path(args.fixed_dir)
    output = (
        Path(args.output)
        if args.output is not None
        else fixed_dir / "fixed_free_control_summary.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output}")
    replay_path = fixed_dir / "initial_point_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if not replay["passed"]:
        raise ValueError("The fixed/free initial-point replay did not pass.")
    free = summarize_arm(
        args.free_samples,
        args.free_diagnostics,
        case="free",
        seed=args.seed,
    )
    fixed = summarize_arm(
        fixed_dir / "samples.npz",
        fixed_dir / "diagnostics.json",
        case="fixed",
        seed=args.seed,
    )
    report = {
        "mode": "m8_v2_fixed_free_clean_control_summary",
        "conditional_intervention": True,
        "initial_point_replay_passed": True,
        "arms": {"free": free, "fixed": fixed},
        "interpretation": interpret(fixed, free),
    }
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
