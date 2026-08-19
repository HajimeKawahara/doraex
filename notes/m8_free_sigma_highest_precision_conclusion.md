# M8 free-sigma highest-precision conclusion

Date: 2026-08-20

## Scope

This note closes the sampler investigation for the frozen M8 linearized,
analytically marginalized target. It does not validate the Taylor-linearized
radiative-transfer approximation. Nonlinear-forward development and validation
belong to a separate workstream.

## Conclusion

The free `sigma_log_p` NUTS pathology was caused primarily by numerical error in
the large float32 low-rank likelihood calculation, not by an unavoidable scale
coordinate pathology. Setting the dedicated sampling process to
`JAX_DEFAULT_MATMUL_PRECISION=highest` restores a consistent local score and
allows the same free-sigma target to sample efficiently.

The precision intervention remains float32 (`x64=false`) and is process-global.
The v8 isolated low-rank experiment localizes the relevant defect, while v9 and
v10 test the intervention on the full frozen target.

## Evidence

### v8 low-rank precision diagnostic

- Maximum full-target absolute reverse/JVP sigma-score difference:
  `3.4482` (default) to `0.04622` (highest).
- Maximum isolated full-factor difference:
  `3.5144` (default) to `0.05873` (highest).
- Reduced float64 reference difference: approximately `1.4e-11`.
- The default reverse-mode sigma curvature had the wrong sign; highest precision
  recovered the float64-reference curvature.

Summary SHA256:

```text
7db88c6fff97d4fa1421898598d8e1d5bff99ca204cd03d70cacc303023d6efc
```

### v9 matched short control

- One chain, seed 0, 200 warmup and 20 retained draws.
- Initial sigma reverse/JVP score difference: `0.00529385`.
- Divergences: `0`.
- 2,047-step cap: `0 / 20`.
- Median and maximum retained steps: `31 / 31`.

Summary SHA256:

```text
a1a1000166981637e4a8ac988d713c184b44664431d384dd7e94f10a179c8f81
```

### v10 matched long control

- One chain, seed 0, 2,000 warmup and 1,500 retained draws.
- Divergences: `0`.
- 2,047-step cap: `0 / 1500`.
- Retained steps: minimum `7`, median `15`, maximum `15`, total `19,340`.
- Final step size: `0.3346841335`.
- Mean acceptance probability: `0.9579087` for target `0.95`.
- Runtime: `8,730.30 s` (`2.43 h`).
- `sigma_log_p` median: `0.38755`; central 90% interval:
  `[0.29811, 0.50938]`.
- Rank-normalized within-chain ESS for `sigma_log_p`: bulk `1322`, tail
  `1163`.

The matched historical M8 v1 run used the same intended target and schedule but
the inherited/default matmul policy. It had a `99.8%` cap fraction, median
`2,047` steps, final step size `0.00151033`, and runtime `113.19 h`.

Artifact SHA256 values:

```text
b9b703f494af44f495182cc65784e7aceb5b99e62d996ccdde96cdaff21ae7b7  summary
c79d8f2c24e25a35bd5827628af8b2e0e5b14202ca7e8229f4545b338e46209f  samples
140b52e99977c82f931da150f7d9c265cd909ba5f9f28be801290aa9d07a26d3  diagnostics
```

## Interpretation limits

- One chain and one seed do not establish between-chain convergence or exclude
  undiscovered modes.
- The historical v1 artifact did not serialize its ambient matmul policy; the
  default-policy interpretation is supported by its launcher and by the v8
  current-environment reproduction.
- The saved diagnostics do not contain total Hamiltonian energy, so E-BFMI is
  unavailable.
- The inferred `sigma_log_p` lies mostly in a region where prior exact-forward
  screens found substantial Taylor-linearization error. These samples therefore
  characterize the frozen linearized target, not a validated nonlinear physical
  posterior.
- Result directories, logs, sentinels, and provenance manifests remain external
  artifacts and are intentionally not committed to Git.
