# doraex
Doppler Retrieval of Atmosphere using ExoJAX

## Milestone 2-4b

M2-4b is the joint chip0-3 Doppler retrieval with a shared contrast map and a
shared atmosphere (`T0`, `log_p_cloud`, `f_cloud`). Chip-local calibration and
noise terms (`log_w`, `surface_scale`, `sigma_d`) remain chip-specific.

```bash
python examples/luhman16b_yama/run_milestone2_joint_chips.py \
  --m2-4b \
  --chip-indices 0,1,2,3 \
  --nside 8 \
  --num-warmup 2000 \
  --num-samples 1500 \
  --target-accept-prob 0.98 \
  --max-tree-depth 11 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-ell-b 0.3 \
  --fix-geometry-to-milestone1
```

```bash
python examples/luhman16b_yama/make_milestone2_joint_chip_products.py \
  --m2-4b \
  --chip-indices 0,1,2,3 \
  --nside 8 \
  --max-map-samples 1000
```
