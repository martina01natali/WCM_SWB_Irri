# Model conventions

Reference for anyone reading or extending `wcm_swb/model.py`.

## Units

- `doi()` (depth of radar influence) takes its angle argument `theta` in
  **radians** and returns depth in **mm**.
- Soil water content (`WW1`, `WW2`, `WW_fc`, `WW_w`, `WW_sat`, `WW_start`) is in
  m³/m³.
- Backscatter (`s0`, `A`/`C`/`D`-related quantities) is in dB, except the
  Water Cloud Model's intermediate linear-domain terms.

## WW_sat is fixed, not calibrated

`WW_sat` (soil saturation water content) is always derived from
`SaxtonRawls(clay, sand, soc)` and passed into `swb_wcm_core` as a fixed
value — it is never a free/calibrated parameter.

## `Kc0`

`Kc0` is a crop-coefficient multiplier on top of the NDVI-as-Kc proxy
(`ETc = ET0 * Kc0 * NDVI`). It defaults to `1.0`, i.e. NDVI is used directly
as the crop coefficient. Set `calibrate_Kc0: true` in a run's config to
promote it to a free, calibrated parameter (see `01_calibration/README.md`).

## Cost function and sampler

The likelihood is Gaussian: `sigma ~ HalfNormal(1.0)`, `s0_calib ~
Normal(mu=s0_sim, sigma=sigma)`. The sampler is `pm.DEMetropolisZ()`.

## Yearround vs. seasonal

Both calibration modes call the same `run_calibration_stage` function
(one PyMC model, one `pm.sample` call) — they differ only in *how many
times* it's called and *which parameters are frozen* on each call:

- **yearround**: one call, nothing frozen, fit against the full time series.
- **seasonal**: three calls — `cal_bare_1` (freeze A, B, irri_cf, irri_thr at
  0; fit soil/backscatter params on NDVI-below-threshold data), `cal_bare_2`
  (same freeze scheme, but first narrow the C/D priors via a change-detection
  fit — `D_slope` — using stage 1's posterior soil moisture), `cal_veg`
  (freeze the soil/backscatter params at stage 2's posterior; fit
  A/B/irrigation params on NDVI-above-threshold data).

See `wcm_swb/model.py`'s `run_yearround_calibration`/`run_seasonal_calibration`
docstrings for the exact parameter lists.
