# 01_calibration

Bayesian (PyMC) calibration of the WCM-SWB model against Sentinel-1
backscatter and Sentinel-2 NDVI, for one field at a time.

All the actual model/engine code lives in `wcm_swb.model` — this directory
only holds the driver scripts and per-run configuration.

## Running

```bash
export WCM_DATA_ROOT=/path/to/WCM_SWB_Irri   # defaults to the repo root
python run_calibration.py --config configuration_01_calibration_TEMPLATE.json
```

Copy `configuration_01_calibration_TEMPLATE.json` to a new file per run/site
rather than editing the template in place. Use `--field <name>` to override
`options.opt_field` from the command line, or `run_calibration_batch.py` to
run every field matched by `paths.data_input.file_shapes`'s glob (one
subprocess per field).

## Config: `options.opt_mode`

- `"yearround"` — one calibration stage over the full time series, no
  parameters frozen.
- `"seasonal"` — three sequential stages splitting on an NDVI threshold
  (bare-soil parameters first, then vegetation/irrigation parameters with
  the bare-soil ones frozen). See `docs/model_conventions.md` for the exact
  parameter-freeze scheme.

## Config: `calibration.calibrate_Kc0`

`false` (default) fixes the crop-coefficient multiplier `Kc0` at
`calibration.PAR.Kc0_init` (normally `1.0`, i.e. NDVI is used directly as
the crop coefficient). Set to `true` to promote `Kc0` to a free, calibrated
parameter.

## Outputs

Written under `paths.data_output.root_plot / folder_plot / <opt_field> /
<cal_name>/`: `summary.txt` (posterior summary), posterior-peak plots,
`configuration_*.json` (the run config with the posterior injected),
`trace_*.nc` (full PyMC trace), `plot-trace`/`plot-quad`/`plot-pair`
figures, `output_*.nc` (per-timestep model states with a 5-95%
credible band per variable, from re-running the physics for ~1000
posterior draws), and `irrigation_samples_*.nc` (the daily irrigation of
each of those draws, which `02_analysis` uses for quantiles of multi-day
and multi-field irrigation totals).
