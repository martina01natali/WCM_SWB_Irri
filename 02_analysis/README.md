# 02_analysis

Interactive notebooks that aggregate and visualize calibration results from
`01_calibration`: model states and backscatter fit for one field, irrigation
estimates with posterior uncertainty across fields and calibration runs, and
calibrated parameters across fields.

## Prerequisite

Run `01_calibration` first, for every field and calibration mode you want to
compare. Each run writes, per field and stage, under
`<root_plot>/<folder_plot>/<field>/<stage>/`:

- `output_<field>_<obs>_<stage>.nc` — model states, with a 5-95% credible band per variable
- `irrigation_samples_<field>_<obs>_<stage>.nc` — daily irrigation of every posterior-ensemble draw
- `trace_<field>_<obs>_<stage>.nc`, `configuration_<field>_<obs>_<stage>.json`, `summary.txt`

Point `paths.root_output` in the config at the folder that holds the run
folders (by default `01_calibration/output/`, so e.g. `output/yearround/` and
`output/seasonal/` from the template's `folder_plot`). Each run folder is one
entry in the comparisons (labelled via `options.run_labels`).

## Notebooks

All three read `configuration_02_analysis_TEMPLATE.json` (copy it and change
`file_settings` in the first code cell to use your own). They're independent:
run them in any order.

| Notebook | What it shows |
|---|---|
| `02_single_field.ipynb` | One run/field/stage (`options.single_*`): sigma0 obs vs. sim + NDVI, soil moisture and daily irrigation, P/ET0 (all with credible bands); sigma0 fit metrics; seasonal/monthly irrigation totals; the posterior summary. |
| `02_irrigation.ipynb` | Area-weighted irrigation over all fields at 1/7/15-day aggregation, one line + 5-95% band per run; season totals; per-field totals; irrigated-field counts / area share / volume at several thresholds; sigma0 fit metrics across fields per run. |
| `02_parameters.ipynb` | Posterior point estimates of every parameter (and the likelihood's `sigma`) across fields, one box per run; optionally grouped by a shapefile attribute (`options.group_by`, `options.group_map`). |

Each notebook uses the final stage of every run: `cal_yearround` for
year-round runs, `cal_veg` for seasonal runs (it carries the soil parameters
frozen from `cal_bare_2`). `02_single_field` can show any stage.

Set `paths.opt_save_plots: true` to write figures (and CSV tables) under
`paths.folder_analysis` (default `02_analysis/output/`).

## Uncertainty

Irrigation happens on a few days only, so per-timestep quantiles are ~0
almost everywhere and summing them gives a meaningless band for weekly or
seasonal totals. The notebooks therefore sum every posterior draw over each
window (and, for multi-field totals, over fields, draw by draw) and take
quantiles of those totals. This is what `irrigation_samples_*.nc` is for.
The per-timestep `*_q05`/`*_q95` bands in `output_*.nc` are used only for
the single-field time-series plot.
