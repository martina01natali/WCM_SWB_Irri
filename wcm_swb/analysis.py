"""
Result-aggregation helpers for the ``02_analysis`` notebooks.

Everything here reads the files written by ``01_calibration/run_calibration.py``,
laid out as::

    <root_plot>/<folder_plot>/<run>/<field>/<stage>/output_<field>_<obs>_<stage>.nc
                                                   /trace_<field>_<obs>_<stage>.nc
                                                   /configuration_<field>_<obs>_<stage>.json
                                                   /irrigation_samples_<field>_<obs>_<stage>.nc

where ``<run>`` is one calibration run folder (e.g. ``yearround``,
``seasonal``) and ``<stage>`` is ``cal_yearround`` or one of the seasonal
stages (``cal_bare_1``, ``cal_bare_2``, ``cal_veg``).

Irrigation uncertainty comes from the posterior ensemble that
``run_calibration.py`` saves as ``irrigation_samples_*.nc`` (daily irrigation
totals for a random subset of draws from the joint posterior). Window totals
and multi-field aggregates are computed per draw first and then summarised
by their quantiles (5-95% by default), so sparse irrigation events are not
lost as they would be when summing per-timestep quantiles.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

# Stage whose posterior holds the complete final parameter set for each mode:
# `cal_veg` passes the frozen soil parameters from `cal_bare_2` through as
# fixed values, so it (not `cal_bare_2`) is the seasonal mode's final answer.
FINAL_STAGES = ('cal_yearround', 'cal_veg')


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

def load_analysis_config(file_name):
    """Load ``configuration_02_analysis*.json`` and expand ``${VAR}``s in its
    paths (``WCM_DATA_ROOT`` defaults to the repo root, as in
    ``01_calibration``). Returns ``(options, paths)``.
    """
    from wcm_swb.config import get_data_settings

    os.environ.setdefault('WCM_DATA_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    settings = get_data_settings(file_name)
    paths = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in settings['paths'].items()}
    return settings['options'], paths


# ----------------------------------------------------------------------------
# Locating calibration outputs
# ----------------------------------------------------------------------------

def find_outputs(root_output, folder_pattern='*', opt_obs='VV', stages=None):
    """Index every calibration output below ``root_output``.

    ``folder_pattern`` is a glob matched against the run folders directly
    under ``root_output`` (e.g. ``'*'``, ``'yearround*'``). ``stages``
    restricts to the given stage names (default: all stages found).

    Returns a DataFrame with one row per (run, field, stage) and columns
    ``run, field, stage, output, trace, configuration, irrigation_samples``
    (file paths; ``None`` where a file is missing).
    """
    rows = []
    pattern = os.path.join(root_output, folder_pattern, '*', '*', f'output_*_{opt_obs}_*.nc')
    for path_output in sorted(glob.glob(pattern)):
        folder_stage = os.path.dirname(path_output)
        stage = os.path.basename(folder_stage)
        field = os.path.basename(os.path.dirname(folder_stage))
        run = os.path.basename(os.path.dirname(os.path.dirname(folder_stage)))
        if stages is not None and stage not in stages:
            continue
        suffix = f'_{field}_{opt_obs}_{stage}'
        path_trace = os.path.join(folder_stage, f'trace{suffix}.nc')
        path_config = os.path.join(folder_stage, f'configuration{suffix}.json')
        path_samples = os.path.join(folder_stage, f'irrigation_samples{suffix}.nc')
        rows.append({
            'run': run, 'field': field, 'stage': stage, 'output': path_output,
            'trace': path_trace if os.path.exists(path_trace) else None,
            'configuration': path_config if os.path.exists(path_config) else None,
            'irrigation_samples': path_samples if os.path.exists(path_samples) else None,
        })
    return pd.DataFrame(rows, columns=['run', 'field', 'stage', 'output', 'trace', 'configuration',
                                       'irrigation_samples'])


def final_stage_outputs(outputs):
    """Keep only each (run, field)'s final stage (see ``FINAL_STAGES``)."""
    return outputs[outputs.stage.isin(FINAL_STAGES)].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Field geometry
# ----------------------------------------------------------------------------

def field_areas(file_shapes):
    """Area [m2] of every field polygon matched by the ``file_shapes`` glob,
    indexed by field name (the file stem, matching ``options.opt_field`` in
    ``01_calibration``). Areas are computed in the polygons' local UTM zone.

    Any non-geometry attributes of the shapefiles (e.g. a ``Crop`` column)
    are returned alongside ``area``, for grouping in the notebooks.
    """
    import geopandas as gpd

    frames = []
    for f in sorted(glob.glob(file_shapes)):
        gdf = gpd.read_file(f)
        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
        attrs = gdf.drop(columns='geometry').iloc[0].to_dict() if len(gdf) else {}
        attrs.update(field=os.path.basename(f).split('.')[0], area=float(gdf_utm.area.sum()))
        frames.append(attrs)
    if not frames:
        raise FileNotFoundError(f'No shapefiles matched {file_shapes}')
    return pd.DataFrame(frames).set_index('field')


# ----------------------------------------------------------------------------
# Irrigation aggregation
# ----------------------------------------------------------------------------

def read_irrigation_samples(path_samples):
    """Daily irrigation [mm/day] of every posterior-ensemble member, from one
    ``irrigation_samples_*.nc``, as a (days x draws) DataFrame."""
    with xr.open_dataset(path_samples) as ds:
        da = ds['Irrigation'].transpose('Datetime', 'draw')
        return pd.DataFrame(da.values.astype(np.float64), index=pd.DatetimeIndex(da['Datetime'].values))


def area_weighted_ensemble(paths_samples, areas):
    """Area-weighted average irrigation [mm/day] across fields, per draw.

    ``paths_samples`` maps field name -> ``irrigation_samples_*.nc`` path;
    ``areas`` maps field name -> area [m2]. Draw ``j`` of every field is
    combined with draw ``j`` of the others, i.e. fields are treated as
    independent (each was calibrated separately). If fields have different
    ensemble sizes, the ensemble is cut to the smallest.
    """
    ensembles = {field: read_irrigation_samples(path) for field, path in paths_samples.items()}
    if not ensembles:
        raise ValueError('No irrigation_samples files given.')
    n_draws = min(e.shape[1] for e in ensembles.values())
    total, area_tot = None, 0.0
    for field, ens in ensembles.items():
        area = float(areas[field])
        weighted = ens.iloc[:, :n_draws] * area
        total = weighted if total is None else total.add(weighted, fill_value=0.0)
        area_tot += area
    return total / area_tot


def irrigation_quantiles(ensemble, freq, start=None, end=None, quantiles=(0.05, 0.95)):
    """Summarize a (days x draws) irrigation ensemble over ``freq`` windows
    (e.g. '1D', '7D', '15D', 'YS'), optionally clipped to [start, end].

    Every draw is summed over each window first, and the mean / median /
    quantiles are taken across draws of those window totals. Returns a
    DataFrame with columns ``mean, median, q05, q95`` (quantile columns
    named after ``quantiles``).
    """
    if start is not None or end is not None:
        ensemble = ensemble.loc[start:end]
    totals = ensemble.resample(freq).sum()
    out = pd.DataFrame({'mean': totals.mean(axis=1), 'median': totals.median(axis=1)})
    for q in quantiles:
        out[f'q{round(q * 100):02d}'] = totals.quantile(q, axis=1)
    return out


def irrigation_season(year, start_doy, end_doy):
    """(start, end) Timestamps of the irrigation window for one year."""
    year_start = pd.Timestamp(year=int(year), month=1, day=1)
    return (year_start + pd.Timedelta(days=start_doy - 1),
            year_start + pd.Timedelta(days=end_doy) - pd.Timedelta(hours=1))


# ----------------------------------------------------------------------------
# Backscatter goodness of fit
# ----------------------------------------------------------------------------

def kge(sim, obs):
    """Kling-Gupta efficiency (2009 formulation)."""
    sim, obs = np.asarray(sim, dtype=float), np.asarray(obs, dtype=float)
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def fit_metrics(sim, obs):
    """Goodness-of-fit of simulated vs. observed sigma0 on finite pairs.

    Returns a dict with ``n, r, bias, rmse, ubrmsd, kge`` (bias = sim - obs).
    """
    sim, obs = np.asarray(sim, dtype=float), np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    anom = (sim - sim.mean()) - (obs - obs.mean())
    return {
        'n': int(mask.sum()),
        'r': float(np.corrcoef(sim, obs)[0, 1]),
        'bias': float(np.mean(sim - obs)),
        'rmse': float(np.sqrt(np.mean((sim - obs) ** 2))),
        'ubrmsd': float(np.sqrt(np.mean(anom ** 2))),
        'kge': float(kge(sim, obs)),
    }


def sigma0_metrics(outputs, opt_obs='VV'):
    """``fit_metrics`` for every row of a ``find_outputs`` frame.

    Only timesteps flagged in ``Sigma0_mask`` (the observations the stage was
    actually calibrated on the full series; see ``run_calibration.py``) are
    scored.
    """
    rows = []
    for row in outputs.itertuples():
        with xr.open_dataset(row.output) as ds:
            mask = ds['Sigma0_mask'].values.astype(bool)
            sim = ds[f'Sigma0_{opt_obs}'].values[mask]
            obs = ds[f'Sigma0_{opt_obs}_obs'].values[mask]
        rows.append({'run': row.run, 'field': row.field, 'stage': row.stage, **fit_metrics(sim, obs)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Posterior parameters
# ----------------------------------------------------------------------------

def read_posteriors(outputs):
    """Posterior point estimates for every row of a ``find_outputs`` frame.

    Reads ``PAR_dict_posterior[<stage>]`` from each run's
    ``configuration_*.json`` and returns a long DataFrame with columns
    ``run, field, stage, param, value, std, init, range_lo, range_hi, ref,
    calibrated`` (``calibrated`` is False for parameters frozen in that
    stage, which carry their fixed value with ``std == 0``).
    """
    rows = []
    for row in outputs.itertuples():
        if row.configuration is None:
            continue
        with open(row.configuration) as f:
            posterior = json.load(f)['PAR_dict_posterior'][row.stage]
        for param, entry in posterior.items():
            lo, hi = entry['range'] if entry.get('range') is not None else (np.nan, np.nan)
            rows.append({
                'run': row.run, 'field': row.field, 'stage': row.stage, 'param': param,
                'value': float(entry['value']), 'std': float(entry['std']),
                'init': float(entry['init']) if entry['init'] is not None else np.nan,
                'range_lo': float(lo), 'range_hi': float(hi), 'ref': entry['ref'],
                'calibrated': float(entry['std']) > 0,
            })
    return pd.DataFrame(rows)


def natural_key(text):
    """Sort key that orders ``Id_2`` before ``Id_10``."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(text))]
