#!/usr/bin/env python
"""
Single-field WCM-SWB calibration driver.

Builds model inputs for one field from its config, then dispatches to either
the yearround (one-stage) or seasonal (three-stage) calibration engine in
``wcm_swb.model``, and writes each stage's outputs (posterior summary, trace,
plots, an output netCDF with a 5-95% credible band per variable, and the
posterior ensemble's daily irrigation).

Usage
-----
    python run_calibration.py --config configuration_01_calibration_TEMPLATE.json
    python run_calibration.py --config configuration_01_calibration_TEMPLATE.json --field ceregnano

Multi-field batches are handled by ``run_calibration_batch.py``, which
launches one subprocess of this script per field (isolating PyMC/numba
state between runs).

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""

import os
import sys
import json
import glob
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter

from wcm_swb.config import get_data_settings, substitute_keywords
from wcm_swb.soil import SaxtonRawls
from wcm_swb.evapotranspiration import hargre, hamon, apply_get_ET_Hourly
from wcm_swb.inputs import build_input_lonlat, build_input_soil, build_input_MERIDA, build_input_wcm
from wcm_swb.metrics import timeseries
from wcm_swb.model import (
    SiteConstants, coeff_for_freq, PAR_STR,
    run_yearround_calibration, run_seasonal_calibration,
)
from wcm_swb.plotting import plot_trace, plot_parameters_tables, plot_quad, plot_pair, plot_posteriors_peaks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Path to configuration_01_calibration.json')
    p.add_argument('--field', default=None, help='Override options.opt_field from the config')
    return p.parse_args()


def covers_period(index, date_range, tolerance=pd.Timedelta('1D')):
    """True if a meteo time index spans `date_range` (within `tolerance` at
    each end — MERIDA's first step of a year is 01:00)."""
    return (len(index) > 0 and index.min() <= date_range[0] + tolerance
            and index.max() >= date_range[-1] - tolerance)


def resolve_paths(data_settings):
    """Fill in {opt_field}/{opt_year} placeholders and join every data path
    onto `paths.root`. Mutates and returns `data_settings` for convenience."""
    opt = data_settings['options']
    paths = data_settings['paths']
    inputs_cfg = paths['data_input']
    outputs_cfg = paths['data_output']

    date_range = pd.date_range(pd.Timestamp(opt['start_date']), pd.Timestamp(opt['end_date']), freq='h')
    if opt.get('opt_year') is None:
        years = np.unique(date_range.year)
        opt['opt_year'] = '{' + ','.join(str(y) for y in years) + '}' if len(years) > 1 else int(years[0])

    root = os.path.expandvars(paths['root'])
    file_shapes = os.path.join(root, inputs_cfg['file_shapes'])
    if opt.get('opt_field') is None:
        opt['opt_field'] = [f.split('/')[-1].split('.')[0] for f in glob.glob(file_shapes)][0]

    file_sigma0 = substitute_keywords(inputs_cfg['file_sigma0'], **opt)
    file_ndvi = substitute_keywords(inputs_cfg['file_ndvi'], **opt)

    resolved = {
        'file_shapes': os.path.join(root, inputs_cfg['file_shapes'].replace('*', opt['opt_field'])),
        'file_sigma0': os.path.join(root, file_sigma0),
        'file_ndvi': os.path.join(root, file_ndvi),
        'file_reanalysis': os.path.join(root, inputs_cfg['file_reanalysis']),
        'file_MERIDA': [os.path.join(root, substitute_keywords(os.path.expandvars(p), **opt))
                        for p in inputs_cfg['file_MERIDA']],
        'file_SoilGrids': os.path.join(root, substitute_keywords(inputs_cfg['file_SoilGrids'], **opt)),
        'folder_plot': os.path.join(os.path.expandvars(outputs_cfg['root_plot']), outputs_cfg['folder_plot'], opt['opt_field']),
        'filename_logging': paths['filename_logging'],
        'date_range': date_range,
    }
    os.makedirs(resolved['folder_plot'], exist_ok=True)
    return resolved, date_range


def build_inputs(data_settings, resolved):
    """Assemble [P, ET0, Kc, veg, angle], the sigma0 obs/mask, veg series,
    and irrigation-window mask, using wcm_swb's explicit-parameter functions."""
    opt = data_settings['options']
    field_params = data_settings['field_params']
    input_cfg = data_settings['input']
    date_range = resolved['date_range']
    time_freq_out = input_cfg['time_freq_out']

    logging.info('Read lon/lat centroid and soil texture.')
    lat_c, lon_c = build_input_lonlat(resolved['file_shapes'])
    clay, sand, soc, WW_fc_init, WW_w_init = build_input_soil(
        resolved['file_SoilGrids'], resolved['file_shapes'], field_params.get('soil_layer_weighting', 'thickness'))
    WW_start_init = float(np.mean([WW_fc_init, WW_w_init]))
    lam_init, WW_sat_init, Ksat_init = SaxtonRawls(clay, sand, soc)
    if time_freq_out == 'd':
        Ksat_init *= 24

    logging.info('Read MERIDA meteo (cached extraction if it covers the period).')
    df_meteo = None
    if os.path.exists(resolved['file_reanalysis']):
        df_meteo = xr.open_dataset(resolved['file_reanalysis']).to_dataframe()
        if not covers_period(df_meteo.index, date_range):
            logging.info('Cache %s (%s..%s) does not cover %s..%s: rebuilding it from %s',
                         resolved['file_reanalysis'], df_meteo.index.min(), df_meteo.index.max(),
                         date_range[0], date_range[-1], resolved['file_MERIDA'])
            df_meteo = None
    if df_meteo is None:
        missing = [f for f in resolved['file_MERIDA'] if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f'MERIDA files {missing} not found (download them with '
                                    '00_preprocessing/download_merida.py or download_all.py)')
        df_meteo = build_input_MERIDA(resolved['file_MERIDA'], lat_c, lon_c, resolved['file_reanalysis'])
    if not covers_period(df_meteo.index, date_range):
        raise ValueError(f'MERIDA meteo covers {df_meteo.index.min()}..{df_meteo.index.max()}, not the '
                         f'calibration period {date_range[0]}..{date_range[-1]}: download MERIDA for it '
                         '(options.years in configuration_00_preprocessing_MERIDA.json)')
    # only short gaps inside the period are filled
    df_meteo = df_meteo.reindex(date_range).ffill().bfill()

    logging.info('Compute ET0.')
    if time_freq_out == 'd':
        df_meteo_daily = df_meteo.resample('D').mean()
        dates = df_meteo_daily.index
        temp_mean = df_meteo_daily.temperature.values
        temp_min = df_meteo_daily.temperature_min.interpolate('linear').values
        temp_max = df_meteo_daily.temperature_max.interpolate('linear').values

        if field_params['epot'] == 'hargre':
            eto_list = np.array([hargre(lat_c, dates[i], temp_min[i], temp_max[i], temp_mean[i])
                                  for i in range(len(dates))])
        elif field_params['epot'] == 'hamon':
            eto_list = np.array([hamon(temp_mean[i], dates[i].day, lat_c, par=1) for i in range(len(dates))])
        else:
            raise KeyError('field_params.epot must be "hargre" or "hamon".')
        eto_list[eto_list < 0] = 0
        eto_df = pd.DataFrame(timeseries(dates, eto_list)).rename(columns={0: 'Date', 1: 'ET0'}).set_index('Date')

        input_swb = pd.merge(df_meteo_daily, eto_df, left_index=True, right_index=True, how='outer')
        input_swb = input_swb.drop(columns=['rain'])
    elif time_freq_out == 'h':
        df_meteo = df_meteo.reindex(pd.date_range(
            df_meteo.index.date[0], freq='h', periods=len(np.unique(df_meteo.index.date)) * 24
        )).interpolate('linear').bfill().ffill()
        df_meteo.index.rename('datetime', inplace=True)
        df_meteo['ET0'] = df_meteo.groupby(df_meteo.index.date).apply(
            lambda x: apply_get_ET_Hourly(x, lat_centroid=lat_c, lon_centroid=lon_c,
                                           elev=0, shortcrop=1, date_format=opt['format_date'])
        ).reset_index().set_index('datetime').drop(columns='level_0')
        input_swb = df_meteo.drop(columns=['temperature_min', 'temperature_max'])
    else:
        raise KeyError('input.time_freq_out must be "h" or "d".')

    logging.info('Read sigma0 and NDVI.')
    input_wcm = build_input_wcm(resolved['file_sigma0'], resolved['file_ndvi'], input_cfg,
                                 date_range, freq_out=time_freq_out)
    scaling_svgfil = 24 if time_freq_out == 'h' else 1
    input_wcm['NDVI'] = savgol_filter(input_wcm.NDVI.values, 31 * scaling_svgfil, 1)

    input_df = input_swb.merge(input_wcm, left_index=True, right_index=True, how='outer')
    input_df['NDVI'] = input_df.NDVI.ffill()

    t = input_df.index
    Kc = input_df[opt['opt_veg']].values
    P = (input_df.rain_sum.values if time_freq_out == 'd' else input_df.rain.values).copy()
    P[P < 0] = 0
    ET0 = input_df.ET0.values.copy()
    ET0[ET0 < 0] = 0

    # The model uses one fixed reference incidence angle for the whole time
    # series (matching the geometry the sigma0 normalization was performed
    # at) rather than the real per-timestep angle. Pick whichever orbit's
    # mean angle is closest to 40 degrees, so no site-specific orbit number
    # is needed.
    orbit_angles = input_wcm.groupby('orb').angle.mean()
    ref_orbit = (orbit_angles - 40.0).abs().idxmin()
    ref_angle = orbit_angles.loc[ref_orbit]
    angle = np.full(len(P), ref_angle)

    mask_temp = input_df.temperature < 1.5
    mask_rain_s0 = input_df.rain_sum > 40
    if 'VV' in opt['opt_obs']:
        mask_s0 = (input_df[opt['opt_obs']] < -20) | (input_df[opt['opt_obs']] > -5)
        input_df.loc[mask_s0, opt['opt_obs']] = np.nan
    input_df.loc[mask_temp, opt['opt_obs']] = np.nan
    input_df.loc[mask_rain_s0, opt['opt_obs']] = np.nan

    obs = input_df[opt['opt_obs']].values
    mask = np.isfinite(obs)
    veg = input_df.NDVI.values

    inputs = np.array([arr.astype(np.float64) for arr in [P, ET0, Kc, veg, angle]], dtype=np.float64)

    doys = np.array([pd.Timestamp(d).dayofyear for d in t])
    hours = np.array([pd.Timestamp(d).hour for d in t])
    mask_irri_d = (doys > 90) & (doys < 270)
    mask_rain = input_df.rain_sum.values < 10
    if time_freq_out == 'h':
        mask_irri_h = (hours >= 5) & (hours <= 20)
        mask_irri = mask_irri_d & mask_irri_h & mask_rain
    else:
        mask_irri = mask_irri_d & mask_rain

    coeff_r, coeff_i = coeff_for_freq(field_params['freq'])
    site = SiteConstants(
        coeff_r=coeff_r, coeff_i=coeff_i, freq=field_params['freq'], sand=sand, clay=clay,
        depth1_add=field_params['depth1_add'], depth2=field_params['depth2'],
        rho_st=field_params['rho_st'], mask_irri=mask_irri,
    )

    par_init = {
        'A': data_settings['calibration']['PAR']['A_init'],
        'B': data_settings['calibration']['PAR']['B_init'],
        'C': data_settings['calibration']['PAR']['C_init'],
        'D': data_settings['calibration']['PAR']['D_init'],
        'Ksat': data_settings['calibration']['PAR']['Ksat_init'] or Ksat_init,
        'lam': data_settings['calibration']['PAR']['lam_init'] or lam_init,
        'WW_fc': data_settings['calibration']['PAR']['WW_fc_init'] or WW_fc_init,
        'WW_w': data_settings['calibration']['PAR']['WW_w_init'] or WW_w_init,
        'WW_start': data_settings['calibration']['PAR']['WW_start_init'] or WW_start_init,
        'irri_thr': data_settings['calibration']['PAR']['irri_thr_init'],
        'irri_cf': data_settings['calibration']['PAR']['irri_cf_init'],
    }

    freq_scaling = 1. if time_freq_out == 'h' else 24.
    ksat_lo, ksat_hi = 0.3 * freq_scaling, 20.0 * freq_scaling
    if ksat_lo > par_init['Ksat']:
        ksat_lo = par_init['Ksat'] - 5 * np.sqrt(par_init['Ksat'])
    if ksat_hi < par_init['Ksat']:
        ksat_hi = par_init['Ksat'] + 5 * np.sqrt(par_init['Ksat'])
    parspace = {
        'A': [0.0, 5.0], 'B': [0.0, 10.0], 'C': [-30.0, 0.0], 'D': [10.0, 60.0],
        'Ksat': [ksat_lo, ksat_hi], 'lam': [0.09, 0.5], 'WW_fc': [0.3, 0.45],
        'WW_w': [0.05, 0.25], 'WW_start': [0.05, 0.45], 'irri_thr': [0, 1], 'irri_cf': [0, 1],
    }

    return {
        'input_df': input_df, 't': t, 'inputs': inputs, 'obs': obs, 'mask': mask, 'veg': veg,
        'P': P, 'ET0': ET0, 'ww_sat': WW_sat_init, 'site': site, 'par_init': par_init,
        'parspace': parspace, 'scaling_svgfil': scaling_svgfil,
    }


def write_stage_outputs(cal_name, trace, posterior, output_mean, output_q05, output_q95, irri_samples,
                         built, data_settings, resolved):
    """Write summary.txt, posterior-peak plots, config.json (with posterior
    injected), trace.nc, trace/params-table/quad/pair plots, output.nc, and
    irrigation_samples.nc -- one call per calibration stage (1 for
    yearround, 3 for seasonal)."""
    import arviz as az

    opt = data_settings['options']
    out_cfg = data_settings['paths']['data_output']
    folder_plot_cal = os.path.join(resolved['folder_plot'], cal_name) + '/'
    os.makedirs(folder_plot_cal, exist_ok=True)

    trace_posterior = az.extract(trace, group='posterior', combined=True)
    summary = az.summary(trace, var_names=[p for p in PAR_STR if p in trace_posterior.data_vars] + ['sigma'])
    with open(os.path.join(folder_plot_cal, 'summary.txt'), 'w') as f:
        f.write(summary.to_string())

    import matplotlib.pyplot as plt
    for param in trace_posterior:
        plot_posteriors_peaks(trace_posterior[param].values, param, opt['opt_field'], posterior,
                               output_dir=folder_plot_cal, opt_save_plots=out_cfg['opt_save_plots'])
        plt.close()

    filename_dict = lambda output_key: {  # noqa: E731
        'filename_output': output_key, 'root_plot': folder_plot_cal, 'opt_field': opt['opt_field'],
        'opt_obs': opt['opt_obs'], 'add_description': '_' + cal_name,
    }

    if out_cfg['opt_save']:
        data_settings_out = dict(data_settings)
        data_settings_out.setdefault('PAR_dict_posterior', {})[cal_name] = posterior
        path = substitute_keywords(out_cfg['filename_template'], **filename_dict(out_cfg['filename_configuration']))
        with open(path + '.json', 'w') as f:
            json.dump(data_settings_out, f, indent=4, default=str)

        path = substitute_keywords(out_cfg['filename_template'], **filename_dict(out_cfg['filename_trace_nc']))
        trace.to_netcdf(path + '.nc', engine='netcdf4')

    plot_trace(trace, {cal_name: posterior}, cal_name, 'DEMetropolisZ',
               data_settings['calibration']['run_params']['n_draws'],
               data_settings['calibration']['run_params']['n_chains'],
               out_cfg['filename_trace'], folder_plot_cal, opt['opt_field'], opt['opt_obs'],
               add_description=cal_name, filename_template=out_cfg['filename_template'],
               opt_save_plots=out_cfg['opt_save_plots'], extension_plot=out_cfg['extension_plot'],
               layout='constrained', compact=True, combined=True)

    plot_parameters_tables({cal_name: posterior}, out_cfg['filename_table_params'],
                            folder_plot=folder_plot_cal, opt_field=opt['opt_field'], opt_obs=opt['opt_obs'],
                            add_description=cal_name, filename_template=out_cfg['filename_template'],
                            opt_save=out_cfg['opt_save_plots'], extension_plot=out_cfg['extension_plot'],
                            figsize=(10, 10), hspace=0, title_fontsize=14)

    plot_quad(trace, built['input_df'], opt['opt_obs'], built['t'], output_mean['s0'], built['veg'],
              output_mean['WW1'], output_mean['WW2'], output_mean['I'], output_mean['PERC1'],
              output_mean['ET1'], output_mean['PERC2'], output_mean['ET2'], None, output_mean['IRRI'],
              built['ET0'], built['P'], opt['opt_field'], add_description=cal_name,
              opt_save_plots=out_cfg['opt_save_plots'], filename_template=out_cfg['filename_template'],
              folder_plot=folder_plot_cal, filename_quad=out_cfg['filename_quad'],
              extension_plot=out_cfg['extension_plot'])

    plot_pair(trace, opt['opt_field'], opt['opt_obs'], out_cfg['filename_pair'],
              folder_plot=folder_plot_cal, filename_template=out_cfg['filename_template'],
              add_description=cal_name, opt_save_plots=out_cfg['opt_save_plots'],
              extension_plot=out_cfg['extension_plot'], figsize=(10, 8), textsize=14)

    unc_keys = {
        'Soil_Moisture_1': 'WW1', 'Soil_Moisture_2': 'WW2', 'Infiltration': 'I',
        'Percolation_1': 'PERC1', 'Percolation_2': 'PERC2', 'ETc_1': 'ET1', 'ETc_2': 'ET2',
        'Irrigation': 'IRRI', f"Sigma0_{opt['opt_obs']}": 's0',
    }
    output_dict = {
        'Datetime': built['t'],
        'Soil_Moisture_1': output_mean['WW1'], 'Soil_Moisture_2': output_mean['WW2'],
        'Infiltration': output_mean['I'], 'Percolation_1': output_mean['PERC1'],
        'Percolation_2': output_mean['PERC2'], 'ETc_1': output_mean['ET1'], 'ETc_2': output_mean['ET2'],
        'Irrigation': output_mean['IRRI'], f"Sigma0_{opt['opt_obs']}": output_mean['s0'],
        f"Sigma0_{opt['opt_obs']}_obs": built['obs'], 'Sigma0_mask': built['mask'],
        'Rain': built['P'], 'ET0': built['ET0'], f"{opt['opt_veg']}": built['veg'],
        **{f'{nc_var}_q05': output_q05[ok] for nc_var, ok in unc_keys.items()},
        **{f'{nc_var}_q95': output_q95[ok] for nc_var, ok in unc_keys.items()},
    }
    output_ds = xr.Dataset.from_dataframe(pd.DataFrame(output_dict).set_index('Datetime'))
    units = {
        'Soil_Moisture_1': 'm3/m3', 'Soil_Moisture_2': 'm3/m3', 'Infiltration': 'mm/h',
        'Percolation_1': 'mm/h', 'Percolation_2': 'mm/h', 'ETc_1': 'mm/h', 'ETc_2': 'mm/h',
        'Irrigation': 'mm/h', f"Sigma0_{opt['opt_obs']}": 'dB', f"Sigma0_{opt['opt_obs']}_obs": 'dB',
        'Rain': 'mm/h', 'ET0': 'mm/h', f"{opt['opt_veg']}": 'dimensionless',
    }
    for var, unit in units.items():
        output_ds[var].attrs['units'] = unit

    if out_cfg['opt_save']:
        path = substitute_keywords(out_cfg['filename_template'], **filename_dict(out_cfg['filename_output']))
        output_ds.to_netcdf(path + '.nc')

        # Daily irrigation totals of every posterior-ensemble member, so
        # 02_analysis can take quantiles of multi-day (and multi-field)
        # irrigation totals over the ensemble itself -- see
        # wcm_swb.model.propagate_posterior_uncertainty.
        irri_daily = pd.DataFrame(irri_samples.T, index=pd.DatetimeIndex(built['t'])).resample('D').sum()
        samples_ds = xr.Dataset(
            {'Irrigation': (('Datetime', 'draw'), irri_daily.values.astype(np.float32))},
            coords={'Datetime': irri_daily.index, 'draw': np.arange(irri_samples.shape[0])},
        )
        samples_ds['Irrigation'].attrs['units'] = 'mm/day'
        samples_ds['Irrigation'].attrs['description'] = (
            'Daily simulated irrigation for each posterior draw used to build the credible band in output_*.nc')
        path = substitute_keywords(out_cfg['filename_template'], **filename_dict(out_cfg.get('filename_irrigation_samples', 'irrigation_samples')))
        samples_ds.to_netcdf(path + '.nc')


def main():
    args = parse_args()
    data_settings = get_data_settings(args.config)
    if args.field:
        data_settings['options']['opt_field'] = args.field

    resolved, _ = resolve_paths(data_settings)

    if os.path.exists(resolved['filename_logging']):
        os.remove(resolved['filename_logging'])
    logging.basicConfig(filename=resolved['filename_logging'], level=logging.INFO,
                         format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Field: %s, mode: %s', data_settings['options']['opt_field'], data_settings['options']['opt_mode'])

    built = build_inputs(data_settings, resolved)
    run_params = data_settings['calibration']['run_params']
    calibrate_kc0 = data_settings['calibration'].get('calibrate_Kc0', False)
    kc0_init = data_settings['calibration']['PAR'].get('Kc0_init', 1.0)

    opt_mode = data_settings['options']['opt_mode']
    if opt_mode == 'yearround':
        stages = run_yearround_calibration(
            built['inputs'], built['obs'], built['mask'], built['ww_sat'], built['site'],
            built['par_init'], built['parspace'], run_params,
            calibrate_kc0=calibrate_kc0, kc0_init=kc0_init)
    elif opt_mode == 'seasonal':
        veg_thr = min(np.quantile(built['input_df'].NDVI.dropna().values, 0.5), 0.5)
        stages = run_seasonal_calibration(
            built['inputs'], built['obs'], built['veg'], veg_thr, built['t'], built['ww_sat'],
            built['site'], built['par_init'], built['parspace'], run_params, built['scaling_svgfil'],
            calibrate_kc0=calibrate_kc0, kc0_init=kc0_init)
    else:
        raise ValueError('options.opt_mode must be "yearround" or "seasonal".')

    for cal_name, (trace, posterior, output_mean, output_q05, output_q95, irri_samples) in stages.items():
        logging.info('Writing outputs for %s...', cal_name)
        write_stage_outputs(cal_name, trace, posterior, output_mean, output_q05, output_q95, irri_samples,
                             built, data_settings, resolved)

    logging.info('DONE.')


if __name__ == '__main__':
    main()
