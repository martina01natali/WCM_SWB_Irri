#!/usr/bin/env python
"""
MERIDA reanalysis download (RSE SFTP server).

Downloads the monthly, national-scale MERIDA HRES netCDF files (hourly,
0.04 deg) for the variables, years and months listed in the config — one
``curl`` SFTP transfer per (variable, month) — into the study site's folder
(``paths.root``, default ``study_sites/<site>/merida/``), and optionally
concatenates each variable's months into one file, which is what
``01_calibration``'s ``paths.data_input.file_MERIDA`` points at.

The period is set in this script's own config
(``configuration_00_preprocessing_MERIDA_TEMPLATE.json``) and is independent
of the Sentinel-1/2 download period.

Credentials are read from the environment variables named in
``server.user_env`` / ``server.password_env`` (default ``MERIDA_USER`` /
``MERIDA_PASSWORD``) and handed to curl on stdin, so they never end up in a
file or in the process list.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md

Usage
-----
    export MERIDA_USER=... MERIDA_PASSWORD=...
    python download_merida.py --config configuration_00_preprocessing_MERIDA_TEMPLATE.json
    python download_merida.py --config ... --merge-only    # skip the download, only (re)merge
"""
import os
import sys
import argparse
import logging
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

from wcm_swb.config import get_data_settings, substitute_keywords

TIME_UNITS = 'hours since 1970-01-01 00:00:00'
TIME_EPOCH = np.datetime64('1970-01-01T00:00:00', 'ns')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='Path to configuration_00_preprocessing_MERIDA.json')
    p.add_argument('--merge-only', action='store_true', help='Do not download, only merge the monthly files')
    return p.parse_args()


def load_config(file_config):
    """Load the config; output folders are resolved relative to ``paths.root`` (``${ENV_VARS}`` expanded)."""
    data_settings = get_data_settings(file_config)
    paths = data_settings['paths']
    paths['root'] = os.path.expandvars(paths['root'])
    outputs_cfg = paths['data_output']
    for key in ('folder_monthly', 'folder_merged'):
        outputs_cfg[key] = os.path.join(paths['root'], os.path.expandvars(outputs_cfg[key]))
    return data_settings


def list_months(years, months):
    """All (year, month) pairs in chronological order, as 'YYYYMM' strings."""
    for m in months:
        if not 1 <= int(m) <= 12:
            raise ValueError(f'options.months: {m} is not a month (1-12).')
    return [f'{int(y):04d}{int(m):02d}' for y in sorted(years) for m in sorted(months)]


def monthly_file(data_settings, variable, yyyymm):
    outputs_cfg = data_settings['paths']['data_output']
    filename = substitute_keywords(outputs_cfg['filename_monthly'], variable=variable, yyyymm=yyyymm)
    return os.path.join(outputs_cfg['folder_monthly'], filename)


def merged_file(data_settings, variable, yyyymm_list):
    outputs_cfg = data_settings['paths']['data_output']
    filename = substitute_keywords(outputs_cfg['filename_merged'],
                                   label=data_settings['options']['variables'][variable],
                                   start=yyyymm_list[0], end=yyyymm_list[-1])
    return os.path.join(outputs_cfg['folder_merged'], filename)


def read_credentials(server_cfg):
    user = os.environ.get(server_cfg['user_env'])
    password = os.environ.get(server_cfg['password_env'])
    if not user or not password:
        raise ValueError(f"MERIDA credentials not set: export {server_cfg['user_env']}=... "
                         f"{server_cfg['password_env']}=...")
    return user, password


def _curl_quote(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def download_month(data_settings, variable, yyyymm, credentials):
    """Download one monthly file via curl (to ``<file>.<pid>.part``, renamed on success)."""
    server_cfg = data_settings['server']
    url = server_cfg['url'] + substitute_keywords(server_cfg['remote_path'], variable=variable, yyyymm=yyyymm)
    file_out = monthly_file(data_settings, variable, yyyymm)
    file_part = f'{file_out}.{os.getpid()}.part'

    cmd = ['curl', '--fail', '--silent', '--show-error', '--retry', '3', '-K', '-', '-o', file_part]
    if server_cfg['opt_insecure']:
        cmd.insert(1, '--insecure')
    curl_config = f'user = {_curl_quote(credentials[0] + ":" + credentials[1])}\nurl = {_curl_quote(url)}\n'
    result = subprocess.run(cmd, input=curl_config, text=True, capture_output=True)
    if result.returncode != 0:
        if os.path.exists(file_part):
            os.remove(file_part)
        raise RuntimeError(f'curl failed for {url} (exit {result.returncode}): {result.stderr.strip()}')
    os.replace(file_part, file_out)


def download_all(data_settings, yyyymm_list):
    opt = data_settings['options']
    os.makedirs(data_settings['paths']['data_output']['folder_monthly'], exist_ok=True)
    credentials = read_credentials(data_settings['server'])
    failed = []
    for variable in opt['variables']:
        for yyyymm in yyyymm_list:
            file_out = monthly_file(data_settings, variable, yyyymm)
            if os.path.exists(file_out) and not opt['opt_overwrite']:
                print(f'{variable} {yyyymm}: {file_out} exists, skipping.')
                continue
            print(f'{variable} {yyyymm}: downloading...')
            try:
                download_month(data_settings, variable, yyyymm, credentials)
                logging.info('Downloaded %s', file_out)
            except RuntimeError as err:
                print('  ', err)
                logging.error(str(err))
                failed.append((variable, yyyymm))
    return failed


def _decode_absolute_days(values):
    """CDO absolute time, units ``day as %Y%m%d.%f`` (e.g. 20210101.0416667 = 2021-01-01 01:00)."""
    values = np.asarray(values, dtype='float64')
    days = np.floor(values)
    dates = pd.to_datetime(days.astype('int64').astype(str), format='%Y%m%d')
    return (dates + pd.to_timedelta((values - days) * 86400, unit='s')).values


def _open_month(file_month):
    """Open one monthly file lazily, with its time axis as a decoded ``time`` dimension.

    MERIDA files don't agree on the time axis: ``time`` (0.07 deg product),
    ``XTIME`` or ``Times`` (HRES), the latter with CDO absolute units
    (``day as %Y%m%d.%f``) that xarray doesn't decode. The axis is found by its
    ``standard_name``/``axis`` attributes.
    """
    ds = xr.open_dataset(file_month, cache=False)
    names = [n for n in ds.coords if ds[n].ndim == 1 and n == ds[n].dims[0]
             and (n in ('time', 'XTIME', 'Times') or ds[n].attrs.get('standard_name') == 'time'
                  or ds[n].attrs.get('axis') == 'T')]
    if len(names) != 1:
        raise ValueError(f'{file_month}: cannot identify the time axis (candidates: {names})')
    name = names[0]
    if str(ds[name].attrs.get('units', '')).startswith('day as %Y%m%d'):
        ds = ds.assign_coords({name: (name, _decode_absolute_days(ds[name].values), {'standard_name': 'time'})})
    if not np.issubdtype(ds[name].dtype, np.datetime64):
        raise ValueError(f'{file_month}: time axis {name!r} was not decoded (units {ds[name].attrs.get("units")!r})')
    return ds.rename({name: 'time'}) if name != 'time' else ds


def merge_variable(file_list, file_out):
    """Concatenate monthly files along time into ``file_out``, one month in memory at a time.

    Months are written in chronological order of their time stamps (so the
    result is time-sorted); time steps
    already written are dropped, so overlapping months are tolerated. Time
    stamps are rounded to the minute (see below) and stored as float hours
    since 1970; the time axis is renamed ``time`` whatever the file calls it
    (see ``_open_month``).
    """
    datasets = [(f, _open_month(f)) for f in file_list]
    try:
        datasets.sort(key=lambda item: item[1]['time'].values[0])
        first = datasets[0][1]
        (var_name,) = list(first.data_vars)
        var = first[var_name]
        if var.dims != ('time', 'lat', 'lon'):
            raise ValueError(f'{datasets[0][0]}: expected dims (time, lat, lon), got {var.dims}')

        # per-process temp name + exclusive create: a concurrent run on the same
        # output must not truncate this one's file (opening 'w' truncates before
        # HDF5's lock check fails, which corrupted a merge once)
        file_tmp = f'{file_out}.{os.getpid()}.part'
        if os.path.exists(file_tmp):
            raise FileExistsError(f'{file_tmp} exists (stale temp file of an earlier run?) — remove it and retry.')
        try:
            _write_merged(datasets, var_name, file_tmp)
        except BaseException:
            if os.path.exists(file_tmp):
                os.remove(file_tmp)
            raise
        os.replace(file_tmp, file_out)
    finally:
        for _, ds in datasets:
            ds.close()


def _write_merged(datasets, var_name, file_tmp):
    """Write the merged variable to ``file_tmp`` (must not exist yet), month by month."""
    first = datasets[0][1]
    var = first[var_name]
    with netCDF4.Dataset(file_tmp, 'w', clobber=False) as nc:
        nc.setncatts(first.attrs)
        nc.createDimension('time', None)
        for dim in ('lat', 'lon'):
            nc.createDimension(dim, first.sizes[dim])
            v = nc.createVariable(dim, first[dim].dtype, (dim,))
            v.setncatts(first[dim].attrs)
            v[:] = first[dim].values
        t = nc.createVariable('time', 'f8', ('time',))
        t.units, t.calendar, t.standard_name = TIME_UNITS, 'standard', 'time'
        out = nc.createVariable(var_name, var.dtype, ('time', 'lat', 'lon'), zlib=True, complevel=4,
                                fill_value=var.encoding.get('_FillValue'),
                                chunksizes=(1, first.sizes['lat'], first.sizes['lon']))
        out.setncatts(var.attrs)

        last_time = None
        for f, ds in datasets:
            if not (np.array_equal(ds['lat'].values, first['lat'].values)
                    and np.array_equal(ds['lon'].values, first['lon'].values)):
                raise ValueError(f'{f}: lat/lon grid differs from the first month')
            # MERIDA stores time as float (days since the month start; float32 in
            # HRES), so some steps decode slightly off the hour (e.g.
            # 12:59:59.999999999): snap to the minute (the data are hourly)
            times = pd.DatetimeIndex(ds['time'].values).round('min').values
            keep = np.ones(times.size, bool) if last_time is None else times > last_time
            if not keep.any():
                continue
            times = times[keep]
            n = len(t)
            t[n:n + times.size] = (times - TIME_EPOCH) / np.timedelta64(1, 'h')
            out[n:n + times.size] = ds[var_name].values[keep]
            last_time = times[-1]


def merge_all(data_settings, yyyymm_list):
    """Merge each variable's months; returns the variables not merged because months are missing."""
    opt = data_settings['options']
    os.makedirs(data_settings['paths']['data_output']['folder_merged'], exist_ok=True)
    not_merged = []
    for variable in opt['variables']:
        files = [monthly_file(data_settings, variable, m) for m in yyyymm_list]
        present = [f for f in files if os.path.exists(f)]
        missing = [m for f, m in zip(files, yyyymm_list) if not os.path.exists(f)]
        if missing:
            print(f'{variable}: missing months {", ".join(missing)} — not merged.')
            logging.warning('%s: missing months %s, not merged', variable, missing)
            not_merged.append(variable)
            continue
        file_out = merged_file(data_settings, variable, yyyymm_list)
        if os.path.exists(file_out) and not opt['opt_overwrite']:
            print(f'{variable}: {file_out} exists, skipping merge.')
            continue
        print(f'{variable}: merging {len(present)} months into {file_out}...')
        merge_variable(present, file_out)
        logging.info('Merged %s', file_out)
    return not_merged


def main():
    sys.stdout.reconfigure(line_buffering=True)   # show progress when piped to a log
    args = parse_args()
    data_settings = load_config(args.config)
    opt = data_settings['options']
    logging.basicConfig(filename='download_merida.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    yyyymm_list = list_months(opt['years'], opt['months'])
    print(f'{len(yyyymm_list)} month(s), {yyyymm_list[0]}..{yyyymm_list[-1]}; variables:',
          ', '.join(opt['variables']))

    failed, not_merged = [], []
    if not args.merge_only:
        failed = download_all(data_settings, yyyymm_list)
        if failed:
            print('Failed downloads:', ', '.join(f'{v} {m}' for v, m in failed))
    if opt['opt_merge']:
        not_merged = merge_all(data_settings, yyyymm_list)
    logging.info('DONE.')
    return 1 if failed or not_merged else 0


if __name__ == '__main__':
    sys.exit(main())
