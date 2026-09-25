"""
Dataset readers for the WCM-SWB pipeline: Sentinel-1 sigma0, Sentinel-2
NDVI, calibration-output netCDF files, and the MERIDA reanalysis.

``read_nc_s0`` and ``read_nc_ndvi`` take an explicit ``input_config``
argument -- the ``"input"`` section of ``configuration.json`` -- and read the
column mapping (``cols_in_sigma0`` / ``cols_out_sigma0``) from it.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""

import xarray as xr
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# read sigma0 data
def read_nc_s0(file_sigma0, input_config: dict):
    """
    Read one or more Sentinel-1 sigma0 netCDF files and return a merged,
    hourly-indexed dataframe.

    Parameters
    ----------
    file_sigma0 : list of str
        Paths to the sigma0 netCDF file(s) to read and merge.
    input_config : dict
        The ``"input"`` section of the run configuration (configuration.json).
        Must provide ``cols_in_sigma0`` and ``cols_out_sigma0`` (parallel
        lists used to rename the raw netCDF variables to canonical column
        names, e.g. ``datetime``, ``latitude``, ``longitude``, ``orb``,
        ``VV``, ``VH``, ``CR``, ``angle``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``datetime``, columns renamed per ``cols_out_sigma0``
        (suffixed per source file when merging more than one file).
    """
    cols_in_sigma0 = input_config['cols_in_sigma0']
    cols_out_sigma0 = input_config['cols_out_sigma0']

    df_s0 = None
    for f in file_sigma0:
        f_id = ('_').join(f.split('/')[-1].split('.')[-2].split('_')[1:])
        ds = xr.open_dataset(f)
        df = ds.to_dataframe().reset_index()
        df = df.rename(columns={cols_in_sigma0[i]: cols_out_sigma0[i] for i in range(len(cols_in_sigma0))})
        df = df.set_index('datetime')
        df = df.groupby(df.index).mean()
        if df_s0 is None:
            df_s0 = df.copy()
            f_id_first = f_id
        else:
            df_s0 = df_s0.merge(df, on='datetime', suffixes=('_' + f_id_first, '_' + f_id))
    return df_s0


# ----------------------------------------------------------------------------
# read ndvi data
def read_nc_ndvi(file_ndvi, input_config: dict):
    """
    Read one or more NDVI netCDF files and return a merged, hourly,
    linearly-interpolated dataframe of the ``NDVI`` column.

    Parameters
    ----------
    file_ndvi : list of str
        Paths to the NDVI netCDF file(s) to read and merge.
    input_config : dict
        The ``"input"`` section of the run configuration (configuration.json).

    Notes
    -----
    Columns are renamed using ``cols_in_sigma0`` / ``cols_out_sigma0`` (the
    *sigma0* column config) rather than ``cols_in_ndvi`` /
    ``cols_out_ndvi``. This is harmless because the shared leading columns (``datetime``,
    ``latitude``, ``longitude``) map to themselves in both configs, and the
    trailing sigma0-only columns (``orb``, ``VV``, ``VH``, ``CR``, ``angle``)
    are simply absent from the NDVI dataframe, so the rename is a no-op for
    them; the ``NDVI`` column itself is untouched by the rename either way.
    """
    cols_in_sigma0 = input_config['cols_in_sigma0']
    cols_out_sigma0 = input_config['cols_out_sigma0']

    df_ndvi = None
    for f in file_ndvi:
        f_id = ('_').join(f.split('/')[-1].split('.')[-2].split('_')[1:])
        ds = xr.open_dataset(f)
        df = ds.to_dataframe().reset_index()
        df = df.rename(columns={cols_in_sigma0[i]: cols_out_sigma0[i] for i in range(len(cols_in_sigma0))})
        df = df.set_index('datetime')
        df = df.groupby(df.index).mean()
        df = pd.DataFrame(df['NDVI'])
        if df_ndvi is None:
            df_ndvi = df.copy()
            f_id_first = f_id
        else:
            df_ndvi = df_ndvi.merge(df, on='datetime', suffixes=('_' + f_id_first, '_' + f_id))
    df_ndvi = df_ndvi.resample('d').mean().resample('h').interpolate()
    return df_ndvi


# ----------------------------------------------------------------------------
# read output nc file
def read_nc_output(filename_output):
    """
    Re-load a previously-written model output netCDF file into a dataframe.

    Parameters
    ----------
    filename_output : str
        Path to an output netCDF file, expected to follow the pipeline's
        ``..._{field_id}_...`` naming convention (the field id is extracted
        from the 5th underscore-separated token of the filename and used to
        strip the per-field column suffix).

    Returns
    -------
    (str, pandas.DataFrame)
        The extracted field id, and the dataframe indexed by ``Datetime``.
    """
    field_id = filename_output.split('_')[4]
    ds = xr.open_dataset(filename_output)
    df = ds.to_dataframe().reset_index()
    df = df.rename(columns={col: col.replace('_' + field_id, '') for col in df.columns})
    df = df.set_index('Datetime')
    return field_id, df


# ----------------------------------------------------------------------------
# read nc files from MERIDA reanalysis product
def read_nc_MERIDA(file_reanalysis, lat_centroid, lon_centroid):
    """
    Read a MERIDA reanalysis netCDF file and extract the time series for the
    grid cell nearest to (lat_centroid, lon_centroid).

    Parameters
    ----------
    file_reanalysis : str
        Path to the MERIDA netCDF file.
    lat_centroid, lon_centroid : float
        Target latitude/longitude to select the nearest grid cell.

    Returns
    -------
    pandas.DataFrame
        The selected grid cell's time series, with the ``lon``/``lat``
        columns dropped.
    """
    ds_merida = xr.open_dataset(file_reanalysis)
    lats = np.unique(ds_merida.lat.values)
    lons = np.unique(ds_merida.lon.values)
    lon_idx = lons[np.abs(lons - lon_centroid).argmin()]
    lat_idx = lats[np.abs(lats - lat_centroid).argmin()]
    print(f'merida centroid is at: ({lon_idx:.6f}, {lat_idx:.6f})')
    ds_centroid = ds_merida.sel(lon=[lon_idx], lat=[lat_idx])
    df_centroid = ds_centroid.to_dataframe().reset_index().drop(columns=['lon', 'lat'])

    return df_centroid
