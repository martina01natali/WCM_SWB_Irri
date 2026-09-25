"""
Input-assembly layer for the WCM-SWB pipeline. Builds the four model
inputs from raw data sources:

- ``build_input_lonlat`` -- field centroid from a shapefile/geojson.
- ``build_input_soil`` -- field-averaged soil parameters from clipped SoilGrids rasters.
- ``build_input_MERIDA`` -- precipitation/temperature time series from the
  MERIDA reanalysis product, cached to netCDF.
- ``build_input_wcm`` -- merged, resampled sigma0 + NDVI dataframe used to
  drive the WCM-SWB model.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""

import os
import glob
import logging

import numpy as np
import pandas as pd

from wcm_swb.soilgrids import read_aoi, aoi_soil_properties
from wcm_swb.data_readers import read_nc_s0, read_nc_ndvi, read_nc_MERIDA


# ----------------------------------------------------------------------------
# read tif files and extract data based on geometry of shapefile
def build_input_lonlat(file_shapes):
    """
    Compute the (lat, lon) centroid of a field's shapefile/geojson geometry.

    Parameters
    ----------
    file_shapes : str
        Path to a shapefile/geojson readable by geopandas.

    Returns
    -------
    (float, float)
        (lat_centroid, lon_centroid), reprojected to EPSG:4326.
    """
    import geopandas as gpd
    shape_df_WGS84 = gpd.read_file(file_shapes)
    shape_df_WGS84 = shape_df_WGS84.to_crs('EPSG:4326')  # already in projected reference system
    centroid = shape_df_WGS84['geometry'].centroid
    lon_centroid = centroid.x[0]
    lat_centroid = centroid.y[0]
    return lat_centroid, lon_centroid


# ----------------------------------------------------------------------------
def build_input_soil(file_SoilGrids, file_shapes, layer_weighting='thickness'):
    """
    Field-averaged soil texture and water retention from the clipped SoilGrids
    GeoTIFFs written by ``00_preprocessing/download_soilgrids.py``.

    Each layer is averaged over the field polygon (pixels weighted by their
    overlap with it), then the layers are averaged over depth
    (``layer_weighting``: ``'thickness'`` or ``'equal'``, see
    ``wcm_swb.soilgrids.depth_mean``).

    Parameters
    ----------
    file_SoilGrids : str
        Glob matching the field's tifs, e.g.
        ``.../inputs/soilgrids/soilgrids_ceregnano_*.tif``.
    file_shapes : str
        The field's GeoJSON.

    Returns
    -------
    clay [%], sand [%], soc [g/kg], WW_fc [m3/m3] (33 kPa), WW_w [m3/m3] (1500 kPa)
    """
    files = sorted(glob.glob(file_SoilGrids))
    if not files:
        raise FileNotFoundError(f'No SoilGrids tifs matching {file_SoilGrids} '
                                '(run 00_preprocessing/download_soilgrids.py)')
    aoi, crs = read_aoi(file_shapes)
    df = aoi_soil_properties(files, aoi, crs, layer_weighting)
    logging.info('SoilGrids field averages (%s layer weighting):\n%s', layer_weighting, df.to_string())
    missing = {'clay', 'sand', 'soc', 'wwfc', 'www'} - set(df.index)
    if missing:
        raise KeyError(f'SoilGrids variables {sorted(missing)} not found in {file_SoilGrids}')
    profile = df[df.attrs['profile']]
    return tuple(float(profile[v]) for v in ('clay', 'sand', 'soc', 'wwfc', 'www'))


# ----------------------------------------------------------------------------
def build_input_MERIDA(file_MERIDA, lat_centroid, lon_centroid, file_reanalysis):
    """
    Extract precipitation and temperature time series from the MERIDA
    reanalysis product at (lat_centroid, lon_centroid), merge them, derive
    daily min/max temperature and daily rain sum columns, and cache the
    result to ``file_reanalysis`` (netCDF).

    Parameters
    ----------
    file_MERIDA : list of str
        ``[precipitation_file, temperature_file]`` paths.
    lat_centroid, lon_centroid : float
        Target coordinates.
    file_reanalysis : str
        Output path where the merged dataframe is cached as netCDF. Its
        parent directory is created if missing.

    Returns
    -------
    pandas.DataFrame
        Merged MERIDA dataframe indexed by ``datetime``.
    """
    file_reanalysis_dir = '/'.join(file_reanalysis.split('/')[:-1])
    if not os.path.exists(file_reanalysis_dir):
        os.mkdir(file_reanalysis_dir)

    df_merida_rain = read_nc_MERIDA(file_MERIDA[0], lat_centroid, lon_centroid)
    df_merida_temp = read_nc_MERIDA(file_MERIDA[1], lat_centroid, lon_centroid)
    df_merida = df_merida_rain.merge(df_merida_temp, on='time')
    del df_merida_rain, df_merida_temp

    df_merida = df_merida.reset_index().rename(columns={
        'time': 'datetime', 'tp': 'rain', 't2m': 'temperature'
    })

    df_merida = df_merida.set_index('datetime').sort_index().drop(columns='index')

    # MERIDA HRES stores some missing hours as whole fields of 0 K (e.g. 2021-01-11
    # 19:00, 2021-06-22 04:00): treat non-physical temperatures as missing and
    # interpolate them linearly in time
    missing_t = df_merida['temperature'] <= 0
    if missing_t.any():
        logging.warning('MERIDA temperature <= 0 K at %d step(s), interpolated in time: %s',
                        missing_t.sum(), ', '.join(str(t) for t in df_merida.index[missing_t]))
        df_merida['temperature'] = (df_merida['temperature'].mask(missing_t)
                                    .interpolate(method='time', limit_direction='both'))

    # add temperature_min and temperature_max columns
    df_merida['temperature'] = df_merida['temperature'].apply(lambda x: x - 273.16)
    df_merida['temperature_min'] = df_merida.groupby(df_merida.index.date)['temperature'].transform('min')
    df_merida['temperature_max'] = df_merida.groupby(df_merida.index.date)['temperature'].transform('max')

    # add rain_sum column
    df_merida['rain_sum'] = df_merida.groupby(df_merida.index.date)['rain'].transform('sum')
    # ------------------------------------------
    ds_merida = df_merida.to_xarray()
    ds_merida.to_netcdf(file_reanalysis)

    return df_merida


# ----------------------------------------------------------------------------
def build_input_wcm(file_sigma0, file_ndvi, input_config, date_range, freq_out):
    """
    Build the merged sigma0 + NDVI dataframe used to drive the WCM-SWB
    model: reads both sources, resamples/reindexes onto ``date_range`` at
    ``freq_out``, gap-fills NDVI, and merges the two on the datetime index.

    Parameters
    ----------
    file_sigma0 : str
        Path to the sigma0 netCDF file for this field.
    file_ndvi : str
        Path to the NDVI netCDF file for this field.
    input_config : dict
        The ``"input"`` section of the run configuration (configuration.json).
        Forwarded to ``read_nc_s0`` / ``read_nc_ndvi``, which look up
        ``cols_in_sigma0`` / ``cols_out_sigma0`` from it.
    date_range : pandas.DatetimeIndex
        Target index to reindex both series onto.
    freq_out : str
        Output resampling frequency (e.g. ``'h'``).

    Returns
    -------
    pandas.DataFrame
        Merged sigma0 + NDVI dataframe indexed by datetime.
    """
    df_s0 = read_nc_s0([file_sigma0], input_config)
    df_s0 = df_s0.resample(freq_out).first()
    df_so = df_s0.reindex(date_range)
    if 'CR' in df_s0.columns:
        df_s0['CR'] = df_s0.CR.resample(freq_out).interpolate('linear')
        df_s0['CR'] = df_s0.CR.bfill(limit=10).ffill(limit=10)

    logging.info('Read ndvi data...')
    df_ndvi = read_nc_ndvi([file_ndvi], input_config)
    logging.info('Smooth ndvi data...')
    df_ndvi = df_ndvi.resample(freq_out).interpolate('linear')
    df_ndvi = df_ndvi.reindex(date_range)
    df_ndvi['NDVI'] = df_ndvi.NDVI.bfill()
    df_ndvi['NDVI'] = df_ndvi.NDVI.ffill()

    # sigma0 and ndvi dataframes are merged
    input_wcm = pd.merge(df_s0, df_ndvi, right_index=True, left_index=True)
    return input_wcm
