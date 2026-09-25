"""
Shared Google Earth Engine helpers for the 00_preprocessing download scripts.

Kept out of the ``wcm_swb`` package because it needs ``earthengine-api``
(``environment-preprocessing.yml`` only).

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
"""
import os
import glob
import json
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ee

from wcm_swb.config import get_data_settings, substitute_keywords


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', required=True, help='Path to configuration_00_preprocessing.json')
    p.add_argument('--field', default=None, help='Process only this field (shapefile stem)')
    return p.parse_args()


def load_config(file_config):
    """Load the config and expand ``${ENV_VARS}`` in `paths.root`/`options.gee_project`."""
    data_settings = get_data_settings(file_config)
    data_settings['paths']['root'] = os.path.expandvars(data_settings['paths']['root'])
    opt = data_settings['options']
    opt['gee_project'] = os.path.expandvars(opt['gee_project'])
    if opt['gee_project'].startswith('$'):
        raise ValueError('options.gee_project is not set: export GEE_PROJECT=<your Google Cloud project id> '
                         'or write the id in the config.')
    return data_settings


def output_path(data_settings, folder_key, filename_key, opt_field):
    """Resolved output file stem (no extension) for one field; creates its folder."""
    opt = data_settings['options']
    outputs_cfg = data_settings['paths']['data_output']
    folder = os.path.join(data_settings['paths']['root'], outputs_cfg[folder_key])
    os.makedirs(folder, exist_ok=True)
    filename = substitute_keywords(outputs_cfg[filename_key], opt_field=opt_field,
                                   start_date=opt['start_date'], end_date=opt['end_date'])
    return os.path.join(folder, filename)


def plots_folder(data_settings):
    folder = os.path.expandvars(data_settings['paths']['data_output']['folder_plots'])
    os.makedirs(folder, exist_ok=True)
    return folder


def setup_logging(filename_logging):
    if os.path.exists(filename_logging):
        os.remove(filename_logging)
    logging.basicConfig(filename=filename_logging, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')


def initialize_ee(project):
    """Initialize Earth Engine, running the interactive auth flow only if needed."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def read_fields(data_settings, only_field=None):
    """Map field id (shapefile stem) -> ee.Geometry of its first feature.

    Z coordinates, if any, are dropped.
    """
    root = data_settings['paths']['root']
    file_shapes = os.path.join(root, data_settings['paths']['data_input']['file_shapes'])
    fields = {}
    for f in sorted(glob.glob(file_shapes)):
        idx = os.path.splitext(os.path.basename(f))[0]
        if only_field is not None and idx != only_field:
            continue
        with open(f) as fh:
            geometry = json.load(fh)['features'][0]['geometry']
        geometry['coordinates'] = _drop_z(geometry['coordinates'])
        fields[idx] = ee.Geometry(geometry)
    if not fields:
        raise FileNotFoundError(f'No shapefiles matching {file_shapes}'
                                + (f' for field {only_field}' if only_field else ''))
    return fields


def _drop_z(coords):
    if isinstance(coords[0], (int, float)):
        return list(coords[:2])
    return [_drop_z(c) for c in coords]


def reference_projection(collection, band, reference_image=None):
    """Projection (crs, transform, nominal scale) that the whole collection is
    reprojected onto — equivalent to coregistering all images.

    The reference image fixes the sub-pixel offset of the sampling grid, so
    two downloads only give pixel-by-pixel comparable values if they share it.
    `reference_image` is an EE asset id (e.g.
    ``COPERNICUS/S1_GRD_FLOAT/S1A_IW_GRDH_...``). If None, the second-to-last
    image of `collection` is used, i.e. the reference depends on the requested
    period.
    """
    if reference_image:
        image = ee.Image(reference_image)
    else:
        image = ee.Image(collection.toList(collection.size()).get(-2))
    logging.info('Reference image for reprojection: %s', image.get('system:id').getInfo())
    proj = image.select(band).projection()
    info = proj.getInfo()
    scale = ee.Number(proj.nominalScale()).getInfo()
    return info['crs'], info['transform'], scale


def chunked_date_range(start_date, end_date, format_date, chunk_days):
    start = datetime.strptime(start_date, format_date)
    end = datetime.strptime(end_date, format_date)
    while start < end:
        chunk_end = min(start + timedelta(days=chunk_days), end)
        yield start, chunk_end
        start = chunk_end


def to_local_hour(ms, timezone):
    """EE millisecond timestamps -> naive datetimes in `timezone`, floored to the hour.

    The timezone is explicit (Europe/Rome in the templates, DST included) so
    the output doesn't depend on where the script runs.
    """
    t = pd.to_datetime(np.asarray(ms, dtype='float64'), unit='ms', utc=True)
    return t.tz_convert(timezone).tz_localize(None).floor('h')


def get_region_pixels(collection, aoi, scale, data_settings, bands):
    """Pixel values of `bands` over `aoi`, fetched in date chunks via getRegion.

    Returns a DataFrame with columns datetime, longitude, latitude, *bands
    (float, one row per pixel per acquisition hour; masked pixels are NaN).
    """
    opt = data_settings['options']
    blocks, cols = [], None
    for start, end in chunked_date_range(opt['start_date'], opt['end_date'], opt['format_date'], opt['chunk_days']):
        chunk = collection.filterDate(start, end)
        if chunk.size().getInfo() == 0:
            continue
        print('Processing dates:', start.date(), 'to', end.date(), end='\r')
        region = chunk.getRegion(aoi, scale).getInfo()
        cols = region[0]
        blocks.extend(region[1:])
    print()
    if not blocks:
        raise RuntimeError('No images found over the AOI in the requested period.')

    df = pd.DataFrame(blocks, columns=cols)
    df['datetime'] = to_local_hour(df['time'], opt['timezone'])
    df = df.drop(columns=['time', 'id'])
    # several scenes of the same pass can cover one pixel within the same hour
    df = df.groupby(['datetime', 'longitude', 'latitude']).first().reset_index()
    df[bands] = df[bands].astype('float')
    return df[['datetime', 'longitude', 'latitude', *bands]]


def save_tifs(ds, crs, variables, file_stem, data_settings):
    """Optional GeoTIFF exports: first acquisition (all variables) and/or one
    multi-band (one band per date) file per variable."""
    outputs_cfg = data_settings['paths']['data_output']
    if not (outputs_cfg['opt_save_single_tif'] or outputs_cfg['opt_save_multiday_tif']):
        return
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude').rio.write_crs(crs)
    if outputs_cfg['opt_save_single_tif']:
        first = ds.isel(datetime=0)
        first[variables].rio.to_raster(f"{file_stem}_{pd.Timestamp(first.datetime.values):%Y-%m-%d}.tif")
    if outputs_cfg['opt_save_multiday_tif']:
        for var in variables:
            ds[var].rio.to_raster(f'{file_stem}_multi_{var}.tif')
