#!/usr/bin/env python
"""
Sentinel-2 NDVI download (Google Earth Engine).

For each field shapefile, pulls ``COPERNICUS/S2_SR_HARMONIZED`` scenes over
the field, keeps one product per acquisition and tile (preferring ESA's
reprocessed versions, see `prefer_reprocessed`), drops scenes with
scene-level cloud cover not below ``sentinel2.CLOUDY_PIXEL_PERCENTAGE``,
reprojects the collection onto a single
reference grid, computes NDVI = (B8 - B4) / (B8 + B4) per pixel and writes one
netCDF per field with dims (datetime, latitude, longitude) and variable NDVI.

No per-pixel cloud masking (s2cloudless / QA60) is applied: scenes are
filtered on scene-level cloud cover only.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)

Usage
-----
    python download_sentinel2_ndvi_gee.py --config configuration_00_preprocessing_TEMPLATE.json
    python download_sentinel2_ndvi_gee.py --config configuration_00_preprocessing_TEMPLATE.json --field ceregnano
"""
import os
import logging

import pandas as pd
import ee

import gee_common

S2_FIRST_DATE = '2017-03-28'


def prefer_reprocessed(collection):
    """Keep one product per (acquisition, tile): the highest PROCESSING_BASELINE,
    then the most recently generated.

    GEE holds ESA's reprocessed products (e.g. baseline 05.00) next to the
    originally ingested ones (e.g. 02.08) for the same acquisition. Without
    this, which version lands in a pixel depends on the order EE returns them.
    ``system:index`` is ``<sensing time>_<generation time>_<tile>``.
    """
    ids = collection.aggregate_array('system:index').getInfo()
    baselines = collection.aggregate_array('PROCESSING_BASELINE').getInfo()
    best = {}
    for idx, baseline in zip(ids, baselines):
        sensing, generation, tile = idx.split('_')
        rank = (float(baseline), generation)
        if (sensing, tile) not in best or rank > best[(sensing, tile)][0]:
            best[(sensing, tile)] = (rank, idx)
    keep = sorted(idx for _, idx in best.values())
    logging.info('%d products, %d kept after preferring reprocessed versions', len(ids), len(keep))
    # sorted by system:index so that, where tiles overlap, the pixel value taken
    # (first in collection order) is deterministic
    return collection.filter(ee.Filter.inList('system:index', keep)).sort('system:index')


def download_field(aoi, data_settings):
    opt = data_settings['options']
    s2_cfg = data_settings['sentinel2']
    if pd.Timestamp(opt['end_date']) < pd.Timestamp(S2_FIRST_DATE):
        raise ValueError('The requested period is outside the operational period of Sentinel-2.')
    start_date = max(pd.Timestamp(opt['start_date']), pd.Timestamp(S2_FIRST_DATE)).strftime('%Y-%m-%d')

    collection = (ee.ImageCollection(s2_cfg['collection'])
                  .filterBounds(aoi)
                  .filter(ee.Filter.date(start_date, opt['end_date'])))
    # pick the version first, so the cloud filter applies to the product actually used
    collection = (prefer_reprocessed(collection)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cfg['CLOUDY_PIXEL_PERCENTAGE'])))
    crs, transform, scale = gee_common.reference_projection(collection, 'B2', s2_cfg.get('reference_image'))
    collection = (collection
                  .map(lambda image: image.reproject(crs, transform))
                  .map(lambda image: image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI')))
                  .select(['NDVI']))
    return gee_common.get_region_pixels(collection, aoi, scale, data_settings, bands=['NDVI']), crs


def main():
    args = gee_common.parse_args(__doc__)
    data_settings = gee_common.load_config(args.config)
    opt = data_settings['options']
    gee_common.setup_logging('download_sentinel2_ndvi_gee.log')
    gee_common.initialize_ee(opt['gee_project'])
    fields = gee_common.read_fields(data_settings, args.field or opt.get('opt_field'))
    print('Found', len(fields), 'field(s):', ', '.join(fields))

    for idx, aoi in fields.items():
        file_stem = gee_common.output_path(data_settings, 'folder_ndvi', 'filename_ndvi', idx)
        if os.path.exists(file_stem + '.nc') and not opt['opt_overwrite']:
            print(f'Field {idx}: {file_stem}.nc exists, skipping (set options.opt_overwrite to redo).')
            continue
        print('Field', idx)
        logging.info('Field %s: download...', idx)
        df, crs = download_field(aoi, data_settings)

        ds = df.set_index(['datetime', 'latitude', 'longitude'])[['NDVI']].to_xarray()
        ds.attrs.update({
            'source': data_settings['sentinel2']['collection'],
            'CLOUDY_PIXEL_PERCENTAGE': data_settings['sentinel2']['CLOUDY_PIXEL_PERCENTAGE'],
            'timezone': opt['timezone'],
        })
        ds.to_netcdf(file_stem + '.nc')
        logging.info('Saved %s.nc', file_stem)
        gee_common.save_tifs(ds, crs, ['NDVI'], file_stem, data_settings)

    logging.info('DONE.')


if __name__ == '__main__':
    main()
