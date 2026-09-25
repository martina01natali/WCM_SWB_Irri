#!/usr/bin/env python
"""
Sentinel-1 backscatter download (Google Earth Engine) + cross-orbit normalization.

For each field shapefile, pulls every ``COPERNICUS/S1_GRD_FLOAT`` acquisition
over the field, reprojects the collection onto a single reference grid
(coregistration), extracts per-pixel VV/VH/incidence angle, attaches the
relative orbit of each acquisition, and HIST-normalizes the backscatter across
orbits onto a reference orbit (Mladenova et al. 2013; see ``wcm_swb.sigma0``).

Output: one netCDF per field, dims (datetime, latitude, longitude), variables

    orb                     relative orbit number
    VV_notnorm, VH_notnorm  raw backscatter [dB]
    angle                   incidence angle [deg]
    VV, VH                  HIST-normalized backscatter [dB]  <- used by 01_calibration
    CR                      cross ratio VH/VV [linear], from the normalized values

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md

Usage
-----
    python download_sentinel1_gee.py --config configuration_00_preprocessing_TEMPLATE.json
    python download_sentinel1_gee.py --config configuration_00_preprocessing_TEMPLATE.json --field ceregnano
"""
import os
import logging

import numpy as np
import pandas as pd
import ee

from wcm_swb.sigma0 import lin_db, normalize_orbits
import gee_common

# S1_GRD_FLOAT has a scene on 2016-10-01 that makes EE requests fail, so
# 2016-09-30..2016-10-01 is skipped (end dates are exclusive).
GAP_START, GAP_END = '2016-09-30', '2016-10-02'
S1_FIRST_DATE = '2014-10-03'


def date_filters(start_date, end_date):
    """EE date filters covering [start_date, end_date) minus the 2016-10-01 gap."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start < pd.Timestamp(S1_FIRST_DATE):
        raise ValueError(f'start_date must be on or after {S1_FIRST_DATE} (Sentinel-1 operational period).')
    if end <= pd.Timestamp(GAP_START) or start >= pd.Timestamp(GAP_END):
        return [ee.Filter.date(start_date, end_date)]
    filters = []
    if start < pd.Timestamp(GAP_START):
        filters.append(ee.Filter.date(start_date, GAP_START))
    if end > pd.Timestamp(GAP_END):
        filters.append(ee.Filter.date(GAP_END, end_date))
    return filters


def acquisition_metadata(collection, timezone):
    """Relative orbit and pass direction of each acquisition hour."""
    meta = pd.DataFrame({
        'datetime': gee_common.to_local_hour(collection.aggregate_array('system:time_start').getInfo(), timezone),
        'orb': collection.aggregate_array('relativeOrbitNumber_start').getInfo(),
        'pass': collection.aggregate_array('orbitProperties_pass').getInfo(),
    })
    return meta.drop_duplicates(subset=['datetime', 'orb'])


def download_field(aoi, data_settings):
    """Raw per-pixel S1 data for one field, with the orbit of each acquisition."""
    opt = data_settings['options']
    collections = [ee.ImageCollection(data_settings['sentinel1']['collection'])
                   .filterBounds(aoi).filter(f).sort('system:time_start')
                   for f in date_filters(opt['start_date'], opt['end_date'])]
    crs, transform, scale = gee_common.reference_projection(collections[0], 'VV',
                                                            data_settings['sentinel1'].get('reference_image'))

    pixels, meta = [], []
    for col in collections:
        col = col.map(lambda image: image.reproject(crs, transform))
        meta.append(acquisition_metadata(col, opt['timezone']))
        pixels.append(gee_common.get_region_pixels(col, aoi, scale, data_settings, bands=['VV', 'VH', 'angle']))
    meta = pd.concat(meta)
    df = pd.concat(pixels)
    for pol in ['VV', 'VH']:
        df[pol] = lin_db(df[pol])
    df = df.merge(meta[['datetime', 'orb']], on='datetime', how='inner')
    df = df.rename(columns={'VV': 'VV_notnorm', 'VH': 'VH_notnorm'})
    return df, meta, crs


def plot_histograms(df, orb_ref, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ['VV_notnorm', 'VV'], ['not normalized', f'normalized (ref orb: {orb_ref})']):
        for orb, g in df.groupby('orb'):
            nbins = min(int(np.sqrt(len(g))), 100)
            ax.hist(g[col], bins=nbins, density=True, alpha=.5, label=f'orb {orb:.0f}')
        ax.set_title(title)
        ax.set_xlabel(r'$\sigma^0$ VV [dB]')
    axes[0].set_ylabel('Density')
    axes[1].legend(loc='best')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    args = gee_common.parse_args(__doc__)
    data_settings = gee_common.load_config(args.config)
    opt = data_settings['options']
    gee_common.setup_logging('download_sentinel1_gee.log')
    gee_common.initialize_ee(opt['gee_project'])
    fields = gee_common.read_fields(data_settings, args.field or opt.get('opt_field'))
    print('Found', len(fields), 'field(s):', ', '.join(fields))

    for idx, aoi in fields.items():
        file_stem = gee_common.output_path(data_settings, 'folder_sigma0', 'filename_sigma0', idx)
        if os.path.exists(file_stem + '.nc') and not opt['opt_overwrite']:
            print(f'Field {idx}: {file_stem}.nc exists, skipping (set options.opt_overwrite to redo).')
            continue
        print('Field', idx)
        logging.info('Field %s: download...', idx)
        df, meta, crs = download_field(aoi, data_settings)

        for orb, g in meta.groupby('orb'):
            logging.info('orbit %d: %s, %d acquisitions, first at %s', orb, g['pass'].iloc[0], len(g), g['datetime'].min())

        logging.info('Field %s: normalization...', idx)
        df = df.dropna(subset=['VV_notnorm', 'VH_notnorm', 'angle'])
        df, orb_ref, stats = normalize_orbits(df, reference_angle=data_settings['sentinel1']['reference_angle'])
        logging.info('Reference orbit: %d\n%s', orb_ref, stats.to_string())
        print('Reference orbit:', orb_ref)

        df = df.groupby(['datetime', 'latitude', 'longitude']).mean().reset_index()
        variables = ['orb', 'VV_notnorm', 'VH_notnorm', 'angle', 'VV', 'VH', 'CR']
        ds = df.set_index(['datetime', 'latitude', 'longitude'])[variables].to_xarray()
        ds.attrs.update({
            'source': data_settings['sentinel1']['collection'],
            'normalization': 'HIST (Mladenova et al. 2013) across relative orbits',
            'reference_orbit': int(orb_ref),
            'timezone': opt['timezone'],
        })
        ds.to_netcdf(file_stem + '.nc')
        logging.info('Saved %s.nc', file_stem)
        gee_common.save_tifs(ds, crs, ['VV', 'VH', 'CR'], file_stem, data_settings)
        if data_settings['paths']['data_output']['opt_save_plots']:
            plot_histograms(df, orb_ref, os.path.join(gee_common.plots_folder(data_settings), f'hist_s0_{idx}.png'))

    logging.info('DONE.')


if __name__ == '__main__':
    main()
