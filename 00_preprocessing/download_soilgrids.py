#!/usr/bin/env python
"""
SoilGrids 2.0 soil properties download (ISRIC WCS), clipped to each field.

For each field shapefile, requests the SoilGrids 2.0 layers listed in
``soilgrids.properties`` (default: sand, clay, SOC, water content at field
capacity ``wv0033`` (33 kPa) and at wilting point ``wv1500`` (1500 kPa)) for
every depth in ``soilgrids.depths`` (default 0-60 cm) from
https://maps.isric.org, on the native 250 m Homolosine grid (no resampling),
over the smallest block of whole pixels covering the field.

Output: one small GeoTIFF per (variable, depth),
``<folder_soil>/<filename_soil>_<variable>_<depth>.tif``, raw SoilGrids mapped
integers (no-data 0) tagged with property, depth, unit and scale factor. These
are the soil input of ``01_calibration``, which averages them over the field
(``wcm_swb.inputs.build_input_soil``). The same field averages are printed and
logged here as a check.

No Google Earth Engine account is needed. Needs ``rasterio``, ``shapely``,
``pyproj`` (all in ``environment-preprocessing.yml``).

Usage
-----
    python download_soilgrids.py --config configuration_00_preprocessing_TEMPLATE.json
    python download_soilgrids.py --config configuration_00_preprocessing_TEMPLATE.json --field ceregnano
"""
import os
import glob
import time
import logging
import argparse
import urllib.parse
import urllib.request

from rasterio.io import MemoryFile

from wcm_swb.config import get_data_settings, substitute_keywords
from wcm_swb import soilgrids as sg

N_RETRIES = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='Path to configuration_00_preprocessing.json')
    p.add_argument('--field', default=None, help='Process only this field (shapefile stem)')
    return p.parse_args()


def read_fields(data_settings, only_field=None):
    """Map field id (shapefile stem) -> (AOI polygon, its CRS)."""
    root = data_settings['paths']['root']
    file_shapes = os.path.join(root, data_settings['paths']['data_input']['file_shapes'])
    fields = {}
    for f in sorted(glob.glob(file_shapes)):
        idx = os.path.splitext(os.path.basename(f))[0]
        if only_field is not None and idx != only_field:
            continue
        fields[idx] = sg.read_aoi(f)
    if not fields:
        raise FileNotFoundError(f'No shapefiles matching {file_shapes}'
                                + (f' for field {only_field}' if only_field else ''))
    return fields


def get_coverage(wcs_url, prop, coverage_id, snapped_bounds, size):
    """Download one snapped window of a coverage; returns the raw int16 array."""
    x0, y0, x1, y1 = snapped_bounds
    query = urllib.parse.urlencode([
        ('map', f'/map/{prop}.map'), ('SERVICE', 'WCS'), ('VERSION', '2.0.1'),
        ('REQUEST', 'GetCoverage'), ('COVERAGEID', coverage_id), ('FORMAT', 'image/tiff'),
        ('SUBSET', f'x({x0},{x1})'), ('SUBSET', f'y({y0},{y1})'),
        ('SCALESIZE', f'x({size[0]}),y({size[1]})'),
    ])
    url = f'{wcs_url}?{query}'
    for attempt in range(1, N_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                content = resp.read()
            with MemoryFile(content) as mem, mem.open() as src:
                data = src.read(1)
            break
        except Exception as err:  # network errors or an XML error page instead of a tiff
            logging.warning('%s attempt %d failed: %s', coverage_id, attempt, err)
            if attempt == N_RETRIES:
                raise RuntimeError(f'Could not download {coverage_id} from {url}') from err
            time.sleep(5 * attempt)
    if data.shape != (size[1], size[0]):
        raise RuntimeError(f'{coverage_id}: expected {size[1]}x{size[0]} pixels, got {data.shape}')
    return data


def main():
    args = parse_args()
    data_settings = get_data_settings(args.config)
    data_settings['paths']['root'] = os.path.expandvars(data_settings['paths']['root'])
    opt = data_settings['options']
    outputs_cfg = data_settings['paths']['data_output']
    cfg = data_settings['soilgrids']
    if os.path.exists('download_soilgrids.log'):
        os.remove('download_soilgrids.log')
    logging.basicConfig(filename='download_soilgrids.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    fields = read_fields(data_settings, args.field or opt.get('opt_field'))
    print('Found', len(fields), 'field(s):', ', '.join(fields))
    folder = os.path.join(data_settings['paths']['root'], outputs_cfg['folder_soil'])
    os.makedirs(folder, exist_ok=True)

    for idx, (aoi, crs) in fields.items():
        file_stem = os.path.join(folder, substitute_keywords(outputs_cfg['filename_soil'], opt_field=idx))
        files_tif = {(name, prop, depth): f'{file_stem}_{name}_{depth}.tif'
                     for name, prop in cfg['properties'].items() for depth in cfg['depths']}
        if all(os.path.exists(f) for f in files_tif.values()) and not opt['opt_overwrite']:
            print(f'Field {idx}: {file_stem}_*.tif exist, skipping (set options.opt_overwrite to redo).')
            continue
        print('Field', idx)
        aoi_igh = sg.to_igh(aoi, crs)
        snapped, size = sg.snap_to_grid(aoi_igh.bounds)
        logging.info('Field %s: AOI %.1f ha, window %dx%d pixels', idx, aoi_igh.area / 1e4, size[0], size[1])
        for (name, prop, depth), file_tif in files_tif.items():
            coverage_id = f"{prop}_{depth}_{cfg['statistic']}"
            data = get_coverage(cfg['wcs_url'], prop, coverage_id, snapped, size)
            sg.write_layer_tif(file_tif, data, snapped, name, prop, depth)
            logging.info('Saved %s (%s)', file_tif, coverage_id)

        df = sg.aoi_soil_properties(list(files_tif.values()), aoi, crs, layer_weighting='thickness')
        print(df.to_string(float_format='%.4g'))
        logging.info('Field %s averages (thickness-weighted over depth):\n%s', idx, df.to_string())
        for variable, frac in df['min_valid_fraction'].items():
            if frac < 1:
                logging.warning('%s: no data on %.0f%% of the field in at least one layer', variable, 100 * (1 - frac))

    logging.info('DONE.')


if __name__ == '__main__':
    main()
