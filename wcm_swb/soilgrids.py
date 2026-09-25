"""
SoilGrids 2.0 helpers: AOI clipping and spatial / depth averaging.

No network access here: the download is ``00_preprocessing/download_soilgrids.py``,
which stores one small clipped GeoTIFF per (property, depth) in the study
site's ``inputs/soilgrids/`` folder; ``aoi_soil_properties`` turns those into
AOI-averaged values (used by ``wcm_swb.inputs.build_input_soil``).

SoilGrids 2.0 (Poggio et al. 2021, doi:10.5194/soil-7-217-2021; water
retention: Turek et al. 2023, doi:10.1016/j.geoderma.2023.116375) is served
on a 250 m grid in the Interrupted Goode Homolosine projection. That
projection is equal-area, so pixel/AOI overlap areas computed in it are true
ground areas.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""
import json
import math

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.transform import from_origin
import shapely
import shapely.geometry
import shapely.ops

# Native grid of every SoilGrids 2.0 coverage (from WCS DescribeCoverage):
# upper-left corner and pixel size, in metres.
IGH_PROJ4 = '+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs'
GRID_X0 = -19949750.0
GRID_Y0 = 8361000.0
GRID_RES = 250.0
NODATA = -32768
# The native files use NODATA, but the WCS returns 0 where there is no data
# (sea, built-up areas); 0 is not a plausible prediction for any property
# used here, so it is treated as missing too.
NODATA_VALUES = (NODATA, 0)

# Mapped value -> physical unit: (factor, unit). See
# https://www.isric.org/explore/soilgrids/faq-soilgrids#What_do_the_filename_codes_mean
UNIT_CONVERSION = {
    'sand': (0.1, '%'),         # g/kg
    'clay': (0.1, '%'),         # g/kg
    'silt': (0.1, '%'),         # g/kg
    'soc': (0.1, 'g/kg'),       # dg/kg
    'wv0010': (1e-3, 'm3/m3'),  # 10^-3 cm3/cm3, 10 kPa
    'wv0033': (1e-3, 'm3/m3'),  # 10^-3 cm3/cm3, 33 kPa (field capacity)
    'wv1500': (1e-3, 'm3/m3'),  # 10^-3 cm3/cm3, 1500 kPa (wilting point)
}


def depth_bounds(depth):
    """``'15-30cm'`` -> ``(15.0, 30.0)``."""
    top, bottom = depth.removesuffix('cm').split('-')
    return float(top), float(bottom)


def to_igh(geometry, crs_in='EPSG:4326'):
    """Reproject a shapely geometry (lon/lat by default) to the SoilGrids projection."""
    transformer = pyproj.Transformer.from_crs(crs_in, IGH_PROJ4, always_xy=True)
    return shapely.ops.transform(transformer.transform, geometry)


def snap_to_grid(bounds):
    """Expand ``(xmin, ymin, xmax, ymax)`` outward to whole SoilGrids pixels.

    Requesting a snapped box makes the WCS return native pixels instead of
    resampling onto the requested extent.
    Returns ``(xmin, ymin, xmax, ymax), (ncols, nrows)``.
    """
    xmin, ymin, xmax, ymax = bounds
    x0 = GRID_X0 + math.floor((xmin - GRID_X0) / GRID_RES) * GRID_RES
    x1 = GRID_X0 + math.ceil((xmax - GRID_X0) / GRID_RES) * GRID_RES
    y1 = GRID_Y0 - math.floor((GRID_Y0 - ymax) / GRID_RES) * GRID_RES
    y0 = GRID_Y0 - math.ceil((GRID_Y0 - ymin) / GRID_RES) * GRID_RES
    return (x0, y0, x1, y1), (round((x1 - x0) / GRID_RES), round((y1 - y0) / GRID_RES))


def overlap_weights(aoi, snapped_bounds, shape):
    """Fraction of the AOI area falling in each pixel of the snapped window.

    `shape` is ``(nrows, ncols)``, row 0 at the top. Weights sum to 1. Fields
    are often only a few SoilGrids pixels wide, so this area weighting is
    used instead of a pixel-centre-in-polygon mask.
    """
    x0, _, _, y1 = snapped_bounds
    nrows, ncols = shape
    weights = np.zeros(shape)
    for i in range(nrows):
        top = y1 - i * GRID_RES
        for j in range(ncols):
            left = x0 + j * GRID_RES
            pixel = shapely.geometry.box(left, top - GRID_RES, left + GRID_RES, top)
            weights[i, j] = pixel.intersection(aoi).area
    return weights / aoi.area


def spatial_mean(values, weights, nodata_values=NODATA_VALUES):
    """Area-weighted mean of a raw raster window, skipping `nodata_values` pixels.

    Returns ``(mean, covered_fraction)``: the fraction of the AOI area with
    valid data. The mean is NaN if no valid pixel overlaps the AOI.
    """
    values = np.asarray(values, dtype=float)
    valid = ~np.isin(values, nodata_values) & np.isfinite(values) & (weights > 0)
    covered = weights[valid].sum()
    if covered == 0:
        return np.nan, 0.0
    return float((values[valid] * weights[valid]).sum() / covered), float(covered)


def depth_mean(layer_values, depths, weighting='thickness'):
    """Average per-layer values over the profile, skipping NaN layers.

    ``weighting='thickness'`` weights each layer by its thickness (i.e. the
    mean over the 0-bottom profile); ``'equal'`` gives each layer the same
    weight (what ``wcm_swb.inputs.build_input_soil`` does).
    """
    values = np.asarray(layer_values, dtype=float)
    if weighting == 'thickness':
        w = np.array([bottom - top for top, bottom in map(depth_bounds, depths)])
    elif weighting == 'equal':
        w = np.ones(len(depths))
    else:
        raise ValueError(f"layer_weighting must be 'thickness' or 'equal', got {weighting!r}")
    ok = np.isfinite(values)
    if not ok.any():
        return np.nan
    return float((values[ok] * w[ok]).sum() / w[ok].sum())


def read_aoi(file_geojson):
    """AOI polygon (union of all features, Z dropped) and its CRS from a GeoJSON file."""
    with open(file_geojson) as fh:
        gj = json.load(fh)
    crs = gj.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
    if crs.endswith('CRS84'):
        crs = 'EPSG:4326'  # same datum; lon/lat order is handled by always_xy
    geom = shapely.ops.unary_union([shapely.geometry.shape(f['geometry']) for f in gj['features']])
    return shapely.force_2d(geom), crs


def write_layer_tif(file_tif, data, snapped_bounds, variable, prop, depth, nodata=0):
    """Save one raw (mapped-unit) SoilGrids window as a GeoTIFF, tagged with
    what it is so readers don't have to parse the filename."""
    factor, unit = UNIT_CONVERSION[prop]
    x0, _, _, y1 = snapped_bounds
    with rasterio.open(file_tif, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1,
                       dtype=data.dtype, crs=IGH_PROJ4, nodata=nodata,
                       transform=from_origin(x0, y1, GRID_RES, GRID_RES)) as dst:
        dst.write(data, 1)
        dst.update_tags(variable=variable, soilgrids_property=prop, depth=depth,
                        scale_factor=factor, unit=unit)


def aoi_soil_properties(files_tif, aoi, crs='EPSG:4326', layer_weighting='thickness'):
    """AOI-averaged soil properties from clipped SoilGrids GeoTIFFs.

    Each tif (as written by `write_layer_tif`) is averaged over the AOI with
    `overlap_weights`, converted to physical units, then the layers of each
    variable are averaged with `depth_mean`.

    Returns a DataFrame indexed by variable with columns ``unit``, one per
    depth, the profile average (e.g. ``0-60cm``) and ``min_valid_fraction``
    (smallest fraction of the AOI with valid data across the layers). The
    profile column name is stored in ``df.attrs['profile']``.
    """
    aoi_igh = to_igh(aoi, crs)
    layers = {}
    for f in files_tif:
        with rasterio.open(f) as src:
            tags = src.tags()
            if not np.allclose(src.res, GRID_RES):
                raise ValueError(f'{f}: not on the native SoilGrids {GRID_RES:g} m grid')
            weights = overlap_weights(aoi_igh, tuple(src.bounds), src.shape)
            nodata = (src.nodata, NODATA) if src.nodata is not None else NODATA_VALUES
            value, covered = spatial_mean(src.read(1), weights, nodata)
        layers.setdefault(tags['variable'], {})[tags['depth']] = (
            value * float(tags['scale_factor']), covered, tags['unit'])
    if not layers:
        raise FileNotFoundError('No SoilGrids tifs given')

    rows = {}
    all_depths = sorted({d for v in layers.values() for d in v}, key=depth_bounds)
    profile = f'{depth_bounds(all_depths[0])[0]:g}-{depth_bounds(all_depths[-1])[1]:g}cm'
    for variable, by_depth in layers.items():
        depths = sorted(by_depth, key=depth_bounds)
        if depths != all_depths:
            raise ValueError(f'{variable}: layers {depths} differ from the other variables\' {all_depths}')
        row = {'unit': by_depth[depths[0]][2]}
        row.update({d: by_depth[d][0] for d in depths})
        row[profile] = depth_mean([by_depth[d][0] for d in depths], depths, layer_weighting)
        row['min_valid_fraction'] = min(by_depth[d][1] for d in depths)
        rows[variable] = row
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'variable'
    df.attrs['profile'] = profile
    return df

