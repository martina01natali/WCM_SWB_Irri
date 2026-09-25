"""Tests for the SoilGrids clipping/averaging helpers (wcm_swb.soilgrids)."""

import glob
import json
import os

import numpy as np
import pytest
import pyproj
import shapely.geometry
import shapely.ops

from wcm_swb import soilgrids as sg

CEREGNANO_SHAPE = os.path.join(os.path.dirname(__file__), '..', 'study_sites', 'Ceregnano', 'shapefile',
                               'ceregnano.geojson')
R = sg.GRID_RES


def _cell(col, row):
    """Bounds of grid cell (col, row) counted from the SoilGrids grid origin."""
    x = sg.GRID_X0 + col * R
    y = sg.GRID_Y0 - row * R
    return x, y - R, x + R, y


def test_depth_bounds():
    assert sg.depth_bounds('15-30cm') == (15.0, 30.0)


def test_snap_to_grid_keeps_aligned_box():
    b = (*_cell(100, 200)[:2], *_cell(102, 199)[2:])  # 3 cols x 2 rows
    snapped, size = sg.snap_to_grid(b)
    assert snapped == pytest.approx(b)
    assert size == (3, 2)


def test_snap_to_grid_expands_to_whole_pixels():
    x0, y0, _, _ = _cell(100, 200)
    snapped, size = sg.snap_to_grid((x0 + 10, y0 + 10, x0 + 260, y0 + 20))
    assert snapped == pytest.approx((x0, y0, x0 + 2 * R, y0 + R))
    assert size == (2, 1)


def test_overlap_weights_split_between_pixels():
    x0, y0, _, _ = _cell(100, 200)
    # 100 m x 100 m square, 3/4 in the left pixel and 1/4 in the right one
    aoi = shapely.geometry.box(x0 + 175, y0 + 50, x0 + 275, y0 + 150)
    snapped, size = sg.snap_to_grid(aoi.bounds)
    w = sg.overlap_weights(aoi, snapped, (size[1], size[0]))
    assert w.shape == (1, 2)
    assert w.ravel() == pytest.approx([0.75, 0.25])


def test_spatial_mean_skips_nodata_and_reports_coverage():
    weights = np.array([[0.5, 0.25], [0.25, 0.0]])
    values = np.array([[10, sg.NODATA], [0, 99]])  # 0 = WCS no-data; 99 has no weight
    mean, covered = sg.spatial_mean(values, weights)
    assert mean == pytest.approx(10.0)
    assert covered == pytest.approx(0.5)
    mean, covered = sg.spatial_mean(np.full((2, 2), sg.NODATA), weights)
    assert np.isnan(mean) and covered == 0.0


def test_depth_mean_weightings():
    depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm']
    values = [1.0, 2.0, 3.0, 4.0]
    assert sg.depth_mean(values, depths, 'equal') == pytest.approx(2.5)
    assert sg.depth_mean(values, depths, 'thickness') == pytest.approx((5 + 20 + 45 + 120) / 60)
    assert sg.depth_mean([1.0, np.nan, 3.0, 4.0], depths, 'equal') == pytest.approx(8 / 3)
    with pytest.raises(ValueError):
        sg.depth_mean(values, depths, 'bogus')


def test_ceregnano_aoi_area_and_weights():
    """Homolosine is equal-area: the reprojected field keeps its surveyed area."""
    with open(CEREGNANO_SHAPE) as fh:
        feat = json.load(fh)['features'][0]
    aoi = sg.to_igh(shapely.geometry.shape(feat['geometry']))
    assert aoi.area == pytest.approx(feat['properties']['area'], rel=1e-3)
    snapped, size = sg.snap_to_grid(aoi.bounds)
    w = sg.overlap_weights(aoi, snapped, (size[1], size[0]))
    assert w.sum() == pytest.approx(1.0)
    assert size == (2, 3)


CEREGNANO_SOIL = os.path.join(os.path.dirname(__file__), '..', 'study_sites', 'Ceregnano', 'inputs', 'soilgrids',
                              'soilgrids_ceregnano_*.tif')


def test_write_read_roundtrip(tmp_path):
    x0, y0, _, _ = _cell(87297, 13403)  # near Ceregnano: inside the valid projection area
    snapped = (x0, y0, x0 + 2 * R, y0 + R)
    aoi_igh = shapely.geometry.box(x0 + 175, y0 + 50, x0 + 275, y0 + 150)  # 3/4 left, 1/4 right pixel
    aoi = shapely.ops.transform(pyproj.Transformer.from_crs(sg.IGH_PROJ4, 'EPSG:4326', always_xy=True).transform,
                                aoi_igh)
    files = []
    for depth, row in [('0-5cm', [100, 200]), ('5-15cm', [300, 0])]:  # 0 = no data
        f = str(tmp_path / f'x_sand_{depth}.tif')
        sg.write_layer_tif(f, np.array([row], dtype=np.int16), snapped, 'sand', 'sand', depth)
        files.append(f)
    df = sg.aoi_soil_properties(files, aoi, layer_weighting='equal')
    assert df.loc['sand', 'unit'] == '%'
    assert df.loc['sand', '0-5cm'] == pytest.approx(12.5, rel=1e-3)
    assert df.loc['sand', '5-15cm'] == pytest.approx(30.0)
    assert df.loc['sand', '0-15cm'] == pytest.approx((12.5 + 30.0) / 2, rel=1e-3)
    assert df.loc['sand', 'min_valid_fraction'] == pytest.approx(0.75, rel=1e-3)


@pytest.mark.skipif(not glob.glob(CEREGNANO_SOIL), reason='Ceregnano SoilGrids data not downloaded')
def test_build_input_soil_on_ceregnano():
    """The downloaded Ceregnano tifs give the values printed by download_soilgrids.py."""
    from wcm_swb.inputs import build_input_soil
    clay, sand, soc, wwfc, www = build_input_soil(CEREGNANO_SOIL, CEREGNANO_SHAPE)
    assert (clay, sand, soc) == pytest.approx((25.624, 26.940, 17.110), abs=1e-3)
    assert (wwfc, www) == pytest.approx((0.3147, 0.1839), abs=1e-4)
    assert wwfc > www
