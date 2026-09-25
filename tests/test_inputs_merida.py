"""Tests for wcm_swb.inputs.build_input_MERIDA (site extraction from merged MERIDA files)."""

import numpy as np
import pandas as pd
import pytest

from wcm_swb.inputs import build_input_MERIDA

xr = pytest.importorskip('xarray')


def _write(path, var, values, times):
    lat, lon = np.array([45.0, 45.04]), np.array([11.8, 11.84])
    data = np.broadcast_to(np.asarray(values, 'float32')[:, None, None], (len(times), 2, 2)).copy()
    xr.Dataset({var: (('time', 'lat', 'lon'), data)}, coords={'time': times, 'lat': lat, 'lon': lon}).to_netcdf(path)


def test_zero_kelvin_temperature_is_interpolated(tmp_path):
    times = pd.date_range('2021-01-11 17:00', periods=5, freq='h')
    t2m = np.array([280.0, 281.0, 0.0, 283.0, 284.0])     # 19:00 missing, stored as 0 K
    rain = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    _write(tmp_path / 'prec.nc', 'tp', rain, times)
    _write(tmp_path / 'temp.nc', 't2m', t2m, times)

    df = build_input_MERIDA([str(tmp_path / 'prec.nc'), str(tmp_path / 'temp.nc')], 45.0, 11.8,
                            str(tmp_path / 'cache' / 'MERIDA_PREC_TEMP.nc'))
    assert df.loc['2021-01-11 19:00', 'temperature'] == pytest.approx(282.0 - 273.16)
    assert df['temperature_min'].min() == pytest.approx(280.0 - 273.16)
    np.testing.assert_allclose(df['rain'].values, rain)   # precipitation untouched
    assert (tmp_path / 'cache' / 'MERIDA_PREC_TEMP.nc').exists()
