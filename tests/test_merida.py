"""Tests for 00_preprocessing/download_merida.py (no network: curl is exercised on file:// URLs)."""

import os
import sys
import shutil
import importlib.util

import numpy as np
import pandas as pd
import xarray as xr
import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', '00_preprocessing', 'download_merida.py')
_spec = importlib.util.spec_from_file_location('download_merida', _SCRIPT)
download_merida = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_merida)


def _write_month(path, start, periods, value, var='tp'):
    time = pd.date_range(start, periods=periods, freq='h')
    lat = np.array([45.1, 45.0], dtype='float32')
    lon = np.array([11.7, 11.8, 11.9], dtype='float32')
    data = np.full((periods, lat.size, lon.size), value, dtype='float32')
    ds = xr.Dataset({var: (('time', 'lat', 'lon'), data, {'units': 'kg m**-2'})},
                    coords={'time': time, 'lat': lat, 'lon': lon}, attrs={'title': 'MERIDA test'})
    ds.to_netcdf(path)
    return ds


def _settings(tmp_path, url='file://'):
    return {
        'options': {'years': [2023], 'months': [1, 2], 'variables': {'PREC': 'PREC'},
                    'opt_merge': True, 'opt_overwrite': False},
        'server': {'url': url, 'remote_path': str(tmp_path / 'remote' / '{yyyymm}' / '{variable}' /
                                                  'MERIDA_HRES_{variable}_{yyyymm}.nc'),
                   'user_env': 'MERIDA_TEST_USER', 'password_env': 'MERIDA_TEST_PASSWORD', 'opt_insecure': False},
        'paths': {'root': str(tmp_path), 'data_output': {
            'folder_monthly': str(tmp_path / 'monthly'), 'filename_monthly': 'MERIDA_HRES_{variable}_{yyyymm}.nc',
            'folder_merged': str(tmp_path / 'merged'), 'filename_merged': 'MERIDA_{label}_{start}-{end}.nc'}},
    }


def test_load_config_resolves_folders_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv('WCM_DATA_ROOT', '/data')
    template = os.path.join(os.path.dirname(_SCRIPT), 'configuration_00_preprocessing_MERIDA_TEMPLATE.json')
    out = download_merida.load_config(template)['paths']['data_output']
    assert out['folder_monthly'] == '/data/study_sites/Ceregnano/merida/monthly/'
    assert out['folder_merged'] == '/data/study_sites/Ceregnano/merida/'


def test_list_months_sorted_product():
    assert download_merida.list_months([2019, 2018], [12, 3]) == ['201803', '201812', '201903', '201912']


def test_list_months_rejects_bad_month():
    with pytest.raises(ValueError):
        download_merida.list_months([2018], [13])


def test_merge_sorts_and_drops_overlap(tmp_path):
    # given out of order, with a 2-hour overlap between the months
    _write_month(tmp_path / 'feb.nc', '2023-02-01T01:00', 5, 2.0)
    _write_month(tmp_path / 'jan.nc', '2023-01-31T21:00', 6, 1.0)
    out = tmp_path / 'merged.nc'
    download_merida.merge_variable([str(tmp_path / 'feb.nc'), str(tmp_path / 'jan.nc')], str(out))

    with xr.open_dataset(out) as ds:
        expected_time = pd.date_range('2023-01-31T21:00', '2023-02-01T05:00', freq='h')
        np.testing.assert_array_equal(ds['time'].values, expected_time.values)
        np.testing.assert_array_equal(ds['tp'].values[:, 0, 0], [1.0] * 6 + [2.0] * 3)
        assert ds['tp'].attrs['units'] == 'kg m**-2'
        assert ds.attrs['title'] == 'MERIDA test'
        assert ds['lat'].dtype == np.float32


def test_merge_rounds_float_days_time_jitter(tmp_path):
    # MERIDA encodes time as float64 days since the month start; 1.5416666... days
    # decodes to 12:59:59.999999999, which would break PREC/T2 alignment downstream
    ds = _write_month(tmp_path / 'raw.nc', '2023-06-01', 48, 1.0, var='t2m')
    days = np.arange(48) / 24
    days[37] = 1.5416666666666665   # as stored in MERIDA_T2_202306.nc (1 ulp below 37/24)
    ds.assign_coords(time=('time', days, {'units': 'days since 2023-6-1 00:00:00',
                                          'calendar': 'proleptic_gregorian'})).to_netcdf(tmp_path / 'jan.nc')
    with xr.open_dataset(tmp_path / 'jan.nc') as raw:
        assert not (raw['time'].values == pd.DatetimeIndex(raw['time'].values).round('s').values).all()
    download_merida.merge_variable([str(tmp_path / 'jan.nc')], str(tmp_path / 'm.nc'))
    with xr.open_dataset(tmp_path / 'm.nc') as merged:
        np.testing.assert_array_equal(merged['time'].values, pd.date_range('2023-06-01', periods=48, freq='h').values)


def test_merge_renames_hres_xtime(tmp_path):
    ds = _write_month(tmp_path / 'raw.nc', '2021-01-01T01:00', 5, 3.0)
    xtime = ds.rename({'time': 'XTIME'})
    xtime['XTIME'].attrs.update(standard_name='time', axis='T')
    xtime.to_netcdf(tmp_path / 'hres.nc')
    download_merida.merge_variable([str(tmp_path / 'hres.nc')], str(tmp_path / 'm.nc'))
    with xr.open_dataset(tmp_path / 'm.nc') as merged:
        assert merged['tp'].dims == ('time', 'lat', 'lon')
        np.testing.assert_array_equal(merged['time'].values, ds['time'].values)


def test_merge_decodes_cdo_absolute_times(tmp_path):
    # MERIDA HRES 2021: time variable 'Times', units 'day as %Y%m%d.%f' (not decoded by xarray)
    ds = _write_month(tmp_path / 'raw.nc', '2021-01-01T01:00', 30, 4.0)
    absolute = np.array([20210101 + h / 24 if h < 24 else 20210102 + (h - 24) / 24 for h in range(1, 31)])
    ds = ds.rename({'time': 'Times'}).assign_coords(Times=('Times', absolute, {
        'standard_name': 'time', 'units': 'day as %Y%m%d.%f', 'calendar': 'proleptic_gregorian', 'axis': 'T'}))
    ds.to_netcdf(tmp_path / 'hres.nc')
    download_merida.merge_variable([str(tmp_path / 'hres.nc')], str(tmp_path / 'm.nc'))
    with xr.open_dataset(tmp_path / 'm.nc') as merged:
        np.testing.assert_array_equal(merged['time'].values,
                                      pd.date_range('2021-01-01T01:00', periods=30, freq='h').values)


def test_merge_leaves_concurrent_runs_temp_file_alone(tmp_path):
    # another process merging the same output writes <out>.<its pid>.part: this run must not touch it
    _write_month(tmp_path / 'jan.nc', '2023-01-01', 3, 1.0)
    out = tmp_path / 'merged.nc'
    other = tmp_path / f'merged.nc.{os.getpid() + 1}.part'
    other.write_bytes(b'in progress')
    download_merida.merge_variable([str(tmp_path / 'jan.nc')], str(out))
    assert other.read_bytes() == b'in progress'
    with xr.open_dataset(out) as ds:
        assert ds.sizes['time'] == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ['jan.nc', 'merged.nc', other.name]


def test_merge_refuses_existing_own_temp_file(tmp_path):
    _write_month(tmp_path / 'jan.nc', '2023-01-01', 3, 1.0)
    stale = tmp_path / f'merged.nc.{os.getpid()}.part'
    stale.write_bytes(b'stale')
    with pytest.raises(FileExistsError):
        download_merida.merge_variable([str(tmp_path / 'jan.nc')], str(tmp_path / 'merged.nc'))
    assert stale.read_bytes() == b'stale'


def test_merge_rejects_different_grid(tmp_path):
    _write_month(tmp_path / 'a.nc', '2023-01-01', 3, 1.0)
    ds = _write_month(tmp_path / 'b.nc', '2023-01-02', 3, 1.0)
    ds.assign_coords(lon=ds['lon'] + 1).to_netcdf(tmp_path / 'c.nc')
    with pytest.raises(ValueError):
        download_merida.merge_variable([str(tmp_path / 'a.nc'), str(tmp_path / 'c.nc')], str(tmp_path / 'm.nc'))


@pytest.mark.skipif(shutil.which('curl') is None, reason='curl not installed')
def test_download_and_merge_via_curl(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    for yyyymm, start in (('202301', '2023-01-01T01:00'), ('202302', '2023-02-01T01:00')):
        folder = tmp_path / 'remote' / yyyymm / 'PREC'
        folder.mkdir(parents=True)
        _write_month(folder / f'MERIDA_HRES_PREC_{yyyymm}.nc', start, 4, float(yyyymm[-1]))

    with pytest.raises(ValueError, match='MERIDA_TEST_USER'):
        download_merida.download_all(settings, ['202301'])

    monkeypatch.setenv('MERIDA_TEST_USER', 'someone')
    monkeypatch.setenv('MERIDA_TEST_PASSWORD', 'p"a:ss')
    months = ['202301', '202302', '202303']   # 202303 does not exist remotely
    failed = download_merida.download_all(settings, months)
    assert failed == [('PREC', '202303')]
    assert sorted(os.listdir(settings['paths']['data_output']['folder_monthly'])) == [
        'MERIDA_HRES_PREC_202301.nc', 'MERIDA_HRES_PREC_202302.nc']   # no leftover .part

    download_merida.merge_all(settings, months)          # a month is missing -> not merged
    assert not os.path.exists(settings['paths']['data_output']['folder_merged']) or not os.listdir(settings['paths']['data_output']['folder_merged'])
    download_merida.merge_all(settings, months[:2])
    with xr.open_dataset(tmp_path / 'merged' / 'MERIDA_PREC_202301-202302.nc') as ds:
        assert ds.sizes['time'] == 8
        np.testing.assert_array_equal(ds['tp'].values[:, 1, 2], [1.0] * 4 + [2.0] * 4)
