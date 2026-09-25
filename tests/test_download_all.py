"""Tests for the 00_preprocessing/download_all.py launcher (expected outputs, preflight)."""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '00_preprocessing'))
import download_all as da  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATE = os.path.join(REPO, '00_preprocessing', 'configuration_00_preprocessing_TEMPLATE.json')


def _settings(root):
    with open(TEMPLATE) as fh:
        s = json.load(fh)
    s['paths']['root'] = root
    return s


def _merida(root, years=(2021,), merge=True):
    return {
        'options': {'years': list(years), 'months': [1, 2], 'variables': {'PREC': 'PREC', 'T2': 'TEMP'},
                    'opt_merge': merge, 'opt_overwrite': False},
        'server': {'user_env': 'MERIDA_USER_TEST', 'password_env': 'MERIDA_PASSWORD_TEST'},
        'paths': {'root': root, 'data_output': {
            'folder_monthly': 'merida/monthly/', 'filename_monthly': 'MERIDA_HRES_{variable}_{yyyymm}.nc',
            'folder_merged': 'merida/', 'filename_merged': 'MERIDA_{label}_{start}-{end}.nc'}},
    }


CEREGNANO = os.path.join(REPO, 'study_sites', 'Ceregnano')


def test_expected_outputs_ceregnano():
    s = _settings(CEREGNANO)
    fields, _ = da._fields(s)
    assert fields == ['ceregnano']
    for fn in (da.expected_soilgrids, da.expected_sigma0, da.expected_ndvi):
        assert fn(s, fields), fn.__name__
    assert len(da.expected_soilgrids(s, fields)) == 20


@pytest.mark.skipif(not os.path.isdir(os.path.join(CEREGNANO, 'sigma0')), reason='Ceregnano data not downloaded')
def test_expected_outputs_match_downloaded_ceregnano():
    s = _settings(CEREGNANO)
    fields, _ = da._fields(s)
    for fn in (da.expected_soilgrids, da.expected_sigma0, da.expected_ndvi):
        assert all(os.path.exists(f) for f in fn(s, fields)), fn.__name__


def test_expected_merida_paths(tmp_path):
    root = str(tmp_path)
    assert da.expected_merida(_merida(root)) == [
        os.path.join(root, 'merida/', 'MERIDA_PREC_202101-202102.nc'),
        os.path.join(root, 'merida/', 'MERIDA_TEMP_202101-202102.nc')]
    monthly = da.expected_merida(_merida(root, merge=False))
    assert len(monthly) == 4 and monthly[0].endswith('merida/monthly/MERIDA_HRES_PREC_202101.nc')


def test_preflight(tmp_path, monkeypatch):
    root = str(tmp_path)
    s = _settings(root)
    s['options']['gee_project'] = '${GEE_PROJECT_UNSET_FOR_TEST}'
    monkeypatch.delenv('MERIDA_USER_TEST', raising=False)
    problems = da.preflight(['sigma0', 'merida'], s, _merida(root + '/other_site'), [], 'x/*.geojson')
    text = '\n'.join(problems)
    assert 'no field shapefiles' in text
    assert 'gee_project is not set' in text
    assert 'MERIDA_USER_TEST is not set' in text
    assert 'is not the site' in text

    monkeypatch.setenv('GEE_PROJECT_UNSET_FOR_TEST', 'my-project')
    monkeypatch.setenv('MERIDA_USER_TEST', 'u')
    monkeypatch.setenv('MERIDA_PASSWORD_TEST', 'p')
    problems = da.preflight(['sigma0', 'soilgrids'], s, None, ['f'], 'x')
    assert problems == []


def test_site_lock(tmp_path):
    lock = da.acquire_lock(str(tmp_path))
    assert lock and open(lock).read() == str(os.getpid())
    assert da.acquire_lock(str(tmp_path)) is None      # held by a live process (us)
    with open(lock, 'w') as fh:
        fh.write('999999999')                           # no such pid: stale
    lock2 = da.acquire_lock(str(tmp_path))
    assert lock2 == lock and open(lock).read() == str(os.getpid())
    os.remove(lock)
