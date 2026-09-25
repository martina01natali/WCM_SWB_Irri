"""Tests for config loading/substitution and the calibration-stage parameter bookkeeping."""

import json

import pytest

from wcm_swb.config import get_data_settings, substitute_keywords
from wcm_swb.model import build_par_config, PAR_STR


def test_get_data_settings_roundtrip(tmp_path):
    payload = {"options": {"opt_field": "ceregnano"}}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(payload))
    assert get_data_settings(str(f)) == payload


def test_get_data_settings_missing_file():
    with pytest.raises(IOError):
        get_data_settings("/nonexistent/path/config.json")


def test_substitute_keywords():
    template = "{root_plot}{filename_output}_{opt_field}"
    result = substitute_keywords(template, root_plot="/out/", filename_output="trace", opt_field="ceregnano")
    assert result == "/out/trace_ceregnano"


def test_build_par_config_no_freeze():
    par_init = {k: 1.0 for k in PAR_STR}
    parspace = {k: [0.0, 2.0] for k in PAR_STR}
    cfg = build_par_config(par_init, parspace)
    assert all(cfg[k]['obs'] is False for k in PAR_STR)
    assert all(cfg[k]['init_obs'] is None for k in PAR_STR)


def test_build_par_config_freeze_and_override():
    par_init = {k: 1.0 for k in PAR_STR}
    parspace = {k: [0.0, 2.0] for k in PAR_STR}
    cfg = build_par_config(
        par_init, parspace, obs_keys=['A', 'B'], obs_values=[0.0, 0.5],
        overrides={'C': {'range': [-5.0, 5.0], 'init': 2.0}},
    )
    assert cfg['A']['obs'] is True and cfg['A']['init_obs'] == 0.0
    assert cfg['B']['obs'] is True and cfg['B']['init_obs'] == 0.5
    assert cfg['C']['obs'] is False
    assert cfg['C']['range'] == [-5.0, 5.0]
    assert cfg['C']['init'] == 2.0
    assert cfg['D']['obs'] is False and cfg['D']['range'] == [0.0, 2.0]


def test_covers_period():
    import importlib.util
    import os
    import pandas as pd
    path = os.path.join(os.path.dirname(__file__), '..', '01_calibration', 'run_calibration.py')
    spec = importlib.util.spec_from_file_location('run_calibration', path)
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    year_2021 = pd.date_range('2021-01-01', '2021-12-31', freq='h')
    merida_2021 = pd.date_range('2021-01-01 01:00', '2021-12-31 23:00', freq='h')
    assert rc.covers_period(merida_2021, year_2021)
    assert not rc.covers_period(merida_2021, pd.date_range('2018-01-01', '2018-12-31', freq='h'))
    assert not rc.covers_period(pd.DatetimeIndex([]), year_2021)
