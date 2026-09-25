"""Tests for the 02_analysis aggregation helpers (wcm_swb.analysis)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wcm_swb import analysis


def _write_stage(root, run, field, stage, irri_daily, obs='VV'):
    """Minimal fake 01_calibration stage folder: output, config, irrigation samples."""
    folder = root / run / field / stage
    folder.mkdir(parents=True)
    suffix = f'_{field}_{obs}_{stage}'
    t = pd.date_range('2018-01-01', periods=48, freq='h')
    s0_obs = np.linspace(-15, -10, len(t))
    xr.Dataset(
        {f'Sigma0_{obs}': ('Datetime', s0_obs + 0.5), f'Sigma0_{obs}_obs': ('Datetime', s0_obs),
         'Sigma0_mask': ('Datetime', np.ones(len(t), dtype=bool))},
        coords={'Datetime': t},
    ).to_netcdf(folder / f'output{suffix}.nc')
    posterior = {'A': {'range': [0, 5], 'init': 0.2, 'ref': 'median', 'value': 1.5, 'std': 0.1},
                 'C': {'range': [-30, 0], 'init': -15, 'ref': 'median', 'value': -12.0, 'std': 0.0}}
    (folder / f'configuration{suffix}.json').write_text(json.dumps({'PAR_dict_posterior': {stage: posterior}}))
    days = pd.date_range('2018-01-01', periods=irri_daily.shape[0], freq='D')
    xr.Dataset({'Irrigation': (('Datetime', 'draw'), irri_daily.astype(np.float32))},
               coords={'Datetime': days, 'draw': np.arange(irri_daily.shape[1])},
               ).to_netcdf(folder / f'irrigation_samples{suffix}.nc')


def test_find_outputs_and_final_stages(tmp_path):
    irri = np.zeros((10, 4))
    _write_stage(tmp_path, 'yearround', 'Id_2', 'cal_yearround', irri)
    _write_stage(tmp_path, 'seasonal', 'Id_2', 'cal_bare_1', irri)
    _write_stage(tmp_path, 'seasonal', 'Id_2', 'cal_veg', irri)

    outputs = analysis.find_outputs(str(tmp_path))
    assert len(outputs) == 3
    assert outputs.irrigation_samples.notna().all()
    finals = analysis.final_stage_outputs(outputs)
    assert sorted(finals.stage) == ['cal_veg', 'cal_yearround']
    assert len(analysis.find_outputs(str(tmp_path), 'yearround*')) == 1


def test_irrigation_quantiles_use_window_sums_not_summed_quantiles():
    # Sparse irrigation: each draw irrigates 10 mm on exactly one (different) day.
    # Per-day quantiles are all 0 (1% of draws irrigate on any given day), but
    # every draw's 100-day total is 10 mm.
    n_days = n_draws = 100
    ens = pd.DataFrame(np.eye(n_days, n_draws) * 10.0, index=pd.date_range('2018-06-01', periods=n_days))
    daily = analysis.irrigation_quantiles(ens, '1D')
    assert (daily.q95 == 0).all()
    assert daily.q95.sum() == 0 and daily['mean'].sum() == pytest.approx(10)
    total = analysis.irrigation_quantiles(ens, '100D')
    assert len(total) == 1
    assert total.iloc[0][['mean', 'median', 'q05', 'q95']].tolist() == pytest.approx([10, 10, 10, 10])


def test_irrigation_quantiles_clips_to_season():
    ens = pd.DataFrame(np.ones((365, 3)), index=pd.date_range('2018-01-01', periods=365))
    start, end = analysis.irrigation_season(2018, 90, 270)
    assert start == pd.Timestamp('2018-03-31') and end.date() == pd.Timestamp('2018-09-27').date()
    total = analysis.irrigation_quantiles(ens, 'YS', start, end)
    assert total.iloc[0]['mean'] == pytest.approx(181)


def test_area_weighted_ensemble(tmp_path):
    _write_stage(tmp_path, 'r', 'a', 'cal_yearround', np.full((5, 4), 2.0))
    _write_stage(tmp_path, 'r', 'b', 'cal_yearround', np.full((5, 3), 8.0))
    outputs = analysis.find_outputs(str(tmp_path))
    paths = dict(zip(outputs.field, outputs.irrigation_samples))
    ens = analysis.area_weighted_ensemble(paths, {'a': 3.0, 'b': 1.0})
    assert ens.shape == (5, 3)  # cut to the smallest ensemble
    assert np.allclose(ens.values, (2.0 * 3 + 8.0 * 1) / 4)


def test_fit_metrics_perfect_and_biased():
    obs = np.array([-15.0, -12.0, np.nan, -10.0, -13.0])
    perfect = analysis.fit_metrics(obs, obs)
    assert perfect['n'] == 4
    assert perfect['rmse'] == pytest.approx(0) and perfect['kge'] == pytest.approx(1)
    shifted = analysis.fit_metrics(obs + 1.0, obs)
    assert shifted['bias'] == pytest.approx(1) and shifted['rmse'] == pytest.approx(1)
    assert shifted['ubrmsd'] == pytest.approx(0) and shifted['r'] == pytest.approx(1)


def test_sigma0_metrics_and_posteriors(tmp_path):
    _write_stage(tmp_path, 'yearround', 'Id_1', 'cal_yearround', np.zeros((3, 2)))
    outputs = analysis.find_outputs(str(tmp_path))
    metrics = analysis.sigma0_metrics(outputs)
    assert metrics.iloc[0]['bias'] == pytest.approx(0.5)
    params = analysis.read_posteriors(outputs).set_index('param')
    assert params.loc['A', 'value'] == 1.5 and bool(params.loc['A', 'calibrated'])
    assert not bool(params.loc['C', 'calibrated'])


def test_natural_key():
    assert sorted(['Id_10', 'Id_2', 'Id_1'], key=analysis.natural_key) == ['Id_1', 'Id_2', 'Id_10']
