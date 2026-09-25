"""Tests for the cross-orbit HIST normalization (wcm_swb.sigma0)."""

import os

import numpy as np
import pandas as pd
import pytest

from wcm_swb.sigma0 import hist_norm, orbit_statistics, select_reference_orbit, normalize_orbits

CEREGNANO_S0 = os.path.join(os.path.dirname(__file__), '..', 'study_sites', 'Ceregnano', 'sigma0',
                            's0_ceregnano_2017-23.nc')


def _synthetic(counts=None, seed=0):
    rng = np.random.default_rng(seed)
    counts = counts or {1: 500, 2: 500, 3: 500}
    offsets = {1: 0.0, 2: 2.0, 3: -3.0}
    angles = {1: 35.0, 2: 41.0, 3: 45.0}
    rows = []
    for orb, n in counts.items():
        rows.append(pd.DataFrame({
            'orb': orb,
            'angle': angles[orb] + rng.normal(0, 0.1, n),
            'VV_notnorm': -10 + offsets[orb] + rng.normal(0, 1 + orb / 2, n),
            'VH_notnorm': -18 + offsets[orb] + rng.normal(0, 1.5, n),
        }))
    return pd.concat(rows, ignore_index=True)


def test_hist_norm_maps_moments():
    x = np.array([1.0, 2.0, 3.0])
    y = hist_norm(x, x.mean(), x.std(), ref_mean=10.0, ref_std=2.0)
    assert y.mean() == pytest.approx(10.0)
    assert y.std() == pytest.approx(2.0)


def test_normalized_orbits_share_reference_moments():
    out, orb_ref, _ = normalize_orbits(_synthetic())
    assert orb_ref == 2  # mean angle closest to 40 deg
    moments = out.groupby('orb')['VV'].agg(['mean', lambda s: s.std(ddof=0)])
    np.testing.assert_allclose(moments.iloc[:, 0], moments.loc[orb_ref].iloc[0])
    np.testing.assert_allclose(moments.iloc[:, 1], moments.loc[orb_ref].iloc[1])
    np.testing.assert_allclose(out['CR'], 10 ** ((out['VH'] - out['VV']) / 10))


def test_low_count_orbit_not_eligible_as_reference():
    df = _synthetic(counts={1: 500, 2: 100, 3: 500})  # orbit 2 < half the mean count
    stats = orbit_statistics(df.rename(columns={'VV_notnorm': 'VV', 'VH_notnorm': 'VH'}))
    assert select_reference_orbit(stats) == 3


def test_zero_std_fallback():
    df = pd.DataFrame({'orb': [1, 1], 'angle': [40.0, 40.0], 'VV': [-4.0, -4.0], 'VH': [-9.0, -9.0]})
    stats = orbit_statistics(df)
    assert stats.at[1, 'VV_std'] == pytest.approx(2.0)
    assert stats.at[1, 'VH_std'] == pytest.approx(3.0)


@pytest.mark.skipif(not os.path.exists(CEREGNANO_S0), reason='Ceregnano example data not present')
def test_reproduces_ceregnano_example():
    """Regression against the shipped Ceregnano file (made with
    00_preprocessing/download_sentinel1_gee.py): re-normalizing its raw
    backscatter must give back its VV/VH/CR, with orbit 95 as reference."""
    xr = pytest.importorskip('xarray')
    ds = xr.open_dataset(CEREGNANO_S0)
    df = ds.to_dataframe().reset_index().dropna(subset=['VV_notnorm']).reset_index(drop=True)
    out, orb_ref, _ = normalize_orbits(df[['orb', 'angle', 'VV_notnorm', 'VH_notnorm']])
    assert orb_ref == 95 == ds.attrs['reference_orbit']
    for col in ['VV', 'VH', 'CR']:
        np.testing.assert_allclose(out[col], df[col], atol=1e-9)
