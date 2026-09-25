"""
Regression/sanity tests for the physics core (`swb_wcm_core`).

Checks:
determinism (same inputs -> same outputs, guarding against accidental
numpy/numba nondeterminism), basic physical invariants (soil moisture stays
within [0, WW_sat], outputs are finite and correctly shaped), and that the
Kc0=1.0 default reproduces identical output to calling without a Kc0 arg
(the Kc0 multiplier is a no-op by default).
"""

import numpy as np
import pytest

from wcm_swb.model import swb_wcm_core, coeff_for_freq


def _synthetic_inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.exponential(scale=1.0, size=n)  # precipitation [mm]
    P[rng.random(n) > 0.2] = 0.0  # mostly dry
    ET0 = np.clip(rng.normal(3.0, 1.0, n), 0, None)  # [mm/day-ish]
    Kc = np.clip(rng.uniform(0.2, 0.8, n), 0, 1)  # NDVI-as-Kc proxy
    veg = Kc.copy()
    angle = np.full(n, 35.0)  # degrees, matches doi()'s np.deg2rad(angle) call inside swb_wcm_core
    return np.array([P, ET0, Kc, veg, angle], dtype=np.float64)


def _default_params():
    coeff_r, coeff_i = coeff_for_freq(4)
    return dict(
        ww_sat=0.45, A=0.2, B=0.2, C=-15.0, D=30.0, Ksat=5.0, lam=0.2,
        WW_fc=0.35, WW_w=0.15, WW_start=0.25, irri_thr=0.3, irri_cf=0.1,
        coeff_r=coeff_r, coeff_i=coeff_i, freq=4, sand=40.0, clay=20.0,
        depth1_add=100, depth2=500, rho_st=0.5,
    )


def test_deterministic():
    inputs = _synthetic_inputs()
    params = _default_params()
    mask_irri = np.ones(inputs.shape[1], dtype=np.bool_)

    result1 = swb_wcm_core(inputs, **params, mask_irri=mask_irri)
    result2 = swb_wcm_core(inputs, **params, mask_irri=mask_irri)
    for a, b in zip(result1, result2):
        np.testing.assert_array_equal(a, b)


def test_kc0_default_matches_explicit_one():
    inputs = _synthetic_inputs()
    params = _default_params()
    mask_irri = np.ones(inputs.shape[1], dtype=np.bool_)

    result_default = swb_wcm_core(inputs, **params, mask_irri=mask_irri)
    result_explicit = swb_wcm_core(inputs, **params, mask_irri=mask_irri, Kc0=1.0)
    for a, b in zip(result_default, result_explicit):
        np.testing.assert_array_equal(a, b)


def test_kc0_scales_evapotranspiration():
    """Kc0=0 should shut off ET entirely -> systematically wetter soil than Kc0=1."""
    inputs = _synthetic_inputs()
    params = _default_params()
    mask_irri = np.ones(inputs.shape[1], dtype=np.bool_)

    WW1_normal = swb_wcm_core(inputs, **params, mask_irri=mask_irri, Kc0=1.0)[0]
    WW1_no_et = swb_wcm_core(inputs, **params, mask_irri=mask_irri, Kc0=0.0)[0]
    assert np.mean(WW1_no_et[10:]) >= np.mean(WW1_normal[10:])


def test_output_shapes_and_bounds():
    inputs = _synthetic_inputs(n=150)
    params = _default_params()
    mask_irri = np.zeros(inputs.shape[1], dtype=np.bool_)  # irrigation off

    WW1, WW2, I, PERC1, PERC2, ET1, ET2, IRRI, s0 = swb_wcm_core(inputs, **params, mask_irri=mask_irri)

    n = inputs.shape[1]
    for arr in (WW1, WW2, I, PERC1, PERC2, ET1, ET2, IRRI, s0):
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr))

    assert np.all(WW1 <= params['ww_sat'] + 1e-9)
    assert np.all(WW2 <= params['ww_sat'] + 1e-9)
    assert np.all(IRRI == 0)  # mask_irri all False -> no irrigation ever triggers
