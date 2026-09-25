"""
WCM-SWB calibration engine.

Physics: `swb_wcm_core` runs a two-layer soil water balance coupled to the
Water Cloud Model, with `doi` (radar depth of influence) and `calc_Ks_single`
(water-stress coefficient) as helpers. Site-specific constants are passed
explicitly through `SiteConstants`; an optional crop-coefficient multiplier
`Kc0` (default 1.0) can be calibrated on request.

Calibration: `build_pymc_model` and `run_calibration_stage` build and sample
one PyMC model for a given set of free parameters. Two engines use them:

- `run_yearround_calibration` -- one stage, all parameters free;
- `run_seasonal_calibration` -- three stages (``cal_bare_1``,
  ``cal_bare_2``, ``cal_veg``) with a D_slope prior update in between, each
  freezing a different subset of parameters.

See ``docs/model_conventions.md`` for units and the freeze scheme.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""

from typing import NamedTuple

import numpy as np
from numba import njit
import pymc as pm
import arviz as az
import pytensor.tensor as T
from pytensor.compile.ops import as_op
from numpy.ma import masked_array

from wcm_swb.calibration_support import build_PAR_dict_posterior, D_slope

# ----------------------------------------------------------------------------
# Physics core
# ----------------------------------------------------------------------------

# Hallikainen et al. (1985) dielectric-mixing coefficients, by SAR frequency [GHz].
_COEFF = {
    1.4: {'real': [2.862, -0.012, 0.001, 3.803, 0.462, -0.341, 119.006, -0.500, 0.633],
          'img': [0.356, -0.003, -0.008, 5.507, 0.044, -0.002, 17.753, -0.313, 0.206]},
    4: {'real': [2.927, -0.012, -0.001, 5.505, 0.371, 0.062, 114.826, -0.389, -0.547],
        'img': [0.004, 0.001, 0.002, 0.951, 0.005, -0.01, 16.759, 0.192, 0.29]},
    6: {'real': [1.993, 0.002, 0.015, 38.086, -0.176, -0.633, 10.72, 1.256, 1.522],
        'img': [-0.123, 0.002, 0.003, 7.502, -0.058, -0.116, 2.942, 0.452, 0.543]},
}


def coeff_for_freq(freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Dielectric-mixing coefficient arrays for `doi`, at 1.4, 4, or 6 GHz."""
    return (np.array(_COEFF[freq]['real'], dtype=np.float32),
            np.array(_COEFF[freq]['img'], dtype=np.float32))


class SiteConstants(NamedTuple):
    """Per-site quantities `swb_wcm_core` needs but never calibrates."""
    coeff_r: np.ndarray
    coeff_i: np.ndarray
    freq: float
    sand: float
    clay: float
    depth1_add: float
    depth2: float
    rho_st: float
    mask_irri: np.ndarray


@njit
def doi(coeff_r: np.ndarray, coeff_i: np.ndarray, freq: float, sand: float,
        clay: float, water: float, theta: float) -> float:
    """Depth of (radar) influence [mm]. theta in radians, water in m3/m3."""
    c = 299792458.  # m/s
    real = coeff_r[0] + coeff_r[1] * sand + coeff_r[2] * clay + \
           (coeff_r[3] + coeff_r[4] * sand + coeff_r[5] * clay) * water + \
           (coeff_r[6] + coeff_r[7] * sand + coeff_r[8] * clay) * (water ** 2)
    img = coeff_i[0] + coeff_i[1] * sand + coeff_i[2] * clay + \
          (coeff_i[3] + coeff_i[4] * sand + coeff_i[5] * clay) * water + \
          (coeff_i[6] + coeff_i[7] * sand + coeff_i[8] * clay) * (water ** 2)
    depth = c / (2 * np.pi * freq * 1e9) * (np.sqrt(real) / img) * np.cos(theta)
    return depth * 1000


@njit
def calc_Ks_single(WW: float, rho: float, WW_fc: float, WW_w: float) -> float:
    """Water-stress coefficient in [0, 1]."""
    if WW >= ((1 - rho) * WW_fc + rho * WW_w):
        return 1.0
    elif WW > WW_w:
        return (WW - WW_w) / ((1 - rho) * (WW_fc - WW_w))
    else:
        return 0.0


@njit
def swb_wcm_core(inputs, ww_sat, A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start,
                  irri_thr, irri_cf, coeff_r, coeff_i, freq, sand, clay,
                  depth1_add, depth2, rho_st, mask_irri, Kc0=1.0):
    """Two-layer soil-water-balance time loop coupled to a Water Cloud Model.

    `Kc0` is a crop-coefficient multiplier on top of the NDVI-as-Kc proxy,
    defaulting to 1.0 (no effect) unless a caller opts in to calibrating it.
    Site constants (coeff_r, coeff_i, freq, sand, clay, depth1_add, depth2,
    rho_st, mask_irri) are passed explicitly; see `SiteConstants`.

    Returns (WW1, WW2, I, PERC1, PERC2, ET1, ET2, IRRI, s0).
    """
    P, ET0, Kc, veg, angle = inputs

    # WW_start*2/2 ensures WW_start is treated as a plain number by np.full
    WW1 = np.full(len(P), WW_start * 2 / 2, dtype=np.float64)
    WW2 = np.full(len(P), WW_start * 2 / 2, dtype=np.float64)
    I = np.zeros(len(P), dtype=np.float64)
    PERC1 = np.zeros(len(P), dtype=np.float64)
    PERC2 = np.zeros(len(P), dtype=np.float64)
    ET1 = np.zeros(len(P), dtype=np.float64)
    ET2 = np.zeros(len(P), dtype=np.float64)
    IRRI = np.zeros(len(P), dtype=np.float64)
    theta = np.deg2rad(angle)

    for i in range(1, len(P)):

        depth1 = doi(coeff_r, coeff_i, freq, sand, clay, WW1[i - 1], theta[i]) + depth1_add
        depth12 = depth1 + depth2

        # Evapotranspiration
        ETc = ET0[i] * Kc0 * Kc[i]

        y1 = 1 - 0.9 ** (depth1 / 10)  # weight ET by cumulative root fraction
        y2 = 1 - 0.9 ** (depth2 / 10) - y1

        ETc1 = ETc * y1
        Ks1 = calc_Ks_single(WW1[i - 1], rho_st, WW_fc, WW_w)
        ET1[i] = ETc1 * Ks1

        ETc2 = ETc * y2
        Ks2 = calc_Ks_single(WW2[i - 1], rho_st, WW_fc, WW_w)
        ET2[i] = ETc2 * Ks2

        # Irrigation
        TAW_max = (WW_fc - WW_w) * depth12
        if mask_irri[i] and P[i] < 10:  # precipitation threshold 10 mm
            TAW_thr = TAW_max * irri_thr
            TAW12 = (WW1[i - 1] - WW_w) * depth1 + (WW2[i - 1] - WW_w) * depth2
            IRRI[i] = (TAW_max - TAW12) * irri_cf if TAW12 < TAW_thr else 0
        else:
            IRRI[i] = 0

        # Infiltration (I layer only)
        I[i] = P[i] + IRRI[i]

        # Percolation I
        PERC1_base = max(0, Ksat * ((WW1[i - 1]) / ww_sat) ** (3 + 2 / lam)) if WW2[i - 1] < ww_sat else 0
        PERC1_amm = max(0, (WW1[i - 1] - WW_w) * depth1 + I[i] - ET1[i]) if WW1[i - 1] > WW_w else max(0, I[i] - ET1[i])
        PERC1[i] = min(PERC1_base, PERC1_amm)

        # Percolation II
        PERC2_base = max(0, Ksat * ((WW2[i - 1]) / ww_sat) ** (3 + 2 / lam))
        PERC2_amm = max(0, (WW2[i - 1] - WW_w) * depth2 + PERC1[i] - ET2[i]) if WW2[i - 1] > WW_w else max(0, PERC1[i] - ET2[i])
        PERC2[i] = min(PERC2_base, PERC2_amm)

        # Water balance [m3/m3]
        WW1[i] = WW1[i - 1] + (I[i] - PERC1[i] - ET1[i]) / depth1
        if WW1[i] > ww_sat:
            WW1[i] = ww_sat
        WW2[i] = WW2[i - 1] + (PERC1[i] - PERC2[i] - ET2[i]) / depth2
        if WW2[i] > ww_sat:
            WW2[i] = ww_sat

    # Water Cloud Model
    tt = np.exp((-2 * B * veg) / np.cos(theta))
    s0_veg = A * veg * np.cos(theta) * (1 - tt)
    s0_soil_db = C + D * WW1
    s0_soil = tt * 10. ** (s0_soil_db / 10)
    s0 = 10 * np.log10(s0_veg + s0_soil)
    s0[0] = s0[3]  # avoid a NaN in the first element

    return (WW1, WW2, I, PERC1, PERC2, ET1, ET2, IRRI, s0)


_OUTPUT_KEYS = ('WW1', 'WW2', 'I', 'PERC1', 'PERC2', 'ET1', 'ET2', 'IRRI', 's0')
PAR_STR = ('A', 'B', 'C', 'D', 'Ksat', 'lam', 'WW_fc', 'WW_w', 'WW_start', 'irri_thr', 'irri_cf')


# ----------------------------------------------------------------------------
# Parameter-config bookkeeping (free vs. frozen, per calibration stage)
# ----------------------------------------------------------------------------

def build_par_config(par_init: dict, parspace: dict, obs_keys=(), obs_values=None,
                      overrides: dict | None = None) -> dict:
    """Build a PAR_STR -> {range, init, obs, init_obs} dict for one calibration stage.

    `obs_keys`/`obs_values` freeze the listed parameters at fixed values
    (used by the seasonal engine's bare-soil stages); `overrides` optionally
    replaces `range`/`init` for specific keys (used by the D_slope prior
    narrowing between seasonal stages 1 and 2).
    """
    obs_values = list(obs_values) if obs_values else []
    overrides = overrides or {}
    par_config = {}
    for k in PAR_STR:
        entry = {
            'range': parspace.get(k),
            'init': par_init[k],
            'obs': k in obs_keys,
            'init_obs': obs_values[list(obs_keys).index(k)] if k in obs_keys else None,
        }
        if k in overrides:
            entry.update(overrides[k])
        par_config[k] = entry
    return par_config


# ----------------------------------------------------------------------------
# PyMC model construction
# ----------------------------------------------------------------------------

def _make_simulation_op(inputs: np.ndarray, ww_sat: float, mask: np.ndarray,
                         site: SiteConstants, kc0_fixed: float | None):
    """Closure factory for the pytensor black-box wrapper around `swb_wcm_core`.

    When `kc0_fixed` is None, Kc0 is a 12th calibrated scalar; otherwise it's
    baked into the closure and the op exposes only the 11 standard params.
    A new op is defined per model build, closing over that calibration
    stage's `inputs`/`mask`.
    """
    if kc0_fixed is None:
        @as_op(itypes=[T.dscalar] * 12, otypes=[T.dvector])
        def simulation(A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start, irri_thr, irri_cf, Kc0):
            result = swb_wcm_core(inputs, ww_sat, A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start,
                                   irri_thr, irri_cf, site.coeff_r, site.coeff_i, site.freq,
                                   site.sand, site.clay, site.depth1_add, site.depth2,
                                   site.rho_st, site.mask_irri, Kc0)
            s0 = np.asarray(result[-1], dtype=np.float64)
            return masked_array(s0, ~mask, dtype=np.float64).compressed()
    else:
        @as_op(itypes=[T.dscalar] * 11, otypes=[T.dvector])
        def simulation(A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start, irri_thr, irri_cf):
            result = swb_wcm_core(inputs, ww_sat, A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start,
                                   irri_thr, irri_cf, site.coeff_r, site.coeff_i, site.freq,
                                   site.sand, site.clay, site.depth1_add, site.depth2,
                                   site.rho_st, site.mask_irri, kc0_fixed)
            s0 = np.asarray(result[-1], dtype=np.float64)
            return masked_array(s0, ~mask, dtype=np.float64).compressed()
    return simulation


def build_pymc_model(par_config: dict, obs: np.ndarray, mask: np.ndarray, inputs: np.ndarray,
                      ww_sat: float, site: SiteConstants, calibrate_kc0: bool = False,
                      kc0_init: float = 1.0, cd_sigma: float = 5.0):
    """Build the pm.Model for one calibration stage (yearround, or one of the
    three seasonal stages). `cd_sigma` narrows the C/D priors for the
    seasonal engine's second bare-soil stage (1.0 instead of the default 5.0),
    after the D_slope prior update.

    The likelihood is Gaussian on the backscatter residuals (SIGMA cost
    function); sampling uses DEMetropolisZ.

    Returns (model, initvals) — `initvals` is ready to pass to `pm.sample`.
    """
    PAR_cal = par_config

    with pm.Model() as model:
        A = pm.TruncatedNormal('A', mu=PAR_cal['A']['init'], sigma=0.1,
                                lower=PAR_cal['A']['range'][0], upper=PAR_cal['A']['range'][1],
                                observed=PAR_cal['A']['init_obs'])
        B = pm.TruncatedNormal('B', mu=PAR_cal['B']['init'], sigma=0.1,
                                lower=PAR_cal['B']['range'][0], upper=PAR_cal['B']['range'][1],
                                observed=PAR_cal['B']['init_obs'])
        C = pm.TruncatedNormal('C', mu=PAR_cal['C']['init'], sigma=cd_sigma,
                                lower=PAR_cal['C']['range'][0], upper=PAR_cal['C']['range'][1],
                                observed=PAR_cal['C']['init_obs'])
        D = pm.TruncatedNormal('D', mu=PAR_cal['D']['init'], sigma=cd_sigma,
                                lower=PAR_cal['D']['range'][0], upper=PAR_cal['D']['range'][1],
                                observed=PAR_cal['D']['init_obs'])
        Ksat = pm.TruncatedNormal('Ksat', mu=PAR_cal['Ksat']['init'], sigma=np.sqrt(PAR_cal['Ksat']['init']),
                                   lower=PAR_cal['Ksat']['range'][0], upper=PAR_cal['Ksat']['range'][1],
                                   observed=PAR_cal['Ksat']['init_obs'])
        lam = pm.TruncatedNormal('lam', mu=PAR_cal['lam']['init'], sigma=PAR_cal['lam']['init'] / 5,
                                  lower=PAR_cal['lam']['range'][0], upper=PAR_cal['lam']['range'][1],
                                  observed=PAR_cal['lam']['init_obs'])
        WW_fc = pm.TruncatedNormal('WW_fc', mu=PAR_cal['WW_fc']['init'], sigma=0.05,
                                    lower=PAR_cal['WW_fc']['range'][0], upper=PAR_cal['WW_fc']['range'][1],
                                    observed=PAR_cal['WW_fc']['init_obs'])
        WW_w = pm.TruncatedNormal('WW_w', mu=PAR_cal['WW_w']['init'], sigma=0.05,
                                   lower=PAR_cal['WW_w']['range'][0], upper=PAR_cal['WW_w']['range'][1],
                                   observed=PAR_cal['WW_w']['init_obs'])

        # Reparametrized so WW_start is always within [WW_w, WW_fc].
        WW_start_frac = pm.Beta('WW_start_frac', alpha=2, beta=2, initval=0.5)
        WW_start = pm.Deterministic('WW_start', WW_w + WW_start_frac * (WW_fc - WW_w))

        irri_thr = pm.Uniform('irri_thr',
                              lower=PAR_cal['irri_thr']['range'][0], upper=PAR_cal['irri_thr']['range'][1],
                              observed=PAR_cal['irri_thr']['init_obs'])
        irri_cf = pm.Uniform('irri_cf',
                             lower=PAR_cal['irri_cf']['range'][0], upper=PAR_cal['irri_cf']['range'][1],
                             observed=PAR_cal['irri_cf']['init_obs'])

        if calibrate_kc0:
            Kc0 = pm.TruncatedNormal('Kc0', mu=kc0_init, sigma=0.1, lower=0.1, upper=3.0)
            simulation = _make_simulation_op(inputs, ww_sat, mask, site, kc0_fixed=None)
            s0_sim = simulation(A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start, irri_thr, irri_cf, Kc0)
        else:
            simulation = _make_simulation_op(inputs, ww_sat, mask, site, kc0_fixed=kc0_init)
            s0_sim = simulation(A, B, C, D, Ksat, lam, WW_fc, WW_w, WW_start, irri_thr, irri_cf)

        s0_obs = pm.Data('s0_obs', masked_array(obs, ~mask, dtype=np.float64).compressed())
        sigma = pm.HalfNormal('sigma', sigma=1.0, initval=1.0)
        pm.Normal('s0_calib', mu=s0_sim, sigma=sigma, observed=s0_obs)

    initvals = {k: PAR_cal[k]['init'] for k in PAR_cal if not PAR_cal[k]['obs'] and k != 'WW_start'}
    initvals['WW_start_frac'] = 0.5
    initvals['sigma'] = 1.0
    if calibrate_kc0:
        initvals['Kc0'] = kc0_init

    return model, initvals


# ----------------------------------------------------------------------------
# Posterior-uncertainty propagation (re-runs the physics for ~1000 posterior draws)
# ----------------------------------------------------------------------------

def propagate_posterior_uncertainty(trace_posterior, posterior: dict, inputs: np.ndarray,
                                     ww_sat: float, site: SiteConstants, n_total: int,
                                     calibrate_kc0: bool = False, kc0_init: float = 1.0,
                                     n_samples: int = 1000, random_seed: int = 0):
    """Re-run `swb_wcm_core` for a random subset of posterior draws to get a
    5-95% credible band on every output variable, not just point estimates.

    Returns (output_mean, output_q05, output_q95, irri_samples): the first
    three are dicts of per-timestep arrays keyed like `_OUTPUT_KEYS`;
    `irri_samples` is the full (n_samples, n_timesteps) irrigation ensemble,
    kept because irrigation is sparse in time -- quantiles of multi-day
    irrigation totals must be taken over the ensemble's window sums, not by
    summing per-timestep quantiles (which are ~0 almost everywhere).
    """
    par_keys = list(PAR_STR) + (['Kc0'] if calibrate_kc0 else [])
    n_samples = min(n_samples, n_total)
    idx = np.random.default_rng(random_seed).choice(n_total, size=n_samples, replace=False)

    par_arrays = {}
    for k in par_keys:
        if k in trace_posterior:
            par_arrays[k] = trace_posterior[k].values
        else:
            par_arrays[k] = float(posterior[k]['value'])

    def _par_vals(i):
        return [par_arrays[k] if np.isscalar(par_arrays[k]) else float(par_arrays[k][i]) for k in PAR_STR]

    def _kc0_val(i):
        if not calibrate_kc0:
            return kc0_init
        return par_arrays['Kc0'] if np.isscalar(par_arrays['Kc0']) else float(par_arrays['Kc0'][i])

    def _run(i):
        return swb_wcm_core(inputs, ww_sat, *_par_vals(i), site.coeff_r, site.coeff_i, site.freq,
                             site.sand, site.clay, site.depth1_add, site.depth2, site.rho_st,
                             site.mask_irri, _kc0_val(i))

    result0 = _run(idx[0])
    t_len = result0[0].shape[0]
    output_samples = {k: np.empty((n_samples, t_len)) for k in _OUTPUT_KEYS}
    for j, i in enumerate(idx):
        result = _run(i)
        for ki, k in enumerate(_OUTPUT_KEYS):
            output_samples[k][j] = result[ki]

    output_mean = {k: output_samples[k].mean(axis=0) for k in _OUTPUT_KEYS}
    output_q05 = {k: np.quantile(output_samples[k], 0.05, axis=0) for k in _OUTPUT_KEYS}
    output_q95 = {k: np.quantile(output_samples[k], 0.95, axis=0) for k in _OUTPUT_KEYS}
    return output_mean, output_q05, output_q95, output_samples['IRRI']


# ----------------------------------------------------------------------------
# Stage runner (shared by yearround and every seasonal stage)
# ----------------------------------------------------------------------------

def run_calibration_stage(cal_name: str, par_config: dict, obs: np.ndarray, mask: np.ndarray,
                           inputs: np.ndarray, ww_sat: float, site: SiteConstants, run_params: dict,
                           calibrate_kc0: bool = False, kc0_init: float = 1.0, cd_sigma: float = 5.0,
                           random_seed: int = 0):
    """Build, sample, and summarize one calibration stage.

    Returns (trace, posterior, output_mean, output_q05, output_q95, irri_samples).
    """
    model, initvals = build_pymc_model(par_config, obs, mask, inputs, ww_sat, site,
                                        calibrate_kc0=calibrate_kc0, kc0_init=kc0_init, cd_sigma=cd_sigma)
    with model:
        step = pm.DEMetropolisZ()
        trace = pm.sample(draws=run_params['n_draws'], tune=run_params['n_tunes'],
                           chains=run_params['n_chains'], cores=run_params['n_cores'], step=step,
                           discard_tuned_samples=True, progressbar=True, random_seed=random_seed,
                           initvals=initvals)
        trace.extend(pm.sample_prior_predictive(samples=run_params['n_draws'], model=model, random_seed=random_seed))
        trace.extend(pm.sample_posterior_predictive(trace, random_seed=random_seed, progressbar=True))

    trace_posterior = az.extract(trace, group='posterior', combined=True)
    posterior = build_PAR_dict_posterior(cal_name, {cal_name: par_config}, {}, trace_posterior)[cal_name]

    n_total = run_params['n_draws'] * run_params['n_chains']
    output_mean, output_q05, output_q95, irri_samples = propagate_posterior_uncertainty(
        trace_posterior, posterior, inputs, ww_sat, site, n_total,
        calibrate_kc0=calibrate_kc0, kc0_init=kc0_init, random_seed=random_seed)

    return trace, posterior, output_mean, output_q05, output_q95, irri_samples


# ----------------------------------------------------------------------------
# Orchestration: yearround (1 stage) vs. seasonal (3 stages)
# ----------------------------------------------------------------------------

def run_yearround_calibration(inputs: np.ndarray, obs: np.ndarray, mask: np.ndarray, ww_sat: float,
                               site: SiteConstants, par_init: dict, parspace: dict, run_params: dict,
                               calibrate_kc0: bool = False, kc0_init: float = 1.0, random_seed: int = 0):
    """Single-stage calibration over the full time series; nothing frozen."""
    par_config = build_par_config(par_init, parspace)
    return {'cal_yearround': run_calibration_stage(
        'cal_yearround', par_config, obs, mask, inputs, ww_sat, site, run_params,
        calibrate_kc0=calibrate_kc0, kc0_init=kc0_init, random_seed=random_seed)}


def run_seasonal_calibration(inputs: np.ndarray, obs_full: np.ndarray, veg: np.ndarray, veg_thr: float,
                              t, ww_sat: float, site: SiteConstants, par_init: dict, parspace: dict,
                              run_params: dict, scaling_svgfil: int, calibrate_kc0: bool = False,
                              kc0_init: float = 1.0, random_seed: int = 0, d_slope_n_min: int = 50):
    """Three-stage sequential calibration:

    1. `cal_bare_1` — freeze A, B, irri_cf, irri_thr at 0 (no vegetation/irrigation
       signal expected on bare soil); fit the 7 remaining soil/backscatter
       parameters on NDVI-below-threshold observations.
    2. `cal_bare_2` — same freeze scheme, but first narrow the C/D priors via
       `D_slope` change-detection using stage 1's posterior-mean WW1.
    3. `cal_veg` — freeze C, D, Ksat, lam, WW_fc, WW_w, WW_start at stage 2's
       posterior values; fit A, B, irri_cf, irri_thr on NDVI-above-threshold
       observations.
    """
    obs_bare = np.where(veg < veg_thr, obs_full, np.nan)
    mask_bare = np.isfinite(obs_bare)
    bare_obs_keys = ['A', 'B', 'irri_cf', 'irri_thr']

    par1 = build_par_config(par_init, parspace, obs_keys=bare_obs_keys, obs_values=[0, 0, 0, 0])
    trace1, post1, out_mean1, out_q05_1, out_q95_1, irri_samples1 = run_calibration_stage(
        'cal_bare_1', par1, obs_bare, mask_bare, inputs, ww_sat, site, run_params,
        calibrate_kc0, kc0_init, random_seed=random_seed)

    param_0 = {
        'C': {'r': post1['C']['range'], 'm': post1['C']['value']},
        'D': {'r': post1['D']['range'], 'm': post1['D']['value']},
    }
    dd = np.array(t, dtype='datetime64[m]')
    param_update = D_slope(param_0, dd, out_mean1['WW1'], veg, veg_thr, obs_full, d_slope_n_min, scaling_svgfil)

    par2 = build_par_config(
        par_init, parspace, obs_keys=bare_obs_keys, obs_values=[0, 0, 0, 0],
        overrides={'C': {'range': param_update['C']['r'], 'init': param_update['C']['m']},
                   'D': {'range': param_update['D']['r'], 'init': param_update['D']['m']}})
    trace2, post2, out_mean2, out_q05_2, out_q95_2, irri_samples2 = run_calibration_stage(
        'cal_bare_2', par2, obs_bare, mask_bare, inputs, ww_sat, site, run_params,
        calibrate_kc0, kc0_init, cd_sigma=1.0, random_seed=random_seed)

    obs_veg = np.where(veg >= veg_thr, obs_full, np.nan)
    mask_veg_obs = np.isfinite(obs_veg)
    frozen_keys = ['C', 'D', 'Ksat', 'lam', 'WW_fc', 'WW_w', 'WW_start']
    par3 = build_par_config(par_init, parspace, obs_keys=frozen_keys,
                             obs_values=[post2[k]['value'] for k in frozen_keys])
    trace3, post3, out_mean3, out_q05_3, out_q95_3, irri_samples3 = run_calibration_stage(
        'cal_veg', par3, obs_veg, mask_veg_obs, inputs, ww_sat, site, run_params,
        calibrate_kc0, kc0_init, random_seed=random_seed)

    return {
        'cal_bare_1': (trace1, post1, out_mean1, out_q05_1, out_q95_1, irri_samples1),
        'cal_bare_2': (trace2, post2, out_mean2, out_q05_2, out_q95_2, irri_samples2),
        'cal_veg': (trace3, post3, out_mean3, out_q05_3, out_q95_3, irri_samples3),
    }
