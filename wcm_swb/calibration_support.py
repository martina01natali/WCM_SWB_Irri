"""
Posterior-analysis helpers for the calibration engine: mode/HDI estimation,
flat-vs-peaked distribution classification, the posterior-parameter-dict
builder, and the seasonal engine's D_slope change-detection prior update.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""

import numpy as np
from scipy import stats


def mode_kde(values):
    """KDE-based mode of a (possibly multimodal) parameter posterior.

    Returns (mode, error), where error is approximately the peak width.
    """
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(values)
    x = np.linspace(min(values), max(values), 1000)
    kde_values = kde(x)

    peaks, props = find_peaks(kde_values, height=(None, None), width=(None, None))
    peak_h, peak_x = kde_values[peaks], x[peaks]

    peak_left = x[[int(num) for num in props['left_ips']]]
    peak_right = x[[int(num) for num in props['right_ips']]]
    peak_w = peak_right - peak_left

    mode_h = peak_h.max()
    mode_x = peak_x[list(peak_h).index(mode_h)]
    mode_err = peak_w[list(peak_h).index(mode_h)]

    return mode_x, mode_err


def hdi(values, credible_interval=0.95):
    """Highest density interval for a given credible interval."""
    sorted_samples = np.sort(values)
    ci_index = int(np.floor(credible_interval * len(sorted_samples)))
    n_intervals = len(sorted_samples) - ci_index
    intervals = [(sorted_samples[i], sorted_samples[i + ci_index]) for i in range(n_intervals)]
    widths = [interval[1] - interval[0] for interval in intervals]
    min_width_index = np.argmin(widths)
    return intervals[min_width_index]


def flatness_estimation(values, param_range, flatness_kurtosis_threshold=-0.5,
                         flatness_entropy_ratio=0.9, flatness_hdi_ratio=0.5):
    """Classify a posterior marginal as 'flat' (report median) or 'peaked'
    (report mode), using kurtosis, histogram entropy, and HDI-width heuristics.
    """
    kurtosis = stats.kurtosis(values)

    hist, bin_edges = np.histogram(values, bins=50, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    entropy = -np.sum(hist * np.log(hist + 1e-10)) * bin_width
    max_entropy = np.log(param_range + 1e-10)
    entropy_ratio = entropy / max_entropy

    hdi_95 = hdi(values, 0.95)
    hdi_range = hdi_95[1] - hdi_95[0]
    hdi_range_ratio = hdi_range / param_range

    if kurtosis < flatness_kurtosis_threshold or hdi_range_ratio > flatness_hdi_ratio or entropy_ratio > flatness_entropy_ratio:
        shape = "flat"
        point_estimate = np.median(values)
    else:
        shape = "peaked"
        point_estimate = stats.mode(values, keepdims=True).mode[0]

    return {
        "point_estimate": point_estimate,
        "hdi_95": hdi_95,
        "shape": shape,
        "kurtosis": kurtosis,
        "entropy": entropy,
        "entropy_ratio": entropy_ratio[0],
        "hdi_range_ratio": hdi_range_ratio[0],
    }


def build_PAR_dict_posterior(cal_name, PAR_dict_prior, PAR_dict_posterior, trace_posterior):
    """Reduce a stage's posterior trace to one point estimate + spread per
    parameter, picking median (flat posteriors) or KDE mode (peaked ones).
    Frozen/observed parameters pass their fixed value through unchanged.
    """
    for k in PAR_dict_prior[cal_name]:
        if cal_name not in PAR_dict_posterior:
            PAR_dict_posterior[cal_name] = {}
        if k in trace_posterior:
            values = trace_posterior[k].values
            param_range = abs(np.diff(PAR_dict_prior[cal_name][k]['range']))

            shape = flatness_estimation(values, param_range)['shape']
            if shape == 'flat':
                ref = 'median'
                param_value = np.nanmedian(values)
                param_err = np.nanstd(values)
            else:
                ref = 'mode'
                param_value = mode_kde(values)[0]
                param_err = mode_kde(values)[1] / 2
        else:
            ref = 'median'
            param_value = PAR_dict_prior[cal_name][k]['init_obs']
            param_err = 0.0
        PAR_dict_posterior[cal_name][k] = {
            'range': PAR_dict_prior[cal_name][k]['range'],
            'init': PAR_dict_prior[cal_name][k]['init'],
            'ref': ref,
            'value': param_value,
            'std': param_err,
        }
    return PAR_dict_posterior


def D_slope(param_0, DD, WW, veg_obs, veg_thr, s0, N_min, scaling_time=24, D_plusmin=5, C_plusmin=10):
    """Change-detection re-estimate of the WCM's C/D parameters from 6-day
    deltas of simulated soil moisture vs. observed backscatter during
    low-vegetation periods, narrowing their priors around the fit. Used
    between the seasonal engine's cal_bare_1 and cal_bare_2 stages.
    """
    vegetation_quantile = 0.5  # noqa: F841 (currently unused)

    ssm_calc, veg_calc, obs_calc = WW, veg_obs, s0

    delta_ssm = ssm_calc[6 * scaling_time:] - ssm_calc[:-6 * scaling_time]
    delta_obs = obs_calc[6 * scaling_time:] - obs_calc[:-6 * scaling_time]

    cond1 = ~np.isnan(delta_obs)
    cond2 = veg_calc[3 * scaling_time:-3 * scaling_time] < veg_thr

    delta_ssm_nonan = delta_ssm[cond1 & cond2]
    delta_obs_nonan = delta_obs[cond1 & cond2]

    if len(delta_obs_nonan) > N_min:
        pf = np.polyfit(delta_ssm_nonan, delta_obs_nonan, 1)
        param_0['D']['m'] = pf[0]

        param_0['D']['m'] = max(param_0['D']['r'][0], param_0['D']['m'])
        param_0['D']['m'] = min(param_0['D']['r'][1], param_0['D']['m'])

        param_0['C']['r'] = [
            min(param_0['C']['r'][0], param_0['C']['m'] - C_plusmin),
            max(param_0['C']['r'][1], param_0['C']['m'] + C_plusmin),
        ]
        param_0['D']['r'] = [
            max(param_0['D']['r'][0], param_0['D']['m'] - D_plusmin),
            min(param_0['D']['r'][1], param_0['D']['m'] + D_plusmin),
        ]
        if param_0['D']['m'] == param_0['D']['r'][0]:
            param_0['D']['r'][0] = param_0['D']['m'] - D_plusmin
        elif param_0['D']['m'] == param_0['D']['r'][1]:
            param_0['D']['r'][1] = param_0['D']['m'] + D_plusmin

        condveg = veg_calc < veg_thr
        param_0['C']['m'] = np.nanmean(obs_calc[condveg]) - param_0['D']['m'] * np.mean(ssm_calc[condveg])
    return param_0
