"""
Plotting utilities for the WCM-SWB calibration pipeline: trace and
posterior plots, parameter tables, time-series summaries of calibration
outputs, and a histogram-normalization helper.

All figures use the package stylesheet ``wcm_swb/style/wcm_swb.mplstyle``.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az

from wcm_swb.config import substitute_keywords

import importlib.resources
with importlib.resources.as_file(importlib.resources.files('wcm_swb.style') / 'wcm_swb.mplstyle') as _stylesheet_path:
    plt.style.use(str(_stylesheet_path))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_trace(
    trace,
    PAR_dict_posterior=None,
    cal_name=None,
    step_sampler=None,
    n_draws=None,
    n_chains=None,
    filename_trace=None,
    folder_plot=None,
    opt_field=None,
    opt_obs=None,
    add_description="",
    filename_template=None,
    opt_save_plots=False,
    extension_plot=".png",
    max_subplots=100,
    layout="constrained",
    compact=True,
    combined=True,
):
    """
    Plots a trace with custom lines using ArviZ's plot_trace function.

    Parameters:
        trace (arviz.InferenceData): The trace data to be plotted.
        max_subplots (int): Maximum number of subplots for the plot. Default is 100.
        layout (str): Layout setting for the figure (passed to plt.figure). Default is "constrained".
        compact (bool): Whether to use a compact layout for the plot. Default is True.
        combined (bool): Whether to combine chains into a single plot. Default is True.
    """
    az.rcParams["plot.max_subplots"] = max_subplots
    backend_kwargs = {'layout': layout}
    lines = [[k, {}, np.array(PAR_dict_posterior[cal_name][k]['value'])] for k in PAR_dict_posterior[cal_name]]
    az.plot_trace(
        trace,
        compact=compact,
        backend_kwargs=backend_kwargs,
        combined=combined,
        lines=lines,
    )
    plt.suptitle(f'Parameters posterior distributions\nSampler={step_sampler}, draws={n_draws}, chains={n_chains}', x=0.5, y=1.01,)

    # build filename path
    filename_dict = filename_path = None
    filename_dict = {
        'filename_output': filename_trace,
        'root_plot': folder_plot,
        'opt_field': opt_field,
        'opt_obs': opt_obs,
        'add_description': '_' + add_description if add_description != '' else '',
    }
    filename_path = substitute_keywords(filename_template, **filename_dict)
    if opt_save_plots:
        plt.savefig(filename_path + extension_plot, bbox_inches='tight', dpi=300)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_parameters_tables(
    PAR_dict_posterior,
    filename_table_params=None,
    folder_plot=None,
    opt_field=None,
    opt_obs=None,
    add_description="",
    filename_template=None,
    opt_save=False,
    extension_plot=".png",
    figsize=(10, 10),
    hspace=0,
    title_fontsize=14
):
    plt.figure(figsize=(10, 10))  # Adjust figsize as needed
    for i, cal_name in enumerate(PAR_dict_posterior.keys()):

        plt.subplot(3, 1, i + 1)
        df = pd.DataFrame.from_dict(PAR_dict_posterior[cal_name], orient='index')
        df['range'] = df.range.apply(lambda x: np.round(x, 1))
        df = df[df['std'] != 0]
        df = df.round(3)

        # Hide axes
        plt.axis('tight')
        plt.axis('off')

        # Create table
        table = plt.table(cellText=df.values,
                           colLabels=df.columns,
                           rowLabels=df.index,
                           cellLoc='center',
                           loc='center')

        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        # Set bold font for headers
        for (i, j), cell in table.get_celld().items():
            if i == 0 or j == -1:
                cell.set_text_props(weight='bold')
            cell.set_height(0.1)

        plt.title('Calibrated parameters for ' + cal_name, y=.9, fontsize=14)
    plt.subplots_adjust(hspace=0)

    # build filename path
    filename_dict = filename_path = None
    filename_dict = {
        'filename_output': filename_table_params,
        'root_plot': folder_plot,
        'opt_field': opt_field,
        'opt_obs': opt_obs,
        'add_description': '_' + add_description if add_description != '' else '',
    }
    filename_path = substitute_keywords(filename_template, **filename_dict)

    if opt_save:
        plt.savefig(filename_path + extension_plot, bbox_inches='tight', dpi=300)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_quad(
    trace, input_df, opt_obs, t,
    s0, veg, WW1, WW2,
    I, PERC1, ET1, PERC2,
    ET2, df_irriitaly, IRRI, ET0, P,
    opt_field, add_description='',
    opt_save_plots=False,
    filename_template=None,
    folder_plot=None,
    filename_quad=None,
    extension_plot='.png'
):

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), layout='tight', sharex=True)
    ax0, ax1, ax2 = axes

    obs = input_df[opt_obs].values
    mask = np.isfinite(obs)

    # ----------------------- TOP PANEL
    # NDVI
    ax00 = ax0.twinx()
    ax00.plot(t, veg, '-', color='tab:olive', label='NDVI', zorder=-100)
    ax00.set_ylabel('NDVI [-]')
    ax00.legend(loc='upper right')
    # ax00.axes.get_xaxis().set_ticks([])

    # sigma0
    ax0.plot(t[mask], obs[mask], 'ks--', label=r'$\sigma^0$ (obs)', lw=1)
    ax0.plot(t[mask], s0[mask], '-', color='tab:red', label=r'$\sigma^0$ (sim)')
    ax0.set_ylabel(r'$\sigma^0$ [dB]')
    ax0.legend(loc='upper left')
    ax0.set_ylim(-20, -5)
    ax0.set_xlim(t[0], t[-1])
    # ax0.axes.get_xaxis().set_ticks([])

    # ----------------------- MIDDLE PANEL
    depth1 = 50
    depth2 = 500
    ax1.plot(t, WW1, color='tab:cyan', label='WW1')
    ax1.plot(t, WW2, color='tab:blue', label='WW2', zorder=-100)
    ax1.set_ylim(0, .6)
    ax1.set_ylabel('Volumetric water\ncontent ' + r'[m$^3$m$^{-3}$]')
    ax1.legend(loc='upper left')
    # ax1.axes.get_xaxis().set_ticks([])

    ax11 = ax1.twinx()
    df_resample_irri = pd.DataFrame.from_dict({'datetime': t, 'irri': IRRI, 'eto': ET0, 'p': P}).set_index('datetime')
    df_resample_irri.fillna(0, inplace=True)
    df_resample_irri = df_resample_irri.resample('d').sum()
    t_day = df_resample_irri.index
    ax11.plot(t_day, df_resample_irri.irri, color='tab:orange', label='IRRI')
    ax11.set_ylim(0, 15)
    ax11.set_ylabel('Simulated irrigation [mm/day]')
    ax11.legend(loc='upper right')
    # ax11.axes.get_xaxis().set_ticks([])

    # ----------------------- BOTTOM PANEL
    ax2.bar(df_resample_irri.index, df_resample_irri.p, color='tab:gray', label='P')
    ax2.plot(df_resample_irri.index, df_resample_irri.eto, color='tab:green', label='ET0')
    ax2.set_ylim(0, 30)
    ax2.legend(loc='upper left')
    ax2.set_ylabel('Precipitation and\nET0 (FAO-PM) [mm/day]')

    plt.suptitle('Components of wcm, swb in layer 1 (top), layer 2 (middle) and inputs with estimated irrigation (bottom)',
                 fontsize=20)

    for ax, txt in zip([ax0, ax1, ax2], ['a)', 'b)', 'c)']):
        ax.text(-0.07, 1.02, txt, transform=ax.transAxes, fontsize=24, fontweight='bold')

    # build filename path
    if opt_save_plots:
        filename_dict = filename_path = None
        filename_dict = {
            'filename_output': filename_quad,
            'root_plot': folder_plot,
            'opt_field': opt_field,
            'add_description': '_' + add_description if add_description != '' else '',
            'opt_obs': opt_obs,
        }
        filename_path = substitute_keywords(filename_template, **filename_dict)
        plt.savefig(filename_path + extension_plot, bbox_inches='tight', dpi=300)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_pair(
    trace,
    opt_field,
    opt_obs,
    filename_pair,
    folder_plot,
    filename_template,
    add_description='',
    opt_save_plots=False,
    extension_plot='.png',
    figsize=(10, 8),
    textsize=14,
):
    backend_kwargs = {'layout': 'constrained'}
    az.plot_pair(trace, figsize=(10, 8), kind='hexbin',
                 marginals=True, textsize=textsize, backend_kwargs=backend_kwargs)
    plt.suptitle("Posterior correlations", size=18)

    # build filename path
    filename_dict = filename_path = None
    filename_dict = {
        'filename_output': filename_pair,
        'root_plot': folder_plot,
        'opt_field': opt_field,
        'opt_obs': opt_obs,
        'add_description': '_' + add_description if add_description != '' else '',
    }
    filename_path = substitute_keywords(filename_template, **filename_dict)
    if opt_save_plots:
        plt.savefig(filename_path + extension_plot, bbox_inches='tight', dpi=300)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_posteriors_peaks(values, param, opt_field, PAR_dict, output_dir, opt_save_plots):

    with plt.rc_context({'font.size': 16}):
        plt.figure(figsize=(10, 6))

        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde
        from numpy.linalg import LinAlgError

        # Perform Kernel Density Estimation (KDE)
        try:
            # Histogram
            plt.hist(values, bins=50, density=True, alpha=0.5, color='gray', label='Histogram')

            kde = gaussian_kde(values)
            x = np.linspace(min(values), max(values), 1000)
            kde_values = kde(x)

            # Find peaks in the KDE curve
            peaks, props = find_peaks(kde_values, height=(None, None), width=(None, None))
            peak_h, peak_x = kde_values[peaks], x[peaks]  # get peak x, get peak heights

            # find width of peaks
            peak_left = x[[int(num) for num in props['left_ips']]]
            peak_right = x[[int(num) for num in props['right_ips']]]
            peak_w = peak_right - peak_left

            mode_h = peak_h.max(); mode_id = list(peak_h).index(mode_h)
            mode_x = peak_x[mode_id]  # get the highest peak position and error as window width
            mode_err = peak_w[mode_id]

            # KDE curve
            plt.plot(x, kde_values, color='blue', label='KDE')
            ymin, ymax = plt.ylim()

            # peaks' widths
            # choose only mode peak
            peak_left = x[[int(num) for num in props['left_ips']]][mode_id]
            peak_right = x[[int(num) for num in props['right_ips']]][mode_id]
            peak_w = peak_right - peak_left
            y = np.linspace(0, ymax * 1.1, 10)
            plt.gca().fill_betweenx(y, peak_left, peak_right, color='red', alpha=.3, edgecolor=None)

            # get chosen value from PAR_dict
            if param in PAR_dict:
                chosen_value = PAR_dict[param]['value']
                ref = PAR_dict[param]['ref']
            else:
                chosen_value = np.nanmedian(values)
                ref = 'median'
            # Vertical lines for peaks
            for peak in peak_x:
                if peak == mode_x:
                    plt.axvline(peak, color='red', ls='-', label=f'Mode at {peak:.2f}')
                # else:
                #     plt.axvline(peak, color='red', linestyle='--', label=f'Peak at {peak:.2f}')

            # plot also median
            median_x = np.nanmedian(values)
            plt.axvline(median_x, color='orange', ls='-', label=f'Median at {median_x:.2f}')

            # plot chosen value above median or mode
            plt.axvline(chosen_value, color='k', ls='--', label=f'Reference value: {ref}')

            plt.title(f'Distribution of {param} - KDE, local maxima, mode and median')
            plt.ylim(ymin, ymax)
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend(loc='upper right')
            if opt_save_plots:
                out_fn = os.path.join(output_dir, param + f'_{opt_field}' + '.png')
                plt.savefig(out_fn, bbox_inches='tight', dpi=300)
        except (LinAlgError, ValueError):
            pass


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_output_quad(ds, opt_obs='VV', opt_veg='NDVI', title=None, figsize=(19 / 2.54, 10 / 2.54)):
    """Three-panel summary of one ``output_*.nc`` with its 5-95% credible bands.

    Reads directly from the output dataset and spans its full time range.
    Bands are drawn only for variables that have ``_q05``/``_q95``
    companions in ``ds``.

    Panels: (a) sigma0 obs vs. sim + NDVI, (b) layer-1/2 soil moisture +
    daily simulated irrigation, (c) daily precipitation and ET0.
    Returns the figure.
    """
    blue, lightgreen, purple, green, pink, orange, lightblue, red = [
        "#6973d8", "#ac9c3d", "#5a3687", "#56ae6a", "#c068b9", "#ba543d", "#6d90d7", "#b84873"
    ]

    def band(var):
        lo, hi = f'{var}_q05', f'{var}_q95'
        return (ds[lo].values, ds[hi].values) if lo in ds and hi in ds else None

    t = pd.DatetimeIndex(ds['Datetime'].values)
    mask = ds['Sigma0_mask'].values.astype(bool)
    s0_var = f'Sigma0_{opt_obs}'

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # ----------------------- TOP PANEL
    ax0.plot(t[mask], ds[f'{s0_var}_obs'].values[mask], 'ks--', label=rf'$\sigma^0_{{{opt_obs}}}$ (obs)', lw=1)
    if band(s0_var) is not None:
        lo, hi = band(s0_var)
        ax0.fill_between(t[mask], lo[mask], hi[mask], color=red, alpha=0.30, linewidth=0)
    ax0.plot(t[mask], ds[s0_var].values[mask], '-', color=red, label=rf'$\sigma^0_{{{opt_obs}}}$ (sim)')
    ax0.set_ylabel(rf'$\sigma^0_{{{opt_obs}}}$ [dB]')
    ax0.legend(loc='lower left')
    ax00 = ax0.twinx()
    ax00.plot(t, ds[opt_veg].values, '-', color=lightgreen, label=opt_veg, alpha=.7)
    ax00.set_ylabel(f'{opt_veg} [-]')
    ax00.legend(loc='lower right')

    # ----------------------- MIDDLE PANEL
    for var, color, label, alpha in [('Soil_Moisture_1', lightblue, r'$\theta_1$', 0.20),
                                      ('Soil_Moisture_2', purple, r'$\theta_2$', 0.15)]:
        if band(var) is not None:
            ax1.fill_between(t, *band(var), color=color, alpha=alpha, linewidth=0)
        ax1.plot(t, ds[var].values, color=color, label=label)
    ax1.set_ylim(0, .6)
    ax1.set_ylabel('Volumetric water\ncontent ' + r'[m$^3$m$^{-3}$]')
    ax1.legend(loc='lower left')

    daily = pd.DataFrame({'irri': ds['Irrigation'].values, 'eto': ds['ET0'].values, 'p': ds['Rain'].values},
                         index=t)
    if band('Irrigation') is not None:
        daily['irri_q05'], daily['irri_q95'] = band('Irrigation')
    daily = daily.fillna(0).resample('D').sum()

    ax11 = ax1.twinx()
    if 'irri_q05' in daily:
        ax11.fill_between(daily.index, daily.irri_q05, daily.irri_q95, color=orange, alpha=0.20, linewidth=0)
    ax11.plot(daily.index, daily.irri, color=orange, label='IRRI')
    ax11.set_ylim(0, max(25, 1.1 * daily[[c for c in daily if c.startswith('irri')]].values.max()))
    ax11.set_ylabel('Simulated\nirrigation [mm/day]')
    ax11.legend(loc='lower right')

    # ----------------------- BOTTOM PANEL
    ax2.bar(daily.index, daily.p, color='tab:gray', label='P')
    ax2.plot(daily.index, daily.eto, color=green, label='ET0')
    ax2.set_ylim(0, 30)
    ax2.legend(loc='lower left')
    ax2.set_ylabel('Precipitation and\nET0 [mm/day]')
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel('Date')

    for ax, txt in zip([ax0, ax1, ax2], ['a)', 'b)', 'c)']):
        ax.text(-.1, 1.03, txt, transform=ax.transAxes, fontsize=10, fontweight='bold')
    if title:
        fig.suptitle(title)
    fig.subplots_adjust(hspace=0.2)
    return fig


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def HIST_norm(ref_mean, ref_std, obs: list):
    """HIST normalization
    Ref. Mladenova, 2013, https://ieeexplore.ieee.org/document/6264094

    obs = [value, mean, std]
    """
    value, mean, std = obs
    return ref_mean + ref_std / std * (value - mean)
