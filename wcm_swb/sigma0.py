"""
Sentinel-1 backscatter conditioning: cross-orbit HIST normalization.

Pure numpy/pandas (no Earth Engine dependency), so it can be imported and
tested from the main environment.

Each relative orbit sees a field at a different incidence angle, so raw sigma0
distributions are offset between orbits. HIST normalization (Mladenova et al.,
2013, doi:10.1109/TGRS.2012.2205264) maps every orbit's distribution onto the
mean/std of one reference orbit:

    sigma0_norm = ref_mean + ref_std / orb_std * (sigma0 - orb_mean)

Statistics are pooled over all pixels and dates of each orbit, in dB. The
reference orbit is the one whose mean incidence angle is closest to 40 deg,
excluding orbits with fewer than half the mean number of observations per
orbit.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""
import numpy as np
import pandas as pd


def lin_db(x):
    return 10 * np.log10(x)


def db_lin(x):
    return 10 ** (x / 10)


def hist_norm(value, mean, std, ref_mean, ref_std):
    """HIST normalization of `value` from (mean, std) onto (ref_mean, ref_std)."""
    return ref_mean + ref_std / std * (value - mean)


def orbit_statistics(df, pols=('VV', 'VH')):
    """Per-orbit count, mean/std of each polarization (dB) and mean incidence angle.

    `df` needs columns ``orb``, ``angle`` and each of `pols`. A zero std is
    replaced by sqrt(|mean|) to avoid division by zero.
    """
    grouped = df.groupby('orb')
    stats = pd.DataFrame({'count': grouped.size(), 'angle_mean': grouped['angle'].mean()})
    for pol in pols:
        stats[f'{pol}_mean'] = grouped[pol].mean()
        stats[f'{pol}_std'] = grouped[pol].std(ddof=0)
        zero = stats[f'{pol}_std'] == 0
        stats.loc[zero, f'{pol}_std'] = np.sqrt(np.abs(stats.loc[zero, f'{pol}_mean']))
    return stats


def select_reference_orbit(stats, reference_angle=40.0, min_count_frac=0.5):
    """Orbit whose mean incidence angle is closest to `reference_angle`.

    Orbits with fewer than `min_count_frac` x (mean observations per orbit)
    are not eligible as reference (they're still normalized).
    """
    threshold = stats['count'].sum() / len(stats) * min_count_frac
    eligible = stats[stats['count'] >= threshold]
    return (eligible['angle_mean'] - reference_angle).abs().idxmin()


def normalize_orbits(df, pols=('VV', 'VH'), reference_angle=40.0, orb_ref=None):
    """HIST-normalize raw backscatter across orbits.

    Parameters
    ----------
    df : pandas.DataFrame
        Pixel-level observations with columns ``orb``, ``angle`` and the raw
        (not normalized) polarizations named ``{pol}_notnorm``, in dB.
    pols : tuple of str
        Polarizations to normalize.
    reference_angle : float
        Target incidence angle [deg] for automatic reference-orbit selection.
    orb_ref : int, optional
        Force a reference orbit instead of selecting it automatically.

    Returns
    -------
    out : pandas.DataFrame
        Copy of `df` with added normalized ``{pol}`` columns and the cross
        ratio ``CR`` = VH/VV (linear, computed from the normalized dB values,
        Vreugdenhil et al. 2020).
    orb_ref : int
        The reference orbit used.
    stats : pandas.DataFrame
        Per-orbit statistics (see `orbit_statistics`).
    """
    raw = df.rename(columns={f'{pol}_notnorm': pol for pol in pols})
    stats = orbit_statistics(raw, pols)
    if orb_ref is None:
        orb_ref = select_reference_orbit(stats, reference_angle)

    out = df.copy()
    for pol in pols:
        mean = df['orb'].map(stats[f'{pol}_mean'])
        std = df['orb'].map(stats[f'{pol}_std'])
        out[pol] = hist_norm(df[f'{pol}_notnorm'], mean, std,
                             stats.at[orb_ref, f'{pol}_mean'], stats.at[orb_ref, f'{pol}_std'])
    if 'VV' in pols and 'VH' in pols:
        out['CR'] = db_lin(out['VH'] - out['VV'])
    return out, orb_ref, stats
