"""
Statistics, normalization, and curve-fitting helper functions.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""
import math

import numpy as np
import scipy.special as sp


#----------------------------------------------------------------------------
# Statistics and data cleaning / normalization

def lin_db(x):
    """linear to dB"""
    return 10*np.log10(x)


def db_lin(x):
    """dB to linear"""
    return 10**(x/10)


def norm_simple(x):
    """min-max normalization of x data"""
    return (x-np.min(x))/(np.max(x)-np.min(x))


def norm_fit(x, a, b):
    """norm in range a,b

    x = data, a<b
    """
    return a+(x-np.nanmin(x))*(b-a)/(np.nanmax(x)-np.nanmin(x))


def biasvalue(obs, sim):
    """distance between obs' and sim's mean values"""
    if len(obs)==len(sim):
        return np.nanmean(obs-sim)
    else: raise ValueError(
        f'obs and sim must have same first dimension, but have shapes {np.shape(obs)} and {np.shape(sim)}')


def Rvalue(x:list,y:list)->float:
    """compute Pearson's R between x,y data"""
    if len(x)!=len(y):
        raise ValueError(
            f'x and y must have same first dimension,'+
            'but have shapes{np.shape(x)} and {np.shape(y)}')

    mask = ~(np.isnan(x) | np.isnan(y))
    x_filtered = np.array(x)[mask]
    y_filtered = np.array(y)[mask]
    xy_filtered = np.transpose(np.column_stack((x_filtered, y_filtered)))
    return np.corrcoef(xy_filtered)[0][1]


def RMSEvalue(x:list, y:list)->float:
    """compute RMSE between x,y data with eventual nan drop"""
    if len(x)!=len(y):
        raise ValueError(
            f'x and y must have same first dimension,'+
            'but have shapes{np.shape(x)} and {np.shape(y)}')
    return np.nanmean((x - y) ** 2) ** 0.5


def timeseries(dates, data):
    """Returns a matrix (list of type(dates,data)) in the format [dates,data]"""

    if len(dates)==len(data):
        return [[dates[i],data[i]] for i in range(len(dates))]
    else: raise ValueError(
        f'dates and data must have same first dimension, but have shapes {np.shape(dates)} and {np.shape(data)}')


def significant_figures_str(master, slave):
    master_rounded = float("%.1g" % master)
    master_rounded_str = str(master_rounded)
    digits = master_rounded_str[::-1].find('.')
    slave_rounded_str = str(f'%.{digits}f' % slave)

    return [master_rounded_str, slave_rounded_str]


#----------------------------------------------------------------------------
# Data analysis, fit
# Fitting functions


def linear(x,a,b):
    return a+b*x


def gauss(x, A, mean, dev):
    """Not-normalized, shifted gaussian distribution."""

    pdf = (1/(dev*np.sqrt(2*math.pi)))*np.exp(-(x-mean)**2/(2*dev**2))
    return A*pdf


def skew_gauss(x, A, mean, dev, alpha,):
    """Skew, not-normalized and shifted gaussian distribution.

    References:
    - https://www.wolframalpha.com/input?i=skew+gaussian+distribution
    - https://stackoverflow.com/questions/15400850/scipy-optimize-curve-fit-unable-to-fit-shifted-skewed-gaussian-curve
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
    """

    pdf = (1/(dev*np.sqrt(2*np.pi)))*np.exp(-pow((x-mean),2)/(2*pow(dev,2)))
    cdf = sp.erfc((-alpha*(x-mean))/(dev*np.sqrt(2)))
    return A*pdf*cdf
