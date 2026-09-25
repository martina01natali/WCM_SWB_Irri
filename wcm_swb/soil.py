"""
Soil hydraulic property pedotransfer functions (Saxton & Rawls, 2006).

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""
import numpy as np


def SaxtonRawls(Clay, Sand, OM):
    """
    Inputs: sand [%], clay [%], OM [%]
        Note: percentages with base the decimal percent by weight
            e.g. if OM = 50 g/kg = 0.05 = 5 %
    Outputs: lambda, saturation [%], Ksat [mm/h]

    E.g. SaxtonRawls(Sand, Clay, OM)

    ref:
    Saxton, K.E. and Rawls, W.J. (2006), Soil Water Characteristic Estimates
    by Texture and Organic Matter for Hydrologic Solutions. Soil Sci. Soc.
    Am. J., 70: 1569-1578. https://doi.org/10.2136/sssaj2005.0117
    """
    Sand = Sand / 100
    Clay = Clay / 100
    OM   = OM / 10

    SM33t = -0.251 * Sand + 0.195 * Clay + 0.011 * OM + 0.006 * Sand * OM - 0.027 * Clay * OM + 0.452 * Sand * Clay + 0.299
    SM33 = SM33t + 1.283 * SM33t ** 2 - 0.374 * SM33t - 0.015

    SM1500t = -0.024 * Sand + 0.487 * Clay + 0.006 * OM + 0.005 * Sand * OM - 0.013 * Clay * OM + 0.068 * Sand * Clay + 0.031
    SM1500 = SM1500t + 0.14 * SM1500t - 0.02

    SMsat33t = 0.278 * Sand + 0.034 * Clay + 0.022 * OM - 0.018 * Sand * OM - 0.027 * Clay * OM - 0.584 * Sand * Clay + 0.078
    SMsat33 = SMsat33t + 0.636 * SMsat33t - 0.107

    lam = (np.log(SM33) - np.log(SM1500)) / (np.log(1500) - np.log(33))

    SMsat = SM33 + SMsat33 - 0.097 * Sand + 0.043

    Ksat = 1930 * (SMsat - SM33) ** (3 - lam)  # [mm/h]

    return lam, SMsat, Ksat
