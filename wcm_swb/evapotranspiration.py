"""
Potential/reference evapotranspiration estimators: Hamon, Hargreaves, and
hourly ASCE standardized Penman-Monteith (`ET_Hourly`).

Uses the bundled ``pyeto`` package (``wcm_swb.vendor.pyeto``, BSD 3-Clause,
(c) Mark Richards), which is not available on PyPI.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""
import math
from datetime import datetime

import numpy as np
import pandas as pd

from wcm_swb.vendor.pyeto import *


def hamon(tavg, jdate, lat, par=1.2):
    """Hamon Potential Evapotranspiration Equation.

    The Hamon method is also considered as one of the simplest estimates
    of potential Evapotranspiration.

    Params
    ------
    - par: proportionality coefficient (unitless), PAR=1.2 as in
      https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1752-1688.2005.tb03759.x
    - tavg: vector of mean daily temperature (deg C)
    - lat: latitude ()
    - jdate: a day number of the year (julian day of the year), e.g.
      timetj = df.index.day

    Returns
    -------
    Potential evapotranspiration (mm day-1).

    Details: see Haith and Shoemaker (1987).
    """
    var_theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.0086 * (jdate-186)))
    var_pi = np.arcsin(0.39795 * np.cos(var_theta))
    daylighthr = 24 - 24 / np.pi * np.arccos((np.sin(0.8333 * np.pi / 180) + np.sin(lat * np.pi / 180) * np.sin(var_pi)) / (np.cos(lat * np.pi / 180) * np.cos(var_pi)))
    esat = 0.611 * np.exp(17.27 * tavg / (237.3 + tavg))

    return par * 29.8 * daylighthr * (esat / (tavg + 273.2))


# Ref. https://pyeto.readthedocs.io/en/latest/hargreaves.html


def hargre(lat_deg, dates, temp_min, temp_max, temp_mean):
    """Hargreaves-Samani model for ET0 estimation from temperature input.

    Params
    ------
    - lat_deg: float
    - dates: timestamp
    - temp_*: float
    """
    lat = deg2rad(lat_deg)  # Convert latitude in degrees to radians

    try:
        day_of_year = dates.dayofyear
    except AttributeError:
        dates = pd.Timestamp(dates)
        day_of_year = dates.dayofyear
    sol_decli = sol_dec(day_of_year)  # Solar declination
    sha = sunset_hour_angle(lat, sol_decli)
    ird = inv_rel_dist_earth_sun(day_of_year)
    et_radia = et_rad(lat, sol_decli, sha, ird)  # Extraterrestrial radiation
    eto = hargreaves(temp_min, temp_max, temp_mean, et_radia)
    if isinstance(eto, complex):
        eto = np.nan
    return eto

## Example use with pandas.DataFrame df
## Final dataframe is made by custom function timeseries (see wcm_swb.metrics)

# lat_deg = 44.570842547510622 # latitude of Budrio (deg)
# temp_min = df.min()['Temperatura[°C]'].values
# temp_max = df.max()['Temperatura[°C]'].values
# temp_mean = df.mean()['Temperatura[°C]'].values
# dates = df.asfreq().index
# eto = timeseries( dates,
#                  [ hargre(lat_deg, dates[i] , temp_min[i], temp_max[i], temp_mean[i])
#                   for i in range(len(dates)) ] )
# eto_df = pd.DataFrame(eto).rename(columns={0:'Date',1:'EPOT'}).set_index('Date')


# ----------------------------------------------------------------------------

# Define a wrapper function to apply 'get_ET_Hourly' for each group
def apply_get_ET_Hourly(group, lat_centroid, lon_centroid, elev=0, shortcrop=1, date_format='%Y-%m-%d'):
    # Assuming 'get_ET_Hourly' takes the date and temperature vector as inputs
    # Extract the date (any hour will have the same date)
    date = group.index.date[0].strftime(date_format)
    index = group.index
    temp_vector = group.temperature.values

    # check if all 24h values are present
    if not len(temp_vector)==24:
        return pd.Series([np.nan]*len(index), index=index)
    else:
        # Call 'get_ET_Hourly' with the extracted date and temperature vector
        et_obj = ET_Hourly(lat_centroid, lon_centroid, elev, shortcrop, date, temp_vector)
        ET_values = et_obj.get_ET_Hourly()
        # Return the ET values in a Series (to ensure alignment with the DataFrame's index)
        return pd.Series(ET_values, index=index)


class ET_Hourly:
    """
    A class to calculate hourly reference evapotranspiration (ET0) using the ASCE standardized Penman-Monteith method.

    Attributes:
        LatGra (float): Latitude of the grid point in degrees.
        LonGra (float): Longitude of the grid point in degrees.
        Elev (float): Elevation of the grid point in meters.
        ShortCrop (int): Indicator of whether the crop is short (1) or tall (0).
        Date (str): Date for which ET0 is calculated, in '%Y-%m-%d' format.
        VelVen (float): Wind speed in m/s.
        CopNuv (float): Cloud cover fraction.
        VettThourly (numpy.ndarray): Array of hourly temperatures in degrees Celsius.
        Coe_a (float): Coefficient 'a' for the Angstrom formula, default is 0.25.
        Coe_b (float): Coefficient 'b' for the Angstrom formula, default is 0.5.
        albedo (float): Albedo for the reference surface, default is 0.23.
        SteBol (float): Stefan-Boltzmann constant, default is 4.903e-9.
        FofPosit (dict): Dictionary containing position-related parameters and coefficients for ET0 calculation.

    Methods:
        __init__: Initializes the ET_Hourly class with the provided parameters.
        PenMonHour: Calculates the reference evapotranspiration (ET0) for a given date, temperature vector, wind speed, and cloud cover.
        F_data: Calculates solar-related parameters such as relative distance to the sun, solar declination, and equation of time components for a given date and latitude.
        F_ora: Calculates net solar radiation (Rns) and a cloudiness factor (FatNuv) based on the position and solar parameters.
        TenVap: Calculates the saturation vapor pressure given a temperature.
        CalcInputPM: Calculates inputs for the Penman-Monteith equation such as net radiation at the crop surface (Rn), saturation vapor pressure (TenVapSat), and actual vapor pressure (TenVapEff).
        ASCE_PM: Calculates the reference evapotranspiration (ET0) using the ASCE standardized Penman-Monteith equation.
    """

    def __init__(self, LatGra, LonGra, Elev, ShortCrop, Date, VettThourly, VelVen=2.5, CopNuv=0.15):
        self.LatGra = LatGra
        self.LonGra = LonGra
        self.Elev = Elev
        self.ShortCrop = ShortCrop

        self.Date = Date
        self.VelVen = VelVen
        self.CopNuv = CopNuv
        self.VettThourly = np.array(VettThourly)

        self.FofPosit = {
            'LatRad': self.LatGra * np.pi / 180,
            'LonGraW': 360 - self.LonGra,
            'LonTimZonGraW': 360 - 15,  # CET has an equivalent longitude of 15° East
        }
        PreAtm = 101.3 * ((293 - 6.5e-3 * self.Elev) / 293) ** 5.26  # KPa, static approximation
        self.FofPosit['gamma'] = 1.013e-3 * PreAtm / .622 / 2.45

        if self.ShortCrop == 1:
            self.FofPosit['Cn'] = 37
            self.FofPosit['Cd_daytime'] = .24
            self.FofPosit['Cd_nighttime'] = .96
            self.FofPosit['FatG_daytime'] = .1
            self.FofPosit['FatG_nighttime'] = .5
        else:
            self.FofPosit['Cn'] = 66
            self.FofPosit['Cd_daytime'] = .25
            self.FofPosit['Cd_nighttime'] = 1.7
            self.FofPosit['FatG_daytime'] = .04
            self.FofPosit['FatG_nighttime'] = .2

        # Constants
        self.Coe_a = .25  # Recommended values for Angstrom's formula
        self.Coe_b = .5
        self.albedo = .23  # Albedo for the reference surfaces is fixed at a constant 0.23
        self.SteBol = 4.903e-9 # daily Stefan-Boltzman constant


    def get_ET_Hourly(self):
        """should be made into a recursive formula for different dates"""
        CurET0 = self.PenMonHour(self.FofPosit, self.Date, self.VettThourly, self.VelVen, self.CopNuv)
        return CurET0


    def PenMonHour(self,FofPosit,Date,VettThourly,VelVen,CopNuv):

        FofDate = self.F_data(Date, FofPosit['LatRad'], CopNuv)
        Rns, FatNuv = self.F_ora(FofPosit, FofDate)
        delta, Rn, TenVapSat, TenVapEff = self.CalcInputPM(VettThourly, FatNuv, Rns)
        ET0 = self.ASCE_PM(FofPosit, delta, Rn, VettThourly, TenVapSat, TenVapEff, VelVen)

        return ET0


    def F_data(self,Date,LatRad,CopNuv):

        if not Date: Date=self.Date
        if not LatRad: LatRad=self.LatRad
        if not CopNuv: CopNuv=self.CopNuv

        # Parse the date string in the format 'dd/mm/yyyy'
        parsed_date = datetime.strptime(Date, '%Y-%m-%d')
        Year = parsed_date.year
        Day = parsed_date.day
        Month = parsed_date.month

        # Calculate the Julian day
        julian = Day - 32 + math.floor(275 * Month / 9) + 2 * math.floor(3 / (Month + 1)) + \
                 math.floor(Month / 100 - (Year % 4) / 4 + 0.975)

        FofDATE = {}
        FofDATE['DisRelTS'] = 1 + 0.033 * math.cos(2 * math.pi / 365 * julian) # Calculate the relative distance from the sun
        FofDATE['DecSol'] = 0.409 * math.sin(2 * math.pi / 365 * julian - 1.39) # Calculate the solar declination
        b = 2 * math.pi * (julian - 81) / 364    # Calculate the equation of time components
        FofDATE['SeaCor'] = 0.1645 * math.sin(2 * b) - 0.1255 * math.cos(b) - 0.025 * math.sin(b)
        FofDATE['AnOrTra'] = math.acos(-math.tan(LatRad) * math.tan(FofDATE['DecSol']))    # Calculate the sunrise and sunset hour angles
        FofDATE['AnOrLev'] = -FofDATE['AnOrTra']
        FofDATE['InsTeo'] = 24 / math.pi * FofDATE['AnOrTra']    # Calculate the theoretical sunshine duration
        FofDATE['InsEff'] = (1 - CopNuv) * FofDATE['InsTeo']    # Calculate the effective sunshine duration

        return FofDATE


    def F_ora(self, FofPosit, FofDATE):

        # FofPosit=self.FofPosit; FofDATE=self.FofDATE
        Coe_a = self.Coe_a
        Coe_b = self.Coe_b
        albedo = self.albedo

        Ra = np.zeros(24)
        OraStdMid = np.arange(0.5, 24, 1)
        for i in range(24):
            SoTiAnMid = math.pi / 12 * ((OraStdMid[i] + .06667 * (FofPosit['LonTimZonGraW'] - FofPosit['LonGraW']) + FofDATE['SeaCor']) - 12)
            SoTiAnBeg = SoTiAnMid - math.pi / 24
            SoTiAnEnd = SoTiAnMid + math.pi / 24

            if (SoTiAnEnd > FofDATE['AnOrLev']) and (SoTiAnBeg < FofDATE['AnOrTra']):
                if SoTiAnBeg < FofDATE['AnOrLev']:
                    SoTiAnBeg = FofDATE['AnOrLev']
                elif SoTiAnEnd > FofDATE['AnOrTra']:
                    SoTiAnEnd = FofDATE['AnOrTra']

                Ra[i] = (12 / math.pi) * 4.92 * FofDATE['DisRelTS'] * ((SoTiAnEnd - SoTiAnBeg) * math.sin(FofPosit['LatRad']) * math.sin(FofDATE['DecSol']) +
                          math.cos(FofPosit['LatRad']) * math.cos(FofDATE['DecSol']) * (math.sin(SoTiAnEnd) - math.sin(SoTiAnBeg)))

        Rs = (Coe_a + Coe_b * FofDATE['InsEff'] / FofDATE['InsTeo']) * Ra
        Rns = Rs * (1 - albedo)
        FatNuvCost = 1.35 * (Coe_a + Coe_b * FofDATE['InsEff'] / FofDATE['InsTeo']) / (Coe_a + Coe_b) - .35
        FatNuv = FatNuvCost  # Assuming FatNuv is constant for all hours

        return Rns, FatNuv


    def TenVap(self, T):
        """TenVap function, which calculates the saturation vapor pressure"""
        return 0.6108 * np.exp(17.27 * T / (T + 237.3))


    def CalcInputPM(self, VettThourly, FatNuv, Rns):

        SteBol = self.SteBol

        VettThourly = np.array(VettThourly)  # Ensure VettThourly is a numpy array for element-wise operations
        TenVapEff = self.TenVap(np.min(VettThourly))  # Assuming that dew-point temperature is near the daily minimum air temperature
        FatUmi = 0.34 - 0.14 * np.sqrt(TenVapEff)
        Rnl = SteBol / 24 * (VettThourly + 273.2) ** 4 * FatUmi * FatNuv
        Rn = Rns - Rnl
        TenVapSat = self.TenVap(VettThourly)
        delta = 4098 * TenVapSat / (VettThourly + 237.3) ** 2

        return delta, Rn, TenVapSat, TenVapEff


    def ASCE_PM(self, FofPosit, delta, Rn, VettThourly, TenVapSat, TenVapEff, VelVen):

        # FofPosit=self.FofPosit; delta=self.delta; Rn=self.Rn, VettThourly=self.VettThourly;
        # TenVapSat=self.TenVapSat; TenVapEff=self.TenVapEff; VelVen=self.VelVen

        # Determine the coefficient for heat transfer (Cd) based on the condition of Rn
        Cd = np.where(Rn <= 0, FofPosit['Cd_nighttime'], FofPosit['Cd_daytime'])

        # Calculate the soil heat flux (G) based on the condition of Rn
        G = np.where(Rn <= 0, FofPosit['FatG_nighttime'] * Rn, FofPosit['FatG_daytime'] * Rn)

        # Calculate the numerator components of the Penman-Monteith equation
        Num1 = 0.408 * delta * (Rn - G)
        Num2 = FofPosit['gamma'] * FofPosit['Cn'] * VelVen * (TenVapSat - TenVapEff) / (VettThourly + 273)

        # Calculate the denominator of the Penman-Monteith equation
        Den = delta + FofPosit['gamma'] * (1 + Cd * VelVen)

        # Calculate ET0, ensuring it is non-negative
        ET0 = np.where(Num1 + Num2 > 0, (Num1 + Num2) / Den, 0)

        return ET0
