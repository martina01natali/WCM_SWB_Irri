# Data layout

The pipeline expects data under a root directory (`WCM_DATA_ROOT`, see
`README.md`) laid out as:

```
$WCM_DATA_ROOT/
├── study_sites/
│   └── <site_name>/
│       ├── shapefile/<site_name>.geojson   # field polygon (centroid for meteo, polygon for soil)
│       ├── sigma0/s0_<site_name>_....nc    # Sentinel-1 backscatter, incl. VV/VV_notnorm columns
│       ├── ndvi/ndvi_<site_name>_....nc    # Sentinel-2 NDVI
│       ├── merida/MERIDA_PREC_TEMP.nc      # cached, site-extracted MERIDA meteo (see below)
│       ├── merida/MERIDA_{PREC,TEMP}_<start>-<end>.nc, merida/monthly/
│       │                                   # national MERIDA HRES files (see below)
│       └── inputs/soilgrids/soilgrids_<field>_<var>_<depth>.tif
│                                           # SoilGrids clipped to the field (00_preprocessing/download_soilgrids.py)
```

`00_preprocessing/download_all.py` downloads all of these for a site in one run.

Only the field polygon of the example site, **Ceregnano**
(`study_sites/Ceregnano/shapefile/`), is included in the repository; every
other file is downloaded by `00_preprocessing` (see the main `README.md`).

The SoilGrids tifs hold `sand`, `clay`, `soc`, `wwfc` (water content at 33 kPa,
field capacity) and `www` (1500 kPa, wilting point) for the 0-5, 5-15, 15-30 and
30-60 cm layers, on SoilGrids' native 250 m Homolosine grid over the pixels
covering the field; see `00_preprocessing/README.md`.

## Scope

The pipeline calibrates the model on satellite observations and meteorological
forcing only: no in-situ irrigation or soil moisture data are needed, and
none are read.

The repository contains code only, no input data and no results: download
the inputs with `00_preprocessing` and run the pipeline to produce the results
(see `01_calibration/README.md`).

## MERIDA meteo data

`study_sites/<site>/merida/MERIDA_PREC_TEMP.nc` is a small cache of hourly
precipitation/temperature at the MERIDA HRES (0.04°) pixel nearest to
the field centroid, for the calibration period, extracted by `01_calibration`
from the national files (e.g. `merida/MERIDA_{PREC,TEMP}_202101-202112.nc` for
2021, the template period; `01_calibration`'s `file_MERIDA`). Download the
national files (~1 GB per variable and year, plus the monthly downloads in
`merida/monthly/`) with `00_preprocessing/download_merida.py` (or
`download_all.py`).

`01_calibration` uses the cache when it covers the calibration period;
otherwise it rebuilds it from `file_MERIDA` (`wcm_swb.inputs.build_input_MERIDA`),
and stops with an error if the MERIDA files don't cover the period either. So
to calibrate another year, download MERIDA for it and point `file_MERIDA` at the
new files.
