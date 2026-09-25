# 00_preprocessing

Downloads Sentinel-1 backscatter and Sentinel-2 NDVI via Google Earth Engine
(GEE), producing the `sigma0/*.nc` and `ndvi/*.nc` files that
`01_calibration` consumes (see `docs/data_layout.md`), field-averaged
SoilGrids soil properties (ISRIC web service, no GEE needed), and the MERIDA
precipitation/temperature reanalysis (`download_merida.py`, see
[MERIDA meteo](#merida-meteo) — separate config, no GEE).

The repository contains no input data: run this stage first, also for the
Ceregnano example (only its field polygon,
`study_sites/Ceregnano/shapefile/ceregnano.geojson`, is included).

## Setup

```bash
conda env create -f environment-preprocessing.yml
conda activate wcm_swb_preprocessing
pip install -e .                        # from the repo root: the scripts import wcm_swb
export WCM_DATA_ROOT=/path/to/data      # see docs/data_layout.md
export GEE_PROJECT=<your-google-cloud-project-id>
```

The first run opens the GEE authentication flow if no credentials are cached.

## Download everything at once

`download_all.py` runs all the download scripts below for one study site, one
after the other, and can be left unattended:

```bash
cd 00_preprocessing
export WCM_DATA_ROOT=...
set -a; source ~/.config/wcm_swb/credentials.env; set +a   # GEE_PROJECT, MERIDA_USER/PASSWORD: see the main README, "Credentials"
nohup python download_all.py --config configuration_00_preprocessing_TEMPLATE.json \
    --merida-config configuration_00_preprocessing_MERIDA_TEMPLATE.json &
```

- Steps, in order: `soilgrids`, `sigma0`, `ndvi`, `merida`. Everything is
  written inside the site folder (`paths.root`, e.g. `study_sites/<site>/`):
  `inputs/soilgrids/`, `sigma0/`, `ndvi/`, `merida/`.
- A step is **skipped when all its expected output files exist**, so
  re-running only downloads what's missing. `--dry-run` only reports
  present/missing files.
- Before downloading anything, a preflight check stops the run if something
  would make a step fail for sure: no field shapefiles, `GEE_PROJECT` /
  MERIDA credentials unset, `curl` missing, or the MERIDA config pointing at a
  different site (`paths.root` of both configs must be the same). It also warns
  if the MERIDA `options.years` don't cover the Sentinel period.
- A failed step is retried (`--retries`, default 2); the sub-scripts skip what
  they already finished, so a retry resumes. A failure doesn't stop the other
  steps. Only one `download_all.py` runs per site at a time (lock file
  `logs/download_all.lock`). Each step's output goes to `study_sites/<site>/logs/`, plus a
  summary `download_all_<timestamp>.log`; the exit code is non-zero if any
  step failed.
- `--steps sigma0 ndvi` runs a subset; `--field <id>` one field (not MERIDA,
  which is per site).

The MERIDA period is set in its own config (`options.years`/`months`), not by
`options.start_date`/`end_date`.

## Scripts

| Script | Output (per field) |
|---|---|
| `download_sentinel1_gee.py` | `sigma0/s0_<field>_....nc` — `orb`, `VV_notnorm`/`VH_notnorm` (raw, dB), `angle`, `VV`/`VH` (HIST-normalized, dB), `CR` |
| `download_sentinel2_ndvi_gee.py` | `ndvi/ndvi_<field>_....nc` — `NDVI` |
| `download_soilgrids.py` | `inputs/soilgrids/soilgrids_<field>_<var>_<depth>.tif` — sand, clay, SOC, field capacity (33 kPa), wilting point (1500 kPa), 0-60 cm, clipped to the field |
| `download_merida.py` | per site (not per field): `merida/MERIDA_{PREC,TEMP}_<start>-<end>.nc` (+ monthly files in `merida/monthly/`) |

The two GEE scripts and `download_soilgrids.py` read the same config and loop over every shapefile matching
`paths.data_input.file_shapes` (one field per file; the file stem is the field
id, `{opt_field}`). Existing outputs are skipped unless
`options.opt_overwrite` is `true`.

```bash
cd 00_preprocessing
python download_sentinel1_gee.py --config configuration_00_preprocessing_TEMPLATE.json
python download_sentinel2_ndvi_gee.py --config configuration_00_preprocessing_TEMPLATE.json
python download_soilgrids.py --config configuration_00_preprocessing_TEMPLATE.json
# restrict to one field:  --field ceregnano
```

## Sentinel-1 processing

1. All `COPERNICUS/S1_GRD_FLOAT` scenes over the field are reprojected onto one
   reference grid (coregistration) and sampled per pixel (VV, VH, incidence
   angle), in `options.chunk_days`-day requests.
2. The 2016-09-30/2016-10-01 scenes are skipped (they make GEE requests fail).
3. Backscatter is HIST-normalized across relative orbits (Mladenova et al.
   2013, doi:10.1109/TGRS.2012.2205264) onto the orbit whose mean incidence
   angle is closest to `sentinel1.reference_angle` (40°) — implemented in
   `wcm_swb.sigma0`. Orbits with
   less than half the mean number of observations aren't eligible as
   reference.
4. `CR` = VH/VV (linear) from the normalized values (Vreugdenhil et al. 2020).

`01_calibration` uses `VV` (`options.opt_obs`).

Timestamps are floored to the hour in `options.timezone` (`Europe/Rome` in the
templates).

With `paths.data_output.opt_save_plots`, a per-orbit histogram of VV before and
after normalization is written to `folder_plots` as a QC check.

## Sentinel-2 processing

`COPERNICUS/S2_SR_HARMONIZED` scenes, one product per acquisition and tile:
GEE holds ESA's reprocessed products (e.g. processing baseline 05.00) next to
the originally ingested ones (e.g. 02.08), and the reprocessed version is
preferred (highest `PROCESSING_BASELINE`, then latest generation time). Scenes
are then kept if their scene-level cloud cover is below
`sentinel2.CLOUDY_PIXEL_PERCENTAGE` (50%). Where two tiles overlap the field
(Ceregnano: T32TQQ/T32TQR), the first by `system:index` is used for each
pixel. Reprojected onto one grid, NDVI =
(B8 − B4)/(B8 + B4) per pixel. No per-pixel cloud mask is applied.
Sentinel-2 data start on 2017-03-28; earlier start dates are
clamped.

## SoilGrids processing

`download_soilgrids.py` doesn't use GEE (`GEE_PROJECT` isn't needed). For each
property in `soilgrids.properties` and depth in `soilgrids.depths`, the
SoilGrids 2.0 `mean` prediction is requested from the ISRIC WCS
(`maps.isric.org`) over the field's bounding box snapped to the native 250 m
Interrupted Goode Homolosine grid, so the values are the original pixels, not
resampled (verified identical to ISRIC's `files.isric.org` VRTs). The AOI is
the union of all features in the geojson.

Each window is saved as a small GeoTIFF (~2 KB; 20 files, 84 KB for
Ceregnano) in `<site>/inputs/soilgrids/soilgrids_<field>_<variable>_<depth>.tif`:
raw SoilGrids integers, no-data `0`, tagged with `variable`,
`soilgrids_property`, `depth`, `unit` and `scale_factor`. These tifs are
`01_calibration`'s soil input (`paths.data_input.file_SoilGrids`). There,
`wcm_swb.inputs.build_input_soil`:

1. weights each pixel by the fraction of the field area inside it (exact
   polygon intersection; Homolosine is equal-area). Fields are often only a
   few pixels wide (Ceregnano: 16 ha over 6 partially covered pixels), so a
   pixel-centre mask would be too coarse. No-data pixels are skipped;
2. converts to physical units: sand/clay `%`, SOC `g/kg`, water contents
   `m3/m3`;
3. averages the layers over depth with `field_params.soil_layer_weighting`:
   `"thickness"` (default; weights 5, 10, 15, 30 cm, i.e. the 0-60 cm profile
   mean) or `"equal"`.

The download script prints and logs the same field averages (thickness
weighting) as a check, and warns if part of the field has no data.

Field capacity is `wv0033` (water content at 33 kPa), wilting point `wv1500`
(1500 kPa).

## Reference grid (`reference_image`)

Each collection is reprojected onto the grid of one reference image before
sampling. Sentinel-1 scenes are not aligned on a common 10 m grid, so the
reference image decides which source pixel each output pixel samples: two
downloads with different references give identical `angle` but speckle-level
(~1–2 dB) per-pixel differences in backscatter. With `reference_image: null`
the second-to-last image of the requested period is used, so **changing the date range changes the S1 sampling
grid**. Pin `sentinel1.reference_image` to an EE asset id when you need
downloads to line up pixel-by-pixel — e.g. to reproduce or extend the Ceregnano
example data exactly, use the image it was built with:

```
COPERNICUS/S1_GRD_FLOAT/S1A_IW_GRDH_1SDV_20231225T170659_20231225T170724_051814_06424B_0831
```

(the full 2017-2023 period with `reference_image: null` also resolves to it). Sentinel-2 tiles share a fixed 10 m grid, so NDVI is
unaffected. The reference image used is written to the log.

## Normalization check (Mladenova et al. 2013)

The method follows the histogram-based (HIST) normalization of Mladenova et
al. (2013): σ⁰_norm = mean_ref + std_ref · (σ⁰ − mean_θ) / std_θ, matched onto
the 40° class. Mladenova et al. group by 1° incidence-angle bins over one
acquisition; here each relative orbit is one group, pooled over all pixels and
dates of the field. At field scale each orbit spans < 1° (Ceregnano: orbit 95
at 40.2 ± 0.4°, 117 at 39.0 ± 0.2°, 168 at 30.4 ± 0.1°), so the orbit is the
angle bin. Not applied: the paper's per-land-cover-class statistics (one field
= one class) and its noise-floor data-range constraint.

On the Ceregnano example (2017-2023), after normalization every orbit has exactly
orbit 95's VV/VH mean and std (spread of orbit means: 1.58 → 0 dB for VV,
1.19 → 0 dB for VH), and `VV` is an exact affine map of `VV_notnorm` per orbit.

## Optional outputs

`opt_save_single_tif` writes a GeoTIFF of the first acquisition;
`opt_save_multiday_tif` one multi-band GeoTIFF (one band per date) per
variable.

## MERIDA meteo

`download_merida.py` downloads the MERIDA HRES reanalysis (RSE; hourly,
0.04°, 323 × 329 cells over Italy, WRF driven by ERA5) from RSE's SFTP
server: `PREC` (`tp`, kg m⁻² per hour) and `T2` (`t2m`, K), one file per
variable and month (~316 MB each, uncompressed; the merged year is ~1 GB per
variable, compressed). HRES files don't agree on the time axis — 2017 files
call it `XTIME`, 2021 files `Times` with CDO absolute units
(`day as %Y%m%d.%f`, which xarray doesn't decode) — so the merge finds it by
name/`standard_name`/`axis`, decodes it, and writes it as `time`, which
`01_calibration` expects. The server's December 2021 PREC file lacks its last
step (2022-01-01 00:00, i.e. the rain of 31 Dec 23-24h), so the merged 2021
PREC has 8759 steps.

MERIDA also exists as an older 0.07° product; this pipeline uses HRES only,
and a site cache built from one isn't comparable to the other.
The site cache `study_sites/<site>/merida/MERIDA_PREC_TEMP.nc` (see
`docs/data_layout.md`) is built from these HRES files. HRES stores some missing hours as whole fields of 0 K
(2021: 2021-01-11 19:00, 2021-06-22 04:00); `wcm_swb.inputs.build_input_MERIDA`
interpolates such temperatures in time when building the cache. Precipitation
is used as is (the all-zero PREC field at 2021-06-17 04:00 stays 0 mm).

The MERIDA download has its own config,
`configuration_00_preprocessing_MERIDA_TEMPLATE.json`, because the meteo period
is independent of the Sentinel period (e.g. a spin-up year before the first
S1 date):

| Key | Meaning |
|---|---|
| `options.years`, `options.months` | every (year, month) combination is downloaded, e.g. `[2018, 2019]` × `[4, 5, 6]` |
| `options.variables` | remote variable name → label used in the merged file name (`T2` → `TEMP`, the name `01_calibration` expects) |
| `options.opt_merge` | also write one time-sorted file per variable covering all the months |
| `options.opt_overwrite` | re-download/re-merge existing files |
| `server.remote_path` | `{variable}`/`{yyyymm}` template of the file on the server |
| `server.opt_insecure` | `true` = don't verify the server's SSH host key (`curl -k`) |
| `paths.root` | the study site's folder, e.g. `${WCM_DATA_ROOT}/study_sites/Ceregnano` |
| `paths.data_output.folder_monthly` / `folder_merged` | relative to `paths.root`: `merida/monthly/` and `merida/` (git-ignored) |

Needs `curl` built with SFTP support (`curl -V` lists `sftp`). Credentials come
from environment variables, never from the config:

```bash
export MERIDA_USER=... MERIDA_PASSWORD=...
# once: trust the server's host key (curl checks ~/.ssh/known_hosts)
ssh-keyscan merida-sftp.rse-web.it >> ~/.ssh/known_hosts
cd 00_preprocessing
python download_merida.py --config configuration_00_preprocessing_MERIDA_TEMPLATE.json
python download_merida.py --config ... --merge-only    # only (re)build the merged files
```

Monthly files that already exist are skipped, so an interrupted run can be
restarted; failed months are listed at the end, and a variable with missing
months is not merged. The merged files are what
`01_calibration`'s `paths.data_input.file_MERIDA` points at (`[PREC, TEMP]`,
in that order); `01_calibration` extracts the field's grid cell from them into
`study_sites/<site>/merida/MERIDA_PREC_TEMP.nc` if that cache doesn't exist.
The merge compresses (zlib) and holds one month in memory at a time.

The merged files are sorted by time, and time stamps are rounded to the
minute: MERIDA stores time as float days since the month start, so a few
steps per month decode 1 ns early (`12:59:59.999999999`) and would otherwise
not match between PREC and T2.

Note on time: PREC is labelled at the end of the accumulation hour
(`01:00` … next month `00:00`), T2 at the instant (`00:00` … `23:00`). The
files don't state a time zone (WRF output is normally UTC), whereas the
Sentinel files use `Europe/Rome` local time (see above).
