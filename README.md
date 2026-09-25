# WCM_SWB_Irri

A Water Cloud Model coupled to a Soil Water Balance model (WCM-SWB), calibrated
per field with Bayesian MCMC (PyMC) against Sentinel-1 backscatter and
Sentinel-2 NDVI, to estimate irrigation from satellite data. No in-situ
irrigation data are needed: the model is calibrated on satellite observations
and meteorological forcing only.

This repository contains the code and the data-processing workflow developed
for the paper:

> **Field-scale irrigation estimation using Sentinel-1 and Sentinel-2 data
> within a soil water balance model coupled to the Water Cloud Model**
> Natali M. et al., *Remote Sensing of Environment*, 346, 115639 (2026).
> DOI: [10.1016/j.rse.2026.115639](https://doi.org/10.1016/j.rse.2026.115639)

If you use this software, please cite **both** the paper and the software
(see [How to cite](#how-to-cite) and [`CITATION.md`](CITATION.md)).

The repository contains code only: all input data (Sentinel-1/2, SoilGrids,
MERIDA) are downloaded on demand by `00_preprocessing`. One worked example is
set up, the **Ceregnano** field (Italy): only its field polygon is included
(data source and credits in `study_sites/Ceregnano/README.md`).

## Pipeline

```
00_preprocessing  →  01_calibration  →  02_analysis
(download Sentinel-1/2,  (per-field Bayesian   (irrigation estimates,
 SoilGrids, MERIDA)       MCMC calibration)     uncertainty, parameters)
```

Shared code lives in the `wcm_swb` Python package. Each stage has its own
README with the full configuration reference.

## Installation

Requires [conda](https://github.com/conda-forge/miniforge) (Miniforge
recommended). From the repository root:

```bash
conda env create -f environment.yml
conda activate wcm_swb
pip install -e .
python -m ipykernel install --user --name wcm_swb --display-name "Python (wcm_swb)"
```

The last line registers the Jupyter kernel used by the `02_analysis`
notebooks. This environment is enough for calibration and analysis.

To download new data (`00_preprocessing`), create the second environment,
which adds Google Earth Engine and geospatial libraries:

```bash
conda env create -f environment-preprocessing.yml
conda activate wcm_swb_preprocessing
pip install -e .
```

### Data root

All configuration files resolve paths relative to `WCM_DATA_ROOT`. It
defaults to the repository root (data then go to `study_sites/`); set it
only if you keep your data elsewhere:

```bash
export WCM_DATA_ROOT=/path/to/data
```

## Quick start: the Ceregnano example

First download the inputs for the Ceregnano field (needs the Earth Engine
project and MERIDA login described in [Credentials](#credentials)); the
templates are already set up for Ceregnano (Sentinel-1/2 2017-2023, MERIDA
2021):

```bash
conda activate wcm_swb_preprocessing
cd 00_preprocessing
set -a; source ~/.config/wcm_swb/credentials.env; set +a
python download_all.py --config configuration_00_preprocessing_TEMPLATE.json \
    --merida-config configuration_00_preprocessing_MERIDA_TEMPLATE.json
```

Then calibrate:

```bash
conda activate wcm_swb
cd ../01_calibration
python run_calibration.py --config configuration_01_calibration_TEMPLATE.json
```

This calibrates the Ceregnano field for 2021 (year-round mode) and writes the
results to `01_calibration/output/yearround/ceregnano/`. A full run takes a
while; for a quick test, copy the template and reduce
`calibration.run_params` (e.g. `n_tunes`, `n_draws` to 20 and `n_chains`,
`n_cores` to 2).

Then open the notebooks in `02_analysis/` (kernel `Python (wcm_swb)`) to plot
soil moisture, backscatter fit and irrigation estimates with their
uncertainty.

## Running the pipeline on your own site

### 1. Data you must provide

| What | Where | Notes |
|---|---|---|
| **Field polygons** | `study_sites/<site>/shapefile/<field>.geojson` | One polygon per file, WGS84 (EPSG:4326) recommended. The file name (without extension) is the field id used everywhere else (`{opt_field}` in the configs). |
| **Google Earth Engine project** | `GEE_PROJECT`, see [Credentials](#credentials) | For the Sentinel-1/2 downloads. Needs a Google Cloud project registered for Earth Engine; the first run opens the authentication flow. |
| **MERIDA credentials** | `MERIDA_USER`, `MERIDA_PASSWORD`, see [Credentials](#credentials) | Hourly precipitation and temperature from the MERIDA HRES reanalysis (RSE). Covers **Italy only**. Needs `curl` with SFTP support; trust the server key once with `ssh-keyscan merida-sftp.rse-web.it >> ~/.ssh/known_hosts`. |

Soil properties (SoilGrids 2.0) are downloaded without any account.

#### Credentials

Keep the Earth Engine project id and the MERIDA login in a private file
**outside this repository** (never in a config, script or notebook here),
e.g. `~/.config/wcm_swb/credentials.env`:

```bash
GEE_PROJECT=<your-google-cloud-project-id>
MERIDA_USER=<your MERIDA user>
MERIDA_PASSWORD='<your MERIDA password>'
```

Make it readable only by you and load it in the shell before downloading:

```bash
chmod 600 ~/.config/wcm_swb/credentials.env
set -a; source ~/.config/wcm_swb/credentials.env; set +a
```

For Earth Engine, register a Google Cloud project
(<https://developers.google.com/earth-engine/guides/access>) and authenticate
once with `earthengine authenticate` (or let the first download open the
browser flow); `GEE_PROJECT` is read by `options.gee_project` in the
preprocessing config. The download scripts read these variables from the
environment and never write them to files or logs; `download_all.py` checks
they are set before starting.

Everything else is downloaded by `00_preprocessing` into the site folder:

```
study_sites/<site>/
├── shapefile/<field>.geojson          # you provide this
├── sigma0/s0_<field>_....nc           # Sentinel-1 backscatter (VV, VH, HIST-normalized)
├── ndvi/ndvi_<field>_....nc           # Sentinel-2 NDVI
├── inputs/soilgrids/soilgrids_<field>_<var>_<depth>.tif   # SoilGrids, 0-60 cm
└── merida/MERIDA_{PREC,TEMP}_<start>-<end>.nc              # MERIDA meteo
```

See `docs/data_layout.md` for details.

### 2. Download the data (`00_preprocessing`)

Copy the two config templates and set, at least:

- `configuration_00_preprocessing_TEMPLATE.json`: `paths.root` (your site
  folder), `options.start_date` / `end_date`, and the output file names;
- `configuration_00_preprocessing_MERIDA_TEMPLATE.json`: the same
  `paths.root`, and `options.years` / `months` covering your calibration period.

Then download everything in one (resumable) run:

```bash
conda activate wcm_swb_preprocessing
cd 00_preprocessing
set -a; source ~/.config/wcm_swb/credentials.env; set +a   # see Credentials
python download_all.py --config my_preprocessing.json --merida-config my_merida.json
```

Existing files are skipped, so an interrupted run can simply be restarted;
`--dry-run` lists what is present or missing. The individual download scripts
can also be run on their own. See `00_preprocessing/README.md`.

### 3. Calibrate (`01_calibration`)

Copy `configuration_01_calibration_TEMPLATE.json` and set `paths.root`,
`options.opt_field`, `options.start_date` / `end_date`, and the input file
names from step 2. Choose the calibration mode with `options.opt_mode`:

- `"yearround"`: one calibration over the whole period;
- `"seasonal"`: bare-soil parameters first, then vegetation and irrigation
  parameters (split on an NDVI threshold).

```bash
conda activate wcm_swb
cd 01_calibration
python run_calibration.py --config my_calibration.json               # one field
python run_calibration.py --config my_calibration.json --field <id>  # override opt_field
python run_calibration_batch.py --config my_calibration.json         # every field of the site
```

Each run writes, per field, a posterior summary, the full trace, diagnostic
plots, the simulated model states with a 5-95% credible band, and the
posterior irrigation ensemble. See `01_calibration/README.md` for the outputs
and `docs/model_conventions.md` for the model parameters and units.

### 4. Analyse (`02_analysis`)

Copy `configuration_02_analysis_TEMPLATE.json`, point `paths.root_output` at
the calibration output folder, and set the file name in the first cell of the
notebooks:

| Notebook | Shows |
|---|---|
| `02_single_field.ipynb` | One field: backscatter fit, NDVI, soil moisture, irrigation, meteo. |
| `02_irrigation.ipynb` | Irrigation over all fields (1/7/15-day and seasonal totals) with uncertainty. |
| `02_parameters.ipynb` | Calibrated parameters across fields. |

See `02_analysis/README.md`.

## Repository layout

- `wcm_swb/`: the Python package (model, calibration engine, data readers,
  plotting).
- `00_preprocessing/`, `01_calibration/`, `02_analysis/`: pipeline scripts,
  notebooks and configuration templates.
- `study_sites/Ceregnano/shapefile/`: field polygon of the example site
  (the other inputs are downloaded into `study_sites/Ceregnano/`).
- `docs/`: data layout and model conventions.
- `tests/`: unit tests (`pytest tests/`).

## How to cite

This software was developed for, and must be cited together with, the paper
above. When you use it, please cite both the paper and the software (Zenodo):
the full references, with BibTeX, are in [`CITATION.md`](CITATION.md).
Machine-readable metadata are in [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button) and [`.zenodo.json`](.zenodo.json).

## License

GPL-3.0, see `LICENSE`.
