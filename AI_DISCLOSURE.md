# AI usage disclosure statement

This repository was prepared with the assistance of a generative AI coding
tool: **Claude Code** (Anthropic), using the model **Claude Opus 5.5**,
between 22 and 25 September 2026. Commits in which the tool took part carry a
`Co-Authored-By: Claude` trailer.

## What the AI tool did

Working under the author's direction, the tool:

- reorganised the author's research code (the WCM-SWB model and its
  calibration scripts) into the `wcm_swb` Python package and the
  `00_preprocessing` / `01_calibration` / `02_analysis` pipeline, including
  replacing duplicated code with one shared calibration engine;
- wrote the data download scripts (Sentinel-1/2 via Google Earth Engine,
  SoilGrids, MERIDA) and the configuration templates;
- wrote the unit tests, the documentation (README files, `docs/`) and the
  citation metadata (`CITATION.md`, `CITATION.cff`, `.zenodo.json`);
- found and fixed bugs, and ran the tests and the end-to-end example runs
  used to check the code.

## What the AI tool did not do

- The scientific method, the model (soil water balance coupled to the Water
  Cloud Model) and its calibration approach were developed by the authors of
  the associated paper (see [`CITATION.md`](CITATION.md)), not by the AI tool.
- The AI tool did not make scientific or design decisions on its own: the
  author made them (e.g. which data, parameters, priors and calibration modes
  to use) and directed and reviewed the work.
- This statement covers this repository only, not the associated paper.

## Responsibility

The author takes full responsibility for the content of this repository,
including the parts produced with AI assistance.
