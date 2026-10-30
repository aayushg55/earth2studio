<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observation scoring (reference snapshot)

A read-only copy of the observation scorer that the eval recipe's online observation scoring
(`recipes/eval/src/obs_online.py`) calls. In a run, `scoring.online.observations.scorer_root`
points at a HealDA checkout and the recipe imports its `private.obs_scoring` package; this
directory holds that package so the method can be read in one place. It is not imported by the
recipe and does not run on its own.

| file | what it is |
| --- | --- |
| `observations.py` | reading NNJA-AI conventional observations (radiosondes, SYNOP, METAR, BUFR SYNOP), quality marks, unit conversion, radiosonde time match |
| `fields.py` | the gridded systems being scored (ERA5, analysis stores) as one `xr.Dataset` per time |
| `sample.py` | interpolating a field to observation points; station-height change |
| `score.py` | RMSE, MAE, bias and counts per region, field and time match; `score_cycle` used online |
| `report.py` | per-report tables and the consensus gross check (`consensus_keep`) |
| `scripts/obs_scores.py` | command line: mirror the observations, score analyses, compare systems |
| `scripts/obs_figures.py` | aggregating per-report pairs from forecast runs into sums, protocols, figures and scorecards |
| `METHOD.md` | every filter, conversion, match and check, and why |
| `BAD_OBSERVATIONS.md` | gross errors found in the observations and what the check removes |
| `PACKAGE_README.md` | the package's own README |

Dependencies outside this directory (from the HealDA code base): `healda.utils.thermo` (gravity,
dewpoint to specific humidity, the 2 m temperature lapse-rate correction),
`healda.datasets.era5_statics` (ERA5 orography), `healda.datasets.da.hourly_latlon_dataset`
(ERA5 reader), `healda.utils.cache` (atomic cache writes), `healda.config.environment` (PrepBUFR
archive location, VAD and aircraft winds only), `healda.observations.loaders.nnja_conventional`
(withheld stations). The scripts also use the region masks (earth2studio's scorecard regions),
a station-map module and the campaign's start-time sampling.
