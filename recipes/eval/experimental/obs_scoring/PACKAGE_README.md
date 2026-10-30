# Observation-space scoring (`private/obs_scoring`)

Scores gridded fields against conventional observations: radiosondes, surface stations, NEXRAD
VAD winds and aircraft winds. Any field can be scored; several systems scored on the same
observations can be compared directly.

Every filter, conversion and correction between an observation file and a figure, with the
reason for each and the known problems in the data: `METHOD.md`.

## In Python

```python
import pathlib

import pandas as pd

from private.obs_scoring import fields, observations
from private.obs_scoring.sample import required_fields, sample
from private.obs_scoring.score import score
from private.scripts import region_masks

t = pd.Timestamp("2025-07-08T00")
mirror = pathlib.Path("MIRROR")          # written by `obs_scores fetch`
obs = observations.conventional_obs(mirror, t)
# or: observations.prepbufr_obs(ARCHIVE, t, "vad")   (NNJA PrepBUFR parquet archive)

store = fields.open_fields("zarr:ANALYSIS.zarr")    # or "era5", "wb2:PATH", or your own source
values = store.at(t, required_fields(obs))          # xr.Dataset with lat, lon coords
regions = region_masks.load_regions()
table = score(obs, sample(values, obs), regions)                          # per region, field
by_dataset = score(obs, sample(values, obs), regions, by=("dataset", "field"))
```

Any field you can put in an `xr.Dataset` works the same way; `required_fields(obs)` lists the
variables `sample` reads (`d2m` and `sp` for `q2m`, `U`/`V` for wind speed).

| module | role |
| --- | --- |
| `observations` | observation tables (one row per value: `time, lat, lon, station, dataset, field, obs`, plus `pressure, height` for profile winds); NNJA-AI (from the raw mirror `fetch` writes) and PrepBUFR sources; quality control and unit conversion on read |
| `sample` | `required_fields(obs)` and `sample(fields, obs)`: model value per observation row |
| `score` | `score(obs, values, regions, by=("field",))`: RMSE, MAE, bias and count per region and `by` group |
| `fields` | sources for our stores: `open_fields("zarr:PATH" \| "era5" \| "wb2:PATH").at(t, names)` |
| `report` | pooling over times, pairs across systems, gross check, profile plot |

`fields` is an `xr.Dataset` for one time with `lat`/`lon` coordinates on a regular grid and
variables named like the HealDA channels (`Z500`, `T850`, `tas`, `d2m`, ...; `Z` as
geopotential). Any source that produces this can be scored; a HEALPix store after
regridding to lat-lon (earth2grid), for example.

## Command line (`private/scripts/analysis/obs_scores.py`)

For each 6-hourly analysis time t:

1. take the observations in [t - 30 min, t + 30 min), radiosondes from t - 90 min (time
   match, below);
2. read each system's fields valid at t;
3. interpolate the fields to the observation points (bilinear, periodic in longitude);
4. compute RMSE, MAE, bias and count per region and field.

`compare` then pools the per-time scores over the times every system has. `fetch` only
downloads: the NNJA-AI day files are mirrored unchanged, and every rule (quality marks, unit
conversion, time window, station selection) is applied when `score` reads them, so a rule
change needs no new download. VAD and aircraft winds are read straight from the PrepBUFR
archive.

| command | does | writes |
| --- | --- | --- |
| `fetch` | mirrors the NNJA-AI day files (~15 MB per day) | `CACHE/<dataset>/YYYYMMDD/*.parquet`, or a `NO_FILE` marker |
| `score` | scores each system (`--obs conventional` from the mirror, `vad`/`aircraft` from the archive) | `OUT/{system}/scores_{stations}.zarr`; with `--pairs`, `OUT/pairs/{stations}/{system}.parquet` |
| `compare` | pools systems over common times, prints tables, plots profiles | `OUT/profiles_{label}.png` |

Score stores are zarr, one per system: `rmse, mae, bias, count` on
`(time, match, region, [dataset,] field)` (`dataset` with `--by-dataset`), `units` on `field`,
`system` and `stations` as attributes; `count` is 0 where there were no observations.
`report.from_dataset` returns a store as a table and `report.pooled` pools it over times.
A pairs file holds one row per observation over all cycles (`cycle, time, lat, lon,
station, dataset, field, obs`, plus `pressure, height` for VAD and aircraft) and the
system's `model` value, one row group per cycle. Pairs hold the `synoptic` selection; the
`strict` rows are those with `time` in [`cycle` - 30 min, `cycle` + 30 min).

## What can be scored

Observations (`score --obs`):

| `--obs` | source | fields |
| --- | --- | --- |
| `conventional` (default) | NNJA-AI radiosondes (`conv-adpupa-NC002001`) and surface stations (`conv-adpsfc-NC000001/02/07/101`), from the `fetch` mirror | `Z`, `T`, `U`, `V`, wind speed `S` at 10-1000 hPa mandatory levels; `Q` without 10 hPa; `tas`, `sp`, `uas`, `vas`, wind speed `sfcwind`, `q2m` |
| `vad` | NNJA PrepBUFR archive, report type 224 | `U{p}`, `V{p}` in 25 hPa bins |
| `aircraft` | NNJA PrepBUFR archive, types 230, 231, 233, 234, 235 | `U{p}`, `V{p}` in 25 hPa bins |

Systems (`score --system NAME=KIND`):

| kind | reads |
| --- | --- |
| `zarr:PATH` | a lat-lon zarr store with one `(time, lat, lon)` array per field, named as in HealDA (our analysis stores, `ifs_hres_t0_025.zarr`). Times with `run_id < 0` and fields with `is_valid` false are skipped. |
| `era5` | the ERA5 hourly packs named in `.env` |
| `wb2:PATH` | a WeatherBench2-layout store (CF names, `level` dimension) |

Units: `Z` in gpm, `T`/`tas` K, winds m/s, `Q`/`q2m` kg/kg, `sp` Pa.

## Matching rules

They follow NOAA PSL eagle-tools `obs-metrics`:

- radiosondes only at mandatory levels, with no vertical interpolation, every level at the
  station position and report time. The NNJA-AI table has no per-level drift (the
  text-format dump it comes from never carried it); balloons drift a median ~25 km and up to
  ~200 km (99th percentile), so upper-level errors include this displacement. Drifted
  positions exist in PrepBUFR (`XDR`, `YDR`, `HRDR`): copied from the measured drift for
  stations that report in BUFR, and estimated from the reported winds with a 5 m/s ascent
  rate for stations that report only in the text format. See `private/docs/radiosonde_drift.md`;
- surface stations with no elevation correction (see `--elevation-correct` below);
- a value is dropped when its quality mark is > 2, and a missing mark is kept;
- dewpoint becomes specific humidity with the Magnus formula (`healda.utils.thermo`).

One deviation: observed radiosonde geopotential `GP10` is divided by 9.8, the factor NCEP
encodes it with, not 9.80665.

VAD and aircraft winds are placed at their reported position and time. The model is
interpolated between the two pressure levels around the report, linearly in log p
(`--vertical pressure`, the default). Or, with `--vertical height`, linearly in height
between the levels whose model geopotential heights bracket the report height. The pressure
of a VAD report is the standard-atmosphere pressure of its height, so `--vertical height`
places VAD winds more faithfully.

## Radiosonde time match

Every score is written for two selections of the radiosonde reports, along `match`:

| `match` | radiosondes kept | surface stations |
| --- | --- | --- |
| `strict` | report time in [t - 30 min, t + 30 min), as eagle-tools | [t - 30 min, t + 30 min) |
| `synoptic` | report time in [t - 90 min, t + 30 min) | the same as `strict` |

Many 00 UTC soundings carry a 23:00 stamp of the day before (NNJA-AI stamps a radiosonde with
the whole-hour YYGG of its report, and these report 23; `METHOD.md` section 4), and fall outside
the strict window. At
2025-07-01 00 UTC that is 140 stations: T500 is scored at 343 stations with `strict` and 483
with `synoptic` (surface counts identical). Since 2026-05-02 the NNJA-AI day files end
before 21 UTC, so those soundings are absent and the two selections coincide at 00 UTC.
`strict` reproduces the scores before `match` existed exactly (checked on 2026-06-01 and
2025-07-01 00 UTC, all regions and fields). `compare --match` picks one; the default is
`synoptic`. VAD and aircraft winds lie within ±30 min, so both selections are equal.

## Beyond eagle-tools (opt-in, off by default)

eagle-tools does neither of these; both are off unless asked for, so default scores stay
comparable with eagle-tools.

- `score --elevation-correct`: moves the model's `tas` and `sp` from the model terrain to
  the station elevation before comparing. `tas` uses the standard lapse rate (6.5 K/km);
  `sp` is hydrostatic through the layer between them, at the virtual temperature of the
  model's 2 m temperature minus the lapse rate over half the height change and its 2 m
  humidity (`healda.utils.thermo.pressure_at_height`). The
  terrain is ERA5's (`fields.era5_orography`) for every system; for IFS HRES, whose `sp`
  refers to its own finer terrain, that is an approximation. Station elevation is `SELV`;
  the 9999 sentinel counts as missing and such rows are not corrected. Without it, raw `sp`
  is dominated by the terrain mismatch (RMSE 2-4 kPa).
- `score --dedupe-surface`: one surface report per field, time and position (0.01 deg),
  preferring SYNOP fixed, then BUFR SYNOP, mobile SYNOP, METAR. Airports that file both
  SYNOP and METAR are otherwise counted twice.

## Comparing systems

- `--region`: any of the 17 regions of the earth2studio eval recipe
  (`third_party/earth2studio/recipes/eval/scorecard/cfg/regions/standard.yaml`). These are
  the same regions as the forecast scorecards. Observations are assigned by their own
  position.
- `--gross K`: rescores from the pairs on the observations every system has. It drops an
  observation when the median over systems of its departure is more than K scaled MADs from
  that field's median at that time. Use it for comparisons between systems. The observation
  sets carry unflagged gross errors (radiosonde heights 500 m off, `q2m` of -1.6 kg/kg).
  It reads the pairs written by `score --pairs --stations all` and applies `--stations`
  itself.
- `--by-dataset` (on `score` and `compare`): separate scores per observation dataset
  (radiosondes, SYNOP fixed, SYNOP mobile, METAR, BUFR SYNOP), written as
  `{system}/scores_{stations}_by_dataset.zarr`. Pooled surface scores mix all four surface datasets,
  and airports that file both SYNOP and METAR are counted in both.

Pairs must hold the same observations for every system; `compare` refuses otherwise.

## Station holdout

`ObsConfig.nnja_withhold_stations` (or `inference.py --withhold-stations`) drops every
report of a fixed 10% of radiosonde, pibal and land stations (crc32 of the station id,
`is_withheld` in `observations/loaders/nnja_conventional.py`). `score --stations withheld`
scores exactly those stations.

## Running

Prerequisites:

- `nnja-ai` (`uv sync --group private`) for `fetch`;
- the pinned earth2studio checkout for regions (`bash private/scripts/campaign/bootstrap_e2s.sh`);
- run on a compute node in the project container.

```bash
S=private.scripts.analysis.obs_scores
# radiosondes and surface stations
python -m $S fetch --start 2025-07-08 --end 2025-08-05T18 --cache MIRROR
python -m $S score --start 2025-07-08 --end 2025-08-05T18 --cache MIRROR --out OUT --pairs \
    --system healda=zarr:ANALYSIS.zarr --system era5=era5 --system hres=zarr:ifs_hres_t0_025.zarr
python -m $S compare --out OUT --gross 5                  # add --region, --by-dataset, --fields, --match
# VAD winds from the PrepBUFR archive (--archive defaults to the .env NNJA root)
python -m $S score --obs vad --vertical height --start 2025-07-08 --end 2025-08-05T18 \
    --out OUT_VAD --system healda=zarr:ANALYSIS.zarr --system era5=era5
```

Cost: `fetch` ~15 MB per day of observations; `score` ~0.5 s per analysis time per system
plus ~15 s per run (region table, ERA5 readers), parallel over `--workers` (default 16).


## Validation

- **Against eagle-tools itself** (unmodified, HealDA v2, 2025-07-08, four times):
  - all 316,400 of its observation pairs appear here with identical observed values;
  - per-field RMSE agrees to five significant figures;
  - the only differences are the 454 points between 359.75 and 360 E, which eagle leaves
    unscored, and `tas` poleward of 80 degrees (up to 0.48 K, spherical vs lat-lon
    interpolation).
- **Wind speed against eagle-tools** (same day and store, eagle-tools rerun with `wind_speed`
  and `10m_wind_speed`): identical observations; model speed within 0.0015 m/s (0.056 m/s
  poleward of 80 degrees); eagle-tools' own RMSE and ours agree to 5e-4 relative, the
  difference being the points near 360 E that eagle leaves unscored.
- **The 9.8 factor** was checked three ways:
  - BUFR heights are whole metres only when divided by 9.8;
  - the text and BUFR reports of the same soundings agree;
  - model-minus-observation bias is flat with height only with 9.8.
