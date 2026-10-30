# Observation verification: every step, filter and correction

What happens to an observation between the NNJA-AI file and a number in a figure, and why. It
covers analyses (`cli.py`) and forecasts (online scoring in `src/obs_online.py`, figures by
`figures.py`). Module-level detail and the command lines are in `README.md` beside this file.

The scorer's defaults follow NOAA PSL eagle-tools `obs-metrics`, so default scores can be
compared with it. Every step that departs from eagle-tools is optional and named in the output
(`match`, `qc`, `elevation` columns), and every variant is computed side by side. The figures
draw one of two protocols, `ecmwf` (default) and `eagle` (section 6).

## 1. Observations

| set | NNJA-AI dataset | fields |
| --- | --- | --- |
| radiosondes | `conv-adpupa-NC002001` (TAC rawinsonde) | `Z`, `T`, `U`, `V`, speed `S` at the mandatory levels 10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000 hPa; `Q` at the same levels except 10 hPa |
| surface, SYNOP fixed | `conv-adpsfc-NC000001` | `tas`, `sp`, `uas`, `vas`, `sfcwind`, `q2m` |
| surface, SYNOP mobile | `conv-adpsfc-NC000002` | as above |
| surface, METAR | `conv-adpsfc-NC000007` | `tas`, `uas`, `vas`, `sfcwind` (no pressure or dewpoint) |
| surface, BUFR SYNOP | `conv-adpsfc-NC000101` | `tas`, `sp`, `uas`, `vas`, `sfcwind`, `q2m` |

`cli.py fetch` mirrors the day files unchanged into `NNJA_AI_MIRROR`; every rule below is
applied when they are read, so a rule change needs no new download. Campaign jobs have no
network: `fetch --config <campaign config>` fetches the days a run scores, and
`run_campaign.sh` runs it with `--check` to stop a run whose days are missing.

## 2. Filters on read (`observations.py`)

| filter | rule | why |
| --- | --- | --- |
| quality mark | a value whose quality mark is > 2 is dropped; a missing mark is kept | the NCEP marks: 0-2 accepted, 3+ suspect, rejected or not used. As eagle-tools. In NNJA-AI v1 the marks are dump marks and are only ever missing or 14 (SDM purge), so this drops the purges only (section 7.3) |
| mandatory levels | radiosonde values only at the levels above; no significant levels | as eagle-tools; one value per level and station |
| missing values | non-finite values are dropped | |
| station pressure 0 | `sp` <= 0 Pa is missing (BUFR SYNOP carries 0 with no quality mark) | sentinel, not a pressure |
| station elevation | `SELV` >= 9000 m is missing (the 9999 sentinel) | used only by the 2 m temperature variants |
| time window | section 4 | |
| station selection | `all` by default; `withheld` / `kept` for the 10 % station holdout | holdout experiments only |

Nothing else is screened on read: no range check, no background check, no duplicate removal (see
`--dedupe-surface`), no thinning. Section 7 lists the gross errors that pass these filters.

## 3. Conversions on read

| quantity | conversion | why |
| --- | --- | --- |
| radiosonde height | `GP10` / 9.8 (not 9.80665) | NCEP encodes reported gpm × 9.8; checked three ways (`README.md`, Validation) |
| winds | speed and direction to `u = -s sin(d)`, `v = -s cos(d)`; speed kept as `S{p}` / `sfcwind` | |
| humidity | dewpoint to specific humidity with the Magnus formula at the level pressure (radiosondes) or the reported station pressure (surface); no `q2m` without a station pressure | `thermo.specific_humidity_from_dewpoint` |
| longitude | to [0, 360) | |
| time | `OBS_TIMESTAMP` as UTC | NNJA-AI truncates radiosonde times to the hour (section 4) |

## 4. Time matching

For valid time t, surface reports in [t - 30 min, t + 30 min) are scored. Radiosondes are read
from t - 90 min and scored in two selections side by side (`match`):

| `match` | radiosondes | why |
| --- | --- | --- |
| `strict` | [t - 30 min, t + 30 min) | as eagle-tools (protocol `eagle`) |
| `synoptic` (protocol `ecmwf`) | [t - 90 min, t + 30 min) | soundings stamped 23:00 of the day before miss the strict window (below); at 2025-07-01 00 UTC that is 140 stations |

The radiosonde stamp (`OBS_TIMESTAMP`, which NNJA-AI describes only as "Observation timestamp")
is always a whole hour and equals `MSG_DATE`. In 2024-12-30..2026-09-26 (555,482 rows):

- Rows with the TEMP text (`RAWRPT`, 88.5 %): the stamp is the day and hour of the report's own
  YYGG group in 99.95 % of rows. The stamp minus the launch time (`UASDG.UALNHR`, `UALNMN`) has
  a median of +30 min and a 1-99 % range of -23 to +76 min.
- Text rows stamped 23 UTC give 23 as their YYGG.
- Rows without text (11.5 %, all from 87 stations in WMO blocks 50-59): they carry no launch or
  nominal time and are stamped 11, 23 or 17 UTC. How that stamp is formed is not established.

Source: `BAD_OBSERVATIONS.md` section 5.

Radiosondes launch at 00 and 12 UTC; at 06 and 18 UTC only a few, mostly Northern Hemisphere,
stations report. Upper-air curves and scorecards therefore use only leads valid at 00/12 UTC;
the 06/18 UTC points are a different, small station set and alternate with the 00/12 points.

## 5. Model values at the observations (`sample.py`)

- Horizontal: bilinear in latitude and longitude on the model's regular grid, periodic in
  longitude.
- Radiosondes: the model's value on the same pressure level at the station position and report
  time; no vertical interpolation and no balloon drift (the NNJA-AI table has no per-level
  position). Balloons drift a median ~25 km and up to ~200 km, so upper-level errors include
  that displacement.
- `tas`: the model's 2 m temperature at the model terrain height, unless corrected (section 6).
- `q2m`: from the model's `d2m` and `sp`, as for the observation.
- `sfcwind` and `S{p}`: the speed of the interpolated `u`, `v`.
- Every IC source is scored against the same observation files, so every system sees the same
  reports.

Forecasts are scored inside the run at every 6 h lead step, at valid time init + lead, without
storing the forecast. Each run writes per-report pairs (`obs_scores/pairs`) and per-region
score tables (`obs_scores/parts`).

## 6. Processing for the figures (`figures.py`)

`accumulate` computes every variant below side by side; `plot` and `scorecards` draw one
protocol, so switching never needs a rerun.

### Protocols

| | `ecmwf` (default) | `eagle` |
| --- | --- | --- |
| radiosonde match (section 4) | `synoptic` | `strict` |
| consensus gross check (step 4) | on (`--no-gross`: off, file suffix `_nogross`) | off |
| 2 m temperature (step 3) | moved to the station elevation at 6.5 K/km; only stations within 250 m of the ERA5 terrain | at the model terrain, every station |
| file suffix | `_ecmwf` | `_eagle` |

`ecmwf` approximates ECMWF's verification of 2 m temperature (below) and removes the gross
errors of section 7; use it to compare systems. `eagle` reproduces NOAA PSL eagle-tools
`obs-metrics` report for report, except that:

- radiosonde height is `GP10` / 9.8, the factor NCEP encodes it with; eagle-tools divides by
  9.80665 (section 3);
- the points between 359.75 and 360° E, which eagle-tools leaves unscored, are scored (454 of
  316,400 pairs on 2025-07-08, `README.md`, Validation);
- every system is scored on the same reports and inits (step 2); eagle-tools scores each
  system on its own.

With these three, RMSE agrees with eagle-tools to five significant figures (`README.md`,
Validation). Both protocols share eagle-tools' quality marks (> 2 dropped, missing kept),
mandatory levels, no vertical interpolation or balloon drift, and Magnus humidity.

| | eagle-tools | `eagle` | `ecmwf` |
| --- | --- | --- | --- |
| outlier screening | none | none | consensus 5-MAD check |
| radiosonde time window | ±30 min | ±30 min | from 90 min before |
| radiosonde geopotential | `GP10` / 9.80665 | `GP10` / 9.8 | `GP10` / 9.8 |
| reports per system | each system on its own | the same reports and inits for every system | the same |
| longitudes 359.75-360° E | unscored | scored | scored |
| 2 m temperature | at the model terrain | at the model terrain | at the station elevation, stations within 250 m |
| intervals | none | paired day-bootstrap 95 %; scorecards paired t-test over days | the same |

### Steps, per init

1. **Common inits**: only inits every IC source completed.
2. **Same reports**: the IC sources' pairs for an init must hold the same reports in the same
   order (observed value, position, field, lead, times); `accumulate` stops otherwise. A report
   is dropped when any IC source has no finite model value there.
3. **2 m temperature variants** (column `elevation`, `tas` only; every other field has
   `none`). With change = station elevation - ERA5 terrain at the station (0.25° orography,
   bilinear; `sample.station_height_change`):
   - `none`: the model's 2 m temperature at the model terrain, every station (eagle-tools).
   - `lapse6.5_win250`: only reports with |change| <= 250 m (reports without an elevation,
     0.2-0.3 % of `tas` reports, ~500-600 stations in 2025-26, are outside), and the model
     moved to the station elevation, `tas - 0.0065 K/m × change`
     (`thermo.temperature_at_height`). The shift is identical for every IC
     source: all of them are forecast by the same model. The terrain a model's 2 m temperature
     refers to depends on its training data; the ERA5 terrain is used for every model, which is
     exact only for one trained on ERA5.
   - Why: a model's 2 m temperature refers to its grid cell's terrain height, a station's to
     its own; at 6.5 K/km a station 500 m below the model terrain reads 3.25 K warmer than a
     perfect forecast. The window removes the stations where that difference, and the
     correction for it, are largest.
   - Measured (2025, WeatherNext 2, consensus check, HealDA v2 r14 IC, global +24 h RMSE):
     2.26 K uncorrected, 2.06 K at 6.5 K/km, 1.99 K in the 250 m window, 1.96 K with both;
     the 250 m window keeps 84-85 % of the stations and 88 % of the reports. The correction
     raises the warm bias over Europe at 00 UTC (+0.32 → +0.50 K) and over North America at
     12 UTC (+0.63 → +0.80 K); its cause has not been established.
   - Variants measured and no longer computed: 5.5 K/km (the bias rise stays,
     +0.58-0.61 K and +0.77-0.78 K); a 100 m window (keeps 67-69 % of the stations and 72 % of
     the reports); the lapse rate without a window and the window without a lapse rate
     (numbers above); dropping reports more than 15 K from ERA5 at the station height (removes
     0.06-0.2 % of the reports, 96-100 % of which the consensus check also removes; the check
     removes 0.3-0.8 %).
   - Practice elsewhere: ECMWF verifies 2 m temperature with a 6.5 K/km correction, against only
     the SYNOP stations within 250 m of the model height, rejecting observations more than 15 K
     from the analysis (ECMWF Newsletter 178, 2024); its 4D-Var uses 5.5 K/km and stations from
     400 m below to 200 m above the model, and drops departures > 7.5 K from the background.
     `lapse6.5_win250` approximates the verification (all station types, the ERA5 terrain
     instead of each model's, the consensus check instead of the 15 K screen).
   - Online: `elevation_correct: true` in an experiment's `observations` block applies the same
     correction (`sample.py`, the same height-change and lapse-rate helpers; every station, no
     window; and the hydrostatic one to `sp`) inside the run; `fc5d_obs_2025` /
     `fc5d_obs_2026` were scored without it.
4. **Consensus gross check** (column `qc`: `gross`, or `none`, every report as eagle-tools;
   `report.consensus_keep`): per field and valid time (one init's forecasts at one lead), c =
   the median over IC sources of model - obs; a report is dropped when
   |c - median(c)| > 5 × 1.4826 × median(|c - median(c)|). Computed separately for each `match`
   and 2 m temperature variant. The analyses' lead 0 uses the same rule per field and cycle,
   as does `cli.py compare --gross`.
   - Why: NNJA-AI passes reports that are grossly wrong for every model (section 7). In 2025
     they dominate the radiosonde RMSE: global Z250 RMSE is ~75-90 m at every lead without the
     check and rises from 14 to 41 m with it (WN2), and the upper-air curves alternate by lead
     without it. The check uses the consensus of the systems compared, and removes the same
     reports for all of them.
   - Effect: global U850 RMSE at 2025-03-21 00 UTC 14.0 → 3.3 m/s (WN2, HealDA v2 r14 IC); the
     12 h alternation is gone (mean absolute second difference of global Z250 RMSE over leads
     4.4 → 0.26 m); summed over the region, match and lead cells of the summer-2026 WN2
     experiment, 0.48 % of report entries are dropped.
   - Limits:
     - With one or two reports in a group nothing can be dropped. Groups pool every station
       worldwide, so groups with fewer than 5 reports are rare: 3 bad reports in 2025, 29 in 2026.
     - The larger gap is the 06/18 UTC radiosonde groups of 6-15 reports. When several soundings
       in one group are wrong by the same amount, the median and MAD follow them and they are
       kept. This is the case for 61052, 61024 and 61090, whose heights are 300-800 m low. In
       2026 the check kept 98 Z reports more than 200 m from the consensus, all at 06/18 UTC;
       in 2025 it kept 116 at 06/18 UTC (section 7). The figures use radiosondes at 00/12 UTC
       only (section 4), so this affects the sums, not the curves.
     - A wrong report that every model agrees with is invisible to the check. One that only
       some systems disagree with is kept if the median agrees.
     - The check also removes departures that are not gross errors: persistent 50-125 m height
       offsets at some stations, and mountain-station `tas`. ERA5 alone flags 46-58 % of the
       removed reports.
5. **Sums** (`sums/forecast.parquet`; the analyses' lead 0 in `sums/analysis_upper.parquet`):
   squared error, error and report count per field, lead, init day, region, `match`, `qc` and
   `elevation`. Upper-air fields have `match` `synoptic` and `strict`. Surface reports are read
   within the strict window, so the two matches give identical surface sums; they are stored
   once, with `match` `any`, and used by both (`accumulate` stops if a surface report lies
   outside the strict window).

Then `plot` and `scorecards`:

- Regions: global, NH (20-90° N), tropics (20° S-20° N), SH (90-20° S), Europe (35-75° N,
  12.5° W-42.5° E), N America (25-60° N, 120-75° W): the scorer's standard table
  (`region_masks.load_regions`, bounds inclusive, `score.point_regions`). A report belongs to
  the region of its own position. `stations` uses the same table (continents for the bars).
- Upper-air curves and scorecards use leads valid at 00/12 UTC (section 4).
- Vector wind `W{p}`: RMSE of the vector error from the `U{p}` and `V{p}` sums, on cells where
  both exist; 10 m vector wind (`w10m`) likewise from `uas` and `vas`.
- Curves and tables cover every region. Scorecards cover global unless `--regions` names
  others (or `all`). Their rows: Z, T, Q and vector wind per level; 2 m temperature, 10 m
  vector wind and 10 m wind speed. A radiosonde cell spans its 00/12 UTC lead and the 06/18
  UTC one after it.
- RMSE pools all reports of all inits at a lead; intervals are 95 % day-bootstrap ones (init
  days resampled independently, the same days for every IC source, so differences are
  paired); scorecards test the paired difference with a t-test over days. Scores of
  consecutive daily forecasts are correlated over 2-5 days (Geer 2016, Tellus A 68, 30229);
  for `fc5d_obs_2025` (256 inits drawn over the year) a moving-block bootstrap with 5- and
  7-day blocks changed the median r14-vs-v2 interval width by under 4 % and no significance
  verdict.
- The % panels are against the reference IC source (the first scoring reference).

## 7. Known problems in the observations

NNJA-AI v1 conventional data, as found while scoring 2025-26; the findings were also reported to
the NNJA-AI team. Station tables, scripts, patterns and the literature with references are in
`BAD_OBSERVATIONS.md`. The numbers below are WeatherNext 2 forecasts from the four IC sources,
all leads pooled; Aurora 1.5 gives the same stations.

### 7.1 Files

- Whole days are missing in all five datasets: 2025-01-14, 2025-10-21, 2026-05-03..04 and
  2026-07-19..25. `fetch` writes `NO_FILE` for them, and the scores have no reports then.
- Since 2026-05-02 the day files end before 21 UTC. The 23 UTC soundings for the next day's
  00 UTC are therefore absent, and `strict` and `synoptic` coincide then.
- Upper air is the TAC rawinsonde subset only.
- Radiosonde time stamps are whole hours (section 4).

### 7.2 Gross errors that pass the quality marks

| | 2025 (256 inits) | 2026 (254 inits) |
| --- | --- | --- |
| reports removed by the consensus check | 1,184,939 at 14,585 stations | 431,208 at 11,662 stations |
| Z500 RMSE, every report / after the check | 66.7 / 19.2 m | 65.4 / 19.4 m |
| Z500 RMSE without the 3 largest contributors | 52.5 m (40811, 61052, 42886) | 46.5 m (61024, 40373, 61052) |
| Z500 RMSE without the 10 largest | 40.9 m | 30.7 m |

The RMSEs use the HealDA v2 r14 IC forecasts, from `report.py`.

The summer-2026 Z500 contamination comes from these stations, with their share of the Z500
squared error:

- **61024** (16.97° N 7.98° E), 20.3 %, and **61052** (13.48° N 2.17° E), 12.5 %. Their 06 and
  18 UTC soundings carry no mandatory-level temperatures. Their heights are 250-1100 m low,
  increasingly so with height, on most days, and ERA5 agrees with the forecasts.
- **40373** (28.33° N 46.12° E), 17.3 %. Sporadic soundings, mostly in July-August, have Z500
  about 1650 m off and T500 about 64 K off.
- **15614** (7.9 %) and **76225** (4.7 %). Single soundings are thousands of metres off.

Recurring problems:

- **Radiosondes without temperatures.** Rows with Z500 but no T850, T500 or T250 come from
  61052, 65503 (12.35° N 1.52° W), 61024 and 61090. They number 1,017 in the mirror, and 97-100 %
  of their Z500 reports are more than 200 m from ERA5 (median -511 to -758 m). They make up
  60-67 % of all Z500 reports more than 200 m from ERA5. The decoded values equal the report
  text (`50512 /////`).
- **Sporadic gross soundings.** 40811, 40373, 42886, 40800 and 40766 have heights 1,000-6,400 m
  off and some temperatures 26-96 K off, on a few to a few dozen days each.
- **Impossible winds.** 20674 (73.5° N 80.4° E) on 2025-12-18 12 UTC has U850 -153.1 m/s and
  V850 -218.7 m/s. In the same row the heights and temperatures carry the SDM purge mark and the
  winds carry none. U850 of 293.6 m/s on 2025-11-27.
- **Persistent offsets** of 50-125 m at 100-300 hPa: 23933, 29839, 30054, 31977, 36003 and
  others, at 00 and 12 UTC on most days. ERA5 shows the same offset. The check removes up to
  half of these stations' upper-level entries.
- **Surface: impossible values.** 61966 `tas` 175.0 K; 80231 `tas` 173.2 K; 47946 `tas` 126 K low;
  48698 `sfcwind` up to 387 m/s (29.9 % of the 2026 `sfcwind` squared error).
- **Surface: Antarctic BUFR SYNOP winds.** 89859 and 89056 winds are 25-48 m/s too strong;
  89859 holds 33.7 % of the 2025 `sfcwind` squared error.
- **Surface: 93988 `tas`** (43.8° S 176.5° W) is 13.0 K too cold through 2025, and normal in 2026.
- **Mountain stations** (KREDARIC 2513 m, KANIN 2260 m): `tas` is colder than every forecast,
  with mean d -6.4 to -9.1 K; ERA5 d is similar. No 2 m temperature variant is applied here.
- `q2m` of -1.6 kg/kg (surface).

The check cannot remove West African soundings that share a 06/18 UTC group of about 6 reports
(section 6). It keeps 98 such Z reports more than 200 m off in 2026, all from 61052, 61024 and
61090.

### 7.3 What the quality marks are

- **Documentation.** The NNJA-AI metadata describes the radiosonde and surface QM columns as
  "SDMEDIT" or "SDMEDIT/QUIPS quality mark". These are the NCEP data-dump marks: the Senior Duty
  Meteorologist's keep (0) or purge (14) flags and the reject list (12). NCEP's automated
  quality control, the complex QC of heights and temperatures and the analysis' own checks, runs
  later, on PrepBUFR, and is not in these files.
- **In the mirror** (2,493 day files, 2024-12-30..2026-09-26), every QM column of all five
  datasets is either missing or 14. Examples: radiosonde QMGP is 14 on 137,532 level values and
  missing on 5,115,053; BUFR SYNOP QMAT is 14 on 4,072,436 and missing on 67,264,640.
- **Consequence.** Dropping marks > 2 removes exactly the SDM purges. Every scored value has a
  missing mark, and no automated check is reflected.

### 7.4 Standard practice, and what this pipeline does

| | standard practice (`BAD_OBSERVATIONS.md`, references) | here |
| --- | --- | --- |
| quality marks | assimilation and WMO verification use observations after the centre's QC: NCEP CQC hydrostatic, baseline, increment and buddy checks on PrepBUFR; GSI does not use PrepBUFR marks ≥ 4 or ≥ 8 | dump marks only; > 2 dropped, missing kept |
| background check | ECMWF rejects squared normalised departures above 25 (T, wind) or 36 (z) times their expected variance, relaxed under the Huber norm; GSI gross limit 8 for radiosonde T and wind | none |
| verification screening | WMO-No. 485: radiosondes from a common list, "screened to exclude those with large errors", excluding values the centre's analysis rejected; ECMWF verifies against observations its 4D-Var QC accepts | the consensus 5-MAD check of the compared systems (protocol `ecmwf`; `compare --gross`) |
| blacklisting | ECMWF monthly monitoring blocklist; WMO suspect-station limits (height 100-450 m, temperature 10-15 K, wind 35-60 m/s) | none |
| unphysical values | WMO: remove; WeatherReal: range checks | none |
| ML papers | GraphCast, GenCast and FuXi Weather verify against analyses; Aurora uses WeatherReal-ISD; WeatherNext 3 uses provider QC flags; eagle-tools masks marks > 2 (this pipeline's default) | as eagle-tools |

Not done here:

- No hydrostatic, baseline, increment or buddy check.
- No station blocklist.
- No range check.
- No common station list.

Consequences:

- Without the consensus check, a few stations dominate the upper-air RMSE (table above).
- The check misses the 06/18 UTC West African soundings.
- The check removes persistent offsets and representativeness departures that a
  background check against an independent analysis would partly keep: ERA5 alone flags 46-58 %
  of the removed reports.
- A read-time rule dropping radiosonde heights from rows without mandatory-level temperatures
  would remove 60-67 % of the Z500 reports more than 200 m from ERA5. This is not implemented.

## 8. Not done

No variational QC, observation bias correction, observation-error normalisation, thinning or
representativeness weighting; no balloon drift; no duplicate removal unless `--dedupe-surface`;
no background check against an independent analysis. Station counts differ by valid hour and
day, and every RMSE pools all reports of its lead, so a region dominated by dense networks
(Europe, North America) weighs heavily in the global surface numbers.
