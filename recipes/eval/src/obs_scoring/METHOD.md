# Observation verification: every step, filter and correction

What happens to an observation between the NNJA-AI file and a number in a figure, and why. It
covers analyses (`private/scripts/analysis/obs_scores.py`) and forecasts (online scoring in the
earth2studio eval recipe, figures by `private/scripts/campaign/obs_figures.py`). Module-level
detail and the command lines are in `README.md` beside this file; the campaign side is in
`private/scripts/campaign/README.md`, "Observation-verified runs".

The scorer's defaults follow NOAA PSL eagle-tools `obs-metrics`, so default scores can be
compared with it. Every step that departs from eagle-tools is optional and named in the output
(`match`, `qc`, `elevation` columns), and every variant is computed side by side. The figures
draw one of three protocols, `wmo` (default), `ecmwf` and `eagle` (section 6).

## 1. Observations

| set | NNJA-AI dataset | fields |
| --- | --- | --- |
| radiosondes | `conv-adpupa-NC002001` (TAC rawinsonde) | `Z`, `T`, `U`, `V`, speed `S` at the mandatory levels 10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000 hPa; `Q` at the same levels except 10 hPa |
| surface, SYNOP fixed | `conv-adpsfc-NC000001` | `tas`, `sp`, `uas`, `vas`, `sfcwind`, `q2m` |
| surface, SYNOP mobile | `conv-adpsfc-NC000002` | as above |
| surface, METAR | `conv-adpsfc-NC000007` | `tas`, `uas`, `vas`, `sfcwind` (no pressure or dewpoint) |
| surface, BUFR SYNOP | `conv-adpsfc-NC000101` | `tas`, `sp`, `uas`, `vas`, `sfcwind`, `q2m` |

`obs_scores.py fetch` mirrors the day files unchanged into `NNJA_AI_MIRROR`; every rule below is
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
| humidity | dewpoint to specific humidity with the Magnus formula at the level pressure (radiosondes) or the reported station pressure (surface); no `q2m` without a station pressure | `healda.utils.thermo.specific_humidity_from_dewpoint` |
| longitude | to [0, 360) | |
| time | `OBS_TIMESTAMP` as UTC | NNJA-AI truncates radiosonde times to the hour (section 4) |

## 4. Time matching

For valid time t, surface reports in [t - 30 min, t + 30 min) are scored. Radiosondes are read
from t - 90 min and scored in two selections side by side (`match`):

| `match` | radiosondes | why |
| --- | --- | --- |
| `strict` | [t - 30 min, t + 30 min) | as eagle-tools (protocol `eagle`) |
| `synoptic` (protocols `wmo`, `ecmwf`) | [t - 90 min, t + 30 min) | soundings stamped 23:00 of the day before miss the strict window (below); at 2025-07-01 00 UTC that is 140 stations |

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
  that displacement (`private/docs/radiosonde_drift.md`).
- `tas`: the model's 2 m temperature at the model terrain height, unless corrected (section 6).
- `q2m`: from the model's `d2m` and `sp`, as for the observation.
- `sfcwind` and `S{p}`: the speed of the interpolated `u`, `v`.
- Every IC source is scored against the same observation files, so every system sees the same
  reports.

Forecasts are scored inside the run at every 6 h lead step, at valid time init + lead, without
storing the forecast. Each run writes per-report pairs (`obs_scores/pairs`) and per-region
score tables (`obs_scores/parts`).

## 6. Processing for the figures (`obs_figures.py`)

`accumulate` computes every variant below side by side; `plot` and `scorecards` draw one
protocol, so switching never needs a rerun.

### Protocols

| | `wmo` (default) | `ecmwf` | `eagle` |
| --- | --- | --- | --- |
| radiosonde match (section 4) | `synoptic` | `synoptic` | `strict` |
| gross check | reference check (step 5): WMO-No. 485 limits on obs minus ERA5 and on obs minus IFS; rejected only if both exceed them (`--reference intersection`, the default) | consensus (step 4; `--no-gross`: off, file suffix `_nogross`) | off |
| 2 m temperature (step 3) | moved to the station elevation at 6.5 K/km; only stations within 250 m of the ERA5 terrain | as `wmo` | at the model terrain, every station |
| station pressure `sp` (step 3) | moved hydrostatically to the station elevation (the report's 2 m temperature); stations within 250 m | as `wmo` | at the model terrain, every station (as eagle-tools) |
| sums needed | default `accumulate` | default `accumulate` | `accumulate --all-checks` |
| upper-air leads | valid at 00/12 UTC | as `wmo` | as `wmo` |
| file suffix | `_wmo_<RULE>` (`_wmo_intersection`) | `_ecmwf` | `_eagle` |

`wmo` is the protocol of the paper's figures. Its reference check removes the same reports
for every IC source, does not depend on the systems being compared, and does not favour
either reference: with one of them alone, % RMSE differences between IC sources change by at
most 0.6 percentage points, only at leads up to 18 h, and no significance verdict changes
(10-day Aurora 1.5 forecasts from 438 06/18 UTC inits in 2025-26). `ecmwf` approximates ECMWF's verification of 2 m temperature (below) and removes the gross
errors of section 7; use it to compare systems. `eagle` reproduces NOAA PSL eagle-tools
`obs-metrics` report for report, except that:

- radiosonde height is `GP10` / 9.8, the factor NCEP encodes it with; eagle-tools divides by
  9.80665 (section 3);
- the points between 359.75 and 360° E, which eagle-tools leaves unscored, are scored (454 of
  316,400 pairs on 2025-07-08, `README.md`, Validation);
- every system is scored on the same reports and inits (step 2); eagle-tools scores each
  system on its own.

With these three, RMSE agrees with eagle-tools to five significant figures (`README.md`,
Validation). All three protocols share eagle-tools' quality marks (> 2 dropped, missing kept),
mandatory levels, no vertical interpolation or balloon drift, and Magnus humidity.

| | eagle-tools | `eagle` | `ecmwf` | `wmo` |
| --- | --- | --- | --- | --- |
| outlier screening | none | none | consensus 5-MAD check | reference check, ERA5 and IFS intersection |
| radiosonde time window | ±30 min | ±30 min | from 90 min before | from 90 min before |
| radiosonde geopotential | `GP10` / 9.80665 | `GP10` / 9.8 | `GP10` / 9.8 | `GP10` / 9.8 |
| reports per system | each system on its own | the same reports and inits for every system | the same | the same |
| longitudes 359.75-360° E | unscored | scored | scored | scored |
| 2 m temperature, `sp` | at the model terrain | at the model terrain | at the station elevation, stations within 250 m | as `ecmwf` |
| intervals | none | 95 % paired bootstrap over inits (1000 resamples), curves and scorecards | the same | the same |

### Steps, per init

1. **Common inits**: only inits every IC source completed.
2. **Same reports**: the IC sources' pairs for an init must hold the same reports in the same
   order (observed value, position, field, lead, times); `accumulate` stops otherwise. A report
   is dropped when any IC source has no finite model value there.
3. **Station-elevation variants** (column `elevation`, `tas` and `sp`; every other field has
   `none`). With change = station elevation - ERA5 terrain at the station (0.25° orography,
   bilinear; `sample.station_height_change`):
   - `none`: the model's 2 m temperature at the model terrain, every station (eagle-tools).
   - `lapse6.5_win250`: only reports with |change| <= 250 m (reports without an elevation,
     0.2-0.3 % of `tas` reports, ~500-600 stations in 2025-26, are outside), and the model
     moved to the station elevation, `tas - 0.0065 K/m × change`
     (`healda.utils.thermo.temperature_at_height`). The shift is identical for every IC
     source: all of them are forecast by the same model. The terrain a model's 2 m temperature
     refers to depends on its training data; the ERA5 terrain is used for every model, which is
     exact only for one trained on ERA5.
   - `sp` (station pressure) in `lapse6.5_win250`: the same window, and each IC source's model
     `sp` moved hydrostatically from the ERA5 terrain to the station elevation
     (`obs_figures.surface_pressure_at_station`, `thermo.pressure_at_height`) through a layer
     at the report's observed 2 m temperature taken back to the terrain at 6.5 K/km, with the
     report's 2 m humidity for the virtual temperature (0 if not reported); reports without a
     2 m temperature are left out (1 %). In `none`, the model's `sp` at its terrain, as NOAA
     eagle-tools scores it. Measured (2025 ERA5 analysis, stations within 250 m): bias
     -3.7 hPa and robust spread 5.6 hPa at the terrain, 0.0 and 0.55 hPa moved; Aurora 1.5
     +6 h forecasts from 4 July 2026 inits: -3.3 → +0.5 hPa (ERA5 IC), -3.1 → +0.7 hPa
     (HealDA IC). The remaining RMSE (10-16 hPa) is a tail of gross errors, which the 15 hPa
     reference check (step 5) removes.
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
   - Online: `elevation_correct: true` in an experiment's `observations` block moves `tas`
     and `sp` from the ERA5 terrain to the station elevation inside the run
     (`sample.sample`), with no window, and stations without an elevation kept uncorrected.
     `tas` is corrected as above. `sp` is moved hydrostatically
     through a layer at the model's own 2 m temperature and 2 m humidity (`d2m`), not the
     report's, so it differs from `lapse6.5_win250`. No experiment sets it; the figures'
     variants are applied by `accumulate` to pairs at the model terrain.
4. **Consensus gross check** (column `qc`: `gross`, or `none`, every report as eagle-tools;
   `report.consensus_keep`): per field and valid time (one init's forecasts at one lead), c =
   the median over IC sources of model - obs; a report is dropped when
   |c - median(c)| > 5 × 1.4826 × median(|c - median(c)|). Computed separately for each `match`
   and 2 m temperature variant. `obs_scores.py compare --gross` uses the same rule per field
   and cycle.
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
5. **Reference check** (column `qc`: `wmo_intersection`; with `accumulate --all-checks` also
   `wmo_<name>` per reference, `wmo_union` and `none`). The references are the lead-0 pairs of
   reference analyses from `obs_scores.py score --pairs`: by default ERA5 and IFS
   (`REFERENCES`) over the periods `PAIR_PERIODS` (`year2025`, `jansep2026` under
   `_observation_scores/`; each period file adds only the cycles after the previous one's
   last), or `accumulate --reference-pairs NAME=FILE[,FILE] ...`. A report is rejected
   when |observation - reference analysis| exceeds a fixed limit for its field; the same
   reports are removed at every lead and for every IC source. `wmo_intersection` rejects only
   what every reference rejects, `wmo_union` what any rejects. `--protocol wmo --reference
   RULE` draws one (`RULE`: a reference name, `union` or `intersection`; default
   `intersection`).
   - Each reference file's rejected and present report keys are saved next to the period
     (`_observation_scores/<period>/reference_{flags,reports}_<name>.parquet`) and reused
     while the pairs file, the fields and the limits are unchanged.
   - Coverage: `accumulate` prints, per field, the kept reports that are missing from some
     reference's pairs (the check cannot judge them), and stops if they are more than 1 %
     (`UNCHECKED_LIMIT`) of an upper-air field's kept reports. This caught the first 2026
     lead-0 pairs, scored for May-September with an older scorer that kept only radiosondes
     within ±30 min: 13 % of the summer-2026 radiosonde reports (every early launch) had no
     reference value, and two single reports with 60 K and 500 m errors set the +6 h RMSE of
     their days. The 2026 pairs were re-scored with the current scorer (radiosondes from 90 min
     before).
   - Matching: a report is identified by a 64-bit hash of station, field, observation time and
     observed value, the same in the analysis and forecast pairs. A report absent from a
     reference's pairs, or without a finite reference value there, is not rejected by it.
   - Wind: `U{p}`/`V{p}` and `uas`/`vas` are judged on the vector departure of the station's
     two components at the same time and level, and both are rejected together; a missing
     partner counts as 0. `sfcwind` is judged on its own.
   - Limits (`obs_figures.py`: `WMO_Z`, `WMO_WIND`, `Q_LIMITS`, `SURFACE_LIMITS`, `wmo_limit`):

     | hPa | 1000 | 925 | 850 | 700 | 500 | 400 | 300 | 250 | 200 | 150 | 100 | 50 |
     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
     | Z (m), WMO | 100 | 100 | 100 | 100 | 150 | 175 | 200 | 225 | 250 | 275 | 300 | 400 |
     | T (K), WMO | 15 | 15 | 15 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 15 |
     | vector wind (m/s), WMO | 35 | 35 | 35 | 40 | 45 | 50 | 60 | 60 | 50 | 50 | 45 | 45 |
     | Q (g/kg), ours | 12 | 11 | 9.9 | 7.2 | 3.1 | 1.5 | 0.56 | 0.32 | 0.19 | 0.070 | 0.030 | 0.053 |

     Surface: `sp` 15 hPa (WMO), `tas` 15 K (ECMWF's 2 m temperature screen, ECMWF Newsletter
     178; not a WMO land-surface limit; tested on the departure at the model terrain, while
     the scored `tas` is moved to the station elevation), `uas`/`vas` vector and `sfcwind`
     30 m/s and `q2m` 6.3 g/kg (ours).
   - Sources of the WMO limits, WMO-No. 485 (2023 edition, Manual on the WMO Integrated
     Processing and Prediction System), Appendix 2.1.2 (exchange of monitoring results;
     gross-error limits on observed minus first-guess): section 2.1.1 heights, 2.1.2
     temperature (15 K below 700 hPa and at 50 hPa and above, 10 K between), 2.1.3 wind;
     section 3.1.1 land-surface mean sea-level pressure, 15 hPa, and 3.1.2 station-level
     pressure, "as described for MSLP", 15 hPa. Its 4 hPa bias and 6 hPa standard-deviation
     criteria flag suspect stations over a month and are not used. Appendix 2.2.34
     (standardized verification of deterministic NWP) verifies mean sea-level pressure
     against analyses only (section 5.1) and lists no surface pressure among the parameters
     verified against surface observations (section 6.1); MSL is therefore not scored here.
   - Station pressure: the reference analysis' `sp` is moved from the ERA5 terrain to the
     station elevation as in step 3 before the 15 hPa limit is applied, so terrain mismatch is
     not counted as observation error (raw where the report has no 2 m temperature).
     IFS's `sp` refers to its own finer terrain, which the IFS store does not hold; moved
     from the ERA5 terrain it keeps part of the mismatch, so IFS alone rejects about 2.7 times
     as many 2025 reports as ERA5 alone (1.13M against 0.42M of 139M, mostly `sp`). Under the
     intersection rule the `sp` check is therefore in effect ERA5's.
   - Specific humidity limits (our choice; WMO-No. 485 lists none): obs - ERA5 departures,
     2025-01 to 2026-09, all inits, sigma_MAD = 1.4826 × median absolute deviation, one rule at
     every level. For T and Z the WMO limits are 12-26 sigma_MAD (median about 20); the 15 K
     `tas` screen is 10.0 sigma_MAD. The Q limits take the same multiples, rounded to two
     significant figures: 20 sigma_MAD for radiosondes, 10 for `q2m`.

     | | 1000 | 925 | 850 | 700 | 500 | 400 | 300 | 250 | 200 | 150 | 100 | 50 | `q2m` |
     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
     | sigma_MAD (g/kg) | 0.60 | 0.55 | 0.50 | 0.36 | 0.16 | 0.076 | 0.028 | 0.016 | 0.0094 | 0.0035 | 0.0015 | 0.0027 | 0.63 |
     | limit (g/kg) | 12 | 11 | 9.9 | 7.2 | 3.1 | 1.5 | 0.56 | 0.32 | 0.19 | 0.070 | 0.030 | 0.053 | 6.3 |

     sigma_MAD from 2025 alone and from January-September 2026 alone differ from the
     pooled value by at most 4 % at 1000-200 hPa and for `q2m`, and 9 % at 150-50 hPa.

     At 300 hPa 0.55 g/kg is about the saturation specific humidity at -37 °C, so a larger
     departure exceeds the moisture an upper-tropospheric parcel can hold. ERA5, IFS and HealDA
     give sigma_MAD within 15 % of each other at every level.
   - Specific humidity is scored at every level, 1000-50 hPa (`Q_ALL_LEVELS`); the scorecards
     and the profile figure hatch the levels above 200 hPa (`Q_TOP`). There the radiosonde
     reports are dominated by sensor artefacts (July 2025: 1.6-6 % of the 150-50 hPa reports
     are zero, the median ratio to ERA5 ranges from 0.88 to 1.55, the robust spread is 0.5-1.2
     times the median value), and differences there are not interpreted. The 20 sigma_MAD
     limits reject 2.6 %, 8.9 % and 6.6 % of the 150, 100 and 50 hPa reports against ERA5. At
     250-200 hPa the reports are 10-11 % drier than ERA5 at the median, with a robust spread
     of 0.4-0.6 times the median value.
   - Fixed limits do not follow the season: at 300-400 hPa they reject 0.4-0.8 % of the
     July 2025 reports against ERA5, against 0.2-0.4 % over the whole period.
6. **Sums** (`sums/forecast.parquet`, from the forecast runs' own pairs at every lead):
   squared error, error and report count per field, lead, init, region, `match`, `qc` and
   `elevation`. Sums written
   per init day (column `day`) by an earlier `accumulate` stop `plot` and `scorecards`;
   rerun `accumulate`. Upper-air fields have `match` `synoptic` and `strict`. Surface reports are read
   within the strict window, so the two matches give identical surface sums; they are stored
   once, with `match` `any`, and used by both (`accumulate` stops if a surface report lies
   outside the strict window).

Then `plot` and `scorecards`:

- Names in figures: GH (geopotential height as reported by radiosondes, Z / g, in m), T, Wind
  (vector wind), Q, with the level (GH500, Wind 250); surface t2m, 10 m vector wind, 10 m wind
  speed; sp in the tables only. Order: GH, T, Wind, Q, then surface. No protocol names appear in figure text.

- Regions: global, NH (20-90° N), tropics (20° S-20° N), SH (90-20° S), Arctic (60-90° N),
  N America (25-60° N, 120-75° W), Europe (35-75° N, 12.5° W-42.5° E), Asia (0-85° N,
  45-180° E), Australia NZ (45-12.5° S, 120-175° E): the scorer's standard table
  (`region_masks.load_regions`, bounds inclusive, `score.point_regions`). A report belongs to
  the region of its own position. `stations` uses the same table (continents for the bars).
- Upper-air curves, tables and scorecards use only leads valid at 00/12 UTC (section 4); a
  radiosonde lead valid at 06/18 UTC has about 3 % of the reports of a 00/12 UTC one. Surface
  fields keep every 6 h lead. Every lead, lead 0 included, is scored from the forecast runs'
  own pairs: the radiosonde curves of 06/18 UTC inits start at +6 h; at lead 0 of 00/12 UTC
  inits they are the initial analyses' fit.
- Vector wind `W{p}`: RMSE of the vector error from the `U{p}` and `V{p}` sums, on cells where
  both exist; 10 m vector wind (`w10m`) likewise from `uas` and `vas`.
- Radiosonde curves and tables cover every region; surface ones global, N America and
  Europe. Scorecards cover global unless `--regions` names others (or `all`). Their rows: GH,
  T, Wind and Q per level (Q above 200 hPa hatched); t2m, 10 m vector wind and 10 m wind
  speed. Each radiosonde cell is centred on its 00/12 UTC lead and
  reaches halfway to the next one (lead ± 6 h); the first starts at the left edge and the last
  ends at the right edge, so the row has no gaps (`plot_forecast_scorecard.cell_edges`).
  Significance dots sit on the verified lead. `--vmax PCT` fixes the colour scale, so several
  cards can share one.
- RMSE pools all reports of all inits at a lead; intervals are 95 % paired bootstrap
  intervals over inits (1000 resamples; each init is one sample, resampled independently,
  the same inits for every IC source, so differences are paired). Scorecard dots use the same bootstrap per cell: each
  resample draws the inits with replacement, the same draw for both IC sources, recomputes
  RMSE = sqrt(sum sq / sum n) over the drawn inits, and a cell is dotted when the 95 %
  interval of the % RMSE change excludes 0 (`data_utils.paired_bootstrap_significant`).
  Scores of consecutive daily forecasts are correlated over
  2-5 days (Geer 2016, Tellus A 68, 30229); for `fc5d_obs_2025` (256 inits drawn over the
  year) a moving-block bootstrap with 5- and 7-day blocks changed the median r14-vs-v2
  interval width, against a bootstrap over init days, by under 4 % and no significance
  verdict.
- The % panels are against the reference IC source (the first scoring reference).

### Consistency of the inputs (2025-26 Aurora 1.5 10-day experiments)

Checked on 2026-10-31 from every run's `provenance.txt` / `resolved_config.yaml` and one sampled
day per pairs file:

| setting | analysis lead-0 pairs, year2025 | analysis lead-0 pairs, jansep2026 | forecast pairs, 2025 and 2026, 06/18 and 00/12 UTC starts |
| --- | --- | --- | --- |
| scorer | `private/obs_scoring` as of 502df32b | same, re-scored 2026-10-31 | same code (runs' `scorer_root`) |
| radiosonde window | [t - 90, t + 30) min | same (before the re-score: ±30 min for May-September) | same |
| surface window | [t - 30, t + 30) min | same | same |
| fields | Z, T, U, V, speed at 13 mandatory levels, Q 1000-50 hPa, tas, uas, vas, sfcwind, q2m, sp | same | same |
| model values | at the model terrain (no `elevation_correct`) | same | same |
| observations | NNJA-AI mirror, conventional, quality marks > 2 dropped | same | same |
| reference systems | ERA5, IFS (`IFS_HRES_T0_025_ZARR`), HealDA v2 r14 | same | IC sources ERA5, IFS, HealDA v2 r14 |
| e2s pin | - | - | d33bd45d; 78520715 for the 2025 00/12 UTC IFS run (gridded-score changes only) |

`accumulate` applies one setting to every experiment: protocol `wmo` (ERA5 and IFS
intersection), the limits above, the standard regions, `lapse6.5_win250` for tas and sp, the
synoptic radiosonde match, upper air at 00/12 UTC valid times.

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
| background check | ECMWF rejects squared normalised departures above 25 (T, wind) or 36 (z) times their expected variance, relaxed under the Huber norm; GSI gross limit 8 for radiosonde T and wind | protocol `wmo`: fixed limits on obs minus one or more reference analyses (section 6, step 5) |
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
no background check against the forecasting system's own background (the `wmo` reference check
uses analyses). Station counts differ by valid hour and
day, and every RMSE pools all reports of its lead, so a region dominated by dense networks
(Europe, North America) weighs heavily in the global surface numbers.
