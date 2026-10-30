# Gross errors in the NNJA-AI v1 conventional observations

What is wrong in the radiosonde and surface reports the forecasts are scored against, which
reports the figures' consensus check (`METHOD.md` section 6, step 4) removes, what the NNJA-AI
quality marks are, and what standard practice does about such reports. Summary in `METHOD.md`
section 7.

Every number below comes from one of the analysis scripts listed in section 6 (kept with the
campaign notes in `my/obs_quality/`, with their text and csv outputs). Literature claims are
numbered and listed in the references; "abstract only" marks sources whose full text was not
read.

Terms:

- **entry**: one report at one lead of one init. This is the unit the figure sums count.
- **report**: one observation, defined by station, dataset, field, valid time, report time and
  value. A report is verified by several inits and leads.
- **d**: obs minus consensus, where consensus is the median over the four IC sources of the
  forecast at the report. The IC sources are HealDA v2 r14, HealDA v2 final, ERA5 and IFS
  control-fc0.
- **share**: the part of a field's squared error that a station contributes. It uses the HealDA
  v2 r14 IC forecasts, every entry, all leads pooled, with no check applied.
- **ERA5 d**: obs minus the ERA5 analysis at the same report. ERA5 is taken from the lead-0
  analysis pairs, matched on valid time, station, field, report time and value. Every removed
  report was matched.
- **ERA5 flag**: the same 5-MAD rule, applied to ERA5 alone per field and valid time.

Runs:

| | 2025 | summer 2026 |
| --- | --- | --- |
| inits | 256, at 06/18 UTC | 254, 2026-05-15 to 2026-09-20 |
| leads | every 6 h to +120 h | every 6 h to +120 h |
| models | WeatherNext 2, Aurora 1.5 | WeatherNext 2, Aurora 1.5 |

The tables use WeatherNext 2. The Aurora runs give the same top stations: Z500 RMSE without the
1/2/3 largest contributors and the station lists agree to within 1 m (`report_*_aurora15det.txt`).

## 1. What the check removes

| | 2025 | 2026 |
| --- | --- | --- |
| removed entries | 2,578,602 | 1,687,171 |
| removed reports | 1,184,939 | 431,208 |
| stations with a removed report | 14,585 | 11,662 |
| radiosonde stations scored / with a removed Z report | 786 / 666 | 571 / 413 |
| removed reports that ERA5 alone also flags | 58.1 % | 46.3 % |
| Z500 RMSE, all entries / kept (r14 IC, all leads) | 66.7 / 19.2 m | 65.4 / 19.4 m |
| Z500 RMSE valid 00/12 UTC, all / kept | 63.6 / 19.1 m | 53.6 / 18.8 m |
| Z500 RMSE valid 06/18 UTC, all / kept | 138.4 / 23.0 m | 190.8 / 29.3 m |
| Z250 RMSE, all / kept | 86.4 / 26.2 m | 98.2 / 28.2 m |
| T850 RMSE, all / kept | 1.81 / 1.44 K | 2.02 / 1.37 K |
| U850 RMSE, all / kept | 3.49 / 3.16 m/s | 3.24 / 3.09 m/s |
| `tas` RMSE, all / kept | 3.40 / 2.54 K | 3.36 / 2.46 K |
| `sfcwind` RMSE, all / kept | 2.92 / 2.08 m/s | 2.69 / 2.01 m/s |

Where the removed squared error sits, by field:

- **Z (all levels).** The removed entries hold 63-94 % (2025) and 78-96 % (2026) of the squared
  error, although they are only 1.1-3.3 % of the entries.
- **Surface.** For `tas` and `sfcwind` the removed share of squared error is 44-50 %. Only
  0.4-0.65 % of entries are removed.
- **U and V aloft.** The removed share is 6-19 %.
- **Q.** Up to 5.4 % of entries are removed (Q300, 2026).
- `q2m` is not in the forecast pairs.

Sources: `report_2025_wn2.txt`, `report_2026_wn2.txt`, `csv/*_fields.csv`.

Agreement with ERA5 at the removed reports, median |d| / median |ERA5 d|:

| family | 2025 | 2026 |
| --- | --- | --- |
| Z | 101.5 / 57.2 m | 95.9 / 55.2 m |
| T | 6.19 / 1.75 K | 5.90 / 1.35 K |
| U | 17.4 / 4.5 m/s | 16.2 / 3.7 m/s |
| `tas` | 11.4 / 9.5 K | 10.3 / 8.1 K |
| `sfcwind` | 10.3 / 9.5 m/s | 9.4 / 8.2 m/s |

For removed surface reports, ERA5 departs from the observation by almost as much as the
forecasts do. For removed upper-air T, U and V reports, ERA5 is much closer to the observation
than the forecasts are.

### 1.1 Removed reports by station

By number of removed reports (`csv/*_stations_removed.csv`).
Columns: removed reports, days with one, and per main field family the number of removed
reports, the median d and the ERA5-flag share (`station_families.py`, `station_families.txt`).

**2025, radiosondes**

| station | lat, lon | reports | days | valid hours | main families: reports, median d, ERA5 flag |
| --- | --- | --- | --- | --- | --- |
| 61052 | 13.48, 2.17 | 6,400 | 306 | 06, 18 (some 00, 12) | Z 5,247, -592 m, 100 % |
| 23933 | 60.97, 69.07 | 3,489 | 330 | 00, 12 | Z 2,669, -84 m, 91 %; T 672, -4.2 K, 73 % |
| 65503 | 12.35, -1.52 | 2,673 | 249 | 00 (some 06, 12) | Z 2,194, -670 m, 100 % |
| 36003 | 52.28, 76.95 | 2,643 | 307 | 00, 12 | Z 1,057, +83 m, 88 %; T 719, +3.7 K, 83 % |
| 32540 | 53.02, 158.72 | 2,208 | 272 | 00, 12 | Z 1,034, +92 m, 79 %; T 609, +4.8 K, 75 % |
| 47058 | 39.02, 125.82 | 2,135 | 228 | 00 | Z 880, +115 m, 98 %; T 747, +5.2 K, 97 % |
| 70026 | 71.28, -156.79 | 2,135 | 302 | all | Z 790, +31 m, 1 %; T 760, -0.8 K, 6 %; wind 512, 4 % |
| 89009 | -90.00, 0.00 | 1,822 | 327 | 00, 12 | Z 1,206, -82 m, 90 %; U, V 585, ±12 m/s, 93 % |
| 32389 | 56.32, 160.83 | 1,790 | 251 | 00, 12 | Z 910, -101 m, 74 %; T 501, -4.0 K, 63 % |
| 10393 | 52.22, 14.12 | 1,698 | 329 | all | T 450, +0.5 K, 10 %; Q 366, 10 %; Z 218, -50 m (ERA5 d -0.3 m), 6 % |

**2025, surface**

| station | dataset | lat, lon | elevation | reports | days | main families: reports, median d, ERA5 flag |
| --- | --- | --- | --- | --- | --- | --- |
| 89859 | BUFR SYNOP | -74.37, 164.14 | 28 m | 50,857 | 324 | `sfcwind` 17,611, +30.9 m/s, 100 %; `uas` 13,176, +35.7; `tas` 5,410, +11.7 K, 63 %. Over all entries `sfcwind` mean d +47.5 m/s, sd 41.4, 92 % removed, 33.7 % of the global `sfcwind` squared error |
| 93988 | BUFR SYNOP | -43.82, -176.48 | 22 m | 17,048 | 359 | `tas` 8,172, -13.4 K, 100 % (all entries: mean d -13.0 K, sd 2.3, 89 % removed); `sfcwind` 4,474, -11.0 m/s |
| 89834 | BUFR SYNOP | -67.38, 138.73 | | 6,453 | 342 | `sfcwind` 2,884, -12.8 m/s, 94 %; `vas` 2,549, -11.5 m/s |
| 61966 | BUFR SYNOP | -7.30, 72.40 | 8 m | 4,845 | 246 | `uas` 3,087, +9.4 m/s, 87 %; `tas` 461, reported 175.0 K, 100 % |
| KREDARIC | BUFR SYNOP | 46.38, 13.85 | 2513 m | 3,895 | 209 | `tas` 2,759, -10.5 K, 97 % |
| 89056 | BUFR SYNOP | -62.19, -58.98 | 45 m | 3,029 | 352 | `sfcwind` 1,220, +24.9 m/s, 97 % |
| 44292 | BUFR SYNOP | 47.89, 106.33 | | 2,991 | 263 | `uas` 1,316, +9.4 m/s, 87 %; `tas` 569, +10.6 K, 91 % |
| KMYP | METAR | 38.47, -106.32 | | 2,892 | 256 | `sfcwind` 1,363, +10.6 m/s, 95 % |
| 91844 | BUFR SYNOP | -21.20, -159.81 | | 2,760 | 107 | `sfcwind` 1,493, -10.0 m/s, 69 % |
| KMWN | METAR | 44.27, -71.30 | | 2,668 | 351 | `sfcwind` 1,022, +15.8 m/s, 99 % |

**2026, radiosondes**

| station | lat, lon | reports | days | valid hours | main families: reports, median d, ERA5 flag |
| --- | --- | --- | --- | --- | --- |
| 61052 | 13.48, 2.17 | 2,034 | 89 | 06, 18 (some 00, 12) | Z 1,577, -557 m, 97 % |
| 70026 | 71.28, -156.79 | 1,766 | 123 | all | T 634, -1.2 K, 4 %; Z 612, -3 m, 1 % |
| 23933 | 60.97, 69.07 | 1,659 | 126 | 00, 12 | Z 1,200, -78 m, 91 %; T 379, -3.6 K, 83 % |
| 47102 | 37.97, 124.63 | 1,433 | 124 | all | Q 619, 12 %; T 254, -1.2 K, 17 %; Z 187, +15 m, 39 % |
| 10393 | 52.22, 14.12 | 1,376 | 126 | all | T 411, -0.4 K, 10 %; Z 366, -34 m (ERA5 d -0.5 m), 3 % |
| 61024 | 16.97, 7.98 | 1,279 | 107 | 06 (some 12) | Z 1,083, -855 m, 99 % |
| 29839 | 53.50, 83.83 | 1,244 | 122 | 00, 12 | Z 949, +91 m, 89 %; T 226, +3.8 K, 76 % |
| 30054 | 59.45, 112.58 | 1,122 | 117 | 00, 12 | Z 728, -78 m, 83 % |
| 31977 | 43.27, 132.05 | 1,057 | 107 | 00, 12 | Z 666, +102 m, 85 % |
| 36003 | 52.28, 76.95 | 1,039 | 122 | 00, 12 | Z 440, +92 m, 82 % |

**2026, surface**

| station | dataset | lat, lon | elevation | reports | days | main families: reports, median d, ERA5 flag |
| --- | --- | --- | --- | --- | --- | --- |
| 93988 | BUFR SYNOP | -43.82, -176.48 | 22 m | 2,800 | 75 | `sfcwind` 1,550, -9.5 m/s, 62 %; `tas` over all entries mean d -1.0 K, none removed |
| KREDARIC | BUFR SYNOP | 46.38, 13.85 | 2513 m | 2,307 | 127 | `tas` 2,076, -10.5 K, 92 % (all entries: mean d -9.1 K, sd 2.7, 46 % removed) |
| 61966 | BUFR SYNOP | -7.30, 72.40 | 8 m | 2,193 | 114 | `tas` 1,549, reported 175.0 K on 80 days, 100 %; 27.0 % of the global `tas` squared error |
| 78896 | BUFR SYNOP | 16.26, -61.53 | | 2,173 | 124 | `tas` 2,055, -11.3 K, 93 % |
| KANIN | BUFR SYNOP | 46.36, 13.47 | 2260 m | 1,823 | 126 | `tas` 1,726, -10.0 K, 85 % |
| 47946 | BUFR SYNOP | 26.50, 127.90 | | 1,664 | 113 | `sfcwind` 640, -10.1 m/s, 57 %; `tas` 171, -125.8 K, 100 % |
| 44292 | BUFR SYNOP | 47.89, 106.33 | | 1,301 | 96 | `sfcwind` 272, +8.9 m/s; `tas` 150, -10.9 K |
| 89056 | BUFR SYNOP | -62.19, -58.98 | 45 m | 1,238 | 127 | `sfcwind` 454, +27.9 m/s, 97 % |
| 94146 | SYNOP | -12.03, 135.57 | | 1,211 | 126 | `tas` 833, -12.3 K, 99 % |
| KRVAVEC | BUFR SYNOP | 46.30, 14.53 | | 1,088 | 112 | `tas` 961, -8.9 K, 76 % |

Other surface stations with the largest share of a field's squared error:

| station | year | lat, lon | field | values | share |
| --- | --- | --- | --- | --- | --- |
| 48698 | 2026 | 1.37, 103.98 | `sfcwind` | 24.2 to 387 m/s on 127 days, consensus 3 m/s | 29.9 % |
| 80231 | 2026 | 4.71, -73.80 | `tas` | 173.2 K | 4.8 % |
| 80231 | 2025 | 4.71, -73.80 | `tas` | 278 K against 289 K | |
| BED, SHY, PIP, CHH | 2025 | | `tas` | d -42 to -90 K, 67-100 % removed | 0.8-1.0 % each |

80231 has elevation 3845 m. BED, SHY, PIP and CHH are mobile-SYNOP IDs at Indian positions.

### 1.2 Error patterns of the radiosonde stations

Per field, from `report.py` "Detail" and `station_lookup.py`. "Removed %" is taken over every
entry of that station and field.

**West African heights (61052, 65503, 61024, 61090)**

The heights are low, and increasingly so with height. Over the station's removed reports:

| station | year | hours | Z925 | Z850 | Z500 | Z250 | Z200 | Z100 | days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 61052 | 2025 | mostly 06, 18 | -253 m | | -521 m | | -853 m | -619 m | 299 |
| 65503 | 2025 | 00 | | -374 m | -591 m | | -915 m | | 205 |
| 61024 | 2026 | 06 | -257 m | | -782 m | -1033 m | -1090 m | | 97 |

- About half of each station's Z entries are removed: 61052 47 %, 65503 50 %, 61024 50 %.
- At 61052 the 00 and 12 UTC soundings are close to the forecasts. At 61024 the 12 UTC ones are.
- ERA5 d equals d to within a few metres, for example 61024 Z500 -783 m.

The raw rows of these soundings carry no temperature at the mandatory levels (`ttaa_text.py`,
`ttaa_text_examples.txt`):

| sounding | TTAA text | decoded Z500 |
| --- | --- | --- |
| 61024, 2026-06-01 06 UTC | `50512 ///// 36005`, temperature group missing | 5120 m (12 UTC: 5940 m) |
| 61052, 2025-05-15 06 UTC | `50535 /////` | 5350 m |
| 61052, 2025-05-15 18 UTC | `50532 /////` | 5320 m (00 and 12 UTC: 5920 m) |
| 65503, 2025-03-21 00 UTC | `50530 /////` | 5300 m (12 UTC: 5920 m) |

In each case the decoded value equals the report text.

Over the whole mirror, 1,017 raw rows have Z500 but no T850, T500 or T250 (`z_without_t.py`):

- They come from 61052 (620), 65503 (213), 61024 (100) and 61090 (64).
- At lead 0, 97-100 % of their Z500 reports are more than 200 m from ERA5. The median obs - ERA5
  is -511 to -758 m.
- They are 695 of the 1,156 Z500 reports more than 200 m from ERA5 in 2025, and 246 of 367 in
  2026.
- Rows that do carry temperatures exceed 200 m from ERA5 in 0.07-0.23 % of reports.

**Sporadic gross heights (40811, 40373, 42886, 40800, 40766, 15614, 76225, 34731, 40856)**

A few soundings at these stations are thousands of metres off. Each station has a small removed
share but a large share of the squared error.

| station | year | removed reports, Z500 | Z500 d | other |
| --- | --- | --- | --- | --- |
| 40811 (31.33, 48.67) | 2025 | 56 on 56 days, all 12 UTC | q10 -4357 m, q90 +1290 m | T850 d -69 K on 12 days |
| 40373 (28.33, 46.12) | 2026 | 22 on 20 days, 00 and 12 UTC, mostly July-August | -1647 m | T500 d -64 K |
| 42886 (21.92, 84.08) | 2025 | 12 at 00 UTC, Jan and Jun | -3065 m | |
| 40800 (32.51, 51.70) | 2025 | | | Z100-Z300 -3300 to -3570 m on 2025-11-18 and 11-20 |
| 40766 (34.27, 47.12) | 2025 | | | Z50 +5691 m on 2025-12-06; Z250 -5108 m on 2025-09-27 |

- 40373: removed 11 % of entries. Its raw TTAA of 2026-07-15 12 UTC reads `50415 77358` (Z500
  4150 m, temperature group 77358), while the 00 UTC report of that day ends at 700 hPa.
- Single reports at 76225 (Z500 -3846 m), 15614 (Z500 q10 -2635 m), 34731 and 40856 give each
  station 4-8 % of the Z500 squared error from fewer than 20 removed reports.

**Persistent moderate height offsets (Russian network)**

At 00 and 12 UTC on most days, the offset grows with height:

| station | year | Z500 | Z300 | Z250 | Z100 | Z50 | removed % (upper levels) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 23933 | 2026 | -57 | -77 | | -86 | -105 | 25-47 |
| 29839 | 2026 | +66 | | +88 | +101 | +125 | 21-38 |

- The values are in metres, median d over the removed reports.
- ERA5 d is within 5-15 m of d.
- The report text carries the same values, for example 23933 2026-06-01 00 UTC `50558` = 5580 m.
- The same pattern holds at 30054 (Z -78 m), 31977 (+102 m), 36003 (+83 to +92 m), 32540
  (+92 m) and 47058 (+115 m), with ERA5 d within 15 m of d; and at 32389 (-101 m, ERA5 d
  -55 m). Medians over all removed Z reports (`station_families.txt`).
- Whether these offsets are observation errors or errors common to the forecasts and ERA5 has
  not been established. The check removes up to half of such a station's upper-level entries.

**Removals not confirmed by ERA5**

ERA5 alone flags only 1-17 % of the removed reports at 70026, 10393, 47102 and 47104. These
stations lose 1,000-2,100 reports a year, spread over T, Z, Q and wind:

- At 70026 and 47104 the removed reports have a median d near 0.
- At 10393 the removed Z reports have a median d of -34 to -50 m but an ERA5 d of -0.3 to
  -0.5 m.

**Wind**

20674 (73.5, 80.4), 2025:

- U850 +293.6 m/s on 2025-11-27 00 UTC.
- Z1000 reported as 500.00 m on 2025-01-19 and 2025-06-06.
- On 2025-12-18 12 UTC: U850 -153.1 m/s and V850 -218.7 m/s.
  - The 850 hPa wind group of the TTAA text is `03767` (`peek_raw.py`).
  - The same row carries QMGP 14 and QMAT 14 at every level, but no wind quality mark, so the
    wind passes the read filter.

**Surface patterns**

- 93988 `tas`: -13.0 K with sd 2.3 K in 2025. It reports every 10 min, which gives 8,172
  removed `tas` reports. In 2026 the `tas` mean d is -1.0 K and nothing is removed.
- 89859 and 89056 (Antarctic BUFR SYNOP): `sfcwind` +25 to +48 m/s mean d. The co-located
  SYNOP-fixed reports of 89859 have mean d +2.2 m/s.
- 61966 `tas` is reported as 175.0 K in every removed report; 80231 `tas` has a median of
  173.2 K over its removed 2026 reports.
- 48698 `sfcwind`: 24.2 to 387 m/s. The SYNOP-fixed reports of the same station are normal
  (mean d -0.4 m/s).
- Mountain stations KREDARIC (2513 m), KANIN (2260 m) and KRVAVEC: `tas` colder than every
  forecast, mean d over all entries -6.4 to -9.1 K (KREDARIC, KANIN), with a similar ERA5 d. These stations are not covered by the 2 m
  temperature variants (`elevation`), which the inventory does not apply.

## 2. The 2026 Z500 contamination

The summer-2026 Z500 RMSE, with no check, all leads pooled and the r14 IC, is 65.4 m for
WeatherNext 2 and 65.1 m for Aurora. The table removes the stations with the largest share one
by one (`report.py`, "Z500 reference-IC RMSE ... without the k stations"):

| stations removed | RMSE |
| --- | --- |
| none | 65.4 m |
| 61024 (20.3 % of the squared error) | 58.5 m |
| + 40373 (17.3 %) | 51.8 m |
| + 61052 (12.5 %) | 46.5 m |
| + 15614 (7.9 %), 76225 (4.7 %) | 40.3 m |
| + 42492, 71043, 43192, 61090, 42701 | 30.7 m |
| 20 stations | 23.7 m |
| consensus check instead | 19.4 m |

- 61024, 61052 and 61090: 06 and 18 UTC soundings without mandatory-level temperatures,
  section 1.2.
- 40373: sporadic soundings with heights 1,600-3,200 m off at 500-100 hPa and temperatures
  37-96 K off.
- 15614, 76225, 42492, 71043 and 43192: single soundings thousands of metres off.

The same analysis for 2025, from 66.7 m:

| stations removed | RMSE |
| --- | --- |
| 40811 (20.3 %) | 59.5 m |
| + 61052 (10.2 %) | 55.7 m |
| + 42886 (7.8 %) | 52.5 m |
| + 34731, 40856 | 47.3 m |
| 10 stations | 40.9 m |
| 20 stations | 33.8 m |
| consensus check instead | 19.2 m |

## 3. Reports the check cannot remove

Groups (field, valid time of one init) with fewer than 5 reports:

| | entries | reports | reports with \|d\| above the limits |
| --- | --- | --- | --- |
| 2025 | 22,734 | 5,891 | 3 (70026 and 10393, T1000) |
| 2026 | 17,395 | 1,647 | 29: Z50 of 61052 (14), 61024 (10) and 61090 (1) at 06/18 UTC; 70026 T925/T1000 (4) |

The limits are Z 200 m, T 10 K, U and V 30 m/s, Q 0.01 kg/kg, `tas` 15 K and surface winds
30 m/s. Source: `report.py`, "Groups with < 5 reports".

Small groups are rare because a group pools every station worldwide. The larger gap is groups
of 6-15 radiosonde reports at 06/18 UTC, where several of the West African soundings sit in the
same group. There the median and MAD follow them, and they are kept:

| | entries kept with \|d\| above the limits | Z reports kept with \|d\| > 200 m | of those at 06/18 UTC |
| --- | --- | --- | --- |
| 2026 | 898 (281 reports, median group size 6) | 98 | 98: 61052 (47), 61024 (28), 61090 (19) |
| 2025 | 1,483 (1,313 reports, median group size 351) | 477 | 116 (61052: 57) |

ERA5 d for the kept West African reports equals d (-313 to -545 m).

The 2025 note that 61052's May 2025 18 UTC soundings are kept because they are alone in their
group does not hold. The check removed 272 of 61052's 06 UTC and 210 of its 18 UTC Z500 reports.
It kept 57 of its Z reports with |d| > 200 m; the median group size of its kept entries is 11.

## 4. What the NNJA-AI quality marks are

**Documentation**

- The NNJA-AI v1 catalog metadata for `conv-adpupa-NC002001` describes the columns:
  - QMGP: "SDMEDIT quality mark for geopotential";
  - QMAT: "SDMEDIT/QUIPS quality mark for air temperature";
  - QMWN: "SDMEDIT/QUIPS quality mark for wind (does not include NESDIS produced satellite
    winds)";
  - the `conv-adpsfc` tables use the same wording [N1].
- The NCEP code table shared by these marks [10]:
  - 0: keep flag set by the SDM (Senior Duty Meteorologist);
  - 1: good;
  - 2: "not checked, or determined to be of neutral quality";
  - 3: suspect;
  - 4: corrected;
  - 12: on the manually updated reject list;
  - 13: bad by automated QC;
  - 14: "purge (reject) flag placed on datum by ... the NCEP SDM".
- The NCEP dump document says "the final stage of the [dump] process is the application of
  manual quality marks". The automated quality control (CQC, OIQC and the analysis' own checks)
  runs later, in PREPBUFR processing [9, 7].
- No source fetched says explicitly that NNJA-AI v1 conv is built from the dump rather than from
  PrepBUFR. That it is follows from the SDMEDIT descriptions and from the mark values found
  below.
- NNJA-AI's documentation makes no statement about quality control or how to use these marks
  [N1].

**In the mirror** (`raw_qm_time.py`, `qm_counts.csv`; 2,493 day files, 2024-12-30 to 2026-09-26)

Every quality-mark column of all five datasets holds only two values: missing, or 14.

| dataset | mark | value present, mark missing | value present, mark 14 |
| --- | --- | --- | --- |
| NC002001 | QMGP | 5,115,053 | 137,532 |
| NC002001 | QMAT | 4,998,399 | 39,027 |
| NC002001 | QMWN | 5,069,323 | 65,902 |
| NC002001 | QMDD | 4,834,437 | 0 |
| NC000001 | TMPSQ1.QMAT | 10,683,148 | 921,145 |
| NC000007 | MTRTMP.QMAT | 89,310,634 | 1,079,378 |
| NC000101 | QMAT | 67,264,640 | 4,072,436 |
| NC000002 | all | all missing | 0 |

The NC002001 counts are per level value. Of the 14,434 radiosonde rows with QMGP 14 at some
level, 12,313 have no wind mark at any level.

The rule "drop a mark > 2, keep a missing mark" therefore drops exactly the SDM purges. Every
scored report has a missing mark. The marks carry no automated check:

- 4.2 % (2025) and 4.6 % (2026) of the removed radiosonde reports come from a sounding in which
  some other variable carries mark 14 (`raw_join.py`).
- No other mark value occurs.

## 5. Radiosonde time stamps (`raw_join.py`, `raw_join.txt`)

NNJA-AI describes `OBS_TIMESTAMP` only as "Observation timestamp" and `MSG_DATE` as "Message
valid timestamp" [N1]. In the 555,482 radiosonde rows of the mirror:

- `OBS_TIMESTAMP` is always a whole hour, and equals `MSG_DATE` in every row.
- **Rows with report text: 491,862.** These carry `RAWRPT` and the TEMP parts in `UARID`.
  - In 491,595 of them (99.95 %), the stamp equals the day and hour of the report's own YYGG
    group: the TTAA part's, else the first part's.
  - Stamp minus launch time (`UASDG.UALNHR`, `UALNMN`): median +30 min, 1st-99th percentile
    -23 to +76 min.
  - Of the 26,441 such rows stamped 23 UTC, 26,405 have YYGG = 23 in the text. Examples are 89571 and
    94998, launched 23:15 to 23:17.
  - So for these rows the stamp is the report's YYGG hour, not the launch hour.
- **Rows without report text: 63,620.** They come from 87 stations, all in WMO blocks 50-59.
  - They have no launch time and no YYGG.
  - They are stamped 11 UTC (34,906), 23 UTC (26,949) or 17 UTC (1,644).
  - How their stamp is formed could not be established.
  - At lead 0 their Z500 is within 9-12 m RMS of ERA5 (`raw_join.txt`).

## 6. Scripts and outputs

All scripts and outputs are in `my/obs_quality/`.

| script | does | outputs |
| --- | --- | --- |
| `inventory.py` | per init: the four IC sources' pairs, the same alignment, fields and check as `figures.py` (`report.consensus_keep`, synoptic match, `tas` uncorrected) | `inventory/<year>_<model>/` removed, small-group and kept-large entries; per station and per field sums |
| `report.py` | tables above; ERA5 lead-0 pairs at the same reports | `report_<tag>.txt`, `csv/` |
| `station_lookup.py` | per-field detail of named stations | `station_lookup_*.txt` |
| `station_families.py` | removed reports per field family of named stations | `station_families.txt` |
| `raw_qm_time.py` | quality-mark counts; radiosonde time and part columns | `qm_counts.csv`, `sonde_times.parquet` |
| `raw_join.py` | stamp vs YYGG vs launch; marks at removed reports; Z500 vs ERA5 by message parts | `raw_join.txt` |
| `z_without_t.py` | Z500 from rows without mandatory-level temperatures vs ERA5 | `z_without_t.txt` |
| `ttaa_text.py`, `peek_raw.py` | report text and decoded values of single soundings | `ttaa_text_examples.txt` |
| `removed_map.py` | maps | `removed_map_<model>.png`, `removed_map_z_<model>.png` |

## 7. Standard practice

**NCEP complex QC of radiosonde heights and temperatures (CQC)**

- Gandin introduced CQC [1, abstract only]. At NMC, Collins and Gandin's comprehensive
  hydrostatic QC [2, record only; described in 5] was replaced in November 1991 by CQCHT [5].
- CQCHT runs all its checks before deciding anything [3, 4, 5]:
  - **Hydrostatic.** Residual of the thickness between adjacent mandatory levels, computed from
    the reported heights and from the reported temperatures, z_i - z_{i-1} = A + B(T_{i-1} + T_i).
  - **Baseline.** Station elevation against the elevation implied hydrostatically by the
    surface and lowest mandatory level.
  - **Increment.** Observation minus the 6 h forecast.
  - **Horizontal statistical.** Optimal interpolation of increments from up to four neighbours
    within 1000 km.
  - **Vertical statistical.**
- The decision-making algorithm (Office Note 413, operational 25 October 1995) normalises each
  residual by tabulated long-term residual standard deviations [4]:
  - Temperatures are rejected when the mean normalised residual is ≥ 6.0, and given reduced
    weight from 4.0.
  - Heights are rejected at ≥ 7.0, with reduced weight from 5.5.
  - It reports about 370 temperature or height errors per main synoptic time.
- Because the reported heights are computed from the reported temperatures, the hydrostatic
  check "does not react at all to observational errors". The DMA identifies an observation error
  when increments and statistical residuals are large but the hydrostatic residuals are small [4, 5].
- Collins (2001) reports almost 16,000 height and temperature errors detected per month
  [6, abstract only].
- In current PREPBUFR processing, CQCBUFR applies these checks against the 6 h GDAS forecast,
  corrects what it can and flags the rest. The GFS analysis runs its own variational QC [7].

**NCEP GSI**

- An observation fails the gross check when |O - B| / max(ermin, min(ermax, obserror)) > gross
  [13]. In `global_convinfo.txt` gross is 8 for radiosonde t, q and uv, and 5 for ps [12].
- Land-surface t, q and uv (types 181, 187, 281, 287) are monitored, not assimilated
  (iuse -1) [12].
- `read_prepbufr` does not assimilate marks ≥ 8, or ≥ 4 when the OI QC is on [11].
- PREPBUFR marks 4-8 and 10-14 are neither used nor monitored by the GSI analyses [8].

**ECMWF**

- Screening first checks completeness and applies a blocklist [14]:
  - The blocklist has a data-selection part and a monthly-monitoring part. The monitoring part
    lists stations "reporting in an excessively noisy or biased manner compared with the ECMWF
    background".
- Background QC compares the squared normalised departure with multiples of its expected
  variance [14]:
  - u, v, T, rh and q: 9, 16, 25 for flags 1/2/3.
  - z and ps: 12.25, 25, 36.
  - Flag 3 is rejected.
  - Wind direction is checked above 700 hPa for speeds over 15 m/s, at 60/90/120°.
- Four consecutive suspicious layers are rejected, and for geopotential every layer above them
  too [14].
- VarQC [15, 16 abstract only] lowers an observation's weight with its probability of gross
  error; P(G) > 0.75 counts as rejected.
- Since 2009 conventional observations use a Huber-norm VarQC [15, 17 abstract only, 18]. With
  it, first-guess rejection limits were relaxed "from about 5 to typically 15 standard
  deviations" for the observation types it covers [18].
- Monthly radiosonde monitoring flags stations by gross-error rate and RMS departure. A
  "severely persistent" problem over about ten days feeds the monthly blocklist, and an operator
  decides [19, 20].
- 2 m temperature in 4D-Var: departures above 7.5 K are not used, and only stations from 400 m
  below to 200 m above the model terrain are used, at 5.5 K/km [22].
- 2 m temperature verification: SYNOP stations within 250 m of the model height, observations
  more than 15 K from the analysis rejected, 6.5 K/km [22].

**Radiosonde time and drift**

- TAC reports carry neither balloon position nor elapsed time. Ignoring drift in assimilation and
  verification affects upper-level scores [23, abstract only].
- ECMWF splits BUFR ascents into sub-profiles valid at their own position and time, and keeps
  TAC profiles vertical and instantaneous [14].
- GSI uses the PrepBUFR drift position and time [11].

**WMO standardized verification against radiosondes** (WMO-No. 485, Appendix 2.2.33 [25])

- Upper-air fields "shall be verified against a common set of radiosondes", from a list
  maintained by the lead centres and published by the Lead Centre for Deterministic NWP
  Verification [25, 26].
- "The observations used for verification shall be screened to exclude those with large
  errors". Centres are recommended to "exclude values rejected by their objective analysis".
- Verification is at the nearest native grid point, separately for 00 and 12 UTC.
- For SYNOP, centres are "encouraged" to remove unphysical values and stations systematically
  rejected, and to require 90 % data availability [25].
- Appendix 2.1.2 sets the gross-error limits used to monitor suspect stations [25]:
  - height: 100 m at 1000-700 hPa, rising to 450 m at 30 hPa;
  - temperature: 15 K below 700 hPa and 10 K from 700 to 50 hPa;
  - wind: 35-60 m/s.
- ECMWF's upper-air verification against radiosondes uses the assimilation's QC. Its surface
  verification "is mostly model-independent" [20, 21].

**Verification against observations in ML weather papers**

| work | verifies against observations? | QC of the verifying observations |
| --- | --- | --- |
| GraphCast [27], GenCast [31], FuXi Weather [28] | no; against ERA5 or HRES-fc0 | n/a |
| Aurora [29] | station 2 m temperature and 10 m wind, WeatherReal-ISD | none described in the paper |
| WeatherReal [33] | (the dataset Aurora uses) | range limits; an ERA5 distributional check (median, 1.48 MAD); spike, persistence, diurnal and neighbour checks |
| Aardvark [30] | held-out HadISD stations | HadISD's own QC |
| WeatherNext 3 [32] | held-out METAR and Mesonet stations | the providers' QC flags |
| WeatherBench-X [34] | METARs | drops MADIS-flagged values |
| eagle-tools `obs-metrics` [35] | NNJA-AI | masks quality marks > 2 and keeps missing ones; ±30 min matching; no further screening (this pipeline's default) |

## 8. What this pipeline does and does not do

| step | standard practice | here | consequence found |
| --- | --- | --- | --- |
| provider marks | PrepBUFR marks after CQC/OIQC; GSI drops ≥ 4 or ≥ 8 [8, 11] | NNJA-AI v1 has only dump marks; > 2 dropped, missing kept | only SDM purges (14) are removed; the 20674 winds of 2025-12-18 pass with no mark |
| hydrostatic / baseline check | CQCHT [4, 5] | none | West African soundings without mandatory temperatures have nothing to check against; they are 60-67 % of Z500 reports > 200 m from ERA5 |
| background check | ECMWF 5σ-type limits [14]; GSI gross 8 [12]; WMO: exclude what the analysis rejects [25] | none on read; ERA5 15 K screen for `tas` only (optional) | reports 500-5,000 m off stay in unscreened scores: 2026 Z500 65.4 m vs 19.4 m |
| check in the figures | | consensus 5-MAD over the compared IC sources, per field and valid time (protocol `ecmwf`; `cli.py compare --gross`) | removes 1.1-3.3 % of Z entries; misses West African soundings in 06/18 UTC groups of ~6 (98 kept in 2026); removes persistent moderate offsets (23933, 29839: 50-125 m) and representativeness departures (mountain `tas`, wind), which ERA5 alone does not always flag (46-58 % overall; 1-17 % at 70026, 10393, 47102) |
| station blocklist / monitoring | monthly lists [14, 19, 20]; WMO suspect lists [25] | none | 61052, 65503, 61024, 89859, 93988 (2025), 61966, 48698 recur all period |
| common radiosonde list | WMO verification list [25, 26] | every NC002001 station | station set differs by year and hour (571-786 stations) |
| VarQC / Huber | assimilation only [15, 18] | n/a | |
| unphysical values | WMO: remove [25]; WeatherReal ranges [33] | none | `tas` 173-175 K, `sfcwind` up to 387 m/s scored without the check |
| drift and time | ECMWF and GSI use BUFR drift [11, 14] | no drift; time from `OBS_TIMESTAMP` (section 5) | upper-level departures include drift (`METHOD.md` section 5) |

## References

- [N1] NNJA-AI v1 catalog metadata, `conv-adpupa-NC002001` and `conv-adpsfc-NC000001` (`.pmetadata`), https://storage.googleapis.com/gcp-nnja-ai/data/v1/conv/adpupa/NC002001/.pmetadata; NNJA-AI documentation, https://nnja-ai.readthedocs.io. Full text read.

1. Gandin, L. S., 1988: Complex quality control of meteorological observations. *Mon. Wea. Rev.*, 116, 1137-1156. Abstract only.
2. Collins, W. G., and L. S. Gandin, 1990: Comprehensive hydrostatic quality control at the National Meteorological Center. *Mon. Wea. Rev.*, 118, 2752-2767. Bibliographic record only.
3. Collins, W. G., and L. S. Gandin, 1988: The hydrostatic checking of radiosonde heights and temperatures. NMC Office Note 344, https://repository.library.noaa.gov/view/noaa/23556. Abstract only.
4. Collins, W. G., and L. S. Gandin, 1996: Complex quality control for observation errors of rawinsonde temperatures and heights. NCEP Office Note 413, https://repository.library.noaa.gov/view/noaa/11426. Full text read (pp. 1-7).
5. Collins, W. G., and L. S. Gandin, 1998: Complex quality control of rawinsonde heights and temperatures at the National Meteorological Center: quality control of mandatory level data. NCEP Office Note 423, https://repository.library.noaa.gov/view/noaa/11419. Full text read (pp. 1-10).
6. Collins, W. G., 2001: The operational complex quality control of radiosonde heights and temperatures at NCEP. Parts I and II. *J. Appl. Meteor.*, 40, 137-151 and 152-168. Abstract only.
7. NCEP/EMC: PREPBUFR processing at NCEP, https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm. Full text read.
8. NCEP/EMC: PREPBUFR Table 7, quality markers, https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_7.htm. Full text read.
9. NCEP/EMC: Observational data dumping at NCEP, https://www.emc.ncep.noaa.gov/mmb/data_processing/data_dumping.doc/document.htm. Full text read.
10. NOAA-EMC NCEPLIBS-bufr: `tables/bufrtab.CodeFlag_LOC_0_7_1` (code tables 0-33-192 to 0-33-218), https://github.com/NOAA-EMC/NCEPLIBS-bufr. Full text read.
11. NOAA-EMC GSI: `src/gsi/read_prepbufr.F90`, `setupt.f90`, `gsimod.F90`, https://github.com/NOAA-EMC/GSI. Source read.
12. NOAA-EMC GSI-fix: `global_convinfo.txt`, https://github.com/NOAA-EMC/GSI-fix. Full text read.
13. DTC, 2017: GSI User's Guide v3.7, chapter 4, https://dtcenter.org/sites/default/files/community-code/gsi/docs/users-guide/html_v3.7/gsi_ch4.html. Full text read.
14. ECMWF, 2024: IFS Documentation Cy49r1, Part I: Observations (chapter 4, Table 4.3, Appendix A), https://www.ecmwf.int/sites/default/files/elibrary/112024/81623-ifs-documentation-cy49r1-part-i-observations.pdf. Full text read.
15. ECMWF, 2024: IFS Documentation Cy49r1, Part II: Data assimilation (section 5.5), https://www.ecmwf.int/sites/default/files/elibrary/112024/81624-ifs-documentation-cy49r1.pdf. Full text read.
16. Andersson, E., and H. Järvinen, 1999: Variational quality control. *Q. J. R. Meteorol. Soc.*, 125, 697-722. Abstract only.
17. Tavolato, C., and L. Isaksen, 2015: On the use of a Huber norm for observation quality control in the ECMWF 4D-Var. *Q. J. R. Meteorol. Soc.*, 141, 1514-1527. Abstract only.
18. Tavolato, C., and L. Isaksen, 2010: Huber norm quality control in the IFS. *ECMWF Newsletter* 122, 27-31, https://www.ecmwf.int/sites/default/files/elibrary/2010/17474-huber-norm-quality-control-ifs.pdf. Full text read.
19. Ingleby, B., 2017: An assessment of different radiosonde types 2015/2016. ECMWF Tech. Memo. 807. Relevant sections read.
20. Haiden, T., et al., 2018: Use of in situ surface observations at ECMWF. ECMWF Tech. Memo. 834, https://www.ecmwf.int/sites/default/files/elibrary/2018/80920-use-situ-surface-observations-ecmwf.pdf. Relevant sections read.
21. Dahoui, M., et al., 2016: Use of forecast departures in verification against observations. *ECMWF Newsletter* 149, 30-33. Read through a fetch summary.
22. Ingleby, B., et al., 2024: Improved two-metre temperature forecasts in the 2024 upgrade. *ECMWF Newsletter* 178, https://www.ecmwf.int/en/newsletter/178/earth-system-science/improved-two-metre-temperature-forecasts-2024-upgrade. Full text read.
23. Laroche, S., and R. Sarrazin, 2013: Impact of radiosonde balloon drift on numerical weather prediction and verification. *Wea. Forecasting*, 28, 772-782. Abstract only.
24. Ingleby, B., et al., 2016: Progress toward high-resolution, real-time radiosonde reports. *Bull. Amer. Meteor. Soc.*, 97, 2149-2161. Abstract only.
25. WMO, 2023 (updated 2024): Manual on the WMO Integrated Processing and Prediction System, WMO-No. 485, Appendices 2.1.2 and 2.2.33 (2.2.34 in the 2018 update). Full text read.
26. ECMWF Lead Centre for Deterministic NWP Verification, https://wmolcdnv.ecmwf.int/ and "Observation lists", https://confluence.ecmwf.int/x/CApzAQ. Page text read; the station lists were not accessed.
27. Lam, R., et al., 2023: Learning skillful medium-range global weather forecasting (GraphCast). *Science*, 382; arXiv:2212.12794. Full text read.
28. Sun, X., et al., 2024: FuXi Weather: a data-to-forecast machine learning system for global weather. arXiv:2408.05472. Full text read.
29. Bodnar, C., et al., 2025: A foundation model for the Earth system (Aurora). *Nature*; arXiv:2405.13063. Full text read.
30. Allen, A., et al., 2025: End-to-end data-driven weather prediction (Aardvark Weather). *Nature*; arXiv:2404.00411. Full text read.
31. Price, I., et al., 2024: GenCast. arXiv:2312.15796. Full text searched.
32. Rasp, S., et al., 2026: WeatherNext 3. arXiv:2609.03582. Relevant sections read.
33. Jin, W., et al., 2024: WeatherReal. arXiv:2409.09371. Section 3 read.
34. google-research/weatherbenchX, `weatherbenchX/data_loaders/sparse_parquet.py`. Source read.
35. NOAA-PSL/eagle-tools, `src/eagle/tools/obs_metrics.py`, `config/obs_metrics.yaml`. Source read.

Not established:

- What QUIPS stands for.
- How NNJA-AI forms `OBS_TIMESTAMP` for rows without report text.
- The full texts of [1, 2, 6, 16, 17, 23, 24].
- The operational GFS GSI namelist.
- Which ECMWF first-guess limit (Table 4.3 or the Huber-norm relaxation) applies to each type.
- The contents of the WMO verification radiosonde list.
