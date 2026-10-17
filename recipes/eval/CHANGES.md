# Online scoring against observations (branch ag-e2s-obs-online)

## What it adds

`scoring.online.observations` (default absent, so existing configs are unchanged) scores every
lead step of an online run against conventional observations valid at `IC + lead`. Nothing of
the forecast is written.

- `src/obs_online.py`
  - `ObservationScorer`: the same `begin_item` / `update` / `finish_item` interface as
    `OnlineScorer`.
  - `WithObservations`: runs it beside the gridded scorer on the same chunks.
  - `finalize_observation_scores`.
- `main.py`: wraps the online scorer when the block is set, closes the worker pool after the
  run, and finalizes the observation store after `finalize_stats`, including on a resumed run
  with nothing left to do.
- `test/test_obs_online.py`: name mapping, config parsing, and sampling plus scoring on a
  synthetic linear field, then the store layout. The scorer tests need
  `OBS_SCORER_ROOT=<checkout with private/obs_scoring>` and skip without it.

## The observation methodology is external

The recipe does not read, quality-control or interpolate observations itself. A HealDA
checkout's `private.obs_scoring` does all of it, named by `scorer_root`:
- `cycle.cycle_observations`: the NNJA-AI mirror or the PrepBUFR archive, QC, the ±30 min window
  and the station selection.
- `cycle.score_cycle`: `sample` (bilinear, periodic in longitude; `Z/g`; wind speed from the
  interpolated u and v; `q2m` from `d2m` and `sp`) and `score` (RMSE, MAE, bias and count per
  region and field).

The functions are called on the in-memory forecast. They run in spawned worker processes whose
import path starts at `scorer_root`, so the scorer's own `healda` package cannot clash with
packages the forecast process imports. The CPU work also overlaps the next model step.

- **Regions**: the scorecard's `scorecard/cfg/regions/standard.yaml` plus `global`, parsed by
  `src.regions.parse_regions`. This is the file and parser the scorer's `standard_regions`
  uses through its own copy of this recipe.
- **Name mapping**: earth2studio names map to the scorer's (`t2m`→`tas`, `u10m`→`uas`,
  `v10m`→`vas`, `msl`→`pres_msl`, `z500`→`Z500`, ...). Only output variables with a mapping
  are handed over.
- **Ensembles**: each rank scores its own members, with a `member` column and no
  communication. No CRPS at observation points yet.

## Outputs (under `output.path/<store_dir>`, default `obs_scores`)

- `parts/<system>/<init>_r<rank>.parquet`: the scorer's per-cycle score table (`time` = valid
  time, `region`, `[dataset,]` `field`, `rmse`, `mae`, `bias`, `count`, `units`) plus `init`,
  `lead_time` and `member` (ensembles only). Written for each IC before the IC's resume marker.
- `pairs/<stations>/<system>/<init>_r<rank>.parquet` (with `pairs: true`): the scorer's pairs
  columns (`time, lat, lon, station, elevation, dataset, field, obs, cycle, model`) plus `lead`
  (hours) and `init`.
- `<system>/scores_<label>.zarr`, written by `finalize_observation_scores`: the scorer's store
  (`rmse, mae, bias, count` with `units`, attrs `system`, `stations`). Its leading dimensions
  are `(init, lead_time[, member])` in place of `time`, with the valid time as a 2-D `time`
  coordinate.

## Config

```yaml
scoring:
  online:
    observations:
      scorer_root: /path/to/healda   # provides private.obs_scoring
      extra_paths: []                # prepended after scorer_root in the workers
      obs: conventional              # or vad / aircraft
      cache: /path/to/mirror         # NNJA-AI mirror (obs_scores fetch), obs=conventional
      archive: null                  # PrepBUFR archive, obs=vad/aircraft
      stations: all                  # all | withheld | kept
      by_dataset: false
      dedupe_surface: false
      vertical: pressure
      pairs: false
      system: null                   # null -> run_id
      store_dir: obs_scores
      workers: 2                     # worker processes per rank
```

The variables to score must be among the run's output variables. With `output.retain: none`
these are `scoring.variables`, so list the surface fields and the pressure-level fields the
observations need (`t2m u10m v10m d2m sp` and `z/t/u/v/q` on the model levels).

## Differences from the scorer's CLI on a store

- **Grid**: the forecast grid is the model's. For Aurora (720 rows, no 90° S row), reports
  south of 89.75° S get NaN: the South Pole station, 2 surface reports per cycle. Fields the
  model lacks (10 hPa radiosonde levels) also get NaN, as for any store without them.
- **No `elevation_correct`**: it needs the model orography, which a forecast chunk does not carry.
- **Validation** (Aurora 1.5, 2026-06-01 06 UTC, IC sampled at lead 0, against the CLI on the
  same analysis stores): the observation rows are identical and in the same order. Model values
  agree exactly (max |diff| 0) wherever both are finite. Counts and RMSE agree in every region
  and field except those two cases.

## Not done

- No CRPS or ensemble-mean terms at observation points.
- No gross-error check: the scorer applies it in `compare --gross`, from the pairs.
- No per-rank reduction: each IC writes its own small table. `finalize_observation_scores`
  concatenates them.
