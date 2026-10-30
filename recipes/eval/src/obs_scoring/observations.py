# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observation tables for scoring: one row per observation value.

Columns: ``time`` (UTC), ``lat``, ``lon`` (0-360), ``station``, ``dataset``, ``field``
(``Z500``, ``tas``, ...), ``obs``; profile winds (VAD, aircraft) add ``pressure`` (hPa) and
``height`` (m). Radiosondes and surface stations come from NNJA-AI day files mirrored
unchanged by ``fetch``; quality control and unit conversion happen on read and follow NOAA
PSL eagle-tools ``obs-metrics``, except that GP10 is divided by 9.8, NCEP's encoding. VAD
and aircraft winds are read from the NNJA PrepBUFR parquet archive.
"""

from __future__ import annotations

import functools
import multiprocessing
import pathlib
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .thermo import specific_humidity_from_dewpoint

WINDOW = pd.Timedelta("30min")
# NNJA-AI radiosonde stamps are whole hours; many soundings for t are stamped t - 1 h
# (METHOD.md section 4). Upper-air reports are read from
# t - SONDE_EARLY; the `strict` match keeps [t - WINDOW, t + WINDOW), `synoptic` all.
SONDE_EARLY = pd.Timedelta("90min")
MATCHES = ("strict", "synoptic")

# NCEP dumps encode radiosonde geopotential as reported gpm * 9.8.
NCEP_GP10_PER_GPM = 9.8

MAX_QUALITY_MARK = 2
# Pressure range (hPa) of the profile winds kept: the model levels they are interpolated between.
PROFILE_PRESSURE_RANGE = (50.0, 1000.0)


UPPER_AIR = "conv-adpupa-NC002001"

UPPER_AIR_STATION = "UASID.RPID"

LEVELS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)

MOISTURE_LEVELS = LEVELS[1:]


# (value, quality mark) columns; winds are (speed, direction, quality mark).
SURFACE = {
    "conv-adpsfc-NC000001": {
        "tas": ("TMPSQ1.TMDB", "TMPSQ1.QMAT"),
        "sp": ("PRSSQ1.PRES", "PRSSQ1.QMPR"),
        "wind": ("WNDSQ1.WSPD", "WNDSQ1.WDIR", "WNDSQ1.QMWN"),
        "dewpoint": ("TMPSQ1.TMDP", "TMPSQ1.QMDD"),
    },
    "conv-adpsfc-NC000002": {
        "tas": ("TMPSQ1.TMDB", "TMPSQ1.QMAT"),
        "sp": ("PRSSQ1.PRES", "PRSSQ1.QMPR"),
        "wind": ("WNDSQ1.WSPD", "WNDSQ1.WDIR", "WNDSQ1.QMWN"),
        "dewpoint": ("TMPSQ1.TMDP", "TMPSQ1.QMDD"),
    },
    "conv-adpsfc-NC000007": {
        "tas": ("MTRTMP.TMDB", "MTRTMP.QMAT"),
        "wind": ("MTRWND.WSPD", "MTRWND.WDIR", "MTRWND.QMWN"),
    },
    "conv-adpsfc-NC000101": {
        "tas": ("TEMHUMDA.TMDB", "QMAT"),
        "sp": ("PRESDATA.PRESSQ03.PRES", "QMPR"),
        "wind": ("BSYWND1.WSPD", "BSYWND1.WDIR", "QMWN"),
        "dewpoint": ("TEMHUMDA.TMDP", "QMDD"),
    },
}

SURFACE_STATION = {
    "conv-adpsfc-NC000001": "RPID",
    "conv-adpsfc-NC000002": "REPORT_ID",
    "conv-adpsfc-NC000007": "MTRID.RPID",
    "conv-adpsfc-NC000101": "RPID",
}

META = ["LAT", "LON", "OBS_TIMESTAMP", "SELV"]


PREPBUFR_SETS = {"vad": (224,), "aircraft": (230, 231, 233, 234, 235)}

PREPBUFR_COLUMNS = [
    "SID",
    "XOB",
    "YOB",
    "DHR",
    "POB",
    "ZOB",
    "UOB",
    "VOB",
    "WQM",
    "TYP",
]

PRESSURE_BIN = 25.0

OBS_SETS = ("conventional", *PREPBUFR_SETS)
NNJA_AI_DATASETS = (UPPER_AIR, *SURFACE)


def wind_components(speed, direction_deg):
    direction = np.deg2rad(direction_deg)
    return -speed * np.sin(direction), -speed * np.cos(direction)


def passed_qc(frame: pd.DataFrame, value: str, quality: str) -> np.ndarray:
    mark = frame[quality].to_numpy(np.float64)
    good = np.isnan(mark) | (mark <= MAX_QUALITY_MARK)
    return np.where(good, frame[value].to_numpy(np.float64), np.nan)


# Station elevations at or above this are the missing-value sentinel (9999 m).
MAX_STATION_ELEVATION = 9000.0


def station_elevation(frame: pd.DataFrame) -> np.ndarray:
    if "SELV" not in frame:
        return np.full(len(frame), np.nan)
    elevation = frame["SELV"].to_numpy(np.float64)
    return np.where(elevation < MAX_STATION_ELEVATION, elevation, np.nan)


def _long(
    frame: pd.DataFrame, dataset: str, station: str, fields: dict
) -> pd.DataFrame:
    if station in frame:
        station_ids = frame[station].astype(object).to_numpy()
    else:
        station_ids = np.full(len(frame), "", dtype=object)
    meta = {
        "time": pd.to_datetime(frame["OBS_TIMESTAMP"], utc=True)
        .dt.tz_localize(None)
        .to_numpy("datetime64[ns]"),
        "lat": frame["LAT"].to_numpy(np.float64),
        "lon": np.mod(frame["LON"].to_numpy(np.float64), 360.0),
        "station": station_ids,
        "elevation": station_elevation(frame),
    }
    parts = []
    for field, values in fields.items():
        keep = np.isfinite(values)
        part = pd.DataFrame({k: v[keep] for k, v in meta.items()})
        part["dataset"] = dataset
        part["field"] = field
        part["obs"] = values[keep]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def upper_air_fields(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    fields = {}
    for level in LEVELS:
        suffix = f"_PRLC{level * 100}"
        fields[f"Z{level}"] = (
            passed_qc(frame, "GP10" + suffix, "QMGP" + suffix) / NCEP_GP10_PER_GPM
        )
        fields[f"T{level}"] = passed_qc(frame, "TMDB" + suffix, "QMAT" + suffix)
        speed = passed_qc(frame, "WSPD" + suffix, "QMWN" + suffix)
        direction = passed_qc(frame, "WDIR" + suffix, "QMWN" + suffix)
        fields[f"U{level}"], fields[f"V{level}"] = wind_components(speed, direction)
        fields[f"S{level}"] = speed
        if level in MOISTURE_LEVELS:
            dewpoint = passed_qc(frame, "TMDP" + suffix, "QMDD" + suffix)
            fields[f"Q{level}"] = specific_humidity_from_dewpoint(
                dewpoint, float(level)
            )
    return fields


def upper_air_columns() -> list[str]:
    columns = [UPPER_AIR_STATION]
    for level in LEVELS:
        names = ["GP10", "QMGP", "TMDB", "QMAT", "WSPD", "WDIR", "QMWN", "TMDP", "QMDD"]
        columns += [f"{name}_PRLC{level * 100}" for name in names]
    return columns


def surface_fields(frame: pd.DataFrame, spec: dict) -> dict[str, np.ndarray]:
    fields = {"tas": passed_qc(frame, *spec["tas"])}
    if "sp" in spec:
        pressure = passed_qc(frame, *spec["sp"])
        # Some BUFR SYNOP reports carry station pressure 0 with no quality mark.
        sentinel = pressure <= 0
        if sentinel.any():
            print(f"{int(sentinel.sum())} station pressures <= 0 Pa treated as missing")
        fields["sp"] = np.where(sentinel, np.nan, pressure)
    speed = passed_qc(frame, spec["wind"][0], spec["wind"][2])
    direction = passed_qc(frame, spec["wind"][1], spec["wind"][2])
    fields["uas"], fields["vas"] = wind_components(speed, direction)
    fields["sfcwind"] = speed
    if "dewpoint" in spec:
        dewpoint = passed_qc(frame, *spec["dewpoint"])
        fields["q2m"] = specific_humidity_from_dewpoint(dewpoint, fields["sp"] / 100.0)
    return fields


def surface_columns(name: str) -> list[str]:
    columns = [c for columns in SURFACE[name].values() for c in columns]
    return [SURFACE_STATION[name]] + columns


@functools.cache
def nnja_dataset(name: str):
    import nnja_ai

    return nnja_ai.DataCatalog()[name].load_manifest()


def raw_day_dir(cache: pathlib.Path, dataset: str, day: pd.Timestamp) -> pathlib.Path:
    return cache / dataset / f"{day:%Y%m%d}"


def _fetch_day(task) -> str:
    cache, dataset, day = task
    target = raw_day_dir(cache, dataset, day)
    if target.exists():
        return f"{dataset} {day:%Y-%m-%d} present"
    manifest = nnja_dataset(dataset).manifest
    key = day.tz_localize("UTC")
    staging = target.with_name(target.name + ".tmp")
    staging.mkdir(parents=True, exist_ok=True)
    if key not in manifest.index:
        (staging / "NO_FILE").touch()
    else:
        import fsspec

        gcs = fsspec.filesystem("gcs", token="anon")  # noqa: S106 (anonymous access)
        for source in manifest.loc[[key], "file"]:
            gcs.get(source, str(staging / pathlib.PurePosixPath(source).name))
    staging.rename(target)
    return f"{dataset} {day:%Y-%m-%d} fetched"


def needed_days(cycles) -> list[pd.Timestamp]:
    return sorted({d for c in cycles for d in window_days(c)})


def missing_days(cache: pathlib.Path, cycles) -> list[pd.Timestamp]:
    """Days the cycles need that the mirror lacks for at least one NNJA-AI dataset."""
    return [
        day
        for day in needed_days(cycles)
        if not all(raw_day_dir(cache, d, day).exists() for d in NNJA_AI_DATASETS)
    ]


def fetch(cache: pathlib.Path, days, workers: int):
    """Mirror the NNJA-AI day files of ``days``, unchanged, under ``cache``; files already
    there are kept. A day NNJA-AI does not have gets a ``NO_FILE`` marker."""
    tasks = [(cache, dataset, day) for day in days for dataset in NNJA_AI_DATASETS]
    with multiprocessing.Pool(workers) as pool:
        for line in pool.imap_unordered(_fetch_day, tasks):
            print(line, flush=True)


def _read_raw(
    cache: pathlib.Path, dataset: str, day: pd.Timestamp, columns: list[str]
) -> pd.DataFrame | None:
    folder = raw_day_dir(cache, dataset, day)
    if not folder.exists():
        raise FileNotFoundError(
            f"{folder} not fetched; run python -m src.obs_scoring.cli fetch"
        )
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return None
    present = set(pq.read_schema(files[0]).names)
    wanted = [c for c in dict.fromkeys(META + columns) if c in present]
    return pd.concat(
        [pd.read_parquet(f, columns=wanted) for f in files], ignore_index=True
    )


@functools.lru_cache(maxsize=4)
def conventional_day(cache: pathlib.Path, day: pd.Timestamp) -> pd.DataFrame:
    """Radiosonde and surface-station values of one day, quality-checked and converted."""
    parts = []
    frame = _read_raw(cache, UPPER_AIR, day, upper_air_columns())
    if frame is not None:
        parts.append(
            _long(frame, UPPER_AIR, UPPER_AIR_STATION, upper_air_fields(frame))
        )
    for name, spec in SURFACE.items():
        frame = _read_raw(cache, name, day, surface_columns(name))
        if frame is not None:
            fields = surface_fields(frame, spec)
            parts.append(_long(frame, name, SURFACE_STATION[name], fields))
    if not parts:
        columns = ["time", "lat", "lon", "station", "dataset", "field", "obs"]
        return pd.DataFrame({c: [] for c in columns}).astype(
            {"time": "datetime64[ns]", "lat": float, "lon": float, "obs": float}
        )
    table = pd.concat(parts, ignore_index=True)
    table["station"] = table["station"].astype(str)
    return table


def within(time, cycle, early=WINDOW, late=WINDOW) -> np.ndarray:
    """Reports whose time is in [cycle - early, cycle + late); ``cycle`` is one time or
    one per report."""
    time = pd.to_datetime(np.asarray(time))
    return np.asarray((time >= cycle - early) & (time < cycle + late))


def in_window(obs: pd.DataFrame, cycle: pd.Timestamp, window=WINDOW) -> pd.DataFrame:
    return obs[within(obs["time"], cycle, window, window)].reset_index(drop=True)


def match_rows(obs: pd.DataFrame, cycle: pd.Timestamp, match: str) -> np.ndarray:
    """Rows of ``obs`` (read for ``cycle``) in the ``strict`` or ``synoptic`` match."""
    if match == "synoptic":
        return np.ones(len(obs), dtype=bool)
    if match == "strict":
        return within(obs["time"], cycle)
    raise ValueError(f"match {match!r}, not one of {MATCHES}")


def window_days(
    cycle: pd.Timestamp, early=SONDE_EARLY, late=WINDOW
) -> list[pd.Timestamp]:
    first = (cycle - early).normalize()
    last = (cycle + late - pd.Timedelta("1ns")).normalize()
    return list(pd.date_range(first, last, freq="D"))


def conventional_obs(cache: pathlib.Path, cycle: pd.Timestamp) -> pd.DataFrame:
    """Surface reports in [t - WINDOW, t + WINDOW), radiosondes from t - SONDE_EARLY."""
    days = [conventional_day(cache, day) for day in window_days(cycle)]
    obs = pd.concat(days, ignore_index=True)
    sonde = (obs["dataset"] == UPPER_AIR).to_numpy()
    keep = np.where(
        sonde,
        within(obs["time"], cycle, SONDE_EARLY, WINDOW),
        within(obs["time"], cycle),
    )
    return obs[keep].reset_index(drop=True)


def prepbufr_path(archive: str, cycle: pd.Timestamp) -> str:
    return f"{archive}/{cycle:%Y}/gdas.{cycle:%Y%m%d}.t{cycle:%H}z.prepbufr.nr.parquet"


def pressure_bin(pressure_hpa) -> np.ndarray:
    return (PRESSURE_BIN * np.round(np.asarray(pressure_hpa) / PRESSURE_BIN)).astype(
        int
    )


def prepbufr_winds(
    frame: pd.DataFrame, cycle: pd.Timestamp, report_types
) -> pd.DataFrame:
    dhr = frame["DHR"].to_numpy(np.float64)
    pressure = frame["POB"].to_numpy(np.float64)
    keep = (
        frame["TYP"].isin(report_types).to_numpy()
        & (dhr >= -0.5)
        & (dhr < 0.5)
        & (frame["WQM"].to_numpy(np.float64) <= MAX_QUALITY_MARK)
        & (pressure >= PROFILE_PRESSURE_RANGE[0])
        & (pressure <= PROFILE_PRESSURE_RANGE[1])
    )
    frame = frame[keep]
    meta = {
        "time": cycle + pd.to_timedelta(frame["DHR"].to_numpy(np.float64), unit="h"),
        "lat": frame["YOB"].to_numpy(np.float64),
        "lon": np.mod(frame["XOB"].to_numpy(np.float64), 360.0),
        "station": frame["SID"].astype(str).to_numpy(),
        "pressure": frame["POB"].to_numpy(np.float64),
        "height": frame["ZOB"].to_numpy(np.float64),
    }
    labels = pressure_bin(meta["pressure"]).astype(str)
    parts = []
    for variable, column in (("U", "UOB"), ("V", "VOB")):
        values = frame[column].to_numpy(np.float64)
        valid = np.isfinite(values)
        part = pd.DataFrame({k: np.asarray(v)[valid] for k, v in meta.items()})
        part["dataset"] = (
            "prepbufr-" + frame["TYP"].astype(int).astype(str).to_numpy()[valid]
        )
        part["field"] = np.char.add(variable, labels[valid])
        part["obs"] = values[valid]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def prepbufr_obs(archive: str, cycle: pd.Timestamp, obs: str) -> pd.DataFrame | None:
    """Profile winds of ``obs`` ("vad", "aircraft") for one cycle, read from the archive."""
    source = prepbufr_path(archive, cycle)
    if not pathlib.Path(source).exists():
        return None
    report_types = PREPBUFR_SETS[obs]
    frame = pd.read_parquet(
        source, columns=PREPBUFR_COLUMNS, filters=[("TYP", "in", list(report_types))]
    )
    return prepbufr_winds(frame, cycle, report_types)


# Surface datasets in the order dedupe_surface prefers them.
SURFACE_PRIORITY = (
    "conv-adpsfc-NC000001",
    "conv-adpsfc-NC000101",
    "conv-adpsfc-NC000002",
    "conv-adpsfc-NC000007",
)
DEDUPE_DEGREES = 0.01


def dedupe_surface(obs: pd.DataFrame) -> pd.DataFrame:
    """One surface report per field, time and position (to ``DEDUPE_DEGREES``): the one
    from the earliest dataset in ``SURFACE_PRIORITY`` (SYNOP before METAR)."""
    surface = obs[obs["dataset"].isin(SURFACE_PRIORITY)]
    rank = surface["dataset"].map({d: i for i, d in enumerate(SURFACE_PRIORITY)})
    key = pd.DataFrame(
        {
            "field": surface["field"],
            "time": surface["time"],
            "lat": (surface["lat"] / DEDUPE_DEGREES).round(),
            "lon": (surface["lon"] / DEDUPE_DEGREES).round(),
        }
    ).loc[rank.sort_values(kind="stable").index]
    duplicate = key.index[key.duplicated()]
    return obs.drop(index=duplicate).reset_index(drop=True)


# Share of stations a training run withholds, chosen by the crc32 of the station id.
HOLDOUT_PERCENT = 10


def is_withheld(station_ids) -> np.ndarray:
    codes, unique = pd.factorize(
        pd.Series(station_ids, dtype=object), use_na_sentinel=False
    )
    keys = ["" if pd.isna(s) else str(s).strip().upper() for s in unique]
    flags = np.array(
        [bool(k) and zlib.crc32(k.encode()) % 100 < HOLDOUT_PERCENT for k in keys],
        dtype=bool,
    )
    return flags[codes]


def station_mask(obs: pd.DataFrame, stations: str) -> np.ndarray:
    """Rows of ``stations`` ("all", "withheld", "kept"): withheld are the radiosonde and
    surface-station rows whose station a training holdout drops (``is_withheld``)."""
    if stations == "all":
        return np.ones(len(obs), dtype=bool)
    conventional = obs["dataset"].str.startswith("conv-").to_numpy()
    withheld = conventional & is_withheld(obs["station"].to_numpy())
    return withheld if stations == "withheld" else ~withheld
