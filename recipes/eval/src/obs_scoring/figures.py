# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures of an observation-scored experiment, from the forecast pairs its runs wrote.

Run from ``recipes/eval``::

    python -m src.obs_scoring.figures accumulate <experiment> --model M \\
        [--campaign-cfg DIR] [--runs-root DIR] [--out DIR] --orography FILE \\
        [--reference-pairs NAME=FILE[,FILE] ...] [--all-checks] [--workers N]
    python -m src.obs_scoring.figures plot <experiment> --model M \\
        [--campaign-cfg DIR] [--runs-root DIR] [--out DIR] \\
        [--protocol wmo|ecmwf|eagle] [--reference RULE] [--no-gross]

``--campaign-cfg`` (default ``$E2S_CAMPAIGN_CFG``) is a campaign configuration directory:
``experiments/<experiment>.yaml`` (``ic_sources``, ``cycles``, ``scoring_references``,
``run.lead_hours``, optional ``runs_from`` and ``pooled_from``) and
``ic_sources/<ic>.yaml`` (``label``, ``color``). ``--runs-root`` (default
``$E2S_CAMPAIGN_RUNS_ROOT``) holds the runs, named ``<experiment>__<model>__<ic>__<cycles>``
(and ``...__from_*`` segments), each with ``obs_scores/pairs/all/<system>/*.parquet``.
Output: ``--out``, default ``<runs-root>/_figures/<experiment>/obs/<model>_<cycles>/``.

accumulate writes error sums per field, region, lead and init for every radiosonde
time match, gross check and 2 m temperature variant (sums/forecast.parquet), from the
forecast runs' own pairs; every lead, lead 0 included, is the forecasts' own. The gross
checks are
the consensus check (``gross``) and the reference check ``wmo_intersection``: a report is
rejected when observation minus every reference analysis (default: REFERENCES, ERA5 and
IFS) exceeds a fixed limit (WMO-No. 485 where it has one). --all-checks adds the sums
without a check (``none``), of each reference alone (``wmo_<name>``) and of their union
(``wmo_union``). The references' rejected reports are saved per period and reused;
accumulate stops if more than UNCHECKED_LIMIT of an upper-air field's kept reports are
missing from a reference's pairs. The default protocol, wmo, draws ``wmo_intersection``;
``--reference RULE`` another one, ``--protocol ecmwf`` the consensus check.
plot (tables, summary, curves, 2 m temperature bias) draws one protocol
from the sums; the file names end in _<protocol> (and _nogross, or _<rule> for wmo).
The protocols, every processing step and the sums layout: METHOD.md section 6.

An experiment with ``pooled_from`` (a list of experiments with disjoint inits, same
model, cycles and IC sources) has no runs: plot reads the listed experiments'
sums and inits.json and adds them up, and the bootstrap resamples the pooled inits.
Intervals are 95 % paired bootstrap intervals over inits (NBOOT resamples, each
init one sample, the same inits for every system).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from ..obs_online import standard_regions
from . import fields as era5_grid
from .observations import within
from .report import check_same_reports, consensus_keep
from .sample import station_height_change
from .score import point_regions
from .thermo import pressure_at_height, temperature_at_height

NBOOT = 1000
PAIRS_GLOB = "obs_scores/pairs/all/*/*.parquet"
RUN_NAME = "{experiment}__{model}__{ic}__{sample}"
NICE = {
    "fcn3": "FourCastNet 3",
    "aurora15": "Aurora 1.5",
    "aurora15det": "Aurora 1.5 det",
    "aurora": "Aurora v1",
    "wn2": "WeatherNext 2",
    "wn2cyc": "WeatherNext Cyclones",
    "wn2cyc1": "WeatherNext Cyclones (model 1)",
}
# Stored unit -> factor to the unit the figures show, and that unit's label.
DISPLAY_UNITS = {"kg/kg": (1000.0, "g kg\u207b\u00b9"), "Pa": (0.01, "hPa")}
PRETTY_UNITS = {
    "m2/s2": "m\u00b2 s\u207b\u00b2",
    "m/s": "m s\u207b\u00b9",
    "kg/kg": "kg kg\u207b\u00b9",
}


def display_unit(u: str) -> tuple[float, str]:
    return DISPLAY_UNITS.get(u, (1.0, PRETTY_UNITS.get(u, u)))


def ic_style(cfg: Path, name: str) -> dict:
    found = sorted((cfg / "ic_sources").rglob(f"{name}.yaml"))
    return yaml.safe_load(found[0].read_text()) if found else {}


def run_experiment(exp: dict, experiment: str, ic: str, model: str) -> str:
    """The experiment holding the runs of ``ic`` (and ``model``): ``runs_from`` maps a source
    to an experiment, or to ``{model: experiment, default: experiment}``."""
    held = exp.get("runs_from", {}).get(ic, experiment)
    if isinstance(held, dict):
        return held.get(model, held["default"])
    return held


def segments(runs_root: Path, cell: str) -> list[Path]:
    """Existing run directories of a cell, first segment first."""
    first = runs_root / cell
    later = sorted(runs_root.glob(f"{cell}__from_*"))
    return ([first] if first.is_dir() else []) + later


def directory(value: str | None, variable: str) -> Path:
    value = value or os.environ.get(variable)
    if not value:
        raise SystemExit(f"pass the directory or set {variable}")
    return Path(value)


# q2m has a few reports with very large errors; it stays in sums and tables.
SURFACE_PLOTTED = ["tas", "sfcwind"]
LEVELS = [850, 500, 250]
Q_LEVELS = [850, 700, 500]
ALL_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50]
UPPER = [f"{v}{p}" for v in "TZUV" for p in ALL_LEVELS] + [f"Q{p}" for p in ALL_LEVELS]
FIELDS = UPPER + ["tas", "q2m", "sfcwind", "uas", "vas", "sp"]
# Region name in the sums -> the recipe's standard region (obs_online.standard_regions).
REGIONS = {
    "global": "global",
    "NH": "northern_hemisphere",
    "tropics": "tropics",
    "SH": "southern_hemisphere",
    "Arctic": "arctic",
    "N America": "north_america",
    "Europe": "europe",
    "Asia": "asia",
    "Australia NZ": "australia_new_zealand",
}
# Region name -> figure title and short label (WMO verification-area names; the boxes are
# e2s's).
REGION_LABELS = {
    "global": ("Global", "Global"),
    "NH": ("Northern Extra-tropics", "N. Extra-tropics"),
    "tropics": ("Tropics", "Tropics"),
    "SH": ("Southern Extra-tropics", "S. Extra-tropics"),
    "Arctic": ("Arctic", "Arctic"),
    "N America": ("North America", "North America"),
    "Europe": ("Europe", "Europe"),
    "Asia": ("Asia", "Asia"),
    "Australia NZ": ("Australia and New Zealand", "Australia & NZ"),
}
REGION_CAPTION = (
    "Global; Northern Extra-tropics 20-90 N; Tropics 20 S-20 N; Southern Extra-tropics "
    "90-20 S; Arctic 60-90 N; North America 25-60 N, 120-75 W; Europe 35-75 N, "
    "12.5 W-42.5 E; Asia 0-85 N, 45-180 E; Australia and New Zealand (Australia & NZ) "
    "45-12.5 S, 120-175 E (bounds inclusive; a report belongs to the region of its own "
    "position). Australia & NZ has ~9 radiosonde reports per 00/12 UTC valid time, so its "
    "intervals are wide."
)
LEAD0_NOTE = (
    "lead 0, the initial analyses, which assimilated these observations (not "
    "independent verification)"
)
INTERVAL = f"95 % paired bootstrap interval over inits ({NBOOT} resamples)"
GROSS = 5.0
# Radiosonde time matches of the upper-air fields. Surface reports are read within the
# strict window, so their sums are the same for both matches and are stored once, as "any".
MATCHES = ("synoptic", "strict")
SURFACE_MATCH = "any"
# Variants of the station-elevation fields (column ``elevation``; other fields have
# "none"): the largest |station elevation - ERA5 terrain| (m) of a kept station (None: every
# station), whether the model is moved from the ERA5 terrain to the station elevation
# (tas at 6.5 K/km, sp hydrostatically: surface_pressure_at_station), and a label.
ELEVATION_FIELDS = ("tas", "sp")
ELEVATION_VARIANTS = {
    "none": (None, False, "t2m and sp at the model terrain"),
    "lapse6.5_win250": (
        250.0,
        True,
        "t2m (6.5 K/km) and sp (hydrostatic) moved to the station elevation, "
        "stations within 250 m of the ERA5 terrain",
    ),
}


# Protocols drawn by plot (METHOD.md section 6).
@dataclasses.dataclass(frozen=True)
class Protocol:
    match: str  # radiosonde time match, one of MATCHES
    qc: str  # gross check: "none", "gross" (consensus) or "wmo_<reference rule>"
    elevation: str  # tas and sp variant, a key of ELEVATION_VARIANTS


DEFAULT_PROTOCOL = "wmo"
# Reference rule of the wmo check: intersection, union or one reference analysis.
DEFAULT_REFERENCE_RULE = "intersection"
PROTOCOLS = {
    "wmo": Protocol("synoptic", f"wmo_{DEFAULT_REFERENCE_RULE}", "lapse6.5_win250"),
    "ecmwf": Protocol("synoptic", "gross", "lapse6.5_win250"),
    "eagle": Protocol("strict", "none", "none"),
}


def resolve_protocol(
    name=DEFAULT_PROTOCOL, reference=DEFAULT_REFERENCE_RULE, no_gross=False
):
    """The Protocol of ``name`` with the wmo check's reference rule and ``no_gross``
    applied, and the suffix of the file names drawn with it."""
    protocol = PROTOCOLS[name]
    if reference != DEFAULT_REFERENCE_RULE and name != "wmo":
        raise SystemExit(f"--reference applies to protocol wmo, not {name}")
    if no_gross and protocol.qc != "gross":
        raise SystemExit(f"--no-gross applies to protocol ecmwf, not {name}")
    suffix = f"_{name}" + ("_nogross" if no_gross else "")
    if name == "wmo":
        protocol = dataclasses.replace(protocol, qc=f"wmo_{reference}")
        suffix += f"_{reference}"
    if no_gross:
        protocol = dataclasses.replace(protocol, qc="none")
    return protocol, suffix


def protocol_label(protocol):
    """The protocol's choices, one line each, for figure titles."""
    qc = {
        "gross": "5-MAD consensus gross check",
        "none": "no gross check",
        "wmo_union": "WMO-No. 485 limits on obs minus each reference analysis, "
        "rejected if any rejects",
        "wmo_intersection": "WMO-No. 485 limits on obs minus each reference analysis "
        f"({', '.join(REFERENCES)}), rejected only if all reject",
    }.get(
        protocol.qc,
        "WMO-No. 485 limits on obs minus the "
        f"{protocol.qc.removeprefix('wmo_')} analysis",
    )
    window = "t - 30 min" if protocol.match == "strict" else "t - 90 min"
    match = f"radiosondes matched in [{window}, t + 30 min)"
    tas = ELEVATION_VARIANTS[protocol.elevation][2]
    return f"{qc[0].upper()}{qc[1:]}; {match}.\n{tas}."


# The reference analyses of the wmo check by default, and the periods of lead-0 pairs
# (obs_scores.py score --pairs) read by default, under
# $E2S_CAMPAIGN_RUNS_ROOT/_observation_scores/<period>/pairs/all/<source>.parquet.
REFERENCES = ("era5", "ifs_hres")
PAIR_PERIODS = ("year2025", "jansep2026")
# Gross-error limits on observation minus a reference analysis, METHOD.md section 6:
# WMO-No. 485 Appendix 2.1.2 for heights (m), temperature (K), vector wind (m/s) and station
# pressure (15 hPa, on the departure from the analysis moved to the station elevation);
# ECMWF's 15 K for 2 m temperature; the specific humidity and 10 m wind limits are this
# pipeline's own.
WMO_Z = {1000: 100, 925: 100, 850: 100, 700: 100, 500: 150, 400: 175, 300: 200, 250: 225,
         200: 250, 150: 275, 100: 300, 70: 375, 50: 400, 30: 450}  # fmt: skip
WMO_WIND = {1000: 35, 925: 35, 850: 35, 700: 40, 500: 45, 400: 50, 300: 60, 250: 60,
            200: 50, 150: 50, 100: 45, 50: 45}  # fmt: skip
# Q_SD_MULTIPLE (upper air) and 10 (2 m) robust standard deviations of observation minus ERA5 over
# 2025-01 to 2026-09, all inits, rounded to two significant figures; the multiples the WMO and
# 2 m temperature limits are of those departures (METHOD.md section 6).
Q_SD_MULTIPLE = 20
Q_LIMITS = {1000: 0.012, 925: 0.011, 850: 0.0099, 700: 0.0072, 500: 0.0031, 400: 0.0015,
            300: 0.00056, 250: 0.00032, 200: 0.00019, 150: 0.000070, 100: 0.000030,
            50: 0.000053}  # fmt: skip
SURFACE_LIMITS = {"tas": 15.0, "sfcwind": 30.0, "uas": 30.0, "vas": 30.0, "q2m": 0.0063,
                  "sp": 1500.0}  # fmt: skip
# Largest share of the kept upper-air reports of a field without a pair in every reference
# analysis; accumulate stops above it (the reference check cannot judge those reports).
UNCHECKED_LIMIT = 0.01
# Figure names and order of every quantity: GH (radiosonde geopotential height, Z / g),
# T, Wind (vector wind), Q, then surface.
QUANTITIES = {
    **{f"Z{p}": (f"GH{p}", "m", "upper") for p in LEVELS},
    **{f"T{p}": (f"T{p}", "K", "upper") for p in LEVELS},
    **{f"W{p}": (f"Wind {p}", "m/s", "upper") for p in LEVELS},
    **{f"U{p}": (f"U{p}", "m/s", "upper") for p in LEVELS},
    **{f"V{p}": (f"V{p}", "m/s", "upper") for p in LEVELS},
    **{f"Q{p}": (f"Q{p}", "kg/kg", "upper") for p in Q_LEVELS},
    "tas": ("t2m", "K", "surface"),
    "sfcwind": ("10 m wind speed", "m/s", "surface"),
    "sp": ("sp", "Pa", "surface"),
    "q2m": ("q2m", "kg/kg", "surface"),
}
SUMMARY = [
    ("GH", [(f"{p} hPa", f"Z{p}", "global") for p in (250, 500, 850)]),
    ("T", [(f"{p} hPa", f"T{p}", "global") for p in (250, 500, 850)]),
    ("Wind", [(f"{p} hPa", f"W{p}", "global") for p in (250, 500, 850)]),
    ("Q", [(f"{p} hPa", f"Q{p}", "global") for p in (500, 700, 850)]),
    ("Surface", [("t2m", "tas", "global"), ("10 m wind speed", "sfcwind", "global")]),
]
# The paper's summary: two rows of three panels.
SUMMARY_PAPER = [
    (
        "",
        [
            ("GH500", "Z500", "global"),
            ("T850", "T850", "global"),
            ("Wind 250", "W250", "global"),
        ],
    ),
    (
        "",
        [
            ("Q700", "Q700", "global"),
            ("t2m", "tas", "global"),
            ("10 m wind speed", "sfcwind", "global"),
        ],
    ),
]


class Experiment:
    """The experiment's IC sources (label, colour, pairs files) for one model and cycle
    group, and where its figures go."""

    def __init__(
        self,
        experiment,
        model,
        cfg,
        runs,
        out=None,
        samples=None,
        protocol=DEFAULT_PROTOCOL,
        reference=DEFAULT_REFERENCE_RULE,
        no_gross=False,
    ):
        exp = yaml.safe_load((cfg / "experiments" / f"{experiment}.yaml").read_text())
        sample = samples or (exp["cycles"][0] if len(exp["cycles"]) == 1 else None)
        if sample is None:
            raise SystemExit(f"--samples: one of {exp['cycles']}")
        self.runs = runs
        self.systems, self.files = {}, {}
        pooled = exp.get("pooled_from", [])
        for ic in exp["ic_sources"]:
            style = ic_style(cfg, ic)
            if pooled:
                self.systems[ic] = (style["label"], style["color"])
                continue
            cell = RUN_NAME.format(
                experiment=run_experiment(exp, experiment, ic, model),
                model=model,
                ic=ic,
                sample=sample,
            )
            dirs = segments(runs, cell)
            files = sorted(f for d in dirs for f in d.glob(PAIRS_GLOB))
            if not files:
                print(f"{ic}: no pairs under {[str(d) for d in dirs]}; left out")
                continue
            self.systems[ic] = (style["label"], style["color"])
            self.files[ic] = files
        if len(self.systems) < 2:
            raise SystemExit("fewer than two IC sources have pairs")
        refs = [r for r in exp["scoring_references"] if r in self.systems]
        if not refs:
            raise SystemExit(
                f"none of the scoring_references {exp['scoring_references']} has pairs"
            )
        self.reference = refs[0]
        self.first = next(iter(self.systems))
        self.model = NICE.get(model, model)
        self.lead_hours = exp["run"]["lead_hours"]
        self.out = out or runs / "_figures" / experiment / "obs" / f"{model}_{sample}"
        self.protocol, self.suffix = resolve_protocol(protocol, reference, no_gross)
        self.sums = self.out / "sums"
        self.inits = self.out / "inits.json"
        self.pooled_from = pooled
        self.pooled = [
            runs / "_figures" / x / "obs" / f"{model}_{sample}" for x in pooled
        ]

    @property
    def others(self):
        return [s for s in self.systems if s != self.reference]

    @property
    def pairs(self):
        """(subject, baseline) pairs beyond each IC against the reference."""
        if self.first == self.reference:
            return []
        return [(self.first, s) for s in self.others if s != self.first]

    @functools.cached_property
    def inits_info(self):
        if not self.pooled:
            return json.loads(self.inits.read_text())
        parts = [json.loads((d / "inits.json").read_text()) for d in self.pooled]
        parts.sort(key=lambda p: p["first"])
        return {
            "n": sum(p["n"] for p in parts),
            "first": min(p["first"] for p in parts),
            "last": max(p["last"] for p in parts),
            "hours": sorted({h for p in parts for h in p["hours"]}),
            "max_lead": min(p["max_lead"] for p in parts),
            "ranges": [(p["first"], p["last"]) for p in parts],
        }

    def inits_label(self):
        inits = self.inits_info
        hours = "/".join(f"{h:02d}" for h in inits["hours"])
        ranges = inits.get("ranges", [(inits["first"], inits["last"])])
        spans = " and ".join(f"{first[:10]} to {last[:10]}" for first, last in ranges)
        return f"{inits['n']} inits at {hours} UTC, {spans}"

    def systems_label(self):
        return ", ".join(label for label, _ in self.systems.values())

    def protocol_label(self):
        return protocol_label(self.protocol)

    def selected_sums(self, filters=()):
        return selected_sums(self.pooled or [self.out], self.protocol, filters)

    def label(self, s):
        return self.systems[s][0]

    def valid_0012(self, lead):
        # Leads valid at 00/12 UTC; every init of a cycle group has the same hour mod 12.
        return (lead + self.inits_info["hours"][0]) % 12 == 0


def wmo_limit(field):
    if field in SURFACE_LIMITS:
        return SURFACE_LIMITS[field]
    var, level = field[0], field[1:]
    if not level.isdigit():
        return np.inf
    p = int(level)
    if var == "T":
        return 15.0 if p > 700 or p <= 50 else 10.0
    limits = {"Z": WMO_Z, "U": WMO_WIND, "V": WMO_WIND, "Q": Q_LIMITS}.get(var, {})
    return limits.get(p, np.inf)


def wind_partner(field, fields):
    """Index in ``fields`` of the U component of a wind component field, else -1."""
    pairs = {"uas": "uas", "vas": "uas"}
    if field[0] in "UV" and field[1:].isdigit():
        pairs[field] = f"U{field[1:]}"
    u = pairs.get(field)
    return fields.index(u) if u in fields else -1


def string_hashes(column):
    encoded = pc.dictionary_encode(column.combine_chunks(), null_encoding="encode")
    values = encoded.dictionary.to_numpy(zero_copy_only=False).astype(object)
    return pd.util.hash_array(values)[encoded.indices.to_numpy()]


def hash_columns(*columns):
    frame = pd.DataFrame(dict(enumerate(columns)))
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def report_keys(table):
    """One 64-bit hash per report: station, field, observation time and observed value."""
    time = table["time"].cast(pa.timestamp("ns")).to_numpy().view(np.int64)
    station = string_hashes(table["station"])
    field = string_hashes(table["field"])
    return hash_columns(station, field, time, table["obs"].to_numpy())


def station_values(station, time, code, obs, source, target):
    """For each report of field code ``target``, the observed value of field code ``source``
    at the same station and time (NaN if there is none); other rows NaN."""
    out = np.full(len(code), np.nan)
    have, want = code == source, code == target
    lookup = pd.Series(obs[have], hash_columns(station[have], time[have]))
    lookup = lookup[~lookup.index.duplicated()]
    out[want] = lookup.reindex(hash_columns(station[want], time[want])).to_numpy()
    return out


def surface_pressure_at_station(model, tas_obs, q2m_obs, change):
    """Model surface pressure moved from the ERA5 terrain to the station elevation, through a
    layer whose temperature is the report's observed 2 m temperature taken back to the
    terrain at 6.5 K/km, with the report's 2 m humidity (0 if none)."""
    start = temperature_at_height(tas_obs, -change)
    return pressure_at_height(model, start, change, np.nan_to_num(q2m_obs))


def reference_flags(path, fields, terrain):
    """Sorted keys of a reference analysis' reports whose departure exceeds ``wmo_limit``,
    and sorted keys of all its reports. Wind components are judged on the vector departure
    of the station's U and V at the same time and level (a missing partner counts as 0);
    sp on the departure from the analysis moved to the station elevation (raw where the
    report has no 2 m temperature)."""
    t = pq.read_table(
        path,
        columns=["station", "field", "time", "obs", "model", "lat", "lon", "elevation"],
        filters=[("field", "in", fields)],
    )
    t = t.filter(pc.is_finite(t["model"]))
    code = field_codes(t, fields)
    obs, model = t["obs"].to_numpy(), t["model"].to_numpy().copy()
    station = string_hashes(t["station"])
    time = t["time"].cast(pa.timestamp("ns")).to_numpy().view(np.int64)
    sp = code == fields.index("sp")
    tas = station_values(
        station, time, code, obs, fields.index("tas"), fields.index("sp")
    )
    q2m = station_values(
        station, time, code, obs, fields.index("q2m"), fields.index("sp")
    )
    change = station_height_change(
        terrain,
        era5_grid.LAT,
        era5_grid.LON,
        t["lat"].to_numpy()[sp],
        t["lon"].to_numpy()[sp],
        t["elevation"].to_numpy(zero_copy_only=False).astype(float)[sp],
    )
    moved = surface_pressure_at_station(model[sp], tas[sp], q2m[sp], change)
    model[sp] = np.where(np.isfinite(moved), moved, model[sp])
    departure = obs - model
    limit = np.array([wmo_limit(f) for f in fields])[code]
    partner = np.array([wind_partner(f, fields) for f in fields])[code]
    wind = partner >= 0
    flagged = ~wind & (np.abs(departure) > limit)
    sounding = hash_columns(station[wind], time[wind], partner[wind])
    frame = pd.DataFrame(
        {
            "sounding": sounding,
            "is_u": code[wind] == partner[wind],
            "d": departure[wind],
            "limit": limit[wind],
        }
    )
    u, v = (
        frame[frame.is_u == is_u].drop_duplicates("sounding").set_index("sounding")
        for is_u in (True, False)
    )
    both = u.join(v, how="outer", lsuffix="_u", rsuffix="_v")
    vector = np.hypot(both.d_u.fillna(0.0), both.d_v.fillna(0.0))
    bad = both.index[vector > both.limit_u.fillna(both.limit_v)]
    flagged[np.flatnonzero(wind)[np.isin(sounding, bad.to_numpy())]] = True
    print(f"{path}: {flagged.sum():,} of {len(flagged):,} reports flagged", flush=True)
    return np.unique(report_keys(t.filter(flagged))), np.unique(report_keys(t))


def limits_fingerprint(path, fields):
    """Changes with the pairs file, the fields and every limit."""
    stat = Path(path).stat()
    text = repr(
        (stat.st_size, stat.st_mtime_ns, fields, [wmo_limit(f) for f in fields])
    )
    return hashlib.sha256(text.encode()).hexdigest()


def cached_reference(path, fields, terrain):
    """``reference_flags`` of one pairs file, saved as reference_{flags,reports}_<name>.parquet
    in its period directory (<period>/pairs/all/<name>.parquet) and reused while the file,
    the fields and the limits are unchanged."""
    path = Path(path)
    home = path.parents[2] if path.parent.parent.name == "pairs" else path.parent
    saved = [
        home / f"reference_{kind}_{path.stem}.parquet" for kind in ("flags", "reports")
    ]
    fingerprint = limits_fingerprint(path, fields)
    if all(f.exists() for f in saved):
        metadata = pq.read_schema(saved[0]).metadata or {}
        if metadata.get(b"fingerprint", b"").decode() == fingerprint:
            return tuple(pq.read_table(f)["key"].to_numpy() for f in saved)
    keys = reference_flags(path, fields, terrain)
    for f, k in zip(saved, keys):
        table = pa.table({"key": k}).replace_schema_metadata(
            {"fingerprint": fingerprint}
        )
        pq.write_table(table, f)
    return keys


def reference_rules(items, fields, terrain, all_checks):
    """The rejected report keys of each reference check, and per reference the sorted report
    keys of each of its files (for ``covered``). With two or more references:
    ``wmo_intersection`` (all reject), and with ``all_checks`` also ``wmo_<name>`` per
    reference and ``wmo_union``; with one: its ``wmo_<name>``."""
    flags, reports = {}, {}
    for item in items:
        name, paths = item.split("=", 1)
        parts = [cached_reference(p, fields, terrain) for p in paths.split(",")]
        flags[name] = functools.reduce(np.union1d, [f for f, _ in parts])
        reports[name] = [r for _, r in parts]
    rules = {}
    if all_checks or len(flags) == 1:
        rules.update({f"wmo_{name}": keys for name, keys in flags.items()})
    if len(flags) > 1:
        rules["wmo_intersection"] = functools.reduce(np.intersect1d, flags.values())
        if all_checks:
            rules["wmo_union"] = functools.reduce(np.union1d, flags.values())
    # References whose files hold the same reports (one scoring of the same observations)
    # are checked once.
    distinct = []
    for files in reports.values():
        if not any(
            len(files) == len(other)
            and all(np.array_equal(a, b) for a, b in zip(files, other))
            for other in distinct
        ):
            distinct.append(files)
    return rules, distinct


def contains(sorted_keys, keys):
    i = np.searchsorted(sorted_keys, keys).clip(max=max(len(sorted_keys) - 1, 0))
    return sorted_keys[i] == keys if len(sorted_keys) else np.zeros(len(keys), bool)


def covered(keys):
    """Whether every reference analysis has each report (in one of its files)."""
    found = np.ones(len(keys), bool)
    for files in COVERED:
        found &= np.any([contains(f, keys) for f in files], axis=0)
    return found


# Per reference, the sorted report keys of each of its files; set by accumulate before its
# worker processes fork, so the workers share them instead of receiving a copy per init.
COVERED = []


@functools.cache
def figure_regions():
    standard = standard_regions()
    return {name: standard[key] for name, key in REGIONS.items()}


def aligned(tables, order_keys):
    """The systems' tables sorted stably by order_keys, checked to hold the same reports."""
    out = {}
    for s, t in tables.items():
        order = np.lexsort([t[k] for k in reversed(order_keys)])
        out[s] = {k: v[order] for k, v in t.items()}
    check_same_reports(out, ("obs", "lat", "lon", "code"), " after sorting")
    return out


def field_codes(table, fields):
    """Index of each report's field in ``fields``, -1 for other fields."""
    codes = pc.index_in(table["field"], value_set=pa.array(fields))
    return codes.fill_null(-1).to_numpy()


def strict_rows(table):
    """Reports within the strict window of their valid time (``match`` strict)."""
    return within(table["time"].to_numpy(), table["cycle"].to_numpy())


def error_sums(errors, systems, fields, reports, matches, rules, none):
    """Squared and plain error sums and report counts per field, lead and init, for every
    region, gross check and match in ``matches`` ("strict": the reports within the strict
    window, any other: every report). ``errors`` is (system, report); ``reports`` holds
    per-report code, lead, init, group (of the gross check), strict, lat, lon and
    ``rejected`` (report, rule): the reference checks named ``rules``. ``none``: also the
    sums without any check."""
    inits, init_index = np.unique(reports["init"], return_inverse=True)
    lead = reports["lead"].astype(np.int64)
    key = (reports["code"] * (lead.max() + 1) + lead) * len(inits) + init_index
    keys, index = np.unique(key, return_inverse=True)
    first = np.empty(len(keys), np.int64)
    first[index] = np.arange(len(index))
    labels = {
        "field": np.asarray(fields, dtype=object)[reports["code"][first]],
        "lead": lead[first],
        "init": inits[init_index[first]].astype("datetime64[ns]"),
    }
    names, masks = point_regions(figure_regions(), reports["lat"], reports["lon"])
    regions = dict(zip(names, masks))
    everything = np.ones(len(index), bool)
    blocks = []
    for match in matches:
        matched = reports["strict"] if match == "strict" else everything
        if match == "strict" and matched.all() and blocks:
            blocks += [(*b[:-1], "strict") for b in blocks]
            continue
        gross = np.zeros(len(index), bool)
        gross[matched] = consensus_keep(
            errors[:, matched], reports["group"][matched], GROSS
        )
        checks = {"gross": gross, **({"none": matched} if none else {})}
        for i, rule in enumerate(rules):
            checks[rule] = matched & ~reports["rejected"][:, i]
        for qc, checked in checks.items():
            for region, in_region in regions.items():
                selected = checked & in_region
                k = index[selected]
                part = {"n": np.bincount(k, minlength=len(keys))}
                for s, e in zip(systems, errors[:, selected]):
                    part[f"sq_{s}"] = np.bincount(k, e * e, len(keys))
                    part[f"e_{s}"] = np.bincount(k, e, len(keys))
                cells = np.flatnonzero(part["n"])
                part = {c: v[cells] for c, v in part.items()}
                part.update({c: v[cells] for c, v in labels.items()})
                blocks.append((part, region, qc, match))
    sizes = [len(b[0]["n"]) for b in blocks]
    table = pd.DataFrame(
        {c: np.concatenate([b[0][c] for b in blocks]) for c in blocks[0][0]}
    )
    for i, column in enumerate(("region", "qc", "match"), 1):
        table[column] = np.repeat([b[i] for b in blocks], sizes)
    return table


def rejected_by(rules, keys):
    """(report, rule): whether each rule's reference check rejects the report."""
    rejected = np.zeros((len(keys), len(rules)), bool)
    for i, rejected_keys in enumerate(rules.values()):
        rejected[:, i] = contains(rejected_keys, keys)
    return rejected


def init_reports(files, fields, terrain, rules):
    """One init's per-report errors (system, report) and report table: the reports every
    IC source has a model value for, with ``change`` (station elevation minus terrain),
    ``rejected`` (``rejected_by``), ``covered`` (in every reference analysis) and,
    for sp, the same report's observed ``tas_obs`` and ``q2m_obs``."""
    tables = {}
    for s, paths in files.items():
        t = pq.read_table(
            paths,
            columns=["init", "lead", "field", "obs", "model", "lat", "lon"]
            + ["time", "cycle", "elevation", "station"],
        )
        tables[s] = {k: t[k].to_numpy() for k in ("init", "lead", "obs", "model")}
        tables[s]["elevation"] = t["elevation"].to_numpy(zero_copy_only=False)
        tables[s].update(lat=t["lat"].to_numpy(), lon=t["lon"].to_numpy())
        tables[s]["code"] = field_codes(t, fields)
        tables[s]["strict"] = strict_rows(t)
        if s == next(iter(files)):
            tables[s]["key"] = report_keys(t)
            tables[s]["station"] = string_hashes(t["station"])
            tables[s]["time"] = (
                t["time"].cast(pa.timestamp("ns")).to_numpy().view(np.int64)
            )
    tables = aligned(tables, ["lead"])
    first = next(iter(tables.values()))
    keep = (first["code"] >= 0) & np.all(
        [np.isfinite(t["model"]) for t in tables.values()], axis=0
    )
    errors = np.stack([(t["model"] - t["obs"])[keep] for t in tables.values()])
    names = ("code", "lead", "strict", "lat", "lon", "elevation", "obs")
    reports = {k: first[k][keep] for k in names}
    reports["rejected"] = rejected_by(rules, first["key"][keep])
    reports["covered"] = covered(first["key"][keep])
    for name in ("tas", "q2m"):
        reports[f"{name}_obs"] = station_values(
            *(first[k][keep] for k in ("station", "time", "code", "obs")),
            fields.index(name),
            fields.index("sp"),
        )
    reports["init"] = np.full(len(errors[0]), first["init"][0].astype("datetime64[ns]"))
    reports["group"] = reports["lead"].astype(np.int64) * len(fields) + reports["code"]
    reports["change"] = station_height_change(
        terrain,
        era5_grid.LAT,
        era5_grid.LON,
        *(reports[k] for k in ("lat", "lon", "elevation")),
    )
    return first["init"][0], int(first["lead"].max()), errors, reports


def elevation_variants(errors, reports, fields):
    """(variant, errors, reports) of every station-elevation variant: the reports within the
    window (stations without an elevation are outside every window), the model moved to the
    station elevation. tas: a shift, applied to the error; sp: ``surface_pressure_at_station``
    on each system's model value; sp reports without a 2 m temperature are left out."""
    change = reports["change"]
    sp = reports["code"] == fields.index("sp")
    for variant, (window, corrected, _) in ELEVATION_VARIANTS.items():
        inside = (
            np.ones(len(change), bool) if window is None else np.abs(change) <= window
        )
        if corrected:
            inside &= ~sp | np.isfinite(reports["tas_obs"])
        selected = {k: v[inside] for k, v in reports.items()}
        moved = errors[:, inside].copy()
        if corrected:
            c, is_sp = change[inside], sp[inside]
            moved[:, ~is_sp] = temperature_at_height(moved[:, ~is_sp], c[~is_sp])
            obs = selected["obs"][is_sp]
            model = surface_pressure_at_station(
                moved[:, is_sp] + obs,
                selected["tas_obs"][is_sp],
                selected["q2m_obs"][is_sp],
                c[is_sp],
            )
            moved[:, is_sp] = model - obs
        yield variant, moved, selected


def init_sums(files, fields, terrain, rules, all_checks):
    """One init's error sums from each IC source's pairs files of that init, and per field
    the reports kept by the first reference check and how many of them no reference has.
    """
    init, max_lead, errors, reports = init_reports(files, fields, terrain, rules)
    systems = list(files)
    sonde = np.isin(reports["code"], [fields.index(f) for f in UPPER])
    if not reports["strict"][~sonde].all():
        raise SystemExit(f"{init}: surface reports outside the strict window")
    corrected = np.isin(reports["code"], [fields.index(f) for f in ELEVATION_FIELDS])
    surface = ~sonde & ~corrected
    paper = resolve_protocol()[0].qc
    rule = list(rules).index(paper) if paper in rules else 0
    kept = ~reports["rejected"][:, rule] if rules else np.ones(len(sonde), bool)
    code = reports["code"][kept]
    unchecked = (~reports["covered"][kept]).astype(float)
    present = np.unique(code)
    coverage = pd.DataFrame(
        {
            "field": np.asarray(fields, dtype=object)[present],
            "kept": np.bincount(code, minlength=len(fields))[present],
            "unchecked": np.bincount(code, unchecked, len(fields))[present],
        }
    )

    def select(mask):
        return errors[:, mask], {k: v[mask] for k, v in reports.items()}

    tables = []
    for mask, matches in ((sonde, MATCHES), (surface, [SURFACE_MATCH])):
        if mask.any():
            e, r = select(mask)
            tables.append(
                error_sums(e, systems, fields, r, matches, list(rules), all_checks)
            )
    for variant, e, r in elevation_variants(*select(corrected), fields):
        if len(e[0]):
            table = error_sums(
                e,
                systems,
                fields,
                r,
                [SURFACE_MATCH],
                list(rules),
                all_checks,
            )
            tables.append(table.assign(elevation=variant))
    table = pd.concat(tables, ignore_index=True)
    return init, max_lead, table.fillna({"elevation": "none"}), coverage


def files_by_init(files):
    found = {}
    for f in files:
        found.setdefault(f.name.rsplit("_r", 1)[0], []).append(f)
    return found


def period_files(runs, name):
    root = runs / "_observation_scores"
    paths = [root / p / "pairs" / "all" / f"{name}.parquet" for p in PAIR_PERIODS]
    return [p for p in paths if p.is_file()]


def write_inits(e, inits, max_lead):
    print(len(inits), "inits common to all ICs", flush=True)
    e.inits.write_text(
        json.dumps(
            {
                "n": len(inits),
                "first": str(min(inits)),
                "last": str(max(inits)),
                "hours": sorted({t.hour for t in inits}),
                "max_lead": int(max_lead),
            }
        )
    )


def check_sums_schema(path):
    names = pq.read_schema(path).names
    if "init" not in names:
        raise SystemExit(
            f"{path} holds sums per init day (column day), not per init; "
            "rerun accumulate"
        )
    if "elevation" not in names:
        raise SystemExit(f"{path} predates the elevation column; rerun accumulate")


def mode_accumulate(e, a):
    global COVERED
    started = time.monotonic()
    e.sums.mkdir(parents=True, exist_ok=True)
    by_init = {s: files_by_init(e.files[s]) for s in e.systems}
    common = sorted(set.intersection(*(set(f) for f in by_init.values())))
    keys = ["field", "lead", "init", "region", "qc", "match", "elevation"]
    terrain = np.asarray(era5_grid.era5_orography(a.orography), np.float64)
    references = a.reference_pairs or [
        f"{r}={','.join(map(str, period_files(e.runs, r)))}" for r in REFERENCES
    ]
    rules, COVERED = reference_rules(references, FIELDS, terrain, a.all_checks)
    print(
        f"reference checks ready after {time.monotonic() - started:.0f} s", flush=True
    )
    parts, coverage, inits, max_lead = [], [], [], 0
    fork = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(a.workers, mp_context=fork) as pool:
        jobs = [
            pool.submit(
                init_sums,
                {s: by_init[s][i] for s in e.systems},
                FIELDS,
                terrain,
                rules,
                a.all_checks,
            )
            for i in common
        ]
        for done, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            init, lead, part, covered = job.result()
            inits.append(pd.Timestamp(init))
            max_lead = max(max_lead, lead)
            parts.append(part)
            coverage.append(covered)
            if done % 32 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} inits", flush=True)
    coverage = pd.concat(coverage).groupby("field")[["kept", "unchecked"]].sum()
    coverage["share"] = coverage.unchecked / coverage.kept
    print("kept reports without a pair in every reference analysis:", flush=True)
    print(coverage[coverage.unchecked > 0].to_string(), flush=True)
    upper = coverage[coverage.index.isin(UPPER)]
    if (upper.share > UNCHECKED_LIMIT).any():
        bad = upper[upper.share > UNCHECKED_LIMIT].share.round(4).to_dict()
        raise SystemExit(
            f"more than {UNCHECKED_LIMIT:.0%} of kept upper-air reports have no reference "
            f"pair: {bad}; the reference analyses' pairs miss these observations"
        )
    write_inits(e, inits, max_lead)
    # One row per field, lead, init, region, check, match and elevation.
    table = pd.concat(parts, ignore_index=True).groupby(keys, sort=True).sum()
    table.reset_index().to_parquet(e.sums / "forecast.parquet")
    print(e.sums / "forecast.parquet", flush=True)
    print(f"accumulate took {time.monotonic() - started:.0f} s", flush=True)


def selected_sums(dirs, protocol, filters=()):
    """The sums (<dir>/sums/forecast.parquet of each of ``dirs``, whose inits must be
    disjoint) of the protocol's gross check, radiosonde match and 2 m temperature variant;
    ``filters`` (pyarrow filters) narrow the read."""
    parts, seen = [], set()
    for d in dirs:
        path = d / "sums" / "forecast.parquet"
        check_sums_schema(path)
        part = pq.read_table(
            path,
            filters=[
                ("qc", "==", protocol.qc),
                ("match", "in", [protocol.match, SURFACE_MATCH]),
                *filters,
            ],
        ).to_pandas()
        inits = set(part["init"])
        if seen & inits:
            raise SystemExit(f"pooled experiments share forecast inits: {dirs}")
        seen |= inits
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    corrected = table.field.isin(ELEVATION_FIELDS)
    table = table[table.elevation == np.where(corrected, protocol.elevation, "none")]
    if not len(table):
        raise SystemExit(
            f"no sums for qc {protocol.qc}; rerun accumulate (--all-checks)"
        )
    if not (table.field == "tas").any():
        raise SystemExit(
            f"no 2 m temperature sums for {protocol.elevation}; rerun accumulate"
        )
    # Sums written before the source column went still hold it.
    return table.drop(columns=["qc", "match", "elevation", "source"], errors="ignore")


def vector_components(name):
    if name == "w10m":
        return "uas", "vas"
    if name.startswith("W"):
        return f"U{name[1:]}", f"V{name[1:]}"
    return None


def combine(table, name, systems):
    """Per-init sums of one plotted quantity: a field, or vector wind (W{p} from U{p} and
    V{p}, w10m from uas and vas)."""
    components = vector_components(name)
    if components:
        u, v = (
            table[table.field == c].set_index(["region", "lead", "init"])
            for c in components
        )
        cols = [f"sq_{s}" for s in systems]
        both = u.index.intersection(v.index)
        out = u.loc[both, cols] + v.loc[both, cols]
        out["n"] = u.loc[both, "n"]
        for s in systems:
            out[f"e_{s}"] = np.nan
        return out.reset_index()
    return table[table.field == name]


def curve(e, inits, rng):
    """RMSE, bias and % RMSE differences per lead, with 95 % paired bootstrap intervals
    over inits (the same resampled inits for every system)."""
    rows = []
    comparisons = [(s, e.reference, f"pct_{s}") for s in e.others] + [
        (s, b, f"pct_{s}_vs_{b}") for s, b in e.pairs
    ]
    for lead, g in inits.groupby("lead"):
        n = g["n"].to_numpy(float)
        sq = {s: g[f"sq_{s}"].to_numpy() for s in e.systems}
        row = {"lead": lead, "n": int(n.sum()), "inits": len(g)}
        for s in e.systems:
            row[f"rmse_{s}"] = np.sqrt(sq[s].sum() / n.sum())
            row[f"bias_{s}"] = g[f"e_{s}"].sum() / n.sum()
        pick = rng.integers(0, len(g), (NBOOT, len(g)))
        boot = {s: np.sqrt(sq[s][pick].sum(1) / n[pick].sum(1)) for s in e.systems}
        for s in e.systems:
            row[f"rmse_{s}_lo"], row[f"rmse_{s}_hi"] = np.percentile(
                boot[s], [2.5, 97.5]
            )
        for s, b, key in comparisons:
            row[key] = 100 * (row[f"rmse_{s}"] / row[f"rmse_{b}"] - 1)
            row[f"{key}_lo"], row[f"{key}_hi"] = np.percentile(
                100 * (boot[s] / boot[b] - 1), [2.5, 97.5]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def is_upper(name):
    return name[0] in "ZTWUVQ" and name[1:].isdigit()


def curves_table(e, names=tuple(QUANTITIES)):
    """``curve`` of each quantity in ``names`` (fields or vector winds) and region: every
    region upper air, the surface in three; upper air at the leads valid 00/12 UTC."""
    table = e.selected_sums()
    out = []
    for name in names:
        q = combine(table, name, e.systems)
        if q.empty:
            continue
        upper = is_upper(name)
        regions = list(REGIONS) if upper else ["global", "N America", "Europe"]
        for region in regions:
            r = q[q.region == region]
            if upper:
                r = r[e.valid_0012(r.lead)]
            # One bootstrap stream per curve: adding a region leaves the others unchanged.
            seed = hashlib.sha256(f"{name} {region}".encode()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed, "little"))
            out.append(curve(e, r, rng).assign(quantity=name, region=region))
    return pd.concat(out, ignore_index=True)


def panel_letter(i):
    letters = "abcdefghijklmnopqrstuvwxyz"
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def plot_blocks(e, table, blocks, title, path):
    """Labelled blocks of panels; ``blocks`` is ``[(heading, [(column title, quantity,
    region)])]``. Each panel: RMSE with init-bootstrap bands over % vs the reference IC.
    ``title`` goes to <path>.caption.txt, not into the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = max(len(cells) for _, cells in blocks)
    # Drawn at the paper's page width, so the fonts below are their printed sizes.
    with plt.rc_context(
        {"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "lines.markersize": 3, "lines.linewidth": 1.2}
    ):  # fmt: skip
        fig = plt.figure(
            figsize=(7.2 * ncol / 3, 3.0 * len(blocks) + 0.5), layout="constrained"
        )
        subfigs = fig.subfigures(len(blocks), 1, hspace=0.04)
        subfigs = [subfigs] if len(blocks) == 1 else list(subfigs)
        n_inits = e.inits_info["n"]
        letter = 0
        for i, ((heading, cells), block) in enumerate(zip(blocks, subfigs)):
            block.suptitle(heading, fontsize=9, fontweight="bold", x=0.01, ha="left")
            axes = block.subplots(
                2, ncol, sharex=True, gridspec_kw={"height_ratios": [1, 1]}
            )
            axes = axes.reshape(2, ncol)
            for j in range(len(cells), ncol):
                axes[0, j].set_visible(False)
                axes[1, j].set_visible(False)
            for j, (column, name, region) in enumerate(cells):
                _, unit, _ = QUANTITIES[name]
                factor, unit = display_unit(unit)
                part = table[
                    (table.quantity == name) & (table.region == region)
                ].sort_values("lead")
                raw, rel = axes[0, j], axes[1, j]
                days = part.lead / 24
                for s, (slabel, colour) in e.systems.items():
                    raw.plot(
                        days,
                        factor * part[f"rmse_{s}"],
                        ".-",
                        color=colour,
                        label=slabel,
                    )
                    raw.fill_between(
                        days,
                        factor * part[f"rmse_{s}_lo"],
                        factor * part[f"rmse_{s}_hi"],
                        color=colour,
                        alpha=0.15,
                        lw=0,
                    )
                    if s == e.reference:
                        rel.plot(days, 0 * days, "-", color=colour, lw=1)
                        continue
                    rel.plot(days, part[f"pct_{s}"], ".-", color=colour)
                    rel.fill_between(
                        days,
                        part[f"pct_{s}_lo"],
                        part[f"pct_{s}_hi"],
                        color=colour,
                        alpha=0.2,
                        lw=0,
                    )
                for ax in (raw, rel):
                    if (part.lead == 0).any():
                        ax.axvspan(-2 / 24, 2 / 24, color="0.9", zorder=0)
                    ax.grid(alpha=0.3)
                raw.set_title(f"{column} [{unit}]", fontsize=9)
                raw.text(
                    -0.02,
                    1.04,
                    panel_letter(letter),
                    transform=raw.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    ha="right",
                )
                letter += 1
                # Median over leads, as some leads have fewer valid times with reports.
                per_time = part.n.median() / n_inits if len(part) else 0
                raw.text(
                    0.98,
                    0.04,
                    f"~{per_time:,.0f} reports per valid time",
                    transform=raw.transAxes,
                    ha="right",
                    fontsize=6,
                    color="0.4",
                )
                rel.set_xticks(range(0, e.lead_hours // 24 + 1))
                rel.set_xlabel("Lead time [days]" if i == len(blocks) - 1 else "")
                raw.set_ylabel("RMSE" if j == 0 else "")
                if j == 0:
                    rel.set_ylabel(f"\u0394RMSE [%] vs {e.label(e.reference)} IC")
        handles, labels = subfigs[0].axes[0].get_legend_handles_labels()
        # Legend order of the paper figures: the subject, then the other ICs, the reference last.
        reference = e.label(e.reference)
        pairs = sorted(zip(labels, handles), key=lambda p: p[0] == reference)
        # In the first RMSE panel, whose curves start low on the left.
        subfigs[0].axes[0].legend(
            [h for _, h in pairs], [lab for lab, _ in pairs], loc="upper left", fontsize=7,
            frameon=False, handlelength=1.5,
        )  # fmt: skip
        fig.savefig(path, dpi=110)
        fig.savefig(Path(path).with_suffix(".pdf"))
        plt.close(fig)
    Path(path).with_suffix(".caption.txt").write_text(title + "\n")
    print(path)


def plot_curves(e, table, names, regions, title, path):
    """One block per quantity, regions as columns."""
    blocks = [
        (
            QUANTITIES[name][0],
            [(REGION_LABELS[region][0], name, region) for region in regions],
        )
        for name in names
        if name in set(table.quantity)
    ]
    plot_blocks(e, table, blocks, title, path)


def plot_bias(e, table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, region in zip(axes, ["Europe", "N America"]):
        part = table[(table.quantity == "tas") & (table.region == region)].sort_values(
            "lead"
        )
        for s, (slabel, colour) in e.systems.items():
            ax.plot(part.lead, part[f"bias_{s}"], ".-", color=colour, label=slabel)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(-2, 2, color="0.92", zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, e.lead_hours + 1, 24))
        ax.set_xlabel("lead [h]")
        ax.set_title(f"t2m bias (model - station), {region}", fontsize=10)
    axes[0].set_ylabel("bias [K]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    Path(path).with_suffix(".caption.txt").write_text(
        f"{e.model} forecasts from {e.systems_label()} vs surface stations; "
        f"{e.inits_label()}; every 6 h of lead.\n{e.protocol_label()}\n"
    )
    print(path)


def write_markdown(e, table, path):
    leads = [0, 6, 24, 48, 72, 96, 114, 120]
    comparisons = [
        (f"pct_{s}", f"{e.label(s)} vs {e.label(e.reference)}") for s in e.others
    ]
    comparisons += [
        (f"pct_{s}_vs_{b}", f"{e.label(s)} vs {e.label(b)}") for s, b in e.pairs
    ]
    head = ["quantity", "region", "lead h", "reports"]
    head += [f"RMSE {e.label(s)}" for s in e.systems]
    head += [f"{title} % [95 %]" for _, title in comparisons]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for r in table[table.lead.isin(leads)].to_dict("records"):
        cells = [r["quantity"], r["region"], str(r["lead"]), f"{r['n']:,}"]
        cells += [f"{r[f'rmse_{s}']:.3f}" for s in e.systems]
        cells += [
            f"{r[k]:+.1f} [{r[f'{k}_lo']:+.1f}, {r[f'{k}_hi']:+.1f}]"
            for k, _ in comparisons
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(path)


def mode_plot(e, a):
    table = curves_table(e)
    e.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(e.out / f"tables{e.suffix}.csv", index=False, float_format="%.6g")
    write_markdown(e, table, e.out / f"tables{e.suffix}.md")
    inits = e.inits_label()
    head = f"{e.model} forecasts from {e.systems_label()} vs observations at valid time"
    inits = f"{e.protocol_label()}\n{inits}"
    upper_leads = "leads valid 00/12 UTC (one dot every 12 h)"
    plot_blocks(
        e,
        table,
        SUMMARY,
        f"{head}, global.\nUpper air: radiosondes, {upper_leads}. "
        "Surface: stations (one dot every 6 h).\n"
        f"{inits}; same reports for every system; shaded: {LEAD0_NOTE}; "
        f"bands: {INTERVAL}",
        e.out / f"summary_global{e.suffix}.png",
    )
    plot_blocks(
        e,
        table,
        SUMMARY_PAPER,
        f"{head}, global: GH500, T850, Wind 250 (vector wind), Q700 from radiosondes "
        f"({upper_leads}); t2m and 10 m wind speed from surface stations (every 6 h).\n"
        f"{inits}; same reports for every system; shaded: {LEAD0_NOTE}; "
        f"bands: {INTERVAL}",
        e.out / f"summary_paper_global{e.suffix}.png",
    )
    plot_curves(
        e,
        table,
        SURFACE_PLOTTED,
        ["global", "N America", "Europe"],
        f"{head}: surface stations.\n{inits}, one dot every 6 h of lead; same reports for "
        f"every system at each lead.\nShaded: {LEAD0_NOTE}. Bands: {INTERVAL}",
        e.out / f"curves_surface{e.suffix}.png",
    )
    for v, label, levels in (
        ("Z", "GH", LEVELS),
        ("T", "T", LEVELS),
        ("W", "Wind", LEVELS),
        ("U", "U", LEVELS),
        ("V", "V", LEVELS),
        ("Q", "Q", Q_LEVELS),
    ):
        plot_curves(
            e,
            table,
            [f"{v}{p}" for p in levels],
            list(REGIONS),
            f"{head}: radiosonde {label}.\n{inits}; {upper_leads}; "
            f"shaded: {LEAD0_NOTE}. Same reports for every system; bands: {INTERVAL}.\n"
            f"Regions: {REGION_CAPTION}",
            e.out / f"curves_radiosonde_{v}{e.suffix}.png",
        )
    plot_bias(e, table, e.out / f"bias_tas{e.suffix}.png")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("accumulate", "plot"):
        q = sub.add_parser(mode)
        q.add_argument("experiment")
        q.add_argument("--model", required=True, help="model key of the experiment")
        q.add_argument("--samples", help="cycle group; needed if there are several")
        q.add_argument(
            "--campaign-cfg",
            help="campaign configuration directory (default $E2S_CAMPAIGN_CFG)",
        )
        q.add_argument(
            "--runs-root", help="runs directory (default $E2S_CAMPAIGN_RUNS_ROOT)"
        )
        q.add_argument("--out", type=Path, help="output directory")
        if mode == "accumulate":
            q.add_argument(
                "--orography",
                required=True,
                help="ERA5 invariant surface geopotential file "
                "(e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc)",
            )
            q.add_argument(
                "--reference-pairs",
                nargs="+",
                metavar="NAME=FILE[,FILE]",
                help="reference analyses' lead-0 pairs (default: REFERENCES over the "
                "PAIR_PERIODS files)",
            )
            q.add_argument(
                "--all-checks",
                action="store_true",
                help="also the sums without a check (protocol eagle), of each reference "
                "alone (wmo_<name>) and of their union (wmo_union)",
            )
            q.add_argument(
                "--workers",
                type=int,
                default=min(32, os.cpu_count()),
                help="processes reading the pairs, one init each",
            )
        if mode == "plot":
            q.add_argument(
                "--protocol",
                choices=list(PROTOCOLS),
                default=DEFAULT_PROTOCOL,
                help="radiosonde match, gross check and 2 m temperature variant "
                "(METHOD.md section 6)",
            )
            q.add_argument(
                "--reference",
                default=DEFAULT_REFERENCE_RULE,
                metavar="RULE",
                help="protocol wmo: the reference check, a --reference-pairs name, "
                "union or intersection",
            )
            q.add_argument(
                "--no-gross",
                action="store_true",
                help="the protocol without the consensus gross check",
            )
    a = p.parse_args()
    cfg = directory(a.campaign_cfg, "E2S_CAMPAIGN_CFG")
    runs = directory(a.runs_root, "E2S_CAMPAIGN_RUNS_ROOT")
    if a.mode == "plot":
        e = Experiment(
            a.experiment, a.model, cfg, runs, a.out, a.samples, a.protocol,
            a.reference, a.no_gross,
        )  # fmt: skip
    else:
        e = Experiment(a.experiment, a.model, cfg, runs, a.out, a.samples)
    {"accumulate": mode_accumulate, "plot": mode_plot}[a.mode](e, a)


if __name__ == "__main__":
    main()
