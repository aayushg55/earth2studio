# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures of an observation-scored experiment, from the forecast pairs its runs wrote.

    python private/scripts/campaign/obs_figures.py accumulate <experiment> --model M \\
        [--analysis-pairs DIR | NAME=FILE ...] [--workers N]
    python private/scripts/campaign/obs_figures.py plot <experiment> --model M \\
        [--protocol ecmwf|eagle] [--no-gross]
    python private/scripts/campaign/obs_figures.py scorecards <experiment> --model M [same options]
    python private/scripts/campaign/obs_figures.py stations <experiment> --model M --leads 24 120

Output: $E2S_CAMPAIGN_RUNS_ROOT/_figures/<experiment>/obs/<model>_<cycles>/.

accumulate writes error sums per field, region, lead and init day for every radiosonde
time match, gross check and 2 m temperature variant (sums/forecast.parquet); with
--analysis-pairs (obs_scores.py score --pairs output: a directory of <ic source>.parquet,
or NAME=FILE) also the analyses' own lead-0 fit at 00/12 UTC (sums/analysis_upper.parquet).
plot (tables, summary, curves, 2 m temperature bias) and scorecards draw one protocol
from the sums; the file names end in _<protocol> (and _nogross). stations: per-station
maps and the European heatwave figure (station_figures.py). The protocols, every
processing step and the sums layout: private/obs_scoring/METHOD.md section 6.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from samples import REPO_ROOT, require, segments

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE.parent / "era5_e2e" / "paper_plots"))
from healda.utils.thermo import temperature_at_height  # noqa: E402
from private.obs_scoring import fields as era5_grid  # noqa: E402
from private.obs_scoring.observations import within  # noqa: E402
from private.obs_scoring.report import check_same_reports, consensus_keep  # noqa: E402
from private.obs_scoring.sample import station_height_change  # noqa: E402
from private.obs_scoring.score import point_regions  # noqa: E402
from private.scripts.region_masks import load_regions  # noqa: E402

CFG = HERE / "cfg"
RUN_NAME = "{experiment}__{model}__{ic}__{sample}"
PAIRS_GLOB = "obs_scores/pairs/all/*/*.parquet"
# q2m has a few reports with very large errors; it stays in sums and tables.
SURFACE_PLOTTED = ["tas", "sfcwind"]
LEVELS = [850, 500, 250]
Q_LEVELS = [850, 700, 500]
ALL_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50]
Q_ALL_LEVELS = [1000, 925, 850, 700, 500, 400, 300]
UPPER = [f"{v}{p}" for v in "TZUV" for p in ALL_LEVELS] + [
    f"Q{p}" for p in Q_ALL_LEVELS
]
FIELDS = UPPER + ["tas", "q2m", "sfcwind", "uas", "vas"]
# Surface fields under the names the paper plotters use.
SCORECARD_SURFACE = {"tas": "t2m", "sfcwind": "ws10m"}
# Figure name -> the scorer's standard region (region_masks.load_regions).
REGIONS = {
    "global": "global",
    "NH": "northern_hemisphere",
    "tropics": "tropics",
    "SH": "southern_hemisphere",
    "Europe": "europe",
    "N America": "north_america",
}
NBOOT = 1000
GROSS = 5.0
SUMS = ("forecast", "analysis_upper")
# Radiosonde time matches of the upper-air fields. Surface reports are read within the
# strict window, so their sums are the same for both matches and are stored once, as "any".
MATCHES = ("synoptic", "strict")
SURFACE_MATCH = "any"
# 2 m temperature variants (column ``elevation``; other fields have "none"): the largest
# |station elevation - ERA5 terrain| (m) of a kept station (None: every station), whether
# the model is moved from the ERA5 terrain to the station elevation
# (thermo.temperature_at_height, 6.5 K/km), and a label.
ELEVATION_FIELD = "tas"
ELEVATION_VARIANTS = {
    "none": (None, False, "2 m temperature at the model terrain"),
    "lapse6.5_win250": (
        250.0,
        True,
        "2 m temperature moved to the station elevation at 6.5 K/km, "
        "stations within 250 m of the ERA5 terrain",
    ),
}
# Protocols drawn by plot and scorecards (METHOD.md section 6).
PROTOCOLS = {
    "ecmwf": {"match": "synoptic", "qc": "gross", "elevation": "lapse6.5_win250"},
    "eagle": {"match": "strict", "qc": "none", "elevation": "none"},
}
QUANTITIES = {
    "tas": ("2 m temperature", "K", "surface"),
    "sfcwind": ("10 m wind speed", "m/s", "surface"),
    "q2m": ("2 m specific humidity", "kg/kg", "surface"),
    **{f"T{p}": (f"T {p} hPa", "K", "upper") for p in LEVELS},
    **{f"W{p}": (f"vector wind {p} hPa", "m/s", "upper") for p in LEVELS},
    **{f"U{p}": (f"U {p} hPa", "m/s", "upper") for p in LEVELS},
    **{f"V{p}": (f"V {p} hPa", "m/s", "upper") for p in LEVELS},
    **{f"Z{p}": (f"Z {p} hPa", "m", "upper") for p in LEVELS},
    **{f"Q{p}": (f"Q {p} hPa", "kg/kg", "upper") for p in Q_LEVELS},
}
SUMMARY = [
    ("Temperature", [(f"{p} hPa", f"T{p}", "global") for p in (250, 500, 850)]),
    ("Vector wind", [(f"{p} hPa", f"W{p}", "global") for p in (250, 500, 850)]),
    ("Geopotential height", [(f"{p} hPa", f"Z{p}", "global") for p in (250, 500, 850)]),
    ("Specific humidity", [(f"{p} hPa", f"Q{p}", "global") for p in (500, 700, 850)]),
    (
        "Surface",
        [
            ("2 m temperature", "tas", "global"),
            ("10 m wind speed", "sfcwind", "global"),
        ],
    ),
]


class Experiment:
    """The experiment's IC sources (label, colour, pairs files) for one model and cycle
    group, and where its figures go."""

    def __init__(self, a):
        from figures import NICE

        exp = yaml.safe_load((CFG / "experiments" / f"{a.experiment}.yaml").read_text())
        sample = a.samples or (exp["cycles"][0] if len(exp["cycles"]) == 1 else None)
        if sample is None:
            raise SystemExit(f"--samples: one of {exp['cycles']}")
        runs = Path(require("E2S_CAMPAIGN_RUNS_ROOT"))
        self.systems, self.files = {}, {}
        for ic in exp["ic_sources"]:
            style = yaml.safe_load((CFG / "ic_sources" / f"{ic}.yaml").read_text())
            # runs_from, as in figures.py: the experiment holding this source's runs.
            holder = exp.get("runs_from", {}).get(ic, a.experiment)
            cell = RUN_NAME.format(
                experiment=holder, model=a.model, ic=ic, sample=sample
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
        self.reference = refs[0] if refs else next(iter(self.systems))
        self.first = next(iter(self.systems))
        self.model = NICE.get(a.model, a.model)
        self.lead_hours = exp["run"]["lead_hours"]
        self.out = runs / "_figures" / a.experiment / "obs" / f"{a.model}_{sample}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.protocol = getattr(a, "protocol", "ecmwf")
        chosen = dict(PROTOCOLS[self.protocol])
        no_gross = getattr(a, "no_gross", False) and chosen["qc"] == "gross"
        if no_gross:
            chosen["qc"] = "none"
        self.match, self.qc, self.elevation = (
            chosen[k] for k in ("match", "qc", "elevation")
        )
        self.suffix = f"_{self.protocol}" + ("_nogross" if no_gross else "")
        self.sums = self.out / "sums"
        self.inits = self.out / "inits.json"

    @property
    def others(self):
        return [s for s in self.systems if s != self.reference]

    @property
    def pairs(self):
        """(subject, baseline) pairs beyond each IC against the reference."""
        if self.first == self.reference:
            return []
        return [(self.first, s) for s in self.others if s != self.first]

    def inits_info(self):
        return json.loads(self.inits.read_text())

    def inits_label(self):
        inits = self.inits_info()
        hours = "/".join(f"{h:02d}" for h in inits["hours"])
        return f"{inits['n']} inits at {hours} UTC, {inits['first'][:10]} to {inits['last'][:10]}"

    def systems_label(self):
        return ", ".join(label for label, _ in self.systems.values())

    def protocol_label(self):
        """The protocol's choices, one line each, for figure titles."""
        qc = "5-MAD consensus gross check" if self.qc == "gross" else "no gross check"
        window = "t - 30 min" if self.match == "strict" else "t - 90 min"
        match = f"{self.match} radiosonde match [{window}, t + 30 min)"
        tas = ELEVATION_VARIANTS[self.elevation][2]
        return f"Protocol {self.protocol}: {qc}; {match}.\n{tas}."

    def label(self, s):
        return self.systems[s][0]

    def valid_0012(self, lead):
        # Leads valid at 00/12 UTC; every init of a cycle group has the same hour mod 12.
        return (lead + self.inits_info()["hours"][0]) % 12 == 0


@functools.cache
def figure_regions():
    standard = load_regions()
    return {name: standard[key] for name, key in REGIONS.items()}


def aligned(tables, order_keys):
    """The systems' tables restricted to the reports every system holds, in one order.

    A report is matched on order_keys, field, time, position and observed value; repeats of
    a key pair up in order of appearance. Systems scored with different selections (e.g. a
    wider radiosonde window) are compared on the reports they share."""
    keys = [*order_keys, "code", "time", "lat", "lon", "obs"]
    joined = None
    for s, t in tables.items():
        f = pd.DataFrame({k: t[k] for k in keys})
        f["repeat"] = f.groupby(keys, sort=False).cumcount()
        f[s] = np.arange(len(f))
        joined = f if joined is None else joined.merge(f, on=[*keys, "repeat"])
    joined = joined.sort_values([*keys, "repeat"], kind="stable")
    out = {
        s: {k: v[joined[s].to_numpy()] for k, v in t.items()} for s, t in tables.items()
    }
    check_same_reports(out, ("obs", "lat", "lon", "code"), " after matching")
    return out


def field_codes(table, fields):
    """Index of each report's field in ``fields``, -1 for other fields."""
    codes = pc.index_in(table["field"], value_set=pa.array(fields))
    return codes.fill_null(-1).to_numpy()


def strict_rows(table):
    """Reports within the strict window of their valid time (``match`` strict)."""
    return within(table["time"].to_numpy(), table["cycle"].to_numpy())


def error_sums(errors, systems, fields, reports, source, matches):
    """Squared and plain error sums and report counts per field, lead and day, for every
    region, gross check and match in ``matches`` ("strict": the reports within the strict
    window, any other: every report). ``errors`` is (system, report); ``reports`` holds
    per-report code, lead, day, group (of the gross check), strict, lat and lon."""
    days, day_index = np.unique(reports["day"], return_inverse=True)
    lead = reports["lead"].astype(np.int64)
    key = (reports["code"] * (lead.max() + 1) + lead) * len(days) + day_index
    keys, index = np.unique(key, return_inverse=True)
    first = np.empty(len(keys), np.int64)
    first[index] = np.arange(len(index))
    labels = {
        "field": np.asarray(fields, dtype=object)[reports["code"][first]],
        "lead": lead[first],
        "day": days[day_index[first]].astype("datetime64[ns]"),
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
        checks = {"gross": gross, "none": matched}
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
    return table.assign(source=source)


def init_reports(files, fields, terrain):
    """One init's per-report errors (system, report) and report table: the reports every
    IC source has a model value for, with ``change`` (station elevation minus terrain)."""
    tables = {}
    for s, paths in files.items():
        t = pq.read_table(
            paths,
            columns=["init", "lead", "field", "obs", "model", "lat", "lon"]
            + ["time", "cycle", "elevation"],
        )
        tables[s] = {k: t[k].to_numpy() for k in ("init", "lead", "obs", "model")}
        tables[s]["elevation"] = t["elevation"].to_numpy(zero_copy_only=False)
        tables[s].update(
            lat=t["lat"].to_numpy(), lon=t["lon"].to_numpy(), time=t["time"].to_numpy()
        )
        tables[s]["code"] = field_codes(t, fields)
        tables[s]["strict"] = strict_rows(t)
    tables = aligned(tables, ["lead"])
    first = next(iter(tables.values()))
    keep = (first["code"] >= 0) & np.all(
        [np.isfinite(t["model"]) for t in tables.values()], axis=0
    )
    errors = np.stack([(t["model"] - t["obs"])[keep] for t in tables.values()])
    names = ("code", "lead", "strict", "lat", "lon", "elevation")
    reports = {k: first[k][keep] for k in names}
    reports["day"] = np.full(len(errors[0]), first["init"][0].astype("datetime64[D]"))
    reports["group"] = reports["lead"].astype(np.int64) * len(fields) + reports["code"]
    reports["change"] = station_height_change(
        terrain,
        era5_grid.LAT,
        era5_grid.LON,
        *(reports[k] for k in ("lat", "lon", "elevation")),
    )
    return first["init"][0], int(first["lead"].max()), errors, reports


def elevation_variants(errors, reports):
    """(variant, errors, reports) of every 2 m temperature variant: the reports within the
    window (stations without an elevation are outside every window), the model moved to
    the station elevation (a shift, so applied to the error)."""
    change = reports["change"]
    for variant, (window, corrected, _) in ELEVATION_VARIANTS.items():
        inside = (
            np.ones(len(change), bool) if window is None else np.abs(change) <= window
        )
        selected = {k: v[inside] for k, v in reports.items()}
        moved = errors[:, inside]
        if corrected:
            moved = temperature_at_height(moved, change[inside])
        yield variant, moved, selected


def init_sums(files, fields, terrain):
    """One init's error sums from each IC source's pairs files of that init."""
    init, max_lead, errors, reports = init_reports(files, fields, terrain)
    systems = list(files)
    sonde = np.isin(reports["code"], [fields.index(f) for f in UPPER])
    if not reports["strict"][~sonde].all():
        raise SystemExit(f"{init}: surface reports outside the strict window")
    corrected = reports["code"] == fields.index(ELEVATION_FIELD)
    surface = ~sonde & ~corrected

    def select(mask):
        return errors[:, mask], {k: v[mask] for k, v in reports.items()}

    tables = []
    for mask, matches in ((sonde, MATCHES), (surface, [SURFACE_MATCH])):
        if mask.any():
            e, r = select(mask)
            tables.append(error_sums(e, systems, fields, r, "forecast", matches))
    for variant, e, r in elevation_variants(*select(corrected)):
        if len(e[0]):
            table = error_sums(e, systems, fields, r, "forecast", [SURFACE_MATCH])
            tables.append(table.assign(elevation=variant))
    table = pd.concat(tables, ignore_index=True)
    return init, max_lead, table.fillna({"elevation": "none"})


def files_by_init(files):
    found = {}
    for f in files:
        found.setdefault(f.name.rsplit("_r", 1)[0], []).append(f)
    return found


def read_analysis_0012(e, analysis, fields):
    inits = e.inits_info()
    # The days of the forecasts' valid times.
    period = (
        pd.Timestamp(inits["first"]).floor("D"),
        (pd.Timestamp(inits["last"]) + pd.Timedelta(hours=inits["max_lead"])).floor("D")
        + pd.Timedelta(days=1),
    )
    tables = {}
    for s in e.systems:
        t = pq.read_table(
            analysis[s],
            columns=["cycle", "time", "field", "obs", "model", "lat", "lon"],
            filters=[
                ("field", "in", fields),
                ("cycle", ">=", period[0]),
                ("cycle", "<", period[1]),
            ],
        )
        t = t.filter(pc.is_in(pc.hour(t["cycle"]), value_set=pa.array([0, 12])))
        tables[s] = {k: t[k].to_numpy() for k in t.column_names if k != "field"}
        tables[s]["code"] = field_codes(t, fields)
        tables[s]["strict"] = strict_rows(t)
    tables = aligned(tables, ["cycle"])
    first = next(iter(tables.values()))
    keep = np.all([np.isfinite(t["model"]) for t in tables.values()], axis=0)
    errors = np.stack([(t["model"] - t["obs"])[keep] for t in tables.values()])
    reports = {k: first[k][keep] for k in ("code", "strict", "lat", "lon")}
    cycle = first["cycle"][keep]
    reports["lead"] = np.zeros(len(cycle), np.int64)
    reports["day"] = cycle.astype("datetime64[D]")
    reports["group"] = (
        np.unique(cycle, return_inverse=True)[1] * len(fields) + reports["code"]
    )
    table = error_sums(
        errors, list(e.systems), fields, reports, "analysis 00/12", MATCHES
    )
    return table.assign(elevation="none")


def analysis_files(e, items):
    found = {}
    for item in items or []:
        if "=" in item:
            name, path = item.split("=", 1)
            found[name] = Path(path)
        else:
            found.update(
                {s: Path(item) / f"{s}.parquet" for s in e.systems if s not in found}
            )
    found = {s: p for s, p in found.items() if s in e.systems and p.is_file()}
    missing = [s for s in e.systems if s not in found]
    if missing:
        print(f"no analysis pairs for {missing}: radiosonde curves start after lead 0")
        return None
    return found


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


def read_sums(e):
    paths = [e.sums / f"{name}.parquet" for name in SUMS]
    return pd.concat(
        [pd.read_parquet(p) for p in paths if p.exists()], ignore_index=True
    )


def mode_accumulate(e, a):
    analysis = analysis_files(e, a.analysis_pairs)
    e.sums.mkdir(exist_ok=True)
    by_init = {s: files_by_init(e.files[s]) for s in e.systems}
    common = sorted(set.intersection(*(set(f) for f in by_init.values())))
    keys = ["field", "lead", "day", "region", "source", "qc", "match"]
    keys += ["elevation"]
    terrain = np.asarray(era5_grid.era5_orography(), np.float64)
    parts, inits, max_lead = [], [], 0
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        jobs = [
            pool.submit(
                init_sums, {s: by_init[s][i] for s in e.systems}, FIELDS, terrain
            )
            for i in common
        ]
        for done, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            init, lead, part = job.result()
            inits.append(pd.Timestamp(init))
            max_lead = max(max_lead, lead)
            parts.append(part)
            if done % 32 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} inits", flush=True)
    write_inits(e, inits, max_lead)
    # Sums per init, added up over the inits of each day.
    table = pd.concat(parts, ignore_index=True).groupby(keys, sort=True).sum()
    table.reset_index().to_parquet(e.sums / "forecast.parquet")
    print(e.sums / "forecast.parquet", flush=True)
    if analysis:
        path = e.sums / "analysis_upper.parquet"
        read_analysis_0012(e, analysis, UPPER).to_parquet(path)
        print(path, flush=True)


def selected_sums(e):
    """The sums of the protocol's gross check, radiosonde match and 2 m temperature
    variant."""
    table = read_sums(e)
    if "elevation" not in table:
        raise SystemExit(f"{e.sums} predates the elevation column; rerun accumulate")
    corrected = table.field == ELEVATION_FIELD
    table = table[
        (table.qc == e.qc)
        & table.match.isin([e.match, SURFACE_MATCH])
        & (table.elevation == np.where(corrected, e.elevation, "none"))
    ]
    if not (table.field == ELEVATION_FIELD).any():
        raise SystemExit(f"no 2 m temperature sums for {e.elevation}; rerun accumulate")
    return table.drop(columns=["qc", "match", "elevation"])


def vector_components(name):
    if name == "w10m":
        return "uas", "vas"
    if name.startswith("W"):
        return f"U{name[1:]}", f"V{name[1:]}"
    return None


def combine(e, table, name):
    """Day sums of one plotted quantity: a field, or vector wind (W{p} from U{p} and
    V{p}, w10m from uas and vas)."""
    components = vector_components(name)
    if components:
        u, v = (
            table[table.field == c].set_index(["source", "region", "lead", "day"])
            for c in components
        )
        cols = [f"sq_{s}" for s in e.systems]
        both = u.index.intersection(v.index)
        out = u.loc[both, cols] + v.loc[both, cols]
        out["n"] = u.loc[both, "n"]
        for s in e.systems:
            out[f"e_{s}"] = np.nan
        return out.reset_index()
    return table[table.field == name]


def curve(e, days, rng):
    """RMSE, bias and % RMSE differences (with paired day-bootstrap intervals) per lead."""
    rows = []
    comparisons = [(s, e.reference, f"pct_{s}") for s in e.others] + [
        (s, b, f"pct_{s}_vs_{b}") for s, b in e.pairs
    ]
    for lead, g in days.groupby("lead"):
        n = g["n"].to_numpy(float)
        sq = {s: g[f"sq_{s}"].to_numpy() for s in e.systems}
        row = {"lead": lead, "n": int(n.sum()), "days": len(g)}
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


def curves_table(e):
    table = selected_sums(e)
    rng = np.random.default_rng(0)
    out = []
    for name, (_, _, kind) in QUANTITIES.items():
        q = combine(e, table, name)
        if q.empty:
            continue
        regions = (
            ["global", "Europe", "N America"] if kind == "surface" else list(REGIONS)
        )
        for region in regions:
            r = q[q.region == region]
            if kind == "upper":
                valid = (r.source == "forecast") & e.valid_0012(r.lead)
                r = r[valid | (r.source == "analysis 00/12")]
            out.append(curve(e, r, rng).assign(quantity=name, region=region))
    return pd.concat(out, ignore_index=True)


def plot_blocks(e, table, blocks, title, path):
    """Labelled blocks of panels; ``blocks`` is ``[(heading, [(column title, quantity,
    region)])]``. Each panel: RMSE with day-bootstrap bands over % vs the reference IC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = max(len(cells) for _, cells in blocks)
    fig = plt.figure(
        figsize=(4.2 * ncol, 4.6 * len(blocks) + 1.6), layout="constrained"
    )
    subfigs = fig.subfigures(len(blocks), 1, hspace=0.04)
    subfigs = [subfigs] if len(blocks) == 1 else list(subfigs)
    n_inits = e.inits_info()["n"]
    for i, ((heading, cells), block) in enumerate(zip(blocks, subfigs)):
        block.set_facecolor("0.97" if i % 2 else "white")
        block.suptitle(heading, fontsize=12, fontweight="bold", x=0.01, ha="left")
        axes = block.subplots(
            2, ncol, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        axes = axes.reshape(2, ncol)
        for j in range(len(cells), ncol):
            axes[0, j].set_visible(False)
            axes[1, j].set_visible(False)
        for j, (column, name, region) in enumerate(cells):
            _, unit, _ = QUANTITIES[name]
            part = table[
                (table.quantity == name) & (table.region == region)
            ].sort_values("lead")
            raw, rel = axes[0, j], axes[1, j]
            for s, (slabel, colour) in e.systems.items():
                raw.plot(part.lead, part[f"rmse_{s}"], ".-", color=colour, label=slabel)
                raw.fill_between(
                    part.lead,
                    part[f"rmse_{s}_lo"],
                    part[f"rmse_{s}_hi"],
                    color=colour,
                    alpha=0.15,
                    lw=0,
                )
                if s == e.reference:
                    rel.plot(part.lead, 0 * part.lead, "-", color=colour, lw=1)
                    continue
                rel.plot(part.lead, part[f"pct_{s}"], ".-", color=colour)
                rel.fill_between(
                    part.lead,
                    part[f"pct_{s}_lo"],
                    part[f"pct_{s}_hi"],
                    color=colour,
                    alpha=0.2,
                    lw=0,
                )
            for ax in (raw, rel):
                ax.axvspan(-2, 2, color="0.9", zorder=0)
                ax.grid(alpha=0.3)
            raw.set_title(column, fontsize=11)
            # Median over leads: the lead-0 analysis point pools every cycle of the period.
            per_time = part.n.median() / n_inits if len(part) else 0
            raw.text(
                0.98,
                0.04,
                f"~{per_time:,.0f} reports per valid time",
                transform=raw.transAxes,
                ha="right",
                fontsize=7,
                color="0.4",
            )
            rel.set_xticks(range(0, e.lead_hours + 1, 24))
            rel.set_xlabel("lead (h)" if i == len(blocks) - 1 else "")
            if j == 0:
                raw.set_ylabel(f"RMSE ({unit})")
                rel.set_ylabel(f"% vs {e.label(e.reference)} IC")
    handles, labels = subfigs[0].axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=len(e.systems), fontsize=10
    )
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


def plot_curves(e, table, names, regions, title, path):
    """One block per quantity, regions as columns."""
    blocks = [
        (QUANTITIES[name][0], [(region, name, region) for region in regions])
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
        ax.set_xlabel("lead (h)")
        ax.set_title(f"2 m temperature bias (model - station), {region}", fontsize=10)
    axes[0].set_ylabel("bias (K)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{e.model} forecasts from {e.systems_label()} vs surface stations; "
        f"{e.inits_label()}; every 6 h of lead.\n" + e.protocol_label(),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)
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
        f"{inits}; same reports for every system; shaded: lead 0; bands: day-bootstrap 95 %",
        e.out / f"summary_global{e.suffix}.png",
    )
    plot_curves(
        e,
        table,
        SURFACE_PLOTTED,
        ["global", "Europe", "N America"],
        f"{head}: surface stations.\n{inits}, one dot every 6 h of lead; same reports for "
        "every system at each lead.\nShaded: lead 0 (the ICs, stations assimilated). "
        "Bands: day-bootstrap 95 %",
        e.out / f"curves_surface{e.suffix}.png",
    )
    for v, label, levels in (
        ("T", "temperature", LEVELS),
        ("W", "vector wind", LEVELS),
        ("U", "zonal wind", LEVELS),
        ("V", "meridional wind", LEVELS),
        ("Z", "geopotential height", LEVELS),
        ("Q", "specific humidity", Q_LEVELS),
    ):
        plot_curves(
            e,
            table,
            [f"{v}{p}" for p in levels],
            list(REGIONS),
            f"{head}: radiosonde {label}.\n{inits}; {upper_leads}; "
            "shaded lead 0: the analyses at 00/12 UTC. "
            "Same reports for every system; bands: day-bootstrap 95 %",
            e.out / f"curves_radiosonde_{v}{e.suffix}.png",
        )
    plot_bias(e, table, e.out / f"bias_tas{e.suffix}.png")


def scorecard_metrics(e, regions):
    """One forecast metrics file per IC and region, in the paper plotters' layout:
    ``rmse_member(field, lead_time)`` and ``rmse_member_timeseries(field, time,
    lead_time)`` with one time per init day."""
    import xarray as xr

    table = selected_sums(e)
    table = table[table.source == "forecast"]
    vectors = [f"W{p}" for p in ALL_LEVELS] + ["w10m"]
    scalars = [f"{v}{p}" for v in "ZT" for p in ALL_LEVELS] + [
        f"Q{p}" for p in Q_ALL_LEVELS
    ]
    names = {**{f: f for f in scalars}, **SCORECARD_SURFACE}
    table = pd.concat(
        [table[table.field.isin(names)].assign(field=lambda t: t.field.map(names))]
        + [combine(e, table, name).assign(field=name) for name in vectors]
    )
    surface = table.field.isin([*SCORECARD_SURFACE.values(), "w10m"])
    table = table[surface | e.valid_0012(table.lead)]
    paths = {}
    for region in regions:
        part = table[table.region == region]
        out = e.out / "scorecard_metrics" / region.replace(" ", "_")
        out.mkdir(parents=True, exist_ok=True)
        for s in e.systems:
            day = part.groupby(["field", "day", "lead"])[[f"sq_{s}", "n"]].sum()
            whole = part.groupby(["field", "lead"])[[f"sq_{s}", "n"]].sum()
            series = np.sqrt(day[f"sq_{s}"] / day["n"]).to_xarray()
            pooled = np.sqrt(whole[f"sq_{s}"] / whole["n"]).to_xarray()
            # A deterministic forecast is one member: the plotter's rmse_member.
            ds = xr.Dataset({"rmse_member": pooled, "rmse_member_timeseries": series})
            ds = ds.rename({"day": "time", "lead": "lead_time"})
            ds = ds.assign_coords(
                lead_time=ds.lead_time.values.astype("timedelta64[h]").astype(
                    "timedelta64[ns]"
                )
            )
            ds["rmse_member_timeseries"] = ds.rmse_member_timeseries.transpose(
                "field", "time", "lead_time"
            )
            paths[region, s] = out / f"{s}{e.suffix}.nc"
            ds.to_netcdf(paths[region, s])
    return paths


def mode_scorecards(e, a):
    from data_utils import load_metrics
    from plot_forecast_scorecard import plot_scorecard

    regions = list(REGIONS) if a.regions == ["all"] else a.regions
    paths = scorecard_metrics(e, regions)
    out = e.out / "scorecards"
    out.mkdir(exist_ok=True)
    comparisons = [(s, e.reference) for s in e.others] + e.pairs
    for region in regions:
        for s, b in comparisons:
            path = out / f"{s}_vs_{b}_{region.replace(' ', '_')}{e.suffix}.png"
            plot_scorecard(
                load_metrics(paths[region, s]),
                load_metrics(paths[region, b]),
                "rmse_member",
                f"{e.label(s)} IC",
                f"{e.label(b)} IC",
                path,
                max_lead_time=e.lead_hours,
                vmax_percentile=a.vmax_percentile,
                skip_t0=True,
                verification_label=f"protocol {e.protocol}, {region}",
                significance_alpha=0.05,
                panel_titles={"W": "vector wind"},
            )
            print(path)


def mode_stations(e, a):
    from private.scripts.analysis.station_figures import (
        plot_heatwave,
        plot_maps,
        station_table,
    )

    out = e.out / "stations"
    out.mkdir(exist_ok=True)
    for lead in a.leads:
        table = station_table(e.files, filters=[("lead", "==", lead)])
        tag = f"{lead:03d}h"
        what = f"{e.model} +{lead} h forecasts from {e.systems_label()}"
        plot_maps(
            table,
            e.systems,
            e.first,
            out / f"station_rmse_maps_{tag}.png",
            f"{what} vs surface stations at valid time: same reports for every IC "
            "(stations with >= 100 reports mapped)",
            "valid hour (UTC)",
        )
        plot_heatwave(
            table,
            e.systems,
            out / f"europe_heatwave_{tag}.png",
            f"Europe: {what} vs surface stations at valid time (2 m temperature), "
            "same reports for every IC",
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("accumulate", "plot", "scorecards", "stations"):
        q = sub.add_parser(mode)
        q.add_argument("experiment")
        q.add_argument("--model", required=True, help="model key of the experiment")
        q.add_argument("--samples", help="cycle group; needed if there are several")
        if mode == "accumulate":
            q.add_argument("--analysis-pairs", nargs="+", metavar="DIR|NAME=FILE")
            q.add_argument(
                "--workers",
                type=int,
                default=min(32, os.cpu_count()),
                help="processes reading the pairs, one init each",
            )
        if mode in ("plot", "scorecards"):
            q.add_argument(
                "--protocol",
                choices=list(PROTOCOLS),
                default="ecmwf",
                help="radiosonde match, gross check and 2 m temperature variant "
                "(private/obs_scoring/METHOD.md section 6)",
            )
            q.add_argument(
                "--no-gross",
                action="store_true",
                help="the protocol without the consensus gross check",
            )
        if mode == "scorecards":
            q.add_argument("--vmax-percentile", type=float, default=95.0)
            q.add_argument(
                "--regions",
                nargs="+",
                choices=[*REGIONS, "all"],
                default=["global"],
                metavar="REGION",
                help=f"regions to draw, or all ({', '.join(REGIONS)})",
            )
        if mode == "stations":
            q.add_argument("--leads", nargs="+", type=int, default=[24, 120])
    a = p.parse_args()
    e = Experiment(a)
    {
        "accumulate": mode_accumulate,
        "plot": mode_plot,
        "scorecards": mode_scorecards,
        "stations": mode_stations,
    }[a.mode](e, a)


if __name__ == "__main__":
    main()
