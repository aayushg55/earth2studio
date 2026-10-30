# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scores of model values against observations, per region and field, and of one cycle:
the observations valid at a time scored against the fields for that time.

``cli score`` reads the fields from a store; a caller that holds them in memory (a
forecast at each lead step, for example) passes its own ``fields_at``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import observations
from .sample import required_fields, sample

STANDARD_REGIONS = Path(__file__).parents[2] / "scorecard/cfg/regions/standard.yaml"

UNITS = {
    "Z": "m",
    "T": "K",
    "U": "m/s",
    "V": "m/s",
    "S": "m/s",
    "Q": "kg/kg",
    "tas": "K",
    "uas": "m/s",
    "vas": "m/s",
    "sfcwind": "m/s",
    "sp": "Pa",
    "q2m": "kg/kg",
}


def field_units(field: str) -> str:
    return UNITS.get(field, UNITS.get(field[:1], ""))


@functools.cache
def standard_regions() -> dict:
    """The forecast scorecard's standard regions plus ``global``."""
    import yaml

    from ..regions import parse_regions

    return parse_regions(
        {"global": None, **yaml.safe_load(STANDARD_REGIONS.read_text())}
    )


def point_regions(regions: dict, lat, lon) -> tuple[list[str], np.ndarray]:
    """``(names, (R, n) bool)`` for points, with the recipe's box rule: bounds inclusive,
    longitude on [0, 360) wrapping where min > max, a 360 degree span is every longitude.
    A point without a finite position is in no region."""
    lat, lon = np.asarray(lat, np.float64), np.mod(np.asarray(lon, np.float64), 360.0)
    located = np.isfinite(lat) & np.isfinite(lon)

    def in_box(box):
        inside = located.copy()
        if "lat" in box:
            inside &= (lat >= box["lat"][0]) & (lat <= box["lat"][1])
        if "lon" in box and box["lon"][1] - box["lon"][0] < 360.0:
            lo, hi = (b % 360.0 for b in box["lon"])
            inside &= (
                (lon >= lo) & (lon <= hi) if lo <= hi else (lon >= lo) | (lon <= hi)
            )
        return inside

    rows = [
        located if boxes is None else np.any([in_box(b) for b in boxes], axis=0)
        for boxes in regions.values()
    ]
    return list(regions), np.stack(rows)


def score(
    obs: pd.DataFrame,
    model: np.ndarray,
    regions: dict | None = None,
    by: tuple[str, ...] = ("field",),
) -> pd.DataFrame:
    """RMSE, MAE, bias and count of ``model - obs`` per region and per ``by`` group.

    ``regions`` is a parsed region table (``standard_regions``); observations are
    assigned by their own position. Without it everything is ``global``. ``by`` must end
    with ``"field"``; ``("dataset", "field")`` splits by observation dataset.
    """
    error = model - obs["obs"].to_numpy()
    valid = np.isfinite(error)
    names, in_regions = point_regions(
        regions or {"global": None}, obs["lat"], obs["lon"]
    )
    rows = []
    for region, in_box in zip(names, in_regions):
        in_region = valid & in_box
        for key, index in obs.groupby(list(by)).indices.items():
            e = error[index][in_region[index]]
            if not e.size:
                continue
            group = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            rows.append(
                {
                    "region": region,
                    **group,
                    "units": field_units(group["field"]),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "mae": float(np.mean(np.abs(e))),
                    "bias": float(np.mean(e)),
                    "count": int(e.size),
                }
            )
    return pd.DataFrame(rows)


FieldsAt = Callable[[list[str]], xr.Dataset | None]


def cycle_observations(
    source: tuple[str, object],
    cycle: pd.Timestamp,
    stations: str = "all",
    dedupe_surface: bool = False,
) -> pd.DataFrame | None:
    """Observations of ``source`` (``("conventional", MIRROR)`` or ``(OBS_SET,
    ARCHIVE)``) in the window around ``cycle``, after the station selection; None when
    the source has no file for the cycle."""
    kind, location = source
    if kind == "conventional":
        obs = observations.conventional_obs(location, cycle)
    else:
        obs = observations.prepbufr_obs(location, cycle, kind)
    if obs is None:
        return None
    obs = obs[observations.station_mask(obs, stations)].reset_index(drop=True)
    if dedupe_surface:
        obs = observations.dedupe_surface(obs)
    return obs


def score_cycle(
    fields_at: FieldsAt,
    obs: pd.DataFrame,
    cycle: pd.Timestamp,
    regions: dict | None,
    by: tuple[str, ...] = ("field",),
    vertical: str = "pressure",
    elevation_correct: bool = False,
    pairs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Per-region scores of the fields ``fields_at(names)`` returns against ``obs``, with
    ``time`` and ``match`` columns; and, with ``pairs``, ``obs`` with ``cycle`` and the
    ``model`` value."""
    values = fields_at(required_fields(obs, vertical, elevation_correct))
    model = sample(values, obs, vertical, elevation_correct)
    paired = obs.assign(cycle=cycle, model=model) if pairs else None
    table = scores_by_match(obs, model, cycle, regions, by)
    table.insert(0, "time", cycle)
    return table, paired


def scores_by_match(obs, model, cycle, regions, by=("field",)) -> pd.DataFrame:
    """``score`` of each match (``observations.MATCHES``), with a ``match`` column."""
    parts = []
    for match in observations.MATCHES:
        rows = observations.match_rows(obs, cycle, match)
        part = score(obs[rows].reset_index(drop=True), model[rows], regions, by)
        parts.append(part.assign(match=match))
    table = pd.concat(parts, ignore_index=True)
    return table[["match", *table.columns.drop("match")]]
