# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model values at observation points.

``fields`` is an ``xr.Dataset`` for one time with ``lat`` and ``lon`` coordinates on a regular
grid and variables named as the keys of ``fields.E2S_NAMES`` (``Z500``, ``T850``, ``tas``, ``d2m``),
``Z{p}`` as geopotential (m2 s-2). Any source that can produce that can be scored; a HEALPix
store, for example, after regridding to lat-lon.

Compared with NOAA PSL eagle-tools ``obs-metrics``:

- Horizontal: bilinear in latitude and longitude on the regular grid, periodic in longitude.
  eagle-tools regrids with xesmf (ESMF bilinear on the sphere) and its regridder is not
  periodic, so points between the last grid longitude and 360 are NaN there. On a 0.25 deg
  grid the largest differences found were 0.003 m (Z500), 5e-4 K (T850), 7e-4 m/s (U500)
  and 2.9 Pa (sp); poleward of 80 deg, where spherical and lat-lon weights differ, 2 m
  temperature differed by up to 0.48 K.
- Vertical, as in eagle-tools: a radiosonde value at a mandatory pressure level is compared
  with the model field on that level (observed T at 500 hPa with ``T500``). Every level is
  placed at the station position and report time: the NNJA-AI radiosonde table, like the
  text-format dump it comes from, has no per-level drift, though balloons drift tens to
  hundreds of km (about 25 km median, 200 km at the 99th percentile) and up to hours.
  Surface observations are compared with the model's 2 m, 10 m and surface fields at the
  station position, with no correction for the difference between
  station and model terrain height. Profile winds (VAD, aircraft; not in eagle-tools) are
  interpolated between model levels, linearly in log p or in height.
- Model ``q2m`` comes from ``d2m`` and ``sp`` with the Magnus formula, model ``Z`` is
  divided by 9.80665, and model wind speed (``S{p}``, ``sfcwind``) is the magnitude of the
  interpolated u and v, as in eagle-tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .thermo import (
    GRAVITY,
    pressure_at_height,
    specific_humidity_from_dewpoint,
    temperature_at_height,
)

# Model levels for profile (VAD, aircraft) interpolation.
PROFILE_LEVELS = np.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], np.float64
)


def bilinear(field: np.ndarray, lat_grid, lon_grid, lat, lon) -> np.ndarray:
    """Bilinear in (lat, lon) on a regular grid, periodic in lon; NaN outside the lat range."""
    lat_grid = np.asarray(lat_grid, np.float64)
    lon_grid = np.asarray(lon_grid, np.float64)
    nlat, nlon = field.shape
    y = (np.asarray(lat, np.float64) - lat_grid[0]) / (lat_grid[1] - lat_grid[0])
    x = np.mod(np.asarray(lon, np.float64) - lon_grid[0], 360.0) / (
        lon_grid[1] - lon_grid[0]
    )
    inside = (y >= 0) & (y <= nlat - 1)
    i0 = np.clip(np.floor(y).astype(np.int64), 0, nlat - 2)
    wy = np.clip(y - i0, 0.0, 1.0)
    j0 = np.floor(x).astype(np.int64) % nlon
    j1 = (j0 + 1) % nlon
    wx = x - np.floor(x)
    value = (
        (1 - wy) * (1 - wx) * field[i0, j0]
        + (1 - wy) * wx * field[i0, j1]
        + wy * (1 - wx) * field[i0 + 1, j0]
        + wy * wx * field[i0 + 1, j1]
    )
    return np.where(inside, value, np.nan)


def model_field(fields: xr.Dataset, field: str) -> np.ndarray | None:
    if field == "q2m":
        if "d2m" not in fields or "sp" not in fields:
            return None
        dewpoint = np.asarray(fields["d2m"].values, np.float64)
        pressure = np.asarray(fields["sp"].values, np.float64)
        return specific_humidity_from_dewpoint(dewpoint, pressure / 100.0)
    if field not in fields:
        return None
    values = np.asarray(fields[field].values, np.float64)
    if field.startswith("Z") and field[1:].isdigit():
        values = values / GRAVITY
    return values


def wind_speed_components(field: str) -> tuple[str, str] | None:
    if field == "sfcwind":
        return "uas", "vas"
    if field[0] == "S" and field[1:].isdigit():
        return f"U{field[1:]}", f"V{field[1:]}"
    return None


def _grid(fields: xr.Dataset):
    return fields["lat"].values, fields["lon"].values


def log_pressure_interpolate(
    fields: xr.Dataset, variable: str, pressure, lat, lon
) -> np.ndarray:
    levels = PROFILE_LEVELS
    upper = np.clip(np.searchsorted(levels, pressure), 1, len(levels) - 1)
    lower = upper - 1
    weight = (np.log(pressure) - np.log(levels[lower])) / (
        np.log(levels[upper]) - np.log(levels[lower])
    )
    values = np.zeros(len(pressure))
    for k in np.union1d(lower, upper):
        rows = (lower == k) | (upper == k)
        field = model_field(fields, f"{variable}{int(levels[k])}")
        if field is None:
            return np.full(len(pressure), np.nan)
        at = bilinear(field, *_grid(fields), lat[rows], lon[rows])
        share = np.where(lower[rows] == k, 1 - weight[rows], weight[rows])
        values[rows] += share * at
    inside = (pressure >= levels[0]) & (pressure <= levels[-1])
    return np.where(inside, values, np.nan)


def height_interpolate(
    fields: xr.Dataset, variable: str, height, lat, lon
) -> np.ndarray:
    """Linear in height between the model levels whose geopotential heights bracket
    ``height`` (m); NaN below the lowest or above the highest level."""
    heights, values = [], []
    for level in PROFILE_LEVELS:
        z = model_field(fields, f"Z{int(level)}")
        wind = model_field(fields, f"{variable}{int(level)}")
        if z is None or wind is None:
            return np.full(len(height), np.nan)
        heights.append(bilinear(z, *_grid(fields), lat, lon))
        values.append(bilinear(wind, *_grid(fields), lat, lon))
    heights, values = np.stack(heights), np.stack(values)
    above = (heights > height).sum(axis=0) - 1
    inside = (above >= 0) & (above < len(PROFILE_LEVELS) - 1)
    k = np.clip(above, 0, len(PROFILE_LEVELS) - 2)
    columns = np.arange(len(height))
    z_top, z_bottom = heights[k, columns], heights[k + 1, columns]
    weight = (z_top - height) / (z_top - z_bottom)
    value = (1 - weight) * values[k, columns] + weight * values[k + 1, columns]
    return np.where(inside, value, np.nan)


def required_fields(
    obs: pd.DataFrame, vertical: str = "pressure", elevation_correct: bool = False
) -> list[str]:
    """The variables ``sample`` reads for these observations."""
    if "pressure" in obs:
        names = {
            f"{v}{int(p)}" for v in obs["field"].str[0].unique() for p in PROFILE_LEVELS
        }
        if vertical == "height":
            names |= {f"Z{int(p)}" for p in PROFILE_LEVELS}
        return sorted(names)
    names = set()
    for field in obs["field"].unique():
        if field == "q2m":
            names |= {"d2m", "sp"}
        else:
            names |= set(wind_speed_components(field) or (field,))
    if elevation_correct and names & {"tas", "sp"}:
        names |= {"orography", "tas"}
    if elevation_correct and "sp" in names:
        names |= {"d2m"}
    return sorted(names)


def station_height_change(
    terrain: np.ndarray, lat_grid, lon_grid, lat, lon, elevation
) -> np.ndarray:
    """Station elevation minus the terrain (on a regular grid) at each station (m); NaN
    where either is unknown."""
    return np.asarray(elevation, np.float64) - bilinear(
        terrain, lat_grid, lon_grid, lat, lon
    )


def _station_height_change(fields: xr.Dataset, obs: pd.DataFrame, rows) -> np.ndarray:
    """Station elevation minus model terrain at each row; 0 where either is unknown."""
    change = station_height_change(
        np.asarray(fields["orography"].values, np.float64),
        *_grid(fields),
        obs["lat"].to_numpy()[rows],
        obs["lon"].to_numpy()[rows],
        obs["elevation"].to_numpy(np.float64)[rows],
    )
    return np.where(np.isfinite(change), change, 0.0)


def sample(
    fields: xr.Dataset | None,
    obs: pd.DataFrame,
    vertical: str = "pressure",
    elevation_correct: bool = False,
) -> np.ndarray:
    """Model value at each observation row; NaN where ``fields`` lacks the variable.

    Rows with ``pressure`` (profile winds) interpolate between model levels, linearly in
    log p at ``pressure`` or, with ``vertical="height"``, linearly in height at ``height``
    using the model's geopotential. Other rows are bilinear in latitude and longitude.
    With ``elevation_correct``, model ``tas`` and ``sp`` are moved from the model terrain
    (``fields["orography"]``) to the station elevation (``obs["elevation"]``): ``tas`` with
    the standard lapse rate, ``sp`` hydrostatically at the virtual temperature of the
    model's 2 m humidity (``d2m``).
    """
    model = np.full(len(obs), np.nan)
    if fields is None:
        return model
    lat, lon = obs["lat"].to_numpy(), obs["lon"].to_numpy()
    if "pressure" in obs:
        coordinate = obs[vertical].to_numpy(np.float64)
        profile = (
            height_interpolate if vertical == "height" else log_pressure_interpolate
        )
        for variable, rows in obs.groupby(obs["field"].str[0]).indices.items():
            model[rows] = profile(
                fields, variable, coordinate[rows], lat[rows], lon[rows]
            )
        return model
    for field, rows in obs.groupby("field").indices.items():
        components = wind_speed_components(field)
        if components is not None:
            u, v = (model_field(fields, name) for name in components)
            if u is not None and v is not None:
                at = (bilinear(c, *_grid(fields), lat[rows], lon[rows]) for c in (u, v))
                model[rows] = np.hypot(*at)
            continue
        values = model_field(fields, field)
        if values is None:
            continue
        model[rows] = bilinear(values, *_grid(fields), lat[rows], lon[rows])
        correct = elevation_correct and field in ("tas", "sp") and "elevation" in obs
        if correct and "orography" in fields and "tas" in fields:
            change = _station_height_change(fields, obs, rows)
            tas = bilinear(
                model_field(fields, "tas"), *_grid(fields), lat[rows], lon[rows]
            )
            if field == "tas":
                model[rows] = temperature_at_height(tas, change)
            else:
                humidity = 0.0
                if "d2m" in fields:
                    dewpoint = bilinear(
                        model_field(fields, "d2m"), *_grid(fields), lat[rows], lon[rows]
                    )
                    humidity = specific_humidity_from_dewpoint(
                        dewpoint, model[rows] / 100.0
                    )
                model[rows] = pressure_at_height(model[rows], tas, change, humidity)
    return model
