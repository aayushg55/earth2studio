# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Field sources for scoring: each turns a store into ``sample``'s input for one time.

``open_fields("zarr:PATH" | "era5" | "wb2:PATH").at(time, names)`` returns an
``xr.Dataset`` with ``lat``/``lon`` coordinates and the requested variables the store has at
that time, or None when the store has no such time.
"""

from __future__ import annotations

import functools
import numpy as np
import pandas as pd
import xarray as xr

LAT = np.linspace(90, -90, 721)
LON = np.arange(1440) * 0.25
# The variables the ERA5 source reads per time.
ERA5_FIELDS = (
    [
        f"{v}{p}"
        for v in "ZTUV"
        for p in (10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)
    ]
    + [f"Q{p}" for p in (50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)]
    + ["Z600", "U600", "V600", "tas", "uas", "vas", "sp", "d2m"]
)


@functools.cache
def era5_orography() -> np.ndarray:
    """ERA5 terrain height (m) on the 0.25 degree grid, lat 90 to -90."""
    from healda.datasets import era5_statics
    from healda.utils.cache import AtomicFileCache
    from healda.utils.thermo import GRAVITY

    _, fname, var = era5_statics._ERA5_NC[0]
    cache = AtomicFileCache(era5_statics.config.CACHE_DIR)
    path = cache.ensure(f"{era5_statics.ERA5_INVARIANTS_URI}/{fname}", {"anon": True})
    return era5_statics._load_nc(path, var) / np.float32(GRAVITY)


def era5_orography_on(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """ERA5 terrain height (m) interpolated to a regular grid of ``lat`` and ``lon``."""
    terrain = xr.DataArray(
        era5_orography(), coords={"lat": LAT, "lon": LON}, dims=("lat", "lon")
    )
    wrapped = terrain.pad(lon=(0, 1), mode="wrap").assign_coords(
        lon=np.arange(1441) * 0.25
    )
    return wrapped.interp(lat=np.asarray(lat), lon=np.asarray(lon) % 360).values


def store_times(group) -> pd.DatetimeIndex:
    """A store's time axis: int64 epoch seconds (our writer) or datetime64 (the IFS store)."""
    raw = group["time"][:]
    if np.issubdtype(raw.dtype, np.datetime64):
        return pd.DatetimeIndex(raw)
    return pd.to_datetime(raw, unit="s")


def _dataset(values: dict[str, np.ndarray], lat, lon, names) -> xr.Dataset:
    if "orography" in names:
        values = {**values, "orography": era5_orography()}
    return xr.Dataset(
        {name: (("lat", "lon"), array) for name, array in values.items()},
        coords={"lat": np.asarray(lat), "lon": np.asarray(lon)},
    )


class ZarrFields:
    """A lat-lon zarr store with one (time, lat, lon) array per field, named as in HealDA.
    Times whose ``run_id`` is negative and fields whose ``is_valid`` is false are absent."""

    def __init__(self, path: str):
        import zarr

        self.group = zarr.open_group(path, mode="r")
        self.names = set(self.group.array_keys())
        written = (
            self.group["run_id"][:] >= 0
            if "run_id" in self.names
            else np.ones(self.group["time"].shape[0], dtype=bool)
        )
        self.index = {
            t: i for i, (t, w) in enumerate(zip(store_times(self.group), written)) if w
        }
        self.variables = (
            [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in self.group["variable"][:]
            ]
            if "is_valid" in self.names
            else None
        )
        self.lat, self.lon = self.group["lat"][:], self.group["lon"][:]

    def _valid(self, name: str, i: int) -> bool:
        if name not in self.names:
            return False
        if self.variables is None:
            return True
        return name in self.variables and bool(
            self.group["is_valid"][i, self.variables.index(name)]
        )

    def at(self, time: pd.Timestamp, names: list[str]) -> xr.Dataset | None:
        if time not in self.index:
            return None
        i = self.index[time]
        values = {n: self.group[n][i] for n in names if self._valid(n, i)}
        return _dataset(values, self.lat, self.lon, names)


class Era5Fields:
    """ERA5 hourly packs."""

    def __init__(self):
        from healda.datasets.da.hourly_latlon_dataset import (
            HOURLY_PACK_NAME,
            hourly_pack_roots,
        )
        from healda.datasets.era5_hourly_h5 import Era5HourlyH5

        groups = {}
        for name in ERA5_FIELDS:
            pack = HOURLY_PACK_NAME[name]
            groups.setdefault(tuple(hourly_pack_roots([pack])), {})[name] = pack
        self.readers = [
            (list(names), Era5HourlyH5(list(roots), list(names.values())))
            for roots, names in groups.items()
        ]

    def at(self, time: pd.Timestamp, names: list[str]) -> xr.Dataset | None:
        values = {}
        for pack_names, reader in self.readers:
            try:
                block = reader.read(pd.DatetimeIndex([time]))[0]
            except (KeyError, IndexError, ValueError):
                continue
            values.update(zip(pack_names, block))
        if not values:
            return None
        return _dataset({n: values[n] for n in names if n in values}, LAT, LON, names)


class Wb2Fields:
    """A WeatherBench2-layout zarr (level dimension, CF names), via ``wb2_store``."""

    def __init__(self, path: str):
        from private.scripts.analysis.wb2_store import WB2Store

        self.store = WB2Store(path, profile=None)
        self.index = {t: i for i, t in enumerate(self.store.times)}

    def at(self, time: pd.Timestamp, names: list[str]) -> xr.Dataset | None:
        if time not in self.index:
            return None
        i = self.index[time]
        return _dataset(
            {n: self.store[n][i] for n in names if n in self.store}, LAT, LON, names
        )


def open_fields(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "era5":
        return Era5Fields()
    if kind == "wb2":
        return Wb2Fields(rest)
    if kind == "zarr":
        return ZarrFields(rest)
    raise ValueError(f"unknown system kind {kind!r} in {spec!r}")
