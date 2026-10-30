# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Field sources for scoring: each turns a store into ``sample``'s input for one time.

``open_fields("zarr:PATH" | "h5:ROOT[:ROOT...]" | "arco").at(time, names)`` returns an
``xr.Dataset`` with ``lat``/``lon`` coordinates and the requested variables the store has at
that time, or None when the store has no such time.
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from .thermo import GRAVITY

LAT = np.linspace(90, -90, 721)
LON = np.arange(1440) * 0.25
PRESSURE_LEVELS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
# Scorer field name to earth2studio variable name.
E2S_NAMES = {
    **{f"{v}{p}": f"{v.lower()}{p}" for v in "ZTUVQ" for p in PRESSURE_LEVELS},
    "tas": "t2m",
    "uas": "u10m",
    "vas": "v10m",
    "sp": "sp",
    "d2m": "d2m",
}


@functools.cache
def era5_orography() -> np.ndarray:
    """ERA5 terrain height (m) on the 0.25 degree grid, lat 90 to -90, from ARCO ERA5."""
    from earth2studio.data import ARCO_ERA5

    z = ARCO_ERA5(cache=True, verbose=False)(
        time=[pd.Timestamp("2020-01-01")], variable=["z"]
    )
    return (np.asarray(z.values[0, 0]) / GRAVITY).astype(np.float32)


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
    """A lat-lon zarr store with one (time, lat, lon) array per field, named as the keys of
    ``E2S_NAMES`` (``Z500``, ``tas``). Times whose ``run_id`` is negative and fields whose ``is_valid`` is false are absent."""

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


class H5Fields:
    """ERA5 HDF5 packs: per pack directory a ``YYYY.h5`` per year with a ``(time, channel,
    lat, lon)`` dataset and a ``metadata/data.json`` sidecar naming its path (``h5_path``),
    channels and time step (``dhours``). A field is read from the first pack, in the order
    given, that has its channel at that time."""

    def __init__(self, roots: list[str]):
        self.packs = []
        for root in roots:
            with open(os.path.join(root, "metadata", "data.json")) as f:
                meta = json.load(f)
            self.packs.append((root, meta, meta["coords"]["channel"]))
        meta = self.packs[0][1]
        self.lat, self.lon = meta["coords"]["lat"], meta["coords"]["lon"]

    @functools.cache
    def _file(self, path: str):
        import h5py

        return h5py.File(path, "r")

    def _read(self, time: pd.Timestamp, channel: str) -> np.ndarray | None:
        for root, meta, channels in self.packs:
            path = os.path.join(root, f"{time.year}.h5")
            if channel not in channels or not os.path.exists(path):
                continue
            offset = time - pd.Timestamp(year=time.year, month=1, day=1)
            step, rest = divmod(offset, pd.Timedelta(hours=meta["dhours"]))
            data = self._file(path)[meta["h5_path"]]
            if rest == pd.Timedelta(0) and step < data.shape[0]:
                return data[step, channels.index(channel)]
        return None

    def at(self, time: pd.Timestamp, names: list[str]) -> xr.Dataset | None:
        values = {}
        for name in names:
            if name in E2S_NAMES:
                value = self._read(time, E2S_NAMES[name])
                if value is not None:
                    values[name] = value
        if not values:
            return None
        return _dataset(values, self.lat, self.lon, names)


class DataSourceFields:
    """Any earth2studio data source on the 0.25 degree lat-lon grid (ARCO ERA5 by default);
    one request per time."""

    def __init__(self, source=None):
        if source is None:
            from earth2studio.data import ARCO_ERA5

            source = ARCO_ERA5(cache=False, verbose=False)
        self.source = source

    def at(self, time: pd.Timestamp, names: list[str]) -> xr.Dataset | None:
        wanted = [n for n in names if n in E2S_NAMES]
        data = self.source(time=[time], variable=[E2S_NAMES[n] for n in wanted])
        values = {n: np.asarray(data.values[0, i]) for i, n in enumerate(wanted)}
        return _dataset(values, data["lat"].values, data["lon"].values, names)


def open_fields(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "arco":
        return DataSourceFields()
    if kind == "h5":
        return H5Fields(rest.split(":"))
    if kind == "zarr":
        return ZarrFields(rest)
    raise ValueError(f"unknown system kind {kind!r} in {spec!r}")
