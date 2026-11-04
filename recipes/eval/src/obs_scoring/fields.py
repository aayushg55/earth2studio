# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Field sources for scoring: each turns a store into ``sample``'s input for one time.

``open_fields("zarr:PATH").at(time, names)`` returns an ``xr.Dataset`` with ``lat``/``lon``
coordinates and the requested variables the store has at that time, or None when the store
has no such time. Readers of other stores (ERA5 HDF5 packs, WeatherBench2 zarrs) live with
the caller and return the same ``xr.Dataset``.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from .thermo import GRAVITY

LAT = np.linspace(90, -90, 721)
LON = np.arange(1440) * 0.25


@functools.cache
def era5_orography(path: str) -> np.ndarray:
    """ERA5 terrain height (m) on the 0.25 degree grid, lat 90 to -90, from the ERA5
    invariant surface geopotential file at ``path``
    (``e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc``, variable ``Z``).
    """
    if not pathlib.Path(path).is_file():
        raise FileNotFoundError(
            f"ERA5 orography file {path} does not exist; it is the ERA5 invariant surface "
            "geopotential (e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc)"
        )
    with xr.open_dataset(path) as ds:
        geopotential = ds["Z"].squeeze().values.astype(np.float32)
    return np.ascontiguousarray(geopotential) / np.float32(GRAVITY)


def era5_orography_on(lat: np.ndarray, lon: np.ndarray, path: str) -> np.ndarray:
    """ERA5 terrain height (m) of the file at ``path`` interpolated to a regular grid of
    ``lat`` and ``lon``."""
    terrain = xr.DataArray(
        era5_orography(path), coords={"lat": LAT, "lon": LON}, dims=("lat", "lon")
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


def _dataset(
    values: dict[str, np.ndarray], lat, lon, names, orography: str | None
) -> xr.Dataset:
    if "orography" in names:
        if orography is None:
            raise ValueError("the elevation correction needs an ERA5 orography file")
        values = {**values, "orography": era5_orography(orography)}
    return xr.Dataset(
        {name: (("lat", "lon"), array) for name, array in values.items()},
        coords={"lat": np.asarray(lat), "lon": np.asarray(lon)},
    )


class ZarrFields:
    """A lat-lon zarr store with one (time, lat, lon) array per field, named as in HealDA.
    Times whose ``run_id`` is negative and fields whose ``is_valid`` is false are absent.
    ``orography``: the ERA5 orography file (``era5_orography``) for the elevation correction.
    """

    def __init__(self, path: str, orography: str | None = None):
        import zarr

        self.orography = orography
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
        return _dataset(values, self.lat, self.lon, names, self.orography)


def open_fields(spec: str, orography: str | None = None):
    kind, _, rest = spec.partition(":")
    if kind == "zarr":
        return ZarrFields(rest, orography)
    raise ValueError(f"unknown system kind {kind!r} in {spec!r}")
