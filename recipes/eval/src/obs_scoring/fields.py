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
import os
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from .thermo import GRAVITY

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


# ERA5 surface geopotential (m2/s2) on the 0.25 degree grid, on the public NSF NCAR mirror.
ERA5_GEOPOTENTIAL_URI = (
    "s3://nsf-ncar-era5/e5.oper.invariant/197901/"
    "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
)


@functools.cache
def era5_orography() -> np.ndarray:
    """ERA5 terrain height (m) on the 0.25 degree grid, lat 90 to -90, downloaded once into
    the earth2studio cache (``EARTH2STUDIO_CACHE``)."""
    import fsspec

    from earth2studio.data.utils import datasource_cache_root

    local = (
        pathlib.Path(datasource_cache_root())
        / "obs_scoring"
        / pathlib.PurePosixPath(ERA5_GEOPOTENTIAL_URI).name
    )
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(f"{local.name}.tmp.{os.getpid()}")
        fsspec.filesystem("s3", anon=True).get(ERA5_GEOPOTENTIAL_URI, str(partial))
        os.replace(partial, local)
    with xr.open_dataset(local) as ds:
        geopotential = ds["Z"].squeeze().values.astype(np.float32)
    return np.ascontiguousarray(geopotential) / np.float32(GRAVITY)


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
    Times whose ``run_id`` is negative and fields whose ``is_valid`` is false are absent.
    """

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


def open_fields(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "zarr":
        return ZarrFields(rest)
    raise ValueError(f"unknown system kind {kind!r} in {spec!r}")
