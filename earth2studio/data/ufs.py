# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import h5netcdf
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    datasource_cache_root,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_data_inputs,
)
from earth2studio.lexicon import GSIConventionalLexicon, GSISatelliteLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray


@dataclass
class _GSIAsyncTask:
    """Small helper struct for Async tasks"""

    datetime_file: datetime
    datetime_max: datetime
    datetime_min: datetime
    gsi_obs_key: str
    # None on the copies handed to decode workers; modifiers are closures and are
    # applied in the parent process
    gsi_modifier: Callable | None
    gsi_obs_name: str
    e2s_obs_name: str
    satellite: str | None = None


_NS_PER_HOUR = np.int64(3_600_000_000_000)

CYCLE_HOURS = 6

# Below this many diag files a process pool costs more than it saves
_MIN_PARALLEL_FILES = 4


def _decode_gsi_group(
    source_cls: type[_UFSObsBase],
    schema: pa.Schema,
    column_map: dict[str, str],
    channel_indexed_fields: dict[str, str],
    local_path: str,
    obs_names: set[str],
    tasks: list[_GSIAsyncTask],
) -> list[pd.DataFrame]:
    """Decode one diag file into a DataFrame per task.

    Tasks in a group share every column but the one supplying ``observation``, so
    the file is read once and the obs columns fanned out. Runs in a worker process,
    so it takes the source class rather than an instance and the tasks must arrive
    without their modifiers.
    """
    ds = h5netcdf.File(local_path, "r")
    try:
        data: dict[str, pa.Array] = {}
        channel_index_raw: np.ndarray | None = None
        for name, dset in ds.variables.items():
            if name not in column_map and name not in obs_names:
                continue
            # Skip channel-indexed fields; they are expanded below
            if name in channel_indexed_fields:
                continue
            values = np.asarray(dset[:])
            pa_type = source_cls.SCHEMA.field(column_map.get(name, "observation")).type
            # Convert char arrays into strings for DF
            if values.dtype.kind == "S" and values.ndim == 2:
                values = values.view(f"S{values.shape[1]}").ravel()
                values = np.char.rstrip(np.char.decode(values, "utf-8"), "\x00")
            # Apply subclass-specific transformations
            values = source_cls._transform_column(name, values, tasks[0], ds)
            data[name] = pa.array(values, type=pa_type)
            # Stash raw Channel_Index for per-channel expansion
            if name == "Channel_Index":
                channel_index_raw = np.asarray(dset[:])

        # Expand channel-indexed fields using Channel_Index as lookup
        if channel_index_raw is not None:
            idx: np.ndarray = channel_index_raw.astype(np.int32) - 1
            for gsi_name, field_name in channel_indexed_fields.items():
                if gsi_name in ds.variables:
                    lut = np.asarray(
                        ds[gsi_name][:],
                        dtype=schema.field(field_name).type.to_pandas_dtype(),
                    )
                    data[gsi_name] = pa.array(
                        lut[idx], type=schema.field(field_name).type
                    )
    finally:
        ds.close()

    frames: list[pd.DataFrame] = []
    for task in tasks:
        df = pd.DataFrame(
            {
                name: values
                for name, values in data.items()
                if name in column_map or name == task.gsi_obs_name
            }
        )
        df.rename(
            columns={**column_map, task.gsi_obs_name: "observation"},
            inplace=True,
        )
        # Add e2s columns
        df["variable"] = task.e2s_obs_name
        df.attrs["source"] = source_cls.SOURCE_ID
        source_cls._add_task_columns(df, task)

        # Half-open (min, max] so adjacent request windows tile without
        # double-counting observations on a cycle boundary
        mask = (df["time"] > task.datetime_min) & (df["time"] <= task.datetime_max)
        frames.append(df.loc[mask])
    return frames


_GSI_WORKER: dict[str, Any] = {}


def _init_gsi_worker(
    source_cls: type[_UFSObsBase],
    schema: pa.Schema,
    column_map: dict[str, str],
    channel_indexed_fields: dict[str, str],
) -> None:
    """Stash the per-fetch decode context once per worker process."""
    _GSI_WORKER.update(
        source_cls=source_cls,
        schema=schema,
        column_map=column_map,
        channel_indexed_fields=channel_indexed_fields,
    )


def _apply_modifiers(
    tasks: list[_GSIAsyncTask], frames: list[pd.DataFrame]
) -> list[pd.DataFrame]:
    """Apply each task's modifier to its frame, in the parent process."""
    modified = []
    for task, df in zip(tasks, frames):
        assert task.gsi_modifier is not None  # noqa: S101  # only worker copies drop it
        modified.append(task.gsi_modifier(df))
    return modified


def _gsi_decode_worker(
    local_path: str, obs_names: set[str], tasks: list[_GSIAsyncTask]
) -> list[pd.DataFrame]:
    """Decode one diag file with the context stashed by the worker initializer."""
    return _decode_gsi_group(
        _GSI_WORKER["source_cls"],
        _GSI_WORKER["schema"],
        _GSI_WORKER["column_map"],
        _GSI_WORKER["channel_indexed_fields"],
        local_path,
        obs_names,
        tasks,
    )


def _hours_since_to_datetime(values: np.ndarray, origin: datetime) -> np.ndarray:
    """Convert fractional hours since ``origin`` to ``datetime64[ns]``.

    Bit-exact with ``pd.to_timedelta(values, unit="h") + origin``; the rounding
    to 12 decimal places is what reproduces the pandas result.
    """
    hours = values.astype(np.float64, copy=False)
    whole_hours = hours.astype(np.int64)
    fractional_ns = (np.round(hours - whole_hours, 12) * _NS_PER_HOUR).astype(np.int64)
    offset_ns = whole_hours * _NS_PER_HOUR + fractional_ns
    return np.datetime64(origin, "ns") + offset_ns.astype("timedelta64[ns]")


class _UFSObsBase:
    """Base class for GSI data sources.

    This abstract base class provides common functionality for reading NOAA UFS
    GEFS-v13 replay observation data from S3.
    """

    UFS_BUCKET = "noaa-ufs-gefsv13replay-pds"
    SOURCE_ID: str  # To be defined by subclasses
    SCHEMA: pa.Schema  # To be defined by subclasses
    # A GSI diag file is labelled by its synoptic cycle and spans cycle +- 3h
    CYCLE_HALF_WIDTH = timedelta(hours=3)

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        max_workers: int = 24,
        decode_workers: int = 8,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        self.obs_type = "ges"
        self._verbose = verbose
        self._cache = cache
        self._max_workers = max_workers
        self._decode_workers = max(1, decode_workers)
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        # Anonymous obstore S3 store for the public NOAA UFS replay bucket.
        self._store = obstore_store_from_url(
            f"s3://{self.UFS_BUCKET}",
            max_pool_connections=self._max_workers,
            region=self._region,
        )

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    # NOAA UFS GEFSv13 replay archive is a public bucket in us-east-1.
    _region = "us-east-1"

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            DataFrame column names to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).
        """
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_key_set = {task.gsi_obs_key for task in async_tasks}
        fetch_jobs = [self._fetch_remote_file(key) for key in file_key_set]
        await tqdm.gather(
            *fetch_jobs, desc="Fetching GSI files", disable=(not self._verbose)
        )

        df = self._compile_dataframe(async_tasks, variable_list, schema)

        return df

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        """Create async tasks for fetching data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    def _cycle_window(self, t: datetime) -> tuple[datetime, datetime]:
        """Half-open observation window ``(t + lower, t + upper]`` for a request."""
        return t + self._tolerance_lower, t + self._tolerance_upper

    @classmethod
    def _cycles(cls, tmin: datetime, tmax: datetime) -> list[datetime]:
        """Cycles whose ``[T - 3h, T + 3h]`` coverage meets the window ``(tmin, tmax]``.

        A cycle qualifies when ``tmin - 3h < T < tmax + 3h``. Windows that end on a
        cycle edge therefore stop there: widening the request past the edge is what
        pulls in the neighbouring file.
        """
        first = tmin - cls.CYCLE_HALF_WIDTH
        day = first.replace(minute=0, second=0, microsecond=0)
        # Round down to a synoptic label, then step past it since T must exceed `first`
        day = day.replace(hour=(day.hour // CYCLE_HOURS) * CYCLE_HOURS)
        day += timedelta(hours=CYCLE_HOURS)
        last = tmax + cls.CYCLE_HALF_WIDTH
        cycles = []
        while day < last:
            cycles.append(day)
            day += timedelta(hours=CYCLE_HOURS)
        return cycles

    async def _fetch_remote_file(
        self,
        key: str,
        byte_offset: int = 0,
        byte_length: int | None = None,
    ) -> None:
        """Fetches a remote object (by key within UFS_BUCKET) into cache.

        Parameters
        ----------
        key : str
            Object key within UFS_BUCKET to fetch
        byte_offset : int, optional
            Byte offset to start reading from, by default 0
        byte_length : int | None, optional
            Number of bytes to read, by default None (read all)
        """
        cache_path = self.cache_path(key, byte_offset, byte_length)
        try:
            # cache_key keeps the historical sha256(key + offset + length)
            # naming so warm caches remain valid
            await obstore_fetch_to_cache(
                self._store,
                key,
                self.cache,
                byte_offset=byte_offset,
                byte_length=byte_length,
                cache_key=os.path.basename(cache_path),
            )
        except FileNotFoundError:
            self._handle_missing_file(key)

    def _handle_missing_file(self, key: str) -> None:
        """Warn and skip a missing diag file. Archive gaps are expected
        (e.g. various satellite/GPS outages), so shouldn't fully derail
        a call to fetch data.

        Can be overridden by subclasses that require stricter handling.
        """
        uri = f"s3://{self.UFS_BUCKET}/{key}"
        logger.warning(f"File {uri} not found")

    def _compile_dataframe(
        self,
        async_tasks: list[_GSIAsyncTask],
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Compile fetched data into a DataFrame."""
        # Identify schema fields that are per-channel (need Channel_Index lookup)
        channel_indexed_fields: dict[str, str] = {}
        for field in schema:
            if (
                field.metadata
                and b"channel_indexed" in field.metadata
                and b"gsi_name" in field.metadata
            ):
                gsi_name = field.metadata[b"gsi_name"].decode("utf-8")
                channel_indexed_fields[gsi_name] = field.name

        column_map = self._build_column_map(schema)
        groups: list[tuple[str, set[str], list[_GSIAsyncTask]]] = []
        for group in self._group_tasks(async_tasks):
            local_path = self.cache_path(group[0].gsi_obs_key)
            if not pathlib.Path(local_path).is_file():
                logger.warning(
                    "Cached file missing for {}",
                    f"s3://{self.UFS_BUCKET}/{group[0].gsi_obs_key}",
                )
                continue
            groups.append((local_path, {t.gsi_obs_name for t in group}, group))

        if not groups:
            logger.warning(
                "No observation files were available for this request; "
                "returning an empty DataFrame."
            )
            return schema.empty_table().to_pandas()

        frames: list[pd.DataFrame] = []
        parallel = self._decode_workers > 1 and len(groups) >= _MIN_PARALLEL_FILES
        if parallel:
            # Modifiers are closures, which cannot be pickled, so they stay here and
            # the workers receive tasks stripped of them
            specs = [
                [replace(task, gsi_modifier=None) for task in group]
                for _path, _names, group in groups
            ]
            with ProcessPoolExecutor(
                max_workers=min(self._decode_workers, len(groups)),
                initializer=_init_gsi_worker,
                initargs=(type(self), schema, column_map, channel_indexed_fields),
            ) as pool:
                futures = [
                    pool.submit(_gsi_decode_worker, path, names, spec)
                    for (path, names, _group), spec in zip(groups, specs)
                ]
                for (_path, _names, group), future in zip(groups, futures):
                    frames.extend(_apply_modifiers(group, future.result()))
        else:
            for local_path, obs_names, group in groups:
                group_frames = _decode_gsi_group(
                    type(self),
                    schema,
                    column_map,
                    channel_indexed_fields,
                    local_path,
                    obs_names,
                    group,
                )
                frames.extend(_apply_modifiers(group, group_frames))

        result = pd.concat(frames, ignore_index=True)
        return result[[name for name in schema.names if name in result.columns]]

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build mapping from GSI column names to schema column names."""
        column_map = {}
        for field in schema:
            if field.metadata is None or b"gsi_name" not in field.metadata:
                continue
            column_map[field.metadata[b"gsi_name"].decode("utf-8")] = field.name
        # Always include time field for filtering
        time_field = self.SCHEMA.field("time")
        column_map[time_field.metadata[b"gsi_name"].decode("utf-8")] = time_field.name
        return column_map

    @staticmethod
    def _transform_column(
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _transform_column.")

    @staticmethod
    def _add_task_columns(df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add task-specific columns to DataFrame. Override in subclasses."""
        pass

    @staticmethod
    def _group_tasks(
        async_tasks: list[_GSIAsyncTask],
    ) -> list[list[_GSIAsyncTask]]:
        """Group tasks reading the same file over the same window.

        GSI packs several observed quantities into one file (u and v in
        ``diag_conv_uv``, bending angle and level-2 retrievals in
        ``diag_conv_gps``), which would otherwise be decoded once per variable.
        """
        groups: dict[tuple[str, datetime, datetime], list[_GSIAsyncTask]] = {}
        for task in async_tasks:
            key = (task.gsi_obs_key, task.datetime_min, task.datetime_max)
            groups.setdefault(key, []).append(task)
        return list(groups.values())

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification. Can be:
            - None: Returns the full class SCHEMA
            - str: Single field name to select from SCHEMA
            - list[str]: List of field names to select from SCHEMA
            - pa.Schema: Validated against class SCHEMA for compatibility

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields

        Raises
        ------
        KeyError
            If a requested field name is not found in the class SCHEMA
        TypeError
            If a field type in the provided schema doesn't match the class SCHEMA
        ValueError
            If required fields are missing
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            # Validate provided schema against class schema
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not found in class SCHEMA. "
                        f"Available fields: {cls.SCHEMA.names}"
                    )
                expected_type = cls.SCHEMA.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type} from class SCHEMA"
                    )
            return fields

        # fields is list[str] - select fields from class schema
        selected_fields = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected_fields.append(cls.SCHEMA.field(name))

        return pa.schema(selected_fields)

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for GSI based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            start_date = datetime(1980, 1, 1)
            if time < start_date:
                raise ValueError(
                    f"Requested date time {time} needs to be after {start_date} for UFS observations"
                )

    def cache_path(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Gets local cache path given s3 uri

        Parameters
        ----------
        path : str
            s3 uri
        byte_offset : int, optional
            Byte offset of file to read, by default 0
        byte_length : int | None, optional
            Byte length of file to read, by default None

        Returns
        -------
        str
            Local path of cached file
        """
        if not byte_length:
            byte_length = -1
        sha = hashlib.sha256((path + str(byte_offset) + str(byte_length)).encode())
        filename = sha.hexdigest()
        return os.path.join(self.cache, filename)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gsi")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gsi_{self._tmp_cache_hash}"
            )
        return cache_location


class UFSObsConv(_UFSObsBase):
    """NOAA UFS GEFS-v13 replay observations in-situ data

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
    decode_workers : int, optional
        Worker processes used to decode cached diag files, by default 8. Set to 1 to
        decode in the calling process.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.
    verbose : bool, optional
        Log basic progress information, by default True.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-ufs-gefsv13replay-pds/
    - https://psl.noaa.gov/data/ufs_replay/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        ds = UFSObsConv(tolerance=timedelta(hours=2))
        df = ds(datetime(2024, 1, 1, 20), ["u"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.UFSObsConv"
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"gsi_name": "Time"}),
            pa.field(
                "pres", pa.float32(), nullable=True, metadata={"gsi_name": "Pressure"}
            ),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Height"}
            ),
            pa.field(
                "type",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Observation_Type"},
            ),
            pa.field(
                "class",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("station", pa.string(), metadata={"gsi_name": "Station_ID"}),
            pa.field(
                "station_elev",
                pa.float32(),
                nullable=True,
                metadata={"gsi_name": "Station_Elevation"},
            ),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        for v in variable:
            try:
                gsi_name, modifier = GSIConventionalLexicon[v]  # type: ignore
                gsi_platform, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
            except KeyError:
                if v in GSISatelliteLexicon:
                    logger.warning(
                        f"Variable id {v} is a UFS satellite variable, skipping in conventional fetch"
                    )
                    continue
                logger.error(f"Variable id {v} not found in GSI lexicon")
                raise

            for t in time_list:
                tmin, tmax = self._cycle_window(t)
                for day in self._cycles(tmin, tmax):
                    year_key = day.strftime("%Y")
                    month_key = day.strftime("%m")
                    datetime_key = day.strftime("%Y%m%d%H")
                    obs_key = f"{year_key}/{month_key}/{datetime_key}/gsi/diag_{gsi_platform}_{gsi_sensor}_{gsi_product}.{datetime_key}_control.nc4"
                    tasks.append(
                        _GSIAsyncTask(
                            datetime_file=day,
                            datetime_min=tmin,
                            datetime_max=tmax,
                            gsi_obs_key=obs_key,
                            gsi_modifier=modifier,
                            gsi_obs_name=gsi_name,
                            e2s_obs_name=v,
                        )
                    )
        return tasks

    @staticmethod
    def _transform_column(
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values for conventional data."""
        # Convert hours offset to timedelta, and add to datetime of file
        if name == "Time":
            values = _hours_since_to_datetime(values, task.datetime_file)
        # GSI stores Pressure in hPa (mb), convert to Pa
        elif name == "Pressure":
            values = values * 100.0
        return values

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build column map including elev field required for modifiers."""
        column_map = super()._build_column_map(schema)
        # Required for modifier filtering
        elev_field = self.SCHEMA.field("elev")
        column_map[elev_field.metadata[b"gsi_name"].decode("utf-8")] = elev_field.name
        return column_map


class UFSObsSat(_UFSObsBase):
    """NOAA UFS GEFS-v13 replay observations satellite data

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    satellites : list[str], optional
        List of satellite platforms to include, by default includes all platforms.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
    decode_workers : int, optional
        Worker processes used to decode cached diag files, by default 8. Set to 1 to
        decode in the calling process.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.
    verbose : bool, optional
        Log basic progress information, by default True.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-ufs-gefsv13replay-pds/
    - https://psl.noaa.gov/data/ufs_replay/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        # Use all possible satellites
        ds = UFSObsSat(tolerance=timedelta(hours=2))
        df = ds(datetime(2024, 1, 1, 20), ["atms"])

        # Use specific satellite
        ds = UFSObsSat(tolerance=timedelta(hours=2), satellites=["n20"])
        df = ds(datetime(2024, 1, 1, 20), ["atms"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.UFSObsSat"
    VALID_SATELLITES = frozenset(
        [
            "aqua",
            "npp",
            "metop-a",
            "metop-b",
            "metop-c",
            "n15",
            "n16",
            "n17",
            "n18",
            "n19",
            "n20",
        ]
    )
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"gsi_name": "Obs_Time"}),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Elevation"}
            ),
            pa.field(
                "class",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("scan_angle", pa.float32(), metadata={"gsi_name": "Scan_Angle"}),
            pa.field(
                "channel_index",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Channel_Index"},
            ),
            pa.field(
                "sensor_index",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "sensor_chan", "channel_indexed": "true"},
            ),
            pa.field(
                "wavenumber",
                pa.float64(),
                nullable=True,
                metadata={"gsi_name": "wavenumber", "channel_indexed": "true"},
            ),
            pa.field("solza", pa.float32(), metadata={"gsi_name": "Sol_Zenith_Angle"}),
            pa.field(
                "solaza", pa.float32(), metadata={"gsi_name": "Sol_Azimuth_Angle"}
            ),
            pa.field(
                "satellite_za", pa.float32(), metadata={"gsi_name": "Sat_Zenith_Angle"}
            ),
            pa.field(
                "satellite_aza",
                pa.float32(),
                metadata={"gsi_name": "Sat_Azimuth_Angle"},
            ),
            pa.field("satellite", pa.string()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        max_workers: int = 24,
        decode_workers: int = 8,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        if satellites is None:
            satellites = list(self.VALID_SATELLITES)
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {invalid}. "
                    f"Valid satellites are: {sorted(self.VALID_SATELLITES)}"
                )
        self.satellites = satellites
        super().__init__(
            time_tolerance=time_tolerance,
            max_workers=max_workers,
            decode_workers=decode_workers,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        for v in variable:
            try:
                gsi_name, modifier = GSISatelliteLexicon[v]  # type: ignore
                gsi_platforms0, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
                gsi_platforms = [
                    p for p in gsi_platforms0.split(",") if p in self.satellites
                ]
            except KeyError:
                if v in GSIConventionalLexicon:
                    logger.warning(
                        f"Variable id {v} is a UFS conventional variable, skipping in satellite fetch"
                    )
                    continue
                logger.error(f"Variable id {v} not found in GSI lexicon")
                raise

            for gsi_platform in gsi_platforms:
                for t in time_list:
                    tmin, tmax = self._cycle_window(t)
                    for day in self._cycles(tmin, tmax):
                        year_key = day.strftime("%Y")
                        month_key = day.strftime("%m")
                        datetime_key = day.strftime("%Y%m%d%H")
                        obs_key = f"{year_key}/{month_key}/{datetime_key}/gsi/diag_{gsi_sensor}_{gsi_platform}_{gsi_product}.{datetime_key}_control.nc4"
                        tasks.append(
                            _GSIAsyncTask(
                                datetime_file=day,
                                datetime_min=tmin,
                                datetime_max=tmax,
                                gsi_obs_key=obs_key,
                                gsi_modifier=modifier,
                                gsi_obs_name=gsi_name,
                                e2s_obs_name=v,
                                satellite=gsi_platform,
                            )
                        )
        return tasks

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build column map, always including Channel_Index for channel-indexed fields."""
        column_map = super()._build_column_map(schema)
        # Channel_Index is required to expand any channel-indexed fields
        for field in schema:
            if field.metadata and b"channel_indexed" in field.metadata:
                ci_field = self.SCHEMA.field("channel_index")
                ci_gsi = ci_field.metadata[b"gsi_name"].decode("utf-8")
                column_map[ci_gsi] = ci_field.name
                break
        return column_map

    @staticmethod
    def _transform_column(
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values for satellite data."""
        # Convert hours offset to timedelta, and add to datetime of file
        if name == "Obs_Time":
            values = _hours_since_to_datetime(values, task.datetime_file)
        return values

    @staticmethod
    def _add_task_columns(df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add satellite column."""
        df["satellite"] = task.satellite
