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

import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import UFSObsConv, UFSObsSat
from earth2studio.data.ufs import _MIN_PARALLEL_FILES, _hours_since_to_datetime


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        [datetime(year=2024, month=8, day=1, hour=12)],
    ],
)
@pytest.mark.parametrize(
    "variable, tol",
    [
        (["t"], timedelta(hours=1)),
        (["u", "v"], timedelta(hours=2)),
    ],
)
def test_ufsobsconv_fetch(time, variable, tol):
    ds = UFSObsConv(time_tolerance=tol, cache=False, verbose=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

    time_union = pd.DataFrame({"time": np.zeros(df.shape[0])}).astype("bool")
    for t in time:
        df_times = df["time"]
        min_time = t - tol
        max_time = t + tol
        time_union["time"] = time_union["time"] | (
            df_times.ge(min_time) & df_times.le(max_time)
        )

    assert time_union["time"].all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
    ],
)
@pytest.mark.parametrize("variable", [["t"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ufsobsconv_cache(time, variable, cache):
    ds = UFSObsConv(
        time_tolerance=timedelta(hours=1),
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert pathlib.Path(ds.cache).is_dir() == cache

    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ufsobsconv_schema_fields():
    time = datetime(year=2024, month=1, day=1, hour=0)
    tol = timedelta(hours=1)

    ds = UFSObsConv(time_tolerance=tol, cache=False, verbose=False)

    df_full = ds(time, ["t"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ufsobsconv_exceptions():
    ds = UFSObsConv(
        time_tolerance=timedelta(hours=1),
        cache=False,
        verbose=False,
    )

    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 1, 1),
            ["t"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["t2m"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 1, 1), ["t2m"], fields=wrong_type_schema)


def test_ufsobsconv_tolerance_conversion():
    ds_timedelta = UFSObsConv(
        time_tolerance=timedelta(hours=1), cache=False, verbose=False
    )
    assert ds_timedelta._tolerance_lower == timedelta(hours=-1)
    assert ds_timedelta._tolerance_upper == timedelta(hours=1)

    ds_numpy = UFSObsConv(
        time_tolerance=np.timedelta64(1, "h"), cache=False, verbose=False
    )
    assert ds_numpy._tolerance_lower == timedelta(hours=-1)
    assert ds_numpy._tolerance_upper == timedelta(hours=1)

    # Asymmetric tolerance tuple
    ds_asym = UFSObsConv(
        time_tolerance=(np.timedelta64(-3, "h"), np.timedelta64(1, "h")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(hours=-3)
    assert ds_asym._tolerance_upper == timedelta(hours=1)


def test_ufs_time_conversion_matches_pandas():
    values = np.array(
        [-45.0, -0.54487, -0.0001, 0.0, 0.12345679, 3.0],
        dtype=np.float32,
    )
    origin = datetime(2024, 1, 1)
    expected = (pd.to_timedelta(values, unit="h") + origin).to_numpy()

    result = _hours_since_to_datetime(values, origin)

    np.testing.assert_array_equal(result, expected)


def test_ufs_time_conversion_matches_pandas_over_cycle_range():
    """GSI stores whole-cycle offsets in [-3h, +3h]; match pandas across that span."""
    rng = np.random.default_rng(0)
    values = rng.uniform(-3.0, 3.0, 100_000).astype(np.float32)
    origin = datetime(2024, 1, 1)
    expected = (pd.to_timedelta(values, unit="h") + origin).to_numpy()

    result = _hours_since_to_datetime(values, origin)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "tolerance,expected",
    [
        # A 48h HealDA-v2 window is the 8 cycles it is built from, not 9 or 10
        ((timedelta(hours=-45), timedelta(hours=3)), 8),
        # A single 6h frame is a single file
        ((timedelta(hours=-3), timedelta(hours=3)), 1),
        ((timedelta(hours=-45), timedelta(hours=-39)), 1),
        # Sub-cycle windows stay within the one cycle that covers them
        ((timedelta(hours=-1), timedelta(hours=1)), 1),
        # Reaching past a cycle edge is what pulls in the neighbouring file
        ((timedelta(hours=-4), timedelta(hours=4)), 3),
        ((timedelta(hours=-45), timedelta(hours=3, minutes=6)), 9),
    ],
)
def test_ufs_cycle_selection(tolerance, expected):
    analysis = datetime(2024, 1, 1)
    cycles = UFSObsConv._cycles(analysis + tolerance[0], analysis + tolerance[1])

    assert len(cycles) == expected
    assert all(cycle.hour % 6 == 0 for cycle in cycles)
    # Every selected cycle must overlap the requested window
    for cycle in cycles:
        assert cycle + timedelta(hours=3) > analysis + tolerance[0]
        assert cycle - timedelta(hours=3) < analysis + tolerance[1]


def test_ufs_adjacent_windows_select_disjoint_cycles():
    """Contiguous rank-local windows must not both claim the shared cycle file."""
    analysis = datetime(2024, 1, 1)
    left = UFSObsConv._cycles(
        analysis + timedelta(hours=-45), analysis + timedelta(hours=-33)
    )
    right = UFSObsConv._cycles(
        analysis + timedelta(hours=-33), analysis + timedelta(hours=-21)
    )

    assert set(left).isdisjoint(right)


def test_ufs_conv_groups_tasks_sharing_a_file():
    """u/v live in diag_conv_uv, so both must be served by a single decode."""
    source = UFSObsConv(
        time_tolerance=(timedelta(hours=-3), timedelta(hours=3)), cache=False
    )
    tasks = source._create_tasks([datetime(2024, 1, 1)], ["u", "v", "t"])
    groups = source._group_tasks(tasks)

    assert len(tasks) == 3
    assert len(groups) == 2
    uv_group = next(group for group in groups if len(group) > 1)
    assert {task.e2s_obs_name for task in uv_group} == {"u", "v"}
    assert len({task.gsi_obs_key for task in uv_group}) == 1


def test_ufs_sat_tasks_never_share_a_file():
    source = UFSObsSat(time_tolerance=(timedelta(hours=-45), timedelta(hours=3)))
    tasks = source._create_tasks([datetime(2024, 1, 1)], ["atms", "amsua"])
    groups = source._group_tasks(tasks)

    assert len(groups) == len(tasks)


@pytest.mark.parametrize("cls", [UFSObsConv, UFSObsSat])
def test_ufsobs_decode_workers_floor(cls):
    ds = cls(decode_workers=0, cache=False, verbose=False)
    assert ds._decode_workers == 1


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
def test_ufsobssat_decode_workers_agree():
    """Pooled and in-process decode must return the same frame.

    The pool decodes in worker processes and applies the observation modifiers back
    in the parent, so this is the check that the split preserves the result.
    """
    time = datetime(year=2024, month=1, day=1, hour=0)
    # Wide enough to span more cycles than _MIN_PARALLEL_FILES, or the pooled arm
    # falls back to in-process decode and the comparison is vacuous
    kwargs = dict(
        time_tolerance=(timedelta(hours=-21), timedelta(hours=3)),
        satellites=["npp"],
        cache=True,
        verbose=False,
    )

    pooled_source = UFSObsSat(decode_workers=4, **kwargs)
    tasks = pooled_source._create_tasks([time], ["atms"])
    assert len(pooled_source._group_tasks(tasks)) >= _MIN_PARALLEL_FILES

    serial = UFSObsSat(decode_workers=1, **kwargs)(time, ["atms"])
    pooled = pooled_source(time, ["atms"])

    pd.testing.assert_frame_equal(serial, pooled)


@pytest.mark.parametrize("cls", [UFSObsConv, UFSObsSat])
def test_ufsobs_missing_file_warns_not_raises(cls, caplog):
    """A missing diag file warns and returns rather than aborting the fetch.

    GSI archives have gaps (e.g. an absent GNSS-RO ``gps`` cycle or a
    decommissioned satellite platform); both obs sources must tolerate
    them so a bulk request spanning many cycles is not derailed by one
    missing object.
    """
    ds = cls(cache=False, verbose=False)
    key = "2024/03/2024033018/gsi/diag_conv_gps_ges.2024033018_control.nc4"

    # Must not raise (previously the conventional source raised here).
    ds._handle_missing_file(key)
    assert "not found" in caplog.text


@pytest.mark.parametrize("cls", [UFSObsConv, UFSObsSat])
def test_ufsobs_all_files_missing_returns_empty(cls):
    """When every file is skipped, an empty schema-shaped frame is returned.

    Guards against ``pd.concat([])`` raising "No objects to concatenate"
    when a whole request's worth of diag files is absent — consumers get
    a well-formed empty DataFrame to apply their own handling to.
    """
    ds = cls(cache=False, verbose=False)
    schema = ds.resolve_fields(None)

    # No async tasks => no frames compiled => empty result.
    df = ds._compile_dataframe([], ["t"], schema)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == schema.names


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        [datetime(year=2024, month=1, day=1, hour=6)],
    ],
)
@pytest.mark.parametrize(
    "variable, satellites, tol",
    [
        (["atms"], ["npp"], timedelta(hours=1)),
        (["mhs"], ["metop-a", "metop-b"], timedelta(hours=2)),
        (["airs"], ["aqua"], timedelta(hours=1)),
    ],
)
def test_ufsobssat_fetch(time, variable, satellites, tol):
    ds = UFSObsSat(
        time_tolerance=tol, satellites=satellites, cache=False, verbose=False
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert "satellite" in df.columns

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

    if not df.empty:
        time_union = pd.DataFrame({"time": np.zeros(df.shape[0])}).astype("bool")
        for t in time:
            df_times = df["time"]
            min_time = t - tol
            max_time = t + tol
            time_union["time"] = time_union["time"] | (
                df_times.ge(min_time) & df_times.le(max_time)
            )
        assert time_union["time"].all()
        assert set(df["satellite"].unique()).issubset(set(satellites))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
    ],
)
@pytest.mark.parametrize("variable", [["atms"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ufsobssat_cache(time, variable, cache):
    ds = UFSObsSat(
        time_tolerance=timedelta(hours=1),
        satellites=["npp"],
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ufsobssat_schema_fields():
    time = datetime(year=2024, month=1, day=1, hour=0)
    tol = timedelta(hours=1)

    ds = UFSObsSat(time_tolerance=tol, satellites=["npp"], cache=False, verbose=False)

    df_full = ds(time, ["atms"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "satellite", "observation", "variable"]
    df_subset = ds(time, ["atms"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ufsobssat_exceptions():
    ds = UFSObsSat(
        time_tolerance=timedelta(hours=1),
        satellites=["npp"],
        cache=False,
        verbose=False,
    )

    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 1, 1),
            ["atms"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["atms"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 1, 1), ["atms"], fields=wrong_type_schema)

    # Test satellites
    with pytest.raises(ValueError, match="Invalid satellite"):
        UFSObsSat(satellites=["invalid_sat"])

    with pytest.raises(ValueError, match="Invalid satellite"):
        UFSObsSat(satellites=["npp", "invalid_sat"])

    ds = UFSObsSat(cache=False, verbose=False)
    assert set(ds.satellites) == ds.VALID_SATELLITES

    ds = UFSObsSat(satellites=["npp", "n20"], cache=False, verbose=False)
    assert ds.satellites == ["npp", "n20"]


def test_gsi_cache_path():
    ds = UFSObsConv(cache=True, verbose=False)
    path1 = ds.cache_path("s3://bucket/file.nc4")
    path2 = ds.cache_path("s3://bucket/file.nc4", byte_offset=100)
    path3 = ds.cache_path("s3://bucket/file.nc4", byte_offset=100, byte_length=200)

    assert path1 != path2
    assert path2 != path3
    assert all(p.startswith(ds.cache) for p in [path1, path2, path3])
