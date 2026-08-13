# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Profile warm-cache UFSObsConv / UFSObsSat fetch for HealDA-v2."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import earth2studio
import earth2studio.data.ufs as ufs_module
from earth2studio.data import UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.data.utils import prep_data_inputs
from earth2studio.models.da.healda_v2 import (
    FRAME_CONTEXT_HOURS,
    N_WINDOW,
    SAT_UFS_VARIABLES,
    WINDOW_STEP_HOURS,
)
from earth2studio.models.da.healda_v2_utils import CONV_REQUEST_VARIABLES

CONV_FIELDS = np.array(
    ["time", "lat", "lon", "observation", "variable", "type", "elev", "pres"]
)
SAT_FIELDS = np.array(
    [
        "time",
        "lat",
        "lon",
        "observation",
        "variable",
        "sensor_index",
        "satellite",
        "scan_angle",
        "satellite_za",
        "solza",
    ]
)


def healda_window_tolerance() -> tuple[np.timedelta64, np.timedelta64]:
    """Single-rank HealDA-v2 window: (-45h, +3h)."""
    lower_hours = -WINDOW_STEP_HOURS * (N_WINDOW - 1) - FRAME_CONTEXT_HOURS
    return np.timedelta64(lower_hours, "h"), np.timedelta64(FRAME_CONTEXT_HOURS, "h")


def cache_hits(
    source: UFSObsConv | UFSObsSat,
    time_array: np.ndarray,
    variables: np.ndarray,
) -> tuple[int, int]:
    times, variable_list = prep_data_inputs(time_array, variables)
    keys = {task.gsi_obs_key for task in source._create_tasks(times, variable_list)}
    hits = sum(Path(source.cache_path(key)).is_file() for key in keys)
    return hits, len(keys)


def fetch_one(
    name: str,
    source: UFSObsConv | UFSObsSat,
    analysis_time: np.ndarray,
    variables: np.ndarray,
    fields: np.ndarray,
    rows: list[dict],
) -> None:
    hits, keys = cache_hits(source, analysis_time, variables)
    start = time.perf_counter()
    frame = fetch_dataframe(
        source,
        time=analysis_time,
        variable=variables,
        fields=fields,
    )
    wall = time.perf_counter() - start
    rows.append(
        {
            "name": name,
            "analysis": str(analysis_time[0]),
            "cache": f"{hits}/{keys}",
            "obs": len(frame),
            "wall_s": wall,
        }
    )
    print(
        f"\n{name}: cache={hits}/{keys}, rows={len(frame):,}, wall={wall:.3f}s",
        flush=True,
    )


def print_summary(label: str, rows: list[dict]) -> None:
    print(f"\n===== SUMMARY {label} =====", flush=True)
    print(
        f"{'source':<6} {'analysis':<20} {'cache':>8} {'obs':>12} {'wall_s':>8}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['name']:<6} {row['analysis']:<20} {row['cache']:>8} "
            f"{row['obs']:>12,} {row['wall_s']:>8.3f}",
            flush=True,
        )
    total_wall = sum(r["wall_s"] for r in rows)
    total_obs = sum(r["obs"] for r in rows)
    print(
        f"{'TOTAL':<6} {'':<20} {'':>8} {total_obs:>12,} {total_wall:>8.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-time", default="2024-01-01T00:00")
    parser.add_argument(
        "--leads",
        default="0",
        help="Comma-separated lead hours; default one analysis at init",
    )
    parser.add_argument(
        "--window",
        choices=("healda48", "frame6"),
        default="healda48",
        help="healda48 = (-45h,+3h); frame6 = (-3h,+3h)",
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=8,
        help="Decode worker processes; 1 decodes in the calling process",
    )
    args = parser.parse_args()

    if args.window == "healda48":
        tolerance = healda_window_tolerance()
    else:
        tolerance = (np.timedelta64(-3, "h"), np.timedelta64(3, "h"))

    print(f"earth2studio={earth2studio.__file__}", flush=True)
    print(f"ufs={ufs_module.__file__}", flush=True)
    print(f"cache={os.environ.get('EARTH2STUDIO_CACHE', '<default>')}", flush=True)
    print(f"window={args.window} tolerance={tolerance}", flush=True)
    print(f"decode_workers={args.decode_workers}", flush=True)

    init_time = np.datetime64(args.init_time)
    leads = [int(value) for value in args.leads.split(",")]
    conv_variables = np.array(CONV_REQUEST_VARIABLES)
    sat_variables = np.array(SAT_UFS_VARIABLES)

    for pass_index in range(args.passes):
        rows: list[dict] = []
        print(f"\n######## pass {pass_index + 1}/{args.passes} ########", flush=True)
        for lead in leads:
            analysis_time = np.array([init_time + np.timedelta64(lead, "h")])
            print(
                f"\n=== lead={lead}h analysis={analysis_time[0]} ===",
                flush=True,
            )
            fetch_one(
                "conv",
                UFSObsConv(
                    time_tolerance=tolerance,
                    decode_workers=args.decode_workers,
                    verbose=False,
                ),
                analysis_time,
                conv_variables,
                CONV_FIELDS,
                rows,
            )
            fetch_one(
                "sat",
                UFSObsSat(
                    time_tolerance=tolerance,
                    decode_workers=args.decode_workers,
                    verbose=False,
                ),
                analysis_time,
                sat_variables,
                SAT_FIELDS,
                rows,
            )
        print_summary(f"pass{pass_index + 1} {args.window}", rows)


if __name__ == "__main__":
    main()
