# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from earth2studio.models.da.healda_v2 import (
    N_WINDOW,
    WINDOW_STEP_HOURS,
    HealDAv2,
)

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "earth2studio"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-time", default="2024-01-01T00:00")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("EARTH2STUDIO_CACHE", str(DEFAULT_CACHE)),
    )
    parser.add_argument(
        "--model-parallel-size",
        type=int,
        choices=(1, 2, 4, 8),
        required=True,
    )
    parser.add_argument("--random-data", action="store_true")
    parser.add_argument("--conv-obs-per-frame", type=int, default=1_250_000)
    parser.add_argument("--sat-obs-per-frame", type=int, default=1_250_000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def _random_atms_observations(
    model: HealDAv2,
    analysis_time: np.datetime64,
    obs_per_frame: int,
) -> pd.DataFrame:
    frame_times = _local_frame_times(model, analysis_time)
    nobs = obs_per_frame * model.frames_per_rank
    rng = np.random.default_rng(model.model_parallel_rank)
    observations = pd.DataFrame(
        {
            "time": np.repeat(frame_times, obs_per_frame),
            "lat": rng.uniform(-90.0, 90.0, nobs).astype(np.float32),
            "lon": rng.uniform(0.0, 360.0, nobs).astype(np.float32),
            "observation": rng.normal(250.0, 25.0, nobs).astype(np.float32),
            "variable": pd.Categorical.from_codes(
                np.zeros(nobs, dtype=np.int8),
                ["atms"],
            ),
            "sensor_index": rng.integers(1, 23, nobs, dtype=np.uint16),
            "satellite": pd.Categorical.from_codes(
                np.zeros(nobs, dtype=np.int8),
                ["n20"],
            ),
            "scan_angle": rng.uniform(-60.0, 60.0, nobs).astype(np.float32),
            "satellite_za": rng.uniform(0.0, 65.0, nobs).astype(np.float32),
            "solza": rng.uniform(0.0, 180.0, nobs).astype(np.float32),
        }
    )
    observations.attrs["request_time"] = np.array([analysis_time])
    return observations


def _random_conv_observations(
    model: HealDAv2,
    analysis_time: np.datetime64,
    obs_per_frame: int,
) -> pd.DataFrame:
    frame_times = _local_frame_times(model, analysis_time)
    nobs = obs_per_frame * model.frames_per_rank
    rng = np.random.default_rng(model.model_parallel_rank + 1_000)
    observations = pd.DataFrame(
        {
            "time": np.repeat(frame_times, obs_per_frame),
            "lat": rng.uniform(-90.0, 90.0, nobs).astype(np.float32),
            "lon": rng.uniform(0.0, 360.0, nobs).astype(np.float32),
            "observation": rng.normal(270.0, 10.0, nobs).astype(np.float32),
            "variable": pd.Categorical.from_codes(
                np.zeros(nobs, dtype=np.int8),
                ["t"],
            ),
            "type": np.full(nobs, 120, dtype=np.uint16),
            "elev": rng.uniform(0.0, 1_000.0, nobs).astype(np.float32),
            "pres": rng.uniform(5_000.0, 100_000.0, nobs).astype(np.float32),
        }
    )
    observations.attrs["request_time"] = np.array([analysis_time])
    return observations


def make_random_observations(
    model: HealDAv2,
    analysis_time: np.datetime64,
    conv_obs_per_frame: int = 1_250_000,
    sat_obs_per_frame: int = 1_250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create rank-local conventional and satellite benchmark observations."""
    conv_obs = _random_conv_observations(
        model,
        analysis_time,
        conv_obs_per_frame,
    )
    sat_obs = _random_atms_observations(
        model,
        analysis_time,
        sat_obs_per_frame,
    )
    return conv_obs, sat_obs


def _local_frame_times(
    model: HealDAv2,
    analysis_time: np.datetime64,
) -> np.ndarray:
    first_frame = model.model_parallel_rank * model.frames_per_rank
    return np.array(
        [
            analysis_time
            - np.timedelta64(
                WINDOW_STEP_HOURS * (N_WINDOW - 1 - global_frame),
                "h",
            )
            for global_frame in range(
                first_frame,
                first_frame + model.frames_per_rank,
            )
        ]
    )


def _fetch_ufs_observations(
    model: HealDAv2,
    analysis_time: np.datetime64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from earth2studio.data import UFSObsConv, UFSObsSat, fetch_dataframe

    request_time = np.array([analysis_time], dtype="datetime64[ns]")
    tolerance = model.input_time_tolerance(analysis_time)
    conv_schema, sat_schema = model.input_coords()
    conv_obs = fetch_dataframe(
        UFSObsConv(time_tolerance=tolerance, verbose=False),
        time=request_time,
        variable=np.asarray(conv_schema["variable"]),
        fields=np.asarray(list(conv_schema.keys())),
    )
    sat_obs = fetch_dataframe(
        UFSObsSat(time_tolerance=tolerance, verbose=False),
        time=request_time,
        variable=np.asarray(sat_schema["variable"]),
        fields=np.asarray(list(sat_schema.keys())),
    )
    return conv_obs, sat_obs


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != args.model_parallel_size:
        raise ValueError(
            f"torchrun world size {world_size} does not match "
            f"model_parallel_size {args.model_parallel_size}"
        )
    os.environ["EARTH2STUDIO_CACHE"] = args.cache_dir
    model = HealDAv2.load_model(model_parallel_size=args.model_parallel_size)
    analysis_time = np.datetime64(args.analysis_time)
    if args.random_data:
        conv_obs, sat_obs = make_random_observations(
            model,
            analysis_time,
            args.conv_obs_per_frame,
            args.sat_obs_per_frame,
        )
        data_source = "random"
    else:
        conv_obs, sat_obs = _fetch_ufs_observations(model, analysis_time)
        data_source = "ufs"

    tolerance_hours = tuple(
        int(value / np.timedelta64(1, "h"))
        for value in model.input_time_tolerance(analysis_time)
    )
    print(
        f"rank={model.model_parallel_rank} "
        f"local_frames={model.frames_per_rank} "
        f"time_tolerance_hours={tolerance_hours} "
        f"conv_obs={len(conv_obs):,} sat_obs={len(sat_obs):,}",
        flush=True,
    )
    counts = torch.tensor(
        [len(conv_obs), len(sat_obs)],
        dtype=torch.int64,
        device=model.device,
    )
    dist.all_reduce(counts)
    if model.model_parallel_rank == 0:
        print(
            f"data_source={data_source} "
            f"model_parallel_size={model.model_parallel_size} "
            f"global_conv_obs={int(counts[0]):,} "
            f"global_sat_obs={int(counts[1]):,}",
            flush=True,
        )

    result = None
    for index in range(args.warmup):
        result = model(
            conv_obs=conv_obs,
            sat_obs=sat_obs,
            analysis_time=analysis_time,
        )
        torch.cuda.synchronize(model.device)
        if model.model_parallel_rank == 0:
            print(f"warmup={index + 1}/{args.warmup}", flush=True)

    latencies = []
    for index in range(args.iterations):
        dist.barrier()
        start = time.perf_counter()
        result = model(
            conv_obs=conv_obs,
            sat_obs=sat_obs,
            analysis_time=analysis_time,
        )
        torch.cuda.synchronize(model.device)
        latency = torch.tensor(
            (time.perf_counter() - start) * 1000.0,
            dtype=torch.float64,
            device=model.device,
        )
        dist.all_reduce(latency, op=dist.ReduceOp.MAX)
        latencies.append(float(latency.item()))
        if model.model_parallel_rank == 0:
            print(
                f"iteration={index + 1}/{args.iterations} "
                f"latency_ms={latencies[-1]:.3f}",
                flush=True,
            )

    if model.model_parallel_rank == 0:
        if result is None:
            raise RuntimeError("No benchmark iterations completed")
        print(f"mean_latency_ms={sum(latencies) / len(latencies):.3f}", flush=True)
        print(f"analysis_shape={result.shape}", flush=True)


if __name__ == "__main__":
    main()
