# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch
import torch.distributed as dist
from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.healda import (
    VideoHealDA,
    prepare_obs_context,
)
from physicsnemo.experimental.models.healda.obs_context import ObsContext

from earth2studio.models.da.healda_v2 import _enable_context_parallel

N_WINDOW = 8
LEVEL_IN = 6
LEVEL_MODEL = 5
NPIX_IN = 12 * 4**LEVEL_IN
NPIX_MODEL = 12 * 4**LEVEL_MODEL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-parallel-size",
        type=int,
        choices=(1, 2, 4, 8),
        required=True,
    )
    parser.add_argument("--obs-per-frame", type=int, default=2_500_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def _make_obs_context(
    obs_per_frame: int,
    local_frames: int,
    device: torch.device,
) -> tuple[ObsContext, float]:
    nobs = obs_per_frame * local_frames
    total_pixels = local_frames * NPIX_MODEL
    flat_idx = (torch.arange(nobs, dtype=torch.int64, device=device) % total_pixels).to(
        torch.int32
    )
    ids = torch.zeros(nobs, dtype=torch.int64, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    context = prepare_obs_context(
        obs=torch.randn(nobs, device=device),
        float_metadata=torch.randn(nobs, 50, device=device),
        obs_type=ids,
        channel=ids,
        platform=ids,
        flat_idx=flat_idx,
        total_pixels=total_pixels,
    )
    end.record()
    end.synchronize()
    return context, start.elapsed_time(end)


def main() -> None:
    args = _parse_args()
    os.environ.pop("HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR", None)
    DistributedManager.initialize()
    manager = DistributedManager()
    world_size = manager.world_size
    if world_size != args.model_parallel_size:
        raise ValueError(
            f"torchrun world size {world_size} does not match "
            f"model_parallel_size {args.model_parallel_size}"
        )

    local_frames = N_WINDOW // args.model_parallel_size
    group = None
    if args.model_parallel_size > 1:
        mesh = manager.initialize_mesh(
            (1, args.model_parallel_size),
            ("data", "model"),
        )
        group = mesh.get_group("model")

    with torch.device(manager.device):
        model = VideoHealDA(time_length=local_frames)
    if group is not None:
        _enable_context_parallel(model, group, args.model_parallel_size)
    model.eval()

    obs_context, prepare_ms = _make_obs_context(
        args.obs_per_frame,
        local_frames,
        manager.device,
    )
    condition = torch.randn(
        1,
        2,
        local_frames,
        NPIX_IN,
        device=manager.device,
    )
    timestep = torch.zeros(1, device=manager.device)
    second_of_day = torch.rand(1, local_frames, device=manager.device) * 86400.0
    day_of_year = torch.rand(1, local_frames, device=manager.device) * 365.0

    if manager.rank == 0:
        print(
            f"model_parallel_size={args.model_parallel_size} "
            f"local_frames={local_frames} "
            f"obs_per_frame={args.obs_per_frame:,}",
            flush=True,
        )
        print(f"obs_context_prepare_ms={prepare_ms:.3f}", flush=True)

    def forward() -> torch.Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                condition,
                timestep,
                second_of_day,
                day_of_year,
                obs_context,
            )
        if manager.rank == args.model_parallel_size - 1:
            analysis = prediction[:, :, -1].contiguous()
        else:
            analysis = torch.empty_like(prediction[:, :, 0]).contiguous()
        if group is not None:
            dist.broadcast(
                analysis,
                src=args.model_parallel_size - 1,
                group=group,
            )
        return analysis

    for index in range(args.warmup):
        forward()
        torch.cuda.synchronize(manager.device)
        if manager.rank == 0:
            print(f"warmup={index + 1}/{args.warmup}", flush=True)

    torch.cuda.reset_peak_memory_stats(manager.device)
    latencies = []
    analysis = None
    for index in range(args.iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        analysis = forward()
        end.record()
        end.synchronize()
        latency = torch.tensor(
            start.elapsed_time(end),
            dtype=torch.float64,
            device=manager.device,
        )
        dist.all_reduce(latency, op=dist.ReduceOp.MAX)
        latencies.append(float(latency.item()))
        if manager.rank == 0:
            print(
                f"iteration={index + 1}/{args.iterations} latency_ms={latencies[-1]:.3f}",
                flush=True,
            )

    peak_memory = torch.tensor(
        torch.cuda.max_memory_allocated(manager.device) / 2**30,
        device=manager.device,
    )
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    if manager.rank == 0:
        if analysis is None:
            raise RuntimeError("No benchmark iterations completed")
        print(f"mean_latency_ms={sum(latencies) / len(latencies):.3f}", flush=True)
        print(f"max_rank_peak_memory_gib={float(peak_memory.item()):.3f}", flush=True)
        print(f"analysis_shape={tuple(analysis.shape)}", flush=True)


if __name__ == "__main__":
    main()
